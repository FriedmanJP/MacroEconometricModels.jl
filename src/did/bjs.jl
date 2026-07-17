# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Borusyak, Jaravel & Spiess (2024) imputation estimator for staggered DiD.

Estimates unit and time fixed effects on the untreated subsample, imputes
counterfactual Y(0) for treated observations, and aggregates cell-level
treatment effects to event-time ATTs with cohort-size weights.

# Reference
Borusyak, K., Jaravel, X. & Spiess, J. (2024). *REStat* 106(2), 391-401.
"""

using LinearAlgebra, Statistics, Distributions

"""
    _estimate_bjs(pd::PanelData{T}, outcome_col::Int, treat_col::Int;
                   leads::Int=0, horizon::Int=5,
                   control_group::Symbol=:never_treated,
                   cluster::Symbol=:unit,
                   conf_level::Real=0.95) where {T<:AbstractFloat}

Internal Borusyak-Jaravel-Spiess (2024) imputation DiD estimator.

# Algorithm
1. Identify untreated observations (never-treated or not-yet-treated at time t)
2. Estimate unit FE (alpha_i) and time FE (gamma_t) iteratively on untreated subsample
3. Impute counterfactual Y(0) = alpha_hat_i + gamma_hat_t for treated observations
4. Compute cell-level treatment effect tau_{it} = Y_{it} - Y(0)_{it}
5. Compute group-time ATT as mean tau per (cohort, time) cell
6. Aggregate to event-time ATTs using cohort-size weights
7. SEs via diagonal influence function approximation

# Arguments
- `pd`: Panel data with outcome and treatment timing columns
- `outcome_col`: Column index of outcome variable
- `treat_col`: Column index of treatment timing variable

# Keyword Arguments
- `leads`: Number of pre-treatment periods (default: 0)
- `horizon`: Post-treatment horizon (default: 5)
- `control_group`: `:never_treated` (default) or `:not_yet_treated`
- `cluster`: SE clustering: `:unit`, `:time`, `:twoway` (default: `:unit`)
- `conf_level`: Confidence level (default: 0.95)

# Returns
`DIDResult{T}` with event-study ATTs, SEs, CIs, and group-time ATTs.

# Reference
Borusyak, K., Jaravel, X. & Spiess, J. (2024). *REStat* 106(2), 391-401.
"""
function _estimate_bjs(pd::PanelData{T}, outcome_col::Int, treat_col::Int;
                        leads::Int=0, horizon::Int=5,
                        control_group::Symbol=:never_treated,
                        cluster::Symbol=:unit,
                        conf_level::Real=0.95) where {T<:AbstractFloat}

    timing = _extract_treatment_timing(pd, treat_col)

    # Override timing with cohort_id if present
    if pd.cohort_id !== nothing
        for g in 1:pd.n_groups
            mask = pd.group_id .== g
            timing[g] = pd.cohort_id[findfirst(mask)]
        end
    end

    all_times = sort(unique(pd.time_id))
    n_times = length(all_times)

    # Identify cohorts and control groups
    cohorts = sort(unique([t for (_, t) in timing if t > 0]))
    never_treated = [g for (g, t) in timing if t == 0]
    n_cohorts = length(cohorts)
    n_control = length(never_treated)

    if control_group == :never_treated && n_control == 0
        throw(ArgumentError("No never-treated units found. Use control_group=:not_yet_treated"))
    end

    n_treated = length([g for (g, t) in timing if t > 0])

    # =========================================================================
    # Step 1: Build panel lookup and identify untreated observations
    # =========================================================================

    # panel[group][time] = outcome value
    panel = Dict{Int, Dict{Int, T}}()
    for i in 1:pd.T_obs
        g = pd.group_id[i]
        t = pd.time_id[i]
        if !haskey(panel, g)
            panel[g] = Dict{Int, T}()
        end
        panel[g][t] = pd.data[i, outcome_col]
    end

    # Classify each (group, time) as untreated or treated
    # Untreated: unit is never-treated (g_time == 0) OR not yet treated (g_time > t)
    # For control_group == :never_treated, only never-treated units serve as controls
    # For control_group == :not_yet_treated, both never-treated and not-yet-treated units
    # are used in the FE estimation
    all_groups = sort(unique(pd.group_id))

    # Collect untreated (group, time) pairs and their outcomes
    control_y = T[]          # outcome values for control observations
    control_g = Int[]        # group ids for control observations
    control_t = Int[]        # time ids for control observations

    treated_y = T[]          # outcome values for treated observations
    treated_g = Int[]        # group ids for treated observations
    treated_t = Int[]        # time ids for treated observations
    treated_cohort = Int[]   # cohort (treatment time) for treated observations

    for g in all_groups
        g_time = timing[g]
        haskey(panel, g) || continue
        for t in all_times
            haskey(panel[g], t) || continue
            y_val = panel[g][t]

            if g_time == 0
                # Never-treated: always control
                push!(control_y, y_val)
                push!(control_g, g)
                push!(control_t, t)
            elseif g_time > t
                # Not yet treated at time t
                if control_group == :not_yet_treated
                    push!(control_y, y_val)
                    push!(control_g, g)
                    push!(control_t, t)
                elseif control_group == :never_treated
                    # Still use for FE estimation if the unit is not yet treated
                    # In BJS, ALL untreated (i,t) cells are used for FE estimation
                    push!(control_y, y_val)
                    push!(control_g, g)
                    push!(control_t, t)
                end
            else
                # Treated observation: g_time > 0 and t >= g_time
                push!(treated_y, y_val)
                push!(treated_g, g)
                push!(treated_t, t)
                push!(treated_cohort, g_time)
            end
        end
    end

    n_ctrl = length(control_y)
    n_treat = length(treated_y)

    if n_ctrl == 0
        throw(ArgumentError("No untreated observations found for FE estimation"))
    end

    # =========================================================================
    # Step 2: Estimate unit and time FE on untreated subsample (iterative)
    # =========================================================================

    # Unique groups and times in control subsample
    ctrl_groups_unique = sort(unique(control_g))
    ctrl_times_unique = sort(unique(control_t))

    unit_fe = Dict{Int, T}(g => zero(T) for g in ctrl_groups_unique)
    time_fe = Dict{Int, T}(t => zero(T) for t in ctrl_times_unique)

    # Index buckets (ascending k, so accumulation order — and hence every FE
    # iterate — is bit-identical to the previous full 1:n_ctrl scans)
    ctrl_by_time = Dict{Int, Vector{Int}}()
    ctrl_by_group = Dict{Int, Vector{Int}}()
    for k in 1:n_ctrl
        push!(get!(() -> Int[], ctrl_by_time, control_t[k]), k)
        push!(get!(() -> Int[], ctrl_by_group, control_g[k]), k)
    end

    for iter in 1:100
        max_change = zero(T)

        # Update time FE: gamma_t = mean(y - alpha_g) for control obs at t
        for t in ctrl_times_unique
            sum_val = zero(T)
            count_val = 0
            for k in ctrl_by_time[t]
                sum_val += control_y[k] - get(unit_fe, control_g[k], zero(T))
                count_val += 1
            end
            if count_val > 0
                new_val = sum_val / T(count_val)
                max_change = max(max_change, abs(new_val - time_fe[t]))
                time_fe[t] = new_val
            end
        end

        # Update unit FE: alpha_g = mean(y - gamma_t) for control obs of g
        for g in ctrl_groups_unique
            sum_val = zero(T)
            count_val = 0
            for k in ctrl_by_group[g]
                sum_val += control_y[k] - get(time_fe, control_t[k], zero(T))
                count_val += 1
            end
            if count_val > 0
                new_val = sum_val / T(count_val)
                max_change = max(max_change, abs(new_val - unit_fe[g]))
                unit_fe[g] = new_val
            end
        end

        # Convergence check
        if max_change < T(1e-10)
            break
        end
    end

    # =========================================================================
    # Step 3: Impute counterfactual and compute cell-level treatment effects
    # =========================================================================

    tau = Vector{T}(undef, n_treat)
    for k in 1:n_treat
        g = treated_g[k]
        t = treated_t[k]
        alpha_g = get(unit_fe, g, zero(T))
        gamma_t = get(time_fe, t, zero(T))
        y0_hat = alpha_g + gamma_t
        tau[k] = treated_y[k] - y0_hat
    end

    # =========================================================================
    # Step 4: Group-time ATT: average tau per (cohort, time) cell
    # =========================================================================

    group_time_att = fill(T(NaN), n_cohorts, n_times)

    # Bucket treated obs by (cohort, time) once — ascending k preserves the
    # accumulation order of the previous per-cell 1:n_treat scans exactly
    treated_by_cell = Dict{Tuple{Int,Int}, Vector{Int}}()
    for k in 1:n_treat
        push!(get!(() -> Int[], treated_by_cell, (treated_cohort[k], treated_t[k])), k)
    end

    for (ci, g_time) in enumerate(cohorts)
        for (ti, t) in enumerate(all_times)
            cell = get(treated_by_cell, (g_time, t), nothing)
            if cell !== nothing
                group_time_att[ci, ti] = mean(@view tau[cell])
            end
        end
    end

    # =========================================================================
    # Step 5: Aggregate to event-time ATTs with cohort-size weights
    # =========================================================================

    reference_period = -1
    event_times_all = collect(-leads:horizon)
    n_evt = length(event_times_all)
    att_agg = zeros(T, n_evt)
    se_agg = zeros(T, n_evt)

    # =========================================================================
    # BJS (2024) Prop.-6 influence-function variance.
    # The imputation τ̂(e) = (1/N_e) Σ_{o∈treated(e)} (y_o − x_o'β̂) is LINEAR in Y, so its
    # exact variance includes the FE estimation-error term W_e'M⁺W_e that the naive
    # var(τ)/n omits (it treats the fitted α̂_i, γ̂_t as known). Build the untreated
    # two-way-FE design X0 (rank-deficient by 1), its Gram pseudo-inverse M⁺, and the
    # untreated residuals. Point estimates att_agg are unchanged (= mean(τ) over treated(e)).
    # =========================================================================
    ctrl_U = length(ctrl_groups_unique)
    ctrl_P = length(ctrl_times_unique)
    K_fe = ctrl_U + ctrl_P
    unit_col = Dict(g => i for (i, g) in enumerate(ctrl_groups_unique))
    time_col = Dict(t => ctrl_U + j for (j, t) in enumerate(ctrl_times_unique))
    X0 = zeros(T, n_ctrl, K_fe)
    @inbounds for o in 1:n_ctrl
        X0[o, unit_col[control_g[o]]] = one(T)
        X0[o, time_col[control_t[o]]] = one(T)
    end
    Mpinv = Matrix{T}(robust_inv(Hermitian(X0' * X0); silent=true))  # handles the rank-1 deficiency
    beta_fe = Mpinv * (X0' * control_y)
    resid0 = control_y .- X0 * beta_fe
    dof0 = max(n_ctrl - (K_fe - 1), 1)                              # two-way FE rank = U+P-1
    sigma2 = sum(abs2, resid0) / dof0

    _treat_row!(x, g, t) = begin
        fill!(x, zero(T))
        haskey(unit_col, g) && (x[unit_col[g]] = one(T))
        haskey(time_col, t) && (x[time_col[t]] = one(T))
        x
    end

    use_cluster = cluster === :unit
    evt_of = [treated_t[k] - treated_cohort[k] for k in 1:n_treat]
    xrow = zeros(T, K_fe)

    # Build the FULL E×E influence-function covariance of the event-study ATT vector,
    # not just its diagonal. The cross-horizon terms are non-zero because every τ̂(e)
    # shares the SAME untreated two-way-FE estimate β̂ (→ common −W_e'β̂ term); the treated
    # idiosyncratic errors are disjoint across e (each treated obs has a unique event time),
    # so they contribute only on the diagonal. Store W_e per horizon (homoskedastic form)
    # and, for unit clustering, the per-unit IF score columns Ψ (att_vcov = Ψ'Ψ).
    E = length(event_times_all)
    We_mat = zeros(T, K_fe, E)                     # column ei = W_e (0 for reference / empty e)
    Ne_vec = zeros(Int, E)
    all_units = sort(unique(vcat(treated_g, control_g)))
    unit_idx = Dict(g => i for (i, g) in enumerate(all_units))
    Psi = use_cluster ? zeros(T, length(all_units), E) : Matrix{T}(undef, 0, 0)

    for (ei, e) in enumerate(event_times_all)
        e == reference_period && continue
        Te = findall(==(e), evt_of)
        Ne = length(Te)
        Ne == 0 && continue
        Ne_vec[ei] = Ne

        att_agg[ei] = mean(@view tau[Te])          # simple mean of imputed effects (= old value)

        We = @view We_mat[:, ei]
        for k in Te
            We .+= _treat_row!(xrow, treated_g[k], treated_t[k])
        end
        We ./= Ne

        if use_cluster
            # Per-unit IF scores c_e(i)·ê_i: treated obs k∈T_e contribute (1/N_e)(τ_k−τ̂(e))
            # to unit treated_g[k]; untreated obs o contribute −(x_o'M⁺W_e)·ê0_o to unit
            # control_g[o]. Summing within units gives the clustered score column Ψ[:,ei].
            u_e = Mpinv * We
            proj = X0 * u_e
            for k in Te
                Psi[unit_idx[treated_g[k]], ei] += (tau[k] - att_agg[ei]) / Ne
            end
            for o in 1:n_ctrl
                Psi[unit_idx[control_g[o]], ei] -= proj[o] * resid0[o]
            end
        end
    end

    if use_cluster
        att_vcov = Psi' * Psi
    else
        # Homoskedastic Prop.-6: Cov(τ̂(e),τ̂(e')) = σ̂² W_e'M⁺W_{e'}, plus σ̂²/N_e on the
        # diagonal from the disjoint treated errors ⇒ Var(τ̂(e)) = σ̂²(1/N_e + W_e'M⁺W_e).
        att_vcov = sigma2 .* (We_mat' * (Mpinv * We_mat))
        for ei in 1:E
            Ne_vec[ei] > 0 && (att_vcov[ei, ei] += sigma2 / Ne_vec[ei])
        end
    end
    att_vcov = (att_vcov .+ att_vcov') ./ 2         # symmetrize FP asymmetry
    se_agg = sqrt.(max.(diag(att_vcov), zero(T)))   # diagonal = the Prop-6 per-horizon SEs

    # =========================================================================
    # Step 6: CIs and overall ATT
    # =========================================================================

    z = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    ci_lower = att_agg .- z .* se_agg
    ci_upper = att_agg .+ z .* se_agg

    # Overall ATT: equal-weighted mean of post-treatment (e >= 0) ATTs with nonzero SE.
    # SE via the full IF covariance sub-block: sqrt(w'V_post w), w = 1/n_post — the old
    # sqrt(sum(se^2))/n_post assumed independence across horizons and understated it.
    post_idx = findall(>=(0), event_times_all)
    valid_post = post_idx[se_agg[post_idx] .> 0]
    if !isempty(valid_post)
        n_post = length(valid_post)
        overall_att = mean(att_agg[valid_post])
        w = fill(one(T) / n_post, n_post)
        Vp = att_vcov[valid_post, valid_post]
        overall_se = sqrt(max(dot(w, Vp * w), zero(T)))
    else
        overall_att = zero(T)
        overall_se = zero(T)
    end

    DIDResult{T}(att_agg, se_agg, ci_lower, ci_upper, event_times_all, reference_period,
                 group_time_att, cohorts, overall_att, overall_se,
                 pd.T_obs, pd.n_groups, n_treated, n_control,
                 :bjs, pd.varnames[outcome_col], pd.varnames[treat_col],
                 control_group, cluster, T(conf_level), att_vcov)
end
