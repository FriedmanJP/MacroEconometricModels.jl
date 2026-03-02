# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

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

    for iter in 1:100
        max_change = zero(T)

        # Update time FE: gamma_t = mean(y - alpha_g) for control obs at t
        for t in ctrl_times_unique
            sum_val = zero(T)
            count_val = 0
            for k in 1:n_ctrl
                if control_t[k] == t
                    sum_val += control_y[k] - get(unit_fe, control_g[k], zero(T))
                    count_val += 1
                end
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
            for k in 1:n_ctrl
                if control_g[k] == g
                    sum_val += control_y[k] - get(time_fe, control_t[k], zero(T))
                    count_val += 1
                end
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

    for (ci, g_time) in enumerate(cohorts)
        for (ti, t) in enumerate(all_times)
            # Collect tau for this (cohort, time) cell
            tau_cell = T[]
            for k in 1:n_treat
                if treated_cohort[k] == g_time && treated_t[k] == t
                    push!(tau_cell, tau[k])
                end
            end
            if !isempty(tau_cell)
                group_time_att[ci, ti] = mean(tau_cell)
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

    for (ei, e) in enumerate(event_times_all)
        if e == reference_period
            att_agg[ei] = zero(T)
            se_agg[ei] = zero(T)
            continue
        end

        # Collect per-cohort contributions
        att_vals = T[]
        var_vals = T[]
        weights_e = T[]

        for (ci, g_time) in enumerate(cohorts)
            t_target = g_time + e
            ti = findfirst(==(t_target), all_times)
            ti === nothing && continue
            isnan(group_time_att[ci, ti]) && continue

            # Collect individual tau for this cohort at event-time e
            tau_ge = T[]
            for k in 1:n_treat
                if treated_cohort[k] == g_time && treated_t[k] == t_target
                    push!(tau_ge, tau[k])
                end
            end
            isempty(tau_ge) && continue

            n_g = T(length(tau_ge))

            push!(att_vals, mean(tau_ge))
            # Variance of cohort-level ATT at this event-time
            # Guard single-observation variance: use zero when n=1
            v_g = length(tau_ge) > 1 ? var(tau_ge) / n_g : zero(T)
            push!(var_vals, v_g)
            push!(weights_e, n_g)
        end

        if !isempty(att_vals)
            w_total = sum(weights_e)
            w_norm = weights_e ./ w_total
            att_agg[ei] = sum(w_norm .* att_vals)
            # SE: sqrt(sum w_g^2 * var(tau_g(e)) / n_g)
            # var_vals already contains var(tau_g) / n_g
            se_agg[ei] = sqrt(sum((w_norm .^ 2) .* var_vals))
        end
    end

    # =========================================================================
    # Step 6: CIs and overall ATT
    # =========================================================================

    z = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    ci_lower = att_agg .- z .* se_agg
    ci_upper = att_agg .+ z .* se_agg

    # Overall ATT: average of post-treatment (e >= 0) ATTs with nonzero SEs
    post_mask = event_times_all .>= 0
    post_att = att_agg[post_mask]
    post_se = se_agg[post_mask]
    nonzero_post = post_se .> 0
    if any(nonzero_post)
        n_post = count(nonzero_post)
        overall_att = mean(post_att[nonzero_post])
        overall_se = sqrt(sum(post_se[nonzero_post] .^ 2)) / n_post
    else
        overall_att = zero(T)
        overall_se = zero(T)
    end

    DIDResult{T}(att_agg, se_agg, ci_lower, ci_upper, event_times_all, reference_period,
                 group_time_att, cohorts, overall_att, overall_se,
                 pd.T_obs, pd.n_groups, n_treated, n_control,
                 :bjs, pd.varnames[outcome_col], pd.varnames[treat_col],
                 control_group, cluster, T(conf_level))
end
