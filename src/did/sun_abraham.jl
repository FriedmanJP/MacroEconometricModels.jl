# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Sun & Abraham (2021) interaction-weighted estimator for staggered DiD.

Estimates cohort-specific effects δ_{e,l} = CATT(cohort e, relative period l) from ONE
joint saturated TWFE regression (cohort × relative-period interactions, never-treated as
the clean control), then aggregates to event-time ATTs with period-specific cohort-share
weights and reports the requested window. Because the regression is joint, cohort
coefficients carry a proper cross-cohort covariance for the aggregation SEs.

# Reference
Sun, L. & Abraham, S. (2021). *JoE* 225(2), 175-199.
"""

using LinearAlgebra, Statistics, Distributions

"""
    _estimate_sun_abraham(pd::PanelData{T}, outcome_col::Int, treat_col::Int;
                          leads::Int=0, horizon::Int=5,
                          control_group::Symbol=:never_treated,
                          cluster::Symbol=:unit,
                          conf_level::Real=0.95) where {T<:AbstractFloat}

Internal Sun & Abraham (2021) interaction-weighted estimator.

# Algorithm
1. Build ONE saturated design: cohort × relative-period dummies for every treated cohort
   (all l in its support, l ≠ -1); never-treated units are pure zero-row controls.
2. Double-demean (unit + time FE) → OLS → joint δ̂_{e,l} and one clustered covariance V.
3. Period-specific weights: for relative period l, w_{e,l} = n_e / Σ_{e' present at l} n_{e'}
   over the cohorts observed at l (cohorts absent at l get weight 0).
4. Aggregate: ATT(l) = Σ_e w_{e,l} δ̂_{e,l} = (Ω β̂)_l, with covariance att_vcov = Ω V Ω'.
5. Report the requested [-leads, horizon] window.

# Returns
`DIDResult{T}` with interaction-weighted event-time ATTs.

# Reference
Sun, L. & Abraham, S. (2021). *JoE* 225(2), 175-199.
"""
function _estimate_sun_abraham(pd::PanelData{T}, outcome_col::Int, treat_col::Int;
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

    # Identify cohorts and control groups
    cohorts = sort(unique([t for (_, t) in timing if t > 0]))
    n_cohorts = length(cohorts)

    never_treated = Set(g for (g, t) in timing if t == 0)
    if control_group == :never_treated
        isempty(never_treated) &&
            throw(ArgumentError("No never-treated units found. Use control_group=:not_yet_treated"))
    end

    treated_groups = [g for (g, t) in timing if t > 0]
    n_treated = length(treated_groups)
    n_control = length(never_treated)
    cohort_size = Dict(e => count(g -> timing[g] == e, keys(timing)) for e in cohorts)

    reference_period = -1
    event_times_all = collect(-leads:horizon)
    n_evt_all = length(event_times_all)

    # -----------------------------------------------------------------
    # ONE joint saturated interaction-weighted regression (Sun-Abraham 2021):
    #   Y_it = α_i + λ_t + Σ_e Σ_{l≠-1} δ_{e,l}·1{cohort_i=e}·1{t-e=l} + ε_it
    # Every treated cohort e gets its OWN relative-period dummies for all l in its support
    # (l≠-1 reference); never-treated units are pure zero-row controls that anchor λ_t.
    # Because it is a SINGLE regression the cohort coefficients δ̂_{e,l} are jointly
    # estimated, so their cross-cohort covariance (needed for correct aggregation SEs)
    # exists — impossible with the previous per-cohort separate fits. When there are no
    # never-treated units, the last-treated cohort is omitted as the reference control.
    # -----------------------------------------------------------------
    ref_cohort = 0
    if isempty(never_treated) && control_group == :not_yet_treated && !isempty(cohorts)
        ref_cohort = maximum(cohorts)
    end

    col_key = Tuple{Int,Int}[]                 # column k ↔ (cohort e, relative period l)
    col_of = Dict{Tuple{Int,Int}, Int}()
    for e in cohorts
        e == ref_cohort && continue
        for t in all_times
            l = t - e
            l == reference_period && continue
            key = (e, l)
            if !haskey(col_of, key)
                push!(col_key, key)
                col_of[key] = length(col_key)
            end
        end
    end
    K = length(col_key)

    N_obs = pd.T_obs
    y = pd.data[:, outcome_col]
    D = zeros(T, N_obs, K)
    @inbounds for i in 1:N_obs
        e = timing[pd.group_id[i]]
        (e == 0 || e == ref_cohort) && continue         # never-treated / reference: zero row
        l = pd.time_id[i] - e
        l == reference_period && continue
        k = get(col_of, (e, l), 0)
        k > 0 && (D[i, k] = one(T))
    end

    # Double-demean (unit + time FE) then OLS; ONE joint clustered covariance V (K×K).
    y_dm = _double_demean(y, pd.group_id, pd.time_id)
    D_dm = _double_demean_matrix(D, pd.group_id, pd.time_id)
    beta = robust_inv(D_dm' * D_dm; silent=true) * (D_dm' * y_dm)
    resid = y_dm - D_dm * beta
    V = _did_vcov(D_dm, resid, pd.group_id, pd.time_id, cluster)

    # -----------------------------------------------------------------
    # Aggregate to event-time ATTs with PERIOD-SPECIFIC cohort-share weights.
    # For relative period l, only cohorts observed at l (e+l ∈ all_times ⇔ column exists)
    # get weight; the weight is that cohort's size renormalized over the PRESENT cohorts.
    # att = Ω·β̂ and att_vcov = Ω V Ω' carry the full cross-cohort/cross-horizon covariance.
    # -----------------------------------------------------------------
    att = zeros(T, n_evt_all)
    Omega = zeros(T, n_evt_all, K)
    for (ri, l) in enumerate(event_times_all)
        l == reference_period && continue
        present = [(e, col_of[(e, l)]) for e in cohorts if haskey(col_of, (e, l))]
        isempty(present) && continue
        sizes = T[cohort_size[e] for (e, _) in present]
        w = sizes ./ sum(sizes)
        for (m, (_, k)) in enumerate(present)
            att[ri] += w[m] * beta[k]
            Omega[ri, k] = w[m]
        end
    end

    att_vcov = Omega * V * Omega'
    att_vcov = (att_vcov .+ att_vcov') ./ 2
    se = sqrt.(max.(diag(att_vcov), zero(T)))

    # Group-time (cohort × event-time) coefficient matrix for output.
    group_time_att = zeros(T, n_cohorts, n_evt_all)
    for (ci, e) in enumerate(cohorts)
        for (ri, l) in enumerate(event_times_all)
            k = get(col_of, (e, l), 0)
            k > 0 && (group_time_att[ci, ri] = beta[k])
        end
    end

    # Confidence intervals
    z = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    ci_lower = att .- z .* se
    ci_upper = att .+ z .* se

    # Overall ATT: equal-weighted mean of post ATTs with nonzero SE; SE via the full
    # covariance sub-block sqrt(w'V_post w), w = 1/n_post (T068).
    post_idx = findall(>=(0), event_times_all)
    valid_post = post_idx[se[post_idx] .> 0]
    if !isempty(valid_post)
        n_post = length(valid_post)
        overall_att = mean(att[valid_post])
        wv = fill(one(T) / n_post, n_post)
        Vp = att_vcov[valid_post, valid_post]
        overall_se = sqrt(max(dot(wv, Vp * wv), zero(T)))
    else
        overall_att = zero(T)
        overall_se = zero(T)
    end

    DIDResult{T}(att, se, ci_lower, ci_upper, event_times_all, reference_period,
                 group_time_att, cohorts, overall_att, overall_se,
                 pd.T_obs, pd.n_groups, n_treated, n_control,
                 :sun_abraham, pd.varnames[outcome_col], pd.varnames[treat_col],
                 control_group, cluster, T(conf_level), att_vcov)
end
