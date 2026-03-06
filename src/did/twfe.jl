# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Two-Way Fixed Effects (TWFE) Difference-in-Differences estimator.

Estimates event-study regression:
    Y_{it} = alpha_i + gamma_t + sum_k beta_k * 1{t - g_i = k} + X_{it}'delta + epsilon_{it}

where g_i is the treatment adoption time for unit i.
"""

using LinearAlgebra, Statistics, Distributions

"""
    _resolve_varindex(pd::PanelData, name::Union{String, Symbol}) -> Int

Find column index for a variable name in PanelData.
"""
function _resolve_varindex(pd::PanelData, name::Union{String, Symbol})
    s = string(name)
    idx = findfirst(==(s), pd.varnames)
    idx === nothing && throw(ArgumentError("Variable '$s' not found. Available: $(pd.varnames)"))
    idx
end

"""
    _extract_treatment_timing(pd::PanelData{T}, treat_col::Int) -> Dict{Int, Int}

Extract treatment timing from a column of PanelData.

The treatment column should contain: the time period when treatment starts for each
unit, or a sentinel value (0, -1, or NaN/Inf) for never-treated units.

Returns a Dict mapping group_id -> treatment_time (0 for never-treated).
"""
function _extract_treatment_timing(pd::PanelData{T}, treat_col::Int) where {T<:AbstractFloat}
    timing = Dict{Int, Int}()
    for g in 1:pd.n_groups
        mask = pd.group_id .== g
        vals = pd.data[mask, treat_col]
        # Use the first non-NaN value as treatment time
        treat_time = 0
        for v in vals
            if !isnan(v) && !isinf(v) && v > 0
                treat_time = round(Int, v)
                break
            end
        end
        timing[g] = treat_time
    end
    timing
end

"""
    _estimate_twfe(pd::PanelData{T}, outcome_col::Int, treat_col::Int;
                   leads::Int=0, horizon::Int=5, covariate_cols::Vector{Int}=Int[],
                   control_group::Symbol=:never_treated,
                   cluster::Symbol=:unit, conf_level::Real=0.95) where {T}

Internal TWFE event-study regression.

Algorithm:
1. Extract treatment timing per unit
2. Construct event-time dummies (excluding reference period = -1)
3. Apply double-demeaning (unit + time FE) via Frisch-Waugh
4. OLS with clustered SEs
"""
function _estimate_twfe(pd::PanelData{T}, outcome_col::Int, treat_col::Int;
                        leads::Int=0, horizon::Int=5,
                        covariate_cols::Vector{Int}=Int[],
                        control_group::Symbol=:never_treated,
                        cluster::Symbol=:unit,
                        conf_level::Real=0.95) where {T<:AbstractFloat}
    N_obs = pd.T_obs
    timing = _extract_treatment_timing(pd, treat_col)

    # Override timing with cohort_id if present
    if pd.cohort_id !== nothing
        for g in 1:pd.n_groups
            mask = pd.group_id .== g
            timing[g] = pd.cohort_id[findfirst(mask)]
        end
    end

    # Identify treated/control groups
    treated_groups = [g for (g, t) in timing if t > 0]
    control_groups = [g for (g, t) in timing if t == 0]
    n_treated = length(treated_groups)
    n_control = length(control_groups)

    # Event-time grid: [-leads, ..., -1, 0, 1, ..., horizon]
    # Reference period = -1 (omitted)
    reference_period = -1
    event_times_all = collect(-leads:horizon)
    event_times_est = filter(!=(reference_period), event_times_all)
    n_dummies = length(event_times_est)

    # Build regression matrices
    # Construct event-time dummies for each observation
    D = zeros(T, N_obs, n_dummies)
    y = pd.data[:, outcome_col]

    for i in 1:N_obs
        g = pd.group_id[i]
        t = pd.time_id[i]
        g_time = timing[g]
        if g_time > 0  # treated unit
            evt = t - g_time
            for (j, e) in enumerate(event_times_est)
                if evt == e
                    D[i, j] = one(T)
                end
            end
        end
    end

    # Covariates
    n_cov = length(covariate_cols)
    X_cov = n_cov > 0 ? pd.data[:, covariate_cols] : zeros(T, N_obs, 0)

    # Full regressor matrix: [event_time_dummies, covariates]
    X_raw = hcat(D, X_cov)

    # Double-demeaning (Frisch-Waugh for unit + time FE)
    y_dm = _double_demean(y, pd.group_id, pd.time_id)
    X_dm = _double_demean_matrix(X_raw, pd.group_id, pd.time_id)

    # OLS
    XtX_inv = robust_inv(X_dm' * X_dm; silent=true)
    beta = XtX_inv * (X_dm' * y_dm)
    resid = y_dm - X_dm * beta

    # Clustered SEs
    V = _did_vcov(X_dm, resid, pd.group_id, pd.time_id, cluster)
    se_all = sqrt.(max.(diag(V), zero(T)))

    # Extract event-time coefficients
    att_est = beta[1:n_dummies]
    se_est = se_all[1:n_dummies]

    # Insert zero for reference period
    att = zeros(T, length(event_times_all))
    se = zeros(T, length(event_times_all))
    for (j, e) in enumerate(event_times_est)
        idx = findfirst(==(e), event_times_all)
        att[idx] = att_est[j]
        se[idx] = se_est[j]
    end

    # CIs
    z = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    ci_lower = att .- z .* se
    ci_upper = att .+ z .* se

    # Overall ATT: average of post-treatment coefficients
    post_mask = event_times_all .>= 0
    post_att = att[post_mask]
    post_se = se[post_mask]
    n_post = count(post_mask)
    overall_att = mean(post_att)
    # SE of average: sqrt(sum(se^2)/K^2) assuming independence across horizons
    overall_se = sqrt(sum(post_se.^2)) / n_post

    DIDResult{T}(att, se, ci_lower, ci_upper, event_times_all, reference_period,
                 nothing, nothing, overall_att, overall_se,
                 N_obs, pd.n_groups, n_treated, n_control,
                 :twfe, pd.varnames[outcome_col], pd.varnames[treat_col],
                 control_group, cluster, T(conf_level))
end

# =============================================================================
# Double-demeaning helpers
# =============================================================================

"""
    _double_demean(y::Vector{T}, group_id::Vector{Int}, time_id::Vector{Int}) where {T}

Remove unit and time fixed effects via iterative demeaning (Frisch-Waugh).
Converges in a few iterations for balanced panels, more for unbalanced.
"""
function _double_demean(y::Vector{T}, group_id::Vector{Int},
                        time_id::Vector{Int}; max_iter::Int=100,
                        tol::Real=1e-10) where {T<:AbstractFloat}
    y_dm = copy(y)
    groups = unique(group_id)
    times = unique(time_id)

    for iter in 1:max_iter
        y_prev = copy(y_dm)

        # Demean by unit
        for g in groups
            mask = group_id .== g
            y_dm[mask] .-= mean(y_dm[mask])
        end

        # Demean by time
        for t in times
            mask = time_id .== t
            y_dm[mask] .-= mean(y_dm[mask])
        end

        # Check convergence
        if maximum(abs.(y_dm .- y_prev)) < tol
            break
        end
        if iter == max_iter
            @warn "Double-demeaning did not converge in $max_iter iterations" maxlog=1
        end
    end
    y_dm
end

"""
    _double_demean_matrix(X::Matrix{T}, group_id::Vector{Int},
                          time_id::Vector{Int}) where {T}

Apply double-demeaning to each column of X.
"""
function _double_demean_matrix(X::Matrix{T}, group_id::Vector{Int},
                               time_id::Vector{Int}) where {T<:AbstractFloat}
    K = size(X, 2)
    X_dm = similar(X)
    for k in 1:K
        X_dm[:, k] = _double_demean(X[:, k], group_id, time_id)
    end
    X_dm
end
