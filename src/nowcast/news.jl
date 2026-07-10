# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
News decomposition for DFM nowcasting (Bańbura & Modugno 2014).

Decomposes the revision of a nowcast into contributions from new data
releases (news), data revisions, and parameter re-estimation.
"""

# =============================================================================
# Public API
# =============================================================================

"""
    nowcast_news(X_new, X_old, model::NowcastDFM, target_period;
                 target_var=size(X_new,2), groups=nothing,
                 group_names=nothing) -> NowcastNews{T}

Compute news decomposition between two data vintages.

Identifies new data releases (positions where `X_old` is NaN but `X_new` is
not), computes their individual impacts on the nowcast via Kalman gain weights,
and decomposes the total revision.

# Arguments
- `X_new::AbstractMatrix` — new data vintage (T_obs × N)
- `X_old::AbstractMatrix` — old data vintage (same size, more NaN)
- `model::NowcastDFM` — estimated DFM model
- `target_period::Int` — time period for which to compute nowcast

# Keyword Arguments
- `target_var::Int` — target variable index (default: last column)
- `groups::Union{Vector{Int},Nothing}` — group assignment per variable (for aggregation)
- `group_names::Union{Vector{String},Nothing}` — labels for each group (auto-generated if omitted)

# Returns
`NowcastNews{T}` with per-release impacts and total decomposition.

# References
- Bańbura, M. & Modugno, M. (2014). Maximum Likelihood Estimation of Factor
  Models on Datasets with Arbitrary Pattern of Missing Data.
"""
function nowcast_news(X_new::AbstractMatrix, X_old::AbstractMatrix,
                      model::NowcastDFM{T}, target_period::Int;
                      target_var::Int=size(X_new, 2),
                      groups::Union{Vector{Int},Nothing}=nothing,
                      group_names::Union{Vector{String},Nothing}=nothing) where {T<:AbstractFloat}
    T_obs, N = size(X_new)
    size(X_old) == (T_obs, N) || throw(ArgumentError("X_new and X_old must have same size"))
    1 <= target_period <= T_obs || throw(ArgumentError("target_period out of range"))
    1 <= target_var <= N || throw(ArgumentError("target_var out of range"))

    X_new_mat = Matrix{T}(X_new)
    X_old_mat = Matrix{T}(X_old)

    # Standardize with model parameters
    x_new = (X_new_mat .- model.Mx') ./ model.Wx'
    x_old = (X_old_mat .- model.Mx') ./ model.Wx'

    # Identify new data releases: positions where old is NaN but new is not
    i_new = findall((isnan.(X_old_mat)) .& (.!isnan.(X_new_mat)))

    # Run Kalman smoother on both vintages
    y_old = x_old'
    y_new = x_new'

    A, C, Q, R = model.A, model.C, model.Q, model.R
    Z_0, V_0 = model.Z_0, model.V_0
    state_dim = size(A, 1)

    # Old-vintage smoother WITH lagged cross-covariances (needed for the joint news system);
    # the new vintage only needs the smoothed target.
    release_times = [idx[1] for idx in i_new]
    all_times = vcat(target_period, release_times)
    kmax = max(min(maximum(all_times) - minimum(all_times), T_obs - 1), 1)
    x_smooth_old, P_smooth_old, Plag_old, _ = _kalman_smoother_lag(y_old, A, C, Q, R, Z_0, V_0, kmax)
    x_smooth_new, _, _, _ = _kalman_smoother_missing(y_new, A, C, Q, R, Z_0, V_0)

    # Old and new nowcasts (unstandardized)
    now_old = dot(C[target_var, :], x_smooth_old[:, target_period]) * model.Wx[target_var] + model.Mx[target_var]
    now_new = dot(C[target_var, :], x_smooth_new[:, target_period]) * model.Wx[target_var] + model.Mx[target_var]

    # Compute news impacts
    n_releases = length(i_new)
    impact_news = zeros(T, n_releases)
    variable_names = String[]

    if n_releases > 0
        # Joint news system (Bańbura–Modugno): with innovation vector I (new data minus its
        # old-vintage forecast), the smoothed-target revision equals B·I where the news weights
        # are B = Cov(F, I)·Var(I)^{-1}. This splits overlapping information across releases
        # correctly (order-invariant), unlike a per-release scalar Kalman gain.
        release_var = [idx[2] for idx in i_new]
        # Cov(state_ta, state_tb) from the lagged smoother covariances (Plag[j][:,:,t]=Cov(x_t,x_{t-j})).
        function _xcov(ta::Int, tb::Int)
            ta == tb && return P_smooth_old[:, :, ta]
            ta > tb && return Plag_old[ta - tb][:, :, ta]
            return permutedims(Plag_old[tb - ta][:, :, tb])
        end
        I_vec = zeros(T, n_releases)
        for k in 1:n_releases
            t_k, v_k = release_times[k], release_var[k]
            push!(variable_names, "Var$(v_k)_t$(t_k)")
            I_vec[k] = x_new[t_k, v_k] - dot(C[v_k, :], x_smooth_old[:, t_k])
        end
        VarI = zeros(T, n_releases, n_releases)
        for k in 1:n_releases, l in 1:n_releases
            t_k, v_k = release_times[k], release_var[k]
            t_l, v_l = release_times[l], release_var[l]
            VarI[k, l] = C[v_k, :]' * _xcov(t_k, t_l) * C[v_l, :]
            (t_k == t_l) && (VarI[k, l] += R[v_k, v_l])
        end
        CovFI = zeros(T, n_releases)
        for k in 1:n_releases
            t_k, v_k = release_times[k], release_var[k]
            CovFI[k] = C[target_var, :]' * _xcov(target_period, t_k) * C[v_k, :]
        end
        Bw = robust_inv(Symmetric(VarI)) * CovFI     # = Var(I)^{-1} Cov(I, F); news weight per release
        for k in 1:n_releases
            impact_news[k] = Bw[k] * I_vec[k] * model.Wx[target_var]
        end
    end

    # Total revision
    total_revision = now_new - now_old
    sum_news = sum(impact_news)
    impact_revision = zero(T)
    impact_reestimation = total_revision - sum_news - impact_revision

    # Group aggregation
    if groups !== nothing
        n_groups = maximum(groups)
        group_impacts = zeros(T, n_groups)
        for (k, idx) in enumerate(i_new)
            v_k = idx[2]
            if v_k <= length(groups)
                g = groups[v_k]
                group_impacts[g] += impact_news[k]
            end
        end
        if group_names !== nothing && length(group_names) != n_groups
            throw(ArgumentError("group_names length ($(length(group_names))) must match number of groups ($n_groups)"))
        end
        gn = group_names !== nothing ? group_names : ["Group $i" for i in 1:n_groups]
    else
        # Default: one group per variable
        group_impacts = zeros(T, N)
        for (k, idx) in enumerate(i_new)
            v_k = idx[2]
            group_impacts[v_k] += impact_news[k]
        end
        if group_names !== nothing && length(group_names) != N
            throw(ArgumentError("group_names length ($(length(group_names))) must match number of variables ($N)"))
        end
        gn = group_names !== nothing ? group_names : ["Var$i" for i in 1:N]
    end

    NowcastNews{T}(now_old, now_new, impact_news, impact_revision,
                   impact_reestimation, group_impacts, gn, variable_names)
end
