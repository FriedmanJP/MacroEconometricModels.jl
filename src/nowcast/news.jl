# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <wookyung9207@gmail.com>
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
News decomposition for DFM nowcasting (Bańbura & Modugno 2014).

Decomposes the revision of a nowcast into contributions from new data
releases (news), data revisions, and parameter re-estimation.
"""

# =============================================================================
# Public API
# =============================================================================

"""
    nowcast_news(X_new, X_old, model::NowcastDFM, target_period;
                 target_var=size(X_new,2), groups=nothing) -> NowcastNews{T}

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

# Returns
`NowcastNews{T}` with per-release impacts and total decomposition.

# References
- Bańbura, M. & Modugno, M. (2014). Maximum Likelihood Estimation of Factor
  Models on Datasets with Arbitrary Pattern of Missing Data.
"""
function nowcast_news(X_new::AbstractMatrix, X_old::AbstractMatrix,
                      model::NowcastDFM{T}, target_period::Int;
                      target_var::Int=size(X_new, 2),
                      groups::Union{Vector{Int},Nothing}=nothing) where {T<:AbstractFloat}
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

    x_smooth_old, P_smooth_old, _, _ = _kalman_smoother_missing(y_old, A, C, Q, R, Z_0, V_0)
    x_smooth_new, P_smooth_new, _, _ = _kalman_smoother_missing(y_new, A, C, Q, R, Z_0, V_0)

    # Old and new nowcasts (unstandardized)
    now_old = dot(C[target_var, :], x_smooth_old[:, target_period]) * model.Wx[target_var] + model.Mx[target_var]
    now_new = dot(C[target_var, :], x_smooth_new[:, target_period]) * model.Wx[target_var] + model.Mx[target_var]

    # Compute news impacts
    n_releases = length(i_new)
    impact_news = zeros(T, n_releases)
    variable_names = String[]

    if n_releases > 0
        # For each new release, compute its weight and innovation
        for (k, idx) in enumerate(i_new)
            t_k = idx[1]  # time of release
            v_k = idx[2]  # variable of release

            push!(variable_names, "Var$(v_k)_t$(t_k)")

            # Innovation: actual - forecast
            forecast_k = dot(C[v_k, :], x_smooth_old[:, t_k])
            actual_k = x_new[t_k, v_k]
            innovation_k = actual_k - forecast_k

            # Weight: proportional to cross-covariance between target and release
            # Simplified: use Kalman gain approximation
            cov_target_release = C[target_var, :]' * P_smooth_old[:, :, t_k] * C[v_k, :]
            var_release = C[v_k, :]' * P_smooth_old[:, :, t_k] * C[v_k, :] + R[v_k, v_k]
            weight = cov_target_release / max(var_release, T(1e-10))

            impact_news[k] = weight * innovation_k * model.Wx[target_var]
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
    else
        # Default: one group per variable
        group_impacts = zeros(T, N)
        for (k, idx) in enumerate(i_new)
            v_k = idx[2]
            group_impacts[v_k] += impact_news[k]
        end
    end

    NowcastNews{T}(now_old, now_new, impact_news, impact_revision,
                   impact_reestimation, group_impacts, variable_names)
end
