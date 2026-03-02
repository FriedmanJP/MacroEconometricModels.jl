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
Clustered standard error computation for panel DiD estimators.

Supports unit-level, time-level, and two-way (Cameron-Gelbach-Miller 2011)
clustering. Used by TWFE, Event Study LP, and LP-DiD.
"""

using LinearAlgebra

# =============================================================================
# Clustered variance-covariance
# =============================================================================

"""
    _cluster_vcov(X::Matrix{T}, resid::Vector{T}, cluster_ids::Vector{Int};
                  type::Symbol=:unit) where {T}

Compute cluster-robust variance-covariance matrix.

Implements the sandwich estimator V = (X'X)^{-1} M (X'X)^{-1} where
M = sum_g (X_g' u_g)(X_g' u_g)' with small-sample correction G/(G-1) * (N-1)/(N-K).

# Arguments
- `X`: N x K regressor matrix
- `resid`: N x 1 residual vector
- `cluster_ids`: N x 1 integer cluster membership

# Returns
K x K cluster-robust VCV matrix.
"""
function _cluster_vcov(X::Matrix{T}, resid::Vector{T},
                       cluster_ids::Vector{Int}) where {T<:AbstractFloat}
    N, K = size(X)
    clusters = unique(cluster_ids)
    G = length(clusters)

    XtX_inv = robust_inv(X' * X)

    # Meat: sum of outer products of cluster scores
    meat = zeros(T, K, K)
    for g in clusters
        mask = cluster_ids .== g
        X_g = X[mask, :]
        u_g = resid[mask]
        score_g = X_g' * u_g  # K x 1
        meat .+= score_g * score_g'
    end

    # Small-sample correction: G/(G-1) * (N-1)/(N-K)
    correction = T(G) / T(G - 1) * T(N - 1) / T(N - K)

    Matrix{T}(XtX_inv * (correction .* meat) * XtX_inv)
end

"""
    _twoway_cluster_vcov(X::Matrix{T}, resid::Vector{T},
                         unit_ids::Vector{Int}, time_ids::Vector{Int}) where {T}

Two-way cluster-robust VCV via Cameron-Gelbach-Miller (2011).

V_twoway = V_unit + V_time - V_het

where V_het is the heteroskedasticity-robust (HC1) estimator.
"""
function _twoway_cluster_vcov(X::Matrix{T}, resid::Vector{T},
                              unit_ids::Vector{Int},
                              time_ids::Vector{Int}) where {T<:AbstractFloat}
    V_unit = _cluster_vcov(X, resid, unit_ids)
    V_time = _cluster_vcov(X, resid, time_ids)

    # HC1 (observation-level "clustering")
    N, K = size(X)
    obs_ids = collect(1:N)
    V_het = _cluster_vcov(X, resid, obs_ids)

    V_twoway = V_unit .+ V_time .- V_het

    # Ensure positive semi-definiteness
    eigvals_v = eigvals(Symmetric(V_twoway))
    if any(eigvals_v .< 0)
        # Project onto PSD cone
        F = eigen(Symmetric(V_twoway))
        V_twoway = F.vectors * Diagonal(max.(F.values, zero(T))) * F.vectors'
    end

    Matrix{T}(V_twoway)
end

"""
    _did_vcov(X::Matrix{T}, resid::Vector{T}, unit_ids::Vector{Int},
              time_ids::Vector{Int}, cluster::Symbol) where {T}

Dispatch to appropriate clustering method.
"""
function _did_vcov(X::Matrix{T}, resid::Vector{T}, unit_ids::Vector{Int},
                   time_ids::Vector{Int}, cluster::Symbol) where {T<:AbstractFloat}
    if cluster == :unit
        _cluster_vcov(X, resid, unit_ids)
    elseif cluster == :time
        _cluster_vcov(X, resid, time_ids)
    elseif cluster == :twoway
        _twoway_cluster_vcov(X, resid, unit_ids, time_ids)
    else
        throw(ArgumentError("cluster must be :unit, :time, or :twoway, got :$cluster"))
    end
end
