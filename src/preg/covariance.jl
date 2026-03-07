# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Panel covariance estimators: entity-cluster, time-cluster, two-way cluster
(Cameron-Gelbach-Miller 2011), and Driscoll-Kraay (1998) HAC.
"""

using LinearAlgebra

# =============================================================================
# Entity Cluster-Robust Covariance
# =============================================================================

"""
    _panel_cluster_vcov(X, resid, XtXinv, groups) -> Matrix{T}

Entity cluster-robust variance-covariance estimator for panel data.

V = (X'X)^{-1} B (X'X)^{-1}, where B = sum_g (X_g' e_g)(X_g' e_g)'.
Includes G/(G-1) * (n-1)/(n-k) finite-sample correction.

# Arguments
- `X::Matrix{T}` — demeaned regressor matrix (n x k)
- `resid::Vector{T}` — residuals (n)
- `XtXinv::Matrix{T}` — (X'X)^{-1}
- `groups::AbstractVector{Int}` — group assignment for each observation

# References
- Arellano, M. (1987). *Oxford Bulletin of Economics and Statistics* 49(4), 431-434.
- Cameron, A. C. & Miller, D. L. (2015). *JPE* 50, 327-372.
"""
function _panel_cluster_vcov(X::Matrix{T}, resid::Vector{T},
                             XtXinv::Matrix{T}, groups::AbstractVector{Int}) where {T<:AbstractFloat}
    n, k = size(X)
    unique_groups = unique(groups)
    G = length(unique_groups)
    G < 2 && throw(ArgumentError("Need at least 2 clusters for cluster-robust SE"))

    B = zeros(T, k, k)
    for g in unique_groups
        idx = findall(==(g), groups)
        X_g = @view X[idx, :]
        e_g = @view resid[idx]
        score_g = X_g' * e_g
        B .+= score_g * score_g'
    end

    correction = T(G) / T(G - 1) * T(n - 1) / T(n - k)
    B .*= correction

    XtXinv * B * XtXinv
end

# =============================================================================
# Time Cluster-Robust Covariance
# =============================================================================

"""
    _panel_time_cluster_vcov(X, resid, XtXinv, time_ids) -> Matrix{T}

Time cluster-robust variance-covariance estimator for panel data.

Clusters observations by time period rather than entity.
Same formula as entity-cluster but grouping by time.

# Arguments
- `X::Matrix{T}` — demeaned regressor matrix (n x k)
- `resid::Vector{T}` — residuals (n)
- `XtXinv::Matrix{T}` — (X'X)^{-1}
- `time_ids::AbstractVector{Int}` — time period for each observation

# References
- Cameron, A. C. & Miller, D. L. (2015). *JPE* 50, 327-372.
"""
function _panel_time_cluster_vcov(X::Matrix{T}, resid::Vector{T},
                                  XtXinv::Matrix{T}, time_ids::AbstractVector{Int}) where {T<:AbstractFloat}
    n, k = size(X)
    unique_times = unique(time_ids)
    G = length(unique_times)
    G < 2 && throw(ArgumentError("Need at least 2 time clusters for time-cluster SE"))

    B = zeros(T, k, k)
    for t in unique_times
        idx = findall(==(t), time_ids)
        X_t = @view X[idx, :]
        e_t = @view resid[idx]
        score_t = X_t' * e_t
        B .+= score_t * score_t'
    end

    correction = T(G) / T(G - 1) * T(n - 1) / T(n - k)
    B .*= correction

    XtXinv * B * XtXinv
end

# =============================================================================
# Two-Way Cluster-Robust Covariance (Cameron-Gelbach-Miller 2011)
# =============================================================================

"""
    _panel_twoway_vcov(X, resid, XtXinv, groups, time_ids) -> Matrix{T}

Two-way cluster-robust variance-covariance estimator.

V = V_entity + V_time - V_entity_x_time

where V_entity_x_time clusters on the intersection of entity and time
(which, for typical panels with unique (i,t) pairs, equals the
HC1 heteroskedasticity-robust estimator).

# Arguments
- `X::Matrix{T}` — demeaned regressor matrix (n x k)
- `resid::Vector{T}` — residuals (n)
- `XtXinv::Matrix{T}` — (X'X)^{-1}
- `groups::AbstractVector{Int}` — entity group for each observation
- `time_ids::AbstractVector{Int}` — time period for each observation

# References
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
  *JBES* 29(2), 238-249.
"""
function _panel_twoway_vcov(X::Matrix{T}, resid::Vector{T},
                            XtXinv::Matrix{T}, groups::AbstractVector{Int},
                            time_ids::AbstractVector{Int}) where {T<:AbstractFloat}
    V_entity = _panel_cluster_vcov(X, resid, XtXinv, groups)
    V_time = _panel_time_cluster_vcov(X, resid, XtXinv, time_ids)

    # Intersection clustering: unique (entity, time) pairs
    # For balanced panels with unique (i,t), each obs is its own cluster -> HC1
    n, k = size(X)
    interaction = groups .* maximum(time_ids) .+ time_ids
    V_interaction = _panel_cluster_vcov(X, resid, XtXinv, interaction)

    V = V_entity .+ V_time .- V_interaction

    # Ensure positive semi-definiteness
    V = (V .+ V') ./ 2
    V
end

# =============================================================================
# Driscoll-Kraay (1998) HAC Covariance
# =============================================================================

"""
    _panel_driscoll_kraay_vcov(X, resid, XtXinv, groups, time_ids; bandwidth=nothing) -> Matrix{T}

Driscoll-Kraay (1998) heteroskedasticity- and autocorrelation-consistent
covariance estimator for panel data.

Averages moment conditions cross-sectionally at each time period, then
applies Newey-West HAC kernel to the resulting time series of averages.

# Arguments
- `X::Matrix{T}` — demeaned regressor matrix (n x k)
- `resid::Vector{T}` — residuals (n)
- `XtXinv::Matrix{T}` — (X'X)^{-1}
- `groups::AbstractVector{Int}` — entity group for each observation
- `time_ids::AbstractVector{Int}` — time period for each observation
- `bandwidth::Union{Nothing,Int}` — Newey-West bandwidth (default: floor(T^(1/4)))

# References
- Driscoll, J. C. & Kraay, A. C. (1998). *Review of Economics and Statistics*
  80(4), 549-560.
"""
function _panel_driscoll_kraay_vcov(X::Matrix{T}, resid::Vector{T},
                                    XtXinv::Matrix{T}, groups::AbstractVector{Int},
                                    time_ids::AbstractVector{Int};
                                    bandwidth::Union{Nothing,Int}=nothing) where {T<:AbstractFloat}
    n, k = size(X)
    unique_times = sort(unique(time_ids))
    n_times = length(unique_times)

    # Default bandwidth: floor(T^(1/4))
    bw = something(bandwidth, max(1, floor(Int, n_times^(1/4))))

    # Cross-sectional average of moment conditions at each time period
    # h_t = (1/N) sum_i x_{it} e_{it}  (but we use sum, not average, for sandwich)
    H = zeros(T, n_times, k)
    for (j, t) in enumerate(unique_times)
        idx = findall(==(t), time_ids)
        X_t = @view X[idx, :]
        e_t = @view resid[idx]
        # Sum of x_{it} * e_{it} at time t
        for i in eachindex(idx)
            for p in 1:k
                H[j, p] += X_t[i, p] * e_t[i]
            end
        end
    end

    # Newey-West HAC on the time series of moment sums
    S = zeros(T, k, k)

    # Lag 0: S_0 = sum_t h_t h_t'
    for j in 1:n_times
        h_t = @view H[j, :]
        S .+= h_t * h_t'
    end

    # Lags 1..bw: Bartlett kernel w(l) = 1 - l/(bw+1)
    for l in 1:bw
        weight = one(T) - T(l) / T(bw + 1)
        Gamma_l = zeros(T, k, k)
        for j in (l+1):n_times
            h_t = @view H[j, :]
            h_lag = @view H[j-l, :]
            Gamma_l .+= h_t * h_lag'
        end
        S .+= weight .* (Gamma_l .+ Gamma_l')
    end

    XtXinv * S * XtXinv
end

# =============================================================================
# Panel Covariance Dispatch
# =============================================================================

"""
    _panel_vcov(X, resid, XtXinv, groups, time_ids, cov_type; bandwidth=nothing) -> Matrix{T}

Dispatch to the appropriate panel covariance estimator.

# Supported types
- `:ols` — classical homoskedastic: sigma^2 (X'X)^{-1}
- `:cluster` — entity cluster-robust (default)
- `:twoway` — two-way cluster (Cameron-Gelbach-Miller 2011)
- `:driscoll_kraay` — Driscoll-Kraay (1998) HAC
"""
function _panel_vcov(X::Matrix{T}, resid::Vector{T}, XtXinv::Matrix{T},
                     groups::AbstractVector{Int}, time_ids::AbstractVector{Int},
                     cov_type::Symbol;
                     bandwidth::Union{Nothing,Int}=nothing) where {T<:AbstractFloat}
    n, k = size(X)

    if cov_type == :ols
        sigma2 = dot(resid, resid) / T(n - k)
        return sigma2 .* XtXinv
    elseif cov_type == :cluster
        return _panel_cluster_vcov(X, resid, XtXinv, groups)
    elseif cov_type == :twoway
        return _panel_twoway_vcov(X, resid, XtXinv, groups, time_ids)
    elseif cov_type == :driscoll_kraay
        return _panel_driscoll_kraay_vcov(X, resid, XtXinv, groups, time_ids;
                                          bandwidth=bandwidth)
    else
        throw(ArgumentError("cov_type must be :ols, :cluster, :twoway, or :driscoll_kraay; got :$cov_type"))
    end
end
