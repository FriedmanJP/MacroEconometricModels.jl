# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Heteroskedasticity-consistent and cluster-robust covariance estimators for
cross-sectional regression models.

Implements HC0 (White 1980), HC1, HC2, HC3 (MacKinnon & White 1985),
and cluster-robust (Arellano 1987) covariance matrices.
"""

using LinearAlgebra

# =============================================================================
# Hat Matrix Diagonal
# =============================================================================

"""
    _hat_diag(X::Matrix{T}, XtXinv::Matrix{T}) -> Vector{T}

Compute the diagonal of the hat (projection) matrix H = X (X'X)^{-1} X'.

Element h_ii measures the leverage of observation i. Used by HC2 and HC3.
"""
function _hat_diag(X::Matrix{T}, XtXinv::Matrix{T}) where {T<:AbstractFloat}
    n = size(X, 1)
    h = Vector{T}(undef, n)
    @inbounds for i in 1:n
        xi = @view X[i, :]
        h[i] = dot(xi, XtXinv * xi)
        # Clamp to [0, 1) for numerical safety
        h[i] = clamp(h[i], zero(T), one(T) - T(1e-10))
    end
    h
end

# =============================================================================
# Cluster-Robust Covariance
# =============================================================================

"""
    _cluster_vcov(X, resid, XtXinv, clusters) -> Matrix{T}

Cluster-robust (Arellano 1987) covariance estimator.

V = (X'X)^{-1} B (X'X)^{-1}, where B = sum_g (X_g' e_g)(X_g' e_g)'.
Includes the standard G/(G-1) * n/(n-k) finite-sample correction.

# Arguments
- `X::Matrix{T}` — regressor matrix (n x k)
- `resid::Vector{T}` — residuals (n)
- `XtXinv::Matrix{T}` — (X'X)^{-1}
- `clusters::Vector` — cluster assignment for each observation

# References
- Arellano, M. (1987). *Oxford Bulletin of Economics and Statistics* 49(4), 431-434.
- Cameron, A. C. & Miller, D. L. (2015). *JPE* 50, 327-372.
"""
function _cluster_vcov(X::Matrix{T}, resid::Vector{T},
                       XtXinv::Matrix{T}, clusters::AbstractVector) where {T<:AbstractFloat}
    n, k = size(X)
    length(clusters) == n || throw(ArgumentError("clusters must have length n=$n"))

    unique_clusters = unique(clusters)
    G = length(unique_clusters)
    G < 2 && throw(ArgumentError("Need at least 2 clusters for cluster-robust SE"))

    # B = sum_g (X_g' e_g)(X_g' e_g)'
    B = zeros(T, k, k)
    for g in unique_clusters
        idx = findall(==(g), clusters)
        X_g = @view X[idx, :]
        e_g = @view resid[idx]
        score_g = X_g' * e_g   # k x 1
        B .+= score_g * score_g'
    end

    # Finite-sample correction: G/(G-1) * (n-1)/(n-k)
    correction = T(G) / T(G - 1) * T(n - 1) / T(n - k)
    B .*= correction

    XtXinv * B * XtXinv
end

# =============================================================================
# Main Covariance Dispatch
# =============================================================================

"""
    _reg_vcov(X, resid, cov_type, XtXinv; clusters=nothing) -> Matrix{T}

Compute the variance-covariance matrix of OLS/WLS coefficients.

# Supported covariance types
- `:ols` — classical homoskedastic: sigma^2 (X'X)^{-1}
- `:hc0` — White (1980): (X'X)^{-1} (sum e_i^2 x_i x_i') (X'X)^{-1}
- `:hc1` — HC0 with n/(n-k) finite-sample correction
- `:hc2` — HC0 with 1/(1-h_ii) leverage correction
- `:hc3` — HC0 with 1/(1-h_ii)^2 jackknife-like correction
- `:cluster` — cluster-robust (requires `clusters` argument)

# Arguments
- `X::Matrix{T}` — regressor matrix (n x k)
- `resid::Vector{T}` — residuals (n)
- `cov_type::Symbol` — covariance estimator type
- `XtXinv::Matrix{T}` — precomputed (X'X)^{-1}
- `clusters` — cluster assignments (required if `cov_type == :cluster`)

# References
- White, H. (1980). *Econometrica* 48(4), 817-838.
- MacKinnon, J. G. & White, H. (1985). *JBES* 3(3), 305-314.
"""
function _reg_vcov(X::Matrix{T}, resid::Vector{T}, cov_type::Symbol,
                   XtXinv::Matrix{T};
                   clusters::Union{Nothing,AbstractVector}=nothing) where {T<:AbstractFloat}
    n, k = size(X)

    if cov_type == :ols
        sigma2 = dot(resid, resid) / T(n - k)
        return sigma2 .* XtXinv
    end

    if cov_type == :cluster
        clusters === nothing && throw(ArgumentError("clusters required for :cluster cov_type"))
        return _cluster_vcov(X, resid, XtXinv, clusters)
    end

    # Heteroskedasticity-consistent estimators: V = (X'X)^{-1} S (X'X)^{-1}
    # where S = X' diag(omega_i) X
    cov_type in (:hc0, :hc1, :hc2, :hc3) ||
        throw(ArgumentError("cov_type must be :ols, :hc0, :hc1, :hc2, :hc3, or :cluster; got :$cov_type"))

    # Compute leverage if needed
    h = (cov_type == :hc2 || cov_type == :hc3) ? _hat_diag(X, XtXinv) : nothing

    # Build the meat matrix S = X' Omega X
    S = zeros(T, k, k)
    @inbounds for i in 1:n
        xi = @view X[i, :]
        ei = resid[i]

        omega_i = if cov_type == :hc0
            ei^2
        elseif cov_type == :hc1
            ei^2
        elseif cov_type == :hc2
            ei^2 / (one(T) - h[i])
        else  # :hc3
            ei^2 / (one(T) - h[i])^2
        end

        S .+= omega_i .* (xi * xi')
    end

    # HC1 finite-sample correction
    if cov_type == :hc1
        S .*= T(n) / T(n - k)
    end

    XtXinv * S * XtXinv
end
