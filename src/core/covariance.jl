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
Robust Covariance Estimators for Time Series Regression.

This module provides heteroscedasticity and autocorrelation consistent (HAC) covariance
estimators commonly used in time series econometrics:

- Newey-West HAC estimator (Newey & West, 1987)
- White heteroscedasticity-robust estimator (HC0-HC3)
- Driscoll-Kraay estimator for panel data (Driscoll & Kraay, 1998)
- Long-run variance/covariance estimation

References:
- Newey, W. K., & West, K. D. (1987). A Simple, Positive Semi-definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix.
- Newey, W. K., & West, K. D. (1994). Automatic Lag Selection in Covariance Matrix Estimation.
- Driscoll, J. C., & Kraay, A. C. (1998). Consistent Covariance Matrix Estimation with
  Spatially Dependent Panel Data.
"""

using LinearAlgebra, Statistics

# =============================================================================
# Abstract Types
# =============================================================================

"""Abstract supertype for covariance estimators."""
abstract type AbstractCovarianceEstimator end

# =============================================================================
# Covariance Estimator Types
# =============================================================================

"""
    NeweyWestEstimator{T} <: AbstractCovarianceEstimator

Newey-West HAC covariance estimator configuration.

Fields:
- bandwidth: Truncation lag (0 = automatic via Newey-West 1994 formula)
- kernel: Kernel function (:bartlett, :parzen, :quadratic_spectral, :tukey_hanning)
- prewhiten: Use AR(1) prewhitening
"""
struct NeweyWestEstimator{T<:AbstractFloat} <: AbstractCovarianceEstimator
    bandwidth::Int
    kernel::Symbol
    prewhiten::Bool

    function NeweyWestEstimator{T}(bandwidth::Int=0, kernel::Symbol=:bartlett,
                                    prewhiten::Bool=false) where {T<:AbstractFloat}
        bandwidth < 0 && throw(ArgumentError("bandwidth must be non-negative"))
        kernel ∉ (:bartlett, :parzen, :quadratic_spectral, :tukey_hanning) &&
            throw(ArgumentError("kernel must be :bartlett, :parzen, :quadratic_spectral, or :tukey_hanning"))
        new{T}(bandwidth, kernel, prewhiten)
    end
end

NeweyWestEstimator(; bandwidth::Int=0, kernel::Symbol=:bartlett, prewhiten::Bool=false) =
    NeweyWestEstimator{Float64}(bandwidth, kernel, prewhiten)

"""
    WhiteEstimator <: AbstractCovarianceEstimator

White heteroscedasticity-robust covariance estimator (HC0).
Does not correct for serial correlation.
"""
struct WhiteEstimator <: AbstractCovarianceEstimator end

"""
    DriscollKraayEstimator{T} <: AbstractCovarianceEstimator

Driscoll-Kraay standard errors for panel data with cross-sectional dependence.
"""
struct DriscollKraayEstimator{T<:AbstractFloat} <: AbstractCovarianceEstimator
    bandwidth::Int
    kernel::Symbol

    function DriscollKraayEstimator{T}(bandwidth::Int=0, kernel::Symbol=:bartlett) where {T<:AbstractFloat}
        bandwidth < 0 && throw(ArgumentError("bandwidth must be non-negative"))
        new{T}(bandwidth, kernel)
    end
end

DriscollKraayEstimator(; bandwidth::Int=0, kernel::Symbol=:bartlett) =
    DriscollKraayEstimator{Float64}(bandwidth, kernel)

# =============================================================================
# Kernel Functions
# =============================================================================

"""
    kernel_weight(j::Int, bandwidth::Int, kernel::Symbol, ::Type{T}=Float64) -> T

Compute kernel weight for lag j given bandwidth and kernel type.

Kernels:
- :bartlett (Newey-West): w(x) = 1 - |x| for |x| ≤ 1
- :parzen: quartic spline kernel
- :quadratic_spectral (Andrews): optimal for Gaussian data
- :tukey_hanning: cosine kernel
"""
function kernel_weight(j::Int, bandwidth::Int, kernel::Symbol, ::Type{T}=Float64) where {T<:AbstractFloat}
    bandwidth == 0 && return zero(T)
    x = T(j) / T(bandwidth + 1)
    abs(x) > 1 && return zero(T)

    if kernel == :bartlett
        one(T) - abs(x)
    elseif kernel == :parzen
        ax = abs(x)
        ax <= 0.5 ? one(T) - 6ax^2 + 6ax^3 : 2(one(T) - ax)^3
    elseif kernel == :quadratic_spectral
        j == 0 && return one(T)
        z = 6π * x / 5
        25 / (12π^2 * x^2) * (sin(z) / z - cos(z))
    elseif kernel == :tukey_hanning
        (one(T) + cos(π * x)) / 2
    else
        throw(ArgumentError("Unknown kernel: $kernel"))
    end
end

# =============================================================================
# Automatic Bandwidth Selection
# =============================================================================

"""
    optimal_bandwidth_nw(residuals::AbstractVector{T}) -> Int

Compute optimal bandwidth using Newey-West (1994) automatic selection.
"""
function optimal_bandwidth_nw(residuals::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(residuals)
    n < 4 && return 0

    # Estimate AR(1) coefficient
    r_lag = @view residuals[1:end-1]
    r_lead = @view residuals[2:end]
    rho = dot(r_lag, r_lead) / dot(r_lag, r_lag)

    # Newey-West (1994) formula for Bartlett kernel
    rho_abs = min(abs(rho), T(0.99))
    alpha = 4rho_abs^2 / (1 - rho_abs)^4
    m = ceil(Int, 1.1447 * (alpha * n)^(1/3))
    min(m, floor(Int, n^(1/3)))
end

"""
    optimal_bandwidth_nw(residuals::AbstractMatrix{T}) -> Int

Multivariate version: average optimal bandwidth across columns.
"""
function optimal_bandwidth_nw(residuals::AbstractMatrix{T}) where {T<:AbstractFloat}
    n_vars = size(residuals, 2)
    n_vars == 0 && return 0
    round(Int, mean(optimal_bandwidth_nw(@view residuals[:, j]) for j in 1:n_vars))
end

# =============================================================================
# Newey-West HAC Estimator
# =============================================================================

"""
    newey_west(X::AbstractMatrix{T}, residuals::AbstractVector{T};
               bandwidth::Int=0, kernel::Symbol=:bartlett, prewhiten::Bool=false,
               XtX_inv::Union{Nothing,AbstractMatrix{T}}=nothing) -> Matrix{T}

Compute Newey-West HAC covariance matrix.

V_NW = (X'X)^{-1} S (X'X)^{-1}
where S = Γ₀ + Σⱼ₌₁ᵐ w(j) (Γⱼ + Γⱼ')

# Arguments
- `X`: Design matrix (n × k)
- `residuals`: Residuals vector (n × 1)
- `bandwidth`: Truncation lag (0 = automatic selection)
- `kernel`: Kernel function
- `prewhiten`: Use AR(1) prewhitening
- `XtX_inv`: Pre-computed (X'X)^{-1} for performance (optional)

# Returns
Robust covariance matrix (k × k)

# Performance
Pass `XtX_inv` when calling multiple times with the same X to avoid recomputation.
"""
function newey_west(X::AbstractMatrix{T}, residuals::AbstractVector{T};
                    bandwidth::Int=0, kernel::Symbol=:bartlett,
                    prewhiten::Bool=false,
                    XtX_inv::Union{Nothing,AbstractMatrix{T}}=nothing) where {T<:AbstractFloat}
    n, k = size(X)
    @assert length(residuals) == n "X and residuals must have same number of rows"

    bw = bandwidth == 0 ? optimal_bandwidth_nw(residuals) : bandwidth

    # Prewhitening
    u, X_use = if prewhiten && n > 2
        u_lag = @view residuals[1:end-1]
        u_lead = @view residuals[2:end]
        rho = dot(u_lag, u_lead) / dot(u_lag, u_lag)
        u_pw = residuals[2:end] .- rho .* residuals[1:end-1]
        (vcat([residuals[1]], u_pw), X)
    else
        (residuals, X)
    end

    # Use cached XtX_inv if provided, otherwise compute
    XtX_inv_use = isnothing(XtX_inv) ? robust_inv(X_use' * X_use) : XtX_inv

    # Compute S = long-run variance of X'u using optimized BLAS operations
    # Pre-compute X .* u for vectorized access
    Xu = X_use .* u  # n × k matrix

    # Lag-0 autocovariance: Γ₀ = (1/n) Σₜ (xₜuₜ)(xₜuₜ)'
    S = (Xu' * Xu)  # k × k, more efficient than loop

    # Add weighted lag autocovariances using BLAS
    @inbounds for j in 1:bw
        w = kernel_weight(j, bw, kernel, T)
        w == 0 && continue
        # Γⱼ = (1/n) Σₜ (xₜuₜ)(xₜ₋ⱼuₜ₋ⱼ)'
        Xu_t = @view Xu[(j+1):n, :]
        Xu_tj = @view Xu[1:(n-j), :]
        Gamma_j = Xu_t' * Xu_tj  # k × k
        S .+= w * (Gamma_j + Gamma_j')
    end

    # Recolor if prewhitened
    if prewhiten && n > 2
        u_lag = @view residuals[1:end-1]
        u_lead = @view residuals[2:end]
        rho = dot(u_lag, u_lead) / dot(u_lag, u_lag)
        S ./= (1 - rho)^2
    end

    S ./= n
    V = XtX_inv_use * S * XtX_inv_use
    # Ensure symmetry (may have tiny floating-point differences from BLAS)
    (V + V') / 2
end

"""
    newey_west(X::AbstractMatrix{T}, residuals::AbstractMatrix{T}; ...) -> Matrix{T}

Multivariate version for systems of equations.
"""
function newey_west(X::AbstractMatrix{T}, residuals::AbstractMatrix{T};
                    bandwidth::Int=0, kernel::Symbol=:bartlett) where {T<:AbstractFloat}
    n, n_eq = size(residuals)
    n_eq == 1 && return newey_west(X, vec(residuals); bandwidth, kernel)

    k = size(X, 2)
    V_full = zeros(T, k * n_eq, k * n_eq)
    for eq in 1:n_eq
        V_eq = newey_west(X, @view(residuals[:, eq]); bandwidth, kernel)
        idx = ((eq-1)*k + 1):(eq*k)
        V_full[idx, idx] .= V_eq
    end
    V_full
end

# =============================================================================
# White Heteroscedasticity-Robust Estimator
# =============================================================================

"""
    white_vcov(X::AbstractMatrix{T}, residuals::AbstractVector{T}; variant::Symbol=:hc0,
               XtX_inv::Union{Nothing,AbstractMatrix{T}}=nothing) -> Matrix{T}

White heteroscedasticity-robust covariance estimator.

Variants: :hc0, :hc1, :hc2, :hc3

# Arguments
- `X`: Design matrix (n × k)
- `residuals`: Residuals vector (n × 1)
- `variant`: HC variant (:hc0 = standard, :hc1 = small sample, :hc2/:hc3 = leverage-adjusted)
- `XtX_inv`: Pre-computed (X'X)^{-1} for performance (optional)

# Returns
Robust covariance matrix (k × k)

# Performance
Pass `XtX_inv` when calling multiple times with the same X to avoid recomputation.
"""
function white_vcov(X::AbstractMatrix{T}, residuals::AbstractVector{T};
                    variant::Symbol=:hc0,
                    XtX_inv::Union{Nothing,AbstractMatrix{T}}=nothing) where {T<:AbstractFloat}
    n, k = size(X)
    @assert length(residuals) == n

    # Use cached XtX_inv if provided, otherwise compute
    XtX_inv_use = isnothing(XtX_inv) ? robust_inv(X' * X) : XtX_inv

    # Compute leverage if needed (use cached XtX_inv)
    h_diag = if variant in (:hc2, :hc3)
        diag(X * XtX_inv_use * X')
    else
        nothing
    end

    # Vectorized computation for HC0 and HC1 (most common cases)
    if variant == :hc0
        # Omega = X' * Diagonal(u²) * X = (X .* u)' * (X .* u)
        Xu = X .* residuals
        Omega = Xu' * Xu
    elseif variant == :hc1
        Xu = X .* residuals
        Omega = Xu' * Xu * T(n) / T(n - k)
    else
        # HC2 and HC3 require per-observation adjustment
        Omega = zeros(T, k, k)
        @inbounds for t in 1:n
            u2 = residuals[t]^2
            u2_adj = if variant == :hc2
                u2 / (1 - h_diag[t])
            elseif variant == :hc3
                u2 / (1 - h_diag[t])^2
            else
                throw(ArgumentError("Unknown HC variant: $variant"))
            end
            x_t = @view X[t, :]
            Omega .+= u2_adj * (x_t * x_t')
        end
    end

    V = XtX_inv_use * Omega * XtX_inv_use
    # Ensure symmetry (may have tiny floating-point differences from BLAS)
    (V + V') / 2
end

"""
    white_vcov(X::AbstractMatrix{T}, residuals::AbstractMatrix{T}; ...) -> Matrix{T}

Multivariate version.
"""
function white_vcov(X::AbstractMatrix{T}, residuals::AbstractMatrix{T};
                    variant::Symbol=:hc0) where {T<:AbstractFloat}
    n, n_eq = size(residuals)
    n_eq == 1 && return white_vcov(X, vec(residuals); variant)

    k = size(X, 2)
    V_full = zeros(T, k * n_eq, k * n_eq)
    for eq in 1:n_eq
        V_eq = white_vcov(X, @view(residuals[:, eq]); variant)
        idx = ((eq-1)*k + 1):(eq*k)
        V_full[idx, idx] .= V_eq
    end
    V_full
end

# =============================================================================
# Driscoll-Kraay Estimator
# =============================================================================

"""
    driscoll_kraay(X::AbstractMatrix{T}, u::AbstractVector{T};
                   bandwidth::Int=0, kernel::Symbol=:bartlett,
                   XtX_inv::Union{Nothing,AbstractMatrix{T}}=nothing) -> Matrix{T}

Driscoll-Kraay standard errors for time series regression.

In a pure time series context, this is equivalent to Newey-West HAC estimation
applied to the moment conditions X'u. For panel data applications, it would
average across cross-sectional units first, but here we treat the data as a
single time series.

# Arguments
- `X`: Design matrix (T × k)
- `u`: Residuals vector (T × 1)
- `bandwidth`: Bandwidth for kernel. If 0, uses optimal bandwidth selection.
- `kernel`: Kernel function (:bartlett, :parzen, :quadratic_spectral, :tukey_hanning)
- `XtX_inv`: Pre-computed (X'X)^{-1} for performance (optional)

# Returns
Robust covariance matrix (k × k)

# References
- Driscoll, J. C., & Kraay, A. C. (1998). Consistent covariance matrix estimation
  with spatially dependent panel data. Review of Economics and Statistics.

# Performance
Pass `XtX_inv` when calling multiple times with the same X to avoid recomputation.
"""
function driscoll_kraay(X::AbstractMatrix{T}, u::AbstractVector{T};
                        bandwidth::Int=0, kernel::Symbol=:bartlett,
                        XtX_inv::Union{Nothing,AbstractMatrix{T}}=nothing) where {T<:AbstractFloat}
    n, k = size(X)

    # Moment conditions: g_t = X_t' * u_t (k × 1 for each t)
    G = X .* u  # T × k matrix of moment contributions

    # Use cached XtX_inv if provided, otherwise compute
    XtX_inv_use = isnothing(XtX_inv) ? robust_inv(X' * X) : XtX_inv

    # Compute long-run covariance of moment conditions
    S = long_run_covariance(G; bandwidth=bandwidth, kernel=kernel)

    # Sandwich formula: V = n * (X'X)^(-1) * S * (X'X)^(-1)
    V = n * XtX_inv_use * S * XtX_inv_use

    # Ensure symmetry
    (V + V') / 2
end

"""
    driscoll_kraay(X::AbstractMatrix{T}, U::AbstractMatrix{T};
                   bandwidth::Int=0, kernel::Symbol=:bartlett) -> Matrix{T}

Driscoll-Kraay standard errors for multi-equation system.
"""
function driscoll_kraay(X::AbstractMatrix{T}, U::AbstractMatrix{T};
                        bandwidth::Int=0, kernel::Symbol=:bartlett) where {T<:AbstractFloat}
    n, k = size(X)
    n_eq = size(U, 2)

    V = zeros(T, k * n_eq, k * n_eq)
    @inbounds for eq in 1:n_eq
        V_eq = driscoll_kraay(X, @view(U[:, eq]); bandwidth=bandwidth, kernel=kernel)
        idx = ((eq-1)*k + 1):(eq*k)
        V[idx, idx] .= V_eq
    end
    V
end

# =============================================================================
# Covariance Estimator Dispatch
# =============================================================================

"""
    robust_vcov(X::AbstractMatrix{T}, residuals::AbstractVecOrMat{T},
                estimator::AbstractCovarianceEstimator) -> Matrix{T}

Dispatch to appropriate covariance estimator based on estimator type.
"""
function robust_vcov(X::AbstractMatrix{T}, residuals::AbstractVector{T},
                     estimator::NeweyWestEstimator) where {T<:AbstractFloat}
    newey_west(X, residuals; bandwidth=estimator.bandwidth, kernel=estimator.kernel,
               prewhiten=estimator.prewhiten)
end

function robust_vcov(X::AbstractMatrix{T}, residuals::AbstractVector{T},
                     estimator::WhiteEstimator) where {T<:AbstractFloat}
    white_vcov(X, residuals)
end

function robust_vcov(X::AbstractMatrix{T}, residuals::AbstractVector{T},
                     estimator::DriscollKraayEstimator) where {T<:AbstractFloat}
    driscoll_kraay(X, residuals; bandwidth=estimator.bandwidth, kernel=estimator.kernel)
end

function robust_vcov(X::AbstractMatrix{T}, residuals::AbstractMatrix{T},
                     estimator::NeweyWestEstimator) where {T<:AbstractFloat}
    newey_west(X, residuals; bandwidth=estimator.bandwidth, kernel=estimator.kernel)
end

function robust_vcov(X::AbstractMatrix{T}, residuals::AbstractMatrix{T},
                     estimator::WhiteEstimator) where {T<:AbstractFloat}
    white_vcov(X, residuals)
end

function robust_vcov(X::AbstractMatrix{T}, residuals::AbstractMatrix{T},
                     estimator::DriscollKraayEstimator) where {T<:AbstractFloat}
    driscoll_kraay(X, residuals; bandwidth=estimator.bandwidth, kernel=estimator.kernel)
end

# =============================================================================
# Performance Utilities
# =============================================================================

"""
    precompute_XtX_inv(X::AbstractMatrix{T}) -> Matrix{T}

Pre-compute (X'X)^{-1} for use with covariance estimators.

When calling `newey_west`, `white_vcov`, or `driscoll_kraay` multiple times with
the same design matrix X, pre-computing XtX_inv avoids redundant matrix inversions.

# Example
```julia
X = randn(100, 5)
XtX_inv = precompute_XtX_inv(X)

# Use with multiple calls
V1 = newey_west(X, residuals1; XtX_inv=XtX_inv)
V2 = newey_west(X, residuals2; XtX_inv=XtX_inv)
V3 = white_vcov(X, residuals3; XtX_inv=XtX_inv)
```
"""
function precompute_XtX_inv(X::AbstractMatrix{T}) where {T<:AbstractFloat}
    robust_inv(X' * X)
end

# =============================================================================
# Long-Run Variance Estimation
# =============================================================================

"""
    long_run_variance(x::AbstractVector{T}; bandwidth::Int=0, kernel::Symbol=:bartlett) -> T

Estimate long-run variance: S = Σⱼ₌₋∞^∞ γⱼ

Used for unit root tests, cointegration tests, and other applications requiring
consistent variance estimation under serial correlation.

# Arguments
- `x`: Time series vector
- `bandwidth`: Truncation lag (0 = automatic)
- `kernel`: Kernel function

# Returns
Long-run variance estimate (scalar)
"""
function long_run_variance(x::AbstractVector{T}; bandwidth::Int=0,
                           kernel::Symbol=:bartlett) where {T<:AbstractFloat}
    n = length(x)
    n < 2 && return var(x)

    bw = bandwidth == 0 ? optimal_bandwidth_nw(x) : bandwidth
    x_demean = x .- mean(x)
    S = sum(x_demean.^2) / n

    @inbounds for j in 1:bw
        w = kernel_weight(j, bw, kernel, T)
        w == 0 && continue
        gamma_j = sum(x_demean[j+1:n] .* x_demean[1:n-j]) / n
        S += 2w * gamma_j
    end

    max(S, zero(T))
end

"""
    long_run_covariance(X::AbstractMatrix{T}; bandwidth::Int=0, kernel::Symbol=:bartlett) -> Matrix{T}

Estimate long-run covariance matrix of multivariate time series.

# Arguments
- `X`: Multivariate time series (T × k)
- `bandwidth`: Truncation lag (0 = automatic)
- `kernel`: Kernel function

# Returns
Long-run covariance matrix (k × k)

# Performance
Uses BLAS matrix operations for lag autocovariance computation.
"""
function long_run_covariance(X::AbstractMatrix{T}; bandwidth::Int=0,
                             kernel::Symbol=:bartlett) where {T<:AbstractFloat}
    n, k = size(X)
    n < 2 && return cov(X)

    bw = bandwidth == 0 ? optimal_bandwidth_nw(X) : bandwidth
    X_demean = X .- mean(X, dims=1)

    # Lag-0 autocovariance using BLAS
    S = (X_demean' * X_demean) / n

    # Pre-allocate for weighted lag autocovariances
    # Using views into X_demean for efficient BLAS operations
    @inbounds for j in 1:bw
        w = kernel_weight(j, bw, kernel, T)
        w == 0 && continue
        # Γⱼ = (1/n) X[j+1:n,:]' * X[1:n-j,:]
        X_t = @view X_demean[(j+1):n, :]
        X_tj = @view X_demean[1:(n-j), :]
        Gamma_j = (X_t' * X_tj) / n  # BLAS gemm
        # S += w * (Γⱼ + Γⱼ')
        @. S += w * (Gamma_j + Gamma_j')
    end

    # Ensure positive semi-definite using eigendecomposition
    # Only compute if needed (check for negative eigenvalues)
    S_sym = Hermitian((S + S') / 2)
    F = eigen(S_sym)
    if minimum(F.values) < 0
        D = max.(F.values, zero(T))
        S = F.vectors * Diagonal(D) * F.vectors'
    end

    Matrix(S)
end
