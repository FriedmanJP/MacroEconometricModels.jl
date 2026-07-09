# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

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
- Andrews, D. W. K., & Monahan, J. C. (1992). An Improved Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix Estimator. Econometrica 60(4):953-966.
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
    # Only compact-support kernels truncate at |x|>1. The quadratic-spectral kernel
    # has infinite support and is nonzero for all x, so it must NOT short-circuit here.
    kernel != :quadratic_spectral && abs(x) > 1 && return zero(T)

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
    optimal_bandwidth_nw(residuals::AbstractVector{T}; kernel::Symbol=:bartlett) -> Int

Automatic HAC truncation-lag selection via the Andrews (1991) parametric AR(1)
plug-in:

    m = ĉ_kernel · (α̂(q) · n)^{1/(2q+1)}

where `q` is the kernel's characteristic (Parzen) exponent — `q=1` for Bartlett,
`q=2` for Parzen / quadratic-spectral / Tukey–Hanning — `ĉ_kernel` is the Andrews
(1991, Table I) kernel constant, and `α̂(q)` is the AR(1) plug-in

    α̂(1) = 4ρ̂² / (1−ρ̂²)²     (Bartlett)
    α̂(2) = 4ρ̂² / (1−ρ̂)⁴      (Parzen / QS / Tukey–Hanning)

`kernel` MUST match the kernel used to weight the sample autocovariances, so the
selected bandwidth is kernel-consistent (earlier versions always used the
Bartlett constant with the `q=2` plug-in regardless of kernel — a mislabel).

A loose Schwert (1989) ceiling `⌊12·(n/100)^{1/4}⌋` guards against runaway
bandwidths on near-integrated series; this replaces the earlier degenerate
`⌊n^{1/3}⌋` clamp, which bound almost always and threw away the plug-in.

Reference: Andrews, D.W.K. (1991), *Econometrica* 59(3):817–858, Table I.
"""
function optimal_bandwidth_nw(residuals::AbstractVector{T};
                              kernel::Symbol=:bartlett) where {T<:AbstractFloat}
    n = length(residuals)
    n < 4 && return 0

    # AR(1) coefficient (same estimator the prewhitening path uses)
    r_lag = @view residuals[1:end-1]
    r_lead = @view residuals[2:end]
    rho = dot(r_lag, r_lead) / dot(r_lag, r_lag)
    rho_abs = min(abs(rho), T(0.99))

    # Andrews (1991) plug-in: kernel-specific constant + characteristic exponent q
    m = if kernel == :bartlett
        alpha = 4rho_abs^2 / (1 - rho_abs^2)^2                # α(1), q=1
        ceil(Int, T(1.1447) * (alpha * n)^(one(T) / 3))
    elseif kernel == :parzen
        alpha = 4rho_abs^2 / (1 - rho_abs)^4                  # α(2), q=2
        ceil(Int, T(2.6614) * (alpha * n)^(one(T) / 5))
    elseif kernel == :quadratic_spectral
        alpha = 4rho_abs^2 / (1 - rho_abs)^4
        ceil(Int, T(1.3221) * (alpha * n)^(one(T) / 5))
    elseif kernel == :tukey_hanning
        alpha = 4rho_abs^2 / (1 - rho_abs)^4
        ceil(Int, T(1.7462) * (alpha * n)^(one(T) / 5))
    else
        throw(ArgumentError("Unknown kernel: $kernel"))
    end

    m = max(m, 0)
    # Loose Schwert (1989) safety ceiling (not the degenerate n^(1/3) clamp)
    schwert = floor(Int, 12 * (T(n) / 100)^(one(T) / 4))
    min(m, schwert)
end

"""
    optimal_bandwidth_nw(residuals::AbstractMatrix{T}; kernel::Symbol=:bartlett) -> Int

Multivariate version: average the kernel-consistent optimal bandwidth across columns.
"""
function optimal_bandwidth_nw(residuals::AbstractMatrix{T};
                              kernel::Symbol=:bartlett) where {T<:AbstractFloat}
    n_vars = size(residuals, 2)
    n_vars == 0 && return 0
    round(Int, mean(optimal_bandwidth_nw(@view residuals[:, j]; kernel=kernel) for j in 1:n_vars))
end

# =============================================================================
# Newey-West HAC Estimator
# =============================================================================

"""
    _prewhiten_moments(G::AbstractMatrix{T}; radius_cap=0.97) -> (Ghat, A)

Andrews–Monahan (1992) VAR(1) prewhitening of the moment matrix `G` (rows `g_t`,
`n × k`). Fits `g_t = A g_{t-1} + ε_t` by multivariate least squares and returns
the `(n-1) × k` whitened residuals `Ĝ = Glead − Glag·B` together with the VAR(1)
matrix `A = B'` (where `B = (Glag'Glag)^{-1} Glag'Glead`). The whitened series is
NOT spliced with a first observation — it has exactly `n-1` rows.

Returns `(nothing, nothing)` when the fitted VAR(1) is not stable (spectral radius
`≥ radius_cap`), so the caller can fall back to no prewhitening rather than
recolor through a near-singular `(I − A)^{-1}`.
"""
function _prewhiten_moments(G::AbstractMatrix{T}; radius_cap::T=T(0.97)) where {T<:AbstractFloat}
    n = size(G, 1)
    n ≤ 2 && return (nothing, nothing)
    Glag  = @view G[1:n-1, :]
    Glead = @view G[2:n, :]
    # Multiple-response LS: g_t' ≈ g_{t-1}' B ⇒ B = (Glag'Glag)^{-1}(Glag'Glead) (k×k),
    # so the VAR(1) matrix in g_t = A g_{t-1} + ε_t is A = B'.
    B = robust_inv(Glag' * Glag) * (Glag' * Glead)
    A = Matrix{T}(B')
    ev = maximum(abs, eigvals(A))
    (isfinite(ev) && ev < radius_cap) || return (nothing, nothing)
    Ghat = Glead .- Glag * B                 # (n-1)×k whitened residuals (no spliced row)
    return (Ghat, A)
end

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
- `prewhiten`: Andrews–Monahan (1992) VAR(1) prewhitening of the moment vector
  `x_t·u_t` with matrix recoloring `S = D S* D'`, `D = (I − Â)^{-1}`; skipped with
  a warning when the fitted moment VAR(1) is near-unit-root (non-stable)
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

    bw = bandwidth == 0 ? optimal_bandwidth_nw(residuals; kernel=kernel) : bandwidth

    # Bread uses the ORIGINAL design matrix — moment prewhitening does not change X'X.
    XtX_inv_use = isnothing(XtX_inv) ? robust_inv(X' * X) : XtX_inv

    # Moment matrix g_t = x_t·u_t (rows), n × k.
    G = X .* residuals

    # Andrews–Monahan (1992) VAR(1) prewhitening of the moment vector. Fall back to the
    # raw moments (with a warning) if the fitted VAR(1) is near a unit root.
    M = G
    recolor_A = nothing
    if prewhiten
        Ghat, A = _prewhiten_moments(G)
        if Ghat === nothing
            @warn "Andrews–Monahan prewhitening skipped: moment VAR(1) is non-stable (near unit root); falling back to no prewhitening." maxlog=1
        else
            M = Ghat
            recolor_A = A
        end
    end

    # Long-run variance S* of the (possibly whitened) moments.
    m = size(M, 1)
    S = M' * M
    # QS kernel has infinite support → sum all lags to m-1; compact kernels truncate at bw.
    jmax = kernel == :quadratic_spectral ? (m - 1) : bw
    @inbounds for j in 1:jmax
        w = kernel_weight(j, bw, kernel, T)
        w == 0 && continue
        Mt  = @view M[(j+1):m, :]
        Mtj = @view M[1:(m-j), :]
        Gamma_j = Mt' * Mtj  # k × k
        S .+= w * (Gamma_j + Gamma_j')
    end

    # Recolor: S = D S* D', D = (I − Â)^{-1}.
    if recolor_A !== nothing
        D = robust_inv(Matrix{T}(I, k, k) - recolor_A)
        S = D * S * D'
        S = (S + S') / 2
    end

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

    bw = bandwidth == 0 ? optimal_bandwidth_nw(x; kernel=kernel) : bandwidth
    x_demean = x .- mean(x)
    S = sum(x_demean.^2) / n

    # QS kernel has infinite support → sum all lags to n-1; compact kernels truncate at bw.
    jmax = kernel == :quadratic_spectral ? (n - 1) : bw
    @inbounds for j in 1:jmax
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

    bw = bandwidth == 0 ? optimal_bandwidth_nw(X; kernel=kernel) : bandwidth
    X_demean = X .- mean(X, dims=1)

    # Lag-0 autocovariance using BLAS
    S = (X_demean' * X_demean) / n

    # Pre-allocate for weighted lag autocovariances. Using views into X_demean for
    # efficient BLAS operations. QS kernel has infinite support → sum all lags to n-1;
    # compact kernels truncate at bw.
    jmax = kernel == :quadratic_spectral ? (n - 1) : bw
    @inbounds for j in 1:jmax
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

# =============================================================================
# Covariance Estimator Registry
# =============================================================================

"""
    _COV_REGISTRY

Registry mapping Symbol names to covariance estimator types.
Use `register_cov_estimator!` to add custom estimators.
"""
const _COV_REGISTRY = Dict{Symbol, Type}(
    :newey_west => NeweyWestEstimator,
    :white => WhiteEstimator,
    :driscoll_kraay => DriscollKraayEstimator,
)

"""
    register_cov_estimator!(name::Symbol, ::Type{T}) where {T<:AbstractCovarianceEstimator}

Register a custom covariance estimator type for use in LP and other estimators.

# Example
```julia
struct MyCovEstimator <: AbstractCovarianceEstimator end
register_cov_estimator!(:my_cov, MyCovEstimator)
```
"""
function register_cov_estimator!(name::Symbol, ::Type{E}) where {E<:AbstractCovarianceEstimator}
    _COV_REGISTRY[name] = E
end
