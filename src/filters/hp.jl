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
Hodrick-Prescott filter for trend-cycle decomposition.

Solves min_τ Σ(yₜ - τₜ)² + λ Σ(τₜ₊₁ - 2τₜ + τₜ₋₁)²
via the sparse linear system (I + λ D'D) τ = y.

Reference: Hodrick, Robert J., and Edward C. Prescott. 1997.
"Postwar U.S. Business Cycles: An Empirical Investigation."
*Journal of Money, Credit and Banking* 29 (1): 1–16.
"""

# =============================================================================
# Internal: Second-Difference Penalty Matrix
# =============================================================================

"""
    _hp_penalty(T_obs, lambda) -> SparseMatrixCSC

Build the sparse penalty matrix I + λ D'D where D is the (T-2)×T
second-difference operator. The result is a T×T pentadiagonal SPD matrix.
"""
function _hp_penalty(T_obs::Int, lambda::T) where {T<:AbstractFloat}
    # D is (T_obs-2) x T_obs second-difference matrix
    # D'D is T_obs x T_obs pentadiagonal
    # Build I + lambda * D'D directly using sparse diagonals
    e = ones(T, T_obs)

    # Diagonal bands of D'D (pentadiagonal):
    # Main diagonal: [1, 5, 6, ..., 6, 5, 1] scaled by lambda, plus I
    # Off-diagonal ±1: [-2, -4, -4, ..., -4, -2] scaled by lambda
    # Off-diagonal ±2: [1, 1, ..., 1] scaled by lambda

    d0 = ones(T, T_obs)  # main diagonal (starts with I)
    d1 = zeros(T, T_obs - 1)  # sub/super diagonal ±1
    d2 = zeros(T, T_obs - 2)  # sub/super diagonal ±2

    # Fill D'D contributions
    # D'D[i,i] for interior points
    d0[1] += lambda
    d0[2] += T(5) * lambda
    @inbounds for i in 3:(T_obs - 2)
        d0[i] += T(6) * lambda
    end
    d0[T_obs - 1] += T(5) * lambda
    d0[T_obs] += lambda

    # D'D off-diagonal ±1
    d1[1] += T(-2) * lambda
    @inbounds for i in 2:(T_obs - 2)
        d1[i] += T(-4) * lambda
    end
    d1[T_obs - 1] += T(-2) * lambda

    # D'D off-diagonal ±2
    @inbounds for i in 1:(T_obs - 2)
        d2[i] += lambda
    end

    spdiagm(0 => d0, 1 => d1, -1 => d1, 2 => d2, -2 => d2)
end

# =============================================================================
# Public API
# =============================================================================

"""
    hp_filter(y::AbstractVector; lambda=1600.0) -> HPFilterResult

Apply the Hodrick-Prescott filter to decompose time series `y` into trend and cycle.

Solves the optimization problem:
```math
\\min_\\tau \\sum_{t=1}^T (y_t - \\tau_t)^2 + \\lambda \\sum_{t=2}^{T-1} (\\tau_{t+1} - 2\\tau_t + \\tau_{t-1})^2
```

Implementation uses a sparse pentadiagonal Cholesky factorization for O(T) cost.

# Arguments
- `y::AbstractVector`: Time series data (length ≥ 3)

# Keywords
- `lambda::Real=1600.0`: Smoothing parameter. Common values: 6.25 (annual),
  1600 (quarterly, default), 129600 (monthly)

# Returns
- `HPFilterResult{T}` with fields `trend`, `cycle`, `lambda`, `T_obs`

# Examples
```julia
y = cumsum(randn(200))
result = hp_filter(y)
result = hp_filter(y; lambda=6.25)  # annual data
```

# References
- Hodrick, R. J., & Prescott, E. C. (1997). *JMCB* 29(1): 1–16.
"""
function hp_filter(y::AbstractVector{T}; lambda::Real=T(1600)) where {T<:AbstractFloat}
    T_obs = length(y)
    T_obs < 3 && throw(ArgumentError("HP filter requires at least 3 observations, got $T_obs"))
    lambda < 0 && throw(ArgumentError("lambda must be non-negative, got $lambda"))

    lam = T(lambda)

    # Special case: lambda = 0 means no smoothing → trend = y
    if lam == zero(T)
        return HPFilterResult(copy(Vector{T}(y)), zeros(T, T_obs), lam, T_obs)
    end

    # Build and solve (I + λ D'D) τ = y via sparse Cholesky
    # CHOLMOD (SuiteSparse) only supports Float64 — promote for Julia 1.10 LTS compatibility
    A64 = _hp_penalty(T_obs, Float64(lam))
    tau = Vector{T}(A64 \ Vector{Float64}(y))
    cyc = Vector{T}(y) .- tau

    HPFilterResult(tau, cyc, lam, T_obs)
end

# Float64 fallback for non-float input
hp_filter(y::AbstractVector; kwargs...) = hp_filter(Float64.(y); kwargs...)
