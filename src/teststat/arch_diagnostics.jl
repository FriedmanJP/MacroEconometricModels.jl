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
Diagnostic tests for ARCH/GARCH effects.
"""

# =============================================================================
# ARCH-LM Test (Engle 1982)
# =============================================================================

"""
    arch_lm_test(y_or_model, q=5)

ARCH-LM test for conditional heteroskedasticity (Engle 1982).

H₀: No ARCH effects (α₁ = ... = αq = 0)
H₁: ARCH(q) effects present

Test statistic: T·R² from regression of ε²ₜ on ε²ₜ₋₁,...,ε²ₜ₋q, distributed χ²(q).

# Arguments
- `y_or_model`: Raw data vector or AbstractVolatilityModel (uses standardized residuals)
- `q`: Number of lags (default 5)

# Returns
Named tuple `(statistic, pvalue, q)`.

# Example
```julia
result = arch_lm_test(randn(500), 5)
println("p-value: ", result.pvalue)
```
"""
function arch_lm_test(y::AbstractVector{T}, q::Int=5) where {T<:AbstractFloat}
    q < 1 && throw(ArgumentError("Number of lags q must be ≥ 1"))
    n = length(y)
    n < q + 2 && throw(ArgumentError("Need at least q+2 observations"))

    eps_sq = (y .- mean(y)) .^ 2
    _arch_lm_core(eps_sq, q)
end

function arch_lm_test(m::AbstractVolatilityModel, q::Int=5)
    _arch_lm_core(m.standardized_residuals .^ 2, q)
end

function _arch_lm_core(eps_sq::Vector{T}, q::Int) where {T}
    n = length(eps_sq)
    n_eff = n - q

    # Build regression: ε²ₜ on [1, ε²ₜ₋₁, ..., ε²ₜ₋q]
    X = ones(T, n_eff, q + 1)
    for lag in 1:q
        X[:, lag+1] = eps_sq[q+1-lag:n-lag]
    end
    y_reg = eps_sq[q+1:n]

    # OLS
    XtX_inv = robust_inv(X' * X)
    beta = XtX_inv * (X' * y_reg)
    fitted = X * beta
    resid = y_reg .- fitted

    # R²
    ss_res = sum(abs2, resid)
    ss_tot = sum(abs2, y_reg .- mean(y_reg))
    r2 = one(T) - ss_res / ss_tot

    statistic = T(n_eff) * r2
    pvalue = one(T) - cdf(Chisq(q), statistic)

    (statistic=statistic, pvalue=pvalue, q=q)
end

arch_lm_test(y::AbstractVector, q::Int=5) = arch_lm_test(Float64.(y), q)

# =============================================================================
# Ljung-Box Test on Squared Residuals
# =============================================================================

"""
    ljung_box_squared(z_or_model, K=10)

Ljung-Box test on squared (standardized) residuals.

H₀: No serial correlation in z²ₜ
H₁: Serial correlation present in z²ₜ

Test statistic: Q = n(n+2) Σₖ ρ̂²ₖ/(n-k), distributed χ²(K).

# Arguments
- `z_or_model`: Standardized residuals vector or AbstractVolatilityModel
- `K`: Number of lags (default 10)

# Returns
Named tuple `(statistic, pvalue, K)`.
"""
function ljung_box_squared(z::AbstractVector{T}, K::Int=10) where {T<:AbstractFloat}
    K < 1 && throw(ArgumentError("Number of lags K must be ≥ 1"))
    n = length(z)
    n < K + 2 && throw(ArgumentError("Need at least K+2 observations"))

    z_sq = z .^ 2
    z_sq_centered = z_sq .- mean(z_sq)
    gamma0 = sum(abs2, z_sq_centered) / n

    Q = zero(T)
    for k in 1:K
        rho_k = sum(z_sq_centered[1:n-k] .* z_sq_centered[k+1:n]) / (n * gamma0)
        Q += rho_k^2 / (n - k)
    end
    Q *= T(n) * T(n + 2)

    pvalue = one(T) - cdf(Chisq(K), Q)

    (statistic=Q, pvalue=pvalue, K=K)
end

function ljung_box_squared(m::AbstractVolatilityModel, K::Int=10)
    ljung_box_squared(m.standardized_residuals, K)
end

ljung_box_squared(z::AbstractVector, K::Int=10) = ljung_box_squared(Float64.(z), K)
