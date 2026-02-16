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
Shared test data generators for MacroEconometricModels.jl test suite.

All functions are pure (no global state) and accept an explicit `rng` argument
for reproducibility across threaded test groups.
"""

using Random, LinearAlgebra

# =============================================================================
# VAR DGP generators
# =============================================================================

"""
    make_var1_data(; T=200, n=3, seed=42) -> Matrix{Float64}

Generate VAR(1) data with diagonal coefficient matrix A = 0.5*I.
Returns T×n matrix.
"""
function make_var1_data(; T::Int=200, n::Int=3, seed::Int=42)
    rng = Random.MersenneTwister(seed)
    A = 0.5 * I(n)
    Y = zeros(T, n)
    for t in 2:T
        Y[t, :] = A * Y[t-1, :] + randn(rng, n)
    end
    Y
end

"""
    make_var_data(A::AbstractMatrix, T::Int; c=zeros(size(A,1)), seed=42) -> Matrix{Float64}

Generate VAR(1) data with specified coefficient matrix A and intercept c.
"""
function make_var_data(A::AbstractMatrix, T::Int; c::Vector{Float64}=zeros(size(A, 1)), seed::Int=42)
    rng = Random.MersenneTwister(seed)
    n = size(A, 1)
    Y = zeros(T, n)
    for t in 2:T
        Y[t, :] = c + A * Y[t-1, :] + randn(rng, n)
    end
    Y
end

# =============================================================================
# Univariate DGP generators
# =============================================================================

"""
    make_ar1_data(; n=500, phi=0.7, c=0.5, sigma=1.0, seed=42) -> Vector{Float64}

Generate stationary AR(1) process: yₜ = c + φ yₜ₋₁ + σ εₜ.
"""
function make_ar1_data(; n::Int=500, phi::Float64=0.7, c::Float64=0.5,
                        sigma::Float64=1.0, seed::Int=42)
    rng = Random.MersenneTwister(seed)
    y = zeros(n)
    y[1] = c / (1 - phi) + randn(rng)
    for t in 2:n
        y[t] = c + phi * y[t-1] + sigma * randn(rng)
    end
    y
end

"""
    make_random_walk(; n=200, seed=42) -> Vector{Float64}

Generate I(1) random walk: yₜ = yₜ₋₁ + εₜ.
"""
function make_random_walk(; n::Int=200, seed::Int=42)
    rng = Random.MersenneTwister(seed)
    cumsum(randn(rng, n))
end

"""
    make_cointegrated_data(; T_obs=200, n=3, rank=1, seed=42) -> Matrix{Float64}

Generate n-dimensional I(1) data with `rank` cointegrating relationships.
"""
function make_cointegrated_data(; T_obs::Int=200, n::Int=3, rank::Int=1, seed::Int=42)
    rng = Random.MersenneTwister(seed)
    Y = cumsum(randn(rng, T_obs, n), dims=1)
    for r in 1:min(rank, n - 1)
        Y[:, r + 1] = Y[:, 1] + 0.1 * randn(rng, T_obs)
    end
    Y
end

# =============================================================================
# Factor model DGP
# =============================================================================

"""
    make_factor_data(; T=200, N=20, r=3, noise=0.5, seed=42) -> (X, F_true, Lambda_true)

Generate factor model data: X = F Λ' + noise * E.
"""
function make_factor_data(; T::Int=200, N::Int=20, r::Int=3,
                           noise::Float64=0.5, seed::Int=42)
    rng = Random.MersenneTwister(seed)
    F_true = randn(rng, T, r)
    Lambda_true = randn(rng, N, r)
    E = randn(rng, T, N)
    X = F_true * Lambda_true' + noise * E
    (X=X, F_true=F_true, Lambda_true=Lambda_true)
end

# =============================================================================
# Volatility DGP generators
# =============================================================================

"""
    simulate_arch1(; n=1000, omega=0.1, alpha1=0.3, mu=0.0, seed=42) -> Vector{Float64}

Simulate ARCH(1) process.
"""
function simulate_arch1(; n::Int=1000, omega::Float64=0.1, alpha1::Float64=0.3,
                         mu::Float64=0.0, seed::Int=42)
    rng = Random.MersenneTwister(seed)
    y = zeros(n)
    h = zeros(n)
    h[1] = omega / (1 - alpha1)
    y[1] = mu + sqrt(h[1]) * randn(rng)
    for t in 2:n
        h[t] = omega + alpha1 * (y[t-1] - mu)^2
        y[t] = mu + sqrt(h[t]) * randn(rng)
    end
    y
end

"""
    simulate_garch11(; n=1000, omega=0.01, alpha1=0.05, beta1=0.90, mu=0.0, seed=42) -> Vector{Float64}

Simulate GARCH(1,1) process.
"""
function simulate_garch11(; n::Int=1000, omega::Float64=0.01, alpha1::Float64=0.05,
                           beta1::Float64=0.90, mu::Float64=0.0, seed::Int=42)
    rng = Random.MersenneTwister(seed)
    y = zeros(n)
    h = zeros(n)
    h[1] = omega / (1 - alpha1 - beta1)
    y[1] = mu + sqrt(h[1]) * randn(rng)
    for t in 2:n
        h[t] = omega + alpha1 * (y[t-1] - mu)^2 + beta1 * h[t-1]
        y[t] = mu + sqrt(h[t]) * randn(rng)
    end
    y
end
