# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Shared test data generators for MacroEconometricModels.jl test suite.

All functions are pure (no global state) and accept an explicit `rng` argument
for reproducibility across threaded test groups.
"""

using Random, LinearAlgebra, Statistics, Distributions, DataFrames

# DGP-01 (#790): shared truth-returning DGP library. Every simulator takes
# rng::AbstractRNG first, burns in, and returns (data, truth...) — see
# test/dgp/ALLOWLIST.md for the white-noise lint allowlist.
if !isdefined(@__MODULE__, :dgp_var)  # include guard: fixtures may load twice
    for _dgp_file in ("dgp_var.jl", "dgp_svar_nongauss.jl", "dgp_univariate.jl",
                      "dgp_cointegration.jl", "dgp_volatility.jl",
                      "dgp_factors.jl", "dgp_lp.jl", "dgp_micro.jl",
                      "dgp_regime.jl", "dgp_gmm.jl", "dgp_truth.jl")
        include(joinpath(@__DIR__, "dgp", _dgp_file))
    end
end

# Safe println that silently catches IOError when stdout pipe is closed
# (happens in threaded parallel test execution on macOS CI)
function _tprint(args...)
    try
        println(args...)
    catch e
        e isa Base.IOError || rethrow()
    end
end

# =============================================================================
# VAR DGP generators
# =============================================================================

"""
    make_var1_data(; T=200, n=3, seed=42) -> Matrix{Float64}

Legacy shim (DGP-01 #790): VAR(1) with `A = 0.5*I`, now with burn-in and
Cholesky-scaled innovations via `dgp_var`. New tests should call `dgp_var`
directly (it also returns `A`, `Sigma`, `B0`, the shocks).
"""
function make_var1_data(; T::Int=200, n::Int=3, seed::Int=42)
    dgp_var(Random.MersenneTwister(seed); A=Matrix{Float64}(0.5 * I(n)),
            B0=Matrix{Float64}(I(n)), T=T).Y
end

# make_var_data removed in DGP-01 (#790): it had zero callers. Use dgp_var.

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
    make_cointegrated_data(; T_obs=200, n=3, rank=1, seed=42, alpha=nothing,
                            beta=nothing, Gamma=nothing) -> Matrix{Float64}

Legacy shim (DGP-01 #790): parametrised VECM via `dgp_vecm` (moderate
adjustment, non-zero `Gamma`, distinct trends) replacing the degenerate
instant-adjustment corner. New tests should call `dgp_vecm` directly.
"""
function make_cointegrated_data(; T_obs::Int=200, n::Int=3, rank::Int=1, seed::Int=42,
                                alpha=nothing, beta=nothing, Gamma=nothing)
    r = min(rank, n - 1)
    # Default β: each of columns 2..r+1 cointegrates against y₁ (distinct trends).
    be = if beta === nothing
        B = zeros(n, r)
        B[1, :] .= 1.0
        for j in 1:r
            B[j + 1, j] = -1.0
        end
        B
    else
        beta
    end
    al = if alpha === nothing
        A = zeros(n, r)
        for j in 1:r
            A[j + 1, j] = -0.3
        end
        A
    else
        alpha
    end
    Ga = Gamma === nothing ? [Matrix{Float64}(0.2 * I, n, n)] : Gamma
    dgp_vecm(Random.MersenneTwister(seed); alpha=al, beta=be, Gamma=Ga, T=T_obs).Y
end

# =============================================================================
# Factor model DGP
# =============================================================================

"""
    make_factor_data(; T=200, N=20, r=3, noise=0.5, seed=42) -> (X, F_true, Lambda_true, A)

Legacy shim (DGP-01 #790): factors are now VAR(1) (not iid) via
`dgp_dynamic_factors`. New tests should call `dgp_dynamic_factors` directly.
"""
function make_factor_data(; T::Int=200, N::Int=20, r::Int=3,
                           noise::Float64=0.5, seed::Int=42)
    A = Matrix{Float64}(0.7 * I, r, r)
    d = dgp_dynamic_factors(Random.MersenneTwister(seed); A=A, r=r, N=N, T=T,
                            idio_sd=noise, signal_share=1.0 / (1.0 + noise^2))
    (X=d.X, F_true=d.F, Lambda_true=d.Lambda, A=d.A[1])
end

# =============================================================================
# Volatility DGP generators
# =============================================================================

"""
    simulate_arch1(; n=1000, omega=0.1, alpha1=0.3, mu=0.0, seed=42) -> Vector{Float64}

Legacy shim (DGP-01 #790): ARCH(1) with burn-in via `dgp_garch_family`
(which also returns the variance path `h` this shim discards).
"""
function simulate_arch1(; n::Int=1000, omega::Float64=0.1, alpha1::Float64=0.3,
                         mu::Float64=0.0, seed::Int=42)
    dgp_garch_family(Random.MersenneTwister(seed); kind=:arch, omega=omega,
                     alpha=alpha1, beta=0.0, mu=mu, T=n).y
end

"""
    simulate_garch11(; n=1000, omega=0.01, alpha1=0.05, beta1=0.90, mu=0.0, seed=42) -> Vector{Float64}

Legacy shim (DGP-01 #790): GARCH(1,1) with burn-in via `dgp_garch_family`
(which also returns the variance path `h` this shim discards).
"""
function simulate_garch11(; n::Int=1000, omega::Float64=0.01, alpha1::Float64=0.05,
                           beta1::Float64=0.90, mu::Float64=0.0, seed::Int=42)
    dgp_garch_family(Random.MersenneTwister(seed); kind=:garch, omega=omega,
                     alpha=alpha1, beta=beta1, mu=mu, T=n).y
end
