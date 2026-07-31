# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# T250 (#349): Primiceri TVP-VAR with stochastic volatility / Cogley-Sargent SV-BVAR.

using Test
using MacroEconometricModels
using LinearAlgebra
using Random
using Statistics

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

const _M = MacroEconometricModels

"""Homoskedastic constant-coefficient VAR(1)."""
function _tvp_sim_const(T_obs::Int; seed::Int=1)
    rng = Random.MersenneTwister(seed)
    A = [0.5 0.1; 0.2 0.4]
    L = [0.5 0.0; 0.2 0.4]
    Y = zeros(T_obs, 2)
    for t in 2:T_obs
        Y[t, :] = A * Y[t-1, :] + L * randn(rng, 2)
    end
    return Y
end

"""VAR(1) whose shock standard deviations jump by `scale` at the midpoint."""
function _tvp_sim_break(T_obs::Int; seed::Int=2, scale::Float64=3.0)
    rng = Random.MersenneTwister(seed)
    A = [0.5 0.1; 0.2 0.4]
    sd = [0.5, 0.4]
    Y = zeros(T_obs, 2)
    for t in 2:T_obs
        sc = t > T_obs ÷ 2 ? scale : 1.0
        Y[t, :] = A * Y[t-1, :] .+ sc .* sd .* randn(rng, 2)
    end
    return Y
end

@testset "TVP-VAR with stochastic volatility" begin

# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

@testset "design matrices place each equation's coefficients contiguously" begin
    Y = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0]
    y, Xt, T_eff, k = _M._tvp_design(Y, 1)
    n, m = 2, 1 + 2 * 1
    @test T_eff == 3
    @test k == n * m
    @test y == Y[2:end, :]

    # X_t' = I_n ⊗ z_t' with z_t = [1, y_{t-1}']
    for t in 1:T_eff
        z = vcat(1.0, Y[t, :])
        expected = zeros(n, k)
        for i in 1:n
            expected[i, ((i-1)*m+1):(i*m)] = z
        end
        @test Xt[t] == expected
    end

    # p = 2 stacks lags in order
    y2, Xt2, T2, k2 = _M._tvp_design(Y, 2)
    @test T2 == 2
    @test k2 == 2 * (1 + 2 * 2)
    @test Xt2[1][1, 1:5] == vcat(1.0, Y[2, :], Y[1, :])
end

@testset "A_t reconstruction from stacked free elements" begin
    a = [0.3, -0.7, 0.2]                     # a₂₁, a₃₁, a₃₂
    A = _M._tvp_A_matrix(a, 3)
    @test A == [1.0 0.0 0.0; 0.3 1.0 0.0; -0.7 0.2 1.0]
    @test istril(A)
    @test all(diag(A) .== 1.0)
    @test _M._tvp_A_matrix(Float64[], 1) == reshape([1.0], 1, 1)
end

@testset "Carter-Kohn FFBS: Q = 0 gives an exactly constant path at the GLS posterior" begin
    rng = Random.MersenneTwister(5)
    T_eff, k = 60, 2
    b_true = [1.5, -0.8]
    Xt = [randn(rng, 1, k) for _ in 1:T_eff]
    Rt = [fill(0.25, 1, 1) for _ in 1:T_eff]
    y = Matrix{Float64}(undef, T_eff, 1)
    for t in 1:T_eff
        y[t, 1] = (Xt[t]*b_true)[1] + 0.5 * randn(rng)
    end

    Q = zeros(2, 2)                           # no drift
    # A weak but CONDITIONED prior. Q = 0 makes P_{t+1|t} = P_{t|t}, so the backward
    # conditional covariance is a cancellation of two identical matrices; its floating-point
    # residue — and hence the jitter the square root injects — scales with the prior
    # variance (P0 = 1e6·I leaves ~4e-4 of drift, P0 = 100·I only ~2e-6).
    P0 = 100.0 * Matrix{Float64}(I, 2, 2)
    path = _M._tvp_ffbs_rw(y, Xt, Rt, Q, zeros(2), P0, Random.MersenneTwister(7))

    # With Q = 0 the state cannot move: the backward gain G = P_{t|t}P_{t+1|t}^{-1} is the
    # identity and the conditional covariance vanishes, so every row equals the last.
    for t in 1:(T_eff-1)
        @test path[t, :] ≈ path[T_eff, :] atol = 1e-4
    end

    # ... and the drawn value is a draw from the full-sample GLS posterior, so many draws
    # centre on the GLS point estimate.
    XtX = zeros(2, 2); Xty = zeros(2)
    for t in 1:T_eff
        w = 1 / Rt[t][1, 1]
        XtX .+= w .* (Xt[t]' * Xt[t])
        Xty .+= w .* vec(Xt[t]' * y[t, :])
    end
    gls = XtX \ Xty
    draws = reduce(hcat, [_M._tvp_ffbs_rw(y, Xt, Rt, Q, zeros(2), P0,
                                          Random.MersenneTwister(100 + d))[1, :]
                          for d in 1:400])
    @test vec(mean(draws; dims=2)) ≈ gls atol = 0.06
end

@testset "Carter-Kohn FFBS: large Q lets the state track the data" begin
    rng = Random.MersenneTwister(9)
    T_eff = 80
    Xt = [ones(1, 1) for _ in 1:T_eff]
    Rt = [fill(0.01, 1, 1) for _ in 1:T_eff]
    b_path = [t <= 40 ? 0.0 : 5.0 for t in 1:T_eff]      # a level break
    y = reshape([b_path[t] + 0.1 * randn(rng) for t in 1:T_eff], T_eff, 1)

    path = _M._tvp_ffbs_rw(y, Xt, Rt, fill(1.0, 1, 1), [0.0], fill(1.0, 1, 1),
                           Random.MersenneTwister(3))
    @test mean(path[1:35, 1]) < 1.0
    @test mean(path[45:end, 1]) > 4.0
end

# ─────────────────────────────────────────────────────────────────────────────
# Sampler behaviour
# ─────────────────────────────────────────────────────────────────────────────

@testset "constant-coefficient SV-BVAR matches OLS and the conjugate BVAR" begin
    Y = _tvp_sim_const(FAST ? 300 : 600)
    post = estimate_tvpvar(Y, 1; tvp=false, sv=true,
                           n_draws=FAST ? 120 : 300, n_burn=FAST ? 120 : 300,
                           varnames=["y1", "y2"], rng=Random.MersenneTwister(11))

    @test post isa TVPVARPosterior{Float64}
    @test !post.tvp && post.sv
    @test post.n == 2 && post.p == 1

    # tvp=false ⇒ the coefficient path is exactly constant within each draw
    for d in 1:size(post.B_draws, 1)
        @test post.B_draws[d, 1, :] ≈ post.B_draws[d, post.T_eff, :] atol = 1e-12
    end

    B = dropdims(mean(post.B_draws; dims=1); dims=1)[1, :]
    A1_sv = [B[2] B[3]; B[5] B[6]]

    # Compare on the SAME estimation window the sampler uses (post-training)
    Yw = Y[(post.n_train - post.p + 1):end, :]
    ols = estimate_var(Yw, 1)
    A1_ols = Matrix(ols.B[2:3, :]')
    @test A1_sv ≈ A1_ols atol = 0.05
    @test [B[1], B[4]] ≈ ols.B[1, :] atol = 0.05

    bv = estimate_bvar(Yw, 1; n_draws=300, rng=Random.MersenneTwister(11))
    Bb = dropdims(mean(bv.B_draws; dims=1); dims=1)
    @test A1_sv ≈ Matrix(Bb[2:3, :]') atol = 0.05
end

@testset "stochastic volatility tracks a known break" begin
    Y = _tvp_sim_break(FAST ? 200 : 300; scale=3.0)
    post = estimate_tvpvar(Y, 1; n_draws=FAST ? 120 : 250, n_burn=FAST ? 120 : 250,
                           varnames=["y1", "y2"], rng=Random.MersenneTwister(21))
    @test post.tvp && post.sv

    vol, qs = volatility_path(post)
    @test size(vol) == (post.T_eff, 2)
    @test size(qs) == (post.T_eff, 2, 3)
    @test all(vol .> 0)
    # Bands are ordered
    @test all(qs[:, :, 1] .<= qs[:, :, 2] .<= qs[:, :, 3])

    Te = post.T_eff
    first_q = vec(mean(vol[1:(Te ÷ 4), :]; dims=1))
    last_q = vec(mean(vol[(3Te ÷ 4):end, :]; dims=1))
    # The DGP triples the shock standard deviations at the midpoint
    ratio = last_q ./ first_q
    @test all(2.0 .< ratio .< 4.5)
    # Levels, not just the ratio: the state is log σ², so σ = exp(h/2)
    @test first_q[1] ≈ 0.5 atol = 0.25
    @test last_q[1] ≈ 1.5 atol = 0.6

    # volatility_path is exactly exp(H/2) averaged over draws
    @test vol ≈ dropdims(mean(exp.(post.H_draws ./ 2); dims=1); dims=1) atol = 1e-12
end

@testset "sv=false freezes the volatilities" begin
    Y = _tvp_sim_const(200)
    post = estimate_tvpvar(Y, 1; tvp=true, sv=false, n_draws=60, n_burn=60,
                           rng=Random.MersenneTwister(31))
    @test !post.sv
    # Every draw keeps the training-sample volatility at every date
    for d in 1:size(post.H_draws, 1)
        @test post.H_draws[d, 1, :] ≈ post.H_draws[d, post.T_eff, :] atol = 1e-12
    end
    @test all(post.W_draws .> 0)          # W is left at its prior value, still positive
end

@testset "drifting coefficients actually drift" begin
    # A coefficient break: the AR(1) persistence of y1 doubles at the midpoint
    rng = Random.MersenneTwister(41)
    T_obs = 400
    Y = zeros(T_obs, 2)
    for t in 2:T_obs
        a11 = t > T_obs ÷ 2 ? 0.8 : 0.1
        Y[t, 1] = a11 * Y[t-1, 1] + 0.3 * randn(rng)
        Y[t, 2] = 0.3 * Y[t-1, 2] + 0.2 * Y[t-1, 1] + 0.3 * randn(rng)
    end
    post = estimate_tvpvar(Y, 1; n_draws=FAST ? 100 : 250, n_burn=FAST ? 100 : 250,
                           k_Q=0.05, rng=Random.MersenneTwister(43))
    B = dropdims(mean(post.B_draws; dims=1); dims=1)      # T_eff × k
    Te = post.T_eff
    a11_path = B[:, 2]                                    # equation 1, own first lag
    @test mean(a11_path[(3Te ÷ 4):end]) > mean(a11_path[1:(Te ÷ 4)])
end

# ─────────────────────────────────────────────────────────────────────────────
# Time-varying impulse responses
# ─────────────────────────────────────────────────────────────────────────────

@testset "time-varying IRFs differ across dates and have usable bands" begin
    Y = _tvp_sim_break(FAST ? 200 : 300; scale=3.0)
    post = estimate_tvpvar(Y, 1; n_draws=FAST ? 100 : 200, n_burn=FAST ? 100 : 200,
                           varnames=["y1", "y2"], rng=Random.MersenneTwister(21))

    early = irf(post, 8; t=5, n_draws=100)
    late = irf(post, 8; t=post.T_eff, n_draws=100)

    @test early isa MacroEconometricModels.BayesianImpulseResponse{Float64}
    @test size(early.point_estimate) == (8, 2, 2)
    @test early.variables == ["y1", "y2"]
    @test occursin("y1", early.shocks[1])

    # Bands finite and ordered
    for r in (early, late)
        @test all(isfinite, r.quantiles)
        @test all(r.quantiles[:, :, :, 1] .<= r.quantiles[:, :, :, end])
        @test r.n_effective > 0
        @test r.n_effective + r.n_failed == r.n_requested
    end

    # The volatility break makes the late impact response much larger
    @test late.point_estimate[1, 1, 1] > 2 * early.point_estimate[1, 1, 1]

    # The impact matrix is lower triangular by construction (A_t recursive ordering),
    # so the first variable does not respond to the second shock on impact.
    @test abs(early.point_estimate[1, 1, 2]) < 1e-10
    @test abs(late.point_estimate[1, 1, 2]) < 1e-10

    @test_throws ArgumentError irf(post, 0)
    @test_throws ArgumentError irf(post, 8; t=0)
    @test_throws ArgumentError irf(post, 8; t=post.T_eff + 1)
end

# ─────────────────────────────────────────────────────────────────────────────
# Interface
# ─────────────────────────────────────────────────────────────────────────────

@testset "reproducibility, validation and display" begin
    Y = _tvp_sim_const(200)

    a = estimate_tvpvar(Y, 1; n_draws=40, n_burn=40, rng=Random.MersenneTwister(77))
    b = estimate_tvpvar(Y, 1; n_draws=40, n_burn=40, rng=Random.MersenneTwister(77))
    @test a.B_draws == b.B_draws
    @test a.H_draws == b.H_draws

    # Shapes
    n, p = 2, 1
    k = n * (1 + n * p)
    @test size(a.B_draws) == (40, a.T_eff, k)
    @test size(a.A_draws) == (40, a.T_eff, n * (n - 1) ÷ 2)
    @test size(a.H_draws) == (40, a.T_eff, n)
    @test size(a.Q_draws) == (40, k, k)
    @test size(a.W_draws) == (40, n)
    @test _M.n_draws(a) == 40

    # thin keeps the requested number of draws
    th = estimate_tvpvar(Y, 1; n_draws=20, n_burn=20, thin=2, rng=Random.MersenneTwister(7))
    @test size(th.B_draws, 1) == 20

    @test_throws ArgumentError estimate_tvpvar(Y, 0)
    @test_throws ArgumentError estimate_tvpvar(Y, 1; n_draws=0)
    @test_throws ArgumentError estimate_tvpvar(Y, 1; thin=0)
    @test_throws ArgumentError estimate_tvpvar(Y, 1; n_train=size(Y, 1) + 5)
    @test_throws ArgumentError estimate_tvpvar(Y, 1; n_train=3)
    @test_throws ArgumentError estimate_tvpvar(Y, 1; varnames=["only_one"])
    @test_throws ArgumentError estimate_tvpvar(reshape(Y[:, 1], :, 1), 1)

    out = sprint(show, a)
    @test occursin("Time-Varying Parameter VAR", out)
    @test occursin("Primiceri", out)
    @test occursin("Posterior mean", out)
    @test report(a) === nothing

    cs = estimate_tvpvar(Y, 1; tvp=false, n_draws=20, n_burn=20,
                         rng=Random.MersenneTwister(5))
    @test occursin("Cogley-Sargent", sprint(show, cs))
end

end  # @testset "TVP-VAR with stochastic volatility"
