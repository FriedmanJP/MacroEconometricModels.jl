# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# T252 (#351): Giannone-Lenza-Primiceri hierarchical hyperparameter optimization.

using Test
using MacroEconometricModels
using LinearAlgebra
using Random
using Statistics
using Distributions

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

const _M = MacroEconometricModels

# DGP-03 (#792): the file's historical design (non-diagonal dynamics,
# non-identity impact so shock orderings matter) on the shared simulator.
const _GLP_A = [0.6 0.1; 0.15 0.5]
const _GLP_B0 = [0.4 0.0; 0.1 0.3]

function _glp_sim(rng::AbstractRNG, T_obs::Int)
    return dgp_var(rng; A=_GLP_A, B0=_GLP_B0, T=T_obs).Y
end

"""Genuine random-walk panel (unit-root DGP) for the boundary-pinning test."""
function _glp_rw(rng::AbstractRNG)
    return dgp_var(rng; A=Matrix{Float64}(I, 3, 3), T=150).Y
end

@testset "GLP hierarchical hyperparameter optimization" begin

# ─────────────────────────────────────────────────────────────────────────────
# Hyperprior parameterization
# ─────────────────────────────────────────────────────────────────────────────

@testset "Gamma mode/sd inversion is exact" begin
    for (m, s) in ((0.2, 0.4), (1.0, 1.0), (2.0, 0.5), (0.05, 0.1))
        k, theta = _M._gamma_from_mode_sd(m, s)
        g = Gamma(k, theta)
        @test mode(g) ≈ m atol = 1e-10
        @test std(g) ≈ s atol = 1e-10
        @test k > 1                       # a mode exists only for shape > 1
    end
    @test_throws ArgumentError _M._gamma_from_mode_sd(0.0, 1.0)
    @test_throws ArgumentError _M._gamma_from_mode_sd(1.0, 0.0)
end

@testset "hyperprior density is the sum of the three GLP Gammas" begin
    tau, lambda, mu = 0.3, 1.2, 0.8
    got = _M._glp_log_hyperprior(tau, lambda, mu, Float64)

    expected = 0.0
    for (val, spec) in ((tau, _M._GLP_HYPERPRIORS.tau),
                        (lambda, _M._GLP_HYPERPRIORS.lambda),
                        (mu, _M._GLP_HYPERPRIORS.mu))
        k, th = _M._gamma_from_mode_sd(spec.mode, spec.sd)
        expected += logpdf(Gamma(k, th), val)
    end
    @test got ≈ expected atol = 1e-12

    # GLP's published hyperprior settings
    @test _M._GLP_HYPERPRIORS.tau == (mode = 0.2, sd = 0.4)
    @test _M._GLP_HYPERPRIORS.lambda == (mode = 1.0, sd = 1.0)
    @test _M._GLP_HYPERPRIORS.mu == (mode = 1.0, sd = 1.0)

    # Non-positive hyperparameters are outside the support
    @test _M._glp_log_hyperprior(-1.0, 1.0, 1.0, Float64) == -Inf
end

@testset "objective is the negative log posterior, penalized outside the box" begin
    Y = _glp_sim(Random.MersenneTwister(2), 120)
    x = log.([0.3, 1.2, 0.8])
    got = _M._glp_objective(x, Y, 2, 0.5, 2.0, Float64)

    h = MinnesotaHyperparameters(; tau=0.3, decay=0.5, lambda=1.2, mu=0.8, omega=2.0)
    expected = -(log_marginal_likelihood(Y, 2, h) +
                 _M._glp_log_hyperprior(0.3, 1.2, 0.8, Float64))
    @test got ≈ expected atol = 1e-10

    # Outside the search box the objective is a large finite penalty, so the optimizer is
    # repelled rather than stepping into a region where the marginal likelihood is undefined
    lo = _M._GLP_BOUNDS.tau[1]
    hi = _M._GLP_BOUNDS.tau[2]
    @test _M._glp_objective(log.([lo / 2, 1.0, 1.0]), Y, 2, 0.5, 2.0, Float64) == 1e12
    @test _M._glp_objective(log.([hi * 2, 1.0, 1.0]), Y, 2, 0.5, 2.0, Float64) == 1e12
end

# ─────────────────────────────────────────────────────────────────────────────
# The optimizer
# ─────────────────────────────────────────────────────────────────────────────

@testset "joint optimization beats the tau-only grid and the defaults" begin
    Y = _glp_sim(Random.MersenneTwister(1), FAST ? 120 : 200)
    p = 2
    r = optimize_hyperparameters_glp(Y, p)

    @test r isa GLPHyperparameters{Float64}
    @test r.converged
    @test !r.at_bound
    @test r.hyper isa MinnesotaHyperparameters{Float64}
    @test all(>(0), (r.hyper.tau, r.hyper.lambda, r.hyper.mu))

    # AC: the joint optimizer attains at least the marginal likelihood the tau-only grid does
    hg = optimize_hyperparameters(Y, p)
    ml_grid = log_marginal_likelihood(Y, p, hg)
    @test r.log_ml >= ml_grid - 1e-6

    # ... and both beat the package defaults
    @test r.log_ml >= r.log_ml_default
    @test r.log_ml_default ≈ log_marginal_likelihood(Y, p, MinnesotaHyperparameters()) atol = 1e-10

    # log_posterior is the maximized objective: log ML plus the hyperprior
    @test r.log_posterior ≈ r.log_ml +
        _M._glp_log_hyperprior(r.hyper.tau, r.hyper.lambda, r.hyper.mu, Float64) atol = 1e-6

    # The unoptimized hyperparameters are passed through untouched
    @test r.hyper.decay == 0.5
    @test r.hyper.omega == 1.0
    r2 = optimize_hyperparameters_glp(Y, p; decay=1.5, omega=3.0)
    @test r2.hyper.decay == 1.5 && r2.hyper.omega == 3.0

    # Deterministic: no RNG enters the optimizer
    @test optimize_hyperparameters_glp(Y, p).hyper.tau == r.hyper.tau
end

@testset "a pinned hyperparameter is never reported as converged" begin
    # THE invariant, and it cannot depend on any particular draw: whenever a fit
    # lands on the edge of the search box it must NOT be reported as converged.
    # Assert it over a sweep rather than on one dataset.
    n_pinned = 0
    for seed in 1:25
        rc = optimize_hyperparameters_glp(
            _glp_rw(Random.MersenneTwister(seed)), 2; verbose=false)
        rc.at_bound && (n_pinned += 1; @test !rc.converged)
    end

    # Random-walk data drives the overall tightness to the edge of the box —
    # but only for some draws, and WHICH draws is a property of the RNG stream,
    # which is not stable across Julia versions. So search for a pinning draw
    # instead of hard-coding a seed, which makes the end-to-end assertions below
    # independent of the stream.
    Y_rw = nothing
    r = nothing
    for seed in 1:40
        Yc = _glp_rw(Random.MersenneTwister(seed))
        rc = optimize_hyperparameters_glp(Yc, 2; verbose=false)
        if rc.at_bound
            Y_rw, r = Yc, rc
            break
        end
    end
    @test r !== nothing                # a near-RW panel must pin for SOME draw
    @test n_pinned > 0

    @test r.at_bound
    @test !r.converged                 # THE point: a boundary value is not a selection
    # `at_bound` fires on whichever hyperparameter hit the box; on this DGP it is
    # the overall tightness, but assert the general property.
    @test any(v <= b[1] * (1 + 1e-6) || v >= b[2] * (1 - 1e-6)
              for (v, b) in ((r.hyper.tau, _M._GLP_BOUNDS.tau),
                             (r.hyper.lambda, _M._GLP_BOUNDS.lambda),
                             (r.hyper.mu, _M._GLP_BOUNDS.mu)))

    # ... and it warns when asked to
    @test_logs (:warn,) match_mode = :any optimize_hyperparameters_glp(Y_rw, 2)

    out = sprint(show, r)
    @test occursin("GLP (2015) Hyperparameter Optimization", out)
    @test occursin("did NOT converge", out)
    @test occursin("pinned to a bound", out)
    @test report(r) === nothing

    # The tau-only grid returns its own endpoint with no flag at all — the pathology this
    # task removes. (It reports a value; only the GLP path can say it is not a selection.)
    hg = optimize_hyperparameters(Y_rw, 2)
    @test hg.tau ≈ 0.01 atol = 1e-8    # the grid's lower endpoint

    good = optimize_hyperparameters_glp(_glp_sim(Random.MersenneTwister(1), 200), 2)
    @test occursin("Converged", sprint(show, good))
    @test !occursin("did NOT converge", sprint(show, good))
end

@testset "optimizer keywords" begin
    Y = _glp_sim(Random.MersenneTwister(3), 100)
    @test optimize_hyperparameters_glp(Y, 1; starts=1).converged isa Bool
    @test optimize_hyperparameters_glp(Y, 1; max_iter=50) isa GLPHyperparameters
    @test_throws ArgumentError optimize_hyperparameters_glp(Y, 0)
    @test_throws ArgumentError optimize_hyperparameters_glp(Y, 1; starts=0)
    # More restarts can only improve the attained objective
    r1 = optimize_hyperparameters_glp(Y, 1; starts=1)
    r4 = optimize_hyperparameters_glp(Y, 1; starts=4)
    @test r4.log_posterior >= r1.log_posterior - 1e-8
end

# ─────────────────────────────────────────────────────────────────────────────
# Wiring into estimate_bvar
# ─────────────────────────────────────────────────────────────────────────────

@testset "estimate_bvar defaults to GLP, with the grid path unchanged" begin
    Y = _glp_sim(Random.MersenneTwister(5), 150)
    p = 2

    # The default path equals passing the GLP-selected hyperparameters explicitly
    glp_h = optimize_hyperparameters_glp(Y, p).hyper
    a = estimate_bvar(Y, p; prior=:minnesota, n_draws=50, rng=Random.MersenneTwister(1))
    b = estimate_bvar(Y, p; prior=:minnesota, hyper=glp_h, n_draws=50,
                      rng=Random.MersenneTwister(1))
    @test a.B_draws ≈ b.B_draws atol = 1e-12

    # hyperopt=:grid reproduces the historical tau-only path exactly
    grid_h = optimize_hyperparameters(Y, p)
    c = estimate_bvar(Y, p; prior=:minnesota, hyperopt=:grid, n_draws=50,
                      rng=Random.MersenneTwister(1))
    d = estimate_bvar(Y, p; prior=:minnesota, hyper=grid_h, n_draws=50,
                      rng=Random.MersenneTwister(1))
    @test c.B_draws ≈ d.B_draws atol = 1e-12

    # The two selection paths genuinely differ
    @test !isapprox(a.B_draws, c.B_draws; atol=1e-8)

    # Truth recovery (DGP-03 #792): the posterior mean recovers the known design.
    Bm = dropdims(mean(a.B_draws; dims=1); dims=1)
    @test maximum(abs, Matrix(Bm[2:3, :]') - _GLP_A) < 0.2

    # An explicit hyper bypasses selection under either setting
    fixed = MinnesotaHyperparameters(; tau=0.7, lambda=2.0, mu=1.5)
    e = estimate_bvar(Y, p; prior=:minnesota, hyper=fixed, hyperopt=:grid, n_draws=50,
                      rng=Random.MersenneTwister(1))
    f = estimate_bvar(Y, p; prior=:minnesota, hyper=fixed, n_draws=50,
                      rng=Random.MersenneTwister(1))
    @test e.B_draws ≈ f.B_draws atol = 1e-12

    # A non-Minnesota prior is untouched by the hyperparameter machinery
    g1 = estimate_bvar(Y, p; n_draws=50, rng=Random.MersenneTwister(1))
    g2 = estimate_bvar(Y, p; hyperopt=:grid, n_draws=50, rng=Random.MersenneTwister(1))
    @test g1.B_draws ≈ g2.B_draws atol = 1e-12

    @test_throws ArgumentError estimate_bvar(Y, p; hyperopt=:bogus)
end

end  # @testset "GLP hierarchical hyperparameter optimization"
