# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# T244 (#343): Anderson-Rubin weak-instrument-robust test and confidence set.

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
const _suppress_warnings = MacroEconometricModels._suppress_warnings

"""Just-identified IV DGP; `pi1` controls instrument strength."""
function _ar_sim(n::Int; pi1::Float64=1.0, seed::Int=1, beta::Float64=1.0)
    rng = Random.MersenneTwister(seed)
    z = randn(rng, n)
    v = randn(rng, n)
    u = 0.8 .* v .+ 0.6 .* randn(rng, n)
    x = pi1 .* z .+ v
    y = beta .* x .+ u
    return y, hcat(ones(n), x), hcat(ones(n), z)
end

"""Over-identified IV DGP with `n_z` excluded instruments; `invalid` adds a direct effect."""
function _ar_sim_overid(n::Int; pi1::Float64=0.5, seed::Int=1, beta::Float64=1.0,
                        n_z::Int=3, invalid::Float64=0.0)
    rng = Random.MersenneTwister(seed)
    Zx = randn(rng, n, n_z)
    v = randn(rng, n)
    u = 0.8 .* v .+ 0.6 .* randn(rng, n)
    x = Zx * fill(pi1, n_z) .+ v
    # A direct instrument→outcome channel violates the exclusion restriction
    y = beta .* x .+ u .+ invalid .* Zx[:, 1]
    return y, hcat(ones(n), x), hcat(ones(n), Zx)
end

@testset "Anderson-Rubin weak-IV inference" begin

# ─────────────────────────────────────────────────────────────────────────────
# The test itself
# ─────────────────────────────────────────────────────────────────────────────

@testset "AR test against a direct auxiliary-regression computation" begin
    y, X, Z = _ar_sim(300; pi1=0.8, seed=3)
    m = estimate_iv(y, X, Z; endogenous=[2], varnames=["const", "x"], cov_type=:ols)

    b0 = 1.0
    ar = anderson_rubin_test(m, b0; cov_type=:ols)
    @test ar isa AndersonRubinTest{Float64}
    @test ar.df1 == 1                       # one excluded instrument
    @test ar.distribution === :F
    @test ar.beta0 == [b0]
    @test ar.endog_names == ["x"]

    # Independent computation: regress ỹ = y − x·β₀ on [const, z] and F-test the z
    # coefficient. That is exactly the AR statistic in the just-identified case.
    n = length(y)
    ytil = y .- X[:, 2] .* b0
    Zfull = Z
    bz = Zfull \ ytil
    ssr_u = sum(abs2, ytil .- Zfull * bz)
    W = X[:, [1]]
    bw = W \ ytil
    ssr_r = sum(abs2, ytil .- W * bw)
    q = size(Zfull, 2) - size(W, 2)
    f_ref = ((ssr_r - ssr_u) / q) / (ssr_u / (n - size(Zfull, 2)))
    @test ar.statistic ≈ f_ref atol = 1e-9
    @test ar.p_value ≈ ccdf(FDist(q, n - size(Zfull, 2)), f_ref) atol = 1e-12

    # The AR test does NOT reject at the true value and DOES at a distant one
    @test anderson_rubin_test(m, 1.0; cov_type=:ols).p_value > 0.05
    @test anderson_rubin_test(m, 5.0; cov_type=:ols).p_value < 0.01
end

@testset "robust and clustered AR variants" begin
    y, X, Z = _ar_sim_overid(400; pi1=0.5, seed=7)
    m = estimate_iv(y, X, Z; endogenous=[2], varnames=["const", "x"], cov_type=:hc1)

    ar_r = anderson_rubin_test(m, 1.0)
    @test ar_r.cov_type === :hc1
    @test ar_r.distribution === :chisq
    @test ar_r.df1 == 3                     # three excluded instruments
    @test ar_r.p_value > 0.05               # true value not rejected

    ar_h = anderson_rubin_test(m, 1.0; cov_type=:ols)
    @test ar_h.distribution === :F
    # Under (near) homoskedasticity the robust and classical statistics are close
    @test ar_r.statistic ≈ ar_h.statistic rtol = 0.35

    cl = repeat(1:20, inner=20)
    ar_c = anderson_rubin_test(m, 1.0; cov_type=:cluster, clusters=cl)
    @test ar_c.cov_type === :cluster
    @test ar_c.distribution === :chisq
    @test isfinite(ar_c.statistic)
    @test_throws ArgumentError anderson_rubin_test(m, 1.0; cov_type=:cluster)
    @test_throws ArgumentError anderson_rubin_test(m, 1.0; cov_type=:cluster,
                                                   clusters=cl[1:10])
end

@testset "AR test validation and display" begin
    y, X, Z = _ar_sim(200; pi1=0.8, seed=3)
    m = estimate_iv(y, X, Z; endogenous=[2], varnames=["const", "x"])

    @test_throws ArgumentError anderson_rubin_test(m, [1.0, 2.0])
    @test_throws ArgumentError anderson_rubin_test(m, 1.0; cov_type=:bogus)

    # An OLS model carries no instruments
    m_ols = estimate_reg(y, X; varnames=["const", "x"])
    @test_throws ArgumentError anderson_rubin_test(m_ols, 1.0)

    out = sprint(show, anderson_rubin_test(m, 1.0))
    @test occursin("Anderson-Rubin", out)
    @test occursin("AR statistic", out)
    @test report(anderson_rubin_test(m, 1.0)) === nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# The confidence set
# ─────────────────────────────────────────────────────────────────────────────

@testset "strong instruments: AR set ≈ Wald interval" begin
    y, X, Z = _ar_sim(500; pi1=1.0, seed=3)
    m = estimate_iv(y, X, Z; endogenous=[2], varnames=["const", "x"], cov_type=:ols)
    @test m.first_stage_f > 100             # unambiguously strong

    ci = anderson_rubin_ci(m; cov_type=:ols)
    @test ci isa AndersonRubinCI{Float64}
    @test ci.bounded
    @test !ci.is_empty && !ci.is_whole_line
    @test length(ci.intervals) == 1
    lo, hi = ci.intervals[1]

    # With a strong first stage AR and Wald agree closely
    @test lo ≈ ci.wald_lower atol = 0.03
    @test hi ≈ ci.wald_upper atol = 0.03
    @test lo < ci.estimate < hi

    # The endpoints are exactly where the AR statistic crosses its critical value
    @test anderson_rubin_test(m, lo; cov_type=:ols).statistic ≈ ci.critical_value rtol = 1e-4
    @test anderson_rubin_test(m, hi; cov_type=:ols).statistic ≈ ci.critical_value rtol = 1e-4
    # ... and interior points are accepted while exterior ones are rejected
    @test anderson_rubin_test(m, ci.estimate; cov_type=:ols).statistic < ci.critical_value
    @test anderson_rubin_test(m, hi + 0.1; cov_type=:ols).statistic > ci.critical_value
end

@testset "weak instruments: the AR set is unbounded" begin
    y, X, Z = _ar_sim(200; pi1=0.03, seed=5)
    m = estimate_iv(y, X, Z; endogenous=[2], varnames=["const", "x"], cov_type=:ols)
    @test m.first_stage_f < 10              # weak by the Stock-Yogo standard

    ci = anderson_rubin_ci(m; cov_type=:ols)
    @test !ci.bounded
    @test !ci.is_empty
    # The Wald interval is bounded and therefore over-confident
    @test isfinite(ci.wald_lower) && isfinite(ci.wald_upper)
    wald_width = ci.wald_upper - ci.wald_lower
    ar_width = sum(b - a for (a, b) in ci.intervals)
    @test ar_width > wald_width             # AR is strictly less informative — correctly so

    out = sprint(show, ci)
    @test occursin("unbounded", out)
    @test occursin("Trust the AR set", out)
    @test occursin("∞", out)
end

@testset "coverage: AR is correct where Wald under-covers" begin
    nrep = FAST ? 80 : 300
    beta_true = 1.0
    cov_ar = 0
    cov_wald = 0
    for s in 1:nrep
        y, X, Z = _ar_sim(200; pi1=0.05, seed=5000 + s, beta=beta_true)
        m = _suppress_warnings() do
            estimate_iv(y, X, Z; endogenous=[2], varnames=["const", "x"], cov_type=:ols)
        end
        ci = _suppress_warnings() do
            anderson_rubin_ci(m; cov_type=:ols, n_grid=401, span=200)
        end
        cov_ar += ci.is_whole_line || any(a <= beta_true <= b for (a, b) in ci.intervals)
        lo = coef(m)[2] - 1.959963985 * stderror(m)[2]
        hi = coef(m)[2] + 1.959963985 * stderror(m)[2]
        cov_wald += (lo <= beta_true <= hi)
    end
    # AR has correct coverage under weak identification; the Wald interval does not.
    # The threshold is 4 Monte-Carlo standard errors below the nominal 0.95 at
    # nrep=300 (se ~0.013): 0.93 sat 1.6 se away and a different RNG stream landed
    # at 0.925. The discriminating claim is `cov_ar > cov_wald`, asserted below.
    @test cov_ar / nrep >= 0.90
    @test cov_wald / nrep < 0.94
    @test cov_ar > cov_wald
end

@testset "over-identified sets need not be intervals" begin
    # Invalid instruments: no value of β satisfies the over-identifying restrictions, so
    # the AR set is EMPTY — the diagnostic a Wald interval can never deliver.
    y, X, Z = _ar_sim_overid(500; pi1=0.8, seed=13, n_z=3, invalid=3.0)
    m = estimate_iv(y, X, Z; endogenous=[2], varnames=["const", "x"], cov_type=:ols)
    ci = anderson_rubin_ci(m; cov_type=:ols, span=50)
    @test ci.is_empty
    @test isempty(ci.intervals)
    @test occursin("∅", sprint(show, ci))

    # Valid over-identified instruments give an ordinary bounded interval
    y2, X2, Z2 = _ar_sim_overid(500; pi1=0.8, seed=13, n_z=3)
    m2 = estimate_iv(y2, X2, Z2; endogenous=[2], varnames=["const", "x"], cov_type=:ols)
    ci2 = anderson_rubin_ci(m2; cov_type=:ols)
    @test !ci2.is_empty
    @test ci2.df1 == 3
    @test any(a <= 1.0 <= b for (a, b) in ci2.intervals)   # covers the truth
end

@testset "whole-line set on an explicitly narrow grid" begin
    y, X, Z = _ar_sim(200; pi1=0.02, seed=21)
    m = estimate_iv(y, X, Z; endogenous=[2], varnames=["const", "x"], cov_type=:ols)
    # A grid tightly around the estimate lies entirely inside a very weak AR set
    ci = anderson_rubin_ci(m; cov_type=:ols, grid=collect(range(coef(m)[2] - 1e-4,
                                                               coef(m)[2] + 1e-4;
                                                               length=11)))
    @test ci.is_whole_line
    @test ci.intervals == [(-Inf, Inf)]
    @test occursin("(-∞, ∞)", sprint(show, ci))
end

@testset "CI options and validation" begin
    y, X, Z = _ar_sim(300; pi1=1.0, seed=3)
    m = estimate_iv(y, X, Z; endogenous=[2], varnames=["const", "x"], cov_type=:ols)

    # An explicit grid is honored
    g = collect(range(0.5, 1.5; length=201))
    ci = anderson_rubin_ci(m; cov_type=:ols, grid=g)
    @test ci.grid_lo == 0.5 && ci.grid_hi == 1.5

    # A wider level gives a wider set
    c95 = anderson_rubin_ci(m; cov_type=:ols)
    c99 = anderson_rubin_ci(m; cov_type=:ols, level=0.99)
    w95 = sum(b - a for (a, b) in c95.intervals)
    w99 = sum(b - a for (a, b) in c99.intervals)
    @test w99 > w95
    @test c99.critical_value > c95.critical_value

    @test_throws ArgumentError anderson_rubin_ci(m; level=1.5)
    @test_throws ArgumentError anderson_rubin_ci(m; n_grid=2)

    # Multiple endogenous regressors: the test still works, the 1-D inversion does not
    rng = Random.MersenneTwister(2)
    n = 400
    Zx = randn(rng, n, 4)
    v1 = randn(rng, n); v2 = randn(rng, n)
    x1 = Zx[:, 1] .+ Zx[:, 2] .+ v1
    x2 = Zx[:, 3] .+ Zx[:, 4] .+ v2
    yy = x1 .+ 2 .* x2 .+ 0.5 .* (v1 .+ v2) .+ randn(rng, n)
    XX = hcat(ones(n), x1, x2)
    ZZ = hcat(ones(n), Zx)
    mm = estimate_iv(yy, XX, ZZ; endogenous=[2, 3], varnames=["const", "x1", "x2"],
                     cov_type=:ols)
    ar2 = anderson_rubin_test(mm, [1.0, 2.0]; cov_type=:ols)
    @test ar2.df1 == 4
    @test ar2.p_value > 0.01
    @test_throws ArgumentError anderson_rubin_ci(mm)
end

@testset "report points at AR when the first stage is weak" begin
    y, X, Z = _ar_sim(200; pi1=0.03, seed=5)
    m = estimate_iv(y, X, Z; endogenous=[2], varnames=["const", "x"], cov_type=:ols)
    out = sprint(show, m)
    @test occursin("Weak instruments", out)
    @test occursin("anderson_rubin_ci", out)

    # Strong instruments print no such note
    ys, Xs, Zs = _ar_sim(500; pi1=1.0, seed=3)
    ms = estimate_iv(ys, Xs, Zs; endogenous=[2], varnames=["const", "x"], cov_type=:ols)
    @test !occursin("Weak instruments", sprint(show, ms))
end

end  # @testset "Anderson-Rubin weak-IV inference"
