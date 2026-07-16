# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# Tests for the nonlinear time series scaffold (EV-05 / #413):
# two-regime threshold LS, SETAR, Hansen (1996) linearity test, Hansen (2000)
# threshold confidence interval.
#
# Oracle discipline
# -----------------
# The issue names R `tsDyn::setar` as the cross-implementation oracle, but tsDyn
# is not installed in this environment. Primary in-env validation is therefore:
#   (1) analytic RECOVERY property — on a fixed-seed, large-n, low-noise
#       SETAR(2;1,1) DGP the SSR-grid minimiser recovers the true γ and per-regime
#       β within a seeded tolerance (grid over order statistics of q gives the
#       unique concentrated-SSR minimiser);
#   (2) KNOWN-VALUE oracle — the Hansen (2000, Table 1) constants
#       c(.90)=5.94, c(.95)=7.35, c(.99)=10.59 asserted exactly, and cross-checked
#       against their closed form (they solve (1−exp(−c/2))² = α);
#   (3) MONTE-CARLO analytic property for the linearity test — fixed-regressor
#       bootstrap p-values reject on the SETAR DGP and do not over-reject on a
#       linear-AR null.
# No reference numerics are invented.

using Test
using MacroEconometricModels
using Random
using Statistics
using LinearAlgebra

# -----------------------------------------------------------------------------
# Fixed-seed DGPs
# -----------------------------------------------------------------------------

"""Simulate a two-regime SETAR(2;1,1): switch on y_{t-1} ≤ γ."""
function _sim_setar(; n::Int, gamma::Float64=0.0,
                    b1=(0.5, 0.6), b2=(-0.3, -0.4), sigma::Float64=0.3,
                    seed::Int=20240716)
    rng = MersenneTwister(seed)
    y = zeros(n)
    for t in 2:n
        if y[t-1] <= gamma
            y[t] = b1[1] + b1[2] * y[t-1] + sigma * randn(rng)
        else
            y[t] = b2[1] + b2[2] * y[t-1] + sigma * randn(rng)
        end
    end
    return y
end

"""Simulate a linear AR(1) (the linearity-test null)."""
function _sim_ar1(; n::Int, phi::Float64=0.4, sigma::Float64=1.0, seed::Int=1)
    rng = MersenneTwister(seed)
    y = zeros(n)
    for t in 2:n
        y[t] = phi * y[t-1] + sigma * randn(rng)
    end
    return y
end

@testset "Nonlinear: Threshold / SETAR (EV-05)" begin

    # =========================================================================
    # (1) Analytic recovery oracle — SSR grid recovers γ and per-regime β
    # =========================================================================
    @testset "SETAR recovery on fixed-seed DGP" begin
        y = _sim_setar(; n=3000, gamma=0.0, b1=(0.5, 0.6), b2=(-0.3, -0.4),
                       sigma=0.3, seed=20240716)
        m = estimate_setar(y, 1, 1; linearity=false)

        @test m isa ThresholdModel
        @test m isa MacroEconometricModels.AbstractNonlinearTSModel
        @test m.is_setar
        @test m.p == 1 && m.d == 1
        # True threshold is 0.0; the SSR minimiser must land in a tight band.
        @test isapprox(m.gamma, 0.0; atol=0.08)
        # Per-regime coefficients recover the DGP within a seeded tolerance.
        @test isapprox(m.beta1[1], 0.5; atol=0.06)
        @test isapprox(m.beta1[2], 0.6; atol=0.06)
        @test isapprox(m.beta2[1], -0.3; atol=0.06)
        @test isapprox(m.beta2[2], -0.4; atol=0.06)
        # Pooled SSR equals the sum of regime SSRs, residual decomposition holds.
        @test isapprox(m.ssr, m.ssr1 + m.ssr2; rtol=1e-12)
        @test m.n1 + m.n2 == m.n
        @test isapprox(m.sigma2, m.ssr / m.n; rtol=1e-12)
        # Threshold estimate lies inside its own CI.
        @test m.gamma_ci[1] <= m.gamma <= m.gamma_ci[2]
    end

    # =========================================================================
    # Concentrated SSR at γ̂ is the grid minimum (definition of the estimator)
    # =========================================================================
    @testset "γ̂ minimises the concentrated SSR over the grid" begin
        y = _sim_setar(; n=800, seed=99)
        m = estimate_setar(y, 1, 1; linearity=false)
        yy, X, q = MacroEconometricModels._setar_design(Float64.(y), 1, 1, 1)
        grid = MacroEconometricModels._threshold_grid(q, 0.15)
        smin = Inf
        for g in grid
            s, _, ok = MacroEconometricModels._split_ssr(yy, X, q, g; min_obs=2)
            ok && (smin = min(smin, s))
        end
        @test isapprox(m.ssr, smin; rtol=1e-10)
        @test all(g -> begin
            s, _, ok = MacroEconometricModels._split_ssr(yy, X, q, g; min_obs=2)
            !ok || s >= m.ssr - 1e-10
        end, grid)
    end

    # =========================================================================
    # (2) Hansen (2000) tabulated critical constants — asserted EXACTLY
    # =========================================================================
    @testset "Hansen (2000) critical constants 5.94 / 7.35 / 10.59" begin
        @test MacroEconometricModels._hansen2000_crit(0.90) == 5.94
        @test MacroEconometricModels._hansen2000_crit(0.95) == 7.35
        @test MacroEconometricModels._hansen2000_crit(0.99) == 10.59
        @test MacroEconometricModels.HANSEN2000_CRIT == (var"0.90"=5.94, var"0.95"=7.35, var"0.99"=10.59)
        @test_throws ArgumentError MacroEconometricModels._hansen2000_crit(0.80)
        # Cross-check against the closed form: c(α) solves (1 − exp(−c/2))² = α.
        cdf(x) = (1 - exp(-x/2))^2
        @test isapprox(cdf(5.94), 0.90; atol=1e-3)
        @test isapprox(cdf(7.35), 0.95; atol=1e-3)
        @test isapprox(cdf(10.59), 0.99; atol=1e-3)
    end

    # =========================================================================
    # Hansen (2000) CI properties
    # =========================================================================
    @testset "Hansen (2000) threshold CI" begin
        y = _sim_setar(; n=400, seed=3, b1=(1.0, 0.5), b2=(-1.0, -0.5))
        m90 = estimate_setar(y, 1, 1; linearity=false, ci_level=0.90)
        m95 = estimate_setar(y, 1, 1; linearity=false, ci_level=0.95)
        m99 = estimate_setar(y, 1, 1; linearity=false, ci_level=0.99)
        # γ̂ inside every CI; nested CIs widen with the confidence level.
        for m in (m90, m95, m99)
            @test m.gamma_ci[1] <= m.gamma <= m.gamma_ci[2]
        end
        @test m90.gamma_ci[1] >= m95.gamma_ci[1] - 1e-9
        @test m90.gamma_ci[2] <= m95.gamma_ci[2] + 1e-9
        @test m95.gamma_ci[1] >= m99.gamma_ci[1] - 1e-9
        @test m95.gamma_ci[2] <= m99.gamma_ci[2] + 1e-9
        # Heteroskedasticity correction runs and returns a finite interval.
        mh = estimate_setar(y, 1, 1; linearity=false, ci_level=0.95, het=true)
        @test all(isfinite, mh.gamma_ci)
        @test mh.gamma_ci[1] <= mh.gamma <= mh.gamma_ci[2]
    end

    # =========================================================================
    # (3) Hansen (1996) linearity test — rejects SETAR, size on linear null
    # =========================================================================
    @testset "Linearity test rejects the SETAR DGP" begin
        y = _sim_setar(; n=400, seed=3, b1=(1.0, 0.5), b2=(-1.0, -0.5))
        X = hcat(ones(length(y)-1), y[1:end-1])
        lt = hansen_linearity_test(y[2:end], X, y[1:end-1]; reps=200,
                                   rng=MersenneTwister(11))
        @test lt isa HansenLinearityTest
        @test lt.sup_lm > 0
        @test lt.sup_wald > 0
        @test 0.0 <= lt.pvalue_lm <= 1.0
        @test lt.pvalue_lm < 0.05     # strong regime switch ⇒ reject linearity
        @test lt.pvalue_wald < 0.05
    end

    @testset "Linearity test does not over-reject a linear AR(1) null" begin
        # Single fixed-seed linear draw should NOT reject at 5%.
        y = _sim_ar1(; n=300, phi=0.4, seed=7)
        X = hcat(ones(length(y)-1), y[1:end-1])
        lt = hansen_linearity_test(y[2:end], X, y[1:end-1]; reps=300,
                                   rng=MersenneTwister(11))
        @test lt.pvalue_lm > 0.05

        # Small Monte-Carlo: bootstrap p-values must not systematically over-reject.
        MC = FAST ? 12 : 30
        rej10 = 0
        below = Float64[]
        for s in 1:MC
            yy = _sim_ar1(; n=140, phi=0.4, seed=1000 + s)
            Xs = hcat(ones(length(yy)-1), yy[1:end-1])
            l = hansen_linearity_test(yy[2:end], Xs, yy[1:end-1]; reps=99,
                                      rng=MersenneTwister(500 + s))
            push!(below, l.pvalue_lm)
            l.pvalue_lm < 0.10 && (rej10 += 1)
        end
        # Nominal rejection at 10% is ~0.10; allow generous MC slack but flag gross
        # over-rejection (a broken bootstrap would reject almost always).
        @test rej10 / MC <= 0.40
        @test mean(below) > 0.15
    end

    # =========================================================================
    # SETAR auto-delay selection
    # =========================================================================
    @testset "SETAR :auto delay selection" begin
        # DGP switches on y_{t-2}; :auto should prefer d = 2.
        rng = MersenneTwister(42)
        n = 1500
        y = zeros(n)
        for t in 3:n
            if y[t-2] <= 0.0
                y[t] = 0.6 + 0.5 * y[t-1] + 0.3 * randn(rng)
            else
                y[t] = -0.6 - 0.3 * y[t-1] + 0.3 * randn(rng)
            end
        end
        m = estimate_setar(y, 2, :auto; linearity=false)
        @test m.d == 2
        mr = estimate_setar(y, 2, 1:2; linearity=false)
        @test mr.d == 2
    end

    # =========================================================================
    # Generic two-regime threshold API + input validation
    # =========================================================================
    @testset "estimate_threshold generic API and validation" begin
        y = _sim_setar(; n=600, seed=5)
        X = hcat(ones(length(y)-1), y[1:end-1])
        q = y[1:end-1]
        m = estimate_threshold(y[2:end], X, q; linearity=false)
        @test m isa ThresholdModel
        @test !m.is_setar
        @test size(m.X, 2) == 2
        @test length(m.regime) == m.n
        @test count(m.regime) == m.n1

        # trim out of range
        @test_throws ArgumentError estimate_threshold(y[2:end], X, q; trim=0.6, linearity=false)
        @test_throws ArgumentError estimate_threshold(y[2:end], X, q; trim=0.0, linearity=false)
        # dimension mismatch
        @test_throws DimensionMismatch estimate_threshold(y[2:end], X, q[1:end-1]; linearity=false)
        # too few observations for two 2-regressor regimes (need ≥ 2·(k+1) = 6)
        yt = randn(MersenneTwister(1), 5); Xt = hcat(ones(5), randn(MersenneTwister(2), 5))
        @test_throws ArgumentError estimate_threshold(yt, Xt, yt; trim=0.15, linearity=false)
        # SETAR order guard
        @test_throws ArgumentError estimate_setar(y, 0)
    end

    # =========================================================================
    # Forecast (bootstrap simulation)
    # =========================================================================
    @testset "SETAR bootstrap forecast" begin
        y = _sim_setar(; n=800, seed=8)
        m = estimate_setar(y, 1, 1; linearity=false)
        f = forecast(m, 8; reps=500, level=0.90, rng=MersenneTwister(5))
        @test f isa ThresholdForecast
        @test f.horizon == 8
        @test length(f.forecast) == 8
        @test length(f.ci_lower) == 8 == length(f.ci_upper)
        @test all(f.ci_lower .<= f.forecast .+ 1e-8)
        @test all(f.forecast .- 1e-8 .<= f.ci_upper)
        @test all(f.se .>= 0)
        # AbstractForecastResult accessors
        @test point_forecast(f) == f.forecast
        @test forecast_horizon(f) == 8
        # generic threshold cannot forecast
        X = hcat(ones(length(y)-1), y[1:end-1])
        mg = estimate_threshold(y[2:end], X, y[1:end-1]; linearity=false)
        @test_throws ArgumentError forecast(mg, 5)
        @test_throws ArgumentError forecast(m, 0)
    end

    # =========================================================================
    # Information criteria and StatsAPI accessors
    # =========================================================================
    @testset "AIC/BIC and StatsAPI accessors" begin
        y = _sim_setar(; n=500, seed=13)
        m = estimate_setar(y, 1, 1; linearity=false)
        npar = 2 * m.k + 1
        ll = -m.n/2 * (log(2π) + log(m.sigma2) + 1)
        @test isapprox(m.aic, -2ll + 2npar; rtol=1e-10)
        @test isapprox(m.bic, -2ll + log(m.n)*npar; rtol=1e-10)
        @test nobs(m) == m.n
        @test length(residuals(m)) == m.n
        @test length(coef(m)) == 2 * m.k
        @test dof(m) == npar
    end

    # =========================================================================
    # Display, refs, plotting smoke tests
    # =========================================================================
    @testset "report / refs / plot_result" begin
        y = _sim_setar(; n=400, seed=21)
        m = estimate_setar(y, 1, 1; reps=100, rng=MersenneTwister(3))
        @test m.linearity !== nothing

        io = IOBuffer()
        report(io, m)
        s = String(take!(io))
        @test occursin("SETAR", s)
        @test occursin("Regime 1", s)
        @test occursin("Regime 2", s)
        @test occursin("Hansen", s)

        # refs render for the model and the linearity test
        rio = IOBuffer(); refs(rio, m)
        rs = String(take!(rio))
        @test occursin("Hansen", rs)
        @test occursin("Tong", rs)
        rio2 = IOBuffer(); refs(rio2, m.linearity)
        @test occursin("Hansen", String(take!(rio2)))

        # forecast show
        f = forecast(m, 5; reps=200, rng=MersenneTwister(1))
        fio = IOBuffer(); show(fio, f)
        @test occursin("Forecast", String(take!(fio)))

        # both plot views produce non-empty HTML
        p1 = plot_result(m; view=:regimes)
        @test p1 isa MacroEconometricModels.PlotOutput
        @test occursin("svg", p1.html)
        p2 = plot_result(m; view=:ssr)
        @test occursin("svg", p2.html)
        @test_throws ArgumentError plot_result(m; view=:bogus)
    end

end
