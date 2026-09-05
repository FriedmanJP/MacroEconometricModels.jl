# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# T242 (#341): multiplicative seasonal ARIMA — SARIMA(p,d,q)(P,D,Q)ₛ.

using Test
using MacroEconometricModels
using StatsAPI
using LinearAlgebra
using Random
using Statistics

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

const _M = MacroEconometricModels

"""Simulate the airline model (0,1,1)(0,1,1)ₛ with known θ, Θ."""
function _airline_series(n::Int, s::Int, th::Float64, TH::Float64; seed::Int=7, sigma::Float64=1.0)
    rng = Random.MersenneTwister(seed)
    burn = 50
    N = n + burn
    resid = sigma .* randn(rng, N)
    y = zeros(N)
    start = s + 2
    for t in start:N
        w = resid[t] + th * resid[t-1] + TH * resid[t-s] + th * TH * resid[t-s-1]
        y[t] = w + y[t-1] + y[t-s] - y[t-s-1]
    end
    return y[(burn+1):end]
end

@testset "SARIMA" begin

# ─────────────────────────────────────────────────────────────────────────────
# Polynomial algebra — exact analytic oracles
# ─────────────────────────────────────────────────────────────────────────────

@testset "multiplicative polynomial expansion" begin
    # (1 - 0.5L)(1 - 0.8L⁴) = 1 - 0.5L - 0.8L⁴ + 0.4L⁵
    # recursion convention φ = -[coefficients beyond the leading 1]
    @test _M._expand_ar([0.5], [0.8], 4) ≈ [0.5, 0.0, 0.0, 0.8, -0.4]
    # (1 + 0.3L)(1 + 0.6L⁴) = 1 + 0.3L + 0.6L⁴ + 0.18L⁵
    @test _M._expand_ma([0.3], [0.6], 4) ≈ [0.3, 0.0, 0.0, 0.6, 0.18]

    # Degree bookkeeping: p + P·s and q + Q·s
    @test length(_M._expand_ar([0.1, 0.2], [0.3, 0.4], 12)) == 2 + 2 * 12
    @test length(_M._expand_ma([0.1], [0.3, 0.4], 4)) == 1 + 2 * 4

    # No seasonal part is the identity
    @test _M._expand_ar([0.5, -0.2], Float64[], 12) == [0.5, -0.2]
    @test _M._expand_ma([0.5, -0.2], Float64[], 12) == [0.5, -0.2]

    # Pure seasonal: φ appears only at multiples of s
    ar = _M._expand_ar(Float64[], [0.7], 4)
    @test ar ≈ [0.0, 0.0, 0.0, 0.7]
end

@testset "differencing operators" begin
    y = collect(1.0:20.0)
    # A linear series has constant first difference and constant seasonal difference
    @test _M._seasonal_difference(y, 1, 4) ≈ fill(4.0, 16)
    @test _M._seasonal_difference(y, 0, 4) == y
    # (1-L)(1-L⁴) annihilates a linear series
    @test maximum(abs, _M._sarima_difference(y, 1, 1, 4)) < 1e-12
    # Coefficients of (1-L)(1-L⁴) = 1 - L - L⁴ + L⁵
    @test _M._sarima_diff_poly(1, 1, 4, Float64) ≈ [1.0, -1.0, 0.0, 0.0, -1.0, 1.0]
    @test _M._sarima_diff_poly(0, 0, 12, Float64) ≈ [1.0]
    @test _M._sarima_diff_poly(2, 0, 4, Float64) ≈ [1.0, -2.0, 1.0]

    @test_throws ArgumentError _M._seasonal_difference(collect(1.0:4.0), 1, 12)
end

@testset "_undifference inverts the operator exactly" begin
    # Round trip: differencing then undifferencing recovers the tail of a known series
    rng = Random.MersenneTwister(3)
    y = cumsum(cumsum(randn(rng, 80)))          # I(2)-ish, plenty of structure
    for (d, D, s) in ((1, 0, 0), (2, 0, 0), (0, 1, 4), (1, 1, 4), (1, 1, 12))
        delta = _M._sarima_diff_poly(d, D, s, Float64)
        w = _M._sarima_difference(y, d, D, s)
        # Treat the last 5 differenced values as "forecasts" and rebuild the levels
        h = 5
        y_hist = y[1:end-h]
        w_future = w[end-h+1:end]
        @test _M._undifference(y_hist, w_future, delta) ≈ y[end-h+1:end] atol = 1e-8
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Estimation
# ─────────────────────────────────────────────────────────────────────────────

@testset "zero seasonal orders reproduce ARIMA exactly" begin
    rng = Random.MersenneTwister(42)
    n = 200
    resid = randn(rng, n)
    z = zeros(n)
    for t in 2:n
        z[t] = 0.6 * z[t-1] + resid[t] + 0.3 * resid[t-1]
    end

    for (p, d, q) in ((1, 0, 1), (2, 0, 0), (0, 0, 1), (1, 1, 1))
        a = estimate_arima(z, p, d, q; method=:css_mle)
        b = estimate_sarima(z, p, d, q, 0, 0, 0, 0; method=:css_mle)
        # Two independent CSS-MLE paths (`_estimate_arma_internal` vs
        # `_estimate_sarima_internal` at g_tol=1e-8). Windows OpenBLAS lands
        # ~1.4e-10 apart on θ/σ² — just past atol=1e-10 — so match the
        # optimizer's own stopping tolerance rather than demand bit identity.
        @test b.phi ≈ a.phi rtol = 1e-8 atol = 1e-8
        @test b.theta ≈ a.theta rtol = 1e-8 atol = 1e-8
        @test b.c ≈ a.c rtol = 1e-8 atol = 1e-8
        @test b.sigma2 ≈ a.sigma2 rtol = 1e-8 atol = 1e-8
        @test b.loglik ≈ a.loglik rtol = 1e-8 atol = 1e-8
        @test b.y_diff ≈ a.y_diff atol = 1e-12
        # Expanded polynomials collapse to the non-seasonal ones
        @test b.phi_expanded ≈ a.phi
        @test b.theta_expanded ≈ a.theta
    end

    # ... and so do the forecasts
    a = estimate_arima(z, 1, 1, 1; method=:css_mle)
    b = estimate_sarima(z, 1, 1, 1, 0, 0, 0, 0; method=:css_mle)
    fa, fb = forecast(a, 10), forecast(b, 10)
    @test fb.forecast ≈ fa.forecast atol = 1e-9
    @test fb.se ≈ fa.se atol = 1e-9
end

@testset "airline model recovers known seasonal MA parameters" begin
    th, TH = -0.4, -0.6
    y = _airline_series(400, 12, th, TH; seed=7)
    m = estimate_sarima(y, 0, 1, 1, 0, 1, 1, 12; include_intercept=false)

    @test m isa SARIMAModel{Float64}
    @test m.p == 0 && m.d == 1 && m.q == 1
    @test m.P == 0 && m.D == 1 && m.Q == 1 && m.s == 12
    @test m.converged
    @test m.theta[1] ≈ th atol = 0.12
    @test m.Theta[1] ≈ TH atol = 0.12
    @test m.sigma2 ≈ 1.0 atol = 0.25

    # The expanded MA carries the multiplicative cross term θ·Θ at lag s+1
    @test length(m.theta_expanded) == 1 + 12
    @test m.theta_expanded[1] ≈ m.theta[1] atol = 1e-12
    @test m.theta_expanded[12] ≈ m.Theta[1] atol = 1e-12
    @test m.theta_expanded[13] ≈ m.theta[1] * m.Theta[1] atol = 1e-12
    @test all(abs.(m.theta_expanded[2:11]) .< 1e-12)

    # Differencing bookkeeping
    @test length(m.y_diff) == length(y) - 1 - 12
end

@testset "seasonal AR recovery" begin
    rng = Random.MersenneTwister(11)
    n = 450
    resid = randn(rng, n)
    z = zeros(n)
    # (1 - 0.5L)(1 - 0.7L¹²) z = ε  ⇒  z_t = 0.5 z_{t-1} + 0.7 z_{t-12} - 0.35 z_{t-13} + ε
    for t in 14:n
        z[t] = 0.5 * z[t-1] + 0.7 * z[t-12] - 0.35 * z[t-13] + resid[t]
    end
    m = estimate_sarima(z[50:end], 1, 0, 0, 1, 0, 0, 12)
    @test m.phi[1] ≈ 0.5 atol = 0.12
    @test m.Phi[1] ≈ 0.7 atol = 0.12
    # Expanded AR reproduces the multiplicative cross term at lag s+1
    @test m.phi_expanded[13] ≈ -m.phi[1] * m.Phi[1] atol = 1e-12
end

@testset "estimation methods and validation" begin
    y = _airline_series(200, 4, -0.5, -0.4; seed=5)

    m_css = estimate_sarima(y, 0, 1, 1, 0, 1, 1, 4; method=:css, include_intercept=false)
    @test m_css.method === :css
    @test isfinite(m_css.loglik)
    m_mle = estimate_sarima(y, 0, 1, 1, 0, 1, 1, 4; method=:mle, include_intercept=false)
    @test m_mle.method === :mle
    # CSS and MLE agree closely on the parameters. Their reported log-likelihoods are NOT
    # comparable — CSS is conditional on the first max(p+Ps, q+Qs) observations and sums
    # over n−m terms, while MLE is the exact likelihood over all n — so compare the
    # estimates, not the objective values.
    @test m_mle.theta[1] ≈ m_css.theta[1] atol = 0.05
    @test m_mle.Theta[1] ≈ m_css.Theta[1] atol = 0.05
    m_both = estimate_sarima(y, 0, 1, 1, 0, 1, 1, 4; method=:css_mle, include_intercept=false)
    @test m_both.method === :css_mle
    @test m_both.loglik ≈ m_mle.loglik atol = 1.0     # same exact-likelihood scale

    @test_throws ArgumentError estimate_sarima(y, -1, 1, 1, 0, 1, 1, 4)
    @test_throws ArgumentError estimate_sarima(y, 0, 1, 1, 1, 0, 0, 1)   # s < 2 with P > 0
    @test_throws ArgumentError estimate_sarima(y, 0, 1, 1, 0, 1, 1, 4; method=:bogus)
    @test_throws ArgumentError estimate_sarima(randn(MersenneTwister(1), 20), 2, 1, 2, 1, 1, 1, 12)  # too short
end

@testset "StatsAPI interface and display" begin
    y = _airline_series(200, 4, -0.5, -0.4; seed=9)
    m = estimate_sarima(y, 1, 1, 1, 1, 1, 1, 4)

    @test length(coef(m)) == 1 + m.p + m.q + m.P + m.Q
    @test coef(m) ≈ vcat(m.c, m.phi, m.theta, m.Phi, m.Theta)
    @test length(stderror(m)) == length(coef(m))
    @test size(confint(m)) == (length(coef(m)), 2)
    @test nobs(m) == length(y)
    @test dof(m) == m.p + m.q + m.P + m.Q + 2
    @test residuals(m) === m.residuals
    @test fitted(m) === m.fitted
    @test loglikelihood(m) == m.loglik
    @test aic(m) == m.aic && bic(m) == m.bic
    @test ar_order(m) == 1 && ma_order(m) == 1 && diff_order(m) == 1

    out = sprint(show, m)
    @test occursin("SARIMA(1,1,1)(1,1,1)[4]", out)
    @test occursin("Φ[1]", out)
    @test occursin("Θ[1]", out)
    @test report(m) === nothing
    @test occursin("Box", sprint(io -> refs(io, m)))

    @test plot_result(m) isa PlotOutput
end

# ─────────────────────────────────────────────────────────────────────────────
# Forecasting
# ─────────────────────────────────────────────────────────────────────────────

@testset "forecasts reproduce the seasonal pattern with widening bands" begin
    s = 12
    y = _airline_series(300, s, -0.4, -0.6; seed=7)
    m = estimate_sarima(y, 0, 1, 1, 0, 1, 1, s; include_intercept=false)
    h = 2 * s
    f = forecast(m, h)

    @test length(f.forecast) == h
    @test f.horizon == h
    # Prediction intervals widen monotonically with the horizon (seasonal differencing
    # is integrated through the ψ-weights of the full non-differenced operator)
    @test all(diff(f.se) .>= -1e-10)
    @test f.se[end] > f.se[1]
    @test all(f.ci_lower .< f.forecast .< f.ci_upper)

    # The forecast carries a seasonal signature: the season-to-season profile of the
    # forecast correlates with that of the last observed year.
    last_year = y[end-s+1:end] .- mean(y[end-s+1:end])
    fc_year = f.forecast[1:s] .- mean(f.forecast[1:s])
    @test cor(last_year, fc_year) > 0.5

    # A pure-seasonal-random-walk special case is exact: with all ARMA orders zero and
    # no intercept, ŷ_{T+h} = y_{T+h-s} + y_{T+h-1} - y_{T+h-1-s}.
    m0 = estimate_sarima(y, 0, 1, 0, 0, 1, 0, s; include_intercept=false)
    f0 = forecast(m0, 3)
    @test f0.forecast[1] ≈ y[end] + y[end-s+1] - y[end-s] atol = 1e-8

    @test_throws ArgumentError forecast(m, 0)
end

@testset "seasonal differencing widens bands relative to the differenced scale" begin
    s = 4
    y = _airline_series(200, s, -0.5, -0.4; seed=13)
    m = estimate_sarima(y, 0, 1, 1, 0, 1, 1, s; include_intercept=false)
    h = 12
    f = forecast(m, h)

    # On the differenced scale the ARMA is an MA(q+Qs): its forecast variance is flat
    # beyond lag q+Qs. On the original scale the two unit roots make it grow without
    # bound — the point of extending the ψ-weight machinery to the seasonal operator.
    f_diff = _M._forecast_arma(m.y_diff, m.residuals, m.c, m.phi_expanded,
                               m.theta_expanded, m.sigma2, h, 0.95)
    @test f_diff.se[end] ≈ f_diff.se[end-1] atol = 1e-10     # flat MA variance
    @test f.se[end] > 2 * f.se[1]                            # integrated growth
end

# ─────────────────────────────────────────────────────────────────────────────
# Automatic order selection
# ─────────────────────────────────────────────────────────────────────────────

@testset "auto_sarima recovers the airline orders" begin
    s = 12
    y = _airline_series(300, s, -0.4, -0.6; seed=7)
    best = auto_sarima(y, s; max_p=1, max_q=1, max_P=1, max_Q=1)

    @test best isa SARIMAModel{Float64}
    @test best.s == s
    @test best.d == 1 && best.D == 1        # HEGY + KPSS pick both unit roots
    @test best.q >= 1 && best.Q >= 1        # the MA terms are needed
    @test best.method === :css_mle          # winner refit with the requested method

    # Fixed differencing orders are honored
    fixed = auto_sarima(y, s; d=1, D=1, max_p=0, max_q=1, max_P=0, max_Q=1)
    @test fixed.d == 1 && fixed.D == 1 && fixed.p == 0 && fixed.P == 0

    @test_throws ArgumentError auto_sarima(y, s; criterion=:bogus)
    @test_throws ArgumentError auto_sarima(y, 0)
end

@testset "differencing-order selectors" begin
    s = 12
    y = _airline_series(300, s, -0.4, -0.6; seed=7)
    # A doubly integrated seasonal series needs a seasonal difference ...
    @test _M._auto_seasonal_diff(y, s) == 1
    # ... and a regular one afterwards
    @test _M._auto_regular_diff(_M._seasonal_difference(y, 1, s)) == 1
    # White noise needs neither
    rng = Random.MersenneTwister(21)
    @test _M._auto_regular_diff(randn(rng, 200)) == 0
    # HEGY does not apply outside s ∈ {4, 12} or on short samples → 0
    @test _M._auto_seasonal_diff(y, 7) == 0
    @test _M._auto_seasonal_diff(y[1:20], 12) == 0
    @test _M._auto_regular_diff(randn(rng, 5)) == 0
end

end  # @testset "SARIMA"
