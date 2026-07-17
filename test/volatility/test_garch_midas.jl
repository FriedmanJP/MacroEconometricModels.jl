# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-02 (#410): GARCH-MIDAS long/short-run volatility components.
#
# Oracle strategy (per docs/plans/2026-07-16-ev-series-specs.json):
#   (1) Analytic property — short-run g is genuinely unit-mean, i.e.
#       E[(r-μ)²/τ] = 1 (guards the √τ innovation scaling); flat X ⇒ τ constant.
#   (2) Fixed-seed simulated recovery — a >=200-block GARCH-MIDAS path recovers
#       (α, β, θ, m, w) within a stated tolerance.
#   (3) Internal cross-impl degeneracy — flat X collapses to a unit-mean
#       GARCH(1,1) matching `estimate_garch` loglik/params to tolerance.
#   (4) R mfGARCH known-value oracle: `mfGARCH` is NOT installed in this
#       environment (only the Octave/BVAR oracle harness is, per
#       test/oracle/README.md) and the published Engle-Ghysels-Sohn/Conrad-Kleen
#       estimates depend on their proprietary CRSP/NFCI dataset, so no exact
#       decimals are hard-coded here (fabrication is disallowed). The reference
#       cross-check would be, for the mfGARCH `df_financial` example:
#           library(mfGARCH)
#           fit_mfgarch(data = df_financial, y = "return", x = "nfci",
#                       low.freq = "year_week", K = 52, weighting = "beta.restricted")
#       reproducing σ²_{i,t} = τ_t g_{i,t}, τ_t = exp(m + θ Σ φ_k(1,w) RV_{t-k}).
#       Recovery + degeneracy (2,3) pin the estimator against ground truth instead.

using Test
using MacroEconometricModels
using Random
using Statistics
using LinearAlgebra

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

# -----------------------------------------------------------------------------
# GARCH-MIDAS data generator (fixed calendar-block long-run component)
# -----------------------------------------------------------------------------
"""
Simulate a fixed-span GARCH-MIDAS path with an exogenous AR(1) macro driver `X`
(one value per low-frequency block). Returns `(r, X)` where `r` is the stacked
high-frequency return series (length `nblk*m_freq`).
"""
function simulate_garch_midas(; nblk::Int=260, m_freq::Int=22, K::Int=12,
                              mu=0.02, alpha=0.06, beta=0.90,
                              m=-0.5, theta=0.3, w=4.0, seed::Int=11)
    rng = MersenneTwister(seed)
    phi = midas_weights([1.0, w], K; kind=:beta2)
    X = zeros(nblk)
    for t in 2:nblk
        X[t] = 0.7 * X[t-1] + randn(rng)
    end
    tau = fill(exp(m), nblk)
    for t in (K+1):nblk
        tau[t] = exp(m + theta * sum(phi[k] * X[t-k] for k in 1:K))
    end
    n = nblk * m_freq
    r = zeros(n)
    g = 1.0
    resid_prev = 0.0
    tau_prev = tau[1]
    for t in 1:nblk, i in 1:m_freq
        gi = (1 - alpha - beta) + alpha * resid_prev^2 / tau_prev + beta * g
        g = gi
        idx = (t - 1) * m_freq + i
        s2 = tau[t] * gi
        r[idx] = mu + sqrt(s2) * randn(rng)
        resid_prev = r[idx] - mu
        tau_prev = tau[t]
    end
    return r, X
end

@testset "GARCH-MIDAS (EV-02, #410)" begin
    r, X = simulate_garch_midas(nblk=260, m_freq=22, K=12; seed=11)

    # =========================================================================
    # Oracle (1) — unit-mean short-run component & √τ innovation scaling
    # =========================================================================
    @testset "unit-mean short-run g" begin
        m = estimate_garch_midas(r, X; K=12, m_freq=22)
        # E[g] = 1 by the (1-α-β) intercept; E[(r-μ)²/τ] = E[g] = 1.
        @test isapprox(mean(m.g), 1.0; atol=0.05)
        @test isapprox(mean(m.residuals .^ 2 ./ m.tau), 1.0; atol=0.05)
        # τ and g are strictly positive; total variance = τ·g
        @test all(m.tau .> 0)
        @test all(m.g .> 0)
        @test isapprox(m.conditional_variance, m.tau .* m.g; rtol=1e-12)
        # Beta weights sum to one and decay monotonically (w > 1)
        @test isapprox(sum(m.weights), 1.0; atol=1e-10)
        @test m.w > 1
        @test all(diff(m.weights) .<= 1e-12)
        # variance ratio is a share in [0,1]
        @test 0 <= m.variance_ratio <= 1
        @test length(m.ret_idx) == length(m.conditional_variance)
    end

    # =========================================================================
    # Oracle (2) — fixed-seed simulated recovery
    # =========================================================================
    @testset "parameter recovery" begin
        # true: mu=0.02, alpha=0.06, beta=0.90, m=-0.5, theta=0.3, w=4.0
        m = estimate_garch_midas(r, X; K=12, m_freq=22)
        @test m.converged
        @test isapprox(m.mu,      0.02; atol=0.03)
        @test isapprox(m.alpha,   0.06; atol=0.03)
        @test isapprox(m.beta,    0.90; atol=0.05)
        @test isapprox(m.m_const, -0.5; atol=0.25)
        @test isapprox(m.theta,   0.30; atol=0.15)
        @test m.theta > 0                       # sign identified
        # `w` (Beta shape) is only weakly identified; assert the recovered
        # qualitative shape — a decaying weight curve concentrated on recent lags —
        # rather than the exact value.
        @test 1.5 < m.w < 8.0
        @test m.weights[1] > m.weights[end]
        @test sum(m.weights[1:6]) > 0.5
        @test m.alpha + m.beta < 1              # short-run stationarity
        # standard errors are finite and positive
        se = MacroEconometricModels.StatsAPI.stderror(m)
        @test length(se) == 6
        @test all(isfinite, se)
        @test all(se .> 0)
        # hessian-based SEs also finite
        se_h = MacroEconometricModels.StatsAPI.stderror(m; cov_type=:hessian)
        @test all(isfinite, se_h)
    end

    # =========================================================================
    # Oracle (3) — flat X ⇒ constant τ ⇒ collapses to standard GARCH(1,1)
    # =========================================================================
    @testset "degenerate collapse to GARCH(1,1)" begin
        Xflat = ones(length(X))
        md = estimate_garch_midas(r, Xflat; K=12, m_freq=22)
        # flat covariate ⇒ Σ φ_k X = const ⇒ τ_t identical for every block/obs
        @test maximum(abs.(md.tau .- md.tau[1])) < 1e-8
        @test isapprox(md.variance_ratio, 0.0; atol=1e-10)   # no long-run variation

        # GARCH(1,1) on the SAME retained sample must match to tolerance
        rret = r[md.ret_idx]
        mg = estimate_garch(rret, 1, 1)
        @test isapprox(md.loglik, mg.loglik; atol=0.5)
        @test isapprox(md.alpha, mg.alpha[1]; atol=0.02)
        @test isapprox(md.beta,  mg.beta[1];  atol=0.02)
        # implied ω = τ·(1-α-β) matches GARCH ω
        omega_implied = md.tau[1] * (1 - md.alpha - md.beta)
        @test isapprox(omega_implied, mg.omega; rtol=0.15)
    end

    # =========================================================================
    # Realized-variance driver & rolling span run end-to-end
    # =========================================================================
    @testset "realized / rolling spans" begin
        mr = estimate_garch_midas(r; K=6, m_freq=22, rv=:realized, span=:fixed)
        @test all(isfinite, mr.conditional_variance)
        @test isapprox(mean(mr.g), 1.0; atol=0.15)
        @test all(mr.tau .> 0)
        @test isempty(mr.x_lf)              # realized ⇒ no exogenous series stored

        mroll = estimate_garch_midas(r; K=4, m_freq=22, span=:rolling)
        @test all(isfinite, mroll.conditional_variance)
        @test isapprox(mean(mroll.g), 1.0; atol=0.2)
        @test mroll.span == :rolling
        @test length(mroll.ret_idx) == length(mroll.conditional_variance)
    end

    # =========================================================================
    # Forecast, display, refs, plotting
    # =========================================================================
    @testset "forecast" begin
        m = estimate_garch_midas(r, X; K=12, m_freq=22)
        fc = forecast(m, 10)
        @test length(fc.total) == 10
        @test length(fc.long_run) == 10
        @test length(fc.short_run) == 10
        @test all(fc.total .> 0)
        @test all(fc.long_run .== m.tau[end])       # τ held at last block
        # short-run mean-reverts toward 1 ⇒ total → τ_last
        @test isapprox(fc.short_run[end], 1.0; atol=0.2)
        @test isapprox(MacroEconometricModels.StatsAPI.predict(m, 5), forecast(m, 5).total)
        @test_throws ArgumentError forecast(m, 0)
    end

    @testset "display / refs / plot" begin
        m = estimate_garch_midas(r, X; K=12, m_freq=22)
        s = sprint(show, m)
        @test occursin("GARCH-MIDAS", s)
        @test occursin("Variance ratio", s)
        @test occursin("θ", s)
        rtxt = sprint(io -> refs(io, m))
        @test occursin("Engle", rtxt)
        @test occursin("2013", rtxt)
        # plotting dispatches
        p1 = plot_result(m; view=:components)
        @test p1 isa MacroEconometricModels.PlotOutput
        p2 = plot_result(m; view=:weights)
        @test p2 isa MacroEconometricModels.PlotOutput
        @test_throws ArgumentError plot_result(m; view=:nope)
        # StatsAPI accessors
        @test MacroEconometricModels.StatsAPI.nobs(m) == length(m.ret_idx)
        @test length(MacroEconometricModels.StatsAPI.coef(m)) == 6
        @test MacroEconometricModels.StatsAPI.dof(m) == 6
        @test persistence(m) ≈ m.alpha + m.beta
    end

    # =========================================================================
    # Input validation
    # =========================================================================
    @testset "input validation" begin
        @test_throws ArgumentError estimate_garch_midas(r, X; K=1, m_freq=22)
        @test_throws ArgumentError estimate_garch_midas(randn(10), randn(10); K=3, m_freq=2)
        # rv=:macro needs enough low-frequency observations
        @test_throws ArgumentError estimate_garch_midas(r, X[1:3]; K=12, m_freq=22)
        @test_throws ArgumentError estimate_garch_midas(r, X; K=12, m_freq=22, rv=:bogus)
        @test_throws ArgumentError estimate_garch_midas(r; K=6, m_freq=22, span=:bogus)
    end
end
