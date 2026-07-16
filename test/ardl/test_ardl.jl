# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-08 (#416): ARDL estimation + Pesaran–Shin–Smith (2001) bounds test.
#
# Oracle strategy (no invented numerics):
#   (1) Published-value spot-assert of the hard-coded PSS (2001) critical-value
#       tables: Case III, k=1, 5% F-bounds = 4.94 / 5.73 (Table CI(iii)) and
#       t-bounds = -2.86 / -3.22 (Table CII(iii)); the I(0) lower t-bound is
#       k-invariant for a given case.
#   (2) Analytic self-consistency: the ARDL long-run coefficient is the fixed
#       point θ = (Σβ)/(1−Σφ) of the difference equation, so a fixed-seed DGP
#       built with a KNOWN long-run multiplier recovers it (cross-check against
#       the closed form + a finite-difference delta-method SE).
#   (3) Cross-implementation within the test: the PSS bounds F equals an
#       independently-computed restricted/unrestricted-SSR joint F on the
#       conditional ECM design.
#   (4) Degenerate property: a strongly cointegrated pair ⇒ :cointegrated;
#       independent random walks ⇒ :not_cointegrated.
#   All statistics are compared to the bounds only — never a p-value.
#
# NOTE on the Stata `ardl` oracle: the acceptance criterion suggests pinning
# Stata `ardl y x` (Kripfganz–Schneider) coefficients. Stata is not available in
# this environment, so rather than fabricate its output we rely on the analytic
# long-run-recovery oracle (2) — the ARDL long-run estimator's fixed point IS the
# cointegrating coefficient — plus the published PSS table spot-checks (1). The
# Stata command that would reproduce this path is documented below for provenance.

using Test, MacroEconometricModels, Random, LinearAlgebra, Statistics

@testset "ARDL & PSS Bounds Test (EV-08 #416)" begin

    # -------------------------------------------------------------------------
    # Fixed-seed DGPs
    # -------------------------------------------------------------------------
    # Cointegrated system with a KNOWN long-run multiplier θ_true on x.
    # y_t = y_{t-1} − φ·(y_{t-1} − θ·x_{t-1}) + ψ·Δx_t + ε_t  (error-correction DGP)
    function coint_dgp(; T=300, θ=2.0, φ=0.4, ψ=0.5, σ=0.3, seed=20240716)
        rng = MersenneTwister(seed)
        x = cumsum(randn(rng, T))
        y = zeros(T)
        for t in 2:T
            y[t] = y[t-1] - φ * (y[t-1] - θ * x[t-1]) + ψ * (x[t] - x[t-1]) + σ * randn(rng)
        end
        (y, x)
    end

    # -------------------------------------------------------------------------
    # (1) Published PSS (2001) critical-value spot-checks
    # -------------------------------------------------------------------------
    @testset "PSS 2001 critical-value tables" begin
        y, x = coint_dgp()
        m = estimate_ardl(y, x; p=1, q=1, case=3)
        bt = bounds_test(m; level=0.05)          # Case III, k=1

        @test bt.levels == [0.10, 0.05, 0.025, 0.01]
        li5 = 2                                    # index of the 5% column
        # Case III, k=1, 5% F-bounds — PSS (2001) Table CI(iii): 4.94 / 5.73
        @test bt.f_lower[li5] == 4.94
        @test bt.f_upper[li5] == 5.73
        # Case III, k=1, 5% t-bounds — PSS (2001) Table CII(iii): -2.86 / -3.22
        @test bt.t_lower[li5] == -2.86
        @test bt.t_upper[li5] == -3.22
        # 10% and 1% F-bounds (Table CI(iii), k=1)
        @test bt.f_lower[1] == 4.04 && bt.f_upper[1] == 4.78
        @test bt.f_lower[4] == 6.84 && bt.f_upper[4] == 7.84

        # The I(0) t lower-bound is k-invariant for a given case (Case III, 5%).
        m2 = estimate_ardl(y, hcat(x, cumsum(randn(MersenneTwister(1), length(y)))); p=1, q=[1,1], case=3)
        bt2 = bounds_test(m2)
        @test bt2.k == 2
        @test bt2.t_lower[li5] == -2.86            # same I(0) as k=1
        @test bt2.f_lower[li5] == 3.79             # Case III, k=2, 5% (Table CI(iii))
        @test bt2.f_upper[li5] == 4.85
    end

    # -------------------------------------------------------------------------
    # (2) Long-run coefficient recovery + closed-form / delta-method checks
    # -------------------------------------------------------------------------
    @testset "long-run coefficients (analytic)" begin
        θ_true = 2.0
        y, x = coint_dgp(; θ=θ_true, T=400)
        m = estimate_ardl(y, x; p=2, q=2, case=3)
        lr = long_run(m)

        # Recovery of the true long-run multiplier.
        @test isapprox(lr.theta[1], θ_true; atol=0.15)
        @test lr.se[1] > 0

        # Closed-form reconstruction θ = (Σβ_x)/(1−Σφ) from raw coefficients.
        b = m.coef
        denom = 1 - sum(b[m.ar_idx])
        num = sum(b[m.x_idx[1]])
        @test isapprox(lr.theta[1], num / denom; rtol=1e-12)
        @test isapprox(lr.denom, denom; rtol=1e-12)

        # Delta-method SE cross-check via a finite-difference Jacobian of θ(coef).
        V = m.vcov
        θfun(bb) = sum(bb[m.x_idx[1]]) / (1 - sum(bb[m.ar_idx]))
        g = zeros(length(b))
        h = 1e-6
        for i in eachindex(b)
            bp = copy(b); bp[i] += h
            bm = copy(b); bm[i] -= h
            g[i] = (θfun(bp) - θfun(bm)) / (2h)
        end
        se_fd = sqrt(dot(g, V * g))
        @test isapprox(lr.se[1], se_fd; rtol=1e-4)
    end

    # -------------------------------------------------------------------------
    # (3) ECM re-parameterisation + bounds-F cross-check (independent SSR-F)
    # -------------------------------------------------------------------------
    @testset "ECM form & bounds-F cross-check" begin
        y, x = coint_dgp(; T=300)
        m = estimate_ardl(y, x; p=1, q=1, case=3)   # ARDL(1,1), Case III
        ecm = MacroEconometricModels.ecm_form(m)

        # Speed of adjustment α = Σφ − 1 = −(1−Σφ) = −denom; and α < 0 (error-correcting).
        @test isapprox(ecm.alpha, sum(m.coef[m.ar_idx]) - 1; rtol=1e-12)
        @test isapprox(ecm.alpha, -long_run(m).denom; rtol=1e-12)
        @test ecm.alpha < 0

        # Independent computation of the PSS F via restricted/unrestricted SSR on the
        # conditional ECM design (same effective sample). Unrestricted ECM:
        #   Δy_t = c + ρ y_{t-1} + δ x_{t-1} + ω Δx_t + u_t   (exact reparam of the ARDL)
        # Restricted drops the level terms {ρ, δ}. F = ((SSR_r−SSR_u)/2)/(SSR_u/(n−4)).
        N = length(y)
        rows = 2:N                                  # L = max(p,q) = 1
        Δy = [y[t] - y[t-1] for t in rows]
        ylag = [y[t-1] for t in rows]
        xlag = [x[t-1] for t in rows]
        Δx = [x[t] - x[t-1] for t in rows]
        n = length(Δy)
        Xu = hcat(ones(n), ylag, xlag, Δx)
        bu = Xu \ Δy; ru = Δy - Xu * bu; ssr_u = sum(abs2, ru)
        Xr = hcat(ones(n), Δx)
        br = Xr \ Δy; rr = Δy - Xr * br; ssr_r = sum(abs2, rr)
        Fssr = ((ssr_r - ssr_u) / 2) / (ssr_u / (n - 4))

        bt = bounds_test(m)
        @test isapprox(bt.fstat, Fssr; rtol=1e-8)

        # ECM unrestricted SSR must equal the levels-ARDL SSR (exact reparam).
        @test isapprox(ssr_u, m.ssr; rtol=1e-8)

        # Independent t-ratio on the y_{t-1} level from the ECM regression.
        XtXinv = inv(Xu' * Xu)
        se_rho = sqrt((ssr_u / (n - 4)) * XtXinv[2, 2])
        t_ecm = bu[2] / se_rho
        @test isapprox(bt.tstat, t_ecm; rtol=1e-6)
    end

    # -------------------------------------------------------------------------
    # (4) Cointegration decisions (degenerate property oracle)
    # -------------------------------------------------------------------------
    @testset "bounds-test decisions" begin
        # Strongly cointegrated pair ⇒ reject H0 (F above I(1), t below I(1)).
        y, x = coint_dgp(; T=300, φ=0.5)
        bt = bounds_test(estimate_ardl(y, x; p=1, q=0, case=3))
        @test bt.f_decision == :cointegrated
        @test bt.t_decision == :cointegrated
        @test bt.fstat > bt.f_upper[2]

        # Independent random walks ⇒ do not reject (F below I(0)).
        rng = MersenneTwister(99)
        nrej = 0
        for s in 1:20
            xr = cumsum(randn(rng, 250)); yr = cumsum(randn(rng, 250))
            b = bounds_test(estimate_ardl(yr, xr; p=1, q=0, case=3))
            b.f_decision == :cointegrated && (nrej += 1)
        end
        # A correctly-sized 5% test rejects on pure noise only rarely.
        @test nrej <= 3
    end

    # -------------------------------------------------------------------------
    # Estimation / selection mechanics
    # -------------------------------------------------------------------------
    @testset "estimation & IC selection" begin
        y, x = coint_dgp(; T=300)
        m = estimate_ardl(y, x; p=2, q=3, case=3)
        @test m.p == 2 && m.q == [3]
        @test m.n == length(y) - 3            # L = max(2,3) = 3
        @test m.K == length(m.coef) == 1 + 2 + 4   # intercept + 2 AR + (q+1) x lags
        @test length(m.residuals) == m.n
        @test isapprox(m.fitted + m.residuals, m.y; rtol=1e-10)

        # Auto grid selection returns valid orders in range.
        mg = estimate_ardl(y, x; p=:auto, q=:auto, max_p=3, max_q=3, ic=:bic, case=3)
        @test mg.selected
        @test 1 <= mg.p <= 3 && 0 <= mg.q[1] <= 3
        @test mg.ic == :bic

        # Deterministics per case.
        @test estimate_ardl(y, x; p=1, q=1, case=1).trend == :none
        @test estimate_ardl(y, x; p=1, q=1, case=3).intercept_col > 0
        m5 = estimate_ardl(y, x; p=1, q=1, case=5)
        @test m5.trend == :trend && m5.trend_col > 0

        # StatsAPI surface.
        @test nobs(m) == m.n
        @test length(coef(m)) == m.K
        @test length(stderror(m)) == m.K
    end

    # -------------------------------------------------------------------------
    # Multi-regressor & case handling
    # -------------------------------------------------------------------------
    @testset "multi-regressor & cases" begin
        rng = MersenneTwister(5)
        T = 300
        x1 = cumsum(randn(rng, T)); x2 = cumsum(randn(rng, T))
        y = zeros(T)
        for t in 2:T
            y[t] = y[t-1] - 0.4*(y[t-1] - 1.5*x1[t-1] - 0.8*x2[t-1]) + 0.3*randn(rng)
        end
        X = hcat(x1, x2)
        m = estimate_ardl(y, X; p=1, q=[1, 1], case=3, xnames=["x1", "x2"])
        lr = long_run(m)
        @test length(lr.theta) == 2
        @test isapprox(lr.theta[1], 1.5; atol=0.3)
        @test isapprox(lr.theta[2], 0.8; atol=0.3)

        bt = bounds_test(m)
        @test bt.k == 2

        # Case IV (restricted trend) ⇒ t-bounds undefined.
        m4 = estimate_ardl(y, X; p=1, q=[1, 1], case=4)
        bt4 = bounds_test(m4)
        @test bt4.t_decision == :undefined
        @test all(isnan, bt4.t_lower)
        @test !isnan(bt4.f_lower[2])   # F-bounds still defined (Case IV, k=2)
    end

    # -------------------------------------------------------------------------
    # Input validation & error paths
    # -------------------------------------------------------------------------
    @testset "validation" begin
        y, x = coint_dgp(; T=120)
        @test_throws ArgumentError estimate_ardl(y, x; p=1, q=1, case=6)
        @test_throws ArgumentError estimate_ardl(y, x; p=1, q=1, ic=:hqic)
        @test_throws ArgumentError estimate_ardl(y[1:50], x; p=1, q=1)      # length mismatch
        m = estimate_ardl(y, x; p=1, q=1, case=3)
        @test_throws ArgumentError bounds_test(m; cv_source=:narayan)       # not bundled
        @test_throws ArgumentError bounds_test(m; cv_source=:foo)
        @test_throws ArgumentError bounds_test(m; level=0.075)              # not tabulated
        @test_throws ArgumentError bounds_test(m; case=9)
        # k beyond the tabulated range (k>10) is rejected.
        bigX = randn(200, 11)
        mbig = estimate_ardl(cumsum(randn(200)), cumsum(bigX, dims=1); p=1,
                             q=fill(0, 11), case=3)
        @test_throws ArgumentError bounds_test(mbig)
    end

    # -------------------------------------------------------------------------
    # Integer input + vector convenience
    # -------------------------------------------------------------------------
    @testset "input conversion" begin
        yi = round.(Int, cumsum(randn(MersenneTwister(3), 150)) .* 10)
        xi = round.(Int, cumsum(randn(MersenneTwister(4), 150)) .* 10)
        m = estimate_ardl(yi, xi; p=1, q=1, case=3)     # Int vectors → Float64
        @test m isa ARDLModel{Float64}
        @test m.q == [1]
    end

    # -------------------------------------------------------------------------
    # Display & references render without error, and NEVER emit a p-value
    # -------------------------------------------------------------------------
    @testset "display & refs" begin
        y, x = coint_dgp(; T=200)
        m = estimate_ardl(y, x; p=1, q=1, case=3)
        bt = bounds_test(m)

        io = IOBuffer(); show(io, m); s = String(take!(io))
        @test occursin("ARDL", s)
        @test occursin("Long-run", s)
        @test occursin("Error-correction", s)

        io = IOBuffer(); show(io, bt); sb = String(take!(io))
        @test occursin("Bounds", sb)
        @test occursin("I(0)", sb) && occursin("I(1)", sb)
        # A bounds test must never print a COMPUTED p-value: no p-value column
        # header, and the output explicitly documents that none is defined.
        @test !occursin("P>|", sb)
        @test occursin("no p-value is defined", sb)

        io = IOBuffer(); report(io, m); @test !isempty(String(take!(io)))
        io = IOBuffer(); report(io, bt); @test !isempty(String(take!(io)))

        rs = refs(m); @test occursin("Pesaran", rs)
        rb = refs(bt); @test occursin("Pesaran", rb)
    end
end
