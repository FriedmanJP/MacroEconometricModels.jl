# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-32 (#440): parameter-stability & influence diagnostics — Brown-Durbin-Evans
# recursive residuals, CUSUM / CUSUM-of-squares, Chow breakpoint/forecast tests,
# Belsley-Kuh-Welsch influence statistics.

using Test
using MacroEconometricModels
using MacroEconometricModels: recursive_residuals, cusum_test, cusumsq_test,
    chow_test, influence_stats, StabilityResult, InfluenceStats,
    _recursive_residuals, _cusum_a, _cusumsq_c0, refs
using LinearAlgebra, Statistics, Random, Distributions

# =============================================================================
# Fixed-seed oracle DGP (stable model). Regenerated identically here and offline
# for the R / statsmodels cross-check. n=60, k=3, homoskedastic:
#   rng = MersenneTwister(20260717)
#   x1 = randn(rng,60); x2 = randn(rng,60); u = randn(rng,60)
#   y  = 1 .+ 0.8x1 .- 0.5x2 .+ u
# The (y,x1,x2) matrix was written to CSV and read by:
#   (a) base R 4.5.0 — growing-window OLS for recursive residuals (an independent
#       engine), and stats::{hatvalues,rstudent,rstandard,dffits,cooks.distance,
#       dfbetas} for influence.measures;
#   (b) statsmodels 0.14.4 RecursiveLS.resid_recursive (confirms w_t) and the
#       Edgerton-Wells (1994) CUSUMSQ critical value (_cusum_squares_scalars).
# Both engines agree with each other and with the values hard-coded below.
# MersenneTwister is stable across Julia versions, so this reproduces exactly
# what the external engines saw.
# =============================================================================
function _stab_oracle_dgp()
    rng = MersenneTwister(20260717)
    n = 60
    x1 = randn(rng, n)
    x2 = randn(rng, n)
    u = randn(rng, n)
    y = 1.0 .+ 0.8 .* x1 .- 0.5 .* x2 .+ u
    X = hcat(ones(n), x1, x2)
    (y, X)
end

@testset "Reg Stability & Influence (EV-32)" begin

    y, X = _stab_oracle_dgp()
    m = estimate_reg(y, X; varnames = ["const", "x1", "x2"])
    n, k = size(X)

    # -------------------------------------------------------------------------
    # 1. Recursive residuals vs R (growing-window OLS) + statsmodels RecursiveLS.
    #    R: for(t in (k+1):n){ Xp<-cbind(1,x1[1:(t-1)],x2[1:(t-1)]); bp<-solve(t(Xp)%*%Xp,t(Xp)%*%y[1:(t-1)]);
    #          xt<-c(1,x1[t],x2[t]); ft<-1+t(xt)%*%solve(t(Xp)%*%Xp)%*%xt; w<-c(w,(y[t]-sum(xt*bp))/sqrt(ft)) }
    #    statsmodels: RecursiveLS(y,X).fit().resid_recursive[k:]
    # -------------------------------------------------------------------------
    @testset "recursive residuals (R + statsmodels oracle)" begin
        w = recursive_residuals(m)
        @test length(w) == n - k                       # 57
        @test isapprox(w[1],  1.2963899423; atol = 1e-8)   # w_first
        @test isapprox(w[2],  1.1074614078; atol = 1e-8)
        @test isapprox(w[29], -1.1609005733; atol = 1e-8)  # mid
        @test isapprox(w[end], 0.6075939211; atol = 1e-8)  # w_last
        # Identity: Σ w_t² == full-sample OLS SSR (Brown-Durbin-Evans).
        @test isapprox(sum(w .^ 2), 48.6913565359; atol = 1e-7)
        @test isapprox(sum(w .^ 2), m.ssr; atol = 1e-9)
    end

    # -------------------------------------------------------------------------
    # 2. Analytic self-consistency: w_t reproduces the growing-window OLS
    #    one-step forecast error exactly (pure identity; no external engine).
    # -------------------------------------------------------------------------
    @testset "recursive residuals reproduce growing-window forecasts" begin
        w = recursive_residuals(m)
        for t in (k+1):n
            Xp = X[1:(t-1), :]
            bp = (Xp' * Xp) \ (Xp' * y[1:(t-1)])
            xt = X[t, :]
            ft = 1 + dot(xt, (Xp' * Xp) \ xt)
            w_manual = (y[t] - dot(xt, bp)) / sqrt(ft)
            @test isapprox(w[t-k], w_manual; atol = 1e-9)
        end
    end

    # -------------------------------------------------------------------------
    # 3. CUSUM test — stable DGP stays inside the Brown-Durbin-Evans 5% band.
    # -------------------------------------------------------------------------
    @testset "CUSUM (stable, within bounds)" begin
        r = cusum_test(m)
        @test r isa StabilityResult
        @test r.kind == :cusum
        @test length(r.stat_path) == n - k
        @test !r.crossed
        @test r.first_crossing === nothing
        # BDE 5% line coefficient a = 0.948, boundary pair ±a√(n-k)(1+2(t-k)/(n-k)).
        @test _cusum_a(0.05) == 0.948
        nr = n - k
        @test isapprox(r.upper[1], 0.948 * sqrt(nr) * (1 + 2 * 1 / nr); atol = 1e-10)
        @test isapprox(r.lower[end], -0.948 * sqrt(nr) * 3.0; atol = 1e-10)
        @test all(r.upper .> 0) && all(r.lower .< 0)
        # Path stays comfortably inside (statsmodels/R: max|W_t| = 4.9808).
        @test isapprox(maximum(abs, r.stat_path), 4.9807693852; atol = 1e-6)
        @test maximum(abs, r.stat_path) < r.upper[1]
    end

    # -------------------------------------------------------------------------
    # 4. CUSUMSQ test — Edgerton-Wells (1994) c0 vs statsmodels; stable ⇒ inside.
    #    statsmodels: n=(60-3)/2-1=27.5; c0(5%) = 0.2284688685.
    # -------------------------------------------------------------------------
    @testset "CUSUMSQ (Edgerton-Wells c0, stable)" begin
        r = cusumsq_test(m)
        @test r.kind == :cusumsq
        @test isapprox(_cusumsq_c0(n - k, 0.05), 0.2284688685; atol = 1e-9)
        @test isapprox(_cusumsq_c0(n - k, 0.10), 0.2039217452; atol = 1e-9)
        @test isapprox(_cusumsq_c0(n - k, 0.01), 0.2774236381; atol = 1e-9)
        # S_t rises monotonically from >0 to exactly 1.
        @test issorted(r.stat_path)
        @test r.stat_path[1] > 0
        @test isapprox(r.stat_path[end], 1.0; atol = 1e-12)
        # Mean line (t-k)/(n-k) ± c0.
        c0 = _cusumsq_c0(n - k, 0.05)
        @test isapprox(r.upper[end], 1.0 + c0; atol = 1e-9)
        @test !r.crossed
    end

    # -------------------------------------------------------------------------
    # 5. Chow breakpoint F — analytic self-consistency (SSR_u = SSR1+SSR2) and
    #    an F(k, n-2k) reference, plus an induced-break DGP that rejects.
    # -------------------------------------------------------------------------
    @testset "Chow breakpoint F (analytic)" begin
        ssr(yy, XX) = (b = XX \ yy; r = yy .- XX * b; dot(r, r))
        bpt = 30
        ct = chow_test(m, bpt; type = :breakpoint)
        ssr_r = ssr(y, X)
        ssr_u = ssr(y[1:bpt], X[1:bpt, :]) + ssr(y[bpt+1:end], X[bpt+1:end, :])
        F_manual = ((ssr_r - ssr_u) / k) / (ssr_u / (n - 2k))
        @test isapprox(ct.statistic, F_manual; atol = 1e-9)
        @test ct.df == (k, n - 2k)                      # (3, 54)
        @test isapprox(ct.pvalue, ccdf(FDist(k, n - 2k), F_manual); atol = 1e-12)
        @test ct.pvalue > 0.05                          # stable DGP ⇒ fail to reject

        # Induced coefficient break mid-sample ⇒ reject.
        yb = copy(y); yb[31:end] .+= 4.0
        mb = estimate_reg(yb, X; varnames = ["const", "x1", "x2"])
        ctb = chow_test(mb, 30; type = :breakpoint)
        @test ctb.pvalue < 1e-6
    end

    @testset "Chow multiple breaks" begin
        ssr(yy, XX) = (b = XX \ yy; r = yy .- XX * b; dot(r, r))
        ct = chow_test(m, [20, 40]; type = :breakpoint)
        ssr_u = ssr(y[1:20], X[1:20, :]) + ssr(y[21:40], X[21:40, :]) +
                ssr(y[41:end], X[41:end, :])
        ssr_r = ssr(y, X)
        m_breaks = 2
        df1 = k * m_breaks; df2 = n - k * (m_breaks + 1)
        F_manual = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
        @test ct.df == (df1, df2)                       # (6, 51)
        @test isapprox(ct.statistic, F_manual; atol = 1e-9)
    end

    @testset "Chow forecast (predictive failure)" begin
        ssr(yy, XX) = (b = XX \ yy; r = yy .- XX * b; dot(r, r))
        n1 = 45
        cf = chow_test(m, n1; type = :forecast)
        n2 = n - n1
        ssr_r = ssr(y, X); ssr1 = ssr(y[1:n1], X[1:n1, :])
        F_manual = ((ssr_r - ssr1) / n2) / (ssr1 / (n1 - k))
        @test cf.df == (n2, n1 - k)                      # (15, 42)
        @test isapprox(cf.statistic, F_manual; atol = 1e-9)
        # forecast test refuses a vector of breaks.
        @test_throws ArgumentError chow_test(m, [10, 20]; type = :forecast)
    end

    @testset "Chow argument guards" begin
        @test_throws ArgumentError chow_test(m, 0; type = :breakpoint)
        @test_throws ArgumentError chow_test(m, n; type = :breakpoint)
        @test_throws ArgumentError chow_test(m, 30; type = :bogus)
        # a segment shorter than k obs is rejected for :breakpoint.
        @test_throws ArgumentError chow_test(m, 1; type = :breakpoint)
    end

    # -------------------------------------------------------------------------
    # 6. Influence statistics vs R stats::influence.measures (base, R 4.5.0).
    #    fit <- lm(y ~ x1 + x2)
    #    hatvalues(fit); rstudent(fit); rstandard(fit); dffits(fit);
    #    cooks.distance(fit); dfbetas(fit)
    # -------------------------------------------------------------------------
    @testset "influence stats (R influence.measures oracle)" begin
        inf = influence_stats(m)
        @test inf isa InfluenceStats
        @test length(inf.hat) == n
        @test size(inf.dfbetas) == (n, k)
        # Leverages sum to k (trace of the hat matrix).
        @test isapprox(sum(inf.hat), k; atol = 1e-8)

        # obs 1
        @test isapprox(inf.hat[1],              0.0374928919; atol = 1e-8)
        @test isapprox(inf.dffits[1],           0.3136091046; atol = 1e-8)
        @test isapprox(inf.cooksd[1],           0.0319294000; atol = 1e-8)
        @test isapprox(inf.student_external[1], 1.5889717114; atol = 1e-8)
        @test isapprox(inf.student_internal[1], 1.5681352191; atol = 1e-8)
        @test isapprox(inf.dfbetas[1, 1],  0.1684206626; atol = 1e-8)
        @test isapprox(inf.dfbetas[1, 2],  0.2262758024; atol = 1e-8)
        @test isapprox(inf.dfbetas[1, 3],  0.0462857519; atol = 1e-8)
        # obs 10
        @test isapprox(inf.hat[10],              0.0170323487; atol = 1e-8)
        @test isapprox(inf.dffits[10],           0.1721660772; atol = 1e-8)
        @test isapprox(inf.cooksd[10],           0.0097587197; atol = 1e-8)
        @test isapprox(inf.student_external[10], 1.3079162242; atol = 1e-8)
        @test isapprox(inf.dfbetas[10, 1],  0.1705212989; atol = 1e-8)
        @test isapprox(inf.dfbetas[10, 2], -0.0244569181; atol = 1e-8)
        @test isapprox(inf.dfbetas[10, 3], -0.0048573671; atol = 1e-8)
        # obs 60
        @test isapprox(inf.hat[60],              0.0669631979; atol = 1e-8)
        @test isapprox(inf.dffits[60],           0.1752277371; atol = 1e-8)
        @test isapprox(inf.cooksd[60],           0.0103387011; atol = 1e-8)
        @test isapprox(inf.student_external[60], 0.6540849351; atol = 1e-8)

        # Cook's D identity: D_i = (t_int²/k)·(h/(1-h)).
        for i in 1:n
            @test isapprox(inf.cooksd[i],
                (inf.student_internal[i]^2 / k) * (inf.hat[i] / (1 - inf.hat[i]));
                atol = 1e-10)
        end
    end

    # -------------------------------------------------------------------------
    # 7. Property: leverage-1 point is guarded (no Inf) and flagged.
    # -------------------------------------------------------------------------
    @testset "leverage-1 guard" begin
        # A regressor that is nonzero for exactly one observation gives that obs
        # leverage ≈ 1 (a dedicated dummy). Denominators must not blow up.
        rng = MersenneTwister(7)
        nn = 40
        z = randn(rng, nn)
        d = zeros(nn); d[nn] = 1.0            # unique dummy ⇒ h_nn = 1, exact fit
        Xd = hcat(ones(nn), z, d)
        yd = 1 .+ 0.5 .* z .+ randn(rng, nn)
        md = estimate_reg(yd, Xd; varnames = ["const", "z", "d"])
        infd = influence_stats(md)
        @test all(isfinite, infd.hat)
        @test all(isfinite, infd.dffits)
        @test all(isfinite, infd.cooksd)
        @test all(isfinite, infd.dfbetas)
        @test nn in infd.high_leverage      # leverage-1 obs flagged
        @test isapprox(infd.hat[nn], 1.0; atol = 1e-8)
    end

    # -------------------------------------------------------------------------
    # 8. Property: engineered mid-sample mean shift ⇒ CUSUM crosses its bound.
    # -------------------------------------------------------------------------
    @testset "CUSUM crosses under a mean shift" begin
        rng = MersenneTwister(123)
        nn = 120
        x1 = randn(rng, nn)
        yb = 1 .+ 0.5 .* x1 .+ randn(rng, nn)
        yb[61:end] .+= 3.0                   # coefficient/mean break at t=60
        Xb = hcat(ones(nn), x1)
        mb = estimate_reg(yb, Xb; varnames = ["const", "x1"])
        r = cusum_test(mb)
        @test r.crossed
        @test r.first_crossing !== nothing
        @test r.first_crossing > 60          # crossing appears after the break
        # CUSUMSQ (variance-shift sensitive) also reacts to the break.
        rsq = cusumsq_test(mb)
        @test rsq.crossed
    end

    # -------------------------------------------------------------------------
    # 9. Display, plotting, refs.
    # -------------------------------------------------------------------------
    @testset "display / plot / refs" begin
        r = cusum_test(m)
        rsq = cusumsq_test(m)
        inf = influence_stats(m)
        ct = chow_test(m, 30)

        for obj in (r, rsq, inf, ct)
            io = IOBuffer(); show(io, obj); s = String(take!(io))
            @test !isempty(s)
        end

        # plot_result dispatches produce non-empty HTML.
        for obj in (r, rsq, inf)
            p = plot_result(obj)
            @test occursin("svg", p.html) || occursin("<div", p.html)
        end

        # refs render for the new types.
        for obj in (r, rsq, inf, ct)
            io = IOBuffer(); refs(io, obj); s = String(take!(io))
            @test !isempty(s)
        end
        io = IOBuffer(); refs(io, :chow_test); @test !isempty(String(take!(io)))
        # A Chow-test RegDiagnosticResult must resolve to Chow (1960), NOT fall
        # through to the heteroskedasticity/serial-correlation reference list.
        io = IOBuffer(); refs(io, ct); s = String(take!(io))
        @test occursin("Chow", s)
        @test !occursin("White", s)
    end
end
