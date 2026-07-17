# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-30 (#438): explosive / rational-bubble detection — SADF (Phillips-Wu-Yu
# 2011) / GSADF (Phillips-Shi-Yu 2015) with BSADF date-stamping.
#
# ORACLE LAYERING (R `exuber` is NOT installed in this environment; Stata
# unavailable). We therefore validate with:
#   (1) IN-ENV IDENTITY oracle (PRIMARY): the full-sample ADF window statistic
#       ADF(1,T; p=0, const) computed by the sup-ADF kernel must EXACTLY equal
#       the already-validated `adf_test(y; lags=0, regression=:constant)`
#       statistic — an independent, live cross-implementation inside the repo.
#   (2) INDEPENDENT hand-recomputation of the ADF-window t-stat by a plain OLS
#       coded inline in this test.
#   (3) ANALYTIC / PROPERTY oracles: GSADF ≥ SADF (double sup dominates single
#       sup); RIGHT-TAILED behaviour (reject for LARGE stats via UPPER
#       quantiles); an embedded φ=1.05 explosive window is date-stamped with an
#       episode overlapping the true interval; a pure random walk yields no
#       episode with high probability.
#   (4) PUBLISHED-with-citation magnitude check: PSY (2015, IER 56(4), Table 1)
#       report the finite-sample GSADF 95% critical value near ≈2.1 for T≈100
#       (r0≈0.1). Our simulated CV (larger auto r0 ⇒ fewer windows ⇒ slightly
#       lower) must land in a loose published band. This is PUBLISHED-NOT-LIVE.
#   (5) SEEDED-MC reproducibility anchors: with mc_reps and seed fixed the
#       simulated 95% CV reproduces a hard-coded constant within a loose tol.
#       (The SADF/GSADF *statistics* depend only on the data, so they are
#       reproducibility anchors of THIS implementation — no live R run exists.)

using Test, MacroEconometricModels, Random, Statistics, LinearAlgebra

# Independent, from-scratch ADF-window t-statistic (constant + p lags), used as
# a hand-recomputation oracle for the internal sup-ADF kernel.
function _ref_adf_tstat(w::Vector{Float64}, p::Int)
    L = length(w)
    dw = diff(w)
    n = (L - 1) - p
    X = zeros(n, p + 2)
    Y = zeros(n)
    for r in 1:n
        m = p + r                      # index within dw
        Y[r] = dw[m]
        X[r, 1] = 1.0
        X[r, 2] = w[m]                 # level lag w_{t-1}
        for j in 1:p
            X[r, 2 + j] = dw[m - j]
        end
    end
    B = (X'X) \ (X'Y)
    resid = Y - X * B
    s2 = sum(abs2, resid) / (n - (p + 2))
    se = sqrt(s2 * inv(X'X)[2, 2])
    return B[2] / se
end

@testset "SADF/GSADF bubble detection (EV-30)" begin

    # Fixed-seed pure random walk, T=100. Statistics are deterministic in y.
    rng = MersenneTwister(12345)
    y_rw = cumsum(randn(rng, 100))

    @testset "in-env identity + hand-recomputation oracle" begin
        s = sadf_test(y_rw; mc_reps=99, seed=777)
        # (1) full-sample ADF window == already-validated adf_test(lags=0)
        adf_full = s.bsadf[end]                        # ADF(1,T; p=0, const)
        a = adf_test(y_rw; lags=0, regression=:constant)
        @test isapprox(adf_full, a.statistic; atol=1e-8)
        # (2) independent OLS hand-recomputation of the same window statistic
        @test isapprox(adf_full, _ref_adf_tstat(y_rw, 0); atol=1e-8)
        # A short interior window ADF(51,80) hand-recomputation
        s2 = sadf_test(y_rw; r0=0.30, mc_reps=50, seed=1)  # ensures 30-obs window in seq
        # recompute an interior fixed-start window ADF(1,60)
        @test isapprox(_ref_adf_tstat(y_rw[1:60], 0),
                       sadf_test(y_rw; mc_reps=20, seed=1).bsadf[60 - 19 + 1]; atol=1e-8)
    end

    @testset "GSADF ≥ SADF (double-sup dominance)" begin
        s = sadf_test(y_rw; mc_reps=99, seed=777)
        g = gsadf_test(y_rw; mc_reps=99, seed=777)
        # GSADF sup over both endpoints ≥ fixed-start SADF sup
        @test g.statistic >= s.statistic - 1e-10
    end

    @testset "reproducibility anchors (statistics deterministic in y)" begin
        # Hard-coded from THIS implementation on the fixed seed above (see header:
        # no live R `exuber::radf(y, minw=19)` run is available in this env).
        # R call that WOULD produce the analogue: exuber::radf(y, minw = 19).
        s = sadf_test(y_rw; mc_reps=299, seed=777)
        g = gsadf_test(y_rw; mc_reps=299, seed=777)
        @test isapprox(s.statistic, 0.5708301611; atol=1e-6)
        @test isapprox(g.statistic, 0.8105936338; atol=1e-6)
        @test s.r0 ≈ 0.19 atol=1e-9              # auto rule 0.01 + 1.8/√100
        @test g.r2_index[1] == 19                # swindow0 = floor(0.19*100)
    end

    @testset "seeded MC critical-value reproduction + published band" begin
        g = gsadf_test(y_rw; mc_reps=299, seed=777)
        # (5) seeded-MC anchor: deterministic given mc_reps + seed. Loose tol.
        @test isapprox(g.critical_values[5], 1.972; atol=0.15)
        # CV ordering: 90% < 95% < 99%
        @test g.critical_values[10] < g.critical_values[5] < g.critical_values[1]
        # (4) PUBLISHED-NOT-LIVE magnitude band vs PSY (2015) Table 1: finite-
        # sample GSADF 95% CV ≈ 2.1 for T≈100 (r0≈0.1). Our larger auto r0=0.19
        # trims windows, so a broad [1.4, 2.6] published band.
        @test 1.4 < g.critical_values[5] < 2.6
        # cv sequence has one CV per r2 endpoint
        @test length(g.cv_seq) == length(g.bsadf) == length(g.r2_index)
    end

    @testset "right-tailed inference (reject for LARGE stats)" begin
        g = gsadf_test(y_rw; mc_reps=299, seed=777)
        # random walk: statistic below the 90% upper CV ⇒ fail to reject
        @test g.statistic < g.critical_values[10]
        @test g.pvalue > 0.10
        # p-value is a right-tail area: monotone — a huge stat gives tiny p
        y_expl = copy(y_rw)
        for t in 40:70; y_expl[t] = 1.06 * y_expl[t-1]; end
        ge = gsadf_test(y_expl; mc_reps=299, seed=777)
        @test ge.statistic > ge.critical_values[1]   # reject at 1%
        @test ge.pvalue < 0.05
        @test ge.pvalue <= g.pvalue                  # larger stat ⇒ smaller right-tail p
    end

    @testset "date-stamping: embedded explosive window is stamped" begin
        # RW with an embedded φ=1.05 explosive interval [50,90] inside T=150
        # (PSY 2015 date-stamping oracle; loose overlap, seeded).
        function make_bubble(seed; T=150, a=50, b=90, phi=1.05)
            r = MersenneTwister(seed); yb = zeros(T)
            for t in 2:T
                yb[t] = (a <= t <= b ? phi : 1.0) * yb[t-1] + randn(r)
            end
            yb
        end
        found = 0
        seeds = (2024, 7, 55, 101, 3, 42)
        for sd in seeds
            yb = make_bubble(sd)
            g = gsadf_test(yb; mc_reps=199, seed=99)
            # at least one stamped episode overlapping the true window [50,90]
            if any(ep -> ep[1] <= 90 && ep[2] >= 50, g.episodes)
                found += 1
            end
        end
        @test found >= length(seeds) - 1             # loose overlap across seeds
        # episodes are index pairs into y (start ≤ end, within [1,T])
        g = gsadf_test(make_bubble(2024); mc_reps=199, seed=99)
        @test !isempty(g.episodes)
        for (s0, e0) in g.episodes
            @test 1 <= s0 <= e0 <= 150
            @test e0 - s0 + 1 >= ceil(Int, log(150))  # min-duration rule
        end
    end

    @testset "pure random walk yields no episode w.h.p." begin
        noep = 0
        for sd in 1:10
            yr = cumsum(randn(MersenneTwister(3000 + sd), 120))
            r = gsadf_test(yr; mc_reps=149, seed=99)
            isempty(r.episodes) && (noep += 1)
        end
        @test noep >= 8                              # ≥80% clean under the null
    end

    @testset "wild-bootstrap CVs (Phillips-Shi 2020)" begin
        yb = let r = MersenneTwister(2024), T = 120, yy = zeros(T)
            for t in 2:T; yy[t] = (50 <= t <= 75 ? 1.05 : 1.0) * yy[t-1] + randn(r); end
            yy
        end
        gw = gsadf_test(yb; mc_reps=199, seed=99, cv=:wildboot)
        @test gw.cv_method == :wildboot
        @test isfinite(gw.critical_values[5])
        @test gw.critical_values[10] < gw.critical_values[5] < gw.critical_values[1]
        @test gw.statistic > gw.critical_values[5]   # strong bubble still rejects
    end

    @testset "adflag augmenting lags" begin
        g0 = gsadf_test(y_rw; adflag=0, mc_reps=99, seed=5)
        g2 = gsadf_test(y_rw; adflag=2, mc_reps=99, seed=5)
        @test g0.adflag == 0 && g2.adflag == 2
        @test isfinite(g2.statistic)
        @test MacroEconometricModels.StatsAPI.dof(g2) == 4   # adflag + 2
    end

    @testset "CV caching (asymptotic) is keyed and reused" begin
        empty!(MacroEconometricModels._BUBBLE_CV_CACHE)
        g1 = gsadf_test(y_rw; mc_reps=99, seed=777)
        n_after_first = length(MacroEconometricModels._BUBBLE_CV_CACHE)
        @test n_after_first >= 1
        g2 = gsadf_test(y_rw; mc_reps=99, seed=777)  # identical key ⇒ cache hit
        @test g1.critical_values[5] == g2.critical_values[5]
        @test length(MacroEconometricModels._BUBBLE_CV_CACHE) == n_after_first
    end

    @testset "display / refs / plot render" begin
        g = gsadf_test(y_rw; mc_reps=99, seed=777)
        buf = IOBuffer(); show(buf, g); str = String(take!(buf))
        @test occursin("GSADF", str)
        @test occursin("Critical", str)
        # report() forwards to show for AbstractUnitRootTest
        rbuf = IOBuffer(); refs(rbuf, g); @test occursin("Phillips", String(take!(rbuf)))
        p = plot_result(g)
        @test p isa PlotOutput
        s = sadf_test(y_rw; mc_reps=99, seed=777)
        sbuf = IOBuffer(); show(sbuf, s); @test occursin("SADF", String(take!(sbuf)))
        @test plot_result(s) isa PlotOutput
    end

    @testset "argument validation" begin
        @test_throws ArgumentError sadf_test(randn(10))               # T too small
        @test_throws ArgumentError gsadf_test(y_rw; cv=:bogus)
        @test_throws ArgumentError gsadf_test(y_rw; adflag=-1)
        @test_throws ArgumentError gsadf_test(y_rw; r0=1.5)
    end
end
