# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-34 (#442): Equality-of-distribution "Basic statistics" battery
# (location + scale) and rank correlations (Pearson/Spearman/Kendall).
#
# Oracle discipline (no invented numerics). Every pinned constant below was
# produced LIVE from base-R 4.5.0 `stats` at authoring time on the EXACT fixed
# vectors used here (all functions are installed — the strongest oracle of the
# batch). The exact R calls + seeds are given inline. `car::leveneTest` is NOT
# installed, so Levene/Brown-Forsythe are cross-checked by base-R
# `oneway.test(abs(x - center) ~ g, var.equal=TRUE)` on the deviations — the
# algebraically identical formulation — and THAT is pinned. In addition, two
# analytic-property oracles run: (a) at the exact/approx boundary (n=25 signed
# rank) the exact and continuity-corrected-normal p-values agree to a loose tol,
# and (b) Kendall's C-D via the merge-sort counter equals the O(n^2) brute-force
# concordant-minus-discordant count (with the tie-aware tau_b denominator).
#
# Fixed vectors (shared with R):
#   g1 = c(5.1,4.9,6.2,5.7,6.0,5.5); g2 = c(6.1,5.9,7.2,6.8,6.5)
#   g3 = c(7.0,7.5,6.9,8.1,7.3,7.8)

using Test, MacroEconometricModels, Random, Statistics, Distributions
using StatsAPI: pvalue, nobs

const G1 = [5.1, 4.9, 6.2, 5.7, 6.0, 5.5]
const G2 = [6.1, 5.9, 7.2, 6.8, 6.5]
const G3 = [7.0, 7.5, 6.9, 8.1, 7.3, 7.8]

# helper: build (y, g) for the 3-group battery
_yg3() = (vcat(G1, G2, G3), vcat(fill(1, 6), fill(2, 5), fill(3, 6)))
_yg2() = (vcat(G1, G2), vcat(fill(1, 6), fill(2, 5)))

@testset "EV-34 equality tests + rank correlations" begin

    # =====================================================================
    # Location: t-tests  (R: t.test)
    # =====================================================================
    @testset "t-tests" begin
        y, g = _yg2()
        # t.test(g1,g2,var.equal=TRUE)
        r = equality_test(y, g; test=:t, equal_var=true)
        @test r.test_name == :two_sample_t
        @test isapprox(r.statistic, -3.00153178463; atol=1e-8)
        @test isapprox(r.df1, 9.0; atol=1e-10)
        @test isapprox(r.pvalue, 0.0149192621712; atol=1e-9)
        # t.test(g1,g2,var.equal=FALSE)  (Welch, fractional df)
        rw = equality_test(y, g; test=:t, equal_var=false)
        @test rw.test_name == :welch_t
        @test isapprox(rw.statistic, -2.98991081885; atol=1e-8)
        @test isapprox(rw.df1, 8.50420246116; atol=1e-7)   # fractional Float!
        @test isapprox(rw.pvalue, 0.0161847711878; atol=1e-9)
        # one-sample  t.test(g1, mu=5.5)
        r1 = ttest(G1; mu=5.5)
        @test r1.test_name == :one_sample_t
        @test isapprox(r1.statistic, 0.323592400845; atol=1e-8)
        @test isapprox(r1.df1, 5.0)
        @test isapprox(r1.pvalue, 0.759343207411; atol=1e-9)
        # paired  t.test(pa,pb,paired=TRUE)
        pa = [12.0, 15, 14, 10, 13, 16, 11]; pb = [10.0, 14, 13, 12, 11, 15, 9]
        rp = ttest(pa, pb; paired=true)
        @test rp.test_name == :paired_t
        @test isapprox(rp.statistic, 1.87082869339; atol=1e-8)
        @test isapprox(rp.pvalue, 0.110551740402; atol=1e-9)
    end

    # =====================================================================
    # Location: ANOVA  (R: oneway.test)
    # =====================================================================
    @testset "ANOVA (classic + Welch)" begin
        y, g = _yg3()
        # oneway.test(val~grp, var.equal=TRUE)
        r = anova_test(y, g; equal_var=true)
        @test r.test_name == :anova
        @test isapprox(r.statistic, 21.2301740812; atol=1e-7)
        @test isapprox(r.df1, 2.0); @test isapprox(r.df2, 14.0)
        @test isapprox(r.pvalue, 5.76356753617e-05; atol=1e-12)
        # oneway.test(val~grp, var.equal=FALSE)  (Welch ANOVA)
        rw = anova_test(y, g; equal_var=false)
        @test rw.test_name == :welch_anova
        @test isapprox(rw.statistic, 20.7779236956; atol=1e-6)
        @test isapprox(rw.df1, 2.0)
        @test isapprox(rw.df2, 9.01045714829; atol=1e-7)
        @test isapprox(rw.pvalue, 0.000421762966749; atol=1e-11)
    end

    # =====================================================================
    # Scale: variance F / Bartlett / Levene / Brown-Forsythe
    # =====================================================================
    @testset "scale tests" begin
        y2, g2v = _yg2(); y3, g3v = _yg3()
        # var.test(g1,g2)
        rf = equality_test(y2, g2v; test=:f)
        @test isapprox(rf.statistic, 0.926060606061; atol=1e-9)
        @test isapprox(rf.df1, 5.0); @test isapprox(rf.df2, 4.0)
        @test isapprox(rf.pvalue, 0.910295283879; atol=1e-9)
        # bartlett.test(val~grp)
        rb = equality_test(y3, g3v; test=:bartlett)
        @test isapprox(rb.statistic, 0.0667470170645; atol=1e-9)
        @test isapprox(rb.df1, 2.0)
        @test isapprox(rb.pvalue, 0.967177243163; atol=1e-9)
        # Levene (center=mean): oneway.test(abs(val-ave(val,grp,mean))~grp, var.equal=TRUE)
        rl = equality_test(y3, g3v; test=:levene)
        @test rl.test_name == :levene
        @test isapprox(rl.statistic, 0.0343137254902; atol=1e-9)
        @test isapprox(rl.df1, 2.0); @test isapprox(rl.df2, 14.0)
        @test isapprox(rl.pvalue, 0.966349318232; atol=1e-9)
        # Brown-Forsythe (center=median)
        rbf = equality_test(y3, g3v; test=:brown_forsythe)
        @test rbf.test_name == :brown_forsythe
        @test isapprox(rbf.statistic, 0.0338015803336; atol=1e-9)
        @test isapprox(rbf.pvalue, 0.966841958614; atol=1e-9)
    end

    # =====================================================================
    # Nonparametric location: Mann-Whitney / signed-rank / Kruskal-Wallis
    # =====================================================================
    @testset "Mann-Whitney U" begin
        y2, g2v = _yg2()
        # wilcox.test(g1,g2,exact=TRUE): W=3, p=0.0303030303
        rmw = equality_test(y2, g2v; test=:mann_whitney)
        @test rmw.test_name == :mann_whitney
        @test rmw.exact
        @test isapprox(rmw.statistic, 3.0; atol=1e-10)
        @test isapprox(rmw.pvalue, 0.030303030303; atol=1e-9)
        # with ties -> normal fallback. tA,tB below (wilcox.test(tA,tB,correct=TRUE))
        tA = [1.0, 2, 2, 4, 5, 7]; tB = [2.0, 3, 5, 6, 8, 9]
        yt = vcat(tA, tB); gt = vcat(fill(1, 6), fill(2, 6))
        rmt = equality_test(yt, gt; test=:mann_whitney)
        @test !rmt.exact
        @test isapprox(rmt.statistic, 9.5; atol=1e-10)   # R W = 9.5
        @test isapprox(rmt.pvalue, 0.196228348652; atol=1e-8)
    end

    @testset "Wilcoxon signed-rank" begin
        # exact, distinct abs diffs: da-db = 1,3,-2,6,-5,8,-7
        da = [11.0, 13, 9, 16, 10, 18, 7]; db = [10.0, 10, 11, 10, 15, 10, 14]
        y = vcat(da, db); g = vcat(fill(1, 7), fill(2, 7))
        rs = equality_test(y, g; test=:wilcoxon)
        @test rs.test_name == :wilcoxon_signed_rank
        @test rs.exact
        @test isapprox(rs.statistic, 16.0; atol=1e-10)          # R V=16
        @test isapprox(rs.pvalue, 0.8125; atol=1e-9)
        # ties -> normal. pa,pb (diffs 2,1,1,-2,2,1,2): R V=22.5, p=0.165327154
        pa = [12.0, 15, 14, 10, 13, 16, 11]; pb = [10.0, 14, 13, 12, 11, 15, 9]
        yp = vcat(pa, pb); gp = vcat(fill(1, 7), fill(2, 7))
        rsn = equality_test(yp, gp; test=:wilcoxon)
        @test !rsn.exact
        @test isapprox(rsn.statistic, 22.5; atol=1e-10)
        @test isapprox(rsn.pvalue, 0.165327154231; atol=1e-8)
    end

    @testset "Kruskal-Wallis (tie-corrected)" begin
        y3, g3v = _yg3()
        # kruskal.test(val~grp): H=12.1712418301, p=0.00227535
        rk = equality_test(y3, g3v; test=:kruskal_wallis)
        @test isapprox(rk.statistic, 12.1712418301; atol=1e-7)
        @test isapprox(rk.df1, 2.0)
        @test isapprox(rk.pvalue, 0.00227535108586; atol=1e-10)
        # with ties: vt2/gt2
        vt2 = Float64[1, 2, 2, 3, 3, 4, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9]
        gt2 = vcat(fill(1, 6), fill(2, 5), fill(3, 6))
        rkt = equality_test(vt2, gt2; test=:kruskal_wallis)
        @test isapprox(rkt.statistic, 13.176509512; atol=1e-7)
        @test isapprox(rkt.pvalue, 0.00137644009242; atol=1e-10)
    end

    @testset "van der Waerden + Mood median" begin
        y3, g3v = _yg3()
        # van der Waerden (hand-recomputed in R): T=11.617826658, df=2, p=0.00300069
        rv = equality_test(y3, g3v; test=:van_der_waerden)
        @test isapprox(rv.statistic, 11.617826658; atol=1e-7)
        @test isapprox(rv.df1, 2.0)
        @test isapprox(rv.pvalue, 0.00300068906308; atol=1e-10)
        # Mood median (above = x > grand-median; Pearson chisq, no continuity):
        # chi=12.1833333333, df=2, p=0.00226163638
        rmd = equality_test(y3, g3v; test=:median)
        @test isapprox(rmd.statistic, 12.1833333333; atol=1e-7)
        @test isapprox(rmd.pvalue, 0.00226163637794; atol=1e-10)
    end

    @testset "Siegel-Tukey" begin
        y2, g2v = _yg2()
        # ST ranks assigned from extremes inward; group-1 rank sum = 36 (manual R).
        rst = equality_test(y2, g2v; test=:siegel_tukey)
        @test isapprox(rst.statistic, 36.0; atol=1e-10)   # W1 rank sum
        # U1 = 36 - 21 = 15 = mean -> two-sided exact p = 1
        @test isapprox(rst.pvalue, 1.0; atol=1e-9)
    end

    # =====================================================================
    # Association: Pearson / Spearman / Kendall  (R: cor.test)
    # =====================================================================
    @testset "cor_test Pearson" begin
        ax = [10.0, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
        ay = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
        cp = cor_test(ax, ay; method=:pearson)
        @test isapprox(cp.estimate, 0.816420516345; atol=1e-9)
        @test isapprox(cp.statistic, 4.24145528889; atol=1e-8)
        @test isapprox(cp.df, 9.0)
        @test isapprox(cp.pvalue, 0.00216962887308; atol=1e-10)
        @test isapprox(cp.ci_lower, 0.424391213393; atol=1e-8)
        @test isapprox(cp.ci_upper, 0.95069325378656; atol=1e-8)
    end

    @testset "cor_test Spearman" begin
        ax = [10.0, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
        ay = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
        # cor.test(method="spearman", exact=FALSE): rho=0.8181818, S=40, p=0.00208314
        cs = cor_test(ax, ay; method=:spearman)
        @test isapprox(cs.estimate, 0.818181818182; atol=1e-9)
        @test isapprox(cs.statistic, 40.0; atol=1e-8)     # S = Σ dᵢ²
        @test isapprox(cs.pvalue, 0.00208314484048; atol=1e-9)
        # with ties: tx,ty  rho=0.9444444, S=3.1111, p=0.00135589
        tx = [1.0, 2, 2, 3, 4, 4, 5]; ty = [1.0, 1, 2, 3, 3, 4, 5]
        cst = cor_test(tx, ty; method=:spearman)
        @test isapprox(cst.estimate, 0.944444444444; atol=1e-9)
        @test isapprox(cst.statistic, 3.11111111111; atol=1e-8)
        @test isapprox(cst.pvalue, 0.00135588583532; atol=1e-9)
    end

    @testset "cor_test Kendall" begin
        ax = [10.0, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
        ay = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
        # no ties -> exact. cor.test(method="kendall"): tau=0.6363636, T=45, p=0.00570717
        ck = cor_test(ax, ay; method=:kendall)
        @test ck.exact
        @test isapprox(ck.estimate, 0.636363636364; atol=1e-9)   # τ_a
        @test isapprox(ck.statistic, 45.0; atol=1e-10)           # concordant count
        @test isapprox(ck.pvalue, 0.0057071709155; atol=1e-9)
        # ties in both -> tau_b, normal. tx,ty: tau_b=0.8947368, z=2.6705074, p=0.00757367
        tx = [1.0, 2, 2, 3, 4, 4, 5]; ty = [1.0, 1, 2, 3, 3, 4, 5]
        ckt = cor_test(tx, ty; method=:kendall)
        @test !ckt.exact
        @test isapprox(ckt.estimate, 0.894736842105; atol=1e-9)  # τ_b
        @test isapprox(ckt.statistic, 2.67050741821; atol=1e-8)  # z
        @test isapprox(ckt.pvalue, 0.00757366980412; atol=1e-9)
    end

    # =====================================================================
    # Analytic-property oracles
    # =====================================================================
    @testset "property: signed-rank exact≈normal at n=25 boundary" begin
        # d25 (distinct abs values 1..25, random signs; seed 7 in R) as differences
        d25 = Float64[1, -2, -3, 4, -5, 6, -7, 8, 9, 10, 11, -12, 13, 14, 15,
                      -16, 17, -18, 19, 20, -21, 22, 23, 24, -25]
        za = d25; zb = zeros(25)
        y = vcat(za, zb); g = vcat(fill(1, 25), fill(2, 25))
        r = equality_test(y, g; test=:wilcoxon)
        @test r.exact
        @test isapprox(r.statistic, 216.0; atol=1e-10)          # R V=216
        @test isapprox(r.pvalue, 0.156338214874; atol=1e-8)     # exact
        # continuity-corrected normal p (R correct=TRUE) = 0.153849198538
        # exact and normal agree to a loose tolerance at the boundary:
        @test isapprox(r.pvalue, 0.153849198538; atol=5e-3)
    end

    @testset "property: Kendall merge-sort C-D = brute force" begin
        rng = Random.MersenneTwister(2024)
        for _ in 1:40
            n = rand(rng, 5:30)
            x = rand(rng, 1:6, n) .|> Float64   # forces ties in both x and y
            y = rand(rng, 1:6, n) .|> Float64
            # brute-force concordant - discordant (O(n^2))
            cmd = 0
            for i in 1:n-1, j in i+1:n
                sx = sign(x[i] - x[j]); sy = sign(y[i] - y[j])
                cmd += sx * sy
            end
            # τ_b from our implementation and from brute force
            ckt = cor_test(x, y; method=:kendall)
            tot = n * (n - 1) ÷ 2
            # recompute pair-tie counts for the τ_b denominator
            xt = 0; yt = 0
            for v in (x, y)
                cnt = Dict{Float64,Int}()
                for a in v; cnt[a] = get(cnt, a, 0) + 1; end
                s = sum(c * (c - 1) ÷ 2 for c in values(cnt))
                v === x ? (xt = s) : (yt = s)
            end
            denom = sqrt((tot - xt) * (tot - yt))
            tau_b_bf = denom > 0 ? cmd / denom : NaN
            if isfinite(tau_b_bf)
                @test isapprox(ckt.estimate, tau_b_bf; atol=1e-10)
            end
        end
    end

    # =====================================================================
    # Data-container dispatch + display + refs
    # =====================================================================
    @testset "CrossSectionData / display / refs" begin
        y3, g3v = _yg3()
        data = hcat(Float64.(y3), Float64.(g3v))
        d = CrossSectionData(data; varnames=["val", "grp"])
        rc = equality_test(d, :val, :grp; test=:kruskal_wallis)
        @test isapprox(rc.statistic, 12.1712418301; atol=1e-7)
        ax = [10.0, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
        ay = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
        dc = CrossSectionData(hcat(ax, ay); varnames=["x", "y"])
        cc = cor_test(dc, :x, :y; method=:pearson)
        @test isapprox(cc.estimate, 0.816420516345; atol=1e-9)
        # display renders
        io = IOBuffer(); show(io, MIME"text/plain"(), rc); s = String(take!(io))
        @test occursin("Kruskal", s); @test occursin("P-value", s)
        io2 = IOBuffer(); show(io2, MIME"text/plain"(), cc); s2 = String(take!(io2))
        @test occursin("Pearson", s2)
        # refs render
        @test occursin("Kruskal", refs(rc))
        @test occursin("Kendall", refs(cor_test(ax, ay; method=:kendall)))
        # StatsAPI
        @test pvalue(rc) == rc.pvalue
        @test nobs(rc) == 17
    end

    @testset "input validation" begin
        y2, g2v = _yg2()
        @test_throws ArgumentError equality_test(_yg3()[1], _yg3()[2]; test=:t)  # needs 2 groups
        @test_throws ArgumentError equality_test(y2, g2v; test=:nonsense)
        @test_throws ArgumentError cor_test([1.0, 2], [1.0, 2]; method=:pearson) # n<3
        @test_throws ArgumentError cor_test([1.0, 2, 3], [1.0, 2, 3]; method=:bad)
    end
end
