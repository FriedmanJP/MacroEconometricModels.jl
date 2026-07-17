# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-24 (#432): Dumitrescu-Hurlin (2012) heterogeneous-panel Granger
# non-causality test.
#
# Oracle discipline (no invented numerics; see the EV-24 spec's oracle_strategy):
#   (1) CROSS-IMPLEMENTATION vs base R. `plm` cannot be compiled in this
#       environment (its dependency `lmtest` fails to build — same toolchain
#       problem noted for the EV-20/EV-21 panel tests), so the DH statistics were
#       recomputed independently in base R (`lm` + `anova` per unit → F test for
#       the p lagged-x restrictions → W_i = p·F, then averaged/standardized) on
#       the EXACT fixed-seed panel this test builds (`make_dh_panel()` below,
#       exported to /tmp/dh_panel_h.csv). The pinned literals are that R output
#       (10 significant digits). CRITICAL convention: R/`plm::pgrangertest` report
#       the F-based W_i = F; this package uses the χ²(p) form W = p·F, so the
#       oracle rescales by p (documented at each literal). See the R script in the
#       commit message / EV-24 report.
#   (2) ANALYTIC N=1 identity: the per-unit W_i equals p·F where F is the
#       classical joint-significance F statistic, recomputed here from separate
#       restricted/unrestricted RSS — fully independent of the θ'V⁻¹θ Wald path.
#   (3) EXACT MOMENTS: E[W_i], Var[W_i] (DH 2012 eqs. 26-27) match closed forms.
#   (4) Properties: standard-normal size under the non-causality null (seeded MC),
#       power under a causal DGP, 0≤p≤1, the T>2p+5 guard errors (not NaN), and
#       bootstrap reproducibility under a fixed seed.

using Test, MacroEconometricModels, Random, LinearAlgebra, Statistics, DataFrames
using Distributions, DelimitedFiles

# Fixed heterogeneous panel identical to the one fed to the R oracle.
# MersenneTwister is not stable across Julia versions, so the exact (y,x) panel
# the R oracle saw is committed as data/dh_panel.csv (test/gen_ev_fixtures.jl).
# x[t-1], x[t-2] enter y ⇒ x Granger-causes y in every unit.
function make_dh_panel()
    N = 8; T = 30
    d = readdlm(joinpath(@__DIR__, "data", "dh_panel.csv"), ',', Float64)
    ids = repeat(1:N, inner = T)
    tt = repeat(1:T, N)
    DataFrame(id = ids, time = tt, y = d[:, 1], x = d[:, 2])
end

# Independent classical F-test for joint significance of the p lagged-x
# coefficients in a single-unit ADL(p) regression (restricted vs unrestricted
# RSS). Returns (F, m, k). Used for the N=1 analytic identity.
function _naive_dh_F(y::Vector{Float64}, x::Vector{Float64}, p::Int)
    Traw = length(y); m = Traw - p; k = 2p + 1
    yd = y[(p+1):Traw]
    Xu = ones(m, k); Xr = ones(m, p + 1)
    for t in 1:m, l in 1:p
        Xu[t, 1 + l]     = y[p + t - l]
        Xu[t, 1 + p + l] = x[p + t - l]
        Xr[t, 1 + l]     = y[p + t - l]
    end
    ru = yd - Xu * (Xu \ yd); rr = yd - Xr * (Xr \ yd)
    rssu = dot(ru, ru); rssr = dot(rr, rr)
    F = ((rssr - rssu) / p) / (rssu / (m - k))
    return (F, m, k)
end

@testset "Dumitrescu-Hurlin panel Granger non-causality (EV-24)" begin

    df = make_dh_panel()
    pd = xtset(df, :id, :time)

    @testset "Cross-implementation oracle vs base R (χ²(p) = p·F)" begin
        # Provenance: base R on /tmp/dh_panel_h.csv (this exact panel).
        #   for each unit: full <- lm(yd ~ yl1..ylp + xl1..xlp)
        #                  restr <- lm(yd ~ yl1..ylp)
        #                  W_i <- p * anova(restr, full)$F   # χ²(p) rescale of F
        #   Wbar=mean(W_i); Zbar=sqrt(N/(2p))(Wbar-p);
        #   Ztilde=sqrt(N)(Wbar-mean(E))/sqrt(mean(V))  with DH eqs 26-27 moments.
        # R output (Rscript, 10 sig digits):
        #   p=1: Wbar=2.3659047723 Zbar=2.7318095446 Ztilde=2.2212159842
        #        Zbar_p=3.149377e-03 Ztilde_p=1.316817e-02
        #   p=2: Wbar=3.7675389818 Zbar=2.4996776000 Ztilde=1.8508376452
        #        Zbar_p=6.215319e-03 Ztilde_p=3.209646e-02
        r1 = dh_causality_test(pd, :x, :y; p = 1)
        @test isapprox(r1.Wbar,   2.3659047723; atol = 1e-8)
        @test isapprox(r1.Zbar,   2.7318095446; atol = 1e-8)
        @test isapprox(r1.Ztilde, 2.2212159842; atol = 1e-8)
        @test isapprox(r1.Zbar_pvalue,   3.149377e-03; rtol = 1e-5)
        @test isapprox(r1.Ztilde_pvalue, 1.316817e-02; rtol = 1e-5)
        @test r1.N == 8 && r1.p == 1 && r1.nobs == 29 && r1.n_skipped == 0

        r2 = dh_causality_test(pd, :x, :y; p = 2)
        @test isapprox(r2.Wbar,   3.7675389818; atol = 1e-8)
        @test isapprox(r2.Zbar,   2.4996776000; atol = 1e-8)
        @test isapprox(r2.Ztilde, 1.8508376452; atol = 1e-8)
        @test isapprox(r2.Zbar_pvalue,   6.215319e-03; rtol = 1e-5)
        @test isapprox(r2.Ztilde_pvalue, 3.209646e-02; rtol = 1e-5)
        @test r2.nobs == 28
    end

    @testset "Exact finite-T moments (DH 2012 eqs. 26-27)" begin
        # Stated (T, p) = (28, 2): effective sample T=28, p=2.
        #   E[W]   = p(T-2p-1)/(T-2p-3) = 2·23/21
        #   Var[W] = 2p(T-2p-1)²(T-p-3)/[(T-2p-3)²(T-2p-5)]
        #          = 4·23²·23 / (21²·19)
        E28 = MacroEconometricModels._dh_ew(28, 2)
        V28 = MacroEconometricModels._dh_varw(28, 2)
        @test isapprox(E28, 2 * 23 / 21; atol = 1e-12)
        @test isapprox(V28, 4 * 23^2 * 23 / (21^2 * 19); atol = 1e-10)
        # χ²(p) asymptotic limit: as T→∞, E→p and Var→2p.
        @test isapprox(MacroEconometricModels._dh_ew(100000, 3), 3.0; atol = 1e-3)
        @test isapprox(MacroEconometricModels._dh_varw(100000, 3), 6.0; atol = 1e-2)
    end

    @testset "N=1 analytic identity: W_i = p·F (independent RSS F-test)" begin
        for p in (1, 2, 3)
            rng = MersenneTwister(77 + p)
            T = 60
            x = randn(rng, T)
            y = zeros(T)
            for t in (p+1):T
                y[t] = 0.5 * y[t-1] + 0.6 * x[t-1] + randn(rng)
            end
            df1 = DataFrame(id = ones(Int, T), time = 1:T, y = y, x = x)
            pd1 = xtset(df1, :id, :time)
            r = dh_causality_test(pd1, :x, :y; p = p)
            F, m, k = _naive_dh_F(y, x, p)
            @test r.N == 1
            @test isapprox(r.W_i[1], p * F; rtol = 1e-9)   # χ²(p) = p·F
        end
    end

    @testset "N=1 causality direction (rejects when x drives y, not otherwise)" begin
        # x strongly drives y.
        rng = MersenneTwister(2024)
        T = 120
        x = randn(rng, T); y = zeros(T)
        for t in 2:T
            y[t] = 0.3 * y[t-1] + 1.2 * x[t-1] + randn(rng)
        end
        pdc = xtset(DataFrame(id = ones(Int, T), time = 1:T, y = y, x = x), :id, :time)
        rc = dh_causality_test(pdc, :x, :y; p = 1)
        @test rc.Ztilde_pvalue < 0.01          # reject non-causality

        # Independent series: x should NOT Granger-cause y.
        rng2 = MersenneTwister(999)
        xi = randn(rng2, T); yi = zeros(T)
        for t in 2:T
            yi[t] = 0.3 * yi[t-1] + randn(rng2)
        end
        pdi = xtset(DataFrame(id = ones(Int, T), time = 1:T, y = yi, x = xi), :id, :time)
        ri = dh_causality_test(pdi, :x, :y; p = 1)
        @test ri.Ztilde_pvalue > 0.10          # fail to reject
    end

    @testset "Standard-normal size under the non-causality null (seeded MC)" begin
        # Under H0 (x ⊥ causes on y), Z̄ and Z̃ are ~N(0,1); the right-tailed 5%
        # test rejects ≈5% of the time. Monte Carlo over independent panels.
        p = 1; N = 12; T = 40; reps = 300
        rejZbar = 0; rejZtil = 0
        zbars = Float64[]
        rng = MersenneTwister(4242)
        for r in 1:reps
            ids = Int[]; tt = Int[]; ys = Float64[]; xs = Float64[]
            for i in 1:N
                x = randn(rng, T); y = zeros(T)
                for t in 2:T
                    y[t] = 0.4 * y[t-1] + randn(rng)   # y independent of x
                end
                append!(ids, fill(i, T)); append!(tt, 1:T)
                append!(ys, y); append!(xs, x)
            end
            pdm = xtset(DataFrame(id = ids, time = tt, y = ys, x = xs), :id, :time)
            res = dh_causality_test(pdm, :x, :y; p = p)
            push!(zbars, res.Zbar)
            res.Zbar_pvalue   < 0.05 && (rejZbar += 1)
            res.Ztilde_pvalue < 0.05 && (rejZtil += 1)
        end
        # Mean of Z̄ near 0 (loose MC band) and empirical size near nominal 5%.
        @test abs(mean(zbars)) < 0.35
        @test 0.01 < rejZbar / reps < 0.12
        @test 0.01 < rejZtil / reps < 0.12
    end

    @testset "Power under a causal panel DGP" begin
        df = make_dh_panel()          # x → y in every unit
        pd = xtset(df, :id, :time)
        r = dh_causality_test(pd, :x, :y; p = 1)
        @test r.Ztilde_pvalue < 0.05
        @test r.Zbar_pvalue   < 0.01
        @test r.Wbar > r.p            # W̄ above its null mean p
    end

    @testset "Basic properties & guards" begin
        df = make_dh_panel(); pd = xtset(df, :id, :time)
        r = dh_causality_test(pd, :x, :y; p = 2)
        for pv in (r.Zbar_pvalue, r.Ztilde_pvalue)
            @test 0.0 <= pv <= 1.0
            @test isfinite(pv)
        end
        @test length(r.W_i) == r.N
        @test all(w -> w >= 0 && isfinite(w), r.W_i)
        @test isnan(r.bootstrap_pvalue)          # no bootstrap requested

        # p ≥ 1 required.
        @test_throws ArgumentError dh_causality_test(pd, :x, :y; p = 0)
        # Same variable both sides.
        @test_throws ArgumentError dh_causality_test(pd, :x, :x; p = 1)

        # T > 2p+5 guard: with T=30 and a large p every unit's effective sample
        # (T-p) falls below 2p+5 ⇒ errors (not NaN). Need T-p ≤ 2p+5 ⇒ p ≥ 8.33.
        @test_throws ArgumentError dh_causality_test(pd, :x, :y; p = 9)
    end

    @testset "Bootstrap reproducibility (seeded, CSD block bootstrap)" begin
        df = make_dh_panel(); pd = xtset(df, :id, :time)
        b1 = dh_causality_test(pd, :x, :y; p = 1, bootstrap = 200, seed = 11)
        b2 = dh_causality_test(pd, :x, :y; p = 1, bootstrap = 200, seed = 11)
        @test b1.bootstrap == 200 && b1.seed == 11
        @test isfinite(b1.bootstrap_pvalue)
        @test 0.0 <= b1.bootstrap_pvalue <= 1.0
        @test b1.bootstrap_pvalue == b2.bootstrap_pvalue          # reproducible
        # Strongly causal panel ⇒ small bootstrap p-value (reject H0).
        @test b1.bootstrap_pvalue < 0.10
        # Different seed generally changes the value (not asserted equal).
        b3 = dh_causality_test(pd, :x, :y; p = 1, bootstrap = 200, seed = 77)
        @test 0.0 <= b3.bootstrap_pvalue <= 1.0
    end

    @testset "Unbalanced panel (skips short units, averages own-T moments)" begin
        # Drop a few late observations from unit 1 so the panel is unbalanced but
        # still has enough T; the test should run and flag 0 skips (all long).
        df = make_dh_panel()
        df2 = df[.!((df.id .== 1) .& (df.time .> 25)), :]     # unit 1 has T=25
        pd2 = xtset(df2, :id, :time)
        r = dh_causality_test(pd2, :x, :y; p = 1)
        @test r.N == 8
        @test r.Wbar > 0
        @test 0.0 <= r.Ztilde_pvalue <= 1.0
    end

    @testset "String names, show, report, refs" begin
        df = make_dh_panel(); pd = xtset(df, :id, :time)
        rs = dh_causality_test(pd, "x", "y"; p = 1)     # string dispatch
        @test rs.cause == :x && rs.effect == :y
        io = IOBuffer(); show(io, rs)
        s = String(take!(io))
        @test occursin("Dumitrescu-Hurlin", s)
        @test occursin("Granger", s)
        # report() dispatches via AbstractUnitRootTest generic (prints to stdout).
        @test (report(rs); true)
        # StatsAPI accessors.
        @test MacroEconometricModels.StatsAPI.pvalue(rs) == rs.Ztilde_pvalue
        @test MacroEconometricModels.StatsAPI.nobs(rs) == rs.nobs
        @test MacroEconometricModels.StatsAPI.dof(rs) == rs.N
        # refs render.
        io3 = IOBuffer(); refs(io3, rs)
        @test occursin("Dumitrescu", String(take!(io3)))
    end
end
