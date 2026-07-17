# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# EV-11 (#419) — residual-based / parameter-stability cointegration tests:
# Engle–Granger, Phillips–Ouliaris (Ẑ_t, Ẑ_α), Hansen Lc, Park H(p,q).
#
# Oracles:
#   (1) Cross-impl: engle_granger_test matches statsmodels.tsa.stattools.coint (v0.14.4) and
#       _mackinnon_coint_pvalue matches statsmodels.tsa.adfvalues.mackinnonp to <1e-6.
#   (2) Analytic behaviour: reject on cointegrated / spurious DGPs, fail-to-reject on
#       independent random walks / stable cointegration, per each test's null.
#   (3) Published-value spot checks: PO Ẑ_α 5% (constant, N=2) ≈ −20.5 (Phillips–Ouliaris
#       1990 Table Ib); Hansen Lc 5% (constant, k=1) ≈ 0.57 (Hansen 1992 Table 1).
#   (4) Exact identities: Park p-value == χ²(q_add) upper tail of the statistic.

using Test, MacroEconometricModels, Random, Statistics, LinearAlgebra, Distributions
using DelimitedFiles
const MEM = MacroEconometricModels

# ---- DGP helpers (see provenance below) ----
# `coint_pair(12345, 200)` is the exact series fed to statsmodels `coint` for the
# oracle. MersenneTwister is not stable across Julia versions, so the exact series
# for the pinned (seed, T) oracle calls are committed as data/coint_pair_*.csv /
# data/indep_pair_*.csv (test/gen_ev_fixtures.jl); other (seed, T) stay RNG-driven.
function coint_pair(seed::Int, T::Int; beta::Float64=2.0, rho::Float64=0.5)
    f = joinpath(@__DIR__, "data", "coint_pair_$(seed)_$(T).csv")
    if beta == 2.0 && rho == 0.5 && isfile(f)
        d = readdlm(f, ',', Float64)
        return d[:, 1], d[:, 2]
    end
    rng = MersenneTwister(seed)
    x = cumsum(randn(rng, T))
    e = zeros(T)
    for t in 2:T
        e[t] = rho * e[t-1] + randn(rng)
    end
    y = 1.0 .+ beta .* x .+ e
    return y, x
end
function indep_pair(seed::Int, T::Int)
    f = joinpath(@__DIR__, "data", "indep_pair_$(seed)_$(T).csv")
    if isfile(f)
        d = readdlm(f, ',', Float64)
        return d[:, 1], d[:, 2]
    end
    rng = MersenneTwister(seed)
    x = cumsum(randn(rng, T))
    y = cumsum(randn(rng, T))
    return y, x
end

@testset "EV-11 Residual-Based Cointegration Tests" begin

    @testset "Engle-Granger: cross-impl vs statsmodels.coint" begin
        # statsmodels 0.14.4:
        #   d = np.loadtxt('coint.csv'); y,x = d[:,0], d[:,1]  (Julia coint_pair(12345,200))
        #   coint(y, x, trend='c', maxlag=1, autolag=None) -> (-7.184916, 0.0, ...)
        #   coint(y, x, trend='c', maxlag=0, autolag=None) -> (-7.831547, 0.0, ...)
        y, x = coint_pair(12345, 200)
        r1 = engle_granger_test(y, x; trend=:constant, lags=1)
        r0 = engle_granger_test(y, x; trend=:constant, lags=0)
        @test isapprox(r1.statistic, -7.184916; atol=1e-4)
        @test isapprox(r0.statistic, -7.831547; atol=1e-4)
        @test r1.pvalue < 1e-4
        @test r0.pvalue < 1e-4
        @test r1.N == 2 && r1.k == 1 && r1.lags == 1

        # Independent random walks: statsmodels coint(y2,x2,'c',maxlag=1,autolag=None)
        #   -> (-1.755262, 0.651187, ...); maxlag=0 -> (-1.642108, 0.702794, ...)
        yi, xi = indep_pair(999, 200)
        ri1 = engle_granger_test(yi, xi; trend=:constant, lags=1)
        ri0 = engle_granger_test(yi, xi; trend=:constant, lags=0)
        @test isapprox(ri1.statistic, -1.755262; atol=1e-4)
        @test isapprox(ri1.pvalue, 0.651187; atol=1e-4)
        @test isapprox(ri0.statistic, -1.642108; atol=1e-4)
        @test isapprox(ri0.pvalue, 0.702794; atol=1e-4)
    end

    @testset "MacKinnon coint p-value surface vs mackinnonp" begin
        # statsmodels.tsa.adfvalues.mackinnonp(stat, regression, N):
        @test isapprox(MEM._mackinnon_coint_pvalue(-3.5, :constant, 2), 0.03239539; atol=1e-6)
        @test isapprox(MEM._mackinnon_coint_pvalue(-2.0, :constant, 2), 0.52857808; atol=1e-6)
        @test isapprox(MEM._mackinnon_coint_pvalue(-4.5, :constant, 2), 0.00122467; atol=1e-6)
        @test isapprox(MEM._mackinnon_coint_pvalue(-1.0, :constant, 2), 0.90284723; atol=1e-6)
        @test isapprox(MEM._mackinnon_coint_pvalue(-3.5, :trend,    3), 0.19877651; atol=1e-6)
        @test isapprox(MEM._mackinnon_coint_pvalue(-4.5, :trend,    3), 0.01683024; atol=1e-6)
        # N=1 reproduces the univariate ADF surface (constant case).
        @test isapprox(MEM._mackinnon_coint_pvalue(-2.86, :constant, 1),
                       MEM._mackinnon_pvalue(-2.86, :constant); atol=1e-8)
        # Monotone decreasing in the statistic; saturates to [0,1].
        @test MEM._mackinnon_coint_pvalue(-8.0, :constant, 3) < MEM._mackinnon_coint_pvalue(-3.0, :constant, 3)
        @test MEM._mackinnon_coint_pvalue(-40.0, :constant, 2) == 0.0
        @test MEM._mackinnon_coint_pvalue(5.0, :constant, 2) == 1.0
        # Clamp N>6 without error.
        @test 0.0 <= MEM._mackinnon_coint_pvalue(-4.0, :trend, 9) <= 1.0
        @test_throws ArgumentError MEM._mackinnon_coint_pvalue(-3.0, :bogus, 2)
    end

    @testset "Engle-Granger: analytic behaviour + interface" begin
        y, x = coint_pair(2024, 250)
        rc = engle_granger_test(y, x)                    # default :aic
        @test rc.pvalue < 0.05                           # cointegrated -> reject (high power)
        # Aggregate size: independent RWs reject well below the ~100% power on cointegration.
        n_rej = count(1:40) do s
            yy, xx = indep_pair(3000 + s, 200)
            engle_granger_test(yy, xx).pvalue < 0.10
        end
        @test n_rej <= 12                                # empirical size near nominal, not 100%
        # Matrix convenience == vector call.
        rm = engle_granger_test(hcat(y, x); lags=2)
        rv = engle_granger_test(y, x; lags=2)
        @test rm.statistic == rv.statistic
        # trend variants and integer input promotion run.
        @test engle_granger_test(y, x; trend=:none) isa EngleGrangerResult
        @test engle_granger_test(y, x; trend=:trend) isa EngleGrangerResult
        @test engle_granger_test(round.(Int, y), x) isa EngleGrangerResult
        @test nobs(rc) == 250
        @test MEM.StatsAPI.pvalue(rc) == rc.pvalue
        @test occursin("Engle-Granger", sprint(show, rc))
        @test occursin("Engle", sprint(io -> refs(io, rc)))
    end

    @testset "Phillips-Ouliaris: Ẑ_t / Ẑ_α behaviour" begin
        y, x = coint_pair(2024, 250)
        pc = phillips_ouliaris_test(y, x)
        # Cointegrated: both statistics large-negative, both reject.
        @test pc.statistic < -3.0
        @test pc.z_alpha < -20.0
        @test pc.pvalue < 0.05
        @test pc.z_alpha_pvalue < 0.05
        # Aggregate size: independent RWs reject well below cointegrated power on both stats.
        n_zt = 0; n_za = 0
        for s in 1:40
            yy, xx = indep_pair(4000 + s, 200)
            p = phillips_ouliaris_test(yy, xx)
            p.pvalue < 0.10 && (n_zt += 1)
            p.z_alpha_pvalue < 0.10 && (n_za += 1)
        end
        @test n_zt <= 12
        @test n_za <= 12
        # Ẑ_t p-value shares the MacKinnon cointegration surface.
        @test isapprox(pc.pvalue, MEM._mackinnon_coint_pvalue(pc.statistic, :constant, 2); atol=1e-10)
        @test pc.N == 2 && pc.k == 1
        @test occursin("Phillips-Ouliaris", sprint(show, pc))

        # Published spot check: PO_ZA_CV 5% (constant, N=2) ≈ Phillips–Ouliaris (1990) −20.5.
        @test isapprox(MEM.PO_ZA_CV[:constant][2][2], -20.5; atol=2.0)
        # _po_za_pvalue is monotone decreasing and bracketed.
        @test MEM._po_za_pvalue(-50.0, :constant, 2) <= 0.01
        @test MEM._po_za_pvalue(-5.0,  :constant, 2) >  0.10
        @test 0.0 <= MEM._po_za_pvalue(-25.0, :constant, 2) <= 1.0
    end

    @testset "Hansen Lc: stable cointegration vs break" begin
        # Stable cointegration (fixed β): Lc should NOT reject.
        y, x = coint_pair(4321, 300; beta=1.5)
        m_stable = estimate_cointreg(y, x; method=:fmols, trend=:const)
        h_stable = hansen_instability_test(m_stable)
        @test h_stable.statistic >= 0.0
        @test h_stable.pvalue > 0.05                     # do not reject stability

        # Structural break in the cointegrating slope halfway: Lc SHOULD reject.
        rng = MersenneTwister(555); T = 300
        xb = cumsum(randn(rng, T))
        e = 0.3 .* randn(rng, T)
        beta_t = vcat(fill(1.0, T ÷ 2), fill(3.0, T - T ÷ 2))
        yb = beta_t .* xb .+ e
        m_break = estimate_cointreg(yb, xb; method=:fmols, trend=:const)
        h_break = hansen_instability_test(m_break)
        @test h_break.statistic > h_stable.statistic
        @test h_break.pvalue < 0.05                      # reject stability

        @test nobs(h_stable) == 300
        @test occursin("Hansen", sprint(show, h_stable))
        # Published spot check: Lc 5% (constant, k=1) ≈ Hansen (1992) 0.57.
        @test isapprox(MEM.HANSEN_LC_CV[:constant][1][2], 0.57; atol=0.1)
    end

    @testset "Park H(p,q): cointegrated vs spurious" begin
        # Genuine cointegration: added superfluous trends are insignificant -> fail to reject.
        y, x = coint_pair(2024, 250)
        mc = estimate_cointreg(y, x; method=:fmols, trend=:const)
        pk = park_added_test(mc; q_add=2)
        @test pk.pvalue > 0.05
        @test pk.q_add == 2

        # Spurious regression: independent I(1) with divergent trend behaviour -> reject.
        yi, xi = indep_pair(31, 250)
        ms = estimate_cointreg(yi, xi; method=:fmols, trend=:const)
        pks = park_added_test(ms; q_add=2)
        @test pks.statistic > pk.statistic

        # Exact identity: p-value is the χ²(q_add) upper tail of the statistic.
        @test isapprox(pk.pvalue, ccdf(Chisq(pk.q_add), pk.statistic); atol=1e-12)
        @test isapprox(pks.pvalue, ccdf(Chisq(pks.q_add), pks.statistic); atol=1e-12)
        # Aggregate spurious-rejection over many seeds (Park H is asymptotic, so allow slack).
        n_reject = 0
        for s in 1:40
            yy, xx = indep_pair(1000 + s, 200)
            mm = estimate_cointreg(yy, xx; method=:fmols, trend=:const)
            park_added_test(mm; q_add=2).pvalue < 0.10 && (n_reject += 1)
        end
        @test n_reject >= 8                              # spurious trends detected far above size

        @test occursin("Park", sprint(show, pk))
        @test MEM.StatsAPI.pvalue(pk) == pk.pvalue
    end

    @testset "Input validation" begin
        y, x = coint_pair(1, 100)
        @test_throws DimensionMismatch engle_granger_test(y, x[1:50])
        @test_throws ArgumentError phillips_ouliaris_test(reshape(y, :, 1))  # <2 cols
        short = randn(10)
        @test_throws ArgumentError engle_granger_test(short, randn(10))       # too few obs
        mc = estimate_cointreg(y, x; method=:fmols, trend=:const)
        @test_throws ArgumentError park_added_test(mc; q_add=0)
    end
end
