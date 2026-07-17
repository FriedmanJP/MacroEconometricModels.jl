# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# Tests for the smooth-transition autoregression (STAR / LSTR1 / LSTR2 / ESTR)
# module (EV-06 / #414): NLS estimation, the Luukkonen–Saikkonen–Teräsvirta
# (1988) LM3 linearity test, and Teräsvirta's (1994) selection sequence.
#
# Oracle discipline
# -----------------
# The issue names R `tsDyn::lstar(y, m=p, thDelay=d)` as the cross-implementation
# oracle. tsDyn is NOT installed in this environment (verified: R's
# requireNamespace("tsDyn") returns FALSE), so no reference numerics are
# fabricated. Primary validation is analytic-property based, exactly as the spec's
# oracle_strategy prescribes:
#   (1) RECOVERY — on a fixed-seed LSTR1 DGP the estimator recovers (φ₁, φ₂, c)
#       within a stated tolerance and the fitted transition weights track the true
#       G (correlation > 0.99). Because γ is reported on the dimension-free scale
#       (γ/σ̂_s), the *unscaled* slope γ̂/σ̂_s recovers the DGP slope.
#   (2) LARGE-γ NESTING — a near-step (very large γ) LSTR1 DGP collapses G to the
#       indicator, so the STAR regime split (G > 0.5) matches the EV-05
#       `estimate_setar` split, and γ̂ blows up while ĉ ≈ the SETAR threshold.
#   (3) LINEARITY-TEST SIZE/POWER — on a linear-AR null the LM3 χ²/F rejection
#       rate ≈ nominal within a seeded Monte-Carlo band; on the LSTR1 DGP it
#       rejects (power).
# The exact tsDyn call is recorded below for a future in-env oracle:
#   R> library(tsDyn); set.seed(...); fit <- lstar(y, m = 1, thDelay = 1)
#   R> coef(fit)   # const.L, phiL.1 (low regime); const.H, phiH.1 (high regime); gamma; th

using Test
using MacroEconometricModels
using Random
using Statistics
using LinearAlgebra
using DelimitedFiles

# -----------------------------------------------------------------------------
# Fixed-seed DGPs
# -----------------------------------------------------------------------------

"""Simulate an LSTR1 STAR(1): logistic transition on y_{t-1} with location `c`."""
function _sim_lstar(; n::Int, gamma::Float64=8.0, c::Float64=0.0,
                    phi1=(0.5, 0.6), phi2=(-0.4, -0.3), sigma::Float64=0.3,
                    seed::Int=20240716)
    # The n=1200, gamma=15 parameter-recovery DGP is pinned to a committed fixture
    # (test/gen_ev_fixtures.jl) because MersenneTwister is not stable across Julia
    # versions and the recovery tolerances are realization-specific. Other
    # (n, gamma, seed) configurations stay RNG-driven.
    f = joinpath(@__DIR__, "data", "star_lstar_$(n).csv")
    if isfile(f) && gamma == 15.0 && c == 0.0 && phi1 == (0.5, 0.6) &&
       phi2 == (-0.4, -0.3) && sigma == 0.3 && seed == 20240716
        return vec(readdlm(f, ',', Float64))
    end
    rng = MersenneTwister(seed)
    y = zeros(n)
    for t in 2:n
        s = y[t-1]
        G = 1 / (1 + exp(-gamma * (s - c)))
        y[t] = (phi1[1] + phi1[2] * y[t-1]) * (1 - G) +
               (phi2[1] + phi2[2] * y[t-1]) * G + sigma * randn(rng)
    end
    return y
end

"""Simulate a near-step SETAR(2;1,1) as the large-γ limit of LSTR1."""
function _sim_sharp_setar(; n::Int, gamma::Float64=0.2, seed::Int=7)
    rng = MersenneTwister(seed)
    y = zeros(n)
    for t in 2:n
        y[t] = y[t-1] <= gamma ? (0.5 + 0.5 * y[t-1] + 0.3 * randn(rng)) :
                                 (-0.4 - 0.3 * y[t-1] + 0.3 * randn(rng))
    end
    return y
end

"""Simulate a linear AR(1) (the linearity-test null)."""
function _sim_ar1_star(; n::Int, phi::Float64=0.5, sigma::Float64=1.0, seed::Int=1)
    rng = MersenneTwister(seed)
    y = zeros(n)
    for t in 2:n
        y[t] = phi * y[t-1] + sigma * randn(rng)
    end
    return y
end

@testset "STAR (EV-06)" begin

    # -------------------------------------------------------------------------
    @testset "Transition functions G(s;γ,c) shape properties" begin
        s = collect(range(-3.0, 3.0, length=61))
        σs = std(s)
        # LSTR1: monotone increasing, in (0,1), G(c)=0.5.
        G1 = MacroEconometricModels._star_transition(s, 5.0, [0.0], σs, :lstr1)
        @test all(0 .< G1 .< 1)
        @test issorted(G1)
        @test G1[31] ≈ 0.5 atol = 1e-8          # s[31] = 0.0 = c
        # ESTR: in [0,1), symmetric about c, minimum (0) at s=c.
        Ge = MacroEconometricModels._star_transition(s, 5.0, [0.0], σs, :estr)
        @test all(0 .<= Ge .< 1)
        @test Ge[31] ≈ 0.0 atol = 1e-10
        @test Ge[31] == minimum(Ge)
        @test Ge ≈ reverse(Ge) atol = 1e-8      # symmetry about c = 0
        # LSTR2: in (0,1), U-shaped weight (high in the tails, low between c₁,c₂).
        G2 = MacroEconometricModels._star_transition(s, 5.0, [-1.0, 1.0], σs, :lstr2)
        @test all(0 .< G2 .< 1)
        @test G2[1] > G2[31] && G2[end] > G2[31]
    end

    # -------------------------------------------------------------------------
    @testset "Parameter recovery on a fixed-seed LSTR1 DGP" begin
        y = _sim_lstar(n=1200, gamma=15.0, c=0.0, seed=20240716)
        m = estimate_star(y, 1; d=1, type=:lstr1)

        @test m isa STARModel{Float64}
        @test m.trans_type == :lstr1
        @test m.converged
        # φ recovery (true φ₁=[0.5,0.6], φ₂=[-0.4,-0.3]).
        @test isapprox(m.phi1, [0.5, 0.6]; atol=0.1)
        @test isapprox(m.phi2, [-0.4, -0.3]; atol=0.15)
        # Location recovery (true c = 0).
        @test isapprox(m.c[1], 0.0; atol=0.1)
        # Dimension-free slope: γ̂/σ̂_s recovers the DGP slope 15 (loose — γ is
        # notoriously imprecise; require the same order of magnitude and γ̂>0).
        @test m.gamma > 0
        @test 5.0 < m.gamma / m.sigma_s < 40.0
        # Fitted transition tracks the true G.
        Gtrue = [1 / (1 + exp(-15.0 * yl)) for yl in y[1:end-1]]
        @test cor(m.G, Gtrue) > 0.99
        # SEs are finite and positive.
        @test all(isfinite, m.se_phi1) && all(m.se_phi1 .> 0)
        @test all(isfinite, m.se_phi2) && all(m.se_phi2 .> 0)
        @test isfinite(m.se_gamma) && m.se_gamma > 0
        @test all(isfinite, m.se_c)
    end

    # -------------------------------------------------------------------------
    @testset "Large-γ nesting: STAR → SETAR indicator" begin
        y = _sim_sharp_setar(n=600, gamma=0.2, seed=7)
        m = estimate_star(y, 1; d=1, type=:lstr1)
        st = estimate_setar(y, 1, 1; linearity=false)

        # Both use the effective sample t = 2:n (p=1, d=1), s_t = y_{t-1}.
        star_reg = m.G .> 0.5
        setar_reg = y[1:end-1] .> st.gamma
        agreement = mean(star_reg .== setar_reg[1:length(star_reg)])
        @test agreement > 0.95
        # γ̂ collapses toward a step; ĉ ≈ the SETAR threshold.
        @test m.gamma > 20.0
        @test isapprox(m.c[1], st.gamma; atol=0.1)
    end

    # -------------------------------------------------------------------------
    @testset "star_linearity_test — size and power" begin
        # Power: rejects on the LSTR1 DGP.
        y = _sim_lstar(n=400, gamma=8.0, seed=101)
        lt = star_linearity_test(y, 1; d=1)
        @test lt.df == 3                        # 3p = 3 for p=1
        @test lt.stat > 0 && lt.fstat > 0
        @test 0 <= lt.pvalue <= 1 && 0 <= lt.fpvalue <= 1
        @test lt.pvalue < 0.01                  # strong rejection
        @test lt.fpvalue < 0.01

        # Size: Monte-Carlo rejection rate on the linear-AR null ≈ nominal 0.05.
        R = 400
        rej_chi = 0
        rej_f = 0
        for r in 1:R
            yy = _sim_ar1_star(n=300, phi=0.5, seed=1000 + r)
            ltr = star_linearity_test(yy, 1; d=1)
            ltr.pvalue < 0.05 && (rej_chi += 1)
            ltr.fpvalue < 0.05 && (rej_f += 1)
        end
        # Seeded MC band around 0.05 (χ² slightly liberal in small samples; F better).
        @test 0.02 <= rej_chi / R <= 0.10
        @test 0.02 <= rej_f / R <= 0.10

        # Power over a seeded MC on the LSTR1 DGP is essentially 1.
        pw = 0
        for r in 1:150
            yy = _sim_lstar(n=250, gamma=8.0, seed=2000 + r)
            star_linearity_test(yy, 1; d=1).pvalue < 0.05 && (pw += 1)
        end
        @test pw / 150 > 0.9
    end

    # -------------------------------------------------------------------------
    @testset "Teräsvirta (1994) selection sequence (type=:auto)" begin
        y = _sim_lstar(n=600, gamma=8.0, seed=303)
        m = estimate_star(y, 1; type=:auto)
        @test m.trans_type in (:lstr1, :estr, :lstr2)
        @test m.sel_pvalues !== nothing
        @test length(m.sel_pvalues) == 3
        @test all(p -> 0 <= p <= 1, m.sel_pvalues)
        # An asymmetric logistic DGP selects LSTR1.
        @test m.trans_type == :lstr1
        # A fixed type does not run selection.
        mfix = estimate_star(y, 1; type=:estr)
        @test mfix.trans_type == :estr
        @test mfix.sel_pvalues === nothing
    end

    # -------------------------------------------------------------------------
    @testset "ESTR estimation runs end-to-end" begin
        # Symmetric (exponential-transition) DGP: extreme |y_{t-1}| in one regime.
        rng = MersenneTwister(55)
        n = 500
        y = zeros(n)
        for t in 2:n
            G = 1 - exp(-3.0 * (y[t-1])^2)
            y[t] = (0.6 * y[t-1]) * (1 - G) + (-0.5 * y[t-1]) * G + 0.3 * randn(rng)
        end
        m = estimate_star(y, 1; type=:estr)
        @test m.trans_type == :estr
        @test length(m.c) == 1
        @test all(0 .<= m.G .< 1)
        @test isfinite(m.ssr) && m.ssr > 0
    end

    # -------------------------------------------------------------------------
    @testset "Display, refs, StatsAPI, and plotting" begin
        y = _sim_lstar(n=400, gamma=8.0, seed=77)
        m = estimate_star(y, 1; type=:lstr1)

        buf = IOBuffer()
        report(buf, m)
        str = String(take!(buf))
        @test occursin("STAR", str)
        @test occursin("Transition", str)
        @test occursin("LM3", str)

        # refs render both STAR references.
        rbuf = IOBuffer()
        refs(rbuf, m)
        rstr = String(take!(rbuf))
        @test occursin("Teräsvirta", rstr)
        @test occursin("Luukkonen", rstr)

        # StatsAPI.
        @test MacroEconometricModels.StatsAPI.nobs(m) == m.n
        @test length(MacroEconometricModels.StatsAPI.residuals(m)) == m.n
        @test length(MacroEconometricModels.StatsAPI.coef(m)) == 2 * m.k + 1 + length(m.c)
        @test MacroEconometricModels.StatsAPI.dof(m) == 2 * m.k + 1 + length(m.c)

        # plot_result dispatch: both views build a PlotOutput.
        p1 = plot_result(m; view=:transition)
        p2 = plot_result(m; view=:weights)
        @test p1 isa MacroEconometricModels.PlotOutput
        @test p2 isa MacroEconometricModels.PlotOutput
        @test_throws ArgumentError plot_result(m; view=:nope)
    end

    # -------------------------------------------------------------------------
    @testset "Forecast" begin
        y = _sim_lstar(n=400, gamma=8.0, seed=88)
        m = estimate_star(y, 1; type=:lstr1)
        f = forecast(m, 8; reps=500, level=0.90)
        @test f isa STARForecast{Float64}
        @test length(f.forecast) == 8
        @test length(f.ci_lower) == 8 && length(f.ci_upper) == 8
        @test all(f.ci_lower .<= f.forecast .<= f.ci_upper)
        @test all(f.se .>= 0)
        @test f.conf_level == 0.90
        # Display.
        buf = IOBuffer(); report(buf, f)
        @test occursin("Forecast", String(take!(buf)))
    end

    # -------------------------------------------------------------------------
    @testset "External transition variable" begin
        rng = MersenneTwister(9)
        n = 400
        sx = randn(rng, n)               # exogenous transition variable
        y = zeros(n)
        for t in 2:n
            G = 1 / (1 + exp(-6.0 * sx[t]))
            y[t] = (0.5 + 0.4 * y[t-1]) * (1 - G) +
                   (-0.3 - 0.2 * y[t-1]) * G + 0.3 * randn(rng)
        end
        m = estimate_star(y, 1; s=sx, type=:lstr1)
        @test m.sname == "s"
        @test m.n == n - 1               # m0 = p = 1
        @test length(m.s) == m.n
        # A self-exciting forecast is undefined for an external s.
        @test_throws ArgumentError forecast(m, 4)
    end

    # -------------------------------------------------------------------------
    @testset "Argument validation" begin
        y = _sim_lstar(n=200, seed=5)
        @test_throws ArgumentError estimate_star(y, 0)               # p < 1
        @test_throws ArgumentError estimate_star(y, 1; d=0)          # d < 1
        @test_throws ArgumentError estimate_star(y, 1; type=:bogus)  # bad type
        @test_throws DimensionMismatch estimate_star(y, 1; s=randn(10))  # s length
        @test_throws ArgumentError estimate_star(fill(1.0, 100), 1)  # zero-variance s
        @test_throws ArgumentError star_linearity_test(y, 0)
    end
end
