# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-20 (#428): First-generation panel unit-root tests — LLC / IPS / Breitung /
# Fisher / Hadri.
#
# Oracle discipline (no invented numerics; see the EV-20 spec's oracle_strategy):
#   (1) Published-value moment-table spot asserts. The finite-sample moment/
#       adjustment constants baked into the implementation are transcribed from
#       the plm reference implementation (`plm::purtest`, which reproduces IPS
#       2003 Table 3 and LLC 2002 Table 2) and the Stata `xtunitroot` Methods
#       and Formulas; the Hadri (ξ,ζ²) are exact rationals from Hadri (2000).
#       We spot-assert these against the published papers.
#   (2) Analytic/degenerate properties. Fisher with N=1 reduces EXACTLY to the
#       single-series adf_test/pp_test p-value (via the same MacKinnon path);
#       a stationary panel rejects and a random-walk panel fails to reject the
#       unit-root null (LLC/IPS/Breitung/Fisher left-tailed), while Hadri — whose
#       null is STATIONARITY — flips: it rejects a random walk and fails to
#       reject a stationary panel (right-tailed).
#   (3) Null-calibration Monte Carlo. Under the relevant null the standardized
#       statistics are ~N(0,1); we check the empirical mean/sd are in range.
#
# NOTE: a Stata cross-implementation reference (hard-coded statistics on a fixed
# seed) is not included because the local R toolchain is broken (miniconda clang
# cannot compile lmtest/plm) and Stata is unavailable in this environment; the
# formulas themselves come verbatim from the Stata xtunitroot manual, and the
# constants from the plm reference tables, so provenance is preserved.

using Test
using MacroEconometricModels
using StatsAPI
using Random
using Statistics

const _M = MacroEconometricModels

# Balanced panels: random walk (unit-root null true) vs stationary iid.
_rw_panel(rng, T=60, N=20) = cumsum(randn(rng, T, N); dims=1)
_stat_panel(rng, T=60, N=20) = randn(rng, T, N)

@testset "First-Generation Panel Unit Root Tests (EV-20)" begin

    @testset "Published moment-table spot asserts" begin
        # --- IPS (2003) Table 3 (via plm adj.ips.wtbar), intercept case ---
        @test _M._ips_moment(:mean, 10, 0, :constant) == -1.504
        @test _M._ips_moment(:var, 10, 0, :constant) == 1.069
        @test _M._ips_moment(:mean, 100, 0, :constant) == -1.532
        @test _M._ips_moment(:var, 100, 0, :constant) == 0.735
        # A tabulated lag>0 entry (T=25, lag=3, intercept).
        @test _M._ips_moment(:mean, 25, 3, :constant) == -1.433
        @test _M._ips_moment(:var, 25, 3, :constant) == 0.952
        # --- IPS Table 3, trend case ---
        @test _M._ips_moment(:mean, 10, 0, :trend) == -2.166
        @test _M._ips_moment(:var, 10, 0, :trend) == 1.132
        # Linear interpolation in T between grid points 10 and 15 (constant, lag0).
        @test _M._ips_moment(:mean, 12.5, 0, :constant) ≈ -1.509 atol = 1e-12
        # Clamp beyond the grid.
        @test _M._ips_moment(:mean, 5, 0, :constant) == -1.504
        @test _M._ips_moment(:mean, 500, 0, :constant) == -1.532
        # NaN entries are skipped: lag 8 has no T=10 value, so T=10 clamps to the
        # first non-NaN (T=25) entry.
        @test _M._ips_moment(:mean, 10, 8, :constant) == -1.212

        # --- LLC (2002) Table 2 (via plm adj.levinlin) ---
        @test _M._llc_adjustment(25, :constant) == (-0.554, 0.919)
        @test _M._llc_adjustment(500, :constant) == (-0.500, 0.707)
        @test _M._llc_adjustment(25, :trend) == (-0.703, 1.003)
        @test _M._llc_adjustment(25, :none) == (0.004, 1.049)
        # Interpolation between T̃=25 and 30 (constant μ*: -0.554,-0.546).
        @test _M._llc_adjustment(27.5, :constant)[1] ≈ -0.550 atol = 1e-12

        # --- Hadri (2000) exact standardization constants (ξ, ζ²) ---
        # constant: (1/6, 1/45); trend: (1/15, 11/6300). Assert via a fitted result.
        rng = MersenneTwister(11)
        hc = hadri_test(_rw_panel(rng), deterministic = :constant)
        @test hc.xi == 1 / 6
        @test hc.zeta ≈ sqrt(1 / 45) atol = 1e-14
        ht = hadri_test(_rw_panel(rng), deterministic = :trend)
        @test ht.xi == 1 / 15
        @test ht.zeta ≈ sqrt(11 / 6300) atol = 1e-14
    end

    @testset "Fisher N=1 ≡ single-series adf/pp p-value" begin
        rng = MersenneTwister(202)
        y = cumsum(randn(rng, 80))
        X1 = reshape(y, :, 1)
        # Maddala-Wu P = -2 ln(p_1); its χ²(2) upper tail equals p_1 exactly.
        fa = fisher_panel_test(X1; base = :adf, combine = :mw)
        @test fa.n_units == 1
        @test fa.pvalue ≈ adf_test(y).pvalue rtol = 1e-10
        @test fa.individual_pvalues[1] ≈ adf_test(y).pvalue rtol = 1e-12
        # Same identity with the Phillips-Perron base.
        fp = fisher_panel_test(X1; base = :pp, combine = :mw)
        @test fp.pvalue ≈ pp_test(y).pvalue rtol = 1e-10
        # Different determinstic term flows through to the per-unit ADF.
        ft = fisher_panel_test(X1; base = :adf, combine = :mw, deterministic = :trend)
        @test ft.pvalue ≈ adf_test(y; regression = :trend).pvalue rtol = 1e-10
    end

    @testset "Degenerate: stationary rejects, random walk does not (correct tails)" begin
        rng = MersenneTwister(7)
        Xrw = _rw_panel(rng)
        Xst = _stat_panel(rng)

        # Unit-root-null tests: left-tailed. Stationary ⇒ very negative ⇒ reject.
        for f in (llc_test, ips_test, breitung_panel_test)
            rst = f(Xst)
            rrw = f(Xrw)
            @test rst.statistic < rrw.statistic          # stationary more negative
            @test rst.pvalue < 0.05                       # reject unit root
            @test rrw.pvalue > 0.10                       # fail to reject
        end
        # Fisher (Maddala-Wu P): upper-tailed χ². Stationary ⇒ large P ⇒ reject.
        fst = fisher_panel_test(Xst; combine = :mw)
        frw = fisher_panel_test(Xrw; combine = :mw)
        @test fst.P > frw.P
        @test fst.pvalue < 0.05
        @test frw.pvalue > 0.10
        # Choi Z is left-tailed and must agree on direction.
        @test fst.Z < frw.Z
        @test fisher_panel_test(Xst; combine = :choi).pvalue < 0.05

        # Hadri: STATIONARITY null, RIGHT-tailed. Random walk ⇒ reject; stationary ⇒ not.
        # Hadri is well-known for mild finite-sample over-rejection (~7% at 5%), so a
        # single stationary draw is a fragile "fail to reject" oracle; check the
        # rejection RATE over several draws instead (robust + honest).
        hrw = hadri_test(Xrw)
        @test hrw.pvalue < 0.05                            # reject stationarity for a random walk
        @test hrw.statistic > 5.0                          # RW pushes Z strongly positive
        n_rej_stat = count(1:12) do _
            hadri_test(_stat_panel(rng)).pvalue < 0.05
        end
        @test n_rej_stat <= 3                              # mostly fails to reject stationarity
    end

    @testset "Null calibration ~ N(0,1)" begin
        # Under each test's own null the standardized statistic is ~N(0,1). Loose
        # bounds (Monte Carlo error) with a fixed seed for determinism.
        reps = 120
        rng = MersenneTwister(99)
        for (f, gen, det) in (
            (llc_test, _rw_panel, :constant),
            (ips_test, _rw_panel, :constant),
            (breitung_panel_test, _rw_panel, :constant),
        )
            s = Float64[]
            for _ in 1:reps
                push!(s, f(gen(rng, 60, 15); deterministic = det).statistic)
            end
            @test abs(mean(s)) < 0.8          # LLC has a known small finite-T bias
            @test 0.7 < std(s) < 1.4
        end
        # Hadri null = stationarity ⇒ generate stationary panels.
        sh = Float64[]
        for _ in 1:reps
            sh = push!(sh, hadri_test(_stat_panel(rng, 80, 15)).statistic)
        end
        @test abs(mean(sh)) < 0.6
        @test 0.6 < std(sh) < 1.4
    end

    @testset "Trend specification runs and rejects stationary" begin
        rng = MersenneTwister(55)
        Xst = _stat_panel(rng, 70, 18)
        @test llc_test(Xst; deterministic = :trend).pvalue < 0.05
        @test ips_test(Xst; deterministic = :trend).pvalue < 0.05
        @test breitung_panel_test(Xst; deterministic = :trend).pvalue < 0.05
        @test hadri_test(_rw_panel(rng, 70, 18); deterministic = :trend).pvalue < 0.05
        # :none deterministic for LLC/Breitung/Fisher.
        @test llc_test(Xst; deterministic = :none).pvalue < 0.05
        @test breitung_panel_test(Xst; deterministic = :none).pvalue < 0.05
        @test fisher_panel_test(Xst; deterministic = :none).pvalue < 0.05
    end

    @testset "Lag augmentation and prewhitening" begin
        rng = MersenneTwister(321)
        # AR(1)-in-differences (serially correlated) stationary panel.
        T, N = 80, 15
        X = zeros(T, N)
        for i in 1:N
            e = randn(rng, T)
            for t in 2:T
                X[t, i] = 0.3 * X[t-1, i] + e[t] + 0.4 * e[t-1]
            end
        end
        @test ips_test(X; lags = 2).pvalue < 0.05
        @test llc_test(X; lags = 2).pvalue < 0.05
        @test breitung_panel_test(X; lags = 2).pvalue < 0.05
        # :auto per-unit lag selection populates a per-unit lag vector.
        ri = ips_test(X; lags = :auto)
        @test length(ri.lags) == N
        @test all(ri.lags .>= 0)
        rl = llc_test(X; lags = :auto)
        @test length(rl.lags) == N
    end

    @testset "cs_demean heuristic runs" begin
        rng = MersenneTwister(88)
        # Panel with a strong common factor (cross-sectional dependence).
        T, N = 60, 20
        f = cumsum(randn(rng, T))
        X = f .+ randn(rng, T, N)
        r0 = ips_test(X; cs_demean = false)
        r1 = ips_test(X; cs_demean = true)
        @test r0.statistic != r1.statistic         # demeaning changes the statistic
        @test isfinite(r1.statistic)
        @test isfinite(hadri_test(X; cs_demean = true).statistic)
    end

    @testset "PanelData dispatch" begin
        rng = MersenneTwister(404)
        # Build a long-format balanced PanelData via xtset.
        T, N = 50, 12
        Xst = _stat_panel(rng, T, N)
        ids = repeat(1:N, inner = T)
        times = repeat(1:T, outer = N)
        yy = vec(Xst)                              # column-major: unit-major stacking
        df = MacroEconometricModels.DataFrames.DataFrame(id = ids, time = times, y = yy)
        pd = xtset(df, :id, :time)
        rm = llc_test(pd)
        rd = llc_test(Xst)
        @test rm.statistic ≈ rd.statistic rtol = 1e-8
        @test ips_test(pd).n_units == N
        @test hadri_test(pd) isa HadriResult
    end

    @testset "Result types, StatsAPI, show, refs" begin
        rng = MersenneTwister(1234)
        X = _stat_panel(rng)
        rl = llc_test(X); ri = ips_test(X); rb = breitung_panel_test(X)
        rf = fisher_panel_test(X); rh = hadri_test(X)
        @test rl isa LLCResult{Float64} && rl isa MacroEconometricModels.AbstractUnitRootTest
        @test ri isa IPSResult{Float64}
        @test rb isa BreitungPanelResult{Float64}
        @test rf isa FisherPanelResult{Float64}
        @test rh isa HadriResult{Float64}
        for r in (rl, ri, rb, rf, rh)
            @test StatsAPI.nobs(r) == 60
            @test 0 <= StatsAPI.pvalue(r) <= 1
            @test StatsAPI.dof(r) > 0
            s = sprint(show, r)                    # renders without error
            @test occursin("Test", s)
            rs = sprint(refs, r)                   # bibliography renders
            @test length(rs) > 0
        end
        @test StatsAPI.dof(rf) == 2 * rf.n_units    # Fisher χ²(2N)
        # Fisher stores all four combination statistics.
        @test rf.P > 0
        @test isfinite(rf.Z) && isfinite(rf.Lstar) && isfinite(rf.Pm)
        @test length(rf.individual_pvalues) == 20
        # Float32 input converts and runs.
        @test llc_test(Float32.(X)) isa LLCResult{Float32}
    end

    @testset "panel_unit_root_summary runs all eight tests" begin
        rng = MersenneTwister(2468)
        X = _stat_panel(rng, 60, 20)
        s = panel_unit_root_summary(X; r = 1)
        @test s isa PanelUnitRootSummary
        # All five first-generation tests populated.
        @test s.llc isa LLCResult
        @test s.ips isa IPSResult
        @test s.breitung isa BreitungPanelResult
        @test s.fisher isa FisherPanelResult
        @test s.hadri isa HadriResult
        # Second-generation still present.
        @test s.panic isa MacroEconometricModels.PANICResult
        @test s.cips isa MacroEconometricModels.PesaranCIPSResult
        @test s.moon_perron isa MacroEconometricModels.MoonPerronResult
        out = sprint(show, s)
        @test occursin("Panel Unit Root Test Battery", out)
        @test occursin("Levin-Lin-Chu", out)
        @test occursin("Hadri", out)
    end

    @testset "Input validation" begin
        rng = MersenneTwister(1)
        @test_throws ArgumentError llc_test(randn(rng, 60, 20); deterministic = :bogus)
        @test_throws ArgumentError ips_test(randn(rng, 60, 20); deterministic = :none)  # IPS: no :none moments
        @test_throws ArgumentError hadri_test(randn(rng, 60, 20); deterministic = :none)
        @test_throws ArgumentError ips_test(randn(rng, 10, 20))                          # T too small
        @test_throws ArgumentError llc_test(randn(rng, 60, 1))                           # N too small
        @test_throws ArgumentError fisher_panel_test(randn(rng, 60, 20); base = :bogus)
        @test_throws ArgumentError fisher_panel_test(randn(rng, 60, 20); combine = :bogus)
    end
end
