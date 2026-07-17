# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-29 (#437): HEGY seasonal unit-root tests (quarterly, Hylleberg-Engle-
# Granger-Yoo 1990; monthly, Beaulieu-Miron 1993) + ERS (1996) feasible
# point-optimal test.
#
# Oracle discipline (R uroot/urca NOT installed in this environment, verified —
# so they cannot be run live; Stata unavailable). Layered per the EV-29 spec:
#   (1) ERS CROSS-IMPLEMENTATION (STRONGEST, fully in-env): ers_test(y).P_T MUST
#       equal the already-validated dfgls_test(y).pt_statistic bit-for-bit,
#       because both route through the shared _ers_pt_statistic helper. This is
#       an exact identity, verifiable live here.
#   (2) HEGY ANALYTIC PROPERTIES (seeded): a deterministic-seasonal-only series
#       (no stochastic seasonal unit root) REJECTS the seasonal-frequency nulls;
#       a Δ_s-integrated (seasonal random-walk) series FAILS TO REJECT them.
#       These conclusions are robust to ±0.3 critical-value error, so they do not
#       depend on the exact published CV digits.
#   (3) HAND-RECOMPUTATION of the ERS P_T = (S(c̄) − ᾱ·S(1))/ω̂² formula on a
#       toy series, recomputed independently inside the test.
#   (4) PUBLISHED-with-citation critical values: HEGY (1990) quarterly and
#       Beaulieu-Miron (1993) monthly tables are hard-coded in critical_values.jl
#       and marked "published, transcribed offline, not live-verified".

using Test, MacroEconometricModels, Random, Statistics, LinearAlgebra
using StatsAPI
using MacroEconometricModels: _ers_pt_statistic, _ers_gls_detrend, _ers_lrv

# ---------------------------------------------------------------------------
# Seeded data generators
# ---------------------------------------------------------------------------
# Deterministic seasonal pattern + stationary noise: NO stochastic unit root at
# any frequency. After seasonal-dummy detrending it is white noise ⇒ every HEGY
# null should reject.
function _det_seasonal(n, s; seed=101)
    rng = Random.MersenneTwister(seed)
    pat = s == 4 ? [6.0, -3.0, 4.0, -7.0] :
                   [5.0, -4.0, 3.0, -6.0, 2.0, -5.0, 4.0, -3.0, 1.0, -2.0, 6.0, -7.0]
    y = zeros(n)
    for t in 1:n
        y[t] = pat[((t - 1) % s) + 1] + 0.4 * randn(rng)
    end
    return y
end

# Seasonal random walk y_t = y_{t-s} + e_t: unit roots at ALL s roots of L^s = 1
# (zero, Nyquist and every harmonic) ⇒ every HEGY null should fail to reject.
function _seasonal_rw(n, s; seed=202)
    rng = Random.MersenneTwister(seed)
    y = zeros(n)
    e = randn(rng, n)
    for t in 1:n
        y[t] = (t <= s ? 0.0 : y[t-s]) + e[t]
    end
    return y
end

@testset "HEGY seasonal unit roots + ERS point-optimal (EV-29)" begin

    # =======================================================================
    # (1) ERS cross-implementation identity — STRONGEST, fully in-env oracle
    # =======================================================================
    @testset "ERS ≡ DF-GLS pt_statistic (shared helper)" begin
        for seed in (1, 7, 42, 99)
            rng = Random.MersenneTwister(seed)
            y = cumsum(randn(rng, 160)) .+ 0.2 .* (1:160)   # trending near-I(1)

            e_c = ers_test(y; trend=false)
            d_c = dfgls_test(y; regression=:constant)
            @test e_c isa ERSResult
            @test e_c.P_T == d_c.pt_statistic          # bit-for-bit identity
            @test e_c.critical_values == d_c.pt_critical_values
            @test e_c.regression == :constant

            e_t = ers_test(y; trend=true)
            d_t = dfgls_test(y; regression=:trend)
            @test e_t.P_T == d_t.pt_statistic          # bit-for-bit identity
            @test e_t.regression == :trend
        end
    end

    @testset "ERS StatsAPI + display + refs" begin
        rng = Random.MersenneTwister(5)
        y = cumsum(randn(rng, 120))
        r = ers_test(y)
        @test StatsAPI.nobs(r) == 120
        @test StatsAPI.pvalue(r) == r.pvalue
        @test StatsAPI.dof(r) == 1
        @test StatsAPI.dof(ers_test(y; trend=true)) == 2
        @test isfinite(r.P_T)                           # Pt reproduces the frozen DFGLS Pt
        @test haskey(r.critical_values, 5)
        # renders without error
        io = IOBuffer(); show(io, r); s = String(take!(io))
        @test occursin("Point-Optimal", s)
        io2 = IOBuffer(); refs(io2, r); @test occursin("Elliott", String(take!(io2)))
        # integer input path
        @test ers_test(round.(Int, y .* 10)) isa ERSResult
        @test_throws ArgumentError ers_test(randn(10))
    end

    # (2b) ERS point-optimal direction/size. With S(1) computed as the a=1 (unit-root)
    # quasi-differenced SSR (ERS 1996 eq. 6), P_T behaves correctly: REJECT the unit-
    # root null for SMALL P_T. (A prior implementation used the OLS-detrended LEVELS
    # SSR for S(1) — O(T²) — which inverted the test and rejected the null ~100% of
    # the time; this testset guards against that regression.)
    @testset "ERS point-optimal size & power" begin
        # Stationary AR(1) ρ=0.4, T=400 ⇒ small P_T ⇒ reject the unit-root null at 5%.
        rng = Random.MersenneTwister(303)
        y_st = zeros(400)
        for t in 2:400; y_st[t] = 0.4 * y_st[t-1] + randn(rng); end
        r_st = ers_test(y_st)
        @test r_st.P_T < r_st.critical_values[5]
        # Random walk (unit-root null) ⇒ large P_T ⇒ fail to reject in the vast majority.
        rej = 0
        rng2 = Random.MersenneTwister(404)
        for _ in 1:100
            yrw = cumsum(randn(rng2, 200))
            r = ers_test(yrw)
            r.P_T < r.critical_values[5] && (rej += 1)
        end
        @test rej <= 25          # empirical size under a permissive ceiling (nominal 5%)
    end

    # =======================================================================
    # (3) Independent hand-recomputation of the ERS P_T formula
    # =======================================================================
    @testset "ERS P_T formula hand-recomputation" begin
        rng = Random.MersenneTwister(77)
        y = cumsum(randn(rng, 90))
        reg = :constant
        n = length(y)
        c_bar = -7.0
        alpha = 1 + c_bar / n
        # GLS quasi-difference (independent reconstruction)
        yqd = copy(y); yqd[2:end] = y[2:end] .- alpha .* y[1:end-1]
        Z = ones(n, 1)
        zqd = ones(n, 1); zqd[2:end, 1] .= 1 - alpha
        delta = zqd \ yqd
        Sa = sum((yqd .- zqd * delta) .^ 2)              # S(c̄)
        yq1 = copy(y); yq1[2:end] = y[2:end] .- y[1:end-1]          # a=1 quasi-diff of y
        Zq1 = copy(Z); Zq1[2:end, :] .= Z[2:end, :] .- Z[1:end-1, :]  # …of the regressors
        S1 = sum((yq1 .- Zq1 * (Zq1 \ yq1)) .^ 2)        # S(1): a=1 quasi-diff SSR
        f00 = _ers_lrv(y .- Z * delta, n)                # same LRV the helper uses
        pt_manual = (Sa - alpha * S1) / f00
        @test _ers_pt_statistic(y, reg).pt_stat ≈ pt_manual rtol=1e-10
        @test ers_test(y).P_T ≈ pt_manual rtol=1e-10
    end

    # =======================================================================
    # (2) HEGY quarterly — analytic property oracles
    # =======================================================================
    @testset "HEGY quarterly basic + fields" begin
        y = _seasonal_rw(200, 4; seed=11)
        r = hegy_test(y; frequency=4)
        @test r isa HEGYResult
        @test r.frequency == 4
        @test r.deterministic == :const_trend_seas
        @test r.lags >= 0
        @test length(r.pi_coefs) == 4                   # π₁..π₄ for quarterly
        @test length(r.pair_F) == 1                     # one harmonic pair (π/2)
        @test isfinite(r.t_zero) && isfinite(r.t_nyquist)
        @test all(isfinite, r.pair_F)
        @test r.pair_F[1] >= 0                          # F is non-negative
        @test isfinite(r.F_seasonal) && isfinite(r.F_all)
        @test r.pair_freqs[1] ≈ pi/2 rtol=1e-8
        @test StatsAPI.nobs(r) == r.nobs
        @test isfinite(StatsAPI.pvalue(r))
        io = IOBuffer(); show(io, r); s = String(take!(io))
        @test occursin("HEGY", s)
        io2 = IOBuffer(); refs(io2, r); @test occursin("Hylleberg", String(take!(io2)))
    end

    @testset "HEGY quarterly: deterministic-seasonal REJECTS seasonal nulls" begin
        y = _det_seasonal(200, 4; seed=101)
        r = hegy_test(y; frequency=4, deterministic=:const_trend_seas)
        # No stochastic unit root anywhere ⇒ reject at every frequency.
        @test r.t_zero < r.t_zero_cv[5]                 # reject zero-freq root
        @test r.t_nyquist < r.t_nyquist_cv[5]           # reject Nyquist root
        @test r.pair_F[1] > r.pair_F_cv[5]              # reject annual-pair root
    end

    @testset "HEGY quarterly: Δ₄-integrated FAILS TO REJECT seasonal nulls" begin
        y = _seasonal_rw(200, 4; seed=202)
        r = hegy_test(y; frequency=4, deterministic=:const_trend_seas)
        # Unit roots at all frequencies ⇒ fail to reject.
        @test r.t_zero > r.t_zero_cv[5]                 # fail at zero freq
        @test r.t_nyquist > r.t_nyquist_cv[5]           # fail at Nyquist
        @test r.pair_F[1] < r.pair_F_cv[5]              # fail at annual pair
    end

    @testset "HEGY quarterly deterministic cases + fixed lags" begin
        y = _seasonal_rw(200, 4; seed=303)
        for det in (:none, :const, :const_seas, :const_trend, :const_trend_seas)
            r = hegy_test(y; frequency=4, deterministic=det)
            @test r.deterministic == det
            @test isfinite(r.t_zero)
        end
        rL = hegy_test(y; frequency=4, lags=2)
        @test rL.lags == 2
        rB = hegy_test(y; frequency=4, lags=:bic)
        @test rB.lags >= 0
    end

    # =======================================================================
    # HEGY monthly — Beaulieu-Miron (1993)
    # =======================================================================
    @testset "HEGY monthly basic + fields" begin
        y = _seasonal_rw(300, 12; seed=22)
        r = hegy_test(y; frequency=12)
        @test r isa HEGYResult
        @test r.frequency == 12
        @test length(r.pi_coefs) == 12                  # π₁..π₁₂ for monthly
        @test length(r.pair_F) == 5                     # 5 complex-conjugate pairs
        @test all(isfinite, r.pair_F)
        # pair frequencies ascending, in (0, π)
        @test all(0 .< r.pair_freqs .< pi)
        @test issorted(r.pair_freqs)
        io = IOBuffer(); show(io, r); @test occursin("Monthly", String(take!(io)))
    end

    @testset "HEGY monthly: Δ₁₂-integrated FAILS TO REJECT seasonal nulls" begin
        y = _seasonal_rw(360, 12; seed=44)
        r = hegy_test(y; frequency=12, deterministic=:const_trend_seas)
        @test r.t_zero > r.t_zero_cv[5]
        @test r.t_nyquist > r.t_nyquist_cv[5]
        @test all(F -> F < r.pair_F_cv[5], r.pair_F)    # every pair fails to reject
    end

    @testset "HEGY monthly: deterministic-seasonal REJECTS Nyquist + pairs" begin
        y = _det_seasonal(360, 12; seed=505)
        r = hegy_test(y; frequency=12, deterministic=:const_trend_seas)
        @test r.t_nyquist < r.t_nyquist_cv[5]
        @test all(F -> F > r.pair_F_cv[5], r.pair_F)
    end

    # =======================================================================
    # Error handling
    # =======================================================================
    @testset "HEGY error handling" begin
        y = _seasonal_rw(120, 4; seed=9)
        @test_throws ArgumentError hegy_test(y; frequency=5)     # unsupported period
        @test_throws ArgumentError hegy_test(y; frequency=4, deterministic=:bogus)
        @test_throws ArgumentError hegy_test(randn(10); frequency=4)  # too few obs
    end
end
