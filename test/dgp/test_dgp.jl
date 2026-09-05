# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# DGP-01 (#790): simulator self-tests — each DGP satisfies its closed-form
# moments. Every tolerance carries a one-line rationale (the #258 rule).

using Test
using Random
using LinearAlgebra
using Statistics
using DataFrames  # nrow (no-op when fixtures.jl already loaded it)

@testset "DGP library self-tests (DGP-01)" begin

    @testset "dgp_var: shapes, determinism, burn-in" begin
        r1 = dgp_var(MersenneTwister(11); T=100)
        @test size(r1.Y) == (100, 3)
        @test size(r1.eps) == (100, 3)
        @test r1.Sigma ≈ r1.B0 * r1.B0'
        # Same seed -> bit-identical draws (rng-first contract).
        r2 = dgp_var(MersenneTwister(11); T=100)
        @test r1.Y == r2.Y
        r3 = dgp_var(MersenneTwister(12); T=100)
        @test r1.Y != r3.Y
        # VAR(2) vector-of-matrices form.
        rv = dgp_var(MersenneTwister(11); A=[[0.5 0.0; 0.0 0.4], [0.1 0.0; 0.0 0.1]],
                     B0=[1.0 0.0; 0.0 1.0], T=50)
        @test size(rv.Y) == (50, 2) && length(rv.A) == 2
        # Sigma-only form is honored (DGP-02 #791: it was silently ignored
        # before, keeping the default B0); passing both throws.
        S_in = [1.0 0.4; 0.4 0.9]
        rs = dgp_var(MersenneTwister(11); A=[0.5 0.1; 0.0 0.4], Sigma=S_in, T=50)
        @test rs.Sigma ≈ S_in
        @test rs.B0 * rs.B0' ≈ S_in
        @test_throws ArgumentError dgp_var(MersenneTwister(11); Sigma=S_in,
                                           B0=[1.0 0.0; 0.0 1.0])
    end

    @testset "VAR sample autocovariance ≈ Lyapunov Γ₀" begin
        rng = MersenneTwister(21)
        d = dgp_var(rng; T=20000)
        G0 = lyapunov_gamma0(d.A[1], d.Sigma)
        Yc = d.Y .- mean(d.Y, dims=1)
        Shat = (Yc' * Yc) / size(Yc, 1)
        # MC sd of autocov entries ≈ sqrt(ΓᵢᵢΓⱼⱼ/T_eff) ≈ 0.05; 4× margin.
        @test Shat ≈ G0 atol=0.2
    end

    @testset "var_irf/var_fevd/var_hd identities" begin
        rng = MersenneTwister(22)
        d = dgp_var(rng; T=200, burn=0)  # starts at μ: HD identity is exact
        irf = var_irf(d.A, d.B0, 8)
        @test irf[1, :, :] ≈ d.B0  # Θ₀ = B0 exactly
        fevd = var_fevd(d.A, d.B0, 8)
        @test all(abs.(sum(fevd, dims=3) .- 1.0) .< 1e-12)  # rows sum to 1
        hd = var_hd(d.A, d.B0, d.eps)
        @test size(hd) == (200, 3, 3)
        # c = 0 here; HD sums to the demeaned path exactly (identity tolerance).
        @test dropdims(sum(hd, dims=3), dims=3) ≈ d.Y .- d.c' atol=1e-8
    end

    @testset "dgp_arima: lengths and order recovery smoke" begin
        rng = MersenneTwister(23)
        d = dgp_arima(rng; phi=[0.7], theta=[0.3], T=300)
        @test length(d.y) == 300
        @test d.phi == [0.7] && d.theta == [0.3] && d.d == 0
        di = dgp_arima(rng; phi=[0.5], d=1, T=100)
        @test length(di.y) == 100
        ds = dgp_arima(rng; phi=[0.5], Phi=[0.3], s=4, T=120)
        @test length(ds.y) == 120
    end

    @testset "dgp_garch_family: unconditional variance ≈ ω/(1−α−β)" begin
        rng = MersenneTwister(24)
        g = dgp_garch_family(rng; kind=:garch, omega=0.02, alpha=0.08,
                             beta=0.88, T=20000)
        uv = 0.02 / (1 - 0.08 - 0.88)
        # MC se of the variance ≈ uv·√(2/T_eff), persistence inflates ~3×.
        @test mean(g.y .^ 2) ≈ uv rtol=0.1
        @test mean(g.h) ≈ uv rtol=0.1
        @test all(g.h .> 0)
        for kind in (:arch, :egarch, :gjr, :aparch, :igarch, :cgarch,
                     :figarch, :fiegarch)
            gk = dgp_garch_family(MersenneTwister(25); kind=kind, T=200)
            @test length(gk.y) == 200 && length(gk.h) == 200
            @test all(gk.h .> 0)
        end
        gt = dgp_garch_family(MersenneTwister(26); innov=:t, T=200)
        @test length(gt.y) == 200
    end

    @testset "dgp_sv / dgp_mgarch / dgp_midas shapes" begin
        s = dgp_sv(MersenneTwister(27); T=200)
        @test length(s.y) == 200 && length(s.h) == 200
        for kind in (:ccc, :dcc, :bekk)
            m = dgp_mgarch(MersenneTwister(28); kind=kind, T=100)
            @test size(m.Y) == (100, 2) && size(m.H) == (100, 2, 2)
        end
        md = dgp_midas(MersenneTwister(29); T_lf=50)
        @test length(md.y) == 50 && abs(sum(md.w_true) - 1.0) < 1e-12
    end

    @testset "dgp_dynamic_factors: R² ≈ signal share" begin
        rng = MersenneTwister(30)
        f = dgp_dynamic_factors(rng; T=400, N=40)
        @test size(f.X) == (400, 40) && size(f.F) == (400, 2)
        r2 = 1 - sum(var(f.X - f.F * f.Lambda', dims=1)) /
                 sum(var(f.X, dims=1))
        # Variance-ratio noise at T = 400 ≈ 0.03; 3× margin around 0.7.
        @test r2 ≈ 0.7 atol=0.1
        fb = dgp_dynamic_factors(MersenneTwister(31); T=100,
                                 blocks=Dict(1 => collect(1:20)))
        @test all(fb.Lambda[21:40, 1] .== 0)
        # DGP-06: the returned innovations reproduce the factor path exactly
        # (F[t] = A·F[t-1] + L·eps[t] with L the Sigma_F Cholesky factor).
        L = cholesky(Symmetric(f.Sigma_F)).L
        @test f.F[2:end, :] ≈ f.F[1:end-1, :] * f.A[1]' + f.eps[2:end, :] * L'
    end

    @testset "dgp_mixed_frequency_panel: MM identity + NaN pattern" begin
        rng = MersenneTwister(32)
        mp = dgp_mixed_frequency_panel(rng; T=120, ragged=3)
        @test size(mp.Y, 1) == 120
        @test mp.agg_weights == [1.0, 2.0, 3.0, 2.0, 1.0]
        nM = size(mp.Lambda_M, 1)
        # Quarterly column observed iff t ≡ 0 (mod 3) on the kept window.
        for (i, t) in enumerate(201:320)
            mod(t, 3) != 0 && @test isnan(mp.Y[i, nM + 1])
        end
        # MM aggregate of the returned monthly factors ≈ quarterly signal
        # within the idiosyncratic noise (sd 0.2; 5 sd bound), on observed dates.
        mm = mm_aggregate(mp.F, mp.Lambda_Q)
        obs = [i for i in 5:115 if mod(200 + i, 3) == 0]
        @test maximum(abs.(mp.Y[obs, nM + 1] .- mm[obs, 1])) < 1.0
    end

    @testset "dgp_vecm: Δy regression recovers α" begin
        rng = MersenneTwister(33)
        v = dgp_vecm(rng; Gamma=zeros(3, 3), T=5000)
        dY = diff(v.Y, dims=1)
        ec = vec(v.Y[1:end - 1, :] * v.beta)
        # OLS se ≈ σ/√(T·Var(ec)) ≈ 0.01; 5× margin.
        for j in 1:3
            @test dot(ec, dY[:, j]) / dot(ec, ec) ≈ v.alpha[j] atol=0.05
        end
    end

    @testset "dgp_cointreg / dgp_panel_var / dgp_ardl / dgp_nardl / dgp_pmg" begin
        c = dgp_cointreg(MersenneTwister(34); T=100)
        @test length(c.y) == 100 && size(c.X) == (100, 2)
        cs = dgp_cointreg(MersenneTwister(35); T=100, spurious=true)
        @test length(cs.y) == 100
        pv = dgp_panel_var(MersenneTwister(36); N=10, T=25)
        @test size(pv.Y) == (250, 2) && length(pv.id) == 250
        a = dgp_ardl(MersenneTwister(37); T=100)
        @test length(a.y) == 100 && a.theta ≈ (0.8 + 0.4) / (1 - 0.6)
        nd = dgp_nardl(MersenneTwister(38); T=100)
        @test length(nd.y) == 100
        pm = dgp_pmg(MersenneTwister(39); N=5, T=30)
        @test length(pm.Y) == 150
    end

    @testset "dgp_lp_iv / dgp_state_dependent_var / dgp_propensity" begin
        li = dgp_lp_iv(MersenneTwister(40); T=200)
        @test size(li.Y) == (200, 3) && size(li.Z) == (200, 1)
        @test li.pi1 == 1.5 && li.theta == 1.0
        sv = dgp_state_dependent_var(MersenneTwister(41); T=200)
        @test size(sv.Y) == (200, 2) && length(sv.G) == 200
        @test size(sv.irf_exp) == (13, 2, 2)
        pr = dgp_propensity(MersenneTwister(42); n=500)
        @test length(pr.Y) == 500 && pr.att == 1.0
    end

    @testset "dgp_nongaussian_var / dgp_heteroskedastic_var" begin
        ng = dgp_nongaussian_var(MersenneTwister(43); T=500)
        @test size(ng.Y) == (500, 3)
        # t₅ shocks are leptokurtic (kurtosis 9, se ≈ 0.35 at T = 500).
        @test mean(ng.eps .^ 4) / mean(ng.eps .^ 2)^2 > 5.0
        for kind in (:markov, :garch, :smooth, :external)
            hh = dgp_heteroskedastic_var(MersenneTwister(44); kind=kind, T=200)
            @test size(hh.Y) == (200, 3) && size(hh.scales) == (200, 3)
        end
    end

    @testset "dgp_regime_switching: all four kinds return truth" begin
        ms = dgp_regime_switching(MersenneTwister(45); kind=:ms, T=200)
        @test length(ms.y) == 200 && all(s -> s == 1 || s == 2, ms.s)
        st = dgp_regime_switching(MersenneTwister(46); kind=:setar, T=200)
        @test length(st.y) == 200
        ls = dgp_regime_switching(MersenneTwister(47); kind=:lstar, T=200)
        @test all(0 .<= ls.G .<= 1)
        es = dgp_regime_switching(MersenneTwister(48); kind=:estr, T=200)
        @test all(0 .<= es.G .<= 1)
    end

    @testset "dgp_trend_cycle / dgp_ar2_peak / dgp_lagged_pair / dgp_state_space" begin
        tc = dgp_trend_cycle(MersenneTwister(49); T=200)
        @test length(tc.y) == 200
        # AR(2) cycle lag-1 autocorrelation = φ₁/(1−φ₂) ≈ 0.92 (se ≈ 0.03).
        @test cor(tc.cycle[1:199], tc.cycle[2:200]) > 0.8
        ap = dgp_ar2_peak(MersenneTwister(50); T=300)
        @test length(ap.y) == 300 && length(ap.spectrum) == 256
        _, imax = findmax(ap.spectrum)
        @test abs(ap.freqs[imax] - 2pi / 8) < 2pi / 256 * 2  # within 2 bins
        lp = dgp_lagged_pair(MersenneTwister(51); T=300)
        @test length(lp.x) == 300 && lp.d == 3 && lp.gain == 2.0
        ss = dgp_state_space(MersenneTwister(52); T=100)
        @test size(ss.y) == (100, 1) && size(ss.x) == (100, 1)
    end

    @testset "dgp_unit_root_pair: H0/H1 shapes for every kind" begin
        for kind in (:adf, :kpss, :trend, :break_level, :break_trend, :seasonal,
                     :fourier, :explosive, :cointegrated_pair, :granger,
                     :panel_ur, :nongaussian, :heteroskedastic_groups)
            d = dgp_unit_root_pair(MersenneTwister(53); kind=kind, T=120)
            @test d.truth.kind == kind
        end
        g = dgp_unit_root_pair(MersenneTwister(54); kind=:granger, T=500)
        # y₂ loads on lagged y₁ with 0.7: correlation must clear noise (se ≈ 0.045).
        @test cor(g.h1[2][2:end], g.h1[1][1:end - 1]) > 0.3
        @test abs(cor(g.h0[2][2:end], g.h0[1][1:end - 1])) < 0.2
    end

    @testset "dgp_cross_section: all 14 kinds run" begin
        for kind in (:ols, :hc, :cluster, :iv, :logit, :probit, :ordered,
                     :mlogit, :poisson, :nb, :tobit, :truncreg, :heckman,
                     :qreg, :rdd)
            d = dgp_cross_section(MersenneTwister(55); kind=kind, n=300)
            @test kind === :truncreg ? 0 < length(d.y) < 300 : length(d.y) == 300
        end
        lo = dgp_cross_section(MersenneTwister(56); kind=:logit, n=2000)
        @test length(logit_ame(lo.X, lo.beta)) == 2
        @test length(probit_ame(lo.X, lo.beta)) == 2
    end

    @testset "dgp_panel / dgp_staggered_did" begin
        p = dgp_panel(MersenneTwister(57); N=20, T=10)
        @test nrow(p.df) == 200 && p.mundlak == zeros(2)
        pc = dgp_panel(MersenneTwister(58); N=20, T=10, corr_alpha_x=0.7)
        @test pc.mundlak == fill(0.7, 2)
        d = dgp_staggered_did(MersenneTwister(59); N=120, T=25)
        @test nrow(d.df) == 3000
        # e = 0 is observed for every cohort: ATT(0) ≈ mean_g τ(g,0) = 1.25
        # (cohort-share noise ≈ 0.03; 3× margin).
        @test d.att_by_event_time[0] ≈ 1.25 atol=0.1
        # overall_att recomputed independently from cohort assignments.
        num, den = 0.0, 0
        for (i, g) in enumerate(d.cohort_of)
            g === nothing && continue
            for t in g:25
                num += 1.0 + 0.1 * (t - g) + 0.05 * (g - 6)
                den += 1
            end
        end
        @test d.overall_att ≈ num / den
    end

    @testset "dgp_gmm / dgp_pce_draws / dgp_dsge_observed" begin
        g = dgp_gmm(MersenneTwister(60); n=200)
        @test length(g.y) == 200 && size(g.Z, 2) == 3
        go = dgp_gmm(MersenneTwister(61); kind=:ols, n=200)
        @test length(go.y) == 200
        ce = dgp_pce_draws(MersenneTwister(62), [1.0, 2.0, 3.0]; sd=0.1)
        @test size(ce.draws) == (500, 3)
        # Width scales with sd on centred draws: doubling sd doubles the
        # 5–95% range (pooled over coordinates; MC noise ≈ ±10%).
        ce2 = dgp_pce_draws(MersenneTwister(62), [1.0, 2.0, 3.0]; sd=0.2)
        q1 = quantile(vec(ce.draws .- ce.point'), [0.05, 0.95])
        q2 = quantile(vec(ce2.draws .- ce2.point'), [0.05, 0.95])
        @test (q2[2] - q2[1]) / (q1[2] - q1[1]) ≈ 2.0 atol=0.3
        do_ = dgp_dsge_observed(MersenneTwister(63), ones(50, 2); H=[0.25, 0.25])
        @test size(do_.y_obs) == (50, 2)
    end

    @testset "arma_spectrum peaks at the AR(2) frequency" begin
        fr = collect(range(0, pi; length=256))
        sp = arma_spectrum([2 * 0.9 * cos(2pi / 8), -0.9^2], Float64[], 1.0, fr)
        _, imax = findmax(sp)
        @test abs(fr[imax] - 2pi / 8) < 2pi / 256 * 2  # within 2 bins
    end
end
