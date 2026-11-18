# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
using LinearAlgebra
using Statistics
using Random
using StatsAPI

const MEM = MacroEconometricModels
const _suppress_warnings = MEM._suppress_warnings

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

@testset "NGID-07: Fundamentalness & Informational Sufficiency Tests (#907)" begin

    @testset "1. Method Ambiguity, Exports & Input Validation" begin
        @test isempty(Test.detect_ambiguities(MacroEconometricModels))
        for sym in (:FundamentalnessTestResult, :SufficientInformationTestResult,
                    :test_fundamentalness, :test_sufficient_information, :identify_cvm)
            @test isdefined(MacroEconometricModels, sym)
        end

        Y_ok = dgp_var(Xoshiro(90701); A=[0.4 0.0; 0.0 0.4], B0=Matrix{Float64}(I, 2, 2), T=120).Y
        m_ok = estimate_var(Y_ok, 1)
        @test_throws ArgumentError test_fundamentalness(m_ok; max_lag=0)
        @test_throws ArgumentError test_fundamentalness(m_ok; n_boot=0)
        @test_throws ArgumentError test_fundamentalness(m_ok; moments=Symbol[])
        @test_throws ArgumentError test_fundamentalness(m_ok; moments=[:bad])

        X_bad = randn(Xoshiro(90702), 50, 5)
        @test_throws ArgumentError test_sufficient_information(m_ok, X_bad)
        X_good = randn(Xoshiro(90703), 120, 5)
        @test_throws ArgumentError test_sufficient_information(m_ok, X_good; r_max=0)
        @test_throws ArgumentError test_sufficient_information(m_ok, X_good; p_f=0)
    end

    @testset "2. CCE (2017) Higher-Moment Fundamentalness Test (Size & Power)" begin
        _suppress_warnings() do
            # (a) Fundamental VAR(1) with t(5) shocks — size replication
            Phi_fund = [0.5 0.1; -0.1 0.4]
            C_id = [1.0 0.3; 0.2 0.9]
            n_size = FAST ? 30 : 60
            n_rej_size = 0
            for rep in 1:n_size
                sim_f = dgp_ssvarma(; T=400, Phi=Phi_fund, Theta=zeros(2, 2, 0),
                                    C=C_id, dist=:student_t, nu=5.0, seed=907100 + rep)
                mf = estimate_var(sim_f.Y, 1)
                rf = test_fundamentalness(mf; max_lag=6, n_boot=299, seed=907200 + rep)
                if rf.pvalue < 0.05
                    n_rej_size += 1
                end
            end
            size_rate = n_rej_size / n_size
            @test 0.0 <= size_rate <= 0.15

            # (b) Non-fundamental VMA(1) (Θ = [-0.5 0; 1 -2]) approximated by VAR(4) — power > 80%
            Theta_nf = [-0.5 0.0; 1.0 -2.0]
            C_nf = [1.0 0.4; 0.3 0.8]
            n_pow = FAST ? 25 : 50
            n_rej_asym = 0
            n_rej_skew = 0
            for rep in 1:n_pow
                sim_a = dgp_ssvarma(; T=400, Phi=zeros(2, 2, 0), Theta=Theta_nf,
                                    C=C_nf, dist=:mixture_normal_asym, seed=907300 + rep)
                r_asym = test_fundamentalness(estimate_var(sim_a.Y, 4);
                                              max_lag=6, n_boot=199, seed=907400 + rep)
                r_asym.pvalue < 0.05 && (n_rej_asym += 1)

                sim_s = dgp_ssvarma(; T=400, Phi=zeros(2, 2, 0), Theta=Theta_nf,
                                    C=C_nf, dist=:skew_normal, seed=907800 + rep)
                r_skew = test_fundamentalness(estimate_var(sim_s.Y, 4);
                                              max_lag=6, n_boot=199, seed=907900 + rep)
                r_skew.pvalue < 0.05 && (n_rej_skew += 1)
            end
            @test n_rej_asym / n_pow > 0.80
            @test n_rej_skew / n_pow > 0.80

            # Symmetric Student-t(5) power via kurtosis moments
            n_rej_t5 = 0
            for rep in 1:n_pow
                sim_t = dgp_ssvarma(; T=550, Phi=zeros(2, 2, 0), Theta=Theta_nf,
                                    C=C_nf, dist=:student_t, nu=5.0, seed=908100 + rep)
                r_t5 = test_fundamentalness(estimate_var(sim_t.Y, 4);
                                            max_lag=6, moments=[:kurt], n_boot=199, seed=908200 + rep)
                r_t5.pvalue < 0.05 && (n_rej_t5 += 1)
            end
            @test n_rej_t5 / n_pow > 0.80

            # (c) Reproducibility & Float32 support
            sim1 = dgp_ssvarma(; T=250, Phi=Phi_fund, Theta=zeros(2, 2, 0),
                               C=C_id, dist=:student_t, nu=5.0, seed=907501)
            m1 = estimate_var(sim1.Y, 1)
            r_a = test_fundamentalness(m1; max_lag=4, n_boot=199, seed=907502)
            r_b = test_fundamentalness(m1; max_lag=4, n_boot=199, seed=907502)
            @test r_a.statistic == r_b.statistic
            @test r_a.pvalue == r_b.pvalue

            m_f32 = estimate_ssvarma(Float32.(sim1.Y), 1, 0; method=:gmm, seed=907503)
            r_f32 = test_fundamentalness(m_f32; max_lag=4, n_boot=99, seed=907504)
            @test r_f32 isa FundamentalnessTestResult{Float32}
            @test isfinite(r_f32.statistic) && 0.0f0 <= r_f32.pvalue <= 1.0f0

            # (d) SVARMAModel and AbstractARIMAModel dispatch
            sim_nf0 = dgp_ssvarma(; T=300, Phi=zeros(2, 2, 0), Theta=Theta_nf,
                                  C=C_nf, dist=:student_t, nu=5.0, seed=907510)
            m_ss = estimate_ssvarma(sim_nf0.Y, 0, 1; method=:ml, distribution=:student_t, seed=907511)
            r_ss = test_fundamentalness(m_ss; max_lag=6, n_boot=199, seed=907512)
            @test r_ss isa FundamentalnessTestResult{Float64}
            @test r_ss.model_type === :svarma
            @test r_ss.pvalue > 0.05  # Recovered non-fundamental shocks satisfy MDD

            y_uni = sim1.Y[:, 1]
            m_ar = estimate_ar(y_uni, 1)
            r_ar = test_fundamentalness(m_ar; max_lag=4, n_boot=99, seed=907513)
            @test r_ar isa FundamentalnessTestResult
            @test r_ar.n == 1 && isfinite(r_ar.statistic)
        end
    end

    @testset "3. Forni & Gambetti (2014) Informational Sufficiency (FVRSW Permanent Income)" begin
        _suppress_warnings() do
            # FVRSW (2007) Permanent Income DGP:
            # Permanent news η₁ affects surplus s_t with a 1-period lag (s_t non-fundamental alone),
            # while forward-looking consumption c_t reacts contemporaneously (bivariate [s_t, c_t] fundamental).
            rng = Xoshiro(907601)
            burn = 100
            T_sim = 300
            T_tot = T_sim + burn
            eta1 = randn(rng, T_tot)
            eta2 = randn(rng, T_tot)
            s_vec = zeros(T_tot)
            c_vec = zeros(T_tot)
            rho_s = 0.4
            for t in 2:T_tot
                s_vec[t] = rho_s * s_vec[t - 1] + 0.35 * eta1[t] + 1.0 * eta1[t - 1] + 0.30 * eta2[t]
                c_vec[t] = 1.50 * eta1[t] + 0.15 * eta2[t]
            end
            s_obs = s_vec[(burn + 1):end]
            c_obs = c_vec[(burn + 1):end]

            # Build macroeconomic panel X_panel containing consumption c_t, s_t, and noisy factor loadings
            N_panel = 20
            X_panel = zeros(T_sim, N_panel)
            X_panel[:, 1] = c_obs
            X_panel[:, 2] = s_obs
            loadings = randn(rng, N_panel - 2, 2) .* 0.8
            factors_true = hcat(eta1[(burn + 1):end], eta2[(burn + 1):end])
            for j in 3:N_panel
                X_panel[:, j] = factors_true * loadings[j - 2, :] .+ 0.35 .* randn(rng, T_sim)
            end

            # (a) Univariate surplus VAR omits c_t -> strongly rejects sufficiency (p < 0.01)
            m_uni = estimate_var(reshape(s_obs, T_sim, 1), 4)
            res_uni = test_sufficient_information(m_uni, X_panel; r_max=4, p_f=4)
            @test res_uni isa SufficientInformationTestResult{Float64}
            @test res_uni.pvalue < 0.01
            @test res_uni.wald_pvalue < 0.01

            # (b) Consumption c_t enters the VAR -> ceases to reject (p > 0.10)
            m_aug = estimate_var(hcat(s_obs, c_obs), 6)
            res_aug = test_sufficient_information(m_aug, X_panel; r_max=4, p_f=4)
            @test res_aug.pvalue > 0.10
            @test res_aug.wald_pvalue > 0.10

            # (c) FactorModel, raw shocks, ICASVARResult, Vector, and Float32 inputs
            fm = estimate_factors(X_panel, 4; standardize=true)
            res_fm = test_sufficient_information(m_uni, fm; r_max=4, p_f=4)
            @test isapprox(res_fm.statistic, res_uni.statistic; atol=1e-10)

            res_vec = test_sufficient_information(vec(m_uni.U), X_panel; r_max=4, p_f=4)
            @test res_vec.pvalue < 0.01

            sim_svar = dgp_ssvarma(; T=240, Phi=[0.4 0.0; 0.0 0.4], Theta=[-0.5 0.0; 1.0 -2.0],
                                   C=[1.0 0.4; 0.3 0.8], dist=:student_t, nu=5.0, seed=907610)
            m_ssv = estimate_ssvarma(sim_svar.Y, 1, 1; method=:ml, seed=907611)
            Xp_svar = hcat(sim_svar.Y, randn(Xoshiro(907612), 240, 10))
            res_ssv = test_sufficient_information(m_ssv, Xp_svar; r_max=3, p_f=3)
            @test res_ssv isa SufficientInformationTestResult{Float64}

            res_f32 = test_sufficient_information(Float32.(m_aug.U),
                                                  Float32.(X_panel); r_max=3, p_f=3)
            @test res_f32 isa SufficientInformationTestResult{Float32}
        end
    end

    @testset "4. StatsAPI Interface & Pretty Display" begin
        _suppress_warnings() do
            sim = dgp_ssvarma(; T=200, Phi=[0.4 0.0; 0.0 0.4], Theta=zeros(2, 2, 0),
                              C=Matrix{Float64}(I, 2, 2), dist=:student_t, nu=5.0, seed=907701)
            m = estimate_var(sim.Y, 1)
            rf = test_fundamentalness(m; max_lag=4, n_boot=99, seed=907702)

            @test rf isa StatsAPI.HypothesisTest
            @test StatsAPI.nobs(rf) == rf.nobs
            @test StatsAPI.dof(rf) == rf.max_lag * length(rf.moments) * rf.n * rf.n
            @test StatsAPI.pvalue(rf) == rf.pvalue

            buf1 = IOBuffer()
            show(buf1, rf)
            txt1 = String(take!(buf1))
            @test occursin("Fundamentalness Test", txt1)
            @test occursin("Bootstrap P-value", txt1)
            @test report(devnull, rf) === nothing

            Xp = hcat(sim.Y, randn(Xoshiro(907703), 200, 10))
            rsi = test_sufficient_information(m, Xp; r_max=3, p_f=3)
            @test rsi isa StatsAPI.HypothesisTest
            @test StatsAPI.nobs(rsi) == rsi.nobs
            @test StatsAPI.dof(rsi) == rsi.df
            @test StatsAPI.pvalue(rsi) == rsi.pvalue

            buf2 = IOBuffer()
            show(buf2, rsi)
            txt2 = String(take!(buf2))
            @test occursin("Informational Sufficiency Test", txt2)
            @test occursin("HAC Wald", txt2)
            @test report(devnull, rsi) === nothing
        end
    end
end
