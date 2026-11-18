# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test, MacroEconometricModels, LinearAlgebra, Statistics, Random, Distributions, StatsAPI
using MacroEconometricModels: dof_residual

@testset "RI Econometric Estimation & Coibion-Gorodnichenko Diagnostics (#920)" begin

    # =========================================================================
    # 1. Input Validation & Edge Cases
    # =========================================================================
    @testset "Input Validation" begin
        prob = RationalInattentionProblem(0.8, 1.0, 1.0; capacity=0.75)
        sol = solve_ri_kalman(prob)
        sim = simulate(sol, 60; burn=50, seed=101)
        data = hcat(sim.states, sim.actions)

        @test_throws ArgumentError estimate_ri(prob, data; method=:invalid)
        @test_throws ArgumentError estimate_ri(prob, data; estimate_param=:invalid)
        @test_throws ArgumentError estimate_ri(prob, data; cov_type=:invalid)
        @test_throws ArgumentError estimate_ri(prob, data; n_draws=0)
        @test_throws ArgumentError estimate_ri(prob, data; burnin=-1)
        @test_throws ArgumentError estimate_ri(prob, data; step_scale=0.0)
        @test_throws ArgumentError estimate_ri(prob, ones(1, 2))
        @test_throws ArgumentError estimate_ri(prob, fill(NaN, 10, 2))
        @test_throws ArgumentError estimate_ri(prob, vcat(data, [Inf 0.0]))
        @test_throws ArgumentError estimate_ri(prob, data; extra_params_init=[0.5])
        @test_throws ArgumentError estimate_ri(k -> RationalInattentionProblem(0.8, 1.0, 1.0; capacity=k),
                                               data; extra_params_init=[0.5], extra_param_names=["a", "b"])
        @test_throws ArgumentError estimate_ri(prob, data; observables=:bad)
        @test_throws ArgumentError estimate_ri(prob, data; observables=[1, 2, 3])
        @test_throws ArgumentError estimate_ri(prob, data; measurement_error=-0.1)

        # CG test validation
        @test_throws DimensionMismatch coibion_gorodnichenko_test(ones(10), ones(9), ones(10))
        @test_throws ArgumentError coibion_gorodnichenko_test(ones(3), ones(3), ones(3))
        @test_throws ArgumentError coibion_gorodnichenko_test([1.0, 2.0, Inf, 4.0, 5.0], ones(5), zeros(5))
        @test_throws ArgumentError coibion_gorodnichenko_test(randn(20), ones(20), ones(20)) # zero revisions
        @test_throws ArgumentError coibion_gorodnichenko_test(randn(20), randn(20), randn(20); hac_lags=:bad)
        @test_throws ArgumentError coibion_gorodnichenko_test(randn(20), randn(20), randn(20); hac_lags=-1)

        # implied_cg_slope validation
        @test_throws ArgumentError implied_cg_slope(sol; horizon=0)
        @test_throws ArgumentError implied_cg_slope(sol; weights=[1.0, 2.0])
        @test_throws ArgumentError implied_cg_slope(sol; weights=[0.0])
        sol_zero = solve_ri_kalman(RationalInattentionProblem(0.8, 1.0, 1.0; omega=100.0))
        @test isnan(implied_cg_slope(sol_zero))
    end

    # =========================================================================
    # 2. AC 2: Scalar AR(1) Implied CG Slope & Simulated Consensus Recovery
    # =========================================================================
    @testset "Implied CG Slope & Coibion-Gorodnichenko Test" begin
        for rho in (0.5, 0.85, 0.95), kappa in (0.25, 0.75, 1.5, 2.5)
            prob = RationalInattentionProblem(rho, 1.0, 1.0; capacity=kappa, units=:bits)
            sol = solve_ri_kalman(prob)
            G = 1.0 - 2.0^(-2.0 * kappa)
            beta_exact = (1.0 - G) / G
            @test abs(beta_exact - 1.0 / (2.0^(2.0 * kappa) - 1.0)) < 1e-12
            for h in (1, 2, 4)
                slope = implied_cg_slope(sol; horizon=h)
                @test abs(slope - beta_exact) < 1e-10
            end
        end

        # Multivariate system parity against large-sample consensus simulation
        A2 = [0.85 0.1; 0.0 0.65]
        B2 = [1.0 0.0; 0.2 0.8]
        H2 = [1.0, 0.5]
        Bu2 = [0.1, -0.05]
        prob2 = RationalInattentionProblem(A2, B2, H2; B_u=Bu2, beta=0.96, capacity=0.9)
        sol2 = solve_ri_kalman(prob2)
        beta2_pop = implied_cg_slope(sol2; horizon=2, weights=[1.0, 0.5])
        @test isfinite(beta2_pop) && beta2_pop > 0.0

        # Simulate consensus forecasts for bivariate system
        rng2 = Xoshiro(9201)
        T_mc = 8000
        KY2 = sol2.K_gain * sol2.Y_signal'
        Acl2 = A2 + prob2.B_u * sol2.F_policy
        x_arr = zeros(T_mc, 2)
        xbar_arr = zeros(T_mc, 2)
        x_cur = zeros(2)
        xb_cur = zeros(2)
        for t in 1:T_mc
            w_t = randn(rng2, 2)
            x_cur = A2 * x_cur + prob2.B_u * (sol2.F_policy * xb_cur) + B2 * w_t
            xb_prior = Acl2 * xb_cur
            xb_cur = xb_prior + KY2 * (x_cur - xb_prior)
            x_arr[t, :] = x_cur
            xbar_arr[t, :] = xb_cur
        end
        v2 = [1.0, 0.5]
        h_test = 2
        Acl2_h = Acl2^h_test
        idx_t = 100:(T_mc - h_test)
        y_real = [dot(v2, x_arr[t + h_test, :]) for t in idx_t]
        f_now = [dot(v2, Acl2_h * xbar_arr[t, :]) for t in idx_t]
        f_lag = [dot(v2, Acl2_h * Acl2 * xbar_arr[t - 1, :]) for t in idx_t]
        cg2 = coibion_gorodnichenko_test(y_real, f_now, f_lag; hac_lags=4)
        @test abs(cg2.beta_cg - beta2_pop) < 2.5 * cg2.se_cg

        # Scalar AR(1) consensus simulation -> coibion_gorodnichenko_test within 2 HAC SEs
        rho_s = 0.85
        kappa_s = 0.75
        prob_s = RationalInattentionProblem(rho_s, 1.0, 1.0; capacity=kappa_s, units=:bits)
        sol_s = solve_ri_kalman(prob_s)
        beta_pop_s = implied_cg_slope(sol_s; horizon=1)
        G_s = sol_s.K_gain[1, 1]

        rng_s = Xoshiro(9202)
        T_s = 600
        x_s = zeros(T_s)
        xb_s = zeros(T_s)
        for t in 2:T_s
            x_s[t] = rho_s * x_s[t - 1] + randn(rng_s)
            xb_s[t] = (1.0 - G_s) * rho_s * xb_s[t - 1] + G_s * x_s[t]
        end
        y_s = x_s[101:end]
        fn_s = rho_s .* xb_s[100:(end - 1)]
        fl_s = (rho_s^2) .* xb_s[99:(end - 2)]
        cg_s = coibion_gorodnichenko_test(y_s, fn_s, fl_s)
        @test cg_s isa CGTestResult{Float64}
        @test abs(cg_s.beta_cg - beta_pop_s) < 2.0 * cg_s.se_cg
        @test cg_s.pvalue < 0.01
        @test isapprox(cg_s.implied_gain, 1.0 / (1.0 + cg_s.beta_cg); atol=1e-12)
        @test isapprox(cg_s.implied_capacity, -0.5 * log2(1.0 - cg_s.implied_gain); atol=1e-12)
        @test StatsAPI.pvalue(cg_s) == cg_s.pvalue
        @test nobs(cg_s) == cg_s.n_obs
        @test dof(cg_s) == 1
        @test dof_residual(cg_s) == cg_s.n_obs - 2

        # NaN missingness & explicit hac_lags=0
        fn_nan = copy(fn_s)
        fn_nan[10] = NaN
        fn_nan[25] = NaN
        cg_nan = coibion_gorodnichenko_test(y_s, fn_nan, fl_s; hac_lags=0)
        @test cg_nan.n_obs == cg_s.n_obs - 2
        @test cg_nan.hac_lags == 0
        @test isfinite(cg_nan.beta_cg)
    end

    # =========================================================================
    # 3. AC 1: Kalman MLE Recovery (T = 300, κ₀ = 0.75) & Missing Data
    # =========================================================================
    @testset "Kalman MLE Recovery & Full-Information LR Test" begin
        kappa0 = 0.75
        rho0 = 0.85
        prob0 = RationalInattentionProblem(rho0, 1.0, 1.0; capacity=kappa0, units=:bits)
        sol0 = solve_ri_kalman(prob0)

        sim = simulate(sol0, 300; burn=200, seed=9203)
        data_biv = hcat(sim.states, sim.actions)

        # Fixed-problem MLE on bivariate macro + action system
        mle_cap = estimate_ri(prob0, data_biv; method=:mle, estimate_param=:capacity)
        @test mle_cap isa RIEstimationResult{Float64}
        @test mle_cap.converged
        @test mle_cap.method === :mle
        @test mle_cap.estimate_param === :capacity
        @test abs(mle_cap.capacity - kappa0) <= 0.18
        @test mle_cap.lr_stat > 0.0
        @test mle_cap.lr_pvalue < 0.01
        @test isempty(mle_cap.draws)
        @test isnan(mle_cap.acceptance_rate)
        @test mle_cap.manifest === nothing
        @test mle_cap.n_obs == 300
        @test isapprox(mle_cap.aic, -2.0 * mle_cap.loglik + 2.0; atol=1e-10)
        @test isapprox(mle_cap.bic, -2.0 * mle_cap.loglik + log(300.0); atol=1e-10)
        @test mle_cap.stderr[1] > 0.0 && isfinite(mle_cap.stderr[1])

        # Dual ω estimation recovers sol0.omega and matching implied κ
        mle_om = estimate_ri(prob0, data_biv; method=:mle, estimate_param=:omega, cov_type=:sandwich)
        @test mle_om.converged
        @test mle_om.estimate_param === :omega
        @test abs(mle_om.capacity - mle_cap.capacity) < 1e-3
        @test isapprox(mle_om.omega, sol0.omega; rtol=0.35)
        @test mle_om.lr_pvalue < 0.01

        # Missing survey observations (NaN in actions column)
        data_miss = copy(data_biv)
        rng_miss = Xoshiro(9204)
        for t in 1:300
            if rand(rng_miss) < 0.25
                data_miss[t, 2] = NaN
            end
        end
        mle_miss = estimate_ri(prob0, data_miss; method=:mle, observables=:actions_and_states)
        @test mle_miss.converged
        @test abs(mle_miss.capacity - kappa0) <= 0.22
        @test mle_miss.lr_pvalue < 0.01

        # Joint builder estimating (κ, ρ) simultaneously
        build_joint = (kap, th) -> RationalInattentionProblem(tanh(th[1]), 1.0, 1.0; capacity=kap)
        mle_joint = estimate_ri(build_joint, data_biv;
                                method=:mle,
                                extra_params_init=[atanh(0.6)],
                                extra_param_names=["atanh_rho"],
                                init_param=1.0)
        @test mle_joint.converged
        @test length(mle_joint.params) == 2
        @test mle_joint.param_names == ["capacity", "atanh_rho"]
        @test abs(mle_joint.capacity - kappa0) <= 0.18
        @test abs(tanh(mle_joint.params[2]) - rho0) <= 0.12
        @test mle_joint.lr_pvalue < 0.01

        # Univariate actions-only estimation & near-FIRE non-rejection check
        mle_act = estimate_ri(prob0, sim.actions; method=:mle, observables=:actions)
        @test mle_act.converged
        @test mle_act.lr_pvalue < 0.05

        prob_fire = RationalInattentionProblem(0.8, 1.0, 1.0; capacity=15.0, units=:bits)
        sol_fire = solve_ri_kalman(prob_fire)
        sim_fire = simulate(sol_fire, 300; burn=200, seed=9205)
        mle_fire = estimate_ri(prob_fire, sim_fire.actions; method=:mle, observables=:actions)
        @test mle_fire.lr_pvalue > 0.05
    end

    # =========================================================================
    # 4. Bayesian RWMH Estimation & Reproducibility
    # =========================================================================
    @testset "Bayesian Adaptive RWMH & Reproducibility" begin
        kappa0 = 0.75
        prob0 = RationalInattentionProblem(0.85, 1.0, 1.0; capacity=kappa0, units=:bits)
        sol0 = solve_ri_kalman(prob0)
        sim = simulate(sol0, 250; burn=150, seed=9206)
        data = hcat(sim.states, sim.actions)

        b1 = estimate_ri(prob0, data; method=:bayes, n_draws=300, burnin=150, seed=920)
        b2 = estimate_ri(prob0, data; method=:bayes, n_draws=300, burnin=150, seed=920)
        b3 = estimate_ri(prob0, data; method=:bayes, n_draws=300, burnin=150, seed=921)

        @test b1 isa RIEstimationResult{Float64}
        @test b1.method === :bayes
        @test size(b1.draws) == (300, 1)
        @test 0.0 < b1.acceptance_rate < 1.0
        @test b1.manifest isa ReproManifest
        @test b1.manifest.seed == 920
        @test b1.draws == b2.draws
        @test b1.params == b2.params
        @test b1.draws != b3.draws
        @test abs(b1.capacity - kappa0) <= 0.25

        # Custom priors (Dict / Vector)
        b_prior = estimate_ri(prob0, data; method=:bayes,
                              priors=Dict("capacity" => Gamma(2.0, 0.5)),
                              n_draws=200, burnin=100, seed=922)
        @test size(b_prior.draws) == (200, 1)
        @test b_prior.converged
    end

    # =========================================================================
    # 5. StatsAPI, RSER-01, Float32, Display & Ambiguities
    # =========================================================================
    @testset "StatsAPI, RSER-01, Float32, Display & Ambiguities" begin
        @test isempty(Test.detect_ambiguities(MacroEconometricModels))

        prob = RationalInattentionProblem(0.85, 1.0, 1.0; capacity=0.75)
        sol = solve_ri_kalman(prob)
        sim = simulate(sol, 150; burn=100, seed=9207)
        data = hcat(sim.states, sim.actions)

        r_mle = estimate_ri(prob, data; method=:mle)
        r_bayes = estimate_ri(prob, data; method=:bayes, n_draws=120, burnin=60, seed=923)

        for r in (r_mle, r_bayes)
            @test coef(r) == r.params
            @test StatsAPI.coefnames(r) == r.param_names
            @test vcov(r) == r.vcov_mat
            @test stderror(r) == r.stderr
            @test loglikelihood(r) == r.loglik
            @test aic(r) == r.aic
            @test bic(r) == r.bic
            @test nobs(r) == r.n_obs
            @test dof(r) == length(r.params)
            @test dof_residual(r) == r.n_obs - length(r.params)
            @test !islinear(r)
            ci = confint(r; level=0.95)
            @test size(ci) == (length(r.params), 2)
            @test all(ci[:, 1] .< r.params .< ci[:, 2])
        end

        # RSER-01 exact positional constructor
        for obj in (r_mle, r_bayes)
            flds = [getfield(obj, f) for f in fieldnames(RIEstimationResult)]
            rec = RIEstimationResult{Float64}(flds...)
            for f in fieldnames(RIEstimationResult)
                @test isequal(getfield(rec, f), getfield(obj, f))
            end
        end

        cg = coibion_gorodnichenko_test(sim.states[2:end, 1], sim.actions[2:end, 1], sim.actions[1:end-1, 1])
        cg_flds = [getfield(cg, f) for f in fieldnames(CGTestResult)]
        cg_rec = CGTestResult{Float64}(cg_flds...)
        for f in fieldnames(CGTestResult)
            @test isequal(getfield(cg_rec, f), getfield(cg, f))
        end

        # Display & backends
        @test occursin("Rational Inattention Econometric Estimation", sprint(show, r_mle))
        @test occursin("Parameter Estimates", sprint(report, r_mle))
        @test occursin("Posterior Parameter Summary", sprint(report, r_bayes))
        @test occursin("Coibion–Gorodnichenko", sprint(show, cg))
        @test occursin("CG Regression", sprint(report, cg))

        latex_r = with_display_backend(:latex) do; sprint(report, r_mle); end
        html_r = with_display_backend(:html) do; sprint(report, r_bayes); end
        latex_cg = with_display_backend(:latex) do; sprint(report, cg); end
        @test occursin("\\begin{tabular}", latex_r)
        @test occursin("<table", html_r)
        @test occursin("\\begin{tabular}", latex_cg)

        # Float32 precision support
        prob32 = RationalInattentionProblem(0.85f0, 1.0f0, 1.0f0; capacity=0.75f0)
        sol32 = solve_ri_kalman(prob32)
        @test typeof(implied_cg_slope(sol32)) === Float32
        sim32 = simulate(sol32, 120; burn=80, seed=9208)
        r32 = estimate_ri(prob32, hcat(sim32.states, sim32.actions); method=:mle)
        @test r32 isa RIEstimationResult{Float32}
        cg32 = coibion_gorodnichenko_test(Float32.(sim32.states[2:end, 1]),
                                          Float32.(sim32.actions[2:end, 1]),
                                          Float32.(sim32.actions[1:end-1, 1]))
        @test cg32 isa CGTestResult{Float32}
    end
end
