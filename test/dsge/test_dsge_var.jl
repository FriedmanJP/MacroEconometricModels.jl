# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

using Test
using LinearAlgebra
using Random
using Statistics
using Distributions
using SpecialFunctions
using MacroEconometricModels

@testset "ABCD-06: Bayesian DSGE-VAR(λ) (Del Negro & Schorfheide 2004, 2007)" begin
    # -------------------------------------------------------------------------
    # Canonical 3-equation New Keynesian DSGE (3 observables, 3 structural shocks)
    # -------------------------------------------------------------------------
    nk_spec = @dsge begin
        parameters: beta = 0.99, kappa = 0.30, psi1 = 1.50, psi2 = 0.25,
                    rho_r = 0.50, rho_d = 0.70, sig_r = 0.01, sig_d = 0.01, sig_pi = 0.01
        endogenous: y, pi, r, xd
        exogenous: e_r, e_d, e_pi
        steady_state: [0.0, 0.0, 0.0, 0.0]

        y[t] = y[t+1] - (r[t] - pi[t+1]) + xd[t]
        pi[t] = beta * pi[t+1] + kappa * y[t] + sig_pi * e_pi[t]
        r[t] = rho_r * r[t-1] + (1 - rho_r) * (psi1 * pi[t] + psi2 * y[t]) + sig_r * e_r[t]
        xd[t] = rho_d * xd[t-1] + sig_d * e_d[t]
    end

    sol_nk = solve(nk_spec)
    @test is_determined(sol_nk)
    obs_nk = [:y, :pi, :r]
    abcd_nk = abcd_representation(sol_nk, obs_nk)
    @test nobservables(abcd_nk) == 3
    @test nshocks(abcd_nk) == 3

    # Helper: simulate observable series from solved DSGE after burn-in
    function _sim_nk_obs(sol, T_raw::Int; burnin::Int=200, seed::Int=2026)
        sim_all = simulate(sol, T_raw + burnin; seed=seed)
        obs_idx = [findfirst(==(sym), sol.spec.endog) for sym in obs_nk]
        return Matrix{Float64}(sim_all[(burnin + 1):end, obs_idx])
    end

    Y_true = _sim_nk_obs(sol_nk, 250; burnin=250, seed=2026)
    p = 2
    T_eff = size(Y_true, 1) - p

    # =========================================================================
    # 1. Population-Moment NIW Prior & QR Structural Rotation
    # =========================================================================
    @testset "Population-Moment Prior & QR Rotation" begin
        prior_sol = dsge_var_prior(sol_nk, p, 1.5, T_eff; observables=obs_nk)
        prior_abcd = dsge_var_prior(abcd_nk, p, 1.5, T_eff)
        prior_spec = dsge_var_prior(nk_spec, p, 1.5, T_eff; observables=obs_nk)
        pert_nk = perturbation_solver(compute_steady_state(nk_spec); order=1)
        prior_pert = dsge_var_prior(pert_nk, p, 1.5, T_eff; observables=obs_nk)
        ssm_nk = StateSpaceModel(
            abcd_nk.C, Matrix{Float64}(0.01 * I, 3, 3), abcd_nk.A, Matrix{Float64}(I, 3, 3);
            d=abcd_nk.d_obs, c=abcd_nk.c_state, R=abcd_nk.B,
            a1=zeros(nstates(abcd_nk)), P1=Matrix{Float64}(I, nstates(abcd_nk), nstates(abcd_nk)),
        )
        prior_ssm = dsge_var_prior(ssm_nk, p, 1.5, T_eff)

        @test prior_sol isa DSGEVARPrior{Float64}
        k = 1 + 3 * p
        @test size(prior_sol.Gamma_yy) == (3, 3)
        @test size(prior_sol.Gamma_xy) == (k, 3)
        @test size(prior_sol.Gamma_xx) == (k, k)
        @test size(prior_sol.Phi_star) == (k, 3)
        @test size(prior_sol.Sigma_u_star) == (3, 3)
        @test size(prior_sol.Omega_star) == (3, 3)
        @test prior_sol.lambda ≈ 1.5
        @test prior_sol.lambda_min ≈ (3 + k + 1) / T_eff
        @test prior_sol.T_obs == T_eff
        @test prior_sol.p == p
        @test prior_sol.obs_names == ["y", "pi", "r"]
        @test prior_sol.shock_names == ["e_r", "e_d", "e_pi"]

        # Consistency across ABCD / ModelSpec / PerturbationSolution / StateSpaceModel
        for alt in (prior_abcd, prior_spec, prior_pert)
            @test alt.Phi_star ≈ prior_sol.Phi_star atol=1e-10
            @test alt.Sigma_u_star ≈ prior_sol.Sigma_u_star atol=1e-10
            @test alt.Omega_star ≈ prior_sol.Omega_star atol=1e-10
        end
        @test prior_ssm isa DSGEVARPrior{Float64}
        @test size(prior_ssm.Phi_star) == (k, 3)
        @test isposdef(Symmetric(prior_ssm.Sigma_u_star))

        # Symmetry & positive definiteness
        @test isposdef(Symmetric(prior_sol.Gamma_yy))
        @test isposdef(Symmetric(prior_sol.Gamma_xx))
        @test isposdef(Symmetric(prior_sol.Sigma_u_star))

        # Exact population Yule-Walker normal equations
        @test prior_sol.Gamma_xx * prior_sol.Phi_star ≈ prior_sol.Gamma_xy atol=1e-12
        @test prior_sol.Sigma_u_star ≈ prior_sol.Gamma_yy - transpose(prior_sol.Gamma_xy) * prior_sol.Phi_star atol=1e-12

        # QR structural rotation: D = L_star * Omega_star, Omega_star orthogonal, L_star lower tri with diag > 0
        @test prior_sol.Omega_star * transpose(prior_sol.Omega_star) ≈ Matrix{Float64}(I, 3, 3) atol=1e-12
        @test transpose(prior_sol.Omega_star) * prior_sol.Omega_star ≈ Matrix{Float64}(I, 3, 3) atol=1e-12
        L_star = abcd_nk.D * transpose(prior_sol.Omega_star)
        for i in 1:3, j in 1:3
            if j > i
                @test abs(L_star[i, j]) < 1e-12
            elseif i == j
                @test L_star[i, i] > 0
            end
        end
        @test L_star * prior_sol.Omega_star ≈ abcd_nk.D atol=1e-12
        @test L_star * transpose(L_star) ≈ abcd_nk.D * transpose(abcd_nk.D) atol=1e-12

        # Nonzero unconditional mean + exact construct_var_matrices alignment check
        A_nz = [0.65 0.10; -0.05 0.55]
        B_nz = [0.5 0.0; 0.1 0.4]
        C_nz = [1.0 0.0; 0.2 1.0]
        D_nz = [0.5 0.0; 0.1 0.4]
        c_nz = [0.4, -0.2]
        d_nz = [1.5, -0.5]
        abcd_nz = ABCDSystem{Float64}(A_nz, B_nz, C_nz, D_nz, c_nz, d_nz,
                                      ["s1", "s2"], ["y1", "y2"], ["e1", "e2"])
        T_long = 250_000
        rng = Random.Xoshiro(77)
        shocks_nz = randn(rng, T_long, 2)
        states_nz = zeros(T_long, 2)
        y_nz = zeros(T_long, 2)
        for t in 1:T_long
            s_prev = t == 1 ? zeros(2) : states_nz[t - 1, :]
            s_curr = A_nz * s_prev + B_nz * shocks_nz[t, :] + c_nz
            states_nz[t, :] = s_curr
            y_nz[t, :] = C_nz * s_prev + D_nz * shocks_nz[t, :] + d_nz
        end
        Y_sample = y_nz[501:end, :]
        Y_eff_s, X_s = construct_var_matrices(Y_sample, 2)
        T_s = size(Y_eff_s, 1)

        prior_nz = dsge_var_prior(abcd_nz, 2, 1.0, 200; include_intercept=true)
        mu_s = (I - A_nz) \ c_nz
        mu_y = C_nz * mu_s + d_nz
        @test prior_nz.Gamma_xx[1, 1] == 1.0
        @test prior_nz.Gamma_xx[1, 2:3] ≈ mu_y atol=1e-12
        @test prior_nz.Gamma_xx[1, 4:5] ≈ mu_y atol=1e-12
        @test prior_nz.Gamma_xy[1, :] ≈ mu_y atol=1e-12
        @test (transpose(Y_eff_s) * Y_eff_s) / T_s ≈ prior_nz.Gamma_yy rtol=0.02 atol=0.02
        @test (transpose(X_s) * Y_eff_s) / T_s ≈ prior_nz.Gamma_xy rtol=0.02 atol=0.02
        @test (transpose(X_s) * X_s) / T_s ≈ prior_nz.Gamma_xx rtol=0.02 atol=0.02

        # No-intercept prior
        prior_noint = dsge_var_prior(sol_nk, p, 1.5, T_eff; observables=obs_nk, include_intercept=false)
        @test size(prior_noint.Gamma_xx) == (3 * p, 3 * p)
        @test size(prior_noint.Phi_star) == (3 * p, 3)

        # Rectangular n > m (4 observables from 3 shocks -> measurement jitter + me_aux shock)
        prior_tall = dsge_var_prior(sol_nk, p, 2.0, T_eff;
                                    observables=[:y, :pi, :r, :xd], measurement_jitter=1e-6)
        @test size(prior_tall.Sigma_u_star) == (4, 4)
        @test isposdef(Symmetric(prior_tall.Sigma_u_star))
        @test size(prior_tall.Omega_star) == (4, 4)
        @test prior_tall.Omega_star * transpose(prior_tall.Omega_star) ≈ Matrix{Float64}(I, 4, 4) atol=1e-12
        @test length(prior_tall.shock_names) == 4
        @test prior_tall.shock_names[4] == "me_aux_1"

        # Wide m > n (2 observables from 3 shocks)
        prior_wide = dsge_var_prior(sol_nk, p, 1.5, T_eff; observables=[:y, :pi])
        @test size(prior_wide.Omega_star) == (2, 3)
        @test prior_wide.Omega_star * transpose(prior_wide.Omega_star) ≈ Matrix{Float64}(I, 2, 2) atol=1e-12
        @test length(prior_wide.shock_names) == 3

        # lambda = Inf allowed even though finite minimum is positive
        prior_inf = dsge_var_prior(sol_nk, p, Inf, T_eff; observables=obs_nk)
        @test isinf(prior_inf.lambda)

        # Validation errors
        lmin = (3 + (1 + 3 * p) + 1) / T_eff
        @test_throws ArgumentError dsge_var_prior(sol_nk, p, 0.5 * lmin, T_eff; observables=obs_nk)
        @test_throws ArgumentError dsge_var_prior(sol_nk, 0, 1.5, T_eff; observables=obs_nk)
        @test_throws ArgumentError dsge_var_prior(sol_nk, p, 1.5, 0; observables=obs_nk)
        @test_throws ArgumentError dsge_var_prior(sol_nk, p, -1.0, T_eff; observables=obs_nk)
        @test_throws ArgumentError dsge_var_prior(sol_nk, p, NaN, T_eff; observables=obs_nk)
        @test_throws ArgumentError dsge_var_prior(sol_nk, p, 1.5, T_eff; observables=obs_nk, measurement_jitter=-1e-5)
    end

    # =========================================================================
    # 2. Dummy-Observation Parity & Closed-Form Log Marginal Likelihood
    # =========================================================================
    @testset "Dummy-Observation Parity & dsge_var_logml" begin
        for inc_int in (true, false)
            lam = 1.75
            prior = dsge_var_prior(sol_nk, p, lam, T_eff; observables=obs_nk, include_intercept=inc_int)
            n = 3
            k = size(prior.Gamma_xx, 1)
            T_d = lam * T_eff

            # Construct explicit synthetic dummy observations (Y_d, X_d) whose outer-product
            # moments match T_d * (Gamma_yy, Gamma_xy, Gamma_xx)
            M_d = T_d * [prior.Gamma_yy transpose(prior.Gamma_xy);
                         prior.Gamma_xy prior.Gamma_xx]
            M_d = (M_d + transpose(M_d)) / 2
            F = eigen(Hermitian(M_d))
            vals_pos = max.(F.values, 0.0)
            Z_d = Diagonal(sqrt.(vals_pos)) * transpose(F.vectors) # (n+k) x (n+k)
            Y_d = Matrix{Float64}(Z_d[:, 1:n])
            X_d = Matrix{Float64}(Z_d[:, (n + 1):end])

            @test transpose(Y_d) * Y_d ≈ T_d * prior.Gamma_yy rtol=1e-11 atol=1e-12
            @test transpose(X_d) * Y_d ≈ T_d * prior.Gamma_xy rtol=1e-11 atol=1e-12
            @test transpose(X_d) * X_d ≈ T_d * prior.Gamma_xx rtol=1e-11 atol=1e-12

            # Direct dummy-observation NIW marginal likelihood formula
            Y_raw_eff, X_raw_full = construct_var_matrices(Y_true, p)
            X_raw = inc_int ? X_raw_full : X_raw_full[:, 2:end]

            Y_tot = vcat(Y_d, Y_raw_eff)
            X_tot = vcat(X_d, X_raw)

            K_0 = transpose(X_d) * X_d
            B_0 = K_0 \ (transpose(X_d) * Y_d)
            S_0 = transpose(Y_d) * Y_d - transpose(B_0) * K_0 * B_0
            nu_0 = T_d - k

            K_1 = transpose(X_tot) * X_tot
            B_1 = K_1 \ (transpose(X_tot) * Y_tot)
            S_1 = transpose(Y_tot) * Y_tot - transpose(B_1) * K_1 * B_1
            nu_1 = T_eff + nu_0

            _logmvgamma(n_dim, a) = (n_dim * (n_dim - 1) / 4) * log(pi) +
                sum(SpecialFunctions.loggamma(a + (1 - j) / 2) for j in 1:n_dim)

            lml_dummy = (n / 2) * (logdet(K_0) - logdet(K_1)) -
                        (nu_1 / 2) * logdet(S_1) + (nu_0 / 2) * logdet(S_0) -
                        (T_eff * n / 2) * log(pi) +
                        _logmvgamma(n, nu_1 / 2) - _logmvgamma(n, nu_0 / 2)

            lml_api_prior = dsge_var_logml(prior, Y_true)
            lml_api_sol = dsge_var_logml(sol_nk, Y_true, p, lam;
                                         observables=obs_nk, include_intercept=inc_int)

            @test abs(lml_api_prior - lml_dummy) < 1e-10
            @test abs(lml_api_sol - lml_dummy) < 1e-10
        end

        # Asymptotic convergence of finite-lambda NIW marginal likelihood -> Gaussian loglik at lambda = Inf
        lml_large = dsge_var_logml(sol_nk, Y_true, p, 1e6; observables=obs_nk)
        lml_inf = dsge_var_logml(sol_nk, Y_true, p, Inf; observables=obs_nk)
        @test isfinite(lml_large) && isfinite(lml_inf)
        @test abs(lml_large - lml_inf) < 1e-3
    end

    # =========================================================================
    # 3. Misspecification Curve on 3-Equation NK DSGE (T = 250)
    # =========================================================================
    @testset "Misspecification Curve & select_dsge_var_lambda" begin
        prof_true = select_dsge_var_lambda(sol_nk, Y_true, p; observables=obs_nk)
        @test prof_true isa DSGEVARLambdaProfile{Float64}
        @test prof_true.p == p
        @test prof_true.T_obs == T_eff
        @test prof_true.obs_names == ["y", "pi", "r"]
        @test prof_true.lambda_hat >= 2.0
        @test prof_true.logml_hat >= maximum(prof_true.log_marginal_likelihoods) - 1e-12

        # Distorted DGP with strong omitted ad-hoc lag dynamics + cross-correlations
        rng = Random.Xoshiro(999)
        T_burn = 150
        Y_mis_all = zeros(T_burn + 250, 3)
        A1_bad = [0.25 -0.40 0.30; 0.35 0.15 -0.25; -0.20 0.30 0.20]
        A2_bad = [-0.45 0.25 0.0; -0.25 -0.35 0.20; 0.30 -0.20 -0.35]
        S_bad = [0.02, 0.015, 0.02]
        for t in 3:(T_burn + 250)
            eps_mis = randn(rng, 3)
            Y_mis_all[t, :] = A1_bad * Y_mis_all[t - 1, :] +
                              A2_bad * Y_mis_all[t - 2, :] +
                              S_bad .* eps_mis
        end
        Y_mis = Y_mis_all[(T_burn + 1):end, :]

        prof_mis = select_dsge_var_lambda(sol_nk, Y_mis, p; observables=obs_nk)
        @test isfinite(prof_mis.lambda_hat)
        @test prof_mis.lambda_min <= prof_mis.lambda_hat < 1.0
        @test prof_mis.logml_hat > prof_mis.logml_inf

        # Grid-only vs Brent-refined comparison and custom lambda_grid
        prof_noref = select_dsge_var_lambda(sol_nk, Y_mis, p; observables=obs_nk, refine=false)
        @test prof_mis.logml_hat >= prof_noref.logml_hat - 1e-12

        prof_inf_only = select_dsge_var_lambda(sol_nk, Y_true, p; observables=obs_nk, lambda_grid=[Inf])
        @test isinf(prof_inf_only.lambda_hat)
        @test isempty(prof_inf_only.lambdas)

        @test_throws ArgumentError select_dsge_var_lambda(sol_nk, Y_true, p; observables=obs_nk, lambda_grid=Float64[])
        @test_throws ArgumentError select_dsge_var_lambda(sol_nk, Y_true, p; observables=obs_nk, lambda_grid=[-0.1, 1.0])
        @test_throws ArgumentError select_dsge_var_lambda(sol_nk, Y_true, p; observables=obs_nk, lambda_grid=[1e-4])
    end

    # =========================================================================
    # 4. Dogmatic Limit & ABCD-04 Population IRF Convergence
    # =========================================================================
    @testset "Dogmatic Limit & ABCD-04 IRF Convergence" begin
        post_dog = estimate_dsge_var(sol_nk, Y_true, p;
                                     lambda=1e4, observables=obs_nk, n_draws=1000, seed=42)
        @test post_dog isa DSGEVARPosterior{Float64}
        @test post_dog.n_draws == 1000
        @test post_dog.theta_draws === nothing
        @test post_dog.log_posterior === nothing
        @test isempty(post_dog.param_names)
        @test post_dog.acceptance_rate == 1.0
        @test has_uncertainty(post_dog) == true

        B_mean = dropdims(mean(post_dog.B_draws; dims=1); dims=1)
        S_mean = dropdims(mean(post_dog.Sigma_draws; dims=1); dims=1)
        @test maximum(abs, B_mean .- post_dog.dsge_prior.Phi_star) < 1e-3
        @test maximum(abs, S_mean .- post_dog.dsge_prior.Sigma_u_star) < 1e-5

        pop_var = population_var(sol_nk, p; observables=obs_nk)
        pop_irf = compute_irf(pop_var, Matrix{Float64}(post_dog.dsge_prior.Omega_star), 12)
        irf_dog = irf(post_dog, 12; method=:dsge_qr)
        @test size(irf_dog.point_estimate) == (12, 3, 3)
        @test maximum(abs, irf_dog.point_estimate .- pop_irf) < 1e-3

        # Exact lambda = Inf dogmatic limit
        post_inf = estimate_dsge_var(sol_nk, Y_true, p;
                                     lambda=Inf, observables=obs_nk, n_draws=50, seed=42)
        @test isinf(post_inf.lambda)
        @test has_uncertainty(post_inf) == false
        @test uncertainty_bounds(post_inf) === nothing
        @test maximum(abs, post_inf.B_draws[1, :, :] .- post_inf.dsge_prior.Phi_star) < 1e-12
        @test maximum(abs, post_inf.Sigma_draws[1, :, :] .- post_inf.dsge_prior.Sigma_u_star) < 1e-12
        irf_inf = irf(post_inf, 12; method=:dsge_qr)
        @test maximum(abs, irf_inf.point_estimate .- pop_irf) < 1e-8

        # Auto-lambda + seed reproducibility + reproduce()
        post_auto1 = estimate_dsge_var(sol_nk, Y_true, p; lambda=:auto, observables=obs_nk, n_draws=80, seed=101)
        post_auto2 = estimate_dsge_var(sol_nk, Y_true, p; lambda=:auto, observables=obs_nk, n_draws=80, seed=101)
        @test post_auto1.B_draws == post_auto2.B_draws
        @test post_auto1.Sigma_draws == post_auto2.Sigma_draws
        @test post_auto1.B0_draws == post_auto2.B0_draws
        rep_fixed = reproduce(post_auto1)
        @test rep_fixed.matched === true

        # No-intercept posterior + rectangular n > m
        post_noint = estimate_dsge_var(sol_nk, Y_true, p;
                                       lambda=1.5, observables=obs_nk,
                                       include_intercept=false, n_draws=40, seed=7)
        @test size(post_noint.B_draws) == (40, 3 * p, 3)
        @test size(irf(post_noint, 8).point_estimate) == (8, 3, 3)

        Y_4obs = simulate(sol_nk, 180; seed=55)
        post_rect = estimate_dsge_var(sol_nk, Y_4obs, 1;
                                      lambda=2.0, observables=[:y, :pi, :r, :xd],
                                      n_draws=40, seed=7)
        @test size(post_rect.B0_draws) == (40, 4, 4)
        @test size(irf(post_rect, 6).point_estimate) == (6, 4, 4)
    end

    # =========================================================================
    # 5. Joint (θ, Φ, Σ_u) Hierarchical Bayes (:rwmh and :smc)
    # =========================================================================
    @testset "Joint (θ, Φ, Σ_u) Estimation (:rwmh & :smc)" begin
        priors_dict = Dict{Symbol,Distribution}(
            :psi1 => Distributions.Uniform(1.15, 2.50),
            :rho_r => Distributions.Beta(4.0, 4.0),
        )
        prior_dsge = MacroEconometricModels.DSGEPrior(priors_dict)

        post_rwmh = estimate_dsge_var(nk_spec, Y_true, p;
                                      priors=priors_dict, lambda=1.5, method=:rwmh,
                                      observables=obs_nk, n_draws=60, burnin=30,
                                      step_scale=0.05, seed=321)
        post_rwmh_2 = estimate_dsge_var(nk_spec, Y_true, p;
                                        priors=priors_dict, lambda=1.5, method=:rwmh,
                                        observables=obs_nk, n_draws=60, burnin=30,
                                        step_scale=0.05, seed=321)

        @test post_rwmh isa DSGEVARPosterior{Float64}
        @test post_rwmh.method === :rwmh
        @test post_rwmh.theta_draws !== nothing
        @test size(post_rwmh.theta_draws) == (60, 2)
        @test Set(post_rwmh.param_names) == Set([:psi1, :rho_r])
        @test post_rwmh.log_posterior !== nothing && length(post_rwmh.log_posterior) == 60
        @test all(isfinite, post_rwmh.log_posterior)
        @test 0 <= post_rwmh.acceptance_rate <= 1
        @test isfinite(post_rwmh.log_marginal_likelihood)
        @test post_rwmh.theta_draws == post_rwmh_2.theta_draws
        @test post_rwmh.B_draws == post_rwmh_2.B_draws
        @test post_rwmh.B0_draws == post_rwmh_2.B0_draws
        @test reproduce(post_rwmh).matched === missing

        # Prior covering a small indeterminacy corner tests benign solve -Inf handling
        priors_indet = Dict{Symbol,Distribution}(
            :psi1 => Distributions.Uniform(0.85, 2.20),
            :rho_r => Distributions.Uniform(0.10, 0.85),
        )
        post_indet = estimate_dsge_var(nk_spec, Y_true, p;
                                       priors=priors_indet, lambda=1.5, method=:rwmh,
                                       observables=obs_nk, n_draws=40, burnin=20,
                                       step_scale=0.10, seed=88)
        @test all(isfinite, post_indet.log_posterior)

        # :smc mode with DSGEPrior and lambda=:auto
        post_smc = estimate_dsge_var(nk_spec, Y_true, p;
                                     priors=prior_dsge, lambda=:auto, method=:smc,
                                     observables=obs_nk, n_draws=50, n_particles=50,
                                     seed=654)
        @test post_smc isa DSGEVARPosterior{Float64}
        @test post_smc.method === :smc
        @test size(post_smc.theta_draws) == (50, 2)
        @test all(isfinite, post_smc.log_posterior)
        @test isfinite(post_smc.log_marginal_likelihood)
        @test size(irf(post_smc, 8).point_estimate) == (8, 3, 3)

        # Argument validation in joint mode
        @test_throws ArgumentError estimate_dsge_var(nk_spec, Y_true, p;
                                                     priors=priors_dict, method=:bad_method)
        @test_throws ArgumentError estimate_dsge_var(nk_spec, Y_true, p;
                                                     priors=Dict(:nonexistent_param => Normal(0, 1)))
        @test_throws ArgumentError estimate_dsge_var(nk_spec, Y_true, p;
                                                     priors=Dict{Symbol,Distribution}())
    end

    # =========================================================================
    # 6. Ecosystem Delegation (irf, fevd, historical_decomposition, forecast)
    # =========================================================================
    @testset "Ecosystem Delegation & Analysis API" begin
        post = estimate_dsge_var(sol_nk, Y_true, p;
                                 lambda=1.5, observables=obs_nk, n_draws=60, seed=11)

        @test size(post, 1) == 60
        @test length(post) == 60
        @test varnames(post) == ["y", "pi", "r"]
        @test marginal_likelihood(post) == post.log_marginal_likelihood
        pe = point_estimate(post)
        @test size(pe) == (1 + 3 * p, 3)
        @test has_uncertainty(post) == true
        lo, hi = uncertainty_bounds(post)
        @test all(lo .<= pe) && all(pe .<= hi)

        m_mean = posterior_mean_model(post)
        m_med = posterior_median_model(post)
        @test m_mean isa VARModel{Float64}
        @test m_med isa VARModel{Float64}
        @test m_mean.p == p

        # IRF: :dsge_qr, normalize=:unit_effect, :cholesky, :sign
        irf_qr = irf(post, 10; method=:dsge_qr)
        irf_unit = irf(post, 10; method=:dsge_qr, normalize=:unit_effect)
        irf_chol = irf(post, 10; method=:cholesky)
        irf_sign = irf(post, 10; method=:sign, check_func=ir_arr -> true)
        @test irf_qr isa BayesianImpulseResponse{Float64}
        @test irf_unit isa BayesianImpulseResponse{Float64}
        @test irf_chol isa BayesianImpulseResponse{Float64}
        @test irf_sign isa BayesianImpulseResponse{Float64}
        @test size(irf_qr.point_estimate) == (10, 3, 3)
        @test irf_qr.shocks == ["e_r", "e_d", "e_pi"]

        # FEVD: :dsge_qr and :cholesky (proportions sum to 1 across shocks)
        fevd_qr = fevd(post, 8; method=:dsge_qr)
        fevd_chol = fevd(post, 8; method=:cholesky)
        @test fevd_qr isa BayesianFEVD{Float64}
        @test fevd_chol isa BayesianFEVD{Float64}
        @test size(fevd_qr.point_estimate) == (3, 3, 8)
        for h in 1:8, i in 1:3
            @test sum(fevd_qr.point_estimate[i, :, h]) ≈ 1.0 atol=1e-6
        end

        # Historical decomposition: :dsge_qr and :cholesky
        hd_qr = historical_decomposition(post; method=:dsge_qr)
        hd_chol = historical_decomposition(post; method=:cholesky)
        @test hd_qr isa BayesianHistoricalDecomposition{Float64}
        @test hd_chol isa BayesianHistoricalDecomposition{Float64}
        @test size(hd_qr.point_estimate) == (T_eff, 3, 3)
        recon_hd = dropdims(sum(hd_qr.point_estimate; dims=3); dims=3) .+ hd_qr.initial_point_estimate
        @test recon_hd ≈ hd_qr.actual atol=1e-4

        # Wide rectangular system (m != n) rejects method=:dsge_qr in historical_decomposition
        post_wide = estimate_dsge_var(sol_nk, Y_true[:, 1:2], p;
                                      lambda=1.5, observables=[:y, :pi], n_draws=20, seed=12)
        @test_throws ArgumentError historical_decomposition(post_wide; method=:dsge_qr)
        @test historical_decomposition(post_wide; method=:cholesky) isa BayesianHistoricalDecomposition{Float64}

        # Forecast
        fc = forecast(post, 6; seed=23)
        @test fc isa BVARForecast{Float64}
        @test size(fc.forecast) == (6, 3)
        @test fc.horizon == 6
    end

    # =========================================================================
    # 7. Float32 Preservation, RSER-01 Roundtrip, show/report & Ambiguities
    # =========================================================================
    @testset "Float32, RSER-01 Serialization, show/report & Ambiguities" begin
        abcd32 = ABCDSystem{Float32}(
            Float32.(abcd_nk.A), Float32.(abcd_nk.B),
            Float32.(abcd_nk.C), Float32.(abcd_nk.D),
            Float32.(abcd_nk.c_state), Float32.(abcd_nk.d_obs),
            abcd_nk.state_names, abcd_nk.obs_names, abcd_nk.shock_names,
        )
        Y32 = Float32.(Y_true)

        prior32 = dsge_var_prior(abcd32, p, 1.5f0, T_eff)
        @test prior32 isa DSGEVARPrior{Float32}
        @test dsge_var_logml(prior32, Y32) isa Float32

        prof32 = select_dsge_var_lambda(abcd32, Y32, p)
        @test prof32 isa DSGEVARLambdaProfile{Float32}

        post32 = estimate_dsge_var(abcd32, Y32, p; lambda=1.5f0, n_draws=30, seed=9)
        @test post32 isa DSGEVARPosterior{Float32}
        @test irf(post32, 6) isa BayesianImpulseResponse{Float32}
        @test fevd(post32, 6) isa BayesianFEVD{Float32}
        @test forecast(post32, 4; seed=9) isa BVARForecast{Float32}

        # Joint Float32 via ModelSpec + Float32 Y
        priors_j = Dict{Symbol,Distribution}(:psi1 => Distributions.Uniform(1.2, 2.3))
        post_joint = estimate_dsge_var(nk_spec, Y_true, p;
                                       priors=priors_j, lambda=1.5, method=:rwmh,
                                       observables=obs_nk, n_draws=25, burnin=15, seed=71)

        # RSER-01: _generic_construct + _from_serializable + _assert_plain_payload on all 3 structs
        function _assert_rser_roundtrip(obj)
            T_obj = typeof(obj)
            fnames = fieldnames(T_obj)
            args = Any[getfield(obj, f) for f in fnames]
            obj_gc = MacroEconometricModels._generic_construct(T_obj, args)
            @test typeof(obj_gc) === T_obj

            payload = Dict{String,Any}(
                string(f) => MacroEconometricModels._ser_field(getfield(obj, f)) for f in fnames
            )
            MacroEconometricModels._assert_plain_payload(payload)
            obj_re = MacroEconometricModels._from_serializable(
                T_obj, payload, MacroEconometricModels.SERIALIZATION_FORMAT_VERSION)
            @test typeof(obj_re) === T_obj

            for f in fnames
                v_orig = getfield(obj, f)
                v_new = getfield(obj_re, f)
                if v_orig isa AbstractArray{<:AbstractFloat}
                    @test v_new ≈ v_orig
                elseif v_orig isa DSGEVARPrior || v_orig isa MacroEconometricModels.ReproManifest
                    for sf in fieldnames(typeof(v_orig))
                        sv_o = getfield(v_orig, sf)
                        sv_n = getfield(v_new, sf)
                        if sv_o isa AbstractArray{<:AbstractFloat}
                            @test sv_n ≈ sv_o
                        else
                            @test sv_n == sv_o
                        end
                    end
                else
                    @test v_new == v_orig
                end
            end
        end

        _assert_rser_roundtrip(prior32)
        _assert_rser_roundtrip(prof32)
        _assert_rser_roundtrip(post32)
        _assert_rser_roundtrip(post_joint)
        _assert_rser_roundtrip(dsge_var_prior(sol_nk, p, Inf, T_eff; observables=obs_nk))
        _assert_rser_roundtrip(select_dsge_var_lambda(sol_nk, Y_true, p; observables=obs_nk))

        # show and report across text and LaTeX backends
        prior64 = dsge_var_prior(sol_nk, p, 1.5, T_eff; observables=obs_nk)
        prof64 = select_dsge_var_lambda(sol_nk, Y_true, p; observables=obs_nk)
        for obj in (prior64, prof64, post_joint)
            s_text = sprint(show, obj)
            @test !isempty(s_text)
            r_text = sprint(report, obj)
            @test !isempty(r_text)
            r_latex = with_display_backend(:latex) do
                sprint(report, obj)
            end
            @test !isempty(r_latex)
            redirect_stdout(devnull) do
                report(obj)
            end
        end
    end

    # =========================================================================
    # 8. An-Schorfheide / Smets-Wouters Style DSGE-VAR IRF Recovery & Plots
    # =========================================================================
    @testset "An-Schorfheide / Smets-Wouters Style DSGE-VAR Recovery & Plots" begin
        # Pre-linearized AR(1)+Phillips+Taylor NK model (An-Schorfheide 2007 style)
        as_spec = @dsge begin
            parameters: beta = 0.99, tau = 1.0, kappa = 0.35, psi1 = 1.65,
                        rho_r = 0.65, rho_g = 0.75, rho_z = 0.70,
                        sig_r = 0.01, sig_g = 0.01, sig_z = 0.01
            endogenous: y, pi, r, g, z
            exogenous: e_r, e_g, e_z
            steady_state: [0.0, 0.0, 0.0, 0.0, 0.0]
            linear: true

            y[t] = y[t+1] - (1 / tau) * (r[t] - pi[t+1]) + (1 - rho_g) * g[t] + rho_z * z[t]
            pi[t] = beta * pi[t+1] + kappa * (y[t] - g[t])
            r[t] = rho_r * r[t-1] + (1 - rho_r) * (psi1 * pi[t]) + sig_r * e_r[t]
            g[t] = rho_g * g[t-1] + sig_g * e_g[t]
            z[t] = rho_z * z[t-1] + sig_z * e_z[t]
        end
        sol_as = solve(as_spec; method=:gensys)
        @test is_determined(sol_as)

        as_obs = [:y, :pi, :r]
        sim_as = simulate(sol_as, 500; seed=2007)
        as_idx = [findfirst(==(s), sol_as.spec.endog) for s in as_obs]
        Y_as = Matrix{Float64}(sim_as[201:end, as_idx])

        # True-DGP lambda selection picks high weight >= 2.0 and finite profile
        prof_as = select_dsge_var_lambda(sol_as, Y_as, 2; observables=as_obs)
        @test prof_as.lambda_hat >= 2.0
        @test prof_as.logml_hat >= prof_as.logml_inf - 1e-10

        # Posterior IRF bands cover true DSGE-ABCD IRF across horizon
        post_as = estimate_dsge_var(sol_as, Y_as, 2;
                                    lambda=:auto, observables=as_obs,
                                    n_draws=200, seed=2007)
        irf_post_as = irf(post_as, 12; method=:dsge_qr, quantiles=[0.025, 0.5, 0.975])
        irf_true_as = irf(abcd_representation(sol_as, as_obs), 12)
        covered = count(
            irf_post_as.quantiles[h, i, j, 1] - 1e-3 <= irf_true_as.values[h, i, j] <=
            irf_post_as.quantiles[h, i, j, 3] + 1e-3
            for h in 1:12, i in 1:3, j in 1:3
        )
        @test covered >= 90  # >= ~83% of 108 (h,i,j) cells inside 95% DSGE-VAR bands

        # Plot integration across prior, lambda profile, and posterior views
        prior_as = dsge_var_prior(sol_as, 2, 1.5, size(Y_as, 1) - 2; observables=as_obs)
        @test plot_result(prior_as; view=:phi) isa PlotOutput
        @test plot_result(prior_as; view=:sigma) isa PlotOutput
        @test plot_result(prof_as) isa PlotOutput
        @test plot_result(post_as; view=:irf, var="y", shock="e_r", horizon=12) isa PlotOutput
        @test plot_result(post_as; view=:draws, max_panels=4) isa PlotOutput
        @test plot_result(post_as; view=:lambda) isa PlotOutput
    end
end
