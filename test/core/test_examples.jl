# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using MacroEconometricModels
using Test
using Random
using LinearAlgebra
using Statistics
using StatsAPI

@testset "Documentation Examples" begin

    # =========================================================================
    # README Quick Start
    # =========================================================================
    @testset "README Quick Start" begin
        rng = MersenneTwister(42)  # DGP-02: explicit rng
        T, n = 200, 3
        A = [0.5 0.1 0.0; 0.0 0.6 0.1; 0.1 0.0 0.4]
        Y = dgp_var(rng; A=A, B0=Matrix{Float64}(I, n, n), T=T).Y  # DGP-02 #791

        # VAR estimation
        model = estimate_var(Y, 2)
        @test model isa VARModel

        # IRF with bootstrap CI
        irf_result = irf(model, 20; ci_type=:bootstrap, reps=50, rng=rng)
        @test irf_result isa ImpulseResponse

        # Local Projections
        lp_result = estimate_lp(Y, 1, 20; cov_type=:newey_west)
        @test lp_result isa LPModel
        lp_irf_result = lp_irf(lp_result)
        @test lp_irf_result isa LPImpulseResponse

        # Bayesian estimation (reduced draws for speed)
        post = estimate_bvar(Y, 2; prior=:minnesota, n_draws=50, rng=rng)
        @test post isa BVARPosterior
    end

    # =========================================================================
    # Example 1: Three-Variable VAR Analysis (docs/src/examples.md)
    # =========================================================================
    @testset "Three-Variable VAR Analysis" begin
        rng = MersenneTwister(42)  # DGP-02: explicit rng

        T = 200
        n = 3
        p = 2

        A_true = [0.85 0.10 -0.15;
                  0.05 0.70  0.00;
                  0.10 0.20  0.80]

        Σ_true = [1.00 0.50 0.20;
                  0.50 0.80 0.10;
                  0.20 0.10 0.60]

        # Shared DGP with the known (A_true, Σ_true) truth (DGP-02 #791; note
        # dgp_var consumes Sigma only with B0=nothing).
        Y = dgp_var(rng; A=A_true, B0=nothing, Sigma=Σ_true, T=T).Y

        # VAR estimation
        model = fit(VARModel, Y, p)
        @test model isa VARModel
        @test !isnan(loglikelihood(model))
        @test !isnan(aic(model))
        @test !isnan(bic(model))

        # Stability check
        F = companion_matrix(model.B, n, p)
        eigenvalues = eigvals(F)
        @test maximum(abs.(eigenvalues)) < 1  # Stable

        # Cholesky IRF
        H = 20
        irfs = irf(model, H; method=:cholesky)
        @test irfs isa ImpulseResponse
        @test size(irfs.values) == (H, n, n)

        # Known impact truth (DGP-02 #791): estimated impact recovers
        # chol(Σ_true) (max err 0.074 realized; bound 0.15 ≈ Σ̂ noise at T=200).
        Ctrue = cholesky(Symmetric(Σ_true)).L
        @test maximum(abs, tril(irfs.values[1, :, :] - Ctrue)) < 0.15

        # Sign restriction identification
        function check_demand_shock(irf_array)
            return irf_array[1, 1, 1] > 0 && irf_array[1, 2, 1] > 0
        end

        irfs_sign = irf(model, H; method=:sign, check_func=check_demand_shock)
        @test irfs_sign isa ImpulseResponse

        # FEVD
        fevd_result = fevd(model, H; method=:cholesky)
        @test fevd_result isa FEVD
        @test size(fevd_result.proportions) == (n, n, H)  # (variable, shock, horizon)
        # FEVD should sum to 1 for each variable at each horizon
        for h in 1:H
            for i in 1:n
                @test isapprox(sum(fevd_result.proportions[i, :, h]), 1.0, atol=1e-10)
            end
        end
    end

    # =========================================================================
    # Example 2: Bayesian VAR with Minnesota Prior (docs/src/examples.md)
    # =========================================================================
    @testset "Bayesian VAR with Minnesota Prior" begin
        rng = MersenneTwister(42)  # DGP-02: explicit rng

        T = 200
        n = 3
        p = 2

        A_true = [0.85 0.10 -0.15;
                  0.05 0.70  0.00;
                  0.10 0.20  0.80]

        Σ_true = [1.00 0.50 0.20;
                  0.50 0.80 0.10;
                  0.20 0.10 0.60]

        # Shared DGP with the known (A_true, Σ_true) truth (DGP-02 #791).
        Y = dgp_var(rng; A=A_true, B0=nothing, Sigma=Σ_true, T=T).Y

        # Hyperparameter optimization
        best_hyper = optimize_hyperparameters(Y, p; grid_size=5)
        @test best_hyper isa MinnesotaHyperparameters
        @test best_hyper.tau > 0

        # BVAR estimation (reduced draws for speed)
        post = estimate_bvar(Y, p;
            n_draws=50,
            prior=:minnesota,
            hyper=best_hyper,
            rng=rng
        )
        @test post isa BVARPosterior

        # Bayesian IRF with Cholesky
        H = 20
        birf_chol = irf(post, H; method=:cholesky)
        @test birf_chol isa BayesianImpulseResponse
        @test size(birf_chol.quantiles) == (H, n, n, 3)  # 16%, 50%, 84%

        # Posterior-median impact recovers chol(Σ_true) (DGP-02 #791; max err
        # 0.078 realized, bound 0.2 for 50-draw posterior noise).
        Ctrue = cholesky(Symmetric(Σ_true)).L
        med = birf_chol.quantiles[:, :, :, 2]
        @test maximum(abs, tril(med[1, :, :] - Ctrue)) < 0.2
    end

    # =========================================================================
    # Example 3: Local Projections (docs/src/examples.md)
    # =========================================================================
    @testset "Local Projections" begin
        rng = MersenneTwister(42)  # DGP-02: explicit rng

        T = 200
        n = 3

        A1 = [0.5 0.1 -0.1; 0.05 0.6 0.0; 0.1 0.15 0.7]
        A2 = [0.2 0.05 0.0; 0.0 0.2 0.0; 0.05 0.05 0.1]

        # Known VAR(2) truth (DGP-02 #791).
        Y = dgp_var(rng; A=[A1, A2], B0=0.5 * Matrix{Float64}(I, n, n), T=T).Y

        H = 20
        shock_var = 1

        # Standard LP with Newey-West
        lp_model = estimate_lp(Y, shock_var, H; lags=4, cov_type=:newey_west)
        @test lp_model isa LPModel

        lp_result = lp_irf(lp_model; conf_level=0.95)
        @test lp_result isa LPImpulseResponse
        @test size(lp_result.values, 1) == H + 1

        # Cumulative IRF
        cum_irf = cumulative_irf(lp_result)
        @test cum_irf isa LPImpulseResponse

        # Compare VAR vs LP
        var_model = estimate_var(Y, 2)
        var_irf_result = irf(var_model, H; method=:cholesky)
        @test var_irf_result isa ImpulseResponse
    end

    # =========================================================================
    # Example 3b: LP with Instrumental Variables (docs/src/examples.md)
    # =========================================================================
    @testset "LP with Instrumental Variables" begin
        rng = MersenneTwister(123)  # DGP-02: explicit rng

        T = 200

        A1 = [0.5 0.1 -0.1; 0.05 0.6 0.0; 0.1 0.15 0.7]
        B0 = [0.5 0.0 0.0; 0.1 0.5 0.0; 0.05 0.1 0.5]
        sim = dgp_var(rng; A=A1, B0=B0, T=T)  # DGP-02 #791: truth + shocks
        Y = sim.Y

        # VALID instrument (DGP-02 #791): the structural shock to var 3 plus
        # noise — relevant by construction, exogenous to all other shocks. The
        # old Z = 0.5·Y[:,3] + noise loads on lagged shocks of every variable,
        # violating exclusion by construction.
        Z = sim.eps[:, 3:3] + 0.5 * randn(rng, T, 1)

        H = 10
        shock_var = 3

        lpiv_model = estimate_lp_iv(Y, shock_var, Z, H; lags=2, cov_type=:newey_west)
        @test lpiv_model isa LPIVModel

        # Strong valid instrument: first-stage F clears 10 at every horizon
        # (realized min 758 — relevance is structural, not luck).
        weak_test = weak_instrument_test(lpiv_model; threshold=10.0)
        @test hasfield(typeof(weak_test), :F_stats)
        @test hasfield(typeof(weak_test), :passes_threshold)
        @test weak_test.passes_threshold
        @test weak_test.min_F > 10.0

        # LP-IV IRF
        lpiv_result = lp_iv_irf(lpiv_model)
        @test lpiv_result isa LPImpulseResponse
    end

    # =========================================================================
    # Example 3c: Smooth Local Projection (docs/src/examples.md)
    # =========================================================================
    @testset "Smooth Local Projection" begin
        rng = MersenneTwister(42)  # DGP-02: explicit rng

        T = 200
        n = 3
        # Persistent VAR(1) truth (DGP-02 #791).
        Y = dgp_var(rng; A=0.5 * Matrix{Float64}(I, n, n),
                    B0=Matrix{Float64}(I, n, n), T=T).Y

        H = 15

        # Smooth LP with B-splines
        smooth_model = estimate_smooth_lp(Y, 1, H;
            degree=3,
            n_knots=4,
            lambda=1.0,
            lags=2
        )
        @test smooth_model isa SmoothLPModel

        # Cross-validate lambda
        optimal_lambda = cross_validate_lambda(Y, 1, H;
            lambda_grid=[0.01, 0.1, 1.0, 10.0],
            k_folds=3
        )
        @test optimal_lambda > 0

        # Compare smooth vs standard LP
        comparison = compare_smooth_lp(Y, 1, H; lambda=optimal_lambda)
        @test hasfield(typeof(comparison), :variance_reduction)
    end

    # =========================================================================
    # Example 3d: State-Dependent Local Projection (docs/src/examples.md)
    # =========================================================================
    @testset "State-Dependent Local Projection" begin
        rng = MersenneTwister(42)  # DGP-02: explicit rng

        # Regime-dependent truth (DGP-02 #791): expansion dynamics are far more
        # persistent than recession dynamics. The old test ran on a DGP with NO
        # regime dependence, so any regime split it found was pure noise.
        sim = dgp_state_dependent_var(rng; T=2000)
        Y, state_var = sim.Y, vec(sim.z)

        H = 10

        state_model = estimate_state_lp(Y, 1, state_var, H;
            gamma=1.5,
            threshold=:median,
            lags=2
        )
        @test state_model isa StateLPModel

        # Regime-specific IRFs
        irf_both = state_irf(state_model; regime=:both)
        @test hasfield(typeof(irf_both), :expansion)
        @test hasfield(typeof(irf_both), :recession)

        # Known direction: expansion responses cumulate above recession ones
        # (realized 5.16 vs 2.01 — persistence differs by construction).
        @test sum(irf_both.expansion.values) > sum(irf_both.recession.values)

        # Test for regime differences
        diff_test = test_regime_difference(state_model)
        @test hasfield(typeof(diff_test), :joint_test)
    end

    # =========================================================================
    # Example 4: Factor Model (docs/src/examples.md, examples/factor_model_example.jl)
    # =========================================================================
    @testset "Factor Model" begin
        rng = MersenneTwister(42)  # DGP-02: explicit rng

        T = 200
        N = 30
        r_true = 5

        # 5 persistent factors with calibrated signal share (DGP-02 #791).
        A5 = [0.8 0.1 0.0 0.0 0.0; 0.0 0.7 0.1 0.0 0.0; 0.0 0.0 0.75 0.05 0.0;
              0.0 0.0 0.0 0.65 0.1; 0.05 0.0 0.0 0.0 0.7]
        X = dgp_dynamic_factors(rng; A=A5, N=N, T=T).X

        # Estimate factor model
        model = estimate_factors(X, r_true)
        @test model isa FactorModel
        @test model.r == r_true
        @test size(model.factors) == (T, r_true)
        @test size(model.loadings) == (N, r_true)

        # Variance explained
        @test length(model.explained_variance) >= r_true
        @test all(model.explained_variance .>= 0)
        @test model.cumulative_variance[r_true] <= 1.0

        # R² for each variable
        r2_values = r2(model)
        @test length(r2_values) == N
        @test all(r2_values .>= 0)
        @test all(r2_values .<= 1)

        # Fitted values and residuals
        X_fitted = predict(model)
        @test size(X_fitted) == (T, N)
        resid = residuals(model)
        @test size(resid) == (T, N)

        # Model info
        @test nobs(model) == T
        @test dof(model) > 0

        # Information criteria
        max_factors = 10
        ic = ic_criteria(X, max_factors)
        @test ic.r_IC1 >= 1
        @test ic.r_IC2 >= 1
        @test ic.r_IC3 >= 1
        @test length(ic.IC1) == max_factors

        # Scree plot data
        scree_data = scree_plot_data(model)
        @test length(scree_data.factors) >= r_true
        @test length(scree_data.explained_variance) >= r_true
        @test length(scree_data.cumulative_variance) >= r_true

        # Standardized vs non-standardized
        model_std = estimate_factors(X, r_true; standardize=true)
        model_nostd = estimate_factors(X, r_true; standardize=false)
        @test model_std.standardized == true
        @test model_nostd.standardized == false
    end

    # =========================================================================
    # Example 4b: Realistic Macroeconomic Factor Model (examples/factor_model_example.jl)
    # =========================================================================
    @testset "Realistic Macroeconomic Factor Model" begin
        rng = MersenneTwister(42)  # DGP-02: explicit rng

        # 3 persistent factors, N=40, T=400 (DGP-02 #791): IC2 recovers the
        # true r exactly (verified pin), replacing the old vacuous bounds.
        fsim = dgp_dynamic_factors(rng;
            A=[0.8 0.1 0.0; 0.0 0.7 0.1; 0.05 0.0 0.6], N=40, T=400)
        X_macro = fsim.X

        # Determine optimal number of factors
        ic_macro = ic_criteria(X_macro, 8)
        r_optimal = ic_macro.r_IC2
        @test r_optimal == 3

        # Estimate with the recovered number of factors
        model_macro = estimate_factors(X_macro, r_optimal)
        @test model_macro isa FactorModel
        # Strong factors (signal share 0.7): mean R² well above noise.
        @test mean(r2(model_macro)) > 0.5
    end

    # =========================================================================
    # Example 5: GMM Estimation (docs/src/examples.md)
    # =========================================================================
    @testset "GMM Estimation" begin
        rng = MersenneTwister(42)  # DGP-02: explicit rng

        n_obs = 500
        n_params = 2

        # Instruments
        Z = randn(rng, n_obs, 3)

        # Endogenous regressor
        u = randn(rng, n_obs)
        X = hcat(ones(n_obs), Z[:, 1] + 0.5 * u + 0.2 * randn(rng, n_obs))

        # Outcome
        β_true = [1.0, 2.0]
        Y = X * β_true + u

        # Data bundle
        data = (Y = Y, X = X, Z = hcat(ones(n_obs), Z))

        # Moment function
        function moment_conditions(theta, data)
            residuals = data.Y - data.X * theta
            data.Z .* residuals
        end

        # Initial values
        theta0 = zeros(n_params)

        # Two-step GMM
        gmm_result = estimate_gmm(moment_conditions, theta0, data;
            weighting=:two_step,
            hac=true
        )

        @test gmm_result isa GMMModel
        @test gmm_result.converged
        @test length(gmm_result.theta) == n_params

        # Check parameter recovery (allowing for some bias)
        @test isapprox(gmm_result.theta[1], β_true[1], atol=0.5)
        @test isapprox(gmm_result.theta[2], β_true[2], atol=0.5)

        # Standard errors
        se = sqrt.(diag(gmm_result.vcov))
        @test all(se .> 0)

        # J-test
        j_result = j_test(gmm_result)
        @test j_result.J_stat >= 0
        @test j_result.df >= 0
        @test 0 <= j_result.p_value <= 1
    end

    # =========================================================================
    # Example 6: Complete Workflow (docs/src/examples.md)
    # =========================================================================
    @testset "Complete Workflow" begin
        rng = MersenneTwister(2024)  # DGP-02: explicit rng

        T, n = 200, 4
        # VAR(1) truth (DGP-02 #791): AIC recovers the true lag exactly.
        Y = dgp_var(rng; A=[0.6 0.1 0.0 0.0; 0.0 0.5 0.1 0.0;
                            0.0 0.0 0.55 0.05; 0.05 0.0 0.0 0.45],
                    B0=0.3 * Matrix{Float64}(I, n, n), T=T).Y

        # Lag selection
        aics = Float64[]
        bics = Float64[]
        for p in 1:8
            m = fit(VARModel, Y, p)
            push!(aics, aic(m))
            push!(bics, bic(m))
        end
        p_aic = argmin(aics)
        p_bic = argmin(bics)
        @test p_aic == 1
        @test 1 <= p_bic <= 8
        p = p_bic

        # VAR estimation
        model = fit(VARModel, Y, p)
        @test model isa VARModel
        @test !isnan(loglikelihood(model))

        # Frequentist IRF
        H = 20
        irfs = irf(model, H; method=:cholesky)
        @test irfs isa ImpulseResponse

        fevd_res = fevd(model, H; method=:cholesky)
        @test fevd_res isa FEVD

        # Bayesian estimation (reduced for speed)
        best_hyper = optimize_hyperparameters(Y, p; grid_size=5)
        @test best_hyper isa MinnesotaHyperparameters

        post = estimate_bvar(Y, p; n_draws=50,
                              prior=:minnesota, hyper=best_hyper)
        @test post isa BVARPosterior

        # Bayesian IRF
        birf = irf(post, H; method=:cholesky)
        @test birf isa BayesianImpulseResponse

        # LP comparison
        lp_model = estimate_lp(Y, 1, H; lags=p, cov_type=:newey_west)
        lp_result = lp_irf(lp_model)
        @test lp_result isa LPImpulseResponse

        # Smooth LP
        smooth_lp = estimate_smooth_lp(Y, 1, H; lambda=1.0, lags=p)
        smooth_result = smooth_lp_irf(smooth_lp)
        @test smooth_result isa LPImpulseResponse
    end

    # =========================================================================
    # Local Projections Example (examples/local_projections_example.jl)
    # =========================================================================
    @testset "Local Projections Example File" begin
        rng = MersenneTwister(42)  # DGP-02: explicit rng

        T = 200
        n = 3

        A1 = [0.5 0.1 -0.1; 0.05 0.6 0.0; 0.1 0.15 0.7]
        A2 = [0.2 0.05 0.0; 0.0 0.2 0.0; 0.05 0.05 0.1]

        # Known VAR(2) truth (DGP-02 #791).
        Y = dgp_var(rng; A=[A1, A2], B0=0.5 * Matrix{Float64}(I, n, n), T=T).Y

        horizon = 20
        shock_var = 1

        # Basic LP
        lp_model = estimate_lp(Y, shock_var, horizon; lags=2, cov_type=:newey_west)
        lp_irf_result = lp_irf(lp_model)

        @test lp_irf_result isa LPImpulseResponse
        @test size(lp_irf_result.values, 1) == horizon + 1
        @test !isempty(lp_irf_result.ci_lower)
        @test !isempty(lp_irf_result.ci_upper)

        # VAR comparison
        var_model = estimate_var(Y, 2)
        var_irf_result = irf(var_model, horizon; method=:cholesky)
        @test var_irf_result isa ImpulseResponse

        # Cumulative IRF
        cum_irf = cumulative_irf(lp_irf_result)
        @test cum_irf isa LPImpulseResponse
        @test size(cum_irf.values) == size(lp_irf_result.values)
    end
end
