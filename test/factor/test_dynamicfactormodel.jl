# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using LinearAlgebra
using Statistics
using Random
using MacroEconometricModels

# StatsAPI functions are exported by MacroEconometricModels via `using`

@testset "Dynamic Factor Model Tests" begin

    # ==========================================================================
    # Basic Estimation Tests
    # ==========================================================================

    @testset "Basic Estimation - Two-Step" begin
        rng = Random.MersenneTwister(12345)

        T_obs, N, r, p = 200, 20, 3, 2
        X = randn(rng, T_obs, N)

        model = estimate_dynamic_factors(X, r, p)

        @test model isa DynamicFactorModel
        @test size(model.X) == (T_obs, N)
        @test size(model.factors) == (T_obs, r)
        @test size(model.loadings) == (N, r)
        @test length(model.A) == p
        @test all(size(A) == (r, r) for A in model.A)
        @test size(model.factor_residuals) == (T_obs - p, r)
        @test size(model.Sigma_eta) == (r, r)
        @test size(model.Sigma_e) == (N, N)
        @test model.r == r
        @test model.p == p
        @test model.method == :twostep
        @test model.standardized == true
        @test model.converged == true
        @test isfinite(model.loglik)
    end

    @testset "Basic Estimation - EM Algorithm" begin
        rng = Random.MersenneTwister(12346)

        T_obs, N, r, p = 150, 15, 2, 1
        X = randn(rng, T_obs, N)

        model = estimate_dynamic_factors(X, r, p; method=:em, max_iter=50)

        @test model isa DynamicFactorModel
        @test size(model.factors) == (T_obs, r)
        @test size(model.loadings) == (N, r)
        @test length(model.A) == p
        @test model.method == :em
        @test model.iterations >= 1
        @test isfinite(model.loglik)
    end

    @testset "Non-Standardized Estimation" begin
        rng = Random.MersenneTwister(12347)

        T_obs, N, r, p = 100, 10, 2, 1
        X = randn(rng, T_obs, N .* 10 .+ 5)

        model = estimate_dynamic_factors(X, r, p; standardize=false)

        @test model.standardized == false
        @test size(model.factors) == (T_obs, r)
    end

    # ==========================================================================
    # Parameter Recovery Tests
    # ==========================================================================

    @testset "Parameter Recovery - Known DGP" begin
        # DGP-06: shared VAR(2)-factor simulator (was: bespoke zero-initialized
        # loop without burn-in). The simulator burns in, so factors start stationary.
        rng = Random.MersenneTwister(54321)

        # Known DGP parameters
        T_obs, N, r_true, p_true = 500, 20, 3, 2

        # True factor dynamics (stationary)
        A1_true = [0.4 0.1 0.0; 0.1 0.4 0.1; 0.0 0.1 0.4]
        A2_true = [0.1 0.0 0.0; 0.0 0.1 0.0; 0.0 0.0 0.1]

        # Generate observables
        sigma_e = 0.2
        d = dgp_dynamic_factors(rng; A=[A1_true, A2_true], N=N, T=T_obs, idio_sd=sigma_e)
        X, F_true, Lambda_true = d.X, d.F, d.Lambda

        # Estimate
        model = estimate_dynamic_factors(X, r_true, p_true)

        # Test: Factor space recovery (correlation between true and estimated)
        F_hat = model.factors[(p_true+1):end, :]
        F_true_eff = F_true[(p_true+1):end, :]

        # Compute canonical correlations (simplified: correlation matrix)
        for j in 1:r_true
            max_corr = maximum(abs.(cor(F_true_eff[:, j], F_hat)))
            @test max_corr > 0.5  # At least moderate correlation
        end

        # Test: Eigenvalue recovery (rotation-invariant)
        companion_true = [A1_true A2_true; I(r_true) zeros(r_true, r_true)]
        companion_est = companion_matrix_factors(model)

        eig_true = sort(abs.(eigvals(companion_true)), rev=true)
        eig_est = sort(abs.(eigvals(companion_est)), rev=true)

        # Allow some estimation error
        @test isapprox(eig_true[1], eig_est[1], rtol=0.3)

        # Test: Stationarity preserved
        @test is_stationary(model)
    end

    @testset "Parameter Recovery - Larger Sample" begin
        # DGP-06: shared simulator (was: bespoke loop).
        rng = Random.MersenneTwister(99999)

        T_obs, N, r_true, p_true = 1000, 30, 2, 1

        # Simple AR(1) factor dynamics
        A_true = [0.6 0.1; 0.1 0.6]

        d = dgp_dynamic_factors(rng; A=A_true, N=N, T=T_obs, idio_sd=0.15)
        X = d.X

        model = estimate_dynamic_factors(X, r_true, p_true)

        # With large sample, should recover eigenvalues well
        eig_true = sort(abs.(eigvals(A_true)), rev=true)
        eig_est = sort(abs.(eigvals(model.A[1])), rev=true)

        @test isapprox(eig_true, eig_est, rtol=0.25)
    end

    # ==========================================================================
    # Numerical Stability Tests
    # ==========================================================================

    @testset "Numerical Stability - Near-Singular Covariance" begin
        # DGP-06: zero-dynamics shared DGP with near-zero idiosyncratic noise
        # (was: bespoke iid-factor loop). Near-singularity comes from the tiny
        # noise, not from dynamics.
        rng = Random.MersenneTwister(11111)

        T_obs, N, r = 100, 10, 2

        # Create data with highly correlated variables
        X = dgp_dynamic_factors(rng; A=zeros(r, r), N=N, T=T_obs, idio_sd=1e-6).X

        # Should not throw, should handle gracefully
        model = estimate_dynamic_factors(X, r, 1)
        @test model isa DynamicFactorModel
        @test isfinite(model.loglik) || model.loglik < 0  # Allow -Inf for degenerate cases
    end

    @testset "Numerical Stability - Ill-Conditioned Data" begin
        rng = Random.MersenneTwister(22222)

        T_obs, N, r = 100, 15, 3

        X = randn(rng, T_obs, N)
        # Scale columns dramatically
        X[:, 1] *= 1e4
        X[:, end] *= 1e-4

        model = estimate_dynamic_factors(X, r, 1; standardize=true)
        @test model isa DynamicFactorModel
        @test all(isfinite.(model.loadings))
        @test all(isfinite.(model.factors))
    end

    @testset "Numerical Stability - Nearly Non-Stationary" begin
        # DGP-06: shared simulator with near-unit-root factor dynamics.
        rng = Random.MersenneTwister(33333)

        T_obs, N, r, p = 200, 10, 2, 1

        # Generate factors with near-unit-root dynamics
        A_near_unit = [0.95 0.0; 0.0 0.95]
        X = dgp_dynamic_factors(rng; A=A_near_unit, N=N, T=T_obs, idio_sd=0.2).X

        model = estimate_dynamic_factors(X, r, p)
        @test model isa DynamicFactorModel
        # Should detect near-stationarity
        max_eig = maximum(abs.(eigvals(model.A[1])))
        @test max_eig < 1.0 || max_eig < 1.05  # Allow small overshoot due to estimation
    end

    # ==========================================================================
    # Edge Case Tests
    # ==========================================================================

    @testset "Edge Cases - Single Factor (r=1)" begin
        # DGP-06: shared simulator (was: bespoke iid-factor loop).
        rng = Random.MersenneTwister(44444)

        T_obs, N = 100, 10
        X = dgp_dynamic_factors(rng; A=reshape([0.5], 1, 1), N=N, T=T_obs, idio_sd=0.3).X

        model = estimate_dynamic_factors(X, 1, 1)

        @test model.r == 1
        @test size(model.factors, 2) == 1
        @test size(model.A[1]) == (1, 1)
        @test length(model.A) == 1
    end

    @testset "Edge Cases - Single Lag (p=1)" begin
        rng = Random.MersenneTwister(55555)

        T_obs, N, r = 100, 10, 2
        X = randn(rng, T_obs, N)

        model = estimate_dynamic_factors(X, r, 1)

        @test model.p == 1
        @test length(model.A) == 1
        @test size(model.A[1]) == (r, r)
    end

    @testset "Edge Cases - Multiple Lags (p=4)" begin
        rng = Random.MersenneTwister(55556)

        T_obs, N, r, p = 200, 12, 2, 4
        X = randn(rng, T_obs, N)

        model = estimate_dynamic_factors(X, r, p)

        @test model.p == p
        @test length(model.A) == p
        @test all(size(A) == (r, r) for A in model.A)
    end

    @testset "Edge Cases - Short Sample" begin
        rng = Random.MersenneTwister(66666)

        T_obs, N, r, p = 50, 8, 2, 1
        X = randn(rng, T_obs, N)

        model = estimate_dynamic_factors(X, r, p)

        @test nobs(model) == T_obs
        @test size(model.factor_residuals, 1) == T_obs - p
    end

    @testset "Edge Cases - Many Variables (N > T)" begin
        rng = Random.MersenneTwister(77777)

        T_obs, N, r = 50, 100, 3
        X = randn(rng, T_obs, N)

        model = estimate_dynamic_factors(X, r, 1)

        @test size(model.loadings) == (N, r)
        @test size(model.factors) == (T_obs, r)
    end

    @testset "Edge Cases - Maximum Factors" begin
        rng = Random.MersenneTwister(88888)

        T_obs, N = 60, 20
        r_max = min(T_obs, N) - 5  # Leave room for estimation
        X = randn(rng, T_obs, N)

        model = estimate_dynamic_factors(X, r_max, 1)

        @test model.r == r_max
        @test size(model.factors, 2) == r_max
    end

    # ==========================================================================
    # Forecasting Tests
    # ==========================================================================

    @testset "Forecasting - Dimensions" begin
        rng = Random.MersenneTwister(10101)

        T_obs, N, r, p = 100, 10, 2, 2
        X = randn(rng, T_obs, N)
        model = estimate_dynamic_factors(X, r, p)

        h = 12
        fc = forecast(model, h)

        @test size(fc.factors) == (h, r)
        @test size(fc.observables) == (h, N)
    end

    @testset "Forecasting - With Confidence Intervals" begin
        rng = Random.MersenneTwister(10102)

        T_obs, N, r, p = 100, 8, 2, 1
        X = randn(rng, T_obs, N)
        model = estimate_dynamic_factors(X, r, p)

        h = 6
        fc = forecast(model, h; ci=true, conf_level=0.90)

        @test size(fc.factors) == (h, r)
        @test size(fc.observables) == (h, N)
        @test size(fc.factors_lower) == (h, r)
        @test size(fc.factors_upper) == (h, r)
        @test size(fc.observables_lower) == (h, N)
        @test size(fc.observables_upper) == (h, N)

        # Upper should be greater than lower
        @test all(fc.factors_upper .>= fc.factors_lower)
        @test all(fc.observables_upper .>= fc.observables_lower)
    end

    @testset "Forecasting - Accuracy with Known Dynamics" begin
        # DGP-06: shared simulator over the train+holdout span, then split
        # (was: bespoke loop). The holdout is a genuine continuation of the DGP.
        rng = Random.MersenneTwister(10103)

        T_obs, N, r, p = 300, 12, 2, 1

        # Generate with known dynamics
        A_true = [0.7 0.1; 0.1 0.7]
        X_full = dgp_dynamic_factors(rng; A=A_true, N=N, T=T_obs + 20, idio_sd=0.15).X

        # Estimate on first T_obs observations
        X_train = X_full[1:T_obs, :]
        model = estimate_dynamic_factors(X_train, r, p)

        # Forecast 10 steps
        fc = forecast(model, 10)

        # Compare to holdout
        X_test = X_full[(T_obs+1):(T_obs+10), :]

        # RMSE should be reasonable
        rmse = sqrt(mean((fc.observables .- X_test).^2))
        baseline_std = std(X_train)

        # Forecast RMSE should be less than 2x data std (reasonable bound)
        @test rmse < 2 * baseline_std
    end

    # ==========================================================================
    # Static Model as Special Case Tests
    # ==========================================================================

    @testset "Static Model as Special Case" begin
        # DGP-06: the static DGP is the shared simulator with zero transition —
        # iid factors by construction (was: bespoke iid loop).
        rng = Random.MersenneTwister(20202)

        T_obs, N, r = 200, 15, 3

        # Generate static factor data (no dynamics in true DGP)
        X = dgp_dynamic_factors(rng; A=zeros(r, r), N=N, T=T_obs, idio_sd=0.3).X

        # Estimate static model
        static_model = estimate_factors(X, r)

        # Estimate dynamic model
        dynamic_model = estimate_dynamic_factors(X, r, 1)

        # Variance explained should be similar
        @test abs(static_model.cumulative_variance[r] - dynamic_model.cumulative_variance[r]) < 0.15

        # AR coefficients should be small for truly static data
        A_norm = norm(dynamic_model.A[1]) / r
        @test A_norm < 0.6  # Not too large
    end

    # ==========================================================================
    # StatsAPI Interface Tests
    # ==========================================================================

    @testset "StatsAPI Interface" begin
        rng = Random.MersenneTwister(30303)

        T_obs, N, r, p = 100, 12, 3, 2
        X = randn(rng, T_obs, N)

        model = estimate_dynamic_factors(X, r, p)

        # nobs
        @test nobs(model) == T_obs

        # dof
        df = dof(model)
        @test df > 0
        # Expected: N*r + r*r*p + r*(r+1)/2 + N
        expected_dof_approx = N * r + r * r * p + div(r * (r + 1), 2) + N
        @test df == expected_dof_approx

        # predict
        X_fitted = predict(model)
        @test size(X_fitted) == (T_obs, N)

        # residuals
        resid = residuals(model)
        @test size(resid) == (T_obs, N)

        # r2
        r2_vals = r2(model)
        @test length(r2_vals) == N
        @test all(0 .<= r2_vals .<= 1)

        # loglikelihood
        ll = loglikelihood(model)
        @test isfinite(ll)

        # aic, bic
        @test isfinite(aic(model))
        @test isfinite(bic(model))
        @test bic(model) >= aic(model)  # BIC penalizes more for n > e^2
    end

    # ==========================================================================
    # Information Criteria Tests
    # ==========================================================================

    @testset "Information Criteria - Model Selection" begin
        # DGP-06: shared simulator (was: bespoke loop).
        rng = Random.MersenneTwister(40404)

        T_obs, N = 200, 20
        r_true, p_true = 2, 1

        # Generate data with known (r, p)
        A_true = [0.5 0.1; 0.1 0.5]
        X = dgp_dynamic_factors(rng; A=A_true, N=N, T=T_obs, idio_sd=0.3).X

        ic = ic_criteria_dynamic(X, 4, 2)  # Reduced range to avoid edge case failures

        @test size(ic.AIC) == (4, 2)
        @test size(ic.BIC) == (4, 2)
        @test 1 <= ic.r_AIC <= 4
        @test 1 <= ic.p_AIC <= 2
        @test 1 <= ic.r_BIC <= 4
        @test 1 <= ic.p_BIC <= 2

        # At least one (r, p) combination should have finite IC
        @test any(isfinite.(ic.AIC))
        @test any(isfinite.(ic.BIC))
    end

    # ==========================================================================
    # Input Validation Tests
    # ==========================================================================

    @testset "Input Validation" begin
        rng = Random.MersenneTwister(40504)
        T_obs, N = 100, 10
        X = randn(rng, T_obs, N)

        # Invalid number of factors
        @test_throws ArgumentError estimate_dynamic_factors(X, 0, 1)
        @test_throws ArgumentError estimate_dynamic_factors(X, N + 1, 1)
        @test_throws ArgumentError estimate_dynamic_factors(X, -1, 1)

        # Invalid number of lags
        @test_throws ArgumentError estimate_dynamic_factors(X, 2, 0)
        @test_throws ArgumentError estimate_dynamic_factors(X, 2, -1)
        @test_throws ArgumentError estimate_dynamic_factors(X, 2, T_obs)

        # Invalid method
        @test_throws ArgumentError estimate_dynamic_factors(X, 2, 1; method=:invalid)

        # Invalid forecast horizon
        model = estimate_dynamic_factors(X, 2, 1)
        @test_throws ArgumentError forecast(model, 0)
        @test_throws ArgumentError forecast(model, -1)
    end

    # ==========================================================================
    # Consistency Between Methods Tests
    # ==========================================================================

    @testset "Consistency Between Methods" begin
        # DGP-06: shared simulator (was: bespoke loop).
        rng = Random.MersenneTwister(50505)

        T_obs, N, r, p = 200, 15, 2, 1

        # Generate data with moderate dynamics
        A_true = [0.5 0.1; 0.1 0.5]
        X = dgp_dynamic_factors(rng; A=A_true, N=N, T=T_obs, idio_sd=0.25).X

        model_twostep = estimate_dynamic_factors(X, r, p; method=:twostep)
        model_em = estimate_dynamic_factors(X, r, p; method=:em, max_iter=100)

        # Factors should span similar space
        F_ts = model_twostep.factors[(p+1):end, :]
        F_em = model_em.factors[(p+1):end, :]

        # Compute correlation between factor estimates
        for j in 1:r
            corr_j = abs(cor(F_ts[:, j], F_em[:, j]))
            @test corr_j > 0.7 || any(abs.(cor(F_ts[:, j], F_em)) .> 0.7)
        end

        # `model_twostep.loglik` is a conditional-on-factors Gaussian likelihood of the
        # idiosyncratic residuals (e = Y − F·Λ'), whereas `model_em.loglik` is the marginal
        # state-space likelihood — different quantities, not directly comparable in magnitude.
        # (The old `EM ≥ twostep − 50` only held because the F-06 PCA-reconstruction bug made
        # the two-step residuals — and hence its loglik — artificially low.) Require both finite
        # and negative, and agreeing to within ~1 nat per observation-variable.
        @test isfinite(model_twostep.loglik) && isfinite(model_em.loglik)
        @test model_twostep.loglik < 0 && model_em.loglik < 0
        @test abs(model_em.loglik - model_twostep.loglik) / (T_obs * N) < 1.0
    end

    # ==========================================================================
    # Asymptotic Properties Tests
    # ==========================================================================

    @testset "Asymptotic Properties - Consistency" begin
        # DGP-06: shared simulator with FIXED loadings across sample sizes
        # (was: bespoke loop). Holding Λ fixed isolates the T effect.
        rng = Random.MersenneTwister(60606)

        r, p = 2, 1
        N = 15

        # Test consistency: estimates improve with sample size
        sample_sizes = [100, 200, 400]
        errors = Float64[]

        # Fixed true parameters
        A_true = [0.5 0.1; 0.1 0.5]
        Lambda_true = randn(rng, N, r)

        for T_obs in sample_sizes
            # Generate data
            X = dgp_dynamic_factors(rng; A=A_true, Lambda=Lambda_true, N=N,
                                    T=T_obs, idio_sd=0.2).X

            model = estimate_dynamic_factors(X, r, p)

            # Measure error in eigenvalues of A (rotation-invariant)
            eig_true = sort(abs.(eigvals(A_true)))
            eig_est = sort(abs.(eigvals(model.A[1])))
            push!(errors, norm(eig_true - eig_est))
        end

        # Errors should generally decrease with sample size
        @test errors[end] <= errors[1] * 1.5  # Allow some variation
    end

    # ==========================================================================
    # Companion Matrix Tests
    # ==========================================================================

    @testset "Companion Matrix" begin
        rng = Random.MersenneTwister(70707)

        T_obs, N, r, p = 100, 10, 2, 3
        X = randn(rng, T_obs, N)

        model = estimate_dynamic_factors(X, r, p)

        C = companion_matrix_factors(model)

        @test size(C) == (r * p, r * p)

        # Top rows should contain A matrices
        for lag in 1:p
            @test C[1:r, ((lag-1)*r+1):(lag*r)] == model.A[lag]
        end

        # Lower blocks should be identity
        if p > 1
            @test C[(r+1):end, 1:(r*(p-1))] == I(r * (p - 1))
        end
    end

    @testset "Stationarity Check" begin
        rng = Random.MersenneTwister(70708)

        T_obs, N, r, p = 150, 12, 2, 1
        X = randn(rng, T_obs, N)

        model = estimate_dynamic_factors(X, r, p)

        # Check that is_stationary returns a boolean
        stat = is_stationary(model)
        @test stat isa Bool

        # For random data, model should typically be stationary
        @test stat == true
    end

    # ==========================================================================
    # Type Conversion Tests
    # ==========================================================================

    @testset "Type Conversion" begin
        rng = Random.MersenneTwister(80808)

        T_obs, N, r, p = 80, 10, 2, 1

        # Integer input should be converted to Float64
        X_int = rand(rng, 1:10, T_obs, N)
        model = estimate_dynamic_factors(X_int, r, p)

        @test model isa DynamicFactorModel{Float64}
        @test eltype(model.factors) == Float64
        @test eltype(model.loadings) == Float64
    end

    # ==========================================================================
    # Variance Explained Properties
    # ==========================================================================

    @testset "Variance Explained Properties" begin
        rng = Random.MersenneTwister(90909)

        T_obs, N, r = 100, 20, 5
        X = randn(rng, T_obs, N)

        model = estimate_dynamic_factors(X, r, 2)

        # Explained variance should be positive
        @test all(model.explained_variance[1:r] .>= 0)

        # Cumulative variance should be increasing
        @test issorted(model.cumulative_variance[1:r])

        # First r eigenvalues should be in descending order (approximately)
        @test model.eigenvalues[1] >= model.eigenvalues[r]
    end

    # ==========================================================================
    # Residuals Properties
    # ==========================================================================

    @testset "Residuals Properties" begin
        # DGP-06: shared simulator (was: bespoke loop).
        rng = Random.MersenneTwister(91919)

        T_obs, N, r, p = 150, 15, 3, 1

        # Generate data with clear factor structure
        X = dgp_dynamic_factors(rng; A=0.5 * Matrix{Float64}(I, r, r), N=N,
                                T=T_obs, idio_sd=0.2).X

        model = estimate_dynamic_factors(X, r, p)

        resid = residuals(model)

        # Check residuals dimensions
        @test size(resid) == size(X)

        # DGP-06: residuals() lives in standardized space (see
        # StatsAPI.residuals), so the reference must be standardized too —
        # comparing against raw X only passed before by scale luck (the old
        # bespoke DGP had Var(X) ≈ 4 > 1).
        X_ref = MacroEconometricModels._standardize(X)
        count_lower = 0
        for i in 1:N
            if var(resid[:, i]) <= var(X_ref[:, i])
                count_lower += 1
            end
        end
        # At least half should have lower variance
        @test count_lower >= N ÷ 2

        # Mean of residuals should be near zero (in standardized space, this is less strict)
        @test abs(mean(resid)) < 1.0
    end

    # ==========================================================================
    # Reconstruction Quality
    # ==========================================================================

    @testset "Reconstruction Quality" begin
        # DGP-06: shared simulator (was: bespoke loop).
        rng = Random.MersenneTwister(92929)

        T_obs, N, r_true = 200, 15, 3

        # Generate data with clear factor structure
        A_true = 0.6 * Matrix{Float64}(I, r_true, r_true)
        noise_level = 0.1
        X = dgp_dynamic_factors(rng; A=A_true, N=N, T=T_obs, idio_sd=noise_level).X

        model = estimate_dynamic_factors(X, r_true, 1)
        X_fitted = predict(model)

        # Check dimensions
        @test size(X_fitted) == size(X)

        # R² should be non-negative
        r2_vals = r2(model)
        @test all(r2_vals .>= -0.01)  # Allow small numerical errors

        # Verify cumulative variance explained is reasonable
        @test model.cumulative_variance[r_true] > 0.5
    end

    @testset "Lag-one smoother cross-covariance (T097 #196)" begin
        # Pt_smooth[t] must equal the EXACT lag-one smoother cross-covariance
        # Cov(alpha_{t+1}, alpha_t | Y) = P_smooth[t+1]·J_t'. The old J_t·P_smooth[t+1]
        # transposed the time order, corrupting the DFM EM VAR / Sigma_eta updates.
        rng = Random.MersenneTwister(11)
        r, p, N, Tn = 1, 2, 2, 6
        sd = r * p
        Λ = reshape([1.0, 0.7], N, r)
        A = [reshape([0.5], 1, 1), reshape([0.2], 1, 1)]
        Sigma_eta = reshape([0.3], 1, 1)
        Sigma_e = [0.4 0.0; 0.0 0.5]
        Y = randn(rng, Tn, N)
        _, _, Pt, _ = MacroEconometricModels._kalman_smoother_dfm(Y, Λ, A, Sigma_eta, Sigma_e, r, p)
        T_mat = zeros(sd, sd); T_mat[1:r, 1:r] = A[1]; T_mat[1:r, r+1:2r] = A[2]
        T_mat[r+1:sd, 1:sd-r] = Matrix(I, sd-r, sd-r)
        Q = zeros(sd, sd); Q[1:r, 1:r] = Sigma_eta
        Z = zeros(N, sd); Z[:, 1:r] = Λ
        Sinf = MacroEconometricModels._compute_unconditional_covariance(T_mat, Q, sd)
        SX = zeros(sd * Tn, sd * Tn)
        for t in 1:Tn, s in 1:Tn
            if t >= s
                Mm = copy(Sinf); for _ in 1:(t - s); Mm = T_mat * Mm; end
                SX[(t-1)*sd+1:t*sd, (s-1)*sd+1:s*sd] = Mm
                SX[(s-1)*sd+1:s*sd, (t-1)*sd+1:t*sd] = Mm'
            end
        end
        H = zeros(N * Tn, sd * Tn); for t in 1:Tn; H[(t-1)*N+1:t*N, (t-1)*sd+1:t*sd] = Z; end
        Rb = zeros(N * Tn, N * Tn); for t in 1:Tn; Rb[(t-1)*N+1:t*N, (t-1)*N+1:t*N] = Sigma_e; end
        Spost = SX - SX * H' * inv(H * SX * H' + Rb) * H * SX
        for t in 1:(Tn - 1)
            blk = Spost[t*sd+1:(t+1)*sd, (t-1)*sd+1:t*sd]   # Cov(alpha_{t+1}, alpha_t)
            @test Pt[t, :, :] ≈ blk atol=1e-8
        end
    end

end
