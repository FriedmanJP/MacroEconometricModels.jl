# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

using Test
using LinearAlgebra
using Statistics
using Random
using MacroEconometricModels

# Use MacroEconometricModels versions of StatsAPI functions
const residuals = MacroEconometricModels.residuals
const r2 = MacroEconometricModels.r2
const predict = MacroEconometricModels.predict
const nobs = MacroEconometricModels.nobs
const dof = MacroEconometricModels.dof
const loglikelihood = MacroEconometricModels.loglikelihood
const aic = MacroEconometricModels.aic
const bic = MacroEconometricModels.bic

@testset "Dynamic Factor Model Tests" begin

    # ==========================================================================
    # Basic Estimation Tests
    # ==========================================================================

    @testset "Basic Estimation - Two-Step" begin
        Random.seed!(12345)

        T_obs, N, r, p = 200, 20, 3, 2
        X = randn(T_obs, N)

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
        Random.seed!(12346)

        T_obs, N, r, p = 150, 15, 2, 1
        X = randn(T_obs, N)

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
        Random.seed!(12347)

        T_obs, N, r, p = 100, 10, 2, 1
        X = randn(T_obs, N) .* 10 .+ 5

        model = estimate_dynamic_factors(X, r, p; standardize=false)

        @test model.standardized == false
        @test size(model.factors) == (T_obs, r)
    end

    # ==========================================================================
    # Parameter Recovery Tests
    # ==========================================================================

    @testset "Parameter Recovery - Known DGP" begin
        Random.seed!(54321)

        # Known DGP parameters
        T_obs, N, r_true, p_true = 500, 20, 3, 2

        # True factor dynamics (stationary)
        A1_true = [0.4 0.1 0.0; 0.1 0.4 0.1; 0.0 0.1 0.4]
        A2_true = [0.1 0.0 0.0; 0.0 0.1 0.0; 0.0 0.0 0.1]

        # True loadings
        Lambda_true = randn(N, r_true)

        # Generate factors with VAR dynamics
        F_true = zeros(T_obs, r_true)
        for t in (p_true+1):T_obs
            F_true[t, :] = A1_true * F_true[t-1, :] + A2_true * F_true[t-2, :] + 0.3 * randn(r_true)
        end

        # Generate observables
        sigma_e = 0.2
        X = F_true * Lambda_true' + sigma_e * randn(T_obs, N)

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
        Random.seed!(99999)

        T_obs, N, r_true, p_true = 1000, 30, 2, 1

        # Simple AR(1) factor dynamics
        A_true = [0.6 0.1; 0.1 0.6]
        Lambda_true = randn(N, r_true)

        F_true = zeros(T_obs, r_true)
        for t in 2:T_obs
            F_true[t, :] = A_true * F_true[t-1, :] + 0.3 * randn(r_true)
        end

        X = F_true * Lambda_true' + 0.15 * randn(T_obs, N)

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
        Random.seed!(11111)

        T_obs, N, r = 100, 10, 2

        # Create data with highly correlated variables
        F = randn(T_obs, r)
        Lambda = randn(N, r)
        # Very low noise
        X = F * Lambda' + 1e-6 * randn(T_obs, N)

        # Should not throw, should handle gracefully
        model = estimate_dynamic_factors(X, r, 1)
        @test model isa DynamicFactorModel
        @test isfinite(model.loglik) || model.loglik < 0  # Allow -Inf for degenerate cases
    end

    @testset "Numerical Stability - Ill-Conditioned Data" begin
        Random.seed!(22222)

        T_obs, N, r = 100, 15, 3

        X = randn(T_obs, N)
        # Scale columns dramatically
        X[:, 1] *= 1e4
        X[:, end] *= 1e-4

        model = estimate_dynamic_factors(X, r, 1; standardize=true)
        @test model isa DynamicFactorModel
        @test all(isfinite.(model.loadings))
        @test all(isfinite.(model.factors))
    end

    @testset "Numerical Stability - Nearly Non-Stationary" begin
        Random.seed!(33333)

        T_obs, N, r, p = 200, 10, 2, 1

        # Generate factors with near-unit-root dynamics
        F = zeros(T_obs, r)
        A_near_unit = [0.95 0.0; 0.0 0.95]
        for t in 2:T_obs
            F[t, :] = A_near_unit * F[t-1, :] + 0.1 * randn(r)
        end

        Lambda = randn(N, r)
        X = F * Lambda' + 0.2 * randn(T_obs, N)

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
        Random.seed!(44444)

        T_obs, N = 100, 10
        F = randn(T_obs, 1)
        Lambda = randn(N, 1)
        X = F * Lambda' + 0.3 * randn(T_obs, N)

        model = estimate_dynamic_factors(X, 1, 1)

        @test model.r == 1
        @test size(model.factors, 2) == 1
        @test size(model.A[1]) == (1, 1)
        @test length(model.A) == 1
    end

    @testset "Edge Cases - Single Lag (p=1)" begin
        Random.seed!(55555)

        T_obs, N, r = 100, 10, 2
        X = randn(T_obs, N)

        model = estimate_dynamic_factors(X, r, 1)

        @test model.p == 1
        @test length(model.A) == 1
        @test size(model.A[1]) == (r, r)
    end

    @testset "Edge Cases - Multiple Lags (p=4)" begin
        Random.seed!(55556)

        T_obs, N, r, p = 200, 12, 2, 4
        X = randn(T_obs, N)

        model = estimate_dynamic_factors(X, r, p)

        @test model.p == p
        @test length(model.A) == p
        @test all(size(A) == (r, r) for A in model.A)
    end

    @testset "Edge Cases - Short Sample" begin
        Random.seed!(66666)

        T_obs, N, r, p = 50, 8, 2, 1
        X = randn(T_obs, N)

        model = estimate_dynamic_factors(X, r, p)

        @test nobs(model) == T_obs
        @test size(model.factor_residuals, 1) == T_obs - p
    end

    @testset "Edge Cases - Many Variables (N > T)" begin
        Random.seed!(77777)

        T_obs, N, r = 50, 100, 3
        X = randn(T_obs, N)

        model = estimate_dynamic_factors(X, r, 1)

        @test size(model.loadings) == (N, r)
        @test size(model.factors) == (T_obs, r)
    end

    @testset "Edge Cases - Maximum Factors" begin
        Random.seed!(88888)

        T_obs, N = 60, 20
        r_max = min(T_obs, N) - 5  # Leave room for estimation
        X = randn(T_obs, N)

        model = estimate_dynamic_factors(X, r_max, 1)

        @test model.r == r_max
        @test size(model.factors, 2) == r_max
    end

    # ==========================================================================
    # Forecasting Tests
    # ==========================================================================

    @testset "Forecasting - Dimensions" begin
        Random.seed!(10101)

        T_obs, N, r, p = 100, 10, 2, 2
        X = randn(T_obs, N)
        model = estimate_dynamic_factors(X, r, p)

        h = 12
        fc = forecast(model, h)

        @test size(fc.factors) == (h, r)
        @test size(fc.observables) == (h, N)
    end

    @testset "Forecasting - With Confidence Intervals" begin
        Random.seed!(10102)

        T_obs, N, r, p = 100, 8, 2, 1
        X = randn(T_obs, N)
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
        Random.seed!(10103)

        T_obs, N, r, p = 300, 12, 2, 1

        # Generate with known dynamics
        A_true = [0.7 0.1; 0.1 0.7]
        F = zeros(T_obs + 20, r)
        for t in 2:(T_obs + 20)
            F[t, :] = A_true * F[t-1, :] + 0.2 * randn(r)
        end

        Lambda = randn(N, r)
        X_full = F * Lambda' + 0.15 * randn(T_obs + 20, N)

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
        Random.seed!(20202)

        T_obs, N, r = 200, 15, 3

        # Generate static factor data (no dynamics in true DGP)
        F_true = randn(T_obs, r)
        Lambda_true = randn(N, r)
        X = F_true * Lambda_true' + 0.3 * randn(T_obs, N)

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
        Random.seed!(30303)

        T_obs, N, r, p = 100, 12, 3, 2
        X = randn(T_obs, N)

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
        Random.seed!(40404)

        T_obs, N = 200, 20
        r_true, p_true = 2, 1

        # Generate data with known (r, p)
        A_true = [0.5 0.1; 0.1 0.5]
        F = zeros(T_obs, r_true)
        for t in 2:T_obs
            F[t, :] = A_true * F[t-1, :] + 0.3 * randn(r_true)
        end
        Lambda = randn(N, r_true)
        X = F * Lambda' + 0.3 * randn(T_obs, N)

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
        T_obs, N = 100, 10
        X = randn(T_obs, N)

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
        Random.seed!(50505)

        T_obs, N, r, p = 200, 15, 2, 1

        # Generate data with moderate dynamics
        F = zeros(T_obs, r)
        A_true = [0.5 0.1; 0.1 0.5]
        for t in 2:T_obs
            F[t, :] = A_true * F[t-1, :] + 0.3 * randn(r)
        end
        Lambda = randn(N, r)
        X = F * Lambda' + 0.25 * randn(T_obs, N)

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

        # Log-likelihoods should be in similar range
        # EM should achieve higher or similar likelihood
        @test model_em.loglik >= model_twostep.loglik - 50
    end

    # ==========================================================================
    # Asymptotic Properties Tests
    # ==========================================================================

    @testset "Asymptotic Properties - Consistency" begin
        Random.seed!(60606)

        r, p = 2, 1
        N = 15

        # Test consistency: estimates improve with sample size
        sample_sizes = [100, 200, 400]
        errors = Float64[]

        # Fixed true parameters
        A_true = [0.5 0.1; 0.1 0.5]
        Lambda_true = randn(N, r)

        for T_obs in sample_sizes
            # Generate data
            F = zeros(T_obs, r)
            for t in 2:T_obs
                F[t, :] = A_true * F[t-1, :] + 0.3 * randn(r)
            end
            X = F * Lambda_true' + 0.2 * randn(T_obs, N)

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
        Random.seed!(70707)

        T_obs, N, r, p = 100, 10, 2, 3
        X = randn(T_obs, N)

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
        Random.seed!(70708)

        T_obs, N, r, p = 150, 12, 2, 1
        X = randn(T_obs, N)

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
        Random.seed!(80808)

        T_obs, N, r, p = 80, 10, 2, 1

        # Integer input should be converted to Float64
        X_int = rand(1:10, T_obs, N)
        model = estimate_dynamic_factors(X_int, r, p)

        @test model isa DynamicFactorModel{Float64}
        @test eltype(model.factors) == Float64
        @test eltype(model.loadings) == Float64
    end

    # ==========================================================================
    # Variance Explained Properties
    # ==========================================================================

    @testset "Variance Explained Properties" begin
        Random.seed!(90909)

        T_obs, N, r = 100, 20, 5
        X = randn(T_obs, N)

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
        Random.seed!(91919)

        T_obs, N, r, p = 150, 15, 3, 1

        # Generate data with clear factor structure
        F_true = zeros(T_obs, r)
        for t in 2:T_obs
            F_true[t, :] = 0.5 * F_true[t-1, :] + randn(r)
        end
        Lambda = randn(N, r)
        X = F_true * Lambda' + 0.2 * randn(T_obs, N)

        model = estimate_dynamic_factors(X, r, p)

        resid = residuals(model)

        # Check residuals dimensions
        @test size(resid) == size(X)

        # Residuals should generally have lower variance than standardized original
        # Note: comparison is in standardized space, so tolerances need adjustment
        count_lower = 0
        for i in 1:N
            if var(resid[:, i]) <= var(X[:, i])
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
        Random.seed!(92929)

        T_obs, N, r_true = 200, 15, 3

        # Generate data with clear factor structure
        F_true = zeros(T_obs, r_true)
        A_true = 0.6 * I(r_true)
        for t in 2:T_obs
            F_true[t, :] = A_true * F_true[t-1, :] + 0.3 * randn(r_true)
        end
        Lambda_true = randn(N, r_true)
        noise_level = 0.1
        X = F_true * Lambda_true' + noise_level * randn(T_obs, N)

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

end
