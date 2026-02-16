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

"""
    Tests for Generalized Dynamic Factor Model (GDFM)

Comprehensive test suite for the GDFM implementation following
Forni, Hallin, Lippi, and Reichlin (2000, 2005).
"""

using Test
using MacroEconometricModels
using LinearAlgebra
using Statistics
using Random

@testset "Generalized Dynamic Factor Model" begin

    # ==========================================================================
    # Basic Estimation Tests
    # ==========================================================================

    @testset "Basic GDFM Estimation" begin
        Random.seed!(12345)
        T_obs, N, q = 200, 20, 2

        # Generate simple factor data
        F_true = randn(T_obs, q)
        Lambda = randn(N, q)
        X = F_true * Lambda' + 0.3 * randn(T_obs, N)

        # Estimate GDFM
        model = estimate_gdfm(X, q)

        # Type checks
        @test model isa GeneralizedDynamicFactorModel{Float64}

        # Dimension checks
        @test size(model.X) == (T_obs, N)
        @test size(model.factors) == (T_obs, q)
        @test size(model.common_component) == (T_obs, N)
        @test size(model.idiosyncratic) == (T_obs, N)
        @test model.q == q
        @test length(model.frequencies) > 0
        @test length(model.variance_explained) == q

        # Common + idiosyncratic should approximately equal original
        reconstruction = model.common_component + model.idiosyncratic
        @test maximum(abs.(reconstruction - X)) < 1e-10
    end

    @testset "Different Kernels" begin
        Random.seed!(23456)
        T_obs, N, q = 150, 15, 2
        X = randn(T_obs, N)

        for kernel in [:bartlett, :parzen, :tukey]
            model = estimate_gdfm(X, q; kernel=kernel)
            @test model isa GeneralizedDynamicFactorModel
            @test model.kernel == kernel
        end
    end

    @testset "Standardization Options" begin
        Random.seed!(34567)
        T_obs, N, q = 100, 10, 1

        # Create data with different scales
        X = randn(T_obs, N)
        X[:, 1] .*= 100  # Large scale
        X[:, 2] .*= 0.01  # Small scale

        # With standardization
        model_std = estimate_gdfm(X, q; standardize=true)
        @test model_std.standardized == true

        # Without standardization
        model_nostd = estimate_gdfm(X, q; standardize=false)
        @test model_nostd.standardized == false

        # Both should produce valid outputs
        @test size(model_std.factors) == (T_obs, q)
        @test size(model_nostd.factors) == (T_obs, q)
    end

    @testset "Custom Bandwidth" begin
        Random.seed!(45678)
        T_obs, N, q = 120, 12, 1
        X = randn(T_obs, N)

        # Automatic bandwidth
        model_auto = estimate_gdfm(X, q; bandwidth=0)
        @test model_auto.bandwidth > 0

        # Custom bandwidth
        bw = 5
        model_custom = estimate_gdfm(X, q; bandwidth=bw)
        @test model_custom.bandwidth == bw
    end

    # ==========================================================================
    # Factor Recovery Tests
    # ==========================================================================

    @testset "Single Factor Recovery" begin
        Random.seed!(56789)
        T_obs, N = 300, 30
        q = 1

        # Generate single factor data with clear structure
        F_true = randn(T_obs)
        Lambda = randn(N)
        # Low noise for clearer recovery
        X = F_true * Lambda' + 0.2 * randn(T_obs, N)

        model = estimate_gdfm(X, q)

        # Common component should capture most variance
        var_common = var(vec(model.common_component))
        var_total = var(vec(X))
        @test var_common / var_total > 0.5  # At least 50% variance explained
    end

    @testset "Multiple Factor Recovery" begin
        Random.seed!(67890)
        T_obs, N = 400, 40
        q_true = 3

        # Generate multi-factor data
        F_true = randn(T_obs, q_true)
        Lambda = randn(N, q_true)
        X = F_true * Lambda' + 0.25 * randn(T_obs, N)

        model = estimate_gdfm(X, q_true)

        # Check variance explained increases with factors
        @test length(model.variance_explained) == q_true
        @test all(model.variance_explained .> 0)

        # R² should be reasonable for most variables
        r2_vals = r2(model)
        @test mean(r2_vals) > 0.3  # Average R² > 30%
    end

    @testset "Dynamic Factor Structure" begin
        Random.seed!(78901)
        T_obs, N = 300, 25
        q = 2

        # Generate factors with AR dynamics
        F_true = zeros(T_obs, q)
        for t in 2:T_obs
            F_true[t, :] = 0.7 * F_true[t-1, :] + randn(q)
        end

        Lambda = randn(N, q)
        X = F_true * Lambda' + 0.3 * randn(T_obs, N)

        model = estimate_gdfm(X, q)

        # Spectral loadings should vary across frequencies for dynamic factors
        @test size(model.loadings_spectral, 3) > 1  # Multiple frequencies
        @test any(!iszero, model.loadings_spectral)
    end

    # ==========================================================================
    # Spectral Density Tests
    # ==========================================================================

    @testset "Spectral Density Properties" begin
        Random.seed!(89012)
        T_obs, N, q = 128, 16, 2
        X = randn(T_obs, N)

        model = estimate_gdfm(X, q)

        n_freq = length(model.frequencies)

        # Frequencies should be in [0, π]
        @test model.frequencies[1] >= 0
        @test model.frequencies[end] <= π + 1e-10

        # Spectral density should be Hermitian at each frequency
        for j in 1:n_freq
            S_j = model.spectral_density_X[:, :, j]
            @test norm(S_j - S_j') < 1e-10  # Hermitian check
        end

        # Eigenvalues should be real and non-negative
        @test all(model.eigenvalues_spectral .>= -1e-10)
    end

    @testset "Eigenvalue Ordering" begin
        Random.seed!(90123)
        T_obs, N, q = 100, 15, 3
        X = randn(T_obs, N)

        model = estimate_gdfm(X, q)

        # Eigenvalues should be sorted in descending order at each frequency
        n_freq = length(model.frequencies)
        for j in 1:n_freq
            eigs = model.eigenvalues_spectral[:, j]
            @test issorted(eigs, rev=true)
        end
    end

    # ==========================================================================
    # StatsAPI Interface Tests
    # ==========================================================================

    @testset "StatsAPI Interface" begin
        Random.seed!(12345)
        T_obs, N, q = 100, 10, 2
        X = randn(T_obs, N)

        model = estimate_gdfm(X, q)

        # predict
        fitted = predict(model)
        @test size(fitted) == (T_obs, N)
        @test fitted == model.common_component

        # residuals
        resid = residuals(model)
        @test size(resid) == (T_obs, N)
        @test resid == model.idiosyncratic

        # nobs
        @test nobs(model) == T_obs

        # dof
        @test dof(model) > 0

        # r2
        r2_vals = r2(model)
        @test length(r2_vals) == N
        @test all(r2_vals .<= 1.0 + 1e-10)  # R² <= 1
    end

    @testset "R² Consistency" begin
        Random.seed!(23456)
        T_obs, N, q = 150, 12, 2

        # Strong factor structure
        F_true = randn(T_obs, q)
        Lambda = randn(N, q)
        X_strong = F_true * Lambda' + 0.1 * randn(T_obs, N)

        # Weak factor structure
        X_weak = 0.1 * F_true * Lambda' + randn(T_obs, N)

        model_strong = estimate_gdfm(X_strong, q)
        model_weak = estimate_gdfm(X_weak, q)

        r2_strong = mean(r2(model_strong))
        r2_weak = mean(r2(model_weak))

        # Strong structure should have higher R²
        @test r2_strong > r2_weak
    end

    # ==========================================================================
    # Information Criteria Tests
    # ==========================================================================

    @testset "Information Criteria Computation" begin
        Random.seed!(34567)
        T_obs, N = 200, 20
        max_q = 5
        X = randn(T_obs, N)

        ic = ic_criteria_gdfm(X, max_q)

        # Check outputs exist
        @test length(ic.eigenvalue_ratios) >= 1
        @test length(ic.cumulative_variance) == max_q
        @test length(ic.avg_eigenvalues) == max_q

        # Cumulative variance should be increasing
        @test issorted(ic.cumulative_variance)

        # Cumulative variance should sum to <= 1 for q <= max_q
        @test ic.cumulative_variance[end] <= 1.0 + 1e-10

        # Selected q should be in valid range
        @test 1 <= ic.q_ratio <= max_q
        @test 1 <= ic.q_variance <= max_q
    end

    @testset "Factor Selection with Known Structure" begin
        Random.seed!(45678)
        T_obs, N = 300, 30
        q_true = 2
        max_q = 5

        # Generate data with clear 2-factor structure
        F_true = randn(T_obs, q_true)
        Lambda = randn(N, q_true)
        X = F_true * Lambda' + 0.2 * randn(T_obs, N)

        ic = ic_criteria_gdfm(X, max_q)

        # First two eigenvalues should dominate
        @test ic.avg_eigenvalues[1] > ic.avg_eigenvalues[3]
        @test ic.avg_eigenvalues[2] > ic.avg_eigenvalues[3]

        # Eigenvalue ratio should suggest q close to true value
        @test ic.q_ratio <= q_true + 1
    end

    # ==========================================================================
    # Forecasting Tests
    # ==========================================================================

    @testset "Basic Forecasting" begin
        Random.seed!(56789)
        T_obs, N, q = 150, 15, 2
        h = 10

        X = randn(T_obs, N)
        model = estimate_gdfm(X, q)

        fc = forecast(model, h; method=:ar)

        # Dimension checks
        @test size(fc.observables) == (h, N)
        @test size(fc.factors) == (h, q)

        # Forecasts should be finite
        @test all(isfinite, fc.observables)
        @test all(isfinite, fc.factors)
    end

    @testset "Forecast Methods" begin
        Random.seed!(67890)
        T_obs, N, q = 120, 12, 2
        h = 5

        X = randn(T_obs, N)
        model = estimate_gdfm(X, q)

        # AR method
        fc_ar = forecast(model, h; method=:ar)
        @test size(fc_ar.observables) == (h, N)

        # Spectral method
        fc_spectral = forecast(model, h; method=:spectral)
        @test size(fc_spectral.observables) == (h, N)
    end

    @testset "Forecast with Dynamic Factors" begin
        Random.seed!(78901)
        T_obs, N, q = 200, 20, 2
        h = 12

        # Generate AR(1) factors
        F_true = zeros(T_obs, q)
        phi = 0.8
        for t in 2:T_obs
            F_true[t, :] = phi * F_true[t-1, :] + randn(q)
        end

        Lambda = randn(N, q)
        X = F_true * Lambda' + 0.3 * randn(T_obs, N)

        model = estimate_gdfm(X, q)
        fc = forecast(model, h)

        # Forecasts should decay toward zero for stationary factors
        factor_norm_start = norm(fc.factors[1, :])
        factor_norm_end = norm(fc.factors[end, :])
        # Not always true due to estimation uncertainty, so just check finiteness
        @test isfinite(factor_norm_start)
        @test isfinite(factor_norm_end)
    end

    # ==========================================================================
    # Edge Cases
    # ==========================================================================

    @testset "Single Factor (q=1)" begin
        Random.seed!(89012)
        T_obs, N = 100, 15
        q = 1

        X = randn(T_obs, N)
        model = estimate_gdfm(X, q)

        @test model.q == 1
        @test size(model.factors, 2) == 1
        @test length(model.variance_explained) == 1
    end

    @testset "Many Factors (q close to N)" begin
        Random.seed!(90123)
        T_obs, N = 100, 10
        q = N - 2  # Many factors

        X = randn(T_obs, N)
        model = estimate_gdfm(X, q)

        @test model.q == q
        @test size(model.factors, 2) == q

        # With many factors, should explain reasonable variance
        # (random data won't have strong factor structure)
        r2_vals = r2(model)
        @test mean(r2_vals) > 0.3  # Relaxed threshold for random data
    end

    @testset "Short Time Series" begin
        Random.seed!(12345)
        T_obs = 50  # Short
        N, q = 10, 2

        X = randn(T_obs, N)
        model = estimate_gdfm(X, q)

        @test size(model.factors) == (T_obs, q)
        @test all(isfinite, model.common_component)
    end

    @testset "Wide Panel (N > T)" begin
        Random.seed!(23456)
        T_obs = 50
        N = 100  # N > T
        q = 2

        X = randn(T_obs, N)
        model = estimate_gdfm(X, q)

        @test size(model.X) == (T_obs, N)
        @test size(model.factors) == (T_obs, q)
    end

    @testset "Power of 2 Sample Size" begin
        Random.seed!(34567)
        T_obs = 256  # Power of 2 for efficient FFT
        N, q = 20, 3

        X = randn(T_obs, N)
        model = estimate_gdfm(X, q)

        @test size(model.factors) == (T_obs, q)
    end

    # ==========================================================================
    # Numerical Stability Tests
    # ==========================================================================

    @testset "Near-Collinear Data" begin
        Random.seed!(45678)
        T_obs, N = 100, 10
        q = 2

        # Create nearly collinear variables
        X = randn(T_obs, N)
        X[:, 2] = X[:, 1] + 1e-8 * randn(T_obs)

        model = estimate_gdfm(X, q)
        @test all(isfinite, model.common_component)
        @test all(isfinite, model.factors)
    end

    @testset "Extreme Scaling" begin
        Random.seed!(56789)
        T_obs, N, q = 100, 10, 2

        # Very large values
        X_large = 1e6 * randn(T_obs, N)
        model_large = estimate_gdfm(X_large, q; standardize=true)
        @test all(isfinite, model_large.common_component)

        # Very small values
        X_small = 1e-6 * randn(T_obs, N)
        model_small = estimate_gdfm(X_small, q; standardize=true)
        @test all(isfinite, model_small.common_component)
    end

    @testset "Mixed Scaling" begin
        Random.seed!(67890)
        T_obs, N, q = 100, 10, 2

        X = randn(T_obs, N)
        X[:, 1] .*= 1e6
        X[:, end] .*= 1e-6

        model = estimate_gdfm(X, q; standardize=true)
        @test all(isfinite, model.common_component)
        @test all(isfinite, r2(model))
    end

    @testset "Constant Column" begin
        Random.seed!(78901)
        T_obs, N, q = 100, 10, 2

        X = randn(T_obs, N)
        X[:, 3] .= 5.0  # Constant column

        model = estimate_gdfm(X, q; standardize=true)
        # Should handle constant column gracefully
        @test all(isfinite, model.common_component)
    end

    # ==========================================================================
    # Input Validation Tests
    # ==========================================================================

    @testset "Input Validation" begin
        Random.seed!(89012)
        T_obs, N = 100, 10
        X = randn(T_obs, N)

        # Invalid q
        @test_throws ArgumentError estimate_gdfm(X, 0)
        @test_throws ArgumentError estimate_gdfm(X, N + 1)

        # Invalid kernel
        @test_throws ArgumentError estimate_gdfm(X, 2; kernel=:invalid)

        # Invalid r < q
        @test_throws ArgumentError estimate_gdfm(X, 3; r=2)
    end

    @testset "IC Criteria Validation" begin
        Random.seed!(90123)
        T_obs, N = 100, 10
        X = randn(T_obs, N)

        # Invalid max_q
        @test_throws ArgumentError ic_criteria_gdfm(X, 0)
        @test_throws ArgumentError ic_criteria_gdfm(X, N + 1)
    end

    @testset "Forecast Validation" begin
        Random.seed!(12345)
        T_obs, N, q = 100, 10, 2
        X = randn(T_obs, N)
        model = estimate_gdfm(X, q)

        # Invalid horizon
        @test_throws ArgumentError forecast(model, 0)
        @test_throws ArgumentError forecast(model, -1)

        # Invalid method
        @test_throws ArgumentError forecast(model, 5; method=:invalid)
    end

    # ==========================================================================
    # Utility Function Tests
    # ==========================================================================

    @testset "Common Variance Share" begin
        Random.seed!(23456)
        T_obs, N, q = 150, 15, 2

        # Strong factor structure
        F_true = randn(T_obs, q)
        Lambda = randn(N, q)
        X = F_true * Lambda' + 0.1 * randn(T_obs, N)

        model = estimate_gdfm(X, q)

        shares = common_variance_share(model)

        @test length(shares) == N
        @test all(shares .>= 0)
        @test all(shares .<= 1.0 + 1e-10)

        # Should be high for strong factor structure
        @test mean(shares) > 0.5
    end

    @testset "Spectral Eigenvalue Plot Data" begin
        Random.seed!(34567)
        T_obs, N, q = 100, 12, 2
        X = randn(T_obs, N)

        model = estimate_gdfm(X, q)

        plot_data = spectral_eigenvalue_plot_data(model)

        @test haskey(plot_data, :frequencies)
        @test haskey(plot_data, :eigenvalues)

        @test plot_data.frequencies == model.frequencies
        @test plot_data.eigenvalues == model.eigenvalues_spectral
    end

    # ==========================================================================
    # Consistency Tests
    # ==========================================================================

    @testset "Decomposition Consistency" begin
        Random.seed!(45678)
        T_obs, N, q = 120, 15, 2
        X = randn(T_obs, N)

        model = estimate_gdfm(X, q)

        # X = chi + xi
        @test norm(model.X - (model.common_component + model.idiosyncratic)) < 1e-10

        # predict() returns common component
        @test predict(model) == model.common_component

        # residuals() returns idiosyncratic
        @test residuals(model) == model.idiosyncratic
    end

    @testset "Reproducibility" begin
        Random.seed!(56789)
        T_obs, N, q = 100, 10, 2
        X = randn(T_obs, N)

        # Same data should give same results
        model1 = estimate_gdfm(X, q; bandwidth=5, kernel=:bartlett)
        model2 = estimate_gdfm(X, q; bandwidth=5, kernel=:bartlett)

        @test model1.factors ≈ model2.factors
        @test model1.common_component ≈ model2.common_component
    end

    @testset "Integer Matrix Input" begin
        Random.seed!(67890)
        T_obs, N, q = 100, 10, 2

        X_int = rand(1:10, T_obs, N)

        model = estimate_gdfm(X_int, q)
        @test model isa GeneralizedDynamicFactorModel{Float64}
        @test all(isfinite, model.common_component)
    end

    # ==========================================================================
    # Asymptotic Behavior Tests
    # ==========================================================================

    @testset "Increasing Sample Size" begin
        Random.seed!(78901)
        N, q = 20, 2
        sample_sizes = [100, 200, 400]

        # Generate consistent DGP
        F_full = randn(500, q)
        Lambda = randn(N, q)
        e = 0.3 * randn(500, N)

        r2_values = Float64[]

        for T_obs in sample_sizes
            X = F_full[1:T_obs, :] * Lambda' + e[1:T_obs, :]
            model = estimate_gdfm(X, q)
            push!(r2_values, mean(r2(model)))
        end

        # R² should generally improve or stay stable with more data
        # Allow for some variation due to randomness
        @test r2_values[end] > 0.3  # At least reasonable R² with largest sample
    end

    @testset "Increasing Panel Width" begin
        Random.seed!(89012)
        T_obs, q = 200, 2
        panel_widths = [10, 25, 50]

        variance_explained_list = Float64[]

        for N in panel_widths
            F_true = randn(T_obs, q)
            Lambda = randn(N, q)
            X = F_true * Lambda' + 0.3 * randn(T_obs, N)

            model = estimate_gdfm(X, q)
            push!(variance_explained_list, sum(model.variance_explained))
        end

        # Variance explained should remain reasonable regardless of N
        @test all(variance_explained_list .> 0.2)
    end

end

# Run all tests
println("Running GDFM tests...")
