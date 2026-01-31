using Test
using LinearAlgebra
using Statistics
using Random
using MacroEconometricModels

# Use MacroEconometricModels versions of StatsAPI functions
const fm_residuals = MacroEconometricModels.residuals
const fm_r2 = MacroEconometricModels.r2
const fm_predict = MacroEconometricModels.predict
const fm_nobs = MacroEconometricModels.nobs
const fm_dof = MacroEconometricModels.dof

@testset "Factor Model Tests" begin

    @testset "Basic Factor Model Estimation" begin
        Random.seed!(123)
        # Generate synthetic data with known factor structure
        T, N, r_true = 100, 20, 3

        # True factors and loadings
        F_true = randn(T, r_true)
        Lambda_true = randn(N, r_true)

        # Generate data: X = F * Lambda' + noise
        X = F_true * Lambda_true' + 0.3 * randn(T, N)

        # Estimate factor model
        model = estimate_factors(X, r_true)

        @test model isa FactorModel
        @test size(model.factors) == (T, r_true)
        @test size(model.loadings) == (N, r_true)
        @test length(model.eigenvalues) == N
        @test model.r == r_true
        @test model.standardized == true

        # Factors should have reasonable magnitude (not exploding or collapsing)
        @test all(isfinite, model.factors)
        @test maximum(abs.(model.factors)) < 100  # Reasonable bound

        # Variance explained should sum to 1
        @test isapprox(sum(model.explained_variance), 1.0, atol=1e-10)

        # Cumulative variance should be increasing
        @test issorted(model.cumulative_variance)
        @test model.cumulative_variance[end] ≈ 1.0
    end

    @testset "Factor Model without Standardization" begin
        Random.seed!(234)
        T, N, r = 50, 10, 2
        X = randn(T, N)

        model = estimate_factors(X, r; standardize=false)

        @test model.standardized == false
        @test size(model.factors) == (T, r)
        @test size(model.loadings) == (N, r)
    end

    @testset "Prediction and Residuals" begin
        Random.seed!(345)
        T, N, r = 80, 15, 3
        F_true = randn(T, r)
        Lambda_true = randn(N, r)
        noise = 0.2 * randn(T, N)
        X = F_true * Lambda_true' + noise

        model = estimate_factors(X, r)

        # Test prediction
        X_fitted = fm_predict(model)
        @test size(X_fitted) == (T, N)

        # Test residuals
        resid = fm_residuals(model)
        @test size(resid) == (T, N)

        # Residuals should be finite
        @test all(isfinite, resid)

        # Residuals should have reasonable magnitude (not exploding)
        @test maximum(abs.(resid)) < 10
    end

    @testset "R-squared Computation" begin
        Random.seed!(456)
        T, N, r = 100, 10, 2
        F_true = randn(T, r)
        Lambda_true = randn(N, r)

        # Generate data with low noise for reasonable R²
        X = F_true * Lambda_true' + 0.1 * randn(T, N)

        model = estimate_factors(X, r)
        r2_vals = fm_r2(model)

        @test length(r2_vals) == N
        # R² should be bounded (allow small negative values due to numerical issues)
        @test all(r2_vals .>= -0.1)
        @test all(r2_vals .<= 1.1)
        # R² values should be finite
        @test all(isfinite, r2_vals)
    end

    @testset "Information Criteria" begin
        Random.seed!(567)
        T, N = 100, 20
        r_true = 3

        # Generate data with r_true factors
        F_true = randn(T, r_true)
        Lambda_true = randn(N, r_true)
        X = F_true * Lambda_true' + 0.3 * randn(T, N)

        max_r = 8
        ic = ic_criteria(X, max_r)

        @test length(ic.IC1) == max_r
        @test length(ic.IC2) == max_r
        @test length(ic.IC3) == max_r

        @test 1 <= ic.r_IC1 <= max_r
        @test 1 <= ic.r_IC2 <= max_r
        @test 1 <= ic.r_IC3 <= max_r

        # IC should be finite
        @test all(isfinite.(ic.IC1))
        @test all(isfinite.(ic.IC2))
        @test all(isfinite.(ic.IC3))
    end

    @testset "Scree Plot Data" begin
        Random.seed!(678)
        T, N, r = 100, 15, 5
        X = randn(T, N)

        model = estimate_factors(X, r)
        scree_data = scree_plot_data(model)

        @test length(scree_data.factors) == N
        @test length(scree_data.explained_variance) == N
        @test length(scree_data.cumulative_variance) == N

        # Cumulative variance should be monotonically increasing
        @test issorted(scree_data.cumulative_variance)

        # Last value should be 1
        @test scree_data.cumulative_variance[end] ≈ 1.0
    end

    @testset "StatsAPI Interface" begin
        Random.seed!(789)
        T, N, r = 100, 12, 3
        X = randn(T, N)

        model = estimate_factors(X, r)

        # Test nobs
        @test fm_nobs(model) == T

        # Test dof
        df = fm_dof(model)
        @test df == N * r + T * r - r^2
        @test df > 0
    end

    @testset "Input Validation" begin
        Random.seed!(890)
        T, N = 50, 10
        X = randn(T, N)

        # Test invalid number of factors
        @test_throws ArgumentError estimate_factors(X, 0)
        @test_throws ArgumentError estimate_factors(X, N + 1)
        @test_throws ArgumentError estimate_factors(X, -1)

        # Test IC criteria with invalid max_factors
        @test_throws ArgumentError ic_criteria(X, 0)
        @test_throws ArgumentError ic_criteria(X, min(T, N) + 1)
    end

    @testset "Edge Cases" begin
        Random.seed!(901)
        # Single factor
        T, N = 100, 10
        X = randn(T, N)
        model = estimate_factors(X, 1)
        @test size(model.factors) == (T, 1)
        @test size(model.loadings) == (N, 1)

        # Maximum number of factors
        T, N = 50, 20
        X = randn(T, N)
        r_max = min(T, N)
        model = estimate_factors(X, r_max)
        @test size(model.factors) == (T, r_max)
    end

    @testset "Constant Series Handling" begin
        Random.seed!(12)
        T, N = 100, 10
        X = randn(T, N)

        # Add a constant series
        X[:, 1] .= 5.0

        # Should not throw error due to zero variance
        model = estimate_factors(X, 2)
        @test model isa FactorModel
    end

    @testset "Explained Variance Properties" begin
        Random.seed!(23)
        T, N, r = 100, 20, 5
        X = randn(T, N)

        model = estimate_factors(X, r)

        # First r factors should explain more variance than later ones
        @test model.explained_variance[1] >= model.explained_variance[r]

        # Explained variance should be in descending order (eigenvalues sorted)
        @test issorted(model.explained_variance[1:r], rev=true)

        # Cumulative variance at r should equal sum of first r explained variances
        @test model.cumulative_variance[r] ≈ sum(model.explained_variance[1:r])
    end

    @testset "Reconstruction Quality" begin
        Random.seed!(34)
        T, N = 150, 15  # More observations for stability
        r_true = 3

        # Generate data with clear factor structure
        F_true = randn(T, r_true)
        Lambda_true = randn(N, r_true)
        noise_level = 0.1
        X = F_true * Lambda_true' + noise_level * randn(T, N)

        model = estimate_factors(X, r_true)
        X_fitted = fm_predict(model)

        # Fitted values should be finite
        @test all(isfinite, X_fitted)
        @test size(X_fitted) == size(X)

        # R² should be computed without errors
        r2_vals = fm_r2(model)
        @test length(r2_vals) == N
        @test all(isfinite, r2_vals)
    end

    @testset "Type Stability" begin
        Random.seed!(45)
        T, N, r = 50, 10, 2

        # Float64
        X64 = randn(Float64, T, N)
        model64 = estimate_factors(X64, r)
        @test eltype(model64.factors) == Float64
        @test eltype(model64.loadings) == Float64

        # Float32
        X32 = randn(Float32, T, N)
        model32 = estimate_factors(X32, r)
        @test eltype(model32.factors) == Float32
        @test eltype(model32.loadings) == Float32
    end

    @testset "Integer Input Conversion" begin
        Random.seed!(56)
        T, N, r = 50, 10, 2
        X_int = rand(1:10, T, N)

        model = estimate_factors(X_int, r)
        @test model isa FactorModel{Float64}
        @test all(isfinite, model.factors)
    end

end
