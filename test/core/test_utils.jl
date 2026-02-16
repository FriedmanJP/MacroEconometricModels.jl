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

using MacroEconometricModels
using Test
using LinearAlgebra
using Statistics
using Random

@testset "Utility Functions" begin
    Random.seed!(12345)

    # ==========================================================================
    # Input Validation Tests
    # ==========================================================================

    @testset "VAR Input Validation" begin
        # Valid inputs return false (short-circuit && evaluation)
        @test MacroEconometricModels.validate_var_inputs(100, 3, 2) == false

        # Invalid lag order (p < 1)
        @test_throws ArgumentError MacroEconometricModels.validate_var_inputs(100, 3, 0)
        @test_throws ArgumentError MacroEconometricModels.validate_var_inputs(100, 3, -1)

        # Not enough observations
        @test_throws ArgumentError MacroEconometricModels.validate_var_inputs(5, 3, 10)
        @test_throws ArgumentError MacroEconometricModels.validate_var_inputs(3, 3, 2)

        # Invalid number of variables
        @test_throws ArgumentError MacroEconometricModels.validate_var_inputs(100, 0, 2)
        @test_throws ArgumentError MacroEconometricModels.validate_var_inputs(100, -1, 2)

        # Edge case: minimum valid
        @test MacroEconometricModels.validate_var_inputs(4, 1, 2; min_obs_factor=1) == false
    end

    @testset "Factor Model Input Validation" begin
        # Valid inputs return false (short-circuit && evaluation)
        @test MacroEconometricModels.validate_factor_inputs(100, 50, 5) == false

        # Invalid r (too small)
        @test_throws ArgumentError MacroEconometricModels.validate_factor_inputs(100, 50, 0)

        # Invalid r (too large)
        @test_throws ArgumentError MacroEconometricModels.validate_factor_inputs(100, 50, 60)
        @test_throws ArgumentError MacroEconometricModels.validate_factor_inputs(50, 100, 60)

        # Edge case: r = min(T, N)
        @test MacroEconometricModels.validate_factor_inputs(100, 50, 50) == false
    end

    @testset "Dynamic Factor Model Input Validation" begin
        # Valid inputs return false (short-circuit && evaluation)
        @test MacroEconometricModels.validate_dynamic_factor_inputs(100, 50, 5, 2) == false

        # Invalid p (too small)
        @test_throws ArgumentError MacroEconometricModels.validate_dynamic_factor_inputs(100, 50, 5, 0)

        # Invalid p (too large relative to T - r)
        @test_throws ArgumentError MacroEconometricModels.validate_dynamic_factor_inputs(20, 50, 5, 15)
    end

    @testset "Generic Validators" begin
        # validate_positive - returns false when valid (short-circuit && with false condition)
        @test MacroEconometricModels.validate_positive(1.0, "test") == false
        @test MacroEconometricModels.validate_positive(0.001, "test") == false
        @test_throws ArgumentError MacroEconometricModels.validate_positive(0.0, "test")
        @test_throws ArgumentError MacroEconometricModels.validate_positive(-1.0, "test")

        # validate_in_range - returns false when valid
        @test MacroEconometricModels.validate_in_range(0.5, "test", 0.0, 1.0) == false
        @test MacroEconometricModels.validate_in_range(0.0, "test", 0.0, 1.0) == false
        @test MacroEconometricModels.validate_in_range(1.0, "test", 0.0, 1.0) == false
        @test_throws ArgumentError MacroEconometricModels.validate_in_range(-0.1, "test", 0.0, 1.0)
        @test_throws ArgumentError MacroEconometricModels.validate_in_range(1.1, "test", 0.0, 1.0)

        # validate_option - returns false when valid
        @test MacroEconometricModels.validate_option(:foo, "test", (:foo, :bar, :baz)) == false
        @test_throws ArgumentError MacroEconometricModels.validate_option(:invalid, "test", (:foo, :bar))
    end

    # ==========================================================================
    # Matrix Utility Tests
    # ==========================================================================

    @testset "Robust Inverse" begin
        # Regular invertible matrix
        A = [1.0 0.5; 0.5 1.0]
        A_inv = MacroEconometricModels.robust_inv(A)
        @test A * A_inv ≈ I(2) atol=1e-10

        # Near-singular matrix (should use pseudo-inverse)
        B = [1.0 1.0; 1.0 1.0 + 1e-12]
        B_inv = MacroEconometricModels.robust_inv(B)
        @test all(isfinite.(B_inv))

        # Integer matrix (should convert to float)
        C = [2 1; 1 2]
        C_inv = MacroEconometricModels.robust_inv(C)
        @test C_inv isa Matrix{Float64}
        @test Float64.(C) * C_inv ≈ I(2) atol=1e-10
    end

    @testset "Safe Cholesky Decomposition" begin
        # Positive definite matrix
        A_pd = [4.0 2.0; 2.0 3.0]
        L = MacroEconometricModels.safe_cholesky(A_pd)
        @test L * L' ≈ A_pd atol=1e-10
        @test istril(L)

        # Nearly positive semi-definite (needs jitter)
        eigenvals = [1.0, 1e-14]  # One very small eigenvalue
        Q = qr(randn(2, 2)).Q
        A_npsd = Q * Diagonal(eigenvals) * Q'
        A_npsd = (A_npsd + A_npsd') / 2  # Ensure symmetric
        L_npsd = MacroEconometricModels.safe_cholesky(A_npsd)
        @test all(isfinite.(L_npsd))

        # Custom jitter
        A_custom = [1.0 0.9999; 0.9999 1.0]
        L_custom = MacroEconometricModels.safe_cholesky(A_custom; jitter=1e-6)
        @test all(isfinite.(L_custom))
    end

    @testset "Safe Log Determinant" begin
        # Regular positive definite matrix
        A_pd = [4.0 2.0; 2.0 3.0]
        ld = MacroEconometricModels.logdet_safe(A_pd)
        @test ld ≈ log(det(A_pd)) atol=1e-10

        # Nearly singular matrix
        A_sing = [1.0 1.0; 1.0 1.0 + 1e-15]
        ld_sing = MacroEconometricModels.logdet_safe(A_sing)
        @test isfinite(ld_sing) || ld_sing == -Inf
    end

    # ==========================================================================
    # VAR Matrix Construction Tests
    # ==========================================================================

    @testset "VAR Matrix Construction" begin
        T_obs = 100
        n = 3
        p = 2
        Y = randn(T_obs, n)

        Y_eff, X = MacroEconometricModels.construct_var_matrices(Y, p)

        # Check dimensions
        T_eff = T_obs - p
        @test size(Y_eff) == (T_eff, n)
        @test size(X) == (T_eff, 1 + n*p)

        # First column should be ones (intercept)
        @test all(X[:, 1] .== 1.0)

        # Check lag structure
        for t in 1:T_eff
            for lag in 1:p
                cols = (2 + (lag-1)*n):(1 + lag*n)
                @test X[t, cols] ≈ Y[p + t - lag, :]
            end
        end

        # Test with non-Float64 input
        Y_int = rand(1:10, 50, 2)
        Y_eff_int, X_int = MacroEconometricModels.construct_var_matrices(Y_int, 1)
        @test eltype(Y_eff_int) == Float64
        @test eltype(X_int) == Float64

        # Test error for insufficient observations
        @test_throws ArgumentError MacroEconometricModels.construct_var_matrices(Y[1:2, :], 3)
    end

    @testset "AR Coefficient Extraction" begin
        n = 2
        p = 3
        # B matrix: (1 + n*p) x n = 7 x 2
        B = randn(1 + n*p, n)

        A_coeffs = MacroEconometricModels.extract_ar_coefficients(B, n, p)

        @test length(A_coeffs) == p
        for i in 1:p
            @test size(A_coeffs[i]) == (n, n)
        end
    end

    @testset "Companion Matrix Construction" begin
        n = 2
        p = 2
        B = [0.1 0.2;    # intercept
             0.5 0.1;    # A1 row 1
             0.1 0.4;    # A1 row 2
             0.2 0.0;    # A2 row 1
             0.0 0.2]    # A2 row 2

        F = MacroEconometricModels.companion_matrix(B, n, p)

        @test size(F) == (n*p, n*p)

        # Check structure: [A1 A2; I 0]
        A1 = B[2:3, :]'
        A2 = B[4:5, :]'
        @test F[1:n, 1:n] ≈ A1
        @test F[1:n, (n+1):(2n)] ≈ A2
        @test F[(n+1):(2n), 1:n] ≈ I(n)
        @test F[(n+1):(2n), (n+1):(2n)] ≈ zeros(n, n)

        # Test VAR(1)
        B1 = [0.1 0.2; 0.5 0.3; 0.2 0.6]
        F1 = MacroEconometricModels.companion_matrix(B1, 2, 1)
        @test size(F1) == (2, 2)
    end

    # ==========================================================================
    # Statistical Utility Tests
    # ==========================================================================

    @testset "Univariate AR Variance" begin
        # Generate AR(1) data with known variance
        T_ar = 500
        rho = 0.7
        sigma = 1.0
        y = zeros(T_ar)
        for t in 2:T_ar
            y[t] = rho * y[t-1] + sigma * randn()
        end

        ar_std = MacroEconometricModels.univariate_ar_variance(y)
        @test ar_std > 0
        @test isapprox(ar_std, sigma, atol=0.3)  # Should be close to innovation std

        # Short series (< 3 observations) should return std
        y_short = [1.0, 2.0]
        ar_std_short = MacroEconometricModels.univariate_ar_variance(y_short)
        @test ar_std_short ≈ std(y_short)
    end

    # ==========================================================================
    # Name Generation Tests
    # ==========================================================================

    @testset "Default Name Generation" begin
        var_names = MacroEconometricModels.default_var_names(3)
        @test var_names == ["Var 1", "Var 2", "Var 3"]

        var_names_custom = MacroEconometricModels.default_var_names(2; prefix="GDP")
        @test var_names_custom == ["GDP 1", "GDP 2"]

        shock_names = MacroEconometricModels.default_shock_names(2)
        @test shock_names == ["Shock 1", "Shock 2"]

        shock_names_custom = MacroEconometricModels.default_shock_names(3; prefix="MP")
        @test shock_names_custom == ["MP 1", "MP 2", "MP 3"]
    end

    # ==========================================================================
    # Integration Tests
    # ==========================================================================

    @testset "Integration with VAR Estimation" begin
        Random.seed!(54321)
        T_int = 200
        n_int = 2
        p_int = 2

        # Generate VAR data
        Y_int = zeros(T_int, n_int)
        A1 = [0.5 0.1; 0.1 0.4]
        for t in 2:T_int
            Y_int[t, :] = A1 * Y_int[t-1, :] + randn(n_int)
        end

        # The utilities should work correctly within VAR estimation
        model = estimate_var(Y_int, p_int)
        @test model isa VARModel

        # Check that the companion matrix is stable (eigenvalues < 1)
        F = MacroEconometricModels.companion_matrix(model.B, n_int, p_int)
        eigenvalues = eigvals(F)
        @test all(abs.(eigenvalues) .< 1.0)
    end

    @testset "Numerical Stability Edge Cases" begin
        Random.seed!(99999)

        # Very small values
        Y_small = 1e-10 * randn(100, 2)
        Y_eff, X = MacroEconometricModels.construct_var_matrices(Y_small, 2)
        @test all(isfinite.(Y_eff))
        @test all(isfinite.(X))

        # Very large values
        Y_large = 1e10 * randn(100, 2)
        Y_eff_l, X_l = MacroEconometricModels.construct_var_matrices(Y_large, 2)
        @test all(isfinite.(Y_eff_l))
        @test all(isfinite.(X_l))

        # Mixed scales
        Y_mixed = hcat(1e-8 * randn(100), 1e8 * randn(100))
        Y_eff_m, X_m = MacroEconometricModels.construct_var_matrices(Y_mixed, 2)
        @test all(isfinite.(Y_eff_m))
        @test all(isfinite.(X_m))
    end

    # =================================================================
    # NaN/Inf Validation (Issue #22)
    # =================================================================
    @testset "NaN/Inf Validation" begin
        # Matrix with NaN
        Y_nan = [1.0 2.0; NaN 4.0; 5.0 6.0]
        @test_throws ArgumentError MacroEconometricModels._validate_data(Y_nan)

        # Matrix with Inf
        Y_inf = [1.0 2.0; Inf 4.0; 5.0 6.0]
        @test_throws ArgumentError MacroEconometricModels._validate_data(Y_inf)

        # Vector with NaN
        y_nan = [1.0, NaN, 3.0]
        @test_throws ArgumentError MacroEconometricModels._validate_data(y_nan)

        # Vector with Inf
        y_inf = [1.0, Inf, 3.0]
        @test_throws ArgumentError MacroEconometricModels._validate_data(y_inf)

        # Clean data should pass without error
        Y_clean = randn(10, 3)
        @test MacroEconometricModels._validate_data(Y_clean) === nothing

        y_clean = randn(10)
        @test MacroEconometricModels._validate_data(y_clean) === nothing
    end

    # =================================================================
    # safe_cholesky Warning (Issue #23)
    # =================================================================
    @testset "safe_cholesky Warning" begin
        # Nearly singular matrix that needs jitter
        A = [1.0 1.0; 1.0 1.0]  # singular, rank-1
        L = @test_warn r"required jitter" MacroEconometricModels.safe_cholesky(A)
        @test size(L) == (2, 2)
        @test all(isfinite.(L))

        # Well-conditioned matrix — no warning
        B = [2.0 0.5; 0.5 2.0]
        L2 = MacroEconometricModels.safe_cholesky(B)
        @test size(L2) == (2, 2)
        @test L2 * L2' ≈ B atol=1e-10
    end
end
