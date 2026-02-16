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
using MacroEconometricModels
using Random
using LinearAlgebra
using Statistics

const MEM_EP = MacroEconometricModels

@testset "Error Paths" begin

    # =========================================================================
    # core/utils.jl
    # =========================================================================

    @testset "Input validation helpers" begin
        # validate_var_inputs
        @test_throws ArgumentError MEM_EP.validate_var_inputs(100, 3, 0)  # p < 1
        @test_throws ArgumentError MEM_EP.validate_var_inputs(5, 3, 5)    # T <= p + 1
        @test_throws ArgumentError MEM_EP.validate_var_inputs(100, 0, 2)  # n < 1
        # valid case should not throw
        MEM_EP.validate_var_inputs(100, 3, 2)

        # validate_factor_inputs
        @test_throws ArgumentError MEM_EP.validate_factor_inputs(100, 10, 0)   # r < 1
        @test_throws ArgumentError MEM_EP.validate_factor_inputs(100, 10, 11)  # r > min(T,N)
        MEM_EP.validate_factor_inputs(100, 10, 5)

        # validate_dynamic_factor_inputs
        @test_throws ArgumentError MEM_EP.validate_dynamic_factor_inputs(100, 10, 5, 0)  # p < 1
        @test_throws ArgumentError MEM_EP.validate_dynamic_factor_inputs(100, 10, 5, 96)  # p >= T-r

        # validate_positive
        @test_throws ArgumentError MEM_EP.validate_positive(0.0, "test")
        @test_throws ArgumentError MEM_EP.validate_positive(-1.0, "test")

        # validate_in_range
        @test_throws ArgumentError MEM_EP.validate_in_range(2.0, "test", 0.0, 1.0)
        @test_throws ArgumentError MEM_EP.validate_in_range(-1.0, "test", 0.0, 1.0)

        # validate_option
        @test_throws ArgumentError MEM_EP.validate_option(:bad, "test", (:a, :b, :c))
    end

    @testset "robust_inv with singular matrix" begin
        # Singular matrix should use pseudo-inverse via fallback
        A = [1.0 0.0; 0.0 0.0]
        result = MEM_EP.robust_inv(A)
        @test size(result) == (2, 2)
        @test isfinite(result[1, 1])

        # Integer input
        A_int = [1 2; 3 4]
        result_int = MEM_EP.robust_inv(A_int)
        @test size(result_int) == (2, 2)
    end

    @testset "safe_cholesky with difficult matrices" begin
        # PSD matrix that needs jitter
        n = 3
        A = zeros(n, n)
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 0.0  # Not PD
        L = MEM_EP.safe_cholesky(A)
        @test size(L) == (n, n)
    end

    @testset "logdet_safe fallback" begin
        # PSD matrix
        A = [1.0 0.0; 0.0 1.0]
        @test MEM_EP.logdet_safe(A) ≈ 0.0

        # Near-singular: eigenvalue fallback
        A_sing = [1.0 1.0; 1.0 1.0 + 1e-15]
        ld = MEM_EP.logdet_safe(A_sing)
        @test isfinite(ld) || ld == -Inf
    end

    @testset "construct_var_matrices errors" begin
        # Too few observations
        Y = randn(3, 2)
        @test_throws ArgumentError MEM_EP.construct_var_matrices(Y, 5)
    end

    @testset "_validate_var_shock_indices" begin
        vars = ["GDP", "CPI", "FFR"]
        shocks = ["Shock 1", "Shock 2", "Shock 3"]
        vi, si = MEM_EP._validate_var_shock_indices("GDP", "Shock 1", vars, shocks)
        @test vi == 1
        @test si == 1
        @test_throws ArgumentError MEM_EP._validate_var_shock_indices("UNKNOWN", "Shock 1", vars, shocks)
        @test_throws ArgumentError MEM_EP._validate_var_shock_indices("GDP", "UNKNOWN", vars, shocks)
    end

    @testset "Name generation" begin
        @test MEM_EP._default_names(3, "Var") == ["Var 1", "Var 2", "Var 3"]
        @test MEM_EP.default_var_names(2) == ["Var 1", "Var 2"]
        @test MEM_EP.default_shock_names(2) == ["Shock 1", "Shock 2"]
        @test MEM_EP.default_var_names(2; prefix="X") == ["X 1", "X 2"]
    end

    # =========================================================================
    # core/covariance.jl
    # =========================================================================

    @testset "NeweyWestEstimator validation" begin
        @test_throws ArgumentError NeweyWestEstimator{Float64}(-1)  # negative bandwidth
        @test_throws ArgumentError NeweyWestEstimator{Float64}(0, :invalid)  # invalid kernel
    end

    @testset "DriscollKraayEstimator validation" begin
        @test_throws ArgumentError DriscollKraayEstimator{Float64}(-1)
    end

    @testset "kernel_weight edge values" begin
        # bandwidth = 0 always returns 0
        @test MEM_EP.kernel_weight(0, 0, :bartlett) == 0.0
        @test MEM_EP.kernel_weight(1, 0, :bartlett) == 0.0

        # j = 0 should return 1 for all kernels (except when bw=0)
        for k in [:bartlett, :parzen, :tukey_hanning]
            @test MEM_EP.kernel_weight(0, 5, k) ≈ 1.0
        end
        @test MEM_EP.kernel_weight(0, 5, :quadratic_spectral) ≈ 1.0

        # All kernels at boundary
        bw = 5
        for k in [:bartlett, :parzen, :tukey_hanning, :quadratic_spectral]
            w = MEM_EP.kernel_weight(bw, bw, k)
            @test isfinite(w)
            @test w >= 0
        end

        # Unknown kernel error
        @test_throws ArgumentError MEM_EP.kernel_weight(1, 5, :invalid)
    end

    @testset "White HC variants" begin
        Random.seed!(8001)
        X = randn(50, 3)
        u = randn(50)
        for variant in [:hc0, :hc1, :hc2, :hc3]
            V = MEM_EP.white_vcov(X, u; variant=variant)
            @test size(V) == (3, 3)
            @test issymmetric(V) || norm(V - V') < 1e-12
        end
    end

    @testset "Newey-West with prewhitening" begin
        Random.seed!(8002)
        X = randn(50, 3)
        u = randn(50)
        V_pw = MEM_EP.newey_west(X, u; prewhiten=true)
        @test size(V_pw) == (3, 3)
    end

    @testset "Newey-West multivariate" begin
        Random.seed!(8003)
        X = randn(50, 3)
        U = randn(50, 2)
        V = MEM_EP.newey_west(X, U)
        @test size(V) == (6, 6)
    end

    @testset "White multivariate" begin
        Random.seed!(8004)
        X = randn(50, 3)
        U = randn(50, 2)
        V = MEM_EP.white_vcov(X, U)
        @test size(V) == (6, 6)
    end

    @testset "Driscoll-Kraay single and multi" begin
        Random.seed!(8005)
        X = randn(50, 3)
        u = randn(50)
        V = MEM_EP.driscoll_kraay(X, u)
        @test size(V) == (3, 3)

        U = randn(50, 2)
        V2 = MEM_EP.driscoll_kraay(X, U)
        @test size(V2) == (6, 6)
    end

    @testset "precompute_XtX_inv" begin
        Random.seed!(8006)
        X = randn(50, 3)
        XtX_inv = MEM_EP.precompute_XtX_inv(X)
        @test size(XtX_inv) == (3, 3)
        @test norm(XtX_inv - inv(X' * X)) < 1e-10

        # Use with newey_west
        u = randn(50)
        V1 = MEM_EP.newey_west(X, u)
        V2 = MEM_EP.newey_west(X, u; XtX_inv=XtX_inv)
        @test norm(V1 - V2) < 1e-10
    end

    @testset "Long-run variance/covariance" begin
        Random.seed!(8007)
        # Short vector (length 1: var returns NaN)
        x_short = [1.0]
        lrv = MEM_EP.long_run_variance(x_short)
        @test isnan(lrv) || isfinite(lrv)  # implementation-dependent edge case

        # Normal case
        x = randn(100)
        lrv_normal = MEM_EP.long_run_variance(x)
        @test lrv_normal >= 0

        # With different kernels
        for k in [:bartlett, :parzen, :quadratic_spectral, :tukey_hanning]
            lrv_k = MEM_EP.long_run_variance(x; kernel=k)
            @test lrv_k >= 0
        end

        # Multivariate long-run covariance
        X = randn(100, 3)
        lrc = MEM_EP.long_run_covariance(X)
        @test size(lrc) == (3, 3)
        @test issymmetric(lrc) || norm(lrc - lrc') < 1e-10

        # Short multivariate
        X_short = randn(1, 2)
        lrc_short = MEM_EP.long_run_covariance(X_short)
        @test size(lrc_short) == (2, 2)
    end

    @testset "robust_vcov dispatch" begin
        Random.seed!(8008)
        X = randn(50, 3)
        u = randn(50)

        V_nw = MEM_EP.robust_vcov(X, u, NeweyWestEstimator())
        @test size(V_nw) == (3, 3)

        V_w = MEM_EP.robust_vcov(X, u, WhiteEstimator())
        @test size(V_w) == (3, 3)

        V_dk = MEM_EP.robust_vcov(X, u, DriscollKraayEstimator())
        @test size(V_dk) == (3, 3)

        # Multivariate dispatch
        U = randn(50, 2)
        V_nw2 = MEM_EP.robust_vcov(X, U, NeweyWestEstimator())
        @test size(V_nw2) == (6, 6)
        V_w2 = MEM_EP.robust_vcov(X, U, WhiteEstimator())
        @test size(V_w2) == (6, 6)
        V_dk2 = MEM_EP.robust_vcov(X, U, DriscollKraayEstimator())
        @test size(V_dk2) == (6, 6)
    end

    # =========================================================================
    # var/identification.jl — compute_Q errors
    # =========================================================================

    @testset "compute_Q error paths" begin
        Random.seed!(8010)
        Y = randn(100, 3)
        model = estimate_var(Y, 2)
        # Unknown method
        @test_throws Union{ArgumentError, ErrorException} MEM_EP.compute_Q(model, :nonexistent_method, 10, nothing, nothing)
    end

    # =========================================================================
    # ARIMA errors
    # =========================================================================

    @testset "ARIMA validation" begin
        # Too short series
        @test_throws ArgumentError estimate_ar(ones(5), 1)
        # Negative orders (via estimate_arima)
        @test_throws ArgumentError estimate_arima(randn(100), -1, 0, 0)
        @test_throws ArgumentError estimate_arima(randn(100), 0, -1, 0)
        @test_throws ArgumentError estimate_arima(randn(100), 0, 0, -1)
    end

    @testset "Differencing" begin
        y = [1.0, 3.0, 6.0, 10.0, 15.0]
        # d=0: no change
        @test MEM_EP._difference(y, 0) == y
        # d=1: first differences
        d1 = MEM_EP._difference(y, 1)
        @test d1 ≈ [2.0, 3.0, 4.0, 5.0]
        # d=2: double differencing
        d2 = MEM_EP._difference(y, 2)
        @test d2 ≈ [1.0, 1.0, 1.0]
    end

    @testset "ARIMA forecast h<1" begin
        Random.seed!(8020)
        y = randn(100)
        m = estimate_ar(y, 1)
        @test_throws ArgumentError forecast(m, 0)
    end

    # =========================================================================
    # Factor model errors
    # =========================================================================

    @testset "Factor model validation" begin
        Random.seed!(8030)
        X = randn(50, 10)
        # r too large
        @test_throws ArgumentError estimate_factors(X, 51)
        # r = 0
        @test_throws ArgumentError estimate_factors(X, 0)
    end

    # =========================================================================
    # @float_fallback
    # =========================================================================

    @testset "@float_fallback with Int matrix" begin
        Y_int = [1 2; 3 4; 5 6; 7 8; 9 10; 11 12; 13 14; 15 16; 17 18; 19 20]
        Y_big = vcat(Y_int, Y_int, Y_int, Y_int, Y_int)  # 50 obs
        model = estimate_var(Y_big, 1)
        @test model isa VARModel{Float64}
    end

    # =========================================================================
    # companion_matrix and extract_ar_coefficients
    # =========================================================================

    @testset "companion_matrix" begin
        Random.seed!(8040)
        Y = randn(100, 3)
        model = estimate_var(Y, 2)
        F = MEM_EP.companion_matrix(model.B, 3, 2)
        @test size(F) == (6, 6)
        # For stable VAR, all eigenvalues should have modulus < 1
        evals = eigvals(F)
        @test all(abs.(evals) .< 2.0)  # loose bound for random data

        # p=1 case
        model1 = estimate_var(Y, 1)
        F1 = MEM_EP.companion_matrix(model1.B, 3, 1)
        @test size(F1) == (3, 3)
    end

    @testset "extract_ar_coefficients" begin
        Random.seed!(8041)
        Y = randn(100, 3)
        model = estimate_var(Y, 2)
        coeffs = MEM_EP.extract_ar_coefficients(model.B, 3, 2)
        @test length(coeffs) == 2
        @test all(size(A) == (3, 3) for A in coeffs)
    end

    @testset "univariate_ar_variance" begin
        Random.seed!(8042)
        y = randn(100)
        v = MEM_EP.univariate_ar_variance(y)
        @test v > 0
        # Short vector
        v_short = MEM_EP.univariate_ar_variance([1.0, 2.0])
        @test v_short > 0
    end
end
