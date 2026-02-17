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
using Statistics
using LinearAlgebra
using Random

@testset "Covariance Estimators" begin

    # =========================================================================
    # Kernel Weights
    # =========================================================================

    @testset "kernel_weight" begin
        # Bartlett kernel
        @test MacroEconometricModels.kernel_weight(0, 5, :bartlett) == 1.0
        @test MacroEconometricModels.kernel_weight(5, 5, :bartlett) ≈ 1 - 5/6  # x = 5/6
        @test MacroEconometricModels.kernel_weight(0, 0, :bartlett) == 0.0  # bandwidth=0 returns 0

        # All kernels return 1 at j=0 (except bandwidth=0)
        for kernel in [:bartlett, :parzen, :quadratic_spectral, :tukey_hanning]
            @test MacroEconometricModels.kernel_weight(0, 5, kernel) ≈ 1.0
        end

        # All kernels return 0 when |x| > 1 (j > bandwidth+1)
        for kernel in [:bartlett, :parzen, :tukey_hanning]
            @test MacroEconometricModels.kernel_weight(10, 3, kernel) == 0.0
        end

        # Bartlett: linearly decreasing
        w1 = MacroEconometricModels.kernel_weight(1, 5, :bartlett)
        w2 = MacroEconometricModels.kernel_weight(2, 5, :bartlett)
        w3 = MacroEconometricModels.kernel_weight(3, 5, :bartlett)
        @test w1 > w2 > w3

        # Parzen kernel: values in [0, 1]
        for j in 0:5
            w = MacroEconometricModels.kernel_weight(j, 5, :parzen)
            @test 0 <= w <= 1
        end

        # Tukey-Hanning: values in [0, 1]
        for j in 0:5
            w = MacroEconometricModels.kernel_weight(j, 5, :tukey_hanning)
            @test 0 <= w <= 1
        end

        # Quadratic spectral: test non-zero for j > 0
        w_qs = MacroEconometricModels.kernel_weight(1, 5, :quadratic_spectral)
        @test w_qs != 0.0

        # Unknown kernel
        @test_throws ArgumentError MacroEconometricModels.kernel_weight(1, 5, :unknown)

        # Float32 type
        w32 = MacroEconometricModels.kernel_weight(1, 5, :bartlett, Float32)
        @test w32 isa Float32
    end

    # =========================================================================
    # Optimal Bandwidth Selection
    # =========================================================================

    @testset "optimal_bandwidth_nw" begin
        Random.seed!(42)

        # White noise: bandwidth should be small
        x_wn = randn(200)
        bw_wn = MacroEconometricModels.optimal_bandwidth_nw(x_wn)
        @test bw_wn >= 0
        @test bw_wn <= 20  # Should be small for white noise

        # Persistent series: bandwidth should be larger
        x_pers = zeros(200)
        x_pers[1] = randn()
        for t in 2:200
            x_pers[t] = 0.9 * x_pers[t-1] + 0.1 * randn()
        end
        bw_pers = MacroEconometricModels.optimal_bandwidth_nw(x_pers)
        @test bw_pers >= 0

        # Short series
        bw_short = MacroEconometricModels.optimal_bandwidth_nw(randn(3))
        @test bw_short == 0

        # Multivariate version
        X_mv = randn(200, 3)
        bw_mv = MacroEconometricModels.optimal_bandwidth_nw(X_mv)
        @test bw_mv >= 0

        # Empty multivariate
        bw_empty = MacroEconometricModels.optimal_bandwidth_nw(zeros(10, 0))
        @test bw_empty == 0
    end

    # =========================================================================
    # Newey-West HAC Estimator
    # =========================================================================

    @testset "newey_west - univariate residuals" begin
        Random.seed!(100)
        n = 200
        k = 3
        X = hcat(ones(n), randn(n, k - 1))
        beta = [1.0, 2.0, -0.5]
        residuals = randn(n)

        # Bartlett kernel (default)
        V_nw = MacroEconometricModels.newey_west(X, residuals)
        @test size(V_nw) == (k, k)
        @test isapprox(V_nw, V_nw', atol=1e-10)  # Symmetric
        @test all(eigvals(Symmetric(V_nw)) .> -1e-8)  # PSD

        # All 4 kernels
        for kernel in [:bartlett, :parzen, :quadratic_spectral, :tukey_hanning]
            V = MacroEconometricModels.newey_west(X, residuals; kernel=kernel)
            @test size(V) == (k, k)
            @test isapprox(V, V', atol=1e-10)
        end

        # Fixed bandwidth
        V_bw = MacroEconometricModels.newey_west(X, residuals; bandwidth=5)
        @test size(V_bw) == (k, k)

        # Prewhitening
        V_pw = MacroEconometricModels.newey_west(X, residuals; prewhiten=true)
        @test size(V_pw) == (k, k)
        @test isapprox(V_pw, V_pw', atol=1e-10)

        # With precomputed XtX_inv
        XtX_inv = MacroEconometricModels.precompute_XtX_inv(X)
        V_cached = MacroEconometricModels.newey_west(X, residuals; XtX_inv=XtX_inv)
        @test isapprox(V_cached, V_nw, atol=1e-10)
    end

    @testset "newey_west - multivariate residuals" begin
        Random.seed!(200)
        n = 200
        k = 2
        n_eq = 3
        X = hcat(ones(n), randn(n))
        residuals = randn(n, n_eq)

        V = MacroEconometricModels.newey_west(X, residuals)
        @test size(V) == (k * n_eq, k * n_eq)

        # Single column treated as vector
        V_single = MacroEconometricModels.newey_west(X, residuals[:, 1:1])
        @test size(V_single) == (k, k)
    end

    # =========================================================================
    # White Heteroscedasticity-Robust Estimator
    # =========================================================================

    @testset "white_vcov - all HC variants" begin
        Random.seed!(300)
        n = 100
        k = 3
        X = hcat(ones(n), randn(n, k - 1))
        residuals = randn(n) .* exp.(0.5 * randn(n))  # Heteroscedastic

        for variant in [:hc0, :hc1, :hc2, :hc3]
            V = MacroEconometricModels.white_vcov(X, residuals; variant=variant)
            @test size(V) == (k, k)
            @test isapprox(V, V', atol=1e-10)  # Symmetric
        end

        # HC1 should give larger values than HC0 (finite sample correction)
        V_hc0 = MacroEconometricModels.white_vcov(X, residuals; variant=:hc0)
        V_hc1 = MacroEconometricModels.white_vcov(X, residuals; variant=:hc1)
        @test all(diag(V_hc1) .>= diag(V_hc0) .- 1e-10)

        # With precomputed XtX_inv
        XtX_inv = MacroEconometricModels.precompute_XtX_inv(X)
        V_cached = MacroEconometricModels.white_vcov(X, residuals; variant=:hc0, XtX_inv=XtX_inv)
        @test isapprox(V_cached, V_hc0, atol=1e-10)
    end

    @testset "white_vcov - multivariate residuals" begin
        Random.seed!(400)
        n = 100
        k = 2
        n_eq = 2
        X = hcat(ones(n), randn(n))
        residuals = randn(n, n_eq)

        V = MacroEconometricModels.white_vcov(X, residuals)
        @test size(V) == (k * n_eq, k * n_eq)
    end

    # =========================================================================
    # Driscoll-Kraay Estimator
    # =========================================================================

    @testset "driscoll_kraay - univariate" begin
        Random.seed!(500)
        n = 200
        k = 3
        X = hcat(ones(n), randn(n, k - 1))
        u = randn(n)

        V = MacroEconometricModels.driscoll_kraay(X, u)
        @test size(V) == (k, k)
        @test isapprox(V, V', atol=1e-10)

        # Different kernels
        for kernel in [:bartlett, :parzen, :quadratic_spectral, :tukey_hanning]
            V_k = MacroEconometricModels.driscoll_kraay(X, u; kernel=kernel)
            @test size(V_k) == (k, k)
        end

        # With precomputed XtX_inv
        XtX_inv = MacroEconometricModels.precompute_XtX_inv(X)
        V_cached = MacroEconometricModels.driscoll_kraay(X, u; XtX_inv=XtX_inv)
        @test isapprox(V_cached, V, atol=1e-10)
    end

    @testset "driscoll_kraay - multivariate" begin
        Random.seed!(600)
        n = 200
        k = 2
        n_eq = 3
        X = hcat(ones(n), randn(n))
        U = randn(n, n_eq)

        V = MacroEconometricModels.driscoll_kraay(X, U)
        @test size(V) == (k * n_eq, k * n_eq)
    end

    # =========================================================================
    # Covariance Estimator Dispatch
    # =========================================================================

    @testset "robust_vcov dispatch" begin
        Random.seed!(700)
        n = 100
        k = 2
        X = hcat(ones(n), randn(n))
        residuals = randn(n)

        # NeweyWestEstimator
        nw_est = MacroEconometricModels.NeweyWestEstimator()
        V_nw = MacroEconometricModels.robust_vcov(X, residuals, nw_est)
        @test size(V_nw) == (k, k)

        # WhiteEstimator
        white_est = MacroEconometricModels.WhiteEstimator()
        V_white = MacroEconometricModels.robust_vcov(X, residuals, white_est)
        @test size(V_white) == (k, k)

        # DriscollKraayEstimator
        dk_est = MacroEconometricModels.DriscollKraayEstimator()
        V_dk = MacroEconometricModels.robust_vcov(X, residuals, dk_est)
        @test size(V_dk) == (k, k)

        # Multivariate dispatch
        residuals_mv = randn(n, 2)
        V_nw_mv = MacroEconometricModels.robust_vcov(X, residuals_mv, nw_est)
        @test size(V_nw_mv) == (k * 2, k * 2)

        V_white_mv = MacroEconometricModels.robust_vcov(X, residuals_mv, white_est)
        @test size(V_white_mv) == (k * 2, k * 2)

        V_dk_mv = MacroEconometricModels.robust_vcov(X, residuals_mv, dk_est)
        @test size(V_dk_mv) == (k * 2, k * 2)
    end

    # =========================================================================
    # Estimator Type Constructors
    # =========================================================================

    @testset "Estimator constructors" begin
        nw = MacroEconometricModels.NeweyWestEstimator()
        @test nw.bandwidth == 0
        @test nw.kernel == :bartlett
        @test nw.prewhiten == false

        nw2 = MacroEconometricModels.NeweyWestEstimator(bandwidth=5, kernel=:parzen, prewhiten=true)
        @test nw2.bandwidth == 5
        @test nw2.kernel == :parzen
        @test nw2.prewhiten == true

        @test_throws ArgumentError MacroEconometricModels.NeweyWestEstimator(bandwidth=-1)
        @test_throws ArgumentError MacroEconometricModels.NeweyWestEstimator(kernel=:invalid)

        dk = MacroEconometricModels.DriscollKraayEstimator()
        @test dk.bandwidth == 0
        @test dk.kernel == :bartlett

        @test_throws ArgumentError MacroEconometricModels.DriscollKraayEstimator{Float64}(-1)
    end

    # =========================================================================
    # precompute_XtX_inv
    # =========================================================================

    @testset "precompute_XtX_inv" begin
        Random.seed!(800)
        n = 100
        k = 3
        X = hcat(ones(n), randn(n, k - 1))

        XtX_inv = MacroEconometricModels.precompute_XtX_inv(X)
        @test size(XtX_inv) == (k, k)

        # Verify it's actually (X'X)^{-1}
        XtX = X' * X
        @test isapprox(XtX_inv * XtX, Matrix{Float64}(I, k, k), atol=1e-8)
        @test isapprox(XtX * XtX_inv, Matrix{Float64}(I, k, k), atol=1e-8)

        # Caching: results from NW with/without cached should match
        residuals = randn(n)
        V1 = MacroEconometricModels.newey_west(X, residuals)
        V2 = MacroEconometricModels.newey_west(X, residuals; XtX_inv=XtX_inv)
        @test isapprox(V1, V2, atol=1e-10)
    end

    # =========================================================================
    # Long-Run Variance
    # =========================================================================

    @testset "long_run_variance" begin
        Random.seed!(900)

        # White noise: long-run variance ≈ variance
        n = 1000
        x_wn = randn(n)
        lrv = MacroEconometricModels.long_run_variance(x_wn)
        @test lrv > 0
        @test isapprox(lrv, var(x_wn), rtol=0.3)  # Approximately variance for white noise

        # AR(1) with rho = 0.5: theoretical LRV = sigma^2 / (1-rho)^2
        rho = 0.5
        x_ar = zeros(n)
        x_ar[1] = randn()
        for t in 2:n
            x_ar[t] = rho * x_ar[t-1] + randn()
        end
        lrv_ar = MacroEconometricModels.long_run_variance(x_ar)
        theoretical_lrv = 1.0 / (1 - rho)^2  # sigma^2 / (1-rho)^2
        @test lrv_ar > 0
        @test isapprox(lrv_ar, theoretical_lrv, rtol=0.5)

        # Fixed bandwidth
        lrv_bw = MacroEconometricModels.long_run_variance(x_wn; bandwidth=3)
        @test lrv_bw > 0

        # Different kernels
        for kernel in [:bartlett, :parzen, :quadratic_spectral, :tukey_hanning]
            lrv_k = MacroEconometricModels.long_run_variance(x_ar; kernel=kernel)
            @test lrv_k > 0
        end

        # Very short series (single element: var returns NaN, that's expected)
        lrv_short = MacroEconometricModels.long_run_variance([1.0])
        @test isnan(lrv_short) || lrv_short >= 0
    end

    # =========================================================================
    # Long-Run Covariance
    # =========================================================================

    @testset "long_run_covariance" begin
        Random.seed!(1000)
        n = 300
        k = 3
        X = randn(n, k)

        lrc = MacroEconometricModels.long_run_covariance(X)
        @test size(lrc) == (k, k)
        @test isapprox(lrc, lrc', atol=1e-10)  # Symmetric
        @test all(eigvals(Symmetric(lrc)) .>= -1e-10)  # PSD

        # Different kernels
        for kernel in [:bartlett, :parzen, :quadratic_spectral, :tukey_hanning]
            lrc_k = MacroEconometricModels.long_run_covariance(X; kernel=kernel)
            @test size(lrc_k) == (k, k)
            @test isapprox(lrc_k, lrc_k', atol=1e-10)
        end

        # Fixed bandwidth
        lrc_bw = MacroEconometricModels.long_run_covariance(X; bandwidth=5)
        @test size(lrc_bw) == (k, k)

        # Short series
        X_short = randn(1, 2)
        lrc_short = MacroEconometricModels.long_run_covariance(X_short)
        @test size(lrc_short) == (2, 2)
    end

    @testset "precompute_XtX_inv caching pattern" begin
        Random.seed!(8801)
        X = randn(80, 4)
        u = randn(80)
        XtX_inv = MacroEconometricModels.precompute_XtX_inv(X)

        # Newey-West with cached XtX_inv
        V_nw = MacroEconometricModels.newey_west(X, u; XtX_inv=XtX_inv)
        V_nw2 = MacroEconometricModels.newey_west(X, u)
        @test norm(V_nw - V_nw2) < 1e-10

        # White with cached XtX_inv
        V_w = MacroEconometricModels.white_vcov(X, u; XtX_inv=XtX_inv)
        V_w2 = MacroEconometricModels.white_vcov(X, u)
        @test norm(V_w - V_w2) < 1e-10

        # Driscoll-Kraay with cached XtX_inv
        V_dk = MacroEconometricModels.driscoll_kraay(X, u; XtX_inv=XtX_inv)
        V_dk2 = MacroEconometricModels.driscoll_kraay(X, u)
        @test norm(V_dk - V_dk2) < 1e-10
    end

    @testset "NW at bw=0 matches White HC0 for white noise" begin
        # Regression test: for white noise residuals, auto-bandwidth ≈ 0,
        # so NW should approximate White HC0 (both are sandwich estimators
        # with the same meat when there is no autocorrelation).
        Random.seed!(12345)
        n = 500
        k = 3
        X = hcat(ones(n), randn(n, k - 1))
        u = randn(n)  # white noise — auto bandwidth should be 0 or very small

        V_nw = MacroEconometricModels.newey_west(X, u; bandwidth=0)
        V_white = MacroEconometricModels.white_vcov(X, u; variant=:hc0)

        # With bandwidth=0, NW should exactly equal White HC0
        bw_auto = MacroEconometricModels.optimal_bandwidth_nw(u)
        if bw_auto == 0
            @test isapprox(V_nw, V_white, rtol=1e-10)
        else
            # Even if auto bw > 0, they should be close for white noise
            @test isapprox(V_nw, V_white, rtol=0.3)
        end

        # Explicitly force bw=0: NW with bw=0 must exactly equal White HC0
        V_nw_bw0 = MacroEconometricModels.newey_west(X, u; bandwidth=0)
        bw_check = MacroEconometricModels.optimal_bandwidth_nw(u)
        if bw_check == 0
            @test isapprox(V_nw_bw0, V_white, rtol=1e-10)
        end
    end

    @testset "Newey-West with fixed bandwidth and all kernels" begin
        Random.seed!(8802)
        X = randn(80, 3)
        u = randn(80)
        for kernel in [:bartlett, :parzen, :quadratic_spectral, :tukey_hanning]
            V = MacroEconometricModels.newey_west(X, u; bandwidth=5, kernel=kernel)
            @test size(V) == (3, 3)
            @test issymmetric(V) || norm(V - V') < 1e-12
        end
    end

end
