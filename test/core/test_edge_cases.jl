# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <wookyung9207@gmail.com>
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
Edge case tests for MacroEconometricModels.

Tests boundary conditions and numerical edge cases:
1. Univariate models (n=1)
2. Boundary horizons (h=1, h=large)
3. Numerical stability (near-singular covariance, near unit root)
4. Factor model edge cases (minimal factors, factors = variables)
"""

using Test
using MacroEconometricModels
using LinearAlgebra
using Random

Random.seed!(12345)

@testset "Edge Cases" begin

    # =========================================================================
    # 1. Univariate Models (n=1)
    # =========================================================================
    @testset "Univariate VAR" begin
        Y = randn(100, 1)
        model = estimate_var(Y, 2)

        @test nvars(model) == 1
        @test size(model.Sigma) == (1, 1)

        # IRF for univariate should work
        irf_result = irf(model, 10)
        @test size(irf_result.values) == (10, 1, 1)

        # FEVD for univariate: single shock = 100%
        fevd_result = fevd(model, 10)
        @test size(fevd_result.proportions) == (1, 1, 10)
        @test all(fevd_result.proportions .≈ 1.0)  # Single shock explains everything

        # Historical decomposition for univariate
        hd = historical_decomposition(model, 98)
        @test size(hd.contributions) == (98, 1, 1)
        @test verify_decomposition(hd)
    end

    # =========================================================================
    # 2. Boundary Horizons
    # =========================================================================
    @testset "Boundary Horizons" begin
        Y = randn(100, 2)
        model = estimate_var(Y, 2)

        # Minimal horizon (h=1)
        irf1 = irf(model, 1)
        @test size(irf1.values, 1) == 1

        fevd1 = fevd(model, 1)
        @test size(fevd1.proportions, 3) == 1
        # At h=1, FEVD should sum to 1 for each variable
        for i in 1:2
            @test sum(fevd1.proportions[i, :, 1]) ≈ 1.0 atol=1e-10
        end

        # Large horizon (h=100) - tests numerical stability
        irf100 = irf(model, 100)
        @test size(irf100.values, 1) == 100
        # IRF should decay for stationary model
        # Check stationarity via companion matrix eigenvalues
        companion = companion_matrix(model.B, 2, 2)
        max_eig = maximum(abs.(eigvals(companion)))
        if max_eig < 1.0
            @test maximum(abs.(irf100.values[100, :, :])) < maximum(abs.(irf100.values[1, :, :]))
        end

        fevd100 = fevd(model, 100)
        @test size(fevd100.proportions, 3) == 100
        # FEVD proportions should still sum to 1 at large horizons
        for h in [1, 50, 100]
            for i in 1:2
                @test sum(fevd100.proportions[i, :, h]) ≈ 1.0 atol=1e-10
            end
        end
    end

    @testset "Very Short Horizon for HD" begin
        Y = randn(50, 2)
        model = estimate_var(Y, 1)

        # Minimal effective sample
        hd = historical_decomposition(model, 10)
        @test verify_decomposition(hd)
    end

    # =========================================================================
    # 3. Numerical Stability
    # =========================================================================
    @testset "Near-Collinear Variables" begin
        # Create nearly collinear data
        Y = randn(100, 3)
        Y[:, 3] = Y[:, 1] + 0.001 * randn(100)  # Variable 3 almost equals variable 1

        model = estimate_var(Y, 1)

        # Should still work (with possible warnings about singular matrices)
        @test_nowarn begin
            irf_result = irf(model, 10)
            @test size(irf_result.values) == (10, 3, 3)
        end
    end

    @testset "Near Unit Root" begin
        # Generate highly persistent data
        Y_persistent = zeros(200, 2)
        for t in 2:200
            Y_persistent[t, :] = 0.99 * Y_persistent[t-1, :] + 0.1 * randn(2)
        end

        model_persistent = estimate_var(Y_persistent, 1)

        # Check eigenvalues are close to but not exceeding unit circle
        companion = companion_matrix(model_persistent.B, 2, 1)
        max_eig = maximum(abs.(eigvals(companion)))
        @test max_eig < 1.05  # Allow small numerical imprecision

        # FEVD should still work
        @test_nowarn begin
            fevd_result = fevd(model_persistent, 20)
            # Proportions should still sum to 1
            for h in 1:20
                for i in 1:2
                    @test sum(fevd_result.proportions[i, :, h]) ≈ 1.0 atol=1e-8
                end
            end
        end
    end

    @testset "Zero Residual Variance" begin
        # Deterministic process: Y_t = 0.5 * Y_{t-1} exactly
        Y_det = zeros(100, 2)
        Y_det[1, :] = [1.0, 1.0]
        for t in 2:100
            Y_det[t, :] = 0.5 * Y_det[t-1, :] + 1e-10 * randn(2)  # Tiny noise
        end

        model = estimate_var(Y_det, 1)

        # Should handle near-zero variance gracefully
        @test_nowarn begin
            irf_result = irf(model, 10)
            # IRF should still have reasonable values
            @test all(isfinite.(irf_result.values))
        end
    end

    # =========================================================================
    # 4. Factor Model Edge Cases
    # =========================================================================
    @testset "Single Factor" begin
        X = randn(100, 10)
        fm = estimate_factors(X, 1)

        @test size(fm.factors, 2) == 1
        @test fm.r == 1

        # R² should be non-negative
        r2_vals = r2(fm)
        @test all(r2_vals .>= 0.0)
        @test all(r2_vals .<= 1.0)
    end

    @testset "Maximum Factors (r = min(T, N))" begin
        X = randn(50, 10)  # T < N case

        # r = T should be allowed (up to numerical precision issues)
        r_max = min(size(X)...)

        # Near-maximum factors
        fm_nearmax = estimate_factors(X, r_max - 1)
        @test size(fm_nearmax.factors, 2) == r_max - 1

        # With near-maximum factors, should explain most variance
        @test all(r2(fm_nearmax) .> 0.8)
    end

    @testset "Factors Equal Variables" begin
        X = randn(100, 5)

        # When r = N, should explain all variance
        fm_full = estimate_factors(X, 5)
        @test fm_full.r == 5

        # R² should be very high for all variables
        r2_vals = r2(fm_full)
        @test all(r2_vals .> 0.95)
    end

    @testset "Dynamic Factor Model Minimal Settings" begin
        X = randn(100, 10)

        # Minimal factors and lags
        dfm = estimate_dynamic_factors(X, 1, 1)
        @test dfm.r == 1
        @test dfm.p == 1

        # Should be able to forecast
        fc = forecast(dfm, 5)
        @test size(fc.observables) == (5, 10)
    end

    @testset "GDFM with q=1" begin
        X = randn(100, 10)

        gdfm = estimate_gdfm(X, 1)
        @test gdfm.q == 1

        # Variance share should be non-negative
        shares = common_variance_share(gdfm)
        @test all(shares .>= 0.0)
        @test all(shares .<= 1.0)
    end

    # =========================================================================
    # 5. Local Projection Edge Cases
    # =========================================================================
    @testset "LP Minimal Horizon" begin
        Y = randn(100, 2)

        # Minimal horizon h=1 (h=0 not allowed - horizon must be positive)
        lp_model = estimate_lp(Y, 1, 1; lags=2)
        @test lp_model.horizon == 1

        lp_result = lp_irf(lp_model)
        @test size(lp_result.values, 1) == 2  # h=0 and h=1
    end

    @testset "LP Large Horizon" begin
        Y = randn(200, 2)

        # Test with larger horizon relative to sample
        lp_model = estimate_lp(Y, 1, 50; lags=2)
        @test lp_model.horizon == 50

        lp_result = lp_irf(lp_model)
        @test size(lp_result.values, 1) == 51  # h=0 to h=50
    end

    # =========================================================================
    # 6. Unified Interface Edge Cases
    # =========================================================================
    @testset "Unified Interface" begin
        Y = randn(100, 2)
        model = estimate_var(Y, 2)

        # Test point_estimate returns correct types
        irf_result = irf(model, 10)
        @test point_estimate(irf_result) === irf_result.values

        fevd_result = fevd(model, 10)
        @test point_estimate(fevd_result) === fevd_result.proportions

        hd = historical_decomposition(model, 98)
        @test point_estimate(hd) === hd.contributions

        # Test has_uncertainty
        @test has_uncertainty(irf_result) == (irf_result.ci_type != :none)
        @test has_uncertainty(fevd_result) == false
        @test has_uncertainty(hd) == false

        # Test uncertainty_bounds returns nothing when no uncertainty
        if !has_uncertainty(irf_result)
            @test uncertainty_bounds(irf_result) === nothing
        end
        @test uncertainty_bounds(fevd_result) === nothing
        @test uncertainty_bounds(hd) === nothing
    end

    # =========================================================================
    # 7. Sign Restriction Edge Cases
    # =========================================================================
    @testset "Sign Restrictions Empty" begin
        Y = randn(100, 2)
        model = estimate_var(Y, 2)

        # Empty sign restrictions = Cholesky identification
        r = SVARRestrictions(2; signs=SignRestriction[], zeros=ZeroRestriction[])
        result = identify_arias(model, r, 10; n_draws=50)

        @test length(result.Q_draws) >= 1
        @test result.acceptance_rate > 0.0
    end

end
