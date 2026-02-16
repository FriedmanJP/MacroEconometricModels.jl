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

using MacroEconometricModels
using Test
using Random
using LinearAlgebra
using Statistics

@testset "LP Forecasting" begin
    Random.seed!(123)

    # Generate test data
    T_obs = 200
    n = 3
    Y = zeros(T_obs, n)
    Y[1, :] = randn(n)
    for t in 2:T_obs
        Y[t, :] = 0.3 * Y[t-1, :] + randn(n)
    end

    H = 8
    lp = estimate_lp(Y, 1, H; lags=4)

    # =========================================================================
    @testset "Basic forecast with analytical CIs" begin
        shock_path = ones(H)
        fc = forecast(lp, shock_path; ci_method=:analytical)

        @test fc isa LPForecast{Float64}
        @test size(fc.forecasts) == (H, n)
        @test size(fc.ci_lower) == (H, n)
        @test size(fc.ci_upper) == (H, n)
        @test size(fc.se) == (H, n)
        @test fc.horizon == H
        @test fc.shock_var == 1
        @test fc.ci_method == :analytical
        @test fc.conf_level ≈ 0.95

        # All values finite
        @test all(isfinite, fc.forecasts)
        @test all(isfinite, fc.ci_lower)
        @test all(isfinite, fc.ci_upper)
        @test all(isfinite, fc.se)

        # SE should be non-negative
        @test all(fc.se .>= 0)

        # CI lower < forecast < CI upper (for analytical with positive SE)
        for h in 1:H, j in 1:n
            if fc.se[h, j] > 0
                @test fc.ci_lower[h, j] < fc.forecasts[h, j]
                @test fc.forecasts[h, j] < fc.ci_upper[h, j]
            end
        end
    end

    # =========================================================================
    @testset "Forecast with no CIs" begin
        shock_path = ones(H)
        fc = forecast(lp, shock_path; ci_method=:none)

        @test fc.ci_method == :none
        # ci_lower == ci_upper == forecasts when no CIs
        @test fc.ci_lower == fc.forecasts
        @test fc.ci_upper == fc.forecasts

        @test all(isfinite, fc.forecasts)
    end

    # =========================================================================
    @testset "Forecast with bootstrap CIs" begin
        shock_path = ones(H)
        fc = forecast(lp, shock_path; ci_method=:bootstrap, n_boot=100)

        @test fc.ci_method == :bootstrap
        @test all(isfinite, fc.ci_lower)
        @test all(isfinite, fc.ci_upper)
        @test all(isfinite, fc.se)

        # CI bounds should be ordered
        for h in 1:H, j in 1:n
            @test fc.ci_lower[h, j] <= fc.ci_upper[h, j]
        end
    end

    # =========================================================================
    @testset "Zero shock path = unconditional forecast" begin
        zero_path = zeros(H)
        fc_zero = forecast(lp, zero_path; ci_method=:none)

        nonzero_path = ones(H)
        fc_nonzero = forecast(lp, nonzero_path; ci_method=:none)

        # Forecasts should differ when shock path differs
        @test fc_zero.forecasts != fc_nonzero.forecasts
    end

    # =========================================================================
    @testset "Shock path of different magnitudes" begin
        path1 = ones(H)
        path2 = 2.0 * ones(H)

        fc1 = forecast(lp, path1; ci_method=:none)
        fc2 = forecast(lp, path2; ci_method=:none)

        # fc2 forecasts should differ from fc1 (linearity in shock)
        @test fc1.forecasts != fc2.forecasts
    end

    # =========================================================================
    @testset "Forecast from StructuralLP" begin
        slp = structural_lp(Y, H; method=:cholesky, lags=4)

        shock_path = ones(H)
        fc = forecast(slp, 1, shock_path; ci_method=:analytical)

        @test fc isa LPForecast{Float64}
        @test size(fc.forecasts) == (H, n)
        @test all(isfinite, fc.forecasts)

        # Test different shock indices
        for j in 1:n
            fc_j = forecast(slp, j, shock_path; ci_method=:none)
            @test size(fc_j.forecasts) == (H, n)
        end
    end

    # =========================================================================
    @testset "Forecast from StructuralLP with bootstrap" begin
        slp = structural_lp(Y, H; method=:cholesky, lags=4)

        shock_path = ones(H)
        fc = forecast(slp, 1, shock_path; ci_method=:bootstrap, n_boot=50)

        @test fc.ci_method == :bootstrap
        @test all(isfinite, fc.ci_lower)
        @test all(isfinite, fc.ci_upper)
    end

    # =========================================================================
    @testset "Forecast shock_idx bounds check" begin
        slp = structural_lp(Y, H; method=:cholesky, lags=4)
        shock_path = ones(H)

        @test_throws AssertionError forecast(slp, 0, shock_path)
        @test_throws AssertionError forecast(slp, n+1, shock_path)
    end

    # =========================================================================
    @testset "Shock path length check" begin
        @test_throws AssertionError forecast(lp, ones(H+1))
        @test_throws AssertionError forecast(lp, ones(H-1))
    end

    # =========================================================================
    @testset "show method for LPForecast" begin
        shock_path = ones(H)
        fc = forecast(lp, shock_path; ci_method=:analytical)

        buf = IOBuffer()
        show(buf, fc)
        output = String(take!(buf))
        @test occursin("LP Forecast", output)
        @test occursin("analytical", output)
    end

    # =========================================================================
    @testset "show method for LPForecast with no CIs" begin
        shock_path = ones(H)
        fc = forecast(lp, shock_path; ci_method=:none)

        buf = IOBuffer()
        show(buf, fc)
        output = String(take!(buf))
        @test occursin("LP Forecast", output)
        @test occursin("none", output)
    end

    # =========================================================================
    @testset "print_table for LPForecast" begin
        shock_path = ones(H)
        fc = forecast(lp, shock_path; ci_method=:analytical)

        buf = IOBuffer()
        print_table(buf, fc)
        output = String(take!(buf))
        @test occursin("LP Forecast", output)
    end

    # =========================================================================
    @testset "Different confidence levels" begin
        shock_path = ones(H)
        fc_90 = forecast(lp, shock_path; ci_method=:analytical, conf_level=0.90)
        fc_99 = forecast(lp, shock_path; ci_method=:analytical, conf_level=0.99)

        # Same point forecasts
        @test fc_90.forecasts ≈ fc_99.forecasts

        # 99% CI should be wider than 90% CI
        for h in 1:H, j in 1:n
            width_90 = fc_90.ci_upper[h, j] - fc_90.ci_lower[h, j]
            width_99 = fc_99.ci_upper[h, j] - fc_99.ci_lower[h, j]
            if width_90 > 0
                @test width_99 >= width_90 - 1e-10
            end
        end
    end

    # =========================================================================
    @testset "Horizon 1" begin
        lp_h1 = estimate_lp(Y, 1, 1; lags=4)
        shock_path = [1.0]
        fc = forecast(lp_h1, shock_path; ci_method=:analytical)

        @test fc.horizon == 1
        @test size(fc.forecasts) == (1, n)
    end
end
