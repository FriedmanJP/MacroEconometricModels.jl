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

const MEM_IH = MacroEconometricModels

@testset "Internal Helpers" begin

    # =========================================================================
    # ARIMA helpers (src/arima/)
    # =========================================================================

    @testset "ARIMA _count_params" begin
        @test MEM_IH._count_params(1, 0) == 3   # intercept + 1 AR + sigma2
        @test MEM_IH._count_params(2, 1) == 5   # intercept + 2 AR + 1 MA + sigma2
        @test MEM_IH._count_params(0, 0) == 2   # intercept + sigma2
        @test MEM_IH._count_params(1, 0; include_intercept=false) == 2  # 1 AR + sigma2
    end

    @testset "ARIMA pack/unpack roundtrip" begin
        c = 0.5
        phi = [0.7, -0.2]
        theta = [0.3]

        # With intercept
        packed = MEM_IH._pack_arma_params(c, phi, theta)
        @test packed == [0.5, 0.7, -0.2, 0.3]

        # Without intercept
        packed_no = MEM_IH._pack_arma_params(c, phi, theta; include_intercept=false)
        @test packed_no == [0.7, -0.2, 0.3]

        # With log_sigma2
        packed_s = MEM_IH._pack_arma_params(c, phi, theta; log_sigma2=log(0.5))
        @test length(packed_s) == 5
    end

    @testset "ARIMA _compute_aic_bic" begin
        loglik = -100.0
        k = 3
        n = 100
        aic, bic = MEM_IH._compute_aic_bic(loglik, k, n)
        @test aic ≈ 206.0
        @test bic ≈ -2 * loglik + k * log(100.0)
    end

    @testset "ARIMA _roots_inside_unit_circle" begin
        # Empty: always true
        @test MEM_IH._roots_inside_unit_circle(Float64[]) == true
        # Single AR coeff < 1: stationary
        @test MEM_IH._roots_inside_unit_circle([0.5]) == true
        # Single AR coeff > 1: not stationary
        @test MEM_IH._roots_inside_unit_circle([1.5]) == false
        # AR(2): [0.5, 0.3] — stable
        @test MEM_IH._roots_inside_unit_circle([0.5, 0.3]) == true
        # AR(2): [1.5, 0.0] — not stable
        @test MEM_IH._roots_inside_unit_circle([1.5, 0.0]) == false
    end

    @testset "ARIMA _is_stationary/_is_invertible aliases" begin
        @test MEM_IH._is_stationary([0.5]) == true
        @test MEM_IH._is_stationary([1.5]) == false
        @test MEM_IH._is_invertible([0.3, 0.2]) == true
    end

    @testset "ARIMA _truncate_to_stable" begin
        # Already stable: no change
        stable = [0.5, 0.2]
        truncated = MEM_IH._truncate_to_stable(stable)
        @test truncated ≈ stable

        # Unstable: should truncate
        unstable = [1.5, 0.0]
        truncated2 = MEM_IH._truncate_to_stable(unstable)
        @test MEM_IH._roots_inside_unit_circle(truncated2)
    end

    @testset "ARIMA _white_noise_fit" begin
        Random.seed!(9001)
        y = randn(100) .+ 2.0
        c, sigma2, loglik, residuals, fitted = MEM_IH._white_noise_fit(y)
        @test c ≈ mean(y)
        @test sigma2 ≈ var(y; corrected=false) atol = 1e-10
        @test isfinite(loglik)
        @test length(residuals) == 100
        @test all(fitted .≈ c)

        # Without intercept
        c0, sigma2_0, _, _, _ = MEM_IH._white_noise_fit(y; include_intercept=false)
        @test c0 == 0.0
    end

    @testset "ARIMA _confidence_band" begin
        forecasts = [1.0, 2.0, 3.0]
        se = [0.5, 0.6, 0.7]
        lower, upper = MEM_IH._confidence_band(forecasts, se, 0.95)
        @test length(lower) == 3
        @test length(upper) == 3
        @test all(lower .< forecasts)
        @test all(upper .> forecasts)
        # 99% CI should be wider than 95%
        lower99, upper99 = MEM_IH._confidence_band(forecasts, se, 0.99)
        @test all(upper99 .- lower99 .> upper .- lower)
    end

    @testset "ARIMA state space construction" begin
        # AR(1)
        c = 0.5
        phi = [0.7]
        theta = Float64[]
        sigma2 = 1.0
        Z, T_mat, R, Q, H, r = MEM_IH._arma_state_space(c, phi, theta, sigma2, 1, 0)
        @test r == 1
        @test Z[1, 1] == 1.0
        @test T_mat[1, 1] == 0.7

        # ARMA(1,1)
        theta_11 = [0.3]
        Z2, T2, R2, Q2, H2, r2 = MEM_IH._arma_state_space(c, phi, theta_11, sigma2, 1, 1)
        @test r2 == 2
        @test Z2[1, 2] == 0.3
    end

    # =========================================================================
    # Display utilities (src/core/display.jl)
    # =========================================================================

    @testset "Display formatting helpers" begin
        # _fmt returns a number (rounded)
        @test MEM_IH._fmt(3.14159) isa Real
        @test MEM_IH._fmt(3.14159; digits=2) ≈ 3.14

        # _fmt_pct returns a string with %
        @test occursin("%", MEM_IH._fmt_pct(0.5))

        # _format_pvalue
        pv_str = MEM_IH._format_pvalue(0.0001)
        @test pv_str == "<0.001"
        pv_str2 = MEM_IH._format_pvalue(0.5)
        @test pv_str2 isa String
        pv_str3 = MEM_IH._format_pvalue(0.9999)
        @test pv_str3 == ">0.999"

        # _significance_stars
        @test MEM_IH._significance_stars(0.001) == "***"
        @test MEM_IH._significance_stars(0.04) == "**"
        @test MEM_IH._significance_stars(0.08) == "*"
        @test MEM_IH._significance_stars(0.5) == ""
    end

    @testset "Display backend management" begin
        # Save current backend
        orig = MEM_IH.get_display_backend()
        @test orig == :text

        # Switch to LaTeX and back
        MEM_IH.set_display_backend(:latex)
        @test MEM_IH.get_display_backend() == :latex
        MEM_IH.set_display_backend(:html)
        @test MEM_IH.get_display_backend() == :html

        # Reset
        MEM_IH.set_display_backend(:text)
        @test MEM_IH.get_display_backend() == :text
    end

    # =========================================================================
    # VAR construction helpers
    # =========================================================================

    @testset "construct_var_matrices" begin
        Random.seed!(9010)
        Y = randn(50, 3)
        Y_eff, X = MEM_IH.construct_var_matrices(Y, 2)
        @test size(Y_eff) == (48, 3)
        @test size(X) == (48, 1 + 3 * 2)  # intercept + n*p

        # Integer input auto-converts
        Y_int = ones(Int, 50, 3)
        Y_eff_int, X_int = MEM_IH.construct_var_matrices(Y_int, 1)
        @test eltype(Y_eff_int) == Float64
    end

    # =========================================================================
    # Kalman filter helpers (src/arima/kalman.jl)
    # =========================================================================

    @testset "Kalman filter ARMA" begin
        Random.seed!(9020)
        y = randn(100)
        c = 0.0
        phi = [0.5]
        theta = Float64[]
        sigma2 = 1.0
        loglik, residuals, fitted = MEM_IH._kalman_filter_arma(y, c, phi, theta, sigma2)
        @test isfinite(loglik)
        @test length(residuals) == 100
        @test length(fitted) == 100
    end

    # =========================================================================
    # _select_horizons (display utility)
    # =========================================================================

    @testset "_select_horizons" begin
        # Default horizons
        h_default = MEM_IH._select_horizons(20)
        @test h_default isa Vector{Int}
        @test all(h .<= 20 for h in h_default)

        # Small H
        h_small = MEM_IH._select_horizons(3)
        @test h_small == [1, 2, 3]

        # Large H
        h_large = MEM_IH._select_horizons(30)
        @test 24 in h_large
        @test 30 in h_large
    end

    # =========================================================================
    # _matrix_table (display utility)
    # =========================================================================

    @testset "_matrix_table" begin
        buf = IOBuffer()
        M = [1.0 2.0; 3.0 4.0]
        MEM_IH._matrix_table(buf, M, "Test Matrix")
        output = String(take!(buf))
        @test length(output) > 0
    end

    # =========================================================================
    # Optimal bandwidth
    # =========================================================================

    @testset "optimal_bandwidth_nw" begin
        Random.seed!(9030)
        x = randn(100)
        bw = MEM_IH.optimal_bandwidth_nw(x)
        @test bw >= 0
        @test bw <= 100

        # Short vector
        bw_short = MEM_IH.optimal_bandwidth_nw(randn(3))
        @test bw_short == 0

        # Multivariate
        X = randn(100, 3)
        bw_multi = MEM_IH.optimal_bandwidth_nw(X)
        @test bw_multi >= 0

        # Empty multivariate
        X_empty = randn(100, 0)
        bw_empty = MEM_IH.optimal_bandwidth_nw(X_empty)
        @test bw_empty == 0
    end
end
