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

@testset "LP-FEVD (Gorodnichenko & Lee 2019)" begin
    Random.seed!(42)

    # Generate test data with known structure
    T_obs = 200
    n = 3
    Y = zeros(T_obs, n)
    Y[1, :] = randn(n)
    for t in 2:T_obs
        Y[t, :] = 0.3 * Y[t-1, :] + randn(n)
    end

    H = 12
    slp = structural_lp(Y, H; method=:cholesky, lags=4)

    # =========================================================================
    @testset "R² estimator — point estimates only" begin
        f = lp_fevd(slp, H; method=:r2, n_boot=0)

        @test f isa LPFEVD{Float64}
        @test size(f.proportions) == (n, n, H)
        @test size(f.bias_corrected) == (n, n, H)
        @test f.method == :r2
        @test f.horizon == H
        @test f.n_boot == 0
        @test f.bias_correction == true

        # R² should be in [0, 1]
        @test all(0 .<= f.proportions .<= 1)

        # Without bootstrap, bias_corrected == proportions
        @test f.bias_corrected ≈ f.proportions

        # SEs should be zero without bootstrap
        @test all(f.se .== 0)
    end

    # =========================================================================
    @testset "R² estimator — with bootstrap" begin
        f = lp_fevd(slp, H; method=:r2, n_boot=50, bias_correct=true, conf_level=0.90)

        @test f isa LPFEVD{Float64}
        @test f.n_boot == 50
        @test f.conf_level ≈ 0.90
        @test f.bias_correction == true

        # Bias-corrected values should be in [0, 1]
        @test all(0 .<= f.bias_corrected .<= 1)

        # SEs should be non-negative
        @test all(f.se .>= 0)

        # CIs should be ordered and in [0, 1]
        for h in 1:H, i in 1:n, j in 1:n
            @test 0 <= f.ci_lower[i, j, h] <= f.ci_upper[i, j, h] <= 1
        end
    end

    # =========================================================================
    @testset "No bias correction with bootstrap" begin
        f = lp_fevd(slp, H; bias_correct=false, n_boot=50)

        @test f.bias_correction == false
        # bias_corrected should equal proportions when no correction
        @test f.bias_corrected ≈ f.proportions

        # But SEs and CIs should still be computed
        @test any(f.se .> 0)
    end

    # =========================================================================
    @testset "LP-A estimator" begin
        f = lp_fevd(slp, H; method=:lp_a, n_boot=0)

        @test f.method == :lp_a
        @test all(0 .<= f.proportions .<= 1)
        @test all(isfinite, f.proportions)
    end

    # =========================================================================
    @testset "LP-B estimator" begin
        f = lp_fevd(slp, H; method=:lp_b, n_boot=0)

        @test f.method == :lp_b
        @test all(0 .<= f.proportions .<= 1)
        @test all(isfinite, f.proportions)
    end

    # =========================================================================
    @testset "R², LP-A, LP-B produce different values" begin
        f_r2  = lp_fevd(slp, H; method=:r2,  n_boot=0)
        f_lpa = lp_fevd(slp, H; method=:lp_a, n_boot=0)
        f_lpb = lp_fevd(slp, H; method=:lp_b, n_boot=0)

        # Typically not identical (different estimators)
        @test f_r2.proportions != f_lpa.proportions || f_r2.proportions != f_lpb.proportions
    end

    # =========================================================================
    @testset "fevd() dispatches to lp_fevd()" begin
        f1 = fevd(slp, H; n_boot=0)
        f2 = lp_fevd(slp, H; n_boot=0)

        @test f1 isa LPFEVD{Float64}
        @test f1.proportions ≈ f2.proportions
    end

    # =========================================================================
    @testset "Horizon capping" begin
        f = lp_fevd(slp, H + 10; n_boot=0)
        @test f.horizon == H  # capped at IRF horizon
    end

    # =========================================================================
    @testset "Short horizon (H=2)" begin
        slp_short = structural_lp(Y, 2; method=:cholesky, lags=4)
        f = lp_fevd(slp_short, 2; n_boot=0)

        @test f.horizon == 2
        @test size(f.proportions) == (n, n, 2)
        @test all(isfinite, f.proportions)
    end

    # =========================================================================
    @testset "n=2 system" begin
        Y2 = Y[:, 1:2]
        slp2 = structural_lp(Y2, 8; method=:cholesky, lags=4)
        f = lp_fevd(slp2, 8; n_boot=0)

        @test size(f.proportions) == (2, 2, 8)
        @test all(0 .<= f.proportions .<= 1)
    end

    # =========================================================================
    @testset "Bootstrap with explicit var_lags" begin
        f = lp_fevd(slp, 8; n_boot=30, var_lags=2)

        @test f.n_boot == 30
        @test all(isfinite, f.bias_corrected)
        @test all(isfinite, f.se)
    end

    # =========================================================================
    @testset "show method" begin
        f = lp_fevd(slp, H; n_boot=0)
        buf = IOBuffer()
        show(buf, f)
        output = String(take!(buf))
        @test occursin("LP-FEVD", output)
        @test occursin("R²", output)
    end

    # =========================================================================
    @testset "show method with bootstrap" begin
        f = lp_fevd(slp, H; n_boot=30)
        buf = IOBuffer()
        show(buf, f)
        output = String(take!(buf))
        @test occursin("LP-FEVD", output)
        @test occursin("VAR bootstrap", output)
    end

    # =========================================================================
    @testset "print_table method" begin
        f = lp_fevd(slp, H; n_boot=0)
        buf = IOBuffer()
        print_table(buf, f, 1)
        output = String(take!(buf))
        @test occursin("LP-FEVD", output)
        @test occursin("Variable 1", output)
    end

    # =========================================================================
    @testset "print_table with bootstrap CIs" begin
        f = lp_fevd(slp, H; n_boot=30)
        buf = IOBuffer()
        print_table(buf, f, 1)
        output = String(take!(buf))
        @test occursin("Lower", output)
    end

    # =========================================================================
    @testset "Own-shock FEVD is non-trivial" begin
        f = lp_fevd(slp, H; n_boot=0)

        # At impact (h=1), own-shock should explain some variance
        # (diagonal FEVD should not be all zeros)
        diag_sum = sum(f.proportions[i, i, 1] for i in 1:n)
        @test diag_sum > 0
    end

    # =========================================================================
    @testset "Confidence levels affect CI width" begin
        f_90 = lp_fevd(slp, 8; n_boot=50, conf_level=0.90)
        f_99 = lp_fevd(slp, 8; n_boot=50, conf_level=0.99)

        # 99% CIs should generally be at least as wide as 90% CIs
        # (check on average due to bootstrap randomness)
        width_90 = mean(f_90.ci_upper - f_90.ci_lower)
        width_99 = mean(f_99.ci_upper - f_99.ci_lower)
        @test width_99 >= width_90 * 0.8  # allow some tolerance for bootstrap noise
    end
end
