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
using Random
using LinearAlgebra
using Statistics

@testset "Structural LP" begin
    Random.seed!(42)

    # Generate test data with some structure
    T_obs = 200
    n = 3
    Y = zeros(T_obs, n)
    Y[1, :] = randn(n)
    for t in 2:T_obs
        Y[t, :] = 0.3 * Y[t-1, :] + randn(n)
    end

    # =========================================================================
    @testset "structural_lp with Cholesky" begin
        slp = structural_lp(Y, 12; method=:cholesky, lags=4)

        @test slp isa StructuralLP{Float64}
        @test size(slp.irf.values) == (12, n, n)
        @test size(slp.structural_shocks, 2) == n
        @test size(slp.Q) == (n, n)
        @test slp.method == :cholesky
        @test slp.lags == 4
        @test slp.cov_type == :newey_west
        @test length(slp.lp_models) == n
        @test size(slp.se) == (12, n, n)

        # IRF should be finite
        @test all(isfinite, slp.irf.values)

        # SE should be non-negative
        @test all(slp.se .>= 0)

        # Q should be identity for Cholesky
        @test slp.Q ≈ Matrix{Float64}(I, n, n)

        # CI type should be :none by default
        @test slp.irf.ci_type == :none
    end

    # =========================================================================
    @testset "structural_lp with var_lags" begin
        slp = structural_lp(Y, 8; method=:cholesky, lags=2, var_lags=4)

        @test slp.lags == 2
        @test slp.var_model.p == 4
        @test size(slp.irf.values) == (8, n, n)
    end

    # =========================================================================
    @testset "structural_lp with long_run" begin
        slp = structural_lp(Y, 8; method=:long_run, lags=4)

        @test slp isa StructuralLP{Float64}
        @test slp.method == :long_run
        @test size(slp.irf.values) == (8, n, n)
        @test all(isfinite, slp.irf.values)
    end

    # =========================================================================
    @testset "structural_lp with sign restrictions" begin
        # Simple sign restriction: first shock has positive impact on first variable
        check_func = irf -> irf[1, 1, 1] > 0

        slp = structural_lp(Y, 8; method=:sign, lags=4, check_func=check_func)

        @test slp isa StructuralLP{Float64}
        @test slp.method == :sign
        @test size(slp.irf.values) == (8, n, n)
    end

    # =========================================================================
    @testset "structural_lp with fastica" begin
        slp = structural_lp(Y, 8; method=:fastica, lags=4)

        @test slp isa StructuralLP{Float64}
        @test slp.method == :fastica
        @test size(slp.irf.values) == (8, n, n)
        @test all(isfinite, slp.irf.values)
    end

    # =========================================================================
    @testset "structural_lp with bootstrap CIs" begin
        slp = structural_lp(Y, 8; method=:cholesky, lags=4,
                            ci_type=:bootstrap, reps=50)

        @test slp.irf.ci_type == :bootstrap
        @test size(slp.irf.ci_lower) == (8, n, n)
        @test size(slp.irf.ci_upper) == (8, n, n)

        # CI bounds should be finite
        @test all(isfinite, slp.irf.ci_lower)
        @test all(isfinite, slp.irf.ci_upper)

        # Lower <= Upper (generally)
        for h in 1:8, v in 1:n, s in 1:n
            @test slp.irf.ci_lower[h, v, s] <= slp.irf.ci_upper[h, v, s]
        end
    end

    # =========================================================================
    @testset "structural_lp with :white covariance" begin
        slp = structural_lp(Y, 8; method=:cholesky, lags=4, cov_type=:white)

        @test slp.cov_type == :white
        @test all(isfinite, slp.se)
    end

    # =========================================================================
    @testset "irf accessor" begin
        slp = structural_lp(Y, 8; method=:cholesky, lags=4)

        irf_result = irf(slp)
        @test irf_result isa ImpulseResponse{Float64}
        @test irf_result === slp.irf
    end

    # =========================================================================
    @testset "fevd from structural LP (GL2019)" begin
        slp = structural_lp(Y, 12; method=:cholesky, lags=4)

        f = fevd(slp, 12; n_boot=0)
        @test f isa LPFEVD{Float64}
        @test size(f.proportions) == (n, n, 12)

        # R² proportions should be in [0, 1]
        @test all(0 .<= f.proportions .<= 1)

        # Test with shorter horizon
        f_short = fevd(slp, 4; n_boot=0)
        @test size(f_short.proportions) == (n, n, 4)
    end

    # =========================================================================
    @testset "historical_decomposition from structural LP" begin
        slp = structural_lp(Y, 12; method=:cholesky, lags=4)

        T_hd = 50
        hd = historical_decomposition(slp, T_hd)
        @test hd isa HistoricalDecomposition{Float64}
        @test hd.T_eff == T_hd
        @test size(hd.contributions) == (T_hd, n, n)
        @test size(hd.initial_conditions) == (T_hd, n)
        @test size(hd.actual) == (T_hd, n)
        @test size(hd.shocks) == (T_hd, n)
        @test hd.method == :cholesky

        # Verify decomposition identity
        @test verify_decomposition(hd)
    end

    # =========================================================================
    @testset "VAR vs LP IRF comparison" begin
        slp = structural_lp(Y, 12; method=:cholesky, lags=4)
        var_model = estimate_var(Y, 4)
        var_irf = irf(var_model, 12; method=:cholesky)

        # LP and VAR IRFs should be similar but not identical (finite-sample)
        # Check that they're at least correlated
        for shock in 1:n
            for resp in 1:n
                lp_vals = slp.irf.values[:, resp, shock]
                var_vals = var_irf.values[:, resp, shock]
                # At least same sign at impact for most
                if abs(var_vals[1]) > 0.1
                    @test sign(lp_vals[1]) == sign(var_vals[1]) || abs(lp_vals[1]) < 0.2
                end
            end
        end
    end

    # =========================================================================
    @testset "n=2 edge case" begin
        Y2 = Y[:, 1:2]
        slp = structural_lp(Y2, 8; method=:cholesky, lags=4)

        @test size(slp.irf.values) == (8, 2, 2)
        @test size(slp.se) == (8, 2, 2)
        @test length(slp.lp_models) == 2

        f = fevd(slp, 8)
        @test size(f.proportions) == (2, 2, 8)
    end

    # =========================================================================
    @testset "large horizon" begin
        slp = structural_lp(Y, 40; method=:cholesky, lags=4)

        @test size(slp.irf.values) == (40, n, n)
        @test all(isfinite, slp.irf.values)
    end

    # =========================================================================
    @testset "show method" begin
        slp = structural_lp(Y, 8; method=:cholesky, lags=4)

        # Should not error
        buf = IOBuffer()
        show(buf, slp)
        output = String(take!(buf))
        @test occursin("Structural Local Projections", output)
        @test occursin("cholesky", output)
        @test occursin("Shock", output)
    end

    # =========================================================================
    @testset "print_table for StructuralLP" begin
        slp = structural_lp(Y, 8; method=:cholesky, lags=4)

        buf = IOBuffer()
        print_table(buf, slp, 1, 1)
        output = String(take!(buf))
        @test occursin("IRF", output)
    end

    # =========================================================================
    @testset "point_estimate / has_uncertainty / uncertainty_bounds" begin
        slp = structural_lp(Y, 8; method=:cholesky, lags=4)

        pe = point_estimate(slp)
        @test pe === slp.irf.values

        @test has_uncertainty(slp) == false

        @test isnothing(uncertainty_bounds(slp))

        # With bootstrap
        slp_ci = structural_lp(Y, 8; method=:cholesky, lags=4,
                               ci_type=:bootstrap, reps=30)
        @test has_uncertainty(slp_ci) == true
        bounds = uncertainty_bounds(slp_ci)
        @test bounds isa Tuple
        @test length(bounds) == 2
    end

    # =========================================================================
    @testset "Float32 input" begin
        Y32 = Float32.(Y)
        slp = structural_lp(Y32, 8; method=:cholesky, lags=4)
        @test slp isa StructuralLP{Float64}  # promoted via fallback
    end

    # =========================================================================
    @testset "nvars accessor" begin
        slp = structural_lp(Y, 8; method=:cholesky, lags=4)
        @test nvars(slp) == n
    end
end
