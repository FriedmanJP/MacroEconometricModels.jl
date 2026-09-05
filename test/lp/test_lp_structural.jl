# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using MacroEconometricModels
using Test
using Random
using LinearAlgebra
using Statistics

@testset "Structural LP" begin
    # Diagonal AR(1) on the shared simulator (DGP-05 #794): same design as
    # the legacy inline loop (0.3 persistence, identity innovations).
    T_obs = 200
    n = 3
    A_slp = 0.3 .* Matrix{Float64}(I, n, n)
    Y = dgp_var(MersenneTwister(42); A=A_slp, B0=Matrix{Float64}(I, n, n),
                T=T_obs).Y

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

        # CI type should be :analytical by default (Newey-West SEs)
        @test slp.irf.ci_type == :analytical
        @test all(isfinite, slp.irf.ci_lower)
        @test all(isfinite, slp.irf.ci_upper)
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
                            ci_type=:bootstrap, reps=(FAST ? 20 : 50))

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
        # Non-identity impact + cross-dynamics truth (DGP-05 #794): the old
        # diagonal-AR file design made every ordering equivalent. Structural
        # LP covers h = 1..12 (probed: matches Θ_1..12, not Θ_0..11).
        A = [0.5 0.1 0.0; 0.1 0.4 0.1; 0.0 0.1 0.3]
        B0 = [0.6 0.0 0.0; 0.2 0.5 0.0; 0.1 0.15 0.4]
        Yb = dgp_var(MersenneTwister(166); A=A, B0=B0, T=2000).Y
        slp = structural_lp(Yb, 12; method=:cholesky, lags=4)
        var_model = estimate_var(Yb, 4)
        var_vals = irf(var_model, 12; method=:cholesky).values
        # Overlap window h = 1..11 (VAR rows are h = 0..11, LP rows h = 1..12).
        truth = var_irf(A, B0, 12)[2:12, :, :]

        # Both recover truth (probed LP 0.036 on MT(42)) — the sign check with
        # its `|| abs < 0.2` escape hatch passed for anti-correlated estimates.
        @test maximum(abs, slp.irf.values[1:11, :, :] - truth) < 0.1
        # ... hence they agree with each other (transitively, < 0.2).
        @test maximum(abs, slp.irf.values[1:11, :, :] - var_vals[2:12, :, :]) < 0.2
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

        # Default is analytical CIs from Newey-West SEs
        @test has_uncertainty(slp) == true
        bounds = uncertainty_bounds(slp)
        @test bounds isa Tuple
        @test length(bounds) == 2

        # With bootstrap
        slp_ci = structural_lp(Y, 8; method=:cholesky, lags=4,
                               ci_type=:bootstrap, reps=(FAST ? 15 : 30))
        @test has_uncertainty(slp_ci) == true
        bounds_boot = uncertainty_bounds(slp_ci)
        @test bounds_boot isa Tuple
        @test length(bounds_boot) == 2
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

    # =========================================================================
    @testset "MC honesty counts (#244)" begin
        # bootstrap path: n_requested == reps, invariant holds
        slp = structural_lp(Y, 6; method=:cholesky, lags=2, ci_type=:bootstrap,
                            reps=30, rng=MersenneTwister(1))
        @test slp.n_requested == 30
        @test slp.n_effective + slp.n_failed == slp.n_requested
        @test 0 <= slp.n_failed <= slp.n_requested
        # count is reproducible under a fixed seed (atomic total is thread-count invariant)
        slp2 = structural_lp(Y, 6; method=:cholesky, lags=2, ci_type=:bootstrap,
                             reps=30, rng=MersenneTwister(1))
        @test slp.n_failed == slp2.n_failed
        # no bootstrap ⇒ zero counts
        slp0 = structural_lp(Y, 6; method=:cholesky, lags=2, ci_type=:none)
        @test (slp0.n_requested, slp0.n_effective, slp0.n_failed) == (0, 0, 0)
        # backward-compatible 9-arg constructor
        b = MacroEconometricModels.StructuralLP{Float64}(slp.irf, slp.structural_shocks,
            slp.var_model, slp.Q, slp.method, slp.lags, slp.cov_type, slp.se, slp.lp_models)
        @test (b.n_requested, b.n_effective, b.n_failed) == (0, 0, 0)
    end
end
