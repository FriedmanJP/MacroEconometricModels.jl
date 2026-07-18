# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# PLT-39 domain lane — structural-analysis plots (irf.jl / fevd.jl / hd.jl):
#   IRF (VAR / Bayesian / LP / StructuralLP), FEVD (VAR / Bayesian / LP),
#   Historical Decomposition (frequentist / Bayesian).
#
# Every testset uses the shared plot_test_helpers.jl assertions (Testing Rules
# 1-7): check_plot + assert_all_json_valid (Rules 1/6/A12), series_count/
# series_names/panel_titles (Rule 2), assert_nan_becomes_null (Rule 4),
# assert_escapes (Rule 5, HOSTILE_NAME), @test_throws for bad selection (Rule 3).
# Structural assertions parse EXTRACTED JSON literals, never raw p.html.

using Test, Random

# Self-bootstrap the shared helpers when run standalone.
isdefined(@__MODULE__, :check_plot) || include(joinpath(@__DIR__, "plot_test_helpers.jl"))

@testset "Plotting — IRF / FEVD / HD" begin
    Random.seed!(42)

    # =========================================================================
    # IRF dispatches
    # =========================================================================
    @testset "ImpulseResponse (VAR)" begin
        m = estimate_var(randn(100, 3), 2)
        r = irf(m, 10; ci_type=:bootstrap, reps=50)

        p = plot_result(r)
        check_plot(p); assert_all_json_valid(p)
        @test length(panel_titles(p.html)) == 9          # 3 vars × 3 shocks

        # Int + String selection resolve to the same single panel (C3).
        pI = plot_result(r; var=1, shock=1)
        pS = plot_result(r; var="y1", shock="y2")
        check_plot(pI); check_plot(pS)
        @test length(panel_titles(pI.html)) == 1
        @test panel_titles(plot_result(r; var=2, shock=2).html) ==
              panel_titles(plot_result(r; var="y2", shock="y2").html)

        # Bad selection → ArgumentError (Rule 3).
        @test_throws ArgumentError plot_result(r; var="nope")
        @test_throws ArgumentError plot_result(r; var=99)
        @test_throws ArgumentError plot_result(r; shock=99)

        # Title override.
        @test occursin("My Custom IRF", plot_result(r; title="My Custom IRF").html)

        # ci_type=:none path.
        check_plot(plot_result(irf(m, 10; ci_type=:none); var=1, shock=1))
    end

    @testset "BayesianImpulseResponse" begin
        post = estimate_bvar(randn(100, 3), 2; n_draws=100)
        r = irf(post, 10)
        p = plot_result(r)
        check_plot(p); assert_all_json_valid(p)
        @test occursin("Bayesian", p.html)
        check_plot(plot_result(r; var=1, shock=1))
    end

    @testset "LPImpulseResponse" begin
        lp = estimate_lp(randn(100, 3), 1, 10; lags=2)
        r = lp_irf(lp)
        p = plot_result(r)
        check_plot(p); assert_all_json_valid(p)
        @test occursin("LP", p.html)
        check_plot(plot_result(r; var=1))
    end

    @testset "StructuralLP" begin
        slp = structural_lp(randn(100, 3), 10; method=:cholesky, lags=2)
        p = plot_result(slp)
        check_plot(p); assert_all_json_valid(p)
        @test occursin("Structural LP", p.html)
    end

    # =========================================================================
    # FEVD dispatches
    # =========================================================================
    @testset "FEVD (VAR)" begin
        m = estimate_var(randn(100, 3), 2)
        f = fevd(m, 10)
        p = plot_result(f)
        check_plot(p); assert_all_json_valid(p)
        @test occursin("FEVD", p.html) || occursin("Variance", p.html)
        # FEVD selects a variable by Int (its signature is Int-only, unlike IRF).
        check_plot(plot_result(f; var=1))
        @test length(panel_titles(plot_result(f; var=1).html)) == 1
    end

    @testset "BayesianFEVD" begin
        post = estimate_bvar(randn(100, 3), 2; n_draws=100)
        f = fevd(post, 10)
        p = plot_result(f)
        check_plot(p); assert_all_json_valid(p)
        @test occursin("Bayesian", p.html)
    end

    @testset "LPFEVD" begin
        slp = structural_lp(randn(100, 3), 10; method=:cholesky, lags=2)
        f = lp_fevd(slp, 10)
        p = plot_result(f)
        check_plot(p); assert_all_json_valid(p)
    end

    # =========================================================================
    # Historical decomposition
    # =========================================================================
    @testset "HistoricalDecomposition (VAR)" begin
        m = estimate_var(randn(100, 3), 2)
        hd_res = historical_decomposition(m, size(m.Y, 1) - m.p)
        p = plot_result(hd_res)
        check_plot(p); assert_all_json_valid(p)
        @test occursin("Historical Decomposition", p.html)
        check_plot(plot_result(hd_res; var=1))
    end

    @testset "BayesianHistoricalDecomposition" begin
        post = estimate_bvar(randn(100, 3), 2; n_draws=100)
        hd_res = historical_decomposition(post, 100 - 2)
        p = plot_result(hd_res)
        check_plot(p); assert_all_json_valid(p)
        @test occursin("Bayesian", p.html)
    end

    # =========================================================================
    # Edge-case battery (Rules 4/5) on directly-constructed result structs
    # =========================================================================
    @testset "NaN → null (IRF line + FEVD area)" begin
        H, n = 6, 2
        vals = randn(H, n, n); vals[2, 1, 1] = NaN
        r = ImpulseResponse{Float64}(vals, zero(vals), zero(vals), H,
                                     ["y1", "y2"], ["y1", "y2"], :none)
        assert_nan_becomes_null(plot_result(r; var=1, shock=1))

        # proportions axis is (n_vars, n_shocks, H); plot var=1 draws props[1, :, :],
        # so the injected NaN must live on variable 1 to appear in the drawn area.
        props = fill(0.5, n, n, H); props[1, 1, 2] = NaN
        f = FEVD{Float64}(copy(props), props, ["y1", "y2"], ["y1", "y2"])
        assert_nan_becomes_null(plot_result(f; var=1))
    end

    @testset "escaping — hostile variable name round-trips (A7/A8)" begin
        m = estimate_var(randn(80, 2), 2; varnames=[HOSTILE_NAME, "y2"])
        assert_escapes(plot_result(irf(m, 6; ci_type=:none)))
        assert_escapes(plot_result(fevd(m, 6)))
        assert_escapes(plot_result(historical_decomposition(m, size(m.Y, 1) - m.p)))
    end
end
