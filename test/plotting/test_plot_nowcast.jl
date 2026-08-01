# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# PLT-39 domain lane — nowcast.jl: NowcastResult (×views) + NowcastNews (×views).
# Multi-view dispatch (C5) throws ArgumentError on unknown views; the ragged-edge
# fixtures carry NaN (missing latest obs), exercising Rule 4 on the real path.
#
# Uses shared plot_test_helpers.jl assertions (Testing Rules 1-7).

using Test, Random

isdefined(@__MODULE__, :check_plot) || include(joinpath(@__DIR__, "plot_test_helpers.jl"))

@testset "Plotting — Nowcast" begin

    # =========================================================================
    # NowcastResult (DFM) — views
    # =========================================================================
    @testset "NowcastResult (default)" begin
        nM, nQ = 4, 1
        Y = randn(Random.MersenneTwister(70), 100, nM + nQ); Y[end, end] = NaN
        nr = nowcast(nowcast_dfm(Y, nM, nQ; r=2, p=1))
        p = plot_result(nr)
        check_plot(p); assert_all_json_valid(p)
        @test occursin("Nowcast", p.html)
    end

    @testset "NowcastResult view=:default with DFM factors" begin
        nM, nQ = 4, 1
        Y = randn(Random.MersenneTwister(70), 100, nM + nQ); Y[end, end] = NaN
        nr = nowcast(nowcast_dfm(Y, nM, nQ; r=2, p=1))
        p = plot_result(nr; view=:default)
        check_plot(p); assert_all_json_valid(p)
        @test occursin("Factor 1", p.html) && occursin("Factor 2", p.html)
    end

    @testset "NowcastResult view=:heatmap (z-score ragged edge)" begin
        nM, nQ = 4, 1
        Y = randn(Random.MersenneTwister(77), 100, nM + nQ)
        Y[end, end] = NaN; Y[end-1:end, 3] .= NaN
        nr = nowcast(nowcast_dfm(Y, nM, nQ; r=2, p=1))
        p = plot_result(nr; view=:heatmap)
        check_plot(p); assert_all_json_valid(p)
        @test occursin("interpolateRdBu", p.html)        # diverging z-score ramp
        @test occursin("#d9d9d9", p.html)                # missing-cell neutral grey
    end

    @testset "NowcastResult view=:contributions" begin
        nM, nQ = 4, 1
        Y = randn(Random.MersenneTwister(71), 100, nM + nQ); Y[end, end] = NaN
        nr = nowcast(nowcast_dfm(Y, nM, nQ; r=2, p=1))
        p = plot_result(nr; view=:contributions)
        check_plot(p); assert_all_json_valid(p)
        @test occursin("Contribution", p.html) || occursin("contribution", p.html)
    end

    @testset "NowcastResult view guards (C5)" begin
        rng = Random.MersenneTwister(72)
        Y = randn(rng, 60, 4); Y[55:60, 3:4] .= NaN
        nr_bvar = nowcast(nowcast_bvar(Y, 2, 2; lags=2, max_iter=20))
        @test_throws ArgumentError plot_result(nr_bvar; view=:contributions)  # DFM-only view

        nM, nQ = 4, 1
        Y2 = randn(Random.MersenneTwister(73), 100, nM + nQ); Y2[end, end] = NaN
        nr = nowcast(nowcast_dfm(Y2, nM, nQ; r=2, p=1))
        @test_throws ArgumentError plot_result(nr; view=:bad)
    end

    # =========================================================================
    # NowcastNews — views
    # =========================================================================
    @testset "NowcastNews (default)" begin
        X_old = randn(100, 5); X_old[end, end] = NaN
        X_new = copy(X_old); X_new[end, end] = 0.5
        nn = nowcast_news(X_new, X_old, nowcast_dfm(X_old, 4, 1; r=2, p=1), 5)
        p = plot_result(nn)
        check_plot(p); assert_all_json_valid(p)
        @test occursin("News", p.html) || occursin("news", p.html)
    end

    @testset "NowcastNews view=:groups (group_names live)" begin
        X_old = randn(Random.MersenneTwister(55), 100, 5); X_old[end, end] = NaN
        X_new = copy(X_old); X_new[end, end] = 0.5
        nn = nowcast_news(X_new, X_old, nowcast_dfm(X_old, 4, 1; r=2, p=1), 5;
                          groups=[1, 1, 2, 2, 3],
                          group_names=["Industry", "Retail", "GDP"])
        p = plot_result(nn; view=:groups)
        check_plot(p); assert_all_json_valid(p)
        @test occursin("Industry", p.html) && occursin("Retail", p.html)
    end

    @testset "NowcastNews view=:individual" begin
        X_old = randn(Random.MersenneTwister(56), 100, 5); X_old[98:100, 1:2] .= NaN
        X_new = copy(X_old); X_new[98:100, 1:2] .= randn(Random.MersenneTwister(57), 3, 2)
        nn = nowcast_news(X_new, X_old, nowcast_dfm(X_old, 4, 1; r=2, p=1), 5)
        p = plot_result(nn; view=:individual)
        check_plot(p); assert_all_json_valid(p)
        @test occursin("Impact", p.html)
    end

    @testset "NowcastNews unknown view → ArgumentError (C5)" begin
        X_old = randn(Random.MersenneTwister(58), 100, 5); X_old[end, end] = NaN
        X_new = copy(X_old); X_new[end, end] = 0.5
        nn = nowcast_news(X_new, X_old, nowcast_dfm(X_old, 4, 1; r=2, p=1), 5)
        @test_throws ArgumentError plot_result(nn; view=:nonexistent)
    end
end
