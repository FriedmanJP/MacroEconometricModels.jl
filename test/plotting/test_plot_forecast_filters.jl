# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# PLT-39 domain lane — forecast fans (forecast.jl) + trend-cycle filters
# (filters.jl): ARIMA / Volatility / VECM / Factor / LP / VAR / BVAR forecasts;
# HP / Hamilton / Beveridge-Nelson / Baxter-King / Boosted-HP filters.
# (X13FilterResult lives in test/filters/test_x13.jl.)
#
# Uses shared plot_test_helpers.jl assertions (Testing Rules 1-7).

using Test, Random

isdefined(@__MODULE__, :check_plot) || include(joinpath(@__DIR__, "plot_test_helpers.jl"))

@testset "Plotting — Forecasts / Filters" begin
    Random.seed!(43)

    # =========================================================================
    # Forecast dispatches
    # =========================================================================
    @testset "ARIMAForecast (+ history)" begin
        y = randn(200)
        fc = forecast(estimate_ar(y, 2), 20)
        p = plot_result(fc)
        check_plot(p); assert_all_json_valid(p)
        # forecast-history context available for every forecast type (PLT-18)
        check_plot(plot_result(fc; history=y, n_history=30))
    end

    @testset "VolatilityForecast (+ history)" begin
        gm = estimate_garch(randn(300), 1, 1)
        fc = forecast(gm, 10)
        check_plot(plot_result(fc)); assert_all_json_valid(plot_result(fc))
        check_plot(plot_result(fc; history=gm.conditional_variance))
    end

    @testset "VECMForecast" begin
        vecm_m = estimate_vecm(cumsum(randn(150, 3), dims=1), 2; rank=1)
        fc = forecast(vecm_m, 10)
        p = plot_result(fc)
        check_plot(p); assert_all_json_valid(p)
        # Int + String selection (C3 — VECMForecast carries names).
        check_plot(plot_result(fc; var=1))
        check_plot(plot_result(fc; var="y2"))
        @test_throws ArgumentError plot_result(fc; var="nope")
        @test_throws ArgumentError plot_result(fc; var=99)
    end

    @testset "FactorForecast (observable + factor)" begin
        fm = estimate_dynamic_factors(randn(200, 20), 2, 1)
        fc = forecast(fm, 10)
        check_plot(plot_result(fc)); assert_all_json_valid(plot_result(fc))
        check_plot(plot_result(fc; type=:observable, var=1))
        p = plot_result(fc; type=:factor)
        check_plot(p)
        @test occursin("Factor", p.html)
        @test_throws ArgumentError plot_result(fc; type=:bogus)
    end

    @testset "LPForecast" begin
        lp = estimate_lp(randn(100, 3), 1, 10; lags=2)
        shock_path = zeros(10); shock_path[1] = 1.0
        fc = forecast(lp, shock_path)
        p = plot_result(fc)
        check_plot(p); assert_all_json_valid(p)
    end

    @testset "VARForecast" begin
        m = estimate_var(randn(100, 3), 2)
        fc = forecast(m, 10)
        check_plot(plot_result(fc)); assert_all_json_valid(plot_result(fc))
        check_plot(plot_result(fc; var=1))
        check_plot(plot_result(fc; var="y2"))
        @test_throws ArgumentError plot_result(fc; var="nope")
    end

    @testset "BVARForecast (point_estimate=:mean)" begin
        post = estimate_bvar(randn(100, 3), 2; n_draws=100)
        fc = forecast(post, 10; point_estimate=:mean)
        p = plot_result(fc)
        check_plot(p); assert_all_json_valid(p)
        @test occursin("mean", p.html)
    end

    # =========================================================================
    # Trend-cycle filters
    # =========================================================================
    @testset "HPFilterResult" begin
        p = plot_result(hp_filter(cumsum(randn(200))))
        check_plot(p); assert_all_json_valid(p)
        @test occursin("Hodrick-Prescott", p.html)
    end

    @testset "HamiltonFilterResult (with + without original)" begin
        y = cumsum(randn(200))
        r = hamilton_filter(y)
        p = plot_result(r; original=y)
        check_plot(p); assert_all_json_valid(p)
        @test occursin("Hamilton", p.html)
        check_plot(plot_result(r))                     # no original= (phantom-original guard)
    end

    @testset "BeveridgeNelsonResult" begin
        p = plot_result(beveridge_nelson(cumsum(randn(200))))
        check_plot(p); assert_all_json_valid(p)
        @test occursin("Beveridge-Nelson", p.html)
    end

    @testset "BaxterKingResult (with + without original)" begin
        y = cumsum(randn(200))
        p = plot_result(baxter_king(y))
        check_plot(p); assert_all_json_valid(p)
        @test occursin("Baxter-King", p.html)
    end

    @testset "BoostedHPResult" begin
        p = plot_result(boosted_hp(cumsum(randn(200))))
        check_plot(p); assert_all_json_valid(p)
        @test occursin("Boosted HP", p.html)
    end

    # =========================================================================
    # Edge cases — history validation + flat series
    # =========================================================================
    @testset "forecast history dimension mismatch → ArgumentError" begin
        m = estimate_var(randn(100, 3), 2)
        fc = forecast(m, 10)
        @test_throws ArgumentError plot_result(fc; history=randn(40, 2))   # wrong n_vars
    end

    @testset "flat (constant) input renders" begin
        p = plot_result(hp_filter(fill(3.0, 120)))
        check_plot(p)
        for (_, lit) in extract_json_blocks(p.html)
            @test !occursin(r"[:\[,]\s*NaN", lit)
        end
    end
end
