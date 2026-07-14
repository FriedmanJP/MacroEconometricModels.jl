# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test, MacroEconometricModels, Random, StatsAPI

@testset "Two-Break ADF Test" begin
    rng = Random.MersenneTwister(88990)

    y_2break = vcat(randn(rng, 80), randn(rng, 60) .+ 3.0, randn(rng, 60) .+ 1.0)
    y_short = vcat(randn(rng, 30), randn(rng, 25) .+ 3.0, randn(rng, 25) .+ 1.0)

    # full-T anchor computed once and shared (deterministic estimator)
    result_A = adf_2break_test(y_2break; model=:level)

    @testset "Model A (level shifts)" begin
        result = result_A
        @test result isa ADF2BreakResult
        @test result.model == :level
        @test result.break1 < result.break2
        @test result.break1 > 0
        @test result.break2 <= length(y_2break)
        @test isfinite(result.statistic)
        @test haskey(result.critical_values, 5)
        @test result.nobs > 0
        @test result.break1_fraction > 0.0
        @test result.break2_fraction <= 1.0
        @test result.break1_fraction < result.break2_fraction
    end

    @testset "Model C (level + trend)" begin
        result = adf_2break_test(y_short; model=:both, lags=1)
        @test result isa ADF2BreakResult
        @test result.model == :both
        @test result.break1 < result.break2
        @test isfinite(result.statistic)
        @test haskey(result.critical_values, 1)
        @test haskey(result.critical_values, 10)
    end

    @testset "Parameters" begin
        result = adf_2break_test(y_short; lags=2)
        @test result.lags == 2

        result_aic = adf_2break_test(y_short; lags=:aic)
        @test result_aic.lags >= 0

        result_bic = adf_2break_test(y_short; lags=:bic)
        @test result_bic.lags >= 0

        result_trim = adf_2break_test(y_short; trim=0.15, lags=1)
        @test isfinite(result_trim.statistic)

        result_maxlags = adf_2break_test(y_short; max_lags=4)
        @test result_maxlags.lags <= 4
    end

    @testset "StatsAPI interface" begin
        result = result_A
        @test nobs(result) == result.nobs
        @test StatsAPI.pvalue(result) == result.pvalue
        @test dof(result) == result.lags + 4
    end

    @testset "Float type promotion" begin
        y_int = round.(Int, y_short .* 10)
        result = adf_2break_test(y_int; lags=1)
        @test result isa ADF2BreakResult{Float64}
    end

    @testset "Error handling" begin
        @test_throws ArgumentError adf_2break_test(randn(rng, 30))
        @test_throws ArgumentError adf_2break_test(y_2break; model=:invalid)
    end

    @testset "show method" begin
        result = result_A
        io = IOBuffer()
        show(io, result)
        output = String(take!(io))
        @test contains(output, "Two-Break ADF")
        @test contains(output, "Break")
    end
end
