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

    # Issue #577: the Narayan-Popp tables do not describe this additive-outlier
    # statistic; they left a driftless random walk rejecting ~70% of the time at
    # 5%. Critical values are now simulated from the null of this implementation
    # (test/oracle/gen_lm_adf2break_cvs.jl).
    @testset "Simulated critical values (#577)" begin
        M = MacroEconometricModels
        grid = M.BREAK_TEST_SIM_T

        for model in (:level, :both), n in (60, 100, 150, 250, 500, 900)
            row = M._adf_2break_cv_row(model, n)
            @test row[1] < row[2] < row[3] < row[4]
            cv = M._adf_2break_cv(model, n, Float64)
            @test cv[1] == row[1] && cv[5] == row[3] && cv[10] == row[4]
        end

        for (i, Tg) in enumerate(grid)
            row = M._adf_2break_cv_row(:both, Tg)
            @test all(collect(row) .≈ M.ADF_2BREAK_SIM_CV[:both][i, :])
            @test all(abs.(collect(M._adf_2break_cv_row(:both, Tg - 1)) .- collect(row)) .< 0.01)
            @test all(abs.(collect(M._adf_2break_cv_row(:both, Tg + 1)) .- collect(row)) .< 0.01)
        end
        @test M._adf_2break_cv_row(:level, 40) == M._adf_2break_cv_row(:level, grid[1])
        @test M._adf_2break_cv_row(:level, 5000) == M._adf_2break_cv_row(:level, grid[end])
        # trend breaks add regressors: Model C needs a more negative cutoff
        @test M._adf_2break_cv_row(:both, 200)[3] < M._adf_2break_cv_row(:level, 200)[3]

        # A driftless random walk (the null) must not reject; white noise must.
        y_null = cumsum(randn(MersenneTwister(577_001), 200))
        y_alt = randn(MersenneTwister(577_002), 200)
        for model in (:level, :both)
            r0 = adf_2break_test(y_null; model=model, lags=0)
            @test r0.statistic > r0.critical_values[5]
            @test r0.pvalue > 0.05
            r1 = adf_2break_test(y_alt; model=model, lags=0)
            @test r1.statistic < r1.critical_values[5]
            @test r1.pvalue < 0.05
        end
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
