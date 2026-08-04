# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test, MacroEconometricModels, Random

@testset "LM Unit Root Tests" begin
    rng = Random.MersenneTwister(77889)

    y_stat = zeros(200)
    y_stat[1] = randn(rng)
    for t in 2:200; y_stat[t] = 0.3 * y_stat[t-1] + randn(rng); end
    y_rw = cumsum(randn(rng, 200))
    y_break = vcat(randn(rng, 100), randn(rng, 100) .+ 3.0)

    @testset "No breaks" begin
        result = lm_unitroot_test(y_stat; breaks=0)
        @test result isa LMUnitRootResult
        @test result.breaks == 0
        @test isempty(result.break_dates)
        @test isempty(result.break_fractions)
        @test isfinite(result.statistic)
        @test haskey(result.critical_values, 5)
        @test result.lags >= 0
        @test result.nobs > 0
    end

    @testset "No breaks - regression options" begin
        result_level = lm_unitroot_test(y_stat; breaks=0, regression=:level)
        @test result_level.regression == :level

        result_both = lm_unitroot_test(y_stat; breaks=0, regression=:both)
        @test result_both.regression == :both
    end

    @testset "No breaks - power" begin
        result_stat = lm_unitroot_test(y_stat; breaks=0)
        @test result_stat.pvalue < 0.50

        result_rw = lm_unitroot_test(y_rw; breaks=0)
        @test result_rw.pvalue > 0.01
    end

    @testset "One break" begin
        result = lm_unitroot_test(y_break; breaks=1, regression=:level)
        @test result isa LMUnitRootResult
        @test result.breaks == 1
        @test length(result.break_dates) == 1
        @test length(result.break_fractions) == 1
        @test result.break_dates[1] > 0
        @test 50 < result.break_dates[1] < 150
        @test isfinite(result.statistic)
        @test haskey(result.critical_values, 1)
        @test haskey(result.critical_values, 5)
        @test haskey(result.critical_values, 10)

        result_both = lm_unitroot_test(y_break; breaks=1, regression=:both)
        @test result_both.breaks == 1
        @test result_both.regression == :both
    end

    @testset "Two breaks" begin
        y_2break = vcat(randn(rng, 70), randn(rng, 60) .+ 3.0, randn(rng, 70) .+ 1.0)
        y_2break_s = vcat(randn(rng, 25), randn(rng, 20) .+ 3.0, randn(rng, 25) .+ 1.0)
        result = lm_unitroot_test(y_2break; breaks=2)
        @test result isa LMUnitRootResult
        @test result.breaks == 2
        @test length(result.break_dates) == 2
        @test length(result.break_fractions) == 2
        @test result.break_dates[1] < result.break_dates[2]
        @test isfinite(result.statistic)

        result_both = lm_unitroot_test(y_2break_s; breaks=2, regression=:both)
        @test result_both.breaks == 2
    end

    @testset "Fixed lags" begin
        result = lm_unitroot_test(y_stat; breaks=0, lags=2)
        @test result.lags == 2

        result1 = lm_unitroot_test(y_break; breaks=1, lags=1)
        @test result1.lags == 1
    end

    @testset "BIC lag selection" begin
        result = lm_unitroot_test(y_stat; breaks=0, lags=:bic)
        @test result isa LMUnitRootResult
        @test result.lags >= 0
    end

    @testset "StatsAPI interface" begin
        result = lm_unitroot_test(y_stat; breaks=0)
        @test StatsAPI.nobs(result) == result.nobs
        @test StatsAPI.pvalue(result) == result.pvalue
        @test StatsAPI.dof(result) >= 0
    end

    @testset "Integer input" begin
        result = lm_unitroot_test(round.(Int, y_stat * 10); breaks=0)
        @test result isa LMUnitRootResult
    end

    @testset "Error handling" begin
        @test_throws ArgumentError lm_unitroot_test(randn(rng, 10); breaks=0)
        @test_throws ArgumentError lm_unitroot_test(y_stat; breaks=3)
        @test_throws ArgumentError lm_unitroot_test(y_stat; breaks=0, regression=:trend)
    end

    # Issue #577: the shipped tables were computed for other statistics, leaving
    # the break searches grossly mis-sized. They are now simulated from the null
    # of this implementation (test/oracle/gen_lm_adf2break_cvs.jl).
    @testset "Simulated critical values (#577)" begin
        M = MacroEconometricModels
        grid = M.BREAK_TEST_SIM_T

        @testset "monotone in significance level" begin
            for breaks in 0:2, reg in (:level, :both), n in (60, 100, 150, 220, 250, 400, 500, 900)
                row = M._lm_unitroot_cv_row(breaks, n, reg)
                @test row[1] < row[2] < row[3] < row[4]
                cv = M._lm_unitroot_critical_values(breaks, n, reg, Float64)
                @test cv[1] == row[1] && cv[5] == row[3] && cv[10] == row[4]
            end
        end

        @testset "interpolation" begin
            for (i, Tg) in enumerate(grid)
                row = M._lm_unitroot_cv_row(1, Tg, :both)
                # exact at a tabulated sample size
                @test all(collect(row) .≈ M.LM_UNITROOT_SIM_CV[(1, :both)][i, :])
                # and continuous across it
                @test all(abs.(collect(M._lm_unitroot_cv_row(1, Tg - 1, :both)) .- collect(row)) .< 0.01)
                @test all(abs.(collect(M._lm_unitroot_cv_row(1, Tg + 1, :both)) .- collect(row)) .< 0.01)
            end
            # clamped, not extrapolated, outside the grid
            @test M._lm_unitroot_cv_row(2, 55, :level) == M._lm_unitroot_cv_row(2, grid[1], :level)
            @test M._lm_unitroot_cv_row(2, 5000, :level) == M._lm_unitroot_cv_row(2, grid[end], :level)
            # searching more break dates needs a more negative critical value
            for reg in (:level, :both)
                @test M._lm_unitroot_cv_row(2, 200, reg)[3] < M._lm_unitroot_cv_row(1, 200, reg)[3]
                @test M._lm_unitroot_cv_row(1, 200, reg)[3] < M._lm_unitroot_cv_row(0, 200, reg)[3]
            end
            # Model C (level + trend break) is more conservative than Model A
            for breaks in 1:2
                @test M._lm_unitroot_cv_row(breaks, 200, :both)[3] <
                      M._lm_unitroot_cv_row(breaks, 200, :level)[3]
            end
        end

        @testset "p-value interpolation" begin
            row = M._lm_unitroot_cv_row(1, 200, :level)
            @test M._break_test_pvalue(row[1] - 1.0, row) == 0.001
            @test M._break_test_pvalue(row[2], row) ≈ 0.025
            @test M._break_test_pvalue(row[3], row) ≈ 0.05
            @test M._break_test_pvalue(row[4], row) ≈ 0.10
            @test M._break_test_pvalue(row[4] + 1.0, row) == 0.20
            ps = [M._break_test_pvalue(s, row) for s in range(row[1], row[4] + 0.5; length = 40)]
            @test issorted(ps)
        end

        @testset "size and power at 5%" begin
            # A driftless random walk (the null) must not reject; white noise must.
            y_null = cumsum(randn(MersenneTwister(577_001), 200))
            y_alt = randn(MersenneTwister(577_002), 200)
            for breaks in 0:2, reg in (:level, :both)
                r0 = lm_unitroot_test(y_null; breaks=breaks, regression=reg, lags=0)
                @test r0.statistic > r0.critical_values[5]
                @test r0.pvalue > 0.05
                r1 = lm_unitroot_test(y_alt; breaks=breaks, regression=reg, lags=0)
                @test r1.statistic < r1.critical_values[5]
                @test r1.pvalue < 0.05
            end
        end
    end
end
