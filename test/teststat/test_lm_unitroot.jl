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
        result = lm_unitroot_test(y_2break; breaks=2)
        @test result isa LMUnitRootResult
        @test result.breaks == 2
        @test length(result.break_dates) == 2
        @test length(result.break_fractions) == 2
        @test result.break_dates[1] < result.break_dates[2]
        @test isfinite(result.statistic)

        result_both = lm_unitroot_test(y_2break; breaks=2, regression=:both)
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
end
