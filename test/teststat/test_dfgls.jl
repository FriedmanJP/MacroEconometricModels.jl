using Test, MacroEconometricModels, Random

@testset "DF-GLS Unit Root Test" begin
    rng = Random.MersenneTwister(66778)

    y_stat = zeros(200)
    y_stat[1] = randn(rng)
    for t in 2:200; y_stat[t] = 0.5 * y_stat[t-1] + randn(rng); end
    y_rw = cumsum(randn(rng, 200))

    @testset "Basic functionality" begin
        result = dfgls_test(y_stat; regression=:constant)
        @test result isa DFGLSResult
        @test result.lags >= 0
        @test result.regression == :constant
        @test isfinite(result.statistic)
        @test isfinite(result.pt_statistic)
        @test isfinite(result.MZt)
        @test haskey(result.critical_values, 5)

        result_t = dfgls_test(y_rw; regression=:trend)
        @test result_t isa DFGLSResult
        @test result_t.regression == :trend
    end

    @testset "Power comparison" begin
        result_stat = dfgls_test(y_stat; regression=:constant)
        @test result_stat.pvalue < 0.50

        result_rw = dfgls_test(y_rw; regression=:constant)
        @test result_rw.pvalue > 0.01
    end

    @testset "Fixed lags" begin
        result = dfgls_test(y_stat; lags=4)
        @test result.lags == 4
    end

    @testset "BIC lag selection" begin
        result = dfgls_test(y_stat; lags=:bic)
        @test result isa DFGLSResult
        @test result.lags >= 0
    end

    @testset "ERS Pt statistic" begin
        result = dfgls_test(y_stat; regression=:constant)
        @test isfinite(result.pt_statistic)
        @test isfinite(result.pt_pvalue)
        @test haskey(result.pt_critical_values, 5)

        result_t = dfgls_test(y_stat; regression=:trend)
        @test isfinite(result_t.pt_statistic)
    end

    @testset "MGLS statistics" begin
        result = dfgls_test(y_stat; regression=:constant)
        @test isfinite(result.MZa)
        @test isfinite(result.MZt)
        @test isfinite(result.MSB)
        @test isfinite(result.MPT)
        @test result.MSB >= 0
        @test haskey(result.mgls_critical_values, :MZa)
        @test haskey(result.mgls_critical_values, :MZt)
        @test haskey(result.mgls_critical_values, :MSB)
        @test haskey(result.mgls_critical_values, :MPT)
    end

    @testset "StatsAPI interface" begin
        result = dfgls_test(y_stat)
        @test StatsAPI.nobs(result) == result.nobs
        @test StatsAPI.pvalue(result) == result.pvalue
        @test StatsAPI.dof(result) >= 1
    end

    @testset "Integer input" begin
        result = dfgls_test(round.(Int, y_stat * 10))
        @test result isa DFGLSResult
    end

    @testset "Error handling" begin
        @test_throws ArgumentError dfgls_test(randn(rng, 10))
        @test_throws ArgumentError dfgls_test(y_stat; regression=:none)
    end
end
