using Test, MacroEconometricModels, Random, LinearAlgebra

@testset "Gregory-Hansen Cointegration Test" begin
    rng = Random.MersenneTwister(99001)

    T_gh = 200
    x = cumsum(randn(rng, T_gh))
    y = vcat(1.0 .+ 0.5 .* x[1:100], 3.0 .+ 1.5 .* x[101:200]) + 0.3 .* randn(rng, T_gh)
    Y = hcat(y, x)

    @testset "Model C (level shift)" begin
        result = gregory_hansen_test(Y; model=:C)
        @test result isa GregoryHansenResult
        @test result.model == :C
        @test result.adf_break > 0
        @test isfinite(result.adf_statistic)
        @test isfinite(result.zt_statistic)
        @test isfinite(result.za_statistic)
        @test haskey(result.adf_critical_values, 5)
        @test haskey(result.za_critical_values, 5)
        @test result.n_regressors == 1
        @test result.nobs == T_gh
    end

    @testset "Model CS (regime shift)" begin
        result = gregory_hansen_test(Y; model=:CS)
        @test result isa GregoryHansenResult
        @test result.model == :CS
        @test isfinite(result.adf_statistic)
        @test isfinite(result.zt_statistic)
        @test isfinite(result.za_statistic)
    end

    @testset "Model CT (level + trend)" begin
        result = gregory_hansen_test(Y; model=:CT)
        @test result isa GregoryHansenResult
        @test result.model == :CT
        @test isfinite(result.adf_statistic)
    end

    @testset "Fixed lags" begin
        result = gregory_hansen_test(Y; model=:C, lags=2)
        @test result isa GregoryHansenResult
        @test isfinite(result.adf_statistic)
    end

    @testset "BIC lag selection" begin
        result = gregory_hansen_test(Y; model=:C, lags=:bic)
        @test result isa GregoryHansenResult
        @test isfinite(result.adf_statistic)
    end

    @testset "Multiple regressors" begin
        x2 = cumsum(randn(rng, T_gh))
        Y3 = hcat(y, x, x2)
        result = gregory_hansen_test(Y3; model=:C)
        @test result isa GregoryHansenResult
        @test result.n_regressors == 2
    end

    @testset "StatsAPI interface" begin
        result = gregory_hansen_test(Y; model=:C)
        @test StatsAPI.nobs(result) == T_gh
        @test StatsAPI.pvalue(result) == result.adf_pvalue
        @test StatsAPI.dof(result) == result.n_regressors
    end

    @testset "Non-Float64 input" begin
        Y_int = hcat(round.(Int, y), round.(Int, x))
        result = gregory_hansen_test(Y_int)
        @test result isa GregoryHansenResult{Float64}
    end

    @testset "show method" begin
        result = gregory_hansen_test(Y; model=:C)
        io = IOBuffer()
        show(io, result)
        output = String(take!(io))
        @test occursin("Gregory-Hansen", output)
    end

    @testset "Error handling" begin
        @test_throws ArgumentError gregory_hansen_test(randn(rng, 50, 1))
        @test_throws ArgumentError gregory_hansen_test(Y; model=:invalid)
        @test_throws ArgumentError gregory_hansen_test(randn(rng, 30, 2))
    end
end
