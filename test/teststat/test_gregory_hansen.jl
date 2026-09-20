# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test, MacroEconometricModels, Random, LinearAlgebra, StatsAPI

@testset "Gregory-Hansen Cointegration Test" begin
    rng = Random.Xoshiro(99001)

    T_gh = 200
    x = cumsum(randn(rng, T_gh))
    y = vcat(1.0 .+ 0.5 .* x[1:100], 3.0 .+ 1.5 .* x[101:200]) + 0.3 .* randn(rng, T_gh)
    Y = hcat(y, x)

    # model=:C fit computed once and shared (deterministic estimator)
    gh_C = gregory_hansen_test(Y; model=:C)

    @testset "Model C (level shift)" begin
        result = gh_C
        @test result isa GregoryHansenResult
        @test result.model == :C
        @test result.adf_break > 0
        # T159: signs only — the DGP has a SLOPE change, so level-shift-only :C is
        # misspecified and correctly fails to reject (−3.56 vs CV −4.61). :CS below
        # is the right spec and carries the rejection + break-recovery pins.
        @test isfinite(result.adf_statistic) && result.adf_statistic < 0  # T159: (see note above)
        @test isfinite(result.zt_statistic) && result.zt_statistic < 0  # T159: (see note above)
        @test isfinite(result.za_statistic) && result.za_statistic < 0  # T159: (see note above)
        @test haskey(result.adf_critical_values, 5)
        @test haskey(result.za_critical_values, 5)
        @test result.n_regressors == 1
        @test result.nobs == T_gh
    end

    @testset "Model CS (regime shift)" begin
        result = gregory_hansen_test(Y; model=:CS)
        @test result isa GregoryHansenResult
        @test result.model == :CS
        # T159: :CS matches the DGP (regime shift) ⇒ decisive rejection (ADF −11.3
        # vs CV −4.95; Za −178 vs CV −47) and all three break searches land within
        # ±10 of the true t=100 (observed 96/101/100; a broken search lands at the
        # trim boundary ~30/170). No zt CV table exists ⇒ sign pin for Zt.
        @test isfinite(result.adf_statistic) && result.adf_statistic < result.adf_critical_values[5]  # T159: (see note above)
        @test isfinite(result.zt_statistic) && result.zt_statistic < 0  # T159: (see note above)
        @test isfinite(result.za_statistic) && result.za_statistic < result.za_critical_values[5]  # T159: (see note above)
        @test abs(result.adf_break - 100) <= 10
        @test abs(result.zt_break - 100) <= 10
        @test abs(result.za_break - 100) <= 10
    end

    @testset "Model CT (level + trend)" begin
        result = gregory_hansen_test(Y; model=:CT)
        @test result isa GregoryHansenResult
        @test result.model == :CT
        # T159: sign pin only — :CT (no slope change) also misses the DGP's regime
        # shift and correctly fails to reject (−4.08 vs CV −4.99).
        @test isfinite(result.adf_statistic) && result.adf_statistic < 0
    end

    @testset "Fixed lags" begin
        result = gregory_hansen_test(Y; model=:C, lags=2)
        @test result isa GregoryHansenResult
        @test isfinite(result.adf_statistic) && result.adf_statistic < 0   # T159: sign pin (−2.68)
    end

    @testset "BIC lag selection" begin
        result = gregory_hansen_test(Y; model=:C, lags=:bic)
        @test result isa GregoryHansenResult
        @test isfinite(result.adf_statistic) && result.adf_statistic < 0   # T159: sign pin (−3.56)
    end

    @testset "Multiple regressors" begin
        x2 = cumsum(randn(rng, T_gh))
        Y3 = hcat(y, x, x2)
        result = gregory_hansen_test(Y3; model=:C)
        @test result isa GregoryHansenResult
        @test result.n_regressors == 2
    end

    @testset "StatsAPI interface" begin
        result = gh_C
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
        result = gh_C
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
