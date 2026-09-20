# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test, MacroEconometricModels, Random, StatsAPI

@testset "DF-GLS Unit Root Test" begin
    rng = Random.Xoshiro(66778)

    y_stat = zeros(200)
    y_stat[1] = randn(rng)
    for t in 2:200; y_stat[t] = 0.5 * y_stat[t-1] + randn(rng); end
    y_rw = cumsum(randn(rng, 200))

    @testset "Basic functionality" begin
        result = dfgls_test(y_stat; regression=:constant)
        @test result isa DFGLSResult
        @test result.lags >= 0
        @test result.regression == :constant
        # T159: tau/MZt are (ρ̂−1)-forms, negative on both DGPs here (fixed seeds;
        # a sign flip gives positives). Pt is a variance ratio > 0.
        @test isfinite(result.statistic) && result.statistic < 0
        @test isfinite(result.pt_statistic) && result.pt_statistic > 0  # T159: (see note above)
        @test isfinite(result.MZt) && result.MZt < 0  # T159: (see note above)
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

        # T159: designed stationary-vs-RW orderings — stationary data looks more
        # stationary on every small-reject (Pt/MPT/MSB) and left-tailed (MZt/MZa)
        # margin. (No tau ordering: MAIC lags leave the stationary tau (−0.62)
        # above the RW tau (−1.05) on this seed.)
        @test result_stat.pt_statistic < result_rw.pt_statistic
        @test result_stat.MZt < result_rw.MZt
        @test result_stat.MZa < result_rw.MZa
        @test result_stat.MSB < result_rw.MSB
        @test result_stat.MPT < result_rw.MPT
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
        # T159: Pt is a variance ratio (> 0); pt_pvalue is a probability (already strong).
        @test isfinite(result.pt_statistic) && result.pt_statistic > 0
        @test isfinite(result.pt_pvalue) && 0 <= result.pt_pvalue <= 1
        @test haskey(result.pt_critical_values, 5)

        result_t = dfgls_test(y_stat; regression=:trend)
        @test isfinite(result_t.pt_statistic) && result_t.pt_statistic > 0  # T159: variance ratio (see above)
    end

    @testset "MGLS statistics" begin
        result = dfgls_test(y_stat; regression=:constant)
        # T159: MZa/MZt are (ρ̂−1)-forms (negative here); MSB/MPT are ratios (> 0).
        # Cross-DGP orderings live in "Power comparison" above.
        @test isfinite(result.MZa) && result.MZa < 0
        @test isfinite(result.MZt) && result.MZt < 0  # T159: (see note above)
        @test isfinite(result.MSB) && result.MSB > 0  # T159: (see note above)
        @test isfinite(result.MPT) && result.MPT > 0  # T159: (see note above)
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
