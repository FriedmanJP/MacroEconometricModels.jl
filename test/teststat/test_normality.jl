# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

using Test
using MacroEconometricModels
using Random
using LinearAlgebra
using Distributions
using StatsAPI: pvalue

@testset "Multivariate Normality Tests" begin
    Random.seed!(12345)

    # Generate Gaussian data (should not reject)
    n_obs, k = 500, 3
    Y_gauss = randn(n_obs + 2, k)
    model_gauss = estimate_var(Y_gauss, 2)

    # Generate non-Gaussian data (t-distributed, should reject)
    Y_nong = rand(TDist(3), n_obs + 2, k)
    model_nong = estimate_var(Y_nong, 2)

    @testset "Jarque-Bera Multivariate" begin
        r = jarque_bera_test(model_gauss)
        @test r isa NormalityTestResult{Float64}
        @test r.test_name == :jarque_bera
        @test r.n_vars == k
        @test r.n_obs == n_obs
        @test r.df == 2k
        @test r.statistic >= 0
        @test 0 <= r.pvalue <= 1
        @test r.components === nothing

        # Raw matrix dispatch
        r2 = jarque_bera_test(model_gauss.U)
        @test r2.statistic == r.statistic
    end

    @testset "Jarque-Bera Component-wise" begin
        r = jarque_bera_test(model_gauss; method=:component)
        @test r.test_name == :jarque_bera
        @test r.df == 2k
        @test r.components isa Vector{Float64}
        @test length(r.components) == k
        @test r.component_pvalues isa Vector{Float64}
        @test length(r.component_pvalues) == k
        @test all(r.components .>= 0)
        @test all(0 .<= r.component_pvalues .<= 1)
    end

    @testset "Mardia Skewness" begin
        r = mardia_test(model_gauss; type=:skewness)
        @test r.test_name == :mardia_skewness
        @test r.statistic >= 0
        @test 0 <= r.pvalue <= 1
        @test r.df == k * (k + 1) * (k + 2) ÷ 6
    end

    @testset "Mardia Kurtosis" begin
        r = mardia_test(model_gauss; type=:kurtosis)
        @test r.test_name == :mardia_kurtosis
        @test 0 <= r.pvalue <= 1
        @test r.df == 1
    end

    @testset "Mardia Combined" begin
        r = mardia_test(model_gauss; type=:both)
        @test r.test_name == :mardia_both
        @test r.statistic >= 0
        @test r.components isa Vector{Float64}
        @test length(r.components) == 2
    end

    @testset "Doornik-Hansen" begin
        r = doornik_hansen_test(model_gauss)
        @test r.test_name == :doornik_hansen
        @test r.df == 2k
        @test r.statistic >= 0
        @test 0 <= r.pvalue <= 1
        @test r.components isa Vector{Float64}
        @test length(r.components) == k
    end

    @testset "Henze-Zirkler" begin
        r = henze_zirkler_test(model_gauss)
        @test r.test_name == :henze_zirkler
        @test r.statistic >= 0
        @test 0 <= r.pvalue <= 1
        @test r.df == 0  # uses log-normal approximation

        # Raw matrix dispatch
        r2 = henze_zirkler_test(model_gauss.U)
        @test r2.statistic == r.statistic
    end

    @testset "Normality Test Suite" begin
        suite = normality_test_suite(model_gauss)
        @test suite isa NormalityTestSuite{Float64}
        @test length(suite.results) == 7
        @test suite.n_vars == k
        @test suite.n_obs == n_obs
        @test suite.residuals == model_gauss.U

        # Raw matrix dispatch
        suite2 = normality_test_suite(model_gauss.U)
        @test length(suite2.results) == 7

        # Test show method
        buf = IOBuffer()
        show(buf, suite)
        s = String(take!(buf))
        @test occursin("Normality Test Suite", s)
        @test occursin("jarque_bera", s)
    end

    @testset "Non-Gaussian data should reject" begin
        # With t(3) distributed data, JB should reject more often
        r_jb = jarque_bera_test(model_nong)
        r_mardia = mardia_test(model_nong; type=:both)

        # At least one test should reject with heavy-tailed data
        all_pvals = [r_jb.pvalue, r_mardia.pvalue]
        @test minimum(all_pvals) < 0.10  # at least one test is sensitive
    end

    @testset "StatsAPI pvalue" begin
        r = jarque_bera_test(model_gauss)
        @test pvalue(r) == r.pvalue
    end

    @testset "Show methods" begin
        r = jarque_bera_test(model_gauss)
        buf = IOBuffer()
        show(buf, r)
        s = String(take!(buf))
        @test occursin("Normality Test", s)
        @test occursin("Statistic", s)
        @test occursin("P-value", s)
    end

    @testset "Bivariate case" begin
        Y2 = randn(200, 2)
        model2 = estimate_var(Y2, 1)
        suite2 = normality_test_suite(model2)
        @test length(suite2.results) == 7
        @test all(r -> r.n_vars == 2, suite2.results)
    end

    @testset "Raw matrix dispatch" begin
        Random.seed!(5678)
        U_raw = randn(200, 3)
        # mardia_test with raw matrix
        r_mardia = mardia_test(U_raw)
        @test r_mardia isa MacroEconometricModels.NormalityTestResult
        @test r_mardia.n_vars == 3

        # doornik_hansen_test with raw matrix
        r_dh = doornik_hansen_test(U_raw)
        @test r_dh isa MacroEconometricModels.NormalityTestResult
        @test r_dh.n_vars == 3
    end

    @testset "Large k=5 case" begin
        Random.seed!(5679)
        U_large = randn(200, 5)
        r_jb = jarque_bera_test(U_large)
        @test r_jb.n_vars == 5
        r_hz = henze_zirkler_test(U_large)
        @test r_hz.n_vars == 5
        @test 0 <= r_hz.pvalue <= 1
    end

    @testset "Univariate k=1 edge" begin
        Random.seed!(5680)
        U_uni = randn(200, 1)
        r_jb = jarque_bera_test(U_uni)
        @test r_jb.n_vars == 1
        @test r_jb.statistic >= 0
    end
end
