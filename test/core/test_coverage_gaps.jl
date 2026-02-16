# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <wookyung9207@gmail.com>
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

"""
Coverage improvement tests for miscellaneous gaps.

Targets uncovered paths:
- StatsAPI interface for unit root test results (dof, pvalue, nobs)
- BVAR weighted quantiles (threaded path with prod(dims) > 1000)
- BVARPosterior Base.size/Base.length
- Various edge case branches
"""

using Test
using MacroEconometricModels
using Random
using Statistics
using StatsAPI
using LinearAlgebra

Random.seed!(7001)

# =============================================================================
# Unit Root Test StatsAPI Interface
# =============================================================================

@testset "Unit root StatsAPI interface" begin
    Random.seed!(7010)
    y_stationary = randn(200)
    y_unit_root = cumsum(randn(200))

    @testset "ADF StatsAPI" begin
        r = adf_test(y_stationary)
        @test StatsAPI.nobs(r) == r.nobs
        @test StatsAPI.nobs(r) > 0
        @test StatsAPI.pvalue(r) == r.pvalue
        @test 0.0 <= StatsAPI.pvalue(r) <= 1.0

        # dof with :constant regression
        r_const = adf_test(y_stationary; regression=:constant)
        d = StatsAPI.dof(r_const)
        @test d == r_const.lags + 2

        # dof with :none regression
        r_none = adf_test(y_stationary; regression=:none)
        d_none = StatsAPI.dof(r_none)
        @test d_none == r_none.lags + 1

        # dof with :trend regression
        r_trend = adf_test(y_stationary; regression=:trend)
        d_trend = StatsAPI.dof(r_trend)
        @test d_trend == r_trend.lags + 3
    end

    @testset "KPSS StatsAPI" begin
        r = kpss_test(y_stationary)
        @test StatsAPI.nobs(r) > 0
        @test StatsAPI.pvalue(r) == r.pvalue

        # dof with :constant
        r_const = kpss_test(y_stationary; regression=:constant)
        @test StatsAPI.dof(r_const) == 1

        # dof with :trend
        r_trend = kpss_test(y_stationary; regression=:trend)
        @test StatsAPI.dof(r_trend) == 2
    end

    @testset "PP StatsAPI" begin
        r = pp_test(y_stationary)
        @test StatsAPI.nobs(r) > 0
        @test StatsAPI.pvalue(r) == r.pvalue

        # dof with :constant
        r_const = pp_test(y_stationary; regression=:constant)
        @test StatsAPI.dof(r_const) == 2

        # dof with :none
        r_none = pp_test(y_stationary; regression=:none)
        @test StatsAPI.dof(r_none) == 1

        # dof with :trend
        r_trend = pp_test(y_stationary; regression=:trend)
        @test StatsAPI.dof(r_trend) == 3
    end

    @testset "ZA StatsAPI" begin
        r = za_test(y_unit_root)
        @test StatsAPI.nobs(r) > 0
        @test StatsAPI.pvalue(r) == r.pvalue

        # dof with :constant
        r_const = za_test(y_unit_root; regression=:constant)
        @test StatsAPI.dof(r_const) == r_const.lags + 4

        # dof with :trend
        r_trend = za_test(y_unit_root; regression=:trend)
        @test StatsAPI.dof(r_trend) == r_trend.lags + 4

        # dof with :both
        r_both = za_test(y_unit_root; regression=:both)
        @test StatsAPI.dof(r_both) == r_both.lags + 5
    end

    @testset "NgPerron StatsAPI" begin
        r = ngperron_test(y_stationary)
        @test StatsAPI.nobs(r) > 0

        # pvalue calls _ngperron_pvalue
        p = StatsAPI.pvalue(r)
        @test 0.0 <= p <= 1.0

        # dof with :constant
        r_const = ngperron_test(y_stationary; regression=:constant)
        @test StatsAPI.dof(r_const) == 1

        # dof with :trend
        r_trend = ngperron_test(y_stationary; regression=:trend)
        @test StatsAPI.dof(r_trend) == 2
    end

    @testset "Johansen StatsAPI" begin
        Y = randn(100, 3)
        r = johansen_test(Y, 2)
        @test StatsAPI.nobs(r) > 0
        @test StatsAPI.dof(r) == 2  # r.lags

        # pvalue returns minimum trace p-value
        p = StatsAPI.pvalue(r)
        @test p == minimum(r.trace_pvalues)
        @test 0.0 <= p <= 1.0
    end
end

# =============================================================================
# BVAR Weighted Quantiles: Threaded Path
# =============================================================================

@testset "BVAR weighted quantiles threaded" begin
    Random.seed!(7020)

    @testset "compute_posterior_quantiles threaded=true with large array" begin
        # prod(other_dims) = 50 * 5 * 5 = 1250 > 1000 → triggers threaded path
        samples = randn(30, 50, 5, 5)
        q_vec = [0.16, 0.5, 0.84]

        q_out, m_out = MacroEconometricModels.compute_posterior_quantiles(samples, q_vec; threaded=true)
        @test size(q_out) == (50, 5, 5, 3)
        @test size(m_out) == (50, 5, 5)

        # Compare with non-threaded
        q_out2, m_out2 = MacroEconometricModels.compute_posterior_quantiles(samples, q_vec; threaded=false)
        @test q_out ≈ q_out2
        @test m_out ≈ m_out2
    end

    @testset "compute_weighted_quantiles_threaded!" begin
        # Need large enough array: 50 * 5 * 5 = 1250 > 1000
        samples = randn(30, 50, 5, 5)
        weights = rand(30)
        weights ./= sum(weights)

        q_vec = Float64.([0.16, 0.5, 0.84])
        q_out = zeros(50, 5, 5, 3)
        m_out = zeros(50, 5, 5)

        MacroEconometricModels.compute_weighted_quantiles_threaded!(q_out, m_out, samples, weights, q_vec)

        @test size(q_out) == (50, 5, 5, 3)
        @test size(m_out) == (50, 5, 5)

        # Compare with non-threaded
        q_out2 = zeros(50, 5, 5, 3)
        m_out2 = zeros(50, 5, 5)
        MacroEconometricModels.compute_weighted_quantiles!(q_out2, m_out2, samples, weights, q_vec)

        @test q_out ≈ q_out2
        @test m_out ≈ m_out2
    end
end

# =============================================================================
# BVARPosterior Base.size / Base.length
# =============================================================================

@testset "BVARPosterior size and length" begin
    Random.seed!(7030)
    Y = randn(50, 2)
    post = estimate_bvar(Y, 1; n_draws=20)

    @test Base.length(post) == 20
    @test Base.size(post, 1) == 20
    @test_throws ErrorException Base.size(post, 2)
end

# =============================================================================
# BVAR process_posterior_samples with deprecated wrapper
# =============================================================================

@testset "process_posterior_samples deprecated wrapper" begin
    Random.seed!(7040)
    Y = randn(50, 2)
    post = estimate_bvar(Y, 1; n_draws=10)

    # The deprecated (post, p, n, func) signature should still work
    results, n_samples = MacroEconometricModels.process_posterior_samples(
        post, post.p, post.n,
        (m, Q, h) -> MacroEconometricModels.compute_irf(m, Q, h);
        horizon=5, method=:cholesky
    )

    @test length(results) == 10
    @test n_samples == 10
end
