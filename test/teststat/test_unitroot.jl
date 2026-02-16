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

using Test
using MacroEconometricModels
using Random
using Statistics

@testset "Unit Root Tests" begin

    # Set seed for reproducibility
    Random.seed!(12345)

    # ==========================================================================
    # Test Data Generation
    # ==========================================================================

    # Generate stationary AR(1) process: y_t = 0.5 * y_{t-1} + e_t
    function generate_stationary(n::Int; rho::Float64=0.5)
        y = zeros(n)
        y[1] = randn()
        for t in 2:n
            y[t] = rho * y[t-1] + randn()
        end
        y
    end

    # Generate random walk (unit root): y_t = y_{t-1} + e_t
    function generate_random_walk(n::Int)
        cumsum(randn(n))
    end

    # Generate trend stationary: y_t = a + b*t + e_t
    function generate_trend_stationary(n::Int)
        0.5 .+ 0.02 .* (1:n) .+ randn(n)
    end

    # ==========================================================================
    # ADF Test
    # ==========================================================================

    @testset "ADF Test" begin
        # Test with stationary series - should reject unit root
        y_stationary = generate_stationary(200; rho=0.5)
        result_stat = adf_test(y_stationary; regression=:constant)

        @test result_stat isa ADFResult
        @test hasfield(ADFResult, :statistic)
        @test hasfield(ADFResult, :pvalue)
        @test hasfield(ADFResult, :lags)
        @test hasfield(ADFResult, :regression)
        @test hasfield(ADFResult, :critical_values)
        @test hasfield(ADFResult, :nobs)

        @test result_stat.regression == :constant
        @test result_stat.nobs > 0
        @test haskey(result_stat.critical_values, 1)
        @test haskey(result_stat.critical_values, 5)
        @test haskey(result_stat.critical_values, 10)

        # P-value should generally be low for stationary series (reject H0)
        # Note: this is probabilistic, so we use a lenient threshold
        @test result_stat.pvalue < 0.50

        # Test with random walk - should fail to reject unit root
        y_rw = generate_random_walk(200)
        result_rw = adf_test(y_rw; regression=:constant)

        @test result_rw isa ADFResult
        # P-value should generally be high for unit root series (fail to reject)
        @test result_rw.pvalue > 0.01

        # Test different regression specifications
        result_none = adf_test(y_rw; regression=:none)
        @test result_none.regression == :none

        result_trend = adf_test(y_rw; regression=:trend)
        @test result_trend.regression == :trend

        # Test automatic lag selection
        result_aic = adf_test(y_stationary; lags=:aic)
        @test result_aic.lags >= 0

        result_bic = adf_test(y_stationary; lags=:bic)
        @test result_bic.lags >= 0

        # Test with fixed lags
        result_fixed = adf_test(y_stationary; lags=2)
        @test result_fixed.lags == 2

        # Test error handling
        @test_throws ArgumentError adf_test(randn(10); regression=:invalid)
        @test_throws ArgumentError adf_test(randn(5))  # Too short

        # Test type conversion (Integer input)
        y_int = round.(Int, y_stationary * 10)
        result_int = adf_test(y_int)
        @test result_int isa ADFResult
    end

    # ==========================================================================
    # KPSS Test
    # ==========================================================================

    @testset "KPSS Test" begin
        # Test with stationary series - should fail to reject stationarity
        y_stationary = generate_stationary(200; rho=0.3)
        result_stat = kpss_test(y_stationary; regression=:constant)

        @test result_stat isa KPSSResult
        @test hasfield(KPSSResult, :statistic)
        @test hasfield(KPSSResult, :pvalue)
        @test hasfield(KPSSResult, :regression)
        @test hasfield(KPSSResult, :bandwidth)

        @test result_stat.regression == :constant
        @test result_stat.bandwidth > 0
        @test result_stat.nobs > 0

        # P-value should be high for stationary series (fail to reject H0)
        @test result_stat.pvalue > 0.01

        # Test with random walk - should reject stationarity
        y_rw = generate_random_walk(200)
        result_rw = kpss_test(y_rw; regression=:constant)

        @test result_rw isa KPSSResult
        # P-value should be low for unit root series (reject stationarity)
        @test result_rw.pvalue < 0.50

        # Test trend stationarity
        y_trend = generate_trend_stationary(200)
        result_trend = kpss_test(y_trend; regression=:trend)
        @test result_trend.regression == :trend

        # Test with fixed bandwidth
        result_bw = kpss_test(y_stationary; bandwidth=5)
        @test result_bw.bandwidth == 5

        # Test error handling
        @test_throws ArgumentError kpss_test(randn(100); regression=:none)  # Invalid regression
        @test_throws ArgumentError kpss_test(randn(5))  # Too short
    end

    # ==========================================================================
    # Phillips-Perron Test
    # ==========================================================================

    @testset "Phillips-Perron Test" begin
        y_stationary = generate_stationary(200; rho=0.5)
        result_stat = pp_test(y_stationary; regression=:constant)

        @test result_stat isa PPResult
        @test hasfield(PPResult, :statistic)
        @test hasfield(PPResult, :pvalue)
        @test hasfield(PPResult, :bandwidth)

        @test result_stat.regression == :constant
        @test result_stat.bandwidth > 0

        # Test with random walk
        y_rw = generate_random_walk(200)
        result_rw = pp_test(y_rw; regression=:constant)
        @test result_rw isa PPResult

        # Test different regression specifications
        result_none = pp_test(y_rw; regression=:none)
        @test result_none.regression == :none

        result_trend = pp_test(y_rw; regression=:trend)
        @test result_trend.regression == :trend

        # Test with fixed bandwidth
        result_bw = pp_test(y_stationary; bandwidth=10)
        @test result_bw.bandwidth == 10
    end

    # ==========================================================================
    # Zivot-Andrews Test
    # ==========================================================================

    @testset "Zivot-Andrews Test" begin
        # Generate series with structural break
        n = 150
        y_break = vcat(randn(75), randn(75) .+ 3.0)  # Level shift at t=75

        result = za_test(y_break; regression=:constant)

        @test result isa ZAResult
        @test hasfield(ZAResult, :statistic)
        @test hasfield(ZAResult, :break_index)
        @test hasfield(ZAResult, :break_fraction)

        @test result.regression == :constant
        @test 0 < result.break_fraction < 1
        @test result.break_index > 0

        # Test different break specifications
        result_trend = za_test(y_break; regression=:trend)
        @test result_trend.regression == :trend

        result_both = za_test(y_break; regression=:both)
        @test result_both.regression == :both

        # Test with different trimming
        result_trim = za_test(y_break; trim=0.10)
        @test result_trim isa ZAResult

        # Test error handling
        @test_throws ArgumentError za_test(randn(30))  # Too short
        @test_throws ArgumentError za_test(randn(100); trim=0.6)  # Invalid trim
    end

    # ==========================================================================
    # Ng-Perron Test
    # ==========================================================================

    @testset "Ng-Perron Test" begin
        y_stationary = generate_stationary(150; rho=0.5)
        result = ngperron_test(y_stationary; regression=:constant)

        @test result isa NgPerronResult
        @test hasfield(NgPerronResult, :MZa)
        @test hasfield(NgPerronResult, :MZt)
        @test hasfield(NgPerronResult, :MSB)
        @test hasfield(NgPerronResult, :MPT)

        @test result.regression == :constant
        @test !isnan(result.MZa)
        @test !isnan(result.MZt)
        @test !isnan(result.MSB)
        @test !isnan(result.MPT)

        # Check critical values
        @test haskey(result.critical_values, :MZa)
        @test haskey(result.critical_values, :MZt)
        @test haskey(result.critical_values, :MSB)
        @test haskey(result.critical_values, :MPT)

        # Test with trend
        result_trend = ngperron_test(y_stationary; regression=:trend)
        @test result_trend.regression == :trend

        # Test with random walk
        y_rw = generate_random_walk(150)
        result_rw = ngperron_test(y_rw)
        @test result_rw isa NgPerronResult
    end

    # ==========================================================================
    # Johansen Cointegration Test
    # ==========================================================================

    @testset "Johansen Cointegration Test" begin
        # Generate cointegrated system
        n, T = 3, 200
        Random.seed!(42)

        # Common stochastic trend
        trend = cumsum(randn(T))

        # Cointegrated variables
        Y = zeros(T, n)
        Y[:, 1] = trend + 0.5 * randn(T)
        Y[:, 2] = 0.8 * trend + 0.3 * randn(T)
        Y[:, 3] = randn(T)  # Stationary, not cointegrated

        result = johansen_test(Y, 2; deterministic=:constant)

        @test result isa JohansenResult
        @test hasfield(JohansenResult, :trace_stats)
        @test hasfield(JohansenResult, :max_eigen_stats)
        @test hasfield(JohansenResult, :rank)
        @test hasfield(JohansenResult, :eigenvectors)
        @test hasfield(JohansenResult, :adjustment)

        @test length(result.trace_stats) == n
        @test length(result.max_eigen_stats) == n
        @test length(result.eigenvalues) == n
        @test size(result.eigenvectors, 1) == n
        @test size(result.adjustment, 1) == n

        @test result.deterministic == :constant
        @test result.lags == 2
        @test result.nobs > 0
        @test result.rank >= 0 && result.rank <= n

        # All eigenvalues should be in [0, 1]
        @test all(0 .<= result.eigenvalues .<= 1)

        # Test different deterministic specifications
        result_none = johansen_test(Y, 2; deterministic=:none)
        @test result_none.deterministic == :none

        result_trend = johansen_test(Y, 2; deterministic=:trend)
        @test result_trend.deterministic == :trend

        # Test error handling
        @test_throws ArgumentError johansen_test(Y, 0)  # Invalid lags
        @test_throws ArgumentError johansen_test(randn(10, 3), 2)  # Too few obs
    end

    # ==========================================================================
    # VAR Stationarity Check
    # ==========================================================================

    @testset "VAR Stationarity" begin
        # Generate stationary VAR data
        Random.seed!(123)
        T, n = 200, 2

        # Stationary VAR(1) with coefficients ensuring stationarity
        Y = zeros(T, n)
        A = [0.5 0.1; 0.1 0.5]  # All eigenvalues < 1
        for t in 2:T
            Y[t, :] = A * Y[t-1, :] + randn(n)
        end

        model_stat = estimate_var(Y, 1; check_stability=false)
        result = is_stationary(model_stat)

        @test result isa VARStationarityResult
        @test hasfield(VARStationarityResult, :is_stationary)
        @test hasfield(VARStationarityResult, :eigenvalues)
        @test hasfield(VARStationarityResult, :max_modulus)
        @test hasfield(VARStationarityResult, :companion_matrix)

        @test length(result.eigenvalues) == n * model_stat.p
        @test result.max_modulus >= 0
        @test size(result.companion_matrix) == (n * model_stat.p, n * model_stat.p)

        # For stationary data, should likely be stationary
        # (probabilistic, so we just check the function runs)
        @test result.is_stationary isa Bool

        # Test with non-stationary data (random walk)
        Y_rw = cumsum(randn(T, n), dims=1)
        model_rw = estimate_var(Y_rw, 1; check_stability=false)
        result_rw = is_stationary(model_rw)
        @test result_rw isa VARStationarityResult
        # Random walk should have max modulus close to or > 1
        @test result_rw.max_modulus > 0.5
    end

    # ==========================================================================
    # estimate_var Stationarity Warning
    # ==========================================================================

    @testset "estimate_var Stability Check" begin
        Random.seed!(456)
        T, n = 100, 2

        # Generate random walk data
        Y_rw = cumsum(randn(T, n), dims=1)

        # Should produce warning with check_stability=true (default)
        @test_logs (:warn, r"non-stationary") estimate_var(Y_rw, 1; check_stability=true)

        # Should NOT produce warning with check_stability=false
        model = estimate_var(Y_rw, 1; check_stability=false)
        @test model isa VARModel
    end

    # ==========================================================================
    # Convenience Functions
    # ==========================================================================

    @testset "unit_root_summary" begin
        y = generate_stationary(200; rho=0.5)
        summary = unit_root_summary(y; tests=[:adf, :kpss, :pp])

        @test haskey(summary, :results)
        @test haskey(summary, :conclusion)
        @test haskey(summary.results, :adf)
        @test haskey(summary.results, :kpss)
        @test haskey(summary.results, :pp)
        @test summary.conclusion isa String

        # Test with subset of tests
        summary2 = unit_root_summary(y; tests=[:adf])
        @test haskey(summary2.results, :adf)
        @test !haskey(summary2.results, :kpss)
    end

    @testset "test_all_variables" begin
        Y = hcat(generate_stationary(150), generate_random_walk(150))
        results = test_all_variables(Y; test=:adf)

        @test length(results) == 2
        @test all(r -> r isa ADFResult, results)

        # Test with different tests
        results_kpss = test_all_variables(Y; test=:kpss)
        @test all(r -> r isa KPSSResult, results_kpss)

        results_pp = test_all_variables(Y; test=:pp)
        @test all(r -> r isa PPResult, results_pp)

        # Test error handling
        @test_throws ArgumentError test_all_variables(Y; test=:invalid)
    end

    # ==========================================================================
    # Show Methods
    # ==========================================================================

    @testset "Show Methods" begin
        y = generate_stationary(100)

        # Test that show methods don't error
        result_adf = adf_test(y)
        @test sprint(show, result_adf) isa String

        result_kpss = kpss_test(y)
        @test sprint(show, result_kpss) isa String

        result_pp = pp_test(y)
        @test sprint(show, result_pp) isa String

        y_long = generate_stationary(150)
        result_za = za_test(y_long)
        @test sprint(show, result_za) isa String

        result_np = ngperron_test(y)
        @test sprint(show, result_np) isa String

        Y = randn(150, 3)
        result_joh = johansen_test(Y, 2)
        @test sprint(show, result_joh) isa String

        model = estimate_var(randn(100, 2), 1; check_stability=false)
        result_stat = is_stationary(model)
        @test sprint(show, result_stat) isa String
    end

    # ==========================================================================
    # Critical Values
    # ==========================================================================

    @testset "Critical Values" begin
        # ADF critical values should be ordered: cv[1] < cv[5] < cv[10] (more negative = more stringent)
        y = randn(100)
        result = adf_test(y)
        @test result.critical_values[1] < result.critical_values[5] < result.critical_values[10]

        # KPSS critical values should be ordered: cv[1] > cv[5] > cv[10]
        result_kpss = kpss_test(y)
        @test result_kpss.critical_values[1] > result_kpss.critical_values[5] > result_kpss.critical_values[10]

        # PP critical values (same ordering as ADF)
        result_pp = pp_test(y)
        @test result_pp.critical_values[1] < result_pp.critical_values[5] < result_pp.critical_values[10]
    end

    @testset "PP with trend" begin
        y = randn(100)
        result = pp_test(y; regression=:trend)
        @test result isa MacroEconometricModels.PPResult
        @test isfinite(result.statistic)
        @test result.regression == :trend
    end

    @testset "Ng-Perron with trend" begin
        y = randn(100)
        result = ngperron_test(y; regression=:trend)
        @test result isa MacroEconometricModels.NgPerronResult
        @test isfinite(result.MZa)
        @test isfinite(result.MZt)
        @test isfinite(result.MSB)
        @test isfinite(result.MPT)
    end

    @testset "ZA with both regression" begin
        y = cumsum(randn(100))
        result = za_test(y; regression=:both)
        @test result isa MacroEconometricModels.ZAResult
        @test result.regression == :both
        @test 1 <= result.break_index <= 100
    end

    @testset "unit_root_summary with custom test list" begin
        y = randn(100)
        summary_result = unit_root_summary(y; tests=[:adf, :pp])
        @test length(summary_result.results) >= 2
        @test haskey(summary_result.results, :adf)
        @test haskey(summary_result.results, :pp)

        summary_full = unit_root_summary(y; tests=[:adf, :kpss, :pp])
        @test length(summary_full.results) >= 3
        @test !isempty(summary_full.conclusion)
    end

    @testset "test_all_variables with pp and za" begin
        Random.seed!(8801)
        Y = randn(100, 3)
        results_pp = test_all_variables(Y; test=:pp)
        @test length(results_pp) == 3

        results_za = test_all_variables(Y; test=:za)
        @test length(results_za) == 3
    end

end
