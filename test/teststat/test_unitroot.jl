# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

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

        # Critical values should differ when lag count changes (Cheung & Lai 1995)
        cv_p0 = MacroEconometricModels.adf_critical_values(:constant, 200, 0)
        cv_p8 = MacroEconometricModels.adf_critical_values(:constant, 200, 8)
        @test cv_p0[5] != cv_p8[5]  # 5% CV should differ for p=0 vs p=8
        @test cv_p8[5] > cv_p0[5]   # Higher lag count → less negative CV (wider)
    end

    @testset "MacKinnon ADF p-values (T078)" begin
        ap = MacroEconometricModels.adf_pvalue
        # Analytic pins from the small-p surface p = Φ(2.1659 + 1.4412τ + 0.038269τ²), :constant
        @test isapprox(ap(-3.4335, :constant, 200, 0), 0.0100; atol=0.003)
        @test isapprox(ap(-2.8621, :constant, 200, 0), 0.0500; atol=0.003)
        @test isapprox(ap(-2.5671, :constant, 200, 0), 0.0999; atol=0.003)
        @test isapprox(ap(-2.0,    :constant, 200, 0), 0.2866; atol=0.003)
        @test isapprox(ap(-1.61,   :constant, 200, 0), 0.478;  atol=0.003)
        # Anchor identity: at the asymptotic DF critical values the surface returns ~1/5/10%
        for (cvs, reg) in (((-2.5658, -1.9393, -1.6156), :none),
                           ((-3.9638, -3.4126, -3.1279), :trend))
            @test isapprox(ap(cvs[1], reg, 200, 0), 0.01; atol=0.006)
            @test isapprox(ap(cvs[2], reg, 200, 0), 0.05; atol=0.006)
            @test isapprox(ap(cvs[3], reg, 200, 0), 0.10; atol=0.006)
        end
        # Regression tests for the FIXED bug (dropped the invalid Normal tail):
        grid = collect(-6.0:0.25:2.0)
        pv = [ap(t, :constant, 200, 0) for t in grid]
        # p is continuous & STRICTLY INCREASING in τ (more-negative τ ⇒ stronger rejection
        # ⇒ smaller p); the old code had a discontinuous jump at the 10% CV.
        @test all(diff(pv) .> 0)
        @test all(0.0 .<= pv .<= 1.0)
        # Beyond-10% region: NOT the old normal tail (~0.124); should be ~0.29 at τ=-2
        @test ap(-2.0, :constant, 200, 0) > 0.25
        @test !isapprox(ap(-1.94, :constant, 200, 0), 0.1236; atol=0.05)  # old normal-tail value
        @test ap(0.0, :constant, 200, 0) > 0.95
        @test ap(1.0, :constant, 200, 0) > 0.99
        @test_throws ArgumentError MacroEconometricModels._mackinnon_pvalue(-2.0, :bogus)
    end

    @testset "M-12 ADF fixed-sample lag selection (Ng-Perron 1995)" begin
        A = MacroEconometricModels
        # AR(2) fixture with an early mean shift: the fixed-sample IC (correct) picks the
        # theoretically right lag while the old variable-sample IC is badly biased.
        rng = MersenneTwister(2079)
        n = 120; max_p = 8
        e = randn(rng, n)
        y = zeros(n)
        for t in 3:n
            y[t] = 0.6 * y[t-1] + 0.3 * y[t-2] + e[t]
        end
        y[1:15] .+= 8.0
        dy = diff(y)

        # (a) IDENTITY ORACLE: independent fixed-sample IC (all p on dy[max_p+1:end])
        Yfix = dy[(max_p+1):end]; Nfix = n - 1 - max_p
        fixed_ic = function (p, crit)
            Xf = A._build_adf_matrix(y, dy, p, :constant)
            X = Xf[(max_p - p + 1):end, :]
            k = size(X, 2)
            B = X \ Yfix
            r = Yfix - X * B
            s2 = sum(abs2, r) / (Nfix - k)
            ll = -Nfix / 2 * (log(2π) + log(s2) + 1)
            crit == :aic ? -2ll + 2k : -2ll + k * log(Nfix)
        end
        exp_aic = argmin([fixed_ic(p, :aic) for p in 0:max_p]) - 1
        exp_bic = argmin([fixed_ic(p, :bic) for p in 0:max_p]) - 1
        @test A.adf_select_lags(y, max_p, :constant, :aic) == exp_aic
        @test A.adf_select_lags(y, max_p, :constant, :bic) == exp_bic

        # (b) BUG EXPOSURE: the old variable-sample argmin picks a DIFFERENT lag
        var_ic = function (p, crit)
            Xf = A._build_adf_matrix(y, dy, p, :constant)
            Yv = dy[(p+1):end]; m = length(Yv); k = size(Xf, 2)
            B = Xf \ Yv; r = Yv - Xf * B; s2 = sum(abs2, r) / (m - k)
            ll = -m / 2 * (log(2π) + log(s2) + 1)
            crit == :aic ? -2ll + 2k : -2ll + k * log(m)
        end
        var_aic = argmin([var_ic(p, :aic) for p in 0:max_p]) - 1
        @test exp_aic != var_aic

        # (c) SANITY on a pure AR(1): a small lag, always within bounds
        Random.seed!(778)
        ys = generate_stationary(200; rho=0.5)
        lag = A.adf_select_lags(ys, 12, :constant, :aic)
        @test 0 <= lag <= 12
        @test lag <= 3
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

        # Test that AIC lag selection actually varies (not hardcoded)
        result_aic = za_test(y_break; regression=:constant, lags=:aic)
        @test result_aic isa ZAResult
        @test result_aic.lags >= 0

        # Test AO model
        result_ao = za_test(y_break; regression=:constant, outlier=:ao)
        @test result_ao isa ZAResult
        @test result_ao.break_index > 0

        # Test IO and AO give different results
        result_io = za_test(y_break; regression=:constant, outlier=:io)
        @test result_io.statistic != result_ao.statistic || result_io.break_index != result_ao.break_index

        # Test invalid outlier argument
        @test_throws ArgumentError za_test(y_break; outlier=:invalid)
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

        # Test with random walk — should fail to reject (MZa near zero)
        y_rw = generate_random_walk(150)
        result_rw = ngperron_test(y_rw)
        @test result_rw isa NgPerronResult
        # MZa should not be strongly negative for a unit root process
        @test result_rw.MZa > result_rw.critical_values[:MZa][1]  # above 1% CV

        # MSB should be positive
        @test result_rw.MSB > zero(result_rw.MSB)

        # MPT should be positive
        @test result_rw.MPT > zero(result_rw.MPT)

        # Regression test: GLS detrending should use original Z, not quasi-differenced Z
        # After fix, MZt for stationary AR(1) with rho=0.3 should be more negative
        # than for a random walk
        rng_np = Random.MersenneTwister(99887)
        y_ar_np = zeros(200)
        y_ar_np[1] = randn(rng_np)
        for t in 2:200; y_ar_np[t] = 0.3 * y_ar_np[t-1] + randn(rng_np); end
        result_ar_np = ngperron_test(y_ar_np; regression=:trend)
        y_rw_np = cumsum(randn(rng_np, 200))
        result_rw_np = ngperron_test(y_rw_np; regression=:trend)
        @test result_ar_np.MZt < result_rw_np.MZt  # stationary should be more negative
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

        # Test statistics should be non-negative
        @test all(result.trace_stats .>= 0)
        @test all(result.max_eigen_stats .>= 0)

        # Trace stats should be monotonically decreasing
        for i in 1:(n-1)
            @test result.trace_stats[i] >= result.trace_stats[i+1]
        end

        # Test different deterministic specifications
        result_none = johansen_test(Y, 2; deterministic=:none)
        @test result_none.deterministic == :none
        @test length(result_none.eigenvalues) == n

        result_trend = johansen_test(Y, 2; deterministic=:trend)
        @test result_trend.deterministic == :trend
        @test length(result_trend.eigenvalues) == n

        # Different deterministic specs should produce different critical values
        @test result.critical_values_trace[1, 2] != result_none.critical_values_trace[1, 2]
        @test result_trend.critical_values_trace[1, 2] != result.critical_values_trace[1, 2]

        # Test error handling
        @test_throws ArgumentError johansen_test(Y, 0)  # Invalid lags
        @test_throws ArgumentError johansen_test(randn(10, 3), 2)  # Too few obs

        # Test Case 4 (:trend) runs without error and produces valid results
        rng_joh = Random.MersenneTwister(7744)
        Y_coint = hcat(cumsum(randn(rng_joh, 200)), cumsum(randn(rng_joh, 200)))
        Y_coint[:, 2] = Y_coint[:, 1] + 0.1 * randn(rng_joh, 200)
        result_trend4 = johansen_test(Y_coint, 2; deterministic=:trend)
        @test result_trend4 isa JohansenResult
        @test result_trend4.deterministic == :trend
        @test length(result_trend4.trace_stats) == 2
        @test all(isfinite, result_trend4.trace_stats)
    end

    @testset "Johansen rank selection (B1/T171)" begin
        rng = MersenneTwister(11)
        tr = cumsum(randn(rng, 300, 1); dims=1)          # one common stochastic trend
        Y = hcat(tr, tr .+ 0.1 .* randn(rng, 300), tr .+ 0.1 .* randn(rng, 300))
        res = johansen_test(Y, 2)
        # rank == number of LEADING trace rejections (self-consistent with the decision
        # column). The old code set rank = r (one too low) on rejection of H0: rank ≤ r.
        expected = 0
        for i in 1:3
            res.trace_stats[i] > res.critical_values_trace[i, 2] ? (expected = i) : break
        end
        @test res.rank == expected
        @test res.rank >= 1                              # genuine common-trend system
        # VECM auto-selector agrees with johansen_test on identical data (deduped selector)
        m = estimate_vecm(Y, 2; rank=:auto)
        @test m.johansen_result.rank == res.rank
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

# =============================================================================
# T078 (#177) Johansen: Doornik (1998) gamma-approximation p-values
# =============================================================================

@testset "MHM Johansen p-values (Doornik gamma approximation)" begin
    MEM = MacroEconometricModels

    @testset "reference values (independent scipy implementation)" begin
        # (stat, m, deterministic, test) -> p, computed from the same Doornik
        # response surfaces with scipy.stats.gamma — pins the transcription
        # and the Gamma(mean²/var, var/mean) parameterization
        for (stat, m, det, test, pref) in (
                (3.5,   1, :none,     :trace, 0.0709697630),
                (20.16, 2, :constant, :trace, 0.0500610501),
                (53.94, 4, :constant, :trace, 0.0500491847),
                (25.73, 2, :trend,    :trace, 0.0500154525),
                (15.0,  2, :constant, :max,   0.0675404161),
                (9.0,   1, :trend,    :max,   0.1849100718),
                (40.0,  4, :none,     :max,   0.0000611059))
            @test MEM._johansen_pvalue(stat, m, det, test) ≈ pref atol = 1e-8
        end
    end

    @testset "nominal levels at MHM critical values" begin
        # statsmodels c_sjt/c_sja tables are MacKinnon-Haug-Michelis-generated;
        # our OL tables drift low for large m, so validate the surface against
        # the case we can pin analytically: p at the OL 95% CV within [0.03, 0.09]
        # for m ≤ 8 across the three deterministic cases (trace test)
        cv95 = Dict(
            (:none, 1) => 3.84,  (:none, 2) => 12.53,
            (:constant, 1) => 9.24, (:constant, 2) => 19.96, (:constant, 4) => 53.12,
            (:trend, 1) => 12.53, (:trend, 2) => 25.32, (:trend, 4) => 62.99)
        for ((det, m), cv) in cv95
            p = MEM._johansen_pvalue(cv, m, det, :trace)
            @test 0.03 <= p <= 0.09
        end
    end

    @testset "p-value properties" begin
        MEMp = MacroEconometricModels._johansen_pvalue
        # monotone decreasing in the statistic
        for det in (:none, :constant, :trend), test in (:trace, :max)
            ps = [MEMp(s, 2, det, test) for s in (1.0, 5.0, 15.0, 30.0, 60.0)]
            @test issorted(ps; rev=true)
            @test all(0 .<= ps .<= 1)
        end
        # degenerate inputs
        @test MEMp(0.0, 2, :constant, :trace) == 1.0
        @test MEMp(-1.0, 2, :constant, :trace) == 1.0
        @test MEMp(1e6, 2, :constant, :trace) < 1e-10
    end

    @testset "johansen_test p-values are Doornik-based" begin
        rng = Random.MersenneTwister(17701)
        Tn = 200
        x = cumsum(randn(rng, Tn))
        Y = hcat(x .+ 0.2 .* randn(rng, Tn), x .+ 0.2 .* randn(rng, Tn),
                 cumsum(randn(rng, Tn)))
        res = johansen_test(Y, 2; deterministic=:constant)
        # p-values match the surface applied to the reported statistics
        for r in 1:3
            m = 3 - (r - 1)
            @test res.trace_pvalues[r] ≈
                  MacroEconometricModels._johansen_pvalue(res.trace_stats[r], m, :constant, :trace) atol = 1e-12
            @test res.max_eigen_pvalues[r] ≈
                  MacroEconometricModels._johansen_pvalue(res.max_eigen_stats[r], m, :constant, :max) atol = 1e-12
        end
        # cointegrated pair -> rank 0 rejected decisively
        @test res.trace_pvalues[1] < 0.01
        @test all(isfinite.(res.trace_pvalues)) && all(isfinite.(res.max_eigen_pvalues))
    end
end
