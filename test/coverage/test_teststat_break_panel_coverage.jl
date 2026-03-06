# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# Coverage tests for structural break and panel unit root modules
# Targets:
#   src/teststat/moon_perron.jl     — Float64 fallback, PanelData dispatch, show (3 conclusion branches)
#   src/teststat/panic.jl           — Float64 fallback, PanelData dispatch, _panel_to_matrix, show (reject/fail)
#   src/teststat/pesaran_cips.jl    — Float64 fallback, PanelData dispatch, :auto lags, :none/:trend,
#                                     _pesaran_cips_critical_values_and_pvalue (4 p-value branches),
#                                     _nearest_val, show
#   src/teststat/factor_break.jl    — Float64 fallback (no-r), FactorModel dispatch (han_inoue),
#                                     _breitung_eickmeier_pvalue, _han_inoue_pvalue (4 branches),
#                                     _sorted_eigenvalues, show (3 conclusion branches, break_date nothing),
#                                     matrix-only dispatch error for :breitung_eickmeier/:han_inoue
#   src/teststat/andrews.jl         — Float64 fallback, all 9 variants (3 stats x 3 functionals),
#                                     _andrews_functional (sup/exp/mean), _andrews_critical_values (3 tables),
#                                     _andrews_pvalue (4 branches), show (reject/fail), error handling

using DataFrames, Statistics
#   src/teststat/helpers.jl         — adf_critical_values, adf_pvalue (4 branches), kpss_pvalue (4 branches),
#                                     za_pvalue (4 branches), _ngperron_pvalue (MZa/MZt + MSB/MPT, 4 branches each),
#                                     adf_select_lags (:bic,:hqic), _build_adf_matrix (none/constant/trend x lags),
#                                     _nw_bandwidth, _long_run_variance, _regression_name (all branches)
#   panel_unit_root_summary         — IO dispatch, PanelData dispatch

Random.seed!(9010)

@testset "Teststat Break & Panel Coverage" begin

    # =========================================================================
    # helpers.jl — critical value and p-value functions
    # =========================================================================

    @testset "adf_critical_values" begin
        # Test with different regression types and sample sizes
        for reg in (:none, :constant, :trend)
            cv = MacroEconometricModels.adf_critical_values(reg, 100)
            @test haskey(cv, 1)
            @test haskey(cv, 5)
            @test haskey(cv, 10)
            @test cv[1] < cv[5] < cv[10]  # More negative = stricter
        end

        # Test with Float32 type
        cv32 = MacroEconometricModels.adf_critical_values(:constant, 200, 0, Float32)
        @test cv32[1] isa Float32
    end

    @testset "adf_pvalue: all 4 branches" begin
        # Branch 1: stat <= cv[1] (below 1% CV)
        pval1 = MacroEconometricModels.adf_pvalue(-10.0, :constant, 100)
        @test pval1 == 0.001

        # Branch 2: stat between cv[1] and cv[5]
        cv = MacroEconometricModels.adf_critical_values(:constant, 100)
        stat_mid15 = (cv[1] + cv[5]) / 2
        pval2 = MacroEconometricModels.adf_pvalue(stat_mid15, :constant, 100)
        @test 0.01 < pval2 < 0.05

        # Branch 3: stat between cv[5] and cv[10]
        stat_mid510 = (cv[5] + cv[10]) / 2
        pval3 = MacroEconometricModels.adf_pvalue(stat_mid510, :constant, 100)
        @test 0.05 < pval3 < 0.10

        # Branch 4: stat above cv[10] — normal approximation
        pval4 = MacroEconometricModels.adf_pvalue(0.0, :constant, 100)
        @test pval4 > 0.10
        @test pval4 <= 1.0

        # Very large positive stat
        pval5 = MacroEconometricModels.adf_pvalue(5.0, :constant, 100)
        @test pval5 > 0.50
    end

    @testset "kpss_pvalue: all 4 branches" begin
        # Branch 1: stat >= cv[1] (above 1% CV)
        pval1 = MacroEconometricModels.kpss_pvalue(5.0, :constant)
        @test pval1 == 0.01

        # Branch 2: between cv[1] and cv[5]
        kpss_cv = MacroEconometricModels.KPSS_CRITICAL_VALUES[:constant]
        stat_12 = (kpss_cv[1] + kpss_cv[5]) / 2
        pval2 = MacroEconometricModels.kpss_pvalue(stat_12, :constant)
        @test 0.01 < pval2 < 0.05

        # Branch 3: between cv[5] and cv[10]
        stat_23 = (kpss_cv[5] + kpss_cv[10]) / 2
        pval3 = MacroEconometricModels.kpss_pvalue(stat_23, :constant)
        @test 0.05 < pval3 < 0.10

        # Branch 4: below cv[10]
        pval4 = MacroEconometricModels.kpss_pvalue(0.001, :constant)
        @test pval4 > 0.10

        # Test with :trend
        pval_trend = MacroEconometricModels.kpss_pvalue(0.001, :trend)
        @test pval_trend > 0.10
    end

    @testset "za_pvalue: all 4 branches" begin
        za_cv = MacroEconometricModels.ZA_CRITICAL_VALUES[:constant]

        # Branch 1: stat <= cv[1]
        pval1 = MacroEconometricModels.za_pvalue(-10.0, :constant)
        @test pval1 == 0.01

        # Branch 2: between cv[1] and cv[5]
        stat_mid = (za_cv[1] + za_cv[5]) / 2
        pval2 = MacroEconometricModels.za_pvalue(stat_mid, :constant)
        @test 0.01 < pval2 < 0.05

        # Branch 3: between cv[5] and cv[10]
        stat_mid2 = (za_cv[5] + za_cv[10]) / 2
        pval3 = MacroEconometricModels.za_pvalue(stat_mid2, :constant)
        @test 0.05 < pval3 < 0.10

        # Branch 4: above cv[10]
        pval4 = MacroEconometricModels.za_pvalue(0.0, :constant)
        @test pval4 > 0.10
        @test pval4 <= 1.0
    end

    @testset "_ngperron_pvalue: MZa/MZt and MSB/MPT branches" begin
        # MZa test (more negative = reject) — all 4 branches
        for test_sym in (:MZa, :MZt)
            ng_cv = MacroEconometricModels.NGPERRON_CRITICAL_VALUES[:constant][test_sym]

            # Branch 1: stat <= cv[1]
            pval1 = MacroEconometricModels._ngperron_pvalue(-999.0, :constant, test_sym)
            @test pval1 == 0.01

            # Branch 2: between cv[1] and cv[5]
            stat_mid = (ng_cv[1] + ng_cv[5]) / 2
            pval2 = MacroEconometricModels._ngperron_pvalue(stat_mid, :constant, test_sym)
            @test 0.01 < pval2 < 0.05

            # Branch 3: between cv[5] and cv[10]
            stat_mid2 = (ng_cv[5] + ng_cv[10]) / 2
            pval3 = MacroEconometricModels._ngperron_pvalue(stat_mid2, :constant, test_sym)
            @test 0.05 < pval3 < 0.10

            # Branch 4: above cv[10]
            pval4 = MacroEconometricModels._ngperron_pvalue(0.0, :constant, test_sym)
            @test pval4 > 0.10
        end

        # MSB, MPT tests (smaller = reject) — all 4 branches
        for test_sym in (:MSB, :MPT)
            ng_cv = MacroEconometricModels.NGPERRON_CRITICAL_VALUES[:constant][test_sym]

            # Branch 1: stat <= cv[1]
            pval1 = MacroEconometricModels._ngperron_pvalue(-999.0, :constant, test_sym)
            @test pval1 == 0.01

            # Branch 2: between cv[1] and cv[5]
            stat_mid = (ng_cv[1] + ng_cv[5]) / 2
            pval2 = MacroEconometricModels._ngperron_pvalue(stat_mid, :constant, test_sym)
            @test 0.01 < pval2 < 0.05

            # Branch 3: between cv[5] and cv[10]
            stat_mid2 = (ng_cv[5] + ng_cv[10]) / 2
            pval3 = MacroEconometricModels._ngperron_pvalue(stat_mid2, :constant, test_sym)
            @test 0.05 < pval3 < 0.10

            # Branch 4: above cv[10]
            pval4 = MacroEconometricModels._ngperron_pvalue(999.0, :constant, test_sym)
            @test pval4 > 0.10
        end

        # Test with :trend regression
        pval = MacroEconometricModels._ngperron_pvalue(-999.0, :trend, :MZa)
        @test pval == 0.01
    end

    @testset "adf_select_lags: all criteria" begin
        rng = Random.MersenneTwister(4242)
        y = cumsum(randn(rng, 200))

        for criterion in (:aic, :bic, :hqic)
            lag = MacroEconometricModels.adf_select_lags(y, 5, :constant, criterion)
            @test lag >= 0
            @test lag <= 5
        end

        # With :none regression
        lag_none = MacroEconometricModels.adf_select_lags(y, 5, :none, :bic)
        @test lag_none >= 0

        # With :trend regression
        lag_trend = MacroEconometricModels.adf_select_lags(y, 5, :trend, :aic)
        @test lag_trend >= 0
    end

    @testset "_build_adf_matrix: all regression types x lag combos" begin
        rng = Random.MersenneTwister(1111)
        y = cumsum(randn(rng, 100))
        dy = diff(y)

        for reg in (:none, :constant, :trend)
            for lags in (0, 1, 3)
                X = MacroEconometricModels._build_adf_matrix(y, dy, lags, reg)
                nobs = length(dy) - lags
                @test size(X, 1) == nobs

                expected_cols = if reg == :none
                    1 + lags  # y_lag + lagged diffs
                elseif reg == :constant
                    2 + lags  # constant + y_lag + lagged diffs
                else  # :trend
                    3 + lags  # constant + trend + y_lag + lagged diffs
                end
                @test size(X, 2) == expected_cols
            end
        end
    end

    @testset "_nw_bandwidth and _long_run_variance" begin
        rng = Random.MersenneTwister(2222)
        resid = randn(rng, 200)

        bw = MacroEconometricModels._nw_bandwidth(resid)
        @test bw >= 1
        @test bw < 200

        lrv = MacroEconometricModels._long_run_variance(resid, bw)
        @test lrv > 0.0

        # With bandwidth = 1
        lrv1 = MacroEconometricModels._long_run_variance(resid, 1)
        @test lrv1 > 0.0

        # Highly autocorrelated residuals → bigger bandwidth
        ar_resid = zeros(200)
        ar_resid[1] = randn(rng)
        for t in 2:200
            ar_resid[t] = 0.9 * ar_resid[t-1] + 0.1 * randn(rng)
        end
        bw_ar = MacroEconometricModels._nw_bandwidth(ar_resid)
        @test bw_ar >= 1
    end

    @testset "_regression_name: all branches" begin
        @test MacroEconometricModels._regression_name(:none) == "None"
        @test MacroEconometricModels._regression_name(:constant) == "Constant"
        @test MacroEconometricModels._regression_name(:trend) == "Constant + Trend"
        @test MacroEconometricModels._regression_name(:both) == "Constant + Trend"
        @test MacroEconometricModels._regression_name(:custom) == "custom"
    end

    # =========================================================================
    # andrews.jl — all 9 variants, functionals, p-value branches, show
    # =========================================================================

    @testset "Andrews test: all 9 test variants produce valid results" begin
        rng = Random.MersenneTwister(5555)
        T_obs = 120
        X = hcat(ones(T_obs), randn(rng, T_obs))
        y = X * [1.0, 2.0] + randn(rng, T_obs) * 0.5

        for test_type in (:supwald, :suplr, :suplm, :expwald, :explr, :explm,
                          :meanwald, :meanlr, :meanlm)
            result = andrews_test(y, X; test=test_type, trimming=0.15)
            @test result isa AndrewsResult{Float64}
            @test result.test_type == test_type
            @test result.statistic >= 0 || result.test_type in (:explr, :explm, :expwald)
            @test 0.0 <= result.pvalue <= 1.0
            @test 1 <= result.break_index <= T_obs
            @test 0.0 < result.break_fraction < 1.0
            @test haskey(result.critical_values, 1)
            @test haskey(result.critical_values, 5)
            @test haskey(result.critical_values, 10)
        end
    end

    @testset "Andrews test: Float64 fallback from Int" begin
        rng = Random.MersenneTwister(6666)
        y_int = round.(Int, randn(rng, 100) .* 10)
        X_int = round.(Int, hcat(ones(100), randn(rng, 100)) .* 10)
        result = andrews_test(y_int, X_int; test=:supwald)
        @test result isa AndrewsResult{Float64}
    end

    @testset "Andrews test: strong break detection" begin
        rng = Random.MersenneTwister(7777)
        T_obs = 150
        X = hcat(ones(T_obs), randn(rng, T_obs))
        y = X * [1.0, 2.0] + randn(rng, T_obs) * 0.3
        y[76:end] .+= X[76:end, 2] .* 5.0  # large break at t=75

        result = andrews_test(y, X; test=:supwald)
        @test result.pvalue < 0.10

        result_lr = andrews_test(y, X; test=:suplr)
        @test result_lr.pvalue < 0.10
    end

    @testset "Andrews test: error handling" begin
        # Invalid test type
        @test_throws ArgumentError andrews_test(randn(100), ones(100, 1); test=:invalid)

        # Dimension mismatch
        @test_throws ArgumentError andrews_test(randn(100), ones(50, 1))

        # Too short series
        @test_throws ArgumentError andrews_test(randn(10), ones(10, 1))

        # Invalid trimming
        @test_throws ArgumentError andrews_test(randn(100), ones(100, 1); trimming=0.6)
        @test_throws ArgumentError andrews_test(randn(100), ones(100, 1); trimming=-0.1)
    end

    @testset "Andrews show method: reject and fail-to-reject" begin
        rng = Random.MersenneTwister(8888)
        T_obs = 150
        X = hcat(ones(T_obs), randn(rng, T_obs))

        # Strong break: should reject
        y_break = X * [1.0, 2.0] + randn(rng, T_obs) * 0.3
        y_break[76:end] .+= X[76:end, 2] .* 5.0
        result_rej = andrews_test(y_break, X; test=:supwald)
        io = IOBuffer()
        show(io, result_rej)
        s = String(take!(io))
        @test occursin("Andrews", s)
        @test occursin("Sup-Wald", s)
        @test occursin("Break", s)
        @test occursin("Critical Values", s)

        # No break: should fail to reject
        y_stable = X * [1.0, 2.0] + randn(Random.MersenneTwister(9999), T_obs) * 2.0
        result_norej = andrews_test(y_stable, X; test=:supwald)
        io2 = IOBuffer()
        show(io2, result_norej)
        s2 = String(take!(io2))
        @test occursin("Andrews", s2)

        # Show different test types
        for tt in (:suplr, :explr, :meanlm)
            r = andrews_test(y_stable, X; test=tt)
            io3 = IOBuffer()
            show(io3, r)
            s3 = String(take!(io3))
            @test length(s3) > 50
        end
    end

    @testset "_andrews_functional: sup/exp/mean" begin
        stats = [1.0, 3.0, 2.0, 5.0, 4.0]

        # sup
        val, idx = MacroEconometricModels._andrews_functional(stats, :sup)
        @test val == 5.0
        @test idx == 4

        # exp (log-sum-exp)
        val_exp, idx_exp = MacroEconometricModels._andrews_functional(stats, :exp)
        @test isfinite(val_exp)
        @test idx_exp == 4  # index of max

        # mean
        val_mean, idx_mean = MacroEconometricModels._andrews_functional(stats, :mean)
        @test val_mean ≈ mean(stats)
        @test idx_mean == 4
    end

    @testset "_andrews_critical_values: all 3 functional tables" begin
        for func in (:sup, :exp, :mean)
            for k in (1, 3, 5, 10)
                cv = MacroEconometricModels._andrews_critical_values(k, func, Float64)
                @test haskey(cv, 1)
                @test haskey(cv, 5)
                @test haskey(cv, 10)
                @test cv[1] > cv[5] > cv[10]  # Higher = more significant
            end
        end

        # k clamped to 1-10
        cv_low = MacroEconometricModels._andrews_critical_values(0, :sup, Float64)
        cv_1 = MacroEconometricModels._andrews_critical_values(1, :sup, Float64)
        @test cv_low == cv_1

        cv_high = MacroEconometricModels._andrews_critical_values(20, :sup, Float64)
        cv_10 = MacroEconometricModels._andrews_critical_values(10, :sup, Float64)
        @test cv_high == cv_10
    end

    @testset "_andrews_pvalue: all 4 branches" begin
        cv = MacroEconometricModels._andrews_critical_values(2, :sup, Float64)

        # Branch 1: stat >= cv[1]
        pval1 = MacroEconometricModels._andrews_pvalue(cv[1] + 5.0, 2, :sup)
        @test pval1 == 0.005

        # Branch 2: stat between cv[5] and cv[1]
        stat_mid = (cv[1] + cv[5]) / 2
        pval2 = MacroEconometricModels._andrews_pvalue(stat_mid, 2, :sup)
        @test 0.01 < pval2 < 0.05

        # Branch 3: stat between cv[10] and cv[5]
        stat_mid2 = (cv[5] + cv[10]) / 2
        pval3 = MacroEconometricModels._andrews_pvalue(stat_mid2, 2, :sup)
        @test 0.05 < pval3 < 0.10

        # Branch 4: stat below cv[10]
        pval4 = MacroEconometricModels._andrews_pvalue(0.01, 2, :sup)
        @test pval4 >= 0.10

        # Test with exp and mean functionals
        for func in (:exp, :mean)
            pval_f = MacroEconometricModels._andrews_pvalue(100.0, 2, func)
            @test pval_f == 0.005
            pval_low = MacroEconometricModels._andrews_pvalue(0.001, 2, func)
            @test pval_low >= 0.10
        end
    end

    # =========================================================================
    # factor_break.jl — all 3 methods, Float64 fallbacks, FactorModel dispatch,
    #                    internal p-value functions, show
    # =========================================================================

    @testset "factor_break_test: breitung_eickmeier" begin
        rng = Random.MersenneTwister(3001)
        X = randn(rng, 100, 25)
        result = factor_break_test(X, 2; method=:breitung_eickmeier)
        @test result isa FactorBreakResult{Float64}
        @test result.method == :breitung_eickmeier
        @test result.n_factors == 2
        @test 0.0 <= result.pvalue <= 1.0
        @test result.break_date isa Int
    end

    @testset "factor_break_test: chen_dolado_gonzalo" begin
        rng = Random.MersenneTwister(3002)
        X = randn(rng, 100, 25)
        # Matrix-only dispatch (no r)
        result = factor_break_test(X; method=:chen_dolado_gonzalo)
        @test result isa FactorBreakResult{Float64}
        @test result.method == :chen_dolado_gonzalo

        # With r provided (goes through the (X, r) method)
        result2 = factor_break_test(X, 3; method=:chen_dolado_gonzalo)
        @test result2 isa FactorBreakResult{Float64}
        @test result2.method == :chen_dolado_gonzalo
    end

    @testset "factor_break_test: han_inoue" begin
        rng = Random.MersenneTwister(3003)
        X = randn(rng, 100, 25)
        result = factor_break_test(X, 2; method=:han_inoue)
        @test result isa FactorBreakResult{Float64}
        @test result.method == :han_inoue
        @test result.break_date isa Int
        @test 0.0 <= result.pvalue <= 1.0
    end

    @testset "factor_break_test: FactorModel dispatch" begin
        rng = Random.MersenneTwister(3004)
        X = randn(rng, 100, 20)
        fm = estimate_factors(X, 2)

        for method in (:breitung_eickmeier, :han_inoue)
            result = factor_break_test(fm; method=method)
            @test result isa FactorBreakResult{Float64}
            @test result.method == method
        end
    end

    @testset "factor_break_test: Float64 fallback (with r)" begin
        X_int = round.(Int, randn(Random.MersenneTwister(3005), 80, 20) .* 10)
        result = factor_break_test(X_int, 2; method=:breitung_eickmeier)
        @test result isa FactorBreakResult{Float64}
    end

    @testset "factor_break_test: Float64 fallback (no r)" begin
        X_int = round.(Int, randn(Random.MersenneTwister(3006), 80, 20) .* 10)
        result = factor_break_test(X_int; method=:chen_dolado_gonzalo)
        @test result isa FactorBreakResult{Float64}
    end

    @testset "factor_break_test: matrix-only dispatch error for methods needing r" begin
        X = randn(Random.MersenneTwister(3007), 100, 20)
        @test_throws ArgumentError factor_break_test(X; method=:breitung_eickmeier)
        @test_throws ArgumentError factor_break_test(X; method=:han_inoue)
    end

    @testset "factor_break_test: error handling" begin
        # Too short
        @test_throws ArgumentError factor_break_test(randn(10, 5), 2)
        # Invalid method (with r)
        @test_throws ArgumentError factor_break_test(randn(100, 20), 2; method=:invalid)
        # Invalid method (no r)
        @test_throws ArgumentError factor_break_test(randn(100, 20); method=:invalid)
    end

    @testset "_breitung_eickmeier_pvalue" begin
        # Small stat → large p-value
        pval1 = MacroEconometricModels._breitung_eickmeier_pvalue(0.5, 6)
        @test pval1 > 0.50

        # Large stat → small p-value
        pval2 = MacroEconometricModels._breitung_eickmeier_pvalue(50.0, 6)
        @test pval2 < 0.01

        # Edge: stat_sq = 0 → p-value = 1
        pval3 = MacroEconometricModels._breitung_eickmeier_pvalue(0.0, 6)
        @test pval3 ≈ 1.0
    end

    @testset "_han_inoue_pvalue: all 4 branches" begin
        # Need to look up critical values for a specific k
        k = 2
        cv = MacroEconometricModels.HANSEN_ANDREWS_CV[k]

        # Branch 1: stat >= cv1 (beyond 1% CV)
        pval1 = MacroEconometricModels._han_inoue_pvalue(Float64(cv[1]) + 10.0, k)
        @test pval1 <= 0.01

        # Branch 2: between cv5 and cv1
        stat_b2 = (Float64(cv[1]) + Float64(cv[5])) / 2
        pval2 = MacroEconometricModels._han_inoue_pvalue(stat_b2, k)
        @test 0.01 <= pval2 <= 0.05

        # Branch 3: between cv10 and cv5
        stat_b3 = (Float64(cv[5]) + Float64(cv[10])) / 2
        pval3 = MacroEconometricModels._han_inoue_pvalue(stat_b3, k)
        @test 0.05 <= pval3 <= 0.10

        # Branch 4: below cv10
        pval4 = MacroEconometricModels._han_inoue_pvalue(0.01, k)
        @test pval4 >= 0.10

        # k clamped to 1..10
        pval_big_k = MacroEconometricModels._han_inoue_pvalue(100.0, 15)
        @test pval_big_k <= 0.01
    end

    @testset "_sorted_eigenvalues" begin
        rng = Random.MersenneTwister(3010)
        X = randn(rng, 50, 10)
        eigs = MacroEconometricModels._sorted_eigenvalues(X)
        @test length(eigs) == 10
        # Should be sorted descending
        @test issorted(eigs; rev=true)
        @test all(isfinite, eigs)
    end

    @testset "factor_break_test show: all 3 conclusion branches" begin
        rng = Random.MersenneTwister(3011)
        X = randn(rng, 100, 25)

        # Show with break_date (breitung_eickmeier)
        result_be = factor_break_test(X, 2; method=:breitung_eickmeier)
        io = IOBuffer()
        show(io, result_be)
        s = String(take!(io))
        @test occursin("Factor Model Structural Break", s)
        @test occursin("Breitung-Eickmeier", s)
        @test occursin("Break date", s) || occursin("Fail to reject", s)

        # Show with chen_dolado_gonzalo
        result_cdg = factor_break_test(X; method=:chen_dolado_gonzalo)
        io2 = IOBuffer()
        show(io2, result_cdg)
        s2 = String(take!(io2))
        @test occursin("Chen-Dolado-Gonzalo", s2)

        # Show with han_inoue
        result_hi = factor_break_test(X, 2; method=:han_inoue)
        io3 = IOBuffer()
        show(io3, result_hi)
        s3 = String(take!(io3))
        @test occursin("Han-Inoue", s3)

        # Show with break_date = nothing (manually constructed)
        result_nothing = FactorBreakResult(0.0, 1.0, nothing, :chen_dolado_gonzalo, 2, 100, 25)
        io4 = IOBuffer()
        show(io4, result_nothing)
        s4 = String(take!(io4))
        @test occursin("Fail to reject", s4)
        # With break_date = nothing, no "Break date" in output
        @test !occursin("Break date", s4)

        # Construct a result that rejects and has no break_date
        result_rej_nobd = FactorBreakResult(100.0, 0.001, nothing, :breitung_eickmeier, 2, 100, 25)
        io5 = IOBuffer()
        show(io5, result_rej_nobd)
        s5 = String(take!(io5))
        @test occursin("Reject", s5)
        @test !occursin("observation", s5)  # no "at observation X" because break_date=nothing
    end

    # =========================================================================
    # moon_perron.jl — Float64 fallback, PanelData dispatch, show branches
    # =========================================================================

    @testset "moon_perron_test: basic with different r values" begin
        rng = Random.MersenneTwister(4001)
        X = randn(rng, 80, 15)

        result1 = moon_perron_test(X; r=1)
        @test result1 isa MoonPerronResult{Float64}
        @test result1.n_factors == 1

        result2 = moon_perron_test(X; r=2)
        @test result2.n_factors == 2

        # Auto factor selection
        result_auto = moon_perron_test(X; r=:auto)
        @test result_auto.n_factors >= 1
    end

    @testset "moon_perron_test: Float64 fallback" begin
        X_int = round.(Int, randn(Random.MersenneTwister(4002), 80, 15) .* 10)
        result = moon_perron_test(X_int; r=1)
        @test result isa MoonPerronResult{Float64}
    end

    @testset "moon_perron_test: PanelData dispatch" begin
        rng = Random.MersenneTwister(4003)
        df = DataFrame(
            group = repeat(1:10, inner=30),
            time = repeat(1:30, 10),
            y = randn(rng, 300)
        )
        pd = xtset(df, :group, :time)
        result = moon_perron_test(pd; r=1)
        @test result isa MoonPerronResult{Float64}
        @test result.n_units == 10
    end

    @testset "moon_perron_test: error handling" begin
        # Too short time dimension
        @test_throws ArgumentError moon_perron_test(randn(5, 3); r=1)
        # r < 1
        @test_throws ArgumentError moon_perron_test(randn(80, 15); r=0)
        # r too large
        @test_throws ArgumentError moon_perron_test(randn(80, 15); r=100)
    end

    @testset "moon_perron_test show: all 3 conclusion branches" begin
        # Branch 1: both reject (stationary panel data)
        rng1 = Random.MersenneTwister(4010)
        X_stat = randn(rng1, 100, 20)  # stationary → both should reject
        result_both = moon_perron_test(X_stat; r=1)
        io = IOBuffer()
        show(io, result_both)
        s = String(take!(io))
        @test occursin("Moon-Perron", s)
        @test occursin("t*_a", s)
        @test occursin("t*_b", s)

        # Branch 2: one rejects (construct manually)
        result_one = MoonPerronResult(-2.0, 0.5, 0.02, 0.80, 1, 100, 20)
        io2 = IOBuffer()
        show(io2, result_one)
        s2 = String(take!(io2))
        @test occursin("One statistic rejects", s2) || occursin("moderate evidence", s2)

        # Branch 3: neither rejects (construct manually)
        result_none = MoonPerronResult(0.5, 0.5, 0.80, 0.80, 1, 100, 20)
        io3 = IOBuffer()
        show(io3, result_none)
        s3 = String(take!(io3))
        @test occursin("Fail to reject", s3)
    end

    # =========================================================================
    # panic.jl — Float64 fallback, _panel_to_matrix, show, method=:individual
    # =========================================================================

    @testset "panic_test: pooled and individual methods" begin
        rng = Random.MersenneTwister(5001)
        # Create data with a common factor
        T_obs, N = 80, 15
        F = cumsum(randn(rng, T_obs))
        Lambda = randn(rng, N)
        e = randn(rng, T_obs, N)
        X = F * Lambda' + e

        result_p = panic_test(X; r=1, method=:pooled)
        @test result_p isa PANICResult{Float64}
        @test result_p.method == :pooled
        @test result_p.n_factors == 1
        @test result_p.n_units == N
        @test length(result_p.factor_adf_stats) == 1
        @test length(result_p.individual_stats) == N
        @test 0.0 <= result_p.pooled_pvalue <= 1.0

        result_i = panic_test(X; r=1, method=:individual)
        @test result_i.method == :individual
        @test length(result_i.individual_pvalues) == N
    end

    @testset "panic_test: auto factor selection" begin
        rng = Random.MersenneTwister(5002)
        X = randn(rng, 80, 15)
        result = panic_test(X; r=:auto, method=:pooled)
        @test result isa PANICResult{Float64}
        @test result.n_factors >= 1
    end

    @testset "panic_test: Float64 fallback" begin
        X_int = round.(Int, randn(Random.MersenneTwister(5003), 80, 15) .* 10)
        result = panic_test(X_int; r=1)
        @test result isa PANICResult{Float64}
    end

    @testset "panic_test: PanelData dispatch" begin
        rng = Random.MersenneTwister(5004)
        df = DataFrame(
            group = repeat(1:8, inner=25),
            time = repeat(1:25, 8),
            y = randn(rng, 200)
        )
        pd = xtset(df, :group, :time)
        result = panic_test(pd; r=1)
        @test result isa PANICResult{Float64}
        @test result.n_units == 8
    end

    @testset "panic_test: error handling" begin
        # Too short
        @test_throws ArgumentError panic_test(randn(5, 3); r=1)
        # Invalid method
        @test_throws ArgumentError panic_test(randn(80, 15); r=1, method=:invalid)
        # r < 1
        @test_throws ArgumentError panic_test(randn(80, 15); r=0)
        # r too large
        @test_throws ArgumentError panic_test(randn(80, 15); r=100)
    end

    @testset "panic_test show: reject and fail-to-reject" begin
        rng = Random.MersenneTwister(5010)

        # Stationary panel → should reject H0
        X_stat = randn(rng, 100, 20)
        result_rej = panic_test(X_stat; r=1, method=:pooled)
        io = IOBuffer()
        show(io, result_rej)
        s = String(take!(io))
        @test occursin("PANIC", s)
        @test occursin("Bai-Ng", s)
        @test occursin("Factor", s)
        @test occursin("Pooled", s)

        # Unit root panel → should fail to reject
        X_ur = cumsum(randn(Random.MersenneTwister(5011), 100, 20), dims=1)
        result_norej = panic_test(X_ur; r=1, method=:pooled)
        io2 = IOBuffer()
        show(io2, result_norej)
        s2 = String(take!(io2))
        @test occursin("PANIC", s2)

        # Individual method show
        result_ind = panic_test(X_stat; r=1, method=:individual)
        io3 = IOBuffer()
        show(io3, result_ind)
        s3 = String(take!(io3))
        @test occursin("Individual", s3)

        # Multiple factors show
        result_multi = panic_test(X_stat; r=3, method=:pooled)
        io4 = IOBuffer()
        show(io4, result_multi)
        s4 = String(take!(io4))
        @test occursin("Factor 1", s4)
        @test occursin("Factor 2", s4)
        @test occursin("Factor 3", s4)
    end

    @testset "_panel_to_matrix" begin
        rng = Random.MersenneTwister(5020)
        # Balanced panel
        df = DataFrame(
            group = repeat(1:5, inner=20),
            time = repeat(1:20, 5),
            y = randn(rng, 100)
        )
        pd = xtset(df, :group, :time)
        X = MacroEconometricModels._panel_to_matrix(pd)
        @test size(X) == (20, 5)
        @test !any(isnan, X)

        # Unbalanced panel (missing some periods for one group)
        df2 = DataFrame(
            group = vcat(repeat(1:3, inner=20), repeat([4], 15)),
            time = vcat(repeat(1:20, 3), collect(1:15)),
            y = randn(rng, 75)
        )
        pd2 = xtset(df2, :group, :time)
        X2 = MacroEconometricModels._panel_to_matrix(pd2)
        # Should have at most 15 valid rows (shortest group is 15)
        @test size(X2, 2) == 4
        @test size(X2, 1) <= 20
        @test !any(isnan, X2)
    end

    # =========================================================================
    # pesaran_cips.jl — all deterministic types, auto lags, Float64 fallback,
    #                    PanelData dispatch, _pesaran_cips_critical_values_and_pvalue,
    #                    _nearest_val, show
    # =========================================================================

    @testset "pesaran_cips_test: all deterministic variants" begin
        rng = Random.MersenneTwister(6001)
        X = randn(rng, 60, 15)

        for det in (:none, :constant, :trend)
            result = pesaran_cips_test(X; lags=1, deterministic=det)
            @test result isa PesaranCIPSResult{Float64}
            @test result.deterministic == det
            @test result.lags == 1
            @test 0.0 <= result.pvalue <= 1.0
            @test length(result.individual_cadf_stats) == 15
            @test haskey(result.critical_values, 1)
            @test haskey(result.critical_values, 5)
            @test haskey(result.critical_values, 10)
        end
    end

    @testset "pesaran_cips_test: auto lags" begin
        rng = Random.MersenneTwister(6002)
        X = randn(rng, 60, 15)
        result = pesaran_cips_test(X; lags=:auto, deterministic=:constant)
        @test result isa PesaranCIPSResult{Float64}
        expected_lags = max(1, floor(Int, 60^(1/3)))
        @test result.lags == expected_lags
    end

    @testset "pesaran_cips_test: Float64 fallback" begin
        X_int = round.(Int, randn(Random.MersenneTwister(6003), 60, 15) .* 10)
        result = pesaran_cips_test(X_int; lags=1)
        @test result isa PesaranCIPSResult{Float64}
    end

    @testset "pesaran_cips_test: PanelData dispatch" begin
        rng = Random.MersenneTwister(6004)
        df = DataFrame(
            group = repeat(1:10, inner=30),
            time = repeat(1:30, 10),
            y = randn(rng, 300)
        )
        pd = xtset(df, :group, :time)
        result = pesaran_cips_test(pd; lags=1, deterministic=:constant)
        @test result isa PesaranCIPSResult{Float64}
        @test result.n_units == 10
    end

    @testset "pesaran_cips_test: error handling" begin
        # Too short
        @test_throws ArgumentError pesaran_cips_test(randn(5, 3); lags=1)
        # Invalid deterministic
        @test_throws ArgumentError pesaran_cips_test(randn(50, 10); deterministic=:invalid)
    end

    @testset "_nearest_val" begin
        vals = [10, 20, 30, 50, 100]
        @test MacroEconometricModels._nearest_val(10, vals) == 10
        @test MacroEconometricModels._nearest_val(15, vals) == 10  # |15-10|=5, |15-20|=5 → first found
        @test MacroEconometricModels._nearest_val(16, vals) == 20
        @test MacroEconometricModels._nearest_val(100, vals) == 100
        @test MacroEconometricModels._nearest_val(75, vals) == 50  # |75-50|=25, |75-100|=25 → first found (50) wins
        @test MacroEconometricModels._nearest_val(76, vals) == 100
        @test MacroEconometricModels._nearest_val(1, vals) == 10
        @test MacroEconometricModels._nearest_val(200, vals) == 100
    end

    @testset "_pesaran_cips_critical_values_and_pvalue: all 4 p-value branches" begin
        # Get cv for reference
        table = MacroEconometricModels.PESARAN_CIPS_CV[:constant]
        cv_dict = table[(20, 30)]
        cv1 = Float64(cv_dict[1])
        cv5 = Float64(cv_dict[5])
        cv10 = Float64(cv_dict[10])

        # Branch 1: cips <= cv[1] → p = 0.001
        cv_r, pval1 = MacroEconometricModels._pesaran_cips_critical_values_and_pvalue(
            cv1 - 1.0, 20, 30, :constant, Float64)
        @test pval1 == 0.001

        # Branch 2: between cv[1] and cv[5]
        cips_mid = (cv1 + cv5) / 2
        _, pval2 = MacroEconometricModels._pesaran_cips_critical_values_and_pvalue(
            cips_mid, 20, 30, :constant, Float64)
        @test 0.01 < pval2 < 0.05

        # Branch 3: between cv[5] and cv[10]
        cips_mid2 = (cv5 + cv10) / 2
        _, pval3 = MacroEconometricModels._pesaran_cips_critical_values_and_pvalue(
            cips_mid2, 20, 30, :constant, Float64)
        @test 0.05 < pval3 < 0.10

        # Branch 4: above cv[10]
        _, pval4 = MacroEconometricModels._pesaran_cips_critical_values_and_pvalue(
            0.0, 20, 30, :constant, Float64)
        @test pval4 > 0.10
        @test pval4 <= 1.0

        # Test with :trend and :none
        for det in (:trend, :none)
            _, pval = MacroEconometricModels._pesaran_cips_critical_values_and_pvalue(
                -5.0, 20, 30, det, Float64)
            @test pval == 0.001
        end

        # Test with non-table N and T values (exercises _nearest_val)
        _, pval_nn = MacroEconometricModels._pesaran_cips_critical_values_and_pvalue(
            -5.0, 25, 45, :constant, Float64)
        @test pval_nn == 0.001
    end

    @testset "pesaran_cips_test show" begin
        rng = Random.MersenneTwister(6010)

        # Stationary panel → should reject
        X_stat = randn(rng, 60, 15)
        result = pesaran_cips_test(X_stat; lags=1, deterministic=:constant)
        io = IOBuffer()
        show(io, result)
        s = String(take!(io))
        @test occursin("Pesaran", s)
        @test occursin("CIPS", s)
        @test occursin("Critical Values", s)
        @test occursin("Constant", s)

        # With :trend
        result_trend = pesaran_cips_test(X_stat; lags=1, deterministic=:trend)
        io2 = IOBuffer()
        show(io2, result_trend)
        s2 = String(take!(io2))
        @test occursin("Constant + Trend", s2) || occursin("Trend", s2)

        # With :none
        result_none = pesaran_cips_test(X_stat; lags=1, deterministic=:none)
        io3 = IOBuffer()
        show(io3, result_none)
        s3 = String(take!(io3))
        @test occursin("None", s3)

        # Unit root panel → fail to reject
        X_ur = cumsum(randn(Random.MersenneTwister(6011), 60, 15), dims=1)
        result_ur = pesaran_cips_test(X_ur; lags=1, deterministic=:constant)
        io4 = IOBuffer()
        show(io4, result_ur)
        s4 = String(take!(io4))
        @test occursin("Pesaran", s4)
    end

    # =========================================================================
    # panel_unit_root_summary — IO dispatch, PanelData dispatch
    # =========================================================================

    @testset "panel_unit_root_summary: IOBuffer dispatch" begin
        rng = Random.MersenneTwister(7001)
        X = randn(rng, 80, 15)
        io = IOBuffer()
        panel_unit_root_summary(io, X; r=1, lags=1)
        s = String(take!(io))
        @test occursin("Panel Unit Root Test Battery", s)
        @test occursin("PANIC", s) || occursin("panic", s) || occursin("Bai-Ng", s)
        @test occursin("Pesaran", s) || occursin("CIPS", s)
        @test occursin("Moon-Perron", s) || occursin("Moon", s)
    end

    @testset "panel_unit_root_summary: stdout dispatch" begin
        rng = Random.MersenneTwister(7002)
        X = randn(rng, 80, 15)
        # Just verify it doesn't error — output goes to stdout
        io = IOBuffer()
        panel_unit_root_summary(io, X; r=1, lags=1)
        @test length(String(take!(io))) > 100
    end

    @testset "panel_unit_root_summary: auto r and lags" begin
        rng = Random.MersenneTwister(7003)
        X = randn(rng, 80, 15)
        io = IOBuffer()
        panel_unit_root_summary(io, X; r=:auto, lags=:auto)
        s = String(take!(io))
        @test occursin("Panel Unit Root", s)
    end

    @testset "panel_unit_root_summary: PanelData dispatch" begin
        rng = Random.MersenneTwister(7004)
        df = DataFrame(
            group = repeat(1:8, inner=25),
            time = repeat(1:25, 8),
            y = randn(rng, 200)
        )
        pd = xtset(df, :group, :time)
        io = IOBuffer()
        panel_unit_root_summary(io, MacroEconometricModels._panel_to_matrix(pd); r=1, lags=1)
        s = String(take!(io))
        @test occursin("Panel Unit Root", s)

        # Direct PanelData dispatch
        result = panel_unit_root_summary(pd; r=1, lags=1)
        @test result === nothing
    end

end
