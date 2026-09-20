# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test, MacroEconometricModels, Random, StatsAPI

@testset "Fourier Unit Root Tests" begin
    rng = Random.Xoshiro(44556)

    # Stationary AR(1) with moderate persistence
    y_stat = zeros(200)
    y_stat[1] = randn(rng)
    for t in 2:200; y_stat[t] = 0.5 * y_stat[t-1] + randn(rng); end

    # Random walk (unit root)
    y_rw = cumsum(randn(rng, 200))

    # Stationary with smooth sinusoidal component
    y_smooth = zeros(200)
    for t in 1:200; y_smooth[t] = 2.0 * sin(2pi * t / 200) + randn(rng); end

    @testset "Fourier ADF Test" begin
        result = fourier_adf_test(y_stat; regression=:constant)
        @test result isa FourierADFResult
        @test result.frequency >= 1
        @test result.lags >= 0
        @test result.nobs > 0
        @test haskey(result.critical_values, 5)
        @test haskey(result.critical_values, 1)
        @test haskey(result.critical_values, 10)
        # T159: DF-type tau is negative on stationary data; the Fourier-term F is an
        # SSR ratio ≥ 0; both p-values are probabilities.
        @test isfinite(result.statistic) && result.statistic < 0
        @test isfinite(result.f_statistic) && result.f_statistic >= 0  # T159: (see note above)
        @test isfinite(result.pvalue) && 0 <= result.pvalue <= 1  # T159: (see note above)
        @test isfinite(result.f_pvalue) && 0 <= result.f_pvalue <= 1  # T159: (see note above)
        @test result.regression == :constant

        result_t = fourier_adf_test(y_rw; regression=:trend)
        @test result_t isa FourierADFResult
        @test result_t.regression == :trend

        # T159: designed comparisons (same :constant spec) — stationary tau is more
        # negative (−3.77 vs −1.86); the sinusoidal DGP yields a far more
        # significant Fourier F (p-values are cross-spec comparable: 0.001 vs 0.2).
        result_rw_c = fourier_adf_test(y_rw; regression=:constant)
        @test result.statistic < result_rw_c.statistic
        result_smooth = fourier_adf_test(y_smooth)
        @test result_smooth.f_pvalue < result.f_pvalue

        result_f2 = fourier_adf_test(y_smooth; fmax=2)
        @test result_f2.frequency <= 2

        # Fixed lags
        result_fixed = fourier_adf_test(y_stat; lags=2)
        @test result_fixed.lags == 2

        # BIC selection
        result_bic = fourier_adf_test(y_stat; lags=:bic)
        @test result_bic.lags >= 0

        # Integer input fallback
        result_int = fourier_adf_test(round.(Int, y_stat * 10))
        @test result_int isa FourierADFResult

        # Too short
        @test_throws ArgumentError fourier_adf_test(randn(rng, 30))

        # Invalid regression
        @test_throws ArgumentError fourier_adf_test(y_stat; regression=:none)

        # Invalid fmax
        @test_throws ArgumentError fourier_adf_test(y_stat; fmax=0)
        @test_throws ArgumentError fourier_adf_test(y_stat; fmax=6)

        # StatsAPI interface
        @test StatsAPI.nobs(result) == result.nobs
        @test StatsAPI.pvalue(result) == result.pvalue
    end

    @testset "Fourier KPSS Test" begin
        result = fourier_kpss_test(y_stat; regression=:constant)
        @test result isa FourierKPSSResult
        @test result.frequency >= 1
        @test result.bandwidth > 0
        # T159: KPSS LM and the Fourier F are non-negative by construction; p-values
        # are probabilities. RW inflates the LM two orders of magnitude (0.149 vs 2.88).
        @test isfinite(result.statistic) && result.statistic >= 0
        @test isfinite(result.f_statistic) && result.f_statistic >= 0  # T159: (see note above)
        @test isfinite(result.pvalue) && 0 <= result.pvalue <= 1  # T159: (see note above)
        @test isfinite(result.f_pvalue) && 0 <= result.f_pvalue <= 1  # T159: (see note above)
        @test result.regression == :constant
        result_rw_c = fourier_kpss_test(y_rw; regression=:constant)
        @test result.statistic < result_rw_c.statistic
        @test haskey(result.critical_values, 1)
        @test haskey(result.critical_values, 5)
        @test haskey(result.critical_values, 10)

        result_t = fourier_kpss_test(y_rw; regression=:trend)
        @test result_t isa FourierKPSSResult
        @test result_t.regression == :trend

        # Custom bandwidth
        result_bw = fourier_kpss_test(y_stat; bandwidth=5)
        @test result_bw.bandwidth == 5

        # Smooth series
        result_smooth = fourier_kpss_test(y_smooth)
        @test result_smooth isa FourierKPSSResult

        # Integer input fallback
        result_int = fourier_kpss_test(round.(Int, y_stat * 10))
        @test result_int isa FourierKPSSResult

        # Too short
        @test_throws ArgumentError fourier_kpss_test(randn(rng, 30))

        # Invalid regression
        @test_throws ArgumentError fourier_kpss_test(y_stat; regression=:none)

        # StatsAPI interface
        @test StatsAPI.nobs(result) == result.nobs
        @test StatsAPI.pvalue(result) == result.pvalue
    end
end
