# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
using StatsAPI
using Random

@testset "Structural Break & Panel Unit Root Types" begin
    @testset "AndrewsResult construction" begin
        r = AndrewsResult(3.45, 0.02, 50, 0.5, :supwald,
            Dict(1 => 12.0, 5 => 8.5, 10 => 6.8),
            fill(2.0, 80), 0.15, 100, 3)
        @test r isa AndrewsResult{Float64}
        @test r.statistic == 3.45
        @test r.pvalue == 0.02
        @test r.break_index == 50
        @test r.break_fraction == 0.5
        @test r.test_type == :supwald
        @test r.trimming == 0.15
        @test r.nobs == 100
        @test r.n_params == 3
        @test length(r.stat_sequence) == 80
        @test length(r.critical_values) == 3
        @test StatsAPI.nobs(r) == 100
        @test StatsAPI.pvalue(r) == 0.02
        @test StatsAPI.dof(r) == 3
    end

    @testset "BaiPerronResult construction" begin
        r = BaiPerronResult(2, [30, 70], [(25, 35), (65, 75)],
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            [10.0, 8.0], [0.01, 0.03],
            [9.0], [0.02],
            [100.0, 90.0, 85.0], [100.0, 92.0, 88.0],
            0.15, 100)
        @test r isa BaiPerronResult{Float64}
        @test r.n_breaks == 2
        @test length(r.break_dates) == 2
        @test r.break_dates == [30, 70]
        @test length(r.break_cis) == 2
        @test r.break_cis[1] == (25, 35)
        @test length(r.regime_coefs) == 3
        @test length(r.supf_stats) == 2
        @test length(r.sequential_stats) == 1
        @test length(r.bic_values) == 3
        @test length(r.lwz_values) == 3
        @test r.trimming == 0.15
        @test StatsAPI.nobs(r) == 100
        @test StatsAPI.pvalue(r) == 0.01
        @test StatsAPI.dof(r) == 2

        # Edge case: empty supf_pvalues
        r2 = BaiPerronResult(0, Int[], Tuple{Int,Int}[],
            [Float64[]], [Float64[]], Float64[], Float64[],
            Float64[], Float64[], [100.0], [100.0],
            0.15, 100)
        @test StatsAPI.pvalue(r2) == 1.0
    end

    @testset "PANICResult construction" begin
        r = PANICResult([-3.5], [0.01], 2.1, 0.03,
            fill(-2.0, 10), fill(0.1, 10),
            1, :pooled, 100, 10)
        @test r isa PANICResult{Float64}
        @test r.n_factors == 1
        @test r.method == :pooled
        @test r.n_units == 10
        @test length(r.factor_adf_stats) == 1
        @test length(r.individual_stats) == 10
        @test r.pooled_statistic == 2.1
        @test StatsAPI.nobs(r) == 100
        @test StatsAPI.pvalue(r) == 0.03
        @test StatsAPI.dof(r) == 1
    end

    @testset "PesaranCIPSResult construction" begin
        r = PesaranCIPSResult(-2.5, 0.04, fill(-1.8, 5),
            Dict(1 => -2.88, 5 => -2.32, 10 => -2.07),
            1, :constant, 50, 5)
        @test r isa PesaranCIPSResult{Float64}
        @test r.cips_statistic == -2.5
        @test r.lags == 1
        @test r.deterministic == :constant
        @test r.n_units == 5
        @test length(r.individual_cadf_stats) == 5
        @test StatsAPI.nobs(r) == 50
        @test StatsAPI.pvalue(r) == 0.04
        @test StatsAPI.dof(r) == 3  # lags + 2 for :constant
    end

    @testset "MoonPerronResult construction" begin
        r = MoonPerronResult(-2.1, -1.9, 0.02, 0.03, 2, 100, 20)
        @test r isa MoonPerronResult{Float64}
        @test r.t_a_statistic == -2.1
        @test r.t_b_statistic == -1.9
        @test r.pvalue_a == 0.02
        @test r.pvalue_b == 0.03
        @test r.n_factors == 2
        @test r.n_units == 20
        @test StatsAPI.nobs(r) == 100
        @test StatsAPI.pvalue(r) == 0.02
        @test StatsAPI.dof(r) == 2
    end

    @testset "FactorBreakResult construction" begin
        r = FactorBreakResult(4.5, 0.01, 50, :breitung_eickmeier, 3, 200, 20)
        @test r isa FactorBreakResult{Float64}
        @test r.statistic == 4.5
        @test r.break_date == 50
        @test r.method == :breitung_eickmeier
        @test r.n_factors == 3
        @test r.n_vars == 20
        @test StatsAPI.nobs(r) == 200
        @test StatsAPI.pvalue(r) == 0.01
        @test StatsAPI.dof(r) == 3

        # break_date can be nothing
        r2 = FactorBreakResult(3.2, 0.05, nothing, :chen_dolado_gonzalo, 2, 100, 15)
        @test r2.break_date === nothing
        @test r2.method == :chen_dolado_gonzalo
    end

    @testset "Float32 parametric types" begin
        r = AndrewsResult(Float32(3.45), Float32(0.02), 50, Float32(0.5),
            :supwald, Dict(1 => Float32(12.0), 5 => Float32(8.5), 10 => Float32(6.8)),
            fill(Float32(2.0), 80), Float32(0.15), 100, 3)
        @test r isa AndrewsResult{Float32}

        r2 = MoonPerronResult(Float32(-2.1), Float32(-1.9),
            Float32(0.02), Float32(0.03), 2, 100, 20)
        @test r2 isa MoonPerronResult{Float32}
    end

    @testset "Critical value tables exist (original)" begin
        # Hansen/Andrews sup-Wald
        @test haskey(MacroEconometricModels.HANSEN_ANDREWS_CV, 1)
        @test haskey(MacroEconometricModels.HANSEN_ANDREWS_CV, 5)
        @test haskey(MacroEconometricModels.HANSEN_ANDREWS_CV, 10)
        @test MacroEconometricModels.HANSEN_ANDREWS_CV[1][5] == 8.85
        @test MacroEconometricModels.HANSEN_ANDREWS_CV[1][1] == 12.35
        @test MacroEconometricModels.HANSEN_ANDREWS_CV[1][10] == 7.12

        # Andrews-Ploberger exp-Wald
        @test haskey(MacroEconometricModels.ANDREWS_PLOBERGER_EXP_CV, 1)
        @test haskey(MacroEconometricModels.ANDREWS_PLOBERGER_EXP_CV, 10)
        @test MacroEconometricModels.ANDREWS_PLOBERGER_EXP_CV[1][5] == 2.56

        # Andrews-Ploberger mean-Wald
        @test haskey(MacroEconometricModels.ANDREWS_PLOBERGER_MEAN_CV, 1)
        @test haskey(MacroEconometricModels.ANDREWS_PLOBERGER_MEAN_CV, 10)
        @test MacroEconometricModels.ANDREWS_PLOBERGER_MEAN_CV[1][5] == 3.72

        # Bai-Perron sup-F (q- and trimming-indexed, T086/#185)
        supf_q = MacroEconometricModels.BAIPERRON_SUPF_CV_Q
        @test sort(collect(keys(supf_q))) == [0.05, 0.10, 0.15, 0.20, 0.25]
        @test supf_q[0.15][0.05][1, 1] == 8.58    # q=1, l=1, 5% — canonical value
        @test supf_q[0.15][0.01][1, 1] == 12.29
        @test supf_q[0.15][0.10][1, 1] == 7.04
        @test size(supf_q[0.15][0.05]) == (10, 5)  # q=1..10, l=1..5 at ε=0.15
        @test size(supf_q[0.05][0.05]) == (10, 9)  # l up to 9 at ε=0.05
        # CVs increase with q at fixed l
        @test all(diff(supf_q[0.15][0.05][:, 1]) .> 0)

        # Bai-Perron sequential (column j = supF(j | j-1))
        seqf_q = MacroEconometricModels.BAIPERRON_SEQF_CV_Q
        @test seqf_q[0.15][0.05][1, 2] == 10.13   # q=1, seqF(2|1), 5%
        @test seqf_q[0.15][0.05][1, 1] == 8.58    # seqF(1|0) == supF(1)
        @test size(seqf_q[0.15][0.05]) == (10, 10)

        # Pesaran CIPS
        @test haskey(MacroEconometricModels.PESARAN_CIPS_CV, :constant)
        @test haskey(MacroEconometricModels.PESARAN_CIPS_CV, :trend)
        @test haskey(MacroEconometricModels.PESARAN_CIPS_CV, :none)
        @test haskey(MacroEconometricModels.PESARAN_CIPS_CV[:constant], (20, 30))
        @test haskey(MacroEconometricModels.PESARAN_CIPS_CV[:constant], (100, 100))
        @test MacroEconometricModels.PESARAN_CIPS_CV[:constant][(10, 20)][5] == -2.32
        @test MacroEconometricModels.PESARAN_CIPS_CV[:trend][(10, 20)][5] == -2.82

        # Check all N x T combinations exist for each deterministic
        for det in (:constant, :trend, :none)
            for N in (10, 20, 30, 50, 100)
                for T in (20, 30, 50, 70, 100)
                    @test haskey(MacroEconometricModels.PESARAN_CIPS_CV[det], (N, T))
                    cv = MacroEconometricModels.PESARAN_CIPS_CV[det][(N, T)]
                    @test haskey(cv, 1)
                    @test haskey(cv, 5)
                    @test haskey(cv, 10)
                    # For unit root tests, 1% CV should be more negative than 10%
                    @test cv[1] <= cv[5] <= cv[10]
                end
            end
        end
    end
end

@testset "Andrews Structural Break Tests" begin
    Random.seed!(42)

    @testset "Known structural break" begin
        T_obs = 100
        X = hcat(ones(T_obs), randn(T_obs))
        y = X * [1.0, 2.0] + randn(T_obs) * 0.5
        y[51:end] .+= X[51:end, 2] .* 3.0  # break at t=50

        result = andrews_test(y, X; test=:supwald, trimming=0.15)
        @test result isa AndrewsResult{Float64}
        @test result.test_type == :supwald
        @test result.nobs == T_obs
        @test result.n_params == 2
        @test result.trimming == 0.15
        @test length(result.stat_sequence) > 0
        @test 0.0 <= result.pvalue <= 1.0
        @test 1 <= result.break_index <= T_obs
        @test result.pvalue < 0.10  # strong break should reject
    end

    @testset "No break (stable series)" begin
        T_obs = 200
        X = hcat(ones(T_obs), randn(T_obs))
        y = X * [1.0, 2.0] + randn(T_obs) * 0.5
        result = andrews_test(y, X; test=:supwald)
        @test result isa AndrewsResult{Float64}
        @test result.pvalue >= 0.0
    end

    @testset "All 9 test variants" begin
        T_obs = 100
        X = hcat(ones(T_obs), randn(T_obs))
        y = X * [1.0, 2.0] + randn(T_obs)
        for test_type in [:supwald, :suplr, :suplm, :expwald, :explr, :explm, :meanwald, :meanlr, :meanlm]
            result = andrews_test(y, X; test=test_type)
            @test result.test_type == test_type
            @test result.statistic >= 0
            @test 0.0 <= result.pvalue <= 1.0
        end
    end

    @testset "Error handling" begin
        @test_throws ArgumentError andrews_test(randn(100), ones(100, 1); test=:invalid)
        @test_throws ArgumentError andrews_test(randn(10), ones(5, 1))
    end

    @testset "Float64 fallback" begin
        result = andrews_test(round.(Int, randn(100) .* 10), round.(Int, randn(100, 2) .* 10))
        @test result isa AndrewsResult{Float64}
    end
end

@testset "Bai-Perron Multiple Break Tests" begin
    Random.seed!(123)

    @testset "Two known breaks" begin
        T_obs = 150
        X = hcat(ones(T_obs), collect(1:T_obs) ./ T_obs)
        y = zeros(T_obs)
        y[1:50] = X[1:50, :] * [1.0, 2.0] + randn(50) * 0.3
        y[51:100] = X[51:100, :] * [3.0, -1.0] + randn(50) * 0.3
        y[101:150] = X[101:150, :] * [0.0, 4.0] + randn(50) * 0.3

        result = bai_perron_test(y, X; max_breaks=3, trimming=0.15)
        @test result isa BaiPerronResult{Float64}
        @test result.nobs == T_obs
        @test 1 <= result.n_breaks <= 3
        @test length(result.break_dates) == result.n_breaks
        @test length(result.break_cis) == result.n_breaks
        @test length(result.regime_coefs) == result.n_breaks + 1
        @test length(result.supf_stats) <= 3
        @test length(result.bic_values) >= 2
    end

    @testset "No breaks" begin
        T_obs = 100
        X = ones(T_obs, 1)
        y = 2.0 .+ randn(T_obs) * 0.5
        result = bai_perron_test(y, X; max_breaks=3)
        @test result isa BaiPerronResult{Float64}
        @test result.n_breaks >= 0
    end

    @testset "Float64 fallback" begin
        result = bai_perron_test(round.(Int, randn(100) .* 10), round.(Int, randn(100, 1) .* 10))
        @test result isa BaiPerronResult{Float64}
    end

    @testset "Single known break" begin
        Random.seed!(456)
        T_obs = 100
        X = ones(T_obs, 1)
        y = vcat(fill(1.0, 50), fill(5.0, 50)) + randn(T_obs) * 0.3
        result = bai_perron_test(y, X; max_breaks=3, trimming=0.15)
        @test result isa BaiPerronResult{Float64}
        @test result.n_breaks >= 1
        # Break should be near observation 50
        if result.n_breaks >= 1
            @test abs(result.break_dates[1] - 50) <= 10
        end
    end

    @testset "LWZ criterion" begin
        Random.seed!(789)
        T_obs = 100
        X = ones(T_obs, 1)
        y = vcat(fill(2.0, 50), fill(4.0, 50)) + randn(T_obs) * 0.5
        result = bai_perron_test(y, X; max_breaks=3, criterion=:lwz)
        @test result isa BaiPerronResult{Float64}
        @test length(result.lwz_values) >= 2
    end

    @testset "Error handling" begin
        @test_throws ArgumentError bai_perron_test(randn(10), ones(10, 1))  # too short
        @test_throws ArgumentError bai_perron_test(randn(100), ones(50, 1))  # size mismatch
        @test_throws ArgumentError bai_perron_test(randn(100), ones(100, 1); criterion=:invalid)
    end

    @testset "Regime coefficients and SEs" begin
        Random.seed!(111)
        T_obs = 120
        X = ones(T_obs, 1)
        y = vcat(fill(3.0, 60), fill(7.0, 60)) + randn(T_obs) * 0.2
        result = bai_perron_test(y, X; max_breaks=2, trimming=0.15)
        @test result isa BaiPerronResult{Float64}
        # Each regime should have 1 coefficient (intercept only)
        for coefs in result.regime_coefs
            @test length(coefs) == 1
        end
        for ses in result.regime_ses
            @test length(ses) == 1
            @test all(ses .>= 0)
        end
    end

    @testset "sup-F and sequential statistics" begin
        Random.seed!(222)
        T_obs = 150
        X = ones(T_obs, 1)
        y = vcat(fill(1.0, 50), fill(4.0, 50), fill(2.0, 50)) + randn(T_obs) * 0.3
        result = bai_perron_test(y, X; max_breaks=4, trimming=0.15)
        @test all(result.supf_stats .>= 0)
        @test all(0 .<= result.supf_pvalues .<= 1)
        if !isempty(result.sequential_stats)
            @test all(result.sequential_stats .>= 0)
            @test all(0 .<= result.sequential_pvalues .<= 1)
        end
    end

    @testset "Display method" begin
        Random.seed!(333)
        T_obs = 100
        X = ones(T_obs, 1)
        y = vcat(fill(2.0, 50), fill(5.0, 50)) + randn(T_obs) * 0.5
        result = bai_perron_test(y, X; max_breaks=2)
        io = IOBuffer()
        show(io, result)
        output = String(take!(io))
        @test occursin("Bai-Perron", output)
        @test occursin("Observations", output)
    end

    @testset "Bai (1997) CIs valid ranges" begin
        # CIs should be valid ranges
        rng_bp = Random.MersenneTwister(55443)
        T_bp = 100
        X_bp = ones(T_bp, 1)
        y_bp = vcat(2.0 * ones(50), 5.0 * ones(50)) + 0.5 * randn(rng_bp, T_bp)
        result_bp = bai_perron_test(y_bp, X_bp; max_breaks=3)
        if result_bp.n_breaks > 0
            for (lo, hi) in result_bp.break_cis
                @test lo >= 1
                @test hi <= T_bp
                @test lo < hi
            end
        end
    end
end

@testset "Factor Break Tests" begin
    Random.seed!(42)

    @testset "Breitung-Eickmeier" begin
        # Panel with stable loadings
        T_obs, N = 100, 20
        X = randn(T_obs, N)
        result = factor_break_test(X, 2; method=:breitung_eickmeier)
        @test result isa FactorBreakResult{Float64}
        @test result.method == :breitung_eickmeier
        @test result.n_factors == 2
        @test result.nobs == T_obs
        @test result.n_vars == N
        @test 0.0 <= result.pvalue <= 1.0
        @test result.break_date isa Int
    end

    @testset "FactorModel dispatch" begin
        X = randn(80, 15)
        fm = estimate_factors(X, 2)
        result = factor_break_test(fm; method=:breitung_eickmeier)
        @test result isa FactorBreakResult{Float64}
    end

    @testset "Chen-Dolado-Gonzalo" begin
        X = randn(100, 20)
        result = factor_break_test(X; method=:chen_dolado_gonzalo)
        @test result isa FactorBreakResult{Float64}
        @test result.method == :chen_dolado_gonzalo
    end

    @testset "Han-Inoue" begin
        X = randn(100, 20)
        result = factor_break_test(X, 2; method=:han_inoue)
        @test result isa FactorBreakResult{Float64}
        @test result.method == :han_inoue
        @test result.break_date isa Int
    end

    @testset "Error handling" begin
        @test_throws ArgumentError factor_break_test(randn(10, 5), 2)  # too short
        @test_throws ArgumentError factor_break_test(randn(100, 20), 2; method=:invalid)
    end

    @testset "Float64 fallback" begin
        result = factor_break_test(round.(Int, randn(80, 15) .* 10), 2)
        @test result isa FactorBreakResult{Float64}
    end
end

@testset "PANIC Panel Unit Root" begin
    Random.seed!(42)

    @testset "Basic PANIC" begin
        T_obs, N = 100, 20
        F = cumsum(randn(T_obs))
        Lambda = randn(N)
        e = randn(T_obs, N)
        X = F * Lambda' + e

        result = panic_test(X; r=1, method=:pooled)
        @test result isa PANICResult{Float64}
        @test result.n_factors == 1
        @test result.method == :pooled
        @test result.nobs == T_obs
        @test result.n_units == N
        @test length(result.factor_adf_stats) == 1
        @test length(result.individual_stats) == N
        @test 0.0 <= result.pooled_pvalue <= 1.0
    end

    @testset "Auto factor selection" begin
        X = randn(80, 15)
        result = panic_test(X; r=:auto)
        @test result isa PANICResult{Float64}
        @test result.n_factors >= 1
    end

    @testset "Individual method" begin
        X = randn(60, 10)
        result = panic_test(X; r=1, method=:individual)
        @test result.method == :individual
        @test length(result.individual_pvalues) == 10
    end

    @testset "PanelData dispatch" begin
        using DataFrames
        df = DataFrame(group=repeat(1:5, inner=20), time=repeat(1:20, 5), y=randn(100))
        pd = xtset(df, :group, :time)
        result = panic_test(pd; r=1)
        @test result isa PANICResult{Float64}
        @test result.n_units == 5
    end

    @testset "Error handling" begin
        @test_throws ArgumentError panic_test(randn(5, 2); r=1)
        @test_throws ArgumentError panic_test(randn(50, 3); r=0)
    end
end

@testset "Pesaran CIPS" begin
    Random.seed!(42)

    @testset "Stationary panel" begin
        X = randn(50, 20)
        result = pesaran_cips_test(X; lags=1, deterministic=:constant)
        @test result isa PesaranCIPSResult{Float64}
        @test result.nobs == 50
        @test result.n_units == 20
        @test length(result.individual_cadf_stats) == 20
        @test 0.0 <= result.pvalue <= 1.0
    end

    @testset "Deterministic variants" begin
        X = randn(50, 10)
        for det in [:none, :constant, :trend]
            result = pesaran_cips_test(X; lags=1, deterministic=det)
            @test result.deterministic == det
        end
    end

    @testset "Error handling" begin
        @test_throws ArgumentError pesaran_cips_test(randn(5, 2); lags=1)
        @test_throws ArgumentError pesaran_cips_test(randn(50, 3); deterministic=:invalid)
    end
end

@testset "Moon-Perron" begin
    Random.seed!(42)

    @testset "Basic test" begin
        X = randn(80, 15)
        result = moon_perron_test(X; r=1)
        @test result isa MoonPerronResult{Float64}
        @test result.n_factors == 1
        @test result.nobs == 80
        @test result.n_units == 15
        @test 0.0 <= result.pvalue_a <= 1.0
        @test 0.0 <= result.pvalue_b <= 1.0
    end

    @testset "Auto factor selection" begin
        X = randn(60, 10)
        result = moon_perron_test(X; r=:auto)
        @test result isa MoonPerronResult{Float64}
    end

    @testset "Error handling" begin
        @test_throws ArgumentError moon_perron_test(randn(5, 2); r=1)
    end
end

# =============================================================================
# T086 (#185): Bai-Perron q-indexed critical values
# =============================================================================

@testset "T086: Bai-Perron q-indexed p-values" begin
    MEM = MacroEconometricModels

    @testset "q changes the p-value (old code used q=1 for all)" begin
        # supF(1) = 10.0 at ε=0.15: significant for q=1 (5% cv 8.58) but not
        # for q=3 (5% cv 13.47-ish) — the q=1-only table was too liberal
        p_q1 = MEM._baiperron_pvalue(10.0, 1, 1, 0.15, :supf)
        p_q3 = MEM._baiperron_pvalue(10.0, 1, 3, 0.15, :supf)
        @test p_q1 < 0.05
        @test p_q3 > 0.05
        @test p_q1 < p_q3   # same stat, more breaking regressors -> less evidence
    end

    @testset "level interpolation pins" begin
        # exactly at tabulated CVs the p-value equals the nominal level
        supf = MEM.BAIPERRON_SUPF_CV_Q[0.15]
        for (lvl, p_expect) in ((0.10, 0.10), (0.05, 0.05), (0.025, 0.025))
            for q in (1, 2, 5), l in (1, 3)
                @test MEM._baiperron_pvalue(supf[lvl][q, l], l, q, 0.15, :supf) ≈ p_expect atol = 1e-10
            end
        end
        @test MEM._baiperron_pvalue(supf[0.01][2, 2], 2, 2, 0.15, :supf) ≈ 0.01 atol = 1e-10
    end

    @testset "trimming axis + guards" begin
        # different tabulated trimmings give different p-values
        p15 = MEM._baiperron_pvalue(9.0, 1, 1, 0.15, :supf)
        p05 = MEM._baiperron_pvalue(9.0, 1, 1, 0.05, :supf)
        @test p15 != p05
        # q out of range errors informatively
        @test_throws ArgumentError MEM._baiperron_pvalue(9.0, 1, 11, 0.15, :supf)
        # beyond tabulated breaks for large trimming -> NaN
        @test isnan(MEM._baiperron_pvalue(9.0, 4, 1, 0.25, :supf))
    end

    @testset "bai_perron_test threads q = k" begin
        rng = Random.MersenneTwister(18501)
        n = 200
        x = randn(rng, n)
        y = vcat(0.5 .+ 0.3 .* x[1:100], 2.5 .+ 0.3 .* x[101:200]) .+ 0.4 .* randn(rng, n)
        X = hcat(ones(n), x)   # k = q = 2
        r = bai_perron_test(y, X; max_breaks=3)
        # p-value of supF(1) equals the q=2 table lookup
        @test r.supf_pvalues[1] ≈
              MEM._baiperron_pvalue(r.supf_stats[1], 1, 2, 0.15, :supf) atol = 1e-12
        @test r.n_breaks >= 1   # the level shift is found

        # q = k is genuinely threaded: 11 regressors exceed the tabulated range
        X11 = hcat(ones(n), randn(rng, n, 10))
        @test_throws ArgumentError bai_perron_test(y, X11; max_breaks=1)
    end
end
