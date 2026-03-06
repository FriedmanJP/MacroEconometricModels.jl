# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# Coverage tests for nowcast/bvar_nowcast.jl and nowcast/bridge.jl
# Targets specific uncovered branches: interior NaN interpolation,
# first/last-row NaN, fallback paths, short columns, singular XtX,
# and multiple monthly indicator combinations.

using Test
using MacroEconometricModels
using Random
using LinearAlgebra
using Statistics

Random.seed!(9002)

# =============================================================================
# 1. BVAR Nowcast: Fallback Path via Public API
# =============================================================================

@testset "BVAR Nowcast Coverage" begin

    @testset "Fallback path: too few complete rows (all rows have some NaN)" begin
        # When t_complete < lags + 2, the fallback fills NaN with column means.
        # Every row has at least one NaN, so t_complete = 0 -> fallback path.
        # After fallback fills NaN and resets t_complete = T_obs, the
        # _bvar_smooth_missing call on the original Ymat (still with NaN)
        # triggers interior NaN interpolation branches.
        rng = Random.MersenneTwister(9040)
        T_obs = 30
        N = 3
        Y = randn(rng, T_obs, N)

        # Make every row have at least one NaN so t_complete = 0
        for t in 1:T_obs
            j = mod(t - 1, N) + 1
            Y[t, j] = NaN
        end

        m = nowcast_bvar(Y, 2, 1; lags=2, max_iter=10)

        @test m isa NowcastBVAR{Float64}
        @test !any(isnan, m.X_sm)
        @test isfinite(m.loglik)
    end

    @testset "Fallback with empty valid set (column all NaN)" begin
        # A column that is entirely NaN triggers the `isempty(valid) ? zero(Tf)` path
        # in the fallback (line 85).
        rng = Random.MersenneTwister(9041)
        T_obs = 20
        N = 3
        Y = randn(rng, T_obs, N)

        # Make column 3 entirely NaN
        Y[:, 3] .= NaN
        # Make every row have NaN to trigger fallback path
        for t in 1:T_obs
            if !isnan(Y[t, 1]) && !isnan(Y[t, 2])
                Y[t, 1] = NaN
            end
        end

        m = nowcast_bvar(Y, 2, 1; lags=2, max_iter=10)

        @test m isa NowcastBVAR{Float64}
        @test !any(isnan, m.X_sm)
        @test all(isfinite.(m.X_sm[:, 3]))
    end

    @testset "Ragged edge with t_lag < 1 in smoothing" begin
        # When t_complete < T_obs and lags are large, the BVAR forecasting
        # loop can hit the t_lag < 1 branch (line 295: zeros fallback).
        rng = Random.MersenneTwister(9055)
        T_obs = 20
        N = 3
        Y = randn(rng, T_obs, N)
        # Ragged edge: last 15 rows have NaN in some columns
        Y[6:T_obs, 2:3] .= NaN
        # t_complete = 5 (row 5 is last clean row)
        # lags = 4, so lags + 2 = 6 > 5 -> fallback path triggers

        m = nowcast_bvar(Y, 2, 1; lags=4, max_iter=10)

        @test m isa NowcastBVAR{Float64}
        @test !any(isnan, m.X_sm)
    end

    @testset "Fallback with NaN in first rows (right-neighbor-only interpolation)" begin
        # Triggers interior NaN interpolation where row 1 has NaN ->
        # lo search finds nothing, only hi neighbor exists.
        rng = Random.MersenneTwister(9042)
        T_obs = 24
        N = 3
        Y = randn(rng, T_obs, N)

        # Row 1 col 1 NaN, plus enough other NaN to force fallback
        Y[1, 1] = NaN
        # Make most rows have NaN to force fallback path
        for t in 2:T_obs
            Y[t, mod(t, N) + 1] = NaN
        end

        m = nowcast_bvar(Y, 2, 1; lags=2, max_iter=10)

        @test m isa NowcastBVAR{Float64}
        @test !any(isnan, m.X_sm)
        @test isfinite(m.X_sm[1, 1])
    end

    @testset "Fallback with NaN in last rows (left-neighbor-only interpolation)" begin
        # Triggers interior NaN interpolation where last row has NaN ->
        # hi search finds nothing, only lo neighbor exists.
        rng = Random.MersenneTwister(9043)
        T_obs = 24
        N = 3
        Y = randn(rng, T_obs, N)

        # Last row col 1 NaN
        Y[T_obs, 1] = NaN
        # Make most rows have NaN to force fallback path
        for t in 1:(T_obs-1)
            Y[t, mod(t, N) + 1] = NaN
        end

        m = nowcast_bvar(Y, 2, 1; lags=2, max_iter=10)

        @test m isa NowcastBVAR{Float64}
        @test !any(isnan, m.X_sm)
    end

    @testset "Fallback with consecutive NaN block" begin
        # Block of consecutive NaN in one column exercises the while loops
        # for lo/hi neighbor search in _bvar_smooth_missing.
        rng = Random.MersenneTwister(9044)
        T_obs = 30
        N = 3
        Y = randn(rng, T_obs, N)

        # Block of NaN in col 1, rows 10-15
        Y[10:15, 1] .= NaN
        # Make every row have at least one NaN to force fallback
        for t in 1:T_obs
            if !any(isnan, Y[t, :])
                Y[t, 2] = NaN
            end
        end

        m = nowcast_bvar(Y, 2, 1; lags=2, max_iter=10)

        @test m isa NowcastBVAR{Float64}
        @test !any(isnan, m.X_sm)
        # Interpolated values should be finite
        for t in 10:15
            @test isfinite(m.X_sm[t, 1])
        end
    end
end

# =============================================================================
# 2. Direct _bvar_smooth_missing Tests
# =============================================================================

@testset "Direct _bvar_smooth_missing coverage" begin

    @testset "Interior NaN: linear interpolation with both neighbors" begin
        rng = Random.MersenneTwister(9200)
        T_obs = 20
        N = 2
        lags = 2

        Y = randn(rng, T_obs, N)
        Y[10, 1] = NaN  # interior NaN

        k = 1 + N * lags
        beta = zeros(Float64, k, N)
        sigma = Matrix{Float64}(I(N))

        # Set t_complete = T_obs so the interior loop processes row 10
        X_sm = MacroEconometricModels._bvar_smooth_missing(Y, beta, sigma, lags, T_obs)

        @test !any(isnan, X_sm)
        expected = Y[9, 1] + (Y[11, 1] - Y[9, 1]) * 1.0 / 2.0
        @test X_sm[10, 1] ≈ expected
    end

    @testset "First-row NaN: right neighbor only (elseif hi <= T_obs)" begin
        rng = Random.MersenneTwister(9210)
        T_obs = 15
        N = 2
        lags = 2

        Y = randn(rng, T_obs, N)
        Y[1, 1] = NaN

        k = 1 + N * lags
        beta = zeros(Float64, k, N)
        sigma = Matrix{Float64}(I(N))

        X_sm = MacroEconometricModels._bvar_smooth_missing(Y, beta, sigma, lags, T_obs)

        @test !any(isnan, X_sm)
        @test X_sm[1, 1] ≈ Y[2, 1]
    end

    @testset "Last-row NaN within t_complete: left neighbor only (elseif lo >= 1)" begin
        rng = Random.MersenneTwister(9220)
        T_obs = 15
        N = 2
        lags = 2

        Y = randn(rng, T_obs, N)
        Y[T_obs, 1] = NaN

        k = 1 + N * lags
        beta = zeros(Float64, k, N)
        sigma = Matrix{Float64}(I(N))

        X_sm = MacroEconometricModels._bvar_smooth_missing(Y, beta, sigma, lags, T_obs)

        @test !any(isnan, X_sm)
        @test X_sm[T_obs, 1] ≈ Y[T_obs - 1, 1]
    end

    @testset "Entire column NaN: column mean fallback (else branch)" begin
        rng = Random.MersenneTwister(9230)
        T_obs = 15
        N = 2
        lags = 2

        Y = randn(rng, T_obs, N)
        Y[:, 2] .= NaN

        k = 1 + N * lags
        beta = zeros(Float64, k, N)
        sigma = Matrix{Float64}(I(N))

        X_sm = MacroEconometricModels._bvar_smooth_missing(Y, beta, sigma, lags, T_obs)

        @test !any(isnan, X_sm)
        @test all(X_sm[:, 2] .== 0.0)
    end

    @testset "Consecutive NaN block: lo/hi while loops traverse multiple rows" begin
        rng = Random.MersenneTwister(9240)
        T_obs = 20
        N = 2
        lags = 2

        Y = randn(rng, T_obs, N)
        Y[8:12, 1] .= NaN

        k = 1 + N * lags
        beta = zeros(Float64, k, N)
        sigma = Matrix{Float64}(I(N))

        X_sm = MacroEconometricModels._bvar_smooth_missing(Y, beta, sigma, lags, T_obs)

        @test !any(isnan, X_sm)
        lo_val = Y[7, 1]
        hi_val = Y[13, 1]
        for t in 8:12
            expected = lo_val + (hi_val - lo_val) * Float64(t - 7) / Float64(13 - 7)
            @test X_sm[t, 1] ≈ expected
        end
    end

    @testset "Ragged edge: t_complete < T_obs with t_lag < 1 zeros fallback" begin
        rng = Random.MersenneTwister(9250)
        T_obs = 10
        N = 2
        lags = 3

        Y = randn(rng, T_obs, N)
        Y[3:T_obs, :] .= NaN

        k = 1 + N * lags
        beta = 0.01 * randn(rng, k, N)
        sigma = Matrix{Float64}(0.1 * I(N))

        # t_complete = 2, so forecasting starts at t=3
        # For t=3, lag=3: t_lag = 0 < 1 -> zeros branch
        X_sm = MacroEconometricModels._bvar_smooth_missing(Y, beta, sigma, lags, 2)

        @test !any(isnan, X_sm)
        @test all(isfinite.(X_sm))
    end

    @testset "Multiple NaN patterns across columns in interior" begin
        rng = Random.MersenneTwister(9260)
        T_obs = 20
        N = 3
        lags = 2

        Y = randn(rng, T_obs, N)
        Y[1:3, 1] .= NaN    # beginning NaN (right neighbor only)
        Y[10, 2] = NaN       # single interior NaN
        Y[18:20, 3] .= NaN   # end NaN (left neighbor only)

        k = 1 + N * lags
        beta = zeros(Float64, k, N)
        sigma = Matrix{Float64}(I(N))

        X_sm = MacroEconometricModels._bvar_smooth_missing(Y, beta, sigma, lags, T_obs)

        @test !any(isnan, X_sm)
        # Col 1 rows 1-3: filled from right neighbor (row 4)
        @test X_sm[1, 1] ≈ Y[4, 1]
        @test X_sm[2, 1] ≈ Y[4, 1]
        @test X_sm[3, 1] ≈ Y[4, 1]
        # Col 3 rows 18-20: filled from left neighbor (row 17)
        @test X_sm[18, 3] ≈ Y[17, 3]
        @test X_sm[19, 3] ≈ Y[17, 3]
        @test X_sm[20, 3] ≈ Y[17, 3]
    end
end

# =============================================================================
# 3. Bridge Equation Coverage
# =============================================================================

@testset "Bridge Equation Coverage" begin

    @testset "Multiple monthly indicators: pair combinations with 5 monthly vars" begin
        # With nM = 5, _bridge_combinations generates C(5,2) + 5 = 15 equations.
        rng = Random.MersenneTwister(9100)
        T_obs = 120
        nM = 5
        nQ = 2
        Y = randn(rng, T_obs, nM + nQ)

        for t in 1:T_obs
            if mod(t, 3) != 0
                Y[t, (nM+1):(nM+nQ)] .= NaN
            end
        end

        m = nowcast_bridge(Y, nM, nQ; lagM=1, lagQ=1, lagY=1)

        @test m.n_equations == 15
        @test size(m.Y_individual, 2) == 15

        n_valid = count(c -> length(c) > 0, m.coefficients)
        @test n_valid >= 10
    end

    @testset "Bridge combinations: pairs and univariate for various nM" begin
        combos2 = MacroEconometricModels._bridge_combinations(2, 0)
        @test size(combos2, 1) == 3  # 1 pair + 2 univariate

        combos5 = MacroEconometricModels._bridge_combinations(5, 0)
        @test size(combos5, 1) == 15  # C(5,2) + 5 = 10 + 5

        combos1 = MacroEconometricModels._bridge_combinations(1, 0)
        @test size(combos1, 1) == 1  # 0 pairs + 1 univariate
        @test combos1[1, 1] == 1
        @test combos1[1, 2] == 1

        # Verify pair structure for nM=3: 3 pairs + 3 singles = 6
        combos3 = MacroEconometricModels._bridge_combinations(3, 0)
        @test size(combos3, 1) == 6
        # First 3 rows should be pairs
        @test combos3[1, :] == [1, 2]
        @test combos3[2, :] == [1, 3]
        @test combos3[3, :] == [2, 3]
        # Last 3 rows should be univariate (i == i)
        @test combos3[4, :] == [1, 1]
        @test combos3[5, :] == [2, 2]
        @test combos3[6, :] == [3, 3]
    end

    @testset "Interior NaN in monthly columns triggers interpolation" begin
        rng = Random.MersenneTwister(9110)
        T_obs = 90
        nM = 3
        nQ = 1
        Y = randn(rng, T_obs, nM + nQ)

        for t in 1:T_obs
            mod(t, 3) != 0 && (Y[t, nM + 1] = NaN)
        end

        Y[15, 1] = NaN         # single interior gap
        Y[30:32, 2] .= NaN     # block gap
        Y[1:2, 3] .= NaN       # beginning gap (backward fill)

        m = nowcast_bridge(Y, nM, nQ; lagM=1, lagQ=0, lagY=1)

        @test m isa NowcastBridge{Float64}
        @test !any(isnan, m.X_sm[1:T_obs, 1:nM])
    end

    @testset "End-of-column NaN in monthly: forward fill" begin
        rng = Random.MersenneTwister(9120)
        T_obs = 60
        nM = 2
        nQ = 1
        Y = randn(rng, T_obs, nM + nQ)

        for t in 1:T_obs
            mod(t, 3) != 0 && (Y[t, nM + 1] = NaN)
        end

        Y[58:60, 1] .= NaN

        m = nowcast_bridge(Y, nM, nQ; lagM=1, lagQ=0, lagY=1)

        @test !any(isnan, m.X_sm[:, 1:nM])
    end

    @testset "Column mean fallback in bridge fill (entire column NaN)" begin
        rng = Random.MersenneTwister(9130)
        T_obs = 60
        nM = 3
        nQ = 1
        Y = randn(rng, T_obs, nM + nQ)

        for t in 1:T_obs
            mod(t, 3) != 0 && (Y[t, nM + 1] = NaN)
        end

        Y[:, 2] .= NaN

        m = nowcast_bridge(Y, nM, nQ; lagM=1, lagQ=0, lagY=1)

        @test m isa NowcastBridge{Float64}
        @test all(isfinite.(m.X_sm[:, 2]))
    end

    @testset "Singular XtX: collinear monthly indicators" begin
        rng = Random.MersenneTwister(9140)
        T_obs = 90
        nM = 3
        nQ = 1
        Y = randn(rng, T_obs, nM + nQ)

        # Make columns 1 and 2 identical (perfectly collinear)
        Y[:, 2] .= Y[:, 1]

        for t in 1:T_obs
            mod(t, 3) != 0 && (Y[t, nM + 1] = NaN)
        end

        m = nowcast_bridge(Y, nM, nQ; lagM=1, lagQ=0, lagY=1)

        @test m isa NowcastBridge{Float64}
        @test !all(isnan, m.Y_nowcast)
    end

    @testset "Very short sample: some equations skipped" begin
        rng = Random.MersenneTwister(9150)
        T_obs = 18  # only 6 quarters
        nM = 3
        nQ = 1
        Y = randn(rng, T_obs, nM + nQ)

        for t in 1:T_obs
            mod(t, 3) != 0 && (Y[t, nM + 1] = NaN)
        end

        # Remove some quarterly targets
        Y[6, nM + 1] = NaN
        Y[9, nM + 1] = NaN
        Y[12, nM + 1] = NaN

        m = nowcast_bridge(Y, nM, nQ; lagM=1, lagQ=0, lagY=1)

        @test m isa NowcastBridge{Float64}
    end

    @testset "Multiple quarterly variables with pairs" begin
        rng = Random.MersenneTwister(9160)
        T_obs = 120
        nM = 4
        nQ = 3
        Y = randn(rng, T_obs, nM + nQ)

        for t in 1:T_obs
            if mod(t, 3) != 0
                Y[t, (nM+1):(nM+nQ)] .= NaN
            end
        end

        m = nowcast_bridge(Y, nM, nQ; lagM=1, lagQ=1, lagY=1)

        @test m.n_equations == 10  # C(4,2) + 4 = 10
        @test m.nQ == 3
        @test length(m.Y_nowcast) == T_obs ÷ 3
    end
end

# =============================================================================
# 4. Direct _bridge_fill_monthly Tests
# =============================================================================

@testset "Direct _bridge_fill_monthly coverage" begin

    @testset "Interior gap: linear interpolation" begin
        Y = Float64[1 10; 2 20; NaN NaN; 4 40; 5 50]
        nM = 2
        X = MacroEconometricModels._bridge_fill_monthly(Y, nM)

        @test !any(isnan, X)
        @test X[3, 1] ≈ 3.0
        @test X[3, 2] ≈ 30.0
    end

    @testset "Beginning NaN: backward fill" begin
        Y = Float64[NaN 10; NaN 20; 3 30; 4 40]
        nM = 2
        X = MacroEconometricModels._bridge_fill_monthly(Y, nM)

        @test !any(isnan, X)
        @test X[1, 1] ≈ 3.0
        @test X[2, 1] ≈ 3.0
    end

    @testset "End NaN: forward fill" begin
        Y = Float64[1 10; 2 20; NaN NaN]
        nM = 2
        X = MacroEconometricModels._bridge_fill_monthly(Y, nM)

        @test !any(isnan, X)
        @test X[3, 1] ≈ 2.0
        @test X[3, 2] ≈ 20.0
    end

    @testset "Entire column NaN: zero fallback" begin
        Y = Float64[1 NaN; 2 NaN; 3 NaN]
        nM = 2
        X = MacroEconometricModels._bridge_fill_monthly(Y, nM)

        @test !any(isnan, X)
        @test all(X[:, 2] .== 0.0)
    end

    @testset "Multiple consecutive NaN: interpolation across block" begin
        Y = zeros(Float64, 8, 1)
        Y[1, 1] = 2.0
        Y[2:6, 1] .= NaN
        Y[7, 1] = 8.0
        Y[8, 1] = 10.0

        X = MacroEconometricModels._bridge_fill_monthly(Y, 1)

        @test !any(isnan, X)
        for t in 2:6
            expected = 2.0 + (8.0 - 2.0) * Float64(t - 1) / Float64(7 - 1)
            @test X[t, 1] ≈ expected
        end
    end
end
