# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using LinearAlgebra
using Statistics
using Random
using MacroEconometricModels

# Use MacroEconometricModels versions of StatsAPI functions
const rfm_residuals = MacroEconometricModels.residuals
const rfm_r2 = MacroEconometricModels.r2
const rfm_predict = MacroEconometricModels.predict
const rfm_nobs = MacroEconometricModels.nobs

@testset "Block-Restricted Factor Model Tests" begin

    @testset "Unrestricted model has block_names = nothing" begin
        rng = Random.MersenneTwister(42)
        X = randn(rng, 100, 20)
        fm = estimate_factors(X, 3)
        @test fm.block_names === nothing
        @test fm isa FactorModel
    end

    @testset "Block-restricted estimation — correct dimensions" begin
        # DGP-06: shared simulator with explicit block-structured loadings
        # (was: bespoke iid-factor loop). Dynamics do not disturb the blocks.
        rng = Random.MersenneTwister(123)
        T_obs, N = 200, 15
        r = 3

        # Generate data with known block structure
        Lambda_true = zeros(N, r)
        Lambda_true[1:5, 1] = randn(rng, 5)
        Lambda_true[6:10, 2] = randn(rng, 5)
        Lambda_true[11:15, 3] = randn(rng, 5)
        A3 = [0.5 0.1 0.0; 0.05 0.5 0.1; 0.0 0.05 0.5]
        X = dgp_dynamic_factors(rng; A=A3, Lambda=Lambda_true, N=N, T=T_obs,
                                idio_sd=0.3).X

        blocks = Dict(:block_A => [1,2,3,4,5], :block_B => [6,7,8,9,10], :block_C => [11,12,13,14,15])
        fm = estimate_factors(X, r; blocks=blocks)

        @test fm isa FactorModel
        @test size(fm.factors) == (T_obs, r)
        @test size(fm.loadings) == (N, r)
        @test fm.r == r
        @test fm.standardized == true
        @test length(fm.eigenvalues) == N
        @test length(fm.explained_variance) == N
        @test length(fm.cumulative_variance) == N
        @test fm.block_names !== nothing
        @test length(fm.block_names) == r
        @test Set(fm.block_names) == Set([:block_A, :block_B, :block_C])
    end

    @testset "Zero restrictions enforced" begin
        # DGP-06: shared simulator with explicit block-structured loadings.
        rng = Random.MersenneTwister(234)
        T_obs, N = 200, 12
        r = 2

        # Generate data with block structure
        Lambda_true = zeros(N, r)
        Lambda_true[1:6, 1] = randn(rng, 6)
        Lambda_true[7:12, 2] = randn(rng, 6)
        X = dgp_dynamic_factors(rng; A=[0.5 0.1; 0.1 0.5], Lambda=Lambda_true,
                                N=N, T=T_obs, idio_sd=0.3).X

        blocks = Dict(:real => [1,2,3,4,5,6], :nominal => [7,8,9,10,11,12])
        fm = estimate_factors(X, r; blocks=blocks)

        # Find which factor index corresponds to each block
        real_idx = findfirst(==(:real), fm.block_names)
        nominal_idx = findfirst(==(:nominal), fm.block_names)

        # Loadings for :real factor should be zero for variables 7-12
        @test all(fm.loadings[7:12, real_idx] .== 0.0)

        # Loadings for :nominal factor should be zero for variables 1-6
        @test all(fm.loadings[1:6, nominal_idx] .== 0.0)

        # Non-zero loadings should exist in the correct blocks
        @test any(fm.loadings[1:6, real_idx] .!= 0.0)
        @test any(fm.loadings[7:12, nominal_idx] .!= 0.0)
    end

    @testset "R-squared reasonable for known DGP" begin
        # DGP-06: shared simulator with explicit strong block loadings.
        rng = Random.MersenneTwister(345)
        T_obs, N = 300, 10
        r = 2

        # Strong factor structure
        Lambda_true = zeros(N, r)
        Lambda_true[1:5, 1] = randn(rng, 5) .* 2.0
        Lambda_true[6:10, 2] = randn(rng, 5) .* 2.0
        X = dgp_dynamic_factors(rng; A=[0.5 0.1; 0.1 0.5], Lambda=Lambda_true,
                                N=N, T=T_obs, idio_sd=0.2).X

        blocks = Dict(:factor1 => [1,2,3,4,5], :factor2 => [6,7,8,9,10])
        fm = estimate_factors(X, r; blocks=blocks)

        r2_vals = rfm_r2(fm)
        @test length(r2_vals) == N
        @test all(isfinite, r2_vals)

        # With strong signal and low noise, R2 should be reasonably high
        @test mean(r2_vals) > 0.3
    end

    @testset "Validation — wrong block count" begin
        rng = Random.MersenneTwister(456)
        X = randn(rng, 100, 10)

        # 2 blocks but r=3
        blocks = Dict(:a => [1,2,3,4,5], :b => [6,7,8,9,10])
        @test_throws ArgumentError estimate_factors(X, 3; blocks=blocks)

        # 3 blocks but r=2
        blocks3 = Dict(:a => [1,2,3], :b => [4,5,6], :c => [7,8,9])
        @test_throws ArgumentError estimate_factors(X, 2; blocks=blocks3)
    end

    @testset "Validation — overlapping indices" begin
        rng = Random.MersenneTwister(567)
        X = randn(rng, 100, 10)

        # Variable 5 in both blocks
        blocks = Dict(:a => [1,2,3,4,5], :b => [5,6,7,8,9])
        @test_throws ArgumentError estimate_factors(X, 2; blocks=blocks)
    end

    @testset "Validation — out-of-range indices" begin
        rng = Random.MersenneTwister(678)
        X = randn(rng, 100, 10)

        # Index 0 is out of range
        blocks = Dict(:a => [0,1,2,3,4], :b => [5,6,7,8,9])
        @test_throws ArgumentError estimate_factors(X, 2; blocks=blocks)

        # Index 11 is out of range for N=10
        blocks2 = Dict(:a => [1,2,3,4,5], :b => [6,7,8,9,11])
        @test_throws ArgumentError estimate_factors(X, 2; blocks=blocks2)
    end

    @testset "Validation — too few variables per block" begin
        rng = Random.MersenneTwister(789)
        X = randn(rng, 100, 10)

        # Block :a has only 1 variable
        blocks = Dict(:a => [1], :b => [2,3,4,5,6,7,8,9,10])
        @test_throws ArgumentError estimate_factors(X, 2; blocks=blocks)
    end

    @testset "Display with block names" begin
        # DGP-06: shared simulator with explicit block-structured loadings.
        rng = Random.MersenneTwister(890)
        T_obs, N = 100, 10
        r = 2

        Lambda_true = zeros(N, r)
        Lambda_true[1:5, 1] = randn(rng, 5)
        Lambda_true[6:10, 2] = randn(rng, 5)
        X = dgp_dynamic_factors(rng; A=[0.5 0.1; 0.1 0.5], Lambda=Lambda_true,
                                N=N, T=T_obs, idio_sd=0.3).X

        blocks = Dict(:real_activity => [1,2,3,4,5], :prices => [6,7,8,9,10])
        fm = estimate_factors(X, r; blocks=blocks)

        # Display should include block names and not error
        io = IOBuffer()
        show(io, fm)
        output = String(take!(io))
        @test contains(output, "Static Factor Model")
        @test contains(output, "Block-Restricted")
        # At least one block name should appear in the output
        @test contains(output, "real_activity") || contains(output, "prices")
    end

    @testset "StatsAPI interface works with restricted model" begin
        # DGP-06: shared simulator with explicit block-structured loadings.
        rng = Random.MersenneTwister(901)
        T_obs, N = 100, 10
        r = 2

        Lambda_true = zeros(N, r)
        Lambda_true[1:5, 1] = randn(rng, 5)
        Lambda_true[6:10, 2] = randn(rng, 5)
        X = dgp_dynamic_factors(rng; A=[0.5 0.1; 0.1 0.5], Lambda=Lambda_true,
                                N=N, T=T_obs, idio_sd=0.3).X

        blocks = Dict(:block1 => [1,2,3,4,5], :block2 => [6,7,8,9,10])
        fm = estimate_factors(X, r; blocks=blocks)

        @test rfm_nobs(fm) == T_obs

        pred = rfm_predict(fm)
        @test size(pred) == (T_obs, N)
        @test all(isfinite, pred)

        resid = rfm_residuals(fm)
        @test size(resid) == (T_obs, N)
        @test all(isfinite, resid)

        r2_vals = rfm_r2(fm)
        @test length(r2_vals) == N
        @test all(isfinite, r2_vals)
    end

    @testset "Without standardization" begin
        # DGP-06: shared simulator with explicit block-structured loadings.
        rng = Random.MersenneTwister(12)
        T_obs, N = 100, 8
        r = 2

        Lambda_true = zeros(N, r)
        Lambda_true[1:4, 1] = randn(rng, 4)
        Lambda_true[5:8, 2] = randn(rng, 4)
        X = dgp_dynamic_factors(rng; A=[0.5 0.1; 0.1 0.5], Lambda=Lambda_true,
                                N=N, T=T_obs, idio_sd=0.3).X

        blocks = Dict(:a => [1,2,3,4], :b => [5,6,7,8])
        fm = estimate_factors(X, r; blocks=blocks, standardize=false)

        @test fm.standardized == false
        @test fm.block_names !== nothing
        @test size(fm.factors) == (T_obs, r)
    end

    @testset "Float32 type stability" begin
        rng = Random.MersenneTwister(23)
        T_obs, N = 100, 8
        r = 2

        X32 = randn(rng, Float32, T_obs, N)
        blocks = Dict(:a => [1,2,3,4], :b => [5,6,7,8])
        fm = estimate_factors(X32, r; blocks=blocks)

        @test fm isa FactorModel{Float32}
        @test eltype(fm.factors) == Float32
        @test eltype(fm.loadings) == Float32
    end

    @testset "Partial coverage — not all variables assigned" begin
        rng = Random.MersenneTwister(34)
        T_obs, N = 100, 10
        r = 2

        X = randn(rng, T_obs, N)
        # Only 8 of 10 variables are assigned to blocks
        blocks = Dict(:a => [1,2,3,4], :b => [5,6,7,8])
        fm = estimate_factors(X, r; blocks=blocks)

        # Variables 9-10 should have zero loadings on all factors
        @test all(fm.loadings[9, :] .== 0.0)
        @test all(fm.loadings[10, :] .== 0.0)
    end

end
