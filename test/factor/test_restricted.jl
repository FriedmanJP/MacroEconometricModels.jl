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
        Random.seed!(42)
        X = randn(100, 20)
        fm = estimate_factors(X, 3)
        @test fm.block_names === nothing
        @test fm isa FactorModel
    end

    @testset "Block-restricted estimation — correct dimensions" begin
        Random.seed!(123)
        T_obs, N = 200, 15
        r = 3

        # Generate data with known block structure
        F_true = randn(T_obs, r)
        Lambda_true = zeros(N, r)
        Lambda_true[1:5, 1] = randn(5)
        Lambda_true[6:10, 2] = randn(5)
        Lambda_true[11:15, 3] = randn(5)
        X = F_true * Lambda_true' + 0.3 * randn(T_obs, N)

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
        Random.seed!(234)
        T_obs, N = 200, 12
        r = 2

        # Generate data with block structure
        F_true = randn(T_obs, r)
        Lambda_true = zeros(N, r)
        Lambda_true[1:6, 1] = randn(6)
        Lambda_true[7:12, 2] = randn(6)
        X = F_true * Lambda_true' + 0.3 * randn(T_obs, N)

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
        Random.seed!(345)
        T_obs, N = 300, 10
        r = 2

        # Strong factor structure
        F_true = randn(T_obs, r)
        Lambda_true = zeros(N, r)
        Lambda_true[1:5, 1] = randn(5) .* 2.0
        Lambda_true[6:10, 2] = randn(5) .* 2.0
        X = F_true * Lambda_true' + 0.2 * randn(T_obs, N)

        blocks = Dict(:factor1 => [1,2,3,4,5], :factor2 => [6,7,8,9,10])
        fm = estimate_factors(X, r; blocks=blocks)

        r2_vals = rfm_r2(fm)
        @test length(r2_vals) == N
        @test all(isfinite, r2_vals)

        # With strong signal and low noise, R2 should be reasonably high
        @test mean(r2_vals) > 0.3
    end

    @testset "Validation — wrong block count" begin
        Random.seed!(456)
        X = randn(100, 10)

        # 2 blocks but r=3
        blocks = Dict(:a => [1,2,3,4,5], :b => [6,7,8,9,10])
        @test_throws ArgumentError estimate_factors(X, 3; blocks=blocks)

        # 3 blocks but r=2
        blocks3 = Dict(:a => [1,2,3], :b => [4,5,6], :c => [7,8,9])
        @test_throws ArgumentError estimate_factors(X, 2; blocks=blocks3)
    end

    @testset "Validation — overlapping indices" begin
        Random.seed!(567)
        X = randn(100, 10)

        # Variable 5 in both blocks
        blocks = Dict(:a => [1,2,3,4,5], :b => [5,6,7,8,9])
        @test_throws ArgumentError estimate_factors(X, 2; blocks=blocks)
    end

    @testset "Validation — out-of-range indices" begin
        Random.seed!(678)
        X = randn(100, 10)

        # Index 0 is out of range
        blocks = Dict(:a => [0,1,2,3,4], :b => [5,6,7,8,9])
        @test_throws ArgumentError estimate_factors(X, 2; blocks=blocks)

        # Index 11 is out of range for N=10
        blocks2 = Dict(:a => [1,2,3,4,5], :b => [6,7,8,9,11])
        @test_throws ArgumentError estimate_factors(X, 2; blocks=blocks2)
    end

    @testset "Validation — too few variables per block" begin
        Random.seed!(789)
        X = randn(100, 10)

        # Block :a has only 1 variable
        blocks = Dict(:a => [1], :b => [2,3,4,5,6,7,8,9,10])
        @test_throws ArgumentError estimate_factors(X, 2; blocks=blocks)
    end

    @testset "Display with block names" begin
        Random.seed!(890)
        T_obs, N = 100, 10
        r = 2

        F_true = randn(T_obs, r)
        Lambda_true = zeros(N, r)
        Lambda_true[1:5, 1] = randn(5)
        Lambda_true[6:10, 2] = randn(5)
        X = F_true * Lambda_true' + 0.3 * randn(T_obs, N)

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
        Random.seed!(901)
        T_obs, N = 100, 10
        r = 2

        F_true = randn(T_obs, r)
        Lambda_true = zeros(N, r)
        Lambda_true[1:5, 1] = randn(5)
        Lambda_true[6:10, 2] = randn(5)
        X = F_true * Lambda_true' + 0.3 * randn(T_obs, N)

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
        Random.seed!(12)
        T_obs, N = 100, 8
        r = 2

        F_true = randn(T_obs, r)
        Lambda_true = zeros(N, r)
        Lambda_true[1:4, 1] = randn(4)
        Lambda_true[5:8, 2] = randn(4)
        X = F_true * Lambda_true' + 0.3 * randn(T_obs, N)

        blocks = Dict(:a => [1,2,3,4], :b => [5,6,7,8])
        fm = estimate_factors(X, r; blocks=blocks, standardize=false)

        @test fm.standardized == false
        @test fm.block_names !== nothing
        @test size(fm.factors) == (T_obs, r)
    end

    @testset "Float32 type stability" begin
        Random.seed!(23)
        T_obs, N = 100, 8
        r = 2

        X32 = randn(Float32, T_obs, N)
        blocks = Dict(:a => [1,2,3,4], :b => [5,6,7,8])
        fm = estimate_factors(X32, r; blocks=blocks)

        @test fm isa FactorModel{Float32}
        @test eltype(fm.factors) == Float32
        @test eltype(fm.loadings) == Float32
    end

    @testset "Partial coverage — not all variables assigned" begin
        Random.seed!(34)
        T_obs, N = 100, 10
        r = 2

        X = randn(T_obs, N)
        # Only 8 of 10 variables are assigned to blocks
        blocks = Dict(:a => [1,2,3,4], :b => [5,6,7,8])
        fm = estimate_factors(X, r; blocks=blocks)

        # Variables 9-10 should have zero loadings on all factors
        @test all(fm.loadings[9, :] .== 0.0)
        @test all(fm.loadings[10, :] .== 0.0)
    end

end
