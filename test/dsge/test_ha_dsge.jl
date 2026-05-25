# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
using LinearAlgebra

@testset "HA-DSGE Types" begin

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: HAGrid — one-asset construction
# ─────────────────────────────────────────────────────────────────────────────

@testset "HAGrid one-asset construction" begin
    g = HAGrid(; assets=(0.0, 100.0, 200), income_states=5)

    # Dimensions
    @test g.n_dims == 1
    @test g.n_income == 5
    @test g.n_points == [200]
    @test length(g.grids) == 1
    @test length(g.grids[1]) == 200
    @test g.labels == [:assets]

    # Bounds
    @test g.bounds[1] == (0.0, 100.0)
    @test g.grids[1][1] ≈ 0.0
    @test g.grids[1][end] ≈ 100.0

    # Sorted
    @test issorted(g.grids[1])

    # Total individual states = n_asset_points × n_income
    @test g.total_individual_states == 200 * 5

    # Double exponential default: denser near zero
    # First 10% of points should cover less than 10% of the range
    idx_10pct = div(200, 10)
    range_10pct = g.grids[1][idx_10pct] - g.grids[1][1]
    total_range = g.grids[1][end] - g.grids[1][1]
    @test range_10pct / total_range < 0.10

    # Linear grid should be uniformly spaced
    g_lin = HAGrid(; assets=(0.0, 100.0, 101), income_states=3, grid_type=:linear)
    @test g_lin.grids[1] ≈ collect(range(0.0, 100.0; length=101))

    # Log grid should also be sorted and denser near zero (less concentrated than double_exp)
    g_log = HAGrid(; assets=(0.0, 100.0, 200), income_states=3, grid_type=:log)
    @test issorted(g_log.grids[1])
    @test g_log.grids[1][1] ≈ 0.0
    @test g_log.grids[1][end] ≈ 100.0
    range_10pct_log = g_log.grids[1][idx_10pct] - g_log.grids[1][1]
    @test range_10pct_log / total_range < 0.15
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: HAGrid — two-asset construction
# ─────────────────────────────────────────────────────────────────────────────

@testset "HAGrid two-asset construction" begin
    g2 = HAGrid(; liquid=(0.0, 50.0, 100), illiquid=(0.0, 200.0, 150), income_states=7)

    @test g2.n_dims == 2
    @test g2.n_income == 7
    @test g2.n_points == [100, 150]
    @test length(g2.grids) == 2
    @test length(g2.grids[1]) == 100   # liquid
    @test length(g2.grids[2]) == 150   # illiquid
    @test g2.labels == [:liquid, :illiquid]
    @test g2.bounds[1] == (0.0, 50.0)
    @test g2.bounds[2] == (0.0, 200.0)
    @test g2.total_individual_states == 100 * 150 * 7

    # Both grids sorted
    @test issorted(g2.grids[1])
    @test issorted(g2.grids[2])
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Rouwenhorst discretization
# ─────────────────────────────────────────────────────────────────────────────

@testset "Rouwenhorst discretization" begin
    inc = rouwenhorst(0.9, 0.2, 7)

    @test inc isa IncomeProcess{Float64}
    @test length(inc.states) == 7
    @test size(inc.transition) == (7, 7)
    @test length(inc.stationary_dist) == 7
    @test inc.labels == :income

    # Transition rows sum to 1
    for i in 1:7
        @test sum(inc.transition[i, :]) ≈ 1.0 atol=1e-12
    end

    # All probabilities non-negative
    @test all(inc.transition .>= 0.0)

    # Stationary distribution sums to 1
    @test sum(inc.stationary_dist) ≈ 1.0 atol=1e-10

    # Stationary distribution is eigenvector: π'P = π'
    pi_check = inc.transition' * inc.stationary_dist
    @test pi_check ≈ inc.stationary_dist atol=1e-10

    # States should be symmetric around zero
    @test inc.states[1] ≈ -inc.states[end] atol=1e-12

    # High persistence: test with rho close to 1
    inc_hp = rouwenhorst(0.99, 0.1, 5)
    for i in 1:5
        @test sum(inc_hp.transition[i, :]) ≈ 1.0 atol=1e-12
    end
    @test sum(inc_hp.stationary_dist) ≈ 1.0 atol=1e-10

    # Minimum case: n=2
    inc2 = rouwenhorst(0.5, 0.3, 2)
    @test size(inc2.transition) == (2, 2)
    @test sum(inc2.transition[1, :]) ≈ 1.0 atol=1e-12
    @test sum(inc2.transition[2, :]) ≈ 1.0 atol=1e-12
    @test sum(inc2.stationary_dist) ≈ 1.0 atol=1e-10
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Tauchen discretization
# ─────────────────────────────────────────────────────────────────────────────

@testset "Tauchen discretization" begin
    inc = tauchen(0.9, 0.2, 7)

    @test inc isa IncomeProcess{Float64}
    @test length(inc.states) == 7
    @test size(inc.transition) == (7, 7)
    @test length(inc.stationary_dist) == 7
    @test inc.labels == :income

    # Transition rows sum to 1
    for i in 1:7
        @test sum(inc.transition[i, :]) ≈ 1.0 atol=1e-12
    end

    # All probabilities non-negative
    @test all(inc.transition .>= 0.0)

    # Stationary distribution sums to 1
    @test sum(inc.stationary_dist) ≈ 1.0 atol=1e-10

    # Stationary distribution is eigenvector: π'P = π'
    pi_check = inc.transition' * inc.stationary_dist
    @test pi_check ≈ inc.stationary_dist atol=1e-10

    # States should be symmetric around zero
    @test inc.states[1] ≈ -inc.states[end] atol=1e-12

    # Custom m parameter
    inc_wide = tauchen(0.9, 0.2, 7; m=4)
    @test abs(inc_wide.states[end]) > abs(inc.states[end])
    for i in 1:7
        @test sum(inc_wide.transition[i, :]) ≈ 1.0 atol=1e-12
    end

    # Minimum case: n=2
    inc2 = tauchen(0.5, 0.3, 2)
    @test size(inc2.transition) == (2, 2)
    @test sum(inc2.stationary_dist) ≈ 1.0 atol=1e-10
end

end # @testset "HA-DSGE Types"
