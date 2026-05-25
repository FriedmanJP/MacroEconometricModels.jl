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

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Interpolation utilities
# ─────────────────────────────────────────────────────────────────────────────

@testset "Interpolation" begin
    x = [1.0, 2.0, 3.0, 4.0]
    y = [10.0, 20.0, 30.0, 40.0]
    @test MacroEconometricModels._linear_interp(x, y, 2.5) ≈ 25.0
    @test MacroEconometricModels._linear_interp(x, y, 1.0) ≈ 10.0
    @test MacroEconometricModels._linear_interp(x, y, 4.0) ≈ 40.0
    @test MacroEconometricModels._linear_interp(x, y, 0.5) ≈ 10.0  # flat extrapolation
    @test MacroEconometricModels._linear_interp(x, y, 5.0) ≈ 40.0  # flat extrapolation

    # Non-linear function
    x2 = [0.0, 1.0, 2.0, 3.0, 4.0]
    y2 = [0.0, 1.0, 4.0, 9.0, 16.0]
    @test MacroEconometricModels._linear_interp(x2, y2, 1.5) ≈ 2.5  # linear interp between 1 and 4
    @test MacroEconometricModels._linear_interp(x2, y2, 0.0) ≈ 0.0
    @test MacroEconometricModels._linear_interp(x2, y2, 4.0) ≈ 16.0

    # Bilinear interpolation
    x1_grid = [0.0, 1.0, 2.0]
    x2_grid = [0.0, 1.0, 2.0]
    z_mat = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
    @test MacroEconometricModels._bilinear_interp(x1_grid, x2_grid, z_mat, 0.0, 0.0) ≈ 1.0
    @test MacroEconometricModels._bilinear_interp(x1_grid, x2_grid, z_mat, 1.0, 1.0) ≈ 5.0
    @test MacroEconometricModels._bilinear_interp(x1_grid, x2_grid, z_mat, 0.5, 0.5) ≈ 3.0
    @test MacroEconometricModels._bilinear_interp(x1_grid, x2_grid, z_mat, 2.0, 2.0) ≈ 9.0
    # Flat extrapolation (clamped)
    @test MacroEconometricModels._bilinear_interp(x1_grid, x2_grid, z_mat, -1.0, 0.0) ≈ 1.0
    @test MacroEconometricModels._bilinear_interp(x1_grid, x2_grid, z_mat, 0.0, 3.0) ≈ 3.0
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 6: EGM one-asset
# ─────────────────────────────────────────────────────────────────────────────

@testset "EGM one-asset" begin
    n_a = 500
    grid = HAGrid(assets=(0.0, 200.0, n_a), income_states=3, grid_type=:linear)
    # Rouwenhorst discretizes log-income; exponentiate for level income
    inc_raw = rouwenhorst(0.966, 0.5, 3)
    e_levels = exp.(inc_raw.states)
    inc = IncomeProcess{Float64}(inc_raw.transition, e_levels, inc_raw.stationary_dist, :income)
    ip = IndividualProblem{Float64}(
        c -> log(c), c -> 1.0/c, m -> 1.0/m, 0.99,
        (a, e, prices) -> (1 + prices[:r]) * a + prices[:w] * e,
        [0.0], nothing, 1
    )
    prices = Dict(:r => 0.01, :w => 1.0)
    c_pol, a_pol = MacroEconometricModels._egm_solve(ip, grid, inc, prices; max_iter=1000, tol=1e-10)

    @test size(c_pol) == (n_a, 3)
    @test size(a_pol) == (n_a, 3)
    @test all(c_pol .> 0)
    @test all(a_pol .>= -1e-10)
    # Higher income → higher consumption at same asset level
    mid = div(n_a, 2)
    @test c_pol[mid, 3] > c_pol[mid, 1]
    # Euler equation error at interior (unconstrained) points
    r = prices[:r]
    euler_checked = 0
    for j in 1:3
        for i in 50:(n_a - 50)
            if a_pol[i, j] > 0.5
                Eu_prime = sum(inc.transition[j, jp] * (1.0 / MacroEconometricModels._linear_interp(
                    grid.grids[1], c_pol[:, jp], a_pol[i, j])) for jp in 1:3)
                euler_resid = abs(1.0 - 0.99 * (1 + r) * Eu_prime / (1.0 / c_pol[i, j]))
                @test euler_resid < 1e-3
                euler_checked += 1
            end
        end
    end
    @test euler_checked > 100  # enough interior points tested

    # Savings should be non-decreasing in assets (for a given income state)
    for j in 1:3
        for i in 2:n_a
            @test a_pol[i, j] >= a_pol[i-1, j] - 1e-10
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Two-asset nested EGM
# ─────────────────────────────────────────────────────────────────────────────

@testset "Two-asset nested EGM" begin
    grid2 = HAGrid(; liquid=(0.0, 20.0, 30), illiquid=(0.0, 50.0, 20), income_states=3)
    inc = rouwenhorst(0.966, 0.5, 3)
    ip2 = IndividualProblem{Float64}(
        c -> log(c), c -> 1.0/c, m -> 1.0/m, 0.99,
        (b, a, e, prices) -> (1 + prices[:r]) * b + prices[:w] * e,
        [0.0, 0.0], nothing, 2
    )
    prices2 = Dict(:r => 0.01, :r_b => 0.01, :r_a => 0.02, :w => 1.0)
    result = MacroEconometricModels._two_asset_egm_solve(ip2, grid2, inc, prices2;
        max_iter=200, tol=1e-6, n_deposit=10)

    @test haskey(result, :consumption)
    @test haskey(result, :liquid_savings)
    @test haskey(result, :deposit)
    @test size(result[:consumption]) == (30, 20, 3)
    @test size(result[:liquid_savings]) == (30, 20, 3)
    @test size(result[:deposit]) == (30, 20, 3)
    # Consumption should be positive
    @test all(result[:consumption] .> 0)
end

end # @testset "HA-DSGE Types"
