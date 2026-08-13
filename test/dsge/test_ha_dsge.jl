# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
using LinearAlgebra
using SparseArrays
using Random
using Distributions

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

# Shared Huggett (1993) credit-limit −2 steady state (T209/#308): three testsets
# (the Table-1 SS loop, SSJ, and Reiter) recompute the identical cl=−2 equilibrium.
# Solve it ONCE here at the stricter (tol=5e-4) bar and reuse everywhere.
# Keep n_a=200 even under FAST — Table 1 atol=0.015 and the #234 @test_broken
# KS-SSJ items are not monotone in the grid (see HA Bayesian note below).
const _HUG_SPEC_M2 = MacroEconometricModels._huggett_example(; credit_limit=-2.0, a_max=8.0, n_a=200)
const _HUG_SS_M2 = compute_steady_state(_HUG_SPEC_M2; max_iter=FAST ? 80 : 200, tol=5e-4)

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

@testset "EGM Euler inversion through budget_fn (#235/T136)" begin
    # The interior endogenous-grid mapping must read the non-asset ("net") income
    # from ip.budget_fn, not a hardcoded `w*e`. Otherwise a nonzero `div` in
    # _hank1_budget is silently dropped and the interior consumption policy is
    # inconsistent (Euler residual ~ div/(1+r)). div=0 must be a no-op.
    grid = HAGrid(assets=(0.0, 50.0, 300), income_states=3)
    ir = rouwenhorst(0.9, 0.2, 3)
    el = exp.(ir.states); el ./= dot(ir.stationary_dist, el)
    inc = IncomeProcess{Float64}(ir.transition, el, ir.stationary_dist, :income)
    ip = IndividualProblem{Float64}(c -> log(c), c -> 1.0/c, m -> 1.0/m, 0.96,
                                    MacroEconometricModels._hank1_budget, [0.0], nothing, 1)

    function _max_euler(ip, grid, inc, prices)
        c_pol, a_pol = MacroEconometricModels._egm_solve(ip, grid, inc, prices;
                                                          max_iter=2000, tol=1e-12)
        ag = grid.grids[1]; n_a = length(ag)
        r = prices[:r]; beta = ip.beta; mx = 0.0
        for j in 1:3, i in 20:(n_a-20)
            a_pol[i, j] <= 0.5 && continue
            emu = sum(inc.transition[j, jp] /
                      MacroEconometricModels._linear_interp(ag, view(c_pol, :, jp), a_pol[i, j])
                      for jp in 1:3)
            mx = max(mx, abs(1 - beta * (1 + r) * emu * c_pol[i, j]))
        end
        return mx, c_pol
    end

    m0, c0 = _max_euler(ip, grid, inc, Dict(:r => 0.02, :w => 1.0, :div => 0.0))
    md, cd = _max_euler(ip, grid, inc, Dict(:r => 0.02, :w => 1.0, :div => 0.5))
    @test m0 < 1e-4              # div = 0: unchanged, tight Euler
    @test md < 1e-4              # div = 0.5: still tight (old code gave ~0.17)
    @test mean(cd) > mean(c0)    # the dividend is real income
end

@testset "EGM warm-start + convergence flag (#238/T139)" begin
    # _egm_solve gains an optional init_policy warm start and returns a trailing
    # convergence flag (Julia drops trailing tuple elements, so 2-tuple call sites
    # are unaffected).
    grid = HAGrid(assets=(0.0, 100.0, 200), income_states=3)
    ir = rouwenhorst(0.9, 0.2, 3); el = exp.(ir.states); el ./= dot(ir.stationary_dist, el)
    inc = IncomeProcess{Float64}(ir.transition, el, ir.stationary_dist, :income)
    ip = IndividualProblem{Float64}(c -> log(c), c -> 1.0/c, m -> 1.0/m, 0.95,
            (a, e, pr) -> (1 + pr[:r]) * a + pr[:w] * e, [0.0], nothing, 1)
    prices = Dict(:r => 0.02, :w => 1.0)

    # Convergence flag reflects convergence
    _, _, conv1 = MacroEconometricModels._egm_solve(ip, grid, inc, prices; max_iter=1, tol=1e-10)
    @test conv1 == false
    cN, _, convN = MacroEconometricModels._egm_solve(ip, grid, inc, prices; max_iter=2000, tol=1e-12)
    @test convN == true

    # 2-tuple destructuring still works (flag dropped)
    c2, a2 = MacroEconometricModels._egm_solve(ip, grid, inc, prices; max_iter=2000, tol=1e-12)
    @test size(c2) == (200, 3) && size(a2) == (200, 3)

    # Cold vs warm converge to the SAME policy; a seeded solve converges at once
    cw, _, convw = MacroEconometricModels._egm_solve(ip, grid, inc, prices;
                        max_iter=2000, tol=1e-12, init_policy=cN)
    @test convw == true
    @test maximum(abs.(cw .- cN)) < 1e-9
    _, _, conv_seed = MacroEconometricModels._egm_solve(ip, grid, inc, prices;
                        max_iter=3, tol=1e-8, init_policy=cN)
    @test conv_seed == true
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Two-asset nested EGM
# ─────────────────────────────────────────────────────────────────────────────

@testset "Two-asset nested EGM" begin
    nl, ni = FAST ? (16, 12) : (30, 20)
    grid2 = HAGrid(; liquid=(0.0, 20.0, nl), illiquid=(0.0, 50.0, ni), income_states=3)
    inc = rouwenhorst(0.966, 0.5, 3)
    ip2 = IndividualProblem{Float64}(
        c -> log(c), c -> 1.0/c, m -> 1.0/m, 0.99,
        (b, a, e, prices) -> (1 + prices[:r]) * b + prices[:w] * e,
        [0.0, 0.0], nothing, 2
    )
    prices2 = Dict(:r => 0.01, :r_b => 0.01, :r_a => 0.02, :w => 1.0)
    result = MacroEconometricModels._two_asset_egm_solve(ip2, grid2, inc, prices2;
        max_iter=FAST ? 80 : 200, tol=1e-6, n_deposit=FAST ? 6 : 10)

    @test haskey(result, :consumption)
    @test haskey(result, :liquid_savings)
    @test haskey(result, :deposit)
    @test size(result[:consumption]) == (nl, ni, 3)
    @test size(result[:liquid_savings]) == (nl, ni, 3)
    @test size(result[:deposit]) == (nl, ni, 3)
    # Consumption should be positive and finite
    @test all(result[:consumption] .> 0)
    @test all(isfinite, result[:consumption])
    @test all(isfinite, result[:deposit])
end

@testset "Two-asset nested EGM: state-dependent deposit (#232/T133)" begin
    # The rewritten solver must (a) yield a genuinely state-dependent deposit
    # d(b,a,e) that varies across the liquid index b (the old code stored a single
    # scalar deposit per (a,e)); and (b) return policies whose *liquid* Euler
    # residual is small (the old code returned near-seed policies with O(1)
    # residuals). Use a well-posed calibration β(1+r_a) < 1 (the r_a=0.02/β=0.99
    # standard case above is explosive for the illiquid asset).
    beta = 0.95
    grid2 = HAGrid(; liquid=(0.0, 20.0, 30), illiquid=(0.0, 50.0, 20), income_states=3)
    inc = rouwenhorst(0.966, 0.5, 3)
    ip2 = IndividualProblem{Float64}(
        c -> log(c), c -> 1.0/c, m -> 1.0/m, beta,
        (b, a, e, prices) -> (1 + prices[:r_b]) * b + prices[:w] * e,
        [0.0, 0.0], MacroEconometricModels._hank2_adjustment_cost, 2
    )
    prices2 = Dict(:r => 0.01, :r_b => 0.01, :r_a => 0.015, :w => 1.0)
    res = MacroEconometricModels._two_asset_egm_solve(ip2, grid2, inc, prices2;
        max_iter=FAST ? 120 : 400, tol=1e-6, n_deposit=FAST ? 6 : 10)
    c = res[:consumption]; b = res[:liquid_savings]; d = res[:deposit]
    b_grid = grid2.grids[1]; a_grid = grid2.grids[2]; e_vals = inc.states
    n_b, n_a, n_e = size(c)
    r_b = prices2[:r_b]; r_a = prices2[:r_a]

    @test res[:converged][1] == 1.0
    @test all(c .> 0)
    # Interior consumption (not the degenerate deposit-everything corner)
    @test count(>(0.1), c) / length(c) > 0.5

    # (1) deposit is NON-constant across the liquid index b for some (a,e)
    nonconst = false
    for je in 1:n_e, ia in 1:n_a
        col = @view d[:, ia, je]
        if maximum(col) - minimum(col) > 1e-6
            nonconst = true; break
        end
    end
    @test nonconst

    # (2) liquid Euler residual is small at genuinely-interior states
    resids = Float64[]
    for je in 1:n_e, ia in 3:(n_a-2), ib in 1:n_b
        bprime = b[ib, ia, je]
        bprime <= b_grid[1] + 1e-6 && continue
        bprime >= b_grid[end] - 1e-6 && continue
        c[ib, ia, je] < 1e-2 && continue
        aprime = (1 + r_a) * a_grid[ia] + d[ib, ia, je]
        cps = [MacroEconometricModels._bilinear_interp(b_grid, a_grid,
                   view(c, :, :, jep), bprime, aprime) for jep in 1:n_e]
        minimum(cps) < 1e-2 && continue
        emu = sum(inc.transition[je, jep] / cps[jep] for jep in 1:n_e)
        push!(resids, abs(1 - beta * (1 + r_b) * emu * c[ib, ia, je]))
    end
    sort!(resids)
    @test length(resids) > 100
    @test resids[cld(length(resids), 2)] < 1e-2          # median
    @test resids[cld(9 * length(resids), 10)] < 1.2e-1   # p90 (illiquid grid limited)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 8: VFI one-asset with Howard improvement
# ─────────────────────────────────────────────────────────────────────────────

@testset "VFI one-asset" begin
    na = FAST ? 40 : 80
    grid = HAGrid(assets=(0.0, 200.0, na), income_states=3)
    inc = rouwenhorst(0.966, 0.5, 3)
    ip = IndividualProblem{Float64}(
        c -> log(c), c -> 1.0/c, m -> 1.0/m, 0.99,
        (a, e, prices) -> (1 + prices[:r]) * a + prices[:w] * e,
        [0.0], nothing, 1
    )
    prices = Dict(:r => 0.01, :w => 1.0)
    V, c_pol, a_pol = MacroEconometricModels._vfi_solve(ip, grid, inc, prices;
                                                         max_iter=FAST ? 80 : 300, tol=1e-6,
                                                         howard_steps=FAST ? 8 : 20)
    @test size(V) == (na, 3)
    @test size(c_pol) == (na, 3)
    @test size(a_pol) == (na, 3)
    @test all(c_pol .> 0)
    @test all(a_pol .>= -1e-10)
    # Higher income → higher consumption
    mid = na ÷ 2
    @test c_pol[mid, 3] > c_pol[mid, 1]
    # Value function increasing in assets
    @test V[div(7 * na, 8), 2] > V[max(1, div(na, 8)), 2]
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 9: Young (2010) distribution tracking
# ─────────────────────────────────────────────────────────────────────────────

@testset "Young (2010) distribution" begin
    grid = HAGrid(assets=(0.0, 200.0, 100), income_states=3)
    inc = rouwenhorst(0.966, 0.5, 3)
    ip = IndividualProblem{Float64}(
        c -> log(c), c -> 1.0/c, m -> 1.0/m, 0.99,
        (a, e, prices) -> (1 + prices[:r]) * a + prices[:w] * e,
        [0.0], nothing, 1
    )
    prices = Dict(:r => 0.01, :w => 1.0)
    c_pol, a_pol = MacroEconometricModels._egm_solve(ip, grid, inc, prices; max_iter=300, tol=1e-8)

    Lambda = MacroEconometricModels._build_transition_matrix(a_pol, grid, inc)
    @test size(Lambda) == (300, 300)
    @test Lambda isa SparseArrays.SparseMatrixCSC
    # Columns sum to 1
    for col in 1:300
        @test sum(Lambda[:, col]) ≈ 1.0 atol=1e-10
    end

    dist, dist_conv = MacroEconometricModels._stationary_dist_young(Lambda)
    @test length(dist) == 300
    @test sum(dist) ≈ 1.0 atol=1e-10
    @test all(dist .>= 0)
    @test dist_conv == true                                              # #240/H-17, #242

    # Forward iteration preserves mass
    dist2 = MacroEconometricModels._forward_iterate(Lambda, dist)
    @test sum(dist2) ≈ 1.0 atol=1e-10
    # Stationary: forward iteration ≈ identity
    @test maximum(abs.(dist2 .- dist)) < 1e-10

    # Aggregate capital
    K = MacroEconometricModels._aggregate(dist, grid; var_index=1)
    @test K > 0
    @test isfinite(K)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 10: HA steady state — bisection on interest rate
# ─────────────────────────────────────────────────────────────────────────────

@testset "HA steady state" begin
    na, ne = FAST ? (50, 3) : (150, 5)
    grid = HAGrid(assets=(0.0, 200.0, na), income_states=ne)
    inc = rouwenhorst(0.966, 0.5, ne)
    ip = IndividualProblem{Float64}(
        c -> log(c), c -> 1.0/c, m -> 1.0/m, 0.99,
        (a, e, prices) -> (1 + prices[:r]) * a + prices[:w] * e,
        [0.0], nothing, 1
    )
    function price_fn(K, params)
        alpha = params[:alpha]; delta = params[:delta]
        Z = params[:Z]; L = params[:L]
        r = alpha * Z * K^(alpha-1) * L^(1-alpha) - delta
        w = (1-alpha) * Z * K^alpha * L^(-alpha)
        Dict(:r => r, :w => w)
    end
    params = Dict(:alpha => 0.36, :delta => 0.025, :Z => 1.0, :L => 1.0)

    ss = MacroEconometricModels._ha_steady_state(
        ip, grid, inc, price_fn, params;
        K_init=10.0, r_bounds=(-0.01, 0.04), max_iter=FAST ? 40 : 100, tol=1e-4
    )

    @test ss isa HASteadyState{Float64}
    @test ss.converged || abs(ss.excess_demand) < 1e-3
    @test ss.prices[:r] > -0.01   # above lower bisection bound
    @test ss.prices[:r] < 0.04    # below upper bisection bound
    @test ss.prices[:w] > 0
    @test sum(ss.distribution) ≈ 1.0 atol=1e-10
    @test all(ss.distribution .>= 0)
    @test ss.aggregates[:K] > 0
    @test ss.euler_error < 1e-2
    @test haskey(ss.policies, :savings)
    @test haskey(ss.policies, :consumption)

    # Value function should be zeros for EGM-based solver
    @test all(ss.value_fn .== 0.0)

    # Aggregate output should be positive
    @test ss.aggregates[:Y] > 0

    # Distribution shape
    @test size(ss.distribution) == (na, ne)

    # Policy shapes
    @test size(ss.policies[:savings]) == (na, ne)
    @test size(ss.policies[:consumption]) == (na, ne)

    # Consumption should be positive everywhere
    @test all(ss.policies[:consumption] .> 0)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 11: Euler error computation
# ─────────────────────────────────────────────────────────────────────────────

@testset "Euler error computation" begin
    grid = HAGrid(assets=(0.0, 200.0, 100), income_states=3)
    inc = rouwenhorst(0.966, 0.5, 3)
    ip = IndividualProblem{Float64}(
        c -> log(c), c -> 1.0/c, m -> 1.0/m, 0.99,
        (a, e, prices) -> (1 + prices[:r]) * a + prices[:w] * e,
        [0.0], nothing, 1
    )
    prices = Dict(:r => 0.01, :w => 1.0)
    c_pol, a_pol = MacroEconometricModels._egm_solve(ip, grid, inc, prices; max_iter=1000, tol=1e-10)

    euler_err = MacroEconometricModels._compute_euler_error(c_pol, a_pol, ip, grid, inc, prices)

    # Euler error should be finite and in log10 units (negative = small error)
    @test isfinite(euler_err)
    # Well-converged EGM should yield small Euler errors (< ~1e-1 → log10 < -1)
    @test euler_err < -1.0
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 12: Krusell-Smith simulation
# ─────────────────────────────────────────────────────────────────────────────

@testset "Krusell-Smith simulation" begin
    grid = HAGrid(assets=(0.0, 200.0, FAST ? 40 : 80), income_states=3)
    # Normalized income (#231): positive states, E[e]=1. The raw log grid has
    # negative states → negative labor income → the household policy collapses to
    # the borrowing constraint and the KS dynamics are degenerate.
    ir = rouwenhorst(0.966, 0.5, 3)
    e_norm = exp.(ir.states); e_norm ./= dot(ir.stationary_dist, e_norm)
    inc = IncomeProcess{Float64}(ir.transition, e_norm, ir.stationary_dist, :income)
    ip = IndividualProblem{Float64}(
        c -> log(c), c -> 1.0/c, m -> 1.0/m, 0.99,
        (a, e, prices) -> (1 + prices[:r]) * a + prices[:w] * e,
        [0.0], nothing, 1
    )
    function price_fn(K, params)
        alpha = params[:alpha]; delta = params[:delta]
        r = alpha * K^(alpha-1) - delta
        w = (1-alpha) * K^alpha
        Dict(:r => r, :w => w)
    end
    params = Dict(:alpha => 0.36, :delta => 0.025, :Z => 1.0, :L => 1.0)
    ss = MacroEconometricModels._ha_steady_state(ip, grid, inc, price_fn, params;
        K_init=10.0, r_bounds=(-0.02, 0.04), max_iter=FAST ? 30 : 60, tol=1e-3)

    result = MacroEconometricModels._krusell_smith_solve(
        ss, ip, grid, inc, price_fn, params;
        T_sim=FAST ? 120 : 300, T_burn=FAST ? 25 : 50, max_outer=FAST ? 2 : 3,
        rho_z=0.95, sigma_z=0.007
    )

    @test haskey(result.plm_coefficients, :K)
    @test length(result.plm_coefficients[:K]) == 3  # z-augmented PLM: [b1, b2, b3]
    @test haskey(result.r_squared, :K)
    @test result.r_squared[:K] > 0.9  # KS typically gets R² > 0.999
    @test result.iterations <= (FAST ? 2 : 3)

    # #229/T130: the household policy — and hence the realized capital path — is a
    # genuine function of the PLM b. Perturbing b, re-solving the (a,e,K,z) policy and
    # re-simulating the cross-section must MOVE {K_t}. The old myopic solver re-solved a
    # stationary EGM at realized prices, so its path was independent of the PLM (the
    # outer loop was vacuous). This assertion fails on that old solver.
    n_z = 3; n_K = 5
    zg, zt = MacroEconometricModels._ks_build_z_grid(0.95, 0.02, n_z)
    Kss = ss.aggregates[:K]
    Kg = Kss .* exp.(collect(range(-0.4, 0.4; length=n_K)))
    css = ss.policies[:consumption]
    c0 = Array{Float64,4}(undef, size(css, 1), size(css, 2), n_K, n_z)
    for lz in 1:n_z, kK in 1:n_K
        @views c0[:, :, kK, lz] .= css
    end
    rng_ks = Random.MersenneTwister(11)
    T_s = FAST ? 60 : 150; zidx = zeros(Int, T_s); zidx[1] = 2; zc = cumsum(zt; dims=2)
    for t in 2:T_s
        u = rand(rng_ks)
        zidx[t] = clamp(searchsortedfirst(view(zc, zidx[t-1], :), u), 1, n_z)
    end
    ks_egm_iter = FAST ? 200 : 1000
    cA, _ = MacroEconometricModels._ks_egm_solve(ip, grid, inc, [0.0, 1.0, 0.0],
        zg, zt, Kg, price_fn, params; max_iter=ks_egm_iter, tol=1e-6, init_policy=c0)
    KA = MacroEconometricModels._ks_simulate(cA, ss, grid, inc, zidx, zg, Kg, price_fn, params)
    cB, _ = MacroEconometricModels._ks_egm_solve(ip, grid, inc, [0.3, 0.85, 0.05],
        zg, zt, Kg, price_fn, params; max_iter=ks_egm_iter, tol=1e-6, init_policy=c0)
    KB = MacroEconometricModels._ks_simulate(cB, ss, grid, inc, zidx, zg, Kg, price_fn, params)
    @test maximum(abs.(cA .- cB)) > 1e-4    # different PLM → materially different policy
    @test maximum(abs.(KA .- KB)) > 1e-3    # different PLM → different realized path
    @test all(isfinite, KA) && all(KA .> 0)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 13: SSJ Jacobian
# ─────────────────────────────────────────────────────────────────────────────

@testset "_egm_backward_step is the _egm_solve fixed point" begin
    # ORACLE: the kernel's own docstring — "iterating it to a fixed point
    # reproduces `_egm_solve`". So ONE step applied to the CONVERGED `_egm_solve`
    # policy must not move it. It used to, whenever the budget carried an offset:
    # the kernel hardcoded `w*e` for non-asset income while `_egm_solve` routes it
    # through `budget_fn` (#235/H-09), so `:div` was silently dropped and every SSJ
    # Jacobian for such a model was differenced around a non-steady-state point.
    spec = load_ha_example(:one_asset_hank)
    for prices in (Dict(:r => 0.01, :w => 1.0, :div => 0.30),   # was |Δc| ≈ 2.6e-1
                   Dict(:r => 0.01, :w => 1.0))                 # plain-budget control
        c, a, _ = MacroEconometricModels._egm_solve(spec.individual, spec.grid,
                                                    spec.income, prices;
                                                    max_iter=2000, tol=1e-12)
        c1, a1 = MacroEconometricModels._egm_backward_step(spec.individual, spec.grid,
                                                           spec.income, prices, c)
        @test maximum(abs, c1 .- c) < 1e-9
        @test maximum(abs, a1 .- a) < 1e-9
    end
end

@testset "SSJ Jacobian" begin
    grid = HAGrid(assets=(0.0, 200.0, FAST ? 40 : 80), income_states=3)
    inc = rouwenhorst(0.966, 0.5, 3)
    ip = IndividualProblem{Float64}(
        c -> log(c), c -> 1.0/c, m -> 1.0/m, 0.99,
        (a, e, prices) -> (1 + prices[:r]) * a + prices[:w] * e,
        [0.0], nothing, 1
    )
    function price_fn_ssj(K, params)
        alpha = params[:alpha]; delta = params[:delta]
        r = alpha * K^(alpha-1) - delta
        w = (1-alpha) * K^alpha
        Dict(:r => r, :w => w)
    end
    params = Dict(:alpha => 0.36, :delta => 0.025, :Z => 1.0, :L => 1.0)
    ss = MacroEconometricModels._ha_steady_state(ip, grid, inc, price_fn_ssj, params;
        K_init=10.0, r_bounds=(-0.02, 0.04), max_iter=FAST ? 30 : 60, tol=1e-3)

    T_h = FAST ? 12 : 30
    J = MacroEconometricModels._ssj_jacobian(ss, ip, grid, inc, :r, :K; T_horizon=T_h, dx=1e-4)
    @test size(J) == (T_h, T_h)
    @test all(isfinite.(J))
    # Contemporaneous effect should be nonzero
    @test abs(J[1,1]) > 1e-8
    # Effects should decay
    @test abs(J[T_h, 1]) < abs(J[1, 1]) + 1.0  # loose: just not exploding

    # #226/T127: the fake-news Jacobian is DENSE with anticipation — households
    # respond BEFORE an announced future price change — so J[t,s] != 0 for some
    # t < s. The old brute force zeroed the t<s block (lower-triangular Toeplitz).
    @test any(abs(J[t, s]) > 1e-10 for t in 1:T_h for s in (t+1):T_h)
    # Default isapprox rtol swallows the ~1e-7 anticipation block on a short
    # FAST horizon; the any(...) check above is the one that pins density.
    FAST || @test !isapprox(J, LowerTriangular(J))
    # Mass conservation of the one-step forward push (column-stochastic Λ, no renorm).
    prices_p = copy(ss.prices); prices_p[:r] += 1e-4
    _, a_pol_p = MacroEconometricModels._egm_solve(ip, grid, inc, prices_p;
                                                   max_iter=200, tol=1e-10)
    Lam_p = MacroEconometricModels._build_transition_matrix(a_pol_p, grid, inc)
    d_ss = vec(ss.distribution); d_ss ./= sum(d_ss)
    @test sum(Lam_p * d_ss) ≈ 1.0 atol=1e-10
    # output_var threading (#240/H-16): a consumption aggregate differs from the
    # asset aggregate (the old code ignored output_var, hardcoding asset aggregation).
    Jc = MacroEconometricModels._ssj_jacobian(ss, ip, grid, inc, :r, :C;
                                              T_horizon=T_h, dx=1e-4)
    @test size(Jc) == (T_h, T_h)
    @test all(isfinite.(Jc))
    @test !isapprox(Jc, J)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 14: Ho-Kalman realization
# ─────────────────────────────────────────────────────────────────────────────

@testset "Ho-Kalman realization" begin
    # Create a known state-space system and verify recovery
    # True system: x_{t+1} = 0.9 x_t + ε_t, y_t = x_t
    T_len = 50
    irf_seq = [reshape([0.9^t], 1, 1) for t in 0:T_len-1]

    G1, impact, C_sol, eu, eigenvalues, C_mat, D =
        MacroEconometricModels._ho_kalman(irf_seq, 1, 1, 5)

    @test size(G1, 1) == size(G1, 2)  # square
    @test size(impact, 2) == 1         # one shock
    @test length(C_sol) == size(G1, 1)
    @test eu == [1, 1]
    @test all(isfinite.(G1))
    @test all(isfinite.(impact))

    # The dominant eigenvalue should be close to 0.9
    max_eig = maximum(abs.(eigenvalues))
    @test abs(max_eig - 0.9) < 0.1

    # #227/T128: _ho_kalman now also returns the output map C (n_vars × k) and the
    # direct feed-through D = h[0]. The realization reproduces the geometric IRF:
    # h[0] = D = 0.9^0, h[k] = C·A^(k-1)·B = 0.9^k.
    @test size(C_mat) == (1, size(G1, 1))
    @test size(D) == (1, 1)
    @test D[1, 1] ≈ 1.0 atol=1e-8
    Ah = Matrix{Float64}(I, size(G1)...)   # at horizon h>=2, Ah = G1^(h-2)
    for h in 1:8
        y_h = h == 1 ? D[1, 1] : (C_mat * (Ah * impact))[1, 1]
        @test isapprox(y_h, 0.9^(h-1); atol=1e-3)
        h >= 2 && (Ah = Ah * G1)
    end
end

@testset "No silent G1 rescale; truthful determinacy (#234/T135)" begin
    # _ho_kalman derives eu from the realization's spectral radius (not a hardcoded
    # [1,1]): a decaying IRF is stable/determinate, a growing IRF is explosive.
    stable = [reshape([0.9^t], 1, 1) for t in 0:49]
    _, _, _, eu_s, eig_s, _, _ = MacroEconometricModels._ho_kalman(stable, 1, 1, 5)
    @test eu_s == [1, 1]
    @test maximum(abs.(eig_s)) < 1
    explosive = [reshape([1.3^t], 1, 1) for t in 0:20]
    _, _, _, eu_x, eig_x, _, _ = MacroEconometricModels._ho_kalman(explosive, 1, 1, 5)
    @test maximum(abs.(eig_x)) > 1
    @test eu_x == [0, 0]              # OLD code hardcoded [1,1] even when explosive

    # _reiter_warn_unstable diagnoses instead of rescaling: warns iff ρ ≥ 1, and
    # returns the TRUE spectral radius (no 0.999/ρ mutation).
    @test_logs MacroEconometricModels._reiter_warn_unstable(
        [0.8 0.0; 0.0 0.5], "stable")                                     # no logs
    rho = @test_logs (:warn,) MacroEconometricModels._reiter_warn_unstable(
        [2.0 0.0; 0.0 0.5], "explosive")
    @test rho ≈ 2.0
end

@testset "HA low-severity batch (#240/T141)" begin
    # H-16: the SSJ Jacobian threads output_var via _ssj_outcome_vector (the old
    # code hardcoded asset aggregation; the bug was latent). Consumption vs
    # savings outputs route to the right policy.
    cpol = Float64[1 4; 2 5; 3 6]; apol = Float64[11 14; 12 15; 13 16]
    @test MacroEconometricModels._ssj_outcome_vector(:C, cpol, apol) == vec(cpol)
    @test MacroEconometricModels._ssj_outcome_vector(:K, cpol, apol) == vec(apol)
    @test MacroEconometricModels._ssj_outcome_vector(:A, cpol, apol) == vec(apol)

    # H-18: _ha_steady_state verifies the r-interval brackets a clearing rate
    # (excess demand K_s − K_d must change sign) instead of returning a spurious
    # midpoint.
    grid = HAGrid(assets=(0.0, 200.0, 40), income_states=3)
    inc = rouwenhorst(0.966, 0.5, 3)
    ip = IndividualProblem{Float64}(c -> log(c), c -> 1.0/c, m -> 1.0/m, 0.99,
            (a, e, pr) -> (1 + pr[:r]) * a + pr[:w] * e, [0.0], nothing, 1)
    pf(K, p) = Dict(:r => p[:alpha]*K^(p[:alpha]-1) - p[:delta],
                    :w => (1-p[:alpha])*K^p[:alpha])
    params = Dict(:alpha => 0.36, :delta => 0.025, :Z => 1.0, :L => 1.0)
    _hass = MacroEconometricModels._ha_steady_state
    # valid bracket → converges
    ss_valid = _hass(ip, grid, inc, pf, params; K_init=10.0, r_bounds=(-0.02, 0.04),
                     max_iter=80, tol=1e-4)
    @test ss_valid isa MacroEconometricModels.HASteadyState
    @test abs(ss_valid.excess_demand) < 1e-3
    # offset interval (both rates above the equilibrium ⇒ excess > 0 at both):
    # the solver WIDENS down to the TRUE clearing rate rather than returning a
    # spurious midpoint of (0.03, 0.05).
    ss_off = _hass(ip, grid, inc, pf, params; K_init=10.0, r_bounds=(0.03, 0.05),
                   max_iter=80, tol=1e-4)
    @test abs(ss_off.excess_demand) < 1e-3
    @test isapprox(ss_off.prices[:r], ss_valid.prices[:r]; atol=1e-3)   # same root, not 0.04
    # non-finite K_d at r_lo (r + δ < 0) is guarded, not thrown
    @test _hass(ip, grid, inc, pf, params; K_init=10.0, r_bounds=(-0.03, 0.04),
                max_iter=80, tol=1e-4) isa MacroEconometricModels.HASteadyState
end

@testset "Stationary distribution single-solve (#242/T143)" begin
    # `_stationary_dist_young` now solves the RIGHT eigenvector of the
    # column-stochastic Λ in ONE sparse LU solve instead of power iteration. It
    # must equal the power-iteration output (the wrong (I−Λ')g=0 transpose would
    # instead give the LEFT eigenvector = uniform, which power iteration rejects).
    grid = HAGrid(assets=(0.0, 200.0, 120), income_states=5)
    inc = rouwenhorst(0.966, 0.5, 5)
    ip = IndividualProblem{Float64}(c -> log(c), c -> 1.0/c, m -> 1.0/m, 0.99,
            (a, e, pr) -> (1 + pr[:r]) * a + pr[:w] * e, [0.0], nothing, 1)
    _, a_pol, _ = MacroEconometricModels._egm_solve(ip, grid, inc,
            Dict(:r => 0.01, :w => 1.0); max_iter=1000, tol=1e-12)
    Lambda = MacroEconometricModels._build_transition_matrix(a_pol, grid, inc)

    g, conv = MacroEconometricModels._stationary_dist_young(Lambda)
    @test conv == true
    @test sum(g) ≈ 1.0 atol=1e-12
    @test all(g .>= 0)
    @test maximum(abs.(Lambda * g .- g)) < 1e-10        # stationary: Λg = g

    # matches an independent power-iteration reference (not the uniform vector)
    d = fill(1.0 / length(g), length(g))
    for _ in 1:200_000
        dn = Lambda * d; dn ./= sum(dn)
        maximum(abs.(dn .- d)) < 1e-14 && (d = dn; break)
        d = dn
    end
    @test maximum(abs.(g .- d)) < 1e-8
    @test maximum(abs.(g .- fill(1.0/length(g), length(g)))) > 1e-3   # NOT uniform
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 14b: HA observation map (#227)
# ─────────────────────────────────────────────────────────────────────────────

@testset "HA observation map irf/fevd/simulate (#227)" begin
    # An SSJ solution carries the Ho-Kalman observation map C_obs/D_obs, so
    # irf/fevd/simulate report the AGGREGATE output (rate r for Huggett), not the
    # abstract reduced state x_1..x_n the old delegating code returned.
    spec = MacroEconometricModels._huggett_example(; credit_limit=-2.0, a_max=8.0, n_a=120)
    ss = compute_steady_state(spec; max_iter=100, tol=1e-3)
    sol = solve(spec; method=:ssj, ss=ss, T_horizon=80, n_reduced=15)

    n_red = size(sol.linear_solution.G1, 1)
    @test size(sol.C_obs) == (1, n_red)
    @test size(sol.D_obs) == (1, 1)
    @test n_red > 1                                  # abstract state is multi-dimensional

    ir = irf(sol, 20)
    @test size(ir.values) == (20, 1, 1)              # ONE aggregate (r), NOT n_red states
    @test ir.variables == ["r"]

    B = sol.linear_solution.impact; G1 = sol.linear_solution.G1; C = sol.C_obs
    # Reproduces the realized rate IRF: impact = D_obs = h[0]; h>=2 = C·A^(h-2)·B.
    @test ir.values[1, 1, 1] ≈ sol.D_obs[1, 1] atol=1e-10
    @test ir.values[2, 1, 1] ≈ (C * B)[1, 1] atol=1e-10
    @test ir.values[3, 1, 1] ≈ (C * G1 * B)[1, 1] atol=1e-10

    fv = fevd(sol, 20)
    @test length(fv.variables) == 1
    @test all(isfinite.(fv.decomposition))

    # simulate reports the aggregate deviation path; a unit impulse gives D_obs.
    sim = simulate(sol, 30; shock_draws=reshape([1.0; zeros(29)], 30, 1))
    @test size(sim) == (30, 1)
    @test sim[1, 1] ≈ sol.D_obs[1, 1] atol=1e-10
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 15: Reiter linearization
# ─────────────────────────────────────────────────────────────────────────────

@testset "Reiter linearization" begin
    nred = FAST ? 10 : 15
    grid = HAGrid(assets=(0.0, 200.0, FAST ? 30 : 50), income_states=3)
    inc = rouwenhorst(0.966, 0.5, 3)
    ip = IndividualProblem{Float64}(
        c -> log(c), c -> 1.0/c, m -> 1.0/m, 0.99,
        (a, e, prices) -> (1 + prices[:r]) * a + prices[:w] * e,
        [0.0], nothing, 1
    )
    function price_fn_reiter(K, params)
        alpha = params[:alpha]; delta = params[:delta]
        r = alpha * K^(alpha-1) - delta
        w = (1-alpha) * K^alpha
        Dict(:r => r, :w => w)
    end
    params = Dict(:alpha => 0.36, :delta => 0.025, :Z => 1.0, :L => 1.0, :rho_z => 0.95)
    ss = MacroEconometricModels._ha_steady_state(ip, grid, inc, price_fn_reiter, params;
        K_init=10.0, r_bounds=(-0.02, 0.04), max_iter=60, tol=1e-3)

    G1, impact, n_red, explained = MacroEconometricModels._reiter_linearize(
        ss, ip, grid, inc; n_reduced=nred, model=:aiyagari, het_params=params
    )
    @test size(G1, 1) == size(G1, 2)  # square
    @test size(G1, 1) <= nred + 5  # reduced dim + aggregates
    @test n_red <= nred
    @test explained > 0.95
    @test maximum(abs.(eigvals(G1))) < 1.0 + 0.01  # approximately stable
    @test size(impact, 1) == size(G1, 1)

    # #230/T131: Aiyagari GE price feedback. The K state column (n_red+1) must be
    # populated — capital feeds back into the distribution via the firm-FOC price
    # channel. The old code left G1[:, n_red+1] identically zero (r never responded).
    @test any(!iszero, G1[1:n_red, n_red + 1])

    # Firm-FOC signs: a higher predetermined K lowers r and raises w.
    dr_dK, dw_dK, dr_dZ, dw_dZ = MacroEconometricModels._aiyagari_foc_derivatives(
        ss.prices[:r], ss.prices[:w], ss.aggregates[:K],
        params[:alpha], params[:delta], params[:Z])
    @test dr_dK < 0
    @test dw_dK > 0
    @test dr_dZ > 0
    @test dw_dZ > 0

    # #236/T137: alpha/delta/rho_z are read from the spec, not hardcoded literals.
    # Varying rho_z changes G1 (the TFP AR(1) diagonal), proving it is read.
    params_lo = merge(params, Dict(:rho_z => 0.80))
    G1b, _, _, _ = MacroEconometricModels._reiter_linearize(
        ss, ip, grid, inc; n_reduced=nred, model=:aiyagari, het_params=params_lo)
    @test !(G1 ≈ G1b)
    # A missing required parameter errors informatively (no magic-number default).
    @test_throws ErrorException MacroEconometricModels._reiter_linearize(
        ss, ip, grid, inc; n_reduced=nred, model=:aiyagari,
        het_params=Dict(:alpha => 0.36, :delta => 0.025, :Z => 1.0))   # no :rho_z
    @test_throws ErrorException MacroEconometricModels._reiter_linearize(
        ss, ip, grid, inc; n_reduced=nred, model=:aiyagari,
        het_params=Dict(:delta => 0.025, :rho_z => 0.95, :Z => 1.0))   # no :alpha
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 16: Display
# ─────────────────────────────────────────────────────────────────────────────

@testset "Display" begin
    grid = HAGrid(assets=(0.0, 200.0, 100), income_states=3)
    inc = rouwenhorst(0.966, 0.5, 3)
    ip = IndividualProblem{Float64}(
        c -> log(c), c -> 1.0/c, m -> 1.0/m, 0.99,
        (a, e, prices) -> (1 + prices[:r]) * a + prices[:w] * e,
        [0.0], nothing, 1
    )
    function price_fn(K, params)
        r = 0.36 * K^(0.36-1) - 0.025; w = 0.64 * K^0.36
        Dict(:r => r, :w => w)
    end
    params = Dict(:alpha => 0.36, :delta => 0.025, :Z => 1.0, :L => 1.0)
    ss = MacroEconometricModels._ha_steady_state(ip, grid, inc, price_fn, params;
        K_init=10.0, r_bounds=(-0.02, 0.04), max_iter=60, tol=1e-3)

    # show doesn't error
    io = IOBuffer()
    show(io, ss)
    s = String(take!(io))
    @test contains(s, "HASteadyState")

    # report doesn't error
    report(ss)

    # Gini coefficient
    gini = MacroEconometricModels._gini_coefficient(vec(ss.distribution), ss.grid)
    @test 0.0 <= gini <= 1.0
    @test isfinite(gini)

    # Wealth percentiles
    p50 = MacroEconometricModels._wealth_percentile(vec(ss.distribution), ss.grid, 0.5)
    p90 = MacroEconometricModels._wealth_percentile(vec(ss.distribution), ss.grid, 0.9)
    @test p90 >= p50  # 90th percentile >= median
    @test isfinite(p50)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 17: Analysis functions
# ─────────────────────────────────────────────────────────────────────────────

@testset "Analysis functions" begin
    grid = HAGrid(assets=(0.0, 200.0, 80), income_states=3)
    inc = rouwenhorst(0.966, 0.5, 3)
    ip = IndividualProblem{Float64}(
        c -> log(c), c -> 1.0/c, m -> 1.0/m, 0.99,
        (a, e, prices) -> (1 + prices[:r]) * a + prices[:w] * e,
        [0.0], nothing, 1
    )
    function price_fn_analysis(K, params)
        r = 0.36 * K^(0.36-1) - 0.025; w = 0.64 * K^0.36
        Dict(:r => r, :w => w)
    end
    params = Dict(:alpha => 0.36, :delta => 0.025, :Z => 1.0, :L => 1.0)
    ss = MacroEconometricModels._ha_steady_state(ip, grid, inc, price_fn_analysis, params;
        K_init=10.0, r_bounds=(-0.02, 0.04), max_iter=60, tol=1e-3)

    # simulate_panel
    panel = MacroEconometricModels.simulate_panel(ss; N_agents=100, T_periods=50,
        rng=Random.MersenneTwister(42))
    @test size(panel) == (100, 50)
    @test all(panel .>= 0)
    @test all(isfinite.(panel))
    # Mean asset holdings should be in a reasonable range
    mean_assets = sum(panel[:, end]) / 100
    @test mean_assets > 0

    # inequality_irf (using steady state directly — simplified)
    ineq = MacroEconometricModels.inequality_irf(ss; T_periods=20)
    @test haskey(ineq, :gini)
    @test haskey(ineq, :p50)
    @test haskey(ineq, :p90)
    @test length(ineq[:gini]) == 20
    @test all(0 .<= ineq[:gini] .<= 1)
    # At steady state, all periods should be identical
    @test all(ineq[:gini] .≈ ineq[:gini][1])
    @test all(ineq[:p50] .≈ ineq[:p50][1])
    @test all(ineq[:p90] .≈ ineq[:p90][1])
    # Percentile ordering
    @test ineq[:p90][1] >= ineq[:p50][1]
    @test ineq[:p50][1] >= ineq[:p10][1]
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 17b: distribution/inequality IRF reduction basis (#233)
# ─────────────────────────────────────────────────────────────────────────────

@testset "distribution IRF reduction basis (#233)" begin
    spec = load_ha_example(:krusell_smith)
    ss = compute_steady_state(spec; r_bounds=(-0.02, 0.04), max_iter=60, tol=1e-3)
    n_a = spec.grid.n_points[1]; n_e = spec.grid.n_income
    reiter_sol = solve(spec; method=:reiter, ss=ss, n_reduced=12)

    # (c) reduction_basis is the REAL U_k: N × n_red with orthonormal columns
    # (was Matrix{T}(I, n_red, n_red), whose row count never equalled n_a·n_e).
    U = reiter_sol.reduction_basis
    @test size(U, 1) == n_a * n_e
    @test size(U, 2) == reiter_sol.n_reduced
    @test U' * U ≈ Matrix{Float64}(I, reiter_sol.n_reduced, reiter_sol.n_reduced) atol=1e-8

    # (a) distribution IRF has nonzero entries after a shock (was identically zero
    # because the projection guard n_full == n_a·n_e always failed on the identity).
    dirf = distribution_irf(reiter_sol, 10)
    @test size(dirf) == (n_a, n_e, 10)
    @test any(!iszero, dirf)

    # (e) inequality IRF: finite Gini in [0,1], p90 >= p50
    ineq = inequality_irf(reiter_sol, 10)
    @test all(0 .<= ineq[:gini] .<= 1)
    @test all(isfinite, ineq[:gini])
    @test all(ineq[:p90] .>= ineq[:p50])

    # (d) SSJ/Ho-Kalman has no distribution basis → both throw informatively.
    ssj_sol = solve(spec; method=:ssj, ss=ss, T_horizon=40, n_reduced=12)
    @test_throws ErrorException distribution_irf(ssj_sol, 10)
    @test_throws ErrorException inequality_irf(ssj_sol, 10)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 18: Built-in examples
# ─────────────────────────────────────────────────────────────────────────────

@testset "Built-in examples" begin
    @testset "Krusell-Smith" begin
        spec = load_ha_example(:krusell_smith)
        @test spec isa HADSGESpec{Float64}
        @test spec.grid.n_dims == 1
        @test spec.grid.n_income == 7
        @test spec.individual.beta ≈ 0.99
        @test length(spec.income.states) == 7
        @test spec.individual.borrowing_constraint[1] ≈ 0.0
        @test spec.grid.n_points == [200]
        @test spec.grid.bounds[1] == (0.0, 1000.0)
        @test spec.het_params[:alpha] ≈ 0.36
        @test spec.het_params[:delta] ≈ 0.025
        @test spec.n_assets == 1
        @test spec.n_income == 7
        # Aggregate spec is a valid DSGESpec
        @test spec.aggregate_spec isa DSGESpec{Float64}
        @test :Y in spec.aggregate_spec.endog
        @test :K in spec.aggregate_spec.endog
    end

    @testset "One-asset HANK" begin
        spec = load_ha_example(:one_asset_hank)
        @test spec isa HADSGESpec{Float64}
        @test spec.grid.n_dims == 1
        @test spec.individual.borrowing_constraint[1] ≈ -2.0
        @test spec.individual.beta ≈ 0.986
        @test spec.grid.bounds[1][1] ≈ -2.0
        @test spec.grid.bounds[1][2] ≈ 1000.0
        @test spec.grid.n_points == [200]
        @test spec.grid.n_income == 7
        @test spec.het_params[:sigma_c] ≈ 1.0
        @test spec.n_assets == 1
    end

    @testset "Two-asset HANK" begin
        spec = load_ha_example(:two_asset_hank)
        @test spec isa HADSGESpec{Float64}
        @test spec.grid.n_dims == 2
        @test spec.individual.adjustment_cost !== nothing
        @test spec.individual.n_asset_dims == 2
        @test spec.individual.borrowing_constraint[1] ≈ -2.0
        @test spec.individual.borrowing_constraint[2] ≈ 0.0
        @test spec.grid.labels == [:liquid, :illiquid]
        @test spec.grid.n_points == [50, 50]
        @test spec.grid.bounds[1] == (-2.0, 50.0)
        @test spec.grid.bounds[2] == (0.0, 100.0)
        @test spec.n_assets == 2
        @test spec.n_income == 7
        # Adjustment cost should return a positive value for nonzero deposit
        chi = spec.individual.adjustment_cost(1.0, 10.0)
        @test chi > 0.0
        @test isfinite(chi)
    end

    @testset "Invalid example" begin
        @test_throws ErrorException load_ha_example(:nonexistent)
    end

    @testset "Income normalization (#231/T132)" begin
        # All four examples must ship a strictly positive income multiplier e
        # (the raw log grid gives half the states negative labor income).
        for name in (:krusell_smith, :one_asset_hank, :two_asset_hank, :huggett)
            spec = load_ha_example(name)
            @test all(spec.income.states .> 0)
        end

        # The three Rouwenhorst examples must have unit-mean income E[e] = 1.
        for name in (:krusell_smith, :one_asset_hank, :two_asset_hank)
            spec = load_ha_example(name)
            @test dot(spec.income.stationary_dist, spec.income.states) ≈ 1.0 atol=1e-10
        end

        # Huggett keeps its bespoke {1.0, 0.1} endowment (mean ≈ 0.8826), NOT normalized.
        spec_h = load_ha_example(:huggett)
        @test dot(spec_h.income.stationary_dist, spec_h.income.states) ≈ 0.8826 atol=1e-3

        # rouwenhorst/tauchen direct calls must still return the symmetric log grid.
        inc = rouwenhorst(0.966, 0.5, 7)
        @test inc.states[1] ≈ -inc.states[end] atol=1e-12
    end

    @testset "Income dispersion units" begin
        # ORACLE (analytic): Rouwenhorst's stationary law is Binomial(n-1, 1/2) on an
        # equispaced grid of half-width ψ = √(n-1)·σ_y, so the state sd is EXACTLY σ_y
        # and the first autocorrelation is EXACTLY ρ. The exp/E[exp] normalization is a
        # pure location shift in logs, so sd(log e) survives it unchanged.
        #
        # This is the detector for the units bug the examples shipped with: `sigma` is
        # the AR(1) INNOVATION sd, but the literal 0.5 in the calibration is the
        # UNCONDITIONAL sd (the Python sequence-jacobian convention). Read the wrong
        # way it gives sd(log e) = 0.5/√(1-0.966²) = 1.9339 — 3.87× too dispersed in
        # logs, 15× in variance — which pinned 5.6% of KS mass on the grid ceiling.
        for name in (:krusell_smith, :one_asset_hank, :two_asset_hank)
            spec = load_ha_example(name)
            p = spec.income.stationary_dist
            z = log.(spec.income.states)
            mu = dot(p, z)
            var_z = dot(p, (z .- mu) .^ 2)
            @test sqrt(var_z) ≈ 0.5 atol=1e-8
            # first autocorrelation of the discretized chain == ρ exactly
            @test (dot(p .* z, spec.income.transition * z) - mu^2) / var_z ≈ 0.966 atol=1e-10
            # top/bottom ratio = exp(2ψ) with ψ = √6 · 0.5
            @test maximum(spec.income.states) / minimum(spec.income.states) ≈
                  exp(2 * sqrt(6) * 0.5) rtol=1e-10
        end
    end

    @testset "sigma_is convention (rouwenhorst / tauchen)" begin
        # Default must be bitwise unchanged from the pre-`sigma_is` implementation.
        for f in (rouwenhorst, tauchen)
            a = f(0.966, 0.5, 7)
            b = f(0.966, 0.5, 7; sigma_is=:innovation)
            @test a.states == b.states
            @test a.transition == b.transition
            @test_throws ArgumentError f(0.9, 0.2, 5; sigma_is=:bogus)
        end

        # :innovation ⇒ half-width √(n-1)·σ/√(1-ρ²);  :unconditional ⇒ √(n-1)·σ.
        @test rouwenhorst(0.966, 0.5, 7).states[end] ≈
              sqrt(6) * 0.5 / sqrt(1 - 0.966^2) atol=1e-12
        @test rouwenhorst(0.9, 0.2, 7; sigma_is=:unconditional).states[end] ≈
              sqrt(6) * 0.2 atol=1e-12
        @test tauchen(0.9, 0.2, 7; sigma_is=:unconditional).states[end] ≈
              3 * 0.2 atol=1e-12

        # The two conventions must agree after the σ_y ↔ σ_ε change of variables.
        # For `tauchen` this also pins the transition matrix, whose CDF is scaled by
        # the INNOVATION sd — passing sd(y) there would silently mis-scale it.
        for (rho, sig) in ((0.9, 0.2), (0.966, 0.13), (0.5, 0.4))
            inn = tauchen(rho, sig, 7)
            unc = tauchen(rho, sig / sqrt(1 - rho^2), 7; sigma_is=:unconditional)
            @test maximum(abs, unc.states .- inn.states) < 1e-12
            @test maximum(abs, unc.transition .- inn.transition) < 1e-12
        end
    end

    @testset ":geometric asset grid" begin
        # ORACLE (analytic): a pivot-geometric grid is equidistant in log(a + piv),
        # so every consecutive ratio equals q = ((a_max+piv)/(a_min+piv))^(1/(n-1)).
        for (lo, hi, n) in ((0.0, 1000.0, 200), (-2.0, 1000.0, 200), (0.0, 200.0, 50))
            g = MacroEconometricModels._make_asset_grid(lo, hi, n, :geometric)
            piv = abs(lo) + 0.25
            q = ((hi + piv) / (lo + piv))^(1 / (n - 1))
            @test all(i -> isapprox((g[i+1] + piv) / (g[i] + piv), q; rtol=1e-12), 1:n-1)
            @test g[1] == lo && g[end] == hi
            @test all(diff(g) .> 0)
        end
        @test_throws ArgumentError MacroEconometricModels._make_asset_grid(
            0.0, 10.0, 5, :nonexistent)

        # The property the shape buys, and the reason the examples use it: raising
        # a_max 5× costs `:double_exp` exactly 5× the bottom spacing (it is a fixed
        # curve rescaled by (a_max - a_min)), but costs `:geometric` only ~1.25×.
        de200 = MacroEconometricModels._make_asset_grid(0.0, 200.0, 200, :double_exp)
        de1000 = MacroEconometricModels._make_asset_grid(0.0, 1000.0, 200, :double_exp)
        gm200 = MacroEconometricModels._make_asset_grid(0.0, 200.0, 200, :geometric)
        gm1000 = MacroEconometricModels._make_asset_grid(0.0, 1000.0, 200, :geometric)
        @test (de1000[2] - de1000[1]) ≈ 5 * (de200[2] - de200[1]) rtol=1e-12
        @test (gm1000[2] - gm1000[1]) < 1.5 * (gm200[2] - gm200[1])
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 18b: Asset-grid adequacy / asset-market clearing
# ─────────────────────────────────────────────────────────────────────────────

@testset "Grid adequacy" begin
    @testset "shipped examples clear the asset market" begin
        # An HA steady state can be EXACTLY stationary and still fail to clear:
        # `_build_transition_matrix` clamps the savings policy at a_max, which
        # conserves mass but destroys assets, and `excess_demand` cannot see it
        # because it is measured on the already-clamped aggregate ∫a dμ.
        #
        # ORACLE: for a stationary Young histogram whose policy never leaves the
        # grid, ∫a'dμ == ∫a dμ is FORCED — stationarity says next period's
        # holdings integrate to ∫a dμ, and with no clamping those holdings ARE a'.
        names = FAST ? (:huggett,) : (:krusell_smith, :one_asset_hank, :huggett)
        for name in names
            ss = compute_steady_state(load_ha_example(name))
            d = ha_grid_diagnostics(ss)
            scale = max(1.0, abs(ss.aggregates[:K]))
            @test d.adequate
            @test d.ceiling_mass < 1e-10
            @test d.n_cells_below == 0
            @test abs(d.clearing_residual) < 1e-8 * scale
            @test abs(ss.aggregates[:A_policy] - ss.aggregates[:K]) < 1e-8 * scale
            @test abs(ss.excess_demand) < 1e-6 * scale
            # NB not `maximum(a_pol) < a_grid[end]`: at a_max = 1000 exactly one
            # cell still overshoots, but it carries zero mass. The measure-zero
            # form is what the theorem requires; the strict-max form is spurious.
        end
    end

    FAST || @testset "steady state is invariant to the bisection bracket" begin
        # ORACLE: the market-clearing rate is a property of the model, not of the
        # interval it is searched over. Any bracket containing the root must give
        # the same answer.
        #
        # It did not. A trial rate with β(1+r) ≥ 1 has no stationary distribution:
        # wealth diverges and every household saves to a_max, so the computed K_s
        # is a pure grid artifact — and that divergent policy was then reused as
        # the EGM warm start (#238), corrupting the excess-demand values at the
        # NEXT, admissible rates and collapsing the bracket on a spurious sign
        # change. On one-asset HANK, r_bounds=(-0.02, 0.04) returned r = 0.010469
        # with excess = -13.2 (converged=false) while (-0.01, 0.04) returned the
        # true r = 0.011523. Raising a_max amplified it: the artifact K_s is a_max.
        spec = load_ha_example(:one_asset_hank)
        beta = spec.individual.beta
        sss = [compute_steady_state(spec; r_bounds=rb)
               for rb in ((-0.01, 0.04), (-0.02, 0.04), (-0.005, 0.05))]
        rs = [ss.prices[:r] for ss in sss]
        Ks = [ss.aggregates[:K] for ss in sss]
        @test maximum(rs) - minimum(rs) < 1e-6
        @test (maximum(Ks) - minimum(Ks)) / minimum(Ks) < 1e-4
        for ss in sss
            @test ss.converged
            @test abs(ss.excess_demand) < 1e-6 * max(1.0, ss.aggregates[:K])
            # ...and the equilibrium must satisfy Aiyagari's existence condition,
            # which is what rules out the divergent branch in the first place.
            @test beta * (1 + ss.prices[:r]) < 1
        end
    end

    @testset "truncation identity (exact)" begin
        # ORACLE: ∫a'dμ − ∫a dμ == ∫max(a'−a_max,0)dμ − ∫max(a_min−a',0)dμ for ANY
        # stationary histogram. Tested on synthetic policies, so no solver is
        # involved and the identity is checked in isolation.
        base = load_ha_example(:krusell_smith)
        ev = base.income.states
        for (amax, n, gt) in ((5.0, 60, :geometric), (20.0, 80, :geometric),
                              (200.0, 200, :double_exp), (50.0, 100, :linear))
            g = HAGrid(; assets=(0.0, amax, n), income_states=7, grid_type=gt)
            ag = g.grids[1]
            # affine policy with fixed point a*_j = 1.5·a_max·e_j ⇒ states truncate
            # differentially (e spans ≈0.3–3.0)
            apol = [max(0.0, 0.15 * amax * ev[j] + 0.9 * ag[i]) for i in 1:n, j in 1:7]
            L = MacroEconometricModels._build_transition_matrix(apol, g, base.income)
            dist, _ = MacroEconometricModels._stationary_dist_young(L; max_iter=100_000,
                                                                    tol=1e-14)
            d = MacroEconometricModels._ha_grid_diagnostics(apol, dist, g)
            @test maximum(abs.(L * dist .- dist)) < 1e-12        # stationary
            @test d.clearing_residual ≈ d.truncation_flux_up - d.truncation_flux_down atol =
                  1e-9 * max(1.0, abs(d.assets_held))
            @test d.n_cells_above > 0
            @test !d.adequate
        end
    end

    FAST || @testset "detects the historical Krusell-Smith defect" begin
        # Rebuild EXACTLY the pre-fix KS spec: sigma = 0.5 read as the INNOVATION
        # sd, on the old [0, 200] :double_exp grid. The diagnostics must catch what
        # `excess_demand` could not.
        base = load_ha_example(:krusell_smith)
        raw = rouwenhorst(0.966, 0.5, 7)                       # old (buggy) convention
        e = exp.(raw.states); e ./= dot(raw.stationary_dist, e)
        old_inc = IncomeProcess{Float64}(raw.transition, e, raw.stationary_dist, :income)
        old = HADSGESpec{Float64}(base.aggregate_spec, base.individual, old_inc,
                                  HAGrid(; assets=(0.0, 200.0, 200), income_states=7),
                                  base.aggregation, base.het_params; model=base.model)
        ss_old = compute_steady_state(old; grid_check=:none)
        d = ha_grid_diagnostics(ss_old)
        @test !d.adequate
        @test d.ceiling_mass > 0.05              # measured 5.574343%
        @test d.relative_residual > 0.015        # measured 1.6843%
        @test abs(ss_old.excess_demand) < 1e-6   # ...and excess_demand is blind to it
        @test d.clearing_residual ≈ d.truncation_flux_up - d.truncation_flux_down atol=1e-9

        logs, _ = Test.collect_test_logs() do
            compute_steady_state(old; grid_check=:warn)
        end
        @test any(l -> occursin("truncates the stationary distribution",
                                string(l.message)), logs)
        @test_throws ArgumentError compute_steady_state(old; grid_check=:error)
    end

    @testset "emitter (synthetic fixture)" begin
        # Hand-computed: a = [0,1,2,3], n_e = 2, a' = [0,1,2,5] in both income
        # states, uniform mass 1/8 ⇒ ∫a dμ = 1.5, ∫a'dμ = 2.0, flux_up = 0.5,
        # ceiling mass = 0.25, 2 cells above. Synthetic on purpose, so this test
        # never needs rewriting when a calibration moves.
        g = HAGrid(; assets=(0.0, 3.0, 4), income_states=2, grid_type=:linear)
        apol = [0.0 0.0; 1.0 1.0; 2.0 2.0; 5.0 5.0]
        d = MacroEconometricModels._ha_grid_diagnostics(apol, fill(1 / 8, 8), g)
        @test d.assets_held ≈ 1.5
        @test d.assets_desired ≈ 2.0
        @test d.truncation_flux_up ≈ 0.5
        @test d.truncation_flux_down ≈ 0.0
        @test d.ceiling_mass ≈ 0.25
        @test d.n_cells_above == 2
        @test d.n_cells_below == 0
        @test !d.adequate

        @test MacroEconometricModels._check_grid_adequacy(d, :none) === d
        @test_throws ArgumentError MacroEconometricModels._check_grid_adequacy(d, :error)
        @test_throws ArgumentError MacroEconometricModels._check_grid_adequacy(d, :bogus)
        logs, _ = Test.collect_test_logs() do
            MacroEconometricModels._check_grid_adequacy(d, :warn)
        end
        @test length(logs) == 1
        msg = string(logs[1].message)
        @test occursin("a_max", msg)                   # message must be actionable
        @test occursin("grid_type=:geometric", msg)
        @test occursin("grid_check=:none", msg)

        # Mirror case: a policy below the grid floor CREATES assets.
        g2 = HAGrid(; assets=(-2.0, 3.0, 4), income_states=2, grid_type=:linear)
        apol2 = [-5.0 -5.0; 0.0 0.0; 1.0 1.0; 2.0 2.0]
        d2 = MacroEconometricModels._ha_grid_diagnostics(apol2, fill(1 / 8, 8), g2)
        @test d2.n_cells_below == 2
        @test d2.truncation_flux_down ≈ 0.75
        @test !d2.adequate

        # Two-asset grids are refused outright.
        g3 = HAGrid(; liquid=(0.0, 5.0, 4), illiquid=(0.0, 5.0, 4), income_states=2)
        @test_throws ArgumentError MacroEconometricModels._ha_grid_diagnostics(
            apol, fill(1 / 8, 8), g3)
    end

    @testset "spec validation: borrowing constraint vs grid floor" begin
        base = load_ha_example(:krusell_smith)
        ip = base.individual
        mk(bc) = IndividualProblem{Float64}(ip.utility, ip.utility_prime,
            ip.utility_prime_inv, ip.beta, ip.budget_fn, [bc], nothing, 1)
        g = HAGrid(; assets=(0.0, 200.0, 50), income_states=7)
        # Below the floor: the Young clamp would create assets out of nothing.
        @test_throws ArgumentError HADSGESpec{Float64}(base.aggregate_spec, mk(-1.0),
            base.income, g, base.aggregation, base.het_params)
        # Above the floor: merely wasteful, so warn.
        logs, _ = Test.collect_test_logs() do
            HADSGESpec{Float64}(base.aggregate_spec, mk(5.0), base.income, g,
                                base.aggregation, base.het_params)
        end
        @test any(l -> occursin("unreachable", string(l.message)), logs)
        # All shipped examples satisfy the check.
        for name in (:krusell_smith, :one_asset_hank, :two_asset_hank, :huggett)
            @test load_ha_example(name) isa HADSGESpec{Float64}
        end
    end

    @testset "two-asset steady state raises an honest error" begin
        # docs/src/dsge_ha.md used to claim `compute_steady_state` auto-selects a
        # VFI solver for two-asset models. It does not — it used to fail with a
        # bare AssertionError on a SHIPPED example.
        @test_throws ArgumentError compute_steady_state(load_ha_example(:two_asset_hank))
    end
end

@testset "Den Haan (2010) accuracy beyond Huggett (#359/T260)" begin
    M = MacroEconometricModels

    @testset "price conventions are provably distinct" begin
        par = Dict(:alpha => 0.36, :delta => 0.025, :Z => 1.0, :L => 1.0)
        p0 = M._ks_prices(M._default_cobb_douglas_price_fn, 40.0, 0.0, par, :effective_capital)
        # z = 0 must agree across conventions
        @test p0 == M._ks_prices(M._default_cobb_douglas_price_fn, 40.0, 0.0, par, :tfp)
        z = 0.02
        pe = M._ks_prices(M._default_cobb_douglas_price_fn, 40.0, z, par, :effective_capital)
        pt = M._ks_prices(M._default_cobb_douglas_price_fn, 40.0, z, par, :tfp)
        # Effective capital: dlog(r+delta)/dz = alpha-1, dlog w/dz = alpha.
        @test log((pe[:r] + 0.025) / (p0[:r] + 0.025)) / z ≈ 0.36 - 1 rtol = 1e-10
        @test log(pe[:w] / p0[:w]) / z ≈ 0.36 rtol = 1e-10
        # TFP: both elasticities are exactly 1.
        @test log((pt[:r] + 0.025) / (p0[:r] + 0.025)) / z ≈ 1.0 rtol = 1e-10
        @test log(pt[:w] / p0[:w]) / z ≈ 1.0 rtol = 1e-10
        # ⇒ NOT related by rescaling z: r rises under TFP and falls under effective capital
        @test pt[:r] > p0[:r]
        @test pe[:r] < p0[:r]
        # the default must be the Krusell-Smith convention (unchanged behaviour)
        @test M._ks_prices(M._default_cobb_douglas_price_fn, 40.0, z, par, :effective_capital) ==
              M._default_cobb_douglas_price_fn(40.0 * exp(z), par)
    end

    @testset "DenHaanAccuracy carries its source" begin
        dh = M.DenHaanAccuracy{Float64}(:K, 0.5, 0.2, 0.01, 0.011,
                                        [1.0, 2.0], [1.0, 2.1], 100, 10)
        @test dh.source === :plm                      # 9-positional contract preserved
        dh2 = M.DenHaanAccuracy{Float64}(:K, 0.5, 0.2, 0.01, 0.011,
                                         [1.0, 2.0], [1.0, 2.1], 100, 10; source=:linear)
        @test dh2.source === :linear
    end

    @testset "Huggett is refused with an informative message" begin
        # Build the solution struct directly: the guard fires on `spec.model`, so running
        # the (expensive) Krusell-Smith PLM fixed point just to reach it wastes ~4 minutes.
        ks_h = M.KrusellSmithSolution{Float64}(
            _HUG_SS_M2, Dict(:K => [0.0, 0.95, 0.0]), Dict(:K => 0.99),
            _HUG_SPEC_M2, true, 1)
        err = try
            den_haan_test(ks_h); nothing
        catch e
            sprint(showerror, e)
        end
        @test err !== nothing
        @test occursin("aiyagari", err)
        @test occursin("distribution-augmented", err)
    end
end

@testset "Adaptive distribution grid (#357/T258)" begin
    M = MacroEconometricModels

    # A Gaussian density on [0, 40] with an analytically known second derivative, sampled
    # finely enough that the discretization is not the binding error.
    a_lo, a_hi, μ0, s0 = 0.0, 40.0, 12.0, 2.0
    xf = collect(range(a_lo, a_hi; length=2001))
    pf = @. exp(-0.5 * ((xf - μ0) / s0)^2) / (s0 * sqrt(2π))
    d2f = @. pf * (((xf - μ0)^2 - s0^2) / s0^4)

    @testset "curvature=0 returns an exactly uniform grid" begin
        x0 = M._make_asset_grid(0.0, 100.0, 60, :double_exp)
        mass = exp.(-((x0 .- 20.0) ./ 3.0) .^ 2); mass ./= sum(mass)
        g0 = adaptive_asset_grid(x0, mass; curvature=0.0)
        @test length(g0) == 60
        @test g0 ≈ collect(range(0.0, 100.0; length=60)) atol = 1e-10
        # ... on any density at all, since the monitor no longer sees it
        g1 = adaptive_asset_grid(xf, pf; curvature=0.0, is_density=true, n=25)
        @test g1 ≈ collect(range(a_lo, a_hi; length=25)) atol = 1e-10
    end

    @testset "equidistribution identity: equal ∫M per cell" begin
        # With curvature=1, no cap and no smoothing the monitor is exactly q/∫q,
        # q = |p''|^{1/2}, so every returned cell must carry the same ∫q.
        q = sqrt.(abs.(d2f))
        h = xf[2] - xf[1]
        C = cumsum(vcat(0.0, (q[1:(end - 1)] .+ q[2:end]) ./ 2 .* h))
        g = adaptive_asset_grid(xf, pf; n=41, curvature=1.0, monitor_cap=Inf,
                                smoothing=0, is_density=true)
        Cg = [(j = searchsortedlast(xf, a); C[j] + (a - xf[j]) * q[j]) for a in g]
        Cg[end] = C[end]
        inc = diff(Cg)
        @test (maximum(inc) - minimum(inc)) / mean(inc) < 1e-2
    end

    @testset "concentration is monotone in `curvature`" begin
        counts = Int[]
        spacings = Float64[]
        for κ in (0.0, 0.5, 0.9, 1.0)
            g = adaptive_asset_grid(xf, pf; n=41, curvature=κ, monitor_cap=Inf,
                                    smoothing=0, is_density=true)
            push!(counts, count(a -> abs(a - μ0) <= 2 * s0, g))
            push!(spacings, minimum(diff(g)))
            @test all(diff(g) .> 0)
            @test g[1] == a_lo && g[end] == a_hi
        end
        @test issorted(counts)                    # more nodes at the peak as κ rises
        @test issorted(spacings; rev=true)        # and finer cells there
        @test counts[end] > 3 * counts[1]         # 27 vs 8
    end

    @testset "beats a uniform grid on piecewise-linear interpolation error" begin
        function pw_lin_err(nodes)
            p_at = [exp(-0.5 * ((a - μ0) / s0)^2) / (s0 * sqrt(2π)) for a in nodes]
            err = 0.0
            for (i, x) in enumerate(xf)
                j = clamp(searchsortedlast(nodes, x), 1, length(nodes) - 1)
                w = (x - nodes[j]) / (nodes[j + 1] - nodes[j])
                err = max(err, abs((1 - w) * p_at[j] + w * p_at[j + 1] - pf[i]))
            end
            return err
        end
        for n in (21, 41, 81)
            e_uni = pw_lin_err(collect(range(a_lo, a_hi; length=n)))
            e_ad = pw_lin_err(adaptive_asset_grid(xf, pf; n=n, curvature=0.9,
                                                  monitor_cap=Inf, smoothing=0, is_density=true))
            @test e_ad < e_uni / 5                # measured 12.5x / 19.1x / 21.0x
        end
    end

    @testset "monitor_cap defuses the borrowing-constraint atom" begin
        # A histogram atom in a cell of width w reports |p''| ~ 1/w², which is a
        # discretization artifact, not curvature. Uncapped it swallows the whole grid.
        x = M._make_asset_grid(0.0, 200.0, 200, :geometric)
        w = M._node_widths(x)
        mass = exp.(-((x .- 30.0) ./ 10.0) .^ 2) .* w
        mass[1] += 0.05 * sum(mass)                       # the atom at the constraint
        mass ./= sum(mass)
        g_cap = adaptive_asset_grid(x, mass; monitor_cap=3.0)
        g_unc = adaptive_asset_grid(x, mass; monitor_cap=Inf)
        @test minimum(diff(g_cap)) > 100 * minimum(diff(g_unc))
        # the capped grid resolves the actual peak; the uncapped one collapses onto the atom
        @test count(a -> 20 <= a <= 40, g_cap) > 4 * count(a -> 20 <= a <= 40, g_unc)  # 65 vs 14
        @test count(<=(1.0), g_unc) > 100      # 153 of 200 nodes inside the bottom 0.5%
        @test count(<=(1.0), g_cap) <= 5       # 1
        @test all(diff(g_cap) .> 0) && all(diff(g_unc) .> 0)
    end

    @testset "structural invariants and input validation" begin
        x = M._make_asset_grid(0.0, 50.0, 40, :geometric)
        mass = fill(1 / 40, 40)
        for n in (3, 17, 40, 97)
            g = adaptive_asset_grid(x, mass; n=n)
            @test length(g) == n
            @test g[1] == 0.0 && g[end] == 50.0
            @test all(diff(g) .> 0)
        end
        @test_throws ArgumentError adaptive_asset_grid(x[1:2], mass[1:2])
        @test_throws ArgumentError adaptive_asset_grid(x, mass[1:39])
        @test_throws ArgumentError adaptive_asset_grid(x, mass; n=2)
        @test_throws ArgumentError adaptive_asset_grid(x, mass; curvature=1.5)
        @test_throws ArgumentError adaptive_asset_grid(x, mass; curvature=-0.1)
        @test_throws ArgumentError adaptive_asset_grid(x, mass; monitor_cap=0.0)
        @test_throws ArgumentError adaptive_asset_grid(x, mass; smoothing=-1)
        @test_throws ArgumentError adaptive_asset_grid(reverse(x), mass)
        # a degenerate (all-zero) density falls back to uniform rather than dividing by zero
        g0 = adaptive_asset_grid(x, zeros(40))
        @test all(isfinite, g0) && all(diff(g0) .> 0)
    end

    @testset "adapt_ha_grid preserves the grid contract" begin
        spec = _HUG_SPEC_M2
        ss = _HUG_SS_M2
        g_new = adapt_ha_grid(spec.grid, ss.distribution)
        @test g_new isa MacroEconometricModels.HAGrid{Float64}
        @test g_new.n_dims == spec.grid.n_dims
        @test g_new.n_income == spec.grid.n_income
        @test g_new.n_points == spec.grid.n_points
        @test g_new.bounds == spec.grid.bounds
        @test g_new.labels == spec.grid.labels
        @test g_new.grids[1][1] == spec.grid.grids[1][1]        # borrowing constraint intact
        @test g_new.grids[1][end] == spec.grid.grids[1][end]
        @test all(diff(g_new.grids[1]) .> 0)
        @test g_new.grids[1] != spec.grid.grids[1]              # nodes actually moved

        # curvature=0 reproduces a uniform grid through the wrapper too
        g_uni = adapt_ha_grid(spec.grid, ss.distribution; curvature=0.0)
        lo, hi = spec.grid.bounds[1]
        @test g_uni.grids[1] ≈ collect(range(lo, hi; length=spec.grid.n_points[1])) atol = 1e-10

        # a coarser grid is allowed
        g_small = adapt_ha_grid(spec.grid, ss.distribution; n_points=[50])
        @test g_small.n_points == [50]
        @test length(g_small.grids[1]) == 50
        @test g_small.total_individual_states == 50 * spec.grid.n_income

        @test_throws ArgumentError adapt_ha_grid(spec.grid, ss.distribution; n_points=[50, 50])
        @test_throws ArgumentError adapt_ha_grid(spec.grid, ss.distribution[1:10])
        @test_throws ArgumentError adapt_ha_grid(spec.grid, zeros(size(ss.distribution)))

        # spec method returns a solvable specification
        spec2 = adapt_ha_grid(spec, ss)
        @test spec2 isa MacroEconometricModels.HADSGESpec{Float64}
        @test spec2.model == spec.model
        @test spec2.distribution == spec.distribution
        @test spec2.grid.grids[1] == g_new.grids[1]
        ss2 = compute_steady_state(spec2; max_iter=200, tol=5e-4)
        @test isfinite(ss2.prices[:r])
        @test isapprox(ss2.prices[:r], ss.prices[:r]; atol=2e-3)
    end

    @testset "two asset dimensions are adapted independently" begin
        grid = MacroEconometricModels.HAGrid(; liquid=(0.0, 20.0, 25), illiquid=(0.0, 60.0, 30),
                                              income_states=3)
        b = grid.grids[1]; a = grid.grids[2]
        dist = [exp(-((bi - 4.0) / 2.0)^2 - ((ai - 25.0) / 6.0)^2) for bi in b, ai in a, _ in 1:3]
        dist ./= sum(dist)
        g2 = adapt_ha_grid(grid, dist)
        @test g2.n_points == [25, 30]
        @test g2.bounds == grid.bounds
        for d in 1:2
            @test all(diff(g2.grids[d]) .> 0)
            @test g2.grids[d][1] == grid.grids[d][1]
            @test g2.grids[d][end] == grid.grids[d][end]
        end
        # nodes cluster on each dimension's own peak
        @test count(x -> abs(x - 4.0) <= 4.0, g2.grids[1]) >
              count(x -> abs(x - 4.0) <= 4.0, collect(range(0.0, 20.0; length=25)))
        @test count(x -> abs(x - 25.0) <= 12.0, g2.grids[2]) >
              count(x -> abs(x - 25.0) <= 12.0, collect(range(0.0, 60.0; length=30)))
    end

    @testset "internal monitor helpers" begin
        # nonuniform second derivative reproduces the uniform formula and is exact on x²
        x = collect(range(0.0, 2.0; length=9))
        @test M._second_derivative_nonuniform(x, x .^ 2)[2:(end - 1)] ≈ fill(2.0, 7) atol = 1e-10
        xn = [0.0, 0.1, 0.3, 0.7, 1.5, 2.0]
        @test M._second_derivative_nonuniform(xn, xn .^ 2)[2:(end - 1)] ≈ fill(2.0, 4) atol = 1e-10
        @test M._second_derivative_nonuniform(x, 3 .* x .+ 1) ≈ zeros(9) atol = 1e-10
        # widths partition the domain
        w = M._node_widths(x)
        @test sum(w) ≈ x[end] - x[1]
        wn = M._node_widths(xn)
        @test sum(wn) ≈ xn[end] - xn[1]
        # the smoother preserves a constant and is a contraction on the range
        v = randn(Random.MersenneTwister(258), 20)
        @test M._smooth3(ones(20)) ≈ ones(20)
        sv = M._smooth3(v)
        @test minimum(sv) >= minimum(v) && maximum(sv) <= maximum(v)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 18c: Endogenous labor supply (GHH / separable) — [T256] #355
# ─────────────────────────────────────────────────────────────────────────────

@testset "Endogenous labor supply" begin
    _ha_ip(base, ls) = IndividualProblem{Float64}(
        base.individual.utility, base.individual.utility_prime,
        base.individual.utility_prime_inv, base.individual.beta,
        base.individual.budget_fn, base.individual.borrowing_constraint,
        nothing, 1; labor=ls)

    @testset "LaborSupply construction" begin
        ls = LaborSupply(; kind=:ghh, psi=2.0, frisch=0.5)
        @test ls isa LaborSupply{Float64}
        @test ls.kind === :ghh && ls.psi ≈ 2.0 && ls.frisch ≈ 0.5 && ls.n_max == Inf
        @test_throws ArgumentError LaborSupply(; kind=:bogus)
        @test_throws ArgumentError LaborSupply(; psi=0.0)
        @test_throws ArgumentError LaborSupply(; frisch=-1.0)
        @test_throws ArgumentError LaborSupply(; n_max=0.0)

        # ORACLE (closed form): ψ n^{1/φ} = w·e ⟹ n = (w e/ψ)^φ
        for (psi, phi, we) in ((2.0, 0.5, 3.0), (1.0, 1.0, 0.4), (3.5, 0.25, 7.0))
            l = LaborSupply(; kind=:ghh, psi=psi, frisch=phi)
            @test labor_supply(l, we) ≈ (we / psi)^phi
            # the separable form multiplies the effective wage by u'(c)
            @test labor_supply(l, we, 2.0) ≈ (2we / psi)^phi
        end
        @test labor_supply(LaborSupply(; psi=1.0, frisch=1.0, n_max=0.75), 10.0) ≈ 0.75
        @test labor_supply(LaborSupply(), -1.0) == 0.0        # non-positive wage ⟹ no work

        # Endogenous labor is one-asset only.
        base = load_ha_example(:krusell_smith)
        @test_throws ArgumentError IndividualProblem{Float64}(
            base.individual.utility, base.individual.utility_prime,
            base.individual.utility_prime_inv, base.individual.beta,
            base.individual.budget_fn, [0.0, 0.0], nothing, 2; labor=LaborSupply())
    end

    FAST || @testset "exogenous-labor paths are untouched" begin
        base = load_ha_example(:krusell_smith)
        @test base.individual.labor === nothing
        # `labor_policy` returns ones, so ∫e·n dμ reduces to ∫e dμ.
        ss = compute_steady_state(base)
        n = labor_policy(base.individual, base.grid, base.income, ss.prices,
                         ss.policies[:consumption])
        @test all(==(1.0), n)
        @test !haskey(ss.policies, :labor)
        @test !haskey(ss.aggregates, :L)
        # The three-argument `_ssj_outcome_vector` form still dispatches as before.
        cpol = Float64[1 4; 2 5; 3 6]; apol = Float64[11 14; 12 15; 13 16]
        @test MacroEconometricModels._ssj_outcome_vector(:C, cpol, apol) == vec(cpol)
        @test MacroEconometricModels._ssj_outcome_vector(:K, cpol, apol) == vec(apol)
        # Labor outputs require an hours policy and say so.
        @test_throws ArgumentError MacroEconometricModels._ssj_outcome_vector(:N, cpol, apol)
        @test_throws ArgumentError MacroEconometricModels._ssj_outcome_vector(
            :L, cpol, apol, cpol)      # hours given, income missing
    end

    @testset "intratemporal FOC holds at every grid point" begin
        # ORACLE (analytic, exact): the household's static first-order condition
        # for hours. GHH: ψ n^{1/φ} = w·e — no wealth effect, so hours depend on
        # the income state alone. Separable: ψ n^{1/φ} = w·e·u'(c), which couples
        # hours to consumption. Both must hold at EVERY (a, e), including the
        # constrained cells where `_egm_solve` runs a joint root-find.
        base = load_ha_example(:krusell_smith)
        prices = Dict(:r => 0.0077, :w => 2.467)
        for kind in (:ghh, :separable)
            ls = LaborSupply(; kind=kind, psi=1.5, frisch=0.5)
            ip = _ha_ip(base, ls)
            c, a, conv = MacroEconometricModels._egm_solve(ip, base.grid, base.income,
                                                            prices; max_iter=3000, tol=1e-12)
            @test conv
            n = labor_policy(ip, base.grid, base.income, prices, c)
            ag = base.grid.grids[1]
            foc_err = 0.0; budget_err = 0.0
            for j in eachindex(base.income.states), i in eachindex(ag)
                we = prices[:w] * base.income.states[j]
                rhs = kind === :ghh ? we : we * ip.utility_prime(c[i, j])
                foc_err = max(foc_err, abs(ls.psi * n[i, j]^(1 / ls.frisch) - rhs))
                # budget identity: c + a' = (1+r)a + w·e·n
                budget_err = max(budget_err, abs((c[i, j] + a[i, j]) -
                                    ((1 + prices[:r]) * ag[i] + we * n[i, j])))
            end
            @test foc_err < 1e-10
            @test budget_err < 1e-9
            @test all(n .> 0)
            # GHH hours are a function of the income state alone (no wealth effect);
            # separable hours must actually vary with assets, or the wealth effect
            # this preference class exists to deliver would be missing.
            spread = maximum(j -> maximum(n[:, j]) - minimum(n[:, j]),
                             eachindex(base.income.states))
            kind === :ghh ? (@test spread < 1e-12) : (@test spread > 1e-3)
        end
    end

    FAST || @testset "steady state clears with endogenous labor" begin
        # ORACLE: with Cobb-Douglas production the firm FOC pins K/L given r —
        # k = (α Z /(r+δ))^{1/(1-α)} — independent of the household side. So the
        # REALIZED K/L from the distribution must reproduce it exactly, and it is
        # aggregate LABOR (∫e·n dμ), not the params[:L] placeholder, that has to
        # enter. Getting this wrong leaves a K/L that misses by the labor gap.
        for (kind, psi) in ((:ghh, 3.0), (:separable, 1.0))
            spec = MacroEconometricModels._endogenous_labor_example(; kind=kind, psi=psi)
            ss = compute_steady_state(spec)
            al = spec.het_params[:alpha]; de = spec.het_params[:delta]
            k_foc = (al / (ss.prices[:r] + de))^(1 / (1 - al))
            # rtol is set by the bisection's own clearing tolerance, not by the
            # identity: K_d = k·L holds exactly, but the solver stops once
            # |K_s − K_d| ≤ rtol·K (≈4.5e-7 here), so K_s/L inherits that slack.
            # 1e-6 still catches a labor gap of any economic size.
            @test ss.aggregates[:K] / ss.aggregates[:L] ≈ k_foc rtol=1e-6
            @test abs(ss.excess_demand) < 1e-6 * max(1.0, ss.aggregates[:K])
            @test ss.converged
            @test ha_grid_diagnostics(ss).adequate
            # Both labor aggregates are reported and are distinct concepts.
            @test haskey(ss.policies, :labor)
            @test ss.aggregates[:L] ≈ dot(vec(ss.policies[:labor] .*
                    reshape(spec.income.states, 1, :)), vec(ss.distribution)) rtol=1e-12
            @test ss.aggregates[:N] ≈ dot(vec(ss.policies[:labor]),
                                          vec(ss.distribution)) rtol=1e-12
            @test ss.aggregates[:L] != ss.aggregates[:N]
            # Y must be built from realized labor, not the params[:L] = 1 default.
            @test ss.aggregates[:Y] ≈ ss.aggregates[:K]^al *
                                      ss.aggregates[:L]^(1 - al) rtol=1e-6
            # Aiyagari existence still holds at the equilibrium.
            @test spec.individual.beta * (1 + ss.prices[:r]) < 1
        end
    end

    FAST || @testset "wage shock moves hours the right way (SSJ)" begin
        # ORACLE (analytic): under GHH, n = (w e/ψ)^φ is PURELY STATIC, so
        #   (i) dN/dw = φ·N/w exactly, and
        #   (ii) the sequence-space Jacobian of hours w.r.t. the wage is DIAGONAL —
        #        there is no anticipation, because hours never depend on the
        #        continuation value.
        spec = load_ha_example(:endogenous_labor)
        ss = compute_steady_state(spec)
        hh = HetBlock(spec, ss; inputs=[:r, :w], outputs=[:A, :C, :N, :L])
        @test hh.ss_outputs[:A] ≈ ss.aggregates[:K] rtol=1e-10
        @test hh.ss_outputs[:N] ≈ ss.aggregates[:N] rtol=1e-10
        @test hh.ss_outputs[:L] ≈ ss.aggregates[:L] rtol=1e-10

        Th = 10
        J = block_jacobian(hh, Th)
        ls = spec.individual.labor
        dN_dw = ls.frisch * ss.aggregates[:N] / ss.prices[:w]
        @test J[(:N, :w)][1, 1] ≈ dN_dw rtol=1e-4          # correct sign AND magnitude
        @test J[(:N, :w)][1, 1] > 0                        # hours rise with the wage
        @test maximum(abs, [J[(:N, :w)][t, s] for t in 1:Th for s in 1:Th if t != s]) < 1e-9
        # A labor output must be rejected on an exogenous-labor block.
        @test_throws ArgumentError HetBlock(load_ha_example(:krusell_smith),
            compute_steady_state(load_ha_example(:krusell_smith)); outputs=[:bogus_output])
    end

    @testset "built-in :endogenous_labor example" begin
        spec = load_ha_example(:endogenous_labor)
        @test spec isa HADSGESpec{Float64}
        @test spec.individual.labor isa LaborSupply{Float64}
        @test spec.individual.labor.kind === :ghh
        @test spec.grid.bounds[1] == (0.0, 2000.0)
        @test spec.n_assets == 1
        @test_throws ErrorException load_ha_example(:not_a_model)
        # ψ = 3 is calibrated so efficiency units land on the L = 1 normalization
        # the exogenous-labor examples impose, making the two comparable.
        FAST && return
        ss = compute_steady_state(spec)
        @test ss.aggregates[:L] ≈ 1.0 atol=0.05
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 19: Plotting
# ─────────────────────────────────────────────────────────────────────────────

@testset "Plotting" begin
    grid = HAGrid(assets=(0.0, 200.0, 80), income_states=3)
    inc = rouwenhorst(0.966, 0.5, 3)
    ip = IndividualProblem{Float64}(
        c -> log(c), c -> 1.0/c, m -> 1.0/m, 0.99,
        (a, e, prices) -> (1 + prices[:r]) * a + prices[:w] * e,
        [0.0], nothing, 1
    )
    function price_fn_plot(K, params)
        r = 0.36 * K^(0.36-1) - 0.025; w = 0.64 * K^0.36
        Dict(:r => r, :w => w)
    end
    params = Dict(:alpha => 0.36, :delta => 0.025, :Z => 1.0, :L => 1.0)
    ss = MacroEconometricModels._ha_steady_state(ip, grid, inc, price_fn_plot, params;
        K_init=10.0, r_bounds=(-0.02, 0.04), max_iter=60, tol=1e-3)

    # Distribution plot (default view)
    p = plot_result(ss)
    @test p isa PlotOutput
    @test !isempty(p.html)
    @test contains(p.html, "Wealth Distribution")

    # Explicit :distribution view
    p1b = plot_result(ss; view=:distribution)
    @test p1b isa PlotOutput

    # Lorenz curve
    p2 = plot_result(ss; view=:lorenz)
    @test p2 isa PlotOutput
    @test contains(p2.html, "Lorenz")

    # Policy function plot
    p3 = plot_result(ss; view=:policy)
    @test p3 isa PlotOutput
    @test contains(p3.html, "Policy")

    # Invalid view
    @test_throws ArgumentError plot_result(ss; view=:invalid)

    # Custom title
    p4 = plot_result(ss; title="Custom Title")
    @test contains(p4.html, "Custom Title")
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 20: @dsge macro parser extensions
# ─────────────────────────────────────────────────────────────────────────────

@testset "@dsge with heterogeneous" begin
    @testset "Rouwenhorst parser" begin
        spec = @dsge begin
            parameters: alpha = 0.36, beta_hh = 0.99, delta = 0.025, rho_z = 0.95, sigma_z = 0.007
            endogenous: Y, K, r, w, Z
            exogenous: eps_Z

            heterogeneous: a in [0.0, 200.0], n_grid = 100, utility = log, discount = beta_hh, borrowing = 0.0

            idiosyncratic: e ~ Rouwenhorst(0.966, 0.5, 5)

            aggregation: K = sum(a)

            Y[t] = Z[t] * K[t-1]^alpha
            r[t] = alpha * Z[t] * K[t-1]^(alpha-1) - delta
            w[t] = (1 - alpha) * Z[t] * K[t-1]^alpha
            Z[t] = rho_z * Z[t-1] + sigma_z * eps_Z[t]
        end

        @test spec isa HADSGESpec{Float64}
        @test spec.grid.n_dims == 1
        @test spec.grid.n_points == [100]
        @test spec.grid.bounds[1] == (0.0, 200.0)
        @test spec.n_income == 5
        @test spec.individual.beta ≈ 0.99
        @test spec.individual.borrowing_constraint[1] ≈ 0.0
        @test spec.individual.n_asset_dims == 1
        @test spec.n_assets == 1
        @test spec.het_params[:alpha] ≈ 0.36
        @test spec.het_params[:delta] ≈ 0.025
        @test spec.aggregate_spec isa DSGESpec{Float64}
        @test :Y in spec.aggregate_spec.endog
        @test :K in spec.aggregate_spec.endog
        @test length(spec.income.states) == 5
        @test size(spec.income.transition) == (5, 5)
    end

    @testset "Tauchen parser" begin
        spec = @dsge begin
            parameters: alpha = 0.36, beta_hh = 0.99, delta = 0.025, rho_z = 0.95, sigma_z = 0.007
            endogenous: Y, K, r, w, Z
            exogenous: eps_Z

            heterogeneous: a in [0.0, 150.0], n_grid = 80, utility = log, discount = beta_hh, borrowing = 0.0

            idiosyncratic: e ~ Tauchen(0.9, 0.3, 7)

            aggregation: K = sum(a)

            Y[t] = Z[t] * K[t-1]^alpha
            r[t] = alpha * Z[t] * K[t-1]^(alpha-1) - delta
            w[t] = (1 - alpha) * Z[t] * K[t-1]^alpha
            Z[t] = rho_z * Z[t-1] + sigma_z * eps_Z[t]
        end

        @test spec isa HADSGESpec{Float64}
        @test spec.grid.n_points == [80]
        @test spec.grid.bounds[1] == (0.0, 150.0)
        @test spec.n_income == 7
        @test length(spec.income.states) == 7
    end

    @testset "Standard @dsge unaffected" begin
        spec_std = @dsge begin
            parameters: rho = 0.9, sigma = 0.01
            endogenous: Y, A
            exogenous: eps_A

            Y[t] = A[t]
            A[t] = rho * A[t-1] + sigma * eps_A[t]
        end
        @test spec_std isa DSGESpec{Float64}
    end

    @testset "CRRA curvature and model routing (#239/T140)" begin
        # σ ≠ 1 CRRA is parsed (old code always forced σ = 1.0 / log utility)
        spec = @dsge begin
            parameters: alpha = 0.36, beta_hh = 0.99, delta = 0.025, rho_z = 0.95, sigma_z = 0.007
            endogenous: Y, K, r, w, Z
            exogenous: eps_Z
            heterogeneous: a in [0.0, 200.0], n_grid = 60, utility = crra(1.5), discount = beta_hh, borrowing = 0.0
            idiosyncratic: e ~ Rouwenhorst(0.966, 0.5, 5)
            aggregation: K = sum(a)
            Y[t] = Z[t] * K[t-1]^alpha
            r[t] = alpha * Z[t] * K[t-1]^(alpha-1) - delta
            w[t] = (1 - alpha) * Z[t] * K[t-1]^alpha
            Z[t] = rho_z * Z[t-1] + sigma_z * eps_Z[t]
        end
        @test spec.individual.utility_prime(2.0) ≈ 2.0^(-1.5)
        @test spec.individual.utility(2.0) ≈ 2.0^(1 - 1.5) / (1 - 1.5)
        @test spec.model == :aiyagari    # default model field

        # model = huggett routes into the ctor; macro-controlled fields match
        spec_h = @dsge begin
            parameters: alpha = 0.36, beta_hh = 0.99322, delta = 0.025, rho_z = 0.9, sigma_z = 0.01
            endogenous: Y, K, r, w, Z
            exogenous: eps_Z
            heterogeneous: a in [-2.0, 4.0], n_grid = 80, utility = crra(1.5), discount = beta_hh, borrowing = -2.0, model = huggett
            idiosyncratic: e ~ Rouwenhorst(0.9, 0.1, 3)
            aggregation: K = sum(a)
            Y[t] = Z[t] * K[t-1]^alpha
            r[t] = alpha * Z[t] * K[t-1]^(alpha-1) - delta
            w[t] = (1 - alpha) * Z[t] * K[t-1]^alpha
            Z[t] = rho_z * Z[t-1] + sigma_z * eps_Z[t]
        end
        @test spec_h.model == :huggett
        @test spec_h.grid.bounds[1] == (-2.0, 4.0)
        @test spec_h.individual.borrowing_constraint[1] ≈ -2.0
        @test spec_h.individual.beta ≈ 0.99322
        @test spec_h.individual.utility_prime(2.0) ≈ 2.0^(-1.5)

        # the built Huggett spec solves
        ss = compute_steady_state(spec_h; max_iter=80, tol=1e-3)
        sol = solve(spec_h; method=:ssj, ss=ss, T_horizon=80, n_reduced=15)
        @test sol isa HADSGESolution
        @test sol.method === :ssj
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 21: solve dispatch
# ─────────────────────────────────────────────────────────────────────────────

@testset "solve dispatch" begin
    spec = load_ha_example(:krusell_smith)
    # Verify method dispatch exists and does not conflict
    @test hasmethod(solve, Tuple{HADSGESpec{Float64}})
    @test hasmethod(solve, Tuple{DSGESpec{Float64}})

    # Verify dispatch is distinct: solve(::HADSGESpec) and solve(::DSGESpec) are different methods
    m1 = which(solve, Tuple{HADSGESpec{Float64}})
    m2 = which(solve, Tuple{DSGESpec{Float64}})
    @test m1 !== m2

    FAST && return

    # Verify unknown method raises error
    ss = MacroEconometricModels._ha_steady_state(
        spec.individual, spec.grid, spec.income,
        MacroEconometricModels._default_cobb_douglas_price_fn, spec.het_params;
        K_init=10.0, r_bounds=(-0.02, 0.04), max_iter=30, tol=1e-2
    )
    @test_throws ErrorException solve(spec; method=:nonexistent, ss=ss)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 22: HA Bayesian estimation
# ─────────────────────────────────────────────────────────────────────────────

@testset "HA Bayesian estimation" begin
    spec = load_ha_example(:krusell_smith)
    # [T206] NOTE: the plan's asset-grid shrink (n_a 200→60/80) was dropped — perturbing the
    # KS-SSJ grid non-monotonically stabilizes the reduced realization and flips the #234
    # @test_broken truncation assertions to unexpected passes (n_a=60 flips T049's L1475;
    # n_a=80 also flips _build_ha_likelihood_fn's ll_val). Per the plan's flip-guard fallback
    # we keep the full-size spec and cut only draws + T_data (+ the shared-solve hoist).
    # The Ho-Kalman spectral radius is CHAOTIC, not monotone, in the calibration: the
    # a_max/grid_type change that fixed the asset-grid truncation was checked against all
    # three @test_broken items (all still -Inf at :geometric a_max=1000, whereas
    # :double_exp a_max=1000 flips all three). Re-measure them after ANY change to
    # a_max, n_a or grid_type — an unexpected pass is reported as a suite FAILURE.

    @testset "_update_ha_params" begin
        param_names = [:alpha]
        theta = [0.30]
        new_spec = MacroEconometricModels._update_ha_params(spec, param_names, theta)
        @test new_spec isa HADSGESpec{Float64}
        @test new_spec.aggregate_spec.param_values[:alpha] ≈ 0.30
        @test new_spec.het_params[:alpha] ≈ 0.36  # het_params has its own copy
        @test new_spec.individual.beta ≈ 0.99  # unchanged

        # Update beta
        param_names2 = [:beta]
        theta2 = [0.98]
        new_spec2 = MacroEconometricModels._update_ha_params(spec, param_names2, theta2)
        @test new_spec2.individual.beta ≈ 0.98
    end

    # Full KS SS + SSJ + MH is the HA-DSGE ceiling. Windows/macOS smoke (FAST)
    # keeps the cheap helper above; Ubuntu still runs the rest.
    FAST && return

    # Compute steady state for generating fake data
    ss = compute_steady_state(spec; K_init=10.0, r_bounds=(-0.02, 0.04), max_iter=50, tol=1e-3)
    K_ss = ss.aggregates[:K]
    T_data = 16
    rng = Random.MersenneTwister(42)
    data_K = K_ss .+ 0.1 .* randn(rng, T_data)  # K with noise

    # [T206] hoist one shared :ssj solve to avoid re-solving in the two helper testsets below.
    sol_shared = solve(spec; method=:ssj, ss=ss, T_horizon=30, n_reduced=10)

    @testset "_build_ha_likelihood_fn" begin
        # Solve model first to have a valid solution for observation equation
        @test sol_shared isa HADSGESolution{Float64}

        param_names = [:alpha]
        ll_fn = MacroEconometricModels._build_ha_likelihood_fn(
            spec, param_names, reshape(data_K, 1, :),
            [:K], nothing, :ssj, (T_horizon=30, n_reduced=10)
        )

        ll_val = ll_fn([0.36])
        # #234 honesty consequence: with the silent G1 eigenvalue rescale removed, the KS-SSJ
        # Ho-Kalman realization is truthfully explosive (reduced ρ≈1.003 ≥ 1) at the small FAST
        # size used here (n_reduced=10), so the Kalman likelihood is honestly -Inf rather than a
        # finite value — this assertion encoded the pre-#234 silently-stabilized behavior.
        # Follow-up: stabilize the reduced realization at small n_reduced (the runtime warning
        # flags a probable incomplete GE block / mis-scaled Jacobian) — NOT a silent rescale.
        @test_broken isfinite(ll_val)
        @test ll_val < 0  # -Inf < 0 still holds; it is the finiteness that broke (see above)

        # Likelihood should handle bad parameter values gracefully
        ll_bad = ll_fn([0.001])  # extreme parameter
        @test ll_bad == -Inf || ll_bad < ll_val + 100  # either fails or worse
    end

    @testset "_build_ha_observation_equation" begin
        sol = sol_shared

        Z, d, H = MacroEconometricModels._build_ha_observation_equation(
            sol, [:K], nothing
        )
        n_states = size(sol.linear_solution.G1, 1)
        @test size(Z) == (1, n_states)
        @test length(d) == 1
        @test size(H) == (1, 1)
        @test d[1] ≈ K_ss atol=1.0  # steady state K
        @test H[1, 1] == 0  # zero default measurement error (T042)
        @test all(iszero, H)

        # #228/T129: Z is the C_obs row for the matched aggregate (:K), NOT a silent
        # unit-loading at an arbitrary reduced-state index.
        @test Z ≈ reshape(sol.C_obs[1, :], 1, :)

        # Custom measurement error
        Z2, d2, H2 = MacroEconometricModels._build_ha_observation_equation(
            sol, [:K], [0.5]
        )
        @test H2[1, 1] ≈ 0.25  # 0.5^2

        # #228/T129: an observable absent from the reduced system's aggregate outputs
        # raises an informative error naming it (the SSJ realization exposes only :K),
        # instead of the old silent arbitrary-index fallback.
        err = try
            MacroEconometricModels._build_ha_observation_equation(sol, [:K, :Y], nothing)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("Y", err.msg)
        @test_throws ErrorException MacroEconometricModels._build_ha_observation_equation(
            sol, [:nonexistent], nothing)
    end

    @testset "estimate_dsge_bayes dispatch" begin
        # Very small run to verify the method dispatches correctly
        priors = Dict(:alpha => Distributions.Normal(0.36, 0.05))
        rng_est = Random.MersenneTwister(123)

        result = estimate_dsge_bayes(
            spec, reshape(data_K, T_data, 1), [0.36];
            priors=priors,
            observables=[:K],
            n_draws=6,
            burnin=2,
            ha_method=:ssj,
            ha_kwargs=(T_horizon=30, n_reduced=10),
            proposal_scale=0.001,
            adapt_interval=50,  # no adaptation in 6 draws
            rng=rng_est
        )

        @test result isa BayesianDSGE{Float64}
        @test result.solved_at === :posterior_mean  # normal path (#149/T050)
        @test size(result.theta_draws, 2) == 1  # one parameter
        @test size(result.theta_draws, 1) == 4  # n_draws - burnin = 6 - 2
        @test length(result.log_posterior) == 4
        @test result.method === :rwmh
        @test result.acceptance_rate >= 0.0
        @test result.acceptance_rate <= 1.0
        @test length(result.param_names) == 1
        @test result.param_names[1] === :alpha

        # Posterior summary should work
        ps = posterior_summary(result)
        @test haskey(ps, :alpha)
        @test isfinite(ps[:alpha][:mean])

        # #136: theta0 as a Dict (order-independent) is accepted through the HA method;
        # a wrong-length positional vector errors informatively before any solve.
        result_dict = estimate_dsge_bayes(
            spec, reshape(data_K, T_data, 1), Dict(:alpha => 0.36);
            priors=priors, observables=[:K], n_draws=6, burnin=2,
            ha_method=:ssj, ha_kwargs=(T_horizon=30, n_reduced=10),
            proposal_scale=0.001, adapt_interval=50, rng=Random.MersenneTwister(7))
        @test result_dict isa BayesianDSGE{Float64}
        @test_throws ArgumentError estimate_dsge_bayes(
            spec, reshape(data_K, T_data, 1), [0.36, 0.9];   # length 2, but 1 prior
            priors=priors, observables=[:K], n_draws=10,
            ha_method=:ssj, ha_kwargs=(T_horizon=30, n_reduced=10))

        # #142: n×T data (1×T_data) resolves identically to T×n (same internal matrix →
        # identical draws under the same rng); a shape matching neither dim to n_obs errors.
        result_nt = estimate_dsge_bayes(
            spec, reshape(data_K, 1, T_data), Dict(:alpha => 0.36);
            priors=priors, observables=[:K], n_draws=6, burnin=2,
            ha_method=:ssj, ha_kwargs=(T_horizon=30, n_reduced=10),
            proposal_scale=0.001, adapt_interval=50, rng=Random.MersenneTwister(7))
        @test result_nt.theta_draws ≈ result_dict.theta_draws
        @test_throws ArgumentError estimate_dsge_bayes(
            spec, randn(3, T_data), [0.36];                  # neither dim == n_obs (1)
            priors=priors, observables=[:K], n_draws=10,
            ha_method=:ssj, ha_kwargs=(T_horizon=30, n_reduced=10))
    end

    @testset "T049: default T_horizon >= 300 (truncation)" begin
        # (A) Pin the signature default cheaply (no horizon-300 solve — those cost minutes):
        #     the signature's ha_kwargs default uses this const.
        @test MacroEconometricModels._HA_DEFAULT_T_HORIZON >= 300

        # (B) Truncation is non-negligible: the likelihood depends on the horizon (compared
        #     at cheap horizons; KS ρ_z=0.95 ⇒ 0.95^30≈0.21 vs 0.95^60≈0.046 tail alive).
        ll30 = MacroEconometricModels._build_ha_likelihood_fn(
            spec, [:alpha], reshape(data_K, 1, :), [:K], nothing, :ssj,
            (T_horizon=30, n_reduced=15))([0.36])
        ll60 = MacroEconometricModels._build_ha_likelihood_fn(
            spec, [:alpha], reshape(data_K, 1, :), [:K], nothing, :ssj,
            (T_horizon=60, n_reduced=15))([0.36])
        # #234 honesty consequence (see the _build_ha_likelihood_fn testset): at these small
        # FAST sizes (n_reduced=15) the truthful KS-SSJ realization is explosive, so both
        # likelihoods are -Inf. Follow-up: stabilize the reduced realization; broken pending that.
        @test_broken isfinite(ll30) && isfinite(ll60)
        @test_broken abs(ll30 - ll60) > 1e-6
    end

    @testset "posterior-mean solution built at the mean, marked (#149/T050)" begin
        # KS always yields a determinate, finite reduced solution for ANY θ (even NaN/Inf),
        # so the mean-solve-fails → highest-posterior-draw branch — which mirrors the
        # unit-tested aggregate [T044]/#143 path — is not reachable with this fast example.
        # We verify the reachable guarantees of the fix: (a) the container is built at the
        # POSTERIOR MEAN θ and marked, NOT silently at the original pre-estimation spec (the
        # removed E-25 bug); (b) when no candidate yields a supported HADSGESolution the
        # helper errors LOUDLY rather than silently substituting.
        post_draws = reshape([0.4, 0.5, 0.6], 3, 1)   # mean = 0.5 (≠ spec's alpha=0.36)
        post_lp    = [-3.0, -1.0, -2.0]
        linear_sol, ss_result, solved_at, theta_used =
            MacroEconometricModels._build_ha_result_solution(
                spec, [:alpha], post_draws, post_lp, [:K], nothing,
                :ssj, (T_horizon=30, n_reduced=10))
        @test solved_at === :posterior_mean
        @test theta_used ≈ [0.5]                    # built at the mean, not spec's 0.36
        @test all(isfinite, linear_sol.G1)

        # No candidate solves (unsupported method ⇒ no HADSGESolution) ⇒ loud error, never a
        # silent original-spec substitution.
        @test_throws ErrorException MacroEconometricModels._build_ha_result_solution(
            spec, [:alpha], reshape([0.36], 1, 1), [0.0], [:K], nothing,
            :badmethod, NamedTuple())
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 23: Clearing closure (Aiyagari regression — refactor must not change behavior)
# ─────────────────────────────────────────────────────────────────────────────

@testset "Clearing closure (Aiyagari regression)" begin
    spec = load_ha_example(:krusell_smith)
    @test spec.model == :aiyagari                       # new field defaults correctly

    FAST && return

    ss = compute_steady_state(spec; r_bounds=(-0.02, 0.04), max_iter=100, tol=1e-3)
    @test ss.aggregates[:K] > 0
    @test isfinite(ss.prices[:r])
    @test haskey(ss.prices, :w)                         # Cobb-Douglas wage still produced
    @test abs(ss.excess_demand) < 5e-3                  # market essentially clears
    @test -0.01 < ss.prices[:r] < 1 / spec.individual.beta - 1  # r* below time-pref rate
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 24: Huggett (1993) — pure-exchange risk-free bond, zero net supply
# ─────────────────────────────────────────────────────────────────────────────

@testset "Huggett (1993) steady state" begin
    # Six model periods per year (Huggett 1993): annualize the per-period rate.
    annualize(rp) = (1 + rp)^6 - 1
    # Table 1 (σ = 1.5): credit limit => equilibrium annual risk-free rate.
    targets = FAST ? [(-2.0, -0.071)] :
              [(-2.0, -0.071), (-4.0, 0.023), (-6.0, 0.034), (-8.0, 0.040)]

    r_annuals = Float64[]
    for (cl, r_target) in targets
        a_max = cl <= -6 ? 18.0 : 8.0
        if cl == -2.0                       # reuse the shared cl=−2 SS (a_max=8.0, n_a=200)
            spec = _HUG_SPEC_M2
            ss = _HUG_SS_M2
        else
            spec = MacroEconometricModels._huggett_example(; credit_limit=cl, a_max=a_max, n_a=200)
            ss = compute_steady_state(spec; max_iter=200, tol=5e-4)
        end
        @test spec.model == :huggett
        @test ss.converged
        @test abs(ss.excess_demand) < 3e-3                 # bond market clears (∫a' ≈ 0)
        r_ann = annualize(ss.prices[:r])
        push!(r_annuals, r_ann)
        # Reproduces Huggett (1993) Table 1 within method/grid tolerance (~1.5pp)
        @test isapprox(r_ann, r_target; atol=0.015)
        # Precautionary saving keeps r* below the time-preference rate (1/β − 1)
        @test r_ann < annualize((1 - spec.individual.beta) / spec.individual.beta)
    end

    # Huggett's comparative static: r* rises as the credit limit loosens.
    @test issorted(r_annuals)

    # load_ha_example(:huggett) is the default (credit limit −2) economy.
    spec0 = load_ha_example(:huggett)
    @test spec0.model == :huggett
    @test spec0.individual.borrowing_constraint[1] == -2.0
    @test spec0.income.states == [1.0, 0.1]
end

@testset "Huggett SSJ" begin
    spec = _HUG_SPEC_M2; ss = _HUG_SS_M2      # reuse shared cl=−2 SS (T209/#308)
    Th = FAST ? 20 : 50
    sol = solve(spec; method=:ssj, ss=ss, T_horizon=Th, n_reduced=FAST ? 10 : 20)
    @test sol isa HADSGESolution
    @test sol.method === :ssj
    @test maximum(abs.(eigvals(sol.linear_solution.G1))) <= 1 + 1e-6  # stable
    @test haskey(sol.jacobians, :H_U)                                  # clearing Jacobian
    @test haskey(sol.jacobians, :H_Z)                                  # shock Jacobian
    # A positive aggregate endowment shock lowers the clearing risk-free rate on impact.
    H_U = sol.jacobians[:H_U]; H_Z = sol.jacobians[:H_Z]
    dr = -(H_U \ (H_Z * [0.9^(t - 1) for t in 1:Th]))
    @test dr[1] < 0
end

@testset "Huggett Reiter" begin
    spec = _HUG_SPEC_M2; ss = _HUG_SS_M2      # reuse shared cl=−2 SS (T209/#308)
    sol = solve(spec; method=:reiter, ss=ss, n_reduced=FAST ? 15 : 30)
    @test sol isa HADSGESolution
    @test sol.method === :reiter
    @test maximum(abs.(eigvals(sol.linear_solution.G1))) <= 1 + 1e-6   # stable
    # #234: eu is now derived from the true spectral radius, so a genuinely stable
    # reduced system reports determinate (not a hardcoded [1,1] on a rescaled G1).
    @test MacroEconometricModels.is_determined(sol.linear_solution)
    @test MacroEconometricModels.is_stable(sol.linear_solution)
    @test sol.explained_variance > 0.5
    @test size(sol.linear_solution.G1, 1) == sol.n_reduced + 1         # state [d̃; w]
end

@testset "Huggett Krusell-Smith" begin
    spec = MacroEconometricModels._huggett_example(; credit_limit=-2.0, a_max=8.0,
                                                    n_a=FAST ? 60 : 100)
    ss = compute_steady_state(spec; max_iter=FAST ? 50 : 100, tol=1e-3)
    sol = solve(spec; method=:krusell_smith, ss=ss,
                T_sim=FAST ? 120 : 300, T_burn=FAST ? 30 : 75, max_outer=FAST ? 2 : 3)
    @test sol isa KrusellSmithSolution
    @test haskey(sol.plm_coefficients, :r)        # PLM forecasts the clearing rate, not K
    @test sol.r_squared[:r] > 0.7                 # rate is near-linear in the endowment shock
    b = sol.plm_coefficients[:r]
    @test abs(b[1] - ss.prices[:r]) < 0.01        # PLM intercept ≈ steady-state rate
    @test b[2] < 0                                # positive endowment shock lowers r
end

@testset "Den Haan (2010) accuracy" begin
    # --- Aiyagari capital model (z-augmented PLM makes the test meaningful) ---
    if !FAST
    ks_spec = load_ha_example(:krusell_smith)
    ss_a = compute_steady_state(ks_spec; r_bounds=(-0.02, 0.04), max_iter=80, tol=1e-3)
    ks = solve(ks_spec; method=:krusell_smith, ss=ss_a, T_sim=200, T_burn=100, max_outer=3)
    @test length(ks.plm_coefficients[:K]) == 3          # z-augmented PLM

    dh = den_haan_test(ks; T_sim=150, T_burn=100)
    @test dh isa DenHaanAccuracy
    @test dh.aggregate === :K
    @test isfinite(dh.dh_max) && dh.dh_max >= dh.dh_mean >= 0
    @test dh.sigma_ref > 0 && dh.sigma_plm > 0
    @test length(dh.ref_path) == 150 && length(dh.plm_path) == 150
    @test dh.sigma_plm > 0.2 * dh.sigma_ref             # PLM reproduces the fluctuations
    @test dh.dh_max < 1.0                               # accurate: well under 1% (Den Haan)
    report(dh)                                          # display smoke test
    end

    # --- Huggett: rate accuracy test is intentionally unsupported (errors clearly) ---
    # Reuse the shared cl=−2 SS — no extra solve (the guard fires on spec.model).
    ks_h = KrusellSmithSolution{Float64}(
        _HUG_SS_M2, Dict(:r => [_HUG_SS_M2.prices[:r], 0.0]), Dict(:r => 1.0),
        _HUG_SPEC_M2, false, 0)
    @test_throws ErrorException den_haan_test(ks_h)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 26 (#352/T253): sequence-space block composition (DAG) + 2nd-order SSJ
# ─────────────────────────────────────────────────────────────────────────────

# A pure-SimpleBlock DAG is exactly solvable by hand, so it pins down every piece
# of the composition machinery (shift matrices, topological sort, forward
# accumulation, the GE solve, and the second-order contraction) against closed
# forms rather than against snapshots.
@testset "SSJ blocks — SimpleBlock algebra" begin
    Th = 12

    # y_t = 2·u_t + 3·z_t ;  q_t = y_t − 0.5·u_{t-1} + 0.25·y_{t+1}
    blk1 = SimpleBlock(x -> [2 * x[1] + 3 * x[2]];
                       inputs=[:u, :z], outputs=[:y],
                       ss_inputs=Dict(:u => 0.0, :z => 0.0), name=:one)
    blk2 = SimpleBlock(x -> [x[3] - 0.5 * x[1] + 0.25 * x[2]];
                       inputs=[:u, :y], outputs=[:q],
                       lags=Dict(:u => [1], :y => [-1, 0]),
                       ss_inputs=Dict(:u => 0.0, :y => 0.0), name=:two)

    # Argument order: inputs in declaration order, lags ascending within an input.
    @test ssj_arg_order(blk1) == [(:u, 0), (:z, 0)]
    @test ssj_arg_order(blk2) == [(:u, 1), (:y, -1), (:y, 0)]
    @test blk1.ss_outputs[:y] == 0.0

    J1 = block_jacobian(blk1, Th)
    @test J1[(:y, :u)] ≈ 2 * Matrix(I, Th, Th)
    @test J1[(:y, :z)] ≈ 3 * Matrix(I, Th, Th)

    # Shift matrices: lag l ⇒ ones on M[t, t-l]; out-of-window entries dropped.
    S_lag = zeros(Th, Th); for t in 2:Th; S_lag[t, t-1] = 1.0; end
    S_lead = zeros(Th, Th); for t in 1:(Th-1); S_lead[t, t+1] = 1.0; end
    J2 = block_jacobian(blk2, Th)
    @test J2[(:q, :u)] ≈ -0.5 .* S_lag
    @test J2[(:q, :y)] ≈ Matrix(I, Th, Th) .+ 0.25 .* S_lead

    model = combine_blocks(blk1, blk2; name=:toy)
    @test [b.name for b in model.blocks] == [:one, :two]      # topological order
    @test model.exogenous == [:u, :z]
    @test model.endogenous == [:y, :q]
    @test model.ss_values[:q] == 0.0

    # Supplying the blocks out of order must not change the sorted DAG.
    @test [b.name for b in combine_blocks(blk2, blk1).blocks] == [:one, :two]

    gej = ssj_jacobian(model; unknowns=[:u], targets=[:q], shocks=[:z], T_horizon=Th)
    # Chain rule by hand: dq/du = ∂q/∂u + (∂q/∂y)(∂y/∂u); dq/dz = (∂q/∂y)(∂y/∂z).
    H_U_hand = J2[(:q, :u)] .+ J2[(:q, :y)] * J1[(:y, :u)]
    H_Z_hand = J2[(:q, :y)] * J1[(:y, :z)]
    @test gej.H_U ≈ H_U_hand
    @test gej.H_Z ≈ H_Z_hand
    @test size(gej.H_U) == (Th, Th) && size(gej.H_Z) == (Th, Th)

    dz = [0.5^(t - 1) for t in 1:Th]
    r1 = ssj_irf(gej, Dict(:z => dz))
    du_hand = -(H_U_hand \ (H_Z_hand * dz))
    @test r1.paths[:u] ≈ du_hand
    @test r1.paths[:y] ≈ J1[(:y, :u)] * du_hand .+ J1[(:y, :z)] * dz
    @test r1.paths[:z] ≈ dz
    @test r1.order == 1 && isempty(r1.correction)
    # A linear DAG clears exactly at first order.
    @test r1.target_residual[:q] < 1e-12
    @test maximum(abs, r1.paths[:q]) < 1e-12

    # Second order on a LINEAR DAG must vanish identically.
    r2 = ssj_irf(gej, Dict(:z => dz); order=2)
    @test r2.order == 2
    @test maximum(abs, r2.correction[:u]) < 1e-9
    @test r2.paths[:u] ≈ du_hand atol=1e-9

    # Convenience single-shock method.
    @test ssj_irf(gej, :z, dz).paths[:u] ≈ du_hand
end

@testset "SSJ blocks — second-order closed form" begin
    Th = 8
    # y_t = u_t + u_t² ;  q_t = y_t − z_t.  Equilibrium: u + u² = z.
    b1 = SimpleBlock(x -> [x[1] + x[1]^2];
                     inputs=[:u], outputs=[:y],
                     ss_inputs=Dict(:u => 0.0), name=:quad)
    b2 = SimpleBlock(x -> [x[1] - x[2]];
                     inputs=[:y, :z], outputs=[:q],
                     ss_inputs=Dict(:y => 0.0, :z => 0.0), name=:clear)
    gej = ssj_jacobian(combine_blocks(b1, b2; name=:quadtoy);
                       unknowns=[:u], targets=[:q], shocks=[:z], T_horizon=Th)
    @test gej.H_U ≈ Matrix(I, Th, Th)           # ∂(u+u²)/∂u = 1 at u=0

    dz = fill(0.05, Th)
    r1 = ssj_irf(gej, Dict(:z => dz))
    @test r1.paths[:u] ≈ dz                     # first order: du = dz

    r2 = ssj_irf(gej, Dict(:z => dz); order=2)
    # Second order: u + u² = z with u = z + u₂ ⇒ u₂ = −z².
    @test r2.correction[:u] ≈ -dz .^ 2 rtol=1e-8
    @test r2.paths[:u] ≈ dz .- dz .^ 2 rtol=1e-8
    # D²y[v,v] = 2·(du¹)² ⇒ the second-order y path is J·u₂ + ½·2z² = −z² + z² = 0.
    @test maximum(abs, r2.correction[:y]) < 1e-8
    # Exact root of u + u² = z:  u* = (√(1+4z) − 1)/2.  Second order beats first.
    u_exact = (sqrt.(1 .+ 4 .* dz) .- 1) ./ 2
    @test maximum(abs, r2.paths[:u] .- u_exact) < maximum(abs, r1.paths[:u] .- u_exact)
    @test r2.target_residual[:q] < r1.target_residual[:q]
end

@testset "SSJ blocks — HetBlock and DAG composition" begin
    spec = load_ha_example(:krusell_smith)
    # Converged (default) tolerance, not the loose tol=1e-4 used elsewhere: the
    # firm SimpleBlock below is built at K_ss = ∫a dμ while `ss.prices` are
    # evaluated at the firm's K_demand, so the two price sets differ by exactly
    # |dp/dK|·|excess_demand|. Testing the block's Cobb-Douglas algebra against
    # the price function therefore needs a steady state where those coincide.
    ss = compute_steady_state(spec; r_bounds=(-0.01, 0.04), max_iter=80)
    Th = 20

    hh = HetBlock(spec, ss; inputs=[:r, :w], outputs=[:A, :C], name=:household)
    @test hh isa HetBlock{Float64}
    @test hh.ss_inputs[:r] == ss.prices[:r]
    @test hh.ss_outputs[:A] ≈ dot(vec(ss.policies[:savings]),
                                  MacroEconometricModels._normalized_distribution(ss))

    # The block Jacobian IS the fake-news Jacobian — no reimplementation drift.
    Jb = block_jacobian(hh, Th)
    @test Set(keys(Jb)) == Set([(:A, :r), (:A, :w), (:C, :r), (:C, :w)])
    @test Jb[(:A, :r)] == MacroEconometricModels._ssj_jacobian(
        ss, spec.individual, spec.grid, spec.income, :r, :A; T_horizon=Th, dx=hh.dx)

    # Nonlinear path evaluation reproduces the steady state on a flat input path.
    flat = Dict(:r => fill(ss.prices[:r], Th), :w => fill(ss.prices[:w], Th))
    base = MacroEconometricModels._block_evaluate(hh, flat, Th)
    @test maximum(abs, base[:A] .- hh.ss_outputs[:A]) < 1e-6
    @test maximum(abs, base[:C] .- hh.ss_outputs[:C]) < 1e-6

    # INDEPENDENT ORACLE: the fake-news Jacobian must equal a central finite
    # difference of the *nonlinear* transition path (backward EGM + forward Young
    # histogram) — two implementations sharing no code beyond the EGM step. The
    # anticipation columns (t < s) are the ones the pre-#226 brute force got wrong.
    hh_fine = HetBlock(spec, ss; inputs=[:r, :w], outputs=[:A], dx=1e-5)
    J_fine = block_jacobian(hh_fine, Th)[(:A, :r)]
    fd_step = 1e-6
    for s in (1, 4, 9)
        pp = deepcopy(flat); pp[:r][s] += fd_step
        pm = deepcopy(flat); pm[:r][s] -= fd_step
        col = (MacroEconometricModels._block_evaluate(hh_fine, pp, Th)[:A] .-
               MacroEconometricModels._block_evaluate(hh_fine, pm, Th)[:A]) ./ (2fd_step)
        @test maximum(abs, col .- J_fine[:, s]) < 1e-5 * maximum(abs, J_fine[:, s])
    end
    @test any(abs(J_fine[t, s]) > 1e-8 for t in 1:Th for s in (t+1):Th)   # anticipation

    # ── Three-block DAG: firm (lagged capital) → household → asset market ────
    alpha = spec.aggregate_spec.param_values[:alpha]
    delta = spec.aggregate_spec.param_values[:delta]
    K_ss = ss.aggregates[:K]
    firm = SimpleBlock(
        x -> [alpha * x[2] * x[1]^(alpha - 1) - delta,
              (1 - alpha) * x[2] * x[1]^alpha,
              x[2] * x[1]^alpha];
        inputs=[:K, :Z], outputs=[:r, :w, :Y],
        lags=Dict(:K => [1]),
        ss_inputs=Dict(:K => K_ss, :Z => 1.0), name=:firm)
    @test firm.ss_outputs[:r] ≈ ss.prices[:r] atol=1e-6
    @test firm.ss_outputs[:w] ≈ ss.prices[:w] atol=1e-6

    hh1 = HetBlock(spec, ss; inputs=[:r, :w], outputs=[:A], name=:household)
    mkt = SimpleBlock(x -> [x[1] - x[2]];
                      inputs=[:A, :K], outputs=[:asset_mkt],
                      ss_inputs=Dict(:A => hh1.ss_outputs[:A], :K => K_ss),
                      name=:asset_market)
    dag = combine_blocks(firm, hh1, mkt; name=:ks_dag)
    @test [b.name for b in dag.blocks] == [:firm, :household, :asset_market]
    @test dag.exogenous == [:K, :Z]
    @test dag.endogenous == [:r, :w, :Y, :A, :asset_mkt]

    # HISTORICAL NOTE: the Krusell-Smith example used to truncate its asset grid
    # (~5.6% of mass pinned at a_max = 200, so ∫a'dμ exceeded ∫a dμ by ~1.7%) and
    # the asset market did NOT clear at the linearization point — this assertion
    # read `dag.ss_values[:asset_mkt] > 1e-3` and the GE assembler's target_tol
    # guard fired. The example now clears; the household block's ∫a'dμ and the
    # steady state's ∫a dμ agree to floating-point. The guard itself is still
    # covered independently on the toy DAG below.
    @test abs(dag.ss_values[:asset_mkt]) < 1e-9

    gej = ssj_jacobian(dag; unknowns=[:K], targets=[:asset_mkt], shocks=[:Z],
                       T_horizon=Th, target_tol=Inf)
    # Forward accumulation vs the chain rule computed by hand from block Jacobians.
    Jf = block_jacobian(firm, Th)
    Jh = block_jacobian(hh1, Th)
    dA_dK = Jh[(:A, :r)] * Jf[(:r, :K)] .+ Jh[(:A, :w)] * Jf[(:w, :K)]
    @test gej.curlyJ[:A][:K] ≈ dA_dK
    @test gej.H_U ≈ dA_dK .- Matrix(I, Th, Th)
    @test gej.curlyJ[:Y][:Z] ≈ Jf[(:Y, :Z)]

    dZ = Dict(:Z => [0.01 * 0.9^(t - 1) for t in 1:Th])
    r1 = ssj_irf(gej, dZ; residual=false)
    # The linearized clearing condition holds exactly whatever the steady-state wedge.
    @test maximum(abs, gej.H_U * r1.paths[:K] .+ gej.H_Z * dZ[:Z]) < 1e-8
    @test r1.paths[:K][1] > 0                    # positive TFP shock raises capital
    @test r1.paths[:r][1] ≈ alpha * 0.01 * K_ss^(alpha - 1) atol=1e-10  # K lagged ⇒ r_1 ← Z_1
    report(dag)                                   # display smoke tests
    report(gej)
    @test occursin("SSJModel", sprint(show, dag))
    @test occursin("SSJGEJacobian", sprint(show, gej))
    @test occursin("HetBlock", sprint(show, hh1))
    @test occursin("SimpleBlock", sprint(show, firm))
end

@testset "SSJ blocks — Huggett GE and second order" begin
    spec = _HUG_SPEC_M2; ss = _HUG_SS_M2       # reuse the shared cl=−2 SS
    Th = 40

    hh = HetBlock(spec, ss; inputs=[:r, :w], outputs=[:A], name=:household)
    bond = SimpleBlock(x -> [x[1]];
                       inputs=[:A], outputs=[:bond_mkt],
                       ss_inputs=Dict(:A => hh.ss_outputs[:A]), name=:bond_market)
    dag = combine_blocks(hh, bond; name=:huggett_dag)
    # Zero net supply: the Huggett steady state genuinely clears, so no warning.
    @test abs(dag.ss_values[:bond_mkt]) < 1e-3
    gej = ssj_jacobian(dag; unknowns=[:r], targets=[:bond_mkt], shocks=[:w],
                       T_horizon=Th, target_tol=1e-2)

    # The two-block DAG reproduces the hard-wired GE close of `_ssj_solve` exactly.
    J_ref_U = MacroEconometricModels._ssj_jacobian(ss, spec.individual, spec.grid,
                                                   spec.income, :r, :A; T_horizon=Th)
    J_ref_Z = MacroEconometricModels._ssj_jacobian(ss, spec.individual, spec.grid,
                                                   spec.income, :w, :A; T_horizon=Th)
    @test gej.H_U == J_ref_U
    @test gej.H_Z == J_ref_Z
    dw = [0.9^(t - 1) for t in 1:Th]
    @test ssj_irf(gej, Dict(:w => dw); residual=false).paths[:r] ≈ -(J_ref_U \ (J_ref_Z * dw))

    # Routing `solve(:ssj)` through the DAG must not start emitting the target guard:
    # for the zero-net-supply close the target level IS ss.excess_demand, already
    # reported by report(ss), so warning again on every solve is pure noise.
    logs, _ = Test.collect_test_logs() do
        solve(spec; method=:ssj, ss=ss, T_horizon=30, n_reduced=12)
    end
    @test !any(occursin("does not vanish in steady state", string(r.message)) for r in logs)

    # ── Second order ────────────────────────────────────────────────────────
    sigma = 0.02
    dZ = Dict(:w => [sigma * 0.9^(t - 1) for t in 1:Th])
    o1 = ssj_irf(gej, dZ)
    o2 = ssj_irf(gej, dZ; order=2)
    @test o2.order == 2
    @test haskey(o2.correction, :r) && haskey(o2.correction, :A)
    # Precautionary saving makes the block genuinely nonlinear: nonzero correction.
    @test maximum(abs, o2.correction[:r]) > 1e-10
    # By construction the target is zero to second order: 𝒥·dU² + ½D²H = 0. Scale
    # against one of the two cancelling terms — NOT against the first-order :A path,
    # which the GE solve itself drives to ~1e-17 (bond_mkt IS A here).
    cancel_scale = maximum(abs, gej.H_U * o2.correction[:r])
    @test cancel_scale > 1e-12
    @test maximum(abs, o2.correction[:bond_mkt]) < 1e-8 * cancel_scale
    # The honest accuracy measure: the nonlinear clearing residual must improve.
    @test o2.target_residual[:bond_mkt] < o1.target_residual[:bond_mkt]

    # dU² is O(σ²) while dU¹ is O(σ), so halving the shock halves the relative
    # correction — this is what "collapses onto the first order" means.
    ratios = Float64[]
    for s in (0.02, 0.01, 0.005)
        rr = ssj_irf(gej, Dict(:w => [s * 0.9^(t - 1) for t in 1:Th]);
                     order=2, residual=false)
        push!(ratios, maximum(abs, rr.correction[:r]) / maximum(abs, rr.first_order[:r]))
    end
    @test issorted(ratios; rev=true)
    @test 1.6 < ratios[1] / ratios[2] < 2.4
    @test 1.6 < ratios[2] / ratios[3] < 2.4

    report(o2)                                    # display smoke test
    @test occursin("SSJImpulseResponse", sprint(show, o2))
end

@testset "SSJ blocks — validation and errors" begin
    ok = SimpleBlock(x -> [x[1]]; inputs=[:a], outputs=[:b],
                     ss_inputs=Dict(:a => 1.0), name=:ok)

    # Construction-time validation
    @test_throws ArgumentError SimpleBlock(x -> [x[1]]; inputs=Symbol[], outputs=[:b],
                                           ss_inputs=Dict{Symbol,Float64}())
    @test_throws ArgumentError SimpleBlock(x -> [x[1]]; inputs=[:a], outputs=Symbol[],
                                           ss_inputs=Dict(:a => 1.0))
    @test_throws ArgumentError SimpleBlock(x -> [x[1]]; inputs=[:a, :a], outputs=[:b],
                                           ss_inputs=Dict(:a => 1.0))
    @test_throws ArgumentError SimpleBlock(x -> [x[1]]; inputs=[:a], outputs=[:b],
                                           ss_inputs=Dict{Symbol,Float64}())    # missing SS
    @test_throws ArgumentError SimpleBlock(x -> [x[1]]; inputs=[:a], outputs=[:b],
                                           ss_inputs=Dict(:a => 1.0),
                                           lags=Dict(:q => [1]))                # unknown lag key
    @test_throws ArgumentError SimpleBlock(x -> [x[1]]; inputs=[:a], outputs=[:b, :c],
                                           ss_inputs=Dict(:a => 1.0))           # arity mismatch

    # DAG assembly
    @test_throws ArgumentError combine_blocks()
    dup = SimpleBlock(x -> [2 * x[1]]; inputs=[:a], outputs=[:b],
                      ss_inputs=Dict(:a => 1.0), name=:dup)
    @test_throws ArgumentError combine_blocks(ok, dup)                # duplicate output
    self = SimpleBlock(x -> [x[1]]; inputs=[:b], outputs=[:b],
                       ss_inputs=Dict(:b => 1.0), name=:self)
    @test_throws ArgumentError combine_blocks(self)                   # self loop
    back = SimpleBlock(x -> [x[1]]; inputs=[:b], outputs=[:a],
                       ss_inputs=Dict(:b => 1.0), name=:back)
    @test_throws ArgumentError combine_blocks(ok, back)               # cycle

    # Inconsistent steady state between producer and consumer is warned about.
    consumer = SimpleBlock(x -> [x[1]]; inputs=[:b], outputs=[:c],
                           ss_inputs=Dict(:b => 5.0), name=:consumer)
    @test_logs (:warn, r"inconsistent steady state") match_mode=:any begin
        combine_blocks(ok, consumer)
    end

    model = combine_blocks(ok; name=:tiny)
    @test_throws ArgumentError ssj_jacobian(model; unknowns=[:a], targets=[:b],
                                            shocks=[:a], T_horizon=4, target_tol=Inf)
    @test_throws ArgumentError ssj_jacobian(model; unknowns=[:b], targets=[:b],
                                            shocks=Symbol[], T_horizon=4, target_tol=Inf)
    @test_throws ArgumentError ssj_jacobian(model; unknowns=[:a], targets=[:a],
                                            shocks=Symbol[], T_horizon=4, target_tol=Inf)
    @test_throws ArgumentError ssj_jacobian(model; unknowns=Symbol[], targets=Symbol[],
                                            shocks=Symbol[], T_horizon=4, target_tol=Inf)
    @test_throws ArgumentError ssj_jacobian(model; unknowns=[:a], targets=[:b],
                                            shocks=Symbol[], T_horizon=1, target_tol=Inf)
    # A non-vanishing target level is warned about (ok's steady-state :b is 1.0).
    @test_logs (:warn, r"does not vanish in steady state") match_mode=:any begin
        ssj_jacobian(model; unknowns=[:a], targets=[:b], shocks=Symbol[], T_horizon=4)
    end

    # A model with no shocks must still assemble and solve (typed empty H_Z path).
    gej = ssj_jacobian(model; unknowns=[:a], targets=[:b], shocks=Symbol[],
                       T_horizon=4, target_tol=Inf)
    @test size(gej.H_Z) == (4, 0)
    @test all(iszero, ssj_irf(gej, Dict{Symbol,Vector{Float64}}()).paths[:a])
    @test_throws ArgumentError ssj_irf(gej, Dict(:zz => zeros(4)))          # undeclared shock
    @test_throws ArgumentError ssj_irf(gej, Dict{Symbol,Vector{Float64}}(); order=3)
    @test_throws ArgumentError ssj_irf(gej, Dict{Symbol,Vector{Float64}}();
                                       order=2, fd_step=0.0)

    # A singular clearing Jacobian is reported, not silently inverted.
    dead = SimpleBlock(x -> [0.0 * x[1]]; inputs=[:a], outputs=[:b],
                       ss_inputs=Dict(:a => 1.0), name=:dead)
    @test_throws ErrorException ssj_jacobian(combine_blocks(dead);
                                             unknowns=[:a], targets=[:b],
                                             shocks=Symbol[], T_horizon=4,
                                             target_tol=Inf)

    # HetBlock validation
    spec = _HUG_SPEC_M2; ss = _HUG_SS_M2
    @test_throws ArgumentError HetBlock(spec, ss; inputs=[:not_a_price], outputs=[:A])
    @test_throws ArgumentError HetBlock(spec, ss; inputs=[:r], outputs=[:nonsense])
    @test_throws ArgumentError HetBlock(spec, ss; inputs=Symbol[], outputs=[:A])
    @test_throws ArgumentError HetBlock(spec, ss; inputs=[:r], outputs=Symbol[])
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 27 (#353/T254): DCEGM — discrete-continuous choice
# ─────────────────────────────────────────────────────────────────────────────

@testset "DCEGM upper envelope" begin
    UE = MacroEconometricModels._upper_envelope
    SEG = MacroEconometricModels._monotone_segments

    @test SEG([1.0, 2.0, 3.0]) == [1:3]
    @test SEG([1.0, 2.0, 3.0, 2.5, 3.5]) == [1:3, 4:5]
    @test SEG([1.0, 2.0, 1.5]) == [1:2]                  # trailing single point dropped
    @test isempty(SEG(Float64[]))

    # A monotone correspondence passes through untouched.
    M1 = [1.0, 2.0, 3.0]; c1 = [0.5, 1.0, 1.5]; v1 = [0.0, 1.0, 2.0]
    Me, ce, ve, nk = UE(M1, c1, v1)
    @test Me == M1 && ce == c1 && ve == v1 && nk == 0

    # Two branches that genuinely CROSS inside their overlap.
    #   A: v = M           on [1, 5]
    #   B: v = 2M − 3.5    on [2, 5]   ⇒  v_A = v_B ⟺ M = 3.5, strictly between knots
    Ma = [1.0, 2.0, 3.0, 4.0, 5.0]; ca = [0.5, 1.0, 1.5, 2.0, 2.5]; va = [1.0, 2.0, 3.0, 4.0, 5.0]
    Mb = [2.0, 3.0, 4.0, 5.0];      cb = [9.0, 9.5, 10.0, 10.5];    vb = [0.5, 2.5, 4.5, 6.5]
    Me, ce, ve, nk = UE(vcat(Ma, Mb), vcat(ca, cb), vcat(va, vb))
    @test nk == 1
    @test all(diff(Me) .> 0)                              # strictly increasing output
    k = findfirst(i -> Me[i+1] == nextfloat(Me[i]), 1:(length(Me)-1))
    @test k !== nothing
    @test Me[k] ≈ 3.5                                     # exact crossing, not a grid point
    @test ve[k] ≈ ve[k+1] ≈ 3.5                           # value is continuous at a kink
    @test ce[k] ≈ 1.75 && ce[k+1] ≈ 9.75                  # consumption jumps
    # Defining property: the envelope dominates every branch everywhere it is defined.
    for (m, v) in zip(Me, ve)
        for (Ms, vs) in ((Ma, va), (Mb, vb))
            Ms[1] <= m <= Ms[end] || continue
            @test v >= MacroEconometricModels._seg_interp(Ms, vs, m) - 1e-12
        end
    end

    # A crossing that lands exactly ON a knot is still a kink: the branches tie at
    # M = 3 and consumption jumps immediately above it. Rounding it away would lose
    # the switching threshold entirely.
    vb_knot = [1.0, 3.0, 5.0, 7.0]                        # B: v = 2M − 3 ⇒ tie at M = 3
    Me, ce, ve, nk = UE(vcat(Ma, Mb), vcat(ca, cb), vcat(va, vb_knot))
    @test nk == 1
    @test all(diff(Me) .> 0)
    k = findfirst(i -> Me[i+1] == nextfloat(Me[i]), 1:(length(Me)-1))
    @test k !== nothing && Me[k] == 3.0
    @test ve[k] ≈ ve[k+1] ≈ 3.0
    @test ce[k] ≈ 1.5 && ce[k+1] ≈ 9.5

    # A switch at a SUPPORT BOUNDARY is not a crossing: branch B starts already
    # dominating, so there is no interior kink to insert.
    Mc_ = [1.0, 2.0, 3.0, 2.5, 3.5, 4.5]
    cc_ = [0.5, 1.0, 1.5, 0.2, 0.3, 0.4]
    vc_ = [0.0, 1.0, 2.0, 3.0, 3.4, 3.8]
    Me, ce, ve, nk = UE(Mc_, cc_, vc_)
    @test nk == 0
    @test all(diff(Me) .> 0)
    @test ve[findfirst(≈(2.5), Me)] ≈ 3.0                 # the dominating branch is kept

    @test_throws ArgumentError UE([1.0, 2.0], [1.0], [1.0, 2.0])
end

@testset "DCEGM retirement model" begin
    prob = dcegm_retirement_model(; n_periods=6, beta=0.98, R=1.0, wage=20.0,
                                  disutility=1.0, a_max=60.0, n_a=250)
    @test prob isa DCEGMProblem{Float64}
    @test prob.options == [:retire, :work] && prob.absorbing == [true, false]
    sol = dcegm_solve(prob)
    @test sol isa DCEGMSolution{Float64}
    @test sol.converged && sol.n_periods == 6

    # ANALYTIC ORACLE: once retired (absorbing, no pension, R = 1, log utility) the
    # problem is deterministic cake-eating with the closed form c_t = M / Σ_{k≤T−t} β^k.
    for t in (6, 5, 3, 1), Mt in (5.0, 20.0, 45.0)
        annuity = sum(0.98^k for k in 0:(6 - t))
        @test dcegm_policy(sol, t, 1, 1, Mt)[1] ≈ Mt / annuity rtol=1e-12
    end

    # The discrete choice makes the WORKING branch non-concave: the envelope deletes
    # secondary segments and inserts switching thresholds. Retirement is absorbing,
    # so its own branch is concave and needs none.
    @test sum(sol.n_kinks[:, 2, :]) > 0
    @test sum(sol.n_kinks[:, 1, :]) == 0

    # At every inserted kink the two value branches coincide while consumption jumps —
    # the defining property of an upper-envelope crossing.
    for t in 1:6, d in 1:2
        Mv = sol.M[t, d, 1]; cv = sol.c[t, d, 1]; vv = sol.v[t, d, 1]
        @test all(diff(Mv) .> 0)
        for i in 1:(length(Mv) - 1)
            Mv[i+1] == nextfloat(Mv[i]) || continue
            @test vv[i] ≈ vv[i+1] rtol=1e-8
            @test abs(cv[i] - cv[i+1]) > 1e-6
        end
    end

    # ── INDEPENDENT ORACLE: dense-grid backward-induction VFI on the same model ──
    # No EGM, no envelope, no Euler equation — just brute-force maximization.
    function _vfi_retirement(; T_end, beta, R, wage, delta, Mmax, nM, nC)
        Mg = collect(range(1e-4, Mmax; length=nM))
        V = fill(-Inf, T_end, nM, 2); C = zeros(T_end, nM, 2); D = zeros(Int, T_end, nM, 2)
        u(c, d) = c > 0 ? log(c) - (d == 2 ? delta : 0.0) : -Inf
        for i in 1:nM, dp in 1:2
            V[T_end, i, dp] = u(Mg[i], 1); C[T_end, i, dp] = Mg[i]; D[T_end, i, dp] = 1
        end
        for t in (T_end-1):-1:1, i in 1:nM, dp in 1:2
            best = -Inf; bc = 0.0; bd = 0
            for d in (dp == 1 ? (1:1) : (1:2))
                inc = d == 2 ? wage : 0.0
                for k in 0:(nC-1)
                    c = Mg[i] * (k + 1) / nC
                    Mn = R * (Mg[i] - c) + inc
                    Vn = if Mn <= Mg[1]; V[t+1, 1, d]
                         elseif Mn >= Mg[end]; V[t+1, end, d]
                         else
                             q = searchsortedfirst(Mg, Mn) - 1
                             w = (Mn - Mg[q]) / (Mg[q+1] - Mg[q])
                             (1 - w) * V[t+1, q, d] + w * V[t+1, q+1, d]
                         end
                    val = u(c, d) + beta * Vn
                    val > best && (best = val; bc = c; bd = d)
                end
            end
            V[t, i, dp] = best; C[t, i, dp] = bc; D[t, i, dp] = bd
        end
        return Mg, C, D
    end
    Mg, C_vfi, D_vfi = _vfi_retirement(; T_end=6, beta=0.98, R=1.0, wage=20.0,
                                      delta=1.0, Mmax=60.0, nM=200, nC=600)
    step = Mg[2] - Mg[1]

    for t in 2:4
        errs = Float64[]; mism = 0
        for (i, m) in enumerate(Mg)
            m < 0.5 && continue
            d = argmax(dcegm_choice_probabilities(sol, t, 2, 1, m))
            d != D_vfi[t, i, 2] && (mism += 1)
            push!(errs, abs(dcegm_policy(sol, t, d, 1, m)[1] - C_vfi[t, i, 2]) /
                        max(C_vfi[t, i, 2], 1e-8))
        end
        @test mism == 0                                        # discrete choice agrees
        @test sort(errs)[cld(length(errs), 2)] < 2e-3          # median within VFI resolution
        # Large disagreements occur only where the policy is genuinely discontinuous:
        # a grid-based VFI cannot resolve a jump, DCEGM locates it exactly.
        @test count(>(1e-2), errs) <= sum(sol.n_kinks[t, :, :]) + 1
    end

    # Retirement threshold vs the VFI switch point, within one oracle grid step.
    for t in (4, 5)
        thr = dcegm_threshold(sol, t, 2, 1; M_lo=0.5, M_hi=60.0)
        idx = findlast(i -> D_vfi[t, i, 2] == 2, 1:length(Mg))
        @test idx !== nothing
        @test isfinite(thr)
        @test abs(thr - Mg[idx]) <= 2 * step
    end
    # Early in life the worker never retires on this bracket — honestly reported as NaN.
    @test isnan(dcegm_threshold(sol, 2, 2, 1; M_lo=0.5, M_hi=60.0))
    @test all(D_vfi[2, i, 2] == 2 for i in 1:length(Mg))   # …and the oracle agrees
    # Retirement is absorbing, so there is no two-option choice left to threshold.
    @test_throws ArgumentError dcegm_threshold(sol, 3, 1, 1; M_lo=1.0, M_hi=10.0)
    @test_throws ArgumentError dcegm_threshold(sol, 3, 2, 1; M_lo=10.0, M_hi=1.0)

    report(sol)                                                # display smoke tests
    @test occursin("DCEGMSolution", sprint(show, sol))
    @test occursin("DCEGMProblem", sprint(show, prob))
end

@testset "DCEGM taste shocks" begin
    base = dcegm_solve(dcegm_retirement_model(; n_periods=5, beta=0.98, R=1.0,
                                              wage=20.0, disutility=1.0,
                                              a_max=60.0, n_a=200))
    Ms = collect(2.0:2.0:55.0)
    devs = Float64[]; spreads = Float64[]
    for lam in (1.0, 0.05, 0.01, 0.002)
        s = dcegm_solve(dcegm_retirement_model(; n_periods=5, beta=0.98, R=1.0,
                                               wage=20.0, disutility=1.0,
                                               a_max=60.0, n_a=200,
                                               taste_shock_scale=lam))
        push!(devs, maximum(abs(dcegm_policy(s, 2, 2, 1, m)[1] -
                                dcegm_policy(base, 2, 2, 1, m)[1]) for m in Ms))
        # Mean distance of the choice probabilities from the deterministic 0/1 rule.
        # The MAXIMUM is the wrong statistic: at the indifference point the
        # probabilities are 1/2 for every λ, so only the *measure* of the interior
        # region shrinks, not its peak.
        push!(spreads, sum(minimum(dcegm_choice_probabilities(s, 3, 2, 1, m))
                           for m in Ms) / length(Ms))
    end
    # The smoothed solution collapses onto the deterministic upper envelope as λ → 0.
    @test issorted(devs; rev=true)
    @test devs[1] > 1.0                                   # λ = 1 genuinely differs
    @test devs[end] < 0.01
    @test issorted(spreads; rev=true)
    @test spreads[end] < 1e-3

    s = dcegm_solve(dcegm_retirement_model(; n_periods=5, a_max=60.0, n_a=150,
                                           taste_shock_scale=0.5))
    p = dcegm_choice_probabilities(s, 3, 2, 1, 30.0)
    @test length(p) == 2 && sum(p) ≈ 1.0 && all(p .>= 0)
    # After retiring, work is infeasible: probability exactly zero, not merely small.
    pr = dcegm_choice_probabilities(s, 3, 1, 1, 30.0)
    @test pr == [1.0, 0.0]
end

@testset "DCEGM distribution and simulation" begin
    prob = dcegm_retirement_model(; n_periods=7, beta=0.98, R=1.02, wage=20.0,
                                  disutility=0.8, sigma=0.15, n_shocks=3,
                                  a_max=80.0, n_a=150)
    @test length(prob.income_process.states) == 3
    @test sum(prob.income_process.stationary_dist) ≈ 1.0
    @test dot(prob.income_process.stationary_dist, prob.income_process.states) ≈ 1.0 rtol=1e-6
    sol = dcegm_solve(prob)

    grid = collect(range(0.01, 80.0; length=120))
    dist = dcegm_simulate(sol, grid)
    @test dist isa DCEGMDistribution{Float64}
    @test dist.n_periods == 7
    # The Young lottery splits off-grid landings between neighbours, so mass is exact.
    for t in 1:7
        @test sum(@view dist.dist[t, :, :, :]) ≈ 1.0 atol=1e-12
        @test sum(@view dist.shares[t, :]) ≈ 1.0 atol=1e-12
    end
    @test all(dist.dist .>= 0)
    # Retirement is absorbing, so its share can only rise with age.
    @test issorted(dist.shares[:, 1])
    @test dist.shares[1, 2] ≈ 1.0                       # everyone starts working
    @test all(isfinite, dist.consumption) && all(dist.consumption .> 0)
    @test all(dist.assets .>= -1e-12)
    report(dist)                                        # display smoke test
    @test occursin("DCEGMDistribution", sprint(show, dist))

    # Custom initial condition: all mass at one node, everyone already retired.
    init = zeros(length(grid), 3)
    init[60, :] .= prob.income_process.stationary_dist
    d2 = dcegm_simulate(sol, grid; init=init, init_option=:retire, n_periods=4)
    @test d2.n_periods == 4
    @test all(d2.shares[:, 1] .≈ 1.0)                   # absorbing: nobody returns to work
    @test sum(@view d2.dist[1, :, :, :]) ≈ 1.0

    @test_throws ArgumentError dcegm_simulate(sol, [3.0, 1.0, 2.0])
    @test_throws ArgumentError dcegm_simulate(sol, [1.0])
    @test_throws ArgumentError dcegm_simulate(sol, grid; n_periods=0)
    @test_throws ArgumentError dcegm_simulate(sol, grid; n_periods=99)
    @test_throws ArgumentError dcegm_simulate(sol, grid; init_option=:nope)
    @test_throws ArgumentError dcegm_simulate(sol, grid; init=zeros(3, 3))
end

@testset "DCEGM infinite horizon and validation" begin
    # Stationary policy: a pension keeps the retired branch finite at the constraint.
    p = dcegm_retirement_model(; n_periods=0, beta=0.95, R=1.01, wage=5.0,
                               disutility=0.5, a_max=40.0, n_a=150, pension=1.0)
    s = dcegm_solve(p; max_iter=300, tol=1e-7)
    @test s.converged
    @test s.iterations > 1 && s.sup_diff < 1e-7
    @test s.n_periods == 1
    # A stationary solution can be simulated for any number of periods.
    d = dcegm_simulate(s, collect(range(0.01, 40.0; length=80)); n_periods=12)
    @test d.n_periods == 12
    @test all(sum(@view d.dist[t, :, :, :]) ≈ 1.0 for t in 1:12)

    # Non-convergence is reported, not silently accepted.
    s1 = dcegm_solve(p; max_iter=1, tol=1e-12)
    @test !s1.converged && s1.iterations == 1

    # ── Constructor validation ──────────────────────────────────────────────
    inc = rouwenhorst(0.5, 0.1, 2)
    base = (utility=(c, d) -> log(c), utility_prime=(c, d) -> 1 / c,
            utility_prime_inv=(m, d) -> 1 / m, income=(d, j) -> 1.0,
            income_process=inc)
    @test_throws ArgumentError DCEGMProblem(; beta=0.95, R=1.0, base...,
        options=Symbol[], absorbing=Bool[], asset_grid=[0.0, 1.0])
    @test_throws ArgumentError DCEGMProblem(; beta=0.95, R=1.0, base...,
        options=[:a, :b], absorbing=[true], asset_grid=[0.0, 1.0])
    @test_throws ArgumentError DCEGMProblem(; beta=0.95, R=1.0, base...,
        options=[:a, :a], absorbing=[true, false], asset_grid=[0.0, 1.0])
    @test_throws ArgumentError DCEGMProblem(; beta=0.95, R=1.0, base...,
        options=[:a], absorbing=[false], asset_grid=[0.0])            # too few points
    @test_throws ArgumentError DCEGMProblem(; beta=0.95, R=1.0, base...,
        options=[:a], absorbing=[false], asset_grid=[1.0, 0.0])       # unsorted
    @test_throws ArgumentError DCEGMProblem(; beta=0.95, R=1.0, base...,
        options=[:a], absorbing=[false], asset_grid=[0.5, 1.0])       # ≠ credit limit
    @test_throws ArgumentError DCEGMProblem(; beta=1.5, R=1.0, base...,
        options=[:a], absorbing=[false], asset_grid=[0.0, 1.0])       # β outside (0,1)
    @test_throws ArgumentError DCEGMProblem(; beta=0.95, R=1.0, base...,
        options=[:a], absorbing=[false], asset_grid=[0.0, 1.0], n_periods=-1)
    @test_throws ArgumentError DCEGMProblem(; beta=0.95, R=1.0, base...,
        options=[:a], absorbing=[false], asset_grid=[0.0, 1.0], taste_shock_scale=-1)
    @test_throws ArgumentError dcegm_retirement_model(; n_shocks=0)
    @test_throws ArgumentError dcegm_retirement_model(; curvature=0.5)

    # A degenerate problem leaves too few usable grid points and says so.
    bad = DCEGMProblem(; beta=0.95, R=1.0,
        utility=(c, d) -> c > 0 ? log(c) : -Inf, utility_prime=(c, d) -> c > 0 ? 1 / c : Inf,
        utility_prime_inv=(m, d) -> m > 0 ? 1 / m : Inf, income=(d, j) -> 0.0,
        options=[:only], absorbing=[true], asset_grid=[0.0, 1.0],
        income_process=inc, n_periods=3)
    @test_throws ErrorException dcegm_solve(bad)
end


# ─────────────────────────────────────────────────────────────────────────────
# Winberry (2018) parametric distribution dynamics (#356/T257)
# ─────────────────────────────────────────────────────────────────────────────

# Small Aiyagari spec: the Winberry end-to-end tests solve TWO steady states and
# TWO linearizations, so the shipped 200x7 examples would dominate this file.
function _win_small_spec(; distribution::Symbol=:young, n_a::Int=80, n_e::Int=3)
    u, up, upi = MacroEconometricModels._crra_utility(1.0)
    income = MacroEconometricModels._unit_mean_lognormal_income(0.90, 0.30, n_e)
    grid = HAGrid(; assets=(0.0, 300.0, n_a), income_states=n_e, grid_type=:geometric)
    ip = IndividualProblem{Float64}(u, up, upi, 0.99,
                                    MacroEconometricModels._ks_budget,
                                    [0.0], nothing, 1)
    agg = MacroEconometricModels._minimal_agg_spec(; alpha=0.36, delta=0.025)
    aggregation = Pair{Symbol,Function}[:K => MacroEconometricModels._agg_var1]
    het = Dict{Symbol,Float64}(:alpha => 0.36, :delta => 0.025, :Z => 1.0, :L => 1.0)
    return HADSGESpec{Float64}(agg, ip, income, grid, aggregation, het;
                                distribution=distribution)
end

@testset "Winberry parametric density (#356/T257)" begin

    @testset "Gauss-Legendre and composite quadrature are exact" begin
        # A k-point Gauss-Legendre rule integrates polynomials of degree 2k-1 exactly.
        for k in 2:6
            x, w = MacroEconometricModels._gauss_legendre(Float64, k)
            @test length(x) == k && length(w) == k
            @test sum(w) ≈ 2.0 atol=1e-14
            for d in 0:(2k - 1)
                exact = iseven(d) ? 2 / (d + 1) : 0.0
                @test sum(w .* x .^ d) ≈ exact atol=1e-12
            end
        end
        # Composite rule on arbitrary (unequal) segments: same exactness, and the
        # weights integrate the domain width.
        edges = [0.0, 0.3, 1.7, 5.0]
        nodes, wts = MacroEconometricModels._composite_quadrature(edges, 4)
        @test length(nodes) == 3 * 4
        @test sum(wts) ≈ 5.0 atol=1e-12
        for d in 0:7
            @test sum(wts .* nodes .^ d) ≈ 5.0^(d + 1) / (d + 1) atol=1e-9
        end
        # Grid-derived rule inherits the asset grid as its segment edges.
        g = HAGrid(; assets=(0.0, 50.0, 40), income_states=2)
        nq, wq = winberry_quadrature(g; n_quad=3)
        @test length(nq) == 39 * 3
        @test sum(wq) ≈ 50.0 atol=1e-10
        @test all(g.grids[1][1] .<= nq .<= g.grids[1][end])
        g2 = HAGrid(; liquid=(0.0, 5.0, 10), illiquid=(0.0, 5.0, 10), income_states=2)
        @test_throws ArgumentError winberry_quadrature(g2)
    end

    @testset "analytic oracles: the max-entropy fit IS the known density" begin
        # (a) Matching mean 0 and variance 1 on a wide symmetric interval must return
        #     the Gaussian exactly: g ∝ exp(−z²/2), i.e. λ = (0, −1/2).
        pd = fit_parametric_density([0.0, 1.0]; bounds=(-8.0, 8.0),
                                    n_segments=200, n_quad=6)
        @test pd.converged
        @test pd.lambda[1] ≈ 0.0 atol=1e-10
        @test pd.lambda[2] ≈ -0.5 atol=1e-7
        @test pd.residual < 1e-10
        for a in (-2.0, -0.5, 0.0, 1.0, 2.5)
            @test parametric_density(pd, a) ≈ exp(-a^2 / 2) / sqrt(2π) rtol=1e-6
        end

        # (b) The exponential distribution with rate 1 has centered moments
        #     (1, 1, 2, 9); the four-moment max-entropy fit must recover exp(−a)
        #     POINTWISE, not merely match the moments. In standardized coordinates
        #     z = a − 1, so the answer is λ = (−1, 0, 0, 0).
        pd4 = fit_parametric_density([1.0, 1.0, 2.0, 9.0]; bounds=(0.0, 40.0),
                                     n_segments=400, n_quad=6, tol=1e-12)
        # `tol` is RELATIVE to the target scale since #514, so the flag is portable
        # on the ill-conditioned four-moment basis again. Measured residual 4.1e-15
        # against an effective tolerance of 9e-12 — 2177x of headroom.
        @test pd4.converged
        @test pd4.residual < 1e-6
        @test pd4.lambda[1] ≈ -1.0 atol=1e-8
        @test all(abs.(pd4.lambda[2:end]) .< 1e-8)
        for a in (0.0, 0.25, 1.0, 2.0, 5.0)
            @test parametric_density(pd4, a) ≈ exp(-a) rtol=1e-6
        end
    end

    @testset "moment round trip (fit ∘ moments = identity)" begin
        nodes, wts = MacroEconometricModels._composite_quadrature(
            collect(range(-6.0, 12.0; length=301)), 5)
        # `converged` used to compare an ABSOLUTE residual against `tol`, which the
        # four-moment basis (Hessian cond ~1e8) could not meet portably — its
        # residual is 2.8e-10 here against a 1e-10 request, so the flag was asserted
        # only for the well-conditioned bases. Since #514 the test is relative to the
        # target scale max(1, max|mu|), which is 3.5 for this basis, so 2.8e-10 sits
        # inside an effective 3.5e-10 and every basis can assert the flag again.
        for targets in ([2.0, 4.0], [2.0, 4.0, 3.0], [1.0, 2.0, 1.5, 14.0])
            pd = MacroEconometricModels._fit_parametric_density(
                copy(targets), nodes, wts; tol=1e-10)
            @test pd.converged
            @test pd.residual < 1e-6            # a hard bound for every basis
            @test parametric_moments(pd, nodes, wts) ≈ targets rtol=1e-7
            # The density integrates to one over the reference interval.
            @test sum(wts .* [parametric_density(pd, a) for a in nodes]) ≈ 1.0 atol=1e-10
        end
    end

    @testset "#514: convergence is relative to the target scale, not absolute" begin
        MEM = MacroEconometricModels
        # The residual is a moment mismatch in STANDARDIZED units, so an absolute
        # threshold demands more relative precision the larger the targets are --
        # and they are largest exactly where the basis is worst conditioned.
        for (mom, want) in (([0.0, 1.0], 1.0), ([2.0, 4.0, 3.0], 1.0),
                            ([1.0, 2.0, 1.5, 14.0], 3.5), ([1.0, 1.0, 2.0, 9.0], 9.0))
            _, _, mu = MEM._standardized_targets(collect(Float64, mom))
            @test max(1.0, maximum(abs, mu)) ≈ want
        end

        # Every basis converges, including the two four-moment ones that could not
        # meet an absolute tolerance.
        pd4 = fit_parametric_density([1.0, 1.0, 2.0, 9.0]; bounds=(0.0, 40.0),
                                     n_segments=400, n_quad=6, tol=1e-12)
        @test pd4.converged
        @test pd4.residual < 1e-12 * 9.0        # inside the RELATIVE tolerance

        # A tolerance no arithmetic can meet still converges, because below
        # sqrt(eps) relative the residual is gradient noise rather than a mismatch
        # the solve could act on. The fit is fully accurate there.
        pd_floor = fit_parametric_density([1.0, 1.0, 2.0, 9.0]; bounds=(0.0, 40.0),
                                          n_segments=400, n_quad=6, tol=1e-30)
        @test pd_floor.converged
        @test pd_floor.residual < sqrt(eps(Float64)) * 9.0
        @test pd_floor.lambda[1] ≈ -1.0 atol=1e-8
        for a in (0.0, 1.0, 5.0)
            @test parametric_density(pd_floor, a) ≈ exp(-a) rtol=1e-6
        end

        # ... and that leniency must NOT rescue a fit that genuinely failed. Both
        # of these stall at an O(1) residual, nowhere near the floor.
        infeasible = fit_parametric_density([0.0, 1.0, 3.0, 1.0]; bounds=(-8.0, 8.0),
                                            tol=1e-30)
        @test !infeasible.converged
        @test infeasible.residual > 1.0
        starved = fit_parametric_density([1.0, 1.0, 2.0, 9.0]; bounds=(0.0, 40.0),
                                         n_segments=400, n_quad=6, max_iter=1)
        @test !starved.converged
        @test starved.residual > 1.0
    end

    @testset "analytic gradient/Hessian match ForwardDiff" begin
        # The fit uses closed-form derivatives of the log-normalizer rather than AD.
        # Cross-check both against ForwardDiff on the same objective (the docstring
        # promises this).
        FD = MacroEconometricModels.ForwardDiff
        nodes, wts = MacroEconometricModels._composite_quadrature(
            collect(range(-5.0, 9.0; length=201)), 5)
        moments = [1.5, 2.25, 1.2, 16.0]
        center, scale, mu = MacroEconometricModels._standardized_targets(moments)
        B = MacroEconometricModels._winberry_basis(nodes, center, scale, mu)
        F(lam) = begin
            u = B * lam
            umax = maximum(u)
            umax + log(sum(wts .* exp.(u .- umax)))
        end
        for lam in ([0.0, -0.5, 0.0, 0.0], [-0.4, -0.3, 0.05, -0.02])
            _, p = MacroEconometricModels._log_normalizer(B * lam, wts)
            grad_analytic = B' * p
            hess_analytic = B' * (B .* p) - grad_analytic * grad_analytic'
            @test grad_analytic ≈ FD.gradient(F, lam) rtol=1e-8
            @test hess_analytic ≈ FD.hessian(F, lam) rtol=1e-6
            # The Hessian is a covariance matrix, hence symmetric PSD.
            @test hess_analytic ≈ hess_analytic' atol=1e-12
            @test minimum(eigvals(Symmetric(hess_analytic))) > -1e-12
        end
        # ∇ = 0 is exactly the moment-matching condition: at the converged λ the
        # analytic gradient vanishes and the fitted density's central moments are
        # the targets.
        pd = MacroEconometricModels._fit_parametric_density(copy(moments), nodes, wts;
                                                            tol=1e-12)
        @test pd.converged
        _, p_star = MacroEconometricModels._log_normalizer(B * pd.lambda, wts)
        @test maximum(abs, B' * p_star) < 1e-11
        @test parametric_moments(pd, nodes, wts) ≈ moments rtol=1e-7
    end

    @testset "input validation" begin
        @test_throws ArgumentError fit_parametric_density([1.0]; bounds=(0.0, 1.0))
        @test_throws ArgumentError fit_parametric_density([1.0, -1.0]; bounds=(0.0, 1.0))
        @test_throws ArgumentError fit_parametric_density([1.0, 1.0])   # no quadrature
        @test_throws ArgumentError fit_parametric_density([1.0, 1.0]; nodes=[0.0, 1.0],
                                                          weights=[0.5])
        nodes, wts = MacroEconometricModels._composite_quadrature([0.0, 4.0], 5)
        @test_throws ArgumentError MacroEconometricModels._fit_parametric_density(
            [1.0, 1.0], nodes, wts; lambda_init=[0.0, 0.0, 0.0])
        @test_throws ArgumentError HADSGESpec{Float64}(
            _win_small_spec().aggregate_spec, _win_small_spec().individual,
            _win_small_spec().income, _win_small_spec().grid,
            _win_small_spec().aggregation, _win_small_spec().het_params;
            distribution=:histogram)
    end

    @testset "histogram ↔ moments" begin
        g = HAGrid(; assets=(0.0, 20.0, 60), income_states=2)
        a = g.grids[1]
        d = zeros(60, 2)
        d[:, 1] .= exp.(-a ./ 3); d[:, 2] .= exp.(-a ./ 8)
        d ./= sum(d)
        M, mass = winberry_moments(d, g; n_moments=4)
        @test size(M) == (2, 4)
        @test sum(mass) ≈ 1.0 atol=1e-12
        # Rows must equal the discrete conditional moments, computed independently.
        for j in 1:2
            p = d[:, j] ./ mass[j]
            m1 = sum(p .* a)
            @test M[j, 1] ≈ m1 rtol=1e-12
            for i in 2:4
                @test M[j, i] ≈ sum(p .* (a .- m1) .^ i) rtol=1e-10
            end
        end
        # Flattened input is accepted and gives the same answer.
        @test first(winberry_moments(vec(d), g; n_moments=4)) ≈ M
        @test_throws ArgumentError winberry_moments(d, g; n_moments=1)

        # Explicit tol for the same reason as above: under the 1e-10 default this fit
        # lands at 7.5e-11 (75% of the threshold), so `converged` is decided by
        # rounding rather than by the fit. The lambda vector and the reconstructed
        # histogram are identical to 4e-10 / 1e-11 across tol in [1e-10, 1e-6].
        fam = fit_winberry(d, g; n_moments=3, tol=1e-8)
        @test fam isa WinberryFamily{Float64}
        @test fam.converged
        @test length(fam.densities) == 2
        @test fam.n_moments == 3
        h = winberry_histogram(fam, g)
        @test length(h) == 120
        @test all(h .>= 0)
        @test sum(h) ≈ 1.0 atol=1e-12
        # Per-income-state mass is preserved by the rendering.
        for j in 1:2
            @test sum(h[((j - 1) * 60 + 1):(j * 60)]) ≈ mass[j] rtol=1e-10
        end
        # The rendered histogram carries roughly the family's own mean.
        @test sum(h .* repeat(a, 2)) ≈ sum(mass .* M[:, 1]) rtol=1e-3
    end

    # Remaining testsets solve shipped HA examples (SS + Reiter). Smoke CI
    # keeps the quadrature / max-entropy oracles above; Ubuntu still runs these.
    FAST && return

    @testset "moment fixed point is genuinely stationary and tracks Young" begin
        spec = _win_small_spec()
        ss = compute_steady_state(spec; grid_check=:none)
        a_pol = ss.policies[:savings]
        nodes, wts = winberry_quadrature(ss.grid; n_quad=4)
        M_young, mass_y = winberry_moments(ss.distribution, ss.grid; n_moments=3)
        K_young = sum(mass_y .* M_young[:, 1])
        @test K_young ≈ ss.aggregates[:K] rtol=1e-10

        errs = Float64[]
        for nm in (2, 3, 4)
            st = MacroEconometricModels._winberry_stationary(
                a_pol, ss.grid, ss.income; n_moments=nm)
            @test st.converged
            @test size(st.moments) == (3, nm)
            # Income-state masses are the ergodic distribution of the income chain.
            @test st.mass ≈ vec(sum(ss.distribution; dims=1)) rtol=1e-8
            # It really is a FIXED POINT: one more application of the law of motion
            # leaves it where it is (this is the property the linearization needs).
            M_next, _, _ = MacroEconometricModels._winberry_forward(
                st.moments, st.mass, a_pol, ss.grid, ss.income, nodes, wts;
                lambda_warm=st.lambdas)
            dev = MacroEconometricModels._winberry_to_state(
                M_next .- st.moments, MacroEconometricModels._winberry_scales(st.moments))
            @test maximum(abs, dev) < 1e-8
            K_w = sum(st.mass .* st.moments[:, 1])
            push!(errs, abs(K_w - K_young) / K_young)
            # A parametric solve started with no guess must find the same point.
            st_cold = MacroEconometricModels._winberry_stationary(
                a_pol, ss.grid, ss.income; n_moments=nm, M_init=nothing)
            @test st_cold.moments ≈ st.moments rtol=1e-6
        end
        # The reduction is accurate, and more moments do not make it worse.
        @test all(errs .< 0.10)
        @test errs[3] <= errs[1] + 1e-12
    end

    @testset "steady state with distribution=:winberry" begin
        spec_y = _win_small_spec()
        spec_w = _win_small_spec(; distribution=:winberry)
        @test spec_y.distribution === :young
        @test spec_w.distribution === :winberry
        ss_y = compute_steady_state(spec_y; grid_check=:none)
        ss_w = compute_steady_state(spec_w; grid_check=:none)

        # The equilibrium is cleared on the histogram either way, so prices and
        # aggregates are identical — only the extra parametric object differs.
        @test ss_y.parametric === nothing
        @test ss_w.parametric isa WinberryFamily{Float64}
        @test ss_w.prices[:r] == ss_y.prices[:r]
        @test ss_w.aggregates[:K] == ss_y.aggregates[:K]
        @test ss_w.parametric.converged
        @test ss_w.parametric.n_moments == 3
        @test length(ss_w.parametric.densities) == 3
        @test sum(ss_w.parametric.mass) ≈ 1.0 atol=1e-12
        @test all(pd -> pd.residual < 1e-9, ss_w.parametric.densities)

        # aggregates[:K_winberry] is the family's OWN stationary aggregate, so the
        # gap against :K is the reduction error — small but not zero.
        @test haskey(ss_w.aggregates, :K_winberry)
        @test !haskey(ss_y.aggregates, :K_winberry)
        rel = abs(ss_w.aggregates[:K_winberry] - ss_w.aggregates[:K]) / ss_w.aggregates[:K]
        @test 0 < rel < 0.10

        # n_moments is honoured, and more moments do not degrade the aggregate.
        ss_w5 = compute_steady_state(spec_w; grid_check=:none, n_moments=5)
        @test ss_w5.parametric.n_moments == 5
        rel5 = abs(ss_w5.aggregates[:K_winberry] - ss_w5.aggregates[:K]) / ss_w5.aggregates[:K]
        @test rel5 <= rel + 1e-10

        # `distribution=` on the call overrides the spec in both directions.
        @test compute_steady_state(spec_y; grid_check=:none,
                                   distribution=:winberry).parametric !== nothing
        @test compute_steady_state(spec_w; grid_check=:none,
                                   distribution=:young).parametric === nothing
        @test_throws ArgumentError compute_steady_state(spec_y; grid_check=:none,
                                                        distribution=:bogus)
    end

    @testset "Reiter linearization on the moment state" begin
        spec_y = _win_small_spec()
        spec_w = _win_small_spec(; distribution=:winberry)
        ss_y = compute_steady_state(spec_y; grid_check=:none)
        ss_w = compute_steady_state(spec_w; grid_check=:none)
        sol_y = solve(spec_y; method=:reiter, ss=ss_y)
        sol_w = solve(spec_w; method=:reiter, ss=ss_w)

        n_e = spec_w.grid.n_income
        # The distribution state is n_income × n_moments — far fewer than the
        # histogram's n_a × n_income, and fewer than the SVD reduction as well.
        @test sol_w.n_reduced == n_e * 3
        @test sol_w.n_reduced < sol_y.n_reduced
        @test sol_w.n_reduced < spec_w.grid.total_individual_states
        @test sol_w.method === :reiter
        @test is_determined(sol_w)
        @test maximum(abs, eigvals(sol_w.linear_solution.G1)) < 1.0
        @test 0.5 < sol_w.explained_variance <= 1.0

        # The reduction basis maps moment deviations back to the full histogram, so
        # distribution IRFs work unchanged — and every column is mass-preserving.
        @test size(sol_w.reduction_basis) == (spec_w.grid.total_individual_states,
                                              sol_w.n_reduced)
        @test maximum(abs, vec(sum(sol_w.reduction_basis; dims=1))) < 1e-8
        di = distribution_irf(sol_w, 6)
        @test size(di) == (spec_w.grid.n_points[1], n_e, 6)
        @test maximum(abs, di) > 0
        @test abs(sum(di[:, :, 1])) < 1e-8

        # Aggregate capital IRFs agree with the Young-based Reiter system. K is the
        # state just after the distribution block in both.
        function _agg_path(sol, H)
            G1 = sol.linear_solution.G1
            x = sol.linear_solution.impact[:, 1]
            out = zeros(H)
            for h in 1:H
                out[h] = x[sol.n_reduced + 1]
                x = G1 * x
            end
            return out
        end
        H = 20
        iy = _agg_path(sol_y, H)
        iw = _agg_path(sol_w, H)
        scale = maximum(abs, iy)
        @test scale > 0
        @test maximum(abs, iw .- iy) / scale < 0.05
        @test cor(iy, iw) > 0.999

        # More moments must not move the aggregate IRF much further away.
        sol_w5 = solve(spec_w; method=:reiter,
                       ss=compute_steady_state(spec_w; grid_check=:none, n_moments=5),
                       n_moments=5)
        @test sol_w5.n_reduced == n_e * 5
        @test maximum(abs, _agg_path(sol_w5, H) .- iy) / scale < 0.05
    end

    @testset "Huggett closure and the built-in examples" begin
        spec = load_ha_example(:huggett; distribution=:winberry)
        @test spec.distribution === :winberry
        @test load_ha_example(:huggett).distribution === :young
        @test load_ha_example(:krusell_smith; distribution=:winberry).distribution === :winberry
        ss = compute_steady_state(spec; grid_check=:none)
        @test ss.parametric isa WinberryFamily{Float64}
        # Huggett is zero net supply: the parametric family's own aggregate must
        # also be (nearly) zero, without ever having been told so.
        @test abs(ss.aggregates[:K_winberry]) < 1e-2
        sol = solve(spec; method=:reiter, ss=ss)
        @test sol.n_reduced == spec.grid.n_income * 3
        @test is_determined(sol)
        @test maximum(abs, eigvals(sol.linear_solution.G1)) < 1.0
    end

    @testset "display" begin
        spec = _win_small_spec(; distribution=:winberry)
        ss = compute_steady_state(spec; grid_check=:none)
        fam = ss.parametric
        str_f = sprint(show, fam)
        @test occursin("WinberryFamily", str_f)
        @test occursin("3 moments", str_f)
        @test occursin("converged=true", str_f)
        str_d = sprint(show, fam.densities[1])
        @test occursin("ParametricDensity", str_d)
        @test occursin("converged=true", str_d)
        # `report` writes to stdout; on Julia 1.12 redirect_stdout no longer accepts
        # an IOBuffer, so capture through a temporary file.
        out = mktemp() do path, f
            redirect_stdout(() -> report(ss), f)
            flush(f)
            read(path, String)
        end
        @test occursin("Winberry Parametric Family", out)
        @test occursin("K_winberry", out)
    end

end

end # @testset "HA-DSGE Types"

@testset "#508: Euler-error metric measures approximation, not round-trip" begin
    MEM = MacroEconometricModels

    @testset "analytic fixture (hand-computed residuals)" begin
        # Everything below is exactly computable with pen and paper.
        #   a_grid = [0,1,2,3,4], one income state, u = log c so u'(c) = 1/c,
        #   beta = 0.96, r = 0.02, c(a) = 1 + a (linear, so the interpolant is
        #   EXACT at midpoints), a'(a) = 1.5 for every a.
        # Then c(a') = 2.5 always, E[u'(c')] = 0.4, and
        #   resid(a) = |1 - beta(1+r)*0.4*(1+a)| = |1 - 0.39168*(1+a)|.
        a_grid = [0.0, 1.0, 2.0, 3.0, 4.0]
        c_pol = reshape([1.0, 2.0, 3.0, 4.0, 5.0], 5, 1)
        a_pol = reshape(fill(1.5, 5), 5, 1)
        ip = MEM.IndividualProblem{Float64}(
            log, c -> 1 / c, u -> 1 / u, 0.96,
            (a, e, p) -> (1 + p[:r]) * a + e, [0.0], nothing, 1)
        grid = MEM.HAGrid{Float64}([a_grid], [5], 1, 1, [(0.0, 4.0)], [:assets])
        income = MEM.IncomeProcess{Float64}(reshape([1.0], 1, 1), [1.0], [1.0], :income)
        prices = Dict(:r => 0.02, :w => 1.0)

        sn = MEM._euler_error_stats(c_pol, a_pol, ip, grid, income, prices; points=:nodes)
        sm = MEM._euler_error_stats(c_pol, a_pol, ip, grid, income, prices; points=:midpoints)

        @test sn.n_evaluated == 5 && sn.n_constrained == 0 && sn.n_offgrid == 0
        @test sm.n_evaluated == 4 && sm.n_constrained == 0 && sm.n_offgrid == 0
        @test sn.max ≈ log10(0.9584) atol = 1e-12
        @test sn.mean ≈ log10(0.505024) atol = 1e-12
        @test sm.max ≈ log10(0.76256) atol = 1e-12
        @test sm.mean ≈ log10(0.39168) atol = 1e-12
        @test sm.points === :midpoints && sn.points === :nodes
        # the scalar wrapper is exactly the `max` field
        @test MEM._compute_euler_error(c_pol, a_pol, ip, grid, income, prices) == sm.max
        @test_throws ArgumentError MEM._euler_error_stats(c_pol, a_pol, ip, grid,
                                                          income, prices; points=:bogus)

        # Off-grid cells are excluded and counted, not scored: with a' = 10 > a_max
        # the last node leaves the grid, and so does the last midpoint, whose
        # interpolated a' is (1.5 + 10)/2 = 5.75.
        a_off = reshape([1.5, 1.5, 1.5, 1.5, 10.0], 5, 1)
        on = MEM._euler_error_stats(c_pol, a_off, ip, grid, income, prices; points=:nodes)
        om = MEM._euler_error_stats(c_pol, a_off, ip, grid, income, prices; points=:midpoints)
        @test on.n_offgrid == 1 && on.n_evaluated == 4
        @test om.n_offgrid == 1 && om.n_evaluated == 3
        # The remaining cells are untouched, so the max is the surviving maximum.
        @test on.max ≈ log10(0.60832) atol = 1e-12
        @test om.max ≈ log10(0.41248) atol = 1e-12

        # Constrained cells are excluded too (the Euler equation is an inequality there).
        a_con = reshape([0.0, 1.5, 1.5, 1.5, 1.5], 5, 1)
        cn = MEM._euler_error_stats(c_pol, a_con, ip, grid, income, prices; points=:nodes)
        @test cn.n_constrained == 1 && cn.n_evaluated == 4
    end

    @testset "shipped examples: the node metric flatters by 2.5-3.8 log10 units" begin
        for (ex, mid, nodes) in ((:krusell_smith, -2.2531, -6.0397),
                                 (:one_asset_hank, -2.2781, -6.0555),
                                 (:huggett, -1.9363, -4.4699))
            ss = compute_steady_state(load_ha_example(ex))
            @test ss.euler !== nothing
            # The headline number is now the off-node one.
            @test ss.euler_error ≈ mid atol = 1e-3
            @test ss.euler.midpoints.max ≈ mid atol = 1e-3
            @test ss.euler.nodes.max ≈ nodes atol = 1e-3
            # The node metric is optimistic by construction, never pessimistic.
            @test ss.euler.nodes.max < ss.euler.midpoints.max
            # mean < max, and both are finite and reported.
            @test ss.euler.midpoints.mean < ss.euler.midpoints.max
            @test isfinite(ss.euler.midpoints.mean)
            @test ss.euler.midpoints.n_evaluated > 0

            # The old convention is still reachable for continuity.
            ss_n = compute_steady_state(load_ha_example(ex); euler_points=:nodes)
            @test ss_n.euler_error ≈ nodes atol = 1e-3
        end
        @test_throws ArgumentError compute_steady_state(load_ha_example(:huggett);
                                                        euler_points=:bogus)
    end

    @testset "a truncating model no longer reports the better accuracy" begin
        # Same pre-fix Krusell-Smith clone the grid diagnostics use. Under the node
        # metric its 22 truncated cells were excused to ~1e-11 while the interior sat
        # at 2.5e-3, so truncation bought accuracy. Off-node it is scored worse than
        # the shipped calibration, which is the point of the change.
        base = load_ha_example(:krusell_smith)
        raw = rouwenhorst(0.966, 0.5, 7)
        e = exp.(raw.states); e ./= dot(raw.stationary_dist, e)
        old_inc = IncomeProcess{Float64}(raw.transition, e, raw.stationary_dist, :income)
        old = HADSGESpec{Float64}(base.aggregate_spec, base.individual, old_inc,
                                  HAGrid(; assets=(0.0, 200.0, 200), income_states=7),
                                  base.aggregation, base.het_params; model=base.model)
        ss_bad = compute_steady_state(old; grid_check=:none)
        ss_good = compute_steady_state(base)

        @test ss_bad.euler_error > ss_good.euler_error      # measured -1.66 vs -2.25
        @test ss_bad.euler.midpoints.n_offgrid > 0          # and its cells do leave the grid
        @test ss_good.euler.midpoints.n_offgrid == 0
        # Under the OLD metric the gap was 3.4 log10 units the other way, which is
        # what made a truncating fit look respectable.
        @test ss_bad.euler.nodes.max ≈ -2.5952 atol = 1e-3
        @test ss_bad.euler.nodes.max < ss_bad.euler.midpoints.max
    end

    @testset "report(ss) names the convention" begin
        ss = compute_steady_state(load_ha_example(:huggett))
        io = IOBuffer(); report(io, ss); s = String(take!(io))
        @test occursin("Euler error", s)
        @test occursin("midpoints", s)          # the convention is stated, not implied
        @test occursin("mean (log10)", s)
        @test occursin("at grid nodes", s)
    end
end
