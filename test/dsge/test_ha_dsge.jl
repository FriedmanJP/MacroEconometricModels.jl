# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
const _hh = MacroEconometricModels._hh
using LinearAlgebra
using SparseArrays
using Random
using Distributions

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

# Shared coarse Huggett VFI SS — reused by the hh_solver=:vfi / Reiter / SSJ smokes.
const _VFI_HUG_SPEC = MacroEconometricModels._huggett_example(; n_a=40)
const _VFI_HUG_SS = compute_steady_state(_VFI_HUG_SPEC; hh_solver=:vfi, max_iter=50,
                                         tol=5e-3, grid_check=:none)

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
    n_a = 120
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
        for i in 10:(n_a - 10)
            if a_pol[i, j] > 0.5
                Eu_prime = sum(inc.transition[j, jp] * (1.0 / MacroEconometricModels._linear_interp(
                    grid.grids[1], c_pol[:, jp], a_pol[i, j])) for jp in 1:3)
                euler_resid = abs(1.0 - 0.99 * (1 + r) * Eu_prime / (1.0 / c_pol[i, j]))
                @test euler_resid < 1e-3
                euler_checked += 1
            end
        end
    end
    @test euler_checked > 20  # enough interior points tested

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
    nl, ni = 16, 12
    grid2 = HAGrid(; liquid=(0.0, 20.0, nl), illiquid=(0.0, 50.0, ni), income_states=3)
    inc = rouwenhorst(0.966, 0.5, 3)
    ip2 = IndividualProblem{Float64}(
        c -> log(c), c -> 1.0/c, m -> 1.0/m, 0.99,
        (b, a, e, prices) -> (1 + prices[:r]) * b + prices[:w] * e,
        [0.0, 0.0], nothing, 2
    )
    prices2 = Dict(:r => 0.01, :r_b => 0.01, :r_a => 0.02, :w => 1.0)
    result = MacroEconometricModels._two_asset_egm_solve(ip2, grid2, inc, prices2;
        max_iter=80, tol=1e-6, n_deposit=6)

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
        max_iter=120, tol=1e-6, n_deposit=6)
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
    na = 40
    grid = HAGrid(assets=(0.0, 200.0, na), income_states=3)
    inc = rouwenhorst(0.966, 0.5, 3)
    ip = IndividualProblem{Float64}(
        c -> log(c), c -> 1.0/c, m -> 1.0/m, 0.99,
        (a, e, prices) -> (1 + prices[:r]) * a + prices[:w] * e,
        [0.0], nothing, 1
    )
    prices = Dict(:r => 0.01, :w => 1.0)
    V, c_pol, a_pol = MacroEconometricModels._vfi_solve(ip, grid, inc, prices;
                                                         max_iter=80, tol=1e-6,
                                                         howard_steps=8)
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
    # Converged flag is last; warm start is accepted
    V2, _, _, conv = MacroEconometricModels._vfi_solve(ip, grid, inc, prices;
                                                       max_iter=80, tol=1e-6,
                                                       howard_steps=8,
                                                       init_value=V)
    @test conv isa Bool
    @test size(V2) == size(V)
end

@testset "hh_solver=:vfi fills value_fn and agrees with EGM SS" begin
    ss_v = _VFI_HUG_SS
    ss_e = compute_steady_state(_VFI_HUG_SPEC; hh_solver=:egm, max_iter=50,
                                tol=5e-3, grid_check=:none)
    @test isfinite(ss_v.prices[:r])
    @test !all(iszero, ss_v.value_fn)
    @test !all(iszero, ss_e.value_fn)   # EGM recovers V by policy evaluation
    na = size(ss_v.value_fn, 1)
    @test ss_v.value_fn[div(7 * na, 8), 1] > ss_v.value_fn[max(1, div(na, 8)), 1]
    @test ss_e.value_fn[div(7 * na, 8), 1] > ss_e.value_fn[max(1, div(na, 8)), 1]
    @test abs(ss_v.prices[:r] - ss_e.prices[:r]) < 5e-3
    # Default remains EGM; EGM-fills-V is already asserted on ss_e
end

@testset "hh_solver=:vfi error paths" begin
    spec = load_ha_example(:huggett)
    @test_throws ArgumentError compute_steady_state(spec; hh_solver=:bogus)
    @test_throws ArgumentError solve(spec; method=:krusell_smith, hh_solver=:vfi,
                                     T_sim=20, T_burn=5, max_outer=1)
    @test_throws ArgumentError compute_steady_state(
        load_ha_example(:two_asset_hank); distribution=:winberry)
end

@testset "VFI with endogenous labor (GHH)" begin
    spec = MacroEconometricModels._endogenous_labor_example(; kind=:ghh, psi=3.0)
    # Coarse grid: rebuild a small one-asset labor problem
    na = 30
    grid = HAGrid(assets=(0.0, 80.0, na), income_states=3)
    inc = MacroEconometricModels._unit_mean_lognormal_income(0.9, 0.2, 3)
    ip = _hh(spec).individual
    prices = Dict(:r => 0.01, :w => 1.0)
    V, c_v, a_v, conv = MacroEconometricModels._vfi_solve(ip, grid, inc, prices;
                                                          max_iter=60,
                                                          tol=1e-6, howard_steps=8)
    c_e, a_e = MacroEconometricModels._egm_solve(ip, grid, inc, prices; max_iter=200, tol=1e-8)
    @test conv isa Bool
    @test all(c_v .> 0)
    mid = na ÷ 2
    @test c_e[mid, 2] > 0.05
    @test abs(c_v[mid, 2] - c_e[mid, 2]) / c_e[mid, 2] < 0.2
    n_v = labor_policy(ip, grid, inc, prices, c_v)
    @test all(n_v .> 0)
end

@testset "VFI with endogenous labor (separable)" begin
    spec = MacroEconometricModels._endogenous_labor_example(; kind=:separable, psi=3.0)
    na = 30
    grid = HAGrid(assets=(0.0, 80.0, na), income_states=3)
    inc = MacroEconometricModels._unit_mean_lognormal_income(0.9, 0.2, 3)
    ip = _hh(spec).individual
    prices = Dict(:r => 0.01, :w => 1.0)
    V, c_v, a_v, conv = MacroEconometricModels._vfi_solve(ip, grid, inc, prices;
                                                          max_iter=60,
                                                          tol=1e-6, howard_steps=8)
    @test conv isa Bool
    @test all(c_v .> 0)
    @test all(isfinite, V)
    @test all(a_v .>= -1e-10)
    n_v = labor_policy(ip, grid, inc, prices, c_v)
    @test all(n_v .>= 0)
    @test any(n_v .> 0)
end

@testset "Two-asset VFI agrees with nested EGM on a convex problem" begin
    nl, ni = 10, 8
    grid2 = HAGrid(; liquid=(0.0, 20.0, nl), illiquid=(0.0, 50.0, ni), income_states=2)
    inc = MacroEconometricModels._unit_mean_lognormal_income(0.9, 0.2, 2)
    ip2 = IndividualProblem{Float64}(
        c -> log(c), c -> 1.0/c, m -> 1.0/m, 0.95,
        (b, a, e, prices) -> (1 + prices[:r_b]) * b + prices[:w] * e,
        [0.0, 0.0], nothing, 2
    )
    prices2 = Dict(:r => 0.01, :r_b => 0.01, :r_a => 0.015, :w => 1.0)
    vfi = MacroEconometricModels._two_asset_vfi_solve(ip2, grid2, inc, prices2;
        max_iter=40, tol=1e-5, howard_steps=6)
    egm = MacroEconometricModels._two_asset_egm_solve(ip2, grid2, inc, prices2;
        max_iter=40, tol=1e-5)
    @test all(vfi[:consumption] .> 0)
    @test all(isfinite, vfi[:value])
    # Mid liquid, low illiquid, first income: consumption in the same ballpark
    ib, ia, je = max(nl ÷ 2, 1), 1, 1
    @test abs(vfi[:consumption][ib, ia, je] - egm[:consumption][ib, ia, je]) /
          max(egm[:consumption][ib, ia, je], 1e-8) < 0.35
end

@testset "Two-asset compute_steady_state closes both markets" begin
    spec = MacroEconometricModels._two_asset_hank_example(;
        n_liquid=8, n_illiquid=6, n_e=2, B_supply=2.0)
    ss = compute_steady_state(spec; max_iter=15, tol=5e-2,
                              grid_check=:none)
    @test ss isa HASteadyState
    @test haskey(ss.prices, :r_a)
    @test haskey(ss.prices, :r_b)
    @test haskey(ss.aggregates, :A)
    @test haskey(ss.aggregates, :B)
    @test size(ss.distribution) == (_hh(spec).grid.n_points[1], _hh(spec).grid.n_points[2], 2)
    @test all(ss.policies[:consumption] .> 0)
    @test isfinite(ss.euler_error)
    @test ss.aggregates[:B_supply] == 2.0
    # Residuals finite; FAST may not fully clear
    @test isfinite(ss.aggregates[:resid_liquid])
    @test isfinite(ss.aggregates[:resid_illiquid])
    gd = ha_grid_diagnostics(ss)
    @test gd isa MacroEconometricModels.HAGridDiagnostics
end

@testset "Two-asset SSJ / Reiter / KS on a coarse SS" begin
    spec = MacroEconometricModels._two_asset_hank_example(;
        n_liquid=6, n_illiquid=5, n_e=2, B_supply=1.0)
    ss = compute_steady_state(spec; max_iter=8, tol=5e-2,
                              grid_check=:none)
    sol_s = solve(spec; method=:ssj, ss=ss, T_horizon=8, n_reduced=3)
    @test sol_s.method === :ssj
    @test isfinite(sol_s.explained_variance)
    J = MacroEconometricModels._ssj_jacobian(ss, _hh(spec).individual, _hh(spec).grid,
                                             _hh(spec).income, :r_b, :B; T_horizon=6)
    @test size(J) == (6, 6)
    @test all(isfinite, J)
    sol_r = solve(spec; method=:reiter, ss=ss, n_reduced=3)
    @test sol_r.method === :reiter
    @test size(sol_r.reduction_basis, 1) == length(vec(ss.distribution))
    sol_k = solve(spec; method=:krusell_smith, ss=ss,
                  T_sim=25, T_burn=5, max_outer=1)
    @test sol_k isa KrusellSmithSolution
    @test haskey(sol_k.plm_coefficients, :K)
    @test isfinite(sol_k.r_squared[:K])
end

@testset "Reiter honors hh_solver=:vfi" begin
    sol = solve(_VFI_HUG_SPEC; method=:reiter, ss=_VFI_HUG_SS, hh_solver=:vfi, n_reduced=4)
    @test sol.method === :reiter
    @test !all(iszero, sol.steady_state.value_fn)
end

@testset "solve forwards hh_solver into the stationary problem" begin
    sol = solve(_VFI_HUG_SPEC; method=:ssj, ss=_VFI_HUG_SS, T_horizon=12, n_reduced=4)
    @test sol.steady_state === _VFI_HUG_SS
    @test !all(iszero, sol.steady_state.value_fn)
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

@testset "Young (2010) two-asset product lottery" begin
    nl, ni, ne = 6, 5, 2
    grid = HAGrid(; liquid=(0.0, 10.0, nl), illiquid=(0.0, 20.0, ni), income_states=ne)
    inc = MacroEconometricModels._unit_mean_lognormal_income(0.9, 0.2, ne)
    b_pol = zeros(nl, ni, ne)
    a_pol = zeros(nl, ni, ne)
    bg, ag = grid.grids[1], grid.grids[2]
    # On-grid: every household stays put
    for je in 1:ne, ia in 1:ni, ib in 1:nl
        b_pol[ib, ia, je] = bg[ib]
        a_pol[ib, ia, je] = ag[ia]
    end
    Λ = MacroEconometricModels._build_transition_matrix(b_pol, a_pol, grid, inc)
    N = nl * ni * ne
    @test size(Λ) == (N, N)
    for col in 1:N
        @test sum(Λ[:, col]) ≈ 1.0 atol=1e-12
    end
    # Mid-grid continuous b' splits liquid mass; a' on-grid stays Dirac
    ib, ia, je = 3, 2, 1
    b_pol[ib, ia, je] = (bg[ib] + bg[ib + 1]) / 2
    Λ2 = MacroEconometricModels._build_transition_matrix(b_pol, a_pol, grid, inc)
    col = MacroEconometricModels._ha_state_index(ib, ia, je, nl, ni)
    @test sum(Λ2[:, col]) ≈ 1.0 atol=1e-12
    dest_lo = MacroEconometricModels._ha_state_index(ib, ia, 1, nl, ni)
    dest_hi = MacroEconometricModels._ha_state_index(ib + 1, ia, 1, nl, ni)
    @test Λ2[dest_lo, col] > 0
    @test Λ2[dest_hi, col] > 0
    d, ok = MacroEconometricModels._stationary_dist_young(Λ)
    @test ok
    @test sum(d) ≈ 1.0 atol=1e-10
    @test length(d) == N
    @test isfinite(MacroEconometricModels._aggregate(d, grid; var_index=1))
    @test isfinite(MacroEconometricModels._aggregate(d, grid; var_index=2))
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

    # EGM recovers V by Howard policy evaluation of the equilibrium policy
    @test !all(iszero, ss.value_fn)
    @test ss.value_fn[div(7 * na, 8), 1] > ss.value_fn[max(1, div(na, 8)), 1]

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
    grid = HAGrid(assets=(0.0, 200.0, 40), income_states=3)
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
        K_init=10.0, r_bounds=(-0.02, 0.04), max_iter=30, tol=1e-3)

    result = MacroEconometricModels._krusell_smith_solve(
        ss, ip, grid, inc, price_fn, params;
        T_sim=120, T_burn=25, max_outer=2,
        rho_z=0.95, sigma_z=0.007
    )

    @test haskey(result.plm_coefficients, :K)
    @test length(result.plm_coefficients[:K]) == 3  # z-augmented PLM: [b1, b2, b3]
    @test haskey(result.r_squared, :K)
    @test result.r_squared[:K] > 0.9  # KS typically gets R² > 0.999
    @test result.iterations <= 2

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
    T_s = 60; zidx = zeros(Int, T_s); zidx[1] = 2; zc = cumsum(zt; dims=2)
    for t in 2:T_s
        u = rand(rng_ks)
        zidx[t] = clamp(searchsortedfirst(view(zc, zidx[t-1], :), u), 1, n_z)
    end
    ks_egm_iter = 200
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
        c, a, _ = MacroEconometricModels._egm_solve(_hh(spec).individual, _hh(spec).grid,
                                                    _hh(spec).income, prices;
                                                    max_iter=2000, tol=1e-12)
        c1, a1 = MacroEconometricModels._egm_backward_step(_hh(spec).individual, _hh(spec).grid,
                                                           _hh(spec).income, prices, c)
        @test maximum(abs, c1 .- c) < 1e-9
        @test maximum(abs, a1 .- a) < 1e-9
    end
end

@testset "SSJ Jacobian" begin
    grid = HAGrid(assets=(0.0, 200.0, 40), income_states=3)
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
        K_init=10.0, r_bounds=(-0.02, 0.04), max_iter=30, tol=1e-3)

    T_h = 12
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
    # Default isapprox rtol swallows the ~1e-7 anticipation block on this
    # short horizon; the any(...) check above is the one that pins density.
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
    spec = MacroEconometricModels._huggett_example(; credit_limit=-2.0, a_max=8.0, n_a=60)
    ss = compute_steady_state(spec; max_iter=100, tol=1e-3)
    sol = solve(spec; method=:ssj, ss=ss, T_horizon=24, n_reduced=8)

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
    nred = 10
    grid = HAGrid(assets=(0.0, 200.0, 30), income_states=3)
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
    n_a = _hh(spec).grid.n_points[1]; n_e = _hh(spec).grid.n_income
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
    ssj_sol = solve(spec; method=:ssj, ss=ss, T_horizon=16, n_reduced=12)
    @test_throws ErrorException distribution_irf(ssj_sol, 10)
    @test_throws ErrorException inequality_irf(ssj_sol, 10)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 18: Built-in examples
# ─────────────────────────────────────────────────────────────────────────────

@testset "Built-in examples" begin
    @testset "Krusell-Smith" begin
        spec = load_ha_example(:krusell_smith)
        @test spec isa ModelSpec
        @test _hh(spec).grid.n_dims == 1
        @test _hh(spec).grid.n_income == 7
        @test _hh(spec).individual.beta ≈ 0.99
        @test length(_hh(spec).income.states) == 7
        @test _hh(spec).individual.borrowing_constraint[1] ≈ 0.0
        @test _hh(spec).grid.n_points == [200]
        @test _hh(spec).grid.bounds[1] == (0.0, 1000.0)
        @test _hh(spec).het_params[:alpha] ≈ 0.36
        @test _hh(spec).het_params[:delta] ≈ 0.025
        @test _hh(spec).n_assets == 1
        @test _hh(spec).n_income == 7
        # Aggregate spec is a valid DSGESpec
        @test :Y in spec.endog
        @test :K in spec.endog
    end

    @testset "One-asset HANK" begin
        spec = load_ha_example(:one_asset_hank)
        @test spec isa ModelSpec
        @test _hh(spec).grid.n_dims == 1
        @test _hh(spec).individual.borrowing_constraint[1] ≈ -2.0
        @test _hh(spec).individual.beta ≈ 0.986
        @test _hh(spec).grid.bounds[1][1] ≈ -2.0
        @test _hh(spec).grid.bounds[1][2] ≈ 1000.0
        @test _hh(spec).grid.n_points == [200]
        @test _hh(spec).grid.n_income == 7
        @test _hh(spec).het_params[:sigma_c] ≈ 1.0
        @test _hh(spec).n_assets == 1
    end

    @testset "Two-asset HANK" begin
        spec = load_ha_example(:two_asset_hank)
        @test spec isa ModelSpec
        @test _hh(spec).grid.n_dims == 2
        @test _hh(spec).individual.adjustment_cost !== nothing
        @test _hh(spec).individual.n_asset_dims == 2
        @test _hh(spec).individual.borrowing_constraint[1] ≈ -2.0
        @test _hh(spec).individual.borrowing_constraint[2] ≈ 0.0
        @test _hh(spec).grid.labels == [:liquid, :illiquid]
        @test _hh(spec).grid.n_points == [50, 50]
        @test _hh(spec).grid.bounds[1] == (-2.0, 50.0)
        @test _hh(spec).grid.bounds[2] == (0.0, 100.0)
        @test _hh(spec).n_assets == 2
        @test _hh(spec).n_income == 7
        # Adjustment cost should return a positive value for nonzero deposit
        chi = _hh(spec).individual.adjustment_cost(1.0, 10.0)
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
            @test all(_hh(spec).income.states .> 0)
        end

        # The three Rouwenhorst examples must have unit-mean income E[e] = 1.
        for name in (:krusell_smith, :one_asset_hank, :two_asset_hank)
            spec = load_ha_example(name)
            @test dot(_hh(spec).income.stationary_dist, _hh(spec).income.states) ≈ 1.0 atol=1e-10
        end

        # Huggett keeps its bespoke {1.0, 0.1} endowment (mean ≈ 0.8826), NOT normalized.
        spec_h = load_ha_example(:huggett)
        @test dot(_hh(spec_h).income.stationary_dist, _hh(spec_h).income.states) ≈ 0.8826 atol=1e-3

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
            p = _hh(spec).income.stationary_dist
            z = log.(_hh(spec).income.states)
            mu = dot(p, z)
            var_z = dot(p, (z .- mu) .^ 2)
            @test sqrt(var_z) ≈ 0.5 atol=1e-8
            # first autocorrelation of the discretized chain == ρ exactly
            @test (dot(p .* z, _hh(spec).income.transition * z) - mu^2) / var_z ≈ 0.966 atol=1e-10
            # top/bottom ratio = exp(2ψ) with ψ = √6 · 0.5
            @test maximum(_hh(spec).income.states) / minimum(_hh(spec).income.states) ≈
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
        beta = _hh(spec).individual.beta
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
        ev = _hh(base).income.states
        for (amax, n, gt) in ((5.0, 60, :geometric), (20.0, 80, :geometric),
                              (200.0, 200, :double_exp), (50.0, 100, :linear))
            g = HAGrid(; assets=(0.0, amax, n), income_states=7, grid_type=gt)
            ag = g.grids[1]
            # affine policy with fixed point a*_j = 1.5·a_max·e_j ⇒ states truncate
            # differentially (e spans ≈0.3–3.0)
            apol = [max(0.0, 0.15 * amax * ev[j] + 0.9 * ag[i]) for i in 1:n, j in 1:7]
            L = MacroEconometricModels._build_transition_matrix(apol, g, _hh(base).income)
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
        old = MacroEconometricModels._replace_household(base; income=old_inc,
            grid=HAGrid(; assets=(0.0, 200.0, 200), income_states=7))
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
        ip = _hh(base).individual
        mk(bc) = IndividualProblem{Float64}(ip.utility, ip.utility_prime,
            ip.utility_prime_inv, ip.beta, ip.budget_fn, [bc], nothing, 1)
        g = HAGrid(; assets=(0.0, 200.0, 50), income_states=7)
        # Below the floor: the Young clamp would create assets out of nothing.
        @test_throws ArgumentError HouseholdSystem{Float64}(mk(-1.0),
            _hh(base).income, g, _hh(base).aggregation, _hh(base).het_params)
        # Above the floor: merely wasteful, so warn.
        logs, _ = Test.collect_test_logs() do
            HouseholdSystem{Float64}(mk(5.0), _hh(base).income, g,
                                     _hh(base).aggregation, _hh(base).het_params)
        end
        @test any(l -> occursin("unreachable", string(l.message)), logs)
        # All shipped examples satisfy the check.
        for name in (:krusell_smith, :one_asset_hank, :two_asset_hank, :huggett)
            @test load_ha_example(name) isa ModelSpec
        end
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
        # Build the solution struct directly: the guard fires on `_hh(spec).model`, so running
        # the (expensive) Krusell-Smith PLM fixed point just to reach it wastes ~4 minutes.
        ks_h = M.KrusellSmithSolution{Float64}(
            _VFI_HUG_SS, Dict(:K => [0.0, 0.95, 0.0]), Dict(:K => 0.99),
            _VFI_HUG_SPEC, true, 1)
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
        spec = _VFI_HUG_SPEC
        ss = _VFI_HUG_SS
        g_new = adapt_ha_grid(_hh(spec).grid, ss.distribution)
        @test g_new isa MacroEconometricModels.HAGrid{Float64}
        @test g_new.n_dims == _hh(spec).grid.n_dims
        @test g_new.n_income == _hh(spec).grid.n_income
        @test g_new.n_points == _hh(spec).grid.n_points
        @test g_new.bounds == _hh(spec).grid.bounds
        @test g_new.labels == _hh(spec).grid.labels
        @test g_new.grids[1][1] == _hh(spec).grid.grids[1][1]        # borrowing constraint intact
        @test g_new.grids[1][end] == _hh(spec).grid.grids[1][end]
        @test all(diff(g_new.grids[1]) .> 0)
        @test g_new.grids[1] != _hh(spec).grid.grids[1]              # nodes actually moved

        # curvature=0 reproduces a uniform grid through the wrapper too
        g_uni = adapt_ha_grid(_hh(spec).grid, ss.distribution; curvature=0.0)
        lo, hi = _hh(spec).grid.bounds[1]
        @test g_uni.grids[1] ≈ collect(range(lo, hi; length=_hh(spec).grid.n_points[1])) atol = 1e-10

        # a coarser grid is allowed
        g_small = adapt_ha_grid(_hh(spec).grid, ss.distribution; n_points=[50])
        @test g_small.n_points == [50]
        @test length(g_small.grids[1]) == 50
        @test g_small.total_individual_states == 50 * _hh(spec).grid.n_income

        @test_throws ArgumentError adapt_ha_grid(_hh(spec).grid, ss.distribution; n_points=[50, 50])
        @test_throws ArgumentError adapt_ha_grid(_hh(spec).grid, ss.distribution[1:10])
        @test_throws ArgumentError adapt_ha_grid(_hh(spec).grid, zeros(size(ss.distribution)))

        # spec method returns a solvable specification
        spec2 = adapt_ha_grid(spec, ss)
        @test spec2 isa ModelSpec
        @test _hh(spec2).model == _hh(spec).model
        @test _hh(spec2).distribution == _hh(spec).distribution
        @test _hh(spec2).grid.grids[1] == g_new.grids[1]
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
        _hh(base).individual.utility, _hh(base).individual.utility_prime,
        _hh(base).individual.utility_prime_inv, _hh(base).individual.beta,
        _hh(base).individual.budget_fn, _hh(base).individual.borrowing_constraint,
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
            _hh(base).individual.utility, _hh(base).individual.utility_prime,
            _hh(base).individual.utility_prime_inv, _hh(base).individual.beta,
            _hh(base).individual.budget_fn, [0.0, 0.0], nothing, 2; labor=LaborSupply())
    end

    FAST || @testset "exogenous-labor paths are untouched" begin
        base = load_ha_example(:krusell_smith)
        @test _hh(base).individual.labor === nothing
        # `labor_policy` returns ones, so ∫e·n dμ reduces to ∫e dμ.
        ss = compute_steady_state(base)
        n = labor_policy(_hh(base).individual, _hh(base).grid, _hh(base).income, ss.prices,
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
            c, a, conv = MacroEconometricModels._egm_solve(ip, _hh(base).grid, _hh(base).income,
                                                            prices; max_iter=3000, tol=1e-12)
            @test conv
            n = labor_policy(ip, _hh(base).grid, _hh(base).income, prices, c)
            ag = _hh(base).grid.grids[1]
            foc_err = 0.0; budget_err = 0.0
            for j in eachindex(_hh(base).income.states), i in eachindex(ag)
                we = prices[:w] * _hh(base).income.states[j]
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
                             eachindex(_hh(base).income.states))
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
            al = _hh(spec).het_params[:alpha]; de = _hh(spec).het_params[:delta]
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
                    reshape(_hh(spec).income.states, 1, :)), vec(ss.distribution)) rtol=1e-12
            @test ss.aggregates[:N] ≈ dot(vec(ss.policies[:labor]),
                                          vec(ss.distribution)) rtol=1e-12
            @test ss.aggregates[:L] != ss.aggregates[:N]
            # Y must be built from realized labor, not the params[:L] = 1 default.
            @test ss.aggregates[:Y] ≈ ss.aggregates[:K]^al *
                                      ss.aggregates[:L]^(1 - al) rtol=1e-6
            # Aiyagari existence still holds at the equilibrium.
            @test _hh(spec).individual.beta * (1 + ss.prices[:r]) < 1
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
        ls = _hh(spec).individual.labor
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
        @test spec isa ModelSpec
        @test _hh(spec).individual.labor isa LaborSupply{Float64}
        @test _hh(spec).individual.labor.kind === :ghh
        @test _hh(spec).grid.bounds[1] == (0.0, 2000.0)
        @test _hh(spec).n_assets == 1
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

        @test spec isa ModelSpec
        @test _hh(spec).grid.n_dims == 1
        @test _hh(spec).grid.n_points == [100]
        @test _hh(spec).grid.bounds[1] == (0.0, 200.0)
        @test _hh(spec).n_income == 5
        @test _hh(spec).individual.beta ≈ 0.99
        @test _hh(spec).individual.borrowing_constraint[1] ≈ 0.0
        @test _hh(spec).individual.n_asset_dims == 1
        @test _hh(spec).n_assets == 1
        @test _hh(spec).het_params[:alpha] ≈ 0.36
        @test _hh(spec).het_params[:delta] ≈ 0.025
        @test :Y in spec.endog
        @test :K in spec.endog
        @test length(_hh(spec).income.states) == 5
        @test size(_hh(spec).income.transition) == (5, 5)
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

        @test spec isa ModelSpec
        @test _hh(spec).grid.n_points == [80]
        @test _hh(spec).grid.bounds[1] == (0.0, 150.0)
        @test _hh(spec).n_income == 7
        @test length(_hh(spec).income.states) == 7
    end

    @testset "Standard @dsge unaffected" begin
        spec_std = @dsge begin
            parameters: rho = 0.9, sigma = 0.01
            endogenous: Y, A
            exogenous: eps_A

            Y[t] = A[t]
            A[t] = rho * A[t-1] + sigma * eps_A[t]
        end
        @test spec_std isa ModelSpec{Float64,NoAgents}
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
        @test _hh(spec).individual.utility_prime(2.0) ≈ 2.0^(-1.5)
        @test _hh(spec).individual.utility(2.0) ≈ 2.0^(1 - 1.5) / (1 - 1.5)
        @test _hh(spec).model == :aiyagari    # default model field

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
        @test _hh(spec_h).model == :huggett
        @test _hh(spec_h).grid.bounds[1] == (-2.0, 4.0)
        @test _hh(spec_h).individual.borrowing_constraint[1] ≈ -2.0
        @test _hh(spec_h).individual.beta ≈ 0.99322
        @test _hh(spec_h).individual.utility_prime(2.0) ≈ 2.0^(-1.5)

        # the built Huggett spec solves
        ss = compute_steady_state(spec_h; max_iter=80, tol=1e-3)
        sol = solve(spec_h; method=:ssj, ss=ss, T_horizon=24, n_reduced=8)
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
    @test hasmethod(solve, Tuple{ModelSpec{Float64}})
    @test MacroEconometricModels.has_kind(spec, HouseholdSystem)

    FAST && return

    # Verify unknown method raises error
    ss = MacroEconometricModels._ha_steady_state(
        _hh(spec).individual, _hh(spec).grid, _hh(spec).income,
        MacroEconometricModels._default_cobb_douglas_price_fn, _hh(spec).het_params;
        K_init=10.0, r_bounds=(-0.02, 0.04), max_iter=30, tol=1e-2
    )
    @test_throws ErrorException solve(spec; method=:nonexistent, ss=ss)

    # KS / one-asset production GE goes through combine_blocks (#636)
    sol = solve(spec; method=:ssj, ss=ss, T_horizon=16, n_reduced=6)
    @test sol.method === :ssj
    @test haskey(sol.jacobians, :H_U)
    @test haskey(sol.jacobians, :H_Z)
    @test size(sol.jacobians[:H_U], 1) == 16
end

end # @testset "HA-DSGE Types"
