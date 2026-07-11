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

# ─────────────────────────────────────────────────────────────────────────────
# Section 8: VFI one-asset with Howard improvement
# ─────────────────────────────────────────────────────────────────────────────

@testset "VFI one-asset" begin
    grid = HAGrid(assets=(0.0, 200.0, 80), income_states=3)
    inc = rouwenhorst(0.966, 0.5, 3)
    ip = IndividualProblem{Float64}(
        c -> log(c), c -> 1.0/c, m -> 1.0/m, 0.99,
        (a, e, prices) -> (1 + prices[:r]) * a + prices[:w] * e,
        [0.0], nothing, 1
    )
    prices = Dict(:r => 0.01, :w => 1.0)
    V, c_pol, a_pol = MacroEconometricModels._vfi_solve(ip, grid, inc, prices;
                                                         max_iter=300, tol=1e-6, howard_steps=20)
    @test size(V) == (80, 3)
    @test size(c_pol) == (80, 3)
    @test size(a_pol) == (80, 3)
    @test all(c_pol .> 0)
    @test all(a_pol .>= -1e-10)
    # Higher income → higher consumption
    @test c_pol[40, 3] > c_pol[40, 1]
    # Value function increasing in assets
    @test V[70, 2] > V[10, 2]
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

    dist = MacroEconometricModels._stationary_dist_young(Lambda)
    @test length(dist) == 300
    @test sum(dist) ≈ 1.0 atol=1e-10
    @test all(dist .>= 0)

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
    grid = HAGrid(assets=(0.0, 200.0, 150), income_states=5)
    inc = rouwenhorst(0.966, 0.5, 5)
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
        K_init=10.0, r_bounds=(-0.01, 0.04), max_iter=100, tol=1e-4
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
    @test size(ss.distribution) == (150, 5)

    # Policy shapes
    @test size(ss.policies[:savings]) == (150, 5)
    @test size(ss.policies[:consumption]) == (150, 5)

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
    grid = HAGrid(assets=(0.0, 200.0, 80), income_states=3)
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
        K_init=10.0, r_bounds=(-0.02, 0.04), max_iter=60, tol=1e-3)

    result = MacroEconometricModels._krusell_smith_solve(
        ss, ip, grid, inc, price_fn, params;
        T_sim=300, T_burn=50, max_outer=3, rho_z=0.95, sigma_z=0.007
    )

    @test haskey(result.plm_coefficients, :K)
    @test length(result.plm_coefficients[:K]) == 3  # z-augmented PLM: [b1, b2, b3]
    @test haskey(result.r_squared, :K)
    @test result.r_squared[:K] > 0.9  # KS typically gets R² > 0.999
    @test result.iterations <= 3

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
    T_s = 150; zidx = zeros(Int, T_s); zidx[1] = 2; zc = cumsum(zt; dims=2)
    for t in 2:T_s
        u = rand(rng_ks)
        zidx[t] = clamp(searchsortedfirst(view(zc, zidx[t-1], :), u), 1, n_z)
    end
    cA, _ = MacroEconometricModels._ks_egm_solve(ip, grid, inc, [0.0, 1.0, 0.0],
        zg, zt, Kg, price_fn, params; max_iter=1000, tol=1e-6, init_policy=c0)
    KA = MacroEconometricModels._ks_simulate(cA, ss, grid, inc, zidx, zg, Kg, price_fn, params)
    cB, _ = MacroEconometricModels._ks_egm_solve(ip, grid, inc, [0.3, 0.85, 0.05],
        zg, zt, Kg, price_fn, params; max_iter=1000, tol=1e-6, init_policy=c0)
    KB = MacroEconometricModels._ks_simulate(cB, ss, grid, inc, zidx, zg, Kg, price_fn, params)
    @test maximum(abs.(cA .- cB)) > 1e-4    # different PLM → materially different policy
    @test maximum(abs.(KA .- KB)) > 1e-3    # different PLM → different realized path
    @test all(isfinite, KA) && all(KA .> 0)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 13: SSJ Jacobian
# ─────────────────────────────────────────────────────────────────────────────

@testset "SSJ Jacobian" begin
    grid = HAGrid(assets=(0.0, 200.0, 80), income_states=3)
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
        K_init=10.0, r_bounds=(-0.02, 0.04), max_iter=60, tol=1e-3)

    T_h = 30
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
    @test !isapprox(J, LowerTriangular(J))
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
    grid = HAGrid(assets=(0.0, 200.0, 50), income_states=3)
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
        ss, ip, grid, inc; n_reduced=15, model=:aiyagari, het_params=params
    )
    @test size(G1, 1) == size(G1, 2)  # square
    @test size(G1, 1) <= 15 + 5  # reduced dim + aggregates
    @test n_red <= 15
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
        ss, ip, grid, inc; n_reduced=15, model=:aiyagari, het_params=params_lo)
    @test !(G1 ≈ G1b)
    # A missing required parameter errors informatively (no magic-number default).
    @test_throws ErrorException MacroEconometricModels._reiter_linearize(
        ss, ip, grid, inc; n_reduced=15, model=:aiyagari,
        het_params=Dict(:alpha => 0.36, :delta => 0.025, :Z => 1.0))   # no :rho_z
    @test_throws ErrorException MacroEconometricModels._reiter_linearize(
        ss, ip, grid, inc; n_reduced=15, model=:aiyagari,
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
        @test spec.grid.bounds[1] == (0.0, 200.0)
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
        @test spec.grid.bounds[1][2] ≈ 50.0
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

    # Compute steady state for generating fake data
    ss = compute_steady_state(spec; K_init=10.0, r_bounds=(-0.02, 0.04), max_iter=50, tol=1e-3)
    K_ss = ss.aggregates[:K]
    T_data = 50
    rng = Random.MersenneTwister(42)
    data_K = K_ss .+ 0.1 .* randn(rng, T_data)  # K with noise

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

    @testset "_build_ha_likelihood_fn" begin
        # Solve model first to have a valid solution for observation equation
        sol = solve(spec; method=:ssj, ss=ss, T_horizon=30, n_reduced=10)
        @test sol isa HADSGESolution{Float64}

        param_names = [:alpha]
        ll_fn = MacroEconometricModels._build_ha_likelihood_fn(
            spec, param_names, reshape(data_K, 1, :),
            [:K], nothing, :ssj, (T_horizon=30, n_reduced=10)
        )

        ll_val = ll_fn([0.36])
        @test isfinite(ll_val)
        @test ll_val < 0  # log likelihood is negative

        # Likelihood should handle bad parameter values gracefully
        ll_bad = ll_fn([0.001])  # extreme parameter
        @test ll_bad == -Inf || ll_bad < ll_val + 100  # either fails or worse
    end

    @testset "_build_ha_observation_equation" begin
        sol = solve(spec; method=:ssj, ss=ss, T_horizon=30, n_reduced=10)

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
            n_draws=20,
            burnin=5,
            ha_method=:ssj,
            ha_kwargs=(T_horizon=30, n_reduced=10),
            proposal_scale=0.001,
            adapt_interval=50,  # no adaptation in 20 draws
            rng=rng_est
        )

        @test result isa BayesianDSGE{Float64}
        @test result.solved_at === :posterior_mean  # normal path (#149/T050)
        @test size(result.theta_draws, 2) == 1  # one parameter
        @test size(result.theta_draws, 1) == 15  # n_draws - burnin = 20 - 5
        @test length(result.log_posterior) == 15
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
            priors=priors, observables=[:K], n_draws=10, burnin=2,
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
            priors=priors, observables=[:K], n_draws=10, burnin=2,
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
        @test isfinite(ll30) && isfinite(ll60)
        @test abs(ll30 - ll60) > 1e-6
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
    targets = [(-2.0, -0.071), (-4.0, 0.023), (-6.0, 0.034), (-8.0, 0.040)]

    r_annuals = Float64[]
    for (cl, r_target) in targets
        a_max = cl <= -6 ? 18.0 : 8.0
        spec = MacroEconometricModels._huggett_example(; credit_limit=cl, a_max=a_max, n_a=400)
        @test spec.model == :huggett
        ss = compute_steady_state(spec; max_iter=200, tol=5e-4)
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
    spec = MacroEconometricModels._huggett_example(; credit_limit=-2.0, a_max=8.0, n_a=200)
    ss = compute_steady_state(spec; max_iter=120, tol=1e-3)
    sol = solve(spec; method=:ssj, ss=ss, T_horizon=100, n_reduced=20)
    @test sol isa HADSGESolution
    @test sol.method === :ssj
    @test maximum(abs.(eigvals(sol.linear_solution.G1))) <= 1 + 1e-6  # stable
    @test haskey(sol.jacobians, :H_U)                                  # clearing Jacobian
    @test haskey(sol.jacobians, :H_Z)                                  # shock Jacobian
    # A positive aggregate endowment shock lowers the clearing risk-free rate on impact.
    H_U = sol.jacobians[:H_U]; H_Z = sol.jacobians[:H_Z]
    dr = -(H_U \ (H_Z * [0.9^(t - 1) for t in 1:100]))
    @test dr[1] < 0
end

@testset "Huggett Reiter" begin
    spec = MacroEconometricModels._huggett_example(; credit_limit=-2.0, a_max=8.0, n_a=200)
    ss = compute_steady_state(spec; max_iter=120, tol=1e-3)
    sol = solve(spec; method=:reiter, ss=ss, n_reduced=30)
    @test sol isa HADSGESolution
    @test sol.method === :reiter
    @test maximum(abs.(eigvals(sol.linear_solution.G1))) <= 1 + 1e-6   # stable
    @test sol.explained_variance > 0.5
    @test size(sol.linear_solution.G1, 1) == sol.n_reduced + 1         # state [d̃; w]
end

@testset "Huggett Krusell-Smith" begin
    spec = MacroEconometricModels._huggett_example(; credit_limit=-2.0, a_max=8.0, n_a=150)
    ss = compute_steady_state(spec; max_iter=100, tol=1e-3)
    sol = solve(spec; method=:krusell_smith, ss=ss, T_sim=800, T_burn=200, max_outer=3)
    @test sol isa KrusellSmithSolution
    @test haskey(sol.plm_coefficients, :r)        # PLM forecasts the clearing rate, not K
    @test sol.r_squared[:r] > 0.7                 # rate is near-linear in the endowment shock
    b = sol.plm_coefficients[:r]
    @test abs(b[1] - ss.prices[:r]) < 0.01        # PLM intercept ≈ steady-state rate
    @test b[2] < 0                                # positive endowment shock lowers r
end

@testset "Den Haan (2010) accuracy" begin
    # --- Aiyagari capital model (z-augmented PLM makes the test meaningful) ---
    ks_spec = load_ha_example(:krusell_smith)
    ss_a = compute_steady_state(ks_spec; r_bounds=(-0.02, 0.04), max_iter=80, tol=1e-3)
    ks = solve(ks_spec; method=:krusell_smith, ss=ss_a, T_sim=500, T_burn=100, max_outer=3)
    @test length(ks.plm_coefficients[:K]) == 3          # z-augmented PLM

    dh = den_haan_test(ks; T_sim=400, T_burn=100)
    @test dh isa DenHaanAccuracy
    @test dh.aggregate === :K
    @test isfinite(dh.dh_max) && dh.dh_max >= dh.dh_mean >= 0
    @test dh.sigma_ref > 0 && dh.sigma_plm > 0
    @test length(dh.ref_path) == 400 && length(dh.plm_path) == 400
    @test dh.sigma_plm > 0.2 * dh.sigma_ref             # PLM reproduces the fluctuations
    @test dh.dh_max < 1.0                               # accurate: well under 1% (Den Haan)
    report(dh)                                          # display smoke test

    # --- Huggett: rate accuracy test is intentionally unsupported (errors clearly) ---
    hug_spec = MacroEconometricModels._huggett_example(; n_a=120)
    ss_h = compute_steady_state(hug_spec; max_iter=80, tol=1e-3)
    ks_h = KrusellSmithSolution{Float64}(ss_h,
        Dict(:r => [ss_h.prices[:r], 0.0]), Dict(:r => 1.0), hug_spec, false, 0)
    @test_throws ErrorException den_haan_test(ks_h)
end

end # @testset "HA-DSGE Types"
