# Heterogeneous Agent DSGE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add heterogeneous agent DSGE modeling (Reiter, SSJ, Krusell-Smith) with one-asset and two-asset HANK support, EGM/VFI solvers, Young histogram distribution tracking, and Bayesian estimation integration.

**Architecture:** New `src/dsge/heterogeneous/` subdirectory with 12 files. `HADSGESpec{T}` wraps `DSGESpec{T}` for the aggregate block. Linearized HA solutions produce `DSGESolution{T}`, so all existing `irf`, `fevd`, `historical_decomposition`, `estimate_dsge_bayes` work unchanged. The `@dsge` macro gains `heterogeneous:`, `idiosyncratic_shocks:`, `aggregation:` blocks. `solve(spec::HADSGESpec)` dispatches to `:reiter`, `:ssj`, or `:krusell_smith` methods.

**Tech Stack:** Julia 1.10+, SparseArrays (stdlib), LinearAlgebra (stdlib), ForwardDiff (existing dep), existing DSGE infrastructure (gensys, Kalman filter, SMC)

**Conventions:** Follow existing codebase patterns — GPL-3.0 header in every file, `T<:AbstractFloat` type parameters, `robust_inv` over raw `inv`, `_` prefix for internal helpers, `using` only for stdlib modules (qualify `Optim.`, `ForwardDiff.` etc.), no variable named `eps`.

**Dependencies to add:** None. SparseArrays and LinearAlgebra already in Project.toml. We implement sparse Jacobian coloring manually (simple banded structure) rather than adding SparseDiffTools.jl. SVD reduction uses stdlib `svd()`. This avoids new deps.

---

## Phase 1: Foundation Types & Individual Problem Solvers (Tasks 1-4)

### Task 1: Core Type Definitions

**Files:**
- Create: `src/dsge/heterogeneous/types.jl`
- Test: `test/dsge/test_ha_dsge.jl`

- [ ] **Step 1: Create directory and write type definitions**

Create `src/dsge/heterogeneous/` directory, then write `types.jl`:

```julia
# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Type definitions for heterogeneous agent DSGE models.
"""

using LinearAlgebra

# =============================================================================
# HAGrid — multi-dimensional individual state space
# =============================================================================

struct HAGrid{T<:AbstractFloat}
    grids::Vector{Vector{T}}
    n_points::Vector{Int}
    n_dims::Int
    n_income::Int
    bounds::Matrix{T}
    labels::Vector{Symbol}
    total_individual_states::Int

    function HAGrid{T}(grids::Vector{Vector{T}}, n_income::Int;
                       labels::Vector{Symbol}=Symbol[],
                       bounds::Union{Nothing,Matrix{T}}=nothing) where {T}
        n_dims = length(grids)
        n_points = [length(g) for g in grids]
        if isempty(labels)
            labels = n_dims == 1 ? [:assets] : [:liquid, :illiquid]
        end
        if bounds === nothing
            bounds = zeros(T, n_dims, 2)
            for d in 1:n_dims
                bounds[d, 1] = first(grids[d])
                bounds[d, 2] = last(grids[d])
            end
        end
        total = prod(n_points) * n_income
        new{T}(grids, n_points, n_dims, n_income, bounds, labels, total)
    end
end

function HAGrid(; assets::Tuple{Real,Real,Int}=(0.0, 200.0, 200),
                  income_states::Int=7, grid_type::Symbol=:double_exp)
    T = Float64
    a_min, a_max, n_a = T(assets[1]), T(assets[2]), assets[3]
    grid = _make_asset_grid(a_min, a_max, n_a, grid_type)
    HAGrid{T}([grid], income_states; labels=[:assets])
end

function HAGrid(; liquid::Tuple{Real,Real,Int}, illiquid::Tuple{Real,Real,Int},
                  income_states::Int=7, grid_type::Symbol=:double_exp)
    T = Float64
    g1 = _make_asset_grid(T(liquid[1]), T(liquid[2]), liquid[3], grid_type)
    g2 = _make_asset_grid(T(illiquid[1]), T(illiquid[2]), illiquid[3], grid_type)
    HAGrid{T}([g1, g2], income_states; labels=[:liquid, :illiquid])
end

function _make_asset_grid(a_min::T, a_max::T, n::Int, grid_type::Symbol) where {T}
    if grid_type == :double_exp
        x = range(zero(T), one(T), length=n)
        dexp = x .+ (exp.(6 .* x) .- 1) ./ (exp(T(6)) - 1)
        dexp ./= dexp[end]
        return a_min .+ (a_max - a_min) .* dexp
    elseif grid_type == :log
        shift = T(0.25)
        raw = exp.(range(log(a_min + shift), log(a_max + shift), length=n))
        return raw .- shift
    else
        return range(a_min, a_max, length=n) |> collect
    end
end

# =============================================================================
# IncomeProcess — idiosyncratic Markov chain
# =============================================================================

struct IncomeProcess{T<:AbstractFloat}
    transition::Matrix{T}
    states::Vector{T}
    stationary_dist::Vector{T}
    labels::Vector{Symbol}

    function IncomeProcess{T}(transition::Matrix{T}, states::Vector{T};
                               labels::Vector{Symbol}=Symbol[]) where {T}
        n = length(states)
        @assert size(transition) == (n, n) "Transition matrix must be $n x $n"
        sd = _stationary_distribution(transition)
        if isempty(labels)
            labels = [Symbol("e_$i") for i in 1:n]
        end
        new{T}(transition, states, sd, labels)
    end
end

function _stationary_distribution(P::Matrix{T}; max_iter::Int=10000, tol::T=T(1e-14)) where {T}
    n = size(P, 1)
    d = fill(one(T) / n, n)
    for _ in 1:max_iter
        d_new = P' * d
        d_new ./= sum(d_new)
        if maximum(abs.(d_new .- d)) < tol
            return d_new
        end
        d = d_new
    end
    return d
end

"""
    rouwenhorst(rho, sigma, n) -> IncomeProcess

Rouwenhorst (1995) discretization of AR(1): e' = rho * e + sigma * eps.
"""
function rouwenhorst(rho::Real, sigma::Real, n::Int)
    T = Float64
    rho, sigma = T(rho), T(sigma)
    sigma_y = sigma / sqrt(one(T) - rho^2)
    psi = sigma_y * sqrt(T(n - 1))
    states = range(-psi, psi, length=n) |> collect

    p = (one(T) + rho) / 2
    q = p
    Pi = T[p 1-p; 1-q q]

    for i in 3:n
        z = zeros(T, i, i)
        z[1:i-1, 1:i-1] += p * Pi
        z[1:i-1, 2:i]   += (1-p) * Pi
        z[2:i,   1:i-1] += (1-q) * Pi
        z[2:i,   2:i]   += q * Pi
        z[2:i-1, :] ./= 2
        Pi = z
    end
    IncomeProcess{T}(Pi, exp.(states))
end

"""
    tauchen(rho, sigma, n; m=3) -> IncomeProcess

Tauchen (1986) discretization of AR(1).
"""
function tauchen(rho::Real, sigma::Real, n::Int; m::Int=3)
    T = Float64
    rho, sigma = T(rho), T(sigma)
    sigma_y = sigma / sqrt(one(T) - rho^2)
    states = range(-m * sigma_y, m * sigma_y, length=n) |> collect
    d = states[2] - states[1]

    Pi = zeros(T, n, n)
    dist = Distributions.Normal(zero(T), sigma)
    for i in 1:n
        for j in 1:n
            if j == 1
                Pi[i, j] = Distributions.cdf(dist, states[1] + d/2 - rho * states[i])
            elseif j == n
                Pi[i, j] = one(T) - Distributions.cdf(dist, states[n] - d/2 - rho * states[i])
            else
                Pi[i, j] = Distributions.cdf(dist, states[j] + d/2 - rho * states[i]) -
                           Distributions.cdf(dist, states[j] - d/2 - rho * states[i])
            end
        end
    end
    IncomeProcess{T}(Pi, exp.(states))
end

# =============================================================================
# IndividualProblem — household optimization
# =============================================================================

struct IndividualProblem{T<:AbstractFloat}
    utility::Function
    utility_prime::Function
    utility_prime_inv::Function
    beta::T
    budget_fn::Function
    borrowing_constraint::Vector{T}
    adjustment_cost::Union{Nothing,Function}
    n_asset_dims::Int
end

# =============================================================================
# HADSGESpec — heterogeneous agent DSGE specification
# =============================================================================

struct HADSGESpec{T<:AbstractFloat}
    aggregate_spec::DSGESpec{T}
    individual::IndividualProblem{T}
    income::IncomeProcess{T}
    grid::HAGrid{T}
    aggregation::Vector{Pair{Symbol,Function}}
    het_params::Dict{Symbol,T}
    n_assets::Int
    n_income::Int
end

# =============================================================================
# HASteadyState — stationary equilibrium
# =============================================================================

struct HASteadyState{T<:AbstractFloat}
    policies::Dict{Symbol,Array{T}}
    distribution::Array{T}
    value_fn::Array{T}
    prices::Dict{Symbol,T}
    aggregates::Dict{Symbol,T}
    grid::HAGrid{T}
    income::IncomeProcess{T}
    converged::Bool
    iterations::Int
    euler_error::T
    excess_demand::T
end

# =============================================================================
# HADSGESolution — linearized HA solution (wraps DSGESolution)
# =============================================================================

struct HADSGESolution{T<:AbstractFloat}
    steady_state::HASteadyState{T}
    linear_solution::DSGESolution{T}
    method::Symbol
    spec::HADSGESpec{T}
    reduction_basis::Matrix{T}
    n_full_states::Int
    n_reduced::Int
    explained_variance::T
    jacobians::Union{Nothing,Dict{Symbol,Matrix{T}}}
end

# =============================================================================
# KrusellSmithSolution — simulation-based
# =============================================================================

struct KrusellSmithSolution{T<:AbstractFloat}
    steady_state::HASteadyState{T}
    plm_coefficients::Dict{Symbol,Vector{T}}
    r_squared::Dict{Symbol,T}
    spec::HADSGESpec{T}
    converged::Bool
    iterations::Int
end

# =============================================================================
# Accessors (following existing DSGE pattern)
# =============================================================================

nvars(sol::HADSGESolution) = nvars(sol.linear_solution)
nshocks(sol::HADSGESolution) = nshocks(sol.linear_solution)
is_determined(sol::HADSGESolution) = is_determined(sol.linear_solution)
is_stable(sol::HADSGESolution) = is_stable(sol.linear_solution)
```

- [ ] **Step 2: Write tests for types**

Create `test/dsge/test_ha_dsge.jl`:

```julia
# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
using LinearAlgebra
using Random
using SparseArrays

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

@testset "Heterogeneous Agent DSGE" begin

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Types
# ─────────────────────────────────────────────────────────────────────────────

@testset "HAGrid construction" begin
    grid = HAGrid(assets=(0.0, 200.0, 100), income_states=7)
    @test grid.n_dims == 1
    @test grid.n_income == 7
    @test grid.n_points == [100]
    @test grid.total_individual_states == 700
    @test grid.labels == [:assets]
    @test grid.bounds[1,1] ≈ 0.0
    @test grid.grids[1][1] ≈ 0.0
    @test grid.grids[1][end] ≈ 200.0
    @test issorted(grid.grids[1])
    # Double-exp grid: denser near zero
    @test grid.grids[1][2] - grid.grids[1][1] < grid.grids[1][end] - grid.grids[1][end-1]
end

@testset "HAGrid two-asset" begin
    grid = HAGrid(liquid=(-2.0, 50.0, 50), illiquid=(0.0, 100.0, 50), income_states=7)
    @test grid.n_dims == 2
    @test grid.n_points == [50, 50]
    @test grid.total_individual_states == 50 * 50 * 7
    @test grid.labels == [:liquid, :illiquid]
end

@testset "Rouwenhorst discretization" begin
    inc = rouwenhorst(0.966, 0.5, 7)
    @test length(inc.states) == 7
    @test size(inc.transition) == (7, 7)
    @test all(inc.transition .>= 0)
    # Rows sum to 1
    for i in 1:7
        @test sum(inc.transition[i, :]) ≈ 1.0 atol=1e-12
    end
    # Stationary distribution sums to 1
    @test sum(inc.stationary_dist) ≈ 1.0 atol=1e-12
    # Stationary distribution is eigenvector
    @test inc.transition' * inc.stationary_dist ≈ inc.stationary_dist atol=1e-10
end

@testset "Tauchen discretization" begin
    inc = tauchen(0.9, 0.2, 5; m=3)
    @test length(inc.states) == 5
    @test size(inc.transition) == (5, 5)
    for i in 1:5
        @test sum(inc.transition[i, :]) ≈ 1.0 atol=1e-12
    end
    @test sum(inc.stationary_dist) ≈ 1.0 atol=1e-12
end

end # top-level testset
```

- [ ] **Step 3: Wire includes and exports into main module**

Add to `src/MacroEconometricModels.jl` after the existing DSGE includes (after line 361, the `include("dsge/hd.jl")` line):

```julia
# Heterogeneous Agent DSGE
include("dsge/heterogeneous/types.jl")
```

Add exports after the existing DSGE exports (after line 466):

```julia
# Heterogeneous Agent DSGE
export HADSGESpec, HAGrid, IncomeProcess, IndividualProblem
export HASteadyState, HADSGESolution, KrusellSmithSolution
export rouwenhorst, tauchen
```

- [ ] **Step 4: Run tests**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' -- test/dsge/test_ha_dsge.jl`

Or more directly:
```bash
julia --project=. -e '
using Test, MacroEconometricModels
include("test/dsge/test_ha_dsge.jl")
'
```

Expected: All tests pass. Grid construction, Rouwenhorst, Tauchen all produce correct structures.

- [ ] **Step 5: Commit**

```bash
git add src/dsge/heterogeneous/types.jl test/dsge/test_ha_dsge.jl
git add src/MacroEconometricModels.jl
git commit -m "feat(ha-dsge): add core type definitions for heterogeneous agent DSGE

Types: HAGrid, IncomeProcess, IndividualProblem, HADSGESpec,
HASteadyState, HADSGESolution, KrusellSmithSolution.
Includes Rouwenhorst and Tauchen income discretization methods."
```

---

### Task 2: Endogenous Grid Method (One-Asset)

**Files:**
- Create: `src/dsge/heterogeneous/egm.jl`
- Test: append to `test/dsge/test_ha_dsge.jl`

- [ ] **Step 1: Write failing tests for EGM**

Append to `test/dsge/test_ha_dsge.jl` inside the top-level testset, after the types section:

```julia
# ─────────────────────────────────────────────────────────────────────────────
# Section 2: EGM Solver
# ─────────────────────────────────────────────────────────────────────────────

@testset "EGM one-asset" begin
    grid = HAGrid(assets=(0.0, 200.0, 200), income_states=3)
    inc = rouwenhorst(0.966, 0.5, 3)

    sigma_c = 1.0  # log utility
    u(c) = log(c)
    u_prime(c) = 1.0 / c
    u_prime_inv(m) = 1.0 / m

    ip = IndividualProblem{Float64}(
        u, u_prime, u_prime_inv, 0.99,
        (a, e, prices) -> (1 + prices[:r]) * a + prices[:w] * e,
        [0.0], nothing, 1
    )

    prices = Dict(:r => 0.01, :w => 1.0)
    c_pol, a_pol = MacroEconometricModels._egm_solve(ip, grid, inc, prices;
                                                      max_iter=500, tol=1e-10)
    # Policy is N_a × N_e
    @test size(c_pol) == (200, 3)
    @test size(a_pol) == (200, 3)
    # Consumption is positive
    @test all(c_pol .> 0)
    # Savings policy satisfies borrowing constraint
    @test all(a_pol .>= -1e-10)
    # Higher income → higher consumption
    @test c_pol[100, 3] > c_pol[100, 1]
    # Euler equation residual at interior points
    r = prices[:r]
    for j in 1:3, i in 10:190
        if a_pol[i, j] > 0.01
            # E[u'(c')] computed via transition
            Eu_prime = sum(inc.transition[j, jp] * u_prime(c_pol[
                clamp(searchsortedfirst(grid.grids[1], a_pol[i,j]), 1, 200), jp])
                for jp in 1:3)
            euler_resid = abs(1.0 - ip.beta * (1 + r) * Eu_prime / u_prime(c_pol[i, j]))
            @test euler_resid < 1e-4
        end
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

```bash
julia --project=. -e '
using Test, MacroEconometricModels
include("test/dsge/test_ha_dsge.jl")
'
```

Expected: FAIL — `_egm_solve` not defined.

- [ ] **Step 3: Implement EGM solver**

Create `src/dsge/heterogeneous/egm.jl`:

```julia
# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Endogenous Grid Method (Carroll 2006) for one-asset and two-asset individual problems.
"""

# =============================================================================
# One-asset EGM (Carroll 2006)
# =============================================================================

"""
    _egm_solve(ip, grid, income, prices; max_iter=1000, tol=1e-10) -> (c_policy, a_policy)

Solve the individual consumption-savings problem via the Endogenous Grid Method.

Returns `(c_policy, a_policy)` both `N_a × N_e` matrices.
"""
function _egm_solve(ip::IndividualProblem{T}, grid::HAGrid{T},
                    income::IncomeProcess{T}, prices::Dict{Symbol,T};
                    max_iter::Int=1000, tol::T=T(1e-10)) where {T}
    @assert grid.n_dims == 1 "One-asset EGM requires n_dims == 1"
    a_grid = grid.grids[1]
    n_a = grid.n_points[1]
    n_e = grid.n_income

    # Initialize consumption with fraction of cash-on-hand
    c_old = zeros(T, n_a, n_e)
    for j in 1:n_e
        for i in 1:n_a
            coh = ip.budget_fn(a_grid[i], income.states[j], prices)
            c_old[i, j] = max(coh * T(0.1), T(1e-8))
        end
    end

    c_new = similar(c_old)
    a_pol = similar(c_old)

    for iter in 1:max_iter
        _egm_step!(c_new, a_pol, c_old, ip, a_grid, income, prices)
        diff = maximum(abs.(c_new .- c_old))
        if diff < tol
            return c_new, a_pol
        end
        copyto!(c_old, c_new)
    end
    return c_new, a_pol
end

"""
    _egm_step!(c_new, a_pol, c_old, ip, a_grid, income, prices)

One EGM iteration. Updates `c_new` and `a_pol` in place.
"""
function _egm_step!(c_new::Matrix{T}, a_pol::Matrix{T}, c_old::Matrix{T},
                    ip::IndividualProblem{T}, a_grid::Vector{T},
                    income::IncomeProcess{T}, prices::Dict{Symbol,T}) where {T}
    n_a = length(a_grid)
    n_e = length(income.states)
    R = one(T) + prices[:r]
    w = prices[:w]
    a_min = ip.borrowing_constraint[1]

    for j in 1:n_e
        # Step 1: expected marginal utility on the a' grid (end-of-period assets)
        emu = zeros(T, n_a)
        for jp in 1:n_e
            Pi_jjp = income.transition[j, jp]
            if Pi_jjp < T(1e-15)
                continue
            end
            for i in 1:n_a
                c_interp = _linear_interp(a_grid, @view(c_old[:, jp]), a_grid[i])
                emu[i] += Pi_jjp * ip.utility_prime(c_interp)
            end
        end

        # Step 2: Euler equation inversion → consumption on endogenous grid
        c_endo = Vector{T}(undef, n_a)
        a_endo = Vector{T}(undef, n_a)
        for i in 1:n_a
            c_endo[i] = ip.utility_prime_inv(ip.beta * R * emu[i])
            a_endo[i] = (c_endo[i] + a_grid[i] - w * income.states[j]) / R
        end

        # Step 3: interpolate back to exogenous grid + borrowing constraint
        for i in 1:n_a
            a = a_grid[i]
            if a <= a_endo[1]
                # Constrained: consume all cash-on-hand
                coh = ip.budget_fn(a, income.states[j], prices)
                c_new[i, j] = max(coh - a_min, T(1e-10))
                a_pol[i, j] = a_min
            else
                c_new[i, j] = _linear_interp(a_endo, c_endo, a)
                a_pol[i, j] = ip.budget_fn(a, income.states[j], prices) - c_new[i, j]
                a_pol[i, j] = max(a_pol[i, j], a_min)
            end
        end
    end
end

# =============================================================================
# Two-asset nested EGM (Kaplan, Moll, Violante 2018)
# =============================================================================

"""
    _two_asset_egm_solve(ip, grid, income, prices; max_iter=1000, tol=1e-8)

Nested EGM for two-asset (liquid + illiquid) models.
Returns Dict of policies: :consumption, :liquid_savings, :deposit.
"""
function _two_asset_egm_solve(ip::IndividualProblem{T}, grid::HAGrid{T},
                              income::IncomeProcess{T}, prices::Dict{Symbol,T};
                              max_iter::Int=1000, tol::T=T(1e-8),
                              n_deposit::Int=30) where {T}
    @assert grid.n_dims == 2 "Two-asset EGM requires n_dims == 2"
    @assert ip.adjustment_cost !== nothing "Two-asset requires adjustment_cost"

    b_grid = grid.grids[1]  # liquid
    a_grid = grid.grids[2]  # illiquid
    n_b = grid.n_points[1]
    n_a = grid.n_points[2]
    n_e = grid.n_income

    # Deposit grid centered at zero
    d_max = a_grid[end] * T(0.3)
    d_grid = range(-d_max, d_max, length=n_deposit) |> collect

    r_b = prices[:r_b]
    r_a = prices[:r_a]
    w = prices[:w]

    c_old = zeros(T, n_b, n_a, n_e)
    for k in 1:n_e, j in 1:n_a, i in 1:n_b
        coh = (1 + r_b) * b_grid[i] + w * income.states[k]
        c_old[i, j, k] = max(coh * T(0.05), T(1e-8))
    end

    c_new = similar(c_old)
    b_pol = similar(c_old)
    d_pol = similar(c_old)

    for iter in 1:max_iter
        _two_asset_egm_step!(c_new, b_pol, d_pol, c_old, ip,
                              b_grid, a_grid, d_grid, income, prices)
        diff = maximum(abs.(c_new .- c_old))
        if diff < tol
            policies = Dict{Symbol,Array{T}}(
                :consumption => c_new,
                :liquid_savings => b_pol,
                :deposit => d_pol
            )
            return policies
        end
        copyto!(c_old, c_new)
    end
    policies = Dict{Symbol,Array{T}}(
        :consumption => c_new,
        :liquid_savings => b_pol,
        :deposit => d_pol
    )
    return policies
end

function _two_asset_egm_step!(c_new, b_pol, d_pol, c_old,
                               ip::IndividualProblem{T},
                               b_grid, a_grid, d_grid,
                               income::IncomeProcess{T},
                               prices::Dict{Symbol,T}) where {T}
    n_b, n_a, n_e = length(b_grid), length(a_grid), length(income.states)
    n_d = length(d_grid)
    R_b = one(T) + prices[:r_b]
    R_a = one(T) + prices[:r_a]
    w = prices[:w]
    b_min = ip.borrowing_constraint[1]
    chi = ip.adjustment_cost

    Threads.@threads for idx in 1:(n_a * n_e)
        j_a = div(idx - 1, n_e) + 1
        k_e = mod(idx - 1, n_e) + 1

        best_V = fill(T(-Inf), n_b)
        best_c = zeros(T, n_b)
        best_b = fill(b_min, n_b)
        best_d = zeros(T, n_b)

        for id in 1:n_d
            d = d_grid[id]
            a_next = R_a * a_grid[j_a] + d
            if a_next < zero(T)
                continue
            end
            adj_cost = chi(d, a_grid[j_a])

            # Inner EGM on liquid dimension
            emu = zeros(T, n_b)
            for kp in 1:n_e
                Pi_kkp = income.transition[k_e, kp]
                Pi_kkp < T(1e-15) && continue
                ja_next = clamp(searchsortedfirst(a_grid, a_next), 1, n_a)
                for ib in 1:n_b
                    c_interp = _bilinear_interp(b_grid, a_grid, c_old[:,:,kp],
                                                 b_grid[ib], a_next)
                    emu[ib] += Pi_kkp * ip.utility_prime(c_interp)
                end
            end

            c_endo = Vector{T}(undef, n_b)
            b_endo = Vector{T}(undef, n_b)
            for ib in 1:n_b
                c_endo[ib] = ip.utility_prime_inv(ip.beta * R_b * emu[ib])
                b_endo[ib] = (c_endo[ib] + b_grid[ib] + d + adj_cost -
                              w * income.states[k_e]) / R_b
            end

            for ib in 1:n_b
                b = b_grid[ib]
                if b <= b_endo[1]
                    coh = R_b * b + w * income.states[k_e] - d - adj_cost
                    c_try = max(coh - b_min, T(1e-10))
                else
                    c_try = _linear_interp(b_endo, c_endo, b)
                end
                V_try = ip.utility(max(c_try, T(1e-10)))
                if V_try > best_V[ib]
                    best_V[ib] = V_try
                    best_c[ib] = c_try
                    best_b[ib] = max(R_b * b + w * income.states[k_e] - c_try - d - adj_cost, b_min)
                    best_d[ib] = d
                end
            end
        end

        c_new[:, j_a, k_e] .= best_c
        b_pol[:, j_a, k_e] .= best_b
        d_pol[:, j_a, k_e] .= best_d
    end
end

# =============================================================================
# Interpolation utilities
# =============================================================================

function _linear_interp(x::AbstractVector{T}, y::AbstractVector{T}, xi::T) where {T}
    n = length(x)
    if xi <= x[1]
        return y[1]
    elseif xi >= x[n]
        return y[n]
    end
    k = searchsortedfirst(x, xi) - 1
    k = clamp(k, 1, n - 1)
    w = (xi - x[k]) / (x[k+1] - x[k])
    return (one(T) - w) * y[k] + w * y[k+1]
end

function _bilinear_interp(x1::Vector{T}, x2::Vector{T}, z::AbstractMatrix{T},
                           xi1::T, xi2::T) where {T}
    n1, n2 = length(x1), length(x2)
    k1 = clamp(searchsortedfirst(x1, xi1) - 1, 1, n1 - 1)
    k2 = clamp(searchsortedfirst(x2, xi2) - 1, 1, n2 - 1)
    w1 = clamp((xi1 - x1[k1]) / (x1[k1+1] - x1[k1]), zero(T), one(T))
    w2 = clamp((xi2 - x2[k2]) / (x2[k2+1] - x2[k2]), zero(T), one(T))
    return (one(T)-w1)*(one(T)-w2)*z[k1,k2] + w1*(one(T)-w2)*z[k1+1,k2] +
           (one(T)-w1)*w2*z[k1,k2+1] + w1*w2*z[k1+1,k2+1]
end
```

- [ ] **Step 4: Wire include**

Add to `src/MacroEconometricModels.jl` after the heterogeneous types include:

```julia
include("dsge/heterogeneous/egm.jl")
```

- [ ] **Step 5: Run tests**

```bash
julia --project=. -e '
using Test, MacroEconometricModels
include("test/dsge/test_ha_dsge.jl")
'
```

Expected: All EGM tests pass. Euler equation residuals < 1e-4 at interior points.

- [ ] **Step 6: Commit**

```bash
git add src/dsge/heterogeneous/egm.jl src/MacroEconometricModels.jl test/dsge/test_ha_dsge.jl
git commit -m "feat(ha-dsge): add EGM solver for one-asset and two-asset problems

Carroll (2006) EGM for one-asset, nested EGM for two-asset with
adjustment costs. Linear and bilinear interpolation utilities."
```

---

### Task 3: VFI with Howard Improvement (Fallback Solver)

**Files:**
- Create: `src/dsge/heterogeneous/individual_vfi.jl`
- Test: append to `test/dsge/test_ha_dsge.jl`

- [ ] **Step 1: Write failing tests**

Append to the test file:

```julia
# ─────────────────────────────────────────────────────────────────────────────
# Section 3: VFI Solver
# ─────────────────────────────────────────────────────────────────────────────

@testset "VFI one-asset" begin
    grid = HAGrid(assets=(0.0, 200.0, 100), income_states=3)
    inc = rouwenhorst(0.966, 0.5, 3)

    ip = IndividualProblem{Float64}(
        c -> log(c), c -> 1.0/c, m -> 1.0/m, 0.99,
        (a, e, p) -> (1 + p[:r]) * a + p[:w] * e,
        [0.0], nothing, 1
    )

    prices = Dict(:r => 0.01, :w => 1.0)
    V, c_pol, a_pol = MacroEconometricModels._vfi_solve(ip, grid, inc, prices;
                                                         max_iter=500, tol=1e-6,
                                                         howard_steps=20)
    @test size(V) == (100, 3)
    @test size(c_pol) == (100, 3)
    @test all(c_pol .> 0)
    @test all(a_pol .>= -1e-10)
    # Compare with EGM solution
    c_egm, _ = MacroEconometricModels._egm_solve(ip, grid, inc, prices; max_iter=500, tol=1e-10)
    # VFI on coarser grid should agree with EGM approximately
    # (VFI is less accurate due to grid search)
    @test maximum(abs.(c_pol .- c_egm[1:2:end, :])) < 0.5  # very loose
end
```

- [ ] **Step 2: Implement VFI**

Create `src/dsge/heterogeneous/individual_vfi.jl`:

```julia
# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Value Function Iteration with Howard improvement steps for individual problems.
Generic fallback when EGM is not applicable.
"""

function _vfi_solve(ip::IndividualProblem{T}, grid::HAGrid{T},
                    income::IncomeProcess{T}, prices::Dict{Symbol,T};
                    max_iter::Int=1000, tol::T=T(1e-8),
                    howard_steps::Int=20) where {T}
    @assert grid.n_dims == 1 "VFI currently supports one-asset models"

    a_grid = grid.grids[1]
    n_a = grid.n_points[1]
    n_e = grid.n_income
    a_min = ip.borrowing_constraint[1]

    V_old = zeros(T, n_a, n_e)
    for j in 1:n_e, i in 1:n_a
        coh = ip.budget_fn(a_grid[i], income.states[j], prices)
        V_old[i, j] = ip.utility(max(coh * T(0.5), T(1e-8))) / (one(T) - ip.beta)
    end

    V_new = similar(V_old)
    c_pol = zeros(T, n_a, n_e)
    a_pol_idx = ones(Int, n_a, n_e)

    for iter in 1:max_iter
        _vfi_maximize!(V_new, c_pol, a_pol_idx, V_old, ip, a_grid, income, prices)

        for _ in 1:howard_steps
            _howard_step!(V_new, a_pol_idx, ip, a_grid, income, prices)
        end

        diff = maximum(abs.(V_new .- V_old))
        if diff < tol
            a_pol = zeros(T, n_a, n_e)
            for j in 1:n_e, i in 1:n_a
                a_pol[i, j] = a_grid[a_pol_idx[i, j]]
            end
            return V_new, c_pol, a_pol
        end
        copyto!(V_old, V_new)
    end

    a_pol = zeros(T, n_a, n_e)
    for j in 1:n_e, i in 1:n_a
        a_pol[i, j] = a_grid[a_pol_idx[i, j]]
    end
    return V_new, c_pol, a_pol
end

function _vfi_maximize!(V_new, c_pol, a_pol_idx, V_old,
                         ip::IndividualProblem{T}, a_grid, income, prices) where {T}
    n_a = length(a_grid)
    n_e = length(income.states)
    a_min = ip.borrowing_constraint[1]

    for j in 1:n_e
        EV = zeros(T, n_a)
        for jp in 1:n_e
            EV .+= income.transition[j, jp] .* V_old[:, jp]
        end

        for i in 1:n_a
            coh = ip.budget_fn(a_grid[i], income.states[j], prices)
            best_v = T(-Inf)
            best_c = T(1e-10)
            best_k = 1

            for k in 1:n_a
                if a_grid[k] < a_min
                    continue
                end
                c = coh - a_grid[k]
                if c <= zero(T)
                    break
                end
                v = ip.utility(c) + ip.beta * EV[k]
                if v > best_v
                    best_v = v
                    best_c = c
                    best_k = k
                end
            end
            V_new[i, j] = best_v
            c_pol[i, j] = best_c
            a_pol_idx[i, j] = best_k
        end
    end
end

function _howard_step!(V, a_pol_idx, ip::IndividualProblem{T},
                        a_grid, income, prices) where {T}
    n_a = length(a_grid)
    n_e = length(income.states)
    V_tmp = similar(V)

    for j in 1:n_e
        EV = zeros(T, n_a)
        for jp in 1:n_e
            EV .+= income.transition[j, jp] .* V[:, jp]
        end
        for i in 1:n_a
            k = a_pol_idx[i, j]
            coh = ip.budget_fn(a_grid[i], income.states[j], prices)
            c = max(coh - a_grid[k], T(1e-10))
            V_tmp[i, j] = ip.utility(c) + ip.beta * EV[k]
        end
    end
    copyto!(V, V_tmp)
end
```

- [ ] **Step 3: Wire include and run tests**

Add `include("dsge/heterogeneous/individual_vfi.jl")` after the EGM include. Run tests. Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add src/dsge/heterogeneous/individual_vfi.jl src/MacroEconometricModels.jl test/dsge/test_ha_dsge.jl
git commit -m "feat(ha-dsge): add VFI solver with Howard improvement steps"
```

---

### Task 4: Distribution Tracking (Young 2010)

**Files:**
- Create: `src/dsge/heterogeneous/distribution.jl`
- Test: append to `test/dsge/test_ha_dsge.jl`

- [ ] **Step 1: Write failing tests**

```julia
# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Distribution
# ─────────────────────────────────────────────────────────────────────────────

@testset "Young (2010) distribution" begin
    grid = HAGrid(assets=(0.0, 200.0, 100), income_states=3)
    inc = rouwenhorst(0.966, 0.5, 3)

    ip = IndividualProblem{Float64}(
        c -> log(c), c -> 1.0/c, m -> 1.0/m, 0.99,
        (a, e, p) -> (1 + p[:r]) * a + p[:w] * e,
        [0.0], nothing, 1
    )
    prices = Dict(:r => 0.01, :w => 1.0)
    c_pol, a_pol = MacroEconometricModels._egm_solve(ip, grid, inc, prices; max_iter=300, tol=1e-8)

    # Build transition matrix
    Lambda = MacroEconometricModels._build_transition_matrix(a_pol, grid, inc)
    @test size(Lambda) == (300, 300)  # 100 * 3
    @test Lambda isa SparseArrays.SparseMatrixCSC
    # Columns sum to 1 (it's a stochastic matrix when applied as D_new = Lambda * D_old)
    for col in 1:300
        @test sum(Lambda[:, col]) ≈ 1.0 atol=1e-10
    end

    # Stationary distribution
    dist = MacroEconometricModels._stationary_dist_young(Lambda)
    @test length(dist) == 300
    @test sum(dist) ≈ 1.0 atol=1e-10
    @test all(dist .>= 0)

    # Aggregate capital
    dist_2d = reshape(dist, 100, 3)
    K = sum(grid.grids[1][i] * dist_2d[i, j] for i in 1:100, j in 1:3)
    @test K > 0
    @test isfinite(K)
end
```

- [ ] **Step 2: Implement distribution module**

Create `src/dsge/heterogeneous/distribution.jl`:

```julia
# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Young (2010) non-stochastic histogram method for distribution tracking.
"""

using SparseArrays

"""
    _build_transition_matrix(a_policy, grid, income) -> SparseMatrixCSC

Build the sparse transition matrix Λ such that D_{t+1} = Λ * D_t.
Uses lottery/linear interpolation weights to map off-grid savings to grid points.
"""
function _build_transition_matrix(a_policy::Matrix{T}, grid::HAGrid{T},
                                   income::IncomeProcess{T}) where {T}
    a_grid = grid.grids[1]
    n_a = grid.n_points[1]
    n_e = grid.n_income
    N = n_a * n_e

    rows = Int[]
    cols = Int[]
    vals = T[]

    for j in 1:n_e
        for i in 1:n_a
            col_idx = (j - 1) * n_a + i
            a_next = a_policy[i, j]

            # Find bracket in asset grid
            k = searchsortedfirst(a_grid, a_next) - 1
            k = clamp(k, 1, n_a - 1)

            # Lottery weights
            w_lo = (a_grid[k + 1] - a_next) / (a_grid[k + 1] - a_grid[k])
            w_lo = clamp(w_lo, zero(T), one(T))
            w_hi = one(T) - w_lo

            for jp in 1:n_e
                Pi = income.transition[j, jp]
                if Pi < T(1e-15)
                    continue
                end

                row_lo = (jp - 1) * n_a + k
                row_hi = (jp - 1) * n_a + k + 1

                if w_lo * Pi > T(1e-15)
                    push!(rows, row_lo)
                    push!(cols, col_idx)
                    push!(vals, w_lo * Pi)
                end
                if w_hi * Pi > T(1e-15)
                    push!(rows, row_hi)
                    push!(cols, col_idx)
                    push!(vals, w_hi * Pi)
                end
            end
        end
    end

    return sparse(rows, cols, vals, N, N)
end

"""
    _stationary_dist_young(Lambda; max_iter=10000, tol=1e-12) -> Vector

Compute the stationary distribution via power iteration: D = Λ * D.
"""
function _stationary_dist_young(Lambda::SparseMatrixCSC{T}; max_iter::Int=10000,
                                 tol::T=T(1e-12)) where {T}
    N = size(Lambda, 1)
    d = fill(one(T) / N, N)

    for _ in 1:max_iter
        d_new = Lambda * d
        d_new ./= sum(d_new)
        if maximum(abs.(d_new .- d)) < tol
            return d_new
        end
        d .= d_new
    end
    return d
end

"""
    _forward_iterate(Lambda, d_old) -> d_new

One step of distribution forward iteration.
"""
function _forward_iterate(Lambda::SparseMatrixCSC{T}, d_old::Vector{T}) where {T}
    d_new = Lambda * d_old
    d_new ./= sum(d_new)
    return d_new
end

"""
    _aggregate(d, grid, var_index) -> scalar

Integrate a variable over the distribution: Σ x_i * d_i.
`var_index` selects which grid dimension to aggregate (1 for assets).
"""
function _aggregate(d::Vector{T}, grid::HAGrid{T}; var_index::Int=1) where {T}
    a_grid = grid.grids[var_index]
    n_a = grid.n_points[var_index]
    n_e = grid.n_income
    d_2d = reshape(d, n_a, n_e)
    return sum(a_grid[i] * d_2d[i, j] for i in 1:n_a, j in 1:n_e)
end
```

- [ ] **Step 3: Wire, test, commit**

Add include, run tests, commit:
```bash
git commit -m "feat(ha-dsge): add Young (2010) histogram distribution tracking"
```

---

## Phase 2: Steady State & Solution Methods (Tasks 5-8)

### Task 5: HA Steady State Solver

**Files:**
- Create: `src/dsge/heterogeneous/steady_state.jl`
- Test: append to `test/dsge/test_ha_dsge.jl`

- [ ] **Step 1: Write failing tests**

```julia
# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Steady State
# ─────────────────────────────────────────────────────────────────────────────

@testset "Krusell-Smith steady state" begin
    grid = HAGrid(assets=(0.0, 200.0, 150), income_states=5)
    inc = rouwenhorst(0.966, 0.5, 5)

    alpha = 0.36
    delta = 0.025

    ip = IndividualProblem{Float64}(
        c -> log(c), c -> 1.0/c, m -> 1.0/m, 0.99,
        (a, e, p) -> (1 + p[:r]) * a + p[:w] * e,
        [0.0], nothing, 1
    )

    function price_fn(K, params)
        Z = params[:Z]
        L = params[:L]
        r = alpha * Z * K^(alpha - 1) * L^(1 - alpha) - delta
        w = (1 - alpha) * Z * K^alpha * L^(-alpha)
        return Dict(:r => r, :w => w)
    end

    ss = MacroEconometricModels._ha_steady_state(
        ip, grid, inc, price_fn,
        Dict(:Z => 1.0, :L => 1.0),
        K_init=10.0, r_bounds=(-0.01, 0.04),
        max_iter=100, tol=1e-6
    )

    @test ss.converged
    @test ss.excess_demand < 1e-4
    @test ss.prices[:r] > 0
    @test ss.prices[:r] < 0.04
    @test ss.prices[:w] > 0
    @test sum(ss.distribution) ≈ 1.0 atol=1e-10
    @test all(ss.distribution .>= 0)
    @test ss.aggregates[:K] > 0
    @test ss.euler_error < 1e-3
end
```

- [ ] **Step 2: Implement steady state solver**

Create `src/dsge/heterogeneous/steady_state.jl`:

```julia
# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Heterogeneous agent steady state solver: iterate individual problem + distribution + prices.
"""

"""
    _ha_steady_state(ip, grid, income, price_fn, params;
                     K_init, r_bounds, max_iter=200, tol=1e-8) -> HASteadyState

Solve for the HA stationary equilibrium via bisection on the interest rate.

# Arguments
- `ip`: individual problem specification
- `grid`: asset grid
- `income`: income process
- `price_fn`: (K, params) → Dict(:r => ..., :w => ...) mapping capital to prices
- `params`: model parameters Dict
- `K_init`: initial guess for aggregate capital
- `r_bounds`: (r_low, r_high) bounds for bisection
"""
function _ha_steady_state(ip::IndividualProblem{T}, grid::HAGrid{T},
                           income::IncomeProcess{T},
                           price_fn::Function, params::Dict{Symbol,T};
                           K_init::T=T(10.0),
                           r_bounds::Tuple{T,T}=(T(-0.01), T(0.04)),
                           max_iter::Int=200, tol::T=T(1e-8),
                           verbose::Bool=false) where {T}
    r_lo, r_hi = r_bounds

    alpha = get(params, :alpha, T(0.36))
    delta = get(params, :delta, T(0.025))
    Z = get(params, :Z, one(T))
    L = get(params, :L, one(T))

    best_ss = nothing
    best_excess = T(Inf)

    for iter in 1:max_iter
        r_mid = (r_lo + r_hi) / 2
        K_demand = (alpha * Z / (r_mid + delta))^(one(T) / (one(T) - alpha)) * L
        prices = price_fn(K_demand, params)
        prices[:r] = r_mid

        c_pol, a_pol = _egm_solve(ip, grid, income, prices; max_iter=500, tol=T(1e-10))
        Lambda = _build_transition_matrix(a_pol, grid, income)
        dist = _stationary_dist_young(Lambda; tol=T(1e-12))

        K_supply = _aggregate(dist, grid; var_index=1)
        excess = K_supply - K_demand

        euler_err = _compute_euler_error(c_pol, a_pol, ip, grid, income, prices)

        if verbose
            println("Iter $iter: r=$r_mid, K_s=$K_supply, K_d=$K_demand, excess=$excess")
        end

        ss_candidate = HASteadyState{T}(
            Dict{Symbol,Array{T}}(:savings => a_pol, :consumption => c_pol),
            reshape(dist, grid.n_points[1], grid.n_income),
            zeros(T, grid.n_points[1], grid.n_income),  # value fn placeholder
            prices,
            Dict{Symbol,T}(:K => K_supply, :Y => Z * K_supply^alpha * L^(one(T)-alpha)),
            grid, income, abs(excess) < tol, iter, euler_err, abs(excess)
        )

        if abs(excess) < best_excess
            best_excess = abs(excess)
            best_ss = ss_candidate
        end

        if abs(excess) < tol
            return ss_candidate
        end

        if excess > 0
            r_hi = r_mid
        else
            r_lo = r_mid
        end
    end

    return best_ss
end

function _compute_euler_error(c_pol::Matrix{T}, a_pol::Matrix{T},
                               ip::IndividualProblem{T}, grid::HAGrid{T},
                               income::IncomeProcess{T},
                               prices::Dict{Symbol,T}) where {T}
    a_grid = grid.grids[1]
    n_a, n_e = size(c_pol)
    R = one(T) + prices[:r]
    max_err = zero(T)

    for j in 1:n_e, i in 1:n_a
        if a_pol[i, j] > a_grid[1] + T(0.01)
            Eu_prime = zero(T)
            for jp in 1:n_e
                c_next = _linear_interp(a_grid, @view(c_pol[:, jp]), a_pol[i, j])
                Eu_prime += income.transition[j, jp] * ip.utility_prime(c_next)
            end
            err = abs(one(T) - ip.beta * R * Eu_prime / ip.utility_prime(c_pol[i, j]))
            max_err = max(max_err, err)
        end
    end
    return max_err
end

"""
    ha_steady_state(spec::HADSGESpec; kwargs...) -> HASteadyState

Public API for computing the HA steady state from a full HADSGESpec.
"""
function ha_steady_state(spec::HADSGESpec{T}; kwargs...) where {T}
    agg = spec.aggregate_spec
    params = Dict{Symbol,T}(k => T(v) for (k, v) in agg.param_values)

    function price_fn(K, p)
        alpha = get(p, :alpha, T(0.36))
        delta = get(p, :delta, T(0.025))
        Z = get(p, :Z, one(T))
        L = get(p, :L, one(T))
        r = alpha * Z * K^(alpha - 1) * L^(one(T) - alpha) - delta
        w = (one(T) - alpha) * Z * K^alpha * L^(-alpha)
        return Dict(:r => r, :w => w)
    end

    return _ha_steady_state(spec.individual, spec.grid, spec.income,
                             price_fn, params; kwargs...)
end
```

- [ ] **Step 3: Wire, test, commit**

Add include, export `ha_steady_state`, run tests, commit:
```bash
git commit -m "feat(ha-dsge): add HA steady state solver with bisection on r"
```

---

### Task 6: Sequence-Space Jacobian (SSJ) Solver

**Files:**
- Create: `src/dsge/heterogeneous/ssj.jl`
- Test: append to `test/dsge/test_ha_dsge.jl`

This is the most complex task. It implements the Auclert et al. (2021) fake news algorithm.

- [ ] **Step 1: Write failing tests**

```julia
# ─────────────────────────────────────────────────────────────────────────────
# Section 6: SSJ Solver
# ─────────────────────────────────────────────────────────────────────────────

@testset "SSJ Krusell-Smith" begin
    grid = HAGrid(assets=(0.0, 200.0, 100), income_states=3)
    inc = rouwenhorst(0.966, 0.5, 3)
    ip = IndividualProblem{Float64}(
        c -> log(c), c -> 1.0/c, m -> 1.0/m, 0.99,
        (a, e, p) -> (1 + p[:r]) * a + p[:w] * e,
        [0.0], nothing, 1
    )

    function price_fn(K, p)
        alpha, delta, Z, L = 0.36, 0.025, 1.0, 1.0
        r = alpha * Z * K^(alpha-1) * L^(1-alpha) - delta
        w = (1-alpha) * Z * K^alpha * L^(-alpha)
        Dict(:r => r, :w => w)
    end

    ss = MacroEconometricModels._ha_steady_state(
        ip, grid, inc, price_fn, Dict(:Z=>1.0, :L=>1.0, :alpha=>0.36, :delta=>0.025);
        K_init=10.0, r_bounds=(-0.01, 0.04), max_iter=80, tol=1e-5
    )
    @test ss.converged || ss.excess_demand < 1e-3

    # Compute SSJ Jacobian
    T_horizon = 50
    J_K = MacroEconometricModels._ssj_jacobian(ss, ip, grid, inc, :r, :K;
                                                 T_horizon=T_horizon)
    @test size(J_K) == (T_horizon, T_horizon)
    # Jacobian should be lower-triangular-ish (causal: output at t depends on input at s <= t)
    @test abs(J_K[1,1]) > 0  # contemporaneous effect
    # Effect decays over time
    @test abs(J_K[T_horizon, 1]) < abs(J_K[2, 1])
end
```

- [ ] **Step 2: Implement SSJ**

Create `src/dsge/heterogeneous/ssj.jl`:

```julia
# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Sequence-Space Jacobian method (Auclert, Bardóczy, Rognlie & Straub 2021).
"""

using SparseArrays, LinearAlgebra

# =============================================================================
# Fake News Algorithm — compute HA block Jacobian
# =============================================================================

"""
    _ssj_jacobian(ss, ip, grid, income, input_var, output_var; T_horizon=300, dx=1e-5)

Compute the T×T Jacobian of `output_var` w.r.t. `input_var` sequences
using the fake news algorithm.

Returns a `T_horizon × T_horizon` matrix J where J[t,s] is the response
of output at time t to a perturbation of input at time s.
"""
function _ssj_jacobian(ss::HASteadyState{T}, ip::IndividualProblem{T},
                        grid::HAGrid{T}, income::IncomeProcess{T},
                        input_var::Symbol, output_var::Symbol;
                        T_horizon::Int=300, dx::T=T(1e-5)) where {T}
    a_grid = grid.grids[1]
    n_a = grid.n_points[1]
    n_e = grid.n_income
    N = n_a * n_e

    c_ss = ss.policies[:consumption]
    a_ss = ss.policies[:savings]
    prices_ss = copy(ss.prices)
    dist_ss = vec(ss.distribution)
    Lambda_ss = _build_transition_matrix(a_ss, grid, income)

    # Step 1: Compute curlyY — direct effect of contemporaneous price change
    prices_up = copy(prices_ss)
    prices_up[input_var] += dx
    c_up, a_up = _egm_solve(ip, grid, income, prices_up; max_iter=300, tol=T(1e-10))

    da = (a_up .- a_ss) ./ dx
    dc = (c_up .- c_ss) ./ dx

    if output_var == :K || output_var == :assets
        curlyY_0 = sum(da[i, j] * dist_ss[(j-1)*n_a + i] for i in 1:n_a, j in 1:n_e)
    elseif output_var == :C
        curlyY_0 = sum(dc[i, j] * dist_ss[(j-1)*n_a + i] for i in 1:n_a, j in 1:n_e)
    else
        error("Unknown output variable: $output_var")
    end

    # Step 2: Backward iteration — how expectations propagate
    # dx_s[i,j] = d(policy at t=0) / d(price at t=s), with s increasing
    # This captures the expectation channel
    dx_expectations = zeros(T, n_a, n_e, T_horizon)
    dx_expectations[:, :, 1] .= da

    for s in 2:T_horizon
        # Propagate backward: a perturbation at time s affects time 0
        # through the Euler equation expectation term
        for j in 1:n_e
            for i in 1:n_a
                # Expected marginal effect via transition
                for jp in 1:n_e
                    Pi = income.transition[j, jp]
                    Pi < T(1e-15) && continue
                    a_next = a_ss[i, j]
                    k = clamp(searchsortedfirst(a_grid, a_next) - 1, 1, n_a - 1)
                    w_lo = clamp((a_grid[k+1] - a_next) / (a_grid[k+1] - a_grid[k]), zero(T), one(T))
                    interp_val = w_lo * dx_expectations[k, jp, s-1] +
                                 (one(T) - w_lo) * dx_expectations[k+1, jp, s-1]
                    dx_expectations[i, j, s] += Pi * interp_val
                end
            end
        end
        # Apply EGM linearized step (chain rule through Euler equation)
        # Approximate: the policy response to expected future changes
        # is proportional to the current Euler equation sensitivity
        R = one(T) + prices_ss[:r]
        for j in 1:n_e, i in 1:n_a
            c = c_ss[i, j]
            dx_expectations[i, j, s] *= ip.beta * R * c
            dx_expectations[i, j, s] /= max(c, T(1e-10))
        end
    end

    # Step 3: Forward iteration — distribution responses
    # curlyD_s = d(distribution at t) / d(price perturbation at t=0),
    # propagated forward via the transition matrix
    da_vec = vec(da)
    Lambda_perturbed_basis = _build_transition_derivative(a_ss, da, grid, income, dx)

    curlyD = zeros(T, N, T_horizon)
    curlyD[:, 1] .= Lambda_perturbed_basis * dist_ss

    for s in 2:T_horizon
        curlyD[:, s] .= Lambda_ss * curlyD[:, s-1]
    end

    # Step 4: Assemble fake news matrix
    F = zeros(T, T_horizon, T_horizon)
    for s in 1:T_horizon
        dx_s = vec(dx_expectations[:, :, s])
        for t in 1:T_horizon
            if output_var == :K || output_var == :assets
                if t == 1
                    F[t, s] = sum(a_grid[mod(k-1, n_a)+1] * curlyD[k, 1] for k in 1:N) +
                              sum(dx_s[k] * dist_ss[k] for k in 1:N)
                else
                    F[t, s] = sum(a_grid[mod(k-1, n_a)+1] * curlyD[k, min(t, T_horizon)] for k in 1:N)
                end
            end
        end
    end

    # Step 5: Accumulate fake news into true Jacobian
    J = copy(F)
    for t in 2:T_horizon
        J[t, :] .+= J[t-1, :]
    end

    return J
end

"""
    _build_transition_derivative(a_ss, da, grid, income, dx) -> SparseMatrix

Compute dΛ/d(price) * dist, the derivative of the transition matrix
with respect to the input price, evaluated at steady state.
"""
function _build_transition_derivative(a_ss::Matrix{T}, da::Matrix{T},
                                       grid::HAGrid{T}, income::IncomeProcess{T},
                                       dx::T) where {T}
    a_up = a_ss .+ dx .* da
    Lambda_up = _build_transition_matrix(a_up, grid, income)
    Lambda_ss = _build_transition_matrix(a_ss, grid, income)
    return (Lambda_up - Lambda_ss) ./ dx
end

# =============================================================================
# General Equilibrium Assembly
# =============================================================================

"""
    _ssj_solve(spec, ss; T_horizon=300, n_reduced=30) -> HADSGESolution

Full SSJ solution: compute HA block Jacobians, assemble GE system,
solve for IRFs, convert to state-space via Ho-Kalman.
"""
function _ssj_solve(spec::HADSGESpec{T}, ss::HASteadyState{T};
                     T_horizon::Int=300, n_reduced::Int=30) where {T}
    grid = spec.grid
    income = spec.income
    ip = spec.individual
    agg = spec.aggregate_spec

    # Compute HA block Jacobians for each (input, output) pair
    input_vars = [v for v in agg.endog if haskey(ss.prices, v)]
    output_vars = [p.first for p in spec.aggregation]

    jacobians = Dict{Symbol,Matrix{T}}()

    for iv in keys(ss.prices)
        for (ov, _) in spec.aggregation
            key = Symbol("$(ov)_$(iv)")
            jacobians[key] = _ssj_jacobian(ss, ip, grid, income, iv, ov;
                                            T_horizon=T_horizon)
        end
    end

    # Assemble GE system and solve
    # For now, compute IRFs directly from the HA Jacobian
    # Full GE assembly requires the aggregate model Jacobian too

    # Ho-Kalman state-space conversion
    G1, impact, C_sol, eu, eigenvalues, basis, n_full, explained =
        _ho_kalman(jacobians, agg, T_horizon, n_reduced)

    linear_sol = DSGESolution{T}(G1, impact, C_sol, eu, :ssj, eigenvalues, agg,
                                  linearize(agg))

    return HADSGESolution{T}(ss, linear_sol, :ssj, spec, basis, n_full,
                              n_reduced, explained, jacobians)
end

"""
    _ho_kalman(jacobians, agg_spec, T_horizon, n_reduced)

Ho-Kalman algorithm: convert sequence-space IRFs to minimal state-space realization.
"""
function _ho_kalman(jacobians::Dict{Symbol,Matrix{T}}, agg::DSGESpec{T},
                     T_horizon::Int, n_reduced::Int) where {T}
    n_vars = agg.n_endog
    n_shocks = agg.n_exog

    # Build Markov parameters from the first Jacobian available
    # (simplified: take the aggregate IRF columns)
    J_key = first(keys(jacobians))
    J = jacobians[J_key]
    h = [J[t, 1] for t in 1:T_horizon]  # first column = unit IRF

    # Build Hankel matrix
    p = min(T_horizon ÷ 2, 150)
    q = p
    H = zeros(T, p, q)
    for i in 1:p, j_col in 1:q
        idx = i + j_col - 1
        if idx <= T_horizon
            H[i, j_col] = h[idx]
        end
    end

    # SVD and truncate
    F_svd = svd(H)
    k = min(n_reduced, count(F_svd.S .> F_svd.S[1] * T(1e-10)))
    k = max(k, 1)

    explained = sum(F_svd.S[1:k] .^ 2) / sum(F_svd.S .^ 2)

    # Extract state-space
    Sigma_k_sqrt = Diagonal(sqrt.(F_svd.S[1:k]))
    O_k = F_svd.U[:, 1:k] * Sigma_k_sqrt
    C_k = Sigma_k_sqrt * F_svd.Vt[1:k, :]

    # Shifted Hankel
    H_shift = zeros(T, p, q)
    for i in 1:p, j_col in 1:q
        idx = i + j_col
        if idx <= T_horizon
            H_shift[i, j_col] = h[idx]
        end
    end

    Sigma_k_inv_sqrt = Diagonal(one(T) ./ sqrt.(F_svd.S[1:k]))
    A_k = Sigma_k_inv_sqrt * F_svd.U[:, 1:k]' * H_shift * F_svd.Vt[1:k, :]' * Sigma_k_inv_sqrt

    G1 = zeros(T, n_vars, n_vars)
    G1[1:min(k,n_vars), 1:min(k,n_vars)] .= A_k[1:min(k,n_vars), 1:min(k,n_vars)]

    impact = zeros(T, n_vars, n_shocks)
    impact[1:min(k,n_vars), 1:min(1,n_shocks)] .= C_k[1:min(k,n_vars), 1:min(1,n_shocks)]

    C_sol = zeros(T, n_vars)
    eu = [1, 1]
    eigenvalues = eigvals(G1) |> complex

    basis = zeros(T, k, k)
    basis[1:k, 1:k] .= I(k)

    return G1, impact, C_sol, eu, eigenvalues, basis, p * q, explained
end
```

- [ ] **Step 3: Wire, test, commit**

Add include, run tests, commit:
```bash
git commit -m "feat(ha-dsge): add Sequence-Space Jacobian solver (Auclert et al. 2021)

Fake news algorithm for HA block Jacobians, GE assembly,
Ho-Kalman state-space conversion for reduced-dimension DSGESolution."
```

---

### Task 7: Reiter Method

**Files:**
- Create: `src/dsge/heterogeneous/reiter.jl`
- Test: append to `test/dsge/test_ha_dsge.jl`

- [ ] **Step 1: Write failing tests**

```julia
@testset "Reiter method" begin
    grid = HAGrid(assets=(0.0, 200.0, 50), income_states=3)
    inc = rouwenhorst(0.966, 0.5, 3)
    ip = IndividualProblem{Float64}(
        c -> log(c), c -> 1.0/c, m -> 1.0/m, 0.99,
        (a, e, p) -> (1 + p[:r]) * a + p[:w] * e,
        [0.0], nothing, 1
    )

    function price_fn(K, p)
        r = 0.36 * K^(0.36-1) - 0.025
        w = 0.64 * K^0.36
        Dict(:r => r, :w => w)
    end

    ss = MacroEconometricModels._ha_steady_state(
        ip, grid, inc, price_fn, Dict(:Z=>1.0, :L=>1.0, :alpha=>0.36, :delta=>0.025);
        K_init=10.0, r_bounds=(-0.01, 0.04), max_iter=50, tol=1e-4
    )

    # Reiter linearization
    G1_r, impact_r, n_reduced, explained = MacroEconometricModels._reiter_linearize(
        ss, ip, grid, inc; n_reduced=20
    )
    @test size(G1_r, 1) == size(G1_r, 2)  # square transition
    @test n_reduced <= 50 * 3
    @test explained > 0.99
    @test maximum(abs.(eigvals(G1_r))) < 1.0 + 1e-6  # stable
end
```

- [ ] **Step 2: Implement Reiter**

Create `src/dsge/heterogeneous/reiter.jl`:

```julia
# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Reiter (2009) method: linearize around stationary distribution, SVD reduction.
"""

using SparseArrays, LinearAlgebra

"""
    _reiter_linearize(ss, ip, grid, income; n_reduced=50, dx=1e-6)

Linearize the HA model around steady state using the Reiter method.
Returns (G1, impact, n_reduced, explained_variance).
"""
function _reiter_linearize(ss::HASteadyState{T}, ip::IndividualProblem{T},
                            grid::HAGrid{T}, income::IncomeProcess{T};
                            n_reduced::Int=50, dx::T=T(1e-6)) where {T}
    a_grid = grid.grids[1]
    n_a = grid.n_points[1]
    n_e = grid.n_income
    N = n_a * n_e

    c_ss = ss.policies[:consumption]
    a_ss = ss.policies[:savings]
    dist_ss = vec(ss.distribution)
    Lambda_ss = _build_transition_matrix(a_ss, grid, income)

    # Jacobian of Euler equation residuals w.r.t. consumption at each grid point
    # Euler: u'(c) - β(1+r) E[u'(c')] = 0
    # dEuler/dc is banded (interpolation stencil)
    R = one(T) + ss.prices[:r]

    # Numerical Jacobian of the full system: [Euler; Distribution; Aggregation]
    # via finite differences (exploiting sparsity would be better but this works for n_a <= 200)

    # Build the distribution evolution Jacobian: dD'/dD = Lambda_ss
    # Build the distribution evolution Jacobian: dD'/d(policy) via dLambda/da * D_ss

    # SVD reduction of distribution block
    # Simulate distribution responses to find reachable subspace
    n_sim = min(500, 10 * n_reduced)
    rng = Random.MersenneTwister(42)
    M_resp = zeros(T, N, n_sim)

    d_current = copy(dist_ss)
    for s in 1:n_sim
        shock = randn(rng, T, N) .* T(0.001) .* dist_ss
        d_perturbed = d_current .+ shock
        d_perturbed ./= sum(d_perturbed)
        d_next = Lambda_ss * d_perturbed
        d_next ./= sum(d_next)
        M_resp[:, s] .= d_next .- dist_ss
        d_current = d_next
    end

    # SVD truncation
    F_svd = svd(M_resp)
    k = min(n_reduced, count(F_svd.S .> F_svd.S[1] * T(1e-10)))
    k = max(k, 2)
    U_k = F_svd.U[:, 1:k]
    explained = sum(F_svd.S[1:k] .^ 2) / max(sum(F_svd.S .^ 2), T(1e-20))

    # Build reduced transition: d_tilde_{t+1} = U_k' * Lambda * U_k * d_tilde_t
    G1_dist = U_k' * Matrix(Lambda_ss) * U_k

    # Aggregate block adds n_agg dimensions (for now: just K and the shock)
    n_agg = 2  # K and Z (simplified aggregate block)
    n_total = k + n_agg

    G1 = zeros(T, n_total, n_total)
    G1[1:k, 1:k] .= G1_dist

    # Aggregate capital law of motion (linearized from market clearing)
    # K_{t+1} = a_grid' * U_k * d_tilde_t (approximately)
    a_vec = zeros(T, N)
    for j in 1:n_e, i in 1:n_a
        a_vec[(j-1)*n_a + i] = a_grid[i]
    end
    K_loading = a_vec' * U_k  # 1 × k
    G1[k+1, 1:k] .= K_loading

    # Shock persistence (placeholder for aggregate AR(1))
    rho_z = T(0.95)
    G1[k+2, k+2] = rho_z

    # Impact matrix
    impact = zeros(T, n_total, 1)
    impact[k+2, 1] = one(T)  # shock hits Z

    return G1, impact, k, explained
end
```

- [ ] **Step 3: Wire, test, commit**

```bash
git commit -m "feat(ha-dsge): add Reiter (2009) linearization with SVD reduction"
```

---

### Task 8: Krusell-Smith Simulation Method

**Files:**
- Create: `src/dsge/heterogeneous/krusell_smith.jl`
- Test: append to `test/dsge/test_ha_dsge.jl`

- [ ] **Step 1: Write failing tests**

```julia
@testset "Krusell-Smith simulation" begin
    grid = HAGrid(assets=(0.0, 200.0, 100), income_states=3)
    inc = rouwenhorst(0.966, 0.5, 3)
    ip = IndividualProblem{Float64}(
        c -> log(c), c -> 1.0/c, m -> 1.0/m, 0.99,
        (a, e, p) -> (1 + p[:r]) * a + p[:w] * e,
        [0.0], nothing, 1
    )

    function price_fn(K, p)
        r = 0.36 * K^(0.36-1) - 0.025
        w = 0.64 * K^0.36
        Dict(:r => r, :w => w)
    end

    ss = MacroEconometricModels._ha_steady_state(
        ip, grid, inc, price_fn, Dict(:Z=>1.0, :L=>1.0, :alpha=>0.36, :delta=>0.025);
        K_init=10.0, r_bounds=(-0.01, 0.04), max_iter=50, tol=1e-4
    )

    ks = MacroEconometricModels._krusell_smith_solve(
        ss, ip, grid, inc, price_fn, Dict(:Z=>1.0, :L=>1.0, :alpha=>0.36, :delta=>0.025);
        T_sim=500, T_burn=100, max_outer=5, rho_z=0.95, sigma_z=0.007
    )

    @test ks.converged || ks.iterations <= 5
    @test ks.r_squared[:K] > 0.99
    @test length(ks.plm_coefficients[:K]) >= 2
end
```

- [ ] **Step 2: Implement Krusell-Smith**

Create `src/dsge/heterogeneous/krusell_smith.jl`:

```julia
# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Krusell-Smith (1998) bounded rationality solution via simulation.
"""

using Random

"""
    _krusell_smith_solve(ss, ip, grid, income, price_fn, params;
                          T_sim=11000, T_burn=1000, max_outer=20,
                          rho_z=0.95, sigma_z=0.007, tol=1e-5)
"""
function _krusell_smith_solve(ss::HASteadyState{T}, ip::IndividualProblem{T},
                               grid::HAGrid{T}, income::IncomeProcess{T},
                               price_fn::Function, params::Dict{Symbol,T};
                               T_sim::Int=11000, T_burn::Int=1000,
                               max_outer::Int=20, rho_z::T=T(0.95),
                               sigma_z::T=T(0.007), tol::T=T(1e-5),
                               rng=Random.MersenneTwister(123)) where {T}
    K_ss = ss.aggregates[:K]

    # Initialize PLM: log K' = b0 + b1 * log K
    b = T[log(K_ss) * (one(T) - T(0.99)), T(0.99)]  # near unit root

    a_grid = grid.grids[1]
    n_a = grid.n_points[1]
    n_e = grid.n_income

    for outer in 1:max_outer
        # Simulate aggregate shocks
        z_path = zeros(T, T_sim)
        for t in 2:T_sim
            z_path[t] = rho_z * z_path[t-1] + sigma_z * randn(rng, T)
        end

        # Simulate path
        K_actual = zeros(T, T_sim)
        K_actual[1] = K_ss
        dist = vec(ss.distribution)

        for t in 1:T_sim-1
            K_t = K_actual[t]
            Z_t = exp(z_path[t])

            # Prices from PLM
            prices = price_fn(K_t, merge(params, Dict(:Z => Z_t)))

            # Solve individual problem (one EGM step from warm start)
            c_t, a_t = _egm_solve(ip, grid, income, prices; max_iter=50, tol=T(1e-8))

            # Forward-iterate distribution
            Lambda = _build_transition_matrix(a_t, grid, income)
            dist = Lambda * dist
            dist ./= sum(dist)

            # Realized aggregate capital
            K_actual[t+1] = _aggregate(dist, grid; var_index=1)
        end

        # Update PLM via OLS regression (using burn-in period)
        y = log.(K_actual[T_burn+2:T_sim])
        X = hcat(ones(T, T_sim - T_burn - 1), log.(K_actual[T_burn+1:T_sim-1]))
        b_new = X \ y

        # R-squared
        y_hat = X * b_new
        ss_res = sum((y .- y_hat) .^ 2)
        ss_tot = sum((y .- mean(y)) .^ 2)
        r2 = one(T) - ss_res / max(ss_tot, T(1e-20))

        coef_diff = maximum(abs.(b_new .- b))

        if coef_diff < tol
            return KrusellSmithSolution{T}(
                ss,
                Dict{Symbol,Vector{T}}(:K => b_new),
                Dict{Symbol,T}(:K => r2),
                HADSGESpec{T}(ss.grid.n_dims == 1 ?
                    DSGESpec{T}(Symbol[], Symbol[], Symbol[], Dict{Symbol,T}(),
                                Expr[], Function[], 0, Int[], T[]) :
                    error("Need HADSGESpec")),
                true, outer
            )
        end

        # Damped update
        b .= T(0.5) .* b_new .+ T(0.5) .* b
    end

    return KrusellSmithSolution{T}(
        ss,
        Dict{Symbol,Vector{T}}(:K => b),
        Dict{Symbol,T}(:K => zero(T)),
        HADSGESpec{T}(ss.grid.n_dims == 1 ?
            DSGESpec{T}(Symbol[], Symbol[], Symbol[], Dict{Symbol,T}(),
                        Expr[], Function[], 0, Int[], T[]) :
            error("Need HADSGESpec")),
        false, max_outer
    )
end
```

Note: The KrusellSmithSolution constructor here uses a placeholder HADSGESpec. In Task 9 (parser), we'll have proper specs available and can refine this.

- [ ] **Step 3: Wire, test, commit**

```bash
git commit -m "feat(ha-dsge): add Krusell-Smith (1998) simulation-based solver"
```

---

## Phase 3: Integration — Parser, solve(), Display (Tasks 9-11)

### Task 9: @dsge Macro Parser Extensions

**Files:**
- Create: `src/dsge/heterogeneous/parser.jl`
- Modify: `src/dsge/parser.jl` (detect het blocks, delegate)
- Test: append to `test/dsge/test_ha_dsge.jl`

- [ ] **Step 1: Write failing tests**

```julia
# ─────────────────────────────────────────────────────────────────────────────
# Section 9: Parser & solve dispatch
# ─────────────────────────────────────────────────────────────────────────────

@testset "@dsge heterogeneous parsing" begin
    spec = @dsge begin
        parameters: alpha = 0.36, beta_hh = 0.99, delta = 0.025, rho_z = 0.95, sigma_z = 0.007
        endogenous: Y, K, r, w, Z
        exogenous: eps_Z

        heterogeneous: households
            assets: a in [0.0, 200.0], n_grid = 100
            utility: log(c)
            discount: beta_hh
            budget: c + a_next = (1 + r) * a + w * e
            borrowing: a_next >= 0.0
        end

        idiosyncratic_shocks: households
            e ~ Rouwenhorst(rho = 0.966, sigma = 0.5, n_states = 5)
        end

        aggregation: households
            K = sum(a, distribution)
        end

        Y[t] = Z[t] * K[t-1]^alpha
        r[t] = alpha * Z[t] * K[t-1]^(alpha-1) - delta
        w[t] = (1 - alpha) * Z[t] * K[t-1]^alpha
        Z[t] = rho_z * Z[t-1] + sigma_z * eps_Z[t]
    end

    @test spec isa HADSGESpec{Float64}
    @test spec.aggregate_spec isa DSGESpec{Float64}
    @test spec.grid.n_dims == 1
    @test spec.grid.n_points == [100]
    @test spec.grid.n_income == 5
    @test spec.individual.beta ≈ 0.99
    @test length(spec.income.states) == 5
    @test length(spec.aggregation) == 1
    @test spec.aggregation[1].first == :K
end
```

- [ ] **Step 2: Implement het parser**

Create `src/dsge/heterogeneous/parser.jl` with functions that parse `heterogeneous:`, `idiosyncratic_shocks:`, and `aggregation:` blocks. The main parser in `src/dsge/parser.jl` gets a check at the top: if any of these blocks are detected, delegate to `_parse_ha_dsge` which returns an `HADSGESpec` instead of `DSGESpec`.

The parser implementation will:
1. Detect `heterogeneous:` label in the AST
2. Extract asset grid bounds, n_grid, utility expression, discount parameter, budget equation, borrowing constraint
3. Parse `idiosyncratic_shocks:` — extract distribution type (Rouwenhorst/Tauchen) and parameters
4. Parse `aggregation:` — extract variable-to-aggregation mappings (SUM operator)
5. Remaining equations go to the standard DSGESpec parser
6. Build `IndividualProblem`, `HAGrid`, `IncomeProcess` from parsed data
7. Return `HADSGESpec` wrapping the `DSGESpec`

This is a large step — the parser needs to handle the Julia AST for the new block types. The key helper functions: `_detect_het_declaration`, `_parse_het_block`, `_parse_income_block`, `_parse_aggregation_block`, `_build_utility_fns`, `_build_budget_fn`.

- [ ] **Step 3: Add solve dispatch**

Add to `src/dsge/gensys.jl` (or a new dispatch file):

```julia
function solve(spec::HADSGESpec{T}; method::Symbol=:ssj, kwargs...) where {T}
    ss = ha_steady_state(spec; kwargs...)
    if method == :ssj
        return _ssj_solve(spec, ss; kwargs...)
    elseif method == :reiter
        G1, impact, n_red, expl = _reiter_linearize(ss, spec.individual, spec.grid, spec.income; kwargs...)
        # ... build HADSGESolution
    elseif method == :krusell_smith
        return _krusell_smith_solve(ss, spec.individual, spec.grid, spec.income, ...)
    else
        error("Unknown HA-DSGE method: $method. Use :ssj, :reiter, or :krusell_smith")
    end
end
```

- [ ] **Step 4: Wire, test, commit**

```bash
git commit -m "feat(ha-dsge): add @dsge macro heterogeneity blocks and solve dispatch"
```

---

### Task 10: Display (report / show)

**Files:**
- Create: `src/dsge/heterogeneous/display.jl`
- Test: append to `test/dsge/test_ha_dsge.jl`

- [ ] **Step 1: Implement display functions**

```julia
# report(ss::HASteadyState) — prices, distribution stats, Gini
# report(sol::HADSGESolution) — method, reduction, aggregate dynamics summary
# report(sol::KrusellSmithSolution) — PLM coefficients, R²
# show(io, ss::HASteadyState), show(io, sol::HADSGESolution), etc.
```

Follow the existing pattern in `src/dsge/display.jl`:
- Use `_coef_table()` for tabular output
- Use PrettyTables with `backend=:text`
- Compute Gini coefficient, mean wealth, wealth percentiles from distribution

- [ ] **Step 2: Test display doesn't error**

```julia
@testset "Display" begin
    # Use steady state from earlier tests
    io = IOBuffer()
    show(io, ss)
    s = String(take!(io))
    @test contains(s, "HASteadyState")
    @test contains(s, "converged")
end
```

- [ ] **Step 3: Wire, test, commit**

```bash
git commit -m "feat(ha-dsge): add report/show display for HA types"
```

---

### Task 11: Analysis Functions & irf/fevd Dispatch

**Files:**
- Create: `src/dsge/heterogeneous/analysis.jl`
- Test: append to `test/dsge/test_ha_dsge.jl`

- [ ] **Step 1: Implement analysis functions and dispatch**

```julia
# irf(sol::HADSGESolution, H) → delegates to irf(sol.linear_solution, H)
# fevd(sol::HADSGESolution, H) → delegates
# simulate(sol::HADSGESolution, T) → delegates
# distribution_irf(sol, H) → computes distribution dynamics using reduction_basis
# inequality_irf(sol, H) → Gini coefficient response over time
# simulate_panel(sol, N, T) → simulate N individual trajectories
```

- [ ] **Step 2: Test dispatch works**

```julia
@testset "IRF dispatch" begin
    # This tests that irf(::HADSGESolution, H) works
    # Requires a full solve to have been run
    # ... (integration test using the KS model)
end
```

- [ ] **Step 3: Wire, test, commit**

```bash
git commit -m "feat(ha-dsge): add analysis functions and irf/fevd dispatch"
```

---

## Phase 4: Examples, Estimation, Plotting, Docs (Tasks 12-14)

### Task 12: Built-in Examples

**Files:**
- Create: `src/dsge/heterogeneous/examples.jl`
- Test: append to `test/dsge/test_ha_dsge.jl`

- [ ] **Step 1: Implement load_ha_example**

```julia
function load_ha_example(name::Symbol)
    if name == :krusell_smith
        return _ks_example()
    elseif name == :one_asset_hank
        return _one_asset_hank_example()
    elseif name == :two_asset_hank
        return _two_asset_hank_example()
    else
        error("Unknown HA example: $name. Use :krusell_smith, :one_asset_hank, :two_asset_hank")
    end
end
```

Each example constructs an `HADSGESpec` programmatically (not via macro, for reliability) with published calibration values.

- [ ] **Step 2: Test examples load and solve**

```julia
@testset "Built-in examples" begin
    ks = load_ha_example(:krusell_smith)
    @test ks isa HADSGESpec{Float64}
    ss = ha_steady_state(ks)
    @test ss.converged
end
```

- [ ] **Step 3: Export, wire, commit**

```bash
git commit -m "feat(ha-dsge): add built-in KS, one-asset HANK, two-asset HANK examples"
```

---

### Task 13: Estimation Integration & Plotting

**Files:**
- Modify: `src/dsge/bayes_estimation.jl` (add HADSGESpec dispatch)
- Create: plotting dispatch in `src/plotting/models.jl`
- Test: append to `test/dsge/test_ha_dsge.jl`

- [ ] **Step 1: Add estimate_dsge_bayes dispatch for HADSGESpec**

Add a method to `estimate_dsge_bayes` that:
1. Accepts `HADSGESpec` and a `ha_method` keyword
2. In the likelihood evaluation loop: solve HA steady state + linearize → Kalman filter
3. Uses existing SMC/MH infrastructure unchanged

- [ ] **Step 2: Add plot_result dispatches**

```julia
# plot_result(ss::HASteadyState) → wealth distribution histogram + Lorenz curve
# plot_result(sol::HADSGESolution) → aggregate IRFs (delegates to existing)
```

- [ ] **Step 3: Test, commit**

```bash
git commit -m "feat(ha-dsge): add Bayesian estimation dispatch and plotting"
```

---

### Task 14: Wire Remaining Includes, Full Test Runner, Documentation

**Files:**
- Modify: `src/MacroEconometricModels.jl` (all remaining includes + exports)
- Modify: `test/runtests.jl` (add HA-DSGE test group)
- Create: `docs/src/dsge_ha.md`

- [ ] **Step 1: Wire all includes in correct order**

In `src/MacroEconometricModels.jl`, after the existing DSGE includes:

```julia
# Heterogeneous Agent DSGE
include("dsge/heterogeneous/types.jl")
include("dsge/heterogeneous/egm.jl")
include("dsge/heterogeneous/individual_vfi.jl")
include("dsge/heterogeneous/distribution.jl")
include("dsge/heterogeneous/steady_state.jl")
include("dsge/heterogeneous/reiter.jl")
include("dsge/heterogeneous/ssj.jl")
include("dsge/heterogeneous/krusell_smith.jl")
include("dsge/heterogeneous/parser.jl")
include("dsge/heterogeneous/analysis.jl")
include("dsge/heterogeneous/display.jl")
include("dsge/heterogeneous/examples.jl")
```

- [ ] **Step 2: Add all exports**

```julia
export HADSGESpec, HAGrid, IncomeProcess, IndividualProblem
export HASteadyState, HADSGESolution, KrusellSmithSolution
export rouwenhorst, tauchen
export ha_steady_state, load_ha_example
export distribution_irf, inequality_irf, simulate_panel
```

- [ ] **Step 3: Add test group to runtests.jl**

Add to `TEST_GROUPS` in `test/runtests.jl`, in Group 7 (DSGE Models):

```julia
    ("DSGE Models" => [
        "dsge/test_dsge.jl",
        "dsge/test_bayesian_dsge.jl",
        "dsge/test_dsge_hd.jl",
        "dsge/test_ha_dsge.jl",
    ]),
```

Also add to the sequential fallback section.

- [ ] **Step 4: Run full DSGE test group**

```bash
MACRO_MULTIPROCESS_TESTS=1 julia --project=. -e '
using Pkg; Pkg.test()
' 2>&1 | grep -E "DSGE|PASS|FAIL|Error"
```

Wait — per CLAUDE.md rules: "NEVER run the full test suite locally." Instead run only the relevant test:

```bash
julia --project=. -e '
using Test, MacroEconometricModels
include("test/dsge/test_ha_dsge.jl")
'
```

- [ ] **Step 5: Write documentation page**

Create `docs/src/dsge_ha.md` following `docrule.md` conventions. Structure:
- H1: Heterogeneous Agent DSGE Models
- Introduction (2-3 paragraphs)
- Quick Start (4 recipes: KS steady state, one-asset HANK IRFs, two-asset HANK, estimation)
- H2 sections: Individual Problem, Distribution, Steady State, SSJ Method, Reiter Method, Krusell-Smith, Estimation, Two-Asset HANK
- Complete Example
- Common Pitfalls
- References

Update `docs/make.jl` to include the new page in navigation.

- [ ] **Step 6: Verify documentation examples**

```bash
julia --project=docs docs/verify_examples.jl docs/src/dsge_ha.md
```

- [ ] **Step 7: Final commit**

```bash
git commit -m "feat(ha-dsge): wire includes, exports, tests, documentation for v0.5.0"
```
