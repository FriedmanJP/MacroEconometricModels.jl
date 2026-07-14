# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Type definitions for Heterogeneous Agent DSGE models — grids, income processes,
individual problems, specifications, and solutions.
"""

using LinearAlgebra

# =============================================================================
# Internal helpers
# =============================================================================

"""
    _make_asset_grid(a_min, a_max, n, grid_type) → Vector{T}

Construct a one-dimensional asset grid on `[a_min, a_max]` with `n` points.

Supported `grid_type`:
- `:double_exp` — double exponential (denser near `a_min`, default)
- `:log` — logarithmic spacing (shifted)
- `:linear` — uniform spacing
"""
function _make_asset_grid(a_min::T, a_max::T, n::Int, grid_type::Symbol) where {T<:AbstractFloat}
    @assert n >= 3 "Need at least 3 grid points"
    @assert a_max > a_min "Upper bound must exceed lower bound"

    if grid_type == :linear
        return collect(range(a_min, a_max; length=n))
    elseif grid_type == :log
        # Shifted log spacing: map [0,1] through log(1+x) then scale
        x = range(zero(T), one(T); length=n)
        raw = @. log(one(T) + x) / log(T(2))  # maps [0,1] → [0,1] with curvature
        return @. a_min + raw * (a_max - a_min)
    elseif grid_type == :double_exp
        # Double exponential: very dense near a_min, sparse at top
        x = range(zero(T), one(T); length=n)
        raw = @. (exp(exp(x) - one(T)) - one(T)) / (exp(exp(one(T)) - one(T)) - one(T))
        return @. a_min + raw * (a_max - a_min)
    else
        throw(ArgumentError("Unknown grid_type: $grid_type. Use :double_exp, :log, or :linear."))
    end
end

"""
    _stationary_distribution(P::Matrix{T}) → Vector{T}

Compute the stationary distribution of a Markov transition matrix `P` via power iteration.
Rows of `P` must sum to 1.
"""
function _stationary_distribution(P::Matrix{T}) where {T<:AbstractFloat}
    n = size(P, 1)
    @assert size(P, 2) == n "Transition matrix must be square"

    # Power iteration on P'
    pi_vec = fill(one(T) / n, n)
    P_t = Matrix{T}(P')
    for _ in 1:10_000
        pi_new = P_t * pi_vec
        pi_new ./= sum(pi_new)
        if maximum(abs.(pi_new .- pi_vec)) < T(1e-12)
            return pi_new
        end
        pi_vec = pi_new
    end
    # Normalize even if not fully converged
    pi_vec ./= sum(pi_vec)
    return pi_vec
end

# =============================================================================
# HAGrid — Multi-dimensional individual state space grid
# =============================================================================

"""
    HAGrid{T}

Multi-dimensional grid over individual state variables (assets and income).

Fields:
- `grids::Vector{Vector{T}}` — grid vectors per asset dimension
- `n_points::Vector{Int}` — number of grid points per asset dimension
- `n_dims::Int` — number of asset dimensions (1 or 2)
- `n_income::Int` — number of income states
- `bounds::Vector{Tuple{T,T}}` — `(min, max)` per asset dimension
- `labels::Vector{Symbol}` — dimension labels (e.g., `[:assets]` or `[:liquid, :illiquid]`)
- `total_individual_states::Int` — product of all asset grid sizes and income states
"""
struct HAGrid{T<:AbstractFloat}
    grids::Vector{Vector{T}}
    n_points::Vector{Int}
    n_dims::Int
    n_income::Int
    bounds::Vector{Tuple{T,T}}
    labels::Vector{Symbol}
    total_individual_states::Int

    function HAGrid{T}(grids, n_points, n_dims, n_income, bounds, labels) where {T<:AbstractFloat}
        @assert n_dims == length(grids) "n_dims must match number of grid vectors"
        @assert n_dims == length(n_points) "n_dims must match length of n_points"
        @assert n_dims == length(bounds) "n_dims must match length of bounds"
        @assert n_dims == length(labels) "n_dims must match length of labels"
        @assert n_dims in (1, 2) "Only 1 or 2 asset dimensions supported"
        @assert n_income >= 1 "Need at least 1 income state"
        total = prod(n_points) * n_income
        new{T}(grids, n_points, n_dims, n_income, bounds, labels, total)
    end
end

"""
    HAGrid(; assets=(0.0, 200.0, 500), income_states=7, grid_type=:double_exp)
    HAGrid(; liquid=(0.0, 50.0, 200), illiquid=(0.0, 200.0, 200), income_states=7, grid_type=:double_exp)

Construct a one-asset or two-asset grid.

For one-asset models, pass `assets=(a_min, a_max, n_points)`.
For two-asset models (e.g., HANK), pass `liquid=...` and `illiquid=...`.

# Arguments
- `assets::Union{Nothing,Tuple{Real,Real,Int}}` — one-asset grid spec (mutually exclusive with liquid/illiquid)
- `liquid::Union{Nothing,Tuple{Real,Real,Int}}` — liquid asset grid spec
- `illiquid::Union{Nothing,Tuple{Real,Real,Int}}` — illiquid asset grid spec
- `income_states::Int` — number of income states (default 7)
- `grid_type::Symbol` — `:double_exp` (default), `:log`, or `:linear`
"""
function HAGrid(; assets::Union{Nothing,Tuple{Real,Real,Int}}=nothing,
                  liquid::Union{Nothing,Tuple{Real,Real,Int}}=nothing,
                  illiquid::Union{Nothing,Tuple{Real,Real,Int}}=nothing,
                  income_states::Int=7,
                  grid_type::Symbol=:double_exp)
    T = Float64

    two_asset = !isnothing(liquid) || !isnothing(illiquid)

    if two_asset
        # Two-asset mode
        isnothing(liquid) && throw(ArgumentError("Two-asset grid requires both `liquid` and `illiquid`"))
        isnothing(illiquid) && throw(ArgumentError("Two-asset grid requires both `liquid` and `illiquid`"))
        !isnothing(assets) && throw(ArgumentError("Cannot specify `assets` together with `liquid`/`illiquid`"))
        b_min, b_max, n_b = T(liquid[1]), T(liquid[2]), liquid[3]
        a_min, a_max, n_a = T(illiquid[1]), T(illiquid[2]), illiquid[3]
        g_b = _make_asset_grid(b_min, b_max, n_b, grid_type)
        g_a = _make_asset_grid(a_min, a_max, n_a, grid_type)
        return HAGrid{T}([g_b, g_a], [n_b, n_a], 2, income_states,
                         [(b_min, b_max), (a_min, a_max)], [:liquid, :illiquid])
    else
        # One-asset mode (default)
        asset_spec = isnothing(assets) ? (0.0, 200.0, 500) : assets
        a_min, a_max, n_a = T(asset_spec[1]), T(asset_spec[2]), asset_spec[3]
        g = _make_asset_grid(a_min, a_max, n_a, grid_type)
        return HAGrid{T}([g], [n_a], 1, income_states, [(a_min, a_max)], [:assets])
    end
end

# =============================================================================
# IncomeProcess — Idiosyncratic Markov chain
# =============================================================================

"""
    IncomeProcess{T}

Discretized idiosyncratic income Markov chain.

Fields:
- `transition::Matrix{T}` — `n × n` transition matrix (rows sum to 1)
- `states::Vector{T}` — `n` income state values
- `stationary_dist::Vector{T}` — stationary distribution
- `labels::Symbol` — label for the process (default `:income`)
"""
struct IncomeProcess{T<:AbstractFloat}
    transition::Matrix{T}
    states::Vector{T}
    stationary_dist::Vector{T}
    labels::Symbol

    function IncomeProcess{T}(transition, states, stationary_dist, labels) where {T<:AbstractFloat}
        n = length(states)
        @assert size(transition) == (n, n) "Transition matrix must be n×n"
        @assert length(stationary_dist) == n "Stationary distribution must have n elements"
        new{T}(transition, states, stationary_dist, labels)
    end
end

# =============================================================================
# rouwenhorst — Rouwenhorst (1995) AR(1) discretization
# =============================================================================

"""
    rouwenhorst(rho, sigma, n) → IncomeProcess{Float64}

Discretize an AR(1) process `y_t = ρ y_{t-1} + σ ε_t` using the Rouwenhorst (1995) method.

More accurate than Tauchen for highly persistent processes (ρ close to 1).

# Arguments
- `rho::Real` — persistence parameter (|ρ| < 1)
- `sigma::Real` — shock standard deviation (σ > 0)
- `n::Int` — number of discrete states (n ≥ 2)

# References
- Rouwenhorst, K. G. (1995). Asset pricing implications of equilibrium business cycle models.
  In *Frontiers of Business Cycle Research* (pp. 294–330). Princeton University Press.
- Kopecky, K. A., & Suen, R. M. H. (2010). Finite state Markov-chain approximations to
  highly persistent processes. *Review of Economic Dynamics*, 13(3), 701–714.
"""
function rouwenhorst(rho::Real, sigma::Real, n::Int)
    T = Float64
    rho_T = T(rho)
    sigma_T = T(sigma)

    @assert abs(rho_T) < one(T) "Persistence must satisfy |rho| < 1"
    @assert sigma_T > zero(T) "Shock std dev must be positive"
    @assert n >= 2 "Need at least 2 states"

    # Unconditional std dev of the AR(1) process
    sigma_y = sigma_T / sqrt(one(T) - rho_T^2)

    # State space: equally spaced on [-ψ, ψ] where ψ = √(n-1) × σ_y
    psi = sqrt(T(n - 1)) * sigma_y
    states = collect(range(-psi, psi; length=n))

    # Build transition matrix recursively
    p = (one(T) + rho_T) / T(2)
    q = p

    if n == 2
        P = [p (one(T)-p); (one(T)-q) q]
    else
        # Start with n=2
        P_prev = [p (one(T)-p); (one(T)-q) q]

        for m in 3:n
            z = zeros(T, m - 1)
            P_new = zeros(T, m, m)

            # Four-corner recursion
            P_new[1:m-1, 1:m-1] .+= p .* P_prev
            P_new[1:m-1, 2:m]   .+= (one(T) - p) .* P_prev
            P_new[2:m,   1:m-1] .+= (one(T) - q) .* P_prev
            P_new[2:m,   2:m]   .+= q .* P_prev

            # Normalize interior rows (divide by 2, since each interior row
            # gets contributions from two recursion terms)
            for i in 2:m-1
                P_new[i, :] ./= T(2)
            end

            P_prev = P_new
        end
        P = P_prev
    end

    # Ensure rows sum exactly to 1
    for i in 1:n
        P[i, :] ./= sum(P[i, :])
    end

    pi_stat = _stationary_distribution(P)

    return IncomeProcess{T}(P, states, pi_stat, :income)
end

# =============================================================================
# tauchen — Tauchen (1986) AR(1) discretization
# =============================================================================

"""
    tauchen(rho, sigma, n; m=3) → IncomeProcess{Float64}

Discretize an AR(1) process `y_t = ρ y_{t-1} + σ ε_t` using the Tauchen (1986) method.

# Arguments
- `rho::Real` — persistence parameter (|ρ| < 1)
- `sigma::Real` — shock standard deviation (σ > 0)
- `n::Int` — number of discrete states (n ≥ 2)
- `m::Real` — state space covers ±m unconditional standard deviations (default 3)

# References
- Tauchen, G. (1986). Finite state Markov-chain approximations to univariate and
  vector autoregressions. *Economics Letters*, 20(2), 177–181.
"""
function tauchen(rho::Real, sigma::Real, n::Int; m::Real=3)
    T = Float64
    rho_T = T(rho)
    sigma_T = T(sigma)
    m_T = T(m)

    @assert abs(rho_T) < one(T) "Persistence must satisfy |rho| < 1"
    @assert sigma_T > zero(T) "Shock std dev must be positive"
    @assert n >= 2 "Need at least 2 states"
    @assert m_T > zero(T) "Coverage parameter m must be positive"

    # Unconditional std dev
    sigma_y = sigma_T / sqrt(one(T) - rho_T^2)

    # State space
    y_max = m_T * sigma_y
    states = collect(range(-y_max, y_max; length=n))
    d = states[2] - states[1]  # step size

    # Standard normal CDF
    normal_dist = Distributions.Normal(zero(T), one(T))

    # Fill transition matrix
    P = zeros(T, n, n)
    for i in 1:n
        for j in 1:n
            if j == 1
                P[i, j] = Distributions.cdf(normal_dist,
                    (states[1] + d / T(2) - rho_T * states[i]) / sigma_T)
            elseif j == n
                P[i, j] = one(T) - Distributions.cdf(normal_dist,
                    (states[n] - d / T(2) - rho_T * states[i]) / sigma_T)
            else
                P[i, j] = Distributions.cdf(normal_dist,
                    (states[j] + d / T(2) - rho_T * states[i]) / sigma_T) -
                           Distributions.cdf(normal_dist,
                    (states[j] - d / T(2) - rho_T * states[i]) / sigma_T)
            end
        end
    end

    # Ensure rows sum exactly to 1
    for i in 1:n
        P[i, :] ./= sum(P[i, :])
    end

    pi_stat = _stationary_distribution(P)

    return IncomeProcess{T}(P, states, pi_stat, :income)
end

# =============================================================================
# IndividualProblem — Household optimization specification
# =============================================================================

# Parameterized on the concrete function-field types (#254 G-14): the EGM/VFI inner loops
# call ip.utility/utility_prime/… on a concretely-typed `ip` argument, so specialization
# removes the dynamic dispatch that abstract ::Function fields forced. Leading {T} is kept —
# IndividualProblem{T}(...) infers FU…FA via the inner ctor, and ::IndividualProblem{T}
# dispatch (used throughout egm/vfi/ssj/reiter/krusell_smith/steady_state) still matches by
# partial parameterization, so no call site changes.
"""
    IndividualProblem{T}

Specification of the individual household optimization problem.

Fields:
- `utility::Function` — `u(c)` utility function
- `utility_prime::Function` — `u'(c)` marginal utility
- `utility_prime_inv::Function` — `(u')⁻¹(v)` inverse marginal utility
- `beta::T` — discount factor
- `budget_fn::Function` — `budget(a, z, prices...)` → available resources
- `borrowing_constraint::Vector{T}` — lower bound per asset dimension
- `adjustment_cost::Union{Nothing,Function}` — optional `χ(d)` portfolio adjustment cost (two-asset)
- `n_asset_dims::Int` — number of asset dimensions (1 or 2)
"""
struct IndividualProblem{T<:AbstractFloat, FU, FUP, FUPI, FB, FA}
    utility::FU
    utility_prime::FUP
    utility_prime_inv::FUPI
    beta::T
    budget_fn::FB
    borrowing_constraint::Vector{T}
    adjustment_cost::FA   # Nothing (one-asset) or a concrete χ(d) function type (two-asset)
    n_asset_dims::Int

    function IndividualProblem{T}(utility, utility_prime, utility_prime_inv, beta,
                                  budget_fn, borrowing_constraint, adjustment_cost,
                                  n_asset_dims) where {T<:AbstractFloat}
        @assert zero(T) < beta < one(T) "Discount factor must be in (0, 1)"
        @assert n_asset_dims in (1, 2) "Only 1 or 2 asset dimensions supported"
        @assert length(borrowing_constraint) == n_asset_dims "Borrowing constraint length must match n_asset_dims"
        new{T, typeof(utility), typeof(utility_prime), typeof(utility_prime_inv),
            typeof(budget_fn), typeof(adjustment_cost)}(
            utility, utility_prime, utility_prime_inv, beta,
            budget_fn, borrowing_constraint, adjustment_cost, n_asset_dims)
    end
end

# =============================================================================
# HADSGESpec — HA-DSGE specification
# =============================================================================

"""
    HADSGESpec{T}

Heterogeneous Agent DSGE model specification. Wraps a representative-agent
`DSGESpec{T}` with individual-level components.

Fields:
- `aggregate_spec::DSGESpec{T}` — aggregate block (equations, params, steady state)
- `individual::IndividualProblem{T}` — household problem
- `income::IncomeProcess{T}` — idiosyncratic income process
- `grid::HAGrid{T}` — individual state space grid
- `aggregation::Vector{Pair{Symbol,Function}}` — maps distribution → aggregate variables
- `het_params::Dict{Symbol,T}` — heterogeneous-agent-specific parameters
- `n_assets::Int` — number of asset dimensions
- `n_income::Int` — number of income states
- `model::Symbol` — model family for clearing/dynamics dispatch (`:aiyagari` default,
  `:huggett` for zero-net-supply pure exchange)
"""
struct HADSGESpec{T<:AbstractFloat}
    aggregate_spec::DSGESpec{T}
    individual::IndividualProblem{T}
    income::IncomeProcess{T}
    grid::HAGrid{T}
    aggregation::Vector{Pair{Symbol,Function}}
    het_params::Dict{Symbol,T}
    n_assets::Int
    n_income::Int
    model::Symbol

    function HADSGESpec{T}(aggregate_spec, individual, income, grid,
                            aggregation, het_params;
                            model::Symbol=:aiyagari) where {T<:AbstractFloat}
        n_assets = grid.n_dims
        n_income = grid.n_income
        @assert individual.n_asset_dims == n_assets "Individual problem asset dims must match grid"
        @assert length(income.states) == n_income "Income states must match grid n_income"
        new{T}(aggregate_spec, individual, income, grid,
               aggregation, het_params, n_assets, n_income, model)
    end
end

# =============================================================================
# HASteadyState — Stationary equilibrium
# =============================================================================

"""
    HASteadyState{T}

Stationary equilibrium of a heterogeneous agent model.

Fields:
- `policies::Dict{Symbol,Array{T}}` — policy functions (e.g., `:savings`, `:consumption`)
- `distribution::Array{T}` — stationary distribution over individual states
- `value_fn::Array{T}` — value function over individual states
- `prices::Dict{Symbol,T}` — equilibrium prices (e.g., `:r`, `:w`)
- `aggregates::Dict{Symbol,T}` — aggregate quantities (e.g., `:K`, `:L`)
- `grid::HAGrid{T}` — grid used for computation
- `income::IncomeProcess{T}` — income process used
- `converged::Bool` — whether the equilibrium computation converged
- `iterations::Int` — number of iterations used
- `euler_error::T` — maximum Euler equation error (log10 units)
- `excess_demand::T` — market clearing residual
"""
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
# HADSGESolution — Linearized HA-DSGE solution
# =============================================================================

"""
    HADSGESolution{T}

Linearized solution of a heterogeneous agent DSGE model, combining dimensionality
reduction of the distribution with a standard linear RE solution.

Fields:
- `steady_state::HASteadyState{T}` — stationary equilibrium
- `linear_solution::DSGESolution{T}` — RE solution of the reduced system
- `method::Symbol` — solution method (e.g., `:reiter`, `:boppart_krusell_mitman`)
- `spec::HADSGESpec{T}` — model specification
- `reduction_basis::Matrix{T}` — basis for distribution reduction (e.g., from SVD)
- `n_full_states::Int` — full state dimension before reduction
- `n_reduced::Int` — reduced state dimension
- `explained_variance::T` — fraction of variance captured by reduction
- `jacobians::Union{Nothing,Dict{Symbol,Matrix{T}}}` — optional Jacobian matrices
- `C_obs::Matrix{T}` — reduced-state → aggregate-output map (Ho-Kalman `C`; identity for Reiter)
- `D_obs::Matrix{T}` — direct shock feed-through to aggregate outputs (`D = h[0]`)
"""
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
    C_obs::Matrix{T}
    D_obs::Matrix{T}
end

# Accessors — delegate to linear_solution
nvars(sol::HADSGESolution) = nvars(sol.linear_solution)
nshocks(sol::HADSGESolution) = nshocks(sol.linear_solution)
is_determined(sol::HADSGESolution) = is_determined(sol.linear_solution)
is_stable(sol::HADSGESolution) = is_stable(sol.linear_solution)

# =============================================================================
# KrusellSmithSolution — Simulation-based HA-DSGE solution
# =============================================================================

"""
    KrusellSmithSolution{T}

Simulation-based solution using the Krusell-Smith (1998) algorithm. Approximates
the distribution via a perceived law of motion (PLM) for aggregate state variables.

Fields:
- `steady_state::HASteadyState{T}` — stationary equilibrium
- `plm_coefficients::Dict{Symbol,Vector{T}}` — PLM regression coefficients per aggregate
- `r_squared::Dict{Symbol,T}` — PLM R² values (accuracy measure)
- `spec::HADSGESpec{T}` — model specification
- `converged::Bool` — whether PLM iteration converged
- `iterations::Int` — number of KS outer loop iterations
"""
struct KrusellSmithSolution{T<:AbstractFloat}
    steady_state::HASteadyState{T}
    plm_coefficients::Dict{Symbol,Vector{T}}
    r_squared::Dict{Symbol,T}
    spec::HADSGESpec{T}
    converged::Bool
    iterations::Int
end
