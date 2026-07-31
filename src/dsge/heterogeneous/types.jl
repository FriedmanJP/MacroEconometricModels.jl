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
    _make_asset_grid(a_min, a_max, n, grid_type; pivot=0.25) → Vector{T}

Construct a one-dimensional asset grid on `[a_min, a_max]` with `n` points.

Supported `grid_type`:
- `:double_exp` — double exponential (denser near `a_min`, default)
- `:geometric` — pivot-geometric: equidistant in `log(a - a_min + pivot)`, so the
  spacing at the bottom grows only *logarithmically* in `a_max`. This matches
  `agrid` in the Python `sequence-jacobian` toolkit. Prefer it when `a_max` must
  be large enough to keep the stationary distribution off the ceiling: the
  `:double_exp` shape is a fixed curve rescaled by `(a_max - a_min)`, so its
  bottom spacing is *linear* in `a_max` and resolution at the borrowing
  constraint degrades as the ceiling is raised.
- `:log` — logarithmic spacing (shifted)
- `:linear` — uniform spacing

`pivot` (default `0.25`) shifts the geometric grid away from the origin; the
effective pivot is `abs(a_min) + pivot`, so a grid starting at a negative
borrowing limit is still dense at the constraint.
"""
function _make_asset_grid(a_min::T, a_max::T, n::Int, grid_type::Symbol;
                          pivot::Real=T(0.25)) where {T<:AbstractFloat}
    @assert n >= 3 "Need at least 3 grid points"
    @assert a_max > a_min "Upper bound must exceed lower bound"

    if grid_type == :geometric
        piv = abs(a_min) + T(pivot)
        @assert piv > zero(T) "pivot must be positive"
        # Geometric in (a + piv): lo = a_min + piv > 0 for any a_min.
        g = exp.(range(log(a_min + piv), log(a_max + piv); length=n)) .- piv
        g[1] = a_min      # exact endpoints (guard against roundoff)
        g[end] = a_max
        return g
    elseif grid_type == :linear
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
        throw(ArgumentError("Unknown grid_type: $grid_type. Use :double_exp, :geometric, :log, or :linear."))
    end
end

"""
    _ar1_unconditional_sd(sigma, rho, sigma_is, caller) → T

Resolve the `sigma_is` convention for an AR(1) discretizer: return the
unconditional standard deviation `sd(y_t)` of `y_t = ρ y_{t-1} + σ ε_t`.

`sigma_is === :innovation` treats `sigma` as `sd(ε_t)` (so `sd(y) = σ/√(1-ρ²)`);
`sigma_is === :unconditional` treats it as `sd(y_t)` itself.
"""
function _ar1_unconditional_sd(sigma::T, rho::T, sigma_is::Symbol, caller::Symbol) where {T<:AbstractFloat}
    if sigma_is === :innovation
        return sigma / sqrt(one(T) - rho^2)
    elseif sigma_is === :unconditional
        return sigma
    else
        throw(ArgumentError("$caller: sigma_is must be :innovation (sigma is the " *
            "standard deviation of the AR(1) innovation eps_t — the default) or " *
            ":unconditional (sigma is sd(y_t) itself, the Python sequence-jacobian " *
            "convention); got :$sigma_is"))
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
- `grid_type::Symbol` — `:double_exp` (default), `:geometric`, `:log`, or `:linear`.
  Use `:geometric` when `a_max` has to be large: unlike `:double_exp`, its spacing
  near `a_min` grows only logarithmically in `a_max`, so raising the ceiling does
  not cost resolution at the borrowing constraint.
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
    rouwenhorst(rho, sigma, n; sigma_is=:innovation) → IncomeProcess{Float64}

Discretize an AR(1) process `y_t = ρ y_{t-1} + σ ε_t` using the Rouwenhorst (1995) method.

More accurate than Tauchen for highly persistent processes (ρ close to 1).

# Arguments
- `rho::Real` — persistence parameter (|ρ| < 1)
- `sigma::Real` — standard deviation (σ > 0); see **Convention** below
- `n::Int` — number of discrete states (n ≥ 2)
- `sigma_is::Symbol` — `:innovation` (default) or `:unconditional`

# Convention

By default `sigma` is the standard deviation of the **innovation** `ε_t`, so the
process itself has `sd(y_t) = σ / √(1 - ρ²)` and the state space spans
`±√(n-1) · sd(y_t)`. Pass `sigma_is=:unconditional` to supply `sd(y_t)` directly.

The two differ sharply at high persistence: at `ρ = 0.966` the unconditional
standard deviation is **3.87×** the innovation standard deviation. `markov_rouwenhorst`
in the Python `sequence-jacobian` toolkit parameterizes by the *unconditional*
standard deviation, so a `sigma` copied from there must be passed with
`sigma_is=:unconditional` (or divided by `√(1 - ρ²)` first).

# References
- Rouwenhorst, K. G. (1995). Asset pricing implications of equilibrium business cycle models.
  In *Frontiers of Business Cycle Research* (pp. 294–330). Princeton University Press.
- Kopecky, K. A., & Suen, R. M. H. (2010). Finite state Markov-chain approximations to
  highly persistent processes. *Review of Economic Dynamics*, 13(3), 701–714.
"""
function rouwenhorst(rho::Real, sigma::Real, n::Int; sigma_is::Symbol=:innovation)
    T = Float64
    rho_T = T(rho)
    sigma_T = T(sigma)

    @assert abs(rho_T) < one(T) "Persistence must satisfy |rho| < 1"
    @assert sigma_T > zero(T) "Shock std dev must be positive"
    @assert n >= 2 "Need at least 2 states"

    # Unconditional std dev of the AR(1) process
    sigma_y = _ar1_unconditional_sd(sigma_T, rho_T, sigma_is, :rouwenhorst)

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
    tauchen(rho, sigma, n; m=3, sigma_is=:innovation) → IncomeProcess{Float64}

Discretize an AR(1) process `y_t = ρ y_{t-1} + σ ε_t` using the Tauchen (1986) method.

# Arguments
- `rho::Real` — persistence parameter (|ρ| < 1)
- `sigma::Real` — standard deviation (σ > 0); see **Convention** below
- `n::Int` — number of discrete states (n ≥ 2)
- `m::Real` — state space covers ±m unconditional standard deviations (default 3)
- `sigma_is::Symbol` — `:innovation` (default) or `:unconditional`

# Convention

Identical to [`rouwenhorst`](@ref): `sigma` is the standard deviation of the
**innovation** `ε_t` by default, giving `sd(y_t) = σ / √(1 - ρ²)`. Pass
`sigma_is=:unconditional` to supply `sd(y_t)` directly (the Python
`sequence-jacobian` convention). At `ρ = 0.966` the two differ by **3.87×**.

# References
- Tauchen, G. (1986). Finite state Markov-chain approximations to univariate and
  vector autoregressions. *Economics Letters*, 20(2), 177–181.
"""
function tauchen(rho::Real, sigma::Real, n::Int; m::Real=3, sigma_is::Symbol=:innovation)
    T = Float64
    rho_T = T(rho)
    sigma_T = T(sigma)
    m_T = T(m)

    @assert abs(rho_T) < one(T) "Persistence must satisfy |rho| < 1"
    @assert sigma_T > zero(T) "Shock std dev must be positive"
    @assert n >= 2 "Need at least 2 states"
    @assert m_T > zero(T) "Coverage parameter m must be positive"

    # Unconditional std dev
    sigma_y = _ar1_unconditional_sd(sigma_T, rho_T, sigma_is, :tauchen)

    # Conditional (innovation) std dev — this is what scales the transition CDF.
    # Branch rather than round-tripping through sigma_y so the :innovation path
    # stays bitwise identical to the pre-`sigma_is` implementation.
    sigma_eps = sigma_is === :innovation ? sigma_T : sigma_y * sqrt(one(T) - rho_T^2)

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
                    (states[1] + d / T(2) - rho_T * states[i]) / sigma_eps)
            elseif j == n
                P[i, j] = one(T) - Distributions.cdf(normal_dist,
                    (states[n] - d / T(2) - rho_T * states[i]) / sigma_eps)
            else
                P[i, j] = Distributions.cdf(normal_dist,
                    (states[j] + d / T(2) - rho_T * states[i]) / sigma_eps) -
                           Distributions.cdf(normal_dist,
                    (states[j] - d / T(2) - rho_T * states[i]) / sigma_eps)
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
# LaborSupply — endogenous labor-supply specification
# =============================================================================

"""
    LaborSupply{T}

Endogenous labor supply for a heterogeneous-agent household problem. Attach one
to an [`IndividualProblem`](@ref) via its `labor` keyword; the default (`nothing`)
leaves labor exogenous and every solver path bitwise unchanged.

Disutility of hours is isoelastic, `v(n) = ψ n^{1 + 1/φ} / (1 + 1/φ)`, so
`v'(n) = ψ n^{1/φ}`.

Fields:
- `kind::Symbol` — `:ghh` or `:separable`
- `psi::T` — disutility scale `ψ > 0`
- `frisch::T` — Frisch elasticity `φ > 0`
- `n_max::T` — upper bound on hours (numerical guard, default `Inf`)

# Preference specifications

**GHH** (Greenwood–Hercowitz–Huffman 1988), `U(c - ψ n^{1+1/φ}/(1+1/φ))`. The
intratemporal condition is

```math
ψ n^{1/φ} = w e \\quad\\Longrightarrow\\quad n(e) = (w e / ψ)^φ
```

independent of consumption and assets — there is **no wealth effect on hours**.
Substituting it out leaves a standard one-dimensional consumption-savings problem
in the composite good `x = c - ψ n^{1+1/φ}/(1+1/φ)`, with net labor income
`ỹ(e) = w e n(e) - ψ n(e)^{1+1/φ}/(1+1/φ)`. This is the tractable default.

**Separable**, `u(c) - v(n)`. The intratemporal condition

```math
ψ n^{1/φ} = w e \\, u'(c)
```

couples hours to consumption, so hours carry a wealth effect. On the
unconstrained EGM branch consumption is known before hours, so `n` is still
explicit; on the constrained branch `c` and `n` must be solved jointly, which
[`_egm_solve`](@ref) does with a bracketed scalar root-find.

See also [`IndividualProblem`](@ref), [`labor_supply`](@ref).
"""
struct LaborSupply{T<:AbstractFloat}
    kind::Symbol
    psi::T
    frisch::T
    n_max::T

    function LaborSupply{T}(kind::Symbol, psi, frisch, n_max) where {T<:AbstractFloat}
        kind in (:ghh, :separable) || throw(ArgumentError(
            "LaborSupply: kind must be :ghh or :separable, got :$kind"))
        psi > 0 || throw(ArgumentError("LaborSupply: psi must be positive, got $psi"))
        frisch > 0 || throw(ArgumentError("LaborSupply: frisch must be positive, got $frisch"))
        n_max > 0 || throw(ArgumentError("LaborSupply: n_max must be positive, got $n_max"))
        new{T}(kind, T(psi), T(frisch), T(n_max))
    end
end

"""
    LaborSupply(; kind=:ghh, psi=1.0, frisch=0.5, n_max=Inf) → LaborSupply{Float64}

Keyword constructor for [`LaborSupply`](@ref). `frisch = 0.5` is a common
macro calibration; `psi` is usually chosen so that mean hours ≈ 1 at the
steady-state wage.
"""
LaborSupply(; kind::Symbol=:ghh, psi::Real=1.0, frisch::Real=0.5, n_max::Real=Inf) =
    LaborSupply{Float64}(kind, psi, frisch, n_max)

"""
    labor_supply(ls::LaborSupply, w_e) → n
    labor_supply(ls::LaborSupply, w_e, u_prime_c) → n

Hours implied by the intratemporal first-order condition at effective wage
`w_e = w·e`. The two-argument form is the GHH condition `ψ n^{1/φ} = w e`; the
three-argument form is the separable condition `ψ n^{1/φ} = w e u'(c)`, and
reduces to the first when `u_prime_c = 1`.

Hours are clamped to `[0, ls.n_max]`.
"""
function labor_supply(ls::LaborSupply{T}, w_e::Real, u_prime_c::Real=one(T)) where {T<:AbstractFloat}
    mrs = T(w_e) * T(u_prime_c)
    mrs <= zero(T) && return zero(T)
    return min((mrs / ls.psi)^ls.frisch, ls.n_max)
end

"""
    _labor_disutility(ls, n) → ψ n^{1+1/φ} / (1 + 1/φ)

Flow disutility of hours. Under GHH this is subtracted from consumption inside
the utility function; under separable preferences it is subtracted from utility.
"""
_labor_disutility(ls::LaborSupply{T}, n::T) where {T<:AbstractFloat} =
    n <= zero(T) ? zero(T) : ls.psi * n^(one(T) + one(T) / ls.frisch) / (one(T) + one(T) / ls.frisch)

"""
    _ghh_net_income(ls, w_e) → (ỹ, n, disutility)

GHH labor income net of the disutility term, `ỹ = w e n - ψ n^{1+1/φ}/(1+1/φ)`,
together with the hours and the disutility that produced it. Because GHH hours
do not depend on consumption or assets, this is a function of the effective wage
alone and can be substituted into the budget before the EGM step.
"""
function _ghh_net_income(ls::LaborSupply{T}, w_e::T) where {T<:AbstractFloat}
    n = labor_supply(ls, w_e)
    d = _labor_disutility(ls, n)
    return (w_e * n - d, n, d)
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
- `labor::Union{Nothing,LaborSupply{T}}` — optional endogenous labor supply
  (default `nothing`, i.e. exogenous labor). Pass via the `labor` keyword.

# Endogenous labor

With `labor = nothing` the household chooses only consumption and savings, and
`budget_fn(a, e, prices)` is the whole budget. With a [`LaborSupply`](@ref)
attached, `budget_fn` is interpreted as the budget **evaluated at `n = 1`**: the
solver reads the gross return and any non-labor offset (e.g. `div`) from it, then
replaces the `w·e` term with the labor income the intratemporal condition
implies. So the same `budget_fn` serves both cases.
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
    labor::Union{Nothing,LaborSupply{T}}

    # `labor` is a KEYWORD with a `nothing` default, so all pre-existing
    # eight-positional-argument call sites keep working unchanged.
    function IndividualProblem{T}(utility, utility_prime, utility_prime_inv, beta,
                                  budget_fn, borrowing_constraint, adjustment_cost,
                                  n_asset_dims;
                                  labor::Union{Nothing,LaborSupply{T}}=nothing) where {T<:AbstractFloat}
        @assert zero(T) < beta < one(T) "Discount factor must be in (0, 1)"
        @assert n_asset_dims in (1, 2) "Only 1 or 2 asset dimensions supported"
        @assert length(borrowing_constraint) == n_asset_dims "Borrowing constraint length must match n_asset_dims"
        labor === nothing || n_asset_dims == 1 || throw(ArgumentError(
            "IndividualProblem: endogenous labor supply is implemented for one-asset " *
            "problems only (got n_asset_dims = $n_asset_dims)."))
        new{T, typeof(utility), typeof(utility_prime), typeof(utility_prime_inv),
            typeof(budget_fn), typeof(adjustment_cost)}(
            utility, utility_prime, utility_prime_inv, beta,
            budget_fn, borrowing_constraint, adjustment_cost, n_asset_dims, labor)
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
- `distribution::Symbol` — distribution representation: `:young` (default, the
  Young 2010 histogram) or `:winberry` (Winberry 2018 parametric moment family;
  see [`WinberryFamily`](@ref))
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
    distribution::Symbol

    function HADSGESpec{T}(aggregate_spec, individual, income, grid,
                            aggregation, het_params;
                            model::Symbol=:aiyagari,
                            distribution::Symbol=:young) where {T<:AbstractFloat}
        distribution in (:young, :winberry) || throw(ArgumentError(
            "HADSGESpec: distribution must be :young or :winberry, got :$distribution."))
        n_assets = grid.n_dims
        n_income = grid.n_income
        @assert individual.n_asset_dims == n_assets "Individual problem asset dims must match grid"
        @assert length(income.states) == n_income "Income states must match grid n_income"

        # The borrowing constraint must coincide with the grid floor. The Young
        # (2010) transition clamps the savings policy into the grid, so a
        # constraint BELOW the floor silently creates assets out of nothing every
        # period — the model would violate its own budget constraint. A
        # constraint above the floor is merely wasteful (unreachable nodes).
        for d in 1:grid.n_dims
            lo = grid.bounds[d][1]
            bc = individual.borrowing_constraint[d]
            scale = max(one(T), abs(lo))
            if bc < lo - sqrt(eps(T)) * scale
                throw(ArgumentError(
                    "HADSGESpec: borrowing_constraint[$d] = $bc lies below the grid " *
                    "lower bound $lo on dimension :$(grid.labels[d]). The Young (2010) " *
                    "transition clamps the savings policy up to the grid floor, so such " *
                    "a model silently creates assets out of nothing every period. Set " *
                    "the grid lower bound equal to the borrowing constraint."))
            elseif bc > lo + sqrt(eps(T)) * scale
                @warn "HADSGESpec: borrowing_constraint[$d] = $bc lies above the grid " *
                      "lower bound $lo on dimension :$(grid.labels[d]); grid nodes below " *
                      "the constraint are unreachable (wasted resolution)." maxlog = 1
            end
        end

        new{T}(aggregate_spec, individual, income, grid,
               aggregation, het_params, n_assets, n_income, model, distribution)
    end
end

# =============================================================================
# ParametricDensity / WinberryFamily — Winberry (2018) moment representation
# =============================================================================

"""
    ParametricDensity{T}

Exponential-family approximation of the asset density within a single income
state (Winberry 2018).  On the standardized variable
`z = (a − center) / scale`,

    g(a) = exp( Σ_i λ_i (z^i − μ_i) − log_norm ),   μ_i = moments[i] / scale^i,

which integrates to one over the reference interval and reproduces `moments`
exactly at a converged fit.

Fields:
- `lambda::Vector{T}` — exponential-family coefficients (length `n_moments`)
- `moments::Vector{T}` — target centered moments `(mean, variance, m_3, …)`
- `center::T` / `scale::T` — standardization, `moments[1]` and `sqrt(moments[2])`
- `log_norm::T` — log normalizer in asset units
- `converged::Bool` — whether the Newton solve for `λ` met its tolerance
- `iterations::Int` — Newton iterations used
- `residual::T` — largest standardized moment residual `max_i |E_g[z^i] − μ_i|`

See [`fit_parametric_density`](@ref), [`parametric_density`](@ref),
[`parametric_moments`](@ref).
"""
struct ParametricDensity{T<:AbstractFloat}
    lambda::Vector{T}
    moments::Vector{T}
    center::T
    scale::T
    log_norm::T
    converged::Bool
    iterations::Int
    residual::T
end

"""
    WinberryFamily{T}

Winberry (2018) parametric representation of a full cross-sectional
distribution: one [`ParametricDensity`](@ref) per income state, plus the
income-state masses.

The distribution state is the `n_income × n_moments` moment matrix rather than
the `n_a × n_income` histogram, which is what shrinks the linearized system.

Fields:
- `densities::Vector{ParametricDensity{T}}` — one density per income state
- `mass::Vector{T}` — income-state masses (sum to one)
- `n_moments::Int` — moments carried per income state
- `nodes::Vector{T}` / `weights::Vector{T}` — reference-grid quadrature (asset units)
- `bounds::Tuple{T,T}` — reference interval
- `converged::Bool` — `true` iff every income state's `λ` solve converged

See [`fit_winberry`](@ref), [`winberry_moments`](@ref), [`winberry_histogram`](@ref).
"""
struct WinberryFamily{T<:AbstractFloat}
    densities::Vector{ParametricDensity{T}}
    mass::Vector{T}
    n_moments::Int
    nodes::Vector{T}
    weights::Vector{T}
    bounds::Tuple{T,T}
    converged::Bool
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
- `parametric::Union{Nothing,WinberryFamily{T}}` — fitted Winberry (2018) family
  when the equilibrium was computed with `distribution=:winberry`, `nothing`
  under the default Young histogram

The `parametric` field is a trailing **keyword** on the constructor
(`HASteadyState{T}(…11 positional…; parametric=nothing)`), so existing
positional call sites are unaffected.
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
    parametric::Union{Nothing,WinberryFamily{T}}

    function HASteadyState{T}(policies, distribution, value_fn, prices, aggregates,
                              grid, income, converged, iterations, euler_error,
                              excess_demand;
                              parametric::Union{Nothing,WinberryFamily{T}}=nothing
                              ) where {T<:AbstractFloat}
        new{T}(policies, distribution, value_fn, prices, aggregates, grid, income,
               converged, iterations, euler_error, excess_demand, parametric)
    end
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
