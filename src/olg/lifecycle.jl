# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
True life-cycle overlapping-generations model with age-dependent EGM.

[`BlanchardOLG`](@ref) is *perpetual youth*: every agent faces the same constant
survival probability, so there is no age structure at all. A **life-cycle** model
in the Auerbach–Kotlikoff / İmrohoroğlu–İmrohoroğlu–Joines tradition has finitely
lived agents whose earnings, mortality, and policies all depend on age
`j = 1, …, J`. That is what pension, demographic, and lifecycle-inequality
questions require.

The household problem is solved by **backward induction over age** — one
endogenous-grid sweep per age, from the terminal age down to age 1. There is no
fixed point over policies: age `J` is known (assets are exhausted), and each
earlier age follows from the next. The only fixed point in the model is the
market-clearing interest rate.

# References
- Auerbach, A. J., & Kotlikoff, L. J. (1987). *Dynamic Fiscal Policy*.
  Cambridge University Press.
- İmrohoroğlu, A., İmrohoroğlu, S., & Joines, D. H. (1995). A life cycle analysis
  of social security. *Economic Theory*, 6(1), 83–114.
- Carroll, C. D. (2006). The method of endogenous gridpoints for solving dynamic
  stochastic optimization problems. *Economics Letters*, 91(3), 312–320.
- Young, E. R. (2010). Solving the incomplete markets model with aggregate
  uncertainty using the Krusell-Smith algorithm and non-stochastic simulations.
  *Journal of Economic Dynamics and Control*, 34(1), 36–41.
"""

using LinearAlgebra

# =============================================================================
# LifeCycleOLG — model specification
# =============================================================================

"""
    LifeCycleOLG{T}

Stationary life-cycle overlapping-generations economy.

Households live at most `J` ages, work through age `J_retire − 1` with a
deterministic age-earnings profile `earnings[j]` and persistent idiosyncratic
productivity, then retire on a pay-as-you-go pension. Survival from age `j` to
`j+1` is `survival[j]`, with `survival[J] = 0`. Firms are competitive with
Cobb-Douglas technology.

# Constructor

    LifeCycleOLG(; J=60, J_retire=45, survival=0.99, earnings=nothing,
                   income=rouwenhorst(0.95, 0.2, 5), a_max=60.0, n_a=200,
                   beta=0.97, sigma=2.0, alpha=0.36, delta=0.06, Z=1.0,
                   n_pop=0.0, replacement=0.4, credit_limit=0.0,
                   annuities=true, grid_type=:double_exp)

| Field | Type | Description |
|---|---|---|
| `J` | `Int` | Maximum age |
| `J_retire` | `Int` | First retired age (`J_retire > J` ⇒ nobody retires) |
| `survival` | `Vector{T}` | `s_j`, length `J`, with `s_J = 0`; a scalar is broadcast |
| `earnings` | `Vector{T}` | Deterministic age-earnings profile `κ_j`, length `J`; defaults to the standard quadratic hump peaking near mid-career, zero after retirement |
| `income` | `IncomeProcess{T}` | Persistent idiosyncratic productivity |
| `grid` | `HAGrid{T}` | Asset grid (`grid_type=:double_exp` packs points near the credit limit) |
| `beta`, `sigma` | `T` | Discount factor and CRRA curvature (`sigma = 1` ⇒ log) |
| `alpha`, `delta`, `Z` | `T` | Capital share, depreciation, TFP |
| `n_pop` | `T` | Population growth rate |
| `replacement` | `T` | Pension as a fraction of average gross labor income (`0` ⇒ no social security) |
| `credit_limit` | `T` | Lower bound on assets |
| `annuities` | `Bool` | `true` ⇒ actuarially fair annuities (survivors earn `(1+r)/s_j`, no accidental bequests); `false` ⇒ accidental bequests rebated lump-sum, solved by an inner fixed point |

# Accidental bequests

With `annuities = true` the assets of the deceased are absorbed by an actuarially
fair annuity, exactly the Blanchard–Yaari device the perpetual-youth model uses,
so the two families nest. With `annuities = false` the assets of those who die are
rebated equally to the living, which makes the transfer a fixed point: policies
depend on the transfer and the transfer on the policies.
"""
struct LifeCycleOLG{T<:AbstractFloat}
    J::Int
    J_retire::Int
    survival::Vector{T}
    earnings::Vector{T}
    income::IncomeProcess{T}
    grid::HAGrid{T}
    beta::T
    sigma::T
    alpha::T
    delta::T
    Z::T
    n_pop::T
    replacement::T
    credit_limit::T
    annuities::Bool
end

"""
    LifeCycleSystem{T} <: AbstractAgentSystem{T}

Finite-horizon life-cycle OLG population. `to_spec` / `@dsge horizon: ages`
(G-03) wrap a [`LifeCycleOLG`](@ref).
"""
struct LifeCycleSystem{T<:AbstractFloat} <: AbstractAgentSystem{T}
    model::LifeCycleOLG{T}
end

# Plot helpers rewritten in #634 call `_hh(ss.spec).grid` / `.income`.
# `LifeCycleSteadyState.spec` is the `LifeCycleOLG` payload itself.
_hh(m::LifeCycleOLG) = m

"""
    to_spec(m::LifeCycleOLG; agent_name::Symbol=:households) -> ModelSpec

Wrap a [`LifeCycleOLG`](@ref) as a [`ModelSpec`](@ref) whose `agents`
NamedTuple holds a [`LifeCycleSystem`](@ref) keyed by `agent_name`.

The aggregate residual system is empty (partial GE). Stationary equilibrium
is [`lifecycle_steady_state`](@ref) of the wrapped payload — `compute_steady_state`
/ `solve` dispatch on this kind lands in G-05.
"""
function to_spec(m::LifeCycleOLG{T}; agent_name::Symbol=:households) where {T}
    params = [:beta, :sigma, :alpha, :delta, :Z, :n_pop, :replacement, :credit_limit]
    param_values = Dict{Symbol,T}(
        :beta => m.beta,
        :sigma => m.sigma,
        :alpha => m.alpha,
        :delta => m.delta,
        :Z => m.Z,
        :n_pop => m.n_pop,
        :replacement => m.replacement,
        :credit_limit => m.credit_limit,
    )
    return ModelSpec{T}(
        Symbol[], Symbol[], params, param_values,
        NamedEquation[], Function[],
        0, Int[], T[];
        agents=NamedTuple{(agent_name,)}((LifeCycleSystem{T}(m),)),
        ir=ModelIR(:discrete, :ages, IRDecl[], IREquation[]),
    )
end

"""
    _lifecycle_earnings(J, J_retire) -> Vector{Float64}

Default deterministic age-earnings profile: a quadratic hump in experience,
normalized to a mean of one over working ages and zero thereafter. Hump-shaped
earnings are what generate the life-cycle consumption hump.
"""
function _lifecycle_earnings(J::Int, J_retire::Int)
    kappa = zeros(Float64, J)
    n_work = min(J_retire - 1, J)
    n_work <= 0 && return kappa
    for j in 1:n_work
        x = (j - 1) / max(n_work - 1, 1)
        kappa[j] = 1.0 + 1.6 * x - 1.8 * x^2       # peaks at roughly 45% of working life
    end
    m = sum(@view kappa[1:n_work]) / n_work
    m > 0 && (kappa[1:n_work] ./= m)
    return kappa
end

"""
    lifecycle_income(rho, sigma, n; method=:rouwenhorst) -> IncomeProcess

Discretize log productivity `log e' = ρ log e + σ ε` and return the chain **in
levels**, normalized so that `E[e] = 1`.

[`rouwenhorst`](@ref) and [`tauchen`](@ref) return the grid in *logs*, symmetric
about zero, so their mean is zero rather than one. Feeding that straight into a
production economy makes aggregate efficiency labor collapse to zero and every
factor price with it. This helper does the `exp`-and-normalize step, and
[`LifeCycleOLG`](@ref) rejects any income process whose states are not positive.
"""
function lifecycle_income(rho::Real, sigma::Real, n::Int; method::Symbol=:rouwenhorst)
    method in (:rouwenhorst, :tauchen) || throw(ArgumentError(
        "`method` must be :rouwenhorst or :tauchen (got :$method)"))
    raw = method === :tauchen ? tauchen(rho, sigma, n) : rouwenhorst(rho, sigma, n)
    e = exp.(raw.states)
    e ./= dot(raw.stationary_dist, e)
    return IncomeProcess{Float64}(raw.transition, e, raw.stationary_dist, :income)
end

"""
    lifecycle_survival(J; age0=21, makeham=0.0002, gompertz=2.7e-5, growth=0.095)
        -> Vector{Float64}

Gompertz–Makeham survival profile `s_j = exp(−μ(x))` with hazard
`μ(x) = makeham + gompertz · exp(growth · x)` at calendar age `x = age0 + j − 1`,
truncated by `s_J = 0`.

The defaults give roughly 0.04% annual mortality at age 21 rising to about 8% at
age 85 — steep enough that late-life mortality actually bends the consumption
path. A flat survival probability cannot produce a life-cycle consumption hump.
"""
function lifecycle_survival(J::Int; age0::Real=21, makeham::Real=0.0002,
                            gompertz::Real=2.7e-5, growth::Real=0.095)
    J >= 2 || throw(ArgumentError("`J` must be at least 2"))
    s = [exp(-(makeham + gompertz * exp(growth * (age0 + j - 1)))) for j in 1:J]
    s = clamp.(Float64.(s), 0.0, 1.0)
    s[J] = 0.0
    return s
end

function LifeCycleOLG(; J::Int=60, J_retire::Int=45,
                        survival::Union{Real,AbstractVector{<:Real}}=0.99,
                        earnings::Union{Nothing,AbstractVector{<:Real}}=nothing,
                        income::IncomeProcess{T}=lifecycle_income(0.95, 0.2, 5),
                        a_max::Real=60.0, n_a::Int=200,
                        beta::Real=0.97, sigma::Real=2.0,
                        alpha::Real=0.36, delta::Real=0.06, Z::Real=1.0,
                        n_pop::Real=0.0, replacement::Real=0.4,
                        credit_limit::Real=0.0, annuities::Bool=true,
                        grid_type::Symbol=:double_exp) where {T<:AbstractFloat}
    J >= 2 || throw(ArgumentError("`J` must be at least 2 (got $J)"))
    J_retire >= 2 || throw(ArgumentError("`J_retire` must be at least 2 (got $J_retire)"))
    0 < beta < 1 || throw(ArgumentError("`beta` must lie in (0, 1), got $beta"))
    sigma > 0 || throw(ArgumentError("`sigma` must be positive, got $sigma"))
    0 < alpha < 1 || throw(ArgumentError("`alpha` must lie in (0, 1), got $alpha"))
    0 <= delta <= 1 || throw(ArgumentError("`delta` must lie in [0, 1], got $delta"))
    n_pop > -1 || throw(ArgumentError("`n_pop` must exceed −1, got $n_pop"))
    replacement >= 0 || throw(ArgumentError("`replacement` must be non-negative"))
    n_a >= 3 || throw(ArgumentError("`n_a` must be at least 3"))
    a_max > credit_limit || throw(ArgumentError("`a_max` must exceed `credit_limit`"))
    # `rouwenhorst`/`tauchen` return LOG states, whose mean is zero. Passing one
    # straight in silently zeroes aggregate efficiency labor and every factor price.
    all(>(0), income.states) || throw(ArgumentError(
        "`income.states` must be productivity LEVELS, all strictly positive (got " *
        "$(round.(income.states; digits=4))). `rouwenhorst`/`tauchen` return log " *
        "states — use `lifecycle_income(rho, sigma, n)` to exponentiate and " *
        "normalize to unit mean."))

    s = if survival isa Real
        0 < survival <= 1 || throw(ArgumentError("scalar `survival` must lie in (0, 1]"))
        fill(T(survival), J)
    else
        length(survival) == J || throw(ArgumentError(
            "`survival` has $(length(survival)) entries but J = $J"))
        all(x -> 0 <= x <= 1, survival) || throw(ArgumentError(
            "every survival probability must lie in [0, 1]"))
        collect(T, survival)
    end
    s[J] = zero(T)                       # nobody survives the terminal age

    kappa = if earnings === nothing
        collect(T, _lifecycle_earnings(J, J_retire))
    else
        length(earnings) == J || throw(ArgumentError(
            "`earnings` has $(length(earnings)) entries but J = $J"))
        collect(T, earnings)
    end

    # Use the package's own grid builder so that `_build_transition_matrix` is fed the
    # IDENTICAL nodes the policies were computed on. Constructing an `HAGrid` from
    # bounds while carrying a separately-built vector would silently map the savings
    # policy onto the wrong nodes.
    grid = HAGrid(; assets=(Float64(credit_limit), Float64(a_max), n_a),
                    income_states=length(income.states), grid_type=grid_type)

    return LifeCycleOLG{T}(J, J_retire, s, kappa, income, grid, T(beta), T(sigma),
                           T(alpha), T(delta), T(Z), T(n_pop), T(replacement),
                           T(credit_limit), annuities)
end

# CRRA marginal utility and its inverse (σ = 1 ⇒ log).
_lc_uprime(c::T, sigma::T) where {T} = c > zero(T) ? c^(-sigma) : T(Inf)
_lc_uprime_inv(m::T, sigma::T) where {T} = m > zero(T) ? m^(-one(T) / sigma) : T(Inf)
function _lc_utility(c::T, sigma::T) where {T}
    c <= zero(T) && return T(-Inf)
    return isapprox(sigma, one(T)) ? log(c) : (c^(one(T) - sigma) - one(T)) / (one(T) - sigma)
end

"""
    _lc_interp(x, y, xi) -> yi

Linear interpolation with **linear** extrapolation above the last knot and flat
extrapolation below the first.

Above the endogenous grid the consumption function is close to linear in wealth,
so extending the last segment is far more accurate than holding it flat — and in a
life-cycle model, where assets peak just before retirement, the top of the grid is
exactly where households spend their highest-wealth years. Below the first knot the
household is credit constrained and the caller uses the analytic branch instead, so
flat is harmless there.
"""
function _lc_interp(x::AbstractVector{T}, y::AbstractVector{T}, xi::T) where {T<:AbstractFloat}
    n = length(x)
    xi <= x[1] && return y[1]
    if xi >= x[n]
        dx = x[n] - x[n-1]
        dx <= zero(T) && return y[n]
        return y[n] + (xi - x[n]) / dx * (y[n] - y[n-1])
    end
    k = clamp(searchsortedfirst(x, xi) - 1, 1, n - 1)
    dx = x[k+1] - x[k]
    dx <= zero(T) && return y[k]
    return y[k] + (xi - x[k]) / dx * (y[k+1] - y[k])
end

# =============================================================================
# Prices, taxes, demographics
# =============================================================================

"Competitive factor prices at capital-labor ratio `K/L` (and optional TFP `Z`)."
function _lc_prices(m::LifeCycleOLG{T}, KL::T, Z::T) where {T<:AbstractFloat}
    r = m.alpha * Z * KL^(m.alpha - one(T)) - m.delta
    w = (one(T) - m.alpha) * Z * KL^m.alpha
    return (r, w)
end
_lc_prices(m::LifeCycleOLG{T}, KL::T) where {T<:AbstractFloat} = _lc_prices(m, KL, m.Z)

"""
    _lc_cohort_mass(m) -> Vector{T}

Stationary population share of each age, `μ_j ∝ ∏_{k<j} s_k / (1+n)^{j-1}`,
normalized to sum to one.
"""
function _lc_cohort_mass(m::LifeCycleOLG{T}) where {T<:AbstractFloat}
    mu = zeros(T, m.J)
    mu[1] = one(T)
    for j in 2:m.J
        mu[j] = mu[j-1] * m.survival[j-1] / (one(T) + m.n_pop)
    end
    s = sum(mu)
    s > zero(T) && (mu ./= s)
    return mu
end

"Aggregate efficiency labor supply per capita, given the cohort masses."
function _lc_labor(m::LifeCycleOLG{T}, mu::Vector{T}) where {T<:AbstractFloat}
    mean_e = dot(m.income.stationary_dist, m.income.states)
    L = zero(T)
    for j in 1:min(m.J_retire - 1, m.J)
        L += mu[j] * m.earnings[j] * mean_e
    end
    return L
end

# =============================================================================
# Backward EGM sweep over age
# =============================================================================

"""
    lifecycle_policies(m, r, w; tau=0, pension=0, transfer=0) -> (c_pol, a_pol)

Solve the household problem by **backward induction over age**: one endogenous-grid
sweep per age, from the terminal age `J` down to age 1.

At the terminal age assets are exhausted, so consumption is all of cash-on-hand
above the credit limit. At every earlier age the Euler equation

```math
u'(c_j) = \\beta\\, s_j\\, R_j\\, E\\!\\left[u'\\!\\left(c_{j+1}(a', e')\\right) \\mid e\\right]
```

is inverted on the exogenous savings grid to give the endogenous cash-on-hand at
which each `a'` is chosen; interpolating back onto the asset grid gives the age-`j`
policy, and assets below the smallest endogenous point are credit constrained.

`R_j` is `(1+r)/s_j` with actuarially fair annuities and `1+r` without — with
annuities the survival probability cancels out of the Euler equation exactly, which
is the Blanchard–Yaari result the perpetual-youth model relies on.

Unlike infinite-horizon EGM this is a **finite sweep**: there is no policy fixed
point, so there is no convergence flag here. The only fixed point in the model is
the market-clearing price.

# Returns
- `c_pol::Array{T,3}` — consumption, `n_a × n_e × J`
- `a_pol::Array{T,3}` — end-of-period assets, `n_a × n_e × J`
"""
function lifecycle_policies(m::LifeCycleOLG{T}, r::Real, w::Real;
                            tau::Real=0.0, pension::Real=0.0,
                            transfer::Real=0.0) where {T<:AbstractFloat}
    a_grid = m.grid.grids[1]
    n_a = length(a_grid)
    e_vals = m.income.states
    n_e = length(e_vals)
    Pi = m.income.transition
    a_min = m.credit_limit
    sigma = m.sigma
    R = one(T) + T(r)
    wT = T(w); tauT = T(tau); penT = T(pension); trT = T(transfer)

    # Gross income at age j in productivity state e (before asset income).
    inc(j, ie) = j < m.J_retire ? (one(T) - tauT) * wT * m.earnings[j] * e_vals[ie] : penT

    c_pol = zeros(T, n_a, n_e, m.J)
    a_pol = zeros(T, n_a, n_e, m.J)

    # ── Terminal age: exhaust assets ────────────────────────────────────────
    # Households may borrow *during* life (down to `credit_limit`) but cannot die in
    # debt, so terminal assets are `max(credit_limit, 0)` — NOT the credit limit
    # itself, which would let them consume an unfunded windfall and break the
    # lifetime budget constraint.
    a_term = max(a_min, zero(T))
    for ie in 1:n_e, i in 1:n_a
        gross = m.annuities ? _lc_gross_return(m, r, m.J - 1) : R
        coh = gross * a_grid[i] + inc(m.J, ie) + trT
        c_pol[i, ie, m.J] = max(coh - a_term, T(1e-12))
        a_pol[i, ie, m.J] = a_term
    end

    emu = zeros(T, n_a)
    c_endo = zeros(T, n_a)
    a_endo = zeros(T, n_a)

    for j in (m.J - 1):-1:1
        gross_next = m.annuities ? _lc_gross_return(m, r, j) : R    # return on a' saved at j
        gross_now = m.annuities ? _lc_gross_return(m, r, j - 1) : R # return on a brought into j
        for ie in 1:n_e
            fill!(emu, zero(T))
            for je in 1:n_e
                p = Pi[ie, je]
                p == zero(T) && continue
                for i in 1:n_a
                    emu[i] += p * _lc_uprime(c_pol[i, je, j+1], sigma)
                end
            end
            for i in 1:n_a
                # With annuities R_j = (1+r)/s_j, so β s_j R_j = β(1+r): survival cancels.
                rhs = m.beta * m.survival[j] * gross_next * emu[i]
                c_endo[i] = _lc_uprime_inv(rhs, sigma)
                a_endo[i] = (c_endo[i] + a_grid[i] - inc(j, ie) - trT) / gross_now
            end
            for i in 1:n_a
                a_val = a_grid[i]
                coh = gross_now * a_val + inc(j, ie) + trT
                if a_val <= a_endo[1]
                    c_pol[i, ie, j] = max(coh - a_min, T(1e-12))
                    a_pol[i, ie, j] = a_min
                else
                    cj = _lc_interp(a_endo, c_endo, a_val)
                    cj = clamp(cj, T(1e-12), coh - a_min)
                    c_pol[i, ie, j] = cj
                    a_pol[i, ie, j] = clamp(coh - cj, a_min, a_grid[end])
                end
            end
        end
    end
    return c_pol, a_pol
end

"Gross return earned on assets saved at age `j` (age 0 means the newborn's zero assets)."
function _lc_gross_return(m::LifeCycleOLG{T}, r::Real, j::Int) where {T<:AbstractFloat}
    R = one(T) + T(r)
    (j < 1 || j > m.J) && return R
    s = m.survival[j]
    return s > zero(T) ? R / s : R
end

# =============================================================================
# Age-extended Young histogram
# =============================================================================

"""
    lifecycle_distribution(m, a_pol; initial_assets=0.0) -> Array{T,3}

Cross-sectional distribution over `(assets, productivity, age)`, `n_a × n_e × J`,
weighted by the stationary cohort masses so the whole array sums to one.

Newborns enter at age 1 with `initial_assets` (split across the two bracketing grid
nodes by the Young lottery) and the stationary productivity distribution. Each
later age is the previous age's distribution pushed through that age's savings
policy. Because survival is independent of assets and productivity, mortality
rescales cohorts without distorting the within-cohort distribution — so the age
dimension enters only through the cohort weights.
"""
function lifecycle_distribution(m::LifeCycleOLG{T}, a_pol::Array{T,3};
                                initial_assets::Real=0.0) where {T<:AbstractFloat}
    a_grid = m.grid.grids[1]
    n_a = length(a_grid)
    n_e = length(m.income.states)
    mu = _lc_cohort_mass(m)

    # Age-1 distribution: newborn assets × stationary productivity.
    phi = zeros(T, n_a * n_e)
    a0 = clamp(T(initial_assets), a_grid[1], a_grid[end])
    k = clamp(searchsortedfirst(a_grid, a0) - 1, 1, n_a - 1)
    dx = a_grid[k+1] - a_grid[k]
    wgt = dx > zero(T) ? (a0 - a_grid[k]) / dx : zero(T)
    for ie in 1:n_e
        pe = T(m.income.stationary_dist[ie])
        phi[(ie - 1) * n_a + k] += pe * (one(T) - wgt)
        phi[(ie - 1) * n_a + k + 1] += pe * wgt
    end

    dist = zeros(T, n_a, n_e, m.J)
    for j in 1:m.J
        dist[:, :, j] = reshape(phi, n_a, n_e) .* mu[j]
        j == m.J && break
        Lambda = _build_transition_matrix(Matrix{T}(@view a_pol[:, :, j]), m.grid, m.income)
        phi = Lambda * phi
        s = sum(phi)
        s > zero(T) && (phi ./= s)          # renormalize within the surviving cohort
    end
    return dist
end

# =============================================================================
# LifeCycleSteadyState
# =============================================================================

"""
    LifeCycleSteadyState{T}

Stationary equilibrium of a [`LifeCycleOLG`](@ref) economy.

| Field | Type | Description |
|---|---|---|
| `r`, `w` | `T` | Equilibrium interest rate and wage |
| `K`, `L` | `T` | Aggregate capital (the integral of `dist` over assets) and efficiency labor, per capita |
| `Y` | `T` | Output |
| `tau`, `pension` | `T` | Payroll tax rate and pension benefit balancing the pay-as-you-go budget |
| `transfer` | `T` | Lump-sum rebate of accidental bequests (`0` under annuities) |
| `c_policy`, `a_policy` | `Array{T,3}` | `n_a × n_e × J` policies |
| `dist` | `Array{T,3}` | `n_a × n_e × J` population distribution, sums to one |
| `cohort_mass` | `Vector{T}` | Stationary population share by age |
| `asset_profile` | `Vector{T}` | Mean assets held at each age |
| `consumption_profile` | `Vector{T}` | Mean consumption at each age |
| `income_profile` | `Vector{T}` | Mean non-asset income at each age |
| `converged` | `Bool` | Whether the market-clearing bisection met `tol` |
| `iterations` | `Int` | Bisection iterations used |
| `excess_demand` | `T` | Final `K_supply − K_demand` |
| `spec` | `LifeCycleOLG{T}` | Model solved |
"""
struct LifeCycleSteadyState{T<:AbstractFloat}
    r::T
    w::T
    K::T
    L::T
    Y::T
    tau::T
    pension::T
    transfer::T
    c_policy::Array{T,3}
    a_policy::Array{T,3}
    dist::Array{T,3}
    cohort_mass::Vector{T}
    asset_profile::Vector{T}
    consumption_profile::Vector{T}
    income_profile::Vector{T}
    converged::Bool
    iterations::Int
    excess_demand::T
    spec::LifeCycleOLG{T}
end

"""
    _lc_supply(m, KL; bequest_iter, bequest_tol) -> NamedTuple

Household capital supply at capital-labor ratio `KL`, together with everything the
steady-state report needs. Without annuities the accidental-bequest rebate is a
fixed point — policies depend on the transfer and the transfer on the policies — so
it is iterated to `bequest_tol` here, inside the price loop.
"""
function _lc_supply(m::LifeCycleOLG{T}, KL::T; bequest_iter::Int=50,
                    bequest_tol::Real=1e-10) where {T<:AbstractFloat}
    r, w = _lc_prices(m, KL)
    mu = _lc_cohort_mass(m)
    L = _lc_labor(m, mu)

    # Pay-as-you-go social security: τ w L = pension × (retired mass).
    mass_ret = sum(@view mu[min(m.J_retire, m.J + 1):end])
    mass_work = sum(@view mu[1:min(m.J_retire - 1, m.J)])
    pension = zero(T); tau = zero(T)
    if m.replacement > zero(T) && mass_ret > zero(T) && mass_work > zero(T)
        pension = m.replacement * w * (L / mass_work)
        tau = pension * mass_ret / (w * L)
    end

    transfer = zero(T)
    local c_pol, a_pol, dist
    for _ in 1:(m.annuities ? 1 : bequest_iter)
        c_pol, a_pol = lifecycle_policies(m, r, w; tau=tau, pension=pension,
                                          transfer=transfer)
        dist = lifecycle_distribution(m, a_pol)
        m.annuities && break
        # Assets of those who die at the end of age j, rebated to the living.
        beq = zero(T)
        for j in 1:m.J
            (one(T) - m.survival[j]) == zero(T) && continue
            beq += (one(T) - m.survival[j]) * (one(T) + r) *
                   sum(@view(dist[:, :, j]) .* a_pol[:, :, j])
        end
        beq /= (one(T) + m.n_pop)
        if abs(beq - transfer) < T(bequest_tol)
            transfer = beq
            break
        end
        transfer = T(0.5) * transfer + T(0.5) * beq       # damped, the map is a contraction
    end

    a_grid = m.grid.grids[1]
    n_a = length(a_grid)
    K_supply = zero(T)
    a_prof = zeros(T, m.J); c_prof = zeros(T, m.J); y_prof = zeros(T, m.J)
    e_vals = m.income.states
    for j in 1:m.J
        mj = sum(@view dist[:, :, j])
        acc = zero(T); cc = zero(T); yy = zero(T)
        for ie in 1:length(e_vals), i in 1:n_a
            wgt = dist[i, ie, j]
            wgt == zero(T) && continue
            acc += wgt * a_grid[i]
            cc += wgt * c_pol[i, ie, j]
            yy += wgt * (j < m.J_retire ? (one(T) - tau) * w * m.earnings[j] * e_vals[ie]
                                        : pension)
        end
        K_supply += acc
        if mj > zero(T)
            a_prof[j] = acc / mj; c_prof[j] = cc / mj; y_prof[j] = yy / mj
        end
    end

    return (r=r, w=w, L=L, K_supply=K_supply, tau=tau, pension=pension,
            transfer=transfer, c_pol=c_pol, a_pol=a_pol, dist=dist, mu=mu,
            a_prof=a_prof, c_prof=c_prof, y_prof=y_prof)
end

"""
    lifecycle_steady_state(m; r_bounds=(-0.02, 0.10), tol=1e-6, max_iter=60,
                              bequest_iter=50, verbose=false) -> LifeCycleSteadyState

Stationary equilibrium of a life-cycle OLG economy.

Bisects on the capital-labor ratio implied by the interest rate until household
capital supply — obtained from the backward EGM sweep and the age-extended Young
histogram — equals firm capital demand. Raising `K/L` lowers the interest rate, so
household supply falls while firm demand `(K/L)·L` rises: excess supply is
decreasing in `K/L` and crosses zero once, which makes bisection globally reliable.

# Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `r_bounds` | `Tuple` | `(-0.02, 0.10)` | Interest-rate bracket, converted to a `K/L` bracket |
| `tol` | `Real` | ``10^{-6}`` | Absolute tolerance on excess capital supply |
| `max_iter` | `Int` | `60` | Bisection iterations |
| `bequest_iter` | `Int` | `50` | Inner iterations for the bequest rebate (`annuities=false` only) |
| `verbose` | `Bool` | `false` | Print bisection diagnostics |

Non-convergence is reported in `converged`/`excess_demand`, never silently
accepted: a life-cycle economy whose asset grid truncates the pre-retirement peak
will fail to clear, and that must be visible.

# Returns
[`LifeCycleSteadyState{T}`](@ref).
"""
function lifecycle_steady_state(m::LifeCycleOLG{T};
                                r_bounds::Tuple{<:Real,<:Real}=(-0.02, 0.10),
                                tol::Real=1e-6, max_iter::Int=60,
                                bequest_iter::Int=50,
                                verbose::Bool=false) where {T<:AbstractFloat}
    r_lo, r_hi = T(r_bounds[1]), T(r_bounds[2])
    r_lo < r_hi || throw(ArgumentError("`r_bounds` must be increasing, got $r_bounds"))
    # r = αZ(K/L)^{α−1} − δ  ⇒  K/L = (αZ / (r + δ))^{1/(1−α)}; r high ⇒ K/L low.
    kl(rv) = (m.alpha * m.Z / (rv + m.delta))^(one(T) / (one(T) - m.alpha))
    (r_lo + m.delta) > zero(T) || throw(ArgumentError(
        "`r_bounds[1]` must exceed −delta = $(-m.delta) for a finite capital-labor ratio"))

    # Excess capital SUPPLY as a function of K/L (increasing in K/L).
    lo = kl(r_hi); hi = kl(r_lo)
    res_lo = _lc_supply(m, lo; bequest_iter=bequest_iter)
    res_hi = _lc_supply(m, hi; bequest_iter=bequest_iter)
    f_lo = res_lo.K_supply - lo * res_lo.L
    f_hi = res_hi.K_supply - hi * res_hi.L

    best = f_lo * f_lo <= f_hi * f_hi ? res_lo : res_hi
    best_kl = f_lo * f_lo <= f_hi * f_hi ? lo : hi
    best_f = f_lo * f_lo <= f_hi * f_hi ? f_lo : f_hi
    iters = 0
    converged = false

    if f_lo * f_hi <= zero(T)
        a, b = lo, hi
        fa = f_lo
        for it in 1:max_iter
            iters = it
            mid = (a + b) / 2
            res = _lc_supply(m, mid; bequest_iter=bequest_iter)
            fm = res.K_supply - mid * res.L
            if abs(fm) < abs(best_f)
                best = res; best_kl = mid; best_f = fm
            end
            verbose && println("lifecycle bisection $it: K/L = $mid, excess = $fm")
            if abs(fm) < T(tol)
                converged = true
                best = res; best_kl = mid; best_f = fm
                break
            end
            if fa * fm <= zero(T)
                b = mid
            else
                a = mid; fa = fm
            end
        end
    else
        @warn "lifecycle_steady_state: excess capital supply does not change sign on the " *
              "requested bracket; returning the closest endpoint" r_bounds=r_bounds excess_lo=f_lo excess_hi=f_hi
    end

    # Report the capital households actually hold — the integral of the age-asset
    # distribution — so the accounting identity holds by construction. The gap
    # against firm demand `best_kl * best.L` is exactly `excess_demand`.
    K = best.K_supply
    Y = m.Z * K^m.alpha * best.L^(one(T) - m.alpha)
    return LifeCycleSteadyState{T}(best.r, best.w, K, best.L, Y, best.tau,
                                   best.pension, best.transfer, best.c_pol,
                                   best.a_pol, best.dist, best.mu, best.a_prof,
                                   best.c_prof, best.y_prof, converged, iters,
                                   best_f, m)
end

# =============================================================================
# Perfect-foresight transition (G-12 / #646)
# =============================================================================

"""
    LifeCycleTransition{T}

Deterministic perfect-foresight path of a [`LifeCycleOLG`](@ref) economy.

All series have length `H+1` (dates `t = 0, …, H`). `K[1]` is the predetermined
initial capital; `K[end]` is household supply after `H` periods and, when the
horizon is long enough, sits at the stationary [`LifeCycleSteadyState`](@ref).
"""
struct LifeCycleTransition{T<:AbstractFloat}
    K::Vector{T}
    r::Vector{T}
    w::Vector{T}
    Y::Vector{T}
    C::Vector{T}
    Z::Vector{T}
    pension::Vector{T}
    transfer::Vector{T}
    tau::T
    converged::Bool
    iterations::Int
    ss::LifeCycleSteadyState{T}
end

export LifeCycleTransition, lifecycle_transition

"Beginning-of-period aggregate capital from a `(a, e, age)` histogram."
function _lc_aggregate_K(a_grid::AbstractVector{T}, dist::Array{T,3}) where {T<:AbstractFloat}
    n_a, n_e, J = size(dist)
    K = zero(T)
    @inbounds for j in 1:J, ie in 1:n_e, i in 1:n_a
        K += dist[i, ie, j] * a_grid[i]
    end
    return K
end

"Cross-sectional mean of a policy `x` against a `(a, e, age)` histogram."
function _lc_aggregate_X(x::AbstractArray{T,3}, dist::Array{T,3}) where {T<:AbstractFloat}
    n_a, n_e, J = size(dist)
    acc = zero(T)
    @inbounds for j in 1:J, ie in 1:n_e, i in 1:n_a
        acc += dist[i, ie, j] * x[i, ie, j]
    end
    return acc
end

"""
    _lc_rescale_dist(m, dist, lambda) -> Array{T,3}

Young-lottery remesh of `dist` after every household's assets are scaled by
`lambda`. Cohort and productivity masses are preserved; nodes that would leave
the grid are clamped.
"""
function _lc_rescale_dist(m::LifeCycleOLG{T}, dist::Array{T,3},
                          lambda::T) where {T<:AbstractFloat}
    isapprox(lambda, one(T)) && return copy(dist)
    a_grid = m.grid.grids[1]
    n_a, n_e, J = size(dist)
    out = zeros(T, n_a, n_e, J)
    @inbounds for j in 1:J, ie in 1:n_e, i in 1:n_a
        wgt = dist[i, ie, j]
        wgt == zero(T) && continue
        a_new = clamp(lambda * a_grid[i], a_grid[1], a_grid[end])
        k = clamp(searchsortedfirst(a_grid, a_new) - 1, 1, n_a - 1)
        dx = a_grid[k+1] - a_grid[k]
        ω = dx > zero(T) ? (a_new - a_grid[k]) / dx : zero(T)
        out[k, ie, j] += wgt * (one(T) - ω)
        out[k+1, ie, j] += wgt * ω
    end
    return out
end

"""
    _lc_forward_dist(m, dist, a_pol) -> Array{T,3}

One-period Young push of a life-cycle histogram. Newborns enter at age 1 with
zero assets and the stationary productivity law; each later age is the previous
age's within-cohort distribution pushed through that age's savings policy and
reweighted by the stationary cohort mass. Demographics stay at
[`_lc_cohort_mass`](@ref) — this is a price/TFP transition, not a demographic one.
"""
function _lc_forward_dist(m::LifeCycleOLG{T}, dist::Array{T,3},
                          a_pol::AbstractArray{T,3}) where {T<:AbstractFloat}
    a_grid = m.grid.grids[1]
    n_a = length(a_grid)
    n_e = length(m.income.states)
    Pi = m.income.transition
    mu = _lc_cohort_mass(m)
    new_dist = zeros(T, n_a, n_e, m.J)

    # Age-1: newborns, zero assets × stationary productivity.
    a0 = clamp(zero(T), a_grid[1], a_grid[end])
    k0 = clamp(searchsortedfirst(a_grid, a0) - 1, 1, n_a - 1)
    dx0 = a_grid[k0+1] - a_grid[k0]
    ω0 = dx0 > zero(T) ? (a0 - a_grid[k0]) / dx0 : zero(T)
    for ie in 1:n_e
        pe = T(m.income.stationary_dist[ie]) * mu[1]
        new_dist[k0, ie, 1] += pe * (one(T) - ω0)
        new_dist[k0+1, ie, 1] += pe * ω0
    end

    tmp = zeros(T, n_a, n_e)
    @inbounds for j in 1:(m.J - 1)
        fill!(tmp, zero(T))
        for ie in 1:n_e, i in 1:n_a
            wgt = dist[i, ie, j]
            wgt == zero(T) && continue
            ap = clamp(a_pol[i, ie, j], a_grid[1], a_grid[end])
            k = clamp(searchsortedfirst(a_grid, ap) - 1, 1, n_a - 1)
            dx = a_grid[k+1] - a_grid[k]
            ω = dx > zero(T) ? (ap - a_grid[k]) / dx : zero(T)
            for je in 1:n_e
                p = Pi[ie, je]
                p == zero(T) && continue
                tmp[k, je] += wgt * (one(T) - ω) * p
                tmp[k+1, je] += wgt * ω * p
            end
        end
        s = sum(tmp)
        scale = (s > zero(T) && mu[j+1] > zero(T)) ? mu[j+1] / s : zero(T)
        for ie in 1:n_e, i in 1:n_a
            new_dist[i, ie, j+1] = tmp[i, ie] * scale
        end
    end
    return new_dist
end

"""
    _lc_policies_at_date!(c_pol, a_pol, m, r_now, r_next, w, tau, pension, transfer, c_next)

Age-EGM sweep at one date of a perfect-foresight path. Continuation marginal
utilities come from next period's consumption policy `c_next` (age `j+1`); the
return on assets brought into the date uses `r_now`, the return on `a'` uses
`r_next`.
"""
function _lc_policies_at_date!(c_pol::AbstractArray{T,3}, a_pol::AbstractArray{T,3},
                               m::LifeCycleOLG{T}, r_now::T, r_next::T, w::T,
                               tau::T, pension::T, transfer::T,
                               c_next::AbstractArray{T,3}) where {T<:AbstractFloat}
    a_grid = m.grid.grids[1]
    n_a = length(a_grid)
    e_vals = m.income.states
    n_e = length(e_vals)
    Pi = m.income.transition
    a_min = m.credit_limit
    sigma = m.sigma
    R_now = one(T) + r_now
    R_next = one(T) + r_next

    inc(j, ie) = j < m.J_retire ? (one(T) - tau) * w * m.earnings[j] * e_vals[ie] : pension

    a_term = max(a_min, zero(T))
    for ie in 1:n_e, i in 1:n_a
        gross = m.annuities ? _lc_gross_return(m, r_now, m.J - 1) : R_now
        coh = gross * a_grid[i] + inc(m.J, ie) + transfer
        c_pol[i, ie, m.J] = max(coh - a_term, T(1e-12))
        a_pol[i, ie, m.J] = a_term
    end

    emu = zeros(T, n_a)
    c_endo = zeros(T, n_a)
    a_endo = zeros(T, n_a)

    for j in (m.J - 1):-1:1
        gross_next = m.annuities ? _lc_gross_return(m, r_next, j) : R_next
        gross_now = m.annuities ? _lc_gross_return(m, r_now, j - 1) : R_now
        for ie in 1:n_e
            fill!(emu, zero(T))
            for je in 1:n_e
                p = Pi[ie, je]
                p == zero(T) && continue
                for i in 1:n_a
                    emu[i] += p * _lc_uprime(c_next[i, je, j+1], sigma)
                end
            end
            for i in 1:n_a
                rhs = m.beta * m.survival[j] * gross_next * emu[i]
                c_endo[i] = _lc_uprime_inv(rhs, sigma)
                a_endo[i] = (c_endo[i] + a_grid[i] - inc(j, ie) - transfer) / gross_now
            end
            for i in 1:n_a
                a_val = a_grid[i]
                coh = gross_now * a_val + inc(j, ie) + transfer
                if a_val <= a_endo[1]
                    c_pol[i, ie, j] = max(coh - a_min, T(1e-12))
                    a_pol[i, ie, j] = a_min
                else
                    cj = _lc_interp(a_endo, c_endo, a_val)
                    cj = clamp(cj, T(1e-12), coh - a_min)
                    c_pol[i, ie, j] = cj
                    a_pol[i, ie, j] = clamp(coh - cj, a_min, a_grid[end])
                end
            end
        end
    end
    return c_pol, a_pol
end

"Pay-as-you-go tax rate (independent of the wage) and the benefit at wage `w`."
function _lc_ss_budget(m::LifeCycleOLG{T}, w::T, L::T, mu::Vector{T}) where {T<:AbstractFloat}
    mass_ret = sum(@view mu[min(m.J_retire, m.J + 1):end])
    mass_work = sum(@view mu[1:min(m.J_retire - 1, m.J)])
    if m.replacement > zero(T) && mass_ret > zero(T) && mass_work > zero(T) && L > zero(T)
        pension = m.replacement * w * (L / mass_work)
        tau = pension * mass_ret / (w * L)
        return tau, pension
    end
    return zero(T), zero(T)
end

"Accidental-bequest rebate implied by `(dist, a_pol)` at the current interest rate."
function _lc_bequest(m::LifeCycleOLG{T}, dist::Array{T,3}, a_pol::AbstractArray{T,3},
                     r::T) where {T<:AbstractFloat}
    beq = zero(T)
    n_a, n_e, J = size(dist)
    @inbounds for j in 1:J
        (one(T) - m.survival[j]) == zero(T) && continue
        acc = zero(T)
        for ie in 1:n_e, i in 1:n_a
            acc += dist[i, ie, j] * a_pol[i, ie, j]
        end
        beq += (one(T) - m.survival[j]) * (one(T) + r) * acc
    end
    return beq / (one(T) + m.n_pop)
end

"""
    lifecycle_transition(m, k0; H=80, Z_path=nothing, ss=nothing, tol=1e-5,
                         max_iter=80, relax=0.5, verbose=false) -> LifeCycleTransition
    lifecycle_transition(m, Z_path; k0=nothing, ss=nothing, ...) -> LifeCycleTransition

Perfect-foresight transition of a life-cycle OLG economy.

The first form starts from a displaced aggregate capital `k0` (the stationary
age–productivity histogram, assets scaled by `k0 / K*`) and a constant TFP
path `Z_t = m.Z`, unless `Z_path` is supplied. The second form starts from the
stationary distribution (`k0 = K*` unless overridden) and a deterministic TFP
path that should return to `m.Z`.

The algorithm shoots on `{K_t}` (Auerbach–Kotlikoff / MIT):

1. Given `{K_t, Z_t}`, set `r_t = α Z_t (K_t/L)^{α−1} − δ` and
   `w_t = (1−α) Z_t (K_t/L)^α`.
2. Solve the household problem **backward** from the terminal stationary
   policy, one age-EGM sweep per date.
3. Push the initial histogram **forward** through those policies.
4. Relax `K_t` toward household asset supply until the path converges.

`K_0` is pinned by the initial distribution. `relax ∈ (0, 1]` is the damping
on the capital-path update. Non-convergence is reported in `converged`, never
silently accepted.

# Keyword Arguments
| Keyword | Default | Description |
|---|---|---|
| `H` | `80` | Horizon (`H+1` dates). Ignored when `Z_path` is given (`H = length(Z_path) − 1`) |
| `Z_path` | `nothing` | Optional TFP path of length `H+1` |
| `ss` | `nothing` | Precomputed [`LifeCycleSteadyState`](@ref); computed if omitted |
| `tol` | ``10^{-5}`` | Absolute tolerance on ``\\max_t |K_t^{\\mathrm{new}} − K_t|`` |
| `max_iter` | `80` | Shooting iterations |
| `relax` | `0.5` | Damping on the capital (and bequest) update |
| `verbose` | `false` | Print shooting diagnostics |
"""
function lifecycle_transition(m::LifeCycleOLG{T}, k0::Real;
                              Z_path::Union{Nothing,AbstractVector}=nothing,
                              H::Int=80,
                              ss::Union{Nothing,LifeCycleSteadyState{T}}=nothing,
                              tol::Real=1e-5, max_iter::Int=80,
                              relax::Real=0.5,
                              verbose::Bool=false) where {T<:AbstractFloat}
    ss0 = ss === nothing ? lifecycle_steady_state(m) : ss
    Z = if Z_path === nothing
        H >= 2 || throw(ArgumentError("`H` must be at least 2, got $H"))
        fill(m.Z, H + 1)
    else
        length(Z_path) >= 3 || throw(ArgumentError(
            "`Z_path` must have at least 3 points (got $(length(Z_path)))"))
        collect(T, Z_path)
    end
    return _lifecycle_transition(m, ss0, T(k0), Z; tol=tol, max_iter=max_iter,
                                 relax=relax, verbose=verbose)
end

function lifecycle_transition(m::LifeCycleOLG{T}, Z_path::AbstractVector;
                              k0::Union{Nothing,Real}=nothing,
                              ss::Union{Nothing,LifeCycleSteadyState{T}}=nothing,
                              tol::Real=1e-5, max_iter::Int=80,
                              relax::Real=0.5,
                              verbose::Bool=false) where {T<:AbstractFloat}
    ss0 = ss === nothing ? lifecycle_steady_state(m) : ss
    k = k0 === nothing ? ss0.K : T(k0)
    length(Z_path) >= 3 || throw(ArgumentError(
        "`Z_path` must have at least 3 points (got $(length(Z_path)))"))
    return _lifecycle_transition(m, ss0, k, collect(T, Z_path);
                                 tol=tol, max_iter=max_iter,
                                 relax=relax, verbose=verbose)
end

function _lifecycle_transition(m::LifeCycleOLG{T}, ss::LifeCycleSteadyState{T},
                               k0::T, Z::Vector{T};
                               tol::Real=1e-5, max_iter::Int=80,
                               relax::Real=0.5,
                               verbose::Bool=false) where {T<:AbstractFloat}
    k0 > zero(T) || throw(ArgumentError("`k0` must be positive, got $k0"))
    all(>(zero(T)), Z) || throw(ArgumentError("every TFP value must be strictly positive"))
    0 < relax <= 1 || throw(ArgumentError("`relax` must lie in (0, 1], got $relax"))
    max_iter >= 1 || throw(ArgumentError("`max_iter` must be at least 1"))
    ss.K > zero(T) || throw(ArgumentError("steady-state capital must be positive"))

    Np1 = length(Z)
    H = Np1 - 1
    a_grid = m.grid.grids[1]
    n_a = length(a_grid)
    n_e = length(m.income.states)
    mu = _lc_cohort_mass(m)
    L = _lc_labor(m, mu)
    L = max(L, T(1e-12))
    relax_T = T(relax)
    tol_T = T(tol)

    lambda = k0 / ss.K
    dist0 = _lc_rescale_dist(m, ss.dist, lambda)
    K0 = _lc_aggregate_K(a_grid, dist0)

    # Initial guess: linear bridge from K0 to the stationary capital.
    K = [K0 + (ss.K - K0) * T(t - 1) / T(H) for t in 1:Np1]
    K[1] = K0
    transfer = fill(ss.transfer, Np1)

    c_pol = zeros(T, n_a, n_e, m.J, Np1)
    a_pol = zeros(T, n_a, n_e, m.J, Np1)
    # Terminal continuation is the stationary policy.
    c_pol[:, :, :, Np1] .= ss.c_policy
    a_pol[:, :, :, Np1] .= ss.a_policy

    tau0, _ = _lc_ss_budget(m, ss.w, L, mu)

    function prices_at(t)
        Kt = max(K[t], T(1e-10))
        return _lc_prices(m, Kt / L, Z[t])
    end

    converged = false
    iters = 0
    dists = Vector{Array{T,3}}(undef, Np1)

    for outer in 1:max_iter
        iters = outer
        rpath = Vector{T}(undef, Np1)
        wpath = Vector{T}(undef, Np1)
        ppath = Vector{T}(undef, Np1)
        for t in 1:Np1
            rpath[t], wpath[t] = prices_at(t)
            _, ppath[t] = _lc_ss_budget(m, wpath[t], L, mu)
        end

        # Backward age-EGM from the terminal stationary policy.
        for t in (Np1 - 1):-1:1
            r_next = rpath[t+1]
            _lc_policies_at_date!(view(c_pol, :, :, :, t), view(a_pol, :, :, :, t),
                                  m, rpath[t], r_next, wpath[t], tau0, ppath[t],
                                  transfer[t], view(c_pol, :, :, :, t+1))
        end

        # Forward histogram from the displaced initial distribution.
        dists[1] = dist0
        for t in 1:H
            dists[t+1] = _lc_forward_dist(m, dists[t], view(a_pol, :, :, :, t))
        end

        K_new = [_lc_aggregate_K(a_grid, dists[t]) for t in 1:Np1]
        K_new[1] = K0
        transfer_new = if m.annuities
            transfer
        else
            [_lc_bequest(m, dists[t], view(a_pol, :, :, :, t), rpath[t]) for t in 1:Np1]
        end

        diffK = maximum(abs.(K_new .- K))
        diffB = m.annuities ? zero(T) : maximum(abs.(transfer_new .- transfer))
        verbose && println("lifecycle transition $outer: max|ΔK| = $diffK, max|Δbeq| = $diffB")

        if diffK < tol_T && diffB < tol_T
            K .= K_new
            transfer = transfer_new
            converged = true
            break
        end
        for t in 2:Np1
            K[t] = relax_T * K_new[t] + (one(T) - relax_T) * K[t]
        end
        if !m.annuities
            for t in 1:Np1
                transfer[t] = relax_T * transfer_new[t] + (one(T) - relax_T) * transfer[t]
            end
        end
    end

    # Final aggregates on the (possibly last-iterate) capital path.
    rpath = Vector{T}(undef, Np1)
    wpath = Vector{T}(undef, Np1)
    ppath = Vector{T}(undef, Np1)
    Ypath = Vector{T}(undef, Np1)
    Cpath = Vector{T}(undef, Np1)
    for t in 1:Np1
        rpath[t], wpath[t] = prices_at(t)
        _, ppath[t] = _lc_ss_budget(m, wpath[t], L, mu)
        Ypath[t] = Z[t] * max(K[t], T(1e-10))^m.alpha * L^(one(T) - m.alpha)
    end
    for t in (Np1 - 1):-1:1
        _lc_policies_at_date!(view(c_pol, :, :, :, t), view(a_pol, :, :, :, t),
                              m, rpath[t], rpath[t+1], wpath[t], tau0, ppath[t],
                              transfer[t], view(c_pol, :, :, :, t+1))
    end
    dists[1] = dist0
    Cpath[1] = _lc_aggregate_X(view(c_pol, :, :, :, 1), dists[1])
    for t in 1:H
        dists[t+1] = _lc_forward_dist(m, dists[t], view(a_pol, :, :, :, t))
        Cpath[t+1] = _lc_aggregate_X(view(c_pol, :, :, :, t+1), dists[t+1])
    end
    if !m.annuities
        transfer = [_lc_bequest(m, dists[t], view(a_pol, :, :, :, t), rpath[t]) for t in 1:Np1]
    end

    if !converged
        @warn "lifecycle_transition did not converge in $iters shooting iterations" maxlog=1
    end

    return LifeCycleTransition{T}(copy(K), rpath, wpath, Ypath, Cpath, copy(Z),
                                  ppath, copy(transfer), tau0, converged, iters, ss)
end

function Base.show(io::IO, m::LifeCycleOLG{T}) where {T}
    print(io, "LifeCycleOLG{$T}: J=", m.J, ", retire at ", m.J_retire,
          ", ", length(m.income.states), " income states, ",
          m.annuities ? "annuities" : "accidental bequests")
end

function Base.show(io::IO, ss::LifeCycleSteadyState{T}) where {T}
    print(io, "LifeCycleSteadyState{$T}: r=", round(ss.r; digits=5),
          ", K/Y=", round(ss.K / ss.Y; digits=3),
          ", converged=", ss.converged)
end

function Base.show(io::IO, tr::LifeCycleTransition{T}) where {T}
    print(io, "LifeCycleTransition{$T}: ", length(tr.K), " periods, K: ",
          round(tr.K[1]; digits=4), " → ", round(tr.K[end]; digits=4),
          ", converged=", tr.converged)
end

"""
    report(ss::LifeCycleSteadyState)

Print the stationary equilibrium: convergence, prices and aggregates, the social
security budget, and the age profiles of assets, consumption, and income.
"""
function report(ss::LifeCycleSteadyState{T}) where {T}
    io = stdout
    m = ss.spec
    head = Any[
        "Converged"              ss.converged ? "Yes" : "No";
        "Bisection iterations"   ss.iterations;
        "Excess capital supply"  _fmt(ss.excess_demand; digits=8);
        "Ages J"                 m.J;
        "Retirement age"         m.J_retire;
        "Annuities"              m.annuities ? "Yes" : "No (bequests rebated)"
    ]
    _pretty_table(io, head;
        title="Life-Cycle OLG Steady State",
        column_labels=["", "Value"],
        alignment=[:l, :r])

    agg = Any[
        "Interest rate r"     _fmt(ss.r; digits=6);
        "Wage w"              _fmt(ss.w; digits=6);
        "Capital K"           _fmt(ss.K; digits=6);
        "Labor L"             _fmt(ss.L; digits=6);
        "Output Y"            _fmt(ss.Y; digits=6);
        "K/Y"                 _fmt(ss.K / ss.Y; digits=6);
        "Payroll tax τ"       _fmt(ss.tau; digits=6);
        "Pension benefit"     _fmt(ss.pension; digits=6);
        "Bequest transfer"    _fmt(ss.transfer; digits=6)
    ]
    _pretty_table(io, agg;
        title="Prices and Aggregates",
        column_labels=["", "Value"],
        alignment=[:l, :r])

    # Age profiles, thinned so the table stays readable for J = 60+.
    step = max(cld(m.J, 12), 1)
    ages = collect(1:step:m.J)
    ages[end] == m.J || push!(ages, m.J)
    data = Matrix{Any}(undef, length(ages), 5)
    for (i, j) in enumerate(ages)
        data[i, 1] = j
        data[i, 2] = j < m.J_retire ? "work" : "retired"
        data[i, 3] = _fmt(ss.asset_profile[j]; digits=4)
        data[i, 4] = _fmt(ss.consumption_profile[j]; digits=4)
        data[i, 5] = _fmt(ss.income_profile[j]; digits=4)
    end
    _pretty_table(io, data;
        title="Age Profiles" * (step > 1 ? " (every $(step)th age)" : ""),
        column_labels=["Age", "Status", "Assets", "Consumption", "Income"],
        alignment=[:r, :l, :r, :r, :r])
    return nothing
end
