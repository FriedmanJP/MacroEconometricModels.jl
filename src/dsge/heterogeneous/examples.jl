# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Built-in example configurations for heterogeneous agent DSGE models.

Provides pre-calibrated `HADSGESpec` specifications for canonical models:
- `:krusell_smith` — Krusell & Smith (1998) incomplete markets
- `:one_asset_hank` — one-asset HANK (Kaplan-Moll-Violante style)
- `:two_asset_hank` — two-asset HANK with portfolio adjustment costs

# References
- Krusell, P., & Smith, A. A. (1998). Income and wealth heterogeneity in the
  macroeconomy. *Journal of Political Economy*, 106(5), 867–896.
- Kaplan, G., Moll, B., & Violante, G. L. (2018). Monetary policy according to
  HANK. *American Economic Review*, 108(3), 697–743.
"""

# =============================================================================
# _minimal_agg_spec — lightweight DSGESpec placeholder for aggregate block
# =============================================================================

"""
    _minimal_agg_spec(; alpha, delta, rho_z, sigma_z) -> DSGESpec{Float64}

Construct a minimal DSGESpec representing a Cobb-Douglas aggregate block.

The individual-level steady state solver (`compute_steady_state`) does not evaluate
the aggregate DSGESpec equations directly -- it uses the price function instead.
This helper creates a lightweight placeholder that records parameter values and
variable names for the aggregate block.
"""
function _minimal_agg_spec(; alpha::Float64=0.36, delta::Float64=0.025,
                             rho_z::Float64=0.95, sigma_z::Float64=0.007)
    endog = [:Y, :K, :r, :w, :Z]
    exog = [:eps_Z]
    params = [:alpha, :delta, :rho_z, :sigma_z]
    param_values = Dict{Symbol,Float64}(:alpha => alpha, :delta => delta,
                                         :rho_z => rho_z, :sigma_z => sigma_z)

    # Placeholder equations (stored as Expr for metadata only)
    equations = [
        :(Y[t] - Z[t] * K[t-1]^alpha),
        :(r[t] - alpha * Z[t] * K[t-1]^(alpha-1) + delta),
        :(w[t] - (1 - alpha) * Z[t] * K[t-1]^alpha),
        :(K[t] - K[t-1]),
        :(Z[t] - rho_z * Z[t-1] - sigma_z * eps_Z[t])
    ]

    # Identity residual functions (not called during steady-state computation)
    dummy_fn = (y_t, y_lag, y_lead, eps, theta) -> 0.0
    residual_fns = [dummy_fn for _ in 1:5]

    n_expect = 0
    forward_indices = Int[]
    steady_state = ones(Float64, 5)  # placeholder

    return DSGESpec{Float64}(endog, exog, params, param_values, equations,
                              residual_fns, n_expect, forward_indices,
                              steady_state, nothing)
end

# =============================================================================
# _crra_utility — CRRA utility function and its derivatives
# =============================================================================

"""
    _crra_utility(sigma_c) -> (u, u_prime, u_prime_inv)

Return CRRA utility `u(c) = c^(1-sigma)/(1-sigma)` (or `log(c)` when sigma=1),
its marginal utility, and the inverse of marginal utility.
"""
function _crra_utility(sigma_c::Float64)
    if sigma_c ≈ 1.0
        u   = (c::Float64) -> log(max(c, 1e-15))
        up  = (c::Float64) -> 1.0 / max(c, 1e-15)
        upi = (m::Float64) -> 1.0 / max(m, 1e-15)
    else
        u   = (c::Float64) -> max(c, 1e-15)^(1.0 - sigma_c) / (1.0 - sigma_c)
        up  = (c::Float64) -> max(c, 1e-15)^(-sigma_c)
        upi = (m::Float64) -> max(m, 1e-15)^(-1.0 / sigma_c)
    end
    return u, up, upi
end

# =============================================================================
# Shared helper functions (named, to avoid closure-over-local issues in 1.12)
# =============================================================================

# Budget function for Krusell-Smith: c + a' = (1+r)*a + w*e
function _ks_budget(a::Float64, e::Float64, prices::Dict{Symbol,Float64})
    (1.0 + prices[:r]) * a + prices[:w] * e
end

# Budget function for one-asset HANK: c + b' = (1+r)*b + w*e + div
function _hank1_budget(b::Float64, e::Float64, prices::Dict{Symbol,Float64})
    (1.0 + prices[:r]) * b + prices[:w] * e + get(prices, :div, 0.0)
end

# Budget function for two-asset HANK: resources from liquid side
function _hank2_budget(b::Float64, a::Float64, e::Float64, prices::Dict{Symbol,Float64})
    (1.0 + get(prices, :r_b, prices[:r])) * b + prices[:w] * e +
    get(prices, :div, 0.0)
end

# Portfolio adjustment cost: chi(d, a) = 0.5 * |d / max(a, 0.01)|^2 * max(a, 0.01)
function _hank2_adjustment_cost(d::Float64, a::Float64)
    a_floor = max(a, 0.01)
    0.5 * (d / a_floor)^2 * a_floor
end

# Aggregation helpers
_agg_var1(dist, grid_arg) = _aggregate(dist, grid_arg; var_index=1)
_agg_var2(dist, grid_arg) = _aggregate(dist, grid_arg; var_index=2)

# =============================================================================
# _ks_example — Krusell & Smith (1998)
# =============================================================================

function _ks_example()
    # Calibration
    alpha = 0.36
    beta  = 0.99
    delta = 0.025

    # CRRA with sigma = 1 (log utility)
    u, up, upi = _crra_utility(1.0)

    # Income process: Rouwenhorst(0.966, 0.5, 7), normalized to a unit-mean
    # multiplier e = exp(z) / E[exp(z)] so that E[e] = 1 and every state gives
    # strictly positive labor income w*e (the raw grid is a symmetric log grid
    # about 0, which would make half the states have negative labor income).
    inc_raw = rouwenhorst(0.966, 0.5, 7)
    e_norm = exp.(inc_raw.states)
    e_norm ./= dot(inc_raw.stationary_dist, e_norm)
    income = IncomeProcess{Float64}(inc_raw.transition, e_norm,
                                    inc_raw.stationary_dist, :income)

    # Asset grid: [0, 200] with 200 points
    grid = HAGrid(; assets=(0.0, 200.0, 200), income_states=7)

    # Individual problem
    individual = IndividualProblem{Float64}(u, up, upi, beta, _ks_budget,
                                            [0.0],    # borrowing constraint: a >= 0
                                            nothing,  # no adjustment cost
                                            1)        # one asset dimension

    # Aggregate block
    agg_spec = _minimal_agg_spec(; alpha=alpha, delta=delta)

    # Aggregation: capital = integral of assets over distribution
    aggregation = Pair{Symbol,Function}[
        :K => _agg_var1
    ]

    # HA-specific parameters
    het_params = Dict{Symbol,Float64}(
        :alpha => alpha, :delta => delta,
        :Z => 1.0, :L => 1.0
    )

    return HADSGESpec{Float64}(agg_spec, individual, income, grid,
                                aggregation, het_params)
end

# =============================================================================
# _one_asset_hank_example — One-asset HANK
# =============================================================================

function _one_asset_hank_example()
    # Calibration
    alpha   = 0.36
    beta    = 0.986
    delta   = 0.025
    sigma_c = 1.0

    # CRRA utility
    u, up, upi = _crra_utility(sigma_c)

    # Income process: Rouwenhorst(0.966, 0.5, 7), normalized to a unit-mean
    # multiplier e = exp(z) / E[exp(z)] so that E[e] = 1 and every state gives
    # strictly positive labor income w*e (the raw grid is a symmetric log grid
    # about 0, which would make half the states have negative labor income).
    inc_raw = rouwenhorst(0.966, 0.5, 7)
    e_norm = exp.(inc_raw.states)
    e_norm ./= dot(inc_raw.stationary_dist, e_norm)
    income = IncomeProcess{Float64}(inc_raw.transition, e_norm,
                                    inc_raw.stationary_dist, :income)

    # Asset grid: [-2, 50] with 200 points (allows borrowing)
    grid = HAGrid(; assets=(-2.0, 50.0, 200), income_states=7)

    # Individual problem — borrowing constraint b >= -2
    individual = IndividualProblem{Float64}(u, up, upi, beta, _hank1_budget,
                                            [-2.0],   # borrowing constraint
                                            nothing,  # no adjustment cost
                                            1)        # one asset dimension

    # Aggregate block
    agg_spec = _minimal_agg_spec(; alpha=alpha, delta=delta)

    # Aggregation
    aggregation = Pair{Symbol,Function}[
        :K => _agg_var1
    ]

    # HA-specific parameters
    het_params = Dict{Symbol,Float64}(
        :alpha => alpha, :delta => delta, :sigma_c => sigma_c,
        :Z => 1.0, :L => 1.0
    )

    return HADSGESpec{Float64}(agg_spec, individual, income, grid,
                                aggregation, het_params)
end

# =============================================================================
# _two_asset_hank_example — Two-asset HANK
# =============================================================================

function _two_asset_hank_example()
    # Calibration
    alpha   = 0.36
    beta    = 0.986
    delta   = 0.025
    sigma_c = 1.0

    # CRRA utility
    u, up, upi = _crra_utility(sigma_c)

    # Income process: Rouwenhorst(0.966, 0.5, 7), normalized to a unit-mean
    # multiplier e = exp(z) / E[exp(z)] so that E[e] = 1 and every state gives
    # strictly positive labor income w*e (the raw grid is a symmetric log grid
    # about 0, which would make half the states have negative labor income).
    inc_raw = rouwenhorst(0.966, 0.5, 7)
    e_norm = exp.(inc_raw.states)
    e_norm ./= dot(inc_raw.stationary_dist, e_norm)
    income = IncomeProcess{Float64}(inc_raw.transition, e_norm,
                                    inc_raw.stationary_dist, :income)

    # Two-asset grid: liquid [-2, 50] (50 pts), illiquid [0, 100] (50 pts)
    grid = HAGrid(; liquid=(-2.0, 50.0, 50), illiquid=(0.0, 100.0, 50),
                    income_states=7)

    # Individual problem — liquid b >= -2, illiquid a >= 0
    individual = IndividualProblem{Float64}(u, up, upi, beta, _hank2_budget,
                                            [-2.0, 0.0],          # borrowing constraints
                                            _hank2_adjustment_cost, # portfolio adjustment cost
                                            2)                     # two asset dimensions

    # Aggregate block
    agg_spec = _minimal_agg_spec(; alpha=alpha, delta=delta)

    # Aggregation: liquid and illiquid separately
    aggregation = Pair{Symbol,Function}[
        :B => _agg_var1,
        :A => _agg_var2
    ]

    # HA-specific parameters
    het_params = Dict{Symbol,Float64}(
        :alpha => alpha, :delta => delta, :sigma_c => sigma_c,
        :Z => 1.0, :L => 1.0
    )

    return HADSGESpec{Float64}(agg_spec, individual, income, grid,
                                aggregation, het_params)
end

# =============================================================================
# _huggett_example — Huggett (1993)
# =============================================================================

"""
    _huggett_income() -> IncomeProcess{Float64}

Two-state endowment process from Huggett (1993): `e ∈ {e_h, e_l} = {1.0, 0.1}` with
transition `π(e_h|e_h) = 0.925`, `π(e_h|e_l) = 0.5` (state order `[e_h, e_l]`). The
stationary distribution puts mass `0.5/0.575 ≈ 0.870` on the high state, giving a mean
endowment of `≈ 0.883` (so six periods ≈ 5.3 = "one year's average endowment", matching
the paper's credit-limit normalization).
"""
function _huggett_income()
    states = [1.0, 0.1]
    P = [0.925 0.075; 0.5 0.5]
    pi_h = 0.5 / (0.5 + 0.075)
    stat = [pi_h, 1.0 - pi_h]
    return IncomeProcess{Float64}(P, states, stat, :endowment)
end

"""
    _huggett_example(; credit_limit=-2.0, a_max=4.0, n_a=300, sigma=1.5,
                       beta=0.99322, rho_e=0.90, sigma_e=0.01) -> HADSGESpec{Float64}

Huggett (1993) pure-exchange, risk-free-bond economy. Agents trade a one-period bond in
**zero net supply** (`∫a' dμ = 0`) subject to a credit limit `ā = credit_limit < 0`.
Income enters the budget as `w·e` with `w = 1` the (steady-state) aggregate endowment
level, so the household problem reuses `_ks_budget` and the equilibrium risk-free rate is
found by bisection (see `_huggett_clearing`). `rho_e`/`sigma_e` parameterize an aggregate
endowment shock used by the dynamic solvers (an extension; the 1993 paper has no
aggregate risk).

Calibration follows Huggett (1993): CRRA `σ = 1.5`, `β = 0.99322` (six model periods per
year, annual `β = 0.96`), endowment `{1.0, 0.1}`, credit limits `{-2,-4,-6,-8}`.
"""
function _huggett_example(; credit_limit::Float64=-2.0, a_max::Float64=4.0,
                            n_a::Int=300, sigma::Float64=1.5,
                            beta::Float64=0.99322, rho_e::Float64=0.90,
                            sigma_e::Float64=0.01)
    u, up, upi = _crra_utility(sigma)
    income = _huggett_income()
    grid = HAGrid(; assets=(credit_limit, a_max, n_a), income_states=2)

    # Pure-exchange household: budget c + a' = (1+r)*a + w*e (w = endowment level)
    individual = IndividualProblem{Float64}(u, up, upi, beta, _ks_budget,
                                            [credit_limit],  # borrowing/credit limit ā
                                            nothing, 1)

    # Minimal aggregate block: endowment-level AR(1) metadata for the dynamic solvers
    agg_spec = _minimal_agg_spec(; rho_z=rho_e, sigma_z=sigma_e)

    # Aggregation: net bond position A = ∫ a dμ (clears at 0)
    aggregation = Pair{Symbol,Function}[:A => _agg_var1]

    het_params = Dict{Symbol,Float64}(
        :sigma_c => sigma, :beta_hh => beta, :credit_limit => credit_limit,
        :rho_e => rho_e, :sigma_e => sigma_e, :Z => 1.0, :L => 1.0
    )

    return HADSGESpec{Float64}(agg_spec, individual, income, grid,
                                aggregation, het_params; model=:huggett)
end

# =============================================================================
# load_ha_example — public API
# =============================================================================

"""
    load_ha_example(name::Symbol) -> HADSGESpec{Float64}

Return a pre-calibrated `HADSGESpec` for a canonical heterogeneous agent model.

# Available models

| Symbol | Model | Reference |
|---|---|---|
| `:krusell_smith` | Incomplete markets, one asset | Krusell & Smith (1998) |
| `:one_asset_hank` | One-asset HANK | Kaplan, Moll & Violante (2018) |
| `:two_asset_hank` | Two-asset HANK with adjustment costs | Kaplan, Moll & Violante (2018) |
| `:huggett` | Pure-exchange risk-free bond, zero net supply | Huggett (1993) |

# Examples

```julia
spec = load_ha_example(:krusell_smith)
ss = compute_steady_state(spec; max_iter=100, tol=1e-4)
report(ss)
```

# References
- Krusell, P., & Smith, A. A. (1998). Income and wealth heterogeneity in the
  macroeconomy. *Journal of Political Economy*, 106(5), 867-896.
- Kaplan, G., Moll, B., & Violante, G. L. (2018). Monetary policy according to
  HANK. *American Economic Review*, 108(3), 697-743.
"""
function load_ha_example(name::Symbol)
    if name === :krusell_smith
        return _ks_example()
    elseif name === :one_asset_hank
        return _one_asset_hank_example()
    elseif name === :two_asset_hank
        return _two_asset_hank_example()
    elseif name === :huggett
        return _huggett_example()
    else
        error("Unknown HA-DSGE example: :$name. " *
              "Available: :krusell_smith, :one_asset_hank, :two_asset_hank, :huggett")
    end
end
