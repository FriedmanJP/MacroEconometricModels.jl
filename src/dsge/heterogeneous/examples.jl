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
- `:endogenous_labor` — Aiyagari with GHH endogenous labor supply

# References
- Krusell, P., & Smith, A. A. (1998). Income and wealth heterogeneity in the
  macroeconomy. *Journal of Political Economy*, 106(5), 867–896.
- Kaplan, G., Moll, B., & Violante, G. L. (2018). Monetary policy according to
  HANK. *American Economic Review*, 108(3), 697–743.
"""

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
# _unit_mean_lognormal_income — shared income discretization for the examples
# =============================================================================

"""
    _unit_mean_lognormal_income(rho, sd_log, n; method=:rouwenhorst) -> IncomeProcess{Float64}

Discretized log-AR(1) idiosyncratic income normalized to a unit-mean multiplier
`e = exp(z) / E[exp(z)]`, so `E[e] = 1` and every state gives strictly positive
labor income `w*e` (the raw grid is symmetric about 0 in logs, which would make
half the states negative if used directly).

`sd_log` is the **unconditional (cross-sectional) standard deviation of
`log e`**, not the innovation standard deviation — the `exp`/`E[exp]`
normalization is a pure location shift in logs, so `sd(log e) == sd_log`
exactly. This is the convention the calibration targets are quoted in and the
one the Python `sequence-jacobian` toolkit uses; the underlying
[`rouwenhorst`](@ref)/[`tauchen`](@ref) primitives default to the *innovation*
convention, so the call below passes `sigma_is=:unconditional` explicitly.
"""
function _unit_mean_lognormal_income(rho::Real, sd_log::Real, n::Int;
                                     method::Symbol=:rouwenhorst)
    raw = method === :tauchen ? tauchen(rho, sd_log, n; sigma_is=:unconditional) :
                                rouwenhorst(rho, sd_log, n; sigma_is=:unconditional)
    e = exp.(raw.states)
    e ./= dot(raw.stationary_dist, e)
    return IncomeProcess{Float64}(raw.transition, e, raw.stationary_dist, :income)
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
    get(prices, :div, 0.0) - get(prices, :tau, 0.0)
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

    # Income: log e is AR(1) with rho = 0.966 and UNCONDITIONAL sd(log e) = 0.5
    # — the Krusell-Smith / sequence-jacobian calibration target — normalized to
    # unit mean so every state gives strictly positive labor income w*e (#231).
    # NB 0.5 is the cross-sectional sd, NOT the innovation sd: passing it
    # positionally to `rouwenhorst` would give sd(log e) = 0.5/sqrt(1-0.966^2)
    # = 1.93, a 15x-too-dispersed income process.
    income = _unit_mean_lognormal_income(0.966, 0.5, 7)

    # Asset grid: [0, 1000] with 200 pivot-geometric points. a_max/K ~ 24 at the
    # equilibrium K, so the savings policy is interior on a set of full measure
    # and the ceiling never binds — required for the asset market to clear, since
    # the Young (2010) transition clamps a' at a_max. `:geometric` rather than
    # `:double_exp` because the latter's bottom spacing scales linearly with
    # a_max and would trade the ceiling problem for a resolution problem at the
    # borrowing constraint.
    grid = HAGrid(; assets=(0.0, 1000.0, 200), income_states=7,
                    grid_type=:geometric)

    # Individual problem
    individual = IndividualProblem{Float64}(u, up, upi, beta, _ks_budget,
                                            [0.0],    # borrowing constraint: a >= 0
                                            nothing,  # no adjustment cost
                                            1)        # one asset dimension

    aggregation = Pair{Symbol,Function}[:K => _agg_var1]
    het_params = Dict{Symbol,Float64}(
        :alpha => alpha, :delta => delta,
        :Z => 1.0, :L => 1.0, :rho_z => 0.95, :sigma_z => 0.007
    )
    hh = HouseholdSystem{Float64}(individual, income, grid, aggregation, het_params)
    return _wrap_ha_spec(hh;
        params=[:alpha, :delta, :rho_z, :sigma_z],
        param_values=Dict{Symbol,Float64}(:alpha => alpha, :delta => delta,
                                          :rho_z => 0.95, :sigma_z => 0.007))
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

    # Income: unconditional sd(log e) = 0.5, as in the KS example (see
    # `_unit_mean_lognormal_income` for the sd convention).
    income = _unit_mean_lognormal_income(0.966, 0.5, 7)

    # Asset grid: [-2, 1000] with 200 pivot-geometric points (allows borrowing).
    # beta = 0.986 here gives beta(1+r) closer to 1 than in the KS example, i.e.
    # a fatter right tail — so the ceiling has to be checked again if beta is
    # ever recalibrated. The shipped [-2, 50] left ~29% of mass on the ceiling.
    grid = HAGrid(; assets=(-2.0, 1000.0, 200), income_states=7,
                    grid_type=:geometric)

    # Individual problem — borrowing constraint b >= -2
    individual = IndividualProblem{Float64}(u, up, upi, beta, _hank1_budget,
                                            [-2.0],   # borrowing constraint
                                            nothing,  # no adjustment cost
                                            1)        # one asset dimension

    aggregation = Pair{Symbol,Function}[:K => _agg_var1]
    het_params = Dict{Symbol,Float64}(
        :alpha => alpha, :delta => delta, :sigma_c => sigma_c,
        :Z => 1.0, :L => 1.0, :rho_z => 0.95, :sigma_z => 0.007
    )
    hh = HouseholdSystem{Float64}(individual, income, grid, aggregation, het_params)
    return _wrap_ha_spec(hh;
        params=[:alpha, :delta, :rho_z, :sigma_z],
        param_values=Dict{Symbol,Float64}(:alpha => alpha, :delta => delta,
                                          :rho_z => 0.95, :sigma_z => 0.007))
end

# =============================================================================
# _two_asset_hank_example — Two-asset HANK
# =============================================================================

function _two_asset_hank_example(; n_liquid::Int=50, n_illiquid::Int=50,
                                   n_e::Int=7, B_supply::Real=2.0)
    # Calibration
    alpha   = 0.36
    beta    = 0.986
    delta   = 0.025
    sigma_c = 1.0

    # CRRA utility
    u, up, upi = _crra_utility(sigma_c)

    # Income: unconditional sd(log e) = 0.5, as in the KS example (see
    # `_unit_mean_lognormal_income` for the sd convention).
    income = _unit_mean_lognormal_income(0.966, 0.5, n_e)

    # Two-asset grid: liquid [-2, 50], illiquid [0, 100]. Production GE
    # (`compute_steady_state`) closes both markets; shrink the grids in tests.
    grid = HAGrid(; liquid=(-2.0, 50.0, n_liquid), illiquid=(0.0, 100.0, n_illiquid),
                    income_states=n_e)

    # Individual problem — liquid b >= -2, illiquid a >= 0
    individual = IndividualProblem{Float64}(u, up, upi, beta, _hank2_budget,
                                            [-2.0, 0.0],          # borrowing constraints
                                            _hank2_adjustment_cost, # portfolio adjustment cost
                                            2)                     # two asset dimensions

    aggregation = Pair{Symbol,Function}[
        :B => _agg_var1,
        :A => _agg_var2
    ]
    het_params = Dict{Symbol,Float64}(
        :alpha => alpha, :delta => delta, :sigma_c => sigma_c,
        :Z => 1.0, :L => 1.0, :B_supply => Float64(B_supply),
        :rho_z => 0.95, :sigma_z => 0.007
    )
    hh = HouseholdSystem{Float64}(individual, income, grid, aggregation, het_params;
                                  model=:two_asset)
    return _wrap_ha_spec(hh;
        params=[:alpha, :delta, :rho_z, :sigma_z],
        param_values=Dict{Symbol,Float64}(:alpha => alpha, :delta => delta,
                                          :rho_z => 0.95, :sigma_z => 0.007))
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

    aggregation = Pair{Symbol,Function}[:A => _agg_var1]
    het_params = Dict{Symbol,Float64}(
        :sigma_c => sigma, :beta_hh => beta, :credit_limit => credit_limit,
        :rho_e => rho_e, :sigma_e => sigma_e, :Z => 1.0, :L => 1.0,
        :rho_z => rho_e, :sigma_z => sigma_e
    )
    hh = HouseholdSystem{Float64}(individual, income, grid, aggregation, het_params;
                                  model=:huggett)
    return _wrap_ha_spec(hh;
        params=[:rho_z, :sigma_z],
        param_values=Dict{Symbol,Float64}(:rho_z => rho_e, :sigma_z => sigma_e,
                                          :alpha => 0.36, :delta => 0.025))
end

# =============================================================================
# load_ha_example — public API
# =============================================================================

"""
    load_ha_example(name::Symbol; distribution=:young) -> ModelSpec{Float64}

Return a pre-calibrated HA `ModelSpec` (one `HouseholdSystem`) for a canonical model.

Pass `distribution=:winberry` to represent the cross-sectional distribution by
the Winberry (2018) parametric moment family instead of the Young (2010)
histogram; everything else about the calibration is unchanged.

# Available models

| Symbol | Model | Reference |
|---|---|---|
| `:krusell_smith` | Incomplete markets, one asset | Krusell & Smith (1998) |
| `:one_asset_hank` | One-asset HANK | Kaplan, Moll & Violante (2018) |
| `:two_asset_hank` | Two-asset HANK with adjustment costs | Kaplan, Moll & Violante (2018) |
| `:huggett` | Pure-exchange risk-free bond, zero net supply | Huggett (1993) |
| `:endogenous_labor` | Aiyagari with GHH endogenous labor supply | Greenwood, Hercowitz & Huffman (1988) |

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
- Greenwood, J., Hercowitz, Z., & Huffman, G. W. (1988). Investment, capacity
  utilization, and the real business cycle. *American Economic Review*, 78(3), 402-417.
"""
function load_ha_example(name::Symbol; distribution::Symbol=:young)
    spec = if name === :krusell_smith
        _ks_example()
    elseif name === :one_asset_hank
        _one_asset_hank_example()
    elseif name === :two_asset_hank
        _two_asset_hank_example()
    elseif name === :huggett
        _huggett_example()
    elseif name === :endogenous_labor
        _endogenous_labor_example()
    else
        error("Unknown HA-DSGE example: :$name. Available: :krusell_smith, " *
              ":one_asset_hank, :two_asset_hank, :huggett, :endogenous_labor")
    end
    distribution === :young && return spec
    return _replace_household(spec; distribution=distribution)
end

# =============================================================================
# _endogenous_labor_example — Aiyagari with GHH endogenous labor supply
# =============================================================================

"""
    _endogenous_labor_example(; kind=:ghh, psi=3.0, frisch=0.5, a_max=2000.0, n_a=200)

Aiyagari economy with **endogenous labor supply**. Identical to
`_ks_example` except that households also choose hours.

`psi = 3.0` is chosen so aggregate labor in efficiency units is `L ≈ 1` at the
steady state, matching the `L = 1` normalization the exogenous-labor examples
impose — which makes the two directly comparable.

`a_max = 2000` rather than 1000: hours raise labor income, so the stationary
wealth distribution has a fatter right tail and the ceiling has to move with it
(see `ha_grid_diagnostics`).
"""
function _endogenous_labor_example(; kind::Symbol=:ghh, psi::Real=3.0,
                                     frisch::Real=0.5, a_max::Real=2000.0,
                                     n_a::Int=200)
    alpha = 0.36
    beta  = 0.99
    delta = 0.025

    u, up, upi = _crra_utility(1.0)

    # Same income process as the Krusell-Smith example: unconditional
    # sd(log e) = 0.5 (see `_unit_mean_lognormal_income` for the sd convention).
    income = _unit_mean_lognormal_income(0.966, 0.5, 7)

    grid = HAGrid(; assets=(0.0, Float64(a_max), n_a), income_states=7,
                    grid_type=:geometric)

    labor = LaborSupply(; kind=kind, psi=psi, frisch=frisch)
    individual = IndividualProblem{Float64}(u, up, upi, beta, _ks_budget,
                                            [0.0], nothing, 1; labor=labor)

    aggregation = Pair{Symbol,Function}[:K => _agg_var1]
    het_params = Dict{Symbol,Float64}(
        :alpha => alpha, :delta => delta, :Z => 1.0, :L => 1.0,
        :psi => Float64(psi), :frisch => Float64(frisch),
        :rho_z => 0.95, :sigma_z => 0.007
    )
    hh = HouseholdSystem{Float64}(individual, income, grid, aggregation, het_params)
    return _wrap_ha_spec(hh;
        params=[:alpha, :delta, :rho_z, :sigma_z],
        param_values=Dict{Symbol,Float64}(:alpha => alpha, :delta => delta,
                                          :rho_z => 0.95, :sigma_z => 0.007))
end
