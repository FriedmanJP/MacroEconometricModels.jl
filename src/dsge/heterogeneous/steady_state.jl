# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Steady state solver for heterogeneous agent models.

Finds the stationary equilibrium by iterating: (1) solve the individual problem
via EGM, (2) compute the stationary distribution, (3) check market clearing,
(4) update prices via bisection on the interest rate.

# References
- Aiyagari, S. R. (1994). Uninsured idiosyncratic risk and aggregate saving.
  *Quarterly Journal of Economics*, 109(3), 659–684.
- Carroll, C. D. (2006). The method of endogenous gridpoints for solving dynamic
  stochastic optimization problems. *Economics Letters*, 91(3), 312–320.
- Young, E. R. (2010). Solving the incomplete markets model with aggregate
  uncertainty using the Krusell–Smith algorithm and non-stochastic simulations.
  *Journal of Economic Dynamics and Control*, 34(1), 36–41.
"""

# =============================================================================
# _compute_euler_error — max Euler equation residual
# =============================================================================

"""
    _compute_euler_error(c_pol, a_pol, ip, grid, income, prices) → T

Compute the maximum Euler equation error (in log10 units) at unconstrained
grid points.

For each (a_i, e_j) where the borrowing constraint does not bind
(a'(a_i, e_j) > a_min + ε), the Euler residual is:

    err_ij = |1 − β(1+r) E[u'(c(a', e'))] / u'(c(a_i, e_j))|

Returns `log10(max err_ij)` over unconstrained points. If no unconstrained
points exist, returns `NaN`.
"""
function _compute_euler_error(c_pol::Matrix{T}, a_pol::Matrix{T},
                               ip::IndividualProblem{T}, grid::HAGrid{T},
                               income::IncomeProcess{T},
                               prices::Dict{Symbol,T}) where {T<:AbstractFloat}
    a_grid = grid.grids[1]
    n_a = length(a_grid)
    n_e = length(income.states)
    a_min = ip.borrowing_constraint[1]

    beta = ip.beta
    u_prime = ip.utility_prime
    r = prices[:r]

    Pi = income.transition

    max_err = zero(T)
    n_checked = 0
    constraint_tol = a_min + T(1e-6)

    @inbounds for j in 1:n_e
        for i in 1:n_a
            # Skip constrained points
            if a_pol[i, j] <= constraint_tol
                continue
            end

            # Expected marginal utility at (a', e')
            emu = zero(T)
            for jp in 1:n_e
                c_tomorrow = _linear_interp(a_grid, view(c_pol, :, jp), a_pol[i, j])
                c_tomorrow = max(c_tomorrow, T(1e-15))
                emu += Pi[j, jp] * u_prime(c_tomorrow)
            end

            # Euler residual
            up_today = u_prime(c_pol[i, j])
            if up_today > zero(T) && isfinite(emu)
                euler_resid = abs(one(T) - beta * (one(T) + r) * emu / up_today)
                if euler_resid > max_err
                    max_err = euler_resid
                end
                n_checked += 1
            end
        end
    end

    if n_checked == 0
        return T(NaN)
    end

    # Return in log10 units
    return max_err > zero(T) ? log10(max_err) : T(-16)
end

# =============================================================================
# _ha_steady_state — bisection on interest rate
# =============================================================================

"""
    _ha_steady_state(ip, grid, income, price_fn, params; K_init, r_bounds, max_iter=200, tol=1e-8, verbose=false)
        → HASteadyState{T}

Find the stationary equilibrium of an Aiyagari (1994) economy by bisecting on
the interest rate until capital supply (from household savings) equals capital
demand (from firm FOC).

# Algorithm
1. Bisect on interest rate r:
   a. `r_mid = (r_lo + r_hi) / 2`
   b. Compute capital demand `K_d` from `price_fn` given `r_mid`
   c. Solve individual problem via EGM: `_egm_solve(ip, grid, income, prices)`
   d. Build transition matrix: `_build_transition_matrix(a_pol, grid, income)`
   e. Compute stationary distribution: `_stationary_dist_young(Lambda)`
   f. Compute capital supply: `K_s = _aggregate(dist, grid; var_index=1)`
   g. Excess demand = `K_s − K_d`
   h. If excess > 0 → r too high → `r_hi = r_mid`; else → `r_lo = r_mid`
2. Converge when `|K_s − K_d| < tol`

# Arguments
- `ip::IndividualProblem{T}` — household problem
- `grid::HAGrid{T}` — asset grid
- `income::IncomeProcess{T}` — income process
- `price_fn::Function` — `(K, params) → Dict{Symbol,T}` mapping capital to prices
- `params::Dict{Symbol,T}` — model parameters (e.g., `:alpha`, `:delta`, `:Z`, `:L`)
- `K_init::T` — initial guess for aggregate capital (used for logging only)
- `r_bounds::Tuple{T,T}` — `(r_low, r_high)` bounds for bisection
- `max_iter::Int` — maximum bisection iterations (default 200)
- `tol` — convergence tolerance on `|K_s − K_d|` (default 1e-8)
- `verbose::Bool` — print iteration progress (default false)
"""
function _ha_steady_state(ip::IndividualProblem{T}, grid::HAGrid{T},
                           income::IncomeProcess{T}, price_fn::Function,
                           params::Dict{Symbol,T};
                           K_init::T=T(10),
                           r_bounds::Tuple{T,T}=(T(-0.01), T(0.04)),
                           max_iter::Int=200,
                           tol::Real=T(1e-8),
                           rtol::Real=T(1e-8),
                           r_atol::Real=T(1e-13),
                           grid_check::Symbol=:none,
                           ceiling_mass_tol::Real=T(1e-6),
                           residual_tol::Real=T(1e-6),
                           verbose::Bool=false,
                           clearing_fn::Union{Nothing,Function}=nothing) where {T<:AbstractFloat}
    grid.n_dims == 1 || throw(ArgumentError(
        "compute_steady_state: the bisection steady-state solver supports one-asset " *
        "models only (got n_dims = $(grid.n_dims)). Two-asset models such as " *
        "load_ha_example(:two_asset_hank) require a two-dimensional market-clearing " *
        "solve, which is not implemented — see docs/src/dsge_ha.md."))
    ip.n_asset_dims == 1 || throw(ArgumentError(
        "compute_steady_state: the bisection steady-state solver supports one-asset " *
        "individual problems only (got n_asset_dims = $(ip.n_asset_dims))."))

    tol_T = T(tol)
    r_lo, r_hi = r_bounds

    # Market-clearing closure: given a trial rate, return (asset demand, prices).
    # Defaults to the Aiyagari firm-FOC rule built from `price_fn`, preserving the
    # original behavior exactly when no explicit closure is supplied.
    clr = isnothing(clearing_fn) ? _aiyagari_clearing(price_fn) : clearing_fn

    # Validate bounds. excess(r) = K_s(r) − K_d(r) is INCREASING in r (a higher
    # rate raises household saving and lowers firm capital demand), so a valid
    # bracket needs excess(r_lo) ≤ 0 ≤ excess(r_hi). The old code only checked
    # r_lo < r_hi and then returned the closest midpoint even when the interval
    # bracketed no root — a spurious rate (#240/H-18). The sign-change check is
    # done below, after the excess closure is defined.
    @assert r_lo < r_hi "r_bounds must satisfy r_lo < r_hi"

    n_a = grid.n_points[1]
    n_e = grid.n_income

    # Storage for final results
    best_c_pol = zeros(T, n_a, n_e)
    best_a_pol = zeros(T, n_a, n_e)
    best_dist = zeros(T, n_a * n_e)
    best_prices = Dict{Symbol,T}()
    best_K_s = zero(T)
    best_K_d = zero(T)
    best_excess = T(Inf)
    converged = false
    final_iter = 0
    warm_c = nothing  # warm-start the inner EGM across bisection iterations (#238)

    # Evaluate excess demand K_s − K_d at a trial rate. A non-finite K_d (firm
    # demand diverges at a too-low rate) is mapped to excess = −∞ (raise r),
    # guarding the escalation path so a valid setup does not throw (#240/H-18).
    function eval_excess(r::T, warm)
        K_d, prices = clr(r, params)
        isfinite(K_d) || return (excess=T(-Inf), K_d=K_d, prices=prices,
                                 c_pol=nothing, a_pol=nothing, dist=nothing, K_s=T(NaN))
        c_pol, a_pol, _ = _egm_solve(ip, grid, income, prices;
                                     max_iter=1000, tol=T(1e-10), init_policy=warm)
        Lambda = _build_transition_matrix(a_pol, grid, income)
        dist, _ = _stationary_dist_young(Lambda; max_iter=10_000, tol=T(1e-12))
        K_s = _aggregate(dist, grid; var_index=1)
        return (excess=K_s - K_d, K_d=K_d, prices=prices,
                c_pol=c_pol, a_pol=a_pol, dist=dist, K_s=K_s)
    end

    # Bracket check + bounded widening (#240/H-18). excess(r) = K_s − K_d is
    # increasing in r, so a market-clearing rate needs excess(r_lo) ≤ 0 ≤
    # excess(r_hi). The old code only asserted r_lo < r_hi and then returned the
    # closest midpoint even when the interval bracketed no root (a spurious rate).
    # If the supplied interval does not bracket, EXPAND it — downward for a
    # too-high r_lo, upward (capped just below 1/β−1, where household saving
    # diverges) for a too-low r_hi — before giving up. This finds the true
    # equilibrium of a valid model whose rate lies outside the default bounds
    # rather than throwing or returning a spurious midpoint.
    r_cap = one(T) / ip.beta - one(T) - T(1e-6)
    res_lo = eval_excess(r_lo, nothing)
    res_hi = eval_excess(r_hi, res_lo.c_pol)
    widen = 0
    while res_lo.excess > zero(T) && widen < 60
        r_lo -= max(r_hi - r_lo, T(1e-3))          # expand downward
        res_lo = eval_excess(r_lo, res_lo.c_pol)
        widen += 1
    end
    widen = 0
    while res_hi.excess < zero(T) && r_hi < r_cap && widen < 60
        r_hi = min(r_hi + max(r_hi - r_lo, T(1e-3)), r_cap)   # expand upward, capped
        res_hi = eval_excess(r_hi, res_hi.c_pol === nothing ? res_lo.c_pol : res_hi.c_pol)
        widen += 1
    end
    if !(res_lo.excess <= zero(T) <= res_hi.excess)
        error("_ha_steady_state: could not bracket a market-clearing rate after " *
              "widening r_bounds to ($r_lo, $r_hi) — excess demand K_s − K_d does " *
              "not change sign (excess(r_lo) = $(res_lo.excess), " *
              "excess(r_hi) = $(res_hi.excess)). The model may admit no interior " *
              "stationary equilibrium.")
    end
    warm_c = res_lo.c_pol !== nothing ? res_lo.c_pol : res_hi.c_pol

    for iter in 1:max_iter
        final_iter = iter

        # Bisection midpoint
        r_mid = (r_lo + r_hi) / T(2)

        # Aiyagari (1994) existence: with β(1+r) ≥ 1 an infinite-horizon household's
        # wealth diverges, so no stationary distribution exists and such a rate can
        # never clear. Solving there anyway yields a K_s that is a pure artifact of
        # the grid ceiling (every household saves to a_max), and — worse — that
        # divergent policy then propagates as the EGM warm start (#238), corrupting
        # the excess-demand evaluations at the *next*, admissible rates and letting
        # the bracket collapse on a spurious sign change. Shrink the bracket without
        # solving and leave `warm_c` untouched.
        if ip.beta * (one(T) + r_mid) >= one(T)
            r_hi = r_mid
            r_hi - r_lo <= T(r_atol) && break
            continue
        end

        res = eval_excess(r_mid, warm_c)
        if res.c_pol === nothing
            # Demand diverges (e.g. r below the marginal-product floor) → raise r.
            r_lo = r_mid
            continue
        end
        warm_c = res.c_pol
        excess = res.excess

        _bisect_msg = "Bisection iter $iter: r = $(round(r_mid; digits=6)), " *
                      "K_s = $(round(res.K_s; digits=4)), K_d = $(round(res.K_d; digits=4)), " *
                      "excess = $(round(excess; digits=6))"
        if verbose
            @info _bisect_msg
        else
            @debug _bisect_msg
        end

        # Store best solution
        if abs(excess) < abs(best_excess)
            best_excess = excess
            copyto!(best_c_pol, res.c_pol)
            copyto!(best_a_pol, res.a_pol)
            copyto!(best_dist, res.dist)
            best_prices = copy(res.prices)
            best_K_s = res.K_s
            best_K_d = res.K_d
        end

        # Check convergence. The threshold is scale-free: an absolute tolerance on
        # K_s − K_d is unmeetable for an economy whose capital stock is O(10-100),
        # because the residual floor is set by the discreteness of the asset grid
        # (~1e-8 absolute), not by the bisection. Scaling by |K_d| makes the same
        # tolerance mean the same thing at any calibration.
        if abs(excess) <= max(tol_T, T(rtol) * max(one(T), abs(res.K_d)))
            converged = true
            break
        end

        # Bisection update: excess is increasing in r, so excess > 0 ⇒ r too high.
        if excess > zero(T)
            r_hi = r_mid
        else
            r_lo = r_mid
        end

        # The bracket has collapsed to floating-point width — further bisection
        # cannot move r, so iterating to max_iter only burns solves.
        r_hi - r_lo <= T(r_atol) && break
    end

    # Compute Euler equation error
    euler_err = _compute_euler_error(best_c_pol, best_a_pol, ip, grid, income, best_prices)

    # Compute output: Cobb-Douglas for production economies, aggregate endowment otherwise
    if best_K_d > zero(T)
        Y_val = get(params, :Z, one(T)) * best_K_d^(get(params, :alpha, T(0.36))) *
                get(params, :L, one(T))^(one(T) - get(params, :alpha, T(0.36)))
    else
        # Pure-exchange (e.g. Huggett): Y = aggregate endowment Σ_j p_j e_j
        inc_marg = vec(sum(reshape(best_dist, n_a, n_e), dims=1))
        Y_val = sum(inc_marg .* income.states)
    end

    # Reshape distribution to N_a × N_e
    dist_reshaped = reshape(best_dist, n_a, n_e)

    # Build result
    policies = Dict{Symbol,Array{T}}(
        :savings => best_a_pol,
        :consumption => best_c_pol
    )
    # Grid adequacy. Computed once, after the loop — never inside `eval_excess`,
    # which runs 30-200 times. `:K` keeps its established meaning (∫a dμ, the
    # aggregate the bisection clears on); `:A_policy` is what the policy actually
    # implies (∫a′ dμ, the aggregate the sequence-space household block reports),
    # and the two coincide exactly iff the grid does not truncate.
    gdiag = _ha_grid_diagnostics(best_a_pol, best_dist, grid;
                                 ceiling_mass_tol=ceiling_mass_tol,
                                 residual_tol=residual_tol)

    aggregates = Dict{Symbol,T}(
        :K => best_K_s,
        :K_demand => best_K_d,
        :Y => Y_val,
        :excess_demand => best_excess,
        :A_policy => gdiag.assets_desired,
        :A_residual => gdiag.clearing_residual
    )

    # Aiyagari (1994) existence condition. With β(1+r) ≥ 1 an infinite-horizon
    # household's wealth diverges, no stationary distribution exists, and NO
    # finite a_max can fix it — a distinct failure from a merely-too-small grid.
    bR = ip.beta * (one(T) + best_prices[:r])
    if bR >= one(T) - T(1e-10)
        @warn "beta*(1+r) = $bR ≥ 1 at the computed steady state: household wealth " *
              "diverges, so no stationary distribution exists and no finite a_max " *
              "will produce one. Check beta, the clearing rule, or r_bounds." maxlog = 1
    end

    grid_check === :none ||
        _check_grid_adequacy(gdiag, grid_check; context="compute_steady_state")

    value_fn = zeros(T, n_a, n_e)  # EGM does not produce a value function

    return HASteadyState{T}(
        policies,
        dist_reshaped,
        value_fn,
        best_prices,
        aggregates,
        grid,
        income,
        converged,
        final_iter,
        euler_err,
        best_excess
    )
end

# =============================================================================
# _default_cobb_douglas_price_fn — standard neoclassical price function
# =============================================================================

"""
    _default_cobb_douglas_price_fn(K, params) → Dict{Symbol,T}

Compute competitive factor prices from a Cobb-Douglas production function:

    Y = Z K^α L^{1−α}
    r = α Z K^{α−1} L^{1−α} − δ
    w = (1−α) Z K^α L^{−α}

Requires `params` to contain `:alpha`, `:delta`, `:Z`, `:L`.
"""
function _default_cobb_douglas_price_fn(K::T, params::Dict{Symbol,T}) where {T<:AbstractFloat}
    alpha = params[:alpha]
    delta = params[:delta]
    Z = params[:Z]
    L = params[:L]

    r = alpha * Z * K^(alpha - one(T)) * L^(one(T) - alpha) - delta
    w = (one(T) - alpha) * Z * K^alpha * L^(-alpha)

    return Dict{Symbol,T}(:r => r, :w => w)
end

# =============================================================================
# Market-clearing closures — pluggable (asset demand, prices) given a trial rate
# =============================================================================

"""
    _aiyagari_clearing(price_fn) → (r_mid, params) -> (K_d, prices)

Default Aiyagari (1994) clearing rule. Inverts the Cobb-Douglas firm FOC to obtain
capital demand `K_d` at the trial rate, then evaluates `price_fn(K_d, params)` and
overwrites `prices[:r]` with the trial rate. Returns `(Inf, …)` when the rate falls
below the marginal-product floor so the caller raises `r`. Numerically identical to
the original hard-coded steady-state logic.
"""
function _aiyagari_clearing(price_fn::Function)
    return function (r_mid::T, params::Dict{Symbol,T}) where {T<:AbstractFloat}
        alpha = get(params, :alpha, T(0.36))
        delta = get(params, :delta, T(0.025))
        Z = get(params, :Z, one(T))
        L = get(params, :L, one(T))
        r_eff = r_mid + delta
        r_eff <= zero(T) && return (T(Inf), Dict{Symbol,T}(:r => r_mid))
        K_d = (r_eff / (alpha * Z * L^(one(T) - alpha)))^(one(T) / (alpha - one(T)))
        prices = price_fn(K_d, params)
        prices[:r] = r_mid
        return (K_d, prices)
    end
end

"""
    _huggett_clearing() → (r_mid, params) -> (0, Dict(:r, :w))

Huggett (1993) zero-net-supply clearing rule for a pure-exchange risk-free-bond
economy: asset demand is identically zero, so the bisection clears `∫a dμ = 0`
(the bisection always measures supply as `_aggregate(dist, grid)`, i.e. holdings
at grid nodes, not the raw policy `∫a′ dμ`). The two coincide here because the
Huggett grid never truncates the policy — see [`ha_grid_diagnostics`](@ref) for
what happens when it does. The aggregate endowment level `w` is fixed at 1 in
steady state (income enters the budget as `w·e`).
"""
function _huggett_clearing()
    return function (r_mid::T, params::Dict{Symbol,T}) where {T<:AbstractFloat}
        return (zero(T), Dict{Symbol,T}(:r => r_mid, :w => one(T)))
    end
end

# =============================================================================
# compute_steady_state — public API (dispatch on HADSGESpec)
# =============================================================================

"""
    compute_steady_state(spec::HADSGESpec{T}; kwargs...) → HASteadyState{T}

Compute the stationary equilibrium of a heterogeneous agent DSGE model.

Extracts the individual problem, grid, income process, and parameters from
`spec`, constructs a default Cobb-Douglas price function if the aggregate block
does not provide one, and delegates to `_ha_steady_state`.

# Keyword Arguments
- `K_init::T` — initial capital guess (default 10.0)
- `r_bounds::Tuple{T,T}` — bisection bounds for r (default (-0.01, 0.04))
- `max_iter::Int` — maximum iterations (default 200)
- `tol` — absolute convergence tolerance on `|K_s − K_d|` (default 1e-8)
- `rtol` — relative convergence tolerance (default 1e-8); the effective threshold
  is `max(tol, rtol * max(1, |K_d|))`, so it means the same thing at any scale
- `r_atol` — stop once the bisection bracket is narrower than this (default 1e-13)
- `grid_check::Symbol` — `:warn` (default), `:none` or `:error`. Checks whether the
  stationary distribution has run into the top of the asset grid; see
  [`ha_grid_diagnostics`](@ref)
- `ceiling_mass_tol` / `residual_tol` — thresholds for that check (default 1e-6)
- `verbose::Bool` — print progress (default false)
- `price_fn::Function` — custom price function; if not supplied, uses Cobb-Douglas

# Aggregates

`aggregates[:K]` is `∫ a dμ`, the aggregate the bisection clears on.
`aggregates[:A_policy]` is `∫ a′ dμ`, the aggregate the savings policy implies —
which is what the sequence-space household block reports. They are equal iff the
asset grid never truncates the policy; `aggregates[:A_residual]` is the
difference. See [`ha_grid_diagnostics`](@ref).
"""
function compute_steady_state(spec::HADSGESpec{T};
                          K_init::T=T(10),
                          r_bounds::Union{Nothing,Tuple{T,T}}=nothing,
                          max_iter::Int=200,
                          tol::Real=T(1e-8),
                          rtol::Real=T(1e-8),
                          r_atol::Real=T(1e-13),
                          grid_check::Symbol=:warn,
                          ceiling_mass_tol::Real=T(1e-6),
                          residual_tol::Real=T(1e-6),
                          verbose::Bool=false,
                          price_fn::Union{Nothing,Function}=nothing,
                          clearing::Union{Nothing,Function}=nothing) where {T<:AbstractFloat}
    pfn = isnothing(price_fn) ? _default_cobb_douglas_price_fn : price_fn

    # Extract parameters: merge het_params with aggregate steady-state params
    params = copy(spec.het_params)

    # Ensure essential parameters exist with sensible defaults
    if !haskey(params, :alpha)
        params[:alpha] = T(0.36)
    end
    if !haskey(params, :delta)
        params[:delta] = T(0.025)
    end
    if !haskey(params, :Z)
        params[:Z] = one(T)
    end
    if !haskey(params, :L)
        params[:L] = one(T)
    end

    # Select the market-clearing closure: explicit override > model family > Aiyagari.
    clr = !isnothing(clearing) ? clearing :
          spec.model === :huggett ? _huggett_clearing() :
          _aiyagari_clearing(pfn)

    # Select bisection bounds: explicit override > model-appropriate default.
    # Huggett's risk-free rate lies below the per-period time-preference rate 1/β − 1.
    rb = !isnothing(r_bounds) ? r_bounds :
         spec.model === :huggett ?
            (T(-0.05), one(T) / spec.individual.beta - one(T) - T(1e-4)) :
            (T(-0.01), T(0.04))

    return _ha_steady_state(
        spec.individual, spec.grid, spec.income, pfn, params;
        K_init=K_init, r_bounds=rb, max_iter=max_iter,
        tol=tol, rtol=rtol, r_atol=r_atol, grid_check=grid_check,
        ceiling_mass_tol=ceiling_mass_tol, residual_tol=residual_tol,
        verbose=verbose, clearing_fn=clr
    )
end
