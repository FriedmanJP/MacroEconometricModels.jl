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
# Euler-equation accuracy (#508)
# =============================================================================

"""
    _euler_error_stats(c_pol, a_pol, ip, grid, income, prices; points=:midpoints)
        → NamedTuple

Euler-equation residuals for a solved one-asset household problem.

For each evaluation point `(a, e_j)` at which the borrowing constraint does not bind,
the residual is

    err = |1 − β(1+r) E[u'(c(a′, e′))] / u'(c(a, e_j))|

and the statistic returned is `log10` of the maximum and of the mean over the evaluated
points.

# `points`
- `:midpoints` (default) — evaluate at the cell midpoints `(aᵢ + aᵢ₊₁)/2`, interpolating
  both `c` and `a′`. This measures **approximation** error.
- `:nodes` — evaluate at the grid nodes. EGM *solves* the Euler equation there, so the
  residual only measures interpolation round-trip error and is optimistically small by
  construction. Retained for continuity with published numbers, not because it is
  informative.

# Excluded cells
Two classes of point carry no information and are counted rather than scored:

- **constrained** (`a′ ≤ a_min + 1e-6`) — the Euler equation holds with inequality there.
- **off-grid** (`a′ > a_max`) — `_linear_interp` flat-extrapolates above the grid, so the
  continuation consumption is looked up at exactly the clamped point the solver itself
  used and the residual collapses to machine precision. Scoring those cells makes a
  *truncating* model look more accurate precisely where it is broken.

Returns `(points, max, mean, n_evaluated, n_constrained, n_offgrid)`; `max` and `mean` are
`NaN` when nothing was evaluated.
"""
function _euler_error_stats(c_pol::Matrix{T}, a_pol::Matrix{T},
                            ip::IndividualProblem{T}, grid::HAGrid{T},
                            income::IncomeProcess{T},
                            prices::Dict{Symbol,T};
                            points::Symbol=:midpoints) where {T<:AbstractFloat}
    points in (:nodes, :midpoints) ||
        throw(ArgumentError("points must be :nodes or :midpoints; got :$points"))

    a_grid = grid.grids[1]
    n_a = length(a_grid)
    n_e = length(income.states)
    a_min = ip.borrowing_constraint[1]
    a_max = a_grid[end]

    beta = ip.beta
    u_prime = ip.utility_prime
    r = prices[:r]
    Pi = income.transition

    # Under GHH preferences marginal utility is U'(x) with x = c − v(n), not
    # U'(c): the Euler equation holds in the composite good. `shift[j]` is v(n_j),
    # which under GHH depends on the income state alone. Zero in every other case,
    # so the residual reduces to the standard one.
    shift = zeros(T, n_e)
    ls = ip.labor
    if ls !== nothing && ls.kind === :ghh
        w = prices[:w]
        for j in 1:n_e
            shift[j] = _labor_disutility(ls, labor_supply(ls, w * income.states[j]))
        end
    end

    eval_pts = points === :nodes ? a_grid :
               T[(a_grid[i] + a_grid[i+1]) / 2 for i in 1:(n_a-1)]

    max_err = zero(T)
    sum_err = zero(T)
    n_checked = 0
    n_constrained = 0
    n_offgrid = 0
    constraint_tol = a_min + T(1e-6)

    @inbounds for j in 1:n_e
        c_j = view(c_pol, :, j)
        a_j = view(a_pol, :, j)
        for (k, a_pt) in enumerate(eval_pts)
            # At a node the policy value IS the stored one; off-node it is the
            # interpolant, which is the whole point of the :midpoints metric.
            c_here = points === :nodes ? c_pol[k, j] : _linear_interp(a_grid, c_j, a_pt)
            a_next = points === :nodes ? a_pol[k, j] : _linear_interp(a_grid, a_j, a_pt)

            if a_next <= constraint_tol
                n_constrained += 1
                continue
            end
            if a_next > a_max
                n_offgrid += 1
                continue
            end

            emu = zero(T)
            for jp in 1:n_e
                c_tomorrow = _linear_interp(a_grid, view(c_pol, :, jp), a_next)
                c_tomorrow = max(c_tomorrow - shift[jp], T(1e-15))
                emu += Pi[j, jp] * u_prime(c_tomorrow)
            end

            up_today = u_prime(max(c_here - shift[j], T(1e-15)))
            if up_today > zero(T) && isfinite(emu)
                euler_resid = abs(one(T) - beta * (one(T) + r) * emu / up_today)
                euler_resid > max_err && (max_err = euler_resid)
                sum_err += euler_resid
                n_checked += 1
            end
        end
    end

    lg(x) = x > zero(T) ? log10(x) : T(-16)
    (points = points,
     max = n_checked == 0 ? T(NaN) : lg(max_err),
     mean = n_checked == 0 ? T(NaN) : lg(sum_err / T(n_checked)),
     n_evaluated = n_checked,
     n_constrained = n_constrained,
     n_offgrid = n_offgrid)
end

"""
    _compute_euler_error(c_pol, a_pol, ip, grid, income, prices; points=:midpoints) → T

Scalar `log10` maximum Euler residual — the `max` field of `_euler_error_stats`.
"""
_compute_euler_error(c_pol::Matrix{T}, a_pol::Matrix{T}, ip::IndividualProblem{T},
                     grid::HAGrid{T}, income::IncomeProcess{T},
                     prices::Dict{Symbol,T}; points::Symbol=:midpoints) where {T<:AbstractFloat} =
    _euler_error_stats(c_pol, a_pol, ip, grid, income, prices; points=points).max

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
                           clearing_fn::Union{Nothing,Function}=nothing,
                           distribution::Symbol=:young,
                           n_moments::Int=3,
                           n_quad::Int=4,
                           winberry_tol::Real=1e-9,
                           euler_points::Symbol=:midpoints,
                           hh_solver::Symbol=:egm) where {T<:AbstractFloat}
    euler_points in (:nodes, :midpoints) || throw(ArgumentError(
        "_ha_steady_state: euler_points must be :nodes or :midpoints, got :$euler_points."))
    hh_solver in (:egm, :vfi) || throw(ArgumentError(
        "_ha_steady_state: hh_solver must be :egm or :vfi, got :$hh_solver"))
    if hh_solver === :vfi && ip.labor !== nothing
        throw(ArgumentError(
            "compute_steady_state: hh_solver=:vfi does not support endogenous labor. " *
            "Use hh_solver=:egm, or drop labor from IndividualProblem."))
    end
    distribution in (:young, :winberry) || throw(ArgumentError(
        "_ha_steady_state: distribution must be :young or :winberry, got :$distribution."))
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
    has_labor = ip.labor !== nothing

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
    best_n_pol = has_labor ? zeros(T, n_a, n_e) : nothing
    best_V = hh_solver === :vfi ? zeros(T, n_a, n_e) : nothing
    best_L = get(params, :L, one(T))
    best_K_s = zero(T)
    best_K_d = zero(T)
    best_excess = T(Inf)
    converged = false
    final_iter = 0
    warm_c = nothing  # warm-start the inner EGM across bisection iterations (#238)

    # Evaluate excess demand K_s − K_d at a trial rate. A non-finite K_d (firm
    # demand diverges at a too-low rate) is mapped to excess = −∞ (raise r),
    # guarding the escalation path so a valid setup does not throw (#240/H-18).
    function eval_excess(r::T, warm, warm_V=nothing)
        p_loc = copy(params)
        K_d, prices = clr(r, p_loc)
        isfinite(K_d) || return (excess=T(-Inf), K_d=K_d, prices=prices,
                                 c_pol=nothing, a_pol=nothing, dist=nothing,
                                 K_s=T(NaN), n_pol=nothing, L=T(NaN),
                                 value_fn=nothing)
        local c_pol, a_pol, dist, K_s
        n_pol = nothing
        V_hh = nothing
        L_agg = get(p_loc, :L, one(T))
        # With endogenous labor, aggregate efficiency units are an outcome of the
        # household problem, so factor demand cannot be evaluated before it. Iterate
        # (solve → aggregate hours → re-price) to a fixed point. Under Cobb-Douglas
        # the wage depends on r alone — the firm FOC pins K/L, and both marginal
        # products are homogeneous of degree zero in it — so this converges on the
        # second pass; the loop is written generally in case a custom clearing rule
        # does make prices depend on L. With exogenous labor it runs exactly once
        # and the whole block reduces to the original code path.
        for _ in 1:(has_labor ? 30 : 1)
            if hh_solver === :vfi
                V_hh, c_pol, a_pol, _ = _vfi_solve(ip, grid, income, prices;
                                                   max_iter=1000, tol=T(1e-8),
                                                   howard_steps=20,
                                                   init_value=warm_V,
                                                   init_policy=warm)
            else
                c_pol, a_pol, _ = _egm_solve(ip, grid, income, prices;
                                             max_iter=1000, tol=T(1e-10),
                                             init_policy=warm)
            end
            Lambda = _build_transition_matrix(a_pol, grid, income)
            dist, _ = _stationary_dist_young(Lambda; max_iter=10_000, tol=T(1e-12))
            K_s = _aggregate(dist, grid; var_index=1)
            has_labor || break

            n_pol = labor_policy(ip, grid, income, prices, c_pol)
            L_new = _aggregate_labor(dist, n_pol, income, grid)
            p_loc[:L] = L_new
            K_d, prices = clr(r, p_loc)
            isfinite(K_d) || break
            converged_L = abs(L_new - L_agg) <= T(1e-12) * max(one(T), abs(L_new))
            L_agg = L_new
            warm = c_pol
            converged_L && break
        end
        return (excess=K_s - K_d, K_d=K_d, prices=prices,
                c_pol=c_pol, a_pol=a_pol, dist=dist, K_s=K_s,
                n_pol=n_pol, L=L_agg, value_fn=V_hh)
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
    # Never evaluate the household problem above the Aiyagari (1994) existence
    # bound: at β(1+r) ≥ 1 wealth diverges, the computed K_s is an artifact of the
    # grid ceiling, and (with endogenous labor) a household driven to zero
    # consumption supplies unbounded hours, so excess demand there is meaningless
    # — it can even come out NEGATIVE and destroy the bracket. Only ever lowers
    # r_hi, so a bracket that was already admissible is untouched.
    if r_hi > r_cap
        width = r_hi - r_lo
        r_hi = r_cap
        # A caller may legitimately supply an interval lying entirely above the
        # bound; drop r_lo with it so the bracket keeps its width and the
        # widening logic below can still find the true clearing rate.
        r_lo >= r_hi && (r_lo = r_hi - max(width, T(1e-3)))
    end
    res_lo = eval_excess(r_lo, nothing)
    res_hi = eval_excess(r_hi, res_lo.c_pol, res_lo.value_fn)
    widen = 0
    while res_lo.excess > zero(T) && widen < 60
        r_lo -= max(r_hi - r_lo, T(1e-3))          # expand downward
        res_lo = eval_excess(r_lo, res_lo.c_pol, res_lo.value_fn)
        widen += 1
    end
    widen = 0
    while res_hi.excess < zero(T) && r_hi < r_cap && widen < 60
        r_hi = min(r_hi + max(r_hi - r_lo, T(1e-3)), r_cap)   # expand upward, capped
        res_hi = eval_excess(r_hi, res_hi.c_pol === nothing ? res_lo.c_pol : res_hi.c_pol,
                            res_hi.value_fn === nothing ? res_lo.value_fn : res_hi.value_fn)
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
    warm_V = res_lo.value_fn !== nothing ? res_lo.value_fn :
             res_hi.value_fn

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

        res = eval_excess(r_mid, warm_c, warm_V)
        if res.c_pol === nothing
            # Demand diverges (e.g. r below the marginal-product floor) → raise r.
            r_lo = r_mid
            continue
        end
        warm_c = res.c_pol
        warm_V = res.value_fn
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
            best_L = res.L
            best_n_pol === nothing || copyto!(best_n_pol, res.n_pol)
            if best_V !== nothing && res.value_fn !== nothing
                copyto!(best_V, res.value_fn)
            end
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

    # Euler-equation accuracy (#508). Both conventions are measured — the off-node
    # statistic is what `euler_error` reports, the node statistic is kept alongside it
    # because it is what every published number for this package used to mean.
    euler_mid = _euler_error_stats(best_c_pol, best_a_pol, ip, grid, income, best_prices;
                                   points=:midpoints)
    euler_nodes = _euler_error_stats(best_c_pol, best_a_pol, ip, grid, income, best_prices;
                                     points=:nodes)
    euler_stats = (midpoints=euler_mid, nodes=euler_nodes)
    euler_err = euler_points === :nodes ? euler_nodes.max : euler_mid.max

    # Compute output: Cobb-Douglas for production economies, aggregate endowment otherwise
    if best_K_d > zero(T)
        # Labor is the REALIZED aggregate when it is endogenous, not params[:L].
        Y_val = get(params, :Z, one(T)) * best_K_d^(get(params, :alpha, T(0.36))) *
                best_L^(one(T) - get(params, :alpha, T(0.36)))
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
    best_n_pol === nothing || (policies[:labor] = best_n_pol)
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

    # Endogenous labor: `:L` is efficiency units ∫e·n dμ (what enters production)
    # and `:N` is mean hours ∫n dμ. They differ whenever income states are not 1.
    if best_n_pol !== nothing
        aggregates[:L] = best_L
        aggregates[:N] = _aggregate_hours(best_dist, best_n_pol, grid)
    end

    # Winberry (2018) parametric family (#356/T257). The equilibrium itself is
    # cleared on the Young histogram — the accurate reference — and the family is
    # fitted afterwards at the equilibrium policy, which is what the issue asks and
    # what keeps the two representations comparable. `M` is the fixed point of the
    # MOMENT law of motion, not the moments of `best_dist`: those are different
    # objects, and `aggregates[:K_winberry]` versus `aggregates[:K]` is exactly the
    # reduction's approximation error. The Young moments serve only as the starting
    # guess (the parametric map has spurious fixed points far from the ergodic set);
    # the returned point is verified stationary to `winberry_tol`.
    best_family = nothing
    if distribution === :winberry
        M0, _ = winberry_moments(best_dist, grid; n_moments=n_moments)
        stat = _winberry_stationary(best_a_pol, grid, income; n_moments=n_moments,
                                    n_quad=n_quad, M_init=M0, tol=winberry_tol)
        nodes_w, wts_w = winberry_quadrature(grid; n_quad=n_quad)
        best_family = _build_family(stat.moments, stat.mass, nodes_w, wts_w, grid;
                                    lambda_warm=stat.lambdas)
        aggregates[:K_winberry] = sum(stat.mass .* view(stat.moments, :, 1))
        stat.converged || @warn "compute_steady_state(distribution=:winberry): the " *
            "moment fixed point did not reach its tolerance; treat " *
            "aggregates[:K_winberry] and any :reiter solution built on it as " *
            "provisional." maxlog = 1
    end

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

    # VFI writes the Bellman V during the solve. EGM recovers V afterwards by
    # Howard policy evaluation of the equilibrium (c, a') policy.
    value_fn = if best_V !== nothing
        best_V
    else
        _policy_value_fn(best_c_pol, best_a_pol, ip, grid, income)
    end

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
        best_excess;
        parametric=best_family,
        euler=euler_stats
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
- `hh_solver::Symbol` — household solver: `:egm` (default) or `:vfi` (one-asset
  Bellman iteration; writes `ss.value_fn`). `:vfi` errors on two-asset models
  and on endogenous labor. Reiter/SSJ Jacobians stay on the EGM kernel.
- `price_fn::Function` — custom price function; if not supplied, uses Cobb-Douglas
- `distribution::Symbol` — override `spec.distribution`: `:young` (default, the
  Young 2010 histogram) or `:winberry` (Winberry 2018 parametric moment family).
  Under `:winberry` the equilibrium is still cleared on the histogram, and the
  parametric family is fitted afterwards at the equilibrium policy as the fixed
  point of the *moment* law of motion. It is returned in `ss.parametric`, and its
  own aggregate appears as `aggregates[:K_winberry]` — the gap against
  `aggregates[:K]` is the reduction's approximation error
- `n_moments::Int` — moments per income state under `:winberry` (default 3)
- `n_quad::Int` — Gauss–Legendre nodes per asset-grid interval (default 4)
- `winberry_tol::Real` — tolerance for the moment fixed point (default 1e-9, in
  standardized moment units)

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
                          clearing::Union{Nothing,Function}=nothing,
                          distribution::Union{Nothing,Symbol}=nothing,
                          n_moments::Int=3,
                          n_quad::Int=4,
                          winberry_tol::Real=1e-9,
                          euler_points::Symbol=:midpoints,
                          hh_solver::Symbol=:egm) where {T<:AbstractFloat}
    hh_solver in (:egm, :vfi) || throw(ArgumentError(
        "compute_steady_state: hh_solver must be :egm or :vfi, got :$hh_solver"))
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
        verbose=verbose, clearing_fn=clr,
        distribution=isnothing(distribution) ? spec.distribution : distribution,
        n_moments=n_moments, n_quad=n_quad, winberry_tol=winberry_tol,
        euler_points=euler_points, hh_solver=hh_solver
    )
end
