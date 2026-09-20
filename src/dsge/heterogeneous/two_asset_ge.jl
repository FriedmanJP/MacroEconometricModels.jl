# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Discrete-time two-asset stationary equilibrium.

Closes the liquid and illiquid markets simultaneously, following the same
`(K, r_b)` damped iteration as [`ct_two_asset_ge`](@ref): illiquid wealth
clears against firm capital and liquid wealth against a fixed bond supply.
The inner household block is nested EGM or two-asset Bellman VFI.

# References
- Kaplan, G., Moll, B., & Violante, G. L. (2018). Monetary Policy According
  to HANK. *American Economic Review*, 108(3), 697–743.
- Young, E. R. (2010). Solving the incomplete markets model with aggregate
  uncertainty using the Krusell–Smith algorithm and non-stochastic simulations.
  *Journal of Economic Dynamics and Control*, 34(1), 36–41.
"""

# =============================================================================
# Illiquid law of motion and liquid Euler residual
# =============================================================================

"""
    _two_asset_a_prime(d_pol, grid, r_a) → Array{T,3}

Next-period illiquid holdings `a' = (1 + r_a) a + d` on the joint grid.
"""
function _two_asset_a_prime(d_pol::AbstractArray{T,3}, grid::HAGrid{T},
                            r_a::T) where {T<:AbstractFloat}
    a_grid = grid.grids[2]
    n_b, n_a, n_e = size(d_pol)
    a_next = similar(d_pol)
    @inbounds for je in 1:n_e, ia in 1:n_a, ib in 1:n_b
        a_next[ib, ia, je] = (one(T) + r_a) * a_grid[ia] + d_pol[ib, ia, je]
    end
    return a_next
end

"""
    _two_asset_euler_error_stats(c_pol, b_pol, a_pol, ip, grid, income, prices;
                                 points=:midpoints) → NamedTuple

Liquid Euler residual of a two-asset policy:

    err = |1 − β(1+r_b) E[u'(c(b', a', e'))] / u'(c)|

Continuation consumption is bilinear in `(b', a')`. Constrained
(`b' ≤ b_min`) and off-grid (`b' > b_max` or `a' > a_max`) cells are
counted, not scored — same convention as [`_euler_error_stats`](@ref).
"""
function _two_asset_euler_error_stats(c_pol::AbstractArray{T,3},
                                      b_pol::AbstractArray{T,3},
                                      a_pol::AbstractArray{T,3},
                                      ip::IndividualProblem{T},
                                      grid::HAGrid{T},
                                      income::IncomeProcess{T},
                                      prices::Dict{Symbol,T};
                                      points::Symbol=:midpoints) where {T<:AbstractFloat}
    points in (:nodes, :midpoints) ||
        throw(ArgumentError("points must be :nodes or :midpoints; got :$points"))
    b_grid = grid.grids[1]
    a_grid = grid.grids[2]
    n_b = length(b_grid)
    n_a = length(a_grid)
    n_e = length(income.states)
    b_min = ip.borrowing_constraint[1]
    b_max = b_grid[end]
    a_max = a_grid[end]
    beta = ip.beta
    u_prime = ip.utility_prime
    r_b = get(prices, :r_b, prices[:r])
    Pi = income.transition
    constraint_tol = b_min + T(1e-6)

    eval_b = points === :nodes ? b_grid :
             T[(b_grid[i] + b_grid[i + 1]) / 2 for i in 1:(n_b - 1)]
    eval_a = points === :nodes ? a_grid :
             T[(a_grid[i] + a_grid[i + 1]) / 2 for i in 1:(n_a - 1)]

    max_err = zero(T)
    sum_err = zero(T)
    n_checked = 0
    n_constrained = 0
    n_offgrid = 0

    @inbounds for je in 1:n_e
        for (ia, a_pt) in enumerate(eval_a), (ib, b_pt) in enumerate(eval_b)
            if points === :nodes
                c_here = c_pol[ib, ia, je]
                bp = b_pol[ib, ia, je]
                ap = a_pol[ib, ia, je]
            else
                c_here = _bilinear_interp(b_grid, a_grid, view(c_pol, :, :, je), b_pt, a_pt)
                bp = _bilinear_interp(b_grid, a_grid, view(b_pol, :, :, je), b_pt, a_pt)
                ap = _bilinear_interp(b_grid, a_grid, view(a_pol, :, :, je), b_pt, a_pt)
            end
            if bp <= constraint_tol
                n_constrained += 1
                continue
            end
            if bp > b_max || ap > a_max
                n_offgrid += 1
                continue
            end
            emu = zero(T)
            for jep in 1:n_e
                ct = _bilinear_interp(b_grid, a_grid, view(c_pol, :, :, jep), bp, ap)
                emu += Pi[je, jep] * u_prime(max(ct, T(1e-15)))
            end
            up_today = u_prime(max(c_here, T(1e-15)))
            if up_today > zero(T) && isfinite(emu)
                resid = abs(one(T) - beta * (one(T) + r_b) * emu / up_today)
                resid > max_err && (max_err = resid)
                sum_err += resid
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

# =============================================================================
# Two-asset Young / EGM / VFI household helper
# =============================================================================

"""
    _two_asset_hh_solve(ip, grid, income, prices; hh_solver=:egm, kwargs...)
        → (c, b', a', d, V[, converged])

Solve the two-asset household problem at given prices and reconstruct `a'`.
With `return_conv=true` (used by the GE closer, #709), append the solver's
convergence flag as a sixth element; the default 5-tuple keeps existing
callers (SSJ, Reiter, Krusell–Smith) untouched.
"""
function _two_asset_hh_solve(ip::IndividualProblem{T}, grid::HAGrid{T},
                             income::IncomeProcess{T}, prices::Dict{Symbol,T};
                             hh_solver::Symbol=:egm,
                             max_iter::Int=200, tol::T=T(1e-6),
                             howard_steps::Int=10,
                             stable_iters::Int=1,
                             init_value=nothing,
                             return_conv::Bool=false) where {T<:AbstractFloat}
    r_a = get(prices, :r_a, prices[:r])
    pol = if hh_solver === :vfi
        stable_iters == 1 || throw(ArgumentError(
            "_two_asset_hh_solve: stable_iters is EGM-only; :vfi converges on " *
            "the Bellman residual natively (got stable_iters=$stable_iters)"))
        _two_asset_vfi_solve(ip, grid, income, prices;
                             max_iter=max_iter, tol=tol, howard_steps=howard_steps,
                             init_value=init_value)
    elseif hh_solver === :egm
        _two_asset_egm_solve(ip, grid, income, prices;
                             max_iter=max_iter, tol=tol, howard_steps=howard_steps,
                             stable_iters=stable_iters,
                             init_value=init_value)
    else
        throw(ArgumentError("_two_asset_hh_solve: hh_solver must be :egm or :vfi, got :$hh_solver"))
    end
    c = pol[:consumption]
    b = pol[:liquid_savings]
    d = pol[:deposit]
    V = pol[:value]
    a_next = _two_asset_a_prime(d, grid, r_a)
    return_conv || return c, b, a_next, d, V
    return c, b, a_next, d, V, Bool(pol[:converged][1])
end

# =============================================================================
# _ha_two_asset_steady_state — damped (K, r_b) closer
# =============================================================================

"""
    _ha_two_asset_steady_state(ip, grid, income, params; kwargs...) → HASteadyState

Stationary equilibrium of a two-asset production economy.

Unknowns `(K, r_b)`. Firm FOCs give `r_a` and `w`; the government budget
sets `τ = r_b * B_supply`. Markets:

- illiquid: `A = ∫a dμ = K`
- liquid: `B = ∫b dμ = B_supply`

# Algorithm — nested bisection (#709)

The original 2D tâtonnement (`r_b += relax_rb*(B_supply−B)`,
`K += relax_K*(A−K)`) is unstable on both markets: liquid demand has slope
~8500/unit at the crossing (40-80× over the fixed-point stability limit,
so `r_b` slams wall-to-wall every iteration) and illiquid demand flips
`0↔a_max` as the `r_a−r_b` premium changes sign (so `K` falls into a
period-6 limit cycle even with a stabilized `r_b`). The closer instead
nests two bisections:

1. Inner: at fixed `K`, bisect `r_b ∈ [−r_cap, +r_cap]` on `B(r_b) − B_supply`
   (increasing: a higher liquid return raises liquid saving), best-tracked
   with no straddle requirement (coarse grids need not bracket).
2. Outer: bisect `K` on `A(K; r_b*(K)) − K` (decreasing: higher capital
   lowers `r_a` and raises the investment target), with bracket validation
   mirroring [`_ha_steady_state`](@ref) (#240/H-18).

Both loops best-track: the returned point is the best evaluated, never the
last midpoint. Household evaluations run pure VFI (`howard_steps=0`) to
genuine policy stability (`stable_iters=20`): Howard evaluation re-targets
`V` on every deposit flip so the residual never clears, and the legacy
one-iteration backstop fires in early transit (residual ~0.1), so both
leave ~0.02-0.09 warm-phase slop that stalls the outer bisection; pure VFI
instead locks the policy into a stable basin (the EGM operator limit-cycles
in `V` on fine grids but the policy along the cycle is stable to ~1.5e-4
in `A`). The default `tol=2e-3` reflects that floor (basin spread × outer
slope ~10). One warm-start `V` is threaded through every solve — cold
starts branch-hop between interior and drainage-trap fixed points (`a=0`
is spuriously absorbing: the `a_grid[2]−a_grid[1]` entry step costs χ≈10
for a 0.46 deposit). `converged = cleared && hh_converged`, mirroring
[`ct_two_asset_ge`](@ref).
"""
function _ha_two_asset_steady_state(ip::IndividualProblem{T}, grid::HAGrid{T},
                                    income::IncomeProcess{T},
                                    params::Dict{Symbol,T};
                                    K_init::T=T(10),
                                    k_lo::Union{Nothing,T}=nothing,
                                    k_hi::Union{Nothing,T}=nothing,
                                    max_iter::Int=60,
                                    tol::Real=T(2e-3),
                                    inner_max_iter::Int=30,
                                    stall_window::Int=8,
                                    k_atol::Real=T(1e-9),
                                    hh_solver::Symbol=:egm,
                                    hh_max_iter::Int=500,
                                    hh_tol::T=T(1e-6),
                                    howard_steps::Int=0,
                                    stable_iters::Int=20,
                                    grid_check::Symbol=:none,
                                    ceiling_mass_tol::Real=T(1e-6),
                                    residual_tol::Real=T(1e-6),
                                    euler_points::Symbol=:midpoints,
                                    verbose::Bool=false,
                                    distribution::Symbol=:young) where {T<:AbstractFloat}
    grid.n_dims == 2 || throw(ArgumentError(
        "_ha_two_asset_steady_state requires n_dims == 2"))
    ip.n_asset_dims == 2 || throw(ArgumentError(
        "_ha_two_asset_steady_state requires n_asset_dims == 2"))
    hh_solver in (:egm, :vfi) || throw(ArgumentError(
        "_ha_two_asset_steady_state: hh_solver must be :egm or :vfi, got :$hh_solver"))
    distribution === :young || throw(ArgumentError(
        "compute_steady_state: two-asset models support distribution=:young only " *
        "(got :$distribution). A joint Winberry family is not implemented."))
    # The stability stop is EGM-only (#709); :vfi converges on the
    # Bellman residual natively.
    stable_eff = hh_solver === :egm ? stable_iters : 1

    alpha = get(params, :alpha, T(0.36))
    delta = get(params, :delta, T(0.025))
    Z = get(params, :Z, one(T))
    L = get(params, :L, one(T))
    B_supply = get(params, :B_supply, one(T))
    tol_T = T(tol)

    firm_ra(K) = alpha * Z * (K / L)^(alpha - one(T)) - delta
    firm_w(K)  = (one(T) - alpha) * Z * (K / L)^alpha
    firm_Y(K)  = Z * K^alpha * L^(one(T) - alpha)

    # Liquid band: the RA Euler bound keeps β(1+r_b) < 1 (else liquid wealth
    # diverges). `r_cap` doubles as the inner bisection bracket.
    r_cap = one(T) / ip.beta - one(T) - T(1e-4)
    # Capital floor: the RA stock, where firm_ra == r_cap. Below it,
    # β(1+r_a) > 1 makes illiquid demand infinite (a pure grid-ceiling
    # artifact), and precautionary saving pushes equilibrium K above it
    # anyway. An explicit k_lo below the floor is clamped up (documented);
    # a bracket lying entirely below it is a loud error, not a silent artifact.
    K_floor = ((alpha * Z) / (r_cap + delta))^(one(T) / (one(T) - alpha)) * L

    n_b = grid.n_points[1]
    n_a = grid.n_points[2]
    n_e = grid.n_income
    n_dist = n_b * n_a * n_e

    # One household evaluation at (K, r_b): solve (warm-started), distribute,
    # aggregate. Returns everything the loops and the best-tracking need.
    function eval_hh(K::T, r_b::T, V_init)
        r_a = firm_ra(K)
        w = firm_w(K)
        tau = r_b * B_supply
        prices = Dict{Symbol,T}(
            :r => r_a, :r_a => r_a, :r_b => r_b, :w => w,
            :tau => tau, :div => zero(T)
        )
        c, b, a_next, d, V, conv = _two_asset_hh_solve(
            ip, grid, income, prices; hh_solver=hh_solver,
            max_iter=hh_max_iter, tol=hh_tol, howard_steps=howard_steps,
            stable_iters=stable_eff,
            init_value=V_init, return_conv=true)
        Lambda = _build_transition_matrix(b, a_next, grid, income)
        dist, _ = _stationary_dist_young(Lambda)
        B = _aggregate(dist, grid; var_index=1)
        A = _aggregate(dist, grid; var_index=2)
        return (B=B, A=A, conv=conv, V=V, c=c, b=b, a=a_next, d=d,
                dist=dist, prices=prices, r_a=r_a, w=w, tau=tau)
    end

    # Inner loop: bisect r_b at fixed K on B(r_b) − B_supply, best-tracked.
    # Returns ((r_b, eval) best by |B − B_supply|, updated warm V).
    #
    # No straddle requirement: on coarse grids the band ends need not bracket
    # (warm-path micro-chaos, or genuinely weak saving). The loop still
    # best-tracks — marching toward the closest end when unstraddled — and
    # the outer joint criterion stays honest (an uncleared inner keeps
    # max(|A−K|, |B−B̄|) above tol, so converged=false with a loud warning).
    # Only grid-infeasible supply throws: B is a convex combination of the
    # liquid nodes, so supply outside [b_min, b_max] is unrepresentable.
    b_node_min, b_node_max = grid.grids[1][1], grid.grids[1][end]
    if !(b_node_min <= B_supply <= b_node_max)
        throw(ArgumentError(
            "_ha_two_asset_steady_state: liquid supply $B_supply lies outside " *
            "the grid-representable range [$b_node_min, $b_node_max] (B is a " *
            "unit-mass average over the liquid nodes). No stationary " *
            "equilibrium exists. Reduce B_supply or extend the liquid grid."))
    end
    function inner_bisect(K::T, V_init)
        lo, hi = -r_cap, r_cap
        e_lo = eval_hh(K, lo, V_init)
        e_hi = eval_hh(K, hi, e_lo.V)
        ex_lo, ex_hi = e_lo.B - B_supply, e_hi.B - B_supply
        best_r = abs(ex_lo) <= abs(ex_hi) ? lo : hi
        best_e = abs(ex_lo) <= abs(ex_hi) ? e_lo : e_hi
        V = e_hi.V
        for _ in 1:inner_max_iter
            mid = (lo + hi) / T(2)
            e = eval_hh(K, mid, V)
            V = e.V
            if abs(e.B - B_supply) < abs(best_e.B - B_supply)
                best_r, best_e = mid, e
            end
            abs(best_e.B - B_supply) <= tol_T && break
            e.B > B_supply ? (hi = mid) : (lo = mid)
        end
        return (r=best_r, e=best_e), V
    end

    # Outer bracket: [K_floor, expand] by default. K_init seeds the expansion.
    klo_raw = something(k_lo, K_floor)
    klo = max(klo_raw, K_floor)   # documented clamp; see K_floor above
    if k_hi === nothing
        khi = max(K_init, klo * T(1.25))
    else
        khi = k_hi
    end
    khi <= K_floor && throw(ArgumentError(
        "_ha_two_asset_steady_state: outer bracket [$klo_raw, $khi] lies below the " *
        "capital floor $K_floor (firm_ra > RA-Euler bound there). No interior " *
        "stationary equilibrium exists below the floor."))
    klo >= khi && throw(ArgumentError(
        "_ha_two_asset_steady_state: outer bracket must satisfy k_lo < k_hi " *
        "(got $klo >= $khi)."))

    # Best-point storage (mirror _ha_steady_state): the reported solution is
    # the best JOINT point by max(|A−K|, |B−B̄|), never the last midpoint.
    best_c = zeros(T, n_b, n_a, n_e)
    best_b = zeros(T, n_b, n_a, n_e)
    best_a = zeros(T, n_b, n_a, n_e)
    best_d = zeros(T, n_b, n_a, n_e)
    best_V = zeros(T, n_b, n_a, n_e)
    best_dist = zeros(T, n_dist)
    best_prices = Dict{Symbol,T}()
    best_K = klo
    best_r_b = zero(T)
    best_resid_a = T(Inf)
    best_resid_b = T(Inf)
    best_joint = T(Inf)
    best_conv = false

    # Bracket validation + widening (mirror #240/H-18). The cold first solve
    # lands on the interior branch exactly here: at the floor, r_a ≈ r_cap
    # makes illiquid entry pay, defeating the a=0 drainage trap.
    V_warm = nothing
    inner_lo, V_warm = inner_bisect(klo, V_warm)
    res_lo = inner_lo.e.A - klo
    inner_hi, V_warm = inner_bisect(khi, V_warm)
    res_hi = inner_hi.e.A - khi
    widen = 0
    while res_lo > zero(T) && klo > K_floor && widen < 25
        klo = max(K_floor, klo - (khi - klo))
        inner_lo, V_warm = inner_bisect(klo, V_warm)
        res_lo = inner_lo.e.A - klo
        widen += 1
    end
    widen = 0
    while res_hi > zero(T) && widen < 25
        khi = khi + (khi - klo) * T(1.5)
        inner_hi, V_warm = inner_bisect(khi, V_warm)
        res_hi = inner_hi.e.A - khi
        widen += 1
    end
    if !(res_lo >= zero(T) >= res_hi)
        throw(ArgumentError(
            "_ha_two_asset_steady_state: could not bracket illiquid clearing " *
            "after widening to [$klo, $khi] — A−K does not change sign " *
            "(A−K at k_lo = $res_lo, at k_hi = $res_hi). The model may admit " *
            "no stationary equilibrium with an interior illiquid market."))
    end

    cleared = false
    final_iter = 0
    since_best = 0
    for it in 1:max_iter
        final_iter = it
        K = (klo + khi) / T(2)
        inner_best, V_warm = inner_bisect(K, V_warm)
        r_b = inner_best.r
        e = inner_best.e
        resid_a = e.A - K
        resid_b = e.B - B_supply
        joint = max(abs(resid_a), abs(resid_b))
        if joint < best_joint
            best_joint = joint
            since_best = 0
            copyto!(best_c, e.c)
            copyto!(best_b, e.b)
            copyto!(best_a, e.a)
            copyto!(best_d, e.d)
            copyto!(best_V, e.V)
            copyto!(best_dist, e.dist)
            best_prices = copy(e.prices)
            best_K = K
            best_r_b = r_b
            best_resid_a = resid_a
            best_resid_b = resid_b
            best_conv = e.conv
        else
            since_best += 1
        end
        _msg = "two-asset GE $it: r_a=$(round(e.r_a; sigdigits=5)) " *
               "r_b=$(round(r_b; sigdigits=5)) K=$(round(K; sigdigits=6)) " *
               "A−K=$(round(resid_a; sigdigits=3)) B−B̄=$(round(resid_b; sigdigits=3))"
        verbose ? (@info _msg) : (@debug _msg)
        if joint <= tol_T
            cleared = true
            break
        end
        resid_a > zero(T) ? (klo = K) : (khi = K)
        # Stalled on micro-branch noise (not converging, just burning solves).
        since_best >= stall_window && break
        khi - klo <= T(k_atol) && break
    end

    # Drainage-trap signature (#709): exact-zero illiquid wealth DESPITE a
    # positive premium violates the household FOC (an infinitesimal deposit
    # gains first-order premium flow at second-order adjustment cost) — the
    # grid's coarse bottom step makes a=0 spuriously absorbing. Never report
    # such a point as converged.
    B = _aggregate(best_dist, grid; var_index=1)
    A = _aggregate(best_dist, grid; var_index=2)
    r_a_best = best_prices[:r_a]
    trapped = A <= T(1e-6) && r_a_best > best_r_b
    if trapped
        @warn "_ha_two_asset_steady_state: solution has A=$A ≈ 0 with " *
              "r_a=$r_a_best > r_b=$best_r_b (positive premium) — the " *
              "a=0 drainage-trap signature. Not reporting converged; the " *
              "discretized model admits a spurious no-illiquid equilibrium " *
              "here (see #709)." maxlog = 1
        cleared = false
    end
    if !cleared
        @warn "_ha_two_asset_steady_state did not clear in $final_iter outer " *
              "iterations: |A−K| = $(abs(best_resid_a)), " *
              "|B−B_supply| = $(abs(best_resid_b)) (tol $tol_T). Returning " *
              "the best evaluated point." maxlog = 1
    end

    Y_val = firm_Y(best_K)

    euler_mid = _two_asset_euler_error_stats(best_c, best_b, best_a, ip, grid,
                                             income, best_prices; points=:midpoints)
    euler_nodes = _two_asset_euler_error_stats(best_c, best_b, best_a, ip, grid,
                                               income, best_prices; points=:nodes)
    euler_stats = (midpoints=euler_mid, nodes=euler_nodes)
    euler_err = euler_points === :nodes ? euler_nodes.max : euler_mid.max

    gdiag = _ha_grid_diagnostics(best_b, best_dist, grid;
                                 ceiling_mass_tol=ceiling_mass_tol,
                                 residual_tol=residual_tol)
    grid_check === :none ||
        _check_grid_adequacy(gdiag, grid_check; context="compute_steady_state(two-asset)")

    policies = Dict{Symbol,Array{T}}(
        :consumption => best_c,
        :liquid_savings => best_b,
        :illiquid_savings => best_a,
        :deposit => best_d,
        :savings => best_a
    )
    dist_reshaped = reshape(best_dist, n_b, n_a, n_e)
    aggregates = Dict{Symbol,T}(
        :K => best_K,
        :A => A,
        :B => B,
        :B_supply => B_supply,
        :Y => Y_val,
        :L => L,
        :excess_demand => best_joint,
        :resid_illiquid => best_resid_a,
        :resid_liquid => best_resid_b,
        :A_policy => dot(vec(best_a), best_dist),
        :B_policy => dot(vec(best_b), best_dist),
        :A_residual => gdiag.clearing_residual,
        :hh_converged => best_conv ? one(T) : zero(T)
    )
    return HASteadyState{T}(
        policies,
        dist_reshaped,
        best_V,
        best_prices,
        aggregates,
        grid,
        income,
        cleared && best_conv,
        final_iter,
        euler_err,
        aggregates[:excess_demand];
        euler=euler_stats
    )
end
