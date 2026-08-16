# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Value Function Iteration (VFI) solver with Howard improvement steps for
heterogeneous agent models.

A fallback solver for cases where the Endogenous Grid Method (EGM) does not
apply (e.g., non-convex problems, discrete choices, adjustment costs with
non-invertible first-order conditions).

# References
- Howard, R. A. (1960). *Dynamic Programming and Markov Processes*. MIT Press.
- Ljungqvist, L., & Sargent, T. J. (2018). *Recursive Macroeconomic Theory*
  (4th ed.). MIT Press, Ch. 4.
"""

# =============================================================================
# Period reward for one-asset VFI (exogenous labor, GHH, or separable)
# =============================================================================

"""
    _vfi_flow(ip, prices, a, e, a_prime) → (u_flow, c)

Period payoff and actual consumption of choosing `a'` at state `(a, e)`.
Returns `u_flow = -Inf` when the choice is infeasible.

- No labor: `u(c)` with `c = budget(a,e) − a'`.
- GHH: the Bellman is in the composite `x = c − v(n)`; `n` depends only on `w e`.
- Separable: hours come from the intratemporal FOC given `c`; `u_flow = u(c) − v(n)`.
"""
function _vfi_flow(ip::IndividualProblem{T}, prices::Dict{Symbol,T},
                   a::T, e::T, a_prime::T) where {T<:AbstractFloat}
    ls = ip.labor
    if ls === nothing
        c = ip.budget_fn(a, e, prices) - a_prime
        c <= zero(T) && return (T(-Inf), c)
        return (ip.utility(c), c)
    end
    w = prices[:w]
    b0 = ip.budget_fn(zero(T), e, prices)
    R = ip.budget_fn(one(T), e, prices) - b0
    we = w * e
    if ls.kind === :ghh
        ytil, _, d = _ghh_net_income(ls, we)
        x = R * a + (b0 - we) + ytil - a_prime
        x <= zero(T) && return (T(-Inf), x)
        return (ip.utility(x), x + d)
    end
    net = b0 - we
    c = _solve_constrained_c(ls, ip.utility_prime, a, 1, T[R], T[net], w, e, a_prime)
    n = labor_supply(ls, we, ip.utility_prime(max(c, T(1e-12))))
    if c <= zero(T) || !isfinite(c)
        return (T(-Inf), c)
    end
    return (ip.utility(c) - _labor_disutility(ls, n), c)
end

"""
    _golden_argmax(f, lo, hi; max_iter=50, tol) → (x, f(x))

Maximize a scalar `f` on `[lo, hi]` by golden-section search. `f` may return
`-Inf` for infeasible points; the search retreats from them.
"""
function _golden_argmax(f, lo::T, hi::T; max_iter::Int=50,
                        tol::T=sqrt(eps(T))) where {T<:AbstractFloat}
    lo >= hi && return (lo, f(lo))
    φ = (sqrt(T(5)) - one(T)) / T(2)
    a, b = lo, hi
    c = b - φ * (b - a)
    d = a + φ * (b - a)
    fc, fd = f(c), f(d)
    for _ in 1:max_iter
        (b - a) <= tol * max(one(T), abs(a), abs(b)) && break
        if fc < fd
            a = c
            c = d
            fc = fd
            d = a + φ * (b - a)
            fd = f(d)
        else
            b = d
            d = c
            fd = fc
            c = b - φ * (b - a)
            fc = f(c)
        end
    end
    return fc >= fd ? (c, fc) : (d, fd)
end

"""
    _vfi_a_hi(ip, prices, a, e, a_min, a_max) → T

Largest feasible end-of-period asset (just inside the resource constraint).
"""
function _vfi_a_hi(ip::IndividualProblem{T}, prices::Dict{Symbol,T},
                   a::T, e::T, a_min::T, a_max::T) where {T<:AbstractFloat}
    ls = ip.labor
    if ls === nothing
        return min(a_max, ip.budget_fn(a, e, prices) - T(1e-12))
    end
    w = prices[:w]
    b0 = ip.budget_fn(zero(T), e, prices)
    R = ip.budget_fn(one(T), e, prices) - b0
    we = w * e
    if ls.kind === :ghh
        ytil, _, _ = _ghh_net_income(ls, we)
        return min(a_max, R * a + (b0 - we) + ytil - T(1e-12))
    end
    ncap = isfinite(ls.n_max) ? ls.n_max : T(1e3)
    return min(a_max, R * a + (b0 - we) + we * ncap - T(1e-12))
end

"""
    _hh_a_policy(ip, grid, income, prices; hh_solver=:egm, kwargs...) → Matrix

One-asset savings policy from EGM or VFI. Shared by the HA steady state and the
Reiter/Winberry price finite differences so `hh_solver=:vfi` is consistent.
"""
function _hh_a_policy(ip::IndividualProblem{T}, grid::HAGrid{T},
                      income::IncomeProcess{T}, prices::Dict{Symbol,T};
                      hh_solver::Symbol=:egm, kwargs...) where {T<:AbstractFloat}
    if hh_solver === :vfi
        _, _, a_pol, _ = _vfi_solve(ip, grid, income, prices; kwargs...)
        return a_pol
    elseif hh_solver === :egm
        _, a_pol, _ = _egm_solve(ip, grid, income, prices; kwargs...)
        return a_pol
    end
    throw(ArgumentError("_hh_a_policy: hh_solver must be :egm or :vfi, got :$hh_solver"))
end

# =============================================================================
# One-asset VFI with Howard improvement
# =============================================================================

"""
    _vfi_solve(ip, grid, income, prices; max_iter=1000, tol=1e-8, howard_steps=20,
               init_value=nothing, init_policy=nothing)
        -> (V, c_policy, a_policy, converged)

Solve a one-asset household savings problem via Value Function Iteration with
Howard (1960) policy-evaluation acceleration.

Returns `N_a x N_e` value function, consumption policy, and savings policy
matrices on the exogenous asset grid.

# Algorithm
1. Initialize value function: `V[i,j] = u(coh * 0.5) / (1 - beta)`.
2. Each iteration:
   a. **Maximize**: discrete search over the savings grid (global, so
      non-convex payoffs stay well-defined), then golden-section refine
      between neighboring nodes so `a'` is continuous. Continuation is
      linearly interpolated. Labor (GHH composite or separable hours)
      enters through [`_vfi_flow`](@ref).
   b. **Howard steps**: Fix the (continuous) policy, iterate the value
      function forward `howard_steps` times.
3. Convergence: `max|V_new - V_old| < tol`.
4. Return value function, consumption policy, and savings policy.

# Arguments
- `ip::IndividualProblem{T}` — household problem specification
- `grid::HAGrid{T}` — asset grid (one-dimensional)
- `income::IncomeProcess{T}` — idiosyncratic income Markov chain
- `prices::Dict{Symbol,T}` — price vector (must contain `:r` and `:w`)
- `max_iter::Int` — maximum number of VFI iterations (default 1000)
- `tol::T` — convergence tolerance on the sup-norm of value function changes (default 1e-8)
- `howard_steps::Int` — number of Howard policy-evaluation steps per iteration (default 20)
"""
function _vfi_solve(ip::IndividualProblem{T}, grid::HAGrid{T},
                     income::IncomeProcess{T}, prices::Dict{Symbol,T};
                     max_iter::Int=1000, tol::T=T(1e-8),
                     howard_steps::Int=20,
                     init_value::Union{Nothing,AbstractMatrix{T}}=nothing,
                     init_policy::Union{Nothing,AbstractMatrix{T}}=nothing) where {T<:AbstractFloat}
    @assert ip.n_asset_dims == 1 "VFI solver requires n_asset_dims == 1"
    @assert grid.n_dims == 1 "VFI solver requires a one-dimensional grid"

    a_grid = grid.grids[1]
    n_a = length(a_grid)
    n_e = length(income.states)
    a_min = ip.borrowing_constraint[1]

    beta = ip.beta
    u = ip.utility
    Pi = income.transition   # n_e x n_e, row-stochastic
    e_vals = income.states

    # Pre-compute cash-on-hand for every (asset, income) pair
    coh = zeros(T, n_a, n_e)
    for j in 1:n_e
        for i in 1:n_a
            coh[i, j] = ip.budget_fn(a_grid[i], e_vals[j], prices)
        end
    end

    # Initialize value function: V[i,j] = u(max(coh, tiny) * 0.5) / (1 - beta)
    # Clamp cash-on-hand to a small positive value for initialization so that
    # log-utility never produces -Inf (which would poison expected values).
    V = zeros(T, n_a, n_e)
    if init_value !== nothing && size(init_value) == (n_a, n_e)
        copyto!(V, init_value)
    else
        for j in 1:n_e
            for i in 1:n_a
                c_init = max(coh[i, j] * T(0.5), T(1e-10))
                V[i, j] = u(c_init) / (one(T) - beta)
            end
        end
    end

    # Pre-compute expected continuation value: EV[i, j] = sum_jp Pi[j, jp] * V[i, jp]
    EV = zeros(T, n_a, n_e)

    # Discrete index (monotone grid scan) plus continuous a' (local refine)
    pol_idx = ones(Int, n_a, n_e)
    a_choice = fill(a_grid[1], n_a, n_e)
    if init_policy !== nothing && size(init_policy) == (n_a, n_e)
        for j in 1:n_e
            for i in 1:n_a
                ap = init_policy[i, j]
                k = searchsortedfirst(a_grid, ap)
                pol_idx[i, j] = clamp(k, 1, n_a)
                a_choice[i, j] = clamp(ap, a_min, a_grid[end])
            end
        end
    end

    # Buffers
    V_new = zeros(T, n_a, n_e)
    converged = false
    final_iter = 0

    for iter in 1:max_iter
        final_iter = iter
        # Compute expected continuation value
        for j in 1:n_e
            for i in 1:n_a
                ev = zero(T)
                for jp in 1:n_e
                    ev += Pi[j, jp] * V[i, jp]
                end
                EV[i, j] = ev
            end
        end

        # ── Maximization step ─────────────────────────────────────────────
        for j in 1:n_e
            ev_j = view(EV, :, j)
            for i in 1:n_a
                best_val = T(-Inf)
                best_idx = 1
                a_i = a_grid[i]
                e_j = e_vals[j]
                hi_feas = _vfi_a_hi(ip, prices, a_i, e_j, a_min, a_grid[end])

                # Discrete global scan; start at the previous row's index
                # (monotonicity) so non-convex payoffs still see the best node.
                lo = i > 1 ? pol_idx[i-1, j] : 1
                for k in lo:n_a
                    u_flow, _ = _vfi_flow(ip, prices, a_i, e_j, a_grid[k])
                    if !isfinite(u_flow)
                        # Remaining larger a' are even less feasible when resources
                        # do not rise with a' (the usual case).
                        ls = ip.labor
                        (ls === nothing || ls.kind === :ghh) && break
                        continue
                    end
                    val = u_flow + beta * EV[k, j]
                    if val > best_val
                        best_val = val
                        best_idx = k
                    end
                end

                if best_val == T(-Inf)
                    best_idx = 1
                    u_fb, _ = _vfi_flow(ip, prices, a_i, e_j, a_grid[1])
                    best_val = (isfinite(u_fb) ? u_fb : u(T(1e-10))) + beta * EV[1, j]
                    a_star = a_grid[1]
                else
                    lo_r = best_idx > 1 ? a_grid[best_idx - 1] : a_grid[1]
                    hi_r = best_idx < n_a ? a_grid[best_idx + 1] : a_grid[n_a]
                    hi_r = min(hi_r, max(hi_feas, lo_r))
                    lo_r = min(lo_r, hi_r)
                    a_star = a_grid[best_idx]
                    if hi_r > lo_r + T(1e-14)
                        f = let ip=ip, prices=prices, a_i=a_i, e_j=e_j,
                                ev_j=ev_j, beta=beta, a_grid=a_grid
                            ap -> begin
                                uf, _ = _vfi_flow(ip, prices, a_i, e_j, ap)
                                isfinite(uf) ? uf + beta * _linear_interp(a_grid, ev_j, ap) :
                                    T(-Inf)
                            end
                        end
                        ap_r, val_r = _golden_argmax(f, lo_r, hi_r)
                        if isfinite(val_r) && val_r >= best_val
                            a_star = ap_r
                            best_val = val_r
                        end
                    end
                end

                V_new[i, j] = best_val
                pol_idx[i, j] = best_idx
                a_choice[i, j] = a_star
            end
        end

        # ── Howard improvement steps ──────────────────────────────────────
        # Fix the continuous policy and iterate V without re-optimizing.
        for _h in 1:howard_steps
            for j in 1:n_e
                for i in 1:n_a
                    ev = zero(T)
                    for jp in 1:n_e
                        ev += Pi[j, jp] * V_new[i, jp]
                    end
                    EV[i, j] = ev
                end
            end

            for j in 1:n_e
                ev_j = view(EV, :, j)
                for i in 1:n_a
                    ap = a_choice[i, j]
                    u_flow, _ = _vfi_flow(ip, prices, a_grid[i], e_vals[j], ap)
                    if !isfinite(u_flow)
                        u_flow = u(T(1e-10))
                    end
                    V_new[i, j] = u_flow + beta * _linear_interp(a_grid, ev_j, ap)
                end
            end
        end

        # ── Convergence check ─────────────────────────────────────────────
        max_diff = zero(T)
        for j in 1:n_e
            for i in 1:n_a
                diff = abs(V_new[i, j] - V[i, j])
                if isfinite(diff) && diff > max_diff
                    max_diff = diff
                end
            end
        end

        copyto!(V, V_new)

        if max_diff < tol
            converged = true
            break
        end
    end

    # ── Extract policies ──────────────────────────────────────────────────
    c_policy = zeros(T, n_a, n_e)
    a_policy = copy(a_choice)
    for j in 1:n_e
        for i in 1:n_a
            _, c = _vfi_flow(ip, prices, a_grid[i], e_vals[j], a_choice[i, j])
            c_policy[i, j] = isfinite(c) ? max(c, T(1e-10)) : T(1e-10)
        end
    end

    return V, c_policy, a_policy, converged
end

"""
    _policy_value_fn(c_pol, a_pol, ip, grid, income; max_iter=200, tol=1e-8) -> Matrix

Howard policy evaluation of a given one-asset household policy. Used to fill
`HASteadyState.value_fn` after EGM, which does not iterate a value function.
"""
function _policy_value_fn(c_pol::AbstractMatrix{T}, a_pol::AbstractMatrix{T},
                          ip::IndividualProblem{T}, grid::HAGrid{T},
                          income::IncomeProcess{T};
                          max_iter::Int=200, tol::T=T(1e-8)) where {T<:AbstractFloat}
    a_grid = grid.grids[1]
    n_a = length(a_grid)
    n_e = length(income.states)
    beta = ip.beta
    u = ip.utility
    Pi = income.transition

    V = zeros(T, n_a, n_e)
    for j in 1:n_e
        for i in 1:n_a
            V[i, j] = u(max(c_pol[i, j], T(1e-10))) / (one(T) - beta)
        end
    end
    V_new = similar(V)
    for _ in 1:max_iter
        max_diff = zero(T)
        for j in 1:n_e
            for i in 1:n_a
                ev = zero(T)
                ap = a_pol[i, j]
                for jp in 1:n_e
                    ev += Pi[j, jp] * _linear_interp(a_grid, view(V, :, jp), ap)
                end
                V_new[i, j] = u(max(c_pol[i, j], T(1e-10))) + beta * ev
                d = abs(V_new[i, j] - V[i, j])
                if isfinite(d) && d > max_diff
                    max_diff = d
                end
            end
        end
        copyto!(V, V_new)
        max_diff < tol && break
    end
    return V
end

# =============================================================================
# Two-asset VFI (grid search over a' and b')
# =============================================================================

"""
    _two_asset_vfi_solve(ip, grid, income, prices; max_iter=200, tol=1e-6,
                         howard_steps=10) → Dict{Symbol,Array}

Bellman VFI on the joint liquid/illiquid state `(b, a, e)`, nested the same
way as [`_two_asset_egm_solve`](@ref): the illiquid deposit `a'` is searched
on the grid; for each candidate the liquid choice `b'` is a continuous
golden-section max of `u(c) + β E[V(b', a', e')]` with linearly interpolated
continuation. Intended as a convex-problem cross-check of nested EGM, not as
a production GE solver.

Returns the same `Dict` keys as `_two_asset_egm_solve`: `:consumption`,
`:liquid_savings`, `:deposit`, `:value`, `:converged`.
"""
function _two_asset_vfi_solve(ip::IndividualProblem{T}, grid::HAGrid{T},
                              income::IncomeProcess{T}, prices::Dict{Symbol,T};
                              max_iter::Int=200, tol::T=T(1e-6),
                              howard_steps::Int=10) where {T<:AbstractFloat}
    @assert ip.n_asset_dims == 2 "Two-asset VFI requires n_asset_dims == 2"
    @assert grid.n_dims == 2 "Two-asset VFI requires a two-dimensional grid"

    b_grid = grid.grids[1]
    a_grid = grid.grids[2]
    n_b = length(b_grid)
    n_a = length(a_grid)
    n_e = length(income.states)
    beta = ip.beta
    u = ip.utility
    Pi = income.transition
    e_vals = income.states
    b_min = ip.borrowing_constraint[1]
    r_a = get(prices, :r_a, prices[:r])
    adj_cost = isnothing(ip.adjustment_cost) ? (d, a) -> zero(T) : ip.adjustment_cost
    budget_fn = ip.budget_fn
    NEG = T(-1e20)

    d_tab = zeros(T, n_a, n_a)
    chi_tab = zeros(T, n_a, n_a)
    for ia in 1:n_a, iap in 1:n_a
        dv = a_grid[iap] - (one(T) + r_a) * a_grid[ia]
        d_tab[iap, ia] = dv
        chi_tab[iap, ia] = adj_cost(dv, a_grid[ia])
    end

    V = zeros(T, n_b, n_a, n_e)
    c_opt = zeros(T, n_b, n_a, n_e)
    b_opt = zeros(T, n_b, n_a, n_e)
    iap_opt = ones(Int, n_b, n_a, n_e)
    for je in 1:n_e, ia in 1:n_a, ib in 1:n_b
        coh = budget_fn(b_grid[ib], a_grid[ia], e_vals[je], prices)
        c0 = max(coh * T(0.5), T(1e-8))
        c_opt[ib, ia, je] = c0
        b_opt[ib, ia, je] = b_grid[ib]
        V[ib, ia, je] = u(c0) / (one(T) - beta)
    end

    EV = zeros(T, n_b, n_a, n_e)
    V_new = zeros(T, n_b, n_a, n_e)
    converged = false

    for iter in 1:max_iter
        fill!(EV, zero(T))
        for je in 1:n_e, jep in 1:n_e
            wgt = Pi[je, jep]
            @inbounds for ia in 1:n_a, ib in 1:n_b
                EV[ib, ia, je] += wgt * V[ib, ia, jep]
            end
        end

        for je in 1:n_e, ia in 1:n_a, ib in 1:n_b
            coh = budget_fn(b_grid[ib], a_grid[ia], e_vals[je], prices)
            best = NEG
            best_iap = 1
            best_c = T(1e-8)
            best_bp = b_min
            for iap in 1:n_a
                cost = d_tab[iap, ia] + chi_tab[iap, ia]
                hi = min(b_grid[end], coh - cost - T(1e-12))
                hi < b_min && continue
                evcol = view(EV, :, iap, je)
                f = let u=u, coh=coh, cost=cost, beta=beta, b_grid=b_grid, evcol=evcol
                    bp -> begin
                        c = coh - bp - cost
                        c <= zero(T) && return T(-Inf)
                        u(c) + beta * _linear_interp(b_grid, evcol, bp)
                    end
                end
                bp_star, val = _golden_argmax(f, b_min, hi)
                if isfinite(val) && val > best
                    best = val
                    best_iap = iap
                    best_bp = bp_star
                    best_c = coh - bp_star - cost
                end
            end
            if best == NEG
                resources = coh
                best_c = max(resources - b_min, T(1e-10))
                best_bp = b_min
                best_iap = ia
                best = u(best_c) + beta * EV[1, ia, je]
            end
            V_new[ib, ia, je] = best
            iap_opt[ib, ia, je] = best_iap
            b_opt[ib, ia, je] = best_bp
            c_opt[ib, ia, je] = best_c
        end

        for _h in 1:howard_steps
            fill!(EV, zero(T))
            for je in 1:n_e, jep in 1:n_e
                wgt = Pi[je, jep]
                @inbounds for ia in 1:n_a, ib in 1:n_b
                    EV[ib, ia, je] += wgt * V_new[ib, ia, jep]
                end
            end
            for je in 1:n_e, ia in 1:n_a, ib in 1:n_b
                iap = iap_opt[ib, ia, je]
                bp = b_opt[ib, ia, je]
                V_new[ib, ia, je] = u(max(c_opt[ib, ia, je], T(1e-10))) +
                    beta * _linear_interp(b_grid, view(EV, :, iap, je), bp)
            end
        end

        max_diff = zero(T)
        @inbounds for i in eachindex(V)
            d = abs(V_new[i] - V[i])
            if isfinite(d) && d > max_diff
                max_diff = d
            end
        end
        copyto!(V, V_new)
        if max_diff < tol
            converged = true
            break
        end
    end

    d_opt = zeros(T, n_b, n_a, n_e)
    for je in 1:n_e, ia in 1:n_a, ib in 1:n_b
        d_opt[ib, ia, je] = d_tab[iap_opt[ib, ia, je], ia]
    end
    return Dict{Symbol,Array{T}}(
        :consumption => c_opt,
        :liquid_savings => b_opt,
        :deposit => d_opt,
        :value => V,
        :converged => T[converged ? one(T) : zero(T)],
    )
end
