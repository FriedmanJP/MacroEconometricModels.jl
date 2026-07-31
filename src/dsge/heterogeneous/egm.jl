# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Endogenous Grid Method (EGM) solvers for heterogeneous agent models.

Implements Carroll (2006) one-asset EGM and a simplified nested EGM for
two-asset models.

# References
- Carroll, C. D. (2006). The method of endogenous gridpoints for solving
  dynamic stochastic optimization problems. *Economics Letters*, 91(3), 312–320.
"""

# =============================================================================
# Interpolation utilities
# =============================================================================

"""
    _linear_interp(x, y, xi) → yi

One-dimensional linear interpolation with flat extrapolation.

Given sorted knots `x` and values `y`, evaluate at query point `xi`.
For `xi` outside the range of `x`, returns `y[1]` or `y[end]` (flat).
"""
function _linear_interp(x::AbstractVector{T}, y::AbstractVector{T}, xi::T) where {T<:AbstractFloat}
    n = length(x)
    @assert n == length(y) "x and y must have the same length"
    @assert n >= 2 "Need at least 2 knots"

    # Flat extrapolation below
    if xi <= x[1]
        return y[1]
    end
    # Flat extrapolation above
    if xi >= x[end]
        return y[end]
    end

    # Find interval: x[k] <= xi < x[k+1]
    k = searchsortedfirst(x, xi) - 1
    k = clamp(k, 1, n - 1)

    # Linear interpolation
    t = (xi - x[k]) / (x[k+1] - x[k])
    return y[k] + t * (y[k+1] - y[k])
end

"""
    _bilinear_interp(x1, x2, z, xi1, xi2) → zi

Two-dimensional bilinear interpolation with flat extrapolation.

Given sorted knots `x1` (length n1) and `x2` (length n2), and values
`z` (n1 × n2 matrix), evaluate at query point `(xi1, xi2)`.
"""
function _bilinear_interp(x1::AbstractVector{T}, x2::AbstractVector{T},
                           z::AbstractMatrix{T}, xi1::T, xi2::T) where {T<:AbstractFloat}
    n1 = length(x1)
    n2 = length(x2)
    @assert size(z) == (n1, n2) "z must be n1 × n2"

    # Clamp query points to grid range
    xi1_c = clamp(xi1, x1[1], x1[end])
    xi2_c = clamp(xi2, x2[1], x2[end])

    # Find intervals
    k1 = searchsortedfirst(x1, xi1_c) - 1
    k1 = clamp(k1, 1, n1 - 1)
    k2 = searchsortedfirst(x2, xi2_c) - 1
    k2 = clamp(k2, 1, n2 - 1)

    # Weights
    t1 = (xi1_c - x1[k1]) / (x1[k1+1] - x1[k1])
    t2 = (xi2_c - x2[k2]) / (x2[k2+1] - x2[k2])

    # Bilinear combination
    return (one(T) - t1) * (one(T) - t2) * z[k1, k2] +
           t1 * (one(T) - t2) * z[k1+1, k2] +
           (one(T) - t1) * t2 * z[k1, k2+1] +
           t1 * t2 * z[k1+1, k2+1]
end

# =============================================================================
# One-asset EGM (Carroll 2006)
# =============================================================================

"""
    _solve_constrained_c(ls, u_prime, a, j, gross_R, net_inc, w, e, a_min) → c

Consumption of a borrowing-constrained household under **separable** preferences
with endogenous labor. The household saves `a_min`, so consumption solves the
budget identity

    c + a_min = R·a + offset + w·e·n(c),    n(c) = (w·e·u'(c)/ψ)^φ

Hours fall as consumption rises (`u'` is decreasing), so
`g(c) = c + a_min − R·a − offset − w·e·n(c)` is strictly increasing: the root is
unique and bisection cannot miss it. This is the one place the intratemporal and
intertemporal conditions genuinely couple — on the unconstrained branch the Euler
inversion delivers `c` first, and hours follow in closed form.
"""
function _solve_constrained_c(ls::LaborSupply{T}, u_prime::F, a::T, j::Int,
                              gross_R::Vector{T}, net_inc::Vector{T}, w::T,
                              e::T, a_min::T) where {T<:AbstractFloat, F}
    we = w * e
    base = gross_R[j] * a + net_inc[j] - a_min
    g(c) = c - base - we * labor_supply(ls, we, u_prime(c))

    lo = T(1e-12)
    g(lo) >= zero(T) && return lo          # already non-negative at the floor
    hi = max(base, one(T))
    for _ in 1:60
        g(hi) >= zero(T) && break
        hi *= T(2)
    end
    g(hi) < zero(T) && return max(hi, T(1e-10))   # unbracketed: return the cap

    for _ in 1:100
        mid = (lo + hi) / T(2)
        (hi - lo) <= eps(T) * max(one(T), hi) && break
        g(mid) < zero(T) ? (lo = mid) : (hi = mid)
    end
    return max((lo + hi) / T(2), T(1e-10))
end

"""
    labor_policy(ip, grid, income, prices, c_pol) → Matrix{T}

Hours worked `n(a, e)` implied by a converged consumption policy, `n_a × n_e`.
Returns a matrix of ones when the individual problem has exogenous labor
(`ip.labor === nothing`), so aggregate labor in efficiency units is `∫ e dμ`
either way.

Hours follow the intratemporal first-order condition in closed form given
consumption: `n = (w e / ψ)^φ` under GHH (independent of `c`) and
`n = (w e u'(c) / ψ)^φ` under separable preferences.
"""
function labor_policy(ip::IndividualProblem{T}, grid::HAGrid{T},
                      income::IncomeProcess{T}, prices::Dict{Symbol,T},
                      c_pol::AbstractMatrix{T}) where {T<:AbstractFloat}
    n_a = grid.n_points[1]
    n_e = length(income.states)
    ls = ip.labor
    ls === nothing && return ones(T, n_a, n_e)

    w = prices[:w]
    u_prime = ip.utility_prime
    n_pol = zeros(T, n_a, n_e)
    for j in 1:n_e
        we = w * income.states[j]
        if ls.kind === :ghh
            nj = labor_supply(ls, we)
            for i in 1:n_a
                n_pol[i, j] = nj
            end
        else
            for i in 1:n_a
                n_pol[i, j] = labor_supply(ls, we, u_prime(c_pol[i, j]))
            end
        end
    end
    return n_pol
end

"""
    _egm_solve(ip, grid, income, prices; max_iter=1000, tol=1e-10, init_policy=nothing)
        → (c_policy::Matrix{T}, a_policy::Matrix{T}, converged::Bool)

Solve a one-asset household savings problem using the Endogenous Grid Method
(Carroll 2006).

Returns `N_a × N_e` consumption and savings policy matrices on the exogenous
asset grid, plus a `converged` flag (`true` iff the fixed-point iteration met
`tol` before exhausting `max_iter`). The flag is the LAST element so existing
`c_pol, a_pol = _egm_solve(...)` / `_, a_pol = _egm_solve(...)` call sites are
unaffected. Pass `init_policy` (an `N_a × N_e` consumption matrix) to warm-start
from a previous solution at nearby prices.

# Algorithm
For each iteration, for each income state `j`:
1. Fix end-of-period asset grid `a'` (the exogenous grid).
2. Compute expected marginal utility: `EMU[i] = β(1+r) Σ_j' π(j,j') u'(c_old(a'_i, e_j'))`.
3. Euler inversion: `c_endo[i] = (u')⁻¹(EMU[i])`.
4. Endogenous beginning-of-period assets: `a_endo[i] = (c_endo[i] + a'_i - w·e_j) / (1+r)`.
5. Interpolate `(a_endo, c_endo)` back onto the exogenous grid.
6. Enforce borrowing constraint: if `a < a_endo[1]`, consume all cash-on-hand
   minus the borrowing limit.
"""
function _egm_solve(ip::IndividualProblem{T}, grid::HAGrid{T},
                     income::IncomeProcess{T}, prices::Dict{Symbol,T};
                     max_iter::Int=1000, tol::T=T(1e-10),
                     init_policy::Union{Nothing,AbstractMatrix{T}}=nothing) where {T<:AbstractFloat}
    @assert ip.n_asset_dims == 1 "One-asset EGM requires n_asset_dims == 1"
    @assert grid.n_dims == 1 "One-asset EGM requires a one-dimensional grid"

    a_grid = grid.grids[1]
    n_a = length(a_grid)
    n_e = length(income.states)
    a_min = ip.borrowing_constraint[1]

    beta = ip.beta
    u_prime = ip.utility_prime
    u_prime_inv = ip.utility_prime_inv

    Pi = income.transition  # n_e × n_e, row-stochastic
    e_vals = income.states

    r = prices[:r]
    w = prices[:w]

    # --- Endogenous labor supply (optional) -------------------------------
    # `budget_fn` is affine in own assets, so its own-asset slope is the gross
    # return and `budget_fn(0, e)` is non-asset income. With endogenous labor
    # `budget_fn` is read at n = 1, so subtracting `w*e` isolates any non-labor
    # offset (e.g. `div` in _hank1_budget); labor income then replaces the `w*e`
    # term. With `labor === nothing` `net_inc[j]` is exactly `budget_fn(0, e_j)`
    # and every formula below reduces to the exogenous-labor one.
    ls = ip.labor
    has_labor = ls !== nothing
    ghh = has_labor && ls.kind === :ghh
    gross_R = zeros(T, n_e)
    net_inc = zeros(T, n_e)       # non-asset income entering the EGM's own good
    ghh_hours = zeros(T, n_e)
    ghh_disutil = zeros(T, n_e)
    for j in 1:n_e
        b0 = ip.budget_fn(zero(T), e_vals[j], prices)
        gross_R[j] = ip.budget_fn(one(T), e_vals[j], prices) - b0
        if !has_labor
            net_inc[j] = b0
        elseif ghh
            # GHH hours depend on the effective wage alone, so the whole net
            # labor income term is precomputable and the EGM runs on the
            # composite good x = c - v(n) with a standard one-dimensional
            # Euler equation. `c_pol` holds x until the final conversion below.
            ytil, nj, dj = _ghh_net_income(ls, w * e_vals[j])
            net_inc[j] = (b0 - w * e_vals[j]) + ytil
            ghh_hours[j] = nj
            ghh_disutil[j] = dj
        else
            # Separable: hours depend on consumption, so only the offset is
            # precomputable; the labor income term is added per grid point.
            net_inc[j] = b0 - w * e_vals[j]
        end
    end

    # Cash-on-hand of the EGM's own good at assets `a` in income state `j`.
    # For separable preferences it also depends on consumption through hours.
    coh_at(a, j) = has_labor ? gross_R[j] * a + net_inc[j] :
                               ip.budget_fn(a, e_vals[j], prices)
    coh_sep(a, j, c) = gross_R[j] * a + net_inc[j] +
                       w * e_vals[j] * labor_supply(ls, w * e_vals[j], u_prime(c))

    # Initialize consumption policy. When a previous solution is supplied
    # (`init_policy`, #238), warm-start from it — successive EGM solves at nearby
    # prices then reach the fixed point in far fewer iterations. Otherwise fall
    # back to the cold 5%-of-cash-on-hand seed.
    c_pol = zeros(T, n_a, n_e)
    if init_policy !== nothing && size(init_policy) == (n_a, n_e)
        copyto!(c_pol, init_policy)
        @inbounds for j in 1:n_e, i in 1:n_a
            c_pol[i, j] = max(c_pol[i, j], T(1e-10))
        end
    else
        for j in 1:n_e
            for i in 1:n_a
                coh = coh_at(a_grid[i], j)
                c_pol[i, j] = max(coh * T(0.05), T(1e-10))
            end
        end
    end

    # Buffers
    c_new = zeros(T, n_a, n_e)
    a_new = zeros(T, n_a, n_e)
    emu = zeros(T, n_a)
    c_endo = zeros(T, n_a)
    a_endo = zeros(T, n_a)

    converged = false
    for iter in 1:max_iter
        # For each income state, run the EGM step
        for j in 1:n_e
            # Step 1: Compute expected marginal utility at each end-of-period asset a'_i
            fill!(emu, zero(T))
            for jp in 1:n_e
                for i in 1:n_a
                    # a'_i is a grid node, so interpolating the current policy there
                    # returns the exact indexed value c_pol[i, jp] (#242 no-op interp).
                    emu[i] += Pi[j, jp] * u_prime(c_pol[i, jp])
                end
            end

            # Step 2: Euler inversion
            for i in 1:n_a
                c_endo[i] = u_prime_inv(beta * (one(T) + r) * emu[i])
            end

            # Step 3: Endogenous beginning-of-period assets.
            # Route the non-asset ("net") income through budget_fn so any offset
            # (e.g. `div` in _hank1_budget) is honoured — the hardcoded `w*e`
            # here silently dropped `div` (#235/H-09). budget_fn is affine in own
            # assets, so net_income = budget_fn(0, e, prices) and the gross return
            # factor is its own-asset slope budget_fn(1,e)-budget_fn(0,e); both are
            # read from the same hook rather than re-inlining `w*e`/`(1+r)`.
            # Under separable preferences hours respond to consumption, so the
            # labor income term varies along the endogenous grid; under GHH (and
            # with exogenous labor) it is already inside `net_inc[j]`.
            if has_labor && !ghh
                we = w * e_vals[j]
                for i in 1:n_a
                    lab_inc = we * labor_supply(ls, we, u_prime(c_endo[i]))
                    a_endo[i] = (c_endo[i] + a_grid[i] - net_inc[j] - lab_inc) / gross_R[j]
                end
            else
                for i in 1:n_a
                    a_endo[i] = (c_endo[i] + a_grid[i] - net_inc[j]) / gross_R[j]
                end
            end

            # Step 4: Interpolate back to exogenous grid + borrowing constraint
            for i in 1:n_a
                a_val = a_grid[i]
                if a_val < a_endo[1]
                    # Constrained: consume all cash-on-hand minus borrowing limit
                    # savings. With separable preferences cash-on-hand itself
                    # depends on consumption through hours, so this is a scalar
                    # root-find rather than a subtraction.
                    if has_labor && !ghh
                        c_new[i, j] = _solve_constrained_c(ls, u_prime, a_val, j,
                                                           gross_R, net_inc, w,
                                                           e_vals[j], a_min)
                    else
                        c_new[i, j] = max(coh_at(a_val, j) - a_min, T(1e-10))
                    end
                    a_new[i, j] = a_min
                else
                    c_new[i, j] = _linear_interp(a_endo, c_endo, a_val)
                    coh = has_labor && !ghh ? coh_sep(a_val, j, c_new[i, j]) :
                                              coh_at(a_val, j)
                    a_new[i, j] = max(coh - c_new[i, j], a_min)
                end
            end
        end

        # Check convergence
        max_diff = zero(T)
        for j in 1:n_e
            for i in 1:n_a
                diff = abs(c_new[i, j] - c_pol[i, j])
                if diff > max_diff
                    max_diff = diff
                end
            end
        end

        # Update policy
        copyto!(c_pol, c_new)

        if max_diff < tol
            converged = true
            break
        end
    end

    # Compute final savings policy from budget constraint
    a_pol = zeros(T, n_a, n_e)
    for j in 1:n_e
        for i in 1:n_a
            coh = has_labor && !ghh ? coh_sep(a_grid[i], j, c_pol[i, j]) :
                                      coh_at(a_grid[i], j)
            a_pol[i, j] = max(coh - c_pol[i, j], a_min)
        end
    end

    # Under GHH the loop above solved for the composite good x = c - v(n); the
    # savings policy is already correct (the budget is written in x), so convert
    # to actual consumption only now: c = x + v(n).
    if ghh
        for j in 1:n_e
            d = ghh_disutil[j]
            for i in 1:n_a
                c_pol[i, j] += d
            end
        end
    end

    # `converged` is appended as the LAST return value (#238/H-17). Julia drops
    # trailing tuple elements, so existing `c_pol, a_pol = _egm_solve(...)` and
    # `_, a_pol = _egm_solve(...)` call sites keep working unchanged; callers that
    # want the flag destructure the third element.
    return c_pol, a_pol, converged
end

# =============================================================================
# Two-asset nested EGM (Auclert et al. 2021 style, modified policy iteration)
# =============================================================================

"""
    _two_asset_egm_solve(ip, grid, income, prices; max_iter=1000, tol=1e-8,
                         n_deposit=30, howard_steps=30)
        → Dict{Symbol,Array{T}}

Solve a two-asset household problem to a *converged* stationary policy over the
joint liquid/illiquid state `(b, a, e)`.

The household chooses consumption `c`, next-period liquid savings `b'` and a
deposit `d` into the illiquid asset (illiquid law of motion
`a' = (1+r_a) a + d`), paying a portfolio adjustment cost `χ(d, a)`:

    max_{c, b', d}  u(c) + β E[V(b', a', e') | e]
    s.t.  c + b' + d + χ(d, a) = budget_fn(b, a, e, prices),  b' ≥ b_min.

# Algorithm (Howard-accelerated nested value-function iteration)
The illiquid deposit is chosen by searching the illiquid grid; for each candidate
`a'` (deposit `d = a' − (1+r_a) a`) the *liquid* choice is solved by the
endogenous grid method on the continuation-value marginal `∂EV/∂b'`
(`= β(1+r_b) E[u'(c')]` by the envelope), giving a continuous `b'` and a tight
liquid Euler residual. The deposit is then selected by maximising the total value
`u(c) + β E[V(b', a', e')]` with `c = budget_fn(b, a, e) − b' − d − χ(d, a)`.
Because utility penalises low consumption (`u(c) → −∞` as `c → 0`), the value
comparison never selects the degenerate "deposit everything, consume nothing"
corner; and because the EGM driver is the smooth continuation-value marginal
(not the consumption policy), it cannot self-reinforce into that corner the way an
iterated policy-marginal nested EGM does. `howard_steps` rounds of policy
evaluation under the fixed policy accelerate convergence (β is close to 1). The
maximiser yields a genuine *state-dependent* deposit `d(b, a, e)` (the optimal
`a'` varies with liquid wealth `b`).

Returns a `Dict` with keys `:consumption`, `:liquid_savings`, `:deposit`,
`:value`, `:converged` — the first three each an `n_b × n_a × n_e` array
(liquid × illiquid × income); `:value` the converged value function;
`:converged` a 0/1 flag stored as `T[flag]`.

The `n_deposit` keyword is retained for API compatibility; deposits are now
searched over the full illiquid grid.
"""
function _two_asset_egm_solve(ip::IndividualProblem{T}, grid::HAGrid{T},
                               income::IncomeProcess{T}, prices::Dict{Symbol,T};
                               max_iter::Int=1000, tol::T=T(1e-8),
                               n_deposit::Int=30,
                               howard_steps::Int=30) where {T<:AbstractFloat}
    @assert ip.n_asset_dims == 2 "Two-asset EGM requires n_asset_dims == 2"
    @assert grid.n_dims == 2 "Two-asset EGM requires a two-dimensional grid"

    b_grid = grid.grids[1]   # liquid
    a_grid = grid.grids[2]   # illiquid
    n_b = length(b_grid)
    n_a = length(a_grid)
    n_e = length(income.states)
    b_min = ip.borrowing_constraint[1]

    beta = ip.beta
    u = ip.utility
    Pi = income.transition
    e_vals = income.states

    r_a = get(prices, :r_a, prices[:r])  # illiquid return

    # Adjustment cost function (default: zero); route resources through budget_fn
    # (net-income hook, #235) so `div` and any price offsets are honoured.
    adj_cost = isnothing(ip.adjustment_cost) ? (d, a) -> zero(T) : ip.adjustment_cost
    budget_fn = ip.budget_fn

    NEG = T(-1e20)  # sentinel value for infeasible choices

    # Policy + value arrays over the joint state (b, a, e)
    c_opt   = zeros(T, n_b, n_a, n_e)
    b_opt   = zeros(T, n_b, n_a, n_e)   # next-period liquid b'
    d_opt   = zeros(T, n_b, n_a, n_e)   # deposit d
    iap_opt = ones(Int, n_b, n_a, n_e)  # chosen illiquid index a' (for Howard eval)
    V       = zeros(T, n_b, n_a, n_e)

    # Precompute deposit d(a', a) and adjustment cost χ(d, a) for every (ia', ia)
    d_tab   = zeros(T, n_a, n_a)   # [iap, ia]
    chi_tab = zeros(T, n_a, n_a)
    for ia in 1:n_a, iap in 1:n_a
        dv = a_grid[iap] - (one(T) + r_a) * a_grid[ia]
        d_tab[iap, ia] = dv
        chi_tab[iap, ia] = adj_cost(dv, a_grid[ia])
    end

    # Initialisation: consume half of a rough resource level; V = u(c)/(1-β).
    for je in 1:n_e, ia in 1:n_a, ib in 1:n_b
        coh = budget_fn(b_grid[ib], a_grid[ia], e_vals[je], prices)
        c0 = max(coh * T(0.5), T(1e-8))
        c_opt[ib, ia, je] = c0
        b_opt[ib, ia, je] = b_grid[ib]
        V[ib, ia, je] = u(c0) / (one(T) - beta)
    end

    u_prime_inv = ip.utility_prime_inv

    # Scratch buffers
    EV      = zeros(T, n_b, n_a, n_e)   # β E[V(b',a',e')|e] over (b', a', e)
    c_new   = zeros(T, n_b, n_a, n_e)
    b_new   = zeros(T, n_b, n_a, n_e)
    d_new   = zeros(T, n_b, n_a, n_e)
    iap_new = ones(Int, n_b, n_a, n_e)
    V_new   = zeros(T, n_b, n_a, n_e)
    V_hnew  = zeros(T, n_b, n_a, n_e)
    dEV_col = zeros(T, n_b)            # ∂EV/∂b' for a fixed (a', e)
    c_endo  = zeros(T, n_b)
    x_endo  = zeros(T, n_b)

    converged = false

    for iter in 1:max_iter
        # Continuation EV[ib', ia', je] = β Σ_{jep} Pi[je,jep] V[ib',ia',jep]
        fill!(EV, zero(T))
        for je in 1:n_e, jep in 1:n_e
            wgt = beta * Pi[je, jep]
            @inbounds for ia in 1:n_a, ib in 1:n_b
                EV[ib, ia, je] += wgt * V[ib, ia, jep]
            end
        end

        # ---- Policy improvement ----
        # Illiquid deposit a' is searched over the grid; for each candidate a'
        # the liquid choice is solved by EGM on the *continuation-value* marginal
        # ∂EV/∂b' (= β(1+r_b) E[u'(c')] by the envelope), giving a continuous b'
        # and a tight liquid Euler. The deposit is then chosen by comparing the
        # total value u(c) + EV(b',a',e). Because u(c) → −∞ as c → 0, the value
        # comparison never selects the degenerate "deposit everything, consume
        # nothing" corner; because the EGM driver is the smooth value marginal
        # (not the consumption policy), it cannot self-reinforce into that corner
        # the way an iterated policy-marginal nested EGM does.
        fill!(V_new, NEG)
        for je in 1:n_e
            for iap in 1:n_a
                EVcol = view(EV, :, iap, je)
                # ∂EV/∂b' via finite differences on the liquid grid
                @inbounds for ibp in 1:n_b
                    if ibp == 1
                        dEV_col[ibp] = (EVcol[2] - EVcol[1]) / (b_grid[2] - b_grid[1])
                    elseif ibp == n_b
                        dEV_col[ibp] = (EVcol[n_b] - EVcol[n_b-1]) / (b_grid[n_b] - b_grid[n_b-1])
                    else
                        dEV_col[ibp] = (EVcol[ibp+1] - EVcol[ibp-1]) / (b_grid[ibp+1] - b_grid[ibp-1])
                    end
                    ce = u_prime_inv(max(dEV_col[ibp], T(1e-12)))  # u'(c) = ∂EV/∂b'
                    c_endo[ibp] = ce
                    x_endo[ibp] = ce + b_grid[ibp]
                end
                @inbounds for ia in 1:n_a
                    a_val = a_grid[ia]
                    avail_const = -d_tab[iap, ia] - chi_tab[iap, ia]
                    for ib in 1:n_b
                        avail = budget_fn(b_grid[ib], a_val, e_vals[je], prices) + avail_const
                        avail <= b_min && continue      # cannot fund b_min with c > 0
                        if avail <= x_endo[1]
                            bprime = b_min
                            c = avail - b_min
                        else
                            c = _linear_interp(x_endo, c_endo, avail)
                            bprime = avail - c
                        end
                        c <= zero(T) && continue
                        val = u(c) + _linear_interp(b_grid, EVcol, bprime)
                        if val > V_new[ib, ia, je]
                            V_new[ib, ia, je] = val
                            c_new[ib, ia, je] = c
                            b_new[ib, ia, je] = bprime
                            d_new[ib, ia, je] = d_tab[iap, ia]
                            iap_new[ib, ia, je] = iap
                        end
                    end
                end
            end
        end

        # Fallback for any state with no feasible deposit (keeps arrays finite)
        @inbounds for je in 1:n_e, ia in 1:n_a, ib in 1:n_b
            if V_new[ib, ia, je] <= NEG / 2
                resources = budget_fn(b_grid[ib], a_grid[ia], e_vals[je], prices)
                c = max(resources - b_min, T(1e-10))
                c_new[ib, ia, je] = c
                b_new[ib, ia, je] = b_min
                d_new[ib, ia, je] = zero(T)
                iap_new[ib, ia, je] = ia
                V_new[ib, ia, je] = u(c) + EV[1, ia, je]
            end
        end

        # Count discrete deposit-choice changes (for a policy-stability stop).
        policy_changes = 0
        @inbounds for idx in eachindex(iap_new)
            iap_new[idx] != iap_opt[idx] && (policy_changes += 1)
        end

        copyto!(c_opt, c_new)
        copyto!(b_opt, b_new)
        copyto!(d_opt, d_new)
        copyto!(iap_opt, iap_new)

        # Convergence on the value function (sup-norm of the Bellman update).
        # The discrete deposit choice can oscillate between near-tied a' values,
        # so a consumption-policy criterion may never settle; the value converges
        # monotonically (VFI contraction) and equals across tied choices. As a
        # backstop, a fully stable discrete policy (no a' changes) is also a fixed
        # point (the continuous EGM/Howard steps are then deterministic).
        max_diff = zero(T)
        @inbounds for idx in eachindex(V_new)
            diff = abs(V_new[idx] - V[idx])
            if diff > max_diff
                max_diff = diff
            end
        end
        copyto!(V, V_new)

        # ---- Howard policy evaluation under the fixed policy ----
        for _ in 1:howard_steps
            @inbounds for je in 1:n_e, ia in 1:n_a, ib in 1:n_b
                iap = iap_opt[ib, ia, je]
                bp = b_opt[ib, ia, je]
                cont = zero(T)
                for jep in 1:n_e
                    cont += Pi[je, jep] * _linear_interp(b_grid, view(V, :, iap, jep), bp)
                end
                V_hnew[ib, ia, je] = u(c_opt[ib, ia, je]) + beta * cont
            end
            copyto!(V, V_hnew)
        end

        if max_diff < tol || (policy_changes == 0 && iter > 1)
            converged = true
            break
        end
    end

    flag = converged ? one(T) : zero(T)
    return Dict{Symbol,Array{T}}(
        :consumption   => c_opt,
        :liquid_savings => b_opt,
        :deposit       => d_opt,
        :value         => V,
        :converged     => T[flag]
    )
end
