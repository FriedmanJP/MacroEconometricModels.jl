# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
DCEGM — the endogenous grid method for discrete-continuous dynamic choice.

The plain endogenous grid method (`_egm_solve`) solves a purely continuous
consumption-savings problem. Adding a *discrete* choice — retire or work, own or
rent, adjust or not — makes the value function non-concave, so the Euler equation
is no longer sufficient: it admits spurious local optima, and the endogenous grid
it produces is non-monotone. Iskhakov, Jørgensen, Rust & Schjerning (2017) solve
this with an **upper-envelope** step that deletes the suboptimal branches of the
endogenous-grid consumption correspondence and locates the exact switching
threshold in the state.

This file provides

1. `_upper_envelope` — the core segment-deletion routine, usable and testable
   in isolation,
2. `DCEGMProblem` — a discrete-continuous household problem,
3. `dcegm_solve` — backward induction (finite horizon) or fixed-point iteration
   (infinite horizon) with per-option EGM plus the upper envelope, optionally
   smoothed by extreme-value taste shocks,
4. `dcegm_simulate` — a Young (2010) histogram whose transition respects the
   discrete choice.

# References
- Iskhakov, F., Jørgensen, T. H., Rust, J., & Schjerning, B. (2017). The
  endogenous grid method for discrete-continuous dynamic choice models with
  (or without) taste shocks. *Quantitative Economics*, 8(2), 317–365.
- Carroll, C. D. (2006). The method of endogenous gridpoints for solving dynamic
  stochastic optimization problems. *Economics Letters*, 91(3), 312–320.
- Young, E. R. (2010). Solving the incomplete markets model with aggregate
  uncertainty using the Krusell-Smith algorithm and non-stochastic simulations.
  *Journal of Economic Dynamics and Control*, 34(1), 36–41.
"""

using LinearAlgebra

# =============================================================================
# Upper envelope — the core DCEGM step
# =============================================================================

"Linear interpolation on a strictly increasing segment; the caller checks support."
function _seg_interp(x::AbstractVector{T}, y::AbstractVector{T}, xi::T) where {T<:AbstractFloat}
    n = length(x)
    xi <= x[1] && return y[1]
    xi >= x[n] && return y[n]
    k = clamp(searchsortedfirst(x, xi) - 1, 1, n - 1)
    dx = x[k+1] - x[k]
    dx <= zero(T) && return y[k]
    return y[k] + (xi - x[k]) / dx * (y[k+1] - y[k])
end

"""
    _monotone_segments(M) -> Vector{UnitRange{Int}}

Split the endogenous grid `M` into maximal runs that are strictly increasing.
A run of a single point carries no interpolable information and is dropped.

Each descent `M[i] ≤ M[i-1]` marks a *loop* in the endogenous-grid
correspondence: the Euler equation has produced two candidate consumption levels
for the same cash-on-hand, only one of which is globally optimal.
"""
function _monotone_segments(M::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(M)
    segs = UnitRange{Int}[]
    n == 0 && return segs
    i0 = 1
    @inbounds for i in 2:n
        if M[i] <= M[i-1]
            i - 1 > i0 && push!(segs, i0:(i-1))
            i0 = i
        end
    end
    n > i0 && push!(segs, i0:n)
    return segs
end

"""
    _upper_envelope(M, c, v) -> (M_env, c_env, v_env, n_kinks)

Upper envelope of a non-monotone endogenous-grid correspondence.

`M`, `c`, `v` are the cash-on-hand, consumption, and value produced by one EGM
sweep over the exogenous post-decision asset grid. When the discrete choice makes
the value function non-concave, `M` is non-monotone: the same cash-on-hand appears
on two or more branches, and the Euler equation cannot tell which is optimal.

The routine splits the correspondence into monotone segments, evaluates every
segment's value function on the union of all grid points, and keeps the maximizer.
Where the identity of the maximizing segment changes between two adjacent grid
points, both segments are linear on that interval (the grid contains every knot),
so the crossing

    M* = M_lo + (M_hi − M_lo) · f(M_lo) / (f(M_lo) − f(M_hi)),   f = v_p − v_q

is exact. `M*` is inserted **twice** — at `M*` with the left branch's consumption
and at `nextfloat(M*)` with the right branch's — because consumption jumps
discontinuously at a discrete-choice threshold while the value function does not.

# Returns
- `M_env`, `c_env`, `v_env` — the envelope, strictly increasing in `M_env`
- `n_kinks::Int` — number of switching thresholds found

# References
- Iskhakov, F., Jørgensen, T. H., Rust, J., & Schjerning, B. (2017).
  *Quantitative Economics*, 8(2), 317–365.
"""
function _upper_envelope(M::AbstractVector{T}, c::AbstractVector{T},
                         v::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(M)
    (length(c) == n && length(v) == n) ||
        throw(ArgumentError("_upper_envelope: M, c, v must have equal length"))
    n >= 2 || return (collect(T, M), collect(T, c), collect(T, v), 0)

    segs = _monotone_segments(M)
    isempty(segs) && return (collect(T, M), collect(T, c), collect(T, v), 0)
    if length(segs) == 1
        r = segs[1]
        return (collect(T, M[r]), collect(T, c[r]), collect(T, v[r]), 0)
    end

    lo = T[M[r[1]] for r in segs]
    hi = T[M[r[end]] for r in segs]
    covers(s, x) = lo[s] <= x <= hi[s]

    grid = sort!(unique(collect(T, M)))
    M_env = T[]; c_env = T[]; v_env = T[]
    prev = 0
    prev_m = zero(T)
    kinks = 0

    for m in grid
        best_v = T(-Inf); best_c = zero(T); best = 0
        for (s, r) in enumerate(segs)
            covers(s, m) || continue
            vs = _seg_interp(view(M, r), view(v, r), m)
            if vs > best_v
                best_v = vs
                best_c = _seg_interp(view(M, r), view(c, r), m)
                best = s
            end
        end
        best == 0 && continue

        if prev != 0 && best != prev && covers(prev, m) && covers(best, prev_m)
            rp = segs[prev]; rq = segs[best]
            f_lo = _seg_interp(view(M, rp), view(v, rp), prev_m) -
                   _seg_interp(view(M, rq), view(v, rq), prev_m)
            f_hi = _seg_interp(view(M, rp), view(v, rp), m) -
                   _seg_interp(view(M, rq), view(v, rq), m)
            den = f_lo - f_hi
            if den != zero(T)
                m_star = prev_m + (m - prev_m) * f_lo / den
                if m_star <= prev_m
                    # The branches tie exactly at the previous knot, which is already
                    # in the envelope carrying the left branch. Consumption jumps
                    # immediately above it, so one point completes the kink.
                    push!(M_env, nextfloat(prev_m))
                    push!(c_env, _seg_interp(view(M, rq), view(c, rq), prev_m))
                    push!(v_env, _seg_interp(view(M, rq), view(v, rq), prev_m))
                    kinks += 1
                elseif m_star < m
                    push!(M_env, m_star)
                    push!(c_env, _seg_interp(view(M, rp), view(c, rp), m_star))
                    push!(v_env, _seg_interp(view(M, rp), view(v, rp), m_star))
                    push!(M_env, nextfloat(m_star))
                    push!(c_env, _seg_interp(view(M, rq), view(c, rq), m_star))
                    push!(v_env, _seg_interp(view(M, rq), view(v, rq), m_star))
                    kinks += 1
                end
            end
        end

        push!(M_env, m); push!(c_env, best_c); push!(v_env, best_v)
        prev = best; prev_m = m
    end

    return (M_env, c_env, v_env, kinks)
end

# =============================================================================
# DCEGMProblem
# =============================================================================

"""
    DCEGMProblem{T}

A discrete-continuous consumption-savings problem in the sense of Iskhakov,
Jørgensen, Rust & Schjerning (2017). Each period the household picks a discrete
option `d` from those its previous choice leaves feasible and a consumption level
`c`, subject to a credit limit:

```math
V_t(M, j, d_-) = \\max_{d \\in D(d_-)} v_t(M, j, d), \\qquad
v_t(M, j, d) = \\max_{c} u(c, d) + \\beta\\, E_j V_{t+1}(R(M - c) + y(d, j'), j', d)
```

where `M` is cash-on-hand, `j` the income state, and `d_-` last period's option.

# Constructor

    DCEGMProblem(; beta, R, utility, utility_prime, utility_prime_inv, income,
                   options, absorbing, asset_grid, income_process,
                   n_periods=0, taste_shock_scale=0.0, credit_limit=0.0)

| Field | Type | Description |
|---|---|---|
| `beta` | `T` | Discount factor |
| `R` | `T` | Gross return on savings |
| `utility` | Function | `u(c, d)` — `d` is the **integer index** into `options` |
| `utility_prime` | Function | `∂u/∂c (c, d)` |
| `utility_prime_inv` | Function | Inverse of `∂u/∂c` in its first argument: `(m, d) -> c` |
| `income` | Function | `y(d, j)` — income received next period after choosing `d`, in next period's income state `j` |
| `options` | `Vector{Symbol}` | Discrete alternatives |
| `absorbing` | `Vector{Bool}` | Option `k` absorbing ⇒ once chosen only `k` stays feasible |
| `asset_grid` | `Vector{T}` | Exogenous post-decision asset grid (must start at `credit_limit`) |
| `income_process` | `IncomeProcess{T}` | Idiosyncratic income states and transition |
| `n_periods` | `Int` | Life-cycle length; `0` ⇒ infinite horizon (stationary policy) |
| `taste_shock_scale` | `T` | Extreme-value taste-shock scale `λ`; `0` ⇒ deterministic upper envelope |
| `credit_limit` | `T` | Lower bound on end-of-period assets |

The utility functions take the option **index**, not its symbol, so they stay
allocation-free inside the backward loop.
"""
struct DCEGMProblem{T<:AbstractFloat,FU,FUP,FUPI,FY}
    beta::T
    R::T
    utility::FU
    utility_prime::FUP
    utility_prime_inv::FUPI
    income::FY
    options::Vector{Symbol}
    absorbing::Vector{Bool}
    asset_grid::Vector{T}
    income_process::IncomeProcess{T}
    n_periods::Int
    taste_shock_scale::T
    credit_limit::T
end

function DCEGMProblem(; beta::Real, R::Real, utility, utility_prime,
                        utility_prime_inv, income,
                        options::AbstractVector{Symbol},
                        absorbing::AbstractVector{Bool},
                        asset_grid::AbstractVector{<:Real},
                        income_process::IncomeProcess{T},
                        n_periods::Int=0,
                        taste_shock_scale::Real=0.0,
                        credit_limit::Real=0.0) where {T<:AbstractFloat}
    opts = collect(Symbol, options)
    abs_ = collect(Bool, absorbing)
    ag = collect(T, asset_grid)
    isempty(opts) && throw(ArgumentError("DCEGMProblem needs at least one discrete option"))
    length(abs_) == length(opts) || throw(ArgumentError(
        "`absorbing` has $(length(abs_)) entries but there are $(length(opts)) options"))
    length(unique(opts)) == length(opts) || throw(ArgumentError(
        "DCEGMProblem has duplicate options: $opts"))
    length(ag) >= 2 || throw(ArgumentError("`asset_grid` needs at least 2 points"))
    issorted(ag) || throw(ArgumentError("`asset_grid` must be sorted ascending"))
    n_periods >= 0 || throw(ArgumentError("`n_periods` must be non-negative (0 ⇒ infinite horizon)"))
    taste_shock_scale >= 0 || throw(ArgumentError("`taste_shock_scale` must be non-negative"))
    0 < beta < 1 || throw(ArgumentError("`beta` must lie in (0, 1), got $beta"))
    isapprox(ag[1], T(credit_limit); atol=sqrt(eps(T))) || throw(ArgumentError(
        "`asset_grid` must start at the credit limit $(credit_limit), got $(ag[1])"))
    return DCEGMProblem{T,typeof(utility),typeof(utility_prime),
                        typeof(utility_prime_inv),typeof(income)}(
        T(beta), T(R), utility, utility_prime, utility_prime_inv, income,
        opts, abs_, ag, income_process, n_periods, T(taste_shock_scale), T(credit_limit))
end

"Options still feasible after choosing option `d` last period."
_dcegm_feasible(prob::DCEGMProblem, d::Int) =
    prob.absorbing[d] ? (d:d) : (1:length(prob.options))

# =============================================================================
# DCEGMSolution
# =============================================================================

"""
    DCEGMSolution{T}

Solution of a [`DCEGMProblem`](@ref).

Policies are stored per period, per discrete option, and per income state on the
option's own **endogenous** cash-on-hand grid, which the upper envelope leaves
non-uniform and generally different across options.

| Field | Type | Description |
|---|---|---|
| `M` | `Array{Vector{T},3}` | `M[t, d, j]` — endogenous cash-on-hand grid |
| `c` | `Array{Vector{T},3}` | `c[t, d, j]` — consumption on that grid |
| `v` | `Array{Vector{T},3}` | `v[t, d, j]` — conditional value `v_t(M, j, d)` |
| `ev_constrained` | `Array{T,3}` | Continuation value at the credit limit, for the constrained branch |
| `n_kinks` | `Array{Int,3}` | Switching thresholds the envelope found per `(t, d, j)` |
| `prob` | `DCEGMProblem{T}` | Problem solved |
| `n_periods` | `Int` | Stored periods (`1` for the stationary infinite-horizon policy) |
| `converged` | `Bool` | Finite horizon: always `true` (backward induction is exact). Infinite horizon: whether the policy fixed point met `tol` |
| `iterations` | `Int` | Infinite-horizon iterations used (`0` for finite horizon) |
| `sup_diff` | `T` | Final sup-norm policy change (infinite horizon) |
"""
struct DCEGMSolution{T<:AbstractFloat}
    M::Array{Vector{T},3}
    c::Array{Vector{T},3}
    v::Array{Vector{T},3}
    ev_constrained::Array{T,3}
    n_kinks::Array{Int,3}
    prob::DCEGMProblem{T}
    n_periods::Int
    converged::Bool
    iterations::Int
    sup_diff::T
end

"""
    dcegm_policy(sol, t, d, j, M) -> (c, v)

Consumption and conditional value of option `d` in income state `j` at cash-on-hand
`M`, period `t`.

Below the smallest endogenous grid point the household is **credit constrained**:
it saves exactly the credit limit and consumes the rest, so consumption and value
are evaluated analytically rather than extrapolated. Above it, the envelope is
interpolated linearly.
"""
function dcegm_policy(sol::DCEGMSolution{T}, t::Int, d::Int, j::Int, M::Real) where {T<:AbstractFloat}
    prob = sol.prob
    Mv = sol.M[t, d, j]
    m = T(M)
    if isempty(Mv) || m <= Mv[1]
        c = m - prob.credit_limit
        c <= zero(T) && return (zero(T), T(-Inf))
        return (c, prob.utility(c, d) + prob.beta * sol.ev_constrained[t, d, j])
    end
    return (_seg_interp(Mv, sol.c[t, d, j], m), _seg_interp(Mv, sol.v[t, d, j], m))
end

"""
    dcegm_choice_probabilities(sol, t, d_prev, j, M) -> Vector{T}

Probability of each discrete option at `(t, M, j)` given last period's option
`d_prev`. Infeasible options get probability zero.

With taste shocks (`taste_shock_scale = λ > 0`) these are the multinomial-logit
probabilities `exp(v_d/λ) / Σ exp(v_d'/λ)`, computed with the log-sum-exp shift.
With `λ = 0` the vector is degenerate at the argmax — the deterministic switching
rule the upper envelope pins down.
"""
function dcegm_choice_probabilities(sol::DCEGMSolution{T}, t::Int, d_prev::Int,
                                    j::Int, M::Real) where {T<:AbstractFloat}
    prob = sol.prob
    n_d = length(prob.options)
    feas = _dcegm_feasible(prob, d_prev)
    vals = fill(T(-Inf), n_d)
    for d in feas
        vals[d] = dcegm_policy(sol, t, d, j, M)[2]
    end
    return _dcegm_softmax(vals, prob.taste_shock_scale)
end

"Multinomial-logit weights over `vals` at scale `lambda` (`lambda = 0` ⇒ argmax)."
function _dcegm_softmax(vals::Vector{T}, lambda::T) where {T<:AbstractFloat}
    n = length(vals)
    p = zeros(T, n)
    vmax = maximum(vals)
    if !isfinite(vmax)
        p[argmax(vals)] = one(T)
        return p
    end
    if lambda <= zero(T)
        best = 1; bv = T(-Inf)
        for k in 1:n
            vals[k] > bv && (bv = vals[k]; best = k)
        end
        p[best] = one(T)
        return p
    end
    s = zero(T)
    @inbounds for k in 1:n
        if isfinite(vals[k])
            p[k] = exp((vals[k] - vmax) / lambda)
            s += p[k]
        end
    end
    s > zero(T) ? (p ./= s) : (p[argmax(vals)] = one(T))
    return p
end

"""
    _dcegm_value_and_slope(sol, t, d_prev, j, M) -> (V, dV_dM)

Smoothed value `V_t(M, j, d_prev)` and its derivative.

With taste shocks `V = λ log Σ_d exp(v_d/λ)`; with `λ = 0` it is the maximum. The
envelope theorem gives `dV/dM = Σ_d P(d) · u_c(c_d(M), d)` in both cases — the
identity that makes the smoothed value function differentiable and lets the
next EGM step use an ordinary Euler equation (Iskhakov et al. 2017, §4).
"""
function _dcegm_value_and_slope(sol::DCEGMSolution{T}, t::Int, d_prev::Int,
                                j::Int, M::T) where {T<:AbstractFloat}
    prob = sol.prob
    n_d = length(prob.options)
    feas = _dcegm_feasible(prob, d_prev)
    vals = fill(T(-Inf), n_d)
    cs = zeros(T, n_d)
    for d in feas
        cd, vd = dcegm_policy(sol, t, d, j, M)
        cs[d] = cd
        vals[d] = vd
    end
    lambda = prob.taste_shock_scale
    p = _dcegm_softmax(vals, lambda)

    vmax = maximum(vals)
    V = if !isfinite(vmax) || lambda <= zero(T)
        vmax
    else
        s = zero(T)
        for d in feas
            isfinite(vals[d]) && (s += exp((vals[d] - vmax) / lambda))
        end
        vmax + lambda * log(s)
    end

    dV = zero(T)
    for d in feas
        p[d] > zero(T) || continue
        if cs[d] <= zero(T)
            # Cash-on-hand has hit the credit limit: consumption goes to zero and the
            # marginal value of a further unit of wealth diverges. Returning 0 here
            # (the opposite limit) inverts the Euler equation into c = ∞ and poisons
            # the whole backward recursion with Inf/NaN.
            return (V, T(Inf))
        end
        dV += p[d] * prob.utility_prime(cs[d], d)
    end
    return (V, dV)
end

# =============================================================================
# dcegm_solve
# =============================================================================

"""
    dcegm_solve(prob; max_iter=500, tol=1e-8, verbose=false) -> DCEGMSolution

Solve a [`DCEGMProblem`](@ref) by the endogenous grid method with an upper
envelope (Iskhakov, Jørgensen, Rust & Schjerning 2017).

For each period, discrete option `d`, and income state `j`, one EGM sweep over the
exogenous post-decision asset grid gives candidate `(M, c, v)` triples from the
Euler equation

```math
u_c(c, d) = \\beta R\\, E_j\\!\\left[\\sum_{d'} P(d' \\mid M') \\, u_c(c'(M', d'), d')\\right]
```

Because the discrete choice makes the continuation value non-concave, the
resulting `M` sequence is non-monotone; `_upper_envelope` deletes the
suboptimal branches and inserts the exact switching thresholds.

With `n_periods > 0` the solver runs backward induction from a terminal period in
which the household consumes all cash-on-hand. With `n_periods == 0` it iterates
the same backward step to a stationary fixed point and reports convergence.

# Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `max_iter` | `Int` | `500` | Infinite-horizon iteration cap |
| `tol` | `Real` | ``10^{-8}`` | Sup-norm policy tolerance (infinite horizon) |
| `verbose` | `Bool` | `false` | Print iteration diagnostics |

# Returns
[`DCEGMSolution{T}`](@ref).

# References
- Iskhakov, F., Jørgensen, T. H., Rust, J., & Schjerning, B. (2017). The
  endogenous grid method for discrete-continuous dynamic choice models with
  (or without) taste shocks. *Quantitative Economics*, 8(2), 317–365.
"""
function dcegm_solve(prob::DCEGMProblem{T}; max_iter::Int=500, tol::Real=1e-8,
                     verbose::Bool=false) where {T<:AbstractFloat}
    n_d = length(prob.options)
    n_e = length(prob.income_process.states)
    finite = prob.n_periods > 0
    n_store = finite ? prob.n_periods : 1

    M = Array{Vector{T},3}(undef, n_store, n_d, n_e)
    c = Array{Vector{T},3}(undef, n_store, n_d, n_e)
    v = Array{Vector{T},3}(undef, n_store, n_d, n_e)
    ev_c = zeros(T, n_store, n_d, n_e)
    kinks = zeros(Int, n_store, n_d, n_e)
    sol = DCEGMSolution{T}(M, c, v, ev_c, kinks, prob, n_store, true, 0, zero(T))

    # ── Terminal period (or the infinite-horizon seed): consume everything ───
    # Assets are run down to the credit limit, so c = M − ā on every grid point and
    # the value is the flow utility alone. `ev_constrained = 0` makes the analytic
    # constrained branch of `dcegm_policy` agree with this exactly below the grid.
    a_grid = prob.asset_grid
    term = n_store
    Mgrid = collect(T, a_grid)
    ct = Mgrid .- prob.credit_limit
    ct[1] = max(ct[1], eps(T))          # the grid starts at ā, where c = 0 is not loggable
    for d in 1:n_d, j in 1:n_e
        M[term, d, j] = copy(Mgrid)
        c[term, d, j] = copy(ct)
        v[term, d, j] = T[prob.utility(x, d) for x in ct]
        ev_c[term, d, j] = zero(T)
        kinks[term, d, j] = 0
    end

    Pi = prob.income_process.transition
    converged = true
    iters = 0
    diff = zero(T)

    if finite
        for t in (prob.n_periods - 1):-1:1
            _dcegm_backward!(sol, t, t + 1, Pi)
        end
    else
        converged = false
        prev_c = [copy(c[1, d, j]) for d in 1:n_d, j in 1:n_e]
        prev_M = [copy(M[1, d, j]) for d in 1:n_d, j in 1:n_e]
        for it in 1:max_iter
            iters = it
            _dcegm_backward!(sol, 1, 1, Pi)
            diff = zero(T)
            for d in 1:n_d, j in 1:n_e
                # Compare on the previous grid: the envelope changes the grid itself.
                for (k, m) in enumerate(prev_M[d, j])
                    diff = max(diff, abs(dcegm_policy(sol, 1, d, j, m)[1] - prev_c[d, j][k]))
                end
                prev_M[d, j] = copy(M[1, d, j])
                prev_c[d, j] = copy(c[1, d, j])
            end
            verbose && println("DCEGM iter $it: sup|Δc| = $diff")
            if diff < T(tol)
                converged = true
                break
            end
        end
    end

    return DCEGMSolution{T}(M, c, v, ev_c, kinks, prob, n_store, converged, iters, diff)
end

"""
    _dcegm_backward!(sol, t, t_next, Pi)

One DCEGM backward step: fill period `t`'s policies from period `t_next`'s, for
every discrete option and income state. Writes in place into `sol`.

Reading and writing the same slice (`t == t_next`) is the infinite-horizon fixed
point; the candidate arrays are built in full before any slice is overwritten, so
the update is well defined.
"""
function _dcegm_backward!(sol::DCEGMSolution{T}, t::Int, t_next::Int,
                          Pi::AbstractMatrix{T}) where {T<:AbstractFloat}
    prob = sol.prob
    n_d = length(prob.options)
    n_e = length(prob.income_process.states)
    a_grid = prob.asset_grid
    n_a = length(a_grid)
    beta, R = prob.beta, prob.R

    new_M = Array{Vector{T},2}(undef, n_d, n_e)
    new_c = Array{Vector{T},2}(undef, n_d, n_e)
    new_v = Array{Vector{T},2}(undef, n_d, n_e)
    new_ev = zeros(T, n_d, n_e)
    new_k = zeros(Int, n_d, n_e)

    Mc = zeros(T, n_a); cc = zeros(T, n_a); vc = zeros(T, n_a)

    for d in 1:n_d, j in 1:n_e
        for (i, a) in enumerate(a_grid)
            EV = zero(T); EdV = zero(T)
            for jp in 1:n_e
                w = Pi[j, jp]
                w == zero(T) && continue
                Mp = R * a + T(prob.income(d, jp))
                Vp, dVp = _dcegm_value_and_slope(sol, t_next, d, jp, Mp)
                EV += w * Vp
                EdV += w * dVp
            end
            rhs = beta * R * EdV
            ci = rhs > zero(T) ? T(prob.utility_prime_inv(rhs, d)) : T(Inf)
            cc[i] = ci
            Mc[i] = a + ci
            vc[i] = prob.utility(ci, d) + beta * EV
            i == 1 && (new_ev[d, j] = EV)     # continuation at the credit limit
        end
        # Drop degenerate candidates before the envelope. Saving exactly the credit
        # limit can imply zero consumption forever (a retiree with no pension), whose
        # value is genuinely −∞; such a point carries no interpolable information and
        # would otherwise contaminate the envelope's max and every later interpolation.
        keep = [i for i in 1:n_a if isfinite(Mc[i]) && isfinite(cc[i]) && isfinite(vc[i])]
        length(keep) >= 2 || error(
            "DCEGM: option :$(prob.options[d]) in income state $j left " *
            "$(length(keep)) usable grid points. Widen `asset_grid`, relax " *
            "`credit_limit`, or give the option a positive income floor.")
        Me, ce, ve, nk = _upper_envelope(Mc[keep], cc[keep], vc[keep])
        new_M[d, j] = Me; new_c[d, j] = ce; new_v[d, j] = ve; new_k[d, j] = nk
    end

    for d in 1:n_d, j in 1:n_e
        sol.M[t, d, j] = new_M[d, j]
        sol.c[t, d, j] = new_c[d, j]
        sol.v[t, d, j] = new_v[d, j]
        sol.ev_constrained[t, d, j] = new_ev[d, j]
        sol.n_kinks[t, d, j] = new_k[d, j]
    end
    return sol
end

"""
    dcegm_threshold(sol, t, d_prev, j; M_lo, M_hi, tol=1e-10) -> T

Cash-on-hand at which the optimal discrete choice switches, found by bisection on
the conditional-value difference over `[M_lo, M_hi]`.

Returns `NaN` when the two options do not cross on the bracket — either one
dominates everywhere, or the bracket is too narrow. Only meaningful with exactly
two feasible options; with taste shocks it locates the cash-on-hand at which the
choice probabilities are equal.

Bracketing is on the **sign** of the value gap, not on its finiteness: just above
a credit limit that implies zero consumption forever the gap is `-Inf`, which is a
perfectly well-signed endpoint and a common one (a retiree with no pension).
Requiring a finite endpoint there would report "no threshold" for models that
plainly have one.
"""
function dcegm_threshold(sol::DCEGMSolution{T}, t::Int, d_prev::Int, j::Int;
                         M_lo::Real, M_hi::Real, tol::Real=1e-10) where {T<:AbstractFloat}
    feas = collect(_dcegm_feasible(sol.prob, d_prev))
    length(feas) == 2 || throw(ArgumentError(
        "dcegm_threshold needs exactly two feasible options after :$(sol.prob.options[d_prev]) " *
        "(got $(length(feas)))"))
    d1, d2 = feas[1], feas[2]
    function gap(m)
        v1 = dcegm_policy(sol, t, d1, j, m)[2]
        v2 = dcegm_policy(sol, t, d2, j, m)[2]
        (v1 == v2) && return zero(T)          # both −Inf ⇒ indifferent, not NaN
        return v1 - v2
    end
    a = T(M_lo); b = T(M_hi)
    a < b || throw(ArgumentError("dcegm_threshold needs M_lo < M_hi (got $M_lo, $M_hi)"))
    fa = gap(a); fb = gap(b)
    (isnan(fa) || isnan(fb)) && return T(NaN)
    sa = sign(fa); sb = sign(fb)
    (sa != zero(T) && sb != zero(T) && sa != sb) || return T(NaN)
    for _ in 1:200
        (b - a) < T(tol) && break
        mid = (a + b) / 2
        sm = sign(gap(mid))
        if sm == zero(T)
            return mid
        elseif sm == sa
            a = mid
        else
            b = mid
        end
    end
    return (a + b) / 2
end

# =============================================================================
# Distribution — Young (2010) histogram respecting the discrete choice
# =============================================================================

"""
    DCEGMDistribution{T}

Distribution of households over cash-on-hand, income state, and discrete option,
returned by [`dcegm_simulate`](@ref).

| Field | Type | Description |
|---|---|---|
| `grid` | `Vector{T}` | Cash-on-hand histogram grid |
| `dist` | `Array{T,4}` | `dist[t, m, j, d]` — mass at period `t` |
| `shares` | `Matrix{T}` | `shares[t, d]` — share choosing option `d` at `t` |
| `consumption` | `Vector{T}` | Mean consumption by period |
| `assets` | `Vector{T}` | Mean end-of-period assets by period |
| `n_periods` | `Int` | Periods simulated |
"""
struct DCEGMDistribution{T<:AbstractFloat}
    grid::Vector{T}
    dist::Array{T,4}
    shares::Matrix{T}
    consumption::Vector{T}
    assets::Vector{T}
    n_periods::Int
end

"""
    dcegm_simulate(sol, grid; init=nothing, init_option=nothing, n_periods=…) -> DCEGMDistribution

Propagate a Young (2010) histogram through a [`DCEGMSolution`](@ref), with the
transition respecting the **discrete** choice: mass at `(M, j, d_-)` is split
across options by [`dcegm_choice_probabilities`](@ref) — degenerately at the
argmax without taste shocks — and each part is then pushed forward by that
option's savings policy and income.

# Arguments
- `sol::DCEGMSolution` — solved problem
- `grid::AbstractVector` — cash-on-hand histogram grid (sorted ascending)
- `init` — initial mass over `(grid, income state)`; defaults to all mass on the
  income process's stationary distribution at the grid point nearest `grid`'s median
- `init_option::Symbol` — option households are treated as having chosen before the
  first period; defaults to the first non-absorbing option
- `n_periods::Int` — periods to simulate; defaults to `sol.n_periods`, and must be
  supplied for an infinite-horizon (stationary) solution

Off-grid landing points are split between the two bracketing nodes in proportion to
distance (the Young lottery), so mass is conserved exactly.

# Returns
[`DCEGMDistribution{T}`](@ref).
"""
function dcegm_simulate(sol::DCEGMSolution{T}, grid::AbstractVector{<:Real};
                        init::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                        init_option::Union{Nothing,Symbol}=nothing,
                        n_periods::Int=sol.n_periods) where {T<:AbstractFloat}
    prob = sol.prob
    g = collect(T, grid)
    issorted(g) || throw(ArgumentError("`grid` must be sorted ascending"))
    length(g) >= 2 || throw(ArgumentError("`grid` needs at least 2 points"))
    n_periods >= 1 || throw(ArgumentError("`n_periods` must be at least 1"))
    n_m = length(g)
    n_d = length(prob.options)
    n_e = length(prob.income_process.states)
    stationary = sol.n_periods == 1 && prob.n_periods == 0
    stationary || n_periods <= sol.n_periods || throw(ArgumentError(
        "`n_periods` = $n_periods exceeds the solved horizon $(sol.n_periods)"))

    d0 = if init_option === nothing
        k = findfirst(!, prob.absorbing)
        k === nothing ? 1 : k
    else
        k = findfirst(==(init_option), prob.options)
        k === nothing && throw(ArgumentError(
            "init_option :$init_option is not one of $(prob.options)"))
        k
    end

    dist = zeros(T, n_periods, n_m, n_e, n_d)
    if init === nothing
        i0 = cld(n_m, 2)
        for j in 1:n_e
            dist[1, i0, j, d0] = T(prob.income_process.stationary_dist[j])
        end
    else
        size(init) == (n_m, n_e) || throw(ArgumentError(
            "`init` must be $(n_m)×$(n_e), got $(size(init))"))
        for i in 1:n_m, j in 1:n_e
            dist[1, i, j, d0] = T(init[i, j])
        end
    end
    s1 = sum(@view dist[1, :, :, :])
    s1 > zero(T) && (dist[1, :, :, :] ./= s1)

    Pi = prob.income_process.transition
    shares = zeros(T, n_periods, n_d)
    cons = zeros(T, n_periods)
    assets = zeros(T, n_periods)

    for t in 1:n_periods
        t_pol = stationary ? 1 : t
        for i in 1:n_m, j in 1:n_e, dp in 1:n_d
            mass = dist[t, i, j, dp]
            mass <= zero(T) && continue
            m = g[i]
            p = dcegm_choice_probabilities(sol, t_pol, dp, j, m)
            for d in 1:n_d
                w = mass * p[d]
                w <= zero(T) && continue
                shares[t, d] += w
                cval, _ = dcegm_policy(sol, t_pol, d, j, m)
                a = max(m - cval, prob.credit_limit)
                cons[t] += w * cval
                assets[t] += w * a
                t == n_periods && continue
                for jp in 1:n_e
                    wj = w * Pi[j, jp]
                    wj <= zero(T) && continue
                    mp = prob.R * a + T(prob.income(d, jp))
                    _young_push!(view(dist, t + 1, :, jp, d), g, mp, wj)
                end
            end
        end
    end

    return DCEGMDistribution{T}(g, dist, shares, cons, assets, n_periods)
end

"Split `mass` landing at `x` between the two bracketing nodes of `g` (Young lottery)."
function _young_push!(col::AbstractVector{T}, g::Vector{T}, x::T, mass::T) where {T<:AbstractFloat}
    n = length(g)
    if x <= g[1]
        col[1] += mass
        return col
    elseif x >= g[n]
        col[n] += mass
        return col
    end
    k = clamp(searchsortedfirst(g, x) - 1, 1, n - 1)
    dx = g[k+1] - g[k]
    w = dx > zero(T) ? (x - g[k]) / dx : zero(T)
    col[k] += mass * (one(T) - w)
    col[k+1] += mass * w
    return col
end

# =============================================================================
# Canonical example — Iskhakov et al. (2017) retirement model
# =============================================================================

"""
    dcegm_retirement_model(; n_periods=20, beta=0.98, R=1.0, wage=20.0,
                             disutility=1.0, sigma=0.0, n_shocks=1,
                             taste_shock_scale=0.0, a_max=50.0, n_a=200,
                             pension=0.0, credit_limit=0.0) -> DCEGMProblem

The canonical retirement model of Iskhakov, Jørgensen, Rust & Schjerning (2017).

A household chooses each period whether to **work** or **retire** — retirement is
absorbing — and how much to consume. Flow utility is `log(c)` less a disutility of
work `δ`, savings earn a gross return `R`, and working delivers `wage · η` next
period, where `η` is a discretized log-normal shock with log standard deviation
`sigma` (`sigma = 0` or `n_shocks = 1` gives the deterministic model). Retirement
pays `pension`.

The discrete choice makes the value function non-concave: near the retirement
threshold the consumption function has the characteristic **secondary kink** that
plain EGM cannot resolve and that the upper envelope removes.

Options are ordered `[:retire, :work]`, so `d = 1` is retirement (absorbing) and
`d = 2` is work.

| Keyword | Type | Default | Description |
|---|---|---|---|
| `n_periods` | `Int` | `20` | Life-cycle length |
| `beta` | `Real` | `0.98` | Discount factor |
| `R` | `Real` | `1.0` | Gross return on savings |
| `wage` | `Real` | `20.0` | Labor income when working |
| `disutility` | `Real` | `1.0` | Flow disutility of work `δ` |
| `sigma` | `Real` | `0.0` | Log standard deviation of the wage shock |
| `n_shocks` | `Int` | `1` | Quadrature nodes for the wage shock |
| `taste_shock_scale` | `Real` | `0.0` | Extreme-value taste-shock scale `λ` |
| `a_max` | `Real` | `50.0` | Top of the post-decision asset grid |
| `n_a` | `Int` | `200` | Asset grid points |
| `pension` | `Real` | `0.0` | Income while retired |
| `credit_limit` | `Real` | `0.0` | Lower bound on end-of-period assets |
| `curvature` | `Real` | `2.0` | Asset-grid curvature; `1` is uniform, higher packs points near the credit limit |

!!! note "Why the grid is curved by default"
    With `pension = 0` a retiree who saves exactly the credit limit consumes zero
    forever, so the value there is `-Inf` and the endogenous grid cannot reach
    below `ā + c(ā)`. On a uniform grid that unresolved wedge is a full asset step
    wide; `curvature = 2` shrinks it by a factor of `n_a`.

# References
- Iskhakov, F., Jørgensen, T. H., Rust, J., & Schjerning, B. (2017).
  *Quantitative Economics*, 8(2), 317–365.
"""
function dcegm_retirement_model(; n_periods::Int=20, beta::Real=0.98, R::Real=1.0,
                                  wage::Real=20.0, disutility::Real=1.0,
                                  sigma::Real=0.0, n_shocks::Int=1,
                                  taste_shock_scale::Real=0.0,
                                  a_max::Real=50.0, n_a::Int=200,
                                  pension::Real=0.0, credit_limit::Real=0.0,
                                  curvature::Real=2.0)
    n_shocks >= 1 || throw(ArgumentError("`n_shocks` must be at least 1"))
    curvature >= 1 || throw(ArgumentError("`curvature` must be at least 1"))
    states, probs = _log_normal_quadrature(Float64(sigma), n_shocks)
    Pi = repeat(reshape(probs, 1, :), n_shocks, 1)      # iid ⇒ identical rows
    income_process = IncomeProcess{Float64}(Pi, states, copy(probs), :income)

    delta = Float64(disutility)
    w = Float64(wage)
    pen = Float64(pension)

    # d = 1 → retire (absorbing), d = 2 → work
    u(c, d) = c > 0 ? log(c) - (d == 2 ? delta : 0.0) : -Inf
    up(c, d) = c > 0 ? 1 / c : Inf
    upinv(m, d) = m > 0 ? 1 / m : Inf
    y(d, j) = d == 2 ? w * states[j] : pen

    # Curved grid, dense near the credit limit. A uniform grid leaves the first
    # asset step unresolved, and just above the credit limit a retiree with no
    # pension consumes zero forever, so that whole step evaluates to −∞.
    lo = Float64(credit_limit); span = Float64(a_max) - lo
    a_grid = [lo + span * (k / (n_a - 1))^Float64(curvature) for k in 0:(n_a-1)]

    return DCEGMProblem(; beta=beta, R=R, utility=u, utility_prime=up,
                          utility_prime_inv=upinv, income=y,
                          options=[:retire, :work], absorbing=[true, false],
                          asset_grid=a_grid, income_process=income_process,
                          n_periods=n_periods, taste_shock_scale=taste_shock_scale,
                          credit_limit=credit_limit)
end

"""
    _log_normal_quadrature(sigma, n) -> (nodes, weights)

Gauss-Hermite quadrature for a unit-mean log-normal shock, `E[η] = 1`. Returns a
single unit node when `n == 1` or `sigma == 0`, so the deterministic model is the
`n = 1` special case rather than a separate code path.
"""
function _log_normal_quadrature(sigma::Float64, n::Int)
    (n == 1 || sigma <= 0) && return ([1.0], [1.0])
    x, w = _gauss_hermite_nodes_weights(n)
    nodes = exp.(sqrt(2.0) * sigma .* x .- 0.5 * sigma^2)
    weights = w ./ sum(w)
    return (nodes, weights)
end

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, p::DCEGMProblem{T}) where {T}
    horizon = p.n_periods > 0 ? "T=$(p.n_periods)" : "infinite horizon"
    print(io, "DCEGMProblem{$T}: options=", p.options, ", ", horizon,
          ", λ=", p.taste_shock_scale)
end

function Base.show(io::IO, s::DCEGMSolution{T}) where {T}
    print(io, "DCEGMSolution{$T}: ", length(s.prob.options), " options, ",
          s.n_periods, " periods, kinks=", sum(s.n_kinks),
          ", converged=", s.converged)
end

function Base.show(io::IO, d::DCEGMDistribution{T}) where {T}
    print(io, "DCEGMDistribution{$T}: ", d.n_periods, " periods, ",
          length(d.grid), " grid points")
end

"""
    report(sol::DCEGMSolution)

Print the problem setup, convergence diagnostics, and the number of
discrete-choice switching thresholds the upper envelope found per period.
"""
function report(sol::DCEGMSolution{T}) where {T}
    io = stdout
    prob = sol.prob
    head = Any[
        "Options"              join(string.(prob.options), ", ");
        "Absorbing"            join(string.(prob.options[prob.absorbing]), ", ");
        "Horizon"              prob.n_periods > 0 ? string(prob.n_periods) : "infinite";
        "Income states"        length(prob.income_process.states);
        "Asset grid points"    length(prob.asset_grid);
        "Taste-shock scale λ"  _fmt(prob.taste_shock_scale; digits=6);
        "Discount factor β"    _fmt(prob.beta; digits=6);
        "Gross return R"       _fmt(prob.R; digits=6);
        "Converged"            sol.converged ? "Yes" : "No";
        "Iterations"           sol.iterations;
        "Total kinks"          sum(sol.n_kinks)
    ]
    _pretty_table(io, head;
        title="DCEGM Solution",
        column_labels=["", "Value"],
        alignment=[:l, :r])

    # Periods where the envelope actually deleted a branch — the non-concave region.
    per = [sum(@view sol.n_kinks[t, :, :]) for t in 1:sol.n_periods]
    active = findall(>(0), per)
    if !isempty(active)
        rows = min(length(active), 15)
        data = Matrix{Any}(undef, rows, 3)
        for i in 1:rows
            t = active[i]
            data[i, 1] = t
            data[i, 2] = per[t]
            data[i, 3] = join(string.(prob.options[[d for d in 1:length(prob.options)
                                                    if sum(@view sol.n_kinks[t, d, :]) > 0]]), ", ")
        end
        _pretty_table(io, data;
            title="Non-Concave Periods (upper-envelope kinks)" *
                  (length(active) > rows ? " — first $rows of $(length(active))" : ""),
            column_labels=["Period", "Kinks", "Options"],
            alignment=[:r, :r, :l])
    end
    return nothing
end

"""
    report(d::DCEGMDistribution)

Print the discrete-choice shares, mean consumption, and mean assets by period.
"""
function report(d::DCEGMDistribution{T}) where {T}
    io = stdout
    n_d = size(d.shares, 2)
    rows = min(d.n_periods, 25)
    data = Matrix{Any}(undef, rows, 3 + n_d)
    for t in 1:rows
        data[t, 1] = t
        for k in 1:n_d
            data[t, 1 + k] = _fmt(d.shares[t, k]; digits=4)
        end
        data[t, end-1] = _fmt(d.consumption[t]; digits=4)
        data[t, end] = _fmt(d.assets[t]; digits=4)
    end
    _pretty_table(io, data;
        title="DCEGM Distribution" *
              (d.n_periods > rows ? " — first $rows of $(d.n_periods) periods" : ""),
        column_labels=vcat(["Period"], ["share_$k" for k in 1:n_d], ["E[c]", "E[a]"]),
        alignment=vcat([:r], fill(:r, n_d + 2)))
    return nothing
end
