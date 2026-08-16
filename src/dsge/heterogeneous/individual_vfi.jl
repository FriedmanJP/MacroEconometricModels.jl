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
   a. **Maximize**: For each `(i, j)`, search over the savings grid for the
      optimal `a'` that maximizes `u(coh - a') + beta * E[V(a', e')]`.
   b. **Howard steps**: Fix the policy, iterate the value function forward
      `howard_steps` times (each is a cheap linear operation).
3. Convergence: `max|V_new - V_old| < tol`.
4. Return value function, consumption policy, and savings policy.

Savings policies are stored as grid indices during iteration and converted to
values upon return.

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

    # Policy indices: for each (i, j), store the index into a_grid of optimal a'
    pol_idx = ones(Int, n_a, n_e)
    if init_policy !== nothing && size(init_policy) == (n_a, n_e)
        for j in 1:n_e
            for i in 1:n_a
                # Snap the warm-start a' onto the nearest grid node
                ap = init_policy[i, j]
                k = searchsortedfirst(a_grid, ap)
                pol_idx[i, j] = clamp(k, 1, n_a)
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
            for i in 1:n_a
                best_val = T(-Inf)
                best_idx = 1

                # Search over savings grid; exploit monotonicity by starting
                # from the previous optimal index (for the prior asset level)
                lo = i > 1 ? pol_idx[i-1, j] : 1

                for k in lo:n_a
                    a_prime = a_grid[k]
                    c = coh[i, j] - a_prime
                    if c <= zero(T)
                        break  # grid is sorted, remaining a' are infeasible
                    end
                    val = u(c) + beta * EV[k, j]
                    if val > best_val
                        best_val = val
                        best_idx = k
                    end
                end

                # If no feasible choice found (coh too low), save at the
                # borrowing constraint and consume max(coh - a_min, tiny).
                if best_val == T(-Inf)
                    best_idx = 1
                    c_fallback = max(coh[i, j] - a_grid[1], T(1e-10))
                    best_val = u(c_fallback) + beta * EV[1, j]
                end

                V_new[i, j] = best_val
                pol_idx[i, j] = best_idx
            end
        end

        # ── Howard improvement steps ──────────────────────────────────────
        # Fix the policy and iterate the value function forward without
        # re-optimizing. Each step: V[i,j] = u(c(i,j)) + beta * E[V(a'(i,j), e')]
        for _h in 1:howard_steps
            # Compute EV from current V_new
            for j in 1:n_e
                for i in 1:n_a
                    ev = zero(T)
                    k = pol_idx[i, j]
                    for jp in 1:n_e
                        ev += Pi[j, jp] * V_new[k, jp]
                    end
                    EV[i, j] = ev
                end
            end

            for j in 1:n_e
                for i in 1:n_a
                    k = pol_idx[i, j]
                    c = coh[i, j] - a_grid[k]
                    c = max(c, T(1e-10))
                    V_new[i, j] = u(c) + beta * EV[i, j]
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
    a_policy = zeros(T, n_a, n_e)
    for j in 1:n_e
        for i in 1:n_a
            k = pol_idx[i, j]
            a_policy[i, j] = a_grid[k]
            c_policy[i, j] = max(coh[i, j] - a_grid[k], T(1e-10))
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
