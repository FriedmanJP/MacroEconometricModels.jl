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
    _egm_solve(ip, grid, income, prices; max_iter=1000, tol=1e-10)
        → (c_policy::Matrix{T}, a_policy::Matrix{T})

Solve a one-asset household savings problem using the Endogenous Grid Method
(Carroll 2006).

Returns `N_a × N_e` consumption and savings policy matrices on the exogenous
asset grid.

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
                     max_iter::Int=1000, tol::T=T(1e-10)) where {T<:AbstractFloat}
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

    # Initialize consumption policy: consume a fraction of cash-on-hand
    c_pol = zeros(T, n_a, n_e)
    for j in 1:n_e
        for i in 1:n_a
            coh = ip.budget_fn(a_grid[i], e_vals[j], prices)
            c_pol[i, j] = max(coh * T(0.05), T(1e-10))
        end
    end

    # Buffers
    c_new = zeros(T, n_a, n_e)
    a_new = zeros(T, n_a, n_e)
    emu = zeros(T, n_a)
    c_endo = zeros(T, n_a)
    a_endo = zeros(T, n_a)

    r = prices[:r]
    w = prices[:w]

    for iter in 1:max_iter
        # For each income state, run the EGM step
        for j in 1:n_e
            # Step 1: Compute expected marginal utility at each end-of-period asset a'_i
            fill!(emu, zero(T))
            for jp in 1:n_e
                for i in 1:n_a
                    # Consumption tomorrow at (a'_i, e_jp) by interpolation on current policy
                    c_tomorrow = _linear_interp(a_grid, view(c_pol, :, jp), a_grid[i])
                    emu[i] += Pi[j, jp] * u_prime(c_tomorrow)
                end
            end

            # Step 2: Euler inversion
            for i in 1:n_a
                c_endo[i] = u_prime_inv(beta * (one(T) + r) * emu[i])
            end

            # Step 3: Endogenous beginning-of-period assets
            for i in 1:n_a
                a_endo[i] = (c_endo[i] + a_grid[i] - w * e_vals[j]) / (one(T) + r)
            end

            # Step 4: Interpolate back to exogenous grid + borrowing constraint
            for i in 1:n_a
                a_val = a_grid[i]
                if a_val < a_endo[1]
                    # Constrained: consume all cash-on-hand minus borrowing limit savings
                    coh = ip.budget_fn(a_val, e_vals[j], prices)
                    c_new[i, j] = max(coh - a_min, T(1e-10))
                    a_new[i, j] = a_min
                else
                    c_new[i, j] = _linear_interp(a_endo, c_endo, a_val)
                    coh = ip.budget_fn(a_val, e_vals[j], prices)
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
            break
        end
    end

    # Compute final savings policy from budget constraint
    a_pol = zeros(T, n_a, n_e)
    for j in 1:n_e
        for i in 1:n_a
            coh = ip.budget_fn(a_grid[i], e_vals[j], prices)
            a_pol[i, j] = max(coh - c_pol[i, j], a_min)
        end
    end

    return c_pol, a_pol
end

# =============================================================================
# Two-asset nested EGM (simplified)
# =============================================================================

"""
    _two_asset_egm_solve(ip, grid, income, prices; max_iter=1000, tol=1e-8, n_deposit=30)
        → Dict{Symbol,Array{T}}

Solve a two-asset household problem using a simplified nested EGM.

For each illiquid deposit choice `d` on a deposit grid:
1. Inner EGM on the liquid dimension (treating the illiquid asset and deposit as
   given parameters).
2. Evaluate lifetime utility for each deposit choice.
3. Pick optimal deposit via max over the deposit grid.

Returns a `Dict` with keys `:consumption`, `:liquid_savings`, `:deposit` —
each an `n_b × n_a × n_e` array (liquid × illiquid × income).
"""
function _two_asset_egm_solve(ip::IndividualProblem{T}, grid::HAGrid{T},
                               income::IncomeProcess{T}, prices::Dict{Symbol,T};
                               max_iter::Int=1000, tol::T=T(1e-8),
                               n_deposit::Int=30) where {T<:AbstractFloat}
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
    u_prime = ip.utility_prime
    u_prime_inv = ip.utility_prime_inv
    Pi = income.transition
    e_vals = income.states

    r_b = get(prices, :r_b, prices[:r])  # liquid return
    r_a = get(prices, :r_a, prices[:r])  # illiquid return
    w = prices[:w]

    # Adjustment cost function (default: zero)
    adj_cost = isnothing(ip.adjustment_cost) ? (d, a) -> zero(T) : ip.adjustment_cost

    # Deposit grid: from negative (withdrawal) to positive
    a_max = a_grid[end]
    d_min = -a_max * T(0.5)
    d_max = a_max * T(0.5)
    deposit_grid = collect(range(d_min, d_max; length=n_deposit))

    # Output arrays
    c_opt = zeros(T, n_b, n_a, n_e)
    b_opt = zeros(T, n_b, n_a, n_e)
    d_opt = zeros(T, n_b, n_a, n_e)

    # Initialize consumption policy for inner EGM (liquid dimension)
    # For each (illiquid asset, deposit, income), run 1D EGM on liquid dimension
    c_inner = zeros(T, n_b, n_e)
    for j in 1:n_e
        for i in 1:n_b
            coh = (one(T) + r_b) * b_grid[i] + w * e_vals[j]
            c_inner[i, j] = max(coh * T(0.05), T(1e-10))
        end
    end

    # For each illiquid asset level and income state, find optimal deposit
    for j in 1:n_e
        for ia in 1:n_a
            a_val = a_grid[ia]

            best_util = T(-Inf)
            best_c = zeros(T, n_b)
            best_b_save = zeros(T, n_b)
            best_d = zero(T)

            for id in 1:n_deposit
                d_val = deposit_grid[id]

                # New illiquid asset after deposit
                a_prime = (one(T) + r_a) * a_val + d_val
                if a_prime < zero(T) || a_prime > a_grid[end]
                    continue  # infeasible
                end

                # Adjustment cost
                chi = adj_cost(d_val, a_val)

                # Inner EGM: solve liquid savings problem given deposit d
                # Effective liquid resources after deposit and adjustment cost
                c_trial = zeros(T, n_b)
                b_save_trial = zeros(T, n_b)

                # EMU for liquid dimension
                emu_liq = zeros(T, n_b)
                for jp in 1:n_e
                    for ib in 1:n_b
                        c_tom = _linear_interp(b_grid, view(c_inner, :, jp), b_grid[ib])
                        emu_liq[ib] += Pi[j, jp] * u_prime(c_tom)
                    end
                end

                # Euler inversion for liquid
                c_endo_liq = zeros(T, n_b)
                b_endo_liq = zeros(T, n_b)
                for ib in 1:n_b
                    c_endo_liq[ib] = u_prime_inv(beta * (one(T) + r_b) * emu_liq[ib])
                    # Endogenous liquid assets
                    b_endo_liq[ib] = (c_endo_liq[ib] + b_grid[ib] + d_val + chi - w * e_vals[j]) / (one(T) + r_b)
                end

                # Interpolate back to exogenous liquid grid
                for ib in 1:n_b
                    b_val = b_grid[ib]
                    coh_liq = (one(T) + r_b) * b_val + w * e_vals[j] - d_val - chi
                    if b_val < b_endo_liq[1] || coh_liq <= zero(T)
                        c_trial[ib] = max(coh_liq - b_min, T(1e-10))
                        b_save_trial[ib] = b_min
                    else
                        c_trial[ib] = _linear_interp(b_endo_liq, c_endo_liq, b_val)
                        b_save_trial[ib] = max(coh_liq - c_trial[ib], b_min)
                    end
                    c_trial[ib] = max(c_trial[ib], T(1e-10))
                end

                # Evaluate utility at median liquid asset point (representative)
                ib_mid = div(n_b, 2)
                util_val = u(c_trial[ib_mid])

                if util_val > best_util
                    best_util = util_val
                    copyto!(best_c, c_trial)
                    copyto!(best_b_save, b_save_trial)
                    best_d = d_val
                end
            end

            # Store optimal policies
            for ib in 1:n_b
                c_opt[ib, ia, j] = best_c[ib]
                b_opt[ib, ia, j] = best_b_save[ib]
                d_opt[ib, ia, j] = best_d
            end
        end
    end

    # Iterate the inner consumption policy
    for iter in 1:max_iter
        c_inner_new = zeros(T, n_b, n_e)
        for j in 1:n_e
            emu_inner = zeros(T, n_b)
            for jp in 1:n_e
                for ib in 1:n_b
                    c_tom = _linear_interp(b_grid, view(c_inner, :, jp), b_grid[ib])
                    emu_inner[ib] += Pi[j, jp] * u_prime(c_tom)
                end
            end
            for ib in 1:n_b
                c_endo_val = u_prime_inv(beta * (one(T) + r_b) * emu_inner[ib])
                b_endo_val = (c_endo_val + b_grid[ib] - w * e_vals[j]) / (one(T) + r_b)

                b_val = b_grid[ib]
                if b_val < b_endo_val
                    coh = (one(T) + r_b) * b_val + w * e_vals[j]
                    c_inner_new[ib, j] = max(coh - b_min, T(1e-10))
                else
                    c_inner_new[ib, j] = _linear_interp(
                        [b_endo_val; b_grid[searchsortedfirst(b_grid, b_endo_val):end]],
                        [c_endo_val; [u_prime_inv(beta * (one(T) + r_b) * emu_inner[k]) for k in searchsortedfirst(b_grid, b_endo_val):n_b]],
                        b_val
                    )
                end
                c_inner_new[ib, j] = max(c_inner_new[ib, j], T(1e-10))
            end
        end

        max_diff = maximum(abs.(c_inner_new .- c_inner))
        copyto!(c_inner, c_inner_new)
        if max_diff < tol
            break
        end
    end

    return Dict{Symbol,Array{T}}(
        :consumption => c_opt,
        :liquid_savings => b_opt,
        :deposit => d_opt
    )
end
