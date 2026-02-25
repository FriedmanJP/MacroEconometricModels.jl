# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

"""
Perfect foresight (deterministic) solver for DSGE models.

Newton solver on the stacked system with block-tridiagonal sparse Jacobian.

For a model `f(y_t, y_{t-1}, y_{t+1}, ε_t, θ) = 0`:
1. Stack T periods of unknowns: `x = [y_1; y_2; ...; y_T]` (T*n unknowns)
2. Initial condition: `y_0 = y_ss` (steady state)
3. Terminal condition: `y_{T+1} = y_ss` (steady state)
4. Shock path: `ε_1, ..., ε_T` (given)
5. Newton iteration: `J * Δx = -F(x)` with block-tridiagonal sparse Jacobian
"""

using SparseArrays

"""
    perfect_foresight(spec::DSGESpec{FT}; T_periods=100, shock_path=nothing,
                       max_iter=100, tol=1e-8, constraints=DSGEConstraint[],
                       solver=nothing) → PerfectForesightPath{FT}

Solve for the deterministic perfect foresight path given a sequence of shocks.

# Keywords
- `T_periods::Int=100` — number of simulation periods
- `shock_path::Matrix` — T_periods × n_exog matrix of shock realizations
- `max_iter::Int=100` — Newton iteration limit
- `tol::Real=1e-8` — convergence tolerance (max abs residual)
- `constraints::Vector{<:DSGEConstraint}` — variable bounds and nonlinear inequalities (requires JuMP + Ipopt)
- `solver::Symbol` — `:ipopt` (NLP) or `:path` (MCP); auto-detected if not specified
"""
function perfect_foresight(spec::DSGESpec{FT};
        T_periods::Int=100,
        shock_path::Union{Nothing,AbstractMatrix}=nothing,
        max_iter::Int=100,
        tol::Real=1e-8,
        constraints::Vector=DSGEConstraint[],
        solver::Union{Nothing,Symbol}=nothing) where {FT<:AbstractFloat}

    n = spec.n_endog
    n_ε = spec.n_exog
    θ = spec.param_values
    y_ss = spec.steady_state

    isempty(y_ss) &&
        throw(ArgumentError("Must compute steady state first (call compute_steady_state)"))

    # Default shock path: zeros (return to SS)
    shocks = if shock_path === nothing
        zeros(FT, T_periods, n_ε)
    else
        @assert size(shock_path, 1) == T_periods "shock_path must have $T_periods rows"
        @assert size(shock_path, 2) == n_ε "shock_path must have $n_ε columns"
        FT.(shock_path)
    end

    # If constraints are provided, dispatch to JuMP extension
    if !isempty(constraints)
        _check_jump_loaded()
        _validate_constraints(spec, constraints)
        chosen = _select_solver(constraints, solver)
        chosen ∉ (:path, :ipopt) &&
            throw(ArgumentError("Unknown solver :$chosen. Valid options: :path, :ipopt"))
        if chosen == :path
            any(c -> c isa NonlinearConstraint, constraints) &&
                throw(ArgumentError(
                    "PATH solver cannot handle NonlinearConstraint. " *
                    "Use solver=:ipopt or remove nonlinear constraints."))
            return _path_perfect_foresight(spec, T_periods, shocks, constraints)
        else
            return _jump_perfect_foresight(spec, T_periods, shocks, constraints)
        end
    end

    # Initialize: all periods at steady state
    # x is a flat vector of length T_periods * n
    # x[(t-1)*n+1 : t*n] = y_t for period t
    x = repeat(y_ss, T_periods)

    # Preallocate residual vector
    F_vec = zeros(FT, T_periods * n)

    converged = false
    iter = 0

    for k in 1:max_iter
        iter = k

        # Evaluate stacked residual F(x)
        _pf_residual!(F_vec, x, spec, shocks, T_periods)

        # Check convergence
        max_resid = maximum(abs, F_vec)
        if max_resid < FT(tol)
            converged = true
            break
        end

        # Build sparse block-tridiagonal Jacobian
        J = _pf_jacobian(x, spec, shocks, T_periods)

        # Newton step: solve J * Δx = -F
        Δx = J \ (-F_vec)

        # Update
        x .+= Δx
    end

    # Reshape solution into T_periods × n matrix
    path_full = reshape(copy(x), n, T_periods)'  # T_periods × n
    deviations_full = path_full .- y_ss'

    # Filter to original variables if augmented
    if spec.augmented
        orig_idx = _original_var_indices(spec)
        path = Matrix{FT}(path_full[:, orig_idx])
        deviations = Matrix{FT}(deviations_full[:, orig_idx])
    else
        path = Matrix{FT}(path_full)
        deviations = Matrix{FT}(deviations_full)
    end

    PerfectForesightPath{FT}(path, deviations, converged, iter, spec)
end

# =============================================================================
# Stacked residual evaluation
# =============================================================================

"""
    _pf_residual!(F, x, spec, shocks, T_periods)

Evaluate the stacked residual vector in-place.

For each period t = 1, ..., T_periods:
  F[(t-1)*n+1 : t*n] = [f_i(y_t, y_{t-1}, y_{t+1}, ε_t, θ) for i in 1:n]

where y_0 = y_ss (initial) and y_{T+1} = y_ss (terminal).
"""
function _pf_residual!(F::Vector{FT}, x::Vector{FT}, spec::DSGESpec{FT},
                       shocks::Matrix{FT}, Tp::Int) where {FT}
    n = spec.n_endog
    n_ε = spec.n_exog
    y_ss = spec.steady_state
    θ = spec.param_values

    for t in 1:Tp
        # Extract y_{t-1}, y_t, y_{t+1}
        y_t = x[(t-1)*n+1 : t*n]

        y_lag = if t == 1
            y_ss  # initial condition
        else
            x[(t-2)*n+1 : (t-1)*n]
        end

        y_lead = if t == Tp
            y_ss  # terminal condition
        else
            x[t*n+1 : (t+1)*n]
        end

        ε_t = shocks[t, :]

        # Evaluate each equation residual
        for i in 1:n
            F[(t-1)*n + i] = spec.residual_fns[i](y_t, y_lag, y_lead, ε_t, θ)
        end
    end
    return nothing
end

# =============================================================================
# Block-tridiagonal sparse Jacobian
# =============================================================================

"""
    _pf_jacobian(x, spec, shocks, T_periods) → SparseMatrixCSC

Build the block-tridiagonal sparse Jacobian of the stacked system.

Each period t contributes three n×n blocks:
- ∂F_t/∂y_{t-1} (sub-diagonal block, except t=1)
- ∂F_t/∂y_t     (diagonal block)
- ∂F_t/∂y_{t+1} (super-diagonal block, except t=T_periods)

Uses central differences for numerical Jacobians.
"""
function _pf_jacobian(x::Vector{FT}, spec::DSGESpec{FT},
                      shocks::Matrix{FT}, Tp::Int) where {FT}
    n = spec.n_endog
    y_ss = spec.steady_state
    θ = spec.param_values
    N = Tp * n  # total unknowns

    # Pre-allocate sparse triplets (I, J, V)
    # Maximum entries: 3 blocks per period × n×n entries = 3*Tp*n^2
    # (first and last periods have only 2 blocks)
    max_nnz = 3 * Tp * n * n
    row_idx = Vector{Int}(undef, max_nnz)
    col_idx = Vector{Int}(undef, max_nnz)
    vals = Vector{FT}(undef, max_nnz)
    cnt = 0

    for t in 1:Tp
        # Extract current state vectors
        y_t = x[(t-1)*n+1 : t*n]

        y_lag = if t == 1
            y_ss
        else
            x[(t-2)*n+1 : (t-1)*n]
        end

        y_lead = if t == Tp
            y_ss
        else
            x[t*n+1 : (t+1)*n]
        end

        ε_t = shocks[t, :]

        # Row offset for period t
        row_off = (t - 1) * n

        # --- Diagonal block: ∂F_t/∂y_t ---
        col_off = (t - 1) * n
        for j in 1:n
            h = max(FT(1e-7), FT(1e-7) * abs(y_t[j]))
            y_plus = copy(y_t)
            y_minus = copy(y_t)
            y_plus[j] += h
            y_minus[j] -= h

            for i in 1:n
                fn = spec.residual_fns[i]
                f_plus = fn(y_plus, y_lag, y_lead, ε_t, θ)
                f_minus = fn(y_minus, y_lag, y_lead, ε_t, θ)
                dfdx = (f_plus - f_minus) / (2h)
                if abs(dfdx) > FT(1e-15)
                    cnt += 1
                    row_idx[cnt] = row_off + i
                    col_idx[cnt] = col_off + j
                    vals[cnt] = dfdx
                end
            end
        end

        # --- Sub-diagonal block: ∂F_t/∂y_{t-1} (only for t >= 2) ---
        if t >= 2
            col_off_lag = (t - 2) * n
            y_lag_base = x[(t-2)*n+1 : (t-1)*n]

            for j in 1:n
                h = max(FT(1e-7), FT(1e-7) * abs(y_lag_base[j]))
                y_lag_plus = copy(y_lag_base)
                y_lag_minus = copy(y_lag_base)
                y_lag_plus[j] += h
                y_lag_minus[j] -= h

                for i in 1:n
                    fn = spec.residual_fns[i]
                    f_plus = fn(y_t, y_lag_plus, y_lead, ε_t, θ)
                    f_minus = fn(y_t, y_lag_minus, y_lead, ε_t, θ)
                    dfdx = (f_plus - f_minus) / (2h)
                    if abs(dfdx) > FT(1e-15)
                        cnt += 1
                        row_idx[cnt] = row_off + i
                        col_idx[cnt] = col_off_lag + j
                        vals[cnt] = dfdx
                    end
                end
            end
        end

        # --- Super-diagonal block: ∂F_t/∂y_{t+1} (only for t <= T_periods-1) ---
        if t <= Tp - 1
            col_off_lead = t * n
            y_lead_base = x[t*n+1 : (t+1)*n]

            for j in 1:n
                h = max(FT(1e-7), FT(1e-7) * abs(y_lead_base[j]))
                y_lead_plus = copy(y_lead_base)
                y_lead_minus = copy(y_lead_base)
                y_lead_plus[j] += h
                y_lead_minus[j] -= h

                for i in 1:n
                    fn = spec.residual_fns[i]
                    f_plus = fn(y_t, y_lag, y_lead_plus, ε_t, θ)
                    f_minus = fn(y_t, y_lag, y_lead_minus, ε_t, θ)
                    dfdx = (f_plus - f_minus) / (2h)
                    if abs(dfdx) > FT(1e-15)
                        cnt += 1
                        row_idx[cnt] = row_off + i
                        col_idx[cnt] = col_off_lead + j
                        vals[cnt] = dfdx
                    end
                end
            end
        end
    end

    # Build sparse matrix from triplets
    sparse(row_idx[1:cnt], col_idx[1:cnt], vals[1:cnt], N, N)
end
