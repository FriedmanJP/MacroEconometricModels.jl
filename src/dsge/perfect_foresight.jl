# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

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
                       solver=nothing, algorithm=nothing) → PerfectForesightPath{FT}

Solve for the deterministic perfect foresight path given a sequence of shocks.

Uses NonlinearSolve.jl as the default Newton solver with block-tridiagonal sparse Jacobian.

# Keywords
- `T_periods::Int=100` — number of simulation periods
- `shock_path::Matrix` — T_periods × n_exog matrix of shock realizations
- `max_iter::Int=100` — Newton iteration limit
- `tol::Real=1e-8` — convergence tolerance (max abs residual)
- `constraints::Vector{<:DSGEConstraint}` — variable bounds and nonlinear inequalities
- `solver::Symbol` — `:nonlinearsolve` (default), `:ipopt` (NLP), or `:path` (MCP)
- `algorithm` — NonlinearSolve.jl algorithm (default: `NewtonRaphson()`); ignored for JuMP solvers
"""
function perfect_foresight(spec::DSGESpec{FT};
        T_periods::Int=100,
        shock_path::Union{Nothing,AbstractMatrix}=nothing,
        max_iter::Int=100,
        tol::Real=1e-8,
        constraints::Vector=DSGEConstraint[],
        solver::Union{Nothing,Symbol}=nothing,
        algorithm=nothing) where {FT<:AbstractFloat}

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

    # If constraints are provided, dispatch to appropriate solver
    if !isempty(constraints)
        _validate_constraints(spec, constraints)
        chosen = _select_solver(constraints, solver)

        if chosen == :nonlinearsolve
            any(c -> c isa NonlinearConstraint, constraints) &&
                throw(ArgumentError(
                    "NonlinearSolve solver cannot handle NonlinearConstraint. " *
                    "Use solver=:nlopt (default) or solver=:ipopt for nonlinear inequality constraints."))
            lower, upper = _extract_bounds(spec, constraints)
            pf = _nonlinearsolve_perfect_foresight(spec, T_periods, shocks;
                        max_iter=max_iter, tol=tol, algorithm=algorithm)
            # Check if bounds are violated in the unconstrained solution
            bounds_ok = true
            for t in 1:T_periods, i in 1:n
                v = pf.path[t, i]
                if (isfinite(lower[i]) && v < lower[i] - FT(1e-6)) ||
                   (isfinite(upper[i]) && v > upper[i] + FT(1e-6))
                    bounds_ok = false
                    break
                end
            end
            bounds_ok && return pf
            # Escalate to projected Newton (always available)
            return _projected_newton_pf(spec, T_periods, shocks, lower, upper;
                        max_iter=max_iter, tol=tol)
        elseif chosen == :nlopt
            return _nlopt_perfect_foresight(spec, T_periods, shocks, constraints;
                        algorithm=algorithm)
        elseif chosen == :path
            _check_jump_loaded()
            any(c -> c isa NonlinearConstraint, constraints) &&
                throw(ArgumentError(
                    "PATH solver cannot handle NonlinearConstraint. " *
                    "Use solver=:nlopt or solver=:ipopt."))
            return _path_perfect_foresight(spec, T_periods, shocks, constraints)
        elseif chosen == :ipopt
            _check_jump_loaded()
            return _jump_perfect_foresight(spec, T_periods, shocks, constraints)
        else
            throw(ArgumentError("Unknown solver :$chosen. " *
                "Valid options: :nonlinearsolve, :nlopt, :path, :ipopt"))
        end
    end

    # Unconstrained: use NonlinearSolve
    return _nonlinearsolve_perfect_foresight(spec, T_periods, shocks;
                max_iter=max_iter, tol=tol, algorithm=algorithm)
end

# =============================================================================
# NonlinearSolve-based perfect foresight solver
# =============================================================================

"""
    _nonlinearsolve_perfect_foresight(spec, T_periods, shocks;
                                       max_iter=100, tol=1e-8, algorithm=nothing)

Solve the stacked perfect foresight system using NonlinearSolve.jl (unconstrained).

Uses the existing `_pf_residual!` and `_pf_jacobian` for the block-tridiagonal
sparse system. Default algorithm is `NewtonRaphson()`.
"""
function _nonlinearsolve_perfect_foresight(spec::DSGESpec{FT}, T_periods::Int,
        shocks::Matrix{FT};
        max_iter::Int=100, tol::Real=1e-8, algorithm=nothing) where {FT<:AbstractFloat}

    n = spec.n_endog
    N = T_periods * n  # total unknowns

    # Initial guess: all periods at steady state
    x0 = repeat(spec.steady_state, T_periods)

    # Wrap _pf_residual! for NonlinearSolve's (F, x, p) signature
    function pf_residual!(F, x, p)
        _pf_residual!(F, x, spec, shocks, T_periods)
        return nothing
    end

    # Compute Jacobian prototype from initial guess for sparsity pattern
    J_proto = _pf_jacobian(x0, spec, shocks, T_periods)

    # Wrap _pf_jacobian for NonlinearSolve's in-place (J, x, p) signature
    function pf_jacobian!(J::SparseMatrixCSC, x, p)
        J_new = _pf_jacobian(x, spec, shocks, T_periods)
        # Zero out existing values, then copy from new sparse matrix
        fill!(nonzeros(J), zero(FT))
        rows_new = rowvals(J_new)
        vals_new = nonzeros(J_new)
        for col in 1:size(J_new, 2)
            for idx in nzrange(J_new, col)
                J[rows_new[idx], col] = vals_new[idx]
            end
        end
        return nothing
    end

    nlfn = NonlinearSolve.NonlinearFunction(pf_residual!; jac=pf_jacobian!, jac_prototype=J_proto)
    prob = NonlinearSolve.NonlinearProblem(nlfn, x0, nothing)

    alg = algorithm !== nothing ? algorithm : NonlinearSolve.NewtonRaphson()
    sol = NonlinearSolve.solve(prob, alg; abstol=FT(tol), maxiters=max_iter)

    converged = NonlinearSolve.SciMLBase.successful_retcode(sol.retcode)
    if !converged
        @warn "Perfect foresight solver did not converge (retcode = $(sol.retcode))"
    end

    # Extract iteration count
    iter = try
        sol.stats.nsteps
    catch
        0
    end

    x = Vector{FT}(sol.u)

    # Reshape solution into T_periods × n matrix
    path_full = reshape(copy(x), n, T_periods)'  # T_periods × n
    deviations_full = path_full .- spec.steady_state'

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
# Projected Newton solver for box-constrained perfect foresight
# =============================================================================

"""
    _projected_newton_pf(spec, T_periods, shocks, lower, upper;
                          max_iter=100, tol=1e-8)

Box-constrained perfect foresight via projected Newton with Armijo backtracking.

Uses the existing block-tridiagonal sparse Jacobian (`_pf_jacobian`) for the
Newton step, then clamps to bounds. Preserves O(T·n) sparsity structure.
"""
function _projected_newton_pf(spec::DSGESpec{FT}, T_periods::Int,
        shocks::Matrix{FT}, lower::Vector{FT}, upper::Vector{FT};
        max_iter::Int=100, tol::Real=1e-8) where {FT<:AbstractFloat}

    n = spec.n_endog
    N = T_periods * n

    # Stack bounds: repeat per-variable bounds across all periods
    lower_stacked = repeat(lower, T_periods)
    upper_stacked = repeat(upper, T_periods)

    # Initial guess: steady state, clamped to bounds
    x = repeat(spec.steady_state, T_periods)
    x .= clamp.(x, lower_stacked, upper_stacked)

    F = zeros(FT, N)
    _pf_residual!(F, x, spec, shocks, T_periods)
    merit = dot(F, F)

    converged = false
    iter = 0
    c_armijo = FT(1e-4)

    for k in 1:max_iter
        iter = k

        # Build sparse Jacobian and compute Newton direction
        J = _pf_jacobian(x, spec, shocks, T_periods)
        d = -(J \ F)  # Newton step (sparse LU)

        # Armijo backtracking line search on merit = ||F(x)||^2
        # Directional derivative: ∇merit · d = 2 * F' * J * d = -2 * F' * F (at Newton step)
        dir_deriv = FT(2) * dot(F, J * d)
        α = FT(1.0)
        x_trial = similar(x)
        F_trial = similar(F)

        for _ls in 1:20
            x_trial .= clamp.(x .+ α .* d, lower_stacked, upper_stacked)
            _pf_residual!(F_trial, x_trial, spec, shocks, T_periods)
            merit_trial = dot(F_trial, F_trial)
            if merit_trial <= merit + c_armijo * α * dir_deriv
                break
            end
            α *= FT(0.5)
        end

        x .= clamp.(x .+ α .* d, lower_stacked, upper_stacked)
        _pf_residual!(F, x, spec, shocks, T_periods)
        merit = dot(F, F)

        if sqrt(merit) < FT(tol)
            converged = true
            break
        end
    end

    if !converged
        throw(ErrorException(
            "Projected Newton PF did not converge after $max_iter iterations " *
            "(||F|| = $(sqrt(merit))). Try solver=:ipopt with JuMP + Ipopt."))
    end

    # Reshape solution
    path_full = reshape(copy(x), n, T_periods)'
    deviations_full = path_full .- spec.steady_state'

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
# NLopt solver for nonlinear-constrained perfect foresight
# =============================================================================

"""
    _nlopt_perfect_foresight(spec, T_periods, shocks, constraints; algorithm=nothing)

Perfect foresight with nonlinear inequality constraints via NLopt LD_SLSQP.

Formulates as a feasibility problem with equality constraints (model equations)
and inequality constraints (NonlinearConstraint). Box bounds from VariableBound.
"""
function _nlopt_perfect_foresight(spec::DSGESpec{FT}, T_periods::Int,
        shocks::Matrix{FT}, constraints::Vector;
        algorithm=nothing) where {FT<:AbstractFloat}

    n = spec.n_endog
    n_ε = spec.n_exog
    N = T_periods * n
    θ = spec.param_values
    y_ss = Float64.(spec.steady_state)
    shocks_f = Float64.(shocks)

    # Warn for large problems
    if N > 1000
        @warn "NLopt PF with $N decision variables may be slow. " *
              "Consider solver=:ipopt with JuMP + Ipopt for large problems."
    end

    alg_sym = algorithm !== nothing ? algorithm : :LD_SLSQP
    opt = NLopt.Opt(alg_sym, N)

    # Objective: constant zero (feasibility problem)
    NLopt.min_objective!(opt, (x, grad) -> begin
        if length(grad) > 0
            fill!(grad, 0.0)
        end
        return 0.0
    end)

    # Box bounds: stack per-variable bounds across all periods
    lower, upper = _extract_bounds(spec, constraints)
    NLopt.lower_bounds!(opt, repeat(Float64.(lower), T_periods))
    NLopt.upper_bounds!(opt, repeat(Float64.(upper), T_periods))

    # Equality constraints: model equations f_i(y_t, y_{t-1}, y_{t+1}, ε_t, θ) = 0
    for t in 1:T_periods
        for i in 1:n
            pf_eq = _build_pf_equation(spec.residual_fns[i], n, n_ε, θ)
            cb = _pf_nlopt_wrap(pf_eq, t, n, n_ε, T_periods, y_ss, shocks_f)
            NLopt.equality_constraint!(opt, cb, 1e-8)
        end
    end

    # Inequality constraints: NonlinearConstraint fn(...) <= 0
    for c in constraints
        if c isa NonlinearConstraint
            for t in 1:T_periods
                pf_nlcon = _build_pf_nlcon(c.fn, n, n_ε, θ)
                cb = _pf_nlopt_wrap(pf_nlcon, t, n, n_ε, T_periods, y_ss, shocks_f)
                NLopt.inequality_constraint!(opt, cb, 1e-8)
            end
        end
    end

    # Tolerances
    NLopt.xtol_rel!(opt, 1e-10)
    NLopt.ftol_rel!(opt, 1e-10)
    NLopt.maxeval!(opt, 3000)

    # Initial guess: steady state, clamped to bounds
    x0 = repeat(y_ss, T_periods)
    lo_stacked = repeat(Float64.(lower), T_periods)
    hi_stacked = repeat(Float64.(upper), T_periods)
    x0 .= clamp.(x0, lo_stacked, hi_stacked)

    (min_val, min_x, ret) = NLopt.optimize(opt, x0)

    if ret ∉ (:SUCCESS, :FTOL_REACHED, :XTOL_REACHED, :STOPVAL_REACHED)
        throw(ErrorException(
            "NLopt PF solver did not converge (return code: $ret). " *
            "Try solver=:ipopt with JuMP + Ipopt for large-scale NLP."))
    end

    converged = true
    x = Vector{FT}(min_x)

    # Reshape solution
    path_full = reshape(copy(x), n, T_periods)'
    deviations_full = path_full .- spec.steady_state'

    if spec.augmented
        orig_idx = _original_var_indices(spec)
        path = Matrix{FT}(path_full[:, orig_idx])
        deviations = Matrix{FT}(deviations_full[:, orig_idx])
    else
        path = Matrix{FT}(path_full)
        deviations = Matrix{FT}(deviations_full)
    end

    PerfectForesightPath{FT}(path, deviations, converged, 0, spec)
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
