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
    perfect_foresight(spec::ModelSpec{FT}; T_periods=100, shock_path=nothing,
                       max_iter=100, tol=1e-8, constraints=DSGEConstraint[],
                       solver=nothing, algorithm=nothing) → PerfectForesightPath{FT}

Solve for the deterministic perfect foresight path given a sequence of shocks.

Uses NonlinearSolve.jl as the default Newton solver with block-tridiagonal sparse Jacobian.

# Keywords
- `T_periods::Int=100` — number of simulation periods
- `shock_path::Matrix` — T_periods × n_exog matrix of shock realizations
- `max_iter::Int=100` — Newton iteration limit
- `tol::Real=1e-8` — convergence tolerance (max abs residual); default is `default_abstol(Float64)`
- `abstol::Real=tol` — absolute residual gate passed to the solver; overrides `tol` for the convergence test
- `constraints::Vector{<:DSGEConstraint}` — variable bounds and nonlinear inequalities
- `solver::Symbol` — `:nonlinearsolve` (default), `:nlopt` (SLSQP), `:ipopt` (NLP), or `:path` (MCP); auto-detected
- `algorithm` — NonlinearSolve.jl algorithm (default: `NewtonRaphson()`); passed through to chosen backend
"""
function perfect_foresight(spec::ModelSpec{FT};
        T_periods::Int=100,
        shock_path::Union{Nothing,AbstractMatrix}=nothing,
        max_iter::Int=100,
        tol::Real=default_abstol(Float64),
        abstol::Real=tol,
        constraints::Vector=DSGEConstraint[],
        solver::Union{Nothing,Symbol}=nothing,
        algorithm=nothing,
        sparsity::Symbol=:auto) where {FT<:AbstractFloat}
    sparsity in (:auto, :dense) || throw(ArgumentError(
        "perfect_foresight: sparsity must be :auto or :dense, got :$sparsity"))

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
                        max_iter=max_iter, tol=tol, abstol=abstol,
                        algorithm=algorithm, sparsity=sparsity)
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
            # Escalate to the semismooth projected Newton (always available).
            # Ipopt is NOT a valid escalation target here: `_jump_perfect_foresight`
            # poses hard equalities + bounds, which is infeasible whenever a bound
            # genuinely binds (the binding case is a complementarity problem, #556).
            return _projected_newton_pf(spec, T_periods, shocks, lower, upper;
                        max_iter=max_iter, tol=tol, abstol=abstol)
        elseif chosen == :nlopt
            return _nlopt_perfect_foresight(spec, T_periods, shocks, constraints;
                        algorithm=algorithm)
        elseif chosen == :path
            _check_path_loaded()
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
                max_iter=max_iter, tol=tol, abstol=abstol, algorithm=algorithm,
                sparsity=sparsity)
end

"""Warn when a perfect-foresight path has not returned to the steady state by the terminal
period (`‖deviations[end,:]‖∞` ≫ tol), which signals the horizon `T_periods` is too short (S-20 / #224)."""
function _pf_terminal_warn(deviations_full::AbstractMatrix{T}, tol::Real) where {T<:AbstractFloat}
    isempty(deviations_full) && return nothing
    terminal_resid = maximum(abs.(@view deviations_full[end, :]))
    terminal_resid > 100 * T(tol) && @warn "Perfect-foresight path has not returned to the " *
        "steady state by the terminal period (‖deviation‖∞ = $terminal_resid ≫ $(100 * T(tol))); " *
        "increase the horizon T for an accurate transition."
    return nothing
end

# =============================================================================
# NonlinearSolve-based perfect foresight solver
# =============================================================================

"""
    _nonlinearsolve_perfect_foresight(spec, T_periods, shocks;
                                       max_iter=100, tol=1e-8, algorithm=nothing)

Solve the stacked perfect foresight system using NonlinearSolve.jl (unconstrained).

Packed residual and a cached colored block-tridiagonal Jacobian. The default
`NewtonRaphson(; linsolve=\\)` path solves each Newton step with the Thomas
kernel (`O(T n³)`); a caller-supplied `algorithm` keeps the same Jacobian
assembly and uses that algorithm's linear solver.
"""
function _nonlinearsolve_perfect_foresight(spec::ModelSpec{FT}, T_periods::Int,
        shocks::Matrix{FT};
        max_iter::Int=100, tol::Real=1e-8, abstol::Real=tol, algorithm=nothing,
        sparsity::Symbol=:auto) where {FT<:AbstractFloat}

    n = spec.n_endog
    N = T_periods * n  # total unknowns

    # Initial guess: all periods at steady state
    x0 = repeat(spec.steady_state, T_periods)
    cache = _pf_make_cache(spec, T_periods; sparsity=sparsity)

    function pf_residual!(F, x, p)
        _pf_residual_packed!(F, x, spec, shocks, T_periods)
        return nothing
    end

    function pf_jacobian!(J, x, p)
        _pf_assemble_jacobian!(J, x, spec, shocks, T_periods, cache)
        return nothing
    end

    # Default: native `\` so `ldiv!` on `_PFStackedJac` runs the BT kernel.
    if algorithm === nothing
        J_proto = _PFStackedJac{FT}(cache.J, cache)
        alg = NonlinearSolve.NewtonRaphson(; linsolve=\)
    else
        J_proto = cache.J
        alg = algorithm
    end

    nlfn = NonlinearSolve.NonlinearFunction(pf_residual!; jac=pf_jacobian!, jac_prototype=J_proto)
    prob = NonlinearSolve.NonlinearProblem(nlfn, x0, nothing)

    sol = NonlinearSolve.solve(prob, alg; abstol=FT(abstol), maxiters=max_iter)

    converged = NonlinearSolve.SciMLBase.successful_retcode(sol.retcode)
    if !converged
        extra = sparsity === :auto ?
            "; sparsity detection may have dropped a kinked entry. Retry with sparsity=:dense" : ""
        @warn "Perfect foresight solver did not converge (retcode = $(sol.retcode))$extra"
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

    _pf_terminal_warn(deviations_full, tol)
    PerfectForesightPath{FT}(path, deviations, converged, iter, spec)
end

# =============================================================================
# Projected Newton solver for box-constrained perfect foresight
# =============================================================================

"""
    _projected_newton_pf(spec, T_periods, shocks, lower, upper;
                          max_iter=100, tol=1e-8)

Box-constrained perfect foresight via semismooth (min-map) Newton with NCP
backtracking.

Solves the mixed complementarity system: at interior points the equation in the
variable's slot holds (`F_j = 0`); at a binding lower bound `F_j ≥ 0`; at a
binding upper bound `F_j ≤ 0`. Each iteration estimates the active set from the
min-map `x - F` and replaces the Jacobian rows of pinned variables with identity
rows (the bound replaces the equation), so the reduced Newton step is consistent
with complementarity — a full step `-(J \\ F)` followed by clamping stalls as
soon as a bound genuinely binds (#556). Preserves the block-tridiagonal
sparsity of `_pf_jacobian`.
"""
function _projected_newton_pf(spec::ModelSpec{FT}, T_periods::Int,
        shocks::Matrix{FT}, lower::Vector{FT}, upper::Vector{FT};
        max_iter::Int=100, tol::Real=1e-8, abstol::Real=tol) where {FT<:AbstractFloat}

    n = spec.n_endog
    N = T_periods * n

    # Stack bounds: repeat per-variable bounds across all periods
    lower_stacked = repeat(lower, T_periods)
    upper_stacked = repeat(upper, T_periods)

    # Initial guess: steady state, clamped to bounds
    x = repeat(spec.steady_state, T_periods)
    x .= clamp.(x, lower_stacked, upper_stacked)

    F = zeros(FT, N)
    cache = _pf_make_cache(spec, T_periods)
    _pf_residual_packed!(F, x, spec, shocks, T_periods)

    # NCP natural residual: at interior points F=0; at lower bound F>=0; at upper bound F<=0
    bound_tol = FT(1e-10)
    function _ncp_residual(x_v, F_v)
        ncp = FT(0)
        for j in 1:N
            at_lower = isfinite(lower_stacked[j]) && x_v[j] <= lower_stacked[j] + bound_tol
            at_upper = isfinite(upper_stacked[j]) && x_v[j] >= upper_stacked[j] - bound_tol
            if at_lower
                ncp = max(ncp, max(FT(0), -F_v[j]))  # F should be >= 0
            elseif at_upper
                ncp = max(ncp, max(FT(0), F_v[j]))   # F should be <= 0
            else
                ncp = max(ncp, abs(F_v[j]))           # F should be = 0
            end
        end
        return ncp
    end

    converged = _ncp_residual(x, F) < FT(abstol)
    iter = 0

    x_trial = similar(x)
    F_trial = similar(F)

    for k in 1:max_iter
        converged && break
        iter = k

        _pf_assemble_jacobian!(cache.J, x, spec, shocks, T_periods, cache)
        J = cache.J

        # Min-map active set: variable j is pinned at a bound when x_j - F_j
        # crosses it. Pinned rows are replaced by identity rows so the step
        # drives x_j exactly onto the bound while the remaining equations are
        # solved on the reduced system (complementarity: bound replaces equation).
        pinned_at = fill(FT(NaN), N)   # NaN = free; otherwise the bound value
        for j in 1:N
            z = x[j] - F[j]
            if isfinite(lower_stacked[j]) && z <= lower_stacked[j]
                pinned_at[j] = lower_stacked[j]
            elseif isfinite(upper_stacked[j]) && z >= upper_stacked[j]
                pinned_at[j] = upper_stacked[j]
            end
        end

        rows, cols, vals = findnz(J)
        keep = [isnan(pinned_at[r]) for r in rows]
        pin_idx = findall(!isnan, pinned_at)
        M = sparse(vcat(rows[keep], pin_idx), vcat(cols[keep], pin_idx),
                   vcat(vals[keep], ones(FT, length(pin_idx))), N, N)
        rhs = [isnan(pinned_at[j]) ? -F[j] : pinned_at[j] - x[j] for j in 1:N]

        d = try
            M \ rhs
        catch err
            err isa SingularException || rethrow()
            -(J \ F)  # fall back to the full (clamped) Newton direction
        end

        # Damped step with backtracking on NCP residual
        α = FT(1.0)
        ncp_old = _ncp_residual(x, F)

        for _ls in 1:20
            x_trial .= clamp.(x .+ α .* d, lower_stacked, upper_stacked)
            _pf_residual_packed!(F_trial, x_trial, spec, shocks, T_periods)
            ncp_new = _ncp_residual(x_trial, F_trial)
            if ncp_new < ncp_old || ncp_new < FT(abstol)
                break
            end
            α *= FT(0.5)
        end

        x .= clamp.(x .+ α .* d, lower_stacked, upper_stacked)
        _pf_residual_packed!(F, x, spec, shocks, T_periods)

        if _ncp_residual(x, F) < FT(abstol)
            converged = true
            break
        end
    end

    if !converged
        final_resid = _ncp_residual(x, F)
        throw(ErrorException(
            "Projected Newton PF did not converge after $max_iter iterations " *
            "(||F||_NCP = $(final_resid)). A binding box constraint is a " *
            "complementarity problem: try a larger max_iter, or solver=:path " *
            "(PATH MCP solver, weak dependency)."))
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

    _pf_terminal_warn(deviations_full, tol)
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
function _nlopt_perfect_foresight(spec::ModelSpec{FT}, T_periods::Int,
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
    NLopt.maxeval!(opt, 10000)

    # Initial guess: steady state, clamped to bounds
    x0 = repeat(y_ss, T_periods)
    lo_stacked = repeat(Float64.(lower), T_periods)
    hi_stacked = repeat(Float64.(upper), T_periods)
    x0 .= clamp.(x0, lo_stacked, hi_stacked)

    (min_val, min_x, ret) = NLopt.optimize(opt, x0)

    # Check solution quality: evaluate residual and verify equations are approximately satisfied
    x_sol = Vector{FT}(min_x)
    F_check = zeros(FT, N)
    _pf_residual!(F_check, x_sol, spec, shocks, N ÷ n)
    max_resid = maximum(abs, F_check)

    if ret ∉ (:SUCCESS, :FTOL_REACHED, :XTOL_REACHED, :STOPVAL_REACHED, :MAXEVAL_REACHED, :ROUNDOFF_LIMITED)
        throw(ErrorException(
            "NLopt PF solver failed (return code: $ret). " *
            "Try solver=:ipopt with JuMP + Ipopt for large-scale NLP."))
    end

    if max_resid > 1e-4
        throw(ErrorException(
            "NLopt PF solver did not find a feasible solution (max |F| = $max_resid, return code: $ret). " *
            "Try solver=:ipopt with JuMP + Ipopt for large-scale NLP."))
    end

    converged = max_resid < FT(1e-6)
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

    _pf_terminal_warn(deviations_full, FT(1e-6))
    PerfectForesightPath{FT}(path, deviations, converged, 0, spec)
end

# =============================================================================
# Stacked residual evaluation
# =============================================================================

_pf_use_threads(n::Int) = Threads.nthreads() > 1 && n > 1

_pf_fd_step(x::T) where {T<:AbstractFloat} = max(T(1e-7), T(1e-7) * abs(x))

"""Views of `(y_t, y_{t-1}, y_{t+1})` into the stacked path (`y_0 = y_{T+1} = y_ss`)."""
function _pf_period_states(x::AbstractVector, t::Int, n::Int, Tp::Int, y_ss)
    y_t = view(x, (t - 1) * n + 1:t * n)
    y_lag = t == 1 ? y_ss : view(x, (t - 2) * n + 1:(t - 1) * n)
    y_lead = t == Tp ? y_ss : view(x, t * n + 1:(t + 1) * n)
    return y_t, y_lag, y_lead
end

function _pf_eval_eqs!(F::AbstractVector, fns, y_t, y_lag, y_lead, ε_t, θ)
    @inbounds for i in eachindex(fns)
        F[i] = fns[i](y_t, y_lag, y_lead, ε_t, θ)
    end
    return F
end

function _pf_residual_at!(F::AbstractVector, x::AbstractVector, t::Int, n::Int, Tp::Int,
                          y_ss, θ, fns, shocks::AbstractMatrix)
    y_t, y_lag, y_lead = _pf_period_states(x, t, n, Tp, y_ss)
    ε_t = view(shocks, t, :)
    off = (t - 1) * n
    @inbounds for i in 1:n
        F[off + i] = fns[i](y_t, y_lag, y_lead, ε_t, θ)
    end
    return nothing
end

"""
    _pf_residual_packed!(F, x, spec, shocks, T_periods)

Stacked residual using views into `x` (no per-period n-vector copies). Threads
over `t` when `Threads.nthreads() > 1`.
"""
function _pf_residual_packed!(F::AbstractVector, x::AbstractVector, spec::ModelSpec,
                              shocks::AbstractMatrix, Tp::Int)
    n = spec.n_endog
    y_ss = spec.steady_state
    θ = spec.param_values
    fns = spec.residual_fns
    if _pf_use_threads(Tp)
        # :dynamic: :static cannot nest or run concurrently (Windows CI
        # is one process with @spawn workers; NonlinearSolve may also
        # evaluate f from a worker).
        Threads.@threads :dynamic for t in 1:Tp
            _pf_residual_at!(F, x, t, n, Tp, y_ss, θ, fns, shocks)
        end
    else
        @inbounds for t in 1:Tp
            _pf_residual_at!(F, x, t, n, Tp, y_ss, θ, fns, shocks)
        end
    end
    return nothing
end

"""
    _pf_residual!(F, x, spec, shocks, T_periods)

Evaluate the stacked residual vector in-place.

For each period t = 1, ..., T_periods:
  F[(t-1)*n+1 : t*n] = [f_i(y_t, y_{t-1}, y_{t+1}, ε_t, θ) for i in 1:n]

where y_0 = y_ss (initial) and y_{T+1} = y_ss (terminal).
"""
function _pf_residual!(F::AbstractVector, x::AbstractVector, spec::ModelSpec,
                       shocks::AbstractMatrix, Tp::Int)
    _pf_residual_packed!(F, x, spec, shocks, Tp)
end

# =============================================================================
# Structural sparsity, coloring, cached CSC
# =============================================================================

"""Per-thread workspace for colored finite differences of one intra-period block."""
struct _PFThreadWS{T}
    y_plus::Vector{T}
    y_minus::Vector{T}
    F_plus::Vector{T}
    F_minus::Vector{T}
    h::Vector{T}
end

function _pf_thread_ws(n::Int, ::Type{T}) where {T}
    _PFThreadWS{T}(zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n))
end

"""
Cached PF Jacobian: intra-period sparsity + coloring, one CSC pattern, and the
`n × n` block arrays used by the Thomas solve.
"""
struct _PFJacCache{T}
    n::Int
    Tp::Int
    pat_lag::BitMatrix
    pat_t::BitMatrix
    pat_lead::BitMatrix
    colors_lag::Vector{Vector{Int}}
    colors_t::Vector{Vector{Int}}
    colors_lead::Vector{Vector{Int}}
    rows_lag::Vector{Vector{Int}}
    rows_t::Vector{Vector{Int}}
    rows_lead::Vector{Vector{Int}}
    J::SparseMatrixCSC{T,Int}
    A_blocks::Vector{Matrix{T}}
    B_blocks::Vector{Matrix{T}}
    C_blocks::Vector{Matrix{T}}
    Y_blocks::Vector{Matrix{T}}
    d_blocks::Vector{Vector{T}}
    Lwork::Matrix{T}
    W::Matrix{T}
    vtmp::Vector{T}
    workspaces::Vector{_PFThreadWS{T}}
end

"""
    _PFStackedJac{T}

Block-tridiagonal stacked Jacobian. `ldiv!` runs the Thomas kernel; `J` is the
cached CSC used as a LU fallback and as the NonlinearSolve prototype.
"""
struct _PFStackedJac{T} <: AbstractMatrix{T}
    J::SparseMatrixCSC{T,Int}
    cache::_PFJacCache{T}
end

Base.size(A::_PFStackedJac) = size(A.J)
Base.getindex(A::_PFStackedJac, i::Integer, j::Integer) = A.J[i, j]
Base.eltype(::Type{_PFStackedJac{T}}) where {T} = T
Base.similar(A::_PFStackedJac{T}) where {T} = _PFStackedJac{T}(similar(A.J), A.cache)
Base.similar(A::_PFStackedJac, ::Type{S}) where {S} = _PFStackedJac{S}(similar(A.J, S), A.cache)
Base.zero(A::_PFStackedJac{T}) where {T} = _PFStackedJac{T}(zero(A.J), A.cache)
Base.copy(A::_PFStackedJac{T}) where {T} = _PFStackedJac{T}(copy(A.J), A.cache)
function Base.fill!(A::_PFStackedJac, x)
    fill!(A.J, x)
    return A
end

function LinearAlgebra.ldiv!(u::AbstractVector, A::_PFStackedJac, b::AbstractVector)
    if !_pf_bt_solve!(u, A.cache, b)
        u .= A.J \ b
    end
    return u
end

function LinearAlgebra.ldiv!(A::_PFStackedJac, b::AbstractVector)
    tmp = copy(b)
    ldiv!(b, A, tmp)
    return b
end

function Base.:(\)(A::_PFStackedJac, b::AbstractVector)
    u = similar(b, promote_type(eltype(A), eltype(b)))
    ldiv!(u, A, b)
    return u
end

"""Greedy column coloring: columns sharing a color have disjoint row supports."""
function _pf_greedy_color(pat::AbstractMatrix{Bool})
    n = size(pat, 2)
    n_rows = size(pat, 1)
    color = zeros(Int, n)
    ncolors = 0
    forbidden = falses(n)
    @inbounds for j in 1:n
        has_nz = false
        for i in 1:n_rows
            if pat[i, j]
                has_nz = true
                break
            end
        end
        has_nz || continue
        fill!(forbidden, false)
        for i in 1:n_rows
            pat[i, j] || continue
            for k in 1:j-1
                c = color[k]
                if c > 0 && pat[i, k]
                    forbidden[c] = true
                end
            end
        end
        c = 1
        while c <= ncolors && forbidden[c]
            c += 1
        end
        if c > ncolors
            ncolors = c
        end
        color[j] = c
    end
    groups = [Int[] for _ in 1:ncolors]
    @inbounds for j in 1:n
        color[j] == 0 && continue
        push!(groups[color[j]], j)
    end
    return groups
end

function _pf_pattern_rows(pat::AbstractMatrix{Bool})
    n = size(pat, 2)
    rows = [Int[] for _ in 1:n]
    @inbounds for j in 1:n, i in 1:size(pat, 1)
        pat[i, j] && push!(rows[j], i)
    end
    return rows
end

function _pf_mark_pattern!(pat::AbstractMatrix{Bool}, F0::AbstractVector{T},
                           F1::AbstractVector{T}, j::Int) where {T}
    @inbounds for i in eachindex(F0)
        thresh = max(T(1e-12), T(1e-10) * (abs(F0[i]) + abs(F1[i])))
        if abs(F1[i] - F0[i]) > thresh
            pat[i, j] = true
        end
    end
    return nothing
end

"""Union structural sparsity of ∂F/∂y_lag, ∂F/∂y_t, ∂F/∂y_lead at several probes."""
function _pf_detect_sparsity(spec::ModelSpec{T}) where {T}
    n = spec.n_endog
    n_ε = spec.n_exog
    fns = spec.residual_fns
    θ = spec.param_values
    y_ss = spec.steady_state
    if length(y_ss) != n
        y_ss = zeros(T, n)
    end

    probes = Vector{Vector{T}}()
    push!(probes, copy(y_ss))
    y2 = similar(y_ss)
    @inbounds for i in 1:n
        y2[i] = y_ss[i] + T(0.1) * (one(T) + abs(y_ss[i]))
    end
    push!(probes, y2)
    y3 = similar(y_ss)
    @inbounds for i in 1:n
        y3[i] = y_ss[i] + T(0.07) * (iseven(i) ? one(T) : -one(T)) * (one(T) + abs(y_ss[i]))
    end
    push!(probes, y3)
    y4 = similar(y_ss)
    @inbounds for i in 1:n
        y4[i] = y_ss[i] + T(0.5) * (one(T) + abs(y_ss[i]))
    end
    push!(probes, y4)
    y5 = similar(y_ss)
    @inbounds for i in 1:n
        y5[i] = y_ss[i] - T(0.5) * (one(T) + abs(y_ss[i]))
    end
    push!(probes, y5)

    ε_probes = (zeros(T, n_ε), n_ε == 0 ? zeros(T, 0) : fill(T(0.1), n_ε))

    pat_lag = falses(n, n)
    pat_t = falses(n, n)
    pat_lead = falses(n, n)
    F0 = zeros(T, n)
    F1 = zeros(T, n)
    y_work = zeros(T, n)

    for y_p in probes, ε_p in ε_probes
        try
            _pf_eval_eqs!(F0, fns, y_p, y_p, y_p, ε_p, θ)
        catch
            fill!(pat_t, true)
            fill!(pat_lag, true)
            fill!(pat_lead, true)
            @goto done_probes
        end
        for j in 1:n
            h = max(T(1e-6), T(1e-4) * (one(T) + abs(y_p[j])))
            copyto!(y_work, y_p)
            y_work[j] += h
            try
                _pf_eval_eqs!(F1, fns, y_work, y_p, y_p, ε_p, θ)
                _pf_mark_pattern!(pat_t, F0, F1, j)
            catch
                pat_t[:, j] .= true
            end
            copyto!(y_work, y_p)
            y_work[j] += h
            try
                _pf_eval_eqs!(F1, fns, y_p, y_work, y_p, ε_p, θ)
                _pf_mark_pattern!(pat_lag, F0, F1, j)
            catch
                pat_lag[:, j] .= true
            end
            copyto!(y_work, y_p)
            y_work[j] += h
            try
                _pf_eval_eqs!(F1, fns, y_p, y_p, y_work, ε_p, θ)
                _pf_mark_pattern!(pat_lead, F0, F1, j)
            catch
                pat_lead[:, j] .= true
            end
        end
    end
    @label done_probes

    # Degenerate probe (all-zero pattern): fall back to dense intra-period blocks.
    if !any(pat_t) && !any(pat_lag) && !any(pat_lead)
        fill!(pat_t, true)
        fill!(pat_lag, true)
        fill!(pat_lead, true)
    end
    return pat_lag, pat_t, pat_lead
end

function _pf_build_csc_pattern(n::Int, Tp::Int, pat_lag, pat_t, pat_lead, ::Type{T}) where {T}
    N = Tp * n
    rows = Int[]
    cols = Int[]
    sizehint!(rows, 3 * Tp * n)
    sizehint!(cols, 3 * Tp * n)
    for t in 1:Tp
        row_off = (t - 1) * n
        col_off = (t - 1) * n
        @inbounds for j in 1:n, i in 1:n
            if pat_t[i, j]
                push!(rows, row_off + i)
                push!(cols, col_off + j)
            end
        end
        if t >= 2
            col_off_lag = (t - 2) * n
            @inbounds for j in 1:n, i in 1:n
                if pat_lag[i, j]
                    push!(rows, row_off + i)
                    push!(cols, col_off_lag + j)
                end
            end
        end
        if t <= Tp - 1
            col_off_lead = t * n
            @inbounds for j in 1:n, i in 1:n
                if pat_lead[i, j]
                    push!(rows, row_off + i)
                    push!(cols, col_off_lead + j)
                end
            end
        end
    end
    # Structural ones: keep every probed (i,j) even if a later FD value is ~0.
    return sparse(rows, cols, ones(T, length(rows)), N, N)
end

function _pf_make_cache(spec::ModelSpec{T}, Tp::Int; sparsity::Symbol=:auto) where {T}
    n = spec.n_endog
    if sparsity === :dense
        pat_lag = trues(n, n)
        pat_t = trues(n, n)
        pat_lead = trues(n, n)
    else
        pat_lag, pat_t, pat_lead = _pf_detect_sparsity(spec)
    end
    colors_lag = _pf_greedy_color(pat_lag)
    colors_t = _pf_greedy_color(pat_t)
    colors_lead = _pf_greedy_color(pat_lead)
    J = _pf_build_csc_pattern(n, Tp, pat_lag, pat_t, pat_lead, T)
    nth = max(1, Threads.nthreads())
    _PFJacCache{T}(
        n, Tp, pat_lag, pat_t, pat_lead,
        colors_lag, colors_t, colors_lead,
        _pf_pattern_rows(pat_lag), _pf_pattern_rows(pat_t), _pf_pattern_rows(pat_lead),
        J,
        [zeros(T, n, n) for _ in 1:Tp],
        [zeros(T, n, n) for _ in 1:Tp],
        [zeros(T, n, n) for _ in 1:Tp],
        [zeros(T, n, n) for _ in 1:Tp],
        [zeros(T, n) for _ in 1:Tp],
        zeros(T, n, n), zeros(T, n, n), zeros(T, n),
        [_pf_thread_ws(n, T) for _ in 1:nth],
    )
end

# =============================================================================
# Colored intra-period FD + in-place CSC write
# =============================================================================

function _pf_color_fd!(Jblock::AbstractMatrix{T}, groups, rows_of,
                       y_slot, y_t, y_lag, y_lead, ε_t, fns, θ, ws::_PFThreadWS{T},
                       which::Symbol) where {T}
    fill!(Jblock, zero(T))
    isempty(groups) && return nothing
    @inbounds for group in groups
        copyto!(ws.y_plus, y_slot)
        copyto!(ws.y_minus, y_slot)
        for (k, j) in enumerate(group)
            hj = _pf_fd_step(y_slot[j])
            ws.h[k] = hj
            ws.y_plus[j] += hj
            ws.y_minus[j] -= hj
        end
        if which === :t
            _pf_eval_eqs!(ws.F_plus, fns, ws.y_plus, y_lag, y_lead, ε_t, θ)
            _pf_eval_eqs!(ws.F_minus, fns, ws.y_minus, y_lag, y_lead, ε_t, θ)
        elseif which === :lag
            _pf_eval_eqs!(ws.F_plus, fns, y_t, ws.y_plus, y_lead, ε_t, θ)
            _pf_eval_eqs!(ws.F_minus, fns, y_t, ws.y_minus, y_lead, ε_t, θ)
        else
            _pf_eval_eqs!(ws.F_plus, fns, y_t, y_lag, ws.y_plus, ε_t, θ)
            _pf_eval_eqs!(ws.F_minus, fns, y_t, y_lag, ws.y_minus, ε_t, θ)
        end
        for (k, j) in enumerate(group)
            inv2h = inv(2 * ws.h[k])
            for i in rows_of[j]
                Jblock[i, j] = (ws.F_plus[i] - ws.F_minus[i]) * inv2h
            end
        end
    end
    return nothing
end

function _pf_period_blocks!(cache::_PFJacCache{T}, x::AbstractVector, t::Int,
                            spec::ModelSpec{T}, shocks::AbstractMatrix,
                            ws::_PFThreadWS{T}) where {T}
    n = cache.n
    Tp = cache.Tp
    fns = spec.residual_fns
    θ = spec.param_values
    y_ss = spec.steady_state
    y_t, y_lag, y_lead = _pf_period_states(x, t, n, Tp, y_ss)
    ε_t = view(shocks, t, :)

    _pf_color_fd!(cache.A_blocks[t], cache.colors_t, cache.rows_t,
                  y_t, y_t, y_lag, y_lead, ε_t, fns, θ, ws, :t)
    if t >= 2
        _pf_color_fd!(cache.B_blocks[t], cache.colors_lag, cache.rows_lag,
                      y_lag, y_t, y_lag, y_lead, ε_t, fns, θ, ws, :lag)
    else
        fill!(cache.B_blocks[t], zero(T))
    end
    if t <= Tp - 1
        _pf_color_fd!(cache.C_blocks[t], cache.colors_lead, cache.rows_lead,
                      y_lead, y_t, y_lag, y_lead, ε_t, fns, θ, ws, :lead)
    else
        fill!(cache.C_blocks[t], zero(T))
    end
    return nothing
end

function _pf_compute_blocks!(cache::_PFJacCache{T}, x::AbstractVector, spec::ModelSpec{T},
                             shocks::AbstractMatrix, Tp::Int) where {T}
    if _pf_use_threads(Tp)
        # Private workspace per period: threadid() is not 1:nthreads() on
        # Julia 1.11+ (interactive pool), so a shared workspaces[tid] races.
        Threads.@threads :dynamic for t in 1:Tp
            ws = _pf_thread_ws(cache.n, T)
            _pf_period_blocks!(cache, x, t, spec, shocks, ws)
        end
    else
        ws = cache.workspaces[1]
        @inbounds for t in 1:Tp
            _pf_period_blocks!(cache, x, t, spec, shocks, ws)
        end
    end
    return nothing
end

function _pf_scatter_blocks!(J::SparseMatrixCSC{T}, cache::_PFJacCache{T}) where {T}
    n = cache.n
    nz = nonzeros(J)
    rv = rowvals(J)
    fill!(nz, zero(T))
    @inbounds for col in 1:size(J, 2)
        t_col = (col - 1) ÷ n + 1
        j = (col - 1) % n + 1
        for p in nzrange(J, col)
            row = rv[p]
            t_row = (row - 1) ÷ n + 1
            i = (row - 1) % n + 1
            Δt = t_row - t_col
            if Δt == 0
                nz[p] = cache.A_blocks[t_row][i, j]
            elseif Δt == 1
                nz[p] = cache.B_blocks[t_row][i, j]
            elseif Δt == -1
                nz[p] = cache.C_blocks[t_row][i, j]
            end
        end
    end
    return nothing
end

function _pf_assemble_jacobian!(J::SparseMatrixCSC, x::AbstractVector, spec::ModelSpec,
                                shocks::AbstractMatrix, Tp::Int, cache::_PFJacCache)
    _pf_compute_blocks!(cache, x, spec, shocks, Tp)
    _pf_scatter_blocks!(J, cache)
    return J
end

function _pf_assemble_jacobian!(J::_PFStackedJac, x::AbstractVector, spec::ModelSpec,
                                shocks::AbstractMatrix, Tp::Int, cache::_PFJacCache)
    _pf_compute_blocks!(cache, x, spec, shocks, Tp)
    _pf_scatter_blocks!(J.J, cache)
    return J
end

# =============================================================================
# Block-tridiagonal Thomas solve  (J u = b)
# =============================================================================

"""
    _pf_bt_solve!(u, cache, b) → Bool

Block Thomas algorithm on the `T` n×n blocks stored in `cache`. Returns `false`
if a diagonal block is singular (caller should fall back to sparse LU).
"""
function _pf_bt_solve!(u::AbstractVector{T}, cache::_PFJacCache{T},
                       b::AbstractVector) where {T}
    n = cache.n
    Tp = cache.Tp
    A = cache.A_blocks
    B = cache.B_blocks
    C = cache.C_blocks
    Y = cache.Y_blocks
    d = cache.d_blocks
    Lwork = cache.Lwork
    W = cache.W
    vtmp = cache.vtmp

    try
        copyto!(Lwork, A[1])
        F1 = lu(Lwork)
        copyto!(d[1], 1, b, 1, n)
        ldiv!(F1, d[1])
        if Tp > 1
            copyto!(Y[1], C[1])
            ldiv!(F1, Y[1])
        end
        for t in 2:Tp
            mul!(W, B[t], Y[t - 1])
            copyto!(Lwork, A[t])
            Lwork .-= W
            Ft = lu(Lwork)
            mul!(d[t], B[t], d[t - 1])
            off = (t - 1) * n
            @inbounds for i in 1:n
                d[t][i] = b[off + i] - d[t][i]
            end
            ldiv!(Ft, d[t])
            if t < Tp
                copyto!(Y[t], C[t])
                ldiv!(Ft, Y[t])
            end
        end
        copyto!(u, (Tp - 1) * n + 1, d[Tp], 1, n)
        for t in (Tp - 1):-1:1
            mul!(vtmp, Y[t], view(u, t * n + 1:(t + 1) * n))
            off = (t - 1) * n
            @inbounds for i in 1:n
                u[off + i] = d[t][i] - vtmp[i]
            end
        end
        return true
    catch err
        (err isa SingularException || err isa LinearAlgebra.LAPACKException) || rethrow()
        return false
    end
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

Uses colored central differences on the probed intra-period sparsity; the CSC
pattern is allocated once and `nonzeros` are written in place.
"""
function _pf_jacobian(x::AbstractVector{FT}, spec::ModelSpec{FT},
                      shocks::AbstractMatrix{FT}, Tp::Int) where {FT}
    cache = _pf_make_cache(spec, Tp)
    _pf_assemble_jacobian!(cache.J, x, spec, shocks, Tp, cache)
    return cache.J
end

