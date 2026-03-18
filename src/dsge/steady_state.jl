# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Numerical steady-state computation for DSGE models via NonlinearSolve.jl.
"""

"""
    compute_steady_state(spec::DSGESpec{T}; initial_guess=nothing, method=:auto,
                          ss_fn=nothing, constraints=DSGEConstraint[],
                          solver=nothing, algorithm=nothing) → DSGESpec{T}

Compute the deterministic steady state: f(y_ss, y_ss, y_ss, 0, θ) = 0.

Returns a new `DSGESpec` with the `steady_state` field filled.

# Keywords
- `initial_guess::Vector{T}` — starting point (default: ones)
- `method::Symbol` — `:auto` (NonlinearSolve), `:analytical`
- `ss_fn::Function` — for `:analytical`, a function `θ → y_ss::Vector`
- `constraints::Vector{<:DSGEConstraint}` — variable bounds and nonlinear inequalities
- `solver::Symbol` — `:nonlinearsolve` (default), `:optim` (Fminbox), `:nlopt` (SLSQP), `:ipopt` (NLP), or `:path` (MCP); auto-detected if not specified
- `algorithm` — NonlinearSolve.jl algorithm (default: `TrustRegion()`); passed through to chosen backend
"""
function compute_steady_state(spec::DSGESpec{T};
        initial_guess::Union{Nothing,AbstractVector}=nothing,
        method::Symbol=:auto,
        ss_fn::Union{Nothing,Function}=nothing,
        constraints::Vector=DSGEConstraint[],
        solver::Union{Nothing,Symbol}=nothing,
        algorithm=nothing) where {T<:AbstractFloat}

    n = spec.n_endog

    # If constraints are provided, dispatch based on solver
    if !isempty(constraints)
        _validate_constraints(spec, constraints)
        chosen = _select_solver(constraints, solver)
        chosen ∉ (:path, :ipopt, :nonlinearsolve, :optim, :nlopt) &&
            throw(ArgumentError("Unknown solver :$chosen. " *
                "Valid options: :nonlinearsolve, :optim, :nlopt, :path, :ipopt"))

        if chosen == :nonlinearsolve
            lower, upper = _extract_bounds(spec, constraints)
            any(c -> c isa NonlinearConstraint, constraints) &&
                throw(ArgumentError(
                    "NonlinearSolve solver cannot handle NonlinearConstraint. " *
                    "Use solver=:nlopt (default) or solver=:ipopt for nonlinear inequality constraints."))
            y_ss = _nonlinearsolve_steady_state(spec, lower, upper;
                        initial_guess=initial_guess, algorithm=algorithm)
            # Check if bounds are violated — escalate to Optim if needed
            bounds_ok = all(i -> (!isfinite(lower[i]) || y_ss[i] >= lower[i] - T(1e-6)) &&
                                  (!isfinite(upper[i]) || y_ss[i] <= upper[i] + T(1e-6)),
                            1:n)
            if !bounds_ok
                y_ss = _optim_steady_state(spec, lower, upper;
                            initial_guess=initial_guess, algorithm=nothing)
            end
        elseif chosen == :optim
            lower, upper = _extract_bounds(spec, constraints)
            any(c -> c isa NonlinearConstraint, constraints) &&
                throw(ArgumentError(
                    "Optim solver cannot handle NonlinearConstraint. " *
                    "Use solver=:nlopt or solver=:ipopt."))
            y_ss = _optim_steady_state(spec, lower, upper;
                        initial_guess=initial_guess, algorithm=algorithm)
        elseif chosen == :nlopt
            y_ss = _nlopt_steady_state(spec, constraints;
                        initial_guess=initial_guess, algorithm=algorithm)
        elseif chosen == :path
            _check_jump_loaded()
            any(c -> c isa NonlinearConstraint, constraints) &&
                throw(ArgumentError(
                    "PATH solver cannot handle NonlinearConstraint. " *
                    "Use solver=:nlopt or solver=:ipopt."))
            y_ss = _path_compute_steady_state(spec, constraints;
                        initial_guess=initial_guess)
        else  # :ipopt
            _check_jump_loaded()
            y_ss = _jump_compute_steady_state(spec, constraints;
                        initial_guess=initial_guess)
        end
        return _update_steady_state(spec, Vector{T}(y_ss))
    end

    θ = spec.param_values

    # Auto-detect: if spec has an analytical ss_fn, use it
    if method == :auto && spec.ss_fn !== nothing
        y_ss = T.(spec.ss_fn(θ))
        @assert length(y_ss) == n "ss_fn must return vector of length $n"
        return _update_steady_state(spec, y_ss)
    end

    if method == :analytical
        ss_fn === nothing && throw(ArgumentError("method=:analytical requires ss_fn"))
        y_ss = T.(ss_fn(θ))
        @assert length(y_ss) == n "ss_fn must return vector of length $n"
        return _update_steady_state(spec, y_ss)
    end

    # Numerical: use NonlinearSolve (unconstrained, infinite bounds)
    lower = fill(T(-Inf), n)
    upper = fill(T(Inf), n)
    y_ss = _nonlinearsolve_steady_state(spec, lower, upper;
                initial_guess=initial_guess, algorithm=algorithm)

    _update_steady_state(spec, y_ss)
end

"""
    _nonlinearsolve_steady_state(spec, lower, upper; initial_guess=nothing, algorithm=nothing)

Solve the steady-state system f(y, y, y, 0, θ) = 0 using NonlinearSolve.jl.

Default algorithm is `TrustRegion()`. Box bounds are applied when finite.
"""
function _nonlinearsolve_steady_state(spec::DSGESpec{T}, lower::Vector{T}, upper::Vector{T};
        initial_guess::Union{Nothing,AbstractVector}=nothing,
        algorithm=nothing) where {T<:AbstractFloat}

    n = spec.n_endog
    if initial_guess !== nothing
        y0 = T.(initial_guess)
    elseif !isempty(spec.steady_state)
        y0 = T.(spec.steady_state)
    else
        y0 = ones(T, n)
    end
    @assert length(y0) == n "initial_guess must have length $n"

    θ = spec.param_values
    ε_zero = zeros(T, spec.n_exog)
    fns = spec.residual_fns

    # Residual function: F[i] = f_i(y, y, y, 0, θ)
    function ss_residual!(F, y, p)
        for i in eachindex(fns)
            F[i] = fns[i](y, y, y, ε_zero, θ)
        end
        return nothing
    end

    # Jacobian via central differences
    function ss_jacobian!(J, y, p)
        F_plus = similar(y)
        F_minus = similar(y)
        for j in 1:n
            h = max(T(1e-7), T(1e-7) * abs(y[j]))
            y_p = copy(y)
            y_m = copy(y)
            y_p[j] += h
            y_m[j] -= h
            ss_residual!(F_plus, y_p, p)
            ss_residual!(F_minus, y_m, p)
            for i in 1:n
                J[i, j] = (F_plus[i] - F_minus[i]) / (2 * h)
            end
        end
        return nothing
    end

    nlfn = NonlinearSolve.NonlinearFunction(ss_residual!; jac=ss_jacobian!)

    # Determine if bounds are finite
    has_finite_bounds = any(isfinite, lower) || any(isfinite, upper)

    if has_finite_bounds
        # Clamp initial guess to bounds
        for i in 1:n
            y0[i] = clamp(y0[i], isfinite(lower[i]) ? lower[i] : T(-1e10),
                                   isfinite(upper[i]) ? upper[i] : T(1e10))
        end
        prob = NonlinearSolve.NonlinearProblem(nlfn, y0, nothing; lb=lower, ub=upper)
    else
        prob = NonlinearSolve.NonlinearProblem(nlfn, y0, nothing)
    end

    alg = algorithm !== nothing ? algorithm : NonlinearSolve.TrustRegion()
    sol = NonlinearSolve.solve(prob, alg; abstol=T(1e-10), maxiters=5000)

    if !NonlinearSolve.SciMLBase.successful_retcode(sol.retcode)
        @warn "Steady state solver did not converge (retcode = $(sol.retcode))"
    end

    return Vector{T}(sol.u)
end

"""
    _optim_steady_state(spec, lower, upper; initial_guess=nothing, algorithm=nothing)

Box-constrained steady state via Optim.jl Fminbox(LBFGS()).
Minimizes sum-of-squared-residuals subject to variable bounds.
"""
function _optim_steady_state(spec::DSGESpec{T}, lower::Vector{T}, upper::Vector{T};
        initial_guess::Union{Nothing,AbstractVector}=nothing,
        algorithm=nothing) where {T<:AbstractFloat}

    n = spec.n_endog
    if initial_guess !== nothing
        y0 = T.(initial_guess)
    elseif !isempty(spec.steady_state)
        y0 = T.(spec.steady_state)
    else
        y0 = ones(T, n)
    end
    @assert length(y0) == n "initial_guess must have length $n"

    # Clamp initial guess to bounds, nudging slightly into interior for Fminbox
    eps_nudge = T(1e-8)
    for i in 1:n
        lo = isfinite(lower[i]) ? lower[i] : T(-1e10)
        hi = isfinite(upper[i]) ? upper[i] : T(1e10)
        y0[i] = clamp(y0[i], lo + eps_nudge, hi - eps_nudge)
    end

    # Build objective: sum of squared residuals (vector interface via _vec_wrap)
    ss_obj_splat = _build_ss_objective(spec.residual_fns, spec.n_exog, spec.param_values)
    obj = _vec_wrap(ss_obj_splat)

    alg = algorithm !== nothing ? algorithm : Optim.Fminbox(Optim.LBFGS())

    # Build OnceDifferentiable with explicit ForwardDiff gradient for Fminbox
    grad! = (G, x) -> ForwardDiff.gradient!(G, obj, x)
    od = Optim.OnceDifferentiable(obj, grad!, y0)
    result = Optim.optimize(od, lower, upper, y0, alg,
                Optim.Options(iterations=5000, g_tol=T(1e-12), show_trace=false))

    if !Optim.converged(result)
        @warn "Optim steady state solver did not converge. " *
              "Try solver=:ipopt with JuMP + Ipopt for large-scale NLP."
    end

    y_ss = Vector{T}(Optim.minimizer(result))

    # Verify residuals are near zero
    final_obj = obj(y_ss)
    if final_obj > T(1e-6)
        @warn "Optim converged to non-zero residual (||F||² = $(final_obj)). " *
              "Steady state may not be accurate."
    end

    return y_ss
end

"""
    _nlopt_steady_state(spec, constraints; initial_guess=nothing, algorithm=nothing)

Constrained steady state via NLopt LD_SLSQP.
Handles both VariableBound (box) and NonlinearConstraint (inequality).
"""
function _nlopt_steady_state(spec::DSGESpec{T}, constraints::Vector;
        initial_guess::Union{Nothing,AbstractVector}=nothing,
        algorithm=nothing) where {T<:AbstractFloat}

    n = spec.n_endog
    # NLopt.jl only supports Float64 — convert everything, then convert back at return
    if initial_guess !== nothing
        y0 = Float64.(initial_guess)
    elseif !isempty(spec.steady_state)
        y0 = Float64.(spec.steady_state)
    else
        y0 = ones(Float64, n)
    end
    @assert length(y0) == n "initial_guess must have length $n"

    # Build objective: sum of squared residuals
    ss_obj_splat = _build_ss_objective(spec.residual_fns, spec.n_exog, spec.param_values)
    nlopt_obj = _nlopt_wrap(ss_obj_splat)

    # Choose algorithm
    alg_sym = algorithm !== nothing ? algorithm : :LD_SLSQP
    opt = NLopt.Opt(alg_sym, n)

    # Set objective
    NLopt.min_objective!(opt, nlopt_obj)

    # Box bounds
    lower, upper = _extract_bounds(spec, constraints)
    NLopt.lower_bounds!(opt, Float64.(lower))
    NLopt.upper_bounds!(opt, Float64.(upper))

    # Nonlinear inequality constraints: fn(y, y, y, 0, θ) <= 0
    for c in constraints
        if c isa NonlinearConstraint
            ss_nlcon_splat = _build_ss_nlcon(c.fn, spec.n_exog, spec.param_values)
            nlopt_con = _nlopt_wrap(ss_nlcon_splat)
            NLopt.inequality_constraint!(opt, nlopt_con, 1e-8)
        end
    end

    # Tolerances
    NLopt.xtol_rel!(opt, 1e-12)
    NLopt.ftol_rel!(opt, 1e-12)
    NLopt.maxeval!(opt, 5000)

    # Clamp initial guess
    lo_f = Float64.(lower)
    hi_f = Float64.(upper)
    for i in 1:n
        y0[i] = clamp(y0[i], isfinite(lo_f[i]) ? lo_f[i] : -1e10,
                               isfinite(hi_f[i]) ? hi_f[i] : 1e10)
    end

    (min_val, min_x, ret) = NLopt.optimize(opt, y0)

    if ret ∉ (:SUCCESS, :FTOL_REACHED, :XTOL_REACHED, :STOPVAL_REACHED)
        throw(ErrorException(
            "NLopt steady state solver did not converge (return code: $ret). " *
            "Try solver=:ipopt with JuMP + Ipopt for large-scale NLP."))
    end

    # Verify residuals are near zero (warn if not — binding constraints make nonzero residual expected)
    if min_val > 1e-6
        @warn "NLopt converged to non-zero residual (||F||² = $(min_val)). " *
              "This is expected when constraints are binding. Verify solution accuracy."
    end

    return Vector{T}(min_x)
end

"""Return a new DSGESpec with updated steady_state field."""
function _update_steady_state(spec::DSGESpec{T}, y_ss::Vector{T}) where {T}
    DSGESpec{T}(
        spec.endog, spec.exog, spec.params, spec.param_values,
        spec.equations, spec.residual_fns,
        spec.n_expect, spec.forward_indices, y_ss, spec.ss_fn;
        original_endog=spec.original_endog,
        original_equations=spec.original_equations,
        augmented=spec.augmented,
        max_lag=spec.max_lag,
        max_lead=spec.max_lead
    )
end
