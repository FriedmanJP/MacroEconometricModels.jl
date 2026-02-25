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
Numerical steady-state computation for DSGE models via Optim.jl.
"""

"""
    compute_steady_state(spec::DSGESpec{T}; initial_guess=nothing, method=:auto,
                          ss_fn=nothing, constraints=DSGEConstraint[],
                          solver=nothing) → DSGESpec{T}

Compute the deterministic steady state: f(y_ss, y_ss, y_ss, 0, θ) = 0.

Returns a new `DSGESpec` with the `steady_state` field filled.

# Keywords
- `initial_guess::Vector{T}` — starting point (default: ones)
- `method::Symbol` — `:auto` (NelderMead → LBFGS), `:analytical`
- `ss_fn::Function` — for `:analytical`, a function `θ → y_ss::Vector`
- `constraints::Vector{<:DSGEConstraint}` — variable bounds and nonlinear inequalities (requires JuMP + Ipopt)
- `solver::Symbol` — `:ipopt` (NLP) or `:path` (MCP); auto-detected if not specified
"""
function compute_steady_state(spec::DSGESpec{T};
        initial_guess::Union{Nothing,AbstractVector}=nothing,
        method::Symbol=:auto,
        ss_fn::Union{Nothing,Function}=nothing,
        constraints::Vector=DSGEConstraint[],
        solver::Union{Nothing,Symbol}=nothing) where {T<:AbstractFloat}

    n = spec.n_endog

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
            y_ss = _path_compute_steady_state(spec, constraints;
                        initial_guess=initial_guess)
        else
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

    # Numerical: minimize sum of squared residuals
    y0 = initial_guess !== nothing ? T.(initial_guess) : ones(T, n)
    @assert length(y0) == n "initial_guess must have length $n"

    ε_zero = zeros(T, spec.n_exog)

    # Objective: sum of squared residuals at SS (y_t = y_{t-1} = y_{t+1} = y)
    function ss_objective(y)
        total = zero(T)
        for fn in spec.residual_fns
            r = fn(y, y, y, ε_zero, θ)
            total += r^2
        end
        total
    end

    # Phase 1: Nelder-Mead (derivative-free, robust to bad starting point)
    result = Optim.optimize(ss_objective, y0, Optim.NelderMead(),
                            Optim.Options(iterations=5000, f_reltol=T(1e-12)))
    y_ss = Optim.minimizer(result)

    # Phase 2: Refine with LBFGS if gradient is available
    if Optim.minimum(result) > T(1e-10)
        result2 = Optim.optimize(ss_objective, y_ss, Optim.LBFGS(),
                                 Optim.Options(iterations=2000, f_reltol=T(1e-14)))
        if Optim.minimum(result2) < Optim.minimum(result)
            y_ss = Optim.minimizer(result2)
        end
    end

    # Verify convergence
    final_resid = ss_objective(y_ss)
    final_resid > T(1e-6) && @warn "Steady state may not have converged (residual = $final_resid)"

    _update_steady_state(spec, y_ss)
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
