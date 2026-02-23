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
OccBin occasionally binding constraint solver — constraint parsing and regime derivation.

Implements the piecewise-linear solution method of Guerrieri & Iacoviello (2015).

References:
- Guerrieri, L., & Iacoviello, M. (2015). OccBin: A toolkit for solving dynamic models
  with occasionally binding constraints easily. Journal of Monetary Economics, 70, 22-38.
"""

# =============================================================================
# Constraint Parsing
# =============================================================================

"""
    parse_constraint(expr::Expr, spec::DSGESpec{T}) → OccBinConstraint{T}

Parse a constraint expression of the form `:(var[t] >= bound)` or `:(var[t] <= bound)`.

# Arguments
- `expr` — constraint expression, e.g. `:(i[t] >= 0)` or `:(y[t] <= 1.0)`
- `spec` — DSGE model specification (used to validate variable names)

# Returns
An `OccBinConstraint{T}` with extracted variable, bound, and direction.

# Throws
- `ArgumentError` if the variable is not found in `spec.endog`
- `ArgumentError` if the expression is not a valid constraint format
"""
function parse_constraint(expr::Expr, spec::DSGESpec{T}) where {T<:AbstractFloat}
    # Parse the constraint expression
    variable, bound_val, direction = _parse_constraint_expr(expr, T)

    # Validate variable exists in endogenous
    variable in spec.endog || throw(ArgumentError(
        "Constrained variable :$variable not found in endogenous variables $(spec.endog)"))

    bound = T(bound_val)

    # Build the binding expression: var[t] = bound
    bind_expr = Expr(:(=), Expr(:ref, variable, :t), bound)

    OccBinConstraint{T}(expr, variable, bound, direction, bind_expr)
end

"""
    _parse_constraint_expr(expr::Expr, ::Type{T}) → (variable, bound, direction)

Extract variable name, bound value, and direction from a constraint expression.

Supported forms:
- `:(var[t] >= bound)` → (:var, bound, :geq)
- `:(var[t] <= bound)` → (:var, bound, :leq)
"""
function _parse_constraint_expr(expr::Expr, ::Type{T}) where {T}
    # Constraint must be a comparison: (call >= lhs rhs) or (call <= lhs rhs)
    expr.head == :call || throw(ArgumentError(
        "Constraint must be a comparison (>= or <=), got expression with head :$(expr.head)"))

    length(expr.args) == 3 || throw(ArgumentError(
        "Constraint must have exactly two operands, got $(length(expr.args) - 1)"))

    op = expr.args[1]
    lhs = expr.args[2]
    rhs = expr.args[3]

    if op === :(>=)
        direction = :geq
    elseif op === :(<=)
        direction = :leq
    else
        throw(ArgumentError(
            "Constraint operator must be >= or <=, got $op"))
    end

    variable = _extract_constrained_var(lhs)
    bound = _eval_bound(rhs, T)

    return (variable, bound, direction)
end

"""
    _extract_constrained_var(lhs) → Symbol

Extract the variable name from the LHS of a constraint (expected form: `var[t]`).
"""
function _extract_constrained_var(lhs)
    if lhs isa Expr && lhs.head == :ref && length(lhs.args) == 2 &&
       lhs.args[1] isa Symbol && lhs.args[2] === :t
        return lhs.args[1]::Symbol
    end
    throw(ArgumentError(
        "Constraint LHS must be var[t], got: $lhs"))
end

"""
    _eval_bound(rhs, ::Type{T}) → T

Evaluate the RHS bound of a constraint to a numeric value.
"""
function _eval_bound(rhs, ::Type{T}) where {T}
    if rhs isa Number
        return T(rhs)
    elseif rhs isa Expr
        # Try to evaluate simple expressions (e.g., -1.0, 1/400)
        try
            val = eval(rhs)
            return T(val)
        catch
            throw(ArgumentError("Cannot evaluate constraint bound expression: $rhs"))
        end
    else
        throw(ArgumentError("Constraint bound must be a numeric value, got: $rhs ($(typeof(rhs)))"))
    end
end

# =============================================================================
# Alternative Regime Derivation
# =============================================================================

"""
    _derive_alternative_regime(spec::DSGESpec{T}, constraint::OccBinConstraint{T}) → DSGESpec{T}

Construct the alternative (binding) regime specification by replacing the constrained
variable's equation with `var[t] = bound`.

When the constraint binds:
1. The equation for the constrained variable is replaced with `var[t] = bound`
2. The corresponding residual function becomes `y_t[var_idx] - bound`
3. Forward-looking indices are updated (binding equation is never forward-looking)
"""
function _derive_alternative_regime(spec::DSGESpec{T}, constraint::OccBinConstraint{T}) where {T}
    var_idx = findfirst(==(constraint.variable), spec.endog)
    var_idx === nothing && throw(ArgumentError(
        "Variable :$(constraint.variable) not found in endogenous variables"))

    bound = constraint.bound

    # Build new equations vector: replace the constrained variable's equation
    new_equations = copy(spec.equations)
    new_equations[var_idx] = constraint.bind_expr

    # Build new residual functions: replace the constrained variable's residual
    new_residual_fns = copy(spec.residual_fns)
    new_residual_fns[var_idx] = (y_t, y_lag, y_lead, epsilon, theta) -> y_t[var_idx] - bound

    # Update forward indices: the binding equation is never forward-looking
    # Remove var_idx from forward_indices if present
    new_forward_indices = filter(!=(var_idx), spec.forward_indices)
    n_expect_new = length(new_forward_indices)

    DSGESpec{T}(spec.endog, spec.exog, spec.params, spec.param_values,
                new_equations, new_residual_fns,
                n_expect_new, new_forward_indices, spec.steady_state, spec.ss_fn)
end

# =============================================================================
# Regime Extraction (Linearized Matrices)
# =============================================================================

"""
    _extract_regime(spec::DSGESpec{T}) → OccBinRegime{T}

Extract the linearized coefficient matrices (A, B, C, D) from a DSGESpec
using numerical Jacobians evaluated at the steady state.

The linearized system is: `B * y_t = C * y_{t-1} + A * y_{t+1} + D * epsilon_t`

Where:
- `A` = Jacobian w.r.t. y_{t+1} (lead/expectation terms)
- `B` = Jacobian w.r.t. y_t (contemporaneous terms)
- `C` = Jacobian w.r.t. y_{t-1} (lagged terms)
- `D` = Jacobian w.r.t. epsilon_t (shock impact)

Note: Uses `_dsge_jacobian` and `_dsge_jacobian_shocks` from linearize.jl.
"""
function _extract_regime(spec::DSGESpec{T}) where {T}
    isempty(spec.steady_state) &&
        throw(ArgumentError("Must compute steady state first (call compute_steady_state)"))

    y_ss = spec.steady_state

    # Compute numerical Jacobians at steady state
    f_0 = _dsge_jacobian(spec, y_ss, :current)     # df/dy_t      → B
    f_1 = _dsge_jacobian(spec, y_ss, :lag)          # df/dy_{t-1}  → A (lag coefficients)
    f_lead = _dsge_jacobian(spec, y_ss, :lead)      # df/dy_{t+1}  → C (lead coefficients)
    f_eps = _dsge_jacobian_shocks(spec, y_ss)       # df/d_epsilon  → D

    # Return OccBinRegime with the convention:
    # A = f_1 (lag), B = f_0 (current), C = f_lead (lead), D = f_eps (shocks)
    OccBinRegime{T}(f_1, f_0, f_lead, f_eps)
end

"""
    _regime_constant(spec::DSGESpec{T}) → Vector{T}

Compute the constant (residual) vector at the steady state for a given regime.

For the reference regime, this is zero (by definition of steady state).
For the alternative (binding) regime, this is non-zero when the constraint
bound differs from the steady-state value.

The full linearized system is: A·ŷ_{t-1} + B·ŷ_t + C·ŷ_{t+1} + D·ε_t + d = 0
where d = f(y_ss, y_ss, y_ss, 0, θ) is the residual at steady state.
"""
function _regime_constant(spec::DSGESpec{T}) where {T}
    y_ss = spec.steady_state
    θ = spec.param_values
    ε_zero = zeros(T, spec.n_exog)
    n = spec.n_endog

    d = zeros(T, n)
    for i in 1:n
        d[i] = spec.residual_fns[i](y_ss, y_ss, y_ss, ε_zero, θ)
    end
    return d
end

# =============================================================================
# One-Constraint OccBin Solver (Guerrieri & Iacoviello 2015)
# =============================================================================

"""
    _map_regime(violvec::BitVector) → (regimes::Vector{Int}, starts::Vector{Int})

Identify contiguous blocks of binding (1) and non-binding (0) periods from a
violation indicator vector.

# Example
```
violvec = BitVector([0, 0, 1, 1, 1, 0, 0, 1, 0, 0])
regimes, starts = _map_regime(violvec)
# regimes = [0, 1, 0, 1, 0]
# starts  = [1, 3, 6, 8, 9]
```
"""
function _map_regime(violvec::BitVector)
    nperiods = length(violvec)
    nperiods == 0 && return (Int[], Int[])

    regimes = Int[violvec[1] ? 1 : 0]
    starts = Int[1]

    for t in 2:nperiods
        current = violvec[t] ? 1 : 0
        if current != regimes[end]
            push!(regimes, current)
            push!(starts, t)
        end
    end

    return (regimes, starts)
end

"""
    _backward_iteration(ref::OccBinRegime{T}, alt::OccBinRegime{T},
                        d_ref::Vector{T}, d_alt::Vector{T},
                        P::Matrix{T}, Q::Matrix{T},
                        violvec::BitVector, shock_path::Matrix{T})

Compute time-varying decision rules by backward iteration from the last
binding period, following Guerrieri & Iacoviello (2015).

The linearized system is: A·ŷ_{t-1} + B·ŷ_t + C·ŷ_{t+1} + D·ε_t + d = 0

where `d` is the constant (residual at steady state), zero for the reference
regime and potentially non-zero for the alternative regime when the constraint
bound differs from the steady-state value.

At the terminal boundary the unconstrained solution applies: ŷ_{t+1} = P·ŷ_t.
Substituting backward through the binding/non-binding regimes yields
time-varying policy matrices P_tv[:,:,t] and constant vectors D_tv[:,t].

# Returns
- `P_tv` — n × n × T_max array of time-varying transition matrices
- `D_tv` — n × T_max matrix of time-varying constants (includes shocks + regime constants)
- `E` — n × n_shocks impact matrix (unused, kept for API compatibility)
"""
function _backward_iteration(ref::OccBinRegime{T}, alt::OccBinRegime{T},
                             d_ref::Vector{T}, d_alt::Vector{T},
                             P::Matrix{T}, Q::Matrix{T},
                             violvec::BitVector, shock_path::Matrix{T}) where {T}
    n = size(P, 1)
    n_shocks = size(Q, 2)
    nperiods = length(violvec)

    # Find the last binding period
    T_max = findlast(violvec)

    # If no binding periods, return trivially: use the linear solution
    if T_max === nothing
        return (zeros(T, n, n, 0), zeros(T, n, 0), Q)
    end

    # Allocate time-varying decision rules
    P_tv = zeros(T, n, n, T_max)
    D_tv = zeros(T, n, T_max)

    # Start at T_max: the next period (T_max+1) uses the unconstrained P
    P_next = P  # ŷ_{T_max+1} = P · ŷ_{T_max}

    # Backward iteration from T_max down to 1
    for t in T_max:-1:1
        # Select regime for period t
        binding = violvec[t]
        rgm = binding ? alt : ref
        d = binding ? d_alt : d_ref

        # (B + C · P_next) · ŷ_t = -A · ŷ_{t-1} - D · ε_t - d - C · D_next
        invmat = robust_inv(rgm.B + rgm.C * P_next)
        P_tv[:, :, t] = -invmat * rgm.A

        # Constant: D_tv_t = -invmat * (D · ε_t + d + C · D_tv_{t+1})
        # At T_max, D_tv_{t+1} = 0 (unconstrained regime has no constant in deviations)
        if t == T_max
            D_tv[:, t] = -invmat * (rgm.D * shock_path[t, :] + d)
        else
            D_tv[:, t] = -invmat * (rgm.D * shock_path[t, :] + d + rgm.C * D_tv[:, t + 1])
        end

        P_next = P_tv[:, :, t]
    end

    return (P_tv, D_tv, Q)
end

"""
    _simulate_piecewise(P_tv, D_tv, P_lin::Matrix{T}, init::Vector{T},
                        nperiods::Int, T_max::Int) → Matrix{T}

Simulate the piecewise-linear path using time-varying decision rules for
periods 1..T_max and the unconditional linear rule for T_max+1..nperiods.

# Returns
- `path` — nperiods × n matrix of simulated deviations from steady state
"""
function _simulate_piecewise(P_tv::Array{T,3}, D_tv::Matrix{T},
                             P_lin::Matrix{T}, init::Vector{T},
                             nperiods::Int, T_max::Int) where {T}
    n = size(P_lin, 1)
    path = zeros(T, nperiods, n)

    # Period 1: use time-varying rule
    path[1, :] = P_tv[:, :, 1] * init + D_tv[:, 1]

    # Periods 2..T_max: time-varying rules
    for t in 2:T_max
        path[t, :] = P_tv[:, :, t] * path[t - 1, :] + D_tv[:, t]
    end

    # Periods T_max+1..nperiods: unconstrained linear rule (no shocks)
    for t in (T_max + 1):nperiods
        path[t, :] = P_lin * path[t - 1, :]
    end

    return path
end

"""
    _simulate_linear(P::Matrix{T}, Q::Matrix{T}, init::Vector{T},
                     shock_path::Matrix{T}, nperiods::Int) → Matrix{T}

Simulate the standard unconstrained linear path:
`y_t = P · y_{t-1} + Q · ε_t`

# Returns
- `path` — nperiods × n matrix of simulated deviations from steady state
"""
function _simulate_linear(P::Matrix{T}, Q::Matrix{T}, init::Vector{T},
                          shock_path::Matrix{T}, nperiods::Int) where {T}
    n = size(P, 1)
    path = zeros(T, nperiods, n)

    y_prev = init
    for t in 1:nperiods
        path[t, :] = P * y_prev + Q * shock_path[t, :]
        y_prev = path[t, :]
    end

    return path
end

"""
    _evaluate_constraint(path::Matrix{T}, P::Matrix{T}, Q::Matrix{T},
                         shock_path::Matrix{T}, spec::DSGESpec{T},
                         constraint::OccBinConstraint{T},
                         violvec_current::BitVector) → BitVector

Evaluate which periods should have the constraint binding, using the "notional"
(shadow) value approach.

For non-binding periods: the simulated value IS the notional value. Check if it
violates the constraint.

For binding periods: the constraint was imposed, so the simulated value satisfies
the bound by construction. Instead, compute the notional value — what the
constrained variable would be if we applied the unconstrained transition
`y_t = P · y_{t-1} + Q · ε_t` to the current state. If the notional value
does NOT violate the constraint, the constraint should not bind.

# Returns
- `violvec` — BitVector of length nperiods; `true` = constraint should bind
"""
function _evaluate_constraint(path::Matrix{T}, P::Matrix{T}, Q::Matrix{T},
                              shock_path::Matrix{T}, spec::DSGESpec{T},
                              constraint::OccBinConstraint{T},
                              violvec_current::BitVector) where {T}
    var_idx = findfirst(==(constraint.variable), spec.endog)
    nperiods = size(path, 1)
    n = size(P, 1)
    bound = constraint.bound
    ss_val = spec.steady_state[var_idx]

    violvec = falses(nperiods)
    for t in 1:nperiods
        if violvec_current[t]
            # Period was binding: compute the notional (shadow) value using
            # the unconstrained transition from the previous-period state
            y_prev = t == 1 ? zeros(T, n) : path[t - 1, :]
            notional_y = P * y_prev + Q * shock_path[t, :]
            notional_val = notional_y[var_idx] + ss_val
            if constraint.direction === :geq
                violvec[t] = notional_val < bound
            else  # :leq
                violvec[t] = notional_val > bound
            end
        else
            # Period was non-binding: use the actual simulated value
            level_val = path[t, var_idx] + ss_val
            if constraint.direction === :geq
                violvec[t] = level_val < bound
            else  # :leq
                violvec[t] = level_val > bound
            end
        end
    end

    return violvec
end

"""
    _guess_verify_one(ref::OccBinRegime{T}, alt::OccBinRegime{T},
                      P::Matrix{T}, Q::Matrix{T},
                      spec::DSGESpec{T}, constraint::OccBinConstraint{T},
                      shock_path::Matrix{T}, nperiods::Int;
                      maxiter::Int=100) → (path, regime_history, converged, iterations)

Run the guess-and-verify loop for a single occasionally binding constraint.

1. Initial guess: violvec = falses(nperiods) (no binding periods)
2. Backward iteration → time-varying decision rules
3. Simulate piecewise-linear path
4. Evaluate constraint → new violvec
5. Repeat until violvec converges or maxiter reached

# Returns
- `path` — nperiods × n simulated piecewise-linear path (deviations from SS)
- `regime_history` — nperiods × 1 matrix of regime indicators (0=slack, 1=binding)
- `converged` — whether the guess-and-verify loop converged
- `iterations` — number of iterations used
"""
function _guess_verify_one(ref::OccBinRegime{T}, alt::OccBinRegime{T},
                           d_ref::Vector{T}, d_alt::Vector{T},
                           P::Matrix{T}, Q::Matrix{T},
                           spec::DSGESpec{T}, constraint::OccBinConstraint{T},
                           shock_path::Matrix{T}, nperiods::Int;
                           maxiter::Int=100) where {T}
    n = size(P, 1)
    init = zeros(T, n)

    # Initial guess: no violations
    violvec = falses(nperiods)
    path = zeros(T, nperiods, n)
    converged = false
    iterations = 0

    for iter in 1:maxiter
        iterations = iter
        violvec_old = copy(violvec)

        # Backward iteration to get time-varying decision rules
        P_tv, D_tv, _ = _backward_iteration(ref, alt, d_ref, d_alt, P, Q, violvec, shock_path)

        T_max = findlast(violvec)

        if T_max === nothing
            # No binding periods: simulate linear path
            path = _simulate_linear(P, Q, init, shock_path, nperiods)
        else
            # Simulate piecewise-linear path
            path = _simulate_piecewise(P_tv, D_tv, P, init, nperiods, T_max)
        end

        # Evaluate constraint using notional (shadow) values for binding periods
        violvec = _evaluate_constraint(path, P, Q, shock_path, spec, constraint, violvec_old)

        # Check convergence: violvec hasn't changed
        if violvec == violvec_old
            converged = true
            break
        end
    end

    if !converged
        @warn "OccBin guess-and-verify did not converge after $maxiter iterations"
    end

    # Check if constraint is binding at the terminal period
    if violvec[end]
        @warn "OccBin: constraint is binding at the terminal period ($nperiods). " *
              "Consider increasing nperiods."
    end

    regime_history = reshape(Int.(violvec), nperiods, 1)

    return (path, regime_history, converged, iterations)
end

"""
    occbin_solve(spec::DSGESpec{T}, constraint::OccBinConstraint{T};
                 shock_path::Matrix{T}=zeros(T, 40, spec.n_exog),
                 nperiods::Int=size(shock_path, 1),
                 maxiter::Int=100) → OccBinSolution{T}

Solve a DSGE model with a single occasionally binding constraint using the
piecewise-linear algorithm of Guerrieri & Iacoviello (2015).

The algorithm:
1. Solve the unconstrained (reference) model via gensys → P, Q
2. Derive the alternative (binding) regime by replacing the constraint equation
3. Extract linearized coefficient matrices for both regimes
4. Run the guess-and-verify loop to find the piecewise-linear solution

# Arguments
- `spec` — DSGE model specification (must have steady state computed)
- `constraint` — the occasionally binding constraint

# Keyword Arguments
- `shock_path` — T_periods × n_exog matrix of shock realizations
- `nperiods` — number of periods to simulate (default: rows of shock_path)
- `maxiter` — maximum guess-and-verify iterations (default: 100)

# Returns
An `OccBinSolution{T}` with linear and piecewise-linear paths.
"""
function occbin_solve(spec::DSGESpec{T}, constraint::OccBinConstraint{T};
                      shock_path::Matrix{T}=zeros(T, 40, spec.n_exog),
                      nperiods::Int=size(shock_path, 1),
                      maxiter::Int=100) where {T<:AbstractFloat}
    # Ensure steady state is computed
    if isempty(spec.steady_state)
        spec = compute_steady_state(spec)
    end

    # Solve reference (unconstrained) model
    sol = solve(spec; method=:gensys)
    is_determined(sol) || @warn "Reference model solution is not determined (eu=$(sol.eu))"

    P = sol.G1       # n × n state transition
    Q = sol.impact    # n × n_shocks impact

    # Derive alternative regime (constraint binding)
    alt_spec = _derive_alternative_regime(spec, constraint)

    # Extract linearized coefficient matrices for both regimes
    ref_regime = _extract_regime(spec)
    alt_regime = _extract_regime(alt_spec)

    # Compute regime constants (d_ref ≈ 0, d_alt may be non-zero)
    d_ref = _regime_constant(spec)
    d_alt = _regime_constant(alt_spec)

    # Pad shock_path if needed
    if size(shock_path, 1) < nperiods
        padded = zeros(T, nperiods, size(shock_path, 2))
        padded[1:size(shock_path, 1), :] = shock_path
        shock_path = padded
    end

    # Simulate linear (unconstrained) path
    init = zeros(T, spec.n_endog)
    linear_path = _simulate_linear(P, Q, init, shock_path, nperiods)

    # Run guess-and-verify loop
    pw_path, regime_history, converged, iterations =
        _guess_verify_one(ref_regime, alt_regime, d_ref, d_alt, P, Q,
                          spec, constraint, shock_path, nperiods; maxiter=maxiter)

    OccBinSolution{T}(
        linear_path, pw_path, spec.steady_state,
        regime_history, converged, iterations,
        spec, spec.varnames
    )
end

"""
    occbin_solve(spec::DSGESpec{T}, constraint::OccBinConstraint{T},
                 alt_spec::DSGESpec{T}; kwargs...) → OccBinSolution{T}

Variant that accepts an explicit alternative regime specification instead of
deriving it automatically from the constraint.
"""
function occbin_solve(spec::DSGESpec{T}, constraint::OccBinConstraint{T},
                      alt_spec::DSGESpec{T};
                      shock_path::Matrix{T}=zeros(T, 40, spec.n_exog),
                      nperiods::Int=size(shock_path, 1),
                      maxiter::Int=100) where {T<:AbstractFloat}
    # Ensure steady states are computed
    if isempty(spec.steady_state)
        spec = compute_steady_state(spec)
    end
    if isempty(alt_spec.steady_state)
        alt_spec = compute_steady_state(alt_spec)
    end

    # Solve reference model
    sol = solve(spec; method=:gensys)
    is_determined(sol) || @warn "Reference model solution is not determined (eu=$(sol.eu))"

    P = sol.G1
    Q = sol.impact

    # Extract regimes and constants
    ref_regime = _extract_regime(spec)
    alt_regime = _extract_regime(alt_spec)
    d_ref = _regime_constant(spec)
    d_alt = _regime_constant(alt_spec)

    # Pad shock_path if needed
    if size(shock_path, 1) < nperiods
        padded = zeros(T, nperiods, size(shock_path, 2))
        padded[1:size(shock_path, 1), :] = shock_path
        shock_path = padded
    end

    # Linear path
    init = zeros(T, spec.n_endog)
    linear_path = _simulate_linear(P, Q, init, shock_path, nperiods)

    # Piecewise-linear path
    pw_path, regime_history, converged, iterations =
        _guess_verify_one(ref_regime, alt_regime, d_ref, d_alt, P, Q,
                          spec, constraint, shock_path, nperiods; maxiter=maxiter)

    OccBinSolution{T}(
        linear_path, pw_path, spec.steady_state,
        regime_history, converged, iterations,
        spec, spec.varnames
    )
end

"""
    occbin_irf(sol, constraints, H, shock_idx; kwargs...) → OccBinIRF{T}

Compute impulse responses under occasionally binding constraints, comparing
the unconstrained linear path with the piecewise-linear OccBin path.

(Full implementation in a subsequent task.)
"""
function occbin_irf end
