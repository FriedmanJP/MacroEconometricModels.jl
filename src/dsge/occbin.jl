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

# =============================================================================
# Solver and IRF stubs (implemented in later tasks)
# =============================================================================

"""
    occbin_solve(sol, constraints, T_periods, shock_path; kwargs...) → OccBinSolution{T}

Solve a DSGE model with occasionally binding constraints using the
guess-and-verify piecewise-linear algorithm of Guerrieri & Iacoviello (2015).

(Full implementation in a subsequent task.)
"""
function occbin_solve end

"""
    occbin_irf(sol, constraints, H, shock_idx; kwargs...) → OccBinIRF{T}

Compute impulse responses under occasionally binding constraints, comparing
the unconstrained linear path with the piecewise-linear OccBin path.

(Full implementation in a subsequent task.)
"""
function occbin_irf end
