# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

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
    parse_constraint(expr::Expr, spec::ModelSpec{T}) → OccBinConstraint{T}

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
function parse_constraint(expr::Expr, spec::ModelSpec{T}) where {T<:AbstractFloat}
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

Evaluate the RHS bound of a constraint to a numeric value of type `T`.

Uses a small recursive numeric evaluator (NOT `eval`) that supports only numeric
literals, unary `+`/`-`, and binary `+ - * / ^`. Runtime `eval` is avoided because
it runs in module global scope (world-age hazard, silent symbol resolution against
package globals, and — worst — it would happily evaluate arbitrary calls such as
`sin(1)`). Each numeric literal leaf is converted to `T` immediately so all
arithmetic runs in `T`; this makes `:(2^-3)` evaluate as `2.0^-3.0 == 0.125`
instead of throwing the `DomainError` that runtime `^(Int, negative Int)` raises.
"""
function _eval_bound(rhs, ::Type{T}) where {T}
    if rhs isa Number
        return T(rhs)                       # leaf → float immediately
    elseif rhs isa Expr && rhs.head === :call && length(rhs.args) >= 2
        op = rhs.args[1]
        vals = [_eval_bound(a, T) for a in rhs.args[2:end]]   # recurse; Symbol/ref leaves throw
        if op === :+
            length(vals) == 1 && return +(vals[1])
            length(vals) == 2 && return vals[1] + vals[2]
        elseif op === :-
            length(vals) == 1 && return -(vals[1])
            length(vals) == 2 && return vals[1] - vals[2]
        elseif op === :* && length(vals) == 2
            return vals[1] * vals[2]
        elseif op === :/ && length(vals) == 2
            return vals[1] / vals[2]
        elseif op === :^ && length(vals) == 2
            return vals[1] ^ vals[2]
        end
        throw(ArgumentError("Unsupported operator/arity in constraint bound: $rhs " *
            "(only unary ± and binary + - * / ^ on numeric literals are allowed)"))
    else
        throw(ArgumentError("Constraint bound must be a numeric literal expression, " *
            "got: $rhs ($(typeof(rhs)))"))
    end
end

# =============================================================================
# Alternative Regime Derivation
# =============================================================================

"""
    _defining_equation_index(spec, var) → Int

Index of the equation that defines endogenous `var`: an attached `:binding`
regime from `@dsge constraint:`, otherwise the unique `defines === var` equation.

# Throws
- `ArgumentError` if zero or several equations define `var` and no `:binding`
  regime disambiguates.
"""
function _defining_equation_index(spec::ModelSpec, var::Symbol)
    bind_idxs = findall(spec.equations) do eq
        haskey(eq.regimes, :binding) &&
            (eq.defines === var || eq.regimes[:binding].defines === var)
    end
    if length(bind_idxs) == 1
        return bind_idxs[1]
    elseif length(bind_idxs) > 1
        names = [spec.equations[i].name for i in bind_idxs]
        throw(ArgumentError(
            "OccBin: multiple :binding regimes for :$var on $(join(names, ", "))."))
    end
    idxs = findall(eq -> eq.defines === var, spec.equations)
    if length(idxs) == 1
        return idxs[1]
    elseif isempty(idxs)
        throw(ArgumentError(
            "OccBin: no equation defines :$var. Name the defining equation " *
            "(e.g. `taylor: $var[t] = ...`) or write `constraint: <eqname> = $var[t] >= bound`."))
    else
        names = [spec.equations[i].name for i in idxs]
        throw(ArgumentError(
            "OccBin: $(length(idxs)) equations define :$var ($(join(names, ", "))). " *
            "Disambiguate with `constraint: <eqname> = $var[t] >= bound`."))
    end
end

"""
    _derive_alternative_regime(spec, constraint) → ModelSpec{T}

Construct the alternative (binding) regime by replacing the constrained
variable's defining equation with `var[t] = bound` (residual `y_t[var_idx] - bound`)
and dropping it from the forward-looking set. The defining equation is the one
with `defines === constraint.variable`, or the equation that already carries a
`:binding` regime from `@dsge constraint:`.
"""
function _derive_alternative_regime(spec::ModelSpec{T}, constraint::OccBinConstraint{T}) where {T}
    var_idx = findfirst(==(constraint.variable), spec.endog)
    var_idx === nothing && throw(ArgumentError(
        "Variable :$(constraint.variable) not found in endogenous variables"))

    bound = constraint.bound
    eq_idx = _defining_equation_index(spec, constraint.variable)

    old_eq = spec.equations[eq_idx]
    if haskey(old_eq.regimes, :binding)
        bind_eq = old_eq.regimes[:binding]
        bind_fn = bind_eq.residual
        bind_expr = bind_eq.expr
    else
        bind_fn = (y_t, y_lag, y_lead, epsilon, theta) -> y_t[var_idx] - bound
        bind_expr = constraint.bind_expr
    end
    new_equations = copy(spec.equations)
    new_equations[eq_idx] = NamedEquation(old_eq.name, old_eq.defines,
                                          bind_expr, bind_fn;
                                          timing=TimingInfo(),
                                          regimes=old_eq.regimes)

    new_residual_fns = copy(spec.residual_fns)
    new_residual_fns[eq_idx] = bind_fn

    new_forward_indices = filter(!=(eq_idx), spec.forward_indices)
    n_expect_new = length(new_forward_indices)

    _copy_model_spec(spec; equations=new_equations, residual_fns=new_residual_fns,
                     n_expect=n_expect_new, forward_indices=new_forward_indices)
end

# =============================================================================
# Regime Extraction (Linearized Matrices)
# =============================================================================

"""
    _extract_regime(spec::ModelSpec{T}) → OccBinRegime{T}

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
function _extract_regime(spec::ModelSpec{T}) where {T}
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
    _regime_constant(spec::ModelSpec{T}) → Vector{T}

Compute the constant (residual) vector at the steady state for a given regime.

For the reference regime, this is zero (by definition of steady state).
For the alternative (binding) regime, this is non-zero when the constraint
bound differs from the steady-state value.

The full linearized system is: A·ŷ_{t-1} + B·ŷ_t + C·ŷ_{t+1} + D·ε_t + d = 0
where d = f(y_ss, y_ss, y_ss, 0, θ) is the residual at steady state.
"""
function _regime_constant(spec::ModelSpec{T}) where {T}
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

        if maximum(abs.(P_tv[:, :, t])) > T(1e10)
            throw(ConvergenceError("OccBin backward iteration diverged (max|P_tv| > 1e10 at period $t). " *
                  "The alternative (binding) regime is likely indeterminate. " *
                  "This commonly occurs in simple NK models at the ZLB. " *
                  "Consider adding model frictions (habits, investment adj. costs) or reducing shock magnitude."))
        end

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
                         shock_path::Matrix{T}, spec::ModelSpec{T},
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
                              shock_path::Matrix{T}, spec::ModelSpec{T},
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
                      spec::ModelSpec{T}, constraint::OccBinConstraint{T},
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
                           spec::ModelSpec{T}, constraint::OccBinConstraint{T},
                           shock_path::Matrix{T}, nperiods::Int;
                           maxiter::Int=100) where {T}
    n = size(P, 1)
    init = zeros(T, n)

    # Initial guess: no violations
    violvec = falses(nperiods)
    path = zeros(T, nperiods, n)
    converged = false
    iterations = 0
    seen = BitVector[]                    # regime patterns already visited (cycle detection)

    for iter in 1:maxiter
        iterations = iter
        violvec_old = copy(violvec)
        push!(seen, BitVector(violvec_old))

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
        # Oscillation: the new guess repeats an earlier (non-immediate) regime pattern, so
        # the fixed-point iteration is cycling and will never settle (S-21 / #224).
        if any(v -> v == violvec, seen)
            @warn "OccBin guess-and-verify is oscillating between regime patterns; " *
                  "stopping without convergence (try curb_retrench or a longer horizon)"
            converged = false
            break
        end
    end

    if !converged
        @warn "OccBin guess-and-verify did not converge after $maxiter iterations"
    end

    regime_history = reshape(Int.(violvec), nperiods, 1)

    return (path, regime_history, converged, iterations)
end

# =============================================================================
# HA / kind guards (#654)
# =============================================================================

"""Nominal policy-rate names OccBin may constrain on an HA aggregate block."""
const _OCCBIN_HA_NOMINAL = (:i, :R)

function _occbin_ha_error(msg::AbstractString)
    throw(ArgumentError(string(msg, " See #654.")))
end

"""Variable name of a constraint `Expr` or `OccBinConstraint`, if parseable."""
function _occbin_constraint_var(c, ::Type{T}) where {T}
    c isa OccBinConstraint && return c.variable
    c isa Expr && return _parse_constraint_expr(c, T)[1]
    return nothing
end

"""
    _occbin_check_kind!(spec, var)

Reject kinds OccBin cannot treat, and shipped real-HANK / `n_endog==0`.

`HouseholdSystem` remaps `solve(; method=:gensys)` to `:ssj` and returns
`HADSGESolution`, which has no `.G1` (it lives on `linear_solution`). Dummy CD
residuals on shipped examples are not an OccBin system. Continuous-time
state constraints stay in the HJB.
"""
function _occbin_check_kind!(spec::ModelSpec, var::Union{Symbol,Nothing})
    if has_kind(spec, ContinuousHouseholdSystem)
        _occbin_ha_error("OccBin does not apply to continuous-time households; " *
            "the state constraint lives in the HJB, not OccBin.")
    end
    if has_kind(spec, DCEGMSystem) || has_kind(spec, LifeCycleSystem)
        _occbin_ha_error("OccBin is not implemented for this agent kind; it " *
            "applies to HA aggregate equations that include a nominal policy rate.")
    end
    has_kind(spec, HouseholdSystem) || return nothing
    ok = spec.n_endog > 0 && var !== nothing &&
         var in _OCCBIN_HA_NOMINAL && var in spec.endog
    ok || _occbin_ha_error(
        "OccBin on heterogeneous-agent models requires a nominal policy rate " *
        "(:i or :R) in the aggregate equations (n_endog > 0). Shipped real-HANK " *
        "examples have real r only, and n_endog == 0 is partial GE.")
    return nothing
end

_occbin_check_kind!(spec::ModelSpec, c, ::Type{T}) where {T} =
    _occbin_check_kind!(spec, _occbin_constraint_var(c, T))

"""Strip `HouseholdSystem` so gensys / SS see only the named aggregate residuals."""
function _occbin_aggregate_spec(spec::ModelSpec{T}) where {T<:AbstractFloat}
    has_kind(spec, HouseholdSystem) || return spec
    return _copy_model_spec(spec; agents=NamedTuple())
end

function _occbin_prepare_spec(spec::ModelSpec{T}) where {T<:AbstractFloat}
    work = _occbin_aggregate_spec(spec)
    isempty(work.steady_state) || return work
    return compute_steady_state(work)
end

"""
G1 / impact of the OccBin reference. Never reads `HADSGESolution.G1` (#654).

For HA, the reference is gensys on the stripped aggregate block (Taylor / NKPC),
not the SSJ reduced `linear_solution` in synthetic coordinates.
"""
function _occbin_transition(sol)
    if sol isa HADSGESolution
        _occbin_ha_error(
            "OccBin on HA must use the aggregate (Taylor / NKPC) block, not " *
            "HADSGESolution.G1. Shipped real-HANK has real r only.")
    end
    return sol.G1, sol.impact
end

function _occbin_reference_solution(spec::ModelSpec{T}) where {T<:AbstractFloat}
    work = _occbin_prepare_spec(spec)
    sol = solve(work; method=:gensys)
    is_determined(sol) || @warn "Reference model solution is not determined (eu=$(sol.eu))"
    return work, sol
end

"""
    occbin_solve(spec::ModelSpec{T}, constraint::OccBinConstraint{T};
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

`constraint` may also be a comparison `Expr` such as `:(i[t] >= 0)`; OccBin
looks up the defining equation by `defines` (or a `:binding` regime from
`@dsge constraint:`).

Household-agent specs (`HouseholdSystem`) run OccBin on the named aggregate
block (Taylor / NKPC) after stripping the household — never via
`HADSGESolution.G1`. Shipped real-HANK (`n_endog==0` or real `r` only) and
continuous-time HJB constraints throw `ArgumentError` citing #654.
"""
function occbin_solve(spec::ModelSpec{T}, constraint::OccBinConstraint{T};
                      shock_path::Matrix{T}=zeros(T, 40, spec.n_exog),
                      nperiods::Int=size(shock_path, 1),
                      maxiter::Int=100) where {T<:AbstractFloat}
    _occbin_check_kind!(spec, constraint.variable)
    spec, sol = _occbin_reference_solution(spec)
    P, Q = _occbin_transition(sol)

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
    orig_nperiods = nperiods
    pw_path, regime_history, converged, iterations =
        _guess_verify_one(ref_regime, alt_regime, d_ref, d_alt, P, Q,
                          spec, constraint, shock_path, nperiods; maxiter=maxiter)

    # Auto-extend if converged but constraint binds at terminal
    if converged && any(regime_history[end, :] .== 1)
        orig_lp, orig_pw, orig_rh = linear_path, pw_path, regime_history
        orig_conv, orig_iter = converged, iterations
        max_nperiods = 2000
        extended = false
        while any(regime_history[end, :] .== 1) && nperiods < max_nperiods
            nperiods = min(nperiods * 2, max_nperiods)
            new_shock_path = zeros(T, nperiods, size(shock_path, 2))
            new_shock_path[1:size(shock_path, 1), :] .= shock_path
            shock_path = new_shock_path
            linear_path = _simulate_linear(P, Q, init, shock_path, nperiods)
            pw_path, regime_history, converged, iterations =
                _guess_verify_one(ref_regime, alt_regime, d_ref, d_alt, P, Q,
                                  spec, constraint, shock_path, nperiods; maxiter=maxiter)
            if !converged
                break
            end
            extended = true
        end
        if extended && converged && !any(regime_history[end, :] .== 1)
            # Successfully extended: truncate to original horizon
            linear_path = linear_path[1:orig_nperiods, :]
            pw_path = pw_path[1:orig_nperiods, :]
            regime_history = regime_history[1:orig_nperiods, :]
        else
            # Fall back to original result
            linear_path, pw_path, regime_history = orig_lp, orig_pw, orig_rh
            converged, iterations = orig_conv, orig_iter
            nperiods = orig_nperiods
            if any(regime_history[end, :] .== 1)
                @warn "OccBin: constraint binding at terminal period ($orig_nperiods). " *
                      "Consider increasing nperiods."
            end
        end
    end

    # Filter output to original variables if augmented
    if spec.augmented
        orig_idx = _original_var_indices(spec)
        vnames = [string(s) for s in spec.original_endog]
        OccBinSolution{T}(
            linear_path[:, orig_idx], pw_path[:, orig_idx], spec.steady_state[orig_idx],
            regime_history, converged, iterations,
            spec, vnames, [constraint]
        )
    else
        OccBinSolution{T}(
            linear_path, pw_path, spec.steady_state,
            regime_history, converged, iterations,
            spec, spec.varnames, [constraint]
        )
    end
end

function occbin_solve(spec::ModelSpec{T}, expr::Expr; kwargs...) where {T<:AbstractFloat}
    _occbin_check_kind!(spec, expr, T)
    occbin_solve(spec, parse_constraint(expr, spec); kwargs...)
end

function occbin_solve(spec::ModelSpec{T}, expr::Expr, alt_spec::ModelSpec{T}; kwargs...) where {T<:AbstractFloat}
    _occbin_check_kind!(spec, expr, T)
    occbin_solve(spec, parse_constraint(expr, spec), alt_spec; kwargs...)
end

"""
    occbin_solve(spec::ModelSpec{T}, constraint::OccBinConstraint{T},
                 alt_spec::ModelSpec{T}; kwargs...) → OccBinSolution{T}

Variant that accepts an explicit alternative regime specification instead of
deriving it automatically from the constraint.
"""
function occbin_solve(spec::ModelSpec{T}, constraint::OccBinConstraint{T},
                      alt_spec::ModelSpec{T};
                      shock_path::Matrix{T}=zeros(T, 40, spec.n_exog),
                      nperiods::Int=size(shock_path, 1),
                      maxiter::Int=100) where {T<:AbstractFloat}
    _occbin_check_kind!(spec, constraint.variable)
    spec, sol = _occbin_reference_solution(spec)
    P, Q = _occbin_transition(sol)
    alt_spec = _occbin_prepare_spec(alt_spec)

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
    orig_nperiods = nperiods
    pw_path, regime_history, converged, iterations =
        _guess_verify_one(ref_regime, alt_regime, d_ref, d_alt, P, Q,
                          spec, constraint, shock_path, nperiods; maxiter=maxiter)

    # Auto-extend if converged but constraint binds at terminal
    if converged && any(regime_history[end, :] .== 1)
        orig_lp, orig_pw, orig_rh = linear_path, pw_path, regime_history
        orig_conv, orig_iter = converged, iterations
        max_nperiods = 2000
        extended = false
        while any(regime_history[end, :] .== 1) && nperiods < max_nperiods
            nperiods = min(nperiods * 2, max_nperiods)
            new_shock_path = zeros(T, nperiods, size(shock_path, 2))
            new_shock_path[1:size(shock_path, 1), :] .= shock_path
            shock_path = new_shock_path
            linear_path = _simulate_linear(P, Q, init, shock_path, nperiods)
            pw_path, regime_history, converged, iterations =
                _guess_verify_one(ref_regime, alt_regime, d_ref, d_alt, P, Q,
                                  spec, constraint, shock_path, nperiods; maxiter=maxiter)
            if !converged
                break
            end
            extended = true
        end
        if extended && converged && !any(regime_history[end, :] .== 1)
            linear_path = linear_path[1:orig_nperiods, :]
            pw_path = pw_path[1:orig_nperiods, :]
            regime_history = regime_history[1:orig_nperiods, :]
        else
            linear_path, pw_path, regime_history = orig_lp, orig_pw, orig_rh
            converged, iterations = orig_conv, orig_iter
            nperiods = orig_nperiods
            if any(regime_history[end, :] .== 1)
                @warn "OccBin: constraint binding at terminal period ($orig_nperiods). " *
                      "Consider increasing nperiods."
            end
        end
    end

    # Filter output to original variables if augmented
    if spec.augmented
        orig_idx = _original_var_indices(spec)
        vnames = [string(s) for s in spec.original_endog]
        OccBinSolution{T}(
            linear_path[:, orig_idx], pw_path[:, orig_idx], spec.steady_state[orig_idx],
            regime_history, converged, iterations,
            spec, vnames, [constraint]
        )
    else
        OccBinSolution{T}(
            linear_path, pw_path, spec.steady_state,
            regime_history, converged, iterations,
            spec, spec.varnames, [constraint]
        )
    end
end

# =============================================================================
# Two-Constraint OccBin Solver (Guerrieri & Iacoviello 2015)
# =============================================================================

"""
    _find_last_binding_two(violvec::BitMatrix) → Int

Find the last period where either constraint binds in a two-constraint setting.

# Returns
- The index of the last period where `violvec[t, 1]` or `violvec[t, 2]` is true,
  or 0 if no constraint ever binds.
"""
function _find_last_binding_two(violvec::BitMatrix)
    nperiods = size(violvec, 1)
    for t in nperiods:-1:1
        if violvec[t, 1] || violvec[t, 2]
            return t
        end
    end
    return 0
end

"""
    _backward_iteration_two(ref, alt1, alt2, alt12,
                             d_ref, d_alt1, d_alt2, d_alt12,
                             P, Q, violvec::BitMatrix, shock_path)

Compute time-varying decision rules by backward iteration for the two-constraint
case, following Guerrieri & Iacoviello (2015).

For each period t, the regime is selected based on (violvec[t,1], violvec[t,2]):
- (false, false) → reference regime (ref, d_ref)
- (true,  false) → alternative 1 (alt1, d_alt1)  — constraint 1 binding
- (false, true)  → alternative 2 (alt2, d_alt2)  — constraint 2 binding
- (true,  true)  → alternative 12 (alt12, d_alt12) — both constraints binding

# Returns
- `P_tv` — n × n × T_max array of time-varying transition matrices
- `D_tv` — n × T_max matrix of time-varying constants
- `E` — n × n_shocks impact matrix (for API compatibility)
"""
function _backward_iteration_two(ref::OccBinRegime{T}, alt1::OccBinRegime{T},
                                  alt2::OccBinRegime{T}, alt12::OccBinRegime{T},
                                  d_ref::Vector{T}, d_alt1::Vector{T},
                                  d_alt2::Vector{T}, d_alt12::Vector{T},
                                  P::Matrix{T}, Q::Matrix{T},
                                  violvec::BitMatrix, shock_path::Matrix{T}) where {T}
    n = size(P, 1)
    n_shocks = size(Q, 2)

    # Find the last binding period (either constraint)
    T_max = _find_last_binding_two(violvec)

    # If no binding periods, return trivially
    if T_max == 0
        return (zeros(T, n, n, 0), zeros(T, n, 0), Q)
    end

    # Allocate time-varying decision rules
    P_tv = zeros(T, n, n, T_max)
    D_tv = zeros(T, n, T_max)

    # Start at T_max: the next period uses the unconstrained P
    P_next = P

    # Backward iteration from T_max down to 1
    for t in T_max:-1:1
        # Select regime for period t based on (c1_binding, c2_binding)
        c1 = violvec[t, 1]
        c2 = violvec[t, 2]

        if c1 && c2
            rgm = alt12
            d = d_alt12
        elseif c1
            rgm = alt1
            d = d_alt1
        elseif c2
            rgm = alt2
            d = d_alt2
        else
            rgm = ref
            d = d_ref
        end

        # (B + C · P_next) · ŷ_t = -A · ŷ_{t-1} - D · ε_t - d - C · D_next
        invmat = robust_inv(rgm.B + rgm.C * P_next)
        P_tv[:, :, t] = -invmat * rgm.A

        if maximum(abs.(P_tv[:, :, t])) > T(1e10)
            throw(ConvergenceError("OccBin backward iteration diverged (max|P_tv| > 1e10 at period $t). " *
                  "The alternative (binding) regime is likely indeterminate. " *
                  "Consider adding model frictions or reducing shock magnitude."))
        end

        # Constant: D_tv_t = -invmat * (D · ε_t + d + C · D_tv_{t+1})
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
    _guess_verify_two(ref, alt1, alt2, alt12,
                       d_ref, d_alt1, d_alt2, d_alt12,
                       P, Q, spec, c1, c2, shock_path, nperiods;
                       maxiter=100, curb_retrench=false)

Run the guess-and-verify loop for two occasionally binding constraints.

1. Initial guess: violvec = falses(nperiods, 2)
2. Backward iteration (4 regimes) → time-varying decision rules
3. Simulate piecewise-linear path
4. Evaluate both constraints independently → new violvec
5. Repeat until violvec converges or maxiter reached

When `curb_retrench=true`, each constraint can only relax (switch from binding
to non-binding) by one period per iteration, which helps prevent oscillation.

# Returns
- `path` — nperiods × n simulated piecewise-linear path (deviations from SS)
- `regime_history` — nperiods × 2 matrix of regime indicators (0=slack, 1=binding)
- `converged` — whether the guess-and-verify loop converged
- `iterations` — number of iterations used
"""
function _guess_verify_two(ref::OccBinRegime{T}, alt1::OccBinRegime{T},
                            alt2::OccBinRegime{T}, alt12::OccBinRegime{T},
                            d_ref::Vector{T}, d_alt1::Vector{T},
                            d_alt2::Vector{T}, d_alt12::Vector{T},
                            P::Matrix{T}, Q::Matrix{T},
                            spec::ModelSpec{T}, c1::OccBinConstraint{T},
                            c2::OccBinConstraint{T},
                            shock_path::Matrix{T}, nperiods::Int;
                            maxiter::Int=100, curb_retrench::Bool=false) where {T}
    n = size(P, 1)
    init = zeros(T, n)

    # Initial guess: no violations for either constraint
    violvec = falses(nperiods, 2)
    path = zeros(T, nperiods, n)
    converged = false
    iterations = 0

    for iter in 1:maxiter
        iterations = iter
        violvec_old = copy(violvec)

        # Backward iteration with 4 regimes
        P_tv, D_tv, _ = _backward_iteration_two(ref, alt1, alt2, alt12,
                                                  d_ref, d_alt1, d_alt2, d_alt12,
                                                  P, Q, violvec, shock_path)

        T_max = _find_last_binding_two(violvec)

        if T_max == 0
            # No binding periods: simulate linear path
            path = _simulate_linear(P, Q, init, shock_path, nperiods)
        else
            # Simulate piecewise-linear path
            path = _simulate_piecewise(P_tv, D_tv, P, init, nperiods, T_max)
        end

        # Evaluate each constraint independently
        violvec_c1_old = BitVector(violvec_old[:, 1])
        violvec_c2_old = BitVector(violvec_old[:, 2])

        violvec_c1_new = _evaluate_constraint(path, P, Q, shock_path, spec, c1, violvec_c1_old)
        violvec_c2_new = _evaluate_constraint(path, P, Q, shock_path, spec, c2, violvec_c2_old)

        # Apply curb_retrench: limit relaxation to one period per constraint per iteration
        if curb_retrench
            for j in 1:2
                old_col = j == 1 ? violvec_c1_old : violvec_c2_old
                new_col = j == 1 ? violvec_c1_new : violvec_c2_new
                relaxed = false
                for t in 1:nperiods
                    if old_col[t] && !new_col[t]
                        if relaxed
                            # Already relaxed one period for this constraint; keep binding
                            new_col[t] = true
                        else
                            relaxed = true
                        end
                    end
                end
                if j == 1
                    violvec_c1_new = new_col
                else
                    violvec_c2_new = new_col
                end
            end
        end

        # Assemble new violvec
        violvec_new = falses(nperiods, 2)
        violvec_new[:, 1] = violvec_c1_new
        violvec_new[:, 2] = violvec_c2_new

        # Check convergence: violvec hasn't changed
        if violvec_new == violvec_old
            violvec = violvec_new
            converged = true
            break
        end

        violvec = violvec_new
    end

    if !converged
        @warn "OccBin two-constraint guess-and-verify did not converge after $maxiter iterations"
    end

    regime_history = Matrix{Int}(violvec)

    return (path, regime_history, converged, iterations)
end

"""
    occbin_solve(spec::ModelSpec{T}, c1::OccBinConstraint{T}, c2::OccBinConstraint{T};
                 shock_path, nperiods, maxiter, curb_retrench) → OccBinSolution{T}

Solve a DSGE model with two occasionally binding constraints using the
piecewise-linear algorithm of Guerrieri & Iacoviello (2015).

The algorithm generalizes the one-constraint solver to 4 regimes:
1. Neither constraint binds (reference)
2. Only constraint 1 binds (alt1)
3. Only constraint 2 binds (alt2)
4. Both constraints bind (alt12)

# Arguments
- `spec` — DSGE model specification (must have steady state computed)
- `c1` — first occasionally binding constraint
- `c2` — second occasionally binding constraint

# Keyword Arguments
- `shock_path` — T_periods × n_exog matrix of shock realizations
- `nperiods` — number of periods to simulate (default: rows of shock_path)
- `maxiter` — maximum guess-and-verify iterations (default: 100)
- `curb_retrench` — limit constraint relaxation to one period per iteration (default: false)

# Returns
An `OccBinSolution{T}` with linear and piecewise-linear paths and a
nperiods × 2 regime history matrix.
"""
function occbin_solve(spec::ModelSpec{T}, e1::Expr, e2::Expr; kwargs...) where {T<:AbstractFloat}
    _occbin_check_kind!(spec, e1, T)
    _occbin_check_kind!(spec, e2, T)
    occbin_solve(spec, parse_constraint(e1, spec), parse_constraint(e2, spec); kwargs...)
end

function occbin_solve(spec::ModelSpec{T}, c1::OccBinConstraint{T}, c2::OccBinConstraint{T};
                      shock_path::Matrix{T}=zeros(T, 40, spec.n_exog),
                      nperiods::Int=size(shock_path, 1),
                      maxiter::Int=100,
                      curb_retrench::Bool=false) where {T<:AbstractFloat}
    _occbin_check_kind!(spec, c1.variable)
    _occbin_check_kind!(spec, c2.variable)
    spec, sol = _occbin_reference_solution(spec)
    P, Q = _occbin_transition(sol)

    # Collision: two constraints that replace the same named defining equation cannot be
    # stacked sequentially. Require the explicit Dict overload.
    eq1 = _defining_equation_index(spec, c1.variable)
    eq2 = _defining_equation_index(spec, c2.variable)
    eq1 == eq2 && throw(ArgumentError(
        "OccBin: constraints on :$(c1.variable) and :$(c2.variable) replace the same defining " *
        "equation ($(spec.equations[eq1].name)). Pass explicit alternative regimes via the " *
        "Dict overload: occbin_solve(spec, c1, c2, Dict((1,0)=>alt1, (0,1)=>alt2, (1,1)=>alt12); ...)."))

    # Derive 3 alternative regime specifications
    alt1_spec = _derive_alternative_regime(spec, c1)      # only c1 binding
    alt2_spec = _derive_alternative_regime(spec, c2)      # only c2 binding
    alt12_spec = _derive_alternative_regime(alt1_spec, c2) # both binding (c1 first, then c2)

    # Extract linearized coefficient matrices for all 4 regimes
    ref_regime = _extract_regime(spec)
    alt1_regime = _extract_regime(alt1_spec)
    alt2_regime = _extract_regime(alt2_spec)
    alt12_regime = _extract_regime(alt12_spec)

    # Compute regime constants
    d_ref = _regime_constant(spec)
    d_alt1 = _regime_constant(alt1_spec)
    d_alt2 = _regime_constant(alt2_spec)
    d_alt12 = _regime_constant(alt12_spec)

    # Pad shock_path if needed
    if size(shock_path, 1) < nperiods
        padded = zeros(T, nperiods, size(shock_path, 2))
        padded[1:size(shock_path, 1), :] = shock_path
        shock_path = padded
    end

    # Simulate linear (unconstrained) path
    init = zeros(T, spec.n_endog)
    linear_path = _simulate_linear(P, Q, init, shock_path, nperiods)

    # Run guess-and-verify loop with 4 regimes
    orig_nperiods = nperiods
    pw_path, regime_history, converged, iterations =
        _guess_verify_two(ref_regime, alt1_regime, alt2_regime, alt12_regime,
                          d_ref, d_alt1, d_alt2, d_alt12,
                          P, Q, spec, c1, c2, shock_path, nperiods;
                          maxiter=maxiter, curb_retrench=curb_retrench)

    # Auto-extend if converged but constraint binds at terminal
    _terminal_binding_two(rh) = rh[end, 1] == 1 || rh[end, 2] == 1
    if converged && _terminal_binding_two(regime_history)
        orig_lp, orig_pw, orig_rh = linear_path, pw_path, regime_history
        orig_conv, orig_iter = converged, iterations
        max_nperiods = 2000
        extended = false
        while _terminal_binding_two(regime_history) && nperiods < max_nperiods
            nperiods = min(nperiods * 2, max_nperiods)
            new_shock_path = zeros(T, nperiods, size(shock_path, 2))
            new_shock_path[1:size(shock_path, 1), :] .= shock_path
            shock_path = new_shock_path
            linear_path = _simulate_linear(P, Q, init, shock_path, nperiods)
            pw_path, regime_history, converged, iterations =
                _guess_verify_two(ref_regime, alt1_regime, alt2_regime, alt12_regime,
                                  d_ref, d_alt1, d_alt2, d_alt12,
                                  P, Q, spec, c1, c2, shock_path, nperiods;
                                  maxiter=maxiter, curb_retrench=curb_retrench)
            if !converged
                break
            end
            extended = true
        end
        if extended && converged && !_terminal_binding_two(regime_history)
            linear_path = linear_path[1:orig_nperiods, :]
            pw_path = pw_path[1:orig_nperiods, :]
            regime_history = regime_history[1:orig_nperiods, :]
        else
            linear_path, pw_path, regime_history = orig_lp, orig_pw, orig_rh
            converged, iterations = orig_conv, orig_iter
            nperiods = orig_nperiods
            if _terminal_binding_two(regime_history)
                @warn "OccBin: constraint binding at terminal period ($orig_nperiods). " *
                      "Consider increasing nperiods."
            end
        end
    end

    # Filter output to original variables if augmented
    if spec.augmented
        orig_idx = _original_var_indices(spec)
        vnames = [string(s) for s in spec.original_endog]
        OccBinSolution{T}(
            linear_path[:, orig_idx], pw_path[:, orig_idx], spec.steady_state[orig_idx],
            regime_history, converged, iterations,
            spec, vnames, [c1, c2]
        )
    else
        OccBinSolution{T}(
            linear_path, pw_path, spec.steady_state,
            regime_history, converged, iterations,
            spec, spec.varnames, [c1, c2]
        )
    end
end

"""
    occbin_solve(spec::ModelSpec{T}, c1::OccBinConstraint{T}, c2::OccBinConstraint{T},
                 alt_specs::Dict; kwargs...) → OccBinSolution{T}

Variant that accepts explicit alternative regime specifications as a Dict mapping
regime indicators to DSGESpec:
- `(1,0)` → alt1_spec (constraint 1 binding only)
- `(0,1)` → alt2_spec (constraint 2 binding only)
- `(1,1)` → alt12_spec (both constraints binding)
"""
function occbin_solve(spec::ModelSpec{T}, c1::OccBinConstraint{T}, c2::OccBinConstraint{T},
                      alt_specs::Dict;
                      shock_path::Matrix{T}=zeros(T, 40, spec.n_exog),
                      nperiods::Int=size(shock_path, 1),
                      maxiter::Int=100,
                      curb_retrench::Bool=false) where {T<:AbstractFloat}
    _occbin_check_kind!(spec, c1.variable)
    _occbin_check_kind!(spec, c2.variable)
    spec, sol = _occbin_reference_solution(spec)
    P, Q = _occbin_transition(sol)

    # Extract alternative specs from dict
    haskey(alt_specs, (1, 0)) || throw(ArgumentError("alt_specs must contain key (1,0) for constraint 1 binding"))
    haskey(alt_specs, (0, 1)) || throw(ArgumentError("alt_specs must contain key (0,1) for constraint 2 binding"))
    haskey(alt_specs, (1, 1)) || throw(ArgumentError("alt_specs must contain key (1,1) for both constraints binding"))

    alt1_spec = _occbin_prepare_spec(alt_specs[(1, 0)])
    alt2_spec = _occbin_prepare_spec(alt_specs[(0, 1)])
    alt12_spec = _occbin_prepare_spec(alt_specs[(1, 1)])

    # Extract regimes and constants
    ref_regime = _extract_regime(spec)
    alt1_regime = _extract_regime(alt1_spec)
    alt2_regime = _extract_regime(alt2_spec)
    alt12_regime = _extract_regime(alt12_spec)

    d_ref = _regime_constant(spec)
    d_alt1 = _regime_constant(alt1_spec)
    d_alt2 = _regime_constant(alt2_spec)
    d_alt12 = _regime_constant(alt12_spec)

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
    orig_nperiods = nperiods
    pw_path, regime_history, converged, iterations =
        _guess_verify_two(ref_regime, alt1_regime, alt2_regime, alt12_regime,
                          d_ref, d_alt1, d_alt2, d_alt12,
                          P, Q, spec, c1, c2, shock_path, nperiods;
                          maxiter=maxiter, curb_retrench=curb_retrench)

    # Auto-extend if converged but constraint binds at terminal
    _terminal_binding_two(rh) = rh[end, 1] == 1 || rh[end, 2] == 1
    if converged && _terminal_binding_two(regime_history)
        orig_lp, orig_pw, orig_rh = linear_path, pw_path, regime_history
        orig_conv, orig_iter = converged, iterations
        max_nperiods = 2000
        extended = false
        while _terminal_binding_two(regime_history) && nperiods < max_nperiods
            nperiods = min(nperiods * 2, max_nperiods)
            new_shock_path = zeros(T, nperiods, size(shock_path, 2))
            new_shock_path[1:size(shock_path, 1), :] .= shock_path
            shock_path = new_shock_path
            linear_path = _simulate_linear(P, Q, init, shock_path, nperiods)
            pw_path, regime_history, converged, iterations =
                _guess_verify_two(ref_regime, alt1_regime, alt2_regime, alt12_regime,
                                  d_ref, d_alt1, d_alt2, d_alt12,
                                  P, Q, spec, c1, c2, shock_path, nperiods;
                                  maxiter=maxiter, curb_retrench=curb_retrench)
            if !converged
                break
            end
            extended = true
        end
        if extended && converged && !_terminal_binding_two(regime_history)
            linear_path = linear_path[1:orig_nperiods, :]
            pw_path = pw_path[1:orig_nperiods, :]
            regime_history = regime_history[1:orig_nperiods, :]
        else
            linear_path, pw_path, regime_history = orig_lp, orig_pw, orig_rh
            converged, iterations = orig_conv, orig_iter
            nperiods = orig_nperiods
            if _terminal_binding_two(regime_history)
                @warn "OccBin: constraint binding at terminal period ($orig_nperiods). " *
                      "Consider increasing nperiods."
            end
        end
    end

    # Filter output to original variables if augmented
    if spec.augmented
        orig_idx = _original_var_indices(spec)
        vnames = [string(s) for s in spec.original_endog]
        OccBinSolution{T}(
            linear_path[:, orig_idx], pw_path[:, orig_idx], spec.steady_state[orig_idx],
            regime_history, converged, iterations,
            spec, vnames, [c1, c2]
        )
    else
        OccBinSolution{T}(
            linear_path, pw_path, spec.steady_state,
            regime_history, converged, iterations,
            spec, spec.varnames, [c1, c2]
        )
    end
end

"""
    occbin_irf(spec::ModelSpec{T}, constraint::OccBinConstraint{T},
               shock_idx::Int, horizon::Int;
               magnitude::Real=one(T), maxiter::Int=100) → OccBinIRF{T}

Compute impulse response functions under an occasionally binding constraint.

Compares the unconstrained linear IRF with the piecewise-linear OccBin IRF.

# Arguments
- `spec` — DSGE model specification
- `constraint` — the occasionally binding constraint
- `shock_idx` — index of the shock to perturb (1-based)
- `horizon` — number of periods for the IRF

# Keyword Arguments
- `magnitude` — size of the shock (default: 1.0)
- `maxiter` — max guess-and-verify iterations (default: 100)
"""
function occbin_irf(spec::ModelSpec{T}, constraint::OccBinConstraint{T},
                    shock_idx::Int, horizon::Int;
                    magnitude::Real=one(T), maxiter::Int=100) where {T<:AbstractFloat}
    1 <= shock_idx <= spec.n_exog || throw(ArgumentError(
        "shock_idx=$shock_idx out of range [1, $(spec.n_exog)]"))

    shock_path = zeros(T, horizon, spec.n_exog)
    shock_path[1, shock_idx] = T(magnitude)

    sol = occbin_solve(spec, constraint; shock_path=shock_path,
                       nperiods=horizon, maxiter=maxiter)

    shock_name = string(spec.exog[shock_idx])
    OccBinIRF{T}(sol.linear_path, sol.piecewise_path, sol.regime_history,
                  sol.varnames, shock_name)
end

"""
    occbin_irf(spec::ModelSpec{T}, c1::OccBinConstraint{T}, c2::OccBinConstraint{T},
               shock_idx::Int, horizon::Int;
               magnitude::Real=one(T), maxiter::Int=100,
               curb_retrench::Bool=false) → OccBinIRF{T}

Two-constraint variant of OccBin IRF.
"""
function occbin_irf(spec::ModelSpec{T}, c1::OccBinConstraint{T}, c2::OccBinConstraint{T},
                    shock_idx::Int, horizon::Int;
                    magnitude::Real=one(T), maxiter::Int=100,
                    curb_retrench::Bool=false) where {T<:AbstractFloat}
    1 <= shock_idx <= spec.n_exog || throw(ArgumentError(
        "shock_idx=$shock_idx out of range [1, $(spec.n_exog)]"))

    shock_path = zeros(T, horizon, spec.n_exog)
    shock_path[1, shock_idx] = T(magnitude)

    sol = occbin_solve(spec, c1, c2; shock_path=shock_path,
                       nperiods=horizon, maxiter=maxiter,
                       curb_retrench=curb_retrench)

    shock_name = string(spec.exog[shock_idx])
    OccBinIRF{T}(sol.linear_path, sol.piecewise_path, sol.regime_history,
                  sol.varnames, shock_name)
end

"""
    irf(sol::OccBinSolution{T}, horizon::Int;
        shock_idx::Int=1, magnitude::Real=one(T),
        maxiter::Int=100) → OccBinIRF{T}

Compute OccBin IRF from a solved OccBin model. Uses the constraint(s) stored in `sol`.
Preferred over `occbin_irf`.

# Arguments
- `sol` — solved OccBin model (contains constraint from `occbin_solve`)
- `horizon` — number of IRF periods

# Keyword Arguments
- `shock_idx` — index of shock to perturb (default: 1)
- `magnitude` — shock size (default: 1.0)
- `maxiter` — max iterations (default: 100)
"""
function irf(sol::OccBinSolution{T}, horizon::Int;
             shock_idx::Int=1, magnitude::Real=one(T),
             maxiter::Int=100) where {T<:AbstractFloat}
    if length(sol.constraints) == 1
        occbin_irf(sol.spec, sol.constraints[1], shock_idx, horizon;
                   magnitude=magnitude, maxiter=maxiter)
    elseif length(sol.constraints) == 2
        occbin_irf(sol.spec, sol.constraints[1], sol.constraints[2],
                   shock_idx, horizon; magnitude=magnitude, maxiter=maxiter)
    else
        error("OccBinSolution has $(length(sol.constraints)) constraints; expected 1 or 2")
    end
end
