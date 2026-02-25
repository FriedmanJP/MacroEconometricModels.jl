# MacroEconometricModels.jl — DSGE Constraint Types
#
# Constraint types for constrained steady-state and perfect-foresight solvers.
# These types are always available; the JuMP extension provides the actual solvers.

"""
    DSGEConstraint{T}

Abstract type for DSGE model constraints (variable bounds and nonlinear inequalities).
"""
abstract type DSGEConstraint{T<:AbstractFloat} end

"""
    VariableBound{T} <: DSGEConstraint{T}

Box constraint on an endogenous variable: `lower <= var <= upper`.

# Fields
- `var_name::Symbol` — endogenous variable name (must exist in `spec.endog`)
- `lower::Union{T, Nothing}` — lower bound (nothing = unbounded below)
- `upper::Union{T, Nothing}` — upper bound (nothing = unbounded above)
"""
struct VariableBound{T} <: DSGEConstraint{T}
    var_name::Symbol
    lower::Union{T, Nothing}
    upper::Union{T, Nothing}
end

"""
    NonlinearConstraint{T} <: DSGEConstraint{T}

Nonlinear inequality constraint: `fn(y, y_lag, y_lead, e, theta) <= 0`.

The function signature matches `@dsge` residual functions.
For steady state: evaluated with `y_lag = y_lead = y`, `e = 0`.
For perfect foresight: evaluated at each period `t`.

# Fields
- `fn::Function` — `(y, y_lag, y_lead, e, theta) -> scalar` (must be <= 0 when satisfied)
- `label::String` — display label for diagnostics
"""
struct NonlinearConstraint{T} <: DSGEConstraint{T}
    fn::Function
    label::String
end

"""
    variable_bound(var::Symbol; lower=nothing, upper=nothing) -> VariableBound{Float64}

Create a variable bound constraint.

# Examples
```julia
variable_bound(:i, lower=0.0)                    # ZLB: i_t >= 0
variable_bound(:c, lower=0.0)                    # positivity
variable_bound(:h, lower=0.0, upper=1.0)         # hours in [0, 1]
```
"""
function variable_bound(var::Symbol; lower::Union{Real,Nothing}=nothing,
                         upper::Union{Real,Nothing}=nothing)
    lower === nothing && upper === nothing &&
        throw(ArgumentError("At least one of lower or upper must be specified"))
    T = Float64
    lo = lower === nothing ? nothing : T(lower)
    hi = upper === nothing ? nothing : T(upper)
    lo !== nothing && hi !== nothing && lo > hi &&
        throw(ArgumentError("lower ($lo) must be <= upper ($hi)"))
    VariableBound{T}(var, lo, hi)
end

"""
    nonlinear_constraint(fn::Function; label="constraint") -> NonlinearConstraint{Float64}

Create a nonlinear inequality constraint `fn(y, y_lag, y_lead, e, theta) <= 0`.

# Examples
```julia
# Collateral constraint: debt <= 0.8 * capital
nonlinear_constraint((y, y_lag, y_lead, e, theta) -> y[3] - 0.8 * y[1]; label="collateral")
```
"""
function nonlinear_constraint(fn::Function; label::String="constraint")
    NonlinearConstraint{Float64}(fn, label)
end

# =============================================================================
# Dispatch stubs — JuMP extension overrides these methods
# =============================================================================

const _JUMP_INSTALL_MSG = "Constrained solving requires JuMP and Ipopt. Install with:\n" *
    "  using Pkg; Pkg.add(\"JuMP\"); Pkg.add(\"Ipopt\")\n" *
    "Then load: import JuMP, Ipopt"

# Function declarations — JuMP extension adds methods to these.
# Without the extension loaded, calling these gives MethodError; callers check first.
function _jump_compute_steady_state end
function _jump_perfect_foresight end
function _path_compute_steady_state end
function _path_perfect_foresight end

function _check_jump_loaded()
    if !hasmethod(_jump_compute_steady_state, Tuple{DSGESpec, Vector})
        throw(ArgumentError(_JUMP_INSTALL_MSG))
    end
    return nothing
end

"""Check if PATHSolver is loaded (extension adds method to `_path_compute_steady_state`)."""
function _path_available()
    return hasmethod(_path_compute_steady_state, Tuple{DSGESpec, Vector})
end

"""
    _select_solver(constraints, solver_override) -> Symbol

Auto-detect solver: VariableBounds-only → :path (if available), NonlinearConstraints → :ipopt.
User override always wins.
"""
function _select_solver(constraints::Vector, solver_override::Union{Nothing,Symbol})
    solver_override !== nothing && return solver_override
    has_nlcon = any(c -> c isa NonlinearConstraint, constraints)
    has_nlcon && return :ipopt
    return _path_available() ? :path : :ipopt
end

"""
    _validate_constraints(spec, constraints)

Check that all variable names in constraints exist in `spec.endog`.
"""
function _validate_constraints(spec::DSGESpec{T}, constraints::Vector) where {T}
    for c in constraints
        if c isa VariableBound
            idx = findfirst(==(c.var_name), spec.endog)
            idx === nothing && throw(ArgumentError(
                "Variable :$(c.var_name) not found in endogenous variables $(spec.endog)"))
        elseif c isa NonlinearConstraint
            try
                y_test = ones(T, spec.n_endog)
                ε_test = zeros(T, spec.n_exog)
                val = c.fn(y_test, y_test, y_test, ε_test, spec.param_values)
                val isa Real || throw(ArgumentError(
                    "Constraint '$(c.label)' fn must return a scalar, got $(typeof(val))"))
            catch e
                e isa ArgumentError && rethrow(e)
                throw(ArgumentError(
                    "Constraint '$(c.label)' fn failed validation: $e"))
            end
        end
    end
    return nothing
end

# =============================================================================
# Shared helpers for JuMP extensions (no JuMP dependency, just closures)
# =============================================================================

"""
Build a callable for the steady-state objective (sum of squared residuals).
Takes n scalar args. Returns scalar. ForwardDiff-compatible via `S<:Real`.
"""
function _build_ss_objective(residual_fns, n_ε, θ)
    function ss_obj(args::S...) where {S<:Real}
        y = collect(args)
        ε_z = zeros(S, n_ε)
        total = zero(S)
        for fn in residual_fns
            try
                r = fn(y, y, y, ε_z, θ)
                total += r^2
            catch e
                (e isa DomainError || e isa InexactError) && return S(NaN)
                rethrow(e)
            end
        end
        return total
    end
    return ss_obj
end

"""
Build a callable for a single residual equation i at steady state.
Takes n scalar args (the SS values). Returns scalar residual.
"""
function _build_ss_residual_i(residual_fn, n_ε, θ)
    function ss_res(args::S...) where {S<:Real}
        y = collect(args)
        ε_z = zeros(S, n_ε)
        try
            return residual_fn(y, y, y, ε_z, θ)
        catch e
            (e isa DomainError || e isa InexactError) && return S(NaN)
            rethrow(e)
        end
    end
    return ss_res
end

"""
Build a callable for a nonlinear constraint at steady state.
"""
function _build_ss_nlcon(cfn, n_ε, θ)
    function ss_nlcon(args::S...) where {S<:Real}
        y = collect(args)
        ε_z = zeros(S, n_ε)
        cfn(y, y, y, ε_z, θ)
    end
    return ss_nlcon
end

"""
Build a callable for one equilibrium equation in the perfect foresight system.
Takes 3n + n_ε scalar args: [y_t; y_lag; y_lead; ε_t].
"""
function _build_pf_equation(fn, n, n_ε, θ)
    function pf_eq(args::S...) where {S<:Real}
        a = collect(args)
        y_t    = a[1:n]
        y_lag  = a[n+1:2n]
        y_lead = a[2n+1:3n]
        ε_t    = a[3n+1:3n+n_ε]
        try
            return fn(y_t, y_lag, y_lead, ε_t, θ)
        catch e
            (e isa DomainError || e isa InexactError) && return S(NaN)
            rethrow(e)
        end
    end
    return pf_eq
end

"""
Build a callable for a nonlinear inequality constraint in the perfect foresight system.
"""
function _build_pf_nlcon(cfn, n, n_ε, θ)
    function pf_nlcon(args::S...) where {S<:Real}
        a = collect(args)
        y_t    = a[1:n]
        y_lag  = a[n+1:2n]
        y_lead = a[2n+1:3n]
        ε_t    = a[3n+1:3n+n_ε]
        cfn(y_t, y_lag, y_lead, ε_t, θ)
    end
    return pf_nlcon
end

"""Extract variable bounds from constraints as lower/upper vectors."""
function _extract_bounds(spec::DSGESpec{T}, constraints::Vector) where {T}
    n = spec.n_endog
    FT = isempty(spec.steady_state) ? Float64 : eltype(spec.steady_state)
    lower = fill(FT(-Inf), n)
    upper = fill(FT(Inf), n)
    for c in constraints
        if c isa VariableBound
            idx = findfirst(==(c.var_name), spec.endog)
            c.lower !== nothing && (lower[idx] = FT(c.lower))
            c.upper !== nothing && (upper[idx] = FT(c.upper))
        end
    end
    return lower, upper
end
