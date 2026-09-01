# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Shared DSGE intermediate representation and `ModelSpec`.

Unified `ModelSpec{T,A}` IR. See issues #630–#637.
"""

# =============================================================================
# Empty agent collection — RA / residual-only models
# =============================================================================

"""
    NoAgents

Type of `NamedTuple()`: a `ModelSpec` with no heterogeneous populations.
"""
const NoAgents = NamedTuple{(), Tuple{}}

# =============================================================================
# Named equations and IR
# =============================================================================

"""
    TimingInfo

Lead/lag metadata for one equation after augmentation.
"""
struct TimingInfo
    max_lag::Int
    max_lead::Int
    has_lead::Bool
end

TimingInfo() = TimingInfo(0, 0, false)

"""
    NamedEquation

One compiled residual with a stable name and optional defining variable.

`residual` is `f(y_t, y_lag, y_lead, ε, θ) → scalar`. Closures close over
index maps, not parameter values. `regimes` holds alternate residuals
(e.g. `:binding` for OccBin).
"""
struct NamedEquation
    name::Symbol
    defines::Union{Nothing,Symbol}
    expr::Expr
    residual::Function
    timing::TimingInfo
    regimes::Dict{Symbol,NamedEquation}
end

function NamedEquation(name::Symbol, defines::Union{Nothing,Symbol}, expr::Expr,
                       residual::Function;
                       timing::TimingInfo=TimingInfo(),
                       regimes::Dict{Symbol,NamedEquation}=Dict{Symbol,NamedEquation}())
    NamedEquation(name, defines, expr, residual, timing, regimes)
end

"""
    IRDecl

One scanned declaration (`parameters`, `heterogeneous`, `clock`, …).
`population` is a free name (`:households`, `:banks`) or `nothing`.
"""
struct IRDecl
    kind::Symbol
    population::Union{Nothing,Symbol}
    payload::Any
end

IRDecl(kind::Symbol, payload) = IRDecl(kind, nothing, payload)

"""
    IREquation

Pre-compile equation: name, optional `defines`, raw LHS/RHS, regimes.
"""
struct IREquation
    name::Symbol
    defines::Union{Nothing,Symbol}
    lhs::Any
    rhs::Any
    regimes::Dict{Symbol,IREquation}
end

function IREquation(name::Symbol, defines::Union{Nothing,Symbol}, lhs, rhs;
                    regimes::Dict{Symbol,IREquation}=Dict{Symbol,IREquation}())
    IREquation(name, defines, lhs, rhs, regimes)
end

"""
    ModelIR

Parser output. `_respec` does **not** recompile this; it is source for `@dsge`
and explicit `compile`. `clock` is `:discrete` or `:continuous`. `horizon` is
`:infinite`, `:finite`, `:ages`, or `:perpetual_youth`.
"""
struct ModelIR
    clock::Symbol
    horizon::Symbol
    declarations::Vector{IRDecl}
    equations::Vector{IREquation}
end

ModelIR() = ModelIR(:discrete, :infinite, IRDecl[], IREquation[])

# =============================================================================
# Agent kind (payloads land in later tasks)
# =============================================================================

"""
    AbstractAgentSystem{T}

Supertype of household / DCEGM / life-cycle / continuous-time / firm /
intermediary populations. The `ModelSpec.agents` NamedTuple holds these.
"""
abstract type AbstractAgentSystem{T<:AbstractFloat} end

# =============================================================================
# ModelSpec
# =============================================================================

"""
    ModelSpec{T,A}

Unified DSGE specification. `A` is a `NamedTuple` of `AbstractAgentSystem`s
(`NoAgents` for residual-only RA models).

RA-hot fields (`residual_fns`, `n_endog`, …) stay on the spec so linearize /
gensys / PF keep `spec.residual_fns[i](...)`.
"""
struct ModelSpec{T<:AbstractFloat, A<:NamedTuple}
    endog::Vector{Symbol}
    exog::Vector{Symbol}
    params::Vector{Symbol}
    param_values::Dict{Symbol,T}
    n_endog::Int
    n_exog::Int
    n_params::Int
    equations::Vector{NamedEquation}
    residual_fns::Vector{Function}
    n_expect::Int
    forward_indices::Vector{Int}
    original_endog::Vector{Symbol}
    original_equations::Vector{NamedEquation}
    n_original_endog::Int
    n_original_eq::Int
    augmented::Bool
    max_lag::Int
    max_lead::Int
    linear::Bool
    ss_fn::Union{Nothing,Function}
    steady_state::Vector{T}
    varnames::Vector{String}
    bellman_utility::Any
    bellman_beta::Any
    bellman_consumption::Union{Nothing,Symbol}
    bellman_controls::Vector{Symbol}
    agents::A
    ir::ModelIR
end

"""Wrap `Vector{Expr}` (programmatic / test constructors) as `NamedEquation`s."""
function _coerce_named_equations(equations, residual_fns)
    eqs = collect(equations)
    isempty(eqs) && return NamedEquation[]
    first(eqs) isa NamedEquation && return NamedEquation[eqs...]
    fns = collect(Function, residual_fns)
    n = length(eqs)
    out = Vector{NamedEquation}(undef, n)
    for i in 1:n
        expr = eqs[i] isa Expr ? eqs[i] : Expr(:call, :+, eqs[i], 0)
        fn = i <= length(fns) ? fns[i] : identity
        out[i] = NamedEquation(Symbol("eq_", i), nothing, expr, fn)
    end
    return out
end

function ModelSpec{T}(endog, exog, params, param_values, equations, residual_fns,
                      n_expect, forward_indices, steady_state,
                      ss_fn::Union{Nothing,Function}=nothing;
                      original_endog::Vector{Symbol}=Symbol[endog...],
                      original_equations=nothing,
                      augmented::Bool=false,
                      max_lag::Int=1,
                      max_lead::Int=1,
                      linear::Bool=false,
                      bellman_utility=nothing,
                      bellman_beta=nothing,
                      bellman_consumption::Union{Nothing,Symbol}=nothing,
                      bellman_controls::AbstractVector=Symbol[],
                      agents::NamedTuple=NamedTuple(),
                      ir::ModelIR=ModelIR(),
                      varnames::Union{Nothing,Vector{String}}=nothing) where {T<:AbstractFloat}
    endog = collect(Symbol, endog)
    exog = collect(Symbol, exog)
    params = collect(Symbol, params)
    residual_fns = collect(Function, residual_fns)
    equations = _coerce_named_equations(equations, residual_fns)
    if original_equations === nothing
        original_equations = equations
    else
        orig = collect(original_equations)
        orig_fns = [eq isa NamedEquation ? eq.residual : identity for eq in orig]
        original_equations = _coerce_named_equations(orig, orig_fns)
    end
    n_endog = length(endog)
    n_exog = length(exog)
    n_params = length(params)
    n_original_endog = length(original_endog)
    n_original_eq = length(original_equations)
    A = typeof(agents)
    if A === NoAgents
        length(equations) == n_endog || throw(ArgumentError(
            "ModelSpec: expected $(n_endog) equations (one per endogenous after " *
            "augmentation), got $(length(equations))"))
        length(residual_fns) == n_endog || throw(ArgumentError(
            "ModelSpec: residual_fns length $(length(residual_fns)) ≠ n_endog $n_endog"))
    end
    if varnames !== nothing && length(varnames) != n_endog
        throw(ArgumentError(
            "varnames length $(length(varnames)) ≠ n_endog $n_endog"))
    end
    length(forward_indices) == n_expect || throw(ArgumentError(
        "ModelSpec: forward_indices length $(length(forward_indices)) ≠ n_expect $n_expect"))
    for i in forward_indices
        1 <= i <= length(equations) || throw(ArgumentError(
            "ModelSpec: forward_indices entry $i is outside 1:$(length(equations))"))
    end
    vnames = varnames === nothing ? [string(s) for s in endog] : copy(varnames)
    return ModelSpec{T,A}(
        endog, exog, params, param_values, n_endog, n_exog, n_params,
        equations, residual_fns, n_expect, collect(Int, forward_indices),
        original_endog, collect(NamedEquation, original_equations),
        n_original_endog, n_original_eq, augmented, max_lag, max_lead, linear,
        ss_fn, steady_state, vnames,
        bellman_utility, bellman_beta, bellman_consumption,
        Symbol[bellman_controls...], agents, ir)
end

"""
    _copy_model_spec(spec::ModelSpec; kwargs...) -> ModelSpec

Copy a compiled `ModelSpec`, overriding any supplied fields. Does **not**
recompile `spec.ir`.
"""
function _copy_model_spec(spec::ModelSpec{T};
                          endog=spec.endog,
                          exog=spec.exog,
                          params=spec.params,
                          param_values=spec.param_values,
                          equations=spec.equations,
                          residual_fns=spec.residual_fns,
                          n_expect=spec.n_expect,
                          forward_indices=spec.forward_indices,
                          steady_state=spec.steady_state,
                          ss_fn=spec.ss_fn,
                          original_endog=spec.original_endog,
                          original_equations=spec.original_equations,
                          augmented=spec.augmented,
                          max_lag=spec.max_lag,
                          max_lead=spec.max_lead,
                          linear=spec.linear,
                          bellman_utility=spec.bellman_utility,
                          bellman_beta=spec.bellman_beta,
                          bellman_consumption=spec.bellman_consumption,
                          bellman_controls=spec.bellman_controls,
                          agents=spec.agents,
                          ir=spec.ir,
                          varnames=spec.varnames) where {T<:AbstractFloat}
    # News-pipeline copies grow `endog` while inheriting the original display
    # names; pad with the new symbols so the constructor check still holds.
    n = length(endog)
    if varnames !== nothing && length(varnames) < n
        varnames = vcat(varnames, [string(s) for s in endog[length(varnames)+1:end]])
    end
    ModelSpec{T}(
        endog, exog, params, param_values, equations, residual_fns,
        n_expect, forward_indices, steady_state, ss_fn;
        original_endog=original_endog,
        original_equations=original_equations,
        augmented=augmented,
        max_lag=max_lag,
        max_lead=max_lead,
        linear=linear,
        bellman_utility=bellman_utility,
        bellman_beta=bellman_beta,
        bellman_consumption=bellman_consumption,
        bellman_controls=bellman_controls,
        agents=agents,
        ir=ir,
        varnames=varnames === nothing ? nothing : copy(varnames),
    )
end

"""
    _respec(spec::ModelSpec, new_pv) -> ModelSpec

Copy the compiled view at a new parameter dictionary. Steady state is cleared.
Does **not** recompile `spec.ir`.
"""
function _respec(spec::ModelSpec{T,A}, new_pv) where {T<:AbstractFloat,A}
    _copy_model_spec(spec; param_values=new_pv, steady_state=T[])
end

function _original_var_indices(spec::ModelSpec)
    spec.augmented || return collect(1:spec.n_endog)
    return [findfirst(==(v), spec.endog) for v in spec.original_endog]
end

_has_agents(spec::ModelSpec) = !isempty(spec.agents)

"""
    has_kind(spec, S) -> Bool

`true` if `spec.agents` holds at least one value of type `S`
(an [`AbstractAgentSystem`](@ref) subtype). Dispatch uses the type, never the
NamedTuple key.
"""
has_kind(spec::ModelSpec, ::Type{S}) where {S} =
    any(v -> v isa S, values(spec.agents))

"""
    agents_of(spec, S)

Iterator over the `spec.agents` values of type `S`.
"""
agents_of(spec::ModelSpec, ::Type{S}) where {S} =
    (v for v in values(spec.agents) if v isa S)

"""
    to_spec(m) -> ModelSpec

Wrap a family constructor (`BlanchardOLG`, `DCEGMProblem`, `LifeCycleOLG`,
`CTAiyagari`, `CTTwoAsset`, `FirmSystem`, `IntermediarySystem`) as a
[`ModelSpec`](@ref). `solve(to_spec(m))` dispatches on the payload kind.
"""
function to_spec end

# =============================================================================
# Shared equation transform / substitution (used by @dsge and load-time recompile)
# =============================================================================

"""
    _parse_time_index(ex) → Int

Parse time index from a `ref` subscript expression.
- `t` → 0
- `(call + t 1)` → 1
- `(call - t 1)` → -1
"""
function _parse_time_index(ex)
    if ex === :t
        return 0
    elseif ex isa Expr && ex.head == :call && length(ex.args) == 3 && ex.args[2] === :t
        op = ex.args[1]
        offset = ex.args[3]
        if op === :(+)
            return Int(offset)
        elseif op === :(-)
            return -Int(offset)
        end
    end
    error("@dsge: unrecognized time index expression: $ex")
end

"""
    _equation_to_residual(eq) → Expr

Transform `LHS = RHS` to `LHS - (RHS)`, creating a residual expression.
"""
function _equation_to_residual(eq::Expr)
    eq.head == :(=) || error("@dsge: equation must be LHS = RHS, got: $eq")
    lhs = eq.args[1]
    rhs = eq.args[2]
    # Unwrap block wrapper if present
    if rhs isa Expr && rhs.head == :block
        inner = filter(a -> !(a isa LineNumberNode), rhs.args)
        length(inner) == 1 || error("@dsge: malformed equation RHS")
        rhs = inner[1]
    end
    return Expr(:call, :(-), lhs, rhs)
end

"""
    _substitute_vars(ex, endog, exog, params) → Expr

Recursively replace:
- `var[t]` → `_y_t_[i]` where `i` = index of `var` in `endog`
- `var[t-1]` → `_y_lag_[i]`
- `var[t+1]` → `_y_lead_[i]`
- `shock[t]` → `_ε_[j]` where `j` = index of `shock` in `exog`
- bare parameter symbols → `_θ_[QuoteNode(name)]`
"""
function _substitute_vars(ex, endog::Vector{Symbol}, exog::Vector{Symbol}, params::Vector{Symbol})
    if ex isa Expr
        # Time-indexed endogenous variable: var[t±k]
        if ex.head == :ref && length(ex.args) == 2 && ex.args[1] isa Symbol
            varname = ex.args[1]::Symbol
            if varname ∈ endog
                idx = findfirst(==(varname), endog)
                offset = _parse_time_index(ex.args[2])
                if offset == 0
                    return Expr(:ref, :_y_t_, idx)
                elseif offset < 0
                    return Expr(:ref, :_y_lag_, idx)
                else  # offset > 0
                    return Expr(:ref, :_y_lead_, idx)
                end
            elseif varname ∈ exog
                jdx = findfirst(==(varname), exog)
                # Shocks should only be [t]
                offset = _parse_time_index(ex.args[2])
                offset == 0 || error("@dsge: exogenous shock $varname can only be indexed at [t], got offset $offset")
                return Expr(:ref, :_ε_, jdx)
            end
            # Not a known variable — could be some other indexing; leave as is
        end

        # Recurse into children
        new_args = Any[]
        for a in ex.args
            push!(new_args, _substitute_vars(a, endog, exog, params))
        end
        return Expr(ex.head, new_args...)

    elseif ex isa Symbol
        # Bare parameter symbol → θ[:name]
        if ex ∈ params
            return Expr(:ref, :_θ_, QuoteNode(ex))
        end
        # Other symbols (operators like +, -, *, ^, numeric constants, etc.) pass through
        return ex
    else
        # Literal values (numbers, etc.)
        return ex
    end
end

# =============================================================================
# Residual AST allowlist + load-time recompile (DSER-02 / #760)
# =============================================================================

const _RESIDUAL_AST_ALLOW = Set{Symbol}([
    :+, :-, :*, :/, :^, :exp, :log, :sqrt, :abs, :max, :min,
    :tanh, :sinh, :cosh, :atan, :sin, :cos, :tan, :erf, :erfc,
    :sign, :floor, :ceil, :round, :mod, :rem, :hypot, :log1p, :expm1,
    :inv, :cbrt, :clamp, :(==), :<, :>, :<=, :>=, :!, :&, :|,
])

"""
    _sanitize_residual_ast(ex, eqname) → ex

Walk `ex` and reject any call / head not on the residual allowlist. A loaded
`ModelSpec` file is executed code: this runs before `Core.eval`.
"""
function _sanitize_residual_ast(ex, eqname::Symbol)
    _sanitize_residual_ast!(ex, eqname)
    ex
end
function _sanitize_residual_ast!(ex::Expr, eqname::Symbol)
    if ex.head === :call
        f = ex.args[1]
        f isa Symbol || throw(SerializationError("ModelSpec equation $eqname: call is not a Symbol"))
        f in _RESIDUAL_AST_ALLOW || throw(SerializationError(
            "ModelSpec equation $eqname: function `$f` is not on the residual AST allowlist"))
        foreach(a -> _sanitize_residual_ast!(a, eqname), ex.args[2:end])
    elseif ex.head in (:ref, :tuple, :vect, :vcat, :hcat, :row, :if, :&&, :||, :comparison, :block)
        foreach(a -> _sanitize_residual_ast!(a, eqname), ex.args)
    elseif ex.head === :(=)
        foreach(a -> _sanitize_residual_ast!(a, eqname), ex.args)
    else
        throw(SerializationError("ModelSpec equation $eqname: Expr head $(ex.head) is not allowed"))
    end
    ex
end
_sanitize_residual_ast!(::Any, ::Symbol) = nothing

"""
    _compile_residual(expr, endog, exog, params) → Function

Rebuild `f(y_t, y_lag, y_lead, ε, θ) → scalar` from a stored residual (or
`LHS = RHS`) expression. Sanitizes the substituted AST, `Core.eval`s in this
module, and wraps with `invokelatest` so the caller is world-age safe.
"""
function _compile_residual(expr::Expr, endog, exog, params)
    endog = Symbol[endog...]
    exog = Symbol[exog...]
    params = Symbol[params...]
    body = expr.head === :(=) ? _equation_to_residual(expr) : expr
    subst = _substitute_vars(body, endog, exog, params)
    _sanitize_residual_ast(subst, :residual)
    fn = Core.eval(MacroEconometricModels,
                   Expr(:->, Expr(:tuple, :_y_t_, :_y_lag_, :_y_lead_, :_ε_, :_θ_), subst))
    (a, b, c, e, θ) -> Base.invokelatest(fn, a, b, c, e, θ)
end

function _recompile_named_equation(eq::NamedEquation, endog, exog, params)
    residual = _compile_residual(eq.expr, endog, exog, params)
    regimes = Dict{Symbol,NamedEquation}()
    for (k, v) in eq.regimes
        regimes[k isa Symbol ? k : Symbol(k)] = _recompile_named_equation(v, endog, exog, params)
    end
    NamedEquation(eq.name, eq.defines, eq.expr, residual; timing=eq.timing, regimes=regimes)
end
