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
    length(forward_indices) == n_expect || throw(ArgumentError(
        "ModelSpec: forward_indices length $(length(forward_indices)) ≠ n_expect $n_expect"))
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
    _respec(spec::ModelSpec, new_pv) -> ModelSpec

Copy the compiled view at a new parameter dictionary. Steady state is cleared.
Does **not** recompile `spec.ir`.
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

function _respec(spec::ModelSpec{T,A}, new_pv) where {T<:AbstractFloat,A}
    _copy_model_spec(spec; param_values=new_pv, steady_state=T[])
end

equation_exprs(spec::ModelSpec) = [eq.expr for eq in spec.equations]

_eq_expr(eq::NamedEquation) = eq.expr
_eq_expr(eq::Expr) = eq

function _original_var_indices(spec::ModelSpec)
    spec.augmented || return collect(1:spec.n_endog)
    return [findfirst(==(v), spec.endog) for v in spec.original_endog]
end

has_agents(spec::ModelSpec) = !isempty(spec.agents)

has_kind(spec::ModelSpec, ::Type{S}) where {S} =
    any(v -> v isa S, values(spec.agents))

agents_of(spec::ModelSpec, ::Type{S}) where {S} =
    (v for v in values(spec.agents) if v isa S)

"""
    to_spec(m) -> ModelSpec

Wrap a family constructor (`BlanchardOLG`, `DCEGMProblem`, `LifeCycleOLG`,
`CTAiyagari`, `CTTwoAsset`) as a [`ModelSpec`](@ref). `solve(to_spec(m))`
dispatches on the payload kind.
"""
function to_spec end
