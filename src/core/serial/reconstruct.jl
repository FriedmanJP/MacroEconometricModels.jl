# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

_as_symbol(x::Symbol) = x
_as_symbol(x::AbstractString) = Symbol(x)

# Resolve a nested `__struct__` tag to its concrete type. Nested types need no
# `_SERIALIZABLE_TYPES` entry — they are looked up by name in the package module.
function _resolve_ser_type(name::AbstractString)
    sym = Symbol(name)
    isdefined(MacroEconometricModels, sym) || throw(SerializationError(
        "serialized nested type '$name' is not defined in this build of MacroEconometricModels"))
    T = getfield(MacroEconometricModels, sym)
    T isa Type || throw(SerializationError("serialized type tag '$name' does not name a type"))
    return T
end

_deser_struct(d::AbstractDict) =
    _from_serializable(_resolve_ser_type(String(d["__struct__"])), d, SERIALIZATION_FORMAT_VERSION)

# Infer a `T<:AbstractFloat` type parameter from the reconstructed field values
# (the first float scalar/array wins). Recurses one-or-more levels into nested
# arrays, tuples, and dict values so `DeterminacyMap.axes::Vector{Vector{T}}`
# still yields `T`. Returns `nothing` for non-float or non-parametric types, in
# which case the un-parameterized constructor is used.
function _infer_float_from_value(a)
    a isa AbstractFloat && return typeof(a)
    if a isa AbstractArray
        E = eltype(a)
        E <: AbstractFloat && return E
        E <: Complex{<:AbstractFloat} && return real(E)
        for x in a
            t = _infer_float_from_value(x)
            t !== nothing && return t
        end
    elseif a isa Tuple    # e.g. PropensityScoreConfig.trimming::Tuple{T,T}
        for t in a
            r = _infer_float_from_value(t)
            r !== nothing && return r
        end
    elseif a isa AbstractDict
        for v in values(a)
            r = _infer_float_from_value(v)
            r !== nothing && return r
        end
    end
    return nothing
end

function _infer_float_param(args)
    for a in args
        t = _infer_float_from_value(a)
        t !== nothing && return t
    end
    return nothing
end

# Construct `T` from positional field values. Parametric types with an inner
# constructor expose only `T{P}(fields...)`; a struct with an un-parameterized
# inner constructor (e.g. `VARModel`) exposes only `T(fields...)`; a struct with
# no inner constructor has both. Try the parameterized form first, fall back.
function _generic_construct(T, args)
    if T isa UnionAll
        P = _infer_float_param(args)
        if P !== nothing
            try
                return T{P}(args...)
            catch e
                (e isa MethodError || e isa TypeError) || rethrow()
            end
        end
    end
    return T(args...)
end

# ─────────────────────────────────────────────────────────────────────────────
# Per-type reconstruction
# ─────────────────────────────────────────────────────────────────────────────

# Generic reconstruction: deserialize each public field and route through the
# type's real constructor (validation preserved). Forward-safe: a v1 loader
# knows exactly the v1 field set via `fieldnames(T)`.
function _from_serializable(::Type{T}, p::AbstractDict, ::Int) where {T}
    args = Any[]
    for f in fieldnames(T)
        key = String(f)
        # A field added to the struct after the file was written must surface as
        # the format's typed error, not a raw KeyError (#538).
        haskey(p, key) || throw(SerializationError(
            "field '$key' of $(nameof(T)) is missing from the payload — the file " *
            "was saved by an older package version, before this field existed. " *
            "Re-create and re-save the object with the current version."))
        push!(args, _deser_field(p[key]))
    end
    return _generic_construct(T, args)
end

# ── explicit overrides where the constructor is not a plain positional call ──

function _from_serializable(::Type{VARModel}, p::AbstractDict, ::Int)
    VARModel(p["Y"], p["p"], p["B"], p["U"], p["Sigma"], p["aic"], p["bic"], p["hqic"], p["varnames"])
end

# `manifest` is a keyword-defaulted constructor argument, not a positional field.
function _from_serializable(::Type{BVARPosterior}, p::AbstractDict, ::Int)
    T = eltype(p["B_draws"])
    mani = _manifest_from_dict(get(p, "manifest", nothing))
    BVARPosterior{T}(p["B_draws"], p["Sigma_draws"], p["n_draws"], p["p"], p["n"],
                     p["data"], _as_symbol(p["prior"]), _as_symbol(p["sampler"]),
                     p["varnames"]; manifest=mani)
end

function _from_serializable(::Type{RegModel}, p::AbstractDict, ::Int)
    T = eltype(p["y"])
    RegModel{T}(p["y"], p["X"], p["beta"], p["vcov_mat"], p["residuals"], p["fitted"],
                p["ssr"], p["tss"], p["r2"], p["adj_r2"], p["f_stat"], p["f_pval"],
                p["loglik"], p["aic"], p["bic"], p["varnames"], _as_symbol(p["method"]),
                _as_symbol(p["cov_type"]), p["weights"], p["Z"], p["endogenous"],
                p["first_stage_f"], p["sargan_stat"], p["sargan_pval"],
                p["cragg_donald_f"], p["kleibergen_paap_f"], p["stock_yogo_10pct"])
end

function _from_serializable(::Type{LogitModel}, p::AbstractDict, ::Int)
    T = eltype(p["y"])
    LogitModel{T}(p["y"], p["X"], p["beta"], p["vcov_mat"], p["residuals"], p["fitted"],
                  p["loglik"], p["loglik_null"], p["pseudo_r2"], p["aic"], p["bic"],
                  p["varnames"], p["converged"], p["iterations"], _as_symbol(p["cov_type"]))
end

function _from_serializable(::Type{ProbitModel}, p::AbstractDict, ::Int)
    T = eltype(p["y"])
    ProbitModel{T}(p["y"], p["X"], p["beta"], p["vcov_mat"], p["residuals"], p["fitted"],
                   p["loglik"], p["loglik_null"], p["pseudo_r2"], p["aic"], p["bic"],
                   p["varnames"], p["converged"], p["iterations"], _as_symbol(p["cov_type"]))
end

function _from_serializable(::Type{LPModel}, p::AbstractDict, ::Int)
    cov = _cov_from_dict(p["cov_estimator"])
    LPModel(p["Y"], p["shock_var"], p["response_vars"], p["horizon"], p["lags"],
            p["B"], p["residuals"], p["vcov"], p["T_eff"], cov, p["varnames"])
end

# Nested config struct inside `StateLPModel`: its inner constructor recomputes the
# derived `F_values` field from (state_var, gamma, threshold, method), so route
# through that 4-argument constructor rather than the full field list.
function _from_serializable(::Type{StateTransition}, p::AbstractDict, ::Int)
    StateTransition(p["state_var"], p["gamma"], p["threshold"], _as_symbol(p["method"]))
end

# `SVARPattern` only exposes a keyword inner constructor (`long_run`).
function _from_serializable(::Type{SVARPattern}, p::AbstractDict, ::Int)
    SVARPattern(p["A"], p["B"]; long_run=_deser_field(p["long_run"]))
end

# Inner constructor validates `:exact/:over/:under/:set`; coerce a stored string.
function _from_serializable(::Type{IdentificationStatus}, p::AbstractDict, ::Int)
    IdentificationStatus(_as_symbol(p["status"]),
                         Vector{Int}(p["ranks"]),
                         Vector{Int}(p["orders"]),
                         Int(p["n_overidentifying"]))
end

# Nested `NamedEquation.residual` is ignored — recompiled from `expr` once the
# owning `ModelSpec` has `endog` / `exog` / `params`. Standalone load uses a
# placeholder; `ModelSpec` reconstruction replaces it.
function _from_serializable(::Type{NamedEquation}, p::AbstractDict, ::Int)
    name = _as_symbol(p["name"])
    defines = p["defines"] === nothing ? nothing : _as_symbol(_deser_field(p["defines"]))
    expr = _deser_field(p["expr"])
    expr isa Expr || throw(SerializationError("NamedEquation.expr is not an Expr"))
    timing = _deser_field(p["timing"])
    timing isa TimingInfo || throw(SerializationError("NamedEquation.timing is not a TimingInfo"))
    regimes_raw = _deser_field(get(p, "regimes", Dict{Symbol,NamedEquation}()))
    regimes = Dict{Symbol,NamedEquation}()
    if regimes_raw isa AbstractDict
        for (k, v) in regimes_raw
            v isa NamedEquation || continue
            regimes[_as_symbol(k)] = v
        end
    end
    NamedEquation(name, defines, expr, identity; timing=timing, regimes=regimes)
end

function _from_serializable(::Type{ModelSpec}, p::AbstractDict, ver::Int)
    _from_serializable_modelspec(p, ver)
end
function _from_serializable(::Type{<:ModelSpec}, p::AbstractDict, ver::Int)
    _from_serializable_modelspec(p, ver)
end

# 18 positionals + keyword accuracy / value-function fields. A generic
# 23-positional call misses the inner constructor.
function _from_serializable(::Type{ProjectionSolution}, p::AbstractDict, ::Int)
    coef = _deser_field(p["coefficients"])
    T = eltype(coef)
    T <: AbstractFloat || (T = Float64)
    ProjectionSolution{T}(
        coef,
        _deser_field(p["state_bounds"]),
        _as_symbol(p["grid_type"]),
        Int(p["degree"]),
        _deser_field(p["collocation_nodes"]),
        T(p["residual_norm"]),
        Int(p["n_basis"]),
        Matrix{Int}(_deser_field(p["multi_indices"])),
        _as_symbol(p["quadrature"]),
        _deser_field(p["spec"]),
        _deser_field(p["linear"]),
        _deser_field(p["impact"]),
        _deser_field(p["steady_state"]),
        Vector{Int}(_deser_field(p["state_indices"])),
        Vector{Int}(_deser_field(p["control_indices"])),
        Bool(p["converged"]),
        Int(p["iterations"]),
        _as_symbol(p["method"]);
        euler_error=T(p["euler_error"]),
        smolyak_levels=Matrix{Int}(_deser_field(p["smolyak_levels"])),
        refinements=Int(p["refinements"]),
        value_fn=_deser_field(p["value_fn"]),
        value_coefficients=_deser_field(p["value_coefficients"]),
    )
end

function _to_serializable(m::ModelSpec)
    d = _capture_fields(m)
    d["had_ss_fn"] = m.ss_fn !== nothing
    return d
end

function _from_serializable_modelspec(p::AbstractDict, ver::Int)
    T = Float64
    ss_raw = p["steady_state"]
    if ss_raw isa AbstractArray && eltype(ss_raw) <: AbstractFloat
        T = eltype(ss_raw)
    else
        pv0 = p["param_values"]
        if pv0 isa AbstractDict
            for v in values(pv0)
                if v isa AbstractFloat
                    T = typeof(v)
                    break
                end
            end
        end
    end

    endog = Symbol[_as_symbol(s) for s in _deser_field(p["endog"])]
    exog = Symbol[_as_symbol(s) for s in _deser_field(p["exog"])]
    params = Symbol[_as_symbol(s) for s in _deser_field(p["params"])]
    pv = _deser_field(p["param_values"])
    param_values = Dict{Symbol,T}(_as_symbol(k) => convert(T, v) for (k, v) in pv)

    function _as_named_eq(eq)
        eq isa NamedEquation && return eq
        eq isa AbstractDict && return _from_serializable(NamedEquation, eq, ver)
        throw(SerializationError("ModelSpec equation payload is not a NamedEquation"))
    end

    equations = NamedEquation[_recompile_named_equation(_as_named_eq(eq), endog, exog, params)
                              for eq in _deser_field(p["equations"])]
    residual_fns = Function[eq.residual for eq in equations]

    original_endog = Symbol[_as_symbol(s) for s in _deser_field(p["original_endog"])]
    # Parser stores `identity` residuals on `original_equations` (pre-augmentation
    # AST, including deep exog news lags). Display uses `.expr`, not `.residual`.
    original_equations = NamedEquation[_as_named_eq(eq)
                                       for eq in _deser_field(p["original_equations"])]

    n_expect = Int(p["n_expect"])
    forward_indices = Int[Int(i) for i in _deser_field(p["forward_indices"])]
    ss_vec = _deser_field(p["steady_state"])
    steady_state = isempty(ss_vec) ? T[] : T[convert(T, x) for x in ss_vec]
    varnames = String[string(s) for s in _deser_field(p["varnames"])]
    bellman_controls = Symbol[_as_symbol(s) for s in _deser_field(p["bellman_controls"])]
    bellman_consumption = let c = _deser_field(p["bellman_consumption"])
        c === nothing ? nothing : _as_symbol(c)
    end
    ir = _deser_field(p["ir"])
    ir isa ModelIR || throw(SerializationError("ModelSpec.ir is not a ModelIR"))
    linear = Bool(p["linear"])
    had_ss_fn = Bool(get(p, "had_ss_fn", false))
    ss_fn = _recompile_ss_fn_from_ir(ir, params, length(endog), linear, had_ss_fn)

    return ModelSpec{T}(
        endog, exog, params, param_values, equations, residual_fns,
        n_expect, forward_indices, steady_state, ss_fn;
        original_endog=original_endog,
        original_equations=original_equations,
        augmented=Bool(p["augmented"]),
        max_lag=Int(p["max_lag"]),
        max_lead=Int(p["max_lead"]),
        linear=linear,
        bellman_utility=_deser_field(p["bellman_utility"]),
        bellman_beta=_deser_field(p["bellman_beta"]),
        bellman_consumption=bellman_consumption,
        bellman_controls=bellman_controls,
        agents=_deser_field(p["agents"]),
        ir=ir,
        varnames=varnames,
    )
end
