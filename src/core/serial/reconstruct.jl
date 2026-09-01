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
# (the first float scalar/array wins). Recurses into nested arrays, tuples,
# NamedTuples, and dict values. Returns `nothing` for non-float or non-parametric
# types, in which case the un-parameterized constructor is used.
function _infer_float_param(args)
    for a in args
        P = _infer_float_param_value(a)
        P !== nothing && return P
    end
    return nothing
end

function _infer_float_param_value(a)
    a isa AbstractFloat && return typeof(a)
    if a isa AbstractArray
        E = eltype(a)
        E <: AbstractFloat && return E
        E <: Complex{<:AbstractFloat} && return real(E)
        for x in a
            P = _infer_float_param_value(x)
            P !== nothing && return P
        end
        return nothing
    end
    if a isa Tuple    # e.g. PropensityScoreConfig.trimming::Tuple{T,T}
        for t in a
            P = _infer_float_param_value(t)
            P !== nothing && return P
        end
        return nothing
    end
    if a isa NamedTuple    # e.g. DispersionTest.nb2::NamedTuple{…,NTuple{4,T}}
        for v in values(a)
            P = _infer_float_param_value(v)
            P !== nothing && return P
        end
        return nothing
    end
    if a isa AbstractDict
        for v in values(a)
            P = _infer_float_param_value(v)
            P !== nothing && return P
        end
        return nothing
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

# Keyword inner constructor: 29 positionals + `fitted` / `fitted_filtered`.
function _from_serializable(::Type{MSRegModel}, p::AbstractDict, ::Int)
    Tf = eltype(p["y"])
    MSRegModel{Tf}(
        _as_symbol(p["model_type"]),
        p["y"], p["X"], p["k_regimes"], p["p"],
        p["mu"], p["coefs"], p["se_coefs"], p["ar"], p["se_ar"],
        p["sigma2"], p["se_sigma2"], p["P"], p["ergodic"], p["expected_durations"],
        p["filtered_prob"], p["smoothed_prob"], p["residuals"],
        p["loglik"], p["aic"], p["bic"],
        p["n"], p["n_params"], p["switching_var"], p["switching_ar"],
        p["converged"], p["iterations"], p["xnames"], p["yname"];
        fitted=p["fitted"],
        fitted_filtered=p["fitted_filtered"],
    )
end
