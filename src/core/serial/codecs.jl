# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# ─────────────────────────────────────────────────────────────────────────────
# Field-value flattening (struct → plain values)
# ─────────────────────────────────────────────────────────────────────────────

# A value whose element type is one of these needs no flattening — JLD2 stores
# numbers/strings/symbols/bools/chars/enums (and arrays/dicts of them) robustly.
_is_plain_eltype(::Type{S}) where {S} = S <: Union{Number,AbstractString,Symbol,Char,Bool,Enum}

# Is `x` a MacroEconometricModels struct we should flatten recursively? Types
# from Base / LinearAlgebra / Distributions (parentmodule ≠ this package) fall
# through to JLD2's own storage; the manifest and covariance estimators have
# their own tagged-dict methods below and never reach here. Functions are
# singleton structs whose parentmodule is this package for named helpers;
# they are handled by the Function codec, not as MEM structs.
function _is_mem_struct(x)
    x isa Function && return false
    T = typeof(x)
    isstructtype(T) || return false
    return parentmodule(T) === MacroEconometricModels
end

# The manifest and LP covariance estimators are flattened by dedicated methods;
# everything else routes through the generic recursive `_ser_field(x)`.
_ser_field(x::ReproManifest) = _manifest_to_dict(x)
_ser_field(x::AbstractCovarianceEstimator) = _cov_to_dict(x)

function _ser_field(ex::Expr)
    clean = Base.remove_linenums!(deepcopy(ex))
    Dict{String,Any}("__expr__" => String(clean.head),
                     "args" => Any[_ser_expr_arg(a) for a in clean.args])
end
_ser_expr_arg(x::Union{Symbol,Number,AbstractString,Bool,Nothing}) = x
_ser_expr_arg(x::QuoteNode) = Dict{String,Any}("__quotenode__" => _ser_expr_arg(x.value))
_ser_expr_arg(x::GlobalRef) = Dict{String,Any}("__globalref__" => [String(nameof(x.mod)), String(x.name)])
_ser_expr_arg(x::Expr) = _ser_field(x)
_ser_expr_arg(x) = throw(SerializationError(
    "Expr leaf of type $(typeof(x)) cannot be serialized; payload must stay plain"))

function _deser_expr(d::AbstractDict)
    args = Any[_deser_expr_arg(a) for a in d["args"]]
    Expr(Symbol(d["__expr__"]), args...)
end
_deser_expr_arg(d::AbstractDict) = haskey(d, "__quotenode__") ? QuoteNode(_deser_expr_arg(d["__quotenode__"])) :
    haskey(d, "__globalref__") ? GlobalRef(getfield(Main, Symbol(d["__globalref__"][1])), Symbol(d["__globalref__"][2])) :
    haskey(d, "__expr__") ? _deser_expr(d) : _deser_field(d)
_deser_expr_arg(x) = x

function _ser_field(nt::NamedTuple)
    Dict{String,Any}("__namedtuple__" => true,
                     "keys" => String[string(k) for k in keys(nt)],
                     "values" => Any[_ser_field(v) for v in values(nt)])
end
function _deser_namedtuple(d::AbstractDict)
    ks = Symbol.(d["keys"])
    vs = Tuple(_deser_field(v) for v in d["values"])
    NamedTuple{Tuple(ks)}(vs)
end

_ser_field(p::Pair) = Dict{String,Any}("__pair__" => true,
    "first" => _ser_field(p.first), "second" => _ser_field(p.second))
_deser_pair(d::AbstractDict) = _deser_field(d["first"]) => _deser_field(d["second"])

function _is_named_function(f::Function)
    n = nameof(f)
    startswith(string(n), "#") && return false
    m = parentmodule(f)
    isdefined(m, n) && getfield(m, n) === f
end
function _ser_field(f::Function)
    _is_named_function(f) || return nothing
    Dict{String,Any}("__function__" => String(nameof(f)),
                     "module" => String(nameof(parentmodule(f))))
end
const _FUNCTION_MODULES = (MacroEconometricModels, Base, Main)
function _deser_function(d::AbstractDict)
    modname = Symbol(d["module"]); fname = Symbol(d["__function__"])
    for m in _FUNCTION_MODULES
        nameof(m) === modname && isdefined(m, fname) && return getfield(m, fname)
    end
    throw(SerializationError(
        "function '$(d["module"]).$(d["__function__"])' referenced by the file is not defined in this session"))
end

function _ser_field(S::SparseMatrixCSC)
    Dict{String,Any}("__sparse__" => true, "m" => S.m, "n" => S.n,
                     "colptr" => S.colptr, "rowval" => S.rowval, "nzval" => S.nzval)
end
_deser_sparse(d::AbstractDict) = SparseMatrixCSC(Int(d["m"]), Int(d["n"]),
    Vector{Int}(d["colptr"]), Vector{Int}(d["rowval"]), d["nzval"])

_ser_field(::Factorization) = nothing

_ser_field(r::UnitRange) = Dict{String,Any}("__unitrange__" => true, "start" => r.start, "stop" => r.stop)
_deser_unitrange(d::AbstractDict) = (d["start"]):(d["stop"])

function _ser_field(x)
    x === nothing && return nothing
    x isa Missing && return missing
    x isa Enum && return x
    (x isa Number || x isa AbstractString || x isa Symbol || x isa Char) && return x
    if x isa AbstractArray
        _is_plain_eltype(eltype(x)) && return x
        return map(_ser_field, x)
    end
    x isa Tuple && return map(_ser_field, x)
    if x isa AbstractDict
        _is_plain_eltype(valtype(x)) && return x
        return Dict(k => _ser_field(v) for (k, v) in x)
    end
    _is_mem_struct(x) && return _struct_to_dict(x)
    return x
end

# Generic public-field capture: {fieldname => flattened value} for every field.
function _capture_fields(m)
    d = Dict{String,Any}()
    for f in fieldnames(typeof(m))
        d[String(f)] = _ser_field(getfield(m, f))
    end
    return d
end

# A nested struct is captured like the top-level payload, plus a `__struct__`
# tag so `_deser_field` can resolve its concrete type on the way back in.
function _struct_to_dict(m)
    d = _capture_fields(m)
    d["__struct__"] = String(nameof(typeof(m)))
    return d
end

_extract_manifest(m) = hasfield(typeof(m), :manifest) ? getfield(m, :manifest) : nothing

# LP covariance estimator ↔ dict. These are small config structs (bandwidth,
# kernel, prewhiten) — capture the type name + fields, rebuild by name.
function _cov_to_dict(c::AbstractCovarianceEstimator)
    d = Dict{String,Any}("__estimator__" => string(nameof(typeof(c))))
    for f in fieldnames(typeof(c))
        d[String(f)] = getfield(c, f)
    end
    return d
end

function _cov_from_dict(d::AbstractDict)
    name = String(d["__estimator__"])
    if name == "NeweyWestEstimator"
        return NeweyWestEstimator{Float64}(Int(d["bandwidth"]), _as_symbol(d["kernel"]), Bool(d["prewhiten"]))
    elseif name == "WhiteEstimator"
        return WhiteEstimator()
    elseif name == "DriscollKraayEstimator"
        return DriscollKraayEstimator{Float64}(Int(d["bandwidth"]), _as_symbol(d["kernel"]))
    end
    throw(SerializationError("unknown covariance estimator '$name' in serialized LPModel"))
end

# ─────────────────────────────────────────────────────────────────────────────
# Field-value reconstruction (plain values → struct)
# ─────────────────────────────────────────────────────────────────────────────

# After mapping `_deser_field` over an array, the result is often `Array{Any}`
# because `_deser_field` is untyped. If every element shares a concrete type
# `E`, narrow to the matching array type (`Vector{E}`, `Matrix{E}`, …).
function _narrow_decoded_array(vals::AbstractArray)
    isempty(vals) && return vals
    E = typeof(first(vals))
    isconcretetype(E) || return vals
    all(v -> typeof(v) === E, vals) || return vals
    eltype(vals) === E && return vals
    out = similar(vals, E)
    copyto!(out, vals)
    return out
end

# Inverse of `_ser_field`: rebuild manifests / covariance estimators / nested
# structs from their tagged dicts; recurse into arrays/dicts of them.
function _deser_field(x)
    if x isa AbstractDict
        haskey(x, "__expr__")        && return _deser_expr(x)
        haskey(x, "__quotenode__")   && return QuoteNode(_deser_expr_arg(x["__quotenode__"]))
        haskey(x, "__globalref__")   && return GlobalRef(getfield(Main, Symbol(x["__globalref__"][1])),
                                                         Symbol(x["__globalref__"][2]))
        haskey(x, "__namedtuple__")  && return _deser_namedtuple(x)
        haskey(x, "__pair__")        && return _deser_pair(x)
        haskey(x, "__function__")    && return _deser_function(x)
        haskey(x, "__sparse__")      && return _deser_sparse(x)
        haskey(x, "__unitrange__")   && return _deser_unitrange(x)
        haskey(x, "__manifest__")    && return _manifest_from_dict(x)
        haskey(x, "__estimator__")   && return _cov_from_dict(x)
        haskey(x, "__struct__")      && return _deser_struct(x)
        _is_plain_eltype(valtype(x)) && return x
        return Dict(k => _deser_field(v) for (k, v) in x)
    end
    if x isa AbstractArray
        _is_plain_eltype(eltype(x)) && return x
        return _narrow_decoded_array(map(_deser_field, x))
    end
    x isa Tuple && return map(_deser_field, x)
    return x
end

function _assert_plain_payload(x)
    if x isa AbstractDict
        foreach(_assert_plain_payload, values(x)); return
    end
    if x isa AbstractArray
        _is_plain_eltype(eltype(x)) && return
        foreach(_assert_plain_payload, x); return
    end
    x isa Tuple && (foreach(_assert_plain_payload, x); return)
    x isa Union{Number,AbstractString,Symbol,Char,Bool,Enum,Nothing,Missing} && return
    throw(SerializationError("non-plain payload leaf $(typeof(x))"))
end
