# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# Versioned result serialization (T248 / #347)
# =============================================================================
# `save_model(result, path)` / `load_model(path)` persist a fitted model to disk
# in a self-describing, version-tagged container so files survive a package
# upgrade. Bare `Base.serialize` is deliberately NOT the on-disk format — it
# breaks across Julia versions and struct-layout changes.
#
# Design:
#   1. `_to_serializable(m)` reduces a result to a `Dict{String,Any}` of PLAIN
#      values only — numbers, strings, symbols, bools, arrays, `nothing`, and
#      nested `Dict`s. No custom structs survive into the payload (the
#      reproducibility manifest and LP covariance estimator are themselves
#      flattened to tagged dicts). Storing only primitives is exactly what makes
#      the format robust across versions.
#   2. `_build_container(m)` wraps that payload with a metadata header: the
#      `format_version`, the package + Julia versions, a timestamp, the result
#      type name, and (when present) the reproducibility manifest.
#   3. The actual disk read/write is a JLD2 weak-dependency backend
#      (`_write_model_container` / `_read_model_container`, overridden in
#      ext/MacroEconometricModelsJLD2Ext.jl). Without JLD2 loaded the stub raises
#      an informative "]add JLD2" error. ALL model↔dict logic and version
#      validation stay here in src, so they are testable without the backend.
#   4. `load_model` validates the `format_version` and type tag and raises a
#      typed `SerializationError` naming the expected-vs-found version on a
#      mismatch, rather than returning a corrupted object.

"""
    SERIALIZATION_FORMAT_VERSION

On-disk schema version written into every model file by [`save_model`](@ref) and
checked by [`load_model`](@ref). Bumped only on a breaking change to the payload
layout; a file whose version this build does not recognize is rejected with a
[`SerializationError`](@ref) rather than mis-read.
"""
const SERIALIZATION_FORMAT_VERSION = 1

# Result types with round-trip support. Maps the stored type tag → the concrete
# type, so `load_model` dispatches `_from_serializable` on the recorded name.
const _SERIALIZABLE_TYPES = Dict{String,Type}(
    "VARModel"       => VARModel,
    "BVARPosterior"  => BVARPosterior,
    "RegModel"       => RegModel,
    "LogitModel"     => LogitModel,
    "ProbitModel"    => ProbitModel,
    "LPModel"        => LPModel,
)

# ─────────────────────────────────────────────────────────────────────────────
# Field-value flattening (struct → plain values) and its inverse helpers
# ─────────────────────────────────────────────────────────────────────────────

# Plain values pass straight through; JLD2 stores numbers/strings/symbols/bools/
# arrays/nothing/nested-dicts robustly. Custom structs are flattened to dicts.
_ser_field(x) = x
_ser_field(x::ReproManifest) = _manifest_to_dict(x)
_ser_field(x::AbstractCovarianceEstimator) = _cov_to_dict(x)

_as_symbol(x::Symbol) = x
_as_symbol(x::AbstractString) = Symbol(x)

# Generic public-field capture: {fieldname => flattened value} for every field.
function _capture_fields(m)
    d = Dict{String,Any}()
    for f in fieldnames(typeof(m))
        d[String(f)] = _ser_field(getfield(m, f))
    end
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
# Per-type reduction and reconstruction
# ─────────────────────────────────────────────────────────────────────────────

# Reduction is generic (public-field capture); reconstruction is explicit per
# type so it goes through the type's real constructor (validation preserved) and
# is forward-safe: a v1 loader knows exactly the v1 field set.
_to_serializable(m::VARModel)      = _capture_fields(m)
_to_serializable(m::BVARPosterior) = _capture_fields(m)
_to_serializable(m::RegModel)      = _capture_fields(m)
_to_serializable(m::LogitModel)    = _capture_fields(m)
_to_serializable(m::ProbitModel)   = _capture_fields(m)
_to_serializable(m::LPModel)       = _capture_fields(m)

function _from_serializable(::Type{VARModel}, p::AbstractDict, ::Int)
    VARModel(p["Y"], p["p"], p["B"], p["U"], p["Sigma"], p["aic"], p["bic"], p["hqic"], p["varnames"])
end

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

# ─────────────────────────────────────────────────────────────────────────────
# Container assembly + validation
# ─────────────────────────────────────────────────────────────────────────────

function _build_container(m)
    tname = string(nameof(typeof(m)))
    haskey(_SERIALIZABLE_TYPES, tname) || throw(SerializationError(
        "save_model does not support $(typeof(m)); supported types: " *
        join(sort(collect(keys(_SERIALIZABLE_TYPES))), ", ")))
    return Dict{String,Any}(
        "format_version"  => SERIALIZATION_FORMAT_VERSION,
        "package_version" => _repro_package_version(),
        "julia_version"   => string(VERSION),
        "created"         => _repro_timestamp(),
        "type"            => tname,
        "manifest"        => _manifest_to_dict(_extract_manifest(m)),
        "payload"         => _to_serializable(m),
    )
end

function _reconstruct_from_container(container::AbstractDict)
    ver = get(container, "format_version", nothing)
    ver isa Integer || throw(SerializationError(
        "not a MacroEconometricModels model file: missing or non-integer format_version"))
    ver == SERIALIZATION_FORMAT_VERSION || throw(SerializationError(
        "unsupported serialization format_version $ver: this build reads version " *
        "$SERIALIZATION_FORMAT_VERSION. Re-save with the current release, or load with a " *
        "package version whose SERIALIZATION_FORMAT_VERSION == $ver."))
    tname = get(container, "type", nothing)
    tname isa AbstractString || throw(SerializationError("serialized model is missing its type tag"))
    haskey(_SERIALIZABLE_TYPES, tname) || throw(SerializationError(
        "serialized type '$tname' is not loadable by this build; supported: " *
        join(sort(collect(keys(_SERIALIZABLE_TYPES))), ", ")))
    payload = get(container, "payload", nothing)
    payload isa AbstractDict || throw(SerializationError("serialized model '$tname' has no payload"))
    return _from_serializable(_SERIALIZABLE_TYPES[tname], payload, ver)
end

# ─────────────────────────────────────────────────────────────────────────────
# Backend stubs (overridden by the JLD2 extension) + public API
# ─────────────────────────────────────────────────────────────────────────────

# Extension entry points — the real methods live in
# ext/MacroEconometricModelsJLD2Ext.jl and override these more-specific-arg stubs.
_write_model_container(path, container) =
    error("save_model requires the JLD2 package. Run `]add JLD2` and `using JLD2` to enable it.")
_read_model_container(path) =
    error("load_model requires the JLD2 package. Run `]add JLD2` and `using JLD2` to enable it.")

"""
    save_model(model, path) -> path

Persist a fitted `model` to `path` in a versioned, self-describing container.
Supported: `VARModel`, `BVARPosterior`, `RegModel`, `LogitModel`, `ProbitModel`,
`LPModel`. The file records the [`SERIALIZATION_FORMAT_VERSION`](@ref), the
package and Julia versions, a timestamp, and — for a randomized result — its
reproducibility manifest. Only public fields are stored; cached factorizations
are recomputed on load rather than persisted.

Requires the JLD2 backend: `]add JLD2` then `using JLD2`.

```julia
using JLD2
m = estimate_var(Y, 2)
save_model(m, "model.jld2")
m2 = load_model("model.jld2")   # identical public fields
```
"""
function save_model(model, path::AbstractString)
    container = _build_container(model)
    _write_model_container(String(path), container)
    return path
end

"""
    load_model(path) -> model

Reconstruct a model saved by [`save_model`](@ref). Validates the stored
`format_version` and type tag; raises a [`SerializationError`](@ref) naming the
expected-versus-found version on an unrecognized format, rather than returning a
corrupted object. Requires the JLD2 backend (`using JLD2`).
"""
function load_model(path::AbstractString)
    isfile(path) || throw(SerializationError("no such model file: $path"))
    container = _read_model_container(String(path))
    container isa AbstractDict || throw(SerializationError(
        "file '$path' does not contain a MacroEconometricModels model container"))
    return _reconstruct_from_container(container)
end
