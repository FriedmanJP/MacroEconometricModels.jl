# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# ─────────────────────────────────────────────────────────────────────────────
# Container assembly + validation
# ─────────────────────────────────────────────────────────────────────────────

_to_serializable(m) = _capture_fields(m)

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
# JLD2 backend + public API
# ─────────────────────────────────────────────────────────────────────────────

function _write_model_container(path::AbstractString, container; compress::Bool=false)
    JLD2.jldopen(path, "w"; compress=compress) do f
        f["container"] = container
    end
    return path
end
function _read_model_container(path::AbstractString)
    JLD2.jldopen(path, "r") do f
        haskey(f, "container") || throw(SerializationError(
            "file '$path' is not a MacroEconometricModels model file (missing 'container' group)"))
        f["container"]
    end
end

"""
    save_model(model, path) -> path

Persist a fitted `model` — or a data container — to `path` in a versioned,
self-describing container. Coverage spans every VAR/BVAR (including mixed-frequency
and TVP), regression/3SLS, panel, volatility (including IGARCH), factor/Bayesian
FAVAR, ARIMA/SARIMA, ARDL/NARDL, STAR/Markov-switching, local-projection, and GMM
model, plus SVAR identification results, innovation-accounting objects (IRF,
FEVD, historical decomposition, LP-IRF/FEVD, Granger/stationarity/PVAR
diagnostics), forecasts (VAR/BVAR/VECM/ARIMA/LP/MIDAS/factor/volatility/threshold/
STAR/MS/conditional/nowcast) and forecast-evaluation results (DM/CW/MZ/encompassing/
combination), nowcast models (DFM/BVAR/bridge) and news decompositions,
Johansen/GPH/local-Whittle companions, and the data containers
(`TimeSeriesData`, `PanelData`, `CrossSectionData`, `IOData`); the full set is
`MacroEconometricModels._SERIALIZABLE_TYPES`. Exported
concrete structs that are not saveable are listed in `_SERIALIZATION_EXCLUDED`
with a reason — permanent exclusions (rendered HTML, workspaces, transient
`reproduce` reports, inline covariance-estimator configs) and pending
`DSER`/`RSER` registrations. The file records the
[`SERIALIZATION_FORMAT_VERSION`](@ref), the package and Julia versions, a
timestamp, and — for a randomized result — its reproducibility manifest. Only
public fields are stored; cached factorizations are recomputed on load, and
compiled equation functions (DSGE) are not yet serializable.

```julia
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
corrupted object.
"""
function load_model(path::AbstractString)
    isfile(path) || throw(SerializationError("no such model file: $path"))
    container = _read_model_container(String(path))
    container isa AbstractDict || throw(SerializationError(
        "file '$path' does not contain a MacroEconometricModels model container"))
    return _reconstruct_from_container(container)
end
