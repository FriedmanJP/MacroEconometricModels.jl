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
    try
        JLD2.jldopen(path, "w"; compress=compress) do f
            f["container"] = container
        end
    catch e
        compress || rethrow()
        msg = sprint(showerror, e)
        occursin(r"compress|CodecZlib|Filter"i, msg) || rethrow()
        throw(SerializationError(
            "save_model(...; compress=true) failed ($msg). JLD2 compress= uses CodecZlib; " *
            "check that a JLD2 0.4–0.6 build with CodecZlib is loaded."))
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
    save_model(model, path; compress::Bool=false) -> path

Persist a fitted `model` — or a data container — to `path` in a versioned,
self-describing container. JLD2 is a package dependency; no extra `using JLD2`
is required.

Coverage spans every VAR/regression/panel/volatility/factor/ARIMA/local-projection/GMM
model, SVAR identification results, DSGE `ModelSpec` and representative-agent
solutions (`DSGESolution`, perturbation, projection, OccBin, …), Bayesian DSGE
results (`BayesianDSGE`, priors, state-space types), HA results (`HASteadyState`,
`HADSGESolution`, `KrusellSmithSolution`, …), sequence-space blocks (`SSJModel`,
`SimpleBlock`, `HetBlock`, `MitBlock`, `SSJGEJacobian`, `SSJImpulseResponse`),
DCEGM / firm / intermediary results, OLG and continuous-time families, and the
data containers (`TimeSeriesData`, `PanelData`, `CrossSectionData`, `IOData`);
the full set is `MacroEconometricModels._SERIALIZABLE_TYPES`. The file records
the [`SERIALIZATION_FORMAT_VERSION`](@ref), the package and Julia versions, a
timestamp, and — for a randomized result — its reproducibility manifest. Only
public fields are stored; cached factorizations and state-space `H_inv` /
`log_det_H` are dropped and recomputed on load. DSGE `ModelSpec` residuals and
`ss_fn` are recompiled from stored equations on load.

Household utilities, SSJ block functions (`SimpleBlock.f`, `MitBlock.evaluate`),
and programmatic `ModelSpec` residuals / `ss_fn` must be **named functions** or
callable structs ([`CRRAUtility`](@ref) and friends). Anonymous closures raise
[`SerializationError`](@ref) at save.

`compress=true` forwards to `JLD2.jldopen(...; compress=true)` (CodecZlib).
The default (`false`) writes an uncompressed file, byte-identical to previous
releases. `load_model` reads both transparently. Posterior draws and dense
Jacobians typically shrink; a size table lives in the
[Data Management](@ref data_page) persistence section.

A loaded DSGE file recompiles stored equation expressions through `Core.eval`
behind an AST allowlist. This is the same class of risk as
`Serialization.deserialize`: only load files you trust.

```julia
m = estimate_var(Y, 2)
save_model(m, "model.jld2")
save_model(m, "model_z.jld2"; compress=true)
m2 = load_model("model.jld2")   # identical public fields
```
"""
function save_model(model, path::AbstractString; compress::Bool=false)
    container = _build_container(model)
    _write_model_container(String(path), container; compress=compress)
    return path
end

"""
    load_model(path) -> model

Reconstruct a model saved by [`save_model`](@ref). Validates the stored
`format_version` and type tag; raises a [`SerializationError`](@ref) naming the
expected-versus-found version on an unrecognized format, rather than returning a
corrupted object.

A DSGE / HA file's equations are recompiled at load with an AST allowlist, but
as with `Serialization.deserialize`, only load files you trust. Named functions
stored in the payload must be defined in `Main` or `MacroEconometricModels` in
the loading session; [`CRRAUtility`](@ref) callables reconstruct without that.
"""
function load_model(path::AbstractString)
    isfile(path) || throw(SerializationError("no such model file: $path"))
    container = _read_model_container(String(path))
    container isa AbstractDict || throw(SerializationError(
        "file '$path' does not contain a MacroEconometricModels model container"))
    return _reconstruct_from_container(container)
end
