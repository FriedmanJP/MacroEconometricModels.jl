# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# ─────────────────────────────────────────────────────────────────────────────
# Container assembly + validation
# ─────────────────────────────────────────────────────────────────────────────

_to_serializable(m) = _capture_fields(m)
function _to_serializable(c::FunctionConstraint)
    _assert_named_function_constraint(c)
    return _capture_fields(c)
end

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
    save_model(model, path; note="", compress=false) -> path
    save_model(objs::AbstractDict{String}, path; note="", compress=false) -> path
    save_model(objs::AbstractVector, path; note="", compress=false) -> path

Persist a fitted `model` — or a data container — to `path` in a versioned,
self-describing container. JLD2 is a package dependency; no extra `using JLD2`
is required. Coverage spans every VAR/BVAR (including mixed-frequency
and TVP), regression/3SLS, panel, volatility (including IGARCH), factor/Bayesian
FAVAR, ARIMA/SARIMA, ARDL/NARDL, STAR/Markov-switching, local-projection, and GMM
model, plus SVAR identification results, innovation-accounting objects (IRF,
FEVD, historical decomposition, LP-IRF/FEVD, Granger/stationarity/PVAR
diagnostics), forecasts (VAR/BVAR/VECM/ARIMA/LP/MIDAS/factor/volatility/threshold/
STAR/MS/conditional/nowcast) and forecast-evaluation results (DM/CW/MZ/encompassing/
combination), nowcast models (DFM/BVAR/bridge) and news decompositions,
cross-section/micro results (robust/penalized/Heckman/Tobit/count/quantile/RDD,
selection, marginal effects, diagnostics, Anderson–Rubin, wild cluster bootstrap,
panel tests), DiD results (TWFE/CS/SA/BJS/dCDH, event-study LP, LP-DiD, Bacon,
pre-trend, negative weights, HonestDiD), filter/spectral/nonparametric results
(HP/Hamilton/BN/BK/boosted-HP/X-13, ACF/periodogram/cross-spectrum, kernel
density/regression/LOWESS, data summary/diagnostics), test-statistic results
(unit-root/panel/cointegration/breaks/diagnostics, including Bai–Perron,
panel-unit-root summaries, and SADF/GSADF bubble tests), counterfactual/OPP
results (policy causal effects, rules/loss, OPP, model bank, path-floor and
named-function constraints), IO leftovers (`FootprintResult`, `IOMultipliers`,
`LinkageResult`, `IOExtension`), LP leftovers (Montiel Olea–Pflueger F, LP-IV AR
bands, B-spline basis, propensity-score config), GMM weighting/parameter
transforms, factor IC results (`HallinLiskaResult`, `BaiNgQResult`,
`AmengualWatsonResult`) and `IdentifiabilityTestResult`, Johansen/GPH/local-Whittle
companions, DSGE `ModelSpec` and
representative-agent solutions (`DSGESolution`, perturbation, projection,
OccBin, …), Bayesian DSGE results (`BayesianDSGE`, priors, state-space types),
HA results (`HASteadyState`, `HADSGESolution`, `KrusellSmithSolution`, …),
sequence-space blocks (`SSJModel`, `SimpleBlock`, `HetBlock`, `MitBlock`,
`SSJGEJacobian`, `SSJImpulseResponse`), DCEGM / firm / intermediary results,
OLG and continuous-time families, and the data containers
(`TimeSeriesData`, `PanelData`, `CrossSectionData`, `IOData`); the full set is
`MacroEconometricModels._SERIALIZABLE_TYPES`, tabulated on the
[API Reference](@ref api_page) Persistence section. Exported
concrete structs that are not saveable are listed in `_SERIALIZATION_EXCLUDED`
with a permanent reason (rendered HTML, workspaces, transient `reproduce`
reports, inline covariance-estimator configs, nested-only sentinels). The file records the
[`SERIALIZATION_FORMAT_VERSION`](@ref), the package and Julia versions, a
timestamp, an optional `note`, and — for a randomized result — its
reproducibility manifest. Only public fields are stored; cached factorizations
and state-space `H_inv` / `log_det_H` are dropped and recomputed on load.
DSGE `ModelSpec` residuals and `ss_fn` are recompiled from stored equations
on load.

`note=` is free-form header metadata (a label, a data vintage); read it back
with [`model_info`](@ref) — it is not reconstructed onto the model. Old files
without a `note` key still load. `compress=true` forwards to
`JLD2.jldopen(...; compress=true)` (CodecZlib). The default (`false`) writes
an uncompressed file. Posterior draws and dense Jacobians typically shrink;
a size table lives in the [Data Management](@ref data_page) persistence section.

A `Dict{String,<:Any}` of named objects, or a `Vector` of objects, is written
as a **bundle**: one file whose `"bundle" => true` header holds an `entries`
dict of per-object containers. [`load_model`](@ref) returns a `Dict{String,Any}`
(vector bundles are keyed `"1"`, `"2"`, …). An unregistered object raises
[`SerializationError`](@ref) naming the key before anything is written.

Household utilities, SSJ block functions (`SimpleBlock.f`, `MitBlock.evaluate`),
and programmatic `ModelSpec` residuals / `ss_fn` must be **named functions** or
callable structs ([`CRRAUtility`](@ref) and friends). Anonymous closures raise
[`SerializationError`](@ref) at save.

A loaded DSGE file recompiles stored equation expressions through `Core.eval`
behind an AST allowlist. This is the same class of risk as
`Serialization.deserialize`: only load files you trust.

```julia
m = estimate_var(Y, 2)
save_model(m, "model.jld2")
save_model(m, "model_z.jld2"; compress=true)
m2 = load_model("model.jld2")   # identical public fields

save_model(Dict("var" => m, "irf" => irf(m, 8)), "session.jld2"; note="vintage")
b = load_model("session.jld2")  # Dict{String,Any}
model_info("session.jld2")["note"] == "vintage"
```
"""
function save_model(model, path::AbstractString; note::AbstractString="", compress::Bool=false)
    container = _build_container(model)
    container["note"] = String(note)
    _write_model_container(String(path), container; compress=compress)
    return path
end

"""
    load_model(path) -> model

Reconstruct a model saved by [`save_model`](@ref). Validates the stored
`format_version` and type tag; raises a [`SerializationError`](@ref) naming the
expected-versus-found version on an unrecognized format, rather than returning a
corrupted object. A bundle file (`"bundle" => true`) returns a `Dict{String,Any}`
of reconstructed objects. Inspect the header without reconstructing with
[`model_info`](@ref).

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
    get(container, "bundle", false) === true && return _load_bundle(container)
    return _reconstruct_from_container(container)
end
