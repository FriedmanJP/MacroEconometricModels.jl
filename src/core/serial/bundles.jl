# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# ─────────────────────────────────────────────────────────────────────────────
# Bundles + header inspection (RSER-12 / #785)
# ─────────────────────────────────────────────────────────────────────────────
# A bundle is a format-1 container recognized by `"bundle" => true` (not a
# `type = "__bundle__"` tag). `entries` is a Dict of named per-object
# containers from `_build_container`. Vector saves are keyed `"1"`, `"2"`, …
# `SERIALIZATION_FORMAT_VERSION` stays 1: a pre-bundle v1 reader rejects a
# bundle with the existing "missing type tag" error.

function save_model(objs::AbstractDict{String,<:Any}, path::AbstractString;
                    note::AbstractString="", compress::Bool=false)
    entries = Dict{String,Any}()
    for (k, v) in objs
        try
            entries[k] = _build_container(v)
        catch e
            e isa SerializationError && throw(SerializationError("bundle key '$k': $(e.msg)"))
            rethrow()
        end
    end
    container = Dict{String,Any}(
        "format_version"  => SERIALIZATION_FORMAT_VERSION,
        "bundle"          => true,
        "note"            => String(note),
        "package_version" => _repro_package_version(),
        "julia_version"   => string(VERSION),
        "created"         => _repro_timestamp(),
        "entries"         => entries,
    )
    _write_model_container(String(path), container; compress=compress)
    return path
end

function save_model(objs::AbstractVector, path::AbstractString;
                    note::AbstractString="", compress::Bool=false)
    save_model(Dict(string(i) => objs[i] for i in eachindex(objs)), path;
               note=note, compress=compress)
end

function _load_bundle(container::AbstractDict)
    ver = get(container, "format_version", nothing)
    ver isa Integer || throw(SerializationError(
        "not a MacroEconometricModels model file: missing or non-integer format_version"))
    ver == SERIALIZATION_FORMAT_VERSION || throw(SerializationError(
        "unsupported serialization format_version $ver: this build reads version " *
        "$SERIALIZATION_FORMAT_VERSION. Re-save with the current release, or load with a " *
        "package version whose SERIALIZATION_FORMAT_VERSION == $ver."))
    src = get(container, "entries", nothing)
    src isa AbstractDict || throw(SerializationError("serialized bundle has no entries dict"))
    out = Dict{String,Any}()
    for (k, v) in src
        key = string(k)
        try
            v isa AbstractDict || throw(SerializationError("entry is not a model container"))
            out[key] = _reconstruct_from_container(v)
        catch e
            e isa SerializationError && throw(SerializationError("bundle key '$key': $(e.msg)"))
            rethrow()
        end
    end
    return out
end

const _MODEL_INFO_KEYS = ("format_version", "package_version", "julia_version",
                          "created", "type", "manifest")

function _container_header(c::AbstractDict; top::Bool=true)
    info = Dict{String,Any}()
    for k in _MODEL_INFO_KEYS
        haskey(c, k) && (info[k] = c[k])
    end
    info["note"] = String(get(c, "note", ""))
    top || return info
    is_bundle = get(c, "bundle", false) === true
    info["bundle"] = is_bundle
    is_bundle || return info
    src = get(c, "entries", nothing)
    entries = Dict{String,Any}()
    if src isa AbstractDict
        for (k, v) in src
            entries[string(k)] = v isa AbstractDict ? _container_header(v; top=false) :
                                 Dict{String,Any}()
        end
    end
    info["entries"] = entries
    return info
end

"""
    model_info(path) -> Dict{String,Any}

Read the metadata header of a file written by [`save_model`](@ref) without
reconstructing the payload. Returns `format_version`, `package_version`,
`julia_version`, `created`, `note`, `manifest`, and `bundle`. A single-object
file also has `type`; a bundle (`bundle === true`) has `entries`: a dict of
per-object headers (type, manifest, versions) with payloads omitted, so a
corrupt payload does not prevent inspection. Missing `note` on a pre-RSER-12
file is reported as `""`.
"""
function model_info(path::AbstractString)
    isfile(path) || throw(SerializationError("no such model file: $path"))
    c = _read_model_container(String(path))
    c isa AbstractDict || throw(SerializationError(
        "file '$path' does not contain a MacroEconometricModels model container"))
    return _container_header(c)
end
