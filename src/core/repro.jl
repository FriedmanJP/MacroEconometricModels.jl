# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# Reproducibility manifest (T246 / #345)
# =============================================================================
# Records HOW a randomized result was produced — the RNG seed, thread count,
# software versions, OS, a UTC timestamp, and the package git revision — so a
# published IRF, posterior, or bootstrap band can later be reproduced and
# audited. For institutional publication workflows (central banks, journals,
# regulators) this provenance is the difference between "a number" and "a number
# I can defend".
#
# The manifest is CHEAP and ALWAYS-ON: the session-constant environment (versions,
# threads, git SHA) is captured once and memoized, so `capture_manifest` costs a
# timestamp lookup after the first call and never slows an estimator. It NEVER
# throws — git discovery degrades to "unknown" outside a checkout.
#
# A seed is NOT recoverable from an `AbstractRNG`, so bit-for-bit reproduction
# requires the estimator to OWN the seed: pass `seed=N` to a randomized estimator
# (`estimate_bvar`, bootstrap `irf`) and it seeds a fresh `MersenneTwister(N)`,
# records `N` in the manifest, and `reproduce(result)` reconstructs it.
#
# This file is included EARLY (before the result-type definitions) because
# `ReproManifest` is a field type of `ImpulseResponse` and `BVARPosterior`. The
# per-type `reproduce` methods live next to their estimators (`core/irf.jl`,
# `bvar/estimation.jl`), where both the type and the estimator are in scope.

"""
    ReproManifest

Provenance record attached to a randomized result. Captures everything needed to
argue a published number is reproducible.

Fields:
- `seed::Union{Int,Nothing}` — canonical RNG seed actually used (`nothing` when
  the caller passed an `rng` directly rather than a `seed`, in which case the
  computation is not bit-for-bit reproducible).
- `n_threads::Int` — `Threads.nthreads()` at compute time (results can be
  thread-count-dependent even at a fixed seed).
- `julia_version::String`, `package_version::String` — the Julia and
  MacroEconometricModels versions.
- `dependency_versions::Dict{String,String}` — versions of key statistical
  dependencies (Distributions, StatsAPI, …).
- `os::String`, `machine::String` — `Sys.KERNEL` and `Sys.MACHINE`.
- `timestamp::String` — UTC ISO-8601 capture time.
- `git_sha::String`, `git_dirty::Bool` — the package's git commit and whether the
  working tree had uncommitted changes (`"unknown"`/`false` outside a checkout).
- `settings::Dict{String,Any}` — extra call parameters needed to re-run
  (e.g. `burnin`/`thin` for a Gibbs BVAR, `reps`/`method` for a bootstrap IRF).

Construct with [`capture_manifest`](@ref); reproduce with [`reproduce`](@ref).
"""
struct ReproManifest
    seed::Union{Int,Nothing}
    n_threads::Int
    julia_version::String
    package_version::String
    dependency_versions::Dict{String,String}
    os::String
    machine::String
    timestamp::String
    git_sha::String
    git_dirty::Bool
    settings::Dict{String,Any}
end

# ─────────────────────────────────────────────────────────────────────────────
# Environment capture (memoized — session-constant, so an estimator pays for it once)
# ─────────────────────────────────────────────────────────────────────────────

_repro_verstr(m::Module) = (v = try Base.pkgversion(m) catch; nothing end;
                            v === nothing ? "unknown" : string(v))

function _repro_package_version()
    v = try Base.pkgversion(@__MODULE__) catch; nothing end
    return v === nothing ? "unknown" : string(v)
end

# Versions of the key statistical dependencies. All modules referenced here are
# `using`/`import`ed at the top of the package module, so the bindings resolve.
function _repro_dependency_versions()
    return Dict{String,String}(
        "Distributions" => _repro_verstr(Distributions),
        "StatsAPI"      => _repro_verstr(StatsAPI),
        "DataFrames"    => _repro_verstr(DataFrames),
        "FFTW"          => _repro_verstr(FFTW),
        "ForwardDiff"   => _repro_verstr(ForwardDiff),
        "Optim"         => _repro_verstr(Optim),
        "Tables"        => _repro_verstr(Tables),
    )
end

# Package git revision + dirty flag, via `git -C <pkgdir>`. Returns
# ("unknown", false) outside a checkout; never throws. stderr is discarded so a
# non-git directory produces no console noise.
function _compute_git_info()
    dir = try pkgdir(@__MODULE__) catch; nothing end
    (dir === nothing || !isdir(dir)) && return ("unknown", false)
    sha = try
        strip(read(pipeline(`git -C $dir rev-parse HEAD`; stderr=devnull), String))
    catch
        return ("unknown", false)
    end
    isempty(sha) && return ("unknown", false)
    dirty = try
        !isempty(strip(read(pipeline(`git -C $dir status --porcelain`; stderr=devnull), String)))
    catch
        false
    end
    return (String(sha), dirty)
end

# Session-constant environment (versions/os/git — NOT the thread count, which is a
# runtime property). Memoized once at runtime so `capture_manifest` stays cheap.
function _compute_static_env()
    sha, dirty = _compute_git_info()
    return (julia_version = string(VERSION),
            package_version = _repro_package_version(),
            dependency_versions = _repro_dependency_versions(),
            os = string(Sys.KERNEL),
            machine = string(Sys.MACHINE),
            git_sha = sha,
            git_dirty = dirty)
end

const _REPRO_ENV_CACHE = Ref{Any}(nothing)
function _repro_static_env()
    # Never memoize during precompilation — a value baked into the sysimage would
    # freeze the build-time git/version state. Recompute (uncached) there.
    ccall(:jl_generating_output, Cint, ()) == 1 && return _compute_static_env()
    c = _REPRO_ENV_CACHE[]
    c === nothing || return c
    env = _compute_static_env()
    _REPRO_ENV_CACHE[] = env
    return env
end

_repro_timestamp() = try
    Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-dd\THH:MM:SS\Z")
catch
    "unknown"
end

"""
    capture_manifest(; seed=nothing, settings=Dict{String,Any}()) -> ReproManifest

Capture a [`ReproManifest`](@ref) for the current environment. `seed` is the
canonical RNG seed the caller used (pass the same integer given to the
estimator's `seed=` kwarg); `settings` holds any extra call parameters needed to
re-run. The environment portion is memoized, so this is effectively free after
the first call and safe to invoke on every randomized estimate. Never throws.
"""
function capture_manifest(; seed::Union{Integer,Nothing}=nothing,
                            settings::AbstractDict=Dict{String,Any}())
    env = _repro_static_env()
    return ReproManifest(
        seed === nothing ? nothing : Int(seed),
        Threads.nthreads(),                # runtime thread count (never cached)
        env.julia_version,
        env.package_version,
        env.dependency_versions,
        env.os,
        env.machine,
        _repro_timestamp(),
        env.git_sha,
        env.git_dirty,
        Dict{String,Any}(settings),
    )
end

"""
    _resolve_repro_rng(rng, seed) -> AbstractRNG

If `seed` is given, return a fresh `MersenneTwister(seed)` (so the draw stream is
reproducible from the recorded seed); otherwise return `rng` unchanged. This is
the single seed-injection point every randomized estimator routes through.
"""
_resolve_repro_rng(rng, seed::Integer) = Random.MersenneTwister(seed)
_resolve_repro_rng(rng, ::Nothing) = rng

# ─────────────────────────────────────────────────────────────────────────────
# Manifest ↔ Dict (plain primitives only — for versioned serialization, #347)
# ─────────────────────────────────────────────────────────────────────────────

function _manifest_to_dict(m::ReproManifest)
    return Dict{String,Any}(
        "__manifest__"        => true,
        "seed"                => m.seed,
        "n_threads"           => m.n_threads,
        "julia_version"       => m.julia_version,
        "package_version"     => m.package_version,
        "dependency_versions" => Dict{String,Any}(m.dependency_versions),
        "os"                  => m.os,
        "machine"             => m.machine,
        "timestamp"           => m.timestamp,
        "git_sha"             => m.git_sha,
        "git_dirty"           => m.git_dirty,
        "settings"            => Dict{String,Any}(m.settings),
    )
end
_manifest_to_dict(::Nothing) = nothing

function _manifest_from_dict(d::AbstractDict)
    depv = Dict{String,String}()
    for (k, v) in get(d, "dependency_versions", Dict{String,Any}())
        depv[String(k)] = String(v)
    end
    seed = get(d, "seed", nothing)
    return ReproManifest(
        seed === nothing ? nothing : Int(seed),
        Int(get(d, "n_threads", 0)),
        String(get(d, "julia_version", "unknown")),
        String(get(d, "package_version", "unknown")),
        depv,
        String(get(d, "os", "unknown")),
        String(get(d, "machine", "unknown")),
        String(get(d, "timestamp", "unknown")),
        String(get(d, "git_sha", "unknown")),
        Bool(get(d, "git_dirty", false)),
        Dict{String,Any}(get(d, "settings", Dict{String,Any}())),
    )
end
_manifest_from_dict(::Nothing) = nothing

# ─────────────────────────────────────────────────────────────────────────────
# reproduce() — re-run a randomized result and compare bit-for-bit
# ─────────────────────────────────────────────────────────────────────────────

"""Per-field comparison of an original result field against its recomputation."""
struct ReproFieldDiff
    name::String
    matched::Bool
    max_abs_diff::Float64   # NaN for non-numeric / shape-mismatched fields
end

"""
    ReproReport

Result of [`reproduce`](@ref). `matched` is `true`/`false` when a recomputation
ran, or `missing` when reproduction could not be attempted (no manifest, no
recorded seed, or a source object was required). `fields` holds the per-field
comparison; `note` explains a mismatch or a thread-count caveat.
"""
struct ReproReport
    matched::Union{Bool,Missing}
    fields::Vector{ReproFieldDiff}
    seed::Union{Int,Nothing}
    threads_captured::Int
    threads_current::Int
    note::String
end

function _repro_field_diff(name::AbstractString, a, b)
    if a isa AbstractArray && b isa AbstractArray && size(a) == size(b)
        d = isempty(a) ? 0.0 : Float64(maximum(abs.(Float64.(a) .- Float64.(b))))
        return ReproFieldDiff(String(name), isequal(a, b), d)
    end
    return ReproFieldDiff(String(name), isequal(a, b), NaN)
end

# Finalize a report from per-field diffs, attaching the thread-count caveat the
# acceptance criteria require (mismatch under a changed thread count is flagged,
# not silently passed).
function _finalize_repro(diffs::Vector{ReproFieldDiff}, m::ReproManifest)
    matched = all(d.matched for d in diffs)
    tc = Threads.nthreads()
    note = if !matched && tc != m.n_threads
        "output differs AND the thread count changed ($(m.n_threads) → $tc); if this " *
        "algorithm is thread-count-sensitive that likely explains the mismatch — re-run " *
        "with JULIA_NUM_THREADS=$(m.n_threads)."
    elseif !matched
        "output differs at the same thread count ($tc); the code or a dependency version " *
        "may have changed since capture."
    elseif tc != m.n_threads
        "matched despite a thread-count change ($(m.n_threads) → $tc): this computation is " *
        "thread-count-invariant."
    else
        "matched bit-for-bit."
    end
    return ReproReport(matched, diffs, m.seed, m.n_threads, tc, note)
end

_no_manifest_report(what::AbstractString) =
    ReproReport(missing, ReproFieldDiff[], nothing, 0, Threads.nthreads(),
        "$what carries no reproducibility manifest; re-run the estimator with a `seed=` " *
        "kwarg to record one.")

_no_seed_report(m::ReproManifest, howto::AbstractString) =
    ReproReport(missing, ReproFieldDiff[], nothing, m.n_threads, Threads.nthreads(),
        "manifest has no recorded seed (an `rng` was passed instead of a `seed`); re-run " *
        "as `$howto` to enable bit-for-bit reproduction.")

_needs_source_report(what::AbstractString, howto::AbstractString) =
    ReproReport(missing, ReproFieldDiff[], nothing, 0, Threads.nthreads(),
        "reproducing a $what needs its source model, which is not retained on the result; " *
        "call `$howto`.")

"""
    reproduce(result) -> ReproReport

Re-run the randomized computation that produced `result` from its stored
reproducibility manifest (same seed) and check the output matches bit-for-bit.
This is the "did my published number actually come from this code" check.

Supported: [`BVARPosterior`](@ref) (self-contained), and a bootstrap
[`ImpulseResponse`](@ref) via the two-argument `reproduce(ir, model)` (the source
`VARModel` is not retained on the IRF result). Returns a [`ReproReport`](@ref);
`matched` is `missing` when reproduction cannot be attempted.
"""
reproduce(x) = ReproReport(missing, ReproFieldDiff[], nothing, 0, Threads.nthreads(),
    "reproduce() is not implemented for $(typeof(x)); supported: BVARPosterior and " *
    "bootstrap ImpulseResponse (reproduce(ir, model)).")

# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

"""Append a one-line reproducibility footer to a result's `show`/`report`. Plain
text (not a table) so it cannot inherit PrettyTables horizontal-crop truncation,
mirroring `_sig_legend`. A no-op when no manifest is attached."""
function _manifest_footer(io::IO, m::ReproManifest)
    seedstr = m.seed === nothing ? "unset" : string(m.seed)
    shastr = m.git_sha == "unknown" ? "unknown" :
             first(m.git_sha, 8) * (m.git_dirty ? "+dirty" : "")
    println(io, "Reproducibility: seed=", seedstr, ", threads=", m.n_threads,
                ", pkg v", m.package_version, ", julia ", m.julia_version, ", git ", shastr)
end
_manifest_footer(::IO, ::Nothing) = nothing

function Base.show(io::IO, m::ReproManifest)
    println(io, "ReproManifest")
    println(io, "  seed         : ", m.seed === nothing ? "unset" : m.seed)
    println(io, "  threads      : ", m.n_threads)
    println(io, "  julia        : ", m.julia_version)
    println(io, "  package      : v", m.package_version)
    println(io, "  git          : ", m.git_sha, m.git_dirty ? " (dirty)" : "")
    println(io, "  os / machine : ", m.os, " / ", m.machine)
    print(io,   "  captured     : ", m.timestamp)
end

function Base.show(io::IO, r::ReproReport)
    status = r.matched === missing ? "N/A" : (r.matched ? "PASS" : "FAIL")
    println(io, "ReproReport: ", status)
    for d in r.fields
        tail = isnan(d.max_abs_diff) ? "" : "  (max|Δ|=" * string(d.max_abs_diff) * ")"
        println(io, "  ", d.matched ? "✓" : "✗", " ", d.name, tail)
    end
    if r.threads_captured > 0
        println(io, "  threads: captured=", r.threads_captured, " current=", r.threads_current)
    end
    print(io, "  ", r.note)
end
