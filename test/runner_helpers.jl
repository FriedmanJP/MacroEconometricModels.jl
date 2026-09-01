# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# Shared by test/runtests.jl (parent orchestrator) and test/core/test_internal_helpers.jl.

using LinearAlgebra

# Monotone expected-duration ranking (heaviest first) for the longest-first work queue (#124).
# Only the ordering matters, not accurate minutes.
function _expected_rank(name::AbstractString)
    name == "HA-DSGE"             && return 100
    name == "HA-DSGE Advanced"    && return 95
    name == "DSGE Core"           && return 90
    name == "DSGE Bayesian & HD"  && return 70
    name == "Extensions (JuMP/Ipopt/PATH)"    && return 60   # cold-load: schedule early
    startswith(name, "Coverage-A")            && return 60
    name == "ARIMA & Tests & Data & Reg"      && return 55
    name == "Plotting"            && return 52   # render + 11 lanes; schedule early to avoid a straggler
    name == "Serialization DSGE"  && return 88  # DSER is DSGE-Core-sized; start with HA
    name == "IRF & VECM"          && return 50
    name == "Serialization"       && return 48  # RSER files pulled out of empirical groups
    name == "Bayesian & SVAR"     && return 45
    name == "Display"             && return 42   # est-heavy compile; schedule with the medium wave
    startswith(name, "Coverage")  && return 20   # light coverage groups last
    name == "Counterfactual"      && return 10   # lightest group (CF-01); schedule last
    return 40                                     # default medium
end

# HA-DSGE and HA-DSGE Advanced are the suite ceiling. Give each 2 OpenBLAS
# threads; every other group stays at 1 so 4-wide dispatch does not oversubscribe.
# Only one 2-thread group so HA-DSGE + Advanced cannot oversubscribe (MSR-28).
_blas_threads_for_group(name::AbstractString) =
    name == "HA-DSGE" ? 2 : 1

_runner_max_conc(cpu_threads::Integer; cap::Integer=4) = min(Int(cpu_threads), cap)

# Do-block argument order: `_with_group_blas(name) do ... end` is
# `_with_group_blas(f, name)`. (Windows CI 31689698555: the reverse
# signature MethodError'd every group.)
function _with_group_blas(f, group_name::AbstractString)
    n = _blas_threads_for_group(group_name)
    old = BLAS.get_num_threads()
    try
        n != old && BLAS.set_num_threads(n)
        return f()
    finally
        n != old && BLAS.set_num_threads(old)
    end
end

# Ubuntu 1.10 Optim-v1 cell (`MACRO_NUMERICAL_CI=1`): skip display/
# plotting/coverage-harness groups. ubuntu LTS keeps the full list.
# Serialization groups are kept — they are their own 1.10 cell.
const _NUMERICAL_SKIP_GROUPS = Set(["Plotting", "Display", "Coverage-A", "Coverage-B"])
const _NUMERICAL_SKIP_CORE = Set([
    "core/test_aqua.jl",
    "core/test_display_backends.jl",
])

function _numerical_groups(groups, numerical::Bool)
    numerical || return groups
    out = Pair{String, Vector{String}}[]
    for (name, files) in groups
        name in _NUMERICAL_SKIP_GROUPS && continue
        fs = if name == "Coverage-C + IO"
            filter(f -> startswith(f, "io/"), files)
        elseif name == "Core & VAR"
            filter(f -> f ∉ _NUMERICAL_SKIP_CORE, files)
        else
            files
        end
        isempty(fs) && continue
        push!(out, String(name) => Vector{String}(fs))
    end
    return out
end

# CI job split (`MACRO_CI_SUITE=dsge|empirical|serialization`): DSGE/HA vs
# round-trip files vs the empirical rest. julia-actions/cache@v3 keys on the
# full matrix (include-matrix), so suite jobs do not share a compiled/ depot.
const _DSGE_SUITE_GROUPS = Set([
    "DSGE Core",
    "DSGE Bayesian & HD",
    "HA-DSGE",
    "HA-DSGE Advanced",
    "Coverage-A",
    "Extensions (JuMP/Ipopt/PATH)",
])
const _SERIALIZATION_SUITE_GROUPS = Set([
    "Serialization",
    "Serialization DSGE",
])

function _ci_suite_groups(groups, suite::AbstractString)
    isempty(suite) && return groups
    suite in ("dsge", "empirical", "serialization") || throw(ArgumentError(
        "MACRO_CI_SUITE must be \"dsge\", \"empirical\", \"serialization\", or empty; got $(repr(suite))"))
    out = Pair{String, Vector{String}}[]
    for (name, files) in groups
        in_dsge = name in _DSGE_SUITE_GROUPS
        in_ser = name in _SERIALIZATION_SUITE_GROUPS
        keep = if suite == "dsge"
            in_dsge
        elseif suite == "serialization"
            in_ser
        else
            !in_dsge && !in_ser
        end
        keep && push!(out, String(name) => Vector{String}(files))
    end
    return out
end

# TEST_GROUPS uses `"name" => files` Pairs, not Tuples. Type the channel from
# the collected vector so put! cannot MethodError (Windows CI, 2026-08-13).
function _make_work_queue(groups)
    queue = sort(collect(groups); by = p -> _expected_rank(first(p)), rev = true)
    work = Channel{eltype(queue)}(max(1, length(queue)))
    for item in queue
        put!(work, item)
    end
    close(work)
    return work
end
