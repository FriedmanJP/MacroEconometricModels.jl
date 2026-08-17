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
    name == "IRF & VECM"          && return 50
    name == "Bayesian & SVAR"     && return 45
    name == "Display"             && return 42   # est-heavy compile; schedule with the medium wave
    startswith(name, "Coverage")  && return 20   # light coverage groups last
    name == "Counterfactual"      && return 10   # lightest group (CF-01); schedule last
    return 40                                     # default medium
end

# HA-DSGE and HA-DSGE Advanced are the suite ceiling. Give each 2 OpenBLAS
# threads; every other group stays at 1 so 4-wide dispatch does not oversubscribe.
_blas_threads_for_group(name::AbstractString) =
    name == "HA-DSGE" || name == "HA-DSGE Advanced" ? 2 : 1

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
