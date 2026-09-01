# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using MacroEconometricModels, Test, Random, LinearAlgebra, DataFrames, Statistics
const _MEM = MacroEconometricModels
_roundtrip(m) = _MEM._reconstruct_from_container(_MEM._build_container(m))

# Recursive, NaN-aware structural equality over public fields — recurses into
# MacroEconometricModels structs, arrays, dicts, and tuples so a round-tripped
# model can be compared field-by-field even when a field is a nested struct.
function _deep_equal(a, b)
    a === nothing && return b === nothing
    a isa Missing && return b isa Missing
    if a isa Number
        return (isnan(a) && b isa Number && isnan(b)) || isequal(a, b)
    end
    (a isa AbstractString || a isa Symbol || a isa Enum || a isa Bool) && return isequal(a, b)
    # UnitRange (e.g. PathFloorConstraint.horizons = 1:typemax(Int)) is an
    # AbstractArray; comparing elementwise would not terminate.
    a isa AbstractRange && return isequal(a, b)
    if a isa AbstractArray
        b isa AbstractArray || return false
        size(a) == size(b) || return false
        return all(_deep_equal(a[i], b[i]) for i in eachindex(a))
    end
    if a isa AbstractDict
        b isa AbstractDict || return false
        Set(keys(a)) == Set(keys(b)) || return false
        return all(_deep_equal(a[k], b[k]) for k in keys(a))
    end
    a isa Tuple && return length(a) == length(b) && all(_deep_equal(a[i], b[i]) for i in eachindex(a))
    if isstructtype(typeof(a)) && parentmodule(typeof(a)) === _MEM
        typeof(a).name === typeof(b).name || return false
        return all(_deep_equal(getfield(a, f), getfield(b, f)) for f in fieldnames(typeof(a)))
    end
    return isequal(a, b)
end

# True when `_from_serializable` for `T` is the generic `where {T}` method,
# not an explicit override (those bind `Type{Concrete}` and are not UnionAll).
_from_serializable_is_generic(::Type{T}) where {T} =
    which(_MEM._from_serializable, Tuple{Type{T}, AbstractDict, Int}).sig isa UnionAll

# Keyword-constructor detector: a registered type with only the generic
# `_from_serializable` must rebuild from positional field values.
function _check_generic_construct(m)
    T = typeof(m)
    Tw = Base.typename(T).wrapper
    haskey(_MEM._SERIALIZABLE_TYPES, string(nameof(T))) || return
    _from_serializable_is_generic(Tw) || return
    args = Any[getfield(m, i) for i in 1:nfields(m)]
    rebuilt = _MEM._generic_construct(Tw, args)
    @test typeof(rebuilt).name === T.name
end

# Round-trip `m` through the container and assert every public field (minus any
# in `skip`, e.g. an intentionally-dropped closure) survives structurally intact.
# `report` text is compared when no fields are skipped. `plot_result` lives on
# `_assert_consumers` / `_assert_plot_equal` so module files are not double-plotted.
function _assert_roundtrip(m; skip::Vector{Symbol}=Symbol[])
    _check_generic_construct(m)
    m2 = _roundtrip(m)
    @test typeof(m2).name === typeof(m).name
    for f in fieldnames(typeof(m))
        f in skip && continue
        @test _deep_equal(getfield(m, f), getfield(m2, f))
    end
    isempty(skip) && _assert_report_equal(m, m2)
    return m2
end

# Prefer 2-arg `report(io, x)` when it exists; many result types only define
# `report(x) = show(stdout, x)`, which `sprint(report, x)` cannot call.
function _assert_report_equal(a, b)
    text(x) = applicable(report, IOBuffer(), x) ? sprint(report, x) : sprint(show, x)
    @test text(a) == text(b)
    a
end
function _assert_plot_equal(a, b)
    # Plot IDs come from a process-wide counter; rewind so two identical objects
    # produce identical HTML rather than `irf_N` vs `irf_N+1`.
    c0 = _MEM._plot_counter[]
    pa = plot_result(a)
    _MEM._plot_counter[] = c0
    pb = plot_result(b)
    @test typeof(pa) === typeof(pb)
    for f in fieldnames(typeof(pa))
        @test _deep_equal(getfield(pa, f), getfield(pb, f))
    end
    pa
end
function _assert_tables_equal(a, b)
    if applicable(long_table, a)
        @test isequal(long_table(a), long_table(b))
    else
        @test isequal(DataFrame(a), DataFrame(b))
    end
end
function _assert_refs_equal(a, b)
    applicable(refs, IOBuffer(), a) || return nothing
    @test sprint(io -> refs(io, a)) == sprint(io -> refs(io, b))
    nothing
end
function _assert_forecast_eval(a, b)
    applicable(point_forecast, a) || return nothing
    pf_a = point_forecast(a)
    pf_b = point_forecast(b)
    col_a = pf_a isa AbstractMatrix ? pf_a[:, 1] : collect(vec(pf_a))
    col_b = pf_b isa AbstractMatrix ? pf_b[:, 1] : collect(vec(pf_b))
    length(col_a) < 2 && return nothing
    actual = col_a .+ 0.1
    @test isequal(forecast_evaluate(actual, col_a).values,
                  forecast_evaluate(actual, col_b).values)
    nothing
end
function _assert_consumers(a, b)
    _assert_report_equal(a, b)
    applicable(plot_result, a) && _assert_plot_equal(a, b)
    applicable(long_table, a) && _assert_tables_equal(a, b)
    _assert_refs_equal(a, b)
    _assert_forecast_eval(a, b)
    b
end

# First non-IO argument type of a `report` / `plot_result` method, unwrapping
# `UnionAll` and skipping `Any` so a fallback `report(::Any)` does not match
# every registered type.
function _dispatch_argtype(m::Method)
    sig = Base.unwrap_unionall(m.sig)
    sig isa DataType || return nothing
    ps = sig.parameters
    length(ps) >= 2 || return nothing
    for i in 2:length(ps)
        T = ps[i]
        T isa TypeVar && (T = T.ub)
        T isa Type || continue
        while T isa UnionAll
            T = T.body
        end
        T isa DataType || continue
        T <: IO && continue
        T === Any && continue
        T <: Function && continue
        return T
    end
    return nothing
end

function _registered_dispatch_names(f::Function)
    names = Set{String}()
    for m in methods(f)
        T = _dispatch_argtype(m)
        T === nothing && continue
        if isabstracttype(T)
            for (name, S) in _MEM._SERIALIZABLE_TYPES
                Sw = S
                while Sw isa UnionAll
                    Sw = Sw.body
                end
                try
                    Sw <: T && push!(names, name)
                catch
                end
            end
        else
            n = string(nameof(T))
            haskey(_MEM._SERIALIZABLE_TYPES, n) && push!(names, n)
        end
    end
    names
end
