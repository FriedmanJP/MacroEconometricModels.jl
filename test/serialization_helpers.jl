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
function _assert_roundtrip(m; skip::Vector{Symbol}=Symbol[])
    _check_generic_construct(m)
    m2 = _roundtrip(m)
    @test typeof(m2).name === typeof(m).name
    for f in fieldnames(typeof(m))
        f in skip && continue
        @test _deep_equal(getfield(m, f), getfield(m2, f))
    end
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
    pa, pb = plot_result(a), plot_result(b)
    @test typeof(pa) === typeof(pb)
    for f in fieldnames(typeof(pa))
        @test _deep_equal(getfield(pa, f), getfield(pb, f))
    end
    pa
end
function _assert_tables_equal(a, b)
    @test DataFrame(a) == DataFrame(b)
end
