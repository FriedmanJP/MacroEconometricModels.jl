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
    # Recompiled residual closures are new function objects; treat any Function
    # pair as matching. Expr codecs drop line numbers.
    a isa Function && return b isa Function
    if a isa Expr
        b isa Expr || return false
        return Base.remove_linenums!(deepcopy(a)) == Base.remove_linenums!(deepcopy(b))
    end
    if a isa Number
        b isa Number || return false
        (isnan(a) && isnan(b)) && return true
        return a == b  # -0.0 == 0.0 after JLD2/container round-trip
    end
    (a isa AbstractString || a isa Symbol || a isa Enum || a isa Bool) && return isequal(a, b)
    if a isa NamedTuple
        b isa NamedTuple || return false
        keys(a) === keys(b) || keys(a) == keys(b) || return false
        return all(_deep_equal(a[k], b[k]) for k in keys(a))
    end
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

# Types that actually went through `_assert_report_equal` / `_assert_plot_equal`
# in this process (kernel coverage + any included module file).
const _REPORT_COVERED = Set{String}()
const _PLOT_COVERED = Set{String}()

_record_helper!(set, x) = (push!(set, string(nameof(typeof(x)))); x)

# Round-trip `m` through the container and assert every public field (minus any
# in `skip`, e.g. an intentionally-dropped closure) survives structurally intact.
# `report` / `plot_result` live on `_assert_consumers` so a `_assert_roundtrip`
# + `_assert_consumers` pair is not double-reported or double-plotted.
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

# Structural round-trip plus consumer equality (`report` always; `plot_result`
# when a dispatch exists). Use this in kernel coverage in place of a bare
# `_assert_roundtrip(estimate_*(...))`.
function _cover(m; skip::Vector{Symbol}=Symbol[])
    m2 = _assert_roundtrip(m; skip=skip)
    if isempty(skip)
        _assert_consumers(m, m2)
    else
        applicable(plot_result, m) && _assert_plot_equal(m, m2)
    end
    return m2
end

# Prefer 2-arg `report(io, x)` when it exists; many result types only define
# `report(x) = show(stdout, x)`, which `sprint(report, x)` cannot call.
function _assert_report_equal(a, b)
    _record_helper!(_REPORT_COVERED, a)
    text(x) = applicable(report, IOBuffer(), x) ? sprint(report, x) : sprint(show, x)
    # Pin :text: Coverage/Display groups call set_display_backend(:latex) on the
    # process default, and the threaded runner shares that Ref across groups.
    _MEM.with_display_backend(:text) do
        @test text(a) == text(b)
    end
    a
end
# Render two plots with the same element IDs (counter rewind under the plot-id lock).
function _paired_plot_result(fa::Function, fb::Function)
    _MEM._with_plot_id_lock() do
        c0 = _MEM._plot_counter[]
        pa = fa()
        _MEM._plot_counter[] = c0
        pb = fb()
        pa, pb
    end
end
function _assert_plot_equal(a, b)
    _record_helper!(_PLOT_COVERED, a)
    pa, pb = _paired_plot_result(() -> plot_result(a), () -> plot_result(b))
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
    ra = try
        sprint(io -> refs(io, a))
    catch
        return nothing
    end
    @test sprint(io -> refs(io, b)) == ra
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
    fast = get(ENV, "MACRO_FAST_TESTS", "") == "1"
    if !fast
        applicable(plot_result, a) && _assert_plot_equal(a, b)
        applicable(long_table, a) && _assert_tables_equal(a, b)
        _assert_refs_equal(a, b)
        _assert_forecast_eval(a, b)
    end
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

# DSER types live in test/dsge/test_dsge_serialization.jl (own harness).
function _dser_coverage_skip()
    src = read(joinpath(dirname(pathof(MacroEconometricModels)), "core", "serial", "registry.jl"), String)
    r = findfirst("# ── DSGE / HA / OLG / CT", src)
    r === nothing && return Set{String}()
    tail = src[first(r):end]
    c = findfirst(")\n", tail)
    block = c === nothing ? tail : tail[1:first(c)]
    Set(String(m.captures[1]) for m in eachmatch(r"\"([A-Za-z0-9]+)\"\s*=>", block))
end

function _ser_test_paths()
    root = @__DIR__
    files = String[]
    for (dir, _, fnames) in walkdir(root)
        basename(dir) == "dsge" && continue
        for f in fnames
            (endswith(f, "_serialization.jl") || f == "test_serialization.jl") || continue
            push!(files, joinpath(dir, f))
        end
    end
    files
end

function _testset_blocks(txt::AbstractString)
    lines = split(txt, '\n')
    idxs = Int[i for (i, ln) in enumerate(lines) if occursin(r"^\s*@testset", ln)]
    isempty(idxs) && return String[txt]
    out = String[]
    idxs[1] > 1 && push!(out, join(lines[1:idxs[1]-1], '\n'))
    for k in eachindex(idxs)
        a = idxs[k]
        b = k < length(idxs) ? idxs[k+1] - 1 : length(lines)
        push!(out, join(lines[a:b], '\n'))
    end
    out
end

function _types_in_block(blk::AbstractString, registered::Set{String})
    found = Set{String}()
    for name in registered
        occursin(Regex("\"" * name * "\"\\s*=>"), blk) && push!(found, name)
        occursin("_from_serializable_is_generic(" * name, blk) && push!(found, name)
        occursin(Regex("\\bisa\\s+" * name * "\\b"), blk) && push!(found, name)
        occursin(Regex("\\b" * name * "\\{"), blk) && push!(found, name)
        occursin(Regex("\\b" * name * "\\("), blk) && push!(found, name)
        occursin(name, blk) && occursin("@testset", blk) && push!(found, name)
    end
    found
end

# Longest-match first so `estimate_garch_midas` is not tagged as GARCHModel.
const _COVER_ESTIMATOR_TYPES = [
    "estimate_garch_midas" => "GarchMidasModel",
    "estimate_gjr_garch" => "GJRGARCHModel",
    "estimate_fiegarch" => "FIEGARCHModel",
    "estimate_figarch" => "FIGARCHModel",
    "estimate_dynamic_factors" => "DynamicFactorModel",
    "estimate_structural_dfm" => "StructuralDFM",
    "estimate_xtcointreg" => "PanelCointRegModel",
    "estimate_cointreg" => "CointRegModel",
    "estimate_factors" => "FactorModel",
    "estimate_gdfm" => "GeneralizedDynamicFactorModel",
    "estimate_favar" => "FAVARModel",
    "estimate_egarch" => "EGARCHModel",
    "estimate_aparch" => "APARCHModel",
    "estimate_cgarch" => "CGARCHModel",
    "estimate_arch" => "ARCHModel",
    "estimate_garch" => "GARCHModel",
    "estimate_dcc" => "MGARCHModel",
    "estimate_sv" => "SVModel",
    "estimate_arfima" => "ARFIMAModel",
    "estimate_arima" => "ARIMAModel",
    "estimate_arma" => "ARMAModel",
    "estimate_ardl" => "ARDLModel",
    "estimate_nardl" => "NARDLModel",
    "estimate_threshold" => "ThresholdModel",
    "estimate_midas" => "MidasModel",
    "estimate_pmg" => "PMGModel",
    "estimate_ologit" => "OrderedLogitModel",
    "estimate_oprobit" => "OrderedProbitModel",
    "estimate_mlogit" => "MultinomialLogitModel",
    "estimate_lp_iv" => "LPIVModel",
    "estimate_smooth_lp" => "SmoothLPModel",
    "estimate_state_lp" => "StateLPModel",
    "estimate_propensity_lp" => "PropensityLPModel",
    "estimate_sur" => "SURModel",
    "estimate_gmm" => "GMMModel",
    "estimate_smm" => "SMMModel",
    "estimate_xtreg" => "PanelRegModel",
    "estimate_xtiv" => "PanelIVModel",
    "estimate_xtlogit" => "PanelLogitModel",
    "estimate_xtprobit" => "PanelProbitModel",
    "estimate_vecm" => "VECMModel",
    "estimate_ar(" => "ARModel",
    "estimate_ma(" => "MAModel",
    "estimate_statespace" => "StateSpaceModel",
    "estimate_logit" => "LogitModel",
    "estimate_probit" => "ProbitModel",
    "estimate_var" => "VARModel",
    "estimate_bvar" => "BVARPosterior",
    "estimate_reg" => "RegModel",
    "estimate_lp(" => "LPModel",
    "estimate_pvar" => "PVARModel",
    "identify_svec" => "SVECResult",
    "estimate_svar" => "SVARModel",
]

const _SID24_TYPES = (
    "ICASVARResult", "NonGaussianMLResult", "NonGaussianGMMResult",
    "MarkovSwitchingSVARResult", "GARCHSVARResult", "SmoothTransitionSVARResult",
    "ExternalVolatilitySVARResult", "ProxySVARResult", "MaxShareResult",
    "AriasSVARResult", "UhligSVARResult", "BayesianSetIdentifiedSVAR",
    "SignIdentifiedSet", "RobustBayesResult",
)

function _scan_ser_helper_coverage()
    registered = Set(keys(_MEM._SERIALIZABLE_TYPES))
    report_cov = Set{String}()
    plot_cov = Set{String}()
    for path in _ser_test_paths()
        txt = read(path, String)
        has_report = occursin("_assert_report_equal", txt) ||
                     occursin("_assert_consumers", txt) || occursin("_cover(", txt)
        has_plot = occursin("_assert_plot_equal", txt) ||
                   occursin("_assert_consumers", txt) || occursin("_cover(", txt)
        if occursin(r"for \(name, r\) in fixtures", txt)
            for m in eachmatch(r"\"([A-Z][A-Za-z0-9]+)\"\s*=>", txt)
                n = m.captures[1]
                n in registered || continue
                has_report && push!(report_cov, n)
                has_plot && push!(plot_cov, n)
            end
            for m in eachmatch(r"const _RSER[^\n]*=\s*\((.*?)\)"s, txt)
                for q in eachmatch(r"\"([A-Z][A-Za-z0-9]+)\"", m.captures[1])
                    n = q.captures[1]
                    n in registered || continue
                    has_report && push!(report_cov, n)
                    has_plot && push!(plot_cov, n)
                end
            end
        end
        for blk in _testset_blocks(txt)
            types = _types_in_block(blk, registered)
            br = occursin("_assert_report_equal", blk) ||
                 occursin("_assert_consumers", blk) || occursin("_cover(", blk)
            bp = occursin("_assert_plot_equal", blk) ||
                 occursin("_assert_consumers", blk) || occursin("_cover(", blk)
            br && union!(report_cov, types)
            bp && union!(plot_cov, types)
        end
        for (est, tname) in _COVER_ESTIMATOR_TYPES
            occursin(est, txt) || continue
            if occursin("_cover(estimate_", txt) || occursin("_cover(" * est, txt) ||
               occursin("_cover(identify_", txt)
                has_report && push!(report_cov, tname)
                has_plot && push!(plot_cov, tname)
            end
        end
        if occursin("_sid24_dummy_objects", txt) &&
           (occursin("_assert_consumers", txt) || occursin("_cover(", txt))
            for n in _SID24_TYPES
                push!(report_cov, n)
                push!(plot_cov, n)
            end
        end
    end
    union!(report_cov, _REPORT_COVERED)
    union!(plot_cov, _PLOT_COVERED)
    report_cov, plot_cov
end
