# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

if !@isdefined(_assert_roundtrip)
    include(joinpath(@__DIR__, "..", "serialization_helpers.jl"))
end

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

const _RSER07 = ("DIDResult", "EventStudyLP", "LPDiDResult", "BaconDecomposition",
                 "PretrendTestResult", "NegativeWeightResult", "HonestDiDResult")
const _RSER07_LEADS = 2
const _RSER07_H = FAST ? 2 : 3

# Staggered panel built through `xtset` so cohort_id / time metadata are real
# xtset output (not a hand-built PanelData). Timing column `treat_time` feeds
# DID/Bacon/NW; binary `treat` feeds LP-DiD.
function _rser07_panel(; n_units=36, n_periods=16, seed=780)
    rng = MersenneTwister(seed)
    n_cohorts = 2
    units_per = n_units ÷ (n_cohorts + 1)
    treat_times = zeros(Int, n_units)
    for c in 1:n_cohorts
        t0 = 5 + 3 * (c - 1)
        for u in ((c - 1) * units_per + 1):(c * units_per)
            treat_times[u] = t0
        end
    end
    N = n_units * n_periods
    id = Vector{Int}(undef, N)
    t = Vector{Int}(undef, N)
    y = Vector{Float64}(undef, N)
    treat_time = Vector{Float64}(undef, N)
    treat = Vector{Float64}(undef, N)
    cohort = Vector{Int}(undef, N)
    row = 1
    for i in 1:n_units
        a = randn(rng)
        for tt in 1:n_periods
            g = treat_times[i]
            te = (g > 0 && tt >= g) ? 1.5 * (1 + 0.1 * (tt - g)) : 0.0
            y[row] = a + 0.1 * tt + te + 0.4 * randn(rng)
            id[row] = i
            t[row] = tt
            treat_time[row] = Float64(g)
            treat[row] = (g > 0 && tt >= g) ? 1.0 : 0.0
            cohort[row] = g
            row += 1
        end
    end
    df = DataFrame(id=id, t=t, y=y, treat_time=treat_time, treat=treat, cohort=cohort)
    xtset(df, :id, :t; cohort=:cohort, frequency=Quarterly)
end

function _assert_panel_meta(a::PanelData, b::PanelData)
    @test b.cohort_id == a.cohort_id
    @test b.cohort_id !== nothing
    @test b.time_id == a.time_id
    @test b.frequency == a.frequency
    @test b.group_id == a.group_id
    @test b.T_obs == a.T_obs
    @test b.n_groups == a.n_groups
    @test b.balanced == a.balanced
    b
end

function _assert_plot_style_equal(a, b; style::Symbol)
    c0 = _MEM._plot_counter[]
    pa = plot_result(a; style=style)
    _MEM._plot_counter[] = c0
    pb = plot_result(b; style=style)
    @test typeof(pa) === typeof(pb)
    for f in fieldnames(typeof(pa))
        @test _deep_equal(getfield(pa, f), getfield(pb, f))
    end
    pa
end

@testset "RSER-07 DiD serialization (#780)" begin
    @testset "registry" begin
        @test length(_RSER07) == 7
        for name in _RSER07
            @test haskey(_MEM._SERIALIZABLE_TYPES, name)
            @test !haskey(_MEM._SERIALIZATION_EXCLUDED, name)
        end
        @test !any(v == "pending RSER-07" for v in values(_MEM._SERIALIZATION_EXCLUDED))
    end

    pd = _rser07_panel(; n_units=FAST ? 15 : 36, n_periods=FAST ? 10 : 16)
    @test pd.cohort_id !== nothing
    @test pd.frequency == Quarterly

    @testset "DIDResult estimator variants" begin
        @test _from_serializable_is_generic(DIDResult)
        methods = FAST ? (:twfe,) : (:twfe, :callaway_santanna, :sun_abraham, :bjs)
        for method in methods
            did = estimate_did(pd, :y, :treat_time; method=method,
                               leads=_RSER07_LEADS, horizon=_RSER07_H)
            @test did isa DIDResult{Float64}
            @test did.method == method
            did2 = _assert_roundtrip(did)
            _assert_consumers(did, did2)
            @test isequal(DataFrame(did), DataFrame(did2))
            @test pretrend_test(did2).statistic == pretrend_test(did).statistic
        end
        if !FAST
            did_dcdh = estimate_did(pd, :y, :treat_time; method=:did_multiplegt,
                                    leads=2, horizon=3, n_boot=10,
                                    rng=MersenneTwister(7801))
            @test did_dcdh isa DIDResult{Float64}
            d2 = _assert_roundtrip(did_dcdh)
            _assert_consumers(did_dcdh, d2)
            @test isequal(DataFrame(did_dcdh), DataFrame(d2))
        end
    end

    @testset "EventStudyLP nested PanelData + Vector{Matrix{T}}" begin
        @test _from_serializable_is_generic(EventStudyLP)
        es = estimate_event_study_lp(pd, :y, :treat_time, _RSER07_H;
                                     leads=_RSER07_LEADS, lags=1)
        @test es isa EventStudyLP{Float64}
        @test es.data isa PanelData{Float64}
        @test es.data.cohort_id !== nothing
        @test es.vcov isa Vector{<:AbstractMatrix}
        args = Any[getfield(es, i) for i in 1:nfields(es)]
        @test _MEM._infer_float_param(args) === Float64
        payload = _MEM._to_serializable(es)
        @test haskey(payload, "data")
        @test payload["data"] isa AbstractDict
        @test payload["data"]["__struct__"] == "PanelData"
        @test payload["data"]["cohort_id"] == es.data.cohort_id
        _MEM._assert_plain_payload(_MEM._build_container(es))
        es2 = _assert_roundtrip(es)
        _assert_consumers(es, es2)
        @test es2.data isa PanelData{Float64}
        _assert_panel_meta(es.data, es2.data)
        @test es2.vcov isa Vector{Matrix{Float64}}
        @test eltype(es2.vcov) === Matrix{Float64}
        @test es2.B isa Vector{Matrix{Float64}}
        @test es2.residuals_per_h isa Vector{Matrix{Float64}}
        pt = pretrend_test(es)
        pt2 = pretrend_test(es2)
        @test _deep_equal(pt, pt2)
        if !FAST
            _assert_plot_style_equal(es, es2; style=:ribbon)
            let path = joinpath(mktempdir(), "event_study_lp.jld2")
                save_model(es, path)
                es3 = load_model(path)
                @test es3 isa EventStudyLP{Float64}
                @test es3.data isa PanelData{Float64}
                _assert_panel_meta(es.data, es3.data)
                _assert_report_equal(es, es3)
                _assert_plot_equal(es, es3)
                @test _deep_equal(pretrend_test(es), pretrend_test(es3))
            end
        end
    end

    @testset "LPDiDResult NamedTuple codec + nested PanelData" begin
        @test _from_serializable_is_generic(LPDiDResult)
        r = estimate_lp_did(pd, :y, :treat, _RSER07_H; pre_window=_RSER07_LEADS, ylags=1,
                            post_pooled=(0, _RSER07_H), pre_pooled=(1, _RSER07_LEADS))
        @test r isa LPDiDResult{Float64}
        @test r.pooled_post !== nothing
        @test r.pooled_pre !== nothing
        @test r.data isa PanelData{Float64}
        payload = _MEM._to_serializable(r)
        @test haskey(payload, "data")
        @test payload["data"]["__struct__"] == "PanelData"
        @test payload["pooled_post"] isa AbstractDict
        @test payload["pooled_post"]["__namedtuple__"] === true
        @test payload["pooled_pre"]["__namedtuple__"] === true
        _MEM._assert_plain_payload(_MEM._build_container(r))
        r2 = _assert_roundtrip(r)
        _assert_consumers(r, r2)
        @test r2.pooled_post isa NamedTuple
        @test r2.pooled_pre isa NamedTuple
        @test r2.pooled_post == r.pooled_post
        @test r2.pooled_pre == r.pooled_pre
        @test keys(r2.pooled_post) === (:coef, :se, :ci_lower, :ci_upper, :nobs)
        @test r2.data isa PanelData{Float64}
        _assert_panel_meta(r.data, r2.data)
        @test r2.vcov isa Vector{Matrix{Float64}}
    end

    @testset "BaconDecomposition" begin
        @test _from_serializable_is_generic(BaconDecomposition)
        bd = bacon_decomposition(pd, :y, :treat_time)
        @test bd isa BaconDecomposition{Float64}
        bd2 = _assert_roundtrip(bd)
        _assert_consumers(bd, bd2)
    end

    @testset "PretrendTestResult" begin
        @test _from_serializable_is_generic(PretrendTestResult)
        did = estimate_did(pd, :y, :treat_time; method=:twfe,
                           leads=_RSER07_LEADS, horizon=_RSER07_H)
        pt = pretrend_test(did)
        @test pt isa PretrendTestResult{Float64}
        pt2 = _assert_roundtrip(pt)
        _assert_consumers(pt, pt2)
        es = estimate_event_study_lp(pd, :y, :treat_time, _RSER07_H;
                                     leads=_RSER07_LEADS, lags=1)
        pte = pretrend_test(es)
        pte2 = _assert_roundtrip(pte)
        _assert_consumers(pte, pte2)
        if !FAST
            @test _deep_equal(pte2, pretrend_test(_roundtrip(es)))
        end
    end

    @testset "NegativeWeightResult Vector{Tuple}" begin
        @test _from_serializable_is_generic(NegativeWeightResult)
        nw = negative_weight_check(pd, :treat_time)
        @test nw isa NegativeWeightResult{Float64}
        @test nw.cohort_time_pairs isa Vector{<:Tuple}
        nw2 = _assert_roundtrip(nw)
        _assert_consumers(nw, nw2)
        @test nw2.cohort_time_pairs isa Vector{<:Tuple}
        @test nw2.cohort_time_pairs == nw.cohort_time_pairs
    end

    FAST && return
    @testset "HonestDiDResult disk" begin
        @test _from_serializable_is_generic(HonestDiDResult)
        did = estimate_did(pd, :y, :treat_time;
                           method=FAST ? :twfe : :callaway_santanna,
                           leads=_RSER07_LEADS, horizon=_RSER07_H,
                           base_period=FAST ? :varying : :universal)
        hd = honest_did(did; Mbar=1.0)
        @test hd isa HonestDiDResult{Float64}
        hd2 = _assert_roundtrip(hd)
        _assert_consumers(hd, hd2)
        if !FAST
            let path = joinpath(mktempdir(), "honest_did.jld2")
                save_model(hd, path)
                hd3 = load_model(path)
                @test hd3 isa HonestDiDResult{Float64}
                _assert_report_equal(hd, hd3)
                _assert_plot_equal(hd, hd3)
            end
        end
    end
end
