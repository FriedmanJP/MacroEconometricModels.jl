# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
plot_result methods for unit-root & structural-break tests (PLT-30):

- sequential-statistic lines (Andrews, Bai–Perron criteria) with a break vline and
  critical-value reference line;
- statistic-vs-critical-value bars (Zivot–Andrews, ADF two-break, Fourier ADF/KPSS);
- grouped statistic-vs-CV bars (Gregory–Hansen, Johansen).

`teststat.jl` keeps the SADF/GSADF bubble monitor; this file owns the break tests.
All rendering reuses the frozen renderers — no renderer is defined here (A1).
"""

# =============================================================================
# Lane-local converters (A5/A6-local — helpers.jl is frozen for Wave 2)
# =============================================================================

"""
    _teststat_seq_json(seq; xstart=1, key="stat") -> String

Line data rows `[{x, <key>}]` for a sequential test-statistic path: the value of
the `i`-th sequence entry attaches to candidate index `xstart + i - 1`. Non-finite
entries serialize to `null` (a visible gap; plotrule Robustness).
"""
function _teststat_seq_json(seq::AbstractVector; xstart::Int=1, key::String="stat")
    rows = Vector{Pair{String,String}}[]
    for (i, v) in enumerate(seq)
        push!(rows, ["x" => _json(xstart + i - 1), key => _json(v)])
    end
    _json_array_of_objects(rows)
end

"""
    _stat_vs_cv_hbar_json(stat_label, stat, cv) -> String

Horizontal single-series bar rows `[{x, v}]` comparing one test statistic to its
critical values: the statistic row (`x = stat_label`) followed by one row per
`(level% CV)` in the `Dict{Int,T}` `cv`, sorted by level. Names read horizontally
(plotrule form table for named categories).
"""
function _stat_vs_cv_hbar_json(stat_label::AbstractString, stat, cv::AbstractDict)
    rows = Vector{Pair{String,String}}[]
    push!(rows, ["x" => _json(String(stat_label)), "v" => _json(stat)])
    for (lvl, val) in sort(collect(cv), by=first)
        push!(rows, ["x" => _json("$(lvl)% CV"), "v" => _json(val)])
    end
    _json_array_of_objects(rows)
end

"""
    _grouped_stat_cv_json(labels, stats, cvs) -> String

Grouped-bar rows `[{x, stat, cv}]` — one category per `labels[i]` carrying the test
statistic `stats[i]` and the matching critical value `cvs[i]`. Feeds a two-series
`_render_bar_js(mode="grouped")` (statistic vs critical value).
"""
function _grouped_stat_cv_json(labels::AbstractVector, stats::AbstractVector, cvs::AbstractVector)
    rows = Vector{Pair{String,String}}[]
    for i in eachindex(labels)
        push!(rows, ["x" => _json(String(labels[i])),
                     "stat" => _json(stats[i]),
                     "cv" => _json(cvs[i])])
    end
    _json_array_of_objects(rows)
end

# Selected critical value from a `Dict{Int,T}` at `level` (percent), falling back to
# the nearest available level when the exact one is absent.
function _cv_at(cv::AbstractDict, level::Int)
    haskey(cv, level) && return cv[level]
    isempty(cv) && return nothing
    ks = sort(collect(keys(cv)))
    cv[ks[argmin(abs.(ks .- level))]]
end

_andrews_label(t::Symbol) = t === :supwald ? "sup-Wald" :
                            t === :expwald ? "exp-Wald" :
                            t === :meanwald ? "mean-Wald" : String(t)

# =============================================================================
# AndrewsResult — sequential Wald path + break vline + CV reference
# =============================================================================

"""
    plot_result(r::AndrewsResult; title="", save_path=nothing)

Andrews (1993) / Andrews–Ploberger structural-break test: the full Wald
`stat_sequence` over the candidate-break range (x = sample index), a solid vertical
break line at the estimated `break_index`, and a dashed horizontal 5 % critical-value
reference line (alert color). The x-axis spans exactly the trimmed candidate range,
so trimming is implicit. Integer x ticks.
"""
function plot_result(r::AndrewsResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    seq = r.stat_sequence
    amax = isempty(seq) ? 1 : argmax(seq)
    xstart = r.break_index - (amax - 1)          # sample index of the first candidate
    id = _next_plot_id("andrews")
    data_json = _teststat_seq_json(seq; xstart=xstart, key="stat")
    s_json = _series_json(["$(_andrews_label(r.test_type)) statistic"],
                          [_PLOT_SERIES[1]]; keys=["stat"])

    reflist = String[]
    cv5 = _cv_at(r.critical_values, 5)
    cv5 !== nothing && push!(reflist,
        "{\"value\":$(_json(cv5)),\"axis\":\"y\",\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"6,4\"}")
    push!(reflist,
        "{\"value\":$(_json(r.break_index)),\"axis\":\"x\",\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"0\"}")
    refs = "[" * join(reflist, ",") * "]"

    js = _render_line_js(id, data_json, s_json; ref_lines_json=refs, integer_x=true,
                         xlabel="Candidate break index", ylabel="Wald statistic")
    isempty(title) && (title = "Andrews Structural-Break Test ($(_andrews_label(r.test_type)))")
    ptitle = "Break at obs $(r.break_index) (fraction $(_fmt(r.break_fraction; digits=3)))"
    p = _make_plot([_PanelSpec(id, ptitle, js)]; title=title)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# BaiPerronResult — criteria path / break-location timeline
# =============================================================================

"""
    plot_result(r::BaiPerronResult; view=:criteria, title="", save_path=nothing)

Bai–Perron (1998, 2003) multiple-break test. `view`:

- `:criteria` (default) — BIC and LWZ information criteria vs the number of breaks
  (integer x from 0), with a solid vertical line at the selected `n_breaks`.
- `:breaks` — a break-location timeline over the sample: each estimated break date is
  a solid vertical line (alert color) and its `break_cis` confidence interval a shaded
  band, on a faint sample baseline.

Unknown `view` throws an `ArgumentError` naming both.
"""
function plot_result(r::BaiPerronResult{T}; view::Symbol=:criteria, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    if view === :criteria
        id = _next_plot_id("bpcrit")
        nb = length(r.bic_values)
        rows = Vector{Pair{String,String}}[]
        for i in 1:nb
            push!(rows, ["x" => _json(i - 1),
                         "bic" => _json(r.bic_values[i]),
                         "lwz" => _json(i <= length(r.lwz_values) ? r.lwz_values[i] : NaN)])
        end
        data_json = _json_array_of_objects(rows)
        s_json = _series_json(["BIC", "LWZ"], [_PLOT_SERIES[1], _PLOT_SERIES[2]];
                              keys=["bic", "lwz"])
        refs = "[{\"value\":$(_json(r.n_breaks)),\"axis\":\"x\"," *
               "\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"0\"}]"
        js = _render_line_js(id, data_json, s_json; ref_lines_json=refs, integer_x=true,
                             xlabel="Number of breaks", ylabel="Information criterion")
        isempty(title) && (title = "Bai–Perron Break-Number Selection")
        ptitle = "Selected breaks: $(r.n_breaks)"
        p = _make_plot([_PanelSpec(id, ptitle, js)]; title=title)
        save_path !== nothing && save_plot(p, save_path)
        return p
    elseif view === :breaks
        id = _next_plot_id("bpbreaks")
        n = max(r.nobs, 2)
        data_json = "[{\"x\":$(_json(1)),\"base\":0},{\"x\":$(_json(n)),\"base\":0}]"
        s_json = _series_json(["Sample"], ["#bbb"]; keys=["base"])
        reflist = String[]; regionlist = String[]
        for (k, bd) in enumerate(r.break_dates)
            push!(reflist,
                "{\"value\":$(_json(bd)),\"axis\":\"x\",\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"0\"}")
            if k <= length(r.break_cis)
                lo, hi = r.break_cis[k]
                push!(regionlist,
                    "{\"x0\":$(_json(lo)),\"x1\":$(_json(hi)),\"color\":$(_json(_PLOT_ALERT)),\"alpha\":0.12}")
            end
        end
        refs = "[" * join(reflist, ",") * "]"
        regions = "[" * join(regionlist, ",") * "]"
        js = _render_line_js(id, data_json, s_json; ref_lines_json=refs,
                             regions_json=regions, integer_x=true,
                             xlabel="Observation", ylabel="")
        isempty(title) && (title = "Bai–Perron Estimated Break Dates")
        ptitle = r.n_breaks == 0 ? "No structural breaks detected" :
                 "$(r.n_breaks) break(s) with 95% confidence intervals"
        p = _make_plot([_PanelSpec(id, ptitle, js)]; title=title)
        save_path !== nothing && save_plot(p, save_path)
        return p
    else
        throw(ArgumentError("Unknown view :$(view). Valid views: :criteria, :breaks"))
    end
end

# =============================================================================
# Single-statistic tests — statistic-vs-CV horizontal bar
# =============================================================================

# Shared builder: one statistic vs its critical-value thresholds (horizontal bar).
function _teststat_bar_panel(id::String, stat_label::AbstractString, stat, cv::AbstractDict)
    data_json = _stat_vs_cv_hbar_json(stat_label, stat, cv)
    s_json = _series_json(["Value"], [_PLOT_SERIES[1]]; keys=["v"])
    _render_bar_js(id, data_json, s_json; mode="grouped", orientation="h",
                   xlabel="Statistic / critical value", ylabel="")
end

"""
    plot_result(r::ZAResult; title="", save_path=nothing)

Zivot–Andrews test: a horizontal bar comparing the (single) test statistic to its
1/5/10 % critical values. There is no stored per-date statistic path, so the break
date and the 5 % reject decision are stated in the panel subtitle (no phantom
series; plotrule C6).
"""
function plot_result(r::ZAResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    id = _next_plot_id("za")
    js = _teststat_bar_panel(id, "ZA statistic", r.statistic, r.critical_values)
    cv5 = _cv_at(r.critical_values, 5)
    rej = cv5 === nothing ? "n/a" : (r.statistic < cv5 ? "reject H₀" : "fail to reject H₀")
    isempty(title) && (title = "Zivot–Andrews Unit-Root Test")
    ptitle = "Break at obs $(r.break_index) (fraction $(_fmt(r.break_fraction; digits=3))) — 5%: $(rej)"
    p = _make_plot([_PanelSpec(id, ptitle, js)]; title=title)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(r::ADF2BreakResult; title="", save_path=nothing)

ADF test with two structural breaks (Narayan–Popp 2010): a horizontal bar of the
test statistic vs its critical values, with the two estimated break dates stated in
the subtitle (no stored statistic path; plotrule C6).
"""
function plot_result(r::ADF2BreakResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    id = _next_plot_id("adf2b")
    js = _teststat_bar_panel(id, "ADF statistic", r.statistic, r.critical_values)
    cv5 = _cv_at(r.critical_values, 5)
    rej = cv5 === nothing ? "n/a" : (r.statistic < cv5 ? "reject H₀" : "fail to reject H₀")
    isempty(title) && (title = "ADF Two-Break Unit-Root Test")
    ptitle = "Breaks at obs $(r.break1), $(r.break2) — 5%: $(rej)"
    p = _make_plot([_PanelSpec(id, ptitle, js)]; title=title)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(r::FourierADFResult; title="", save_path=nothing)

Fourier ADF test (Enders–Lee 2012): a horizontal statistic-vs-CV bar. The fitted
Fourier component is **not** stored on the result, so it is not drawn — the honest
subtitle records the frequency `k` and the joint-significance F-statistic instead of
fabricating a series (plotrule C6, anti-pattern "phantom original").
"""
function plot_result(r::FourierADFResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    id = _next_plot_id("fadf")
    js = _teststat_bar_panel(id, "Fourier ADF statistic", r.statistic, r.critical_values)
    isempty(title) && (title = "Fourier ADF Unit-Root Test")
    ptitle = "k=$(r.frequency), F=$(_fmt(r.f_statistic; digits=3)) — fitted Fourier component not stored"
    p = _make_plot([_PanelSpec(id, ptitle, js)]; title=title)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(r::FourierKPSSResult; title="", save_path=nothing)

Fourier KPSS stationarity test (Becker–Enders–Lee 2006): a horizontal statistic-vs-CV
bar. As with the Fourier ADF, no fitted Fourier component is stored, so the subtitle
reports the frequency and F-statistic honestly rather than drawing a phantom series
(plotrule C6).
"""
function plot_result(r::FourierKPSSResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    id = _next_plot_id("fkpss")
    js = _teststat_bar_panel(id, "Fourier KPSS statistic", r.statistic, r.critical_values)
    isempty(title) && (title = "Fourier KPSS Stationarity Test")
    ptitle = "k=$(r.frequency), F=$(_fmt(r.f_statistic; digits=3)) — fitted Fourier component not stored"
    p = _make_plot([_PanelSpec(id, ptitle, js)]; title=title)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# Gregory–Hansen / Johansen — grouped statistic-vs-CV bars
# =============================================================================

"""
    plot_result(r::GregoryHansenResult; title="", save_path=nothing)

Gregory–Hansen cointegration test with a structural break: a grouped bar comparing
each of the ADF*, Zt* and Za* statistics to its 5 % critical value (the ADF and Zt
statistics share the ADF critical-value table; Za uses its own). The estimated break
date is annotated in the subtitle.
"""
function plot_result(r::GregoryHansenResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    id = _next_plot_id("gh")
    adf_cv5 = _cv_at(r.adf_critical_values, 5)
    za_cv5 = _cv_at(r.za_critical_values, 5)
    labels = ["ADF*", "Zt*", "Za*"]
    stats = [r.adf_statistic, r.zt_statistic, r.za_statistic]
    cvs = [adf_cv5, adf_cv5, za_cv5]
    data_json = _grouped_stat_cv_json(labels, stats, cvs)
    s_json = _series_json(["Statistic", "5% CV"], [_PLOT_SERIES[1], _PLOT_ALERT];
                          keys=["stat", "cv"])
    js = _render_bar_js(id, data_json, s_json; mode="grouped", orientation="v",
                        xlabel="Statistic", ylabel="Value")
    isempty(title) && (title = "Gregory–Hansen Cointegration Test")
    ptitle = "Break at obs $(r.adf_break)"
    p = _make_plot([_PanelSpec(id, ptitle, js)]; title=title)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(r::JohansenResult; title="", save_path=nothing)

Johansen cointegration test: a grouped bar of the trace statistic per null rank
`r ≤ k` against the corresponding 5 % critical value. The estimated cointegration
rank is annotated in the subtitle.
"""
function plot_result(r::JohansenResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    id = _next_plot_id("johansen")
    nranks = length(r.trace_stats)
    ncv = size(r.critical_values_trace, 1)
    labels = ["r≤$(k-1)" for k in 1:nranks]
    stats = collect(r.trace_stats)
    cvs = [k <= ncv ? r.critical_values_trace[k, 2] : T(NaN) for k in 1:nranks]  # col 2 = 5%
    data_json = _grouped_stat_cv_json(labels, stats, cvs)
    s_json = _series_json(["Trace statistic", "5% CV"], [_PLOT_SERIES[1], _PLOT_ALERT];
                          keys=["stat", "cv"])
    js = _render_bar_js(id, data_json, s_json; mode="grouped", orientation="v",
                        xlabel="Null rank", ylabel="Statistic")
    isempty(title) && (title = "Johansen Cointegration Test (trace)")
    ptitle = "Estimated cointegration rank = $(r.rank)"
    p = _make_plot([_PanelSpec(id, ptitle, js)]; title=title)
    save_path !== nothing && save_plot(p, save_path)
    p
end
