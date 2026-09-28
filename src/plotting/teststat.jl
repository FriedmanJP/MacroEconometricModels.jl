# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# Hypothesis-test plotting dispatches (D3.js, zero external deps).
# EV-30 (#438): the signature explosive-bubble chart — the backward sup-ADF
# (BSADF) sequence against its 95% critical-value sequence, with the stamped
# bubble episodes shaded. This is the standard PSY (2015) central-bank monitor.
# =============================================================================

"""
    plot_result(r::BubbleResult; title="", save_path=nothing)

Draw the sup-ADF bubble monitor: the BSADF sequence and its 95% critical-value
sequence as two lines (right-tailed — exuberance where BSADF pierces the CV),
with the date-stamped bubble [`episodes`](@ref BubbleResult) shaded. The x-axis
is the level index of `y` (`r2_index`), so shaded regions line up with the
user's calendar.
"""
function plot_result(r::BubbleResult{T};
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    id = _next_plot_id("bubble")

    # Vertical shading extent for episodes (span the plotted y-range).
    yall = vcat(collect(r.bsadf), collect(r.cv_seq))
    yall = filter(isfinite, yall)
    ylo = isempty(yall) ? -1.0 : minimum(yall)
    yhi = isempty(yall) ? 1.0 : maximum(yall)
    ypad = (yhi - ylo) * 0.10 + eps(Float64)
    shade_lo = ylo - ypad
    shade_hi = yhi + ypad

    in_episode(idx) = any(ep -> ep[1] <= idx <= ep[2], r.episodes)

    rows = Vector{Pair{String,String}}[]
    for k in eachindex(r.r2_index)
        idx = r.r2_index[k]
        shaded = in_episode(idx)
        push!(rows, [
            "x" => _json(idx),
            "bsadf" => _json(r.bsadf[k]),
            "cv95" => _json(r.cv_seq[k]),
            "ep_lo" => (shaded ? _json(shade_lo) : "null"),
            "ep_hi" => (shaded ? _json(shade_hi) : "null"),
        ])
    end
    data_json = _json_array_of_objects(rows)

    seq_name = r.kind == :sadf ? "Recursive ADF" : "BSADF"
    s_json = _series_json([seq_name, "95% Critical Value"],
                          [_PLOT_COLORS[1], _PLOT_COLORS[2]];
                          keys=["bsadf", "cv95"], dash=["", "6,3"])
    bands = "[{\"lo_key\":\"ep_lo\",\"hi_key\":\"ep_hi\",\"color\":\"$(_PLOT_COLORS[4])\",\"alpha\":0.18}]"

    js = _render_line_js(id, data_json, s_json;
                         bands_json=bands,
                         xlabel="Observation index", ylabel="sup-ADF statistic")
    kind_name = r.kind == :sadf ? "SADF" : "GSADF"
    panels = [_PanelSpec(id, "$(kind_name) Bubble Monitor", js)]

    if isempty(title)
        n_ep = length(r.episodes)
        title = string(kind_name, " Explosive-Behaviour Monitor (",
                       n_ep, n_ep == 1 ? " episode" : " episodes", ", n=", r.nobs, ")")
    end
    p = _make_plot(panels; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# Single-statistic unit-root tests — statistic-vs-CV horizontal bar (#841 PR1)
# =============================================================================

# Shared shape: one statistic vs its 1/5/10% critical values via
# `_teststat_bar_panel` (teststat_breaks.jl), with the reject decision at the
# `level`% CV stated in the panel subtitle. Left-tailed tests (ADF, PP, ERS:
# statistic below CV rejects the unit root) pass `left=true`; right-tailed
# tests (KPSS: statistic above CV rejects stationarity) pass `left=false`.
function _unitroot_bar_plot(prefix::String, stat_label::AbstractString,
                             stat::Real, cv::AbstractDict, left::Bool,
                             ftitle::String, ptitle::String;
                             title::String="",
                             save_path::Union{String,Nothing}=nothing,
                             level::Int=5)
    id = _next_plot_id(prefix)
    js = _teststat_bar_panel(id, stat_label, stat, cv)
    c = _cv_at(cv, level)
    rej = c === nothing ? "n/a" :
        ((left ? stat < c : stat > c) ? "reject H₀" : "fail to reject H₀")
    ft = isempty(title) ? ftitle : title
    p = _make_plot([_PanelSpec(id, "$(ptitle) — $(level)%: $(rej)", js)]; title=ft)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(r::ADFResult; title="", save_path=nothing, level=5)

Augmented Dickey–Fuller test: a horizontal bar comparing the ADF t-statistic to
its 1/5/10% critical values (left-tailed: below CV rejects the unit root). The
lag order and regression specification are stated in the panel subtitle alongside
the reject decision at the `level`% CV (default 5).
"""
function plot_result(r::ADFResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    _unitroot_bar_plot("adf", "ADF statistic", r.statistic, r.critical_values, true,
        "Augmented Dickey–Fuller Unit-Root Test",
        "lags=$(r.lags), $(r.regression)";
        title=title, save_path=save_path, level=level)
end

"""
    plot_result(r::KPSSResult; title="", save_path=nothing, level=5)

KPSS stationarity test: a horizontal bar comparing the KPSS statistic to its
1/5/10% critical values (right-tailed: above CV rejects stationarity). The
bandwidth and regression specification are stated in the panel subtitle alongside
the reject decision at the `level`% CV (default 5).
"""
function plot_result(r::KPSSResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    _unitroot_bar_plot("kpss", "KPSS statistic", r.statistic, r.critical_values, false,
        "KPSS Stationarity Test",
        "bandwidth=$(r.bandwidth), $(r.regression)";
        title=title, save_path=save_path, level=level)
end

"""
    plot_result(r::PPResult; title="", save_path=nothing, level=5)

Phillips–Perron test: a horizontal bar comparing the PP statistic to its
1/5/10% critical values (left-tailed: below CV rejects the unit root). The
bandwidth and regression specification are stated in the panel subtitle alongside
the reject decision at the `level`% CV (default 5).
"""
function plot_result(r::PPResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    _unitroot_bar_plot("pp", "PP statistic", r.statistic, r.critical_values, true,
        "Phillips–Perron Unit-Root Test",
        "bandwidth=$(r.bandwidth), $(r.regression)";
        title=title, save_path=save_path, level=level)
end

"""
    plot_result(r::ERSResult; title="", save_path=nothing, level=5)

Elliott–Rothenberg–Stock point-optimal test: a horizontal bar comparing the `P_T`
statistic to its 1/5/10% critical values (left-tailed: below CV rejects the unit
root). The regression specification is stated in the panel subtitle alongside the
reject decision at the `level`% CV (default 5).
"""
function plot_result(r::ERSResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    _unitroot_bar_plot("ers", "ERS P_T statistic", r.P_T, r.critical_values, true,
        "ERS Point-Optimal Unit-Root Test",
        "$(r.regression)";
        title=title, save_path=save_path, level=level)
end

# =============================================================================
# Multi-statistic + panel unit-root tests (#841 PR2)
# =============================================================================

# Standard-normal reference bars for asymptotically N(0,1) statistics (LLC, IPS,
# Breitung, Hadri): the 1/5/10% CVs are universal normal quantiles — the same
# distribution the stored p-values are computed from — not stored per result.
function _normal_bar_plot(prefix::String, stat_label::AbstractString, stat::Real,
                           left::Bool, ftitle::String, ptitle::String;
                           title::String="",
                           save_path::Union{String,Nothing}=nothing,
                           level::Int=5)
    q = level == 1 ? 0.01 : level == 10 ? 0.10 : 0.05
    cv = Dict(1 => quantile(Normal(), left ? 0.01 : 0.99),
              5 => quantile(Normal(), left ? 0.05 : 0.95),
              10 => quantile(Normal(), left ? 0.10 : 0.90))
    id = _next_plot_id(prefix)
    js = _teststat_bar_panel(id, stat_label, stat, cv)
    z = quantile(Normal(), left ? q : 1 - q)
    rej = (left ? stat < z : stat > z) ? "reject H₀" : "fail to reject H₀"
    ft = isempty(title) ? ftitle : title
    p = _make_plot([_PanelSpec(id, "$(ptitle) — $(level)%: $(rej)", js)]; title=ft)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(r::NgPerronResult; title="", save_path=nothing, level=5)

Ng–Perron (2001) tests: a grouped bar comparing each of the four statistics
(`MZa`, `MZt`, `MSB`, `MPT`) to its own `level`% critical value (default 5;
all four reject the unit root below their CV). The regression specification is
stated in the subtitle alongside the count of rejections.
"""
function plot_result(r::NgPerronResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    id = _next_plot_id("ngp")
    labels = ["MZa", "MZt", "MSB", "MPT"]
    stats = [r.MZa, r.MZt, r.MSB, r.MPT]
    cvs = [_cv_at(r.critical_values[k], level) for k in (:MZa, :MZt, :MSB, :MPT)]
    data_json = _grouped_stat_cv_json(labels, stats, cvs)
    s_json = _series_json(["Statistic", "$(level)% CV"], [_PLOT_SERIES[1], _PLOT_ALERT];
                          keys=["stat", "cv"])
    js = _render_bar_js(id, data_json, s_json; mode="grouped", orientation="v",
                        xlabel="Statistic", ylabel="Value")
    nrej = count(i -> cvs[i] !== nothing && stats[i] < cvs[i], eachindex(stats))
    isempty(title) && (title = "Ng–Perron Unit-Root Tests")
    ptitle = "$(r.regression) — $(nrej)/4 reject H₀ at $(level)%"
    p = _make_plot([_PanelSpec(id, ptitle, js)]; title=title)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(r::HEGYResult; title="", save_path=nothing, level=5)

HEGY seasonal unit-root test: two grouped-bar panels — the zero-frequency and
Nyquist t-statistics vs their own `level`% CVs (left-tailed), and the harmonic-pair
joint F-statistics vs the common pair-F CV (right-tailed). The joint `F_seasonal`
and `F_all` values carry no stored CVs, so they are reported in the subtitle,
not drawn.
"""
function plot_result(r::HEGYResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    id1 = _next_plot_id("hegy_t")
    tstats = [r.t_zero, r.t_nyquist]
    tcvs = [_cv_at(r.t_zero_cv, level), _cv_at(r.t_nyquist_cv, level)]
    js1 = _render_bar_js(id1,
        _grouped_stat_cv_json(["t(π₁) zero", "t(π₂) Nyquist"], tstats, tcvs),
        _series_json(["Statistic", "$(level)% CV"], [_PLOT_SERIES[1], _PLOT_ALERT];
                     keys=["stat", "cv"]);
        mode="grouped", orientation="v", xlabel="Frequency", ylabel="t-statistic")
    id2 = _next_plot_id("hegy_f")
    flabels = ["F(ω=$(_fmt(w; digits=3)))" for w in r.pair_freqs]
    fcvs = [_cv_at(r.pair_F_cv, level) for _ in r.pair_F]
    js2 = _render_bar_js(id2,
        _grouped_stat_cv_json(flabels, collect(r.pair_F), fcvs),
        _series_json(["Statistic", "$(level)% CV"], [_PLOT_SERIES[1], _PLOT_ALERT];
                     keys=["stat", "cv"]);
        mode="grouped", orientation="v", xlabel="Harmonic pair", ylabel="F-statistic")
    isempty(title) && (title = "HEGY Seasonal Unit-Root Test")
    ptitle1 = "t-stats vs $(level)% CVs (left-tailed)"
    ptitle2 = "F_seasonal=$(_fmt(r.F_seasonal; digits=3)), " *
              "F_all=$(_fmt(r.F_all; digits=3)) (no stored CVs)"
    p = _make_plot([_PanelSpec(id1, ptitle1, js1),
                    _PanelSpec(id2, ptitle2, js2)]; title=title)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(r::LLCResult; title="", save_path=nothing, level=5)

Levin–Lin–Chu panel unit-root test: the standardized `t*` statistic (N(0,1)
under the unit-root null) against standard-normal 1/5/10% CVs (left-tailed).
The deterministic specification and cross-section size are stated in the
subtitle alongside the reject decision at the `level`% CV (default 5).
"""
function plot_result(r::LLCResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    _normal_bar_plot("llc", "LLC t* statistic", r.statistic, true,
        "Levin–Lin–Chu Panel Unit-Root Test",
        "$(r.deterministic), N=$(r.n_units)";
        title=title, save_path=save_path, level=level)
end

"""
    plot_result(r::IPSResult; title="", save_path=nothing, level=5)

Im–Pesaran–Shin panel unit-root test: the standardized `W_tbar` statistic
(N(0,1) under the unit-root null) against standard-normal CVs (left-tailed).
The mean per-unit t (`tbar`) and cross-section size are stated in the subtitle
alongside the reject decision at the `level`% CV (default 5).
"""
function plot_result(r::IPSResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    _normal_bar_plot("ips", "IPS W_tbar statistic", r.statistic, true,
        "Im–Pesaran–Shin Panel Unit-Root Test",
        "tbar=$(_fmt(r.tbar; digits=3)), N=$(r.n_units)";
        title=title, save_path=save_path, level=level)
end

"""
    plot_result(r::BreitungPanelResult; title="", save_path=nothing, level=5)

Breitung panel unit-root test: the `λ` statistic (N(0,1) under the unit-root
null) against standard-normal CVs (left-tailed). The deterministic
specification and cross-section size are stated in the subtitle alongside the
reject decision at the `level`% CV (default 5).
"""
function plot_result(r::BreitungPanelResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    _normal_bar_plot("breitung", "Breitung λ statistic", r.statistic, true,
        "Breitung Panel Unit-Root Test",
        "$(r.deterministic), N=$(r.n_units)";
        title=title, save_path=save_path, level=level)
end

"""
    plot_result(r::FisherPanelResult; title="", save_path=nothing, level=5)

Fisher-type panel unit-root test: the four combination p-values (`P`, `Z`,
`L*`, `Pm`) as bars against the significance level `α` (default 5%). Each
combination lives on its own statistic scale, so p-values — uniform on [0,1]
— are the comparable quantity. The base test and selected combination are
stated in the subtitle alongside the rejection count.
"""
function plot_result(r::FisherPanelResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    α = level == 1 ? 0.01 : level == 10 ? 0.10 : 0.05
    id = _next_plot_id("fisherp")
    labels = ["P (MW)", "Z (Choi)", "L* (logit)", "Pm"]
    pvs = [r.P_pvalue, r.Z_pvalue, r.Lstar_pvalue, r.Pm_pvalue]
    data_json = _grouped_stat_cv_json(labels, pvs, fill(α, 4))
    s_json = _series_json(["p-value", "α = $(α)"], [_PLOT_SERIES[1], _PLOT_ALERT];
                          keys=["stat", "cv"])
    js = _render_bar_js(id, data_json, s_json; mode="grouped", orientation="v",
                        xlabel="Combination", ylabel="p-value")
    nrej = count(pv -> pv < α, pvs)
    isempty(title) && (title = "Fisher-Type Panel Unit-Root Test")
    ptitle = "base=$(r.base), combine=$(r.combine) — $(nrej)/4 reject at $(level)%"
    p = _make_plot([_PanelSpec(id, ptitle, js)]; title=title)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(r::HadriResult; title="", save_path=nothing, level=5)

Hadri panel stationarity test: the standardized `Z` statistic (N(0,1) under the
stationarity null) against standard-normal CVs (right-tailed: above CV rejects
stationarity). The mean LM statistic and deterministic specification are stated
in the subtitle alongside the reject decision at the `level`% CV (default 5).
"""
function plot_result(r::HadriResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    _normal_bar_plot("hadri", "Hadri Z statistic", r.statistic, false,
        "Hadri Panel Stationarity Test",
        "LM̄=$(_fmt(r.LM; digits=3)), $(r.deterministic)";
        title=title, save_path=save_path, level=level)
end

# =============================================================================
# Panel unit-root set 2 + cointegration (#841 PR3)
# =============================================================================

# Grouped statistics vs per-statistic standard-normal CVs. `right[i]` selects
# the upper tail for `stats[i]` (Pedroni panel-v); all others use the lower
# tail. Rejections are counted with each statistic's own tail.
function _grouped_normal_plot(prefix::String, labels::AbstractVector,
                               stats::AbstractVector, right::AbstractVector{Bool},
                               ftitle::String, ptitle::String;
                               title::String="",
                               save_path::Union{String,Nothing}=nothing,
                               level::Int=5)
    α = level == 1 ? 0.01 : level == 10 ? 0.10 : 0.05
    cvs = [r ? quantile(Normal(), 1 - α) : quantile(Normal(), α) for r in right]
    id = _next_plot_id(prefix)
    js = _render_bar_js(id,
        _grouped_stat_cv_json(labels, stats, cvs),
        _series_json(["Statistic", "$(level)% CV"], [_PLOT_SERIES[1], _PLOT_ALERT];
                     keys=["stat", "cv"]);
        mode="grouped", orientation="v", xlabel="Statistic", ylabel="Value")
    nrej = count(i -> right[i] ? stats[i] > cvs[i] : stats[i] < cvs[i],
                 eachindex(stats))
    ft = isempty(title) ? ftitle : title
    p = _make_plot([_PanelSpec(id, "$(ptitle) — $(nrej)/$(length(stats)) reject H₀ at $(level)%", js)];
                   title=ft)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# P-values as bars against the significance level: the comparable quantity when
# member statistics live on different scales (Fisher combos, PANIC pooling,
# residual-cointegration tests without stored CVs).
function _pvalue_bar_plot(prefix::String, labels::AbstractVector,
                           pvalues::AbstractVector, ftitle::String, ptitle::String;
                           title::String="",
                           save_path::Union{String,Nothing}=nothing,
                           level::Int=5)
    α = level == 1 ? 0.01 : level == 10 ? 0.10 : 0.05
    id = _next_plot_id(prefix)
    js = _render_bar_js(id,
        _grouped_stat_cv_json(labels, pvalues, fill(α, length(pvalues))),
        _series_json(["p-value", "α = $(α)"], [_PLOT_SERIES[1], _PLOT_ALERT];
                     keys=["stat", "cv"]);
        mode="grouped", orientation="v", xlabel="Test", ylabel="p-value")
    nrej = count(pv -> pv < α, pvalues)
    ft = isempty(title) ? ftitle : title
    p = _make_plot([_PanelSpec(id, "$(ptitle) — $(nrej)/$(length(pvalues)) reject at $(level)%", js)];
                   title=ft)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(r::MoonPerronResult; title="", save_path=nothing, level=5)

Moon–Perron panel unit-root test: the two factor-adjusted statistics (`t_a*`,
`t_b*`, N(0,1) under the unit-root null) against standard-normal CVs
(left-tailed). The number of estimated factors is stated in the subtitle
alongside the rejection count at the `level`% CV (default 5).
"""
function plot_result(r::MoonPerronResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    _grouped_normal_plot("moonperron", ["t_a*", "t_b*"],
        [r.t_a_statistic, r.t_b_statistic], [false, false],
        "Moon–Perron Panel Unit-Root Test", "factors=$(r.n_factors), N=$(r.n_units)";
        title=title, save_path=save_path, level=level)
end

"""
    plot_result(r::PANICResult; title="", save_path=nothing, level=5)

PANIC pooled + idiosyncratic unit-root p-values as bars against `α` (default
5%): the pooled p-value first, then per-unit p-values (capped at 11 units with
a cap note). The pooling method and factor count are stated in the subtitle
alongside the rejection count.
"""
function plot_result(r::PANICResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    nshow = min(length(r.individual_pvalues), 11)
    labels = vcat(["pooled"], ["unit $i" for i in 1:nshow])
    pvs = vcat([r.pooled_pvalue], r.individual_pvalues[1:nshow])
    capnote = length(r.individual_pvalues) > nshow ?
        " (showing $(nshow) of $(length(r.individual_pvalues)) units)" : ""
    _pvalue_bar_plot("panic", labels, pvs, "PANIC Panel Unit-Root Test",
        "$(r.method), factors=$(r.n_factors)$(capnote)";
        title=title, save_path=save_path, level=level)
end

"""
    plot_result(r::PesaranCIPSResult; title="", save_path=nothing, level=5)

Pesaran CIPS panel unit-root test: a horizontal bar comparing the CIPS
statistic to its 1/5/10% critical values (left-tailed). Lags, deterministic
case, and cross-section size are stated in the subtitle alongside the reject
decision at the `level`% CV (default 5).
"""
function plot_result(r::PesaranCIPSResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    _unitroot_bar_plot("cips", "CIPS statistic", r.cips_statistic,
        r.critical_values, true,
        "Pesaran CIPS Panel Unit-Root Test",
        "lags=$(r.lags), $(r.deterministic), N=$(r.n_units)";
        title=title, save_path=save_path, level=level)
end

"""
    plot_result(r::KaoResult; title="", save_path=nothing, level=5)

Kao residual-based panel cointegration test: the five DF-type statistics (all
N(0,1), left-tailed) against standard-normal CVs. The pooled AR coefficient
and cross-section size are stated in the subtitle alongside the rejection
count at the `level`% CV (default 5).
"""
function plot_result(r::KaoResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    _grouped_normal_plot("kao", r.names, collect(r.statistics),
        fill(false, length(r.statistics)),
        "Kao Panel Cointegration Test", "ρ̂=$(_fmt(r.rho; digits=3)), N=$(r.n_units)";
        title=title, save_path=save_path, level=level)
end

"""
    plot_result(r::PedroniResult; title="", save_path=nothing, level=5)

Pedroni residual-based panel cointegration test: the seven statistics against
standard-normal CVs — `panel-v` is right-tailed, the other six left-tailed
(the per-bar CV reflects each statistic's own tail). Trend case and
cross-section size are stated in the subtitle alongside the rejection count at
the `level`% CV (default 5).
"""
function plot_result(r::PedroniResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    right = [nm == "panel-v" for nm in r.names]
    _grouped_normal_plot("pedroni", r.names, collect(r.statistics), right,
        "Pedroni Panel Cointegration Test",
        "$(r.trend), N=$(r.n_units) (panel-v right-tailed)";
        title=title, save_path=save_path, level=level)
end

"""
    plot_result(r::WesterlundResult; title="", save_path=nothing, level=5)

Westerlund ECM panel cointegration test: `Gt`, `Ga`, `Pt`, `Pa` (all N(0,1),
left-tailed) against standard-normal CVs. Trend case and bootstrap replication
count are stated in the subtitle alongside the rejection count at the `level`%
CV (default 5).
"""
function plot_result(r::WesterlundResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    _grouped_normal_plot("westerlund", r.names, collect(r.statistics),
        fill(false, length(r.statistics)),
        "Westerlund Panel Cointegration Test",
        "$(r.trend), bootstrap=$(r.bootstrap), N=$(r.n_units)";
        title=title, save_path=save_path, level=level)
end

"""
    plot_result(r::EngleGrangerResult; title="", save_path=nothing, level=5)

Engle–Granger residual cointegration test: the residual-ADF p-value as a bar
against `α` (default 5%). The statistic follows a nonstandard Dickey–Fuller
distribution with no stored CVs, so the MacKinnon p-value is the plottable
quantity. Lags, regressor count, and deterministic case are stated in the
subtitle alongside the statistic value.
"""
function plot_result(r::EngleGrangerResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    _pvalue_bar_plot("englegranger", ["residual ADF"], [r.pvalue],
        "Engle–Granger Cointegration Test",
        "t=$(_fmt(r.statistic; digits=3)), lags=$(r.lags), k=$(r.k), $(r.regression)";
        title=title, save_path=save_path, level=level)
end

"""
    plot_result(r::PhillipsOuliarisResult; title="", save_path=nothing, level=5)

Phillips–Ouliaris residual cointegration test: the `Ẑ_t` and `Ẑ_α` p-values as
bars against `α` (default 5%). Neither statistic carries stored CVs, so the
p-values are the plottable quantity. Kernel, bandwidth, and statistic values
are stated in the subtitle.
"""
function plot_result(r::PhillipsOuliarisResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    _pvalue_bar_plot("phillipsouliaris", ["Ẑ_t", "Ẑ_α"], [r.pvalue, r.z_alpha_pvalue],
        "Phillips–Ouliaris Cointegration Test",
        "Ẑ_t=$(_fmt(r.statistic; digits=3)), Ẑ_α=$(_fmt(r.z_alpha; digits=3)), " *
        "$(r.kernel), bw=$(_fmt(r.bandwidth; digits=3))";
        title=title, save_path=save_path, level=level)
end

"""
    plot_result(r::ParkAddedResult; title="", save_path=nothing, level=5)

Park added-trends cointegration test: the `H(p,q)` Wald p-value (χ² with
`q_add` degrees of freedom, upper tail) as a bar against `α` (default 5%).
The statistic value and trend orders are stated in the subtitle.
"""
function plot_result(r::ParkAddedResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    _pvalue_bar_plot("parkadded", ["H(p,q)"], [r.pvalue],
        "Park Added-Trends Cointegration Test",
        "H=$(_fmt(r.statistic; digits=3)), q_add=$(r.q_add), p=$(r.base_order)";
        title=title, save_path=save_path, level=level)
end

# =============================================================================
# Portmanteau / dependence / causality (#841 PR4)
# =============================================================================

# Right-tailed χ² reference bars: the 1/5/10% CVs are universal quantiles of
# χ²(df) — the same distribution the stored p-values are computed from — not
# stored per result.
function _chisq_bar_plot(prefix::String, stat_label::AbstractString, stat::Real,
                           df::Int, ftitle::String, ptitle::String;
                           title::String="",
                           save_path::Union{String,Nothing}=nothing,
                           level::Int=5)
    cv = Dict(1 => quantile(Chisq(df), 0.99),
              5 => quantile(Chisq(df), 0.95),
              10 => quantile(Chisq(df), 0.90))
    id = _next_plot_id(prefix)
    js = _teststat_bar_panel(id, stat_label, stat, cv)
    α = level == 1 ? 0.01 : level == 10 ? 0.10 : 0.05
    c = quantile(Chisq(df), 1 - α)
    rej = stat > c ? "reject H₀" : "fail to reject H₀"
    ft = isempty(title) ? ftitle : title
    p = _make_plot([_PanelSpec(id, "$(ptitle) — $(level)%: $(rej)", js)]; title=ft)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(r::LjungBoxResult; title="", save_path=nothing, level=5)

Ljung–Box portmanteau test: the Q-statistic against χ² 1/5/10% CVs
(right-tailed: above CV rejects no-autocorrelation). Lags and degrees of
freedom are stated in the subtitle alongside the reject decision at the
`level`% CV (default 5).
"""
function plot_result(r::LjungBoxResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    _chisq_bar_plot("ljungbox", "LB Q statistic", r.statistic, r.df,
        "Ljung–Box Portmanteau Test", "lags=$(r.lags), df=$(r.df)";
        title=title, save_path=save_path, level=level)
end

"""
    plot_result(r::BoxPierceResult; title="", save_path=nothing, level=5)

Box–Pierce portmanteau test: the Q-statistic against χ² 1/5/10% CVs
(right-tailed). Lags and degrees of freedom are stated in the subtitle
alongside the reject decision at the `level`% CV (default 5).
"""
function plot_result(r::BoxPierceResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    _chisq_bar_plot("boxpierce", "BP Q statistic", r.statistic, r.df,
        "Box–Pierce Portmanteau Test", "lags=$(r.lags), df=$(r.df)";
        title=title, save_path=save_path, level=level)
end

"""
    plot_result(r::DurbinWatsonResult; title="", save_path=nothing, level=5)

Durbin–Watson first-order autocorrelation test: the p-value as a bar against
`α` (default 5%). The DW null distribution is nonstandard (regressor-dependent,
no stored CVs), so the p-value is the plottable quantity. The statistic (≈2
under H₀) and sample size are stated in the subtitle.
"""
function plot_result(r::DurbinWatsonResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    _pvalue_bar_plot("durbinwatson", ["DW"], [r.pvalue],
        "Durbin–Watson Autocorrelation Test",
        "DW=$(_fmt(r.statistic; digits=3)) (≈2 under H₀), n=$(r.nobs)";
        title=title, save_path=save_path, level=level)
end

"""
    plot_result(r::BDSResult; title="", save_path=nothing, level=5)

BDS independence test: per-`(m, ε)`-cell two-sided p-values as bars against
`α` (default 5%), capped at 12 cells with a cap note. Cells live on different
correlation-integral scales, so p-values are the comparable quantity. A small
sample (`T < 200`) makes the asymptotic p-values unreliable — flagged in the
subtitle when set.
"""
function plot_result(r::BDSResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    cells = [(i, j) for i in eachindex(r.m) for j in eachindex(r.eps)]
    nshow = min(length(cells), 12)
    labels = ["m=$(r.m[i]),ε=$(_fmt(r.eps[j]; digits=3))" for (i, j) in cells[1:nshow]]
    pvs = [r.pvalue[i, j] for (i, j) in cells[1:nshow]]
    capnote = length(cells) > nshow ?
        " (showing $(nshow) of $(length(cells)) cells)" : ""
    smallnote = r.small_sample ? "; T<200: asymptotic p-values unreliable" : ""
    _pvalue_bar_plot("bds", labels, pvs, "BDS Independence Test",
        "n=$(r.nobs)$(capnote)$(smallnote)";
        title=title, save_path=save_path, level=level)
end

"""
    plot_result(r::VarianceRatioResult; title="", save_path=nothing, level=5)

Variance-ratio random-walk test: per-`q` p-values plus the joint Chow–Denning
p-value as bars against `α` (default 5%). The individual statistics are N(0,1)
but the joint CD statistic follows a Studentized Maximum Modulus law with no
stored CVs, so p-values are the comparable quantity. Uses the robust (`z*`)
branch when reported, matching `StatsAPI.pvalue`. Method and Wright-augmentation
flags are stated in the subtitle.
"""
function plot_result(r::VarianceRatioResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    qpvs = r.robust ? r.z_star_pvalue : r.z_pvalue
    cd = r.robust ? r.cd_star_pvalue : r.cd_pvalue
    labels = vcat(["VR($(q))" for q in r.q], ["CD joint"])
    pvs = vcat(collect(qpvs), [cd])
    branch = r.robust ? "robust" : "homoskedastic"
    wnote = r.wright ? ", Wright R/S" : ""
    _pvalue_bar_plot("varianceratio", labels, pvs, "Variance-Ratio Random-Walk Test",
        "$(branch)$(wnote), n=$(r.nobs)";
        title=title, save_path=save_path, level=level)
end

"""
    plot_result(r::GrangerCausalityResult; title="", save_path=nothing, level=5)

Granger causality Wald test: the χ² statistic against 1/5/10% CVs
(right-tailed: above CV rejects non-causality). Causing/effect variables, lag
order, and test type are stated in the subtitle alongside the reject decision
at the `level`% CV (default 5).
"""
function plot_result(r::GrangerCausalityResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    _chisq_bar_plot("granger", "Wald χ² statistic", r.statistic, r.df,
        "Granger Causality Test",
        "var $(r.cause) → var $(r.effect), p=$(r.p), $(r.test_type)";
        title=title, save_path=save_path, level=level)
end

"""
    plot_result(r::VECMGrangerResult; title="", save_path=nothing, level=5)

VECM Granger causality: short-run (Γ), long-run (α), and joint Wald χ²
statistics, each against its own-df `level`% CV (default 5%; right-tailed).
Causing/effect variables are stated in the subtitle alongside the rejection
count.
"""
function plot_result(r::VECMGrangerResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    α = level == 1 ? 0.01 : level == 10 ? 0.10 : 0.05
    labels = ["Short-run (Γ)", "Long-run (α)", "Strong (joint)"]
    stats = [r.short_run_stat, r.long_run_stat, r.strong_stat]
    dfs = [r.short_run_df, r.long_run_df, r.strong_df]
    cvs = [quantile(Chisq(df), 1 - α) for df in dfs]
    id = _next_plot_id("vecmgranger")
    js = _render_bar_js(id,
        _grouped_stat_cv_json(labels, stats, cvs),
        _series_json(["Statistic", "$(level)% CV"], [_PLOT_SERIES[1], _PLOT_ALERT];
                     keys=["stat", "cv"]);
        mode="grouped", orientation="v", xlabel="Test", ylabel="Wald χ²")
    nrej = count(i -> stats[i] > cvs[i], eachindex(stats))
    isempty(title) && (title = "VECM Granger Causality Test")
    ptitle = "var $(r.cause_var) → var $(r.effect_var) — " *
             "$(nrej)/3 reject H₀ at $(level)%"
    p = _make_plot([_PanelSpec(id, ptitle, js)]; title=title)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# Specification / diagnostic tests (#841 PR5)
# =============================================================================

"""
    plot_result(r::LMTestResult; title="", save_path=nothing, level=5)

Lagrange-multiplier (score) test: the LM statistic against χ² 1/5/10% CVs
(right-tailed: above CV rejects the restrictions). The score norm and sample
size are stated in the subtitle alongside the reject decision at the `level`%
CV (default 5).
"""
function plot_result(r::LMTestResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    _chisq_bar_plot("lmtest", "LM statistic", r.statistic, r.df,
        "LM Specification Test", "‖s‖=$(_fmt(r.score_norm; digits=3)), n=$(r.nobs)";
        title=title, save_path=save_path, level=level)
end

"""
    plot_result(r::LRTestResult; title="", save_path=nothing, level=5)

Likelihood-ratio test: the LR statistic against χ² 1/5/10% CVs (right-tailed).
Restricted/unrestricted parameter counts and sample size are stated in the
subtitle alongside the reject decision at the `level`% CV (default 5).
"""
function plot_result(r::LRTestResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    _chisq_bar_plot("lrtest", "LR statistic", r.statistic, r.df,
        "LR Specification Test",
        "dof $(r.dof_restricted)→$(r.dof_unrestricted), n=$(r.nobs_unrestricted)";
        title=title, save_path=save_path, level=level)
end

"""
    plot_result(r::EqualityTestResult; title="", save_path=nothing, level=5)

Grouped equality-of-distribution test: the p-value as a bar against `α`
(default 5%). Member statistics live on different nulls (t/F/U/V/H/χ²), so
the p-value is the comparable quantity. Test name, method detail, and group
count are stated in the subtitle.
"""
function plot_result(r::EqualityTestResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    _pvalue_bar_plot("equality", [String(r.test_name)], [r.pvalue],
        "Equality-of-Distribution Test",
        "$(r.test_name) ($(r.detail)), $(r.n_groups) groups";
        title=title, save_path=save_path, level=level)
end

"""
    plot_result(r::HansenInstabilityResult; title="", save_path=nothing, level=5)

Hansen parameter-instability test for a cointegrating regression: the `L_c`
p-value as a bar against `α` (default 5%). `L_c` follows a nonstandard
distribution with no stored CVs, so the bracketing p-value is the plottable
quantity. Deterministic cases and parameter counts are stated in the subtitle
alongside the statistic value.
"""
function plot_result(r::HansenInstabilityResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    _pvalue_bar_plot("hanseninst", ["L_c"], [r.pvalue],
        "Hansen Instability Test",
        "L_c=$(_fmt(r.statistic; digits=3)), $(r.regression)/$(r.trend), " *
        "params=$(r.nparam), k=$(r.k)";
        title=title, save_path=save_path, level=level)
end

"""
    plot_result(r::NormalityTestResult; title="", save_path=nothing, level=5)

Multivariate normality test: overall + per-component p-values as bars against
`α` (default 5%). Member tests span χ² and non-χ² nulls, so p-values are the
comparable quantity. Test name, degrees of freedom, and variable count are
stated in the subtitle.
"""
function plot_result(r::NormalityTestResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    labels = ["overall"]
    pvs = [r.pvalue]
    if r.component_pvalues !== nothing
        append!(labels, ["comp $i" for i in eachindex(r.component_pvalues)])
        append!(pvs, collect(r.component_pvalues))
    end
    _pvalue_bar_plot("normality", labels, pvs, "Multivariate Normality Test",
        "$(r.test_name), df=$(r.df), vars=$(r.n_vars)";
        title=title, save_path=save_path, level=level)
end

"""
    plot_result(r::FactorBreakResult; title="", save_path=nothing, level=5)

Factor-model structural-break test: the sup-statistic p-value as a bar against
`α` (default 5%). Sup statistics follow non-χ² nulls with no stored CVs, so
the p-value is the plottable quantity. Method, break date, and factor count
are stated in the subtitle.
"""
function plot_result(r::FactorBreakResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    bd = r.break_date === nothing ? "none" : string(r.break_date)
    _pvalue_bar_plot("factorbreak", ["sup"], [r.pvalue],
        "Factor-Model Break Test",
        "$(r.method), break=$(bd), factors=$(r.n_factors), n=$(r.nobs)";
        title=title, save_path=save_path, level=level)
end

"""
    plot_result(r::PanelTestResult; title="", save_path=nothing, level=5)

Panel specification test: the p-value as a bar against `α` (default 5%). The
dispatch covers heterogeneous panel tests with no common null scale, so the
p-value is the comparable quantity. Test name and description are stated in
the subtitle.
"""
function plot_result(r::PanelTestResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    _pvalue_bar_plot("paneltest", [r.test_name], [r.pvalue],
        "Panel Specification Test", r.description;
        title=title, save_path=save_path, level=level)
end

"""
    plot_result(r::PVARTestResult; title="", save_path=nothing, level=5)

Panel-VAR specification (Hansen J) test: the statistic against χ² 1/5/10% CVs
(right-tailed: above CV rejects the over-identifying restrictions).
Instrument/parameter counts are stated in the subtitle alongside the reject
decision at the `level`% CV (default 5).
"""
function plot_result(r::PVARTestResult{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    _chisq_bar_plot("pvartest", "J statistic", r.statistic, r.df,
        "Panel-VAR Specification Test",
        "$(r.test_name): $(r.n_instruments) instruments, $(r.n_params) params";
        title=title, save_path=save_path, level=level)
end

"""
    plot_result(r::VECMRestrictionTest; title="", save_path=nothing, level=5)

Johansen LR test of a cointegrating-structure restriction: the LR statistic
against χ² 1/5/10% CVs (right-tailed: above CV rejects the restriction).
Restriction kind and rank are stated in the subtitle alongside the reject
decision at the `level`% CV (default 5).
"""
function plot_result(r::VECMRestrictionTest{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    _chisq_bar_plot("vecmrestr", "LR statistic", r.lr_stat, r.df,
        "VECM Restriction Test", "$(r.kind), rank=$(r.rank)";
        title=title, save_path=save_path, level=level)
end

"""
    plot_result(r::VARStationarityResult; title="", save_path=nothing)

VAR stationarity check: companion eigenvalues at `(Re λ, Im λ)` inside a
unit-circle reference (mirrors the PVAR stability scatter). Roots inside the
circle are stable; roots on/outside it (alert color) violate stationarity. The
panel subtitle states stationarity and the largest modulus. There is no
statistic or p-value on this diagnostic, so no `level` keyword applies.
"""
function plot_result(r::VARStationarityResult{T,E}; title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T,E}
    id = _next_plot_id("varstab")
    rows = Vector{Pair{String,String}}[]
    for lambda in r.eigenvalues
        m = abs(lambda)
        grp = (isfinite(m) && m < 1) ? "Inside unit circle" : "On/outside unit circle"
        push!(rows, ["x" => _json(real(lambda)),
                     "y" => _json(imag(lambda)),
                     "group" => _json(grp)])
    end
    data_json = _json_array_of_objects(rows)
    groups_json = "[{\"name\":\"Inside unit circle\",\"color\":$(_json(_PLOT_SERIES[1]))}," *
                  "{\"name\":\"On/outside unit circle\",\"color\":$(_json(_PLOT_ALERT))}]"
    ref_shapes = "[{\"type\":\"circle\",\"cx\":0,\"cy\":0,\"r\":1," *
                 "\"color\":\"#999\",\"dash\":\"4,3\"}]"
    ref_lines = "[{\"value\":0,\"axis\":\"x\",\"color\":\"#bbb\",\"dash\":\"2,2\"}," *
                "{\"value\":0,\"axis\":\"y\",\"color\":\"#bbb\",\"dash\":\"2,2\"}]"
    js = _render_scatter_js(id, data_json, groups_json;
                            ref_lines_json=ref_lines, ref_shapes_json=ref_shapes,
                            xlabel="Real", ylabel="Imaginary")
    status = r.is_stationary ? "STATIONARY" : "NON-STATIONARY"
    ptitle = "Companion eigenvalues ($(status) — max |λ| = $(_fmt(r.max_modulus; digits=3)))"
    isempty(title) && (title = "VAR Stationarity Check")
    p = _make_plot([_PanelSpec(id, ptitle, js)]; title=title)
    save_path !== nothing && save_plot(p, save_path)
    p
end
