# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
plot_result methods for cross-sectional regression types: RegModel, LogitModel,
ProbitModel, MarginalEffects.
"""

# =============================================================================
# RegModel — OLS / WLS / IV Regression Diagnostics
# =============================================================================

"""
    plot_result(m::RegModel{T}; view=:diagnostics, acf_lags=0, title="", save_path=nothing)

Plot OLS/WLS/IV regression residual diagnostics as the shared four-panel figure
(PLT-24): residual-vs-fitted scatter, residual histogram + fitted-normal overlay,
Normal Q-Q (with an A4 45° `line_overlays_json` reference line — no scale-clone), and
the residual ACF. All panels come from the single `_residual_diagnostics_panels`
converter reused across every `residuals`-bearing family (A6).
"""
function plot_result(m::RegModel{T}; view::Symbol=:diagnostics, acf_lags::Int=0,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    view === :diagnostics ||
        throw(ArgumentError("Unknown view :$view for RegModel; use :diagnostics."))
    resid = Float64[Float64(v) for v in m.residuals]
    fitted = Float64[Float64(v) for v in m.fitted]
    panels = _residual_diagnostics_panels(resid, fitted; acf_lags=acf_lags)

    if isempty(title)
        method_str = m.method == :ols ? "OLS" : m.method == :wls ? "WLS" : "IV/2SLS"
        title = "$method_str Regression Diagnostics"
    end

    p = _make_plot(panels; title=title, ncols=2)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# Binary Choice Shared Helper
# =============================================================================

"""
Generate 2-panel binary choice model diagnostic plot:
1. Sorted predicted probabilities colored by outcome
2. Distribution of predicted probabilities by outcome (grouped bar)
"""
function _plot_binary_choice(y::AbstractVector{T}, fitted_probs::AbstractVector{T},
                              model_name::String; title::String="",
                              save_path::Union{String,Nothing}=nothing) where {T}
    n = length(y)

    # Panel 1: Sorted predicted probabilities colored by outcome
    id1 = _next_plot_id("bc_sort")
    order = sortperm(fitted_probs)
    scatter_rows = String[]
    for (rank, idx) in enumerate(order)
        grp = y[idx] > T(0.5) ? "y = 1" : "y = 0"
        push!(scatter_rows, "{\"x\":$(_json(rank)),\"y\":$(_json(fitted_probs[idx])),\"group\":$(_json(grp))}")
    end
    data1 = "[" * join(scatter_rows, ",\n") * "]"
    # Outcome colors from the red-excluded series palette — red stays reserved for
    # reference elements, never a series (plotrule Color: reserved red; PLT-13).
    groups1 = "[{\"name\":\"y = 1\",\"color\":\"$(_PLOT_SERIES[1])\"},{\"name\":\"y = 0\",\"color\":\"$(_PLOT_SERIES[2])\"}]"
    refs1 = "[{\"value\":0.5,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js1 = _render_scatter_js(id1, data1, groups1;
                             ref_lines_json=refs1,
                             xlabel="Observation (sorted by P\u0302)",
                             ylabel="Predicted Probability")
    p1 = _PanelSpec(id1, "Sorted Predicted Probabilities", js1)

    # Panel 2: Distribution by outcome (grouped bar)
    id2 = _next_plot_id("bc_dist")
    n_bins = 10
    bin_counts_0 = zeros(Int, n_bins)
    bin_counts_1 = zeros(Int, n_bins)
    for i in 1:n
        # Map probability to bin [1, n_bins]
        b = clamp(floor(Int, fitted_probs[i] * n_bins) + 1, 1, n_bins)
        if y[i] > T(0.5)
            bin_counts_1[b] += 1
        else
            bin_counts_0[b] += 1
        end
    end
    bar_rows = Vector{Pair{String,String}}[]
    for b in 1:n_bins
        lo = (b - 1) / n_bins
        hi = b / n_bins
        label = string(round((lo + hi) / 2; digits=2))
        push!(bar_rows, ["x" => _json(label), "s0" => _json(bin_counts_0[b]), "s1" => _json(bin_counts_1[b])])
    end
    data2 = _json_array_of_objects(bar_rows)
    s2 = _series_json(["y = 0", "y = 1"], [_PLOT_SERIES[2], _PLOT_SERIES[1]]; keys=["s0", "s1"])
    js2 = _render_bar_js(id2, data2, s2; mode="grouped",
                         xlabel="Predicted Probability", ylabel="Count")
    p2 = _PanelSpec(id2, "Distribution by Outcome", js2)

    if isempty(title)
        title = "$model_name Model Diagnostics"
    end

    p = _make_plot([p1, p2]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# LogitModel
# =============================================================================

"""
    plot_result(m::LogitModel{T}; title="", save_path=nothing)

Plot logit model diagnostics: sorted predicted probabilities by outcome and
distribution of predictions by outcome group.
"""
function plot_result(m::LogitModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    _plot_binary_choice(m.y, m.fitted, "Logit"; title=title, save_path=save_path)
end

# =============================================================================
# ProbitModel
# =============================================================================

"""
    plot_result(m::ProbitModel{T}; title="", save_path=nothing)

Plot probit model diagnostics: sorted predicted probabilities by outcome and
distribution of predictions by outcome group.
"""
function plot_result(m::ProbitModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    _plot_binary_choice(m.y, m.fitted, "Probit"; title=title, save_path=save_path)
end

# =============================================================================
# MarginalEffects
# =============================================================================

"""
    plot_result(me::MarginalEffects{T}; title="", save_path=nothing)

Plot marginal effects as a horizontal coefficient plot with CI error bars.
"""
function plot_result(me::MarginalEffects{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    id = _next_plot_id("me_coef")

    # Build data JSON (skip the intercept, whose marginal effect is NaN)
    rows = String[]
    for i in 1:length(me.effects)
        isfinite(me.effects[i]) || continue
        push!(rows, "{\"name\":$(_json(me.varnames[i])),\"effect\":$(_json(me.effects[i])),\"ci_lo\":$(_json(me.ci_lower[i])),\"ci_hi\":$(_json(me.ci_upper[i]))}")
    end
    data_json = "[" * join(rows, ",\n") * "]"

    pct = round(Int, me.conf_level * 100)
    js = _render_coef_plot_js(id, data_json;
                               xlabel="Effect Size",
                               ylabel="")

    type_str = me.type == :ame ? "Average Marginal Effects" :
               me.type == :mem ? "Marginal Effects at Mean" :
               "Marginal Effects at Representative"
    ptitle = "$type_str ($(pct)% CI)"
    panel = _PanelSpec(id, ptitle, js)

    if isempty(title)
        title = type_str
    end

    p = _make_plot([panel]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# StabilityResult — CUSUM / CUSUM-of-squares path with significance bounds (EV-32)
# =============================================================================

"""
    plot_result(r::StabilityResult{T}; title="", save_path=nothing)

Plot a recursive-residual stability test: the CUSUM (`:cusum`) or CUSUM-of-squares
(`:cusumsq`) statistic path against its upper/lower significance-band lines. The path
straying outside the dashed bounds signals coefficient instability.
"""
function plot_result(r::StabilityResult{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    id = _next_plot_id(r.kind == :cusum ? "cusum" : "cusumsq")
    rows = Vector{Pair{String,String}}[]
    for i in eachindex(r.tindex)
        push!(rows, [
            "x" => _json(r.tindex[i]),
            "stat" => _json(r.stat_path[i]),
            "upper" => _json(r.upper[i]),
            "lower" => _json(r.lower[i]),
        ])
    end
    data = _json_array_of_objects(rows)
    stat_name = r.kind == :cusum ? "CUSUM" : "CUSUM²"
    lvl_pct = round(Int, 100 * r.level)
    series = _series_json(
        [stat_name, "$(lvl_pct)% bound", "$(lvl_pct)% bound"],
        [_PLOT_COLORS[1], "#d62728", "#d62728"];
        keys = ["stat", "upper", "lower"],
        dash = ["", "6,3", "6,3"])
    ylabel = r.kind == :cusum ? "CUSUM statistic" : "CUSUM of squares"
    js = _render_line_js(id, data, series; xlabel = "Observation", ylabel = ylabel)
    ptitle = (r.kind == :cusum ? "CUSUM Test" : "CUSUM of Squares Test") *
             (r.crossed ? " — bound crossed" : " — stable")
    panel = _PanelSpec(id, ptitle, js)
    isempty(title) && (title = r.kind == :cusum ? "CUSUM Stability Test" :
                                                  "CUSUM of Squares Stability Test")
    p = _make_plot([panel]; title = title, ncols = 1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# InfluenceStats — leverage & Cook's distance (EV-32)
# =============================================================================

"""
    plot_result(s::InfluenceStats{T}; title="", save_path=nothing)

Plot observation-level influence diagnostics: a leverage scatter (`h_ii` vs
observation, with the `2k/n` reference line) and a Cook's distance bar panel.
"""
function plot_result(s::InfluenceStats{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    n = s.n
    # Panel 1: leverage scatter with 2k/n reference line.
    id1 = _next_plot_id("infl_lev")
    scatter_rows = String[]
    hi_cut = T(2 * s.k) / T(n)
    for i in 1:n
        grp = s.hat[i] > hi_cut ? "High leverage" : "Leverage"
        push!(scatter_rows, "{\"x\":$(_json(i)),\"y\":$(_json(s.hat[i])),\"group\":$(_json(grp))}")
    end
    data1 = "[" * join(scatter_rows, ",\n") * "]"
    # "High leverage" is an alert category (the observations being flagged), so it
    # legitimately keeps the reserved alert red; "Leverage" uses a series hue (PLT-13).
    groups1 = "[{\"name\":\"Leverage\",\"color\":\"$(_PLOT_SERIES[1])\"}," *
              "{\"name\":\"High leverage\",\"color\":\"$(_PLOT_ALERT)\"}]"
    refs1 = "[{\"value\":$(_json(hi_cut)),\"color\":\"#d62728\",\"dash\":\"6,3\"}]"
    js1 = _render_scatter_js(id1, data1, groups1; ref_lines_json = refs1,
                             xlabel = "Observation", ylabel = "Leverage (h_ii)")
    p1 = _PanelSpec(id1, "Leverage (2k/n cutoff)", js1)

    # Panel 2: Cook's distance bar panel.
    id2 = _next_plot_id("infl_cook")
    bar_rows = Vector{Pair{String,String}}[]
    for i in 1:n
        push!(bar_rows, ["x" => _json(i), "s1" => _json(s.cooksd[i])])
    end
    data2 = _json_array_of_objects(bar_rows)
    s2 = _series_json(["Cook's D"], [_PLOT_COLORS[2]]; keys = ["s1"])
    js2 = _render_bar_js(id2, data2, s2; mode = "stacked",
                         xlabel = "Observation", ylabel = "Cook's distance")
    p2 = _PanelSpec(id2, "Cook's Distance", js2)

    isempty(title) && (title = "Influence Diagnostics")
    p = _make_plot([p1, p2]; title = title, ncols = 1)
    save_path !== nothing && save_plot(p, save_path)
    p
end
