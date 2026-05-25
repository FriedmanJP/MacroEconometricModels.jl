# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
plot_result methods for filter types: HPFilter, Hamilton, BN, BK, BoostedHP.
"""

# =============================================================================
# Shared Filter Plotting
# =============================================================================

"""
Generate 2-panel Figure for a filter result: (1) Original + Trend, (2) Cycle.

- `tr`: trend component
- `cyc`: cycle component
- `filter_name`: display name
- `original`: original series (optional, reconstructed from trend+cycle if nil)
- `offset`: index offset for shorter filters (Hamilton, BK)
- `T_obs`: original series length
"""
function _plot_filter_panels(tr::AbstractVector, cyc::AbstractVector,
                             filter_name::String;
                             original::Union{AbstractVector,Nothing}=nothing,
                             offset::Int=0, T_obs::Int=0)
    orig = original !== nothing ? original : tr .+ cyc
    data_json = _filter_data_json(tr, cyc; original=orig, offset=offset)

    # Panel 1: Original + Trend
    id1 = _next_plot_id("filt_tc")
    s1 = _series_json(["Original", "Trend"], [_PLOT_COLORS[1], _PLOT_COLORS[2]];
                       keys=["orig", "trend"], dash=["", "6,3"])
    js1 = _render_line_js(id1, data_json, s1; xlabel="Period", ylabel="Value")
    p1 = _PanelSpec(id1, "Trend-Cycle Decomposition", js1)

    # Panel 2: Cycle + zero line
    id2 = _next_plot_id("filt_cy")
    s2 = _series_json(["Cycle"], [_PLOT_COLORS[3]]; keys=["cycle"])
    refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js2 = _render_line_js(id2, data_json, s2;
                          ref_lines_json=refs, xlabel="Period", ylabel="Cycle")
    p2 = _PanelSpec(id2, "Cyclical Component", js2)

    [p1, p2]
end

# =============================================================================
# HPFilterResult
# =============================================================================

"""
    plot_result(r::HPFilterResult; title="", save_path=nothing)

Plot HP filter: original+trend and cycle.
"""
function plot_result(r::HPFilterResult{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    panels = _plot_filter_panels(r.trend, r.cycle, "HP Filter";
                                 original=r.trend .+ r.cycle)
    if isempty(title)
        title = "Hodrick-Prescott Filter (λ=$(round(r.lambda, sigdigits=4)))"
    end
    p = _make_plot(panels; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# HamiltonFilterResult
# =============================================================================

"""
    plot_result(r::HamiltonFilterResult; original=nothing, title="", save_path=nothing)

Plot Hamilton filter: original+trend and cycle.

Pass the original series via `original` kwarg since HamiltonFilterResult
doesn't store it.
"""
function plot_result(r::HamiltonFilterResult{T};
                     original::Union{AbstractVector,Nothing}=nothing,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    offset = r.valid_range.start - 1
    panels = _plot_filter_panels(r.trend, r.cycle, "Hamilton Filter";
                                 original=original, offset=offset,
                                 T_obs=r.T_obs)
    if isempty(title)
        title = "Hamilton (2018) Filter (h=$(r.h), p=$(r.p))"
    end
    p = _make_plot(panels; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# BeveridgeNelsonResult
# =============================================================================

"""
    plot_result(r::BeveridgeNelsonResult; title="", save_path=nothing)

Plot BN decomposition: permanent+original and transitory.
"""
function plot_result(r::BeveridgeNelsonResult{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    panels = _plot_filter_panels(r.permanent, r.transitory, "BN Decomposition";
                                 original=r.permanent .+ r.transitory)
    if isempty(title)
        p, d, q = r.arima_order
        title = "Beveridge-Nelson Decomposition (ARIMA($p,$d,$q))"
    end
    p = _make_plot(panels; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# BaxterKingResult
# =============================================================================

"""
    plot_result(r::BaxterKingResult; original=nothing, title="", save_path=nothing)

Plot BK band-pass filter.
"""
function plot_result(r::BaxterKingResult{T};
                     original::Union{AbstractVector,Nothing}=nothing,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    offset = r.valid_range.start - 1
    panels = _plot_filter_panels(r.trend, r.cycle, "BK Filter";
                                 original=original, offset=offset,
                                 T_obs=r.T_obs)
    if isempty(title)
        title = "Baxter-King Band-Pass Filter ([$(r.pl), $(r.pu)], K=$(r.K))"
    end
    p = _make_plot(panels; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# BoostedHPResult
# =============================================================================

"""
    plot_result(r::BoostedHPResult; title="", save_path=nothing)

Plot boosted HP filter.
"""
function plot_result(r::BoostedHPResult{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    panels = _plot_filter_panels(r.trend, r.cycle, "Boosted HP";
                                 original=r.trend .+ r.cycle)
    if isempty(title)
        title = "Boosted HP Filter ($(r.stopping), $(r.iterations) iter)"
    end
    p = _make_plot(panels; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# X13FilterResult
# =============================================================================

"""
    plot_result(r::X13FilterResult; title="", save_path=nothing)

Plot X-13 seasonal adjustment: (1) Original + Trend + SA, (2) Seasonal, (3) Irregular.
"""
function plot_result(r::X13FilterResult{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    n = r.T_obs

    # Panel 1: Original + Trend + SA
    rows1 = Vector{Pair{String,String}}[]
    for i in 1:n
        push!(rows1, ["x" => _json(i), "orig" => _json(r.original[i]),
                       "trend" => _json(r.trend[i]), "sa" => _json(r.adjusted[i])])
    end
    data1 = _json_array_of_objects(rows1)
    id1 = _next_plot_id("x13_tc")
    s1 = _series_json(["Original", "Trend", "Seasonally Adjusted"],
                       [_PLOT_COLORS[1], _PLOT_COLORS[2], _PLOT_COLORS[4]];
                       keys=["orig", "trend", "sa"], dash=["", "6,3", "3,3"])
    js1 = _render_line_js(id1, data1, s1; xlabel="Period", ylabel="Value")
    p1 = _PanelSpec(id1, "Trend-Cycle Decomposition", js1)

    # Panel 2: Seasonal component
    rows2 = Vector{Pair{String,String}}[]
    for i in 1:n
        push!(rows2, ["x" => _json(i), "seasonal" => _json(r.seasonal[i])])
    end
    data2 = _json_array_of_objects(rows2)
    id2 = _next_plot_id("x13_seas")
    s2 = _series_json(["Seasonal"], [_PLOT_COLORS[3]]; keys=["seasonal"])
    ref_val = r.transform == :log ? "1" : "0"
    refs2 = "[{\"value\":$ref_val,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js2 = _render_line_js(id2, data2, s2; ref_lines_json=refs2,
                          xlabel="Period", ylabel="Seasonal")
    p2 = _PanelSpec(id2, "Seasonal Component", js2)

    # Panel 3: Irregular component
    rows3 = Vector{Pair{String,String}}[]
    for i in 1:n
        push!(rows3, ["x" => _json(i), "irregular" => _json(r.irregular[i])])
    end
    data3 = _json_array_of_objects(rows3)
    id3 = _next_plot_id("x13_irr")
    s3 = _series_json(["Irregular"], [_PLOT_COLORS[5]]; keys=["irregular"])
    js3 = _render_line_js(id3, data3, s3; ref_lines_json=refs2,
                          xlabel="Period", ylabel="Irregular")
    p3 = _PanelSpec(id3, "Irregular Component", js3)

    if isempty(title)
        p_o, d_o, q_o, P_o, D_o, Q_o = r.arima_order
        title = "X-13ARIMA-SEATS ($(uppercase(string(r.method))), ($p_o,$d_o,$q_o)($P_o,$D_o,$Q_o)_$(r.frequency))"
    end
    p = _make_plot([p1, p2, p3]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end
