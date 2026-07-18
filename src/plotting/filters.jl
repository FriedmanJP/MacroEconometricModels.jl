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
Generate a 2-panel figure for a filter result: (1) Original + Trend, (2) Cycle.

- `tr`: trend component
- `cyc`: cycle component
- `filter_name`: display name
- `original`: original series (optional). When absent and `offset > 0` (Hamilton/BK
  drop leading observations) the pre-trim original **cannot** be recovered from the
  trimmed trend/cycle, so the Original line is **omitted** rather than fabricated
  (plotrule Anti-Pattern #5); a subtitle notes it. When absent and `offset == 0` it
  is reconstructed as `trend + cycle` and labelled "Original (reconstructed)".
- `offset`: index offset for shorter filters (Hamilton, BK)
- `T_obs`: original series length
- `dates`: optional calendar labels (length ≥ `offset + n`) aligned to the *full*
  series so each drawn point shows its calendar position (PLT-08).
"""
function _plot_filter_panels(tr::AbstractVector, cyc::AbstractVector,
                             filter_name::String;
                             original::Union{AbstractVector,Nothing}=nothing,
                             offset::Int=0, T_obs::Int=0,
                             dates::Union{AbstractVector,Nothing}=nothing)
    n = length(tr)
    reconstructed = false
    if original !== nothing
        orig = original
        has_original = true
    elseif offset == 0
        orig = tr .+ cyc
        has_original = true
        reconstructed = true
    else
        orig = nothing
        has_original = false
    end

    data_json = _filter_data_json(tr, cyc; original=orig, offset=offset)

    # Date axis: label calendar positions offset+1 .. offset+n with dates.
    xlabel = "Period"
    x_ticks_json = "null"
    if dates !== nothing
        if length(dates) < offset + n
            throw(ArgumentError("dates has length $(length(dates)); expected at least " *
                "$(offset + n) to cover the filter's calendar range (offset=$offset, n=$n)"))
        end
        xvals = collect((1:n) .+ offset)
        x_ticks_json = _x_ticks_json(xvals, dates[(1:n) .+ offset])
        xlabel = "Date"
    end

    # Panel 1: Original (or reconstructed) + Trend
    id1 = _next_plot_id("filt_tc")
    if has_original
        orig_label = reconstructed ? "Original (reconstructed)" : "Original"
        s1 = _series_json([orig_label, "Trend"], [_PLOT_COLORS[1], _PLOT_COLORS[2]];
                          keys=["orig", "trend"], dash=["", "6,3"])
    else
        s1 = _series_json(["Trend"], [_PLOT_COLORS[2]]; keys=["trend"], dash=["6,3"])
    end
    js1 = _render_line_js(id1, data_json, s1; xlabel=xlabel, ylabel="Value",
                          x_ticks_json=x_ticks_json)
    subtitle1 = if !has_original
        "Trend-Cycle Decomposition (original not supplied; pass original= to overlay it)"
    elseif reconstructed
        "Trend-Cycle Decomposition (original = trend + cycle)"
    else
        "Trend-Cycle Decomposition"
    end
    p1 = _PanelSpec(id1, subtitle1, js1)

    # Panel 2: Cycle + zero line
    id2 = _next_plot_id("filt_cy")
    s2 = _series_json(["Cycle"], [_PLOT_COLORS[3]]; keys=["cycle"])
    refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js2 = _render_line_js(id2, data_json, s2;
                          ref_lines_json=refs, xlabel=xlabel, ylabel="Cycle",
                          x_ticks_json=x_ticks_json)
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
                     dates::Union{Vector{String},Nothing}=nothing,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    panels = _plot_filter_panels(r.trend, r.cycle, "HP Filter";
                                 original=r.trend .+ r.cycle, dates=dates)
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
                     dates::Union{Vector{String},Nothing}=nothing,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    offset = r.valid_range.start - 1
    panels = _plot_filter_panels(r.trend, r.cycle, "Hamilton Filter";
                                 original=original, offset=offset,
                                 T_obs=r.T_obs, dates=dates)
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
                     dates::Union{Vector{String},Nothing}=nothing,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    panels = _plot_filter_panels(r.permanent, r.transitory, "BN Decomposition";
                                 original=r.permanent .+ r.transitory, dates=dates)
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
                     dates::Union{Vector{String},Nothing}=nothing,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    offset = r.valid_range.start - 1
    panels = _plot_filter_panels(r.trend, r.cycle, "BK Filter";
                                 original=original, offset=offset,
                                 T_obs=r.T_obs, dates=dates)
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
                     dates::Union{Vector{String},Nothing}=nothing,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    panels = _plot_filter_panels(r.trend, r.cycle, "Boosted HP";
                                 original=r.trend .+ r.cycle, dates=dates)
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
