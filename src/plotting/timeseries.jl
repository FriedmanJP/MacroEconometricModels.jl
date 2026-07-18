# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
plot_result method for TimeSeriesData (relocated out of models.jl in the PLT
plotting overhaul so Wave-2 data-view lanes own this file). Adds PLT-08 date
axes: when `d.dates` is populated the x-axis shows calendar labels, otherwise it
falls back to the integer `time_index`.
"""

# =============================================================================
# TimeSeriesData
# =============================================================================

"""
    plot_result(d::TimeSeriesData; vars=nothing, ncols=0, title="", save_path=nothing)

Plot time series data: one panel per variable (capped at 12).

- `vars`: Variable indices or names to plot. `nothing` = all (up to 12).

When `d.dates` is set (via `set_dates!`) the x-axis is labelled with those calendar
strings and the axis title reads "Date"; otherwise the integer `time_index` is used
and the axis title reads "Time".
"""
function plot_result(d::TimeSeriesData{T};
                     vars::Union{Vector,Nothing}=nothing,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    if vars === nothing
        idxs = collect(1:min(d.n_vars, 12))
    else
        idxs = [v isa String ? _resolve_var(v, d.varnames) : v for v in vars]
    end

    # Date axis (PLT-08): calendar labels when d.dates is populated, else integers.
    labels = _date_axis_labels(d)
    x_ticks_json = _x_ticks_json(d.time_index, labels)
    xlabel = labels === nothing ? "Time" : "Date"

    panels = _PanelSpec[]
    for vi in idxs
        id = _next_plot_id("ts")
        ptitle = d.varnames[vi]

        rows = Vector{Pair{String,String}}[]
        for t in 1:d.T_obs
            push!(rows, [
                "x" => _json(d.time_index[t]),
                "v1" => _json(d.data[t, vi])
            ])
        end
        data_json = _json_array_of_objects(rows)
        s_json = _series_json([d.varnames[vi]], [_PLOT_COLORS[mod1(vi, length(_PLOT_COLORS))]];
                              keys=["v1"])

        js = _render_line_js(id, data_json, s_json; xlabel=xlabel, ylabel="",
                             x_ticks_json=x_ticks_json)
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        title = isempty(desc(d)) ? "Time Series Data" : desc(d)
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end
