# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
plot_result method for PanelData (relocated out of models.jl in the PLT plotting
overhaul). The per-(group, time) lookup is pre-indexed once into a
`Dict{Tuple{Int,Int},Int}` instead of an O(N·T·G) `findfirst` scan of the full
panel inside the time×group loop (plotrule Robustness "pre-index once").
"""

# =============================================================================
# PanelData
# =============================================================================

"""
    plot_result(d::PanelData; vars=nothing, dates=nothing, ncols=0, title="", save_path=nothing)

Plot panel data: one panel per variable, with lines for each group.

- `vars`: Variable indices or names to plot. `nothing` = all (up to 6).
- `dates`: optional calendar labels parallel to the sorted unique `time_id` values;
  when supplied the x-axis shows these labels, otherwise the integer time ids.
"""
function plot_result(d::PanelData{T};
                     vars::Union{Vector,Nothing}=nothing,
                     dates::Union{Vector{String},Nothing}=nothing,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    if vars === nothing
        var_idxs = collect(1:min(d.n_vars, 6))
    else
        var_idxs = [v isa String ? _resolve_var(v, d.varnames) : v for v in vars]
    end

    # Group data by group_id
    unique_groups = sort(unique(d.group_id))
    n_groups_plot = min(length(unique_groups), 10)
    groups_to_plot = unique_groups[1:n_groups_plot]

    # Pre-index (group, time) → row ONCE (avoids the O(N·T·G) findfirst scan).
    idx_map = Dict{Tuple{Int,Int},Int}()
    for r in eachindex(d.group_id)
        idx_map[(d.group_id[r], d.time_id[r])] = r
    end

    unique_times = sort(unique(d.time_id))

    # Date axis (PLT-08): labels parallel to the sorted unique times, else integers.
    x_ticks_json = "null"
    xlabel = "Time"
    if dates !== nothing
        length(dates) == length(unique_times) || throw(ArgumentError(
            "dates length ($(length(dates))) must match the number of unique time ids " *
            "($(length(unique_times)))"))
        x_ticks_json = _x_ticks_json(unique_times, dates)
        xlabel = "Date"
    end

    panels = _PanelSpec[]
    for vi in var_idxs
        id = _next_plot_id("pd")
        ptitle = d.varnames[vi]

        # Build data with one column per group
        rows = Vector{Pair{String,String}}[]
        for t in unique_times
            row = Pair{String,String}["x" => _json(t)]
            for (gi, gid) in enumerate(groups_to_plot)
                idx = get(idx_map, (gid, t), nothing)
                val = idx !== nothing ? d.data[idx, vi] : NaN
                push!(row, "g$gi" => _json(val))
            end
            push!(rows, row)
        end
        data_json = _json_array_of_objects(rows)

        group_names = [gi <= length(d.group_names) ? d.group_names[gi] : "Group $gid"
                       for (gi, gid) in enumerate(groups_to_plot)]
        s_json = _series_json(group_names, _PLOT_COLORS[1:n_groups_plot];
                              keys=["g$i" for i in 1:n_groups_plot])

        js = _render_line_js(id, data_json, s_json; xlabel=xlabel, ylabel="",
                             x_ticks_json=x_ticks_json)
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        title = isempty(desc(d)) ? "Panel Data" : desc(d)
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end
