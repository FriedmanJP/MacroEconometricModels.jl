# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
plot_result method for PanelData. The per-(group, time) lookup is pre-indexed once
into a `Dict{Tuple{Int,Int},Int}` (plotrule Robustness "pre-index once"; audit M18)
and reused by every view. PLT-21 adds the analysis views
(`:lines/:quantiles/:spaghetti/:groups/:scatter`) plus `view=:binscatter` (PLT-23);
`:lines` stays the default for back-compatibility.

Shared lane converters/builders (`_resolve_vars`, `_xy_scatter_json`, `_scatter2_panel`,
`_plot_binscatter`) are defined in timeseries.jl / binscatter.jl (same lane).
"""

# =============================================================================
# PanelData
# =============================================================================

const _PANEL_VIEWS = (:lines, :quantiles, :spaghetti, :groups, :scatter, :binscatter)

# View-specific group cap (0 = use the view default). :lines keeps the historical 10.
_panel_max_groups(view::Symbol, max_groups::Int) =
    max_groups > 0 ? max_groups :
    (view === :spaghetti ? 60 : view === :groups ? 12 : 10)

"""
    _panel_index(d) -> (idx_map, unique_groups, unique_times)

Pre-index `(group_id, time_id) → row` ONCE (kills the O(N·T·G) `findfirst` scan) and
return the sorted unique group ids and time ids alongside it.
"""
function _panel_index(d::PanelData)
    idx_map = Dict{Tuple{Int,Int},Int}()
    for r in eachindex(d.group_id)
        idx_map[(d.group_id[r], d.time_id[r])] = r
    end
    (idx_map, sort(unique(d.group_id)), sort(unique(d.time_id)))
end

"""Resolve a group selector (group_id `Int` value, or `String` group name) to a
1-based position in the sorted unique-group list (plotrule C3)."""
function _resolve_group(sel, unique_groups::Vector{Int}, group_names::Vector{String})
    if sel isa Integer
        pos = findfirst(==(Int(sel)), unique_groups)
        pos === nothing && throw(ArgumentError(
            "group id $sel not found; available ids: $unique_groups"))
        return pos
    else
        pos = findfirst(==(String(sel)), group_names)
        pos === nothing && throw(ArgumentError(
            "group '$sel' not found; available: $group_names"))
        return pos
    end
end

"""
    plot_result(d::PanelData; view=:lines, vars=nothing, dates=nothing, max_vars=6,
                max_groups=0, qs=[0.1,0.25,0.5,0.75,0.9], highlight=nothing,
                demean=:none, x=nothing, y=nothing, n_bins=0, controls=nothing,
                fit=:ols, ncols=0, title="", save_path=nothing)

Multi-view plot of a panel container. Unknown `view` throws (plotrule C5).

- `:lines` (default) — one panel per variable, a line per group (cap `max_groups`,
  default 10). `dates` (parallel to the sorted unique times) labels the x-axis (PLT-08).
- `:quantiles` — the cross-sectional quantile fan over time: for each variable a
  nested fan of the cross-group quantiles `qs` (median + IQR + outer band) with
  legend-labelled bands (PLT-16). The x values are the panel `time_id`s.
- `:spaghetti` — one thin line per group; `highlight=` (group ids or names) draws the
  named unit(s) in palette color, all others in neutral grey (cap `max_groups`,
  default 60, surfaced in a note).
- `:groups` — small multiples: one panel per group (cap `max_groups`, default 12),
  each a line of the selected variable(s).
- `:scatter` — a two-variable scatter (`vars=[x, y]`); `demean=:within` removes group
  means, `:between` collapses to one point per group, `:none` pools; the subtitle
  states the transform. `fit=:ols` overlays the least-squares line.
- `:binscatter` — quantile-binned means of `y` on `x` (see the binscatter docstring);
  `demean=:within` bins within-group.

`vars`/`highlight` accept names or indices; bad input throws (plotrule C3).
`save_path` writes the HTML and returns the `PlotOutput` (plotrule C8).
"""
function plot_result(d::PanelData{T};
                     view::Symbol=:lines,
                     vars::Union{Vector,Nothing}=nothing,
                     dates::Union{Vector{String},Nothing}=nothing,
                     max_vars::Int=6, max_groups::Int=0,
                     qs::Vector{Float64}=[0.1, 0.25, 0.5, 0.75, 0.9],
                     highlight::Union{Vector,Nothing}=nothing,
                     demean::Symbol=:none,
                     x=nothing, y=nothing, n_bins::Int=0,
                     controls::Union{Vector,Nothing}=nothing, fit::Symbol=:ols,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    view in _PANEL_VIEWS || throw(ArgumentError(
        "view=:$view not supported for PanelData; valid: $(collect(_PANEL_VIEWS))"))

    if view === :binscatter
        return _plot_binscatter(d.data, d.varnames, x, y;
                                n_bins=n_bins, controls=controls, fit=fit,
                                demean=demean, group_id=d.group_id,
                                title=title, ncols=ncols, save_path=save_path)
    elseif view === :scatter
        return _panel_scatter(d, vars, demean, fit, ncols, title, save_path)
    elseif view === :quantiles
        return _panel_quantiles(d, vars, qs, max_vars, ncols, title, save_path)
    elseif view === :spaghetti
        return _panel_spaghetti(d, vars, highlight, dates,
                                _panel_max_groups(view, max_groups),
                                ncols, title, save_path)
    elseif view === :groups
        return _panel_groups(d, vars, dates, max_vars,
                             _panel_max_groups(view, max_groups),
                             ncols, title, save_path)
    else  # :lines
        return _panel_lines(d, vars, dates, max_vars,
                            _panel_max_groups(view, max_groups),
                            ncols, title, save_path)
    end
end

# -----------------------------------------------------------------------------
# :lines (default) — one panel per variable, a line per group
# -----------------------------------------------------------------------------
function _panel_lines(d::PanelData{T}, vars, dates, max_vars::Int, max_groups::Int,
                      ncols::Int, title::String, save_path) where {T}
    n_vars_shown = min(d.n_vars, max_vars)
    var_idxs = vars === nothing ? collect(1:n_vars_shown) : _resolve_vars(vars, d.varnames)

    idx_map, unique_groups, unique_times = _panel_index(d)
    n_groups_total = length(unique_groups)
    n_groups_plot = min(n_groups_total, max_groups)
    groups_to_plot = unique_groups[1:n_groups_plot]

    x_ticks_json, xlabel = _panel_date_axis(dates, unique_times)

    panels = _PanelSpec[]
    for vi in var_idxs
        id = _next_plot_id("pd")
        rows = Vector{Pair{String,String}}[]
        for t in unique_times
            row = Pair{String,String}["x" => _json(t)]
            for (gi, gid) in enumerate(groups_to_plot)
                idx = get(idx_map, (gid, t), nothing)
                push!(row, "g$gi" => _json(idx !== nothing ? d.data[idx, vi] : NaN))
            end
            push!(rows, row)
        end
        data_json = _json_array_of_objects(rows)
        gnames = [gi <= length(d.group_names) ? d.group_names[gi] : "Group $gid"
                  for (gi, gid) in enumerate(groups_to_plot)]
        s_json = _series_json(gnames, _palette_take(n_groups_plot);
                              keys=["g$i" for i in 1:n_groups_plot])
        js = _render_line_js(id, data_json, s_json; xlabel=xlabel, ylabel="",
                             x_ticks_json=x_ticks_json)
        push!(panels, _PanelSpec(id, d.varnames[vi], js))
    end

    isempty(title) && (title = isempty(desc(d)) ? "Panel Data" : desc(d))
    notes = String[]
    gn = _cap_note("groups", n_groups_plot, n_groups_total, "max_groups")
    isempty(gn) || push!(notes, gn)
    if vars === nothing
        vn = _cap_note("variables", n_vars_shown, d.n_vars, "max_vars")
        isempty(vn) || push!(notes, vn)
    end
    _noted_plot(panels, title, ncols, join(notes, " "), save_path)
end

# Date-axis tick map for the time-based panel views (:lines/:spaghetti/:groups).
function _panel_date_axis(dates::Union{Vector{String},Nothing}, unique_times::Vector{Int})
    dates === nothing && return ("null", "Time")
    length(dates) == length(unique_times) || throw(ArgumentError(
        "dates length ($(length(dates))) must match the number of unique time ids " *
        "($(length(unique_times)))"))
    (_x_ticks_json(unique_times, dates), "Date")
end

# -----------------------------------------------------------------------------
# :quantiles — cross-sectional quantile fan over time
# -----------------------------------------------------------------------------
function _panel_quantiles(d::PanelData{T}, vars, qs::Vector{Float64}, max_vars::Int,
                          ncols::Int, title::String, save_path) where {T}
    all(q -> 0 <= q <= 1, qs) || throw(ArgumentError("qs must lie in [0,1]"))
    n_vars_shown = min(d.n_vars, max_vars)
    var_idxs = vars === nothing ? collect(1:n_vars_shown) : _resolve_vars(vars, d.varnames)

    idx_map, unique_groups, unique_times = _panel_index(d)
    nt = length(unique_times); nq = length(qs)

    panels = _PanelSpec[]
    for vi in var_idxs
        Q = fill(NaN, nt, nq)
        central = fill(NaN, nt)
        for (ti, t) in enumerate(unique_times)
            vals = Float64[]
            for gid in unique_groups
                r = get(idx_map, (gid, t), nothing)
                r === nothing && continue
                v = d.data[r, vi]
                isfinite(v) && push!(vals, Float64(v))
            end
            isempty(vals) && continue
            sorted = sort(vals)
            for (qi, q) in enumerate(qs)
                Q[ti, qi] = _quantile(sorted, q)
            end
            central[ti] = _quantile(sorted, 0.5)
        end
        id = _next_plot_id("pdq")
        fan_data = _fan_data_json(Q, qs, central; xvals=unique_times)
        fan_bands = _fan_bands_json(qs; color=_palette(1))
        js = _render_fan_js(id, fan_data, fan_bands; central_label="Median",
                            xlabel="Time", ylabel=d.varnames[vi])
        push!(panels, _PanelSpec(id, d.varnames[vi], js))
    end

    isempty(title) && (title = "Cross-sectional Quantiles")
    notes = String[]
    if vars === nothing
        vn = _cap_note("variables", n_vars_shown, d.n_vars, "max_vars")
        isempty(vn) || push!(notes, vn)
    end
    _noted_plot(panels, title, ncols, join(notes, " "), save_path)
end

# -----------------------------------------------------------------------------
# :spaghetti — one thin line per group, highlighted units colored
# -----------------------------------------------------------------------------
function _panel_spaghetti(d::PanelData{T}, vars, highlight, dates, max_groups::Int,
                          ncols::Int, title::String, save_path) where {T}
    var_idxs = vars === nothing ? [1] : _resolve_vars(vars, d.varnames)
    idx_map, unique_groups, unique_times = _panel_index(d)
    n_groups_total = length(unique_groups)
    n_groups_plot = min(n_groups_total, max_groups)

    hl_pos = highlight === nothing ? Int[] :
        Int[_resolve_group(h, unique_groups, d.group_names) for h in highlight]

    # Order the drawn groups so highlighted units come FIRST (legend cap keeps them
    # visible), the rest fill up to the cap in id order.
    rest = [g for g in 1:n_groups_total if !(g in hl_pos)]
    order = vcat([g for g in hl_pos if g <= n_groups_total], rest)[1:n_groups_plot]

    x_ticks_json, xlabel = _panel_date_axis(dates, unique_times)

    panels = _PanelSpec[]
    for vi in var_idxs
        id = _next_plot_id("pds")
        rows = Vector{Pair{String,String}}[]
        for t in unique_times
            row = Pair{String,String}["x" => _json(t)]
            for (ci, gpos) in enumerate(order)
                gid = unique_groups[gpos]
                idx = get(idx_map, (gid, t), nothing)
                push!(row, "g$ci" => _json(idx !== nothing ? d.data[idx, vi] : NaN))
            end
            push!(rows, row)
        end
        data_json = _json_array_of_objects(rows)
        gnames = String[gpos <= length(d.group_names) ? d.group_names[gpos] :
                        "Group $(unique_groups[gpos])" for gpos in order]
        colors = String[gpos in hl_pos ? _palette(findfirst(==(gpos), hl_pos)) :
                        "#cccccc" for gpos in order]
        s_json = _series_json(gnames, colors; keys=["g$i" for i in 1:length(order)])
        js = _render_line_js(id, data_json, s_json; xlabel=xlabel, ylabel="",
                             x_ticks_json=x_ticks_json)
        push!(panels, _PanelSpec(id, d.varnames[vi], js))
    end

    isempty(title) && (title = "Spaghetti Plot")
    note = _cap_note("groups", n_groups_plot, n_groups_total, "max_groups")
    _noted_plot(panels, title, ncols, note, save_path)
end

# -----------------------------------------------------------------------------
# :groups — small multiples, one panel per group
# -----------------------------------------------------------------------------
function _panel_groups(d::PanelData{T}, vars, dates, max_vars::Int, max_groups::Int,
                       ncols::Int, title::String, save_path) where {T}
    n_vars_shown = min(d.n_vars, max_vars)
    var_idxs = vars === nothing ? collect(1:n_vars_shown) : _resolve_vars(vars, d.varnames)
    idx_map, unique_groups, unique_times = _panel_index(d)
    n_groups_total = length(unique_groups)
    n_groups_plot = min(n_groups_total, max_groups)
    groups_to_plot = unique_groups[1:n_groups_plot]

    x_ticks_json, xlabel = _panel_date_axis(dates, unique_times)
    vnames = String[d.varnames[vi] for vi in var_idxs]
    vcolors = _palette_take(length(var_idxs))

    panels = _PanelSpec[]
    for (gi, gid) in enumerate(groups_to_plot)
        id = _next_plot_id("pdg")
        rows = Vector{Pair{String,String}}[]
        for t in unique_times
            row = Pair{String,String}["x" => _json(t)]
            idx = get(idx_map, (gid, t), nothing)
            for (vj, vi) in enumerate(var_idxs)
                push!(row, "v$vj" => _json(idx !== nothing ? d.data[idx, vi] : NaN))
            end
            push!(rows, row)
        end
        data_json = _json_array_of_objects(rows)
        s_json = _series_json(vnames, vcolors; keys=["v$j" for j in 1:length(var_idxs)])
        js = _render_line_js(id, data_json, s_json; xlabel=xlabel, ylabel="",
                             x_ticks_json=x_ticks_json)
        gname = gi <= length(d.group_names) ? d.group_names[gi] : "Group $gid"
        push!(panels, _PanelSpec(id, gname, js))
    end

    isempty(title) && (title = "Groups")
    note = _cap_note("groups", n_groups_plot, n_groups_total, "max_groups")
    _noted_plot(panels, title, ncols, note, save_path)
end

# -----------------------------------------------------------------------------
# :scatter — two-variable scatter with within/between demeaning
# -----------------------------------------------------------------------------
function _panel_scatter(d::PanelData{T}, vars, demean::Symbol, fit::Symbol,
                        ncols::Int, title::String, save_path) where {T}
    demean in (:none, :within, :between) || throw(ArgumentError(
        "demean must be :none, :within or :between, got :$demean"))
    idxs = vars === nothing ? throw(ArgumentError(":scatter needs two variables via vars=[x, y]")) :
           _resolve_vars(vars, d.varnames)
    length(idxs) >= 2 || throw(ArgumentError(":scatter needs two variables via vars=[x, y]"))
    ix, iy = idxs[1], idxs[2]
    xv, yv, sub = _panel_scatter_demean(d.data[:, ix], d.data[:, iy], d.group_id, demean)
    panel = _scatter2_panel(xv, yv, d.varnames[ix], d.varnames[iy];
                            fit=fit, subtitle=sub, id_prefix="pdsc")
    # Figure title must differ from the "y vs x" panel title (plotrule Axes/titles).
    isempty(title) && (title = isempty(desc(d)) ? "Scatter" : desc(d))
    p = _make_plot([panel]; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# Within/between demeaning for the panel scatter. Returns (x, y, subtitle).
function _panel_scatter_demean(xcol::AbstractVector, ycol::AbstractVector,
                       group_id::Vector{Int}, demean::Symbol)
    if demean === :none
        return (collect(xcol), collect(ycol), "pooled")
    end
    gmx = Dict{Int,Vector{Float64}}(); gmy = Dict{Int,Vector{Float64}}()
    for r in eachindex(group_id)
        (isfinite(xcol[r]) && isfinite(ycol[r])) || continue
        push!(get!(gmx, group_id[r], Float64[]), Float64(xcol[r]))
        push!(get!(gmy, group_id[r], Float64[]), Float64(ycol[r]))
    end
    if demean === :between
        gs = sort(collect(keys(gmx)))
        xb = Float64[sum(gmx[g]) / length(gmx[g]) for g in gs]
        yb = Float64[sum(gmy[g]) / length(gmy[g]) for g in gs]
        return (xb, yb, "between (group means)")
    else  # :within
        mx = Dict(g => sum(v) / length(v) for (g, v) in gmx)
        my = Dict(g => sum(v) / length(v) for (g, v) in gmy)
        xw = Float64[]; yw = Float64[]
        for r in eachindex(group_id)
            (isfinite(xcol[r]) && isfinite(ycol[r]) && haskey(mx, group_id[r])) || continue
            push!(xw, Float64(xcol[r]) - mx[group_id[r]])
            push!(yw, Float64(ycol[r]) - my[group_id[r]])
        end
        return (xw, yw, "within (group-demeaned)")
    end
end
