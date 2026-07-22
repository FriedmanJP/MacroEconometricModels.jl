# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
plot_result method for TimeSeriesData plus the SHARED data-lane view builders.

This file is included first among the Wave-2 data-lane files (timeseries.jl,
panel.jl, crosssection.jl, binscatter.jl), so the small lane-local converters and
view-assembly helpers used by more than one container method are defined HERE and
reused by the later files (definition precedes use in include order). They are all
`_`-prefixed and documented (plotrule A5: a documented `_plot_*`/`_view_*` assembly
helper may live in the owning plotting file; A6 pure converters normally live in
helpers.jl, but helpers.jl is frozen this wave, so these lane-local converters stay
here). No renderer is defined here (plotrule A1) — every panel goes through a
`_render_*_js` from render.jl.

PLT-20 adds the multi-view entry point (`:line/:scatter/:hist/:density/:corr/:growth`)
plus `view=:binscatter` (PLT-23), on top of PLT-08 date axes.
"""

# =============================================================================
# Shared lane-local converters / view builders
# =============================================================================

"""
    _resolve_vars(vars, names) -> Vector{Int}

Resolve a `vars` selector (`nothing` = all, else a vector of `Int`/`String`) to
column indices through `_resolve_var` (which bounds-checks and throws a friendly
`ArgumentError`, plotrule C3). `nothing` returns every column.
"""
function _resolve_vars(vars::Union{Vector,Nothing}, names::Vector{String})
    vars === nothing && return collect(1:length(names))
    Int[_resolve_var(v, names) for v in vars]
end

"""
    _xy_scatter_json(x, y; group="Data") -> String

`[{x, y, group}, …]` payload for `_render_scatter_js` (non-finite values serialize
to `null` via `_json`). Lane-local converter (helpers.jl frozen this wave).
"""
function _xy_scatter_json(x::AbstractVector, y::AbstractVector; group::String="Data")
    rows = String[]
    for i in eachindex(x)
        push!(rows, "{\"x\":$(_json(x[i])),\"y\":$(_json(y[i])),\"group\":$(_json(group))}")
    end
    "[" * join(rows, ",") * "]"
end

"""
    _regions_json(shade) -> String

Build the `regions_json` shaded-x-range payload `[{x0, x1, color, alpha}, …]` for
`_render_line_js` from a vector of tuples. Each tuple is `(x0, x1)` or
`(x0, x1, color)` (optionally `(x0, x1, color, alpha)`); coordinates are in the
plot's x units (integer `time_index`, not date strings — the date axis only relabels
ticks). Default fill is neutral grey `#d9d9d9` at alpha `0.35`. `nothing` → `"[]"`.
"""
function _regions_json(shade::Union{AbstractVector,Nothing})
    (shade === nothing || isempty(shade)) && return "[]"
    parts = String[]
    for rg in shade
        x0 = rg[1]; x1 = rg[2]
        col = length(rg) >= 3 ? String(rg[3]) : "#d9d9d9"
        alp = length(rg) >= 4 ? Float64(rg[4]) : 0.35
        push!(parts, "{\"x0\":$(_json(x0)),\"x1\":$(_json(x1))," *
                     "\"color\":$(_json(col)),\"alpha\":$(_json(alp))}")
    end
    "[" * join(parts, ",") * "]"
end

"""
    _dist_panels(data, idxs, varnames; density=false, n_bins=0, id_prefix="dist")
        -> Vector{_PanelSpec}

One histogram panel per selected column (`:hist`), or a histogram + KDE-overlay
panel (`:density`), via `_render_histogram_js` + the frozen `_histogram_bins` /
`_kde_line_json` converters. Shared by the TimeSeriesData and CrossSectionData
distribution views.
"""
function _dist_panels(data::AbstractMatrix, idxs::Vector{Int}, varnames::Vector{String};
                      density::Bool=false, n_bins::Int=0, id_prefix::String="dist")
    panels = _PanelSpec[]
    for vi in idxs
        col = @view data[:, vi]
        id = _next_plot_id(id_prefix)
        bins_json = _histogram_bins(col; n_bins=n_bins, density=density)
        if density
            dens_json = _kde_line_json(col)
            s_json = _series_json([varnames[vi], "Density"],
                                  [_palette(1), _palette(2)]; keys=["bar", "d"])
            js = _render_histogram_js(id, bins_json, s_json;
                                      density_json=dens_json, xlabel=varnames[vi],
                                      ylabel="Density")
        else
            s_json = _series_json([varnames[vi]], [_palette(1)]; keys=["bar"])
            js = _render_histogram_js(id, bins_json, s_json;
                                      xlabel=varnames[vi], ylabel="Frequency")
        end
        push!(panels, _PanelSpec(id, varnames[vi], js))
    end
    panels
end

"""
    _scatter2_panel(xv, yv, xlab, ylab; fit=:ols, subtitle="", id_prefix="scat")
        -> _PanelSpec

A single grouped scatter of `(xv, yv)` with an optional OLS fit overlay (drawn
through the renderer's own scales via `line_overlays_json`, plotrule A4). `fit`
is `:ols` (degree-1 line), `:none`; any other value throws. `subtitle`, when given,
is appended to the panel title (e.g. a within/between note). Shared by the
TimeSeriesData, PanelData and CrossSectionData scatter views.
"""
function _scatter2_panel(xv::AbstractVector, yv::AbstractVector,
                         xlab::String, ylab::String;
                         fit::Symbol=:ols, subtitle::String="", id_prefix::String="scat")
    fit in (:ols, :none) || throw(ArgumentError(
        "fit must be :ols or :none, got :$fit"))
    id = _next_plot_id(id_prefix)
    data_json = _xy_scatter_json(xv, yv; group=ylab)
    groups_json = _series_json([ylab], [_palette(1)])
    overlays = fit === :ols ? _ols_fit_line(xv, yv; degree=1, color=_PLOT_ALERT) : "[]"
    js = _render_scatter_js(id, data_json, groups_json;
                            line_overlays_json=overlays, xlabel=xlab, ylabel=ylab)
    ptitle = isempty(subtitle) ? "$ylab vs $xlab" : "$ylab vs $xlab — $subtitle"
    _PanelSpec(id, ptitle, js)
end

"""
    _corr_panel(data, idxs, varnames; id_prefix="corr") -> (_PanelSpec, shown, total)

Pearson correlation heatmap over the selected columns via `_corr_matrix_json` +
`_render_heatmap_js` on the signed diverging `_PLOT_DIVERGING` ramp with a fixed
symmetric `[-1, 1]` domain and midpoint 0 (plotrule Heatmaps / PLT-15). Columns are
capped at 12 (plotrule C7); returns the panel plus `(shown, total)` so the caller
surfaces any truncation in a note. Shared by TimeSeriesData / CrossSectionData.
"""
function _corr_panel(data::AbstractMatrix, idxs::Vector{Int}, varnames::Vector{String};
                     id_prefix::String="corr")
    total = length(idxs)
    shown = min(total, 12)
    use = idxs[1:shown]
    labels = String[varnames[i] for i in use]
    M = data[:, use]
    id = _next_plot_id(id_prefix)
    data_json = _corr_matrix_json(M; labels=labels)
    labs_json = _json(labels)
    js = _render_heatmap_js(id, data_json, labs_json, labs_json;
                            color_domain=Float64[-1.0, 1.0], scale=:diverging,
                            midpoint=0, tip_label="")
    ptitle = _cap_title("Correlation", shown, total)
    (_PanelSpec(id, ptitle, js), shown, total)
end

"""
    _noted_plot(panels, title, ncols, note, save_path) -> PlotOutput

Compose panels, but keep any C7 truncation `note` visible even when there is exactly
one panel — `_render_body_single` drops the figure note, so the note is folded into
the single panel's title instead (plotrule C7 "truncation visible in the output").
Multi-panel figures keep the note in the figure footer as usual.
"""
function _noted_plot(panels::Vector{_PanelSpec}, title::String, ncols::Int,
                     note::String, save_path::Union{String,Nothing})
    # _make_plot now renders the figure note on single- and multi-panel figures
    # alike (C7), so no title-folding workaround is needed here.
    p = _make_plot(panels; title=title, ncols=ncols, note=note)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# TimeSeriesData
# =============================================================================

const _TS_VIEWS = (:line, :scatter, :hist, :density, :corr, :growth, :binscatter)

"""
    plot_result(d::TimeSeriesData; view=:line, vars=nothing, ncols=0, title="",
                save_path=nothing, shade=nothing, fit=:ols, n_bins=0, tcodes=nothing,
                x=nothing, y=nothing, controls=nothing)

Multi-view plot of a time-series container. `view` selects the chart family; an
unknown view throws an `ArgumentError` listing the valid set (plotrule C5).

- `:line` (default) — one line panel per variable. The x-axis shows calendar
  labels when `dates(d)` is populated (via `set_dates!`) and integer `time_index`
  otherwise (PLT-08). `shade=[(x0,x1), …]` draws shaded x-ranges (e.g. recession
  bands) in grey behind the series; coordinates are in `time_index` units.
- `:scatter` — a two-variable scatter; pass `vars=[x, y]` (String or Int). `fit=:ols`
  overlays the least-squares line, `fit=:none` suppresses it.
- `:hist` / `:density` — one histogram per selected variable; `:density` adds a KDE
  curve. `n_bins=0` auto-selects (Freedman–Diaconis).
- `:corr` — Pearson correlation heatmap over the selected variables on a symmetric
  `[-1,1]` diverging scale with a color legend (≤12 variables, capped with a note).
- `:growth` — apply the FRED transformation codes (default `d.tcode`, override with
  `tcodes`) then draw `:line`; the panel notes the transform and any log→difference
  fallback on a non-positive column.
- `:binscatter` — quantile-binned means of `y` against `x` (see the `plot_result`
  binscatter docstring); pass `x`, `y`, optional `controls`.

`vars` accepts variable names or 1-based indices; bad input throws (plotrule C3).
`save_path` also writes the HTML and returns the `PlotOutput` (plotrule C8).
"""
function plot_result(d::TimeSeriesData{T};
                     view::Symbol=:line,
                     vars::Union{Vector,Nothing}=nothing,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     shade::Union{AbstractVector,Nothing}=nothing,
                     fit::Symbol=:ols, n_bins::Int=0,
                     tcodes::Union{Vector{Int},Int,Nothing}=nothing,
                     x=nothing, y=nothing,
                     controls::Union{Vector,Nothing}=nothing) where {T}
    view in _TS_VIEWS || throw(ArgumentError(
        "view=:$view not supported for TimeSeriesData; valid: $(collect(_TS_VIEWS))"))

    if view === :binscatter
        return _plot_binscatter(d.data, d.varnames, x, y;
                                n_bins=n_bins, controls=controls, fit=fit,
                                title=title, ncols=ncols, save_path=save_path)
    elseif view === :scatter
        idxs = _resolve_vars(vars, d.varnames)
        length(idxs) >= 2 || throw(ArgumentError(
            ":scatter needs two variables via vars=[x, y]"))
        ix, iy = idxs[1], idxs[2]
        panel = _scatter2_panel(d.data[:, ix], d.data[:, iy],
                                d.varnames[ix], d.varnames[iy]; fit=fit, id_prefix="ts")
        # Figure title must differ from the "y vs x" panel title (plotrule Axes/titles).
        isempty(title) && (title = isempty(desc(d)) ? "Scatter" : desc(d))
        p = _make_plot([panel]; title=title, ncols=ncols)
        save_path !== nothing && save_plot(p, save_path)
        return p
    elseif view === :hist || view === :density
        idxs = _resolve_vars(vars, d.varnames)
        total = length(idxs)
        shown = min(total, 12)
        panels = _dist_panels(d.data, idxs[1:shown], d.varnames;
                              density=(view === :density), n_bins=n_bins, id_prefix="ts")
        isempty(title) && (title = view === :density ? "Densities" : "Distributions")
        note = _cap_note("variables", shown, total, "vars")
        return _noted_plot(panels, title, ncols, note, save_path)
    elseif view === :corr
        idxs = _resolve_vars(vars, d.varnames)
        panel, shown, total = _corr_panel(d.data, idxs, d.varnames; id_prefix="ts")
        isempty(title) && (title = "Correlation Matrix")
        p = _make_plot([panel]; title=title, ncols=ncols)
        save_path !== nothing && save_plot(p, save_path)
        return p
    elseif view === :growth
        tc = tcodes === nothing ? d.tcode :
             (tcodes isa Int ? fill(tcodes, d.n_vars) : tcodes)
        dg = apply_tcode(d, tc)
        # Note any tcode that fell back (log-based → first difference) on a
        # non-positive column: apply_tcode records the *effective* tcode in dg.tcode.
        fell = [dg.varnames[j] for j in 1:dg.n_vars if dg.tcode[j] != tc[j]]
        note = isempty(fell) ? "" :
            "Log transform fell back to first difference on: $(join(fell, ", "))."
        return _ts_line(dg; vars=vars, ncols=ncols, title=title,
                        save_path=save_path, shade=shade, note=note,
                        subtitle_tcode=true)
    else  # :line
        return _ts_line(d; vars=vars, ncols=ncols, title=title,
                        save_path=save_path, shade=shade, note="")
    end
end

"""
    _ts_line(d; vars, ncols, title, save_path, shade, note, subtitle_tcode=false)

The `:line`/`:growth` panel assembler: one date-axis line per selected variable,
capped at 12 with a `C7` note, optional shaded x-regions. `subtitle_tcode` appends
the per-variable FRED transform label to the panel title (used by `:growth`).
"""
function _ts_line(d::TimeSeriesData{T}; vars::Union{Vector,Nothing},
                  ncols::Int, title::String, save_path::Union{String,Nothing},
                  shade::Union{AbstractVector,Nothing}, note::String,
                  subtitle_tcode::Bool=false) where {T}
    idxs = _resolve_vars(vars, d.varnames)
    total = length(idxs)
    shown = min(total, 12)
    idxs = idxs[1:shown]

    labels = _date_axis_labels(d)
    x_ticks_json = _x_ticks_json(d.time_index, labels)
    xlabel = labels === nothing ? "Time" : "Date"
    regions = _regions_json(shade)

    panels = _PanelSpec[]
    for vi in idxs
        id = _next_plot_id("ts")
        rows = Vector{Pair{String,String}}[]
        for t in 1:d.T_obs
            push!(rows, ["x" => _json(d.time_index[t]), "v1" => _json(d.data[t, vi])])
        end
        data_json = _json_array_of_objects(rows)
        s_json = _series_json([d.varnames[vi]], [_palette(vi)]; keys=["v1"])
        js = _render_line_js(id, data_json, s_json; xlabel=xlabel, ylabel="",
                             x_ticks_json=x_ticks_json, regions_json=regions)
        ptitle = subtitle_tcode ? "$(d.varnames[vi]) [$(_tcode_label(d.tcode[vi]))]" :
                                  d.varnames[vi]
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        title = isempty(desc(d)) ? "Time Series Data" : desc(d)
    end
    notes = String[]
    isempty(note) || push!(notes, note)
    cn = _cap_note("variables", shown, total, "vars")
    isempty(cn) || push!(notes, cn)

    _noted_plot(panels, title, ncols, join(notes, " "), save_path)
end

# Human-readable FRED transformation-code label for a growth-view panel subtitle.
_tcode_label(tc::Int) = get(Dict(
    1 => "level", 2 => "Δ", 3 => "Δ²", 4 => "log",
    5 => "Δlog", 6 => "Δ²log", 7 => "Δ(x/x₋₁−1)"), tc, "tcode $tc")
