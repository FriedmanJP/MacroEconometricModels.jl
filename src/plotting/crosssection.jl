# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
plot_result for CrossSectionData — the FIRST dispatch for this container (PLT-22).
Mirrors the TimeSeriesData view set minus the time-specific views. The shared
distribution / scatter / correlation builders live in timeseries.jl (`_dist_panels`,
`_scatter2_panel`, `_corr_panel`) and are reused here; binscatter is in binscatter.jl.
No renderer is defined here (plotrule A1).
"""

# =============================================================================
# CrossSectionData
# =============================================================================

const _CS_VIEWS = (:hist, :density, :scatter, :corr, :pairs, :binscatter)

"""
    plot_result(d::CrossSectionData; view=:hist, vars=nothing, ncols=0, title="",
                save_path=nothing, fit=:ols, n_bins=0, x=nothing, y=nothing,
                controls=nothing)

Multi-view plot of a cross-sectional container. Unknown `view` throws (plotrule C5).
A distribution is the sensible default (no natural time axis).

- `:hist` (default) / `:density` — one histogram per selected variable; `:density`
  adds a KDE curve. `n_bins=0` auto-selects (Freedman–Diaconis).
- `:scatter` — a two-variable scatter (`vars=[x, y]`, String or Int); `fit=:ols`
  overlays the least-squares line, `fit=:none` suppresses it.
- `:corr` — Pearson correlation heatmap over the selected variables, symmetric
  `[-1,1]` diverging scale with a color legend (≤12 variables, capped with a note).
- `:pairs` — scatter-matrix over ≤6 variables (excess folded with a title note):
  the diagonal is each variable's histogram, off-diagonal cells are pairwise scatters.
- `:binscatter` — quantile-binned means of `y` on `x` (see the binscatter docstring).

`vars` accepts names or 1-based indices; bad input throws (plotrule C3). `save_path`
writes the HTML and returns the `PlotOutput` (plotrule C8).
"""
function plot_result(d::CrossSectionData{T};
                     view::Symbol=:hist,
                     vars::Union{Vector,Nothing}=nothing,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     fit::Symbol=:ols, n_bins::Int=0,
                     x=nothing, y=nothing,
                     controls::Union{Vector,Nothing}=nothing) where {T}
    view in _CS_VIEWS || throw(ArgumentError(
        "view=:$view not supported for CrossSectionData; valid: $(collect(_CS_VIEWS))"))

    if view === :binscatter
        return _plot_binscatter(d.data, d.varnames, x, y;
                                n_bins=n_bins, controls=controls, fit=fit,
                                title=title, ncols=ncols, save_path=save_path)
    elseif view === :scatter
        idxs = _resolve_vars(vars, d.varnames)
        length(idxs) >= 2 || throw(ArgumentError(":scatter needs two variables via vars=[x, y]"))
        ix, iy = idxs[1], idxs[2]
        panel = _scatter2_panel(d.data[:, ix], d.data[:, iy],
                                d.varnames[ix], d.varnames[iy]; fit=fit, id_prefix="cs")
        # Figure title must differ from the "y vs x" panel title (plotrule Axes/titles).
        isempty(title) && (title = isempty(desc(d)) ? "Scatter" : desc(d))
        p = _make_plot([panel]; title=title, ncols=ncols)
        save_path !== nothing && save_plot(p, save_path)
        return p
    elseif view === :hist || view === :density
        idxs = _resolve_vars(vars, d.varnames)
        total = length(idxs); shown = min(total, 12)
        panels = _dist_panels(d.data, idxs[1:shown], d.varnames;
                              density=(view === :density), n_bins=n_bins, id_prefix="cs")
        isempty(title) && (title = view === :density ? "Densities" : "Distributions")
        note = _cap_note("variables", shown, total, "vars")
        p = _make_plot(panels; title=title, ncols=ncols, note=note)
        save_path !== nothing && save_plot(p, save_path)
        return p
    elseif view === :corr
        idxs = _resolve_vars(vars, d.varnames)
        panel, _, _ = _corr_panel(d.data, idxs, d.varnames; id_prefix="cs")
        isempty(title) && (title = "Correlation Matrix")
        p = _make_plot([panel]; title=title, ncols=ncols)
        save_path !== nothing && save_plot(p, save_path)
        return p
    else  # :pairs
        idxs = _resolve_vars(vars, d.varnames)
        return _cs_pairs(d, idxs, ncols, title, save_path)
    end
end

# -----------------------------------------------------------------------------
# :pairs — scatter matrix (diagonal histograms, off-diagonal scatters), ≤6 vars
# -----------------------------------------------------------------------------
function _cs_pairs(d::CrossSectionData{T}, idxs::Vector{Int},
                   ncols::Int, title::String, save_path) where {T}
    total = length(idxs)
    shown = min(total, 6)                       # a 7×7 = 49-panel matrix is unreadable (C7)
    use = idxs[1:shown]

    panels = _PanelSpec[]
    for iy in use          # rows
        for ix in use      # columns
            if ix == iy
                id = _next_plot_id("csp")
                bins_json = _histogram_bins(@view d.data[:, ix])
                s_json = _series_json([d.varnames[ix]], [_palette(1)]; keys=["bar"])
                js = _render_histogram_js(id, bins_json, s_json;
                                          xlabel=d.varnames[ix], ylabel="Frequency")
                push!(panels, _PanelSpec(id, d.varnames[ix], js))
            else
                push!(panels, _scatter2_panel(d.data[:, ix], d.data[:, iy],
                                              d.varnames[ix], d.varnames[iy];
                                              fit=:none, id_prefix="csp"))
            end
        end
    end

    if isempty(title)
        title = shown < total ? "Pairs (first $shown of $total)" : "Pairs"
    end
    # Force the grid to be the scatter-matrix (shown × shown) layout.
    ncols == 0 && (ncols = shown)
    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end
