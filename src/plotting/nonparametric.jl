# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# plot_result for nonparametric fits (EV-33, #441)
#   - KernelDensity     → density curve over the grid
#   - KernelRegression  → scatter of (x,y) + fitted line + SE band (EViews-style)
#   - LowessFit         → scatter of (x,y) + LOWESS line
# =============================================================================

# Build a JSON array of {x, y, group} scatter rows.
function _np_scatter_json(x::AbstractVector, y::AbstractVector, group::String)
    rows = String[]
    for i in eachindex(x)
        push!(rows, "{\"x\":$(_json(x[i])),\"y\":$(_json(y[i])),\"group\":$(_json(group))}")
    end
    "[" * join(rows, ",\n") * "]"
end

# Build a JSON array of {x, y, lo, hi} for the fitted line (band optional).
function _np_fit_json(x::AbstractVector, fit::AbstractVector,
                      lo::Union{AbstractVector,Nothing},
                      hi::Union{AbstractVector,Nothing})
    rows = String[]
    for i in eachindex(x)
        lo_s = lo === nothing ? "null" : _json(lo[i])
        hi_s = hi === nothing ? "null" : _json(hi[i])
        push!(rows, "{\"x\":$(_json(x[i])),\"y\":$(_json(fit[i])),\"lo\":$lo_s,\"hi\":$hi_s}")
    end
    "[" * join(rows, ",\n") * "]"
end

# Build the scatter renderer's `curve_overlays_json` payload for a fitted curve
# (+ optional [lo,hi] band) from the `_np_fit_json` points. The relocated scatter
# renderer draws the polyline and band through its OWN scales (plotrule A1/A4) —
# no post-hoc scale-clone.
function _np_curve_overlay_json(fit_json::String; color::String=_PLOT_COLORS[2],
                                band::Bool=false)
    "[{\"points\":$(fit_json),\"color\":$(_json(color)),\"band\":$(band ? "true" : "false"),\"alpha\":0.15}]"
end

"""
    plot_result(r::KernelDensity; title="", save_path=nothing)

Plot the kernel density estimate as a curve over the grid.
"""
function plot_result(r::KernelDensity{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    id = _next_plot_id("kde")
    rows = Vector{Pair{String,String}}[]
    for i in eachindex(r.x)
        push!(rows, ["x" => _json(r.x[i]), "density" => _json(r.density[i])])
    end
    data_json = _json_array_of_objects(rows)
    s_json = _series_json(["Density"], [_PLOT_COLORS[1]]; keys=["density"])
    js = _render_line_js(id, data_json, s_json; xlabel="x", ylabel="Density")
    klab = get(_NP_KERNEL_LABEL, r.kernel, string(r.kernel))
    panels = [_PanelSpec(id, "Kernel Density ($klab, h=$(_fmt(r.bandwidth)))", js)]
    if isempty(title)
        title = "Kernel Density Estimate (n=$(r.nobs))"
    end
    p = _make_plot(panels; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(r::KernelRegression; bands=true, title="", save_path=nothing)

Plot the scatter of `(x, y)` with the nonparametric fit and (optionally) the
pointwise standard-error band — the EViews-style nonparametric-fit chart.
"""
function plot_result(r::KernelRegression{T}; bands::Bool=true, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    id = _next_plot_id("kreg")
    scatter_json = _np_scatter_json(r.xdata, r.ydata, "Data")
    groups = _series_json(["Data"], [_PLOT_COLORS[1]])
    lo = bands ? (r.fitted .- 1.96 .* r.se) : nothing
    hi = bands ? (r.fitted .+ 1.96 .* r.se) : nothing
    fit_json = _np_fit_json(r.x, r.fitted, lo, hi)
    curve = _np_curve_overlay_json(fit_json; color=_PLOT_COLORS[2], band=bands)
    js = _render_scatter_js(id, scatter_json, groups;
                            curve_overlays_json=curve, xlabel="x", ylabel="y")
    mlab = get(_NP_METHOD_LABEL, r.method, string(r.method))
    panels = [_PanelSpec(id, "$mlab fit (h=$(_fmt(r.bandwidth)))", js)]
    if isempty(title)
        title = "Nonparametric Regression (n=$(r.nobs))"
    end
    p = _make_plot(panels; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(r::LowessFit; title="", save_path=nothing)

Plot the scatter of `(x, y)` with the LOWESS smoother line.
"""
function plot_result(r::LowessFit{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    id = _next_plot_id("lowess")
    scatter_json = _np_scatter_json(r.x, r.ydata, "Data")
    groups = _series_json(["Data"], [_PLOT_COLORS[1]])
    fit_json = _np_fit_json(r.x, r.fitted, nothing, nothing)
    curve = _np_curve_overlay_json(fit_json; color=_PLOT_COLORS[2], band=false)
    js = _render_scatter_js(id, scatter_json, groups;
                            curve_overlays_json=curve, xlabel="x", ylabel="y")
    panels = [_PanelSpec(id, "LOWESS (f=$(_fmt(r.span)), iter=$(r.iter))", js)]
    if isempty(title)
        title = "LOWESS Smoother (n=$(r.nobs))"
    end
    p = _make_plot(panels; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end
