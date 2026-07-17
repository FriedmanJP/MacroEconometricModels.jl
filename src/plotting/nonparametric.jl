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

# Overlay a fitted line (+ optional shaded band) onto an existing scatter panel,
# recomputing the shared domain from BOTH the scatter points and the fit so the
# curve and its band always lie inside the drawn axes.
function _render_np_overlay_js(id::String, scatter_json::String, fit_json::String;
                               color::String=_PLOT_COLORS[2], has_band::Bool=false)
    """
(function() {
    const container = d3.select('#$(id)');
    const svgEl = container.select('svg');
    const gEl = svgEl.select('g');
    const W = +svgEl.attr('width');
    const margin = {top:10, right:15, bottom:35, left:55};
    const w = W - margin.left - margin.right;
    const h = Math.min(w * 0.6, 250);

    const pts = $(scatter_json);
    const fit = $(fit_json);
    const hasBand = $(has_band ? "true" : "false");

    const xVals = pts.map(d => d.x).concat(fit.map(d => d.x));
    let yVals = pts.map(d => d.y).concat(fit.map(d => d.y));
    if (hasBand) {
        fit.forEach(d => { if(d.lo!==null) yVals.push(d.lo); if(d.hi!==null) yVals.push(d.hi); });
    }
    const xExt = d3.extent(xVals);
    const xPad = (xExt[1]-xExt[0])*0.08 || 1;
    const x = d3.scaleLinear().domain([xExt[0]-xPad, xExt[1]+xPad]).range([0,w]);
    const yExt = d3.extent(yVals);
    const yPad = (yExt[1]-yExt[0])*0.08 || 0.01;
    const y = d3.scaleLinear().domain([yExt[0]-yPad, yExt[1]+yPad]).range([h,0]);

    if (hasBand) {
        const area = d3.area().x(d=>x(d.x))
            .y0(d=>y(d.lo!==null ? d.lo : d.y)).y1(d=>y(d.hi!==null ? d.hi : d.y))
            .defined(d=>d.lo!==null && d.hi!==null);
        gEl.append('path').datum(fit).attr('d',area)
            .attr('fill','$(color)').attr('opacity',0.15);
    }
    const line = d3.line().x(d=>x(d.x)).y(d=>y(d.y));
    gEl.append('path').datum(fit).attr('d',line)
        .attr('fill','none').attr('stroke','$(color)').attr('stroke-width',2.2);
})();
"""
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
    js_scatter = _render_scatter_js(id, scatter_json, groups; xlabel="x", ylabel="y")
    lo = bands ? (r.fitted .- 1.96 .* r.se) : nothing
    hi = bands ? (r.fitted .+ 1.96 .* r.se) : nothing
    fit_json = _np_fit_json(r.x, r.fitted, lo, hi)
    js_line = _render_np_overlay_js(id, scatter_json, fit_json;
                                    color=_PLOT_COLORS[2], has_band=bands)
    mlab = get(_NP_METHOD_LABEL, r.method, string(r.method))
    panels = [_PanelSpec(id, "$mlab fit (h=$(_fmt(r.bandwidth)))", js_scatter * "\n" * js_line)]
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
    js_scatter = _render_scatter_js(id, scatter_json, groups; xlabel="x", ylabel="y")
    fit_json = _np_fit_json(r.x, r.fitted, nothing, nothing)
    js_line = _render_np_overlay_js(id, scatter_json, fit_json;
                                    color=_PLOT_COLORS[2], has_band=false)
    panels = [_PanelSpec(id, "LOWESS (f=$(_fmt(r.span)), iter=$(r.iter))",
                         js_scatter * "\n" * js_line)]
    if isempty(title)
        title = "LOWESS Smoother (n=$(r.nobs))"
    end
    p = _make_plot(panels; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end
