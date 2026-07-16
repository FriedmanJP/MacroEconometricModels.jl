# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
plot_result methods for penalized regression (`PenalizedRegModel`): coefficient path vs
`log λ` (`view=:path`) and the cross-validation MSE curve (`view=:cv`).
"""

# =============================================================================
# Vertical-marker overlay (draws x-position guide lines matching _render_line_js scales)
# =============================================================================

"""Append a D3 snippet drawing dashed **vertical** guide lines at given `log λ` x-positions,
reconstructing the exact x-scale of `_render_line_js` (no x-padding, same margins)."""
function _penalized_vlines_js(id::String, data_json::String,
                              marks::Vector{Tuple{Float64,String,String}})
    mark_js = join(["{v:$(m[1]),color:'$(m[2])',dash:'$(m[3])'}" for m in marks], ",")
    """
(function() {
    const container = d3.select('#$(id)');
    const svgEl = container.select('svg');
    const gEl = svgEl.select('g');
    const W = +svgEl.attr('width');
    const margin = {top:10, right:15, bottom:35, left:55};
    const w = W - margin.left - margin.right;
    const h = Math.min(w * 0.6, 250);
    const data = $(data_json);
    const xVals = data.map(d => d.x);
    const x = d3.scaleLinear().domain(d3.extent(xVals)).range([0, w]);
    const marks = [$(mark_js)];
    marks.forEach(m => {
        gEl.append('line')
            .attr('x1', x(m.v)).attr('x2', x(m.v))
            .attr('y1', 0).attr('y2', h)
            .attr('stroke', m.color).attr('stroke-width', 1.2)
            .attr('stroke-dasharray', m.dash);
    });
})();
"""
end

# =============================================================================
# PenalizedRegModel
# =============================================================================

"""
    plot_result(m::PenalizedRegModel{T}; view=:path, title="", save_path=nothing)

Visualize a penalized-regression fit.

- `view=:path` — each regressor's natural-scale coefficient as a function of `log λ`, with a
  dashed vertical marker at the selected `λ` (the classic `glmnet` regularization path).
- `view=:cv` — the cross-validation mean-squared-error curve vs `log λ`, with dashed markers at
  `λ_min` and the 1-SE-rule `λ` (requires a CV-selected model).
"""
function plot_result(m::PenalizedRegModel{T};
                     view::Symbol=:path, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    if view == :path
        p = _plot_penalized_path(m; title=title)
    elseif view == :cv
        p = _plot_penalized_cv(m; title=title)
    else
        throw(ArgumentError("view must be :path or :cv; got :$view"))
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

function _plot_penalized_path(m::PenalizedRegModel{T}; title::String="") where {T}
    id = _next_plot_id("pen_path")
    L = length(m.lambda_path)
    p = length(m.beta)
    logλ = [log(max(m.lambda_path[k], floatmin(T))) for k in 1:L]

    rows = Vector{Pair{String,String}}[]
    for k in 1:L
        row = Pair{String,String}["x" => _json(logλ[k])]
        for j in 1:p
            push!(row, "s$j" => _json(m.coef_path[j, k]))
        end
        push!(rows, row)
    end
    data = _json_array_of_objects(rows)

    colors = [_PLOT_COLORS[mod1(j, length(_PLOT_COLORS))] for j in 1:p]
    series = _series_json(m.varnames, colors; keys=["s$j" for j in 1:p])

    refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"   # zero coefficient line
    js = _render_line_js(id, data, series;
                         ref_lines_json=refs,
                         xlabel="log(λ)", ylabel="Coefficient")
    js *= "\n" * _penalized_vlines_js(id, data,
        [(Float64(log(max(m.lambda, floatmin(T)))), "#666", "5,4")])
    panel = _PanelSpec(id, "Coefficient Path", js)

    isempty(title) && (title = "Penalized Regression Coefficient Path")
    _make_plot([panel]; title=title, ncols=1)
end

function _plot_penalized_cv(m::PenalizedRegModel{T}; title::String="") where {T}
    m.cv_mse === nothing &&
        throw(ArgumentError("model was not selected by CV; refit with select=:cv for view=:cv"))
    id = _next_plot_id("pen_cv")
    L = length(m.lambda_path)
    logλ = [log(max(m.lambda_path[k], floatmin(T))) for k in 1:L]

    rows = Vector{Pair{String,String}}[]
    for k in 1:L
        push!(rows, [
            "x" => _json(logλ[k]),
            "mse" => _json(m.cv_mse[k]),
            "lo" => _json(m.cv_mse[k] - m.cv_se[k]),
            "hi" => _json(m.cv_mse[k] + m.cv_se[k]),
        ])
    end
    data = _json_array_of_objects(rows)
    series = _series_json(["CV MSE"], [_PLOT_COLORS[1]]; keys=["mse"])
    bands = "[{\"lo_key\":\"lo\",\"hi_key\":\"hi\",\"color\":\"$(_PLOT_COLORS[1])\"}]"

    js = _render_line_js(id, data, series;
                         bands_json=bands,
                         xlabel="log(λ)", ylabel="Mean CV MSE")
    js *= "\n" * _penalized_vlines_js(id, data, [
        (Float64(log(max(m.lambda_min, floatmin(T)))), "#2ca02c", "5,4"),
        (Float64(log(max(m.lambda_1se, floatmin(T)))), "#999", "3,3"),
    ])
    panel = _PanelSpec(id, "Cross-Validation Curve", js)

    isempty(title) && (title = "Penalized Regression CV Curve")
    _make_plot([panel]; title=title, ncols=1)
end
