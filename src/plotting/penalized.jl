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

    # Zero coefficient line + dashed vertical marker at the selected log(λ), both via
    # the line renderer's ref-line options (axis:"x" for the vertical — plotrule A4,
    # no scale-clone).
    logλ_sel = log(max(m.lambda, floatmin(T)))
    refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}," *
           "{\"value\":$(_json(logλ_sel)),\"color\":\"#666\",\"dash\":\"5,4\",\"axis\":\"x\"}]"
    js = _render_line_js(id, data, series;
                         ref_lines_json=refs,
                         xlabel="log(λ)", ylabel="Coefficient")
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

    # Dashed vertical markers at log(λ_min) and the 1-SE-rule log(λ), via the line
    # renderer's axis:"x" ref-line option (plotrule A4, no scale-clone).
    logλ_min = log(max(m.lambda_min, floatmin(T)))
    logλ_1se = log(max(m.lambda_1se, floatmin(T)))
    refs = "[{\"value\":$(_json(logλ_min)),\"color\":\"#2ca02c\",\"dash\":\"5,4\",\"axis\":\"x\"}," *
           "{\"value\":$(_json(logλ_1se)),\"color\":\"#999\",\"dash\":\"3,3\",\"axis\":\"x\"}]"
    js = _render_line_js(id, data, series;
                         bands_json=bands, ref_lines_json=refs,
                         xlabel="log(λ)", ylabel="Mean CV MSE")
    panel = _PanelSpec(id, "Cross-Validation Curve", js)

    isempty(title) && (title = "Penalized Regression CV Curve")
    _make_plot([panel]; title=title, ncols=1)
end
