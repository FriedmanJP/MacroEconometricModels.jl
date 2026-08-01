# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
plot_result methods for MIDAS models: the fitted weight curve (`:weights`) and
actual-vs-fitted (`:fit`).
"""

# =============================================================================
# MidasModel — weight curve & actual-vs-fitted
# =============================================================================

"""
    plot_result(m::MidasModel; view=:weights, title="", save_path=nothing)

Visualize a MIDAS regression.

- `view=:weights` — the realized weight curve `wₖ` versus high-frequency lag `k`.
- `view=:fit` — actual versus fitted low-frequency target.
- `view=:diagnostics` — the shared four-panel residual diagnostics (PLT-24).
"""
function plot_result(m::MidasModel{T};
                     view::Symbol=:weights, acf_lags::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    if view === :weights
        p = _midas_weight_plot(m, title)
    elseif view === :fit
        p = _midas_fit_plot(m, title)
    elseif view === :diagnostics
        resid = Float64[Float64(v) for v in m.residuals]
        fitted = Float64[Float64(v) for v in m.fitted]
        panels = _residual_diagnostics_panels(resid, fitted; acf_lags=acf_lags)
        isempty(title) && (title = "MIDAS Residual Diagnostics")
        p = _make_plot(panels; title=title, ncols=2)
    else
        throw(ArgumentError("unknown view $view — use :weights, :fit, or :diagnostics"))
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

function _midas_weight_plot(m::MidasModel{T}, title::String) where {T}
    id = _next_plot_id("midas_w")
    K = m.K
    rows = Vector{Pair{String,String}}[]
    for k in 1:K
        push!(rows, [
            "x" => _json(k),
            "w" => _json(m.w[k]),
            "zero" => "0",
        ])
    end
    data_json = _json_array_of_objects(rows)
    s_json = _series_json(["Weight wₖ"], [_PLOT_COLORS[1]]; keys=["w"])
    refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js = _render_line_js(id, data_json, s_json;
                         ref_lines_json=refs,
                         xlabel="HF lag k (most-recent-first)", ylabel="Weight")
    isempty(title) && (title = "MIDAS Weight Curve ($(m.weights_kind), K=$K)")
    _make_plot([_PanelSpec(id, title, js)]; title=title)
end

function _midas_fit_plot(m::MidasModel{T}, title::String) where {T}
    id = _next_plot_id("midas_fit")
    n = length(m.y)
    rows = Vector{Pair{String,String}}[]
    for i in 1:n
        push!(rows, [
            "x" => _json(i),
            "actual" => _json(m.y[i]),
            "fitted" => _json(m.fitted[i]),
        ])
    end
    data_json = _json_array_of_objects(rows)
    s_json = _series_json(["Actual", "Fitted"],
                          [_PLOT_COLORS[1], _PLOT_COLORS[2]];
                          keys=["actual", "fitted"], dash=["", "6,3"])
    js = _render_line_js(id, data_json, s_json;
                         xlabel="Low-frequency period", ylabel="Value")
    isempty(title) && (title = "MIDAS Actual vs Fitted")
    _make_plot([_PanelSpec(id, title, js)]; title=title)
end
