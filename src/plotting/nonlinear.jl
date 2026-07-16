# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
plot_result methods for nonlinear time series models (`ThresholdModel`).
"""

# =============================================================================
# ThresholdModel — regime-shaded series and SSR-profile views
# =============================================================================

"""
    plot_result(m::ThresholdModel; view=:regimes, title="", save_path=nothing)

Visualise a fitted [`ThresholdModel`](@ref).

- `view=:regimes` (default): the dependent series coloured by regime
  (`q ≤ γ̂` vs `q > γ̂`), a shared reference line at the split.
- `view=:ssr`: the concentrated SSR profile `S(γ)` over the threshold grid, with a
  reference line marking the SSR-minimising γ̂.
"""
function plot_result(m::ThresholdModel{T}; view::Symbol=:regimes,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    if view === :regimes
        p = _plot_threshold_regimes(m; title=title)
    elseif view === :ssr
        p = _plot_threshold_ssr(m; title=title)
    else
        throw(ArgumentError("Unknown view :$view for ThresholdModel; use :regimes or :ssr."))
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

function _plot_threshold_regimes(m::ThresholdModel{T}; title::String="") where {T}
    n = m.n
    rows = Vector{Pair{String,String}}[]
    for t in 1:n
        push!(rows, [
            "x"  => _json(t),
            "r1" => (m.regime[t] ? _json(m.y[t]) : "null"),
            "r2" => (m.regime[t] ? "null" : _json(m.y[t])),
        ])
    end
    data_json = _json_array_of_objects(rows)

    id = _next_plot_id("thr_reg")
    s = _series_json(["Regime 1 ($(m.qname) ≤ $(_fmt(m.gamma)))",
                      "Regime 2 ($(m.qname) > $(_fmt(m.gamma)))"],
                     [_PLOT_COLORS[1], _PLOT_COLORS[4]]; keys=["r1", "r2"])
    js = _render_line_js(id, data_json, s; xlabel="t", ylabel="y")
    panel = _PanelSpec(id, "Series by Regime", js)

    ttl = isempty(title) ?
        (m.is_setar ? "SETAR(2; $(m.p), $(m.p)) — Regime Classification" :
                      "Threshold Regression — Regime Classification") : title
    _make_plot([panel]; title=ttl, ncols=1)
end

function _plot_threshold_ssr(m::ThresholdModel{T}; title::String="") where {T}
    grid = _threshold_grid(m.q, m.trim)
    min_obs = m.k + 1
    rows = Vector{Pair{String,String}}[]
    for g in grid
        s, _, ok = _split_ssr(m.y, m.X, m.q, T(g); min_obs=min_obs)
        ok || continue
        push!(rows, ["x" => _json(g), "ssr" => _json(s)])
    end
    data_json = _json_array_of_objects(rows)

    id = _next_plot_id("thr_ssr")
    s = _series_json(["S(γ)"], [_PLOT_COLORS[2]]; keys=["ssr"])
    # γ̂ is a vertical marker; _render_line_js only draws horizontal ref lines, so
    # annotate γ̂ in the axis label instead of passing a value that would skew the y-range.
    js = _render_line_js(id, data_json, s; xlabel="γ (threshold, γ̂=$(_fmt(m.gamma)))",
                         ylabel="Concentrated SSR")
    panel = _PanelSpec(id, "SSR Profile", js)

    ttl = isempty(title) ? "Threshold SSR Profile S(γ)" : title
    _make_plot([panel]; title=ttl, ncols=1)
end
