# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# ARIMA plots (PLT-25 / #487) — the SOLE `plot_result(::AbstractARIMAModel)`
# method (single-method rule, plotrule §3): it covers `:fit`, `:resid`, `:roots`
# and `:diagnostics`; the `:diagnostics` branch CALLS the shared, frozen
# `_residual_diagnostics_panels` helper (helpers.jl) rather than duplicating it.
# A separate `plot_result(::ARIMAOrderSelection)` draws the IC (p,q) grid.
# =============================================================================

"""
    _arima_ar_coeffs(m) -> Vector{Float64}
    _arima_ma_coeffs(m) -> Vector{Float64}

AR (`phi`) and MA (`theta`) coefficient vectors of an ARIMA-family model, returned
as `Float64`. Robust across the non-parametric `AbstractARIMAModel` hierarchy: a
model missing the field (e.g. `MAModel` has no `phi`) yields an empty vector.
"""
_arima_ar_coeffs(m::AbstractARIMAModel) = hasproperty(m, :phi) ? Float64[Float64(c) for c in m.phi] : Float64[]
_arima_ma_coeffs(m::AbstractARIMAModel) = hasproperty(m, :theta) ? Float64[Float64(c) for c in m.theta] : Float64[]

"""
    _arima_inverse_roots(coeffs) -> Vector{Complex{Float64}}

Inverse roots of the AR (or MA) characteristic polynomial, computed as the
eigenvalues of the coefficient companion matrix — the exact primitive the package's
own stationarity/invertibility check (`_roots_inside_unit_circle`, arima/kalman.jl)
uses, so a modulus `< 1` means stable and `> 1` flags nonstationarity /
noninvertibility. No `Polynomials` dependency. Empty input → no roots.
"""
function _arima_inverse_roots(coeffs::AbstractVector)
    isempty(coeffs) && return Complex{Float64}[]
    c = Float64[Float64(v) for v in coeffs]
    n = length(c)
    n == 1 && return Complex{Float64}[complex(c[1], 0.0)]
    F = zeros(Float64, n, n)
    F[1, :] = c
    for i in 2:n
        F[i, i-1] = 1.0
    end
    Complex{Float64}[complex(z) for z in eigvals(F)]
end

"""Compact model label, e.g. `ARIMA(2,0,1)` — used as the figure title."""
_arima_label(m::AbstractARIMAModel) = "ARIMA($(ar_order(m)),$(diff_order(m)),$(ma_order(m)))"

# The series the model was actually fit to: the differenced series for an integrated
# model (so `fitted` is on the same scale), the level otherwise.
_arima_fit_target(m::AbstractARIMAModel) =
    hasproperty(m, :y_diff) ? Float64[Float64(v) for v in m.y_diff] : Float64[Float64(v) for v in m.y]

"""
    plot_result(m::AbstractARIMAModel; view=:fit, title="", ncols=0, save_path=nothing)

Diagnostic plots for an ARIMA-family model. `view` selects:

- `:fit` (default) — one-step fitted values overlaid on the actual series (two-line
  panel, integer observation axis; the integrated model is shown on its differenced
  scale, noted in the panel title).
- `:resid` — residual ACF and PACF as two vertical-bar panels (`_render_vbar_js` +
  `acf`/`pacf`), lags from 1 with ±CI reference lines.
- `:roots` — inverse roots of the AR and MA polynomials on the complex plane with a
  unit-circle reference (`_render_scatter_js` + `ref_shapes_json`); any root outside
  the circle is drawn in the alert color and the panel title is flagged UNSTABLE.
- `:diagnostics` — the shared four-panel residual figure (residual-vs-fitted,
  histogram + normal overlay, Normal Q-Q, residual ACF) via `_residual_diagnostics_panels`.

Unknown `view` throws `ArgumentError`.
"""
function plot_result(m::AbstractARIMAModel; view::Symbol=:fit, title::String="",
                     ncols::Int=0, save_path::Union{String,Nothing}=nothing)
    if view === :fit
        p = _plot_arima_fit(m; title=title, ncols=ncols)
    elseif view === :resid
        p = _plot_arima_resid(m; title=title, ncols=ncols)
    elseif view === :roots
        p = _plot_arima_roots(m; title=title, ncols=ncols)
    elseif view === :diagnostics
        panels = _residual_diagnostics_panels(m.residuals, m.fitted; varname="")
        ft = isempty(title) ? "$(_arima_label(m)) — Residual Diagnostics" : title
        p = _make_plot(panels; title=ft, ncols=ncols)
    else
        throw(ArgumentError("unknown view :$view; valid views: :fit, :resid, :roots, :diagnostics"))
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

function _plot_arima_fit(m::AbstractARIMAModel; title::String, ncols::Int)
    f = Float64[Float64(v) for v in m.fitted]
    nf = length(f)
    target = _arima_fit_target(m)
    ny = length(target)
    ya = (nf <= ny) ? target[ny-nf+1:end] : target        # align fitted to the tail
    rows = Vector{Pair{String,String}}[]
    for i in 1:min(nf, length(ya))
        push!(rows, ["x" => _json(i), "actual" => _json(ya[i]), "fit" => _json(f[i])])
    end
    data_json = _json_array_of_objects(rows)
    s_json = "[{\"name\":$(_json("Actual")),\"color\":$(_json(_PLOT_COLORS[1])),\"key\":\"actual\",\"dash\":\"\"}," *
             "{\"name\":$(_json("Fitted")),\"color\":$(_json(_PLOT_COLORS[2])),\"key\":\"fit\",\"dash\":\"6,3\"}]"
    id = _next_plot_id("arima_fit")
    js = _render_line_js(id, data_json, s_json; integer_x=true,
                         xlabel="Observation", ylabel="Value")
    diffnote = diff_order(m) > 0 ? " (differenced)" : ""
    ptitle = "Fitted vs Actual" * diffnote
    ft = isempty(title) ? _arima_label(m) : title
    _make_plot([_PanelSpec(id, ptitle, js)]; title=ft, ncols=ncols)
end

# One correlogram (ACF/PACF/draw-ACF) vertical-bar panel: bars from zero, integer lag
# ticks (from 1), ±CI horizontal reference lines. Shared by the ARIMA `:resid` view
# and the MCMC `:acf` view (same PLT lane).
function _correlogram_vbar_panel(prefix::String, lags::AbstractVector,
                                 vals::AbstractVector, ci::Real, ylabel::String,
                                 ptitle::String)
    rows = Vector{Pair{String,String}}[]
    for i in eachindex(lags)
        push!(rows, ["x" => _json(lags[i]), "y" => _json(vals[i])])
    end
    data_json = _json_array_of_objects(rows)
    refs = ci > 0 ?
        "[{\"value\":$(_json(ci)),\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"5,4\"}," *
        "{\"value\":$(_json(-ci)),\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"5,4\"}]" : "[]"
    id = _next_plot_id(prefix)
    js = _render_vbar_js(id, data_json; bar_color=_PLOT_COLORS[1], ref_lines_json=refs,
                         integer_x=true, xlabel="Lag", ylabel=ylabel)
    _PanelSpec(id, ptitle, js)
end

function _plot_arima_resid(m::AbstractARIMAModel; title::String, ncols::Int)
    r = Float64[Float64(v) for v in m.residuals if isfinite(v)]
    panels = _PanelSpec[]
    if length(r) >= 3
        nlags = max(1, min(24, fld(length(r), 4)))
        ar = acf(r; lags=nlags)
        pa = pacf(r; lags=nlags)
        push!(panels, _correlogram_vbar_panel("arima_acf", ar.lags, ar.acf, ar.ci,
                                             "Residual ACF", "Residual ACF"))
        push!(panels, _correlogram_vbar_panel("arima_pacf", pa.lags, pa.pacf, pa.ci,
                                             "Residual PACF", "Residual PACF"))
    else
        # Degenerate: too few residuals for a correlogram — emit empty panels rather
        # than throwing, so the figure still renders (plotrule Robustness).
        push!(panels, _correlogram_vbar_panel("arima_acf", Int[], Float64[], 0.0,
                                             "Residual ACF", "Residual ACF"))
        push!(panels, _correlogram_vbar_panel("arima_pacf", Int[], Float64[], 0.0,
                                             "Residual PACF", "Residual PACF"))
    end
    ft = isempty(title) ? "$(_arima_label(m)) — Residual Correlogram" : title
    _make_plot(panels; title=ft, ncols=ncols)
end

function _plot_arima_roots(m::AbstractARIMAModel; title::String, ncols::Int)
    ar_roots = _arima_inverse_roots(_arima_ar_coeffs(m))
    ma_roots = _arima_inverse_roots(_arima_ma_coeffs(m))
    rows = Vector{Pair{String,String}}[]
    used = Set{String}()
    outside = Ref(false)
    _push_root = (z, base) -> begin
        grp = abs(z) > 1 ? "Outside unit circle" : base
        abs(z) > 1 && (outside[] = true)
        push!(used, grp)
        push!(rows, ["x" => _json(real(z)), "y" => _json(imag(z)), "group" => _json(grp)])
    end
    for z in ar_roots; _push_root(z, "AR inverse root"); end
    for z in ma_roots; _push_root(z, "MA inverse root"); end
    data_json = _json_array_of_objects(rows)

    # Groups in canonical order, entity-stable colors; outside-circle roots = alert.
    order = ["AR inverse root" => _PLOT_COLORS[1],
             "MA inverse root" => _PLOT_COLORS[2],
             "Outside unit circle" => _PLOT_ALERT]
    gparts = String[]
    for (nm, col) in order
        nm in used || continue
        push!(gparts, "{\"name\":$(_json(nm)),\"color\":$(_json(col))}")
    end
    groups_json = "[" * join(gparts, ",") * "]"

    shapes = "[{\"type\":\"circle\",\"cx\":0,\"cy\":0,\"r\":1,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    refs = "[{\"value\":0,\"axis\":\"x\",\"color\":\"#ccc\",\"dash\":\"2,2\"}," *
           "{\"value\":0,\"axis\":\"y\",\"color\":\"#ccc\",\"dash\":\"2,2\"}]"
    id = _next_plot_id("arima_roots")
    js = _render_scatter_js(id, data_json, groups_json; ref_lines_json=refs,
                            ref_shapes_json=shapes, xlabel="Real", ylabel="Imaginary")
    ptitle = "Inverse Roots" * (outside[] ? " — UNSTABLE" : "")
    ft = isempty(title) ? _arima_label(m) : title
    _make_plot([_PanelSpec(id, ptitle, js)]; title=ft, ncols=ncols)
end

# =============================================================================
# ARIMA order-selection IC grid (heatmap)
# =============================================================================

"""
    _plot_ic_grid_json(M, row_labels, col_labels) -> String

Heatmap payload `[{x, y, v}, …]` for an information-criterion `(p,q)` matrix: `x` is
the column (q) label, `y` is the row (p) label, `v` is the IC value. Non-finite cells
(failed fits are stored as `Inf`) serialize to `null` so they render as the neutral
"missing" grey (plotrule Heatmaps rule 2).
"""
function _plot_ic_grid_json(M::AbstractMatrix, row_labels::Vector{String},
                            col_labels::Vector{String})
    rows = Vector{Pair{String,String}}[]
    for i in axes(M, 1), j in axes(M, 2)
        push!(rows, ["x" => _json(col_labels[j]), "y" => _json(row_labels[i]),
                     "v" => _json(M[i, j])])          # _json(Number) → "null" for NaN/Inf
    end
    _json_array_of_objects(rows)
end

"""
    plot_result(sel::ARIMAOrderSelection; view=:aic, title="", ncols=0, save_path=nothing)

Heatmap of the AIC (`view=:aic`, default) or BIC (`view=:bic`) grid over the
`(p, q)` order search. Rows are AR orders `p`, columns are MA orders `q`. Information
criteria are unsigned with no meaningful midpoint, so the ramp is a **sequential**
single hue with a color legend (plotrule PLT-15 / Heatmaps rule 3), and the selected
minimum-IC cell is marked. Unknown `view` throws `ArgumentError`.
"""
function plot_result(sel::ARIMAOrderSelection; view::Symbol=:aic, title::String="",
                     ncols::Int=0, save_path::Union{String,Nothing}=nothing)
    if view === :aic
        M = sel.aic_matrix; bp = sel.best_p_aic; bq = sel.best_q_aic; lbl = "AIC"
    elseif view === :bic
        M = sel.bic_matrix; bp = sel.best_p_bic; bq = sel.best_q_bic; lbl = "BIC"
    else
        throw(ArgumentError("unknown view :$view; valid views: :aic, :bic"))
    end
    P = size(M, 1) - 1
    Q = size(M, 2) - 1
    row_labels = String["p=$(p)" for p in 0:P]
    col_labels = String["q=$(q)" for q in 0:Q]
    data_json = _plot_ic_grid_json(M, row_labels, col_labels)
    best_json = "{\"x\":$(_json("q=$(bq)")),\"y\":$(_json("p=$(bp)"))}"
    id = _next_plot_id("arima_ic")
    js = _render_heatmap_js(id, data_json, _json(row_labels), _json(col_labels);
                            scale=:sequential, best_cell_json=best_json,
                            xlabel="MA order q", ylabel="AR order p", tip_label=lbl)
    ptitle = "$(lbl) grid — best p=$(bp), q=$(bq)"
    ft = isempty(title) ? "ARIMA Order Selection" : title
    p = _make_plot([_PanelSpec(id, ptitle, js)]; title=ft, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end
