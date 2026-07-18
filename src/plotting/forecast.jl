# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
plot_result methods for forecast types: ARIMAForecast, VolatilityForecast,
VECMForecast, FactorForecast, LPForecast.
"""

# =============================================================================
# Shared history/fan panel builder (PLT-18) — every multi-panel forecast type can
# now draw a pre-sample "History" line joined to the dashed forecast fan, exactly
# like ARIMA/Volatility, plus a vertical forecast-origin reference line.
# =============================================================================

# Validate a `history=` argument against the forecast width and the panels drawn.
# A matrix must have one column per forecast series; a bare vector is only allowed
# when a single variable is selected (plotrule Robustness: misaligned aux series).
function _fc_validate_history(history, n_panels::Int, width::Int)
    history === nothing && return nothing
    if history isa AbstractMatrix
        size(history, 2) == width || throw(ArgumentError(
            "history has $(size(history, 2)) columns; expected $width to match the forecast width"))
    elseif history isa AbstractVector
        n_panels == 1 || throw(ArgumentError(
            "a vector history is only valid when a single variable is selected; " *
            "$(n_panels) panels are drawn — pass a T×$(width) matrix or select one var"))
    end
    nothing
end

# The history column for panel `vi` (bare vector passes through; matrix → column vi).
_fc_history_col(history, vi::Int) = history === nothing ? nothing :
    (history isa AbstractVector ? history : @view history[:, vi])

# Build one forecast panel's JS: solid History line (when supplied) + dashed forecast
# fan + CI band + a vertical forecast-origin reference line when history is present.
function _forecast_panel_js(id::String, fc::AbstractVector, lo::AbstractVector,
                            hi::AbstractVector; has_ci::Bool, fc_name::String,
                            history::Union{AbstractVector,Nothing}, n_history::Int,
                            base_color::String=_PLOT_SERIES[1],
                            xlabel::String, ylabel::String)
    data_json = _forecast_data_json(fc, lo, hi; history=history, n_history=n_history)
    if history === nothing
        s_json = _series_json([fc_name], [base_color]; keys=["fc"])
        bands = has_ci ?
            "[{\"lo_key\":\"ci_lo\",\"hi_key\":\"ci_hi\",\"color\":\"$(base_color)\",\"alpha\":$(_PLOT_CI_ALPHA)}]" : "[]"
        refs = "[]"
    else
        s_json = _series_json(["History", fc_name], [_PLOT_SERIES[1], _PLOT_SERIES[2]];
                              keys=["hist", "fc"], dash=["", "6,3"])
        bands = has_ci ?
            "[{\"lo_key\":\"ci_lo\",\"hi_key\":\"ci_hi\",\"color\":\"$(_PLOT_SERIES[2])\",\"alpha\":$(_PLOT_CI_ALPHA)}]" : "[]"
        # Vertical reference line at the forecast origin (x = 0 in the joined domain).
        refs = "[{\"value\":0,\"color\":\"$(_PLOT_ALERT)\",\"dash\":\"5,3\",\"axis\":\"x\"}]"
    end
    _render_line_js(id, data_json, s_json; bands_json=bands, ref_lines_json=refs,
                    xlabel=xlabel, ylabel=ylabel)
end

# =============================================================================
# ARIMAForecast
# =============================================================================

"""
    plot_result(fc::ARIMAForecast; history=nothing, n_history=50, title="", save_path=nothing)

Plot ARIMA forecast with CI fan. Pass original series as `history` to show context.
"""
function plot_result(fc::ARIMAForecast{T};
                     history::Union{AbstractVector,Nothing}=nothing,
                     n_history::Int=50, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    id = _next_plot_id("arima_fc")
    data_json = _forecast_data_json(fc.forecast, fc.ci_lower, fc.ci_upper;
                                     history=history, n_history=n_history)

    series_keys = String[]
    series_names = String[]
    series_colors = String[]
    series_dash = String[]

    if history !== nothing
        push!(series_keys, "hist"); push!(series_names, "History")
        push!(series_colors, _PLOT_COLORS[1]); push!(series_dash, "")
    end
    push!(series_keys, "fc"); push!(series_names, "Forecast")
    push!(series_colors, _PLOT_COLORS[2]); push!(series_dash, "6,3")

    s_json = _series_json(series_names, series_colors; keys=series_keys, dash=series_dash)
    bands = "[{\"lo_key\":\"ci_lo\",\"hi_key\":\"ci_hi\",\"color\":\"$(_PLOT_COLORS[2])\",\"alpha\":$(_PLOT_CI_ALPHA)}]"

    js = _render_line_js(id, data_json, s_json;
                         bands_json=bands, xlabel="Period", ylabel="Value")

    if isempty(title)
        ci_pct = round(Int, 100 * fc.conf_level)
        title = "ARIMA Forecast (h=$(fc.horizon), $(ci_pct)% CI)"
    end

    p = _make_plot([_PanelSpec(id, title, js)]; title=title)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# VolatilityForecast
# =============================================================================

"""
    plot_result(fc::VolatilityForecast; history=nothing, n_history=50, title="", save_path=nothing)

Plot volatility forecast (conditional variance).
"""
function plot_result(fc::VolatilityForecast{T};
                     history::Union{AbstractVector,Nothing}=nothing,
                     n_history::Int=50, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    id = _next_plot_id("vol_fc")
    data_json = _forecast_data_json(fc.forecast, fc.ci_lower, fc.ci_upper;
                                     history=history, n_history=n_history)

    series_keys = String[]
    series_names = String[]
    series_colors = String[]
    series_dash = String[]

    if history !== nothing
        push!(series_keys, "hist"); push!(series_names, "History")
        push!(series_colors, _PLOT_COLORS[1]); push!(series_dash, "")
    end
    push!(series_keys, "fc"); push!(series_names, "Forecast σ²")
    push!(series_colors, _PLOT_COLORS[2]); push!(series_dash, "6,3")

    s_json = _series_json(series_names, series_colors; keys=series_keys, dash=series_dash)
    bands = "[{\"lo_key\":\"ci_lo\",\"hi_key\":\"ci_hi\",\"color\":\"$(_PLOT_COLORS[2])\",\"alpha\":$(_PLOT_CI_ALPHA)}]"

    js = _render_line_js(id, data_json, s_json;
                         bands_json=bands, xlabel="Horizon", ylabel="Conditional Variance")

    if isempty(title)
        ci_pct = round(Int, 100 * fc.conf_level)
        title = "Volatility Forecast ($(fc.model_type), h=$(fc.horizon), $(ci_pct)% CI)"
    end

    p = _make_plot([_PanelSpec(id, title, js)]; title=title)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# VARForecast
# =============================================================================

"""
    plot_result(fc::VARForecast; var=nothing, history=nothing, n_history=50, ncols=0, title="", save_path=nothing)

Plot VAR forecast with bootstrap CI bands. Pass `history` (a `T×n_vars` matrix, or a
vector when a single `var` is selected) to draw a pre-sample History line joined to
the forecast fan, with a vertical forecast-origin reference line.
"""
function plot_result(fc::VARForecast{T};
                     var::Union{Int,String,Nothing}=nothing,
                     history::Union{AbstractVector,AbstractMatrix,Nothing}=nothing,
                     n_history::Int=50,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    h, n_vars = size(fc.forecast)
    vars_to_plot = var === nothing ? (1:n_vars) : [_resolve_var(var, fc.varnames)]
    _fc_validate_history(history, length(vars_to_plot), n_vars)
    has_ci = fc.ci_method != :none

    panels = _PanelSpec[]
    for vi in vars_to_plot
        id = _next_plot_id("var_fc")
        ptitle = fc.varnames[vi]
        js = _forecast_panel_js(id, fc.forecast[:, vi], fc.ci_lower[:, vi], fc.ci_upper[:, vi];
                                has_ci=has_ci, fc_name="Forecast",
                                history=_fc_history_col(history, vi), n_history=n_history,
                                xlabel="Horizon", ylabel="Forecast")
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        title = fc.ci_method == :none ? "VAR Forecast" :
                "VAR Forecast ($(fc.ci_method) CI)"
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# BVARForecast
# =============================================================================

"""
    plot_result(fc::BVARForecast; var=nothing, history=nothing, n_history=50, ncols=0, title="", save_path=nothing)

Plot Bayesian VAR forecast with posterior credible bands. Pass `history` (a
`T×n_vars` matrix, or a vector when a single `var` is selected) to draw a pre-sample
History line joined to the forecast fan, with a vertical forecast-origin line.
"""
function plot_result(fc::BVARForecast{T};
                     var::Union{Int,String,Nothing}=nothing,
                     history::Union{AbstractVector,AbstractMatrix,Nothing}=nothing,
                     n_history::Int=50,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    h, n_vars = size(fc.forecast)
    vars_to_plot = var === nothing ? (1:n_vars) : [_resolve_var(var, fc.varnames)]
    _fc_validate_history(history, length(vars_to_plot), n_vars)
    pe_label = fc.point_estimate == :median ? "Posterior median" : "Posterior mean"

    panels = _PanelSpec[]
    for vi in vars_to_plot
        id = _next_plot_id("bvar_fc")
        ptitle = fc.varnames[vi]
        js = _forecast_panel_js(id, fc.forecast[:, vi], fc.ci_lower[:, vi], fc.ci_upper[:, vi];
                                has_ci=true, fc_name=pe_label,
                                history=_fc_history_col(history, vi), n_history=n_history,
                                xlabel="Horizon", ylabel="Forecast")
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        ci_pct = round(Int, 100 * fc.conf_level)
        title = "Bayesian VAR Forecast ($(ci_pct)% credible interval)"
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# VECMForecast
# =============================================================================

"""
    plot_result(fc::VECMForecast; var=nothing, history=nothing, n_history=50, ncols=0, title="", save_path=nothing)

Plot VECM forecast in levels with CI bands. Pass `history` (a `T×n_vars` matrix, or a
vector when a single `var` is selected) to draw a pre-sample History line joined to
the forecast fan, with a vertical forecast-origin line.
"""
function plot_result(fc::VECMForecast{T};
                     var::Union{Int,String,Nothing}=nothing,
                     history::Union{AbstractVector,AbstractMatrix,Nothing}=nothing,
                     n_history::Int=50,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    h, n_vars = size(fc.levels)
    vars_to_plot = var === nothing ? (1:n_vars) : [_resolve_var(var, fc.varnames)]
    _fc_validate_history(history, length(vars_to_plot), n_vars)
    has_ci = fc.ci_method != :none

    panels = _PanelSpec[]
    for vi in vars_to_plot
        id = _next_plot_id("vecm_fc")
        ptitle = fc.varnames[vi]
        js = _forecast_panel_js(id, fc.levels[:, vi], fc.ci_lower[:, vi], fc.ci_upper[:, vi];
                                has_ci=has_ci, fc_name="Forecast",
                                history=_fc_history_col(history, vi), n_history=n_history,
                                xlabel="Horizon", ylabel="Level")
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        title = fc.ci_method == :none ? "VECM Forecast" :
                "VECM Forecast ($(fc.ci_method) CI)"
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# FactorForecast
# =============================================================================

"""
    plot_result(fc::FactorForecast; type=:both, var=nothing, ncols=0, title="",
                n_obs=6, save_path=nothing)

Plot factor model forecast.

- `type=:both`: plot factor forecasts and top observable forecasts (default)
- `type=:factor`: plot factor forecasts only
- `type=:observable`: plot observable forecasts only
- `n_obs`: max number of observables to show when `type=:both` (default 6)
- `history`/`n_history`: pre-sample context for the **observable** panels (a
  `T×n_observables` matrix, or a vector when a single observable is selected); the
  latent-factor panels have no observed history so they ignore it.
- `var`: select a single panel by 1-based index (`Int`) or synthetic name
  (`"Factor k"` / `"Observable k"`, `String`); resolved through `_resolve_var`
  (out-of-range/unknown → `ArgumentError`). An `Int` selects the same index in
  **both** the factor and observable loops under `type=:both`; select a single
  `type` when addressing a panel by name.
"""
function plot_result(fc::FactorForecast{T};
                     type::Symbol=:both, var::Union{Int,String,Nothing}=nothing,
                     history::Union{AbstractVector,AbstractMatrix,Nothing}=nothing,
                     n_history::Int=50,
                     ncols::Int=0, title::String="",
                     n_obs::Int=6,
                     save_path::Union{String,Nothing}=nothing) where {T}
    type in (:factor, :observable, :both) ||
        throw(ArgumentError("Unknown type: $type. Expected :factor, :observable, or :both"))
    has_ci = fc.ci_method != :none

    panels = _PanelSpec[]

    # Factor panels (latent — no observed history)
    if type == :factor || type == :both
        h_f, n_factors = size(fc.factors)
        if var !== nothing
            fvars = [_resolve_var(var, String["Factor $i" for i in 1:n_factors])]
        else
            fvars = 1:n_factors
        end
        for vi in fvars
            id = _next_plot_id("fac_fc")
            ptitle = "Factor $vi"
            js = _forecast_panel_js(id, fc.factors[:, vi], fc.factors_lower[:, vi],
                                    fc.factors_upper[:, vi]; has_ci=has_ci, fc_name="Forecast",
                                    history=nothing, n_history=n_history,
                                    base_color=_PLOT_SERIES[1], xlabel="Horizon", ylabel="Factor")
            push!(panels, _PanelSpec(id, ptitle, js))
        end
    end

    # Observable panels (carry optional history)
    if type == :observable || type == :both
        h_o, n_obs_total = size(fc.observables)
        if var !== nothing
            ovars = [_resolve_var(var, String["Observable $i" for i in 1:n_obs_total])]
        elseif type == :both
            ovars = 1:min(n_obs_total, n_obs)
        else
            ovars = 1:min(n_obs_total, 6)
        end
        _fc_validate_history(history, length(ovars), n_obs_total)
        for vi in ovars
            id = _next_plot_id("obs_fc")
            ptitle = "Observable $vi"
            js = _forecast_panel_js(id, fc.observables[:, vi], fc.observables_lower[:, vi],
                                    fc.observables_upper[:, vi]; has_ci=has_ci, fc_name="Forecast",
                                    history=_fc_history_col(history, vi), n_history=n_history,
                                    base_color=_PLOT_SERIES[2], xlabel="Horizon", ylabel="Observable")
            push!(panels, _PanelSpec(id, ptitle, js))
        end
    end

    if isempty(title)
        ci_part = fc.ci_method == :none ? "" : " ($(fc.ci_method) CI)"
        title = "Factor Model Forecast$ci_part"
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# LPForecast
# =============================================================================

"""
    plot_result(fc::LPForecast; var=nothing, history=nothing, n_history=50, ncols=0, title="", save_path=nothing)

Plot LP direct multi-step forecast. Pass `history` (a `T×n_response` matrix, or a
vector when a single `var` is selected) to draw a pre-sample History line joined to
the forecast fan, with a vertical forecast-origin line.
"""
function plot_result(fc::LPForecast{T};
                     var::Union{Int,String,Nothing}=nothing,
                     history::Union{AbstractVector,AbstractMatrix,Nothing}=nothing,
                     n_history::Int=50,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    h, n_resp = size(fc.forecast)
    # Names for selection are the response-variable names (one per forecast column).
    resp_names = String[fc.varnames[fc.response_vars[vi]] for vi in 1:n_resp]
    vars_to_plot = var === nothing ? (1:n_resp) : [_resolve_var(var, resp_names)]
    _fc_validate_history(history, length(vars_to_plot), n_resp)
    has_ci = fc.ci_method != :none

    panels = _PanelSpec[]
    for vi in vars_to_plot
        id = _next_plot_id("lp_fc")
        ptitle = fc.varnames[fc.response_vars[vi]]
        js = _forecast_panel_js(id, fc.forecast[:, vi], fc.ci_lower[:, vi], fc.ci_upper[:, vi];
                                has_ci=has_ci, fc_name="LP Forecast",
                                history=_fc_history_col(history, vi), n_history=n_history,
                                xlabel="Horizon", ylabel="Forecast")
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        ci_pct = round(Int, 100 * fc.conf_level)
        title = "LP Forecast (h=$(fc.horizon), $(ci_pct)% CI)"
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# ForecastEvaluation — the `plot_result` dispatch moved to plotting/fceval.jl (PLT-38),
# where it gained the `view=:metrics/:theil` API alongside the other forecast-evaluation
# and Local-Projection plots. Keeping a single method there avoids a same-type dispatch
# collision that would silently override on include.
# =============================================================================
