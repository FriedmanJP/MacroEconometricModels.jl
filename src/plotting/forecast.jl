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
                            xlabel::String, ylabel::String,
                            extra_line::Union{AbstractVector,Nothing}=nothing,
                            extra_name::String="Reference")
    data_json = _forecast_data_json(fc, lo, hi; history=history, n_history=n_history,
                                    extra=extra_line)
    has_extra = extra_line !== nothing
    if history === nothing
        names = has_extra ? [fc_name, extra_name] : [fc_name]
        colors = has_extra ? [base_color, _PLOT_SERIES[3]] : [base_color]
        keys = has_extra ? ["fc", "extra"] : ["fc"]
        dash = has_extra ? ["", "4,4"] : [""]
        s_json = _series_json(names, colors; keys=keys, dash=dash)
        bands = has_ci ?
            "[{\"lo_key\":\"ci_lo\",\"hi_key\":\"ci_hi\",\"color\":\"$(base_color)\",\"alpha\":$(_PLOT_CI_ALPHA)}]" : "[]"
        refs = "[]"
    else
        names = has_extra ? ["History", fc_name, extra_name] : ["History", fc_name]
        colors = has_extra ? [_PLOT_SERIES[1], _PLOT_SERIES[2], _PLOT_SERIES[3]] :
                             [_PLOT_SERIES[1], _PLOT_SERIES[2]]
        keys = has_extra ? ["hist", "fc", "extra"] : ["hist", "fc"]
        dash = has_extra ? ["", "6,3", "4,4"] : ["", "6,3"]
        s_json = _series_json(names, colors; keys=keys, dash=dash)
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
# ConditionalForecast
# =============================================================================

"""
    plot_result(fc::ConditionalForecast; var=nothing, history=nothing, n_history=50, ncols=0, title="", save_path=nothing)

Plot a Waggoner-Zha conditional forecast with its bands. The unconditional forecast is
drawn alongside as a reference line, so the effect of the conditioning path is visible
panel by panel.
"""
function plot_result(fc::ConditionalForecast{T};
                     var::Union{Int,String,Nothing}=nothing,
                     history::Union{AbstractVector,AbstractMatrix,Nothing}=nothing,
                     n_history::Int=50,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    h, n_vars = size(fc.forecast)
    vars_to_plot = var === nothing ? (1:n_vars) : [_resolve_var(var, fc.varnames)]
    _fc_validate_history(history, length(vars_to_plot), n_vars)
    constrained = Set(c.variable::Int for c in fc.conditions)

    panels = _PanelSpec[]
    for vi in vars_to_plot
        id = _next_plot_id("cond_fc")
        ptitle = vi in constrained ? "$(fc.varnames[vi]) (conditioned)" : fc.varnames[vi]
        js = _forecast_panel_js(id, fc.forecast[:, vi], fc.ci_lower[:, vi], fc.ci_upper[:, vi];
                                has_ci=true, fc_name="Conditional",
                                history=_fc_history_col(history, vi), n_history=n_history,
                                xlabel="Horizon", ylabel="Forecast",
                                extra_line=fc.unconditional[:, vi],
                                extra_name="Unconditional")
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        ci_pct = round(Int, 100 * fc.conf_level)
        title = "Conditional Forecast ($(ci_pct)% band, $(fc.identification))"
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

# =============================================================================
# GrowthAtRiskResult (QFM-07)
# =============================================================================

const _GAR_PLOT_VIEWS = (:fan, :gar, :density)

# Build the time-series fan / Growth-at-Risk panel for GrowthAtRiskResult.
function _plot_gar_fan(g::GrowthAtRiskResult{T};
                       quantile_idx::Int=0,
                       ncols::Int=0,
                       title::String="") where {T}
    T_eff = g.n_eff
    M = length(g.taus)
    perm = sortperm(g.taus)

    q_idx = quantile_idx > 0 ? quantile_idx :
            (M >= 1 ? argmin(abs.(Float64.(g.taus) .- 0.5)) : 0)
    q_label = q_idx > 0 ? (quantile_idx > 0 ? "Q(τ=$(_fmt(g.taus[q_idx]; digits=2)))" :
                                              "Median (τ=$(_fmt(g.taus[q_idx]; digits=2)))") : ""

    has_outer = M >= 2
    lo_idx = has_outer ? perm[1] : 0
    hi_idx = has_outer ? perm[end] : 0

    has_inner = M >= 4
    ilo_idx = has_inner ? perm[max(2, fld(M, 4))] : 0
    ihi_idx = has_inner ? perm[min(M - 1, M + 1 - max(2, fld(M, 4)))] : 0
    if has_inner && (ilo_idx == ihi_idx || (ilo_idx == lo_idx && ihi_idx == hi_idx))
        has_inner = false
    end

    rows = Vector{Pair{String,String}}[]
    for t in 1:T_eff
        es_v = g.expected_shortfall[t]
        gar_v = g.gar[t]
        tail_ok = isfinite(es_v) && isfinite(gar_v)
        row = Pair{String,String}[
            "x" => _json(t),
            "actual" => _json(g.actual_target[t]),
            "gar" => _json(gar_v),
            "es" => _json(es_v),
            "tail_lo" => tail_ok ? _json(min(es_v, gar_v)) : "null",
            "tail_hi" => tail_ok ? _json(max(es_v, gar_v)) : "null",
        ]
        if q_idx > 0
            push!(row, "q_sel" => _json(g.fitted_quantiles[t, q_idx]))
        end
        if has_outer
            push!(row, "q_lo" => _json(g.fitted_quantiles[t, lo_idx]))
            push!(row, "q_hi" => _json(g.fitted_quantiles[t, hi_idx]))
        end
        if has_inner
            push!(row, "q_ilo" => _json(g.fitted_quantiles[t, ilo_idx]))
            push!(row, "q_ihi" => _json(g.fitted_quantiles[t, ihi_idx]))
        end
        push!(rows, row)
    end
    data_json = _json_array_of_objects(rows)

    gar_pct = round(Int, 100 * Float64(g.alpha))
    s_names = String["Actual ($(g.target_name))", "GaR ($(gar_pct)%)", "Expected Shortfall"]
    s_colors = String[_PLOT_SERIES[1], _PLOT_SERIES[2], _PLOT_SERIES[4]]
    s_keys = String["actual", "gar", "es"]
    s_dash = String["", "", "5,3"]
    if q_idx > 0
        push!(s_names, q_label)
        push!(s_colors, _PLOT_SERIES[3])
        push!(s_keys, "q_sel")
        push!(s_dash, "4,3")
    end
    s_json = _series_json(s_names, s_colors; keys=s_keys, dash=s_dash)

    band_parts = String[]
    if has_outer
        lbl = "$(round(Int, 100 * Float64(g.taus[lo_idx])))–$(round(Int, 100 * Float64(g.taus[hi_idx])))% Fan"
        push!(band_parts, "{\"lo_key\":\"q_lo\",\"hi_key\":\"q_hi\",\"color\":$(_json(_PLOT_SERIES[1])),\"alpha\":0.12,\"name\":$(_json(lbl))}")
    end
    if has_inner
        lbl = "$(round(Int, 100 * Float64(g.taus[ilo_idx])))–$(round(Int, 100 * Float64(g.taus[ihi_idx])))% Fan"
        push!(band_parts, "{\"lo_key\":\"q_ilo\",\"hi_key\":\"q_ihi\",\"color\":$(_json(_PLOT_SERIES[1])),\"alpha\":0.22,\"name\":$(_json(lbl))}")
    end
    push!(band_parts, "{\"lo_key\":\"tail_lo\",\"hi_key\":\"tail_hi\",\"color\":$(_json(_PLOT_ALERT)),\"alpha\":0.16,\"name\":\"ES–GaR Tail\"}")
    bands_json = "[" * join(band_parts, ",") * "]"
    refs_json = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"

    id = _next_plot_id("gar_fan")
    js = _render_line_js(id, data_json, s_json;
                         bands_json=bands_json, ref_lines_json=refs_json,
                         integer_x=true, xlabel="Period", ylabel=g.target_name)
    ptitle = "$(g.target_name) (h=$(g.horizon), α=$(_fmt(g.alpha; digits=2)), $(g.density))"
    ftitle = isempty(title) ? "Growth-at-Risk Forecast Fan — $(g.target_name)" : title
    return _make_plot([_PanelSpec(id, ptitle, js)]; title=ftitle, ncols=(ncols <= 0 ? 1 : ncols))
end

# Build predictive density / quantile curve + OOS tail-risk bar summary for GrowthAtRiskResult.
function _plot_gar_density(g::GrowthAtRiskResult{T};
                           quantile_idx::Int=0,
                           ncols::Int=0,
                           title::String="") where {T}
    M = length(g.taus)
    perm = sortperm(g.taus)
    panels = _PanelSpec[]

    # Panel 1: OOS T+h|T predictive PDF (when parametric density params exist) or quantile curve
    id1 = _next_plot_id("gar_dens")
    has_skewt = g.density === :skewed_t && length(g.oos_density_params) == 4 &&
                all(isfinite, g.oos_density_params) &&
                g.oos_density_params[2] > zero(T) && g.oos_density_params[4] > zero(T)
    has_norm = g.density === :normal && length(g.oos_density_params) == 2 &&
               all(isfinite, g.oos_density_params) && g.oos_density_params[2] > zero(T)

    if has_skewt || has_norm
        mu = Float64(g.oos_density_params[1])
        sig = Float64(g.oos_density_params[2])
        fin_q = Float64[Float64(v) for v in g.oos_quantiles if isfinite(v)]
        lo_candidates = Float64[mu - 3.5 * sig]
        hi_candidates = Float64[mu + 3.5 * sig]
        isfinite(g.oos_expected_shortfall) && push!(lo_candidates, Float64(g.oos_expected_shortfall) - sig)
        isfinite(g.oos_gar) && push!(lo_candidates, Float64(g.oos_gar) - sig)
        if !isempty(fin_q)
            push!(lo_candidates, minimum(fin_q) - sig)
            push!(hi_candidates, maximum(fin_q) + 1.5 * sig)
        end
        y_lo = minimum(lo_candidates)
        y_hi = maximum(hi_candidates)
        y_hi <= y_lo && (y_hi = y_lo + 1.0)
        y_grid = collect(range(y_lo, y_hi; length=120))
        dens_vals = if has_skewt
            alp = Float64(g.oos_density_params[3])
            nu = Float64(g.oos_density_params[4])
            Float64[_skewt_pdf(yg, mu, sig, alp, nu) for yg in y_grid]
        else
            nd = Distributions.Normal(mu, sig)
            Float64[Distributions.pdf(nd, yg) for yg in y_grid]
        end
        data1 = _keyed_rows_json(y_grid, ["pdf" => dens_vals])
        dlabel = has_skewt ? "Skewed-t Density" : "Normal Density"
        s1 = _series_json([dlabel], [_PLOT_SERIES[1]]; keys=["pdf"])
        ref_parts = String[]
        if isfinite(g.oos_gar)
            push!(ref_parts, "{\"value\":$(_json(Float64(g.oos_gar))),\"color\":$(_json(_PLOT_SERIES[2])),\"dash\":\"5,3\",\"axis\":\"x\"}")
        end
        if isfinite(g.oos_expected_shortfall)
            push!(ref_parts, "{\"value\":$(_json(Float64(g.oos_expected_shortfall))),\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"4,2\",\"axis\":\"x\"}")
        end
        if quantile_idx > 0 && isfinite(g.oos_quantiles[quantile_idx])
            push!(ref_parts, "{\"value\":$(_json(Float64(g.oos_quantiles[quantile_idx]))),\"color\":$(_json(_PLOT_SERIES[3])),\"dash\":\"3,3\",\"axis\":\"x\"}")
        end
        refs1 = "[" * join(ref_parts, ",") * "]"
        js1 = _render_line_js(id1, data1, s1; ref_lines_json=refs1,
                              xlabel=g.target_name, ylabel="Density")
        push!(panels, _PanelSpec(id1, "Predictive Density (T+$(g.horizon)|T)", js1))
    else
        # Empirical quantile curve across taus
        if M >= 2
            taus_s = Float64[Float64(g.taus[j]) for j in perm]
            oq_s = Float64[Float64(g.oos_quantiles[j]) for j in perm]
            data1 = _keyed_rows_json(taus_s, ["q" => oq_s])
            s1 = _series_json(["OOS Quantiles"], [_PLOT_SERIES[1]]; keys=["q"])
            ref_parts = String[]
            isfinite(g.oos_gar) && push!(ref_parts, "{\"value\":$(_json(Float64(g.oos_gar))),\"color\":$(_json(_PLOT_SERIES[2])),\"dash\":\"5,3\",\"axis\":\"y\"}")
            isfinite(g.oos_expected_shortfall) && push!(ref_parts, "{\"value\":$(_json(Float64(g.oos_expected_shortfall))),\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"4,2\",\"axis\":\"y\"}")
            if quantile_idx > 0
                push!(ref_parts, "{\"value\":$(_json(Float64(g.taus[quantile_idx]))),\"color\":$(_json(_PLOT_SERIES[3])),\"dash\":\"3,3\",\"axis\":\"x\"}")
            end
            js1 = _render_line_js(id1, data1, s1;
                                  ref_lines_json="[" * join(ref_parts, ",") * "]",
                                  xlabel="Quantile τ", ylabel=g.target_name)
        else
            rows1 = [["x" => _json("τ=$(_fmt(g.taus[1]; digits=2))"),
                      "q" => _json(g.oos_quantiles[1])]]
            s1 = _series_json(["OOS Quantile"], [_PLOT_SERIES[1]]; keys=["q"])
            js1 = _render_bar_js(id1, _json_array_of_objects(rows1), s1;
                                 mode="grouped", xlabel="Quantile", ylabel=g.target_name)
        end
        push!(panels, _PanelSpec(id1, "Conditional Quantile Function (T+$(g.horizon)|T)", js1))
    end

    # Panel 2: OOS Tail Risk Summary (horizontal bar chart)
    id2 = _next_plot_id("gar_sum")
    gar_pct = round(Int, 100 * Float64(g.alpha))
    bar_rows = Vector{Pair{String,String}}[
        ["x" => _json("GaR ($(gar_pct)%)"), "val" => _json(g.oos_gar)],
        ["x" => _json("Expected Shortfall"), "val" => _json(g.oos_expected_shortfall)],
    ]
    if quantile_idx > 0
        push!(bar_rows, ["x" => _json("Q(τ=$(_fmt(g.taus[quantile_idx]; digits=2)))"),
                         "val" => _json(g.oos_quantiles[quantile_idx])])
    elseif M >= 1
        med_i = argmin(abs.(Float64.(g.taus) .- 0.5))
        push!(bar_rows, ["x" => _json("Q(τ=$(_fmt(g.taus[med_i]; digits=2)))"),
                         "val" => _json(g.oos_quantiles[med_i])])
    end
    if isfinite(g.oos_downside_entropy)
        push!(bar_rows, ["x" => _json("Downside Entropy"), "val" => _json(g.oos_downside_entropy)])
    end
    if isfinite(g.oos_upside_entropy)
        push!(bar_rows, ["x" => _json("Upside Entropy"), "val" => _json(g.oos_upside_entropy)])
    end
    s2 = _series_json(["OOS T+$(g.horizon)|T"], [_PLOT_SERIES[2]]; keys=["val"])
    js2 = _render_bar_js(id2, _json_array_of_objects(bar_rows), s2;
                         mode="grouped", orientation="h", xlabel="Value", ylabel="")
    push!(panels, _PanelSpec(id2, "Tail Risk Summary (T+$(g.horizon)|T)", js2))

    ftitle = isempty(title) ?
        "Growth-at-Risk Predictive Density — $(g.target_name) (h=$(g.horizon), $(g.density))" : title
    return _make_plot(panels; title=ftitle, ncols=(ncols <= 0 ? 2 : ncols))
end

"""
    plot_result(g::GrowthAtRiskResult; view=:fan, quantile_idx=0, ncols=0, title="", save_path=nothing)

Plot Growth-at-Risk conditional quantile forecasts and predictive density metrics.

# Views
- `:fan` / `:gar` — time series of `actual_target`, `gar`, and `expected_shortfall`
  (with shaded ES–GaR lower-tail ribbon) overlaid on `fitted_quantiles` fan bands
  (highlighting `quantile_idx` when `quantile_idx > 0`).
- `:density` — `T+h|T` predictive density (`:skewed_t` or `:normal`) or quantile
  curve (`:none`) with `oos_gar` / `oos_expected_shortfall` markers, paired with an
  out-of-sample tail-risk summary bar chart.

# Keyword Arguments
- `view::Symbol = :fan` — exhibit view (`:fan`, `:gar`, or `:density`)
- `quantile_idx::Int = 0` — optional 1-based index into `g.taus` to highlight (`0` = median/auto)
- `ncols::Int = 0` — number of grid columns (`0` = auto)
- `title::String = ""` — custom figure title (`""` = auto-generated)
- `save_path::Union{String,Nothing} = nothing` — optional output HTML path
"""
function plot_result(g::GrowthAtRiskResult{T};
                     view::Symbol=:fan,
                     quantile_idx::Int=0,
                     ncols::Int=0,
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    view in _GAR_PLOT_VIEWS || throw(ArgumentError(
        "Unknown view :$view for GrowthAtRiskResult. Valid views: $(_GAR_PLOT_VIEWS)"))
    M = length(g.taus)
    (0 <= quantile_idx <= M) || throw(ArgumentError(
        "quantile_idx must be in 0:$M, got $quantile_idx"))

    p = if view === :density
        _plot_gar_density(g; quantile_idx=quantile_idx, ncols=ncols, title=title)
    else
        _plot_gar_fan(g; quantile_idx=quantile_idx, ncols=ncols, title=title)
    end
    save_path !== nothing && save_plot(p, save_path)
    return p
end
