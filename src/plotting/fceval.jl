# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
PLT-38 — forecast-evaluation & Local-Projection plotting extras. Adds the
`ForecastEvaluation` metric/Theil views (the single `plot_result(::ForecastEvaluation)`
method now lives here, not in forecast.jl), Mincer–Zarnowitz efficiency lines, the
Diebold–Mariano / Clark–West single-statistic dot-whiskers, forecast-combination
weights, the two-regime state-dependent LP IRF, and the LP-family delegation to each
type's own `ImpulseResponse` builder. Renderers are the frozen render.jl primitives (A1);
converters are the frozen helpers.jl converters plus lane-local `_`-prefixed helpers (A5).
"""

# =============================================================================
# ForecastEvaluation — metric / Theil-decomposition views
# =============================================================================

"""
    plot_result(ev::ForecastEvaluation; view=:metrics, metric=nothing, title="", save_path=nothing)

Plot a multi-model forecast evaluation.

- `view=:metrics` (default) — a grouped bar comparing the models within each accuracy
  metric (one bar group per metric, one bar per model). Pass `metric="RMSE"` to draw a
  single ranked metric instead.
- `view=:theil` — a stacked bar of each model's Theil MSE decomposition (bias / variance
  / covariance proportions, summing to 1).

Unknown `view` → `ArgumentError` (C5).
"""
function plot_result(ev::ForecastEvaluation{T};
                     view::Symbol=:metrics, metric::Union{String,Nothing}=nothing,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    M = length(ev.models)
    if view === :metrics && metric !== nothing
        # Single ranked metric (best/smallest first).
        k = findfirst(==(metric), ev.metrics)
        k === nothing && throw(ArgumentError("metric must be one of $(ev.metrics); got \"$metric\""))
        ord = sortperm(ev.values[:, k])
        id = _next_plot_id("fceval_rank")
        rows = Vector{Pair{String,String}}[]
        for j in ord
            push!(rows, ["x" => _json(ev.models[j]), "s1" => _json(ev.values[j, k])])
        end
        data = _json_array_of_objects(rows)
        s = _series_json([metric], [_PLOT_COLORS[2]]; keys=["s1"])
        js = _render_bar_js(id, data, s; mode="stacked", xlabel="Model", ylabel=metric)
        isempty(title) && (title = "Forecast Accuracy — $metric (n=$(ev.n))")
        p = _make_plot([_PanelSpec(id, title, js)]; title=title)
        save_path !== nothing && save_plot(p, save_path)
        return p
    elseif view === :metrics
        # Grouped bar: one group per metric, one bar per model.
        id = _next_plot_id("fceval_metrics")
        rows = Vector{Pair{String,String}}[]
        for (mi, mname) in enumerate(ev.metrics)
            row = Pair{String,String}["x" => _json(mname)]
            for j in 1:M
                push!(row, "s$j" => _json(ev.values[j, mi]))
            end
            push!(rows, row)
        end
        data = _json_array_of_objects(rows)
        s = _series_json(ev.models, _colors_for(ev.models); keys=["s$j" for j in 1:M])
        js = _render_bar_js(id, data, s; mode="grouped", xlabel="Metric", ylabel="Value")
        isempty(title) && (title = "Forecast Accuracy Metrics (n=$(ev.n))")
        p = _make_plot([_PanelSpec(id, title, js)]; title=title)
        save_path !== nothing && save_plot(p, save_path)
        return p
    elseif view === :theil
        id = _next_plot_id("fceval_theil")
        comp = ["Bias", "Variance", "Covariance"]
        rows = Vector{Pair{String,String}}[]
        for j in 1:M
            push!(rows, ["x" => _json(ev.models[j]),
                         "s1" => _json(ev.decomp[j, 1]),
                         "s2" => _json(ev.decomp[j, 2]),
                         "s3" => _json(ev.decomp[j, 3])])
        end
        data = _json_array_of_objects(rows)
        s = _series_json(comp, _colors_for(comp); keys=["s1", "s2", "s3"])
        js = _render_bar_js(id, data, s; mode="stacked", xlabel="Model",
                            ylabel="Theil MSE proportion")
        isempty(title) && (title = "Theil MSE Decomposition (n=$(ev.n))")
        p = _make_plot([_PanelSpec(id, title, js)]; title=title)
        save_path !== nothing && save_plot(p, save_path)
        return p
    end
    throw(ArgumentError("Unknown view :$view for ForecastEvaluation; use :metrics or :theil."))
end

# =============================================================================
# MincerZarnowitzResult — efficiency line vs the 45° reference
# =============================================================================

"""
    plot_result(r::MincerZarnowitzResult; title="", save_path=nothing)

Plot the Mincer–Zarnowitz efficiency line `actual = a + b·forecast` against the 45°
reference (`a=0, b=1`, forecast optimality) over a normalized forecast range. The result
stores no actual/forecast series, so **no scatter of points is drawn** (C6); the fitted
coefficients and the joint Wald p-value are annotated via `_fmt`/`_format_pvalue` (C9).
"""
function plot_result(r::MincerZarnowitzResult{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    xlo, xhi = -3.0, 3.0
    id = _next_plot_id("mz_line")
    rows = Vector{Pair{String,String}}[]
    for x in (xlo, xhi)
        push!(rows, ["x" => _json(x),
                     "eff" => _json(Float64(r.a) + Float64(r.b) * x),
                     "ident" => _json(x)])
    end
    data = _json_array_of_objects(rows)
    s = _series_json(["Efficiency a+b·fc", "45° (a=0, b=1)"],
                     [_PLOT_COLORS[1], _PLOT_ALERT]; keys=["eff", "ident"], dash=["", "6,3"])
    js = _render_line_js(id, data, s; xlabel="Forecast (normalized)", ylabel="Actual")
    ann = "a=$(_fmt(r.a)), b=$(_fmt(r.b)), Wald p=$(_format_pvalue(r.pvalue_wald))"
    isempty(title) && (title = "Mincer–Zarnowitz Efficiency")
    note = "$ann — no scatter (no forecast/actual points stored)."
    p = _make_plot([_PanelSpec(id, "Mincer–Zarnowitz — $ann", js)]; title=title, note=note)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# DMTestResult / ClarkWestResult — single-statistic dot-whisker
# =============================================================================

# Lane-local (A5): draw one loss-differential statistic ± its SE as a horizontal
# dot-whisker with a zero reference (a single test number is an annotation-backed dot,
# not a bar chart of one value — form table).
function _single_stat_dotwhisker(name::String, point::Real, se::Real, ann::String,
                                 title::String, save_path)
    id = _next_plot_id("fc_dm")
    lo = Float64(point) - 1.96 * Float64(se)
    hi = Float64(point) + 1.96 * Float64(se)
    data = "[{\"name\":$(_json(name)),\"effect\":$(_json(Float64(point)))," *
           "\"ci_lo\":$(_json(lo)),\"ci_hi\":$(_json(hi))}]"
    js = _render_coef_plot_js(id, data; ref_value=0, xlabel="Loss differential", ylabel="")
    p = _make_plot([_PanelSpec(id, ann, js)]; title=isempty(title) ? name : title)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(r::DMTestResult; title="", save_path=nothing)

Plot the Diebold–Mariano test as a single dot-whisker: the mean loss differential `dbar`
± its 95% interval (`√(lrvar/T_obs)`) against a zero line, with the DM statistic and
p-value annotated (C9).
"""
function plot_result(r::DMTestResult{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    se = sqrt(max(Float64(r.lrvar) / max(r.T_obs, 1), 0.0))
    ann = "DM: d̄=$(_fmt(r.dbar)), stat=$(_fmt(r.statistic)), p=$(_format_pvalue(r.pvalue))"
    _single_stat_dotwhisker("DM loss diff.", r.dbar, se, ann, title, save_path)
end

"""
    plot_result(r::ClarkWestResult; title="", save_path=nothing)

Plot the Clark–West test as a single dot-whisker: the adjusted mean differential `fbar`
± its 95% interval (`√(lrvar/T_obs)`) against a zero line, with the CW statistic and
p-value annotated (C9).
"""
function plot_result(r::ClarkWestResult{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    se = sqrt(max(Float64(r.lrvar) / max(r.T_obs, 1), 0.0))
    ann = "CW: f̄=$(_fmt(r.fbar)), stat=$(_fmt(r.statistic)), p=$(_format_pvalue(r.pvalue))"
    _single_stat_dotwhisker("CW adj. diff.", r.fbar, se, ann, title, save_path)
end

# =============================================================================
# ForecastCombination — weights (diverging) + MSE bars
# =============================================================================

"""
    plot_result(fc::ForecastCombination; title="", save_path=nothing)

Plot forecast-combination results: a horizontal bar of each model's combination weight
(Granger–Ramanathan weights may be negative ⇒ a diverging bar with a zero line, names
read horizontally) and a horizontal bar of each model's standalone MSE.
"""
function plot_result(fc::ForecastCombination{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    M = length(fc.models)
    # Panel 1: weights (horizontal, diverging around 0)
    id1 = _next_plot_id("fc_w")
    rows1 = Vector{Pair{String,String}}[]
    for j in 1:M
        push!(rows1, ["x" => _json(fc.models[j]), "s1" => _json(fc.weights[j])])
    end
    data1 = _json_array_of_objects(rows1)
    s1 = _series_json(["Weight"], [_PLOT_COLORS[1]]; keys=["s1"])
    js1 = _render_bar_js(id1, data1, s1; mode="stacked", orientation="h",
                         xlabel="Combination weight", ylabel="")
    p1 = _PanelSpec(id1, "Combination Weights ($(fc.method))", js1)

    # Panel 2: standalone MSE (horizontal, all positive)
    id2 = _next_plot_id("fc_mse")
    rows2 = Vector{Pair{String,String}}[]
    for j in 1:M
        push!(rows2, ["x" => _json(fc.models[j]), "s1" => _json(fc.mse[j])])
    end
    data2 = _json_array_of_objects(rows2)
    s2 = _series_json(["MSE"], [_PLOT_COLORS[2]]; keys=["s1"])
    js2 = _render_bar_js(id2, data2, s2; mode="stacked", orientation="h",
                         xlabel="Standalone MSE", ylabel="")
    p2 = _PanelSpec(id2, "Standalone MSE", js2)

    isempty(title) && (title = "Forecast Combination ($(fc.method))")
    p = _make_plot([p1, p2]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# StateLPModel — two-regime state-dependent IRF
# =============================================================================

"""
    plot_result(m::StateLPModel; regime=:both, var=nothing, conf_level=0.95, ncols=0,
                title="", save_path=nothing)

Plot state-dependent Local-Projection IRFs. With `regime=:both` (default) the expansion
and recession responses (with CI bands, colour- and dash-distinct) are overlaid per
response variable; `regime=:expansion`/`:recession` draws a single regime. `var` selects
the response by index or name (Int/String, `_resolve_var`, C3). Integer horizon ticks,
zero reference line.
"""
function plot_result(m::StateLPModel{T};
                     regime::Symbol=:both, var::Union{Int,String,Nothing}=nothing,
                     conf_level::Real=0.95, ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    regime in (:both, :expansion, :recession) ||
        throw(ArgumentError("regime must be :both, :expansion, or :recession, got :$regime"))
    resp_names = String[m.varnames[m.response_vars[j]] for j in 1:length(m.response_vars)]
    vars_to_plot = var === nothing ? (1:length(resp_names)) : [_resolve_var(var, resp_names)]
    H1 = m.horizon + 1
    refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"

    panels = _PanelSpec[]
    for vi in vars_to_plot
        id = _next_plot_id("state_lp")
        rows = Vector{Pair{String,String}}[]
        if regime === :both
            ir = state_irf(m; regime=:both, conf_level=conf_level)
            for h in 1:H1
                push!(rows, ["x" => _json(h - 1),
                             "exp" => _json(ir.expansion.values[h, vi]),
                             "exp_lo" => _json(ir.expansion.ci_lower[h, vi]),
                             "exp_hi" => _json(ir.expansion.ci_upper[h, vi]),
                             "rec" => _json(ir.recession.values[h, vi]),
                             "rec_lo" => _json(ir.recession.ci_lower[h, vi]),
                             "rec_hi" => _json(ir.recession.ci_upper[h, vi])])
            end
            data = _json_array_of_objects(rows)
            s = _series_json(["Expansion", "Recession"],
                             [_PLOT_SERIES[1], _PLOT_SERIES[2]];
                             keys=["exp", "rec"], dash=["", "6,3"])
            bands = "[{\"lo_key\":\"exp_lo\",\"hi_key\":\"exp_hi\",\"name\":\"Expansion CI\"," *
                    "\"color\":\"$(_PLOT_SERIES[1])\",\"alpha\":$(_PLOT_CI_ALPHA)}," *
                    "{\"lo_key\":\"rec_lo\",\"hi_key\":\"rec_hi\",\"name\":\"Recession CI\"," *
                    "\"color\":\"$(_PLOT_SERIES[2])\",\"alpha\":$(_PLOT_CI_ALPHA)}]"
        else
            ir = state_irf(m; regime=regime, conf_level=conf_level)
            for h in 1:H1
                push!(rows, ["x" => _json(h - 1),
                             "irf" => _json(ir.values[h, vi]),
                             "lo" => _json(ir.ci_lower[h, vi]),
                             "hi" => _json(ir.ci_upper[h, vi])])
            end
            data = _json_array_of_objects(rows)
            rlabel = regime === :expansion ? "Expansion" : "Recession"
            s = _series_json([rlabel], [_PLOT_SERIES[1]]; keys=["irf"])
            bands = "[{\"lo_key\":\"lo\",\"hi_key\":\"hi\",\"name\":\"$(rlabel) CI\"," *
                    "\"color\":\"$(_PLOT_SERIES[1])\",\"alpha\":$(_PLOT_CI_ALPHA)}]"
        end
        js = _render_line_js(id, data, s; bands_json=bands, ref_lines_json=refs,
                             integer_x=true, xlabel="Horizon", ylabel="Response")
        push!(panels, _PanelSpec(id, "$(resp_names[vi]) ← $(m.varnames[m.shock_var])", js))
    end

    if isempty(title)
        rlab = regime === :both ? "expansion vs recession" : String(regime)
        title = "State-Dependent LP IRF ($rlab)"
    end
    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# LP-family delegation to each type's ImpulseResponse builder (PLT-38) + the LPModel /
# LPIVModel residual-diagnostics view (PLT-24, single-method rule — combined here).
# =============================================================================

# Lane-local (A5): coerce an LP residual object (Vector, Matrix, or per-horizon
# Vector{Matrix}) to a Float64 residual vector for the chosen horizon/response column.
function _lp_resid_vector(r, h::Int, vi::Int)
    obj = r isa AbstractVector && !isempty(r) && r[1] isa AbstractMatrix ?
          r[clamp(h + 1, 1, length(r))] : r
    if obj isa AbstractMatrix
        col = clamp(vi, 1, size(obj, 2))
        return Float64[Float64(v) for v in @view obj[:, col]]
    end
    Float64[Float64(v) for v in obj]
end

function _lp_diag_plot(m, resid_obj, h::Int, resp_names, var, acf_lags, title, save_path)
    vi = _resolve_var(var, resp_names, 1)
    resid = _lp_resid_vector(resid_obj, h, vi)
    fitted = zeros(Float64, length(resid))  # direct-h LP has no single stored fitted series
    panels = _residual_diagnostics_panels(resid, fitted; varname=resp_names[vi], acf_lags=acf_lags)
    isempty(title) && (title = "LP Residual Diagnostics (h=$h)")
    p = _make_plot(panels; title=title, ncols=2)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(m::LPModel; view=:irf, var=nothing, h=0, conf_level=0.95, ncols=0,
                title="", save_path=nothing)

Plot a Local Projection. `view=:irf` (default) builds `lp_irf(m)` and delegates to the
`LPImpulseResponse` plot; `view=:diagnostics` draws the four-panel residual figure for
horizon `h` and response variable `var` (PLT-24).
"""
function plot_result(m::LPModel{T}; view::Symbol=:irf,
                     var::Union{Int,String,Nothing}=nothing, h::Int=0,
                     conf_level::Real=0.95, ncols::Int=0, acf_lags::Int=0,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    if view === :irf
        return plot_result(lp_irf(m; conf_level=conf_level); var=var, ncols=ncols,
                           title=title, save_path=save_path)
    elseif view === :diagnostics
        resp_names = String[m.varnames[m.response_vars[j]] for j in 1:length(m.response_vars)]
        return _lp_diag_plot(m, residuals(m, h), h, resp_names, var, acf_lags, title, save_path)
    end
    throw(ArgumentError("Unknown view :$view for LPModel; use :irf or :diagnostics."))
end

"""
    plot_result(m::LPIVModel; view=:irf, var=nothing, h=0, conf_level=0.95, ncols=0,
                title="", save_path=nothing)

Plot an IV Local Projection. `view=:irf` (default) delegates to the `lp_iv_irf`
`LPImpulseResponse` plot; `view=:diagnostics` draws the residual figure (PLT-24).
"""
function plot_result(m::LPIVModel{T}; view::Symbol=:irf,
                     var::Union{Int,String,Nothing}=nothing, h::Int=0,
                     conf_level::Real=0.95, ncols::Int=0, acf_lags::Int=0,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    if view === :irf
        return plot_result(lp_iv_irf(m; conf_level=conf_level); var=var, ncols=ncols,
                           title=title, save_path=save_path)
    elseif view === :diagnostics
        resp_names = String[m.varnames[m.response_vars[j]] for j in 1:length(m.response_vars)]
        return _lp_diag_plot(m, residuals(m), h, resp_names, var, acf_lags, title, save_path)
    end
    throw(ArgumentError("Unknown view :$view for LPIVModel; use :irf or :diagnostics."))
end

"""
    plot_result(m::SmoothLPModel; var=nothing, ncols=0, title="", save_path=nothing)

Plot a smooth (penalized) Local Projection — delegates to the `smooth_lp_irf`
`LPImpulseResponse` plot (PLT-38).
"""
function plot_result(m::SmoothLPModel{T};
                     var::Union{Int,String,Nothing}=nothing, conf_level::Real=0.95,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    plot_result(smooth_lp_irf(m; conf_level=conf_level); var=var, ncols=ncols,
                title=title, save_path=save_path)
end

"""
    plot_result(m::PropensityLPModel; var=nothing, ncols=0, title="", save_path=nothing)

Plot a propensity-score (doubly-robust) Local Projection — delegates to the
`propensity_irf` `LPImpulseResponse` plot (PLT-38).
"""
function plot_result(m::PropensityLPModel{T};
                     var::Union{Int,String,Nothing}=nothing, conf_level::Real=0.95,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    plot_result(propensity_irf(m; conf_level=conf_level); var=var, ncols=ncols,
                title=title, save_path=save_path)
end
