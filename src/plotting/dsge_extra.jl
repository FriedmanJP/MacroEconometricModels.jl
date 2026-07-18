# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
plot_result methods for additional DSGE outputs (PLT-33). The existing DSGE plots
(`BayesianDSGE`, models) stay in `models.jl`; these dispatches are additive and
live here for parallel-lane file isolation:

- `PerfectForesightPath` — per-variable transition lines (`view=:levels` from `path`,
  with a per-panel steady-state reference line; `view=:deviations` from `deviations`,
  with a zero line).
- `KalmanSmootherResult` — smoothed state means ± 1.96·√diag(cov) bands, with the
  observed data overlaid when supplied. The struct carries no names, so `varnames`
  is a **required** keyword.
- `DSGEEstimation` — a parameter dot-and-whisker (`θ ± 1.96·stderror`) with a J-test
  annotation (an honest parameter plot; the struct stores no moments, plotrule C6).

(`BayesianDSGESimulation` is plotted by the Bayesian nested-fan dispatch in
`bayesfan.jl` (PLT-28), which owns the Bayesian-fan family.)

All rendering reuses the frozen line / coef / quantile-fan renderers (A1).
"""

# =============================================================================
# Lane-local converters (A5/A6) — helpers.jl is frozen for Wave 2
# =============================================================================

"""
    _pf_path_json(col; xstart=1) -> String

Line `data_json` `[{x, val}, …]` for one variable's transition path `col` over
periods `xstart, xstart+1, …`. Non-finite values serialize to `null` (visible gap).
"""
function _pf_path_json(col::AbstractVector; xstart::Int=1)
    rows = Vector{Pair{String,String}}[]
    for (i, v) in enumerate(col)
        push!(rows, ["x" => _json(xstart + i - 1), "val" => _json(v)])
    end
    _json_array_of_objects(rows)
end

"""
    _smoothed_state_json(states, covs, i; obs=nothing) -> String

Line+band `data_json` `[{x, sm, lo, hi[, obs]}, …]` for state `i` of a smoother
result: `states` is `n×T`, `covs` is `n×n×T`; the band is `sm ± 1.96·√covs[i,i,t]`.
When `obs` (length `T`) is supplied it is emitted as an `obs` overlay column.
Non-finite means/variances serialize to `null` (a visible gap; plotrule A7).
"""
function _smoothed_state_json(states::AbstractMatrix, covs::AbstractArray{<:Any,3},
                              i::Int; obs::Union{AbstractVector,Nothing}=nothing)
    Tn = size(states, 2)
    rows = Vector{Pair{String,String}}[]
    for t in 1:Tn
        m = states[i, t]
        v = covs[i, i, t]
        sd = (isfinite(v) && v > 0) ? sqrt(v) : NaN
        lo = (isfinite(m) && isfinite(sd)) ? m - 1.96 * sd : NaN
        hi = (isfinite(m) && isfinite(sd)) ? m + 1.96 * sd : NaN
        row = Pair{String,String}["x" => _json(t), "sm" => _json(m),
                                  "lo" => _json(lo), "hi" => _json(hi)]
        obs !== nothing && push!(row, "obs" => _json(obs[t]))
        push!(rows, row)
    end
    _json_array_of_objects(rows)
end

"""
    _dsge_coef_json(names, est, se; z=1.96) -> String

Coefficient-plot `data_json` `[{name, effect, ci_lo, ci_hi}, …]` for a parameter
estimate/standard-error pair (`ci = est ± z·se`). Used for the DSGE-GMM parameter
plot; DSGE deep parameters carry no intercept row, so none is omitted.
"""
function _dsge_coef_json(names::AbstractVector, est::AbstractVector,
                         se::AbstractVector; z::Real=1.96)
    rows = Vector{Pair{String,String}}[]
    for i in eachindex(names)
        push!(rows, ["name" => _json(string(names[i])), "effect" => _json(est[i]),
                     "ci_lo" => _json(est[i] - z * se[i]),
                     "ci_hi" => _json(est[i] + z * se[i])])
    end
    _json_array_of_objects(rows)
end

# Which variable indices to draw, honoring an optional `var` selector and a panel cap
# (plotrule C7). Returns (indices, note): `note` is a figure-level cap note or "".
function _dsge_extra_indices(var, names::Vector{String}, max_panels::Int)
    if var !== nothing
        return ([_resolve_var(var, names)], "")
    end
    n = length(names)
    shown = min(n, max_panels)
    idx = collect(1:shown)
    note = _cap_note("variables", shown, n, "max_panels")
    (idx, note)
end

# =============================================================================
# PerfectForesightPath
# =============================================================================

"""
    plot_result(pf::PerfectForesightPath; view=:levels, var=nothing, max_panels=12,
                ncols=0, title="", save_path=nothing)

Per-variable deterministic transition paths. `view=:levels` (default) draws `pf.path`
with a horizontal steady-state reference line per panel (from `pf.spec.steady_state`);
`view=:deviations` draws `pf.deviations` with a zero line. `var` selects one variable
by `Int` or name (`pf.spec.varnames`); otherwise up to `max_panels` variables are
drawn, with any cap surfaced in a figure note (plotrule C7). Unknown `view` throws an
`ArgumentError`.
"""
function plot_result(pf::PerfectForesightPath{T};
                     view::Symbol=:levels, var::Union{Int,String,Nothing}=nothing,
                     max_panels::Int=12, ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    view in (:levels, :deviations) ||
        throw(ArgumentError("Unknown view :$view. Valid views: :levels, :deviations"))
    names = pf.spec.varnames
    M = view === :levels ? pf.path : pf.deviations
    ss = pf.spec.steady_state
    idx, note = _dsge_extra_indices(var, names, max_panels)
    panels = _PanelSpec[]
    for j in idx
        id = _next_plot_id("pf_path")
        data_json = _pf_path_json(M[:, j])
        series_json = _series_json([names[j]], [_PLOT_SERIES[1]]; keys=["val"])
        if view === :levels
            ssj = (j <= length(ss)) ? ss[j] : zero(T)
            refs = "[{\"value\":$(_json(ssj)),\"axis\":\"y\",\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"5,4\"}]"
        else
            refs = "[{\"value\":0,\"axis\":\"y\",\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"4,3\"}]"
        end
        js = _render_line_js(id, data_json, series_json; ref_lines_json=refs,
                             xlabel="Period", ylabel=view === :levels ? "Level" : "Deviation")
        push!(panels, _PanelSpec(id, names[j], js))
    end
    vlabel = view === :levels ? "levels" : "deviations from SS"
    ftitle = isempty(title) ? "Perfect-Foresight Transition Path ($vlabel)" : title
    p = _make_plot(panels; title=ftitle, ncols=ncols, note=note)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# KalmanSmootherResult
# =============================================================================

"""
    plot_result(r::KalmanSmootherResult; varnames, data=nothing, var=nothing,
                max_panels=12, ncols=0, title="", save_path=nothing)

Smoothed-state plot: per state, the smoothed mean with a ±1.96·√diag(covariance)
band, optionally overlaying the observed `data`. The struct carries no names, so
`varnames` is **required** and must have length `size(smoothed_states, 1)` — otherwise
an `ArgumentError` is thrown. `data` (when supplied) must be `T_obs × n` or
`n × T_obs`; a mismatched shape throws (plotrule Robustness). `var` selects one state
by `Int` or name; otherwise up to `max_panels` states are drawn (cap noted, C7).
"""
function plot_result(r::KalmanSmootherResult{T};
                     varnames::Union{Nothing,Vector{String}}=nothing,
                     data::Union{Nothing,AbstractMatrix}=nothing,
                     var::Union{Int,String,Nothing}=nothing,
                     max_panels::Int=12, ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    n, Tn = size(r.smoothed_states)
    varnames === nothing && throw(ArgumentError(
        "varnames is required — KalmanSmootherResult stores no variable names " *
        "(pass varnames::Vector{String} of length $n)"))
    length(varnames) == n || throw(ArgumentError(
        "varnames has length $(length(varnames)); expected $n (= number of states)"))
    obs_mat = nothing
    if data !== nothing
        if size(data) == (Tn, n)
            obs_mat = data
        elseif size(data) == (n, Tn)
            obs_mat = permutedims(data)
        else
            throw(ArgumentError("data has size $(size(data)); expected ($Tn, $n) or ($n, $Tn)"))
        end
    end
    idx, note = _dsge_extra_indices(var, varnames, max_panels)
    panels = _PanelSpec[]
    for j in idx
        id = _next_plot_id("ks_state")
        obs = obs_mat === nothing ? nothing : @view obs_mat[:, j]
        data_json = _smoothed_state_json(r.smoothed_states, r.smoothed_covariances, j; obs=obs)
        snames = obs === nothing ? ["Smoothed"] : ["Smoothed", "Observed"]
        skeys = obs === nothing ? ["sm"] : ["sm", "obs"]
        sdash = obs === nothing ? [""] : ["", "6,3"]
        series_json = _series_json(snames, _colors_for(snames); keys=skeys, dash=sdash)
        band_color = _colors_for(snames)[1]
        bands = "[{\"lo_key\":\"lo\",\"hi_key\":\"hi\",\"color\":$(_json(band_color))," *
                "\"alpha\":$(_json(_PLOT_CI_ALPHA)),\"name\":\"±1.96 s.e.\"}]"
        js = _render_line_js(id, data_json, series_json; bands_json=bands, integer_x=true,
                             xlabel="Period", ylabel=varnames[j])
        push!(panels, _PanelSpec(id, varnames[j], js))
    end
    ftitle = isempty(title) ? "Kalman-Smoothed States (±95% band)" : title
    p = _make_plot(panels; title=ftitle, ncols=ncols, note=note)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# DSGEEstimation (GMM)
# =============================================================================

"""
    plot_result(est::DSGEEstimation; title="", save_path=nothing)

Parameter coefficient plot for a GMM-estimated DSGE: `θ ± 1.96·stderror(est)` as a
horizontal dot-and-whisker over `param_names`, with the Hansen J-test annotated in
the panel title (`J = … (p = …)`, rounded via `_fmt`/`_format_pvalue`, plotrule C9).
This is a **parameter** plot — the struct stores no empirical/model moments, so no
moment-fit exhibit is drawn (plotrule C6; moment fit lives in the GMM/SMM plot).
"""
function plot_result(est::DSGEEstimation{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    id = _next_plot_id("dsge_est")
    se = StatsAPI.stderror(est)
    data_json = _dsge_coef_json(est.param_names, est.theta, se)
    js = _render_coef_plot_js(id, data_json; ref_value=0, xlabel="Estimate", ylabel="")
    ptitle = "Estimated Parameters ($(est.method)); J = $(_fmt(est.J_stat; digits=3)) (p = $(_format_pvalue(est.J_pvalue)))"
    panel = _PanelSpec(id, ptitle, js)
    ftitle = isempty(title) ? "DSGE-GMM Parameter Estimates" : title
    p = _make_plot([panel]; title=ftitle, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end
