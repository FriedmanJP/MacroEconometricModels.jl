# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

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
- `DSGEEstimation` — a parameter dot-and-whisker (`θ ± z·stderror`, `conf_level`) with a J-test
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

Parameter coefficient plot for a GMM-estimated DSGE: `θ ± z·stderror(est)` at
confidence level `conf_level` (default 0.95) as a
horizontal dot-and-whisker over `param_names`, with the Hansen J-test annotated in
the panel title (`J = … (p = …)`, rounded via `_fmt`/`_format_pvalue`, plotrule C9).
This is a **parameter** plot — the struct stores no empirical/model moments, so no
moment-fit exhibit is drawn (plotrule C6; moment fit lives in the GMM/SMM plot).
"""
function plot_result(est::DSGEEstimation{T};
                     title::String="", save_path::Union{String,Nothing}=nothing,
                     conf_level::Real=0.95) where {T}
    id = _next_plot_id("dsge_est")
    z = _conf_z(conf_level)
    se = StatsAPI.stderror(est)
    data_json = _dsge_coef_json(est.param_names, est.theta, se; z=z)
    js = _render_coef_plot_js(id, data_json; ref_value=0, xlabel="Estimate", ylabel="")
    ptitle = "Estimated Parameters ($(est.method)); J = $(_fmt(est.J_stat; digits=3)) (p = $(_format_pvalue(est.J_pvalue)))"
    panel = _PanelSpec(id, ptitle, js)
    ftitle = isempty(title) ? "DSGE-GMM Parameter Estimates" : title
    p = _make_plot([panel]; title=ftitle, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end


# =============================================================================
# DeterminacyMap ([T268])
# =============================================================================

"""
    plot_result(m::DeterminacyMap; title="", save_path=nothing) -> PlotOutput

Determinacy map: a heatmap of the verdict over the swept parameter grid.

Cells carry the ordered verdict codes ([`DETERMINACY_CODES`](@ref)) — `1` determinate, `0`
indeterminate, `-1` no stable solution — on a single-hue sequential ramp with a fixed domain
`[-1, 1]`, so the colours mean the same thing across two maps of the same model. The domain is
pinned rather than data-driven for exactly that reason: a sweep that happens to contain only
determinate cells must not repaint them as if they were the worst outcome present.

Grid points where the model could not be solved serialize as `null`, which the heatmap renderer
draws in neutral grey and labels "Missing" (plotrule Heatmaps) — a failed solve is genuinely
absent information, not a fourth region.

A one-parameter sweep is drawn as a single row.
"""
function plot_result(m::DeterminacyMap{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    n1, n2 = size(m.verdict)
    id = _next_plot_id("dsge_determinacy")

    # Rows = first parameter (descending down the axis, so low values sit at the bottom as on a
    # conventional plot); columns = second parameter, or a single unlabeled row when 1-D.
    row_vals = m.axes[1]
    row_labels_asc = String[_fmt(v; digits=3) for v in row_vals]
    col_labels = _dm_is_1d(m) ? [""] : String[_fmt(v; digits=3) for v in m.axes[2]]

    rows = Vector{Pair{String,String}}[]
    for i in 1:n1, j in 1:n2
        v = m.verdict[i, j]
        vj = v == DETERMINACY_CODES.failed ? "null" : _json(float(v))
        push!(rows, Pair{String,String}["x" => _json(col_labels[j]),
                                        "y" => _json(row_labels_asc[i]),
                                        "v" => vj])
    end
    data_json = _json_array_of_objects(rows)

    xlab = _dm_is_1d(m) ? "" : string(m.params[2])
    js = _render_heatmap_js(id, data_json, _json(reverse(row_labels_asc)), _json(col_labels);
                            xlabel=xlab, ylabel=string(m.params[1]),
                            tip_label=isempty(xlab) ? string(m.params[1]) : xlab,
                            scale=:sequential, color_domain=[-1.0, 1.0])
    ptitle = "1 = determinate, 0 = indeterminate, -1 = no stable solution"
    panel = _PanelSpec(id, ptitle, js)

    ftitle = isempty(title) ? "Determinacy Region (" * join(string.(m.params), " × ") * ")" :
             title
    p = _make_plot([panel]; title=ftitle, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# ABCD State-Space & DSGE-VAR Plotting Helpers (#915)
# (No docstrings on internal helpers — docs/src/api/visualization.md uses @autodocs)
# =============================================================================

# Format scalar values for panel/figure titles cleanly (plotrule C9).
_abcd_fmt_num(x::Real; digits::Int=3) =
    isnan(x) ? "n/a" : (isinf(x) ? (x > 0 ? "∞" : "−∞") : _fmt(float(x); digits=digits))

# Resolve scalar or vector Int/String/Symbol selector against `names` via `_resolve_var` (C3).
function _abcd_resolve_sel(sel, names::Vector{String})
    if sel === nothing
        return collect(1:length(names))
    elseif sel isa AbstractVector
        isempty(sel) && throw(ArgumentError("Selection cannot be empty"))
        return Int[_resolve_var(v isa Symbol ? string(v) : (v isa Integer ? Int(v) : v), names) for v in sel]
    else
        v = sel isa Symbol ? string(sel) : (sel isa Integer ? Int(sel) : sel)
        return Int[_resolve_var(v, names)]
    end
end

# Complex-plane unit-circle scatter panel shared by ABCDSystem, InvertibilityResult,
# and InnovationsRepresentation.
function _abcd_unit_circle_scatter(prefix::String, ptitle::String,
                                   re_vals::AbstractVector{<:Real},
                                   im_vals::AbstractVector{<:Real};
                                   inside_label::String="Inside unit circle (|λ| < 1)",
                                   outside_label::String="On/outside unit circle (|λ| ≥ 1)",
                                   tol::Real=1e-8)
    rows = Vector{Pair{String,String}}[]
    has_in = false
    has_out = false
    for i in eachindex(re_vals)
        rv = Float64(re_vals[i])
        iv = Float64(im_vals[i])
        m = hypot(rv, iv)
        is_in = isfinite(m) && (m < 1.0 - Float64(tol))
        grp = is_in ? inside_label : outside_label
        is_in ? (has_in = true) : (has_out = true)
        push!(rows, ["x" => _json(rv), "y" => _json(iv), "group" => _json(grp)])
    end
    data_json = _json_array_of_objects(rows)
    gparts = String[]
    if has_in || !has_out
        push!(gparts, "{\"name\":$(_json(inside_label)),\"color\":$(_json(_PLOT_SERIES[1]))}")
    end
    if has_out
        push!(gparts, "{\"name\":$(_json(outside_label)),\"color\":$(_json(_PLOT_ALERT))}")
    end
    groups_json = "[" * join(gparts, ",") * "]"
    ref_shapes = "[{\"type\":\"circle\",\"cx\":0,\"cy\":0,\"r\":1,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    ref_lines = "[{\"value\":0,\"axis\":\"x\",\"color\":\"#bbb\",\"dash\":\"2,2\"}," *
                "{\"value\":0,\"axis\":\"y\",\"color\":\"#bbb\",\"dash\":\"2,2\"}]"
    id = _next_plot_id(prefix)
    js = _render_scatter_js(id, data_json, groups_json;
                            ref_lines_json=ref_lines, ref_shapes_json=ref_shapes,
                            xlabel="Real", ylabel="Imaginary")
    _PanelSpec(id, ptitle, js)
end

# Horizontal singular-value bar panel for matrix M (returns _PanelSpec).
function _abcd_sval_panel(prefix::String, ptitle::String, M::AbstractMatrix)
    id = _next_plot_id(prefix)
    if isempty(M)
        data_json = "[]"
    else
        sv = if all(isfinite, M)
            Float64.(svdvals(Matrix{Float64}(M)))
        else
            fill(NaN, min(size(M)...))
        end
        labels = String["σ$i" for i in eachindex(sv)]
        data_json = _keyed_rows_json(labels, ["sv" => sv])
    end
    s_json = _series_json(["Singular value"], [_PLOT_SERIES[1]]; keys=["sv"])
    js = _render_bar_js(id, data_json, s_json; mode="grouped", orientation="h",
                        xlabel="Singular value", ylabel="")
    _PanelSpec(id, ptitle, js)
end

# Regressor-by-equation labels for DSGEVARPosterior B_draws in fixed-theta mode.
function _dsgevar_coef_labels(post::DSGEVARPosterior)
    vn = post.varnames
    k = size(post.B_draws, 2)
    has_int = (k == 1 + post.n * post.p)
    coef_names = String[]
    has_int && push!(coef_names, _INTERCEPT_LABEL)
    for l in 1:post.p, v in 1:post.n
        push!(coef_names, "$(vn[v]).L$l")
    end
    while length(coef_names) < k
        push!(coef_names, "x$(length(coef_names)+1)")
    end
    labels = String[]
    for eq in 1:post.n, j in 1:k
        push!(labels, "$(vn[eq]) ← $(coef_names[j])")
    end
    labels
end

# Build JSON rows [{x, q1..qk, med[, dog]}, ...] for DSGEVARPosterior IRF overlay.
function _dsgevar_irf_panel_json(xs::AbstractVector, qmat::AbstractMatrix,
                                 levels::AbstractVector, central::AbstractVector,
                                 dogmatic::Union{AbstractVector,Nothing})
    Tn, _ = size(qmat)
    perm = sortperm(Float64[Float64(l) for l in levels])
    rows = Vector{Pair{String,String}}[]
    for t in 1:Tn
        row = Pair{String,String}["x" => _json(xs[t])]
        for (ci, j) in enumerate(perm)
            push!(row, "q$ci" => _json(qmat[t, j]))
        end
        push!(row, "med" => _json(central[t]))
        dogmatic !== nothing && push!(row, "dog" => _json(dogmatic[t]))
        push!(rows, row)
    end
    _json_array_of_objects(rows)
end

# Shared marginal log-likelihood curve panel over lambda grid (DSGEVARLambdaProfile
# and fixed-theta DSGEVARPosterior view=:lambda).
function _dsgevar_lambda_panel(prefix::String, lambdas::AbstractVector,
                               logmls::AbstractVector, lambda_hat::Real,
                               logml_hat::Real, logml_inf::Real, lambda_min::Real)
    lam_hat_str = isinf(lambda_hat) ? "∞" : _abcd_fmt_num(lambda_hat)
    ml_hat_str = _abcd_fmt_num(logml_hat)
    ml_inf_str = _abcd_fmt_num(logml_inf)

    if isempty(lambdas)
        lmin = isfinite(lambda_min) ? Float64(lambda_min) : 0.1
        xs = [lmin, lmin + 1.0]
        y_inf = isfinite(logml_inf) ? Float64(logml_inf) : NaN
        data_json = _keyed_rows_json(xs, ["inf_line" => [y_inf, y_inf]])
        s_json = _series_json(["Log ML (λ = ∞)"], [_PLOT_SERIES[2]];
                              keys=["inf_line"], dash=["6,3"])
        refs = "[]"
        note = "No finite λ grid points evaluated (dogmatic limit λ = ∞ only)."
    else
        xs = Float64[Float64(v) for v in lambdas]
        lmls = Float64[Float64(v) for v in logmls]
        if isfinite(logml_inf)
            cols = Pair{String,Vector{Float64}}[
                "lml" => lmls,
                "inf_line" => fill(Float64(logml_inf), length(xs)),
            ]
            s_json = _series_json(["Log ML(λ)", "Log ML(λ = ∞)"],
                                  [_PLOT_SERIES[1], _PLOT_SERIES[2]];
                                  keys=["lml", "inf_line"], dash=["", "6,3"])
        else
            cols = Pair{String,Vector{Float64}}["lml" => lmls]
            s_json = _series_json(["Log ML(λ)"], [_PLOT_SERIES[1]]; keys=["lml"])
        end
        data_json = _keyed_rows_json(xs, cols)
        fin_xs = [v for v in xs if isfinite(v)]
        if !isempty(fin_xs) && isfinite(lambda_hat) &&
           minimum(fin_xs) <= Float64(lambda_hat) <= maximum(fin_xs)
            refs = "[{\"value\":$(_json(Float64(lambda_hat))),\"axis\":\"x\"," *
                   "\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"4,3\"}]"
        else
            refs = "[]"
        end
        note = ""
    end

    id = _next_plot_id(prefix)
    js = _render_line_js(id, data_json, s_json; ref_lines_json=refs,
                         xlabel="DSGE Prior Weight λ", ylabel="Log Marginal Likelihood")
    ptitle = "Log Marginal Likelihood Profile (λ̂ = $(lam_hat_str), " *
             "log ML(λ̂) = $(ml_hat_str), log ML(∞) = $(ml_inf_str))"
    (_PanelSpec(id, ptitle, js), note)
end

# =============================================================================
# ABCDSystem (#915)
# =============================================================================

"""
    plot_result(sys::ABCDSystem; view=:eigenvalues, title="", ncols=0, save_path=nothing)

Plot canonical state-space diagnostics for an [`ABCDSystem`](@ref).

# Views
- `:eigenvalues` (default) — complex-plane scatter of state-transition eigenvalues
  `eigvals(sys.A)` relative to the unit circle (inside in blue, on/outside in alert red),
  with state-space dimensions `(n, k, m)` annotated in the title.
- `:singular_values` — horizontal bar chart(s) of singular values of the impact matrix
  `D` (`k × m`) and, when `n > 0`, the state-transition matrix `A` (`n × n`).

Unknown `view` throws an `ArgumentError`.
"""
function plot_result(sys::ABCDSystem{T};
                     view::Symbol=:eigenvalues,
                     title::String="", ncols::Int=0,
                     save_path::Union{String,Nothing}=nothing) where {T}
    n = nstates(sys)
    k = nobservables(sys)
    m = nshocks(sys)
    if view === :eigenvalues
        if n == 0
            re_vals = Float64[]
            im_vals = Float64[]
            mx = 0.0
            stat_lbl = "STATIC (n=0)"
        elseif all(isfinite, sys.A)
            ev = Complex{Float64}.(eigvals(Matrix{Float64}(sys.A)))
            re_vals = real.(ev)
            im_vals = imag.(ev)
            mx = maximum(abs, ev)
            stat_lbl = mx < 1.0 - 1e-8 ? "STABLE" : "UNSTABLE"
        else
            re_vals = Float64[NaN]
            im_vals = Float64[NaN]
            mx = NaN
            stat_lbl = "NON-FINITE"
        end
        ptitle = n == 0 ?
            "State Transition Eigenvalues (STATIC — n=0)" :
            "State Transition Eigenvalues ($(stat_lbl) — max |λ| = $(_abcd_fmt_num(mx)))"
        panel = _abcd_unit_circle_scatter("abcd_eig", ptitle, re_vals, im_vals)
        panels = _PanelSpec[panel]
        ftitle = isempty(title) ? "ABCD State-Space System (n=$(n), k=$(k), m=$(m))" : title
        p = _make_plot(panels; title=ftitle, ncols=ncols <= 0 ? 1 : ncols)
    elseif view === :singular_values
        panels = _PanelSpec[
            _abcd_sval_panel("abcd_svd_d", "Impact Matrix D Singular Values ($(k) × $(m))", sys.D)
        ]
        if n > 0
            push!(panels, _abcd_sval_panel("abcd_svd_a", "State Transition A Singular Values ($(n) × $(n))", sys.A))
        end
        ftitle = isempty(title) ? "ABCD Singular Values & Dimensions (n=$(n), k=$(k), m=$(m))" : title
        p = _make_plot(panels; title=ftitle, ncols=ncols <= 0 ? length(panels) : ncols)
    else
        throw(ArgumentError("Unknown view :$view. Valid views: :eigenvalues, :singular_values"))
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# InvertibilityResult (#915)
# =============================================================================

"""
    plot_result(r::InvertibilityResult; view=:eigenvalues, title="", ncols=0, save_path=nothing)

Plot the FVRSW (2007) / Komunjer–Ng (2011) invertibility test results on the complex plane.

# Views
- `:eigenvalues` (default) — complex-plane scatter of finite transmission zeros /
  eigenvalues of `M = A - B D⁻¹ C` (`r.zero_re`, `r.zero_im`) relative to the unit circle
  (`|z| < 1` in blue vs `|z| ≥ 1` in alert red), with infinite zeros (`n_infinite_zeros`)
  and `rank(D)` highlighted in the subtitle and figure note.

Unknown `view` throws an `ArgumentError`.
"""
function plot_result(r::InvertibilityResult{T};
                     view::Symbol=:eigenvalues,
                     title::String="", ncols::Int=0,
                     save_path::Union{String,Nothing}=nothing) where {T}
    view === :eigenvalues ||
        throw(ArgumentError("Unknown view :$view. Valid views: :eigenvalues"))
    verdict = r.invertible ? "INVERTIBLE" : "NON-INVERTIBLE"
    inf_str = r.n_infinite_zeros > 0 ? ", $(r.n_infinite_zeros) infinite zero(s)" : ""
    ptitle = "Transmission Zeros / λ(M) — $(verdict) (max |z| = $(_abcd_fmt_num(r.max_modulus))$(inf_str))"
    panel = _abcd_unit_circle_scatter(
        "abcd_inv", ptitle, r.zero_re, r.zero_im;
        inside_label="|z| < 1 (invertible)",
        outside_label="|z| ≥ 1 (non-invertible)",
        tol=r.tol,
    )
    note_parts = String[]
    if r.n_infinite_zeros > 0
        push!(note_parts, "$(r.n_infinite_zeros) infinite transmission zero(s) (delayed impact; rank(D) = $(r.rank_D)/$(min(r.n_obs, r.n_shocks))).")
    elseif r.status === :non_invertible_underdetermined
        push!(note_parts, "Underdetermined system (k=$(r.n_obs) < m=$(r.n_shocks)) — no finite transmission zeros.")
    elseif isempty(r.zero_re)
        push!(note_parts, "No finite transmission zeros.")
    end
    if !isempty(r.obs_names)
        obs_shown = length(r.obs_names) > 6 ? vcat(r.obs_names[1:6], ["…"]) : r.obs_names
        push!(note_parts, "Observables: " * join(obs_shown, ", "))
    end
    note = join(note_parts, " ")
    ftitle = isempty(title) ?
        "FVRSW Invertibility Check ($(r.status), n=$(r.n_states), k=$(r.n_obs), m=$(r.n_shocks))" : title
    p = _make_plot([panel]; title=ftitle, ncols=ncols <= 0 ? 1 : ncols, note=note)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# InnovationsRepresentation (#915)
# =============================================================================

"""
    plot_result(innov::InnovationsRepresentation; view=:eigenvalues, title="", ncols=0, save_path=nothing)

Plot diagnostics for a Hansen–Sargent / FVRSW (2007) Wold [`InnovationsRepresentation`](@ref).

# Views
- `:eigenvalues` (default) — complex-plane scatter of closed-loop innovations eigenvalues
  `λ(A - K C)` against the unit circle (guaranteed inside/on the unit circle by DARE stability).
- `:gain` — diverging heatmap of the `n × k` steady-state Kalman gain matrix `K` across states
  (rows) and observables (columns), with color-scale legend.

Unknown `view` throws an `ArgumentError`.
"""
function plot_result(innov::InnovationsRepresentation{T};
                     view::Symbol=:eigenvalues,
                     title::String="", ncols::Int=0,
                     save_path::Union{String,Nothing}=nothing) where {T}
    n = size(innov.A, 1)
    k = size(innov.C, 1)
    inv_str = innov.invertible ? "invertible" : "non-invertible"
    if view === :eigenvalues
        if n == 0
            re_vals = Float64[]
            im_vals = Float64[]
            mx = 0.0
        elseif all(isfinite, innov.A) && all(isfinite, innov.K) && all(isfinite, innov.C)
            M_cl = Matrix{Float64}(innov.A) - Matrix{Float64}(innov.K) * Matrix{Float64}(innov.C)
            ev = Complex{Float64}.(eigvals(M_cl))
            re_vals = real.(ev)
            im_vals = imag.(ev)
            mx = maximum(abs, ev)
        else
            re_vals = Float64[NaN]
            im_vals = Float64[NaN]
            mx = NaN
        end
        ptitle = n == 0 ?
            "Closed-Loop Eigenvalues λ(A − KC) (n=0 static, $(inv_str))" :
            "Closed-Loop Eigenvalues λ(A − KC) (max |λ| = $(_abcd_fmt_num(mx)), $(inv_str))"
        panel = _abcd_unit_circle_scatter(
            "innov_eig", ptitle, re_vals, im_vals;
            inside_label="Closed-loop |λ| < 1",
            outside_label="On/outside |λ| ≥ 1",
        )
        ftitle = isempty(title) ? "Wold Innovations Representation (n=$(n), k=$(k))" : title
        p = _make_plot([panel]; title=ftitle, ncols=ncols <= 0 ? 1 : ncols)
    elseif view === :gain
        (n > 0 && k > 0) || throw(ArgumentError(
            "Cannot plot Kalman gain K of size ($n, $k); use view=:eigenvalues"))
        row_labels = copy(innov.state_names)
        col_labels = copy(innov.obs_names)
        data_json = _matrix_heat_json(Matrix{Float64}(innov.K), row_labels, col_labels)
        id = _next_plot_id("innov_k")
        js = _render_heatmap_js(id, data_json, _json(row_labels), _json(col_labels);
                                xlabel="Observable", ylabel="State", tip_label="",
                                scale=:diverging, midpoint=0)
        fin_loss = [Float64(v) for v in innov.information_loss if isfinite(v)]
        max_loss = isempty(fin_loss) ? NaN : maximum(fin_loss)
        ptitle = "Steady-State Kalman Gain K ($(n) × $(k); max info loss = $(_abcd_fmt_num(max_loss)))"
        ftitle = isempty(title) ? "Wold Innovations Kalman Gain ($(inv_str))" : title
        p = _make_plot([_PanelSpec(id, ptitle, js)]; title=ftitle, ncols=ncols <= 0 ? 1 : ncols)
    else
        throw(ArgumentError("Unknown view :$view. Valid views: :eigenvalues, :gain"))
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# ShockRecoverabilityResult (#915)
# =============================================================================

"""
    plot_result(r::ShockRecoverabilityResult; view=:leads, shock=nothing, max_shocks=8,
                ncols=0, title="", save_path=nothing)

Plot Sims (2012) structural shock recoverability ``R_j^2(\\ell) \\in [0, 1]``.

# Views
- `:leads` (default) — lead curve of `r.r2_matrix` across leads `ℓ = 0:r.max_lead`
  against `0` and `1` reference bounds, one line per shock (capped at `max_shocks`
  when `shock === nothing`; falls back to contemporaneous bars when `max_lead == 0`).
- `:contemp` — horizontal bar chart comparing contemporaneous `R²(0)` (and terminal
  `R²(max_lead)` when `max_lead > 0`) across shocks.

`shock` selects one or more structural shocks by integer index or name (`r.shock_names`).
Unknown `view` throws an `ArgumentError`.
"""
function plot_result(r::ShockRecoverabilityResult{T};
                     view::Symbol=:leads,
                     shock::Union{Int,String,AbstractVector,Nothing}=nothing,
                     max_shocks::Int=8,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    view in (:leads, :contemp) ||
        throw(ArgumentError("Unknown view :$view. Valid views: :leads, :contemp"))
    m = length(r.shock_names)
    m > 0 || throw(ArgumentError("ShockRecoverabilityResult has 0 shocks"))

    if shock === nothing
        shown = min(m, max(1, max_shocks))
        idx = collect(1:shown)
        note = _cap_note("shocks", shown, m, "max_shocks")
    else
        idx = _abcd_resolve_sel(shock, r.shock_names)
        note = ""
    end

    sel_names = r.shock_names[idx]
    all_colors = _colors_for(r.shock_names)
    sel_colors = all_colors[idx]
    inv_str = r.is_invertible ? "invertible" : "non-invertible"

    if view === :leads && r.max_lead >= 1
        xs = collect(0:r.max_lead)
        cols = Pair{String,Vector{Float64}}[
            "s$k" => Float64[Float64(r.r2_matrix[h, si]) for h in 1:(r.max_lead + 1)]
            for (k, si) in enumerate(idx)
        ]
        data_json = _keyed_rows_json(xs, cols)
        dashes = String[mod1(k, 2) == 1 ? "" : "6,3" for k in 1:length(idx)]
        s_json = _series_json(sel_names, sel_colors; keys=["s$k" for k in 1:length(idx)], dash=dashes)
        refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}," *
               "{\"value\":1,\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"5,4\"}]"
        id = _next_plot_id("recov_leads")
        js = _render_line_js(id, data_json, s_json; ref_lines_json=refs, integer_x=true,
                             xlabel="Lead ℓ", ylabel="Recoverability R²(ℓ)")
        fin0 = Float64[Float64(v) for v in r.r2_contemp[idx] if isfinite(v)]
        min_r0 = isempty(fin0) ? NaN : minimum(fin0)
        ptitle = _cap_title(
            "Recoverability Profile R²(ℓ) — min R²(0) = $(_abcd_fmt_num(min_r0)) ($(inv_str))",
            length(idx), m)
        panel = _PanelSpec(id, ptitle, js)
    else
        r0_vals = Float64[Float64(r.r2_contemp[si]) for si in idx]
        if r.max_lead >= 1
            rmax_vals = Float64[Float64(r.r2_matrix[end, si]) for si in idx]
            data_json = _keyed_rows_json(sel_names, ["r0" => r0_vals, "rmax" => rmax_vals])
            s_json = _series_json(["R²(0)", "R²(ℓ=$(r.max_lead))"],
                                  [_PLOT_SERIES[1], _PLOT_SERIES[2]]; keys=["r0", "rmax"])
            ptitle = _cap_title("Contemporaneous vs Lead-$(r.max_lead) R² ($(inv_str))", length(idx), m)
        else
            data_json = _keyed_rows_json(sel_names, ["r0" => r0_vals])
            s_json = _series_json(["R²(0)"], [_PLOT_SERIES[1]]; keys=["r0"])
            ptitle = _cap_title("Contemporaneous Recoverability R²(0) ($(inv_str))", length(idx), m)
        end
        id = _next_plot_id("recov_bar")
        js = _render_bar_js(id, data_json, s_json; mode="grouped", orientation="h",
                            xlabel="Recoverability R²", ylabel="Shock")
        panel = _PanelSpec(id, ptitle, js)
    end

    ftitle = isempty(title) ? "Sims (2012) Shock Recoverability ($(inv_str))" : title
    p = _make_plot([panel]; title=ftitle, ncols=ncols <= 0 ? 1 : ncols, note=note)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# ObservableSearchResult (#915)
# =============================================================================

"""
    plot_result(res::ObservableSearchResult; metric=:min_R2, max_candidates=15,
                ncols=0, title="", save_path=nothing)

Plot the ranked candidate observable subsets from [`search_invertible_observables`](@ref) as a
horizontal bar chart color-coded by invertibility (`Invertible` in blue vs `Non-invertible` in
alert red, paired with `[✓]`/`[✗]` tags on the subset labels).

# Keywords
- `metric::Symbol=:min_R2` — bar metric: `:min_R2` (minimum contemporaneous `R²(0)` across target
  shocks), `:mean_R2`, or `:rho` (spectral radius / max transmission zero modulus).
- `max_candidates::Int=15` — maximum number of top-ranked subsets to display (capped visibly per
  plotrule C7).

Unknown `metric` throws an `ArgumentError`.
"""
function plot_result(res::ObservableSearchResult{T};
                     metric::Symbol=:min_R2,
                     max_candidates::Int=15,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    metric in (:min_R2, :mean_R2, :rho) ||
        throw(ArgumentError("Unknown metric :$metric. Valid metrics: :min_R2, :mean_R2, :rho"))
    max_candidates >= 1 || throw(ArgumentError("max_candidates must be >= 1; got $max_candidates"))

    n_eval = length(res.observables)
    shown = min(n_eval, max_candidates)
    vals_vec = metric === :min_R2 ? res.min_R2 : (metric === :mean_R2 ? res.mean_R2 : res.rho)
    mlabel = metric === :min_R2 ? "Min R²(0)" : (metric === :mean_R2 ? "Mean R²(0)" : "Max |λ(M)| (ρ)")

    rows = Vector{Pair{String,String}}[]
    for i in 1:shown
        tag = res.is_invertible[i] ? "[✓]" : "[✗]"
        label = "$(tag) #$(i) " * join(string.(res.observables[i]), ", ")
        v = Float64(vals_vec[i])
        if res.is_invertible[i]
            push!(rows, ["x" => _json(label), "inv" => _json(v), "noninv" => "0"])
        else
            push!(rows, ["x" => _json(label), "inv" => "0", "noninv" => _json(v)])
        end
    end
    data_json = _json_array_of_objects(rows)
    s_json = _series_json(["Invertible", "Non-invertible"], [_PLOT_SERIES[1], _PLOT_ALERT];
                          keys=["inv", "noninv"])
    id = _next_plot_id("obs_search")
    js = _render_bar_js(id, data_json, s_json; mode="stacked", orientation="h",
                        xlabel=mlabel, ylabel="Candidate Set")

    n_inv = count(res.is_invertible)
    ptitle = _cap_title("Top Observable Sets by $(mlabel) ($(n_inv)/$(n_eval) invertible)", shown, n_eval)
    note = _cap_note("candidate subsets", shown, n_eval, "max_candidates")
    mode_lbl = res.used_greedy ? "greedy" : "exhaustive"
    ftitle = isempty(title) ? "Observable-Set Invertibility Search (k=$(res.k), $(mode_lbl))" : title
    p = _make_plot([_PanelSpec(id, ptitle, js)]; title=ftitle, ncols=ncols <= 0 ? 1 : ncols, note=note)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# DSGEVARIRFAudit (#915)
# =============================================================================

"""
    plot_result(audit::DSGEVARIRFAudit; view=:irf_comparison, var=nothing, shock=nothing,
                max_panels=12, ncols=0, title="", save_path=nothing)

Plot Ravenna (2007) three-way DSGE vs. population `SVAR(p)` IRF distortion decomposition.

# Views
- `:irf_comparison` (default) — overlays true DSGE (`irf_dsge`), Procrustes best-rotated Wold
  (`irf_wold_best`), identified Wold (`irf_wold_id` when `identification != :best_rotation`),
  and population `SVAR(p)` (`irf_svar`) impulse responses over horizons `0:horizon-1`. Supports
  `var` and `shock` selection by `Int` or `String` (bounds-checked) and caps un-filtered
  grids at `max_panels`.
- `:wedges` — horizon-by-horizon RMSE curves decomposing total distortion (`rmse_by_horizon`)
  into non-invertibility (`noninv_by_horizon`), lag-truncation (`trunc_by_horizon`), and
  identification (`id_by_horizon`) wedges. `var`/`shock` do not apply to `:wedges`.

Unknown `view` throws an `ArgumentError`.
"""
function plot_result(audit::DSGEVARIRFAudit{T};
                     view::Symbol=:irf_comparison,
                     var::Union{Int,String,AbstractVector,Nothing}=nothing,
                     shock::Union{Int,String,AbstractVector,Nothing}=nothing,
                     max_panels::Int=12,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    inv_str = audit.is_invertible ? "invertible" : "non-invertible"
    xs = collect(0:(audit.horizon - 1))
    refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"

    if view === :irf_comparison
        vars_to_plot = _abcd_resolve_sel(var, audit.obs_names)
        shocks_to_plot = _abcd_resolve_sel(shock, audit.shock_names)
        pairs = [(vi, si) for vi in vars_to_plot for si in shocks_to_plot]
        n_total = length(pairs)
        shown = (var === nothing && shock === nothing) ? min(n_total, max(1, max_panels)) : n_total
        note = (var === nothing && shock === nothing) ?
            _cap_note("response-shock panels", shown, n_total, "max_panels") : ""
        include_id = (audit.identification !== :best_rotation)

        panels = _PanelSpec[]
        for (vi, si) in pairs[1:shown]
            id = _next_plot_id("audit_irf")
            cols = Pair{String,Vector{Float64}}[
                "dsge" => Float64.(audit.irf_dsge[:, vi, si]),
                "wold_best" => Float64.(audit.irf_wold_best[:, vi, si]),
            ]
            snames = String["DSGE (True)", "Wold (best rot.)"]
            scols = String[_PLOT_SERIES[1], _PLOT_SERIES[2]]
            skeys = String["dsge", "wold_best"]
            sdash = String["", "6,3"]
            if include_id
                push!(cols, "wold_id" => Float64.(audit.irf_wold_id[:, vi, si]))
                push!(snames, "Wold ($(audit.identification))")
                push!(scols, _PLOT_SERIES[4])
                push!(skeys, "wold_id")
                push!(sdash, "2,2")
            end
            push!(cols, "svar" => Float64.(audit.irf_svar[:, vi, si]))
            push!(snames, "SVAR($(audit.p))")
            push!(scols, _PLOT_SERIES[3])
            push!(skeys, "svar")
            push!(sdash, "8,3,2,3")

            data_json = _keyed_rows_json(xs, cols)
            s_json = _series_json(snames, scols; keys=skeys, dash=sdash)
            js = _render_line_js(id, data_json, s_json; ref_lines_json=refs, integer_x=true,
                                 xlabel="Horizon", ylabel="Response")
            push!(panels, _PanelSpec(id, "$(audit.obs_names[vi]) ← $(audit.shock_names[si])", js))
        end

        auto_ncols = length(shocks_to_plot) == 1 ?
            min(length(vars_to_plot), 3) :
            (shown == n_total ? length(shocks_to_plot) : min(length(shocks_to_plot), 3))
        ftitle = isempty(title) ?
            "DSGE vs SVAR($(audit.p)) IRF Comparison ($(audit.identification), $(inv_str))" : title
        p = _make_plot(panels; title=ftitle, ncols=ncols <= 0 ? auto_ncols : ncols, note=note)
    elseif view === :wedges
        (var === nothing && shock === nothing) ||
            throw(ArgumentError("var/shock apply only to view=:irf_comparison"))
        cols = Pair{String,Vector{Float64}}[
            "tot" => Float64.(audit.rmse_by_horizon),
            "ni" => Float64.(audit.noninv_by_horizon),
            "tr" => Float64.(audit.trunc_by_horizon),
            "id" => Float64.(audit.id_by_horizon),
        ]
        data_json = _keyed_rows_json(xs, cols)
        s_json = _series_json(
            ["Total RMSE", "Non-invertibility", "Lag truncation", "Identification"],
            [_PLOT_SERIES[1], _PLOT_SERIES[2], _PLOT_SERIES[3], _PLOT_SERIES[4]];
            keys=["tot", "ni", "tr", "id"],
            dash=["", "6,3", "3,3", "8,3,2,3"],
        )
        id = _next_plot_id("audit_wedges")
        js = _render_line_js(id, data_json, s_json; ref_lines_json=refs, integer_x=true,
                             xlabel="Horizon", ylabel="RMSE")
        ptitle = "RMSE Wedges — total=$(_abcd_fmt_num(audit.total_rmse)), " *
                 "non-inv=$(_abcd_fmt_num(audit.noninvertibility_wedge)), " *
                 "trunc=$(_abcd_fmt_num(audit.truncation_wedge)), " *
                 "id=$(_abcd_fmt_num(audit.identification_wedge))"
        ftitle = isempty(title) ?
            "Ravenna (2007) IRF Distortion Wedges (p=$(audit.p), $(audit.identification))" : title
        p = _make_plot([_PanelSpec(id, ptitle, js)]; title=ftitle, ncols=ncols <= 0 ? 1 : ncols)
    else
        throw(ArgumentError("Unknown view :$view. Valid views: :irf_comparison, :wedges"))
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# DSGEVARPrior (#915)
# =============================================================================

"""
    plot_result(prior::DSGEVARPrior; view=:phi, title="", ncols=0, save_path=nothing)

Plot Del Negro & Schorfheide (2004) [`DSGEVARPrior`](@ref) population matrices with
diverging color scales and annotated metadata (`λ`, `λ_min`, `p`, `T_obs`).

# Views
- `:phi` (default) — diverging heatmap of the `k × n` population VAR coefficient projection `Φ*`.
- `:sigma` — diverging heatmap of the `n × n` population innovation covariance `Σ_u*`.
- `:both` — two-panel comparison showing both `Φ*` and `Σ_u*` side by side.

Unknown `view` throws an `ArgumentError`.
"""
function plot_result(prior::DSGEVARPrior{T};
                     view::Symbol=:phi,
                     title::String="", ncols::Int=0,
                     save_path::Union{String,Nothing}=nothing) where {T}
    view in (:phi, :sigma, :both) ||
        throw(ArgumentError("Unknown view :$view. Valid views: :phi, :sigma, :both"))
    n = length(prior.obs_names)
    k = size(prior.Phi_star, 1)
    has_int = (k == 1 + n * prior.p)
    lam_str = isinf(prior.lambda) ? "∞" : _abcd_fmt_num(prior.lambda)
    lmin_str = _abcd_fmt_num(prior.lambda_min)

    panels = _PanelSpec[]
    if view === :phi || view === :both
        phi_rows = String[]
        has_int && push!(phi_rows, _INTERCEPT_LABEL)
        for l in 1:prior.p, v in 1:n
            push!(phi_rows, "$(prior.obs_names[v]).L$l")
        end
        while length(phi_rows) < k
            push!(phi_rows, "x$(length(phi_rows) + 1)")
        end
        id_phi = _next_plot_id("dsgevar_phi")
        data_phi = _matrix_heat_json(Matrix{Float64}(prior.Phi_star), phi_rows, prior.obs_names)
        js_phi = _render_heatmap_js(id_phi, data_phi, _json(phi_rows), _json(prior.obs_names);
                                    xlabel="Equation", ylabel="Regressor", tip_label="",
                                    scale=:diverging, midpoint=0)
        pt_phi = "Prior Mean Φ* (λ=$(lam_str), λ_min=$(lmin_str), p=$(prior.p), T=$(prior.T_obs))"
        push!(panels, _PanelSpec(id_phi, pt_phi, js_phi))
    end
    if view === :sigma || view === :both
        id_sig = _next_plot_id("dsgevar_sig")
        data_sig = _matrix_heat_json(Matrix{Float64}(prior.Sigma_u_star), prior.obs_names, prior.obs_names)
        js_sig = _render_heatmap_js(id_sig, data_sig, _json(prior.obs_names), _json(prior.obs_names);
                                    xlabel="Observable", ylabel="Observable", tip_label="",
                                    scale=:diverging, midpoint=0)
        pt_sig = "Innovation Covariance Σ_u* (λ=$(lam_str), n=$(n))"
        push!(panels, _PanelSpec(id_sig, pt_sig, js_sig))
    end

    ftitle = isempty(title) ? "DSGE-VAR(p=$(prior.p)) Prior (λ=$(lam_str), T=$(prior.T_obs))" : title
    p = _make_plot(panels; title=ftitle, ncols=ncols <= 0 ? length(panels) : ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# DSGEVARLambdaProfile (#915)
# =============================================================================

"""
    plot_result(prof::DSGEVARLambdaProfile; title="", ncols=0, save_path=nothing)

Plot the closed-form DSGE-VAR(`λ`) log marginal likelihood profile over `prof.lambdas`,
marking the optimal weight `λ̂` (`prof.lambda_hat`) with a vertical reference line and the
dogmatic DSGE limit `λ = ∞` (`prof.logml_inf`) with a dashed horizontal line.
"""
function plot_result(prof::DSGEVARLambdaProfile{T};
                     title::String="", ncols::Int=0,
                     save_path::Union{String,Nothing}=nothing) where {T}
    panel, note = _dsgevar_lambda_panel(
        "dsgevar_lam", prof.lambdas, prof.log_marginal_likelihoods,
        prof.lambda_hat, prof.logml_hat, prof.logml_inf, prof.lambda_min,
    )
    ftitle = isempty(title) ?
        "DSGE-VAR(p=$(prof.p)) Marginal Likelihood Profile (T=$(prof.T_obs))" : title
    p = _make_plot([panel]; title=ftitle, ncols=ncols <= 0 ? 1 : ncols, note=note)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# DSGEVARPosterior (#915)
# =============================================================================

"""
    plot_result(post::DSGEVARPosterior; view=:irf, var=nothing, shock=nothing,
                horizon=20, method=:dsge_qr, quantiles=[0.05, 0.16, 0.5, 0.84, 0.95],
                stat=:median, params=nothing, thin=1, max_panels=12,
                ncols=0, title="", save_path=nothing)

Plot Del Negro & Schorfheide (2004, 2007) Bayesian [`DSGEVARPosterior`](@ref) results.

# Views
- `:irf` (default) — posterior DSGE-VAR impulse response credible bands (`quantiles`,
  central line via `stat=:median` or `:mean`) overlaid with the dogmatic DSGE-VAR (`λ = ∞`)
  IRF curve (when `method` is `:dsge_qr` or `:cholesky`). Supports `var` and `shock` selection
  by `Int` or `String` (bounds-checked) and caps un-filtered grids at `max_panels`.
- `:draws` — posterior trace and histogram/KDE panels for DSGE structural parameters
  (`theta_draws`, joint mode) or VAR coefficients (`B_draws`, fixed-`θ` mode). Supports
  `params`, `thin`, and `max_panels`. `var`/`shock` do not apply.
- `:lambda` — in joint mode (`log_posterior !== nothing`), plots the log-posterior trace
  and density histogram; in fixed-`θ` conjugate mode, plots the closed-form marginal
  likelihood profile over `λ` marking `post.lambda` against `λ = ∞`. `var`/`shock`/`params`
  do not apply.

Unknown `view` throws an `ArgumentError`.
"""
function plot_result(post::DSGEVARPosterior{T};
                     view::Symbol=:irf,
                     var::Union{Int,String,AbstractVector,Nothing}=nothing,
                     shock::Union{Int,String,AbstractVector,Nothing}=nothing,
                     horizon::Int=20,
                     method::Symbol=:dsge_qr,
                     quantiles::AbstractVector{<:Real}=[0.05, 0.16, 0.5, 0.84, 0.95],
                     stat::Symbol=:median,
                     params=nothing,
                     thin::Int=1,
                     max_panels::Int=12,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    lam_str = isinf(post.lambda) ? "∞" : _abcd_fmt_num(post.lambda)

    if view === :irf
        params === nothing || throw(ArgumentError("params applies only to view=:draws"))
        horizon >= 1 || throw(ArgumentError("horizon must be >= 1; got $horizon"))
        stat in (:median, :mean) || throw(ArgumentError("stat must be :median or :mean; got :$stat"))

        q_req = Float64[Float64(q) for q in quantiles]
        birf = irf(post, horizon; method=method, quantiles=q_req, point_estimate=stat)
        vars_to_plot = _abcd_resolve_sel(var, birf.variables)
        shocks_to_plot = _abcd_resolve_sel(shock, birf.shocks)
        pairs = [(vi, si) for vi in vars_to_plot for si in shocks_to_plot]
        n_total = length(pairs)
        shown = (var === nothing && shock === nothing) ? min(n_total, max(1, max_panels)) : n_total
        note = (var === nothing && shock === nothing) ?
            _cap_note("response-shock panels", shown, n_total, "max_panels") : ""

        # Dogmatic DSGE-VAR (λ = ∞) IRF overlay from prior moments
        irf_dog = nothing
        if method === :dsge_qr || method === :cholesky
            pr = post.dsge_prior
            if all(isfinite, pr.Sigma_u_star) && (method === :cholesky || all(isfinite, pr.Omega_star))
                L_star = Matrix{Float64}(safe_cholesky(Hermitian(Matrix{Float64}(pr.Sigma_u_star)); silent=true))
                B0_star = method === :dsge_qr ? L_star * Matrix{Float64}(pr.Omega_star) : L_star
                if size(B0_star, 2) == length(birf.shocks)
                    Phi_pad = _dsgevar_pad_B(Matrix{Float64}(pr.Phi_star), post.p, post.n)
                    irf_dog = _dsgevar_structural_irf(Phi_pad, B0_star, post.n, post.p, horizon)
                end
            else
                irf_dog = fill(NaN, horizon, post.n, length(birf.shocks))
            end
        end

        xs = collect(0:(horizon - 1))
        refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
        central_name = stat === :mean ? "DSGE-VAR Mean" : "DSGE-VAR Median"
        panels = _PanelSpec[]
        for (vi, si) in pairs[1:shown]
            id = _next_plot_id("dsgevar_irf")
            qmat = birf.quantiles[1:horizon, vi, si, :]
            central = stat === :mean ?
                Float64.(birf.point_estimate[1:horizon, vi, si]) :
                _median_from_quantiles(qmat, birf.quantile_levels, Float64.(birf.point_estimate[1:horizon, vi, si]))
            dog_vec = irf_dog === nothing ? nothing : Float64.(irf_dog[1:horizon, vi, si])
            data_json = _dsgevar_irf_panel_json(xs, qmat, birf.quantile_levels, central, dog_vec)

            snames = dog_vec === nothing ? String[central_name] : String[central_name, "Dogmatic DSGE (λ=∞)"]
            scols = dog_vec === nothing ? String[_PLOT_SERIES[1]] : String[_PLOT_SERIES[1], _PLOT_SERIES[2]]
            skeys = dog_vec === nothing ? String["med"] : String["med", "dog"]
            sdash = dog_vec === nothing ? String[""] : String["", "6,3"]
            s_json = _series_json(snames, scols; keys=skeys, dash=sdash)
            bands = _fan_bands_named_json(birf.quantile_levels; color=_PLOT_SERIES[1])
            js = _render_line_js(id, data_json, s_json; bands_json=bands, ref_lines_json=refs,
                                 integer_x=true, xlabel="Horizon", ylabel="Response")
            push!(panels, _PanelSpec(id, "$(birf.variables[vi]) ← $(birf.shocks[si])", js))
        end

        auto_ncols = length(shocks_to_plot) == 1 ?
            min(length(vars_to_plot), 3) :
            (shown == n_total ? length(shocks_to_plot) : min(length(shocks_to_plot), 3))
        ftitle = isempty(title) ?
            "Bayesian DSGE-VAR($(post.p)) Impulse Responses (λ = $(lam_str), $(method))" : title
        p = _make_plot(panels; title=ftitle, ncols=ncols <= 0 ? auto_ncols : ncols, note=note)

    elseif view === :draws
        (var === nothing && shock === nothing) ||
            throw(ArgumentError("var/shock apply only to view=:irf"))
        thin >= 1 || throw(ArgumentError("thin must be >= 1; got $thin"))

        if post.theta_draws !== nothing && !isempty(post.param_names)
            labels = String[string(nm) for nm in post.param_names]
            drawfun = i -> Float64[Float64(v) for v in @view post.theta_draws[:, i]]
            kind_str = "DSGE Parameter Draws"
        else
            labels = _dsgevar_coef_labels(post)
            k_reg = size(post.B_draws, 2)
            drawfun = function (gi)
                eq = div(gi - 1, k_reg) + 1
                j = mod(gi - 1, k_reg) + 1
                Float64[Float64(v) for v in @view post.B_draws[:, j, eq]]
            end
            kind_str = "VAR Coefficient Draws"
        end

        param_cap = max(1, fld(max(2, max_panels), 2))
        sel, total, capped = _resolve_mcmc_params(labels, params; cap=param_cap)
        panels = _PanelSpec[]
        for i in sel
            d = drawfun(i)
            push!(panels, _mcmc_trace_panel("$(labels[i]) (Trace)", d, thin))
            d_thinned = d[1:thin:end]
            fin = Float64[v for v in d_thinned if isfinite(v)]
            pm = isempty(fin) ? nothing : mean(fin)
            push!(panels, _predictive_hist_panel("dsgevar_hist", "$(labels[i]) (Posterior)",
                                                 d_thinned; barname="Draws", observed=pm))
        end
        note = capped ? _cap_note("parameters", length(sel), total, "max_panels") : ""
        ftitle = isempty(title) ?
            "DSGE-VAR($(post.p)) $(kind_str) (λ = $(lam_str), $(post.method))" : title
        p = _make_plot(panels; title=ftitle, ncols=ncols <= 0 ? 2 : ncols, note=note)

    elseif view === :lambda
        (var === nothing && shock === nothing && params === nothing) ||
            throw(ArgumentError("var/shock apply only to view=:irf; params applies only to view=:draws"))
        thin >= 1 || throw(ArgumentError("thin must be >= 1; got $thin"))

        if post.log_posterior !== nothing
            lp = Float64[Float64(v) for v in post.log_posterior]
            lp_thinned = lp[1:thin:end]
            fin = Float64[v for v in lp_thinned if isfinite(v)]
            pm = isempty(fin) ? nothing : mean(fin)
            panels = _PanelSpec[
                _mcmc_trace_panel("Log Posterior Trace", lp, thin),
                _predictive_hist_panel("dsgevar_lp_hist", "Log Posterior Density",
                                       lp_thinned; barname="Draws", observed=pm),
            ]
            ftitle = isempty(title) ?
                "DSGE-VAR($(post.p)) Log Posterior (λ = $(lam_str), log ML = $(_abcd_fmt_num(post.log_marginal_likelihood)))" :
                title
            p = _make_plot(panels; title=ftitle, ncols=ncols <= 0 ? 2 : ncols)
        else
            pr = post.dsge_prior
            can_eval = !isempty(post.data) && size(post.data, 1) > post.p &&
                       size(post.data, 2) == post.n && all(isfinite, post.data) &&
                       all(isfinite, pr.Gamma_yy) && all(isfinite, pr.Gamma_xy) &&
                       all(isfinite, pr.Gamma_xx) && all(isfinite, pr.Phi_star) &&
                       all(isfinite, pr.Sigma_u_star)
            if can_eval
                inc_int = _dsgevar_has_intercept(pr)
                Y_eff, X, _, XtY, XtX, T_eff_data = _dsgevar_data_stats(
                    Float64, post.data, post.p, post.n, inc_int)
                lmin = Float64(pr.lambda_min)
                base_grid = Float64[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 25.0, 50.0]
                isfinite(post.lambda) && push!(base_grid, Float64(post.lambda))
                lmin_safe = lmin * (1.0 + 1e-6)
                lam_vec = sort!(unique!(vcat([lmin_safe], filter(>(lmin_safe), base_grid))))
                Gyy = Matrix{Float64}(pr.Gamma_yy)
                Gxy = Matrix{Float64}(pr.Gamma_xy)
                Gxx = Matrix{Float64}(pr.Gamma_xx)
                Ps = Matrix{Float64}(pr.Phi_star)
                Su = Matrix{Float64}(pr.Sigma_u_star)
                lml_vec = Float64[_dsgevar_logml_eval(Gyy, Gxy, Gxx, Ps, Su, pr.T_obs, lv,
                                                      Y_eff, X, XtY, XtX, T_eff_data) for lv in lam_vec]
                lml_inf = _dsgevar_logml_eval(Gyy, Gxy, Gxx, Ps, Su, pr.T_obs, Inf,
                                              Y_eff, X, XtY, XtX, T_eff_data)
                panel, note = _dsgevar_lambda_panel(
                    "dsgevar_lam_post", lam_vec, lml_vec,
                    Float64(post.lambda), Float64(post.log_marginal_likelihood), lml_inf, lmin,
                )
                ftitle = isempty(title) ?
                    "DSGE-VAR($(post.p)) Prior Weight λ Profile (λ = $(lam_str), log ML = $(_abcd_fmt_num(post.log_marginal_likelihood)))" :
                    title
                p = _make_plot([panel]; title=ftitle, ncols=ncols <= 0 ? 1 : ncols, note=note)
            else
                # Fallback when data is non-finite or too short for profile re-evaluation
                logdets = Float64[
                    all(isfinite, @view post.Sigma_draws[s, :, :]) ?
                        logdet_safe(Matrix{Float64}(@view post.Sigma_draws[s, :, :])) : NaN
                    for s in 1:post.n_draws
                ]
                ld_thinned = logdets[1:thin:end]
                fin = Float64[v for v in ld_thinned if isfinite(v)]
                pm = isempty(fin) ? nothing : mean(fin)
                panels = _PanelSpec[
                    _mcmc_trace_panel("log |Σ_u| Trace", logdets, thin),
                    _predictive_hist_panel("dsgevar_ld_hist", "log |Σ_u| Density",
                                           ld_thinned; barname="Draws", observed=pm),
                ]
                ftitle = isempty(title) ?
                    "DSGE-VAR($(post.p)) Innovation Scale (λ = $(lam_str), log ML = $(_abcd_fmt_num(post.log_marginal_likelihood)))" :
                    title
                p = _make_plot(panels; title=ftitle, ncols=ncols <= 0 ? 2 : ncols)
            end
        end
    else
        throw(ArgumentError("Unknown view :$view. Valid views: :irf, :draws, :lambda"))
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end
