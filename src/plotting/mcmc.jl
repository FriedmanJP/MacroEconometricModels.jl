# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# MCMC / posterior diagnostic plots (PLT-26 / #488).
#
# `plot_result(::BayesianDSGE)` was relocated here out of models.jl by the PLT
# plotting overhaul (Wave-1) so this lane owns the file. It now dispatches on
# `view` (:trace/:density/:running/:acf). `:density` keeps the PLT-05 prior-vs-
# posterior figure (posterior KDE via the shared `_kde_line` helper, prior overlay,
# posterior-mean reference line). New `plot_result` methods cover `BVARPosterior`
# and `MCMCDiagnostics`. No renderer is defined here (plotrule A1); statistics come
# from `_kde_line` / `acf` (plotrule A5).
# =============================================================================

# -----------------------------------------------------------------------------
# Parameter selection (Symbol/String/Int; nothing → all, capped — C3/C4/C7)
# -----------------------------------------------------------------------------

"""
    _resolve_mcmc_params(labels, params; cap=12) -> (sel::Vector{Int}, total::Int, capped::Bool)

Resolve a `params` selector against the ordered `labels`. `params === nothing`
selects all, truncated to `cap` (with `capped=true` so the caller can surface the
cap — plotrule C7). A vector (or scalar) of `Int`/`Symbol`/`String` selects those
parameters; an out-of-range index or unknown name throws `ArgumentError` (C3). An
explicit selection is never capped (C4 — `params` is honored verbatim).
"""
function _resolve_mcmc_params(labels::Vector{String}, params; cap::Int=12)
    n = length(labels)
    if params === nothing
        sel = collect(1:n)
        capped = n > cap
        capped && (sel = sel[1:cap])
        return (sel, n, capped)
    end
    raw = params isa AbstractVector ? collect(params) : [params]
    sel = Int[]
    for pr in raw
        if pr isa Integer
            (1 <= pr <= n) ||
                throw(ArgumentError("parameter index $(pr) out of range 1:$(n)"))
            push!(sel, Int(pr))
        else
            s = string(pr)
            j = findfirst(==(s), labels)
            j === nothing &&
                throw(ArgumentError("unknown parameter \"$(s)\"; available: $(labels)"))
            push!(sel, j)
        end
    end
    (sel, n, false)
end

# -----------------------------------------------------------------------------
# Per-parameter view panels (trace / running mean / draw-ACF / density)
# -----------------------------------------------------------------------------

# Trace: draw index vs value, with a posterior-mean horizontal reference line
# (axis:"y"). `thin` subsamples the retained sequence; the x-axis keeps the
# original iteration numbers so thinning changes point count, not the span.
function _mcmc_trace_panel(label::String, draws::AbstractVector, thin::Int)
    idxs = 1:thin:length(draws)
    rows = Vector{Pair{String,String}}[]
    for k in idxs
        push!(rows, ["x" => _json(k), "val" => _json(Float64(draws[k]))])
    end
    data = _json_array_of_objects(rows)
    fin = Float64[Float64(v) for v in draws if isfinite(v)]
    pm = isempty(fin) ? 0.0 : mean(fin)
    s = "[{\"name\":$(_json("draw")),\"color\":$(_json(_PLOT_COLORS[1])),\"key\":\"val\",\"dash\":\"\"}]"
    refs = "[{\"value\":$(_json(pm)),\"axis\":\"y\",\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"4,3\"}]"
    id = _next_plot_id("mcmc_trace")
    js = _render_line_js(id, data, s; ref_lines_json=refs, integer_x=true,
                         xlabel="Draw", ylabel="Value")
    _PanelSpec(id, label, js)
end

# Running (cumulative) mean of the retained sequence — the convergence eye-test —
# with the final posterior mean as a reference line. Non-finite draws are skipped in
# the accumulation so one bad draw does not poison the whole curve.
function _mcmc_running_panel(label::String, draws::AbstractVector, thin::Int)
    idxs = 1:thin:length(draws)
    rows = Vector{Pair{String,String}}[]
    s = 0.0; c = 0
    for k in idxs
        v = Float64(draws[k])
        if isfinite(v); s += v; c += 1; end
        push!(rows, ["x" => _json(k), "val" => _json(c > 0 ? s / c : NaN)])
    end
    data = _json_array_of_objects(rows)
    fin = Float64[Float64(v) for v in draws if isfinite(v)]
    pm = isempty(fin) ? 0.0 : mean(fin)
    ser = "[{\"name\":$(_json("running mean")),\"color\":$(_json(_PLOT_COLORS[1])),\"key\":\"val\",\"dash\":\"\"}]"
    refs = "[{\"value\":$(_json(pm)),\"axis\":\"y\",\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"4,3\"}]"
    id = _next_plot_id("mcmc_run")
    js = _render_line_js(id, data, ser; ref_lines_json=refs, integer_x=true,
                         xlabel="Draw", ylabel="Running mean")
    _PanelSpec(id, label, js)
end

# Draw-ACF via the spectral `acf` + the shared correlogram vbar panel (arima.jl,
# same lane). Assesses chain mixing.
function _mcmc_acf_panel(label::String, draws::AbstractVector, thin::Int)
    dr = Float64[Float64(draws[k]) for k in 1:thin:length(draws)]
    fin = Float64[v for v in dr if isfinite(v)]
    if length(fin) >= 3
        nlags = max(1, min(40, fld(length(fin), 4)))
        a = acf(fin; lags=nlags)
        return _correlogram_vbar_panel("mcmc_acf", a.lags, a.acf, a.ci, "Draw ACF", label)
    end
    _correlogram_vbar_panel("mcmc_acf", Int[], Float64[], 0.0, "Draw ACF", label)
end

# -----------------------------------------------------------------------------
# Shared trace/running/acf assembly (BayesianDSGE + BVARPosterior)
# -----------------------------------------------------------------------------

"""
    _plot_mcmc(labels, drawfun, view, params, thin, title, ncols, save_path, figbase; density_panels)

Assemble the requested `view` (`:trace`/`:running`/`:acf`/`:density`) for the
parameters selected from `labels` (draws fetched lazily via `drawfun(i)`). `:density`
delegates to the model-specific `density_panels(sel)` closure. Surfaces the parameter
cap in the figure title (C7). Unknown `view` throws `ArgumentError` (C5).
"""
function _plot_mcmc(labels::Vector{String}, drawfun, view::Symbol, params,
                    thin::Int, title::String, ncols::Int, save_path,
                    figbase::String; density_panels)
    thin >= 1 || throw(ArgumentError("thin must be >= 1, got $(thin)"))
    sel, total, capped = _resolve_mcmc_params(labels, params)
    if view === :density
        panels = density_panels(sel)
        vlabel = "Prior vs Posterior"
    elseif view === :trace
        panels = _PanelSpec[_mcmc_trace_panel(labels[i], drawfun(i), thin) for i in sel]
        vlabel = "Trace"
    elseif view === :running
        panels = _PanelSpec[_mcmc_running_panel(labels[i], drawfun(i), thin) for i in sel]
        vlabel = "Running Mean"
    elseif view === :acf
        panels = _PanelSpec[_mcmc_acf_panel(labels[i], drawfun(i), thin) for i in sel]
        vlabel = "Draw ACF"
    else
        throw(ArgumentError("unknown view :$view; valid views: :trace, :density, :running, :acf"))
    end
    capnote = capped ? " (showing $(length(sel)) of $(total) parameters)" : ""
    ft = isempty(title) ? "$(figbase) — $(vlabel)$(capnote)" : title * capnote
    p = _make_plot(panels; title=ft, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# -----------------------------------------------------------------------------
# BayesianDSGE
# -----------------------------------------------------------------------------

_safe_pdf(d, x) = try
    v = pdf(d, x); isfinite(v) ? Float64(v) : 0.0
catch
    0.0
end

# Prior (dashed) + posterior KDE (solid) overlay per parameter, with a posterior-mean
# vertical reference line (axis:"x"). PLT-05 behavior, PLT-19 shared `_kde_line`.
function _bayesdsge_density_panels(result::BayesianDSGE, sel::Vector{Int}, labels::Vector{String})
    panels = _PanelSpec[]
    for i in sel
        draws = Float64[Float64(v) for v in @view result.theta_draws[:, i]]
        d = result.priors.distributions[i]
        xs, kde_vals = _kde_line(draws)
        rows = Vector{Pair{String,String}}[]
        for g in eachindex(xs)
            push!(rows, ["x" => _json(xs[g]), "post" => _json(kde_vals[g]),
                         "prior" => _json(_safe_pdf(d, xs[g]))])
        end
        data = _json_array_of_objects(rows)
        fin = Float64[v for v in draws if isfinite(v)]
        pm = isempty(fin) ? 0.0 : mean(fin)
        s = "[{\"name\":$(_json("Posterior")),\"color\":$(_json(_PLOT_COLORS[1])),\"key\":\"post\",\"dash\":\"\"}," *
            "{\"name\":$(_json("Prior")),\"color\":$(_json(_PLOT_COLORS[2])),\"key\":\"prior\",\"dash\":\"6,3\"}]"
        refs = "[{\"value\":$(_json(pm)),\"axis\":\"x\",\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"4,3\"}]"
        id = _next_plot_id("bayes_dens")
        js = _render_line_js(id, data, s; ref_lines_json=refs, xlabel=labels[i], ylabel="Density")
        push!(panels, _PanelSpec(id, labels[i], js))
    end
    panels
end

"""
    plot_result(result::BayesianDSGE; view=:trace, params=nothing, thin=1, title="", ncols=0, save_path=nothing)

MCMC diagnostics for a Bayesian DSGE posterior. `view` selects `:trace` (default,
draw sequence + posterior-mean line), `:density` (prior dashed vs posterior-KDE solid
with a posterior-mean vertical line), `:running` (cumulative mean convergence check),
or `:acf` (draw autocorrelation). `params` (Symbol/String/Int, scalar or vector)
selects parameters; `nothing` shows all, capped at 12 with the cap noted in the title.
`thin` subsamples the retained draw sequence. Unknown `view` throws `ArgumentError`.
"""
function plot_result(result::BayesianDSGE{T}; view::Symbol=:trace, params=nothing,
                     thin::Int=1, title::String="", ncols::Int=0,
                     save_path::Union{String,Nothing}=nothing) where {T}
    labels = String[string(p) for p in result.param_names]
    drawfun = i -> Float64[Float64(v) for v in @view result.theta_draws[:, i]]
    dpan = sel -> _bayesdsge_density_panels(result, sel, labels)
    _plot_mcmc(labels, drawfun, view, params, thin, title, ncols, save_path,
               "Bayesian DSGE"; density_panels=dpan)
end

# -----------------------------------------------------------------------------
# BVARPosterior
# -----------------------------------------------------------------------------

"""
    _bvar_param_labels(post) -> Vector{String}

Per-coefficient labels for a `BVARPosterior`, flattened in equation-major order:
`"<depvar> ← <regressor>"` where the regressor labels mirror the `report()`
construction (`_INTERCEPT_LABEL`, then `varname.L{lag}`; bvar/types.jl).
"""
function _bvar_param_labels(post::BVARPosterior)
    vn = post.varnames
    k = size(post.B_draws, 2)
    coef_names = String[_INTERCEPT_LABEL]
    for l in 1:post.p, v in 1:post.n
        push!(coef_names, "$(vn[v]).L$(l)")
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

# Posterior-only KDE per coefficient (no prior overlay — a BVAR posterior carries no
# per-coefficient prior density), with a posterior-mean vertical line.
function _bvar_density_panels(sel::Vector{Int}, drawfun, labels::Vector{String})
    panels = _PanelSpec[]
    for i in sel
        draws = drawfun(i)
        xs, dens = _kde_line(draws)
        rows = Vector{Pair{String,String}}[]
        for g in eachindex(xs)
            push!(rows, ["x" => _json(xs[g]), "post" => _json(dens[g])])
        end
        data = _json_array_of_objects(rows)
        fin = Float64[v for v in draws if isfinite(v)]
        pm = isempty(fin) ? 0.0 : mean(fin)
        s = "[{\"name\":$(_json("Posterior")),\"color\":$(_json(_PLOT_COLORS[1])),\"key\":\"post\",\"dash\":\"\"}]"
        refs = "[{\"value\":$(_json(pm)),\"axis\":\"x\",\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"4,3\"}]"
        id = _next_plot_id("bvar_dens")
        js = _render_line_js(id, data, s; ref_lines_json=refs, xlabel=labels[i], ylabel="Density")
        push!(panels, _PanelSpec(id, labels[i], js))
    end
    panels
end

"""
    plot_result(post::BVARPosterior; view=:trace, params=nothing, thin=1, title="", ncols=0, save_path=nothing)

MCMC diagnostics for a Bayesian VAR posterior. Same views as the DSGE method
(`:trace`/`:density`/`:running`/`:acf`); `:density` is the posterior KDE (no prior
overlay). Coefficients are addressed by the mirrored `"<depvar> ← <regressor>"`
labels (Symbol/String/Int selection). Unknown `view` throws `ArgumentError`.
"""
function plot_result(post::BVARPosterior{T}; view::Symbol=:trace, params=nothing,
                     thin::Int=1, title::String="", ncols::Int=0,
                     save_path::Union{String,Nothing}=nothing) where {T}
    labels = _bvar_param_labels(post)
    k = size(post.B_draws, 2)
    drawfun = function (gi)
        eq = div(gi - 1, k) + 1
        j = mod(gi - 1, k) + 1
        Float64[Float64(v) for v in @view post.B_draws[:, j, eq]]
    end
    dpan = sel -> _bvar_density_panels(sel, drawfun, labels)
    _plot_mcmc(labels, drawfun, view, params, thin, title, ncols, save_path,
               "Bayesian VAR"; density_panels=dpan)
end

# -----------------------------------------------------------------------------
# MCMCDiagnostics summary (horizontal R-hat / ESS with threshold reference lines)
# -----------------------------------------------------------------------------

"""
    _threshold_coef_panel(prefix, ptitle, labels, effects, ci_lo, ci_hi, ref_value; logx=false, flag=:none) -> _PanelSpec

Horizontal dot/bar panel via the frozen `_render_coef_plot_js` (plotrule A1): one row
per named entity read horizontally (plotrule PLT-06 / form table), a value dot with a
`[ci_lo, ci_hi]` whisker, and the alert-colored reference line at `ref_value`. `flag`
(`:above`/`:below`/`:none`) counts entities past the threshold and surfaces the count
in the panel title. Shared by `MCMCDiagnostics` here and the PLT-27 static-Bayesian
plots (bayes.jl, same lane).
"""
function _threshold_coef_panel(prefix::String, ptitle::String, labels::Vector{String},
                               effects::AbstractVector, ci_lo::AbstractVector,
                               ci_hi::AbstractVector, ref_value::Real;
                               logx::Bool=false, flag::Symbol=:none)
    rows = Vector{Pair{String,String}}[]
    nflag = 0
    for i in eachindex(labels)
        e = Float64(effects[i])
        if flag === :above && isfinite(e) && e > ref_value
            nflag += 1
        elseif flag === :below && isfinite(e) && e < ref_value
            nflag += 1
        end
        push!(rows, ["name" => _json(labels[i]), "effect" => _json(e),
                     "ci_lo" => _json(Float64(ci_lo[i])), "ci_hi" => _json(Float64(ci_hi[i]))])
    end
    data = _json_array_of_objects(rows)
    id = _next_plot_id(prefix)
    js = _render_coef_plot_js(id, data; ref_value=ref_value, logx=logx,
                              xlabel=ptitle, ylabel="")
    note = (flag !== :none && nflag > 0) ? " ($(nflag) flagged)" : ""
    _PanelSpec(id, ptitle * note, js)
end

"""
    plot_result(d::MCMCDiagnostics; title="", ncols=0, save_path=nothing)

Convergence-diagnostic summary: horizontal panels of split-R̂ (reference line at
`1.01`), bulk-ESS and tail-ESS (reference line at `400`, the Vehtari et al. 2021
thresholds). Parameters past a threshold are counted in the panel title; the
threshold reference line is drawn in the alert color.
"""
function plot_result(d::MCMCDiagnostics{T}; title::String="", ncols::Int=0,
                     save_path::Union{String,Nothing}=nothing) where {T}
    labels = String[string(p) for p in d.param_names]
    panels = _PanelSpec[
        _threshold_coef_panel("mcmc_rhat", "R-hat", labels, d.rhat, d.rhat, d.rhat,
                              1.01; flag=:above),
        _threshold_coef_panel("mcmc_essb", "ESS (bulk)", labels, d.ess_bulk,
                              d.ess_bulk, d.ess_bulk, 400.0; flag=:below),
        _threshold_coef_panel("mcmc_esst", "ESS (tail)", labels, d.ess_tail,
                              d.ess_tail, d.ess_tail, 400.0; flag=:below),
    ]
    ft = isempty(title) ? "MCMC Convergence Diagnostics ($(length(labels)) parameters)" : title
    p = _make_plot(panels; title=ft, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end
