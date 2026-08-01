# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# Prior/posterior & predictive-check plots (PLT-27 / #489). Five NEW dispatches
# for the Bayesian-DSGE diagnostic result types. No renderer is defined here
# (plotrule A1): horizontal dot/bar panels go through `_threshold_coef_panel`
# (mcmc.jl, same lane) → the frozen `_render_coef_plot_js`; predictive histograms
# through `_render_histogram_js` with the frozen `_histogram_bins`/`_kde_line_json`
# converters. Numbers embedded in titles are `_fmt`'d, never `_json`'d (C9).
# =============================================================================

# One prior/posterior-predictive statistic panel: a density-scaled histogram of the
# (prior- or posterior-) predicted statistic with a KDE overlay, plus an optional
# vertical reference line at the observed value (axis:"x", PLT-05 support).
function _predictive_hist_panel(prefix::String, ptitle::String, vals::AbstractVector;
                                barname::String="Draws",
                                observed::Union{Nothing,Real}=nothing)
    bins = _histogram_bins(vals; density=true)
    dens = _kde_line_json(vals)
    series = "[{\"name\":$(_json(barname)),\"color\":$(_json(_PLOT_COLORS[1]))}," *
             "{\"name\":$(_json("Density")),\"color\":$(_json(_PLOT_COLORS[2]))}]"
    refs = "[]"
    if observed !== nothing && isfinite(observed)
        refs = "[{\"value\":$(_json(Float64(observed))),\"axis\":\"x\"," *
               "\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"4,3\"}]"
    end
    id = _next_plot_id(prefix)
    js = _render_histogram_js(id, bins, series; density_json=dens, ref_lines_json=refs,
                              xlabel=ptitle, ylabel="Density")
    _PanelSpec(id, ptitle, js)
end

# -----------------------------------------------------------------------------
# PriorPosteriorOverlap — weak-identification symptom
# -----------------------------------------------------------------------------

"""
    plot_result(o::PriorPosteriorOverlap; title="", ncols=0, save_path=nothing)

Horizontal bar of the prior/posterior overlap coefficient per parameter (names read
horizontally, plotrule PLT-06) with a reference line at the flag `threshold` drawn in
the alert color. An overlap at or above the threshold means the data barely moved the
prior (weak identification); the flagged count and the `_fmt`'d threshold are stated
in the figure title (C7/C9).
"""
function plot_result(o::PriorPosteriorOverlap{T}; title::String="", ncols::Int=0,
                     save_path::Union{String,Nothing}=nothing) where {T}
    labels = String[string(p) for p in o.param_names]
    ov = Float64[Float64(v) for v in o.overlap]
    panel = _threshold_coef_panel("pp_overlap", "Prior/posterior overlap", labels,
                                  ov, ov, ov, Float64(o.threshold); flag=:above)
    nflag = count(o.flagged)
    ft = isempty(title) ?
        "Prior/Posterior Overlap ($(nflag) of $(length(labels)) flagged, threshold $(_fmt(o.threshold)))" :
        title
    p = _make_plot([panel]; title=ft, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# -----------------------------------------------------------------------------
# PriorPredictiveResult — distribution of each summary statistic
# -----------------------------------------------------------------------------

"""
    plot_result(r::PriorPredictiveResult; view=:density, title="", ncols=0, save_path=nothing)

Histogram + KDE of each prior-predictive summary statistic (one panel per column of
`stats`), capped at 12 statistics with the cap noted in the title (C7). Only
`view=:density` is defined; any other view throws `ArgumentError`.
"""
function plot_result(r::PriorPredictiveResult{T}; view::Symbol=:density, title::String="",
                     ncols::Int=0, save_path::Union{String,Nothing}=nothing) where {T}
    view === :density ||
        throw(ArgumentError("unknown view :$view; valid views: :density"))
    names = r.stat_names
    nstat = length(names)
    cap = 12
    shown = min(nstat, cap)
    panels = _PanelSpec[]
    for j in 1:shown
        col = Float64[Float64(v) for v in @view r.stats[:, j]]
        push!(panels, _predictive_hist_panel("prior_pred", names[j], col; barname="Prior draws"))
    end
    capnote = nstat > cap ? " (showing $(shown) of $(nstat) statistics)" : ""
    ft = isempty(title) ? "Prior Predictive Distribution$(capnote)" : title * capnote
    p = _make_plot(panels; title=ft, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# -----------------------------------------------------------------------------
# PosteriorPredictiveCheck — replicated vs observed statistics
# -----------------------------------------------------------------------------

"""
    plot_result(ppc::PosteriorPredictiveCheck; title="", ncols=0, save_path=nothing)

For each summary statistic, the replicated-data distribution (histogram + KDE) with a
vertical line at the observed value (axis:"x"); the posterior-predictive p-value is
`_fmt`'d into the panel title (C9). Capped at 12 statistics with the cap noted (C7).
"""
function plot_result(ppc::PosteriorPredictiveCheck{T}; title::String="", ncols::Int=0,
                     save_path::Union{String,Nothing}=nothing) where {T}
    names = ppc.stat_names
    nstat = length(names)
    cap = 12
    shown = min(nstat, cap)
    panels = _PanelSpec[]
    for j in 1:shown
        col = Float64[Float64(v) for v in @view ppc.replicated[:, j]]
        obs = Float64(ppc.observed[j])
        pv = ppc.p_values[j]
        pvs = isnan(pv) ? "n/a" : _fmt(pv)
        ptitle = "$(names[j]) (p = $(pvs))"
        push!(panels, _predictive_hist_panel("post_pred", ptitle, col;
                                             barname="Replicated", observed=obs))
    end
    capnote = nstat > cap ? " (showing $(shown) of $(nstat) statistics)" : ""
    ft = isempty(title) ? "Posterior Predictive Check$(capnote)" : title * capnote
    p = _make_plot(panels; title=ft, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# -----------------------------------------------------------------------------
# IdentificationDiagnostics — Iskrev rank / singular values
# -----------------------------------------------------------------------------

"""
    plot_result(idg::IdentificationDiagnostics; title="", ncols=0, save_path=nothing)

Horizontal bar of the Jacobian singular values on a **log scale** with the effective
`tol` reference line (alert color); a value below `tol` marks an unidentified
direction and is counted in the panel title. The figure title states the rank
verdict (`rank / n_params`, identified vs UNIDENTIFIED).
"""
function plot_result(idg::IdentificationDiagnostics{T}; title::String="", ncols::Int=0,
                     save_path::Union{String,Nothing}=nothing) where {T}
    sv = Float64[Float64(v) for v in idg.singular_values]
    labels = String["σ$(i)" for i in eachindex(sv)]
    panel = _threshold_coef_panel("ident_sv", "Singular value (log scale)", labels,
                                  sv, sv, sv, Float64(idg.tol); logx=true, flag=:below)
    verdict = idg.identified ? "identified" : "UNIDENTIFIED"
    ft = isempty(title) ?
        "Identification — $(verdict) (rank $(idg.rank)/$(idg.n_params))" : title
    p = _make_plot([panel]; title=ft, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# -----------------------------------------------------------------------------
# PosteriorMode — mode + Laplace 95% interval
# -----------------------------------------------------------------------------

"""
    plot_result(pm::PosteriorMode; title="", ncols=0, save_path=nothing)

Horizontal dot-and-whisker of each parameter's posterior mode with its Laplace 95%
interval (`mode ± 1.96·√diag(inv_hessian)`), zero reference line. A non-positive
diagonal of the inverse Hessian yields a `NaN` (→ `null`) interval end rather than a
crash.
"""
function plot_result(pm::PosteriorMode{T}; title::String="", ncols::Int=0,
                     save_path::Union{String,Nothing}=nothing) where {T}
    labels = String[string(p) for p in pm.param_names]
    mode = Float64[Float64(v) for v in pm.mode]
    dv = diag(pm.inv_hessian)
    se = Float64[d >= 0 ? sqrt(Float64(d)) : NaN for d in dv]
    lo = mode .- 1.96 .* se
    hi = mode .+ 1.96 .* se
    panel = _threshold_coef_panel("post_mode", "Posterior mode (95% Laplace)", labels,
                                  mode, lo, hi, 0.0; flag=:none)
    ft = isempty(title) ? "Posterior Mode (Laplace 95% intervals)" : title
    p = _make_plot([panel]; title=ft, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end
