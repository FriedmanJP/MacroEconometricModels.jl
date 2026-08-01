# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
plot_result methods for the micro / panel / limited-dependent-variable coefficient
family (PLT-36). Every dispatch is a **horizontal** dot-and-whisker (forest) plot so
coefficient names read horizontally, via the frozen `_render_coef_plot_js` (A1):

- `PanelRegModel`, `PanelIVModel`, `PMGModel` — coefficient plot from the slope
  vector, its standard errors, and the coefficient names.
- `OrderedModel` (ordered logit/probit) — slope coefficients; cutpoints in a note.
- `MultinomialLogitModel` — one coefficient facet per non-base outcome (or a single
  chosen `category`).
- `MultinomialMarginalEffects` — per-outcome AME facets; **dots only** when standard
  errors are unavailable (`se === nothing`), with an honest subtitle (plotrule C6).
- `OddsRatio` — a forest plot on a **log x-axis** with the reference line at **1**.
- `HeckmanModel` — `view=:outcome` / `:selection` coefficient panels (two equations).
- `TobitModel`, `TruncRegModel`, `CointRegModel` — single coefficient panels.
- `SURModel`, `ThreeSLSModel` — one coefficient panel per equation.

The intercept row is omitted everywhere (mirroring `_display_intercept` in the `show`
methods). Numbers embedded in titles are rounded via `_fmt` (plotrule C9).
"""

# =============================================================================
# Lane-local converters (A5/A6) — helpers.jl is frozen for Wave 2
# =============================================================================

"""
    _coef_ci_json(names, est, ci_lo, ci_hi) -> String

Coefficient-plot `data_json` `[{name, effect, ci_lo, ci_hi}, …]` from an estimate
vector and explicit CI bounds, **omitting the intercept row** (mirroring
`_display_intercept`). Non-finite values serialize to `null` (plotrule A7). All rows
are kept when every name is an intercept (degenerate single-parameter model).
"""
function _coef_ci_json(names::AbstractVector, est::AbstractVector,
                       ci_lo::AbstractVector, ci_hi::AbstractVector)
    keep = [i for i in eachindex(names) if _display_intercept(string(names[i])) != _INTERCEPT_LABEL]
    isempty(keep) && (keep = collect(eachindex(names)))
    rows = Vector{Pair{String,String}}[]
    for i in keep
        push!(rows, ["name" => _json(string(names[i])), "effect" => _json(est[i]),
                     "ci_lo" => _json(ci_lo[i]), "ci_hi" => _json(ci_hi[i])])
    end
    _json_array_of_objects(rows)
end

"""
    _coef_plot_json(names, est, se; z=1.96) -> String

Coefficient-plot `data_json` from an estimate / standard-error pair, with symmetric
`ci = est ± z·se`; intercept omitted (via `_coef_ci_json`).
"""
_coef_plot_json(names::AbstractVector, est::AbstractVector, se::AbstractVector; z::Real=1.96) =
    _coef_ci_json(names, est, est .- z .* se, est .+ z .* se)

# Diagonal standard errors from a covariance matrix (nonnegative-clamped sqrt).
_diag_se(V::AbstractMatrix) = sqrt.(max.(diag(V), zero(eltype(V))))

# Build a single coef panel from an est/se pair.
function _coef_panel(prefix::String, names::AbstractVector, est::AbstractVector,
                     se::AbstractVector; z::Real=1.96, xlabel::String="Coefficient",
                     ptitle::String)
    id = _next_plot_id(prefix)
    js = _render_coef_plot_js(id, _coef_plot_json(names, est, se; z=z);
                              ref_value=0, xlabel=xlabel, ylabel="")
    _PanelSpec(id, ptitle, js)
end

# Resolve a multinomial `category` selector (Int index or label) to a non-base column
# index 1..J-1. `nonbase_labels` are the string labels of categories 2..J.
_resolve_category(category, nonbase_labels::Vector{String}) =
    _resolve_var(category isa AbstractString ? String(category) : category, nonbase_labels)

# =============================================================================
# Panel / panel-IV / PMG
# =============================================================================

"""
    plot_result(m::PanelRegModel; view=:coef, acf_lags=0, title="", save_path=nothing)

`view=:coef` (default) draws a horizontal coefficient plot of the panel-regression
slopes `β ± 1.96·SE` (SE from `diag(vcov_mat)`), with a zero reference line; intercept
omitted. `view=:diagnostics` draws the shared four-panel residual diagnostics
(residual-vs-fitted, histogram + fitted normal, Normal Q-Q, residual ACF) from the
model's stored `residuals`/`fitted` (PLT-24). Unknown `view` throws an `ArgumentError`.
"""
function plot_result(m::PanelRegModel{T};
                     view::Symbol=:coef, acf_lags::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    if view === :coef
        panel = _coef_panel("preg_coef", m.varnames, m.beta, _diag_se(m.vcov_mat);
                            ptitle="Coefficients ($(m.method), 95% CI)")
        ftitle = isempty(title) ? "Panel Regression Coefficients" : title
        p = _make_plot([panel]; title=ftitle, ncols=1)
    elseif view === :diagnostics
        panels = _residual_diagnostics_panels(m.residuals, m.fitted; acf_lags=acf_lags)
        ftitle = isempty(title) ? "Panel Regression Residual Diagnostics ($(m.method))" : title
        p = _make_plot(panels; title=ftitle, ncols=2)
    else
        throw(ArgumentError("Unknown view :$view. Valid views: :coef, :diagnostics"))
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(m::PanelIVModel; view=:coef, acf_lags=0, title="", save_path=nothing)

`view=:coef` (default) draws a horizontal coefficient plot of the panel-IV slopes
`β ± 1.96·SE`; intercept omitted. `view=:diagnostics` draws the shared four-panel
residual diagnostics from the model's stored `residuals`/`fitted` (PLT-24). Unknown
`view` throws an `ArgumentError`.
"""
function plot_result(m::PanelIVModel{T};
                     view::Symbol=:coef, acf_lags::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    if view === :coef
        panel = _coef_panel("piv_coef", m.varnames, m.beta, _diag_se(m.vcov_mat);
                            ptitle="Coefficients ($(m.method), 95% CI)")
        ftitle = isempty(title) ? "Panel IV Coefficients" : title
        p = _make_plot([panel]; title=ftitle, ncols=1)
    elseif view === :diagnostics
        panels = _residual_diagnostics_panels(m.residuals, m.fitted; acf_lags=acf_lags)
        ftitle = isempty(title) ? "Panel IV Residual Diagnostics ($(m.method))" : title
        p = _make_plot(panels; title=ftitle, ncols=2)
    else
        throw(ArgumentError("Unknown view :$view. Valid views: :coef, :diagnostics"))
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(m::PMGModel; title="", save_path=nothing)

Horizontal coefficient plot of the PMG/MG long-run coefficients `θ ± 1.96·SE`
(`theta`/`theta_se` over `xnames`), with a zero reference line.
"""
function plot_result(m::PMGModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    panel = _coef_panel("pmg_coef", m.xnames, m.theta, m.theta_se;
                        xlabel="Long-run coefficient",
                        ptitle="Long-run Coefficients ($(m.method), 95% CI)")
    ftitle = isempty(title) ? "PMG Long-run Coefficients" : title
    p = _make_plot([panel]; title=ftitle, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# Ordered logit / probit
# =============================================================================

"""
    plot_result(m::OrderedModel; title="", save_path=nothing)

Two horizontal dot-and-whisker panels: the ordered-model slopes `β ± 1.96·SE` (SE
from the `β` block of `diag(vcov_mat)`), and the estimated `cutpoints ± 1.96·SE` (the
`α` block), each with a zero reference line. The cutpoints live on the latent-index
scale, so they get their own panel rather than sharing the coefficient axis.
"""
function plot_result(m::OrderedModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    K = length(m.beta)
    se_all = _diag_se(m.vcov_mat)
    p1 = _coef_panel("ord_coef", m.varnames, m.beta, se_all[1:K];
                     ptitle="Coefficients (95% CI)")
    cut_names = String["α$i" for i in 1:length(m.cutpoints)]
    p2 = _coef_panel("ord_cut", cut_names, m.cutpoints, se_all[K+1:end];
                     xlabel="Threshold", ptitle="Cutpoints (95% CI)")
    ftitle = isempty(title) ? "Ordered Model Coefficients" : title
    p = _make_plot([p1, p2]; title=ftitle, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# Multinomial logit — per-outcome facets
# =============================================================================

"""
    plot_result(m::MultinomialLogitModel; category=nothing, ncols=0, title="", save_path=nothing)

Per-outcome coefficient facets: one horizontal coefficient panel per non-base
category (columns of `β`, SE from the matching block of `diag(vcov_mat)`), titled
`"<cat> (vs <base>)"`. When `category` (Int index or label among the non-base
categories) is given, only that outcome's panel is drawn (plotrule C3).
"""
function plot_result(m::MultinomialLogitModel{T};
                     category::Union{Int,String,Nothing}=nothing, ncols::Int=0,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    K, Jm1 = size(m.beta)
    se_all = _diag_se(m.vcov_mat)
    base = string(m.categories[1])
    nonbase = String[string(m.categories[j + 1]) for j in 1:Jm1]
    cols = category === nothing ? collect(1:Jm1) : [_resolve_category(category, nonbase)]
    panels = _PanelSpec[]
    for j in cols
        se_j = se_all[(j - 1) * K + 1:(j - 1) * K + K]
        panel = _coef_panel("mlogit_coef", m.varnames, m.beta[:, j], se_j;
                            ptitle="$(nonbase[j]) (vs $base)")
        push!(panels, panel)
    end
    ftitle = isempty(title) ? "Multinomial Logit Coefficients" : title
    p = _make_plot(panels; title=ftitle, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# Multinomial marginal effects — per-outcome facets (dots-only when SE missing)
# =============================================================================

"""
    plot_result(me::MultinomialMarginalEffects; category=nothing, ncols=0, title="", save_path=nothing)

Per-outcome average-marginal-effect facets from `effects` (K × J, base = column 1),
one panel per non-base category. When `me.se === nothing` the whiskers collapse to
the point (no fabricated CI) and the subtitle says so (plotrule C6). `category`
selects one non-base outcome by Int index or label.
"""
function plot_result(me::MultinomialMarginalEffects{T};
                     category::Union{Int,String,Nothing}=nothing, ncols::Int=0,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    K, J = size(me.effects)
    nonbase = String[string(me.categories[j]) for j in 2:J]
    base = string(me.categories[1])
    has_se = me.se !== nothing
    sel = category === nothing ? collect(1:(J - 1)) : [_resolve_category(category, nonbase)]
    panels = _PanelSpec[]
    for c in sel
        j = c + 1                       # column into effects/se (skip base col 1)
        id = _next_plot_id("mme_coef")
        eff = me.effects[:, j]
        if has_se
            se_j = me.se[:, j]
            data_json = _coef_plot_json(me.varnames, eff, se_j)
            sub = ""
        else
            data_json = _coef_ci_json(me.varnames, eff, eff, eff)  # collapsed whisker
            sub = " — SE unavailable, points only"
        end
        js = _render_coef_plot_js(id, data_json; ref_value=0, xlabel="dy/dx", ylabel="")
        push!(panels, _PanelSpec(id, "$(nonbase[c]) (vs $base)$sub", js))
    end
    ftitle = isempty(title) ? "Multinomial Marginal Effects" : title
    p = _make_plot(panels; title=ftitle, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# Odds ratios — forest plot on a log x-axis (reference at 1)
# =============================================================================

"""
    plot_result(r::OddsRatio; title="", save_path=nothing)

Forest plot of logit odds ratios on a **log x-axis** with the reference line at **1**
(not 0): `or` with `[ci_lower, ci_upper]`, intercept omitted. Uses the coef renderer's
`logx=true` / `ref_value=1` options (plotrule A4).
"""
function plot_result(r::OddsRatio{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    id = _next_plot_id("odds_forest")
    data_json = _coef_ci_json(r.varnames, r.or, r.ci_lower, r.ci_upper)
    js = _render_coef_plot_js(id, data_json; logx=true, ref_value=1,
                              xlabel="Odds ratio (log scale)", ylabel="")
    ci_pct = round(Int, 100 * r.conf_level)
    panel = _PanelSpec(id, "Odds Ratios ($(ci_pct)% CI, ref = 1)", js)
    ftitle = isempty(title) ? "Odds Ratios" : title
    p = _make_plot([panel]; title=ftitle, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# Heckman selection — two-equation view
# =============================================================================

"""
    plot_result(m::HeckmanModel; view=:outcome, title="", save_path=nothing)

Coefficient plot for one equation of the Heckman selection model:
`view=:outcome` (default) draws the outcome equation (`beta`/`vcov_beta`/`outcome_names`),
`view=:selection` the selection probit (`gamma`/`vcov_gamma`/`select_names`). Unknown
`view` throws an `ArgumentError`.
"""
function plot_result(m::HeckmanModel{T};
                     view::Symbol=:outcome, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    if view === :outcome
        panel = _coef_panel("heck_out", m.outcome_names, m.beta, _diag_se(m.vcov_beta);
                            ptitle="Outcome Equation (95% CI)")
        ftitle = isempty(title) ? "Heckman — Outcome Equation" : title
    elseif view === :selection
        panel = _coef_panel("heck_sel", m.select_names, m.gamma, _diag_se(m.vcov_gamma);
                            ptitle="Selection Equation (95% CI)")
        ftitle = isempty(title) ? "Heckman — Selection Equation" : title
    else
        throw(ArgumentError("Unknown view :$view. Valid views: :outcome, :selection"))
    end
    p = _make_plot([panel]; title=ftitle, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# Tobit / truncated / cointegrating regression — single coefficient panels
# =============================================================================

"""
    plot_result(m::TobitModel; title="", save_path=nothing)

Horizontal coefficient plot of the Tobit slopes `β ± 1.96·SE`; intercept omitted.
"""
function plot_result(m::TobitModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    # vcov_mat is (k+1)×(k+1) with σ last — take the β block only.
    se_beta = _diag_se(m.vcov_mat)[1:length(m.beta)]
    panel = _coef_panel("tobit_coef", m.varnames, m.beta, se_beta;
                        ptitle="Coefficients (95% CI)")
    ftitle = isempty(title) ? "Tobit Coefficients" : title
    p = _make_plot([panel]; title=ftitle, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(m::TruncRegModel; title="", save_path=nothing)

Horizontal coefficient plot of the truncated-regression slopes `β ± 1.96·SE`.
"""
function plot_result(m::TruncRegModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    # vcov_mat is (k+1)×(k+1) with σ last — take the β block only.
    se_beta = _diag_se(m.vcov_mat)[1:length(m.beta)]
    panel = _coef_panel("trunc_coef", m.varnames, m.beta, se_beta;
                        ptitle="Coefficients (95% CI)")
    ftitle = isempty(title) ? "Truncated Regression Coefficients" : title
    p = _make_plot([panel]; title=ftitle, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(m::CointRegModel; title="", save_path=nothing)

Horizontal coefficient plot of the cointegrating-regression coefficients
`coef ± 1.96·SE` (SE from `diag(vcov)`); intercept/trend rows omitted.
"""
function plot_result(m::CointRegModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    panel = _coef_panel("cr_coef", m.varnames, m.coef, _diag_se(m.vcov);
                        ptitle="Cointegrating Coefficients ($(m.method), 95% CI)")
    ftitle = isempty(title) ? "Cointegrating Regression Coefficients" : title
    p = _make_plot([panel]; title=ftitle, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# SUR / 3SLS — one coefficient panel per equation
# =============================================================================

# Shared per-equation coef figure for the system estimators.
function _system_coef_plot(eqnames::Vector{String}, varnames::Vector{Vector{String}},
                           betas::Vector{Vector{T}}, ses::Vector{Vector{T}},
                           label::String, title::String,
                           save_path::Union{String,Nothing}, ncols::Int) where {T}
    panels = _PanelSpec[]
    for j in eachindex(eqnames)
        panel = _coef_panel("sys_coef", varnames[j], betas[j], ses[j];
                            ptitle="$(eqnames[j]) (95% CI)")
        push!(panels, panel)
    end
    ftitle = isempty(title) ? label : title
    p = _make_plot(panels; title=ftitle, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(m::SURModel; ncols=0, title="", save_path=nothing)

One horizontal coefficient panel per equation of a seemingly-unrelated-regression
system (`betas`/`ses`/`varnames` per equation); intercepts omitted.
"""
plot_result(m::SURModel{T}; ncols::Int=0, title::String="",
            save_path::Union{String,Nothing}=nothing) where {T} =
    _system_coef_plot(m.eqnames, m.varnames, m.betas, m.ses,
                      "SUR System Coefficients", title, save_path, ncols)

"""
    plot_result(m::ThreeSLSModel; ncols=0, title="", save_path=nothing)

One horizontal coefficient panel per equation of a three-stage-least-squares system.
"""
plot_result(m::ThreeSLSModel{T}; ncols::Int=0, title::String="",
            save_path::Union{String,Nothing}=nothing) where {T} =
    _system_coef_plot(m.eqnames, m.varnames, m.betas, m.ses,
                      "3SLS System Coefficients", title, save_path, ncols)
