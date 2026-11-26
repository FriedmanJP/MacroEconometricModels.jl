# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

"""
plot_result methods for model types: ARCH/GARCH/EGARCH/GJR/SV,
FactorModel, DynamicFactorModel, TimeSeriesData, PanelData.
"""

# =============================================================================
# Volatility Model Diagnostics (shared)
# =============================================================================

"""
Generate 3-panel volatility diagnostic Figure:
1. Returns
2. Conditional volatility (σₜ)
3. Standardized residuals (with ±2σ reference lines)
"""
function _plot_volatility_diagnostics(y::AbstractVector, cond_var::AbstractVector,
                                      model_name::String; title::String="",
                                      save_path::Union{String,Nothing}=nothing)
    data_json = _volatility_data_json(y, cond_var)

    # Panel 1: Returns
    id1 = _next_plot_id("vol_ret")
    s1 = _series_json(["Returns"], [_PLOT_COLORS[1]]; keys=["ret"])
    refs1 = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js1 = _render_line_js(id1, data_json, s1;
                          ref_lines_json=refs1, xlabel="", ylabel="Returns")
    p1 = _PanelSpec(id1, "Returns", js1)

    # Panel 2: Conditional volatility
    id2 = _next_plot_id("vol_sig")
    s2 = _series_json(["Cond. Volatility (σ)"], [_PLOT_COLORS[2]]; keys=["vol"])
    js2 = _render_line_js(id2, data_json, s2; xlabel="", ylabel="σₜ")
    p2 = _PanelSpec(id2, "Conditional Volatility", js2)

    # Panel 3: Standardized residuals
    id3 = _next_plot_id("vol_zr")
    s3 = _series_json(["Std. Residuals"], [_PLOT_COLORS[3]]; keys=["std_resid"])
    refs3 = "[{\"value\":2,\"color\":\"#d62728\",\"dash\":\"4,3\"},{\"value\":-2,\"color\":\"#d62728\",\"dash\":\"4,3\"},{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js3 = _render_line_js(id3, data_json, s3;
                          ref_lines_json=refs3, xlabel="Period",
                          ylabel="zₜ = εₜ/σₜ")
    p3 = _PanelSpec(id3, "Standardized Residuals", js3)

    if isempty(title)
        title = "$model_name — Diagnostic Plots"
    end

    p = _make_plot([p1, p2, p3]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# Volatility-model view helpers (lane-local, A5) — the standardized-residual
# diagnostics figure (PLT-24) and the news-impact curve (PLT-37). Both compose the
# frozen render/converter primitives; no renderer is defined here (A1).
# =============================================================================

# Standardized-residual four-panel diagnostics for a GARCH-family model (PLT-24). Every
# GARCH-family struct carries `standardized_residuals`; `fitted` is the mean equation
# (a constant μ for GARCH-MIDAS, which has no `fitted` field). Routed through the shared
# `_residual_diagnostics_panels` converter (A6).
function _vol_diag_plot(m, model_name::String; title::String="",
                        save_path::Union{String,Nothing}=nothing)
    sr = Float64[Float64(v) for v in m.standardized_residuals]
    ft = hasproperty(m, :fitted) ? Float64[Float64(v) for v in m.fitted] :
                                   fill(Float64(m.mu), length(sr))
    n = min(length(sr), length(ft))
    panels = _residual_diagnostics_panels(sr[end-n+1:end], ft[end-n+1:end]; standardized=true)
    isempty(title) && (title = "$model_name — Residual Diagnostics")
    p = _make_plot(panels; title=title, ncols=2)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# News-impact curve figure (PLT-37): the conditional variance σ²ₜ as a function of the
# previous shock εₜ₋₁, with a vertical reference line at ε=0. Asymmetric families
# (GJR/EGARCH/APARCH) show a steeper negative-shock branch (C6 — the docstring's
# "asymmetric parabola" becomes visible). `news_impact_curve` returns `(shocks, variance)`.
function _news_impact_plot(m, model_name::String; title::String="",
                           save_path::Union{String,Nothing}=nothing)
    nic = news_impact_curve(m)
    id = _next_plot_id("nic")
    rows = Vector{Pair{String,String}}[]
    for i in eachindex(nic.shocks)
        push!(rows, ["x" => _json(nic.shocks[i]), "niv" => _json(nic.variance[i])])
    end
    data = _json_array_of_objects(rows)
    s = _series_json(["News impact σ²"], [_PLOT_COLORS[1]]; keys=["niv"])
    refs = "[{\"value\":0,\"color\":\"$(_PLOT_ALERT)\",\"dash\":\"5,3\",\"axis\":\"x\"}]"
    js = _render_line_js(id, data, s; ref_lines_json=refs,
                         xlabel="Shock εₜ₋₁", ylabel="Conditional variance σ²ₜ")
    isempty(title) && (title = "$model_name — News Impact Curve")
    p = _make_plot([_PanelSpec(id, title, js)]; title=title)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# ARCHModel
# =============================================================================

"""
    plot_result(m::ARCHModel; view=:default, title="", save_path=nothing)

Plot ARCH model diagnostics.

- `view=:default` — returns, conditional volatility, standardized residuals (3 panels).
- `view=:diagnostics` — the shared four-panel residual diagnostics on the standardized
  residuals (PLT-24).
"""
function plot_result(m::ARCHModel{T}; view::Symbol=:default,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    if view === :default
        return _plot_volatility_diagnostics(m.y, m.conditional_variance, "ARCH($(m.q))";
                                            title=title, save_path=save_path)
    elseif view === :diagnostics
        return _vol_diag_plot(m, "ARCH($(m.q))"; title=title, save_path=save_path)
    end
    throw(ArgumentError("unknown view :$view for ARCHModel — use :default or :diagnostics"))
end

# =============================================================================
# GARCHModel
# =============================================================================

"""
    plot_result(m::GARCHModel; view=:default, title="", save_path=nothing)

Plot GARCH model diagnostics.

- `view=:default` — returns, conditional volatility, standardized residuals.
- `view=:diagnostics` — four-panel standardized-residual diagnostics (PLT-24).
- `view=:news_impact` — the news-impact curve σ²ₜ(εₜ₋₁) with a vline at ε=0 (PLT-37).
"""
function plot_result(m::GARCHModel{T}; view::Symbol=:default,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    if view === :default
        return _plot_volatility_diagnostics(m.y, m.conditional_variance, "GARCH($(m.p),$(m.q))";
                                            title=title, save_path=save_path)
    elseif view === :diagnostics
        return _vol_diag_plot(m, "GARCH($(m.p),$(m.q))"; title=title, save_path=save_path)
    elseif view === :news_impact
        return _news_impact_plot(m, "GARCH($(m.p),$(m.q))"; title=title, save_path=save_path)
    end
    throw(ArgumentError("unknown view :$view for GARCHModel — use :default, :diagnostics, or :news_impact"))
end

# =============================================================================
# EGARCHModel
# =============================================================================

"""
    plot_result(m::EGARCHModel; view=:default, title="", save_path=nothing)

Plot EGARCH model diagnostics. `view` ∈ (`:default`, `:diagnostics`, `:news_impact`) —
the asymmetric news-impact curve (steeper negative-shock branch) is drawn by
`view=:news_impact` (PLT-24/PLT-37).
"""
function plot_result(m::EGARCHModel{T}; view::Symbol=:default,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    if view === :default
        return _plot_volatility_diagnostics(m.y, m.conditional_variance, "EGARCH($(m.p),$(m.q))";
                                            title=title, save_path=save_path)
    elseif view === :diagnostics
        return _vol_diag_plot(m, "EGARCH($(m.p),$(m.q))"; title=title, save_path=save_path)
    elseif view === :news_impact
        return _news_impact_plot(m, "EGARCH($(m.p),$(m.q))"; title=title, save_path=save_path)
    end
    throw(ArgumentError("unknown view :$view for EGARCHModel — use :default, :diagnostics, or :news_impact"))
end

# =============================================================================
# GJRGARCHModel
# =============================================================================

"""
    plot_result(m::GJRGARCHModel; view=:default, title="", save_path=nothing)

Plot GJR-GARCH model diagnostics. `view` ∈ (`:default`, `:diagnostics`, `:news_impact`)
— the leverage asymmetry is visible in `view=:news_impact` (PLT-24/PLT-37).
"""
function plot_result(m::GJRGARCHModel{T}; view::Symbol=:default,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    if view === :default
        return _plot_volatility_diagnostics(m.y, m.conditional_variance, "GJR-GARCH($(m.p),$(m.q))";
                                            title=title, save_path=save_path)
    elseif view === :diagnostics
        return _vol_diag_plot(m, "GJR-GARCH($(m.p),$(m.q))"; title=title, save_path=save_path)
    elseif view === :news_impact
        return _news_impact_plot(m, "GJR-GARCH($(m.p),$(m.q))"; title=title, save_path=save_path)
    end
    throw(ArgumentError("unknown view :$view for GJRGARCHModel — use :default, :diagnostics, or :news_impact"))
end

# =============================================================================
# FIGARCHModel / FIEGARCHModel (EV-14, #422)
# =============================================================================

"""
    plot_result(m::FIGARCHModel; view=:default, title="", save_path=nothing)

Plot FIGARCH conditional-volatility diagnostics. `view` ∈ (`:default`, `:diagnostics`).
"""
function plot_result(m::FIGARCHModel{T}; view::Symbol=:default,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    if view === :default
        return _plot_volatility_diagnostics(m.y, m.conditional_variance, "FIGARCH($(m.p),d,$(m.q))";
                                            title=title, save_path=save_path)
    elseif view === :diagnostics
        return _vol_diag_plot(m, "FIGARCH($(m.p),d,$(m.q))"; title=title, save_path=save_path)
    end
    throw(ArgumentError("unknown view :$view for FIGARCHModel — use :default or :diagnostics"))
end

"""
    plot_result(m::FIEGARCHModel; view=:default, title="", save_path=nothing)

Plot FIEGARCH conditional-volatility diagnostics. `view` ∈ (`:default`, `:diagnostics`).
"""
function plot_result(m::FIEGARCHModel{T}; view::Symbol=:default,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    if view === :default
        return _plot_volatility_diagnostics(m.y, m.conditional_variance, "FIEGARCH($(m.p),d,$(m.q))";
                                            title=title, save_path=save_path)
    elseif view === :diagnostics
        return _vol_diag_plot(m, "FIEGARCH($(m.p),d,$(m.q))"; title=title, save_path=save_path)
    end
    throw(ArgumentError("unknown view :$view for FIEGARCHModel — use :default or :diagnostics"))
end

# =============================================================================
# IGARCHModel / APARCHModel / CGARCHModel (EV-15, #423)
# =============================================================================

"""
    plot_result(m::IGARCHModel; view=:default, title="", save_path=nothing)

Plot IGARCH conditional-volatility diagnostics. `view` ∈ (`:default`, `:diagnostics`,
`:news_impact`) (PLT-24/PLT-37).
"""
function plot_result(m::IGARCHModel{T}; view::Symbol=:default,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    if view === :default
        return _plot_volatility_diagnostics(m.y, m.conditional_variance, "IGARCH($(m.p),$(m.q))";
                                            title=title, save_path=save_path)
    elseif view === :diagnostics
        return _vol_diag_plot(m, "IGARCH($(m.p),$(m.q))"; title=title, save_path=save_path)
    elseif view === :news_impact
        return _news_impact_plot(m, "IGARCH($(m.p),$(m.q))"; title=title, save_path=save_path)
    end
    throw(ArgumentError("unknown view :$view for IGARCHModel — use :default, :diagnostics, or :news_impact"))
end

"""
    plot_result(m::APARCHModel; view=:default, title="", save_path=nothing)

Plot APARCH conditional-volatility diagnostics. `view` ∈ (`:default`, `:diagnostics`,
`:news_impact`) — the power/asymmetry shape is visible in `view=:news_impact`
(PLT-24/PLT-37).
"""
function plot_result(m::APARCHModel{T}; view::Symbol=:default,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    if view === :default
        return _plot_volatility_diagnostics(m.y, m.conditional_variance, "APARCH($(m.p),$(m.q))";
                                            title=title, save_path=save_path)
    elseif view === :diagnostics
        return _vol_diag_plot(m, "APARCH($(m.p),$(m.q))"; title=title, save_path=save_path)
    elseif view === :news_impact
        return _news_impact_plot(m, "APARCH($(m.p),$(m.q))"; title=title, save_path=save_path)
    end
    throw(ArgumentError("unknown view :$view for APARCHModel — use :default, :diagnostics, or :news_impact"))
end

"""
    plot_result(m::CGARCHModel; view=:components, title="", save_path=nothing)

Plot a Component-GARCH(1,1) fit.

- `view=:components` — stacked-area decomposition of the permanent (`√q`) and
  transitory (`√max(σ²−q,0)`) volatility contributions over the sample.
- `view=:default` — the standard 3-panel volatility diagnostic figure.
- `view=:diagnostics` — four-panel standardized-residual diagnostics (PLT-24).
- `view=:news_impact` — the news-impact curve σ²ₜ(εₜ₋₁) (PLT-37).
"""
function plot_result(m::CGARCHModel{T}; view::Symbol=:components,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    if view === :components
        p = _cgarch_components_plot(m, title)
    elseif view === :default
        return _plot_volatility_diagnostics(m.y, m.conditional_variance, "Component-GARCH(1,1)";
                                             title=title, save_path=save_path)
    elseif view === :diagnostics
        return _vol_diag_plot(m, "Component-GARCH(1,1)"; title=title, save_path=save_path)
    elseif view === :news_impact
        return _news_impact_plot(m, "Component-GARCH(1,1)"; title=title, save_path=save_path)
    else
        throw(ArgumentError("unknown view $view — use :components, :default, :diagnostics, or :news_impact"))
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

function _cgarch_components_plot(m::CGARCHModel{T}, title::String) where {T}
    id = _next_plot_id("cg_comp")
    n = length(m.conditional_variance)
    rows = Vector{Pair{String,String}}[]
    for i in 1:n
        perm = sqrt(max(m.permanent[i], zero(T)))
        trans = sqrt(max(m.transitory[i], zero(T)))
        push!(rows, ["x" => _json(i), "permanent" => _json(perm), "transitory" => _json(trans)])
    end
    data_json = _json_array_of_objects(rows)
    s_json = _series_json(["Permanent √q", "Transitory √(σ²−q)"],
                          [_PLOT_COLORS[1], _PLOT_COLORS[2]]; keys=["permanent", "transitory"])
    js = _render_area_js(id, data_json, s_json;
                         xlabel="Observation", ylabel="Volatility contribution")
    isempty(title) && (title = "Component-GARCH Volatility Decomposition")
    _make_plot([_PanelSpec(id, title, js)]; title=title)
end

# =============================================================================
# GarchMidasModel (EV-02, #410)
# =============================================================================

"""
    plot_result(m::GarchMidasModel; view=:components, title="", save_path=nothing)

Plot a GARCH-MIDAS fit.

- `view=:components` — overlay total conditional volatility `√σ²` and the
  long-run component `√τ` over the retained sample.
- `view=:weights` — the fitted Beta MIDAS weight curve `φ_k` versus lag `k`.
- `view=:diagnostics` — four-panel standardized-residual diagnostics (PLT-24).
"""
function plot_result(m::GarchMidasModel{T}; view::Symbol=:components,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    if view === :components
        p = _garch_midas_components_plot(m, title)
    elseif view === :weights
        p = _garch_midas_weight_plot(m, title)
    elseif view === :diagnostics
        return _vol_diag_plot(m, "GARCH-MIDAS"; title=title, save_path=save_path)
    else
        throw(ArgumentError("unknown view $view — use :components, :weights, or :diagnostics"))
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

function _garch_midas_components_plot(m::GarchMidasModel{T}, title::String) where {T}
    id = _next_plot_id("gm_comp")
    n = length(m.conditional_variance)
    rows = Vector{Pair{String,String}}[]
    for i in 1:n
        push!(rows, [
            "x" => _json(i),
            "total" => _json(sqrt(max(m.conditional_variance[i], zero(T)))),
            "longrun" => _json(sqrt(max(m.tau[i], zero(T)))),
        ])
    end
    data_json = _json_array_of_objects(rows)
    s_json = _series_json(["Total √σ²", "Long-run √τ"],
                          [_PLOT_COLORS[1], _PLOT_COLORS[2]];
                          keys=["total", "longrun"], dash=["", "6,3"])
    js = _render_line_js(id, data_json, s_json;
                         xlabel="Retained HF observation", ylabel="Volatility")
    isempty(title) && (title = "GARCH-MIDAS Volatility Components")
    _make_plot([_PanelSpec(id, title, js)]; title=title)
end

function _garch_midas_weight_plot(m::GarchMidasModel{T}, title::String) where {T}
    id = _next_plot_id("gm_w")
    rows = Vector{Pair{String,String}}[]
    for k in 1:m.K
        push!(rows, ["x" => _json(k), "w" => _json(m.weights[k]), "zero" => "0"])
    end
    data_json = _json_array_of_objects(rows)
    s_json = _series_json(["Weight φₖ"], [_PLOT_COLORS[1]]; keys=["w"])
    refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js = _render_line_js(id, data_json, s_json;
                         ref_lines_json=refs, xlabel="MIDAS lag k", ylabel="Weight")
    isempty(title) && (title = "GARCH-MIDAS Beta Weights (K=$(m.K))")
    _make_plot([_PanelSpec(id, title, js)]; title=title)
end

# =============================================================================
# SVModel
# =============================================================================

"""
    plot_result(m::SVModel; view=:default, title="", save_path=nothing)

Plot SV model diagnostics.

- `view=:default` — returns, posterior volatility with credible band, standardized
  residuals (3 panels).
- `view=:diagnostics` — four-panel residual diagnostics on the already-standardized
  residuals `zₜ = yₜ/√σ̄²ₜ` (PLT-24).
"""
function plot_result(m::SVModel{T}; view::Symbol=:default,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    if view === :diagnostics
        sr = Float64[Float64(m.y[i]) / sqrt(max(Float64(m.volatility_mean[i]), 1e-12))
                     for i in eachindex(m.y)]
        ft = zeros(Float64, length(sr))   # SV assumes a zero conditional mean
        panels = _residual_diagnostics_panels(sr, ft; standardized=true)
        isempty(title) && (title = "Stochastic Volatility — Residual Diagnostics")
        p = _make_plot(panels; title=title, ncols=2)
        save_path !== nothing && save_plot(p, save_path)
        return p
    elseif view !== :default
        throw(ArgumentError("unknown view :$view for SVModel — use :default or :diagnostics"))
    end
    data_json = _sv_data_json(m.y, m.volatility_mean, m.volatility_quantiles,
                               m.quantile_levels)
    nq = length(m.quantile_levels)

    # Panel 1: Returns
    id1 = _next_plot_id("sv_ret")
    s1 = _series_json(["Returns"], [_PLOT_COLORS[1]]; keys=["ret"])
    refs1 = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js1 = _render_line_js(id1, data_json, s1;
                          ref_lines_json=refs1, xlabel="", ylabel="Returns")
    p1 = _PanelSpec(id1, "Returns", js1)

    # Panel 2: Posterior volatility with CI band
    id2 = _next_plot_id("sv_vol")
    s2 = _series_json(["Posterior mean σ"], [_PLOT_COLORS[2]]; keys=["vol_mean"])
    bands2 = nq >= 2 ?
        "[{\"lo_key\":\"q1\",\"hi_key\":\"q$(nq)\",\"color\":\"$(_PLOT_COLORS[2])\",\"alpha\":$(_PLOT_CI_ALPHA)}]" : "[]"
    js2 = _render_line_js(id2, data_json, s2;
                          bands_json=bands2, xlabel="", ylabel="σₜ")
    p2 = _PanelSpec(id2, "Stochastic Volatility", js2)

    # Panel 3: Standardized residuals
    id3 = _next_plot_id("sv_zr")
    s3 = _series_json(["Std. Residuals"], [_PLOT_COLORS[3]]; keys=["std_resid"])
    refs3 = "[{\"value\":2,\"color\":\"#d62728\",\"dash\":\"4,3\"},{\"value\":-2,\"color\":\"#d62728\",\"dash\":\"4,3\"},{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js3 = _render_line_js(id3, data_json, s3;
                          ref_lines_json=refs3, xlabel="Period", ylabel="zₜ")
    p3 = _PanelSpec(id3, "Standardized Residuals", js3)

    if isempty(title)
        title = "Stochastic Volatility Model — Diagnostic Plots"
    end

    p = _make_plot([p1, p2, p3]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# FactorModel
# =============================================================================

"""
    plot_result(fm::FactorModel; title="", save_path=nothing)

Plot factor model: scree plot (eigenvalues) + extracted factor series.

Caps (raise to show more; the drawn count appears in the panel title, plotrule C7):
`max_factors=5` factor series, `max_eig=10` scree eigenvalues.
"""
function plot_result(fm::FactorModel{T};
                     max_factors::Int=5, max_eig::Int=10,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    # Panel 1: Scree plot (eigenvalues as bar chart)
    id1 = _next_plot_id("fm_scree")
    n_e_total = length(fm.eigenvalues)
    n_eig = min(n_e_total, max_eig)
    rows1 = Vector{Pair{String,String}}[]
    for i in 1:n_eig
        push!(rows1, ["x" => _json("PC $i"), "eig" => _json(fm.eigenvalues[i])])
    end
    data1 = _json_array_of_objects(rows1)
    s1 = _series_json(["Eigenvalue"], [_PLOT_COLORS[1]]; keys=["eig"])
    js1 = _render_bar_js(id1, data1, s1; mode="grouped", ylabel="Eigenvalue")
    p1 = _PanelSpec(id1, _cap_title("Scree Plot", n_eig, n_e_total), js1)

    # Panel 2: Factor series
    id2 = _next_plot_id("fm_fac")
    T_obs, r = size(fm.factors)
    n_plot = min(r, max_factors)
    fac_names = ["Factor $i" for i in 1:n_plot]
    fac_colors = _palette_take(n_plot)
    data2 = _timeseries_data_json(fm.factors[:, 1:n_plot], fac_names)
    s2 = _series_json(fac_names, fac_colors; keys=["v$i" for i in 1:n_plot])
    js2 = _render_line_js(id2, data2, s2; xlabel="Period", ylabel="Factor Value")
    p2 = _PanelSpec(id2, _cap_title("Extracted Factors", n_plot, r), js2)

    if isempty(title)
        title = "Static Factor Model (r=$(fm.r))"
    end

    p = _make_plot([p1, p2]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# DynamicFactorModel
# =============================================================================

"""
    plot_result(fm::DynamicFactorModel; title="", save_path=nothing)

Plot dynamic factor model: scree + factor series.

Caps (raise to show more; the drawn count appears in the panel title, plotrule C7):
`max_factors=5` factor series, `max_eig=10` scree eigenvalues.
"""
function plot_result(fm::DynamicFactorModel{T};
                     max_factors::Int=5, max_eig::Int=10,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    # Panel 1: Scree
    id1 = _next_plot_id("dfm_scree")
    n_e_total = length(fm.eigenvalues)
    n_eig = min(n_e_total, max_eig)
    rows1 = Vector{Pair{String,String}}[]
    for i in 1:n_eig
        push!(rows1, ["x" => _json("PC $i"), "eig" => _json(fm.eigenvalues[i])])
    end
    data1 = _json_array_of_objects(rows1)
    s1 = _series_json(["Eigenvalue"], [_PLOT_COLORS[1]]; keys=["eig"])
    js1 = _render_bar_js(id1, data1, s1; mode="grouped", ylabel="Eigenvalue")
    p1 = _PanelSpec(id1, _cap_title("Scree Plot", n_eig, n_e_total), js1)

    # Panel 2: Factor series
    id2 = _next_plot_id("dfm_fac")
    T_obs, r = size(fm.factors)
    n_plot = min(r, max_factors)
    fac_names = ["Factor $i" for i in 1:n_plot]
    data2 = _timeseries_data_json(fm.factors[:, 1:n_plot], fac_names)
    s2 = _series_json(fac_names, _palette_take(n_plot); keys=["v$i" for i in 1:n_plot])
    js2 = _render_line_js(id2, data2, s2; xlabel="Period", ylabel="Factor Value")
    p2 = _PanelSpec(id2, _cap_title("Extracted Factors (VAR($(fm.p)))", n_plot, r), js2)

    if isempty(title)
        title = "Dynamic Factor Model (r=$(fm.r), p=$(fm.p))"
    end

    p = _make_plot([p1, p2]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# NOTE: plot_result(::TimeSeriesData) relocated to plotting/timeseries.jl and
# plot_result(::PanelData) relocated to plotting/panel.jl (PLT plotting overhaul,
# Wave-2 data-view lanes). See those files.

# =============================================================================
# OccBinIRF
# Convenience constructors for OccBinIRF and OccBinSolution (mock & testing ergonomics)
function OccBinIRF{T}(linear::AbstractMatrix, piecewise::AbstractMatrix,
                      regime_history::AbstractVector, varnames::Vector{String},
                      shock_name::String) where {T<:AbstractFloat}
    rh = reshape(Int.(regime_history), :, 1)
    return OccBinIRF{T}(Matrix{T}(linear), Matrix{T}(piecewise), rh,
                        varnames, shock_name)
end

function OccBinIRF{T}(linear::AbstractMatrix, piecewise::AbstractMatrix,
                      regime_history::AbstractVecOrMat,
                      varnames::Vector{String}) where {T<:AbstractFloat}
    rh = regime_history isa AbstractVector ?
        reshape(Int.(regime_history), :, 1) : Matrix{Int}(regime_history)
    return OccBinIRF{T}(Matrix{T}(linear), Matrix{T}(piecewise), rh,
                        varnames, "shock")
end

function OccBinIRF(linear::AbstractMatrix{T}, piecewise::AbstractMatrix{T},
                   regime_history::AbstractVecOrMat,
                   varnames::Vector{String}) where {T<:AbstractFloat}
    return OccBinIRF{T}(linear, piecewise, regime_history, varnames)
end

function OccBinIRF(linear::AbstractMatrix{T}, piecewise::AbstractMatrix{T},
                   regime_history::AbstractVector, varnames::Vector{String},
                   shock_name::String) where {T<:AbstractFloat}
    return OccBinIRF{T}(linear, piecewise, regime_history, varnames, shock_name)
end

function OccBinSolution{T}(linear_path::AbstractMatrix, piecewise_path::AbstractMatrix,
                           regime_history::AbstractVecOrMat,
                           varnames::Vector{String}) where {T<:AbstractFloat}
    rh = regime_history isa AbstractVector ?
        reshape(Int.(regime_history), :, 1) : Matrix{Int}(regime_history)
    n_vars = length(varnames)
    syms = Symbol[Symbol(v) for v in varnames]
    spec = ModelSpec{T,NoAgents}(
        syms, Symbol[], Symbol[], Dict{Symbol,T}(),
        n_vars, 0, 0,
        NamedEquation[], Function[],
        0, Int[],
        syms, NamedEquation[],
        n_vars, 0,
        false, 0, 0, false,
        nothing, zeros(T, n_vars), varnames,
        nothing, nothing, nothing, Symbol[],
        NamedTuple(),
        ModelIR(),
    )
    return OccBinSolution{T}(
        Matrix{T}(linear_path), Matrix{T}(piecewise_path), zeros(T, n_vars),
        rh, true, 1, spec, varnames,
        OccBinConstraint{T}[],
    )
end

function OccBinSolution(linear_path::AbstractMatrix{T}, piecewise_path::AbstractMatrix{T},
                        regime_history::AbstractVecOrMat,
                        varnames::Vector{String}) where {T<:AbstractFloat}
    return OccBinSolution{T}(linear_path, piecewise_path, regime_history, varnames)
end

"""
    plot_result(oirf::OccBinIRF; var=nothing, ncols=0, title="", save_path=nothing)

Plot OccBin IRF comparison: linear (dashed) vs piecewise-linear (solid) with
shaded binding-period rectangles for each variable.
"""
function plot_result(oirf::OccBinIRF{T};
                     var::Union{Int,String,AbstractVector,Nothing}=nothing,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    H, n_vars = size(oirf.piecewise)
    vars_to_plot = if var === nothing
        collect(1:n_vars)
    elseif var isa AbstractVector
        Int[_resolve_var(v, oirf.varnames) for v in var]
    else
        [_resolve_var(var, oirf.varnames)]
    end

    panels = _PanelSpec[]
    for j in vars_to_plot
        id = _next_plot_id("oirf")
        ptitle = oirf.varnames[j]

        # Build data JSON: {h, lin, pw, bind}
        binding = vec(any(oirf.regime_history .> 0; dims=2))
        rows = Vector{Pair{String,String}}[]
        for h in 1:H
            push!(rows, [
                "h" => _json(h),
                "lin" => _json(oirf.linear[h, j]),
                "pw" => _json(oirf.piecewise[h, j]),
                "bind" => _json(binding[h] ? 1 : 0),
            ])
        end
        data_json = _json_array_of_objects(rows)

        js = _render_occbin_panel_js(id, data_json;
                                     xlabel="Horizon", ylabel=oirf.varnames[j])
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        title = "OccBin IRF — Shock: $(oirf.shock_name)"
    end
    if ncols <= 0
        ncols = length(vars_to_plot) == 2 ? 2 : min(length(vars_to_plot), 3)
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    return p
end

# =============================================================================
# OccBinSolution
# =============================================================================

"""
    plot_result(sol::OccBinSolution; var=nothing, ncols=0, title="", save_path=nothing)

Plot OccBin solution comparison: linear (dashed) vs piecewise-linear (solid) with
shaded binding-period rectangles for each variable.
"""
function plot_result(sol::OccBinSolution{T};
                     var::Union{Int,String,AbstractVector,Nothing}=nothing,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    nperiods, n_vars = size(sol.piecewise_path)
    vars_to_plot = if var === nothing
        collect(1:n_vars)
    elseif var isa AbstractVector
        Int[_resolve_var(v, sol.varnames) for v in var]
    else
        [_resolve_var(var, sol.varnames)]
    end

    panels = _PanelSpec[]
    for j in vars_to_plot
        id = _next_plot_id("osol")
        ptitle = sol.varnames[j]

        # Build data JSON: {h, lin, pw, bind}
        binding = vec(any(sol.regime_history .> 0; dims=2))
        rows = Vector{Pair{String,String}}[]
        for h in 1:nperiods
            push!(rows, [
                "h" => _json(h),
                "lin" => _json(sol.linear_path[h, j]),
                "pw" => _json(sol.piecewise_path[h, j]),
                "bind" => _json(binding[h] ? 1 : 0),
            ])
        end
        data_json = _json_array_of_objects(rows)

        js = _render_occbin_panel_js(id, data_json;
                                     xlabel="Period", ylabel=sol.varnames[j])
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        title = "OccBin Piecewise-Linear Solution"
    end
    if ncols <= 0
        ncols = length(vars_to_plot) == 2 ? 2 : min(length(vars_to_plot), 3)
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    return p
end

# NOTE: plot_result(::BayesianDSGE) relocated to plotting/mcmc.jl (PLT plotting
# overhaul, Wave-2 MCMC lane; PLT-05 posterior-mean vline applied there).

# =============================================================================
# FAVARModel
# =============================================================================

"""
    plot_result(m::FAVARModel; title="", save_path=nothing)

Plot FAVAR model: extracted factor series.

Caps `max_factors=5` factor series; the drawn count appears in the panel title when
truncated (plotrule C7).
"""
function plot_result(m::FAVARModel{T};
                     max_factors::Int=5,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    r = m.n_factors
    n_plot = min(r, max_factors)
    fac_names = ["Factor $i" for i in 1:n_plot]
    fac_colors = _palette_take(n_plot)
    data2 = _timeseries_data_json(m.factors[:, 1:n_plot], fac_names)
    s2 = _series_json(fac_names, fac_colors; keys=["v$i" for i in 1:n_plot])

    id = _next_plot_id("favar_fac")
    js = _render_line_js(id, data2, s2; xlabel="Period", ylabel="Factor Value")
    p1 = _PanelSpec(id, _cap_title("Extracted Factors", n_plot, r), js)

    if isempty(title)
        title = "FAVAR Model (r=$(m.n_factors), p=$(m.p))"
    end

    p = _make_plot([p1]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# BayesianFAVAR
# =============================================================================

"""
    plot_result(m::BayesianFAVAR; title="", save_path=nothing)

Plot Bayesian FAVAR: posterior mean factor series with 68% credible intervals.

Caps `max_factors=5` factor panels; a figure note reports any truncation (C7).
"""
function plot_result(m::BayesianFAVAR{T};
                     max_factors::Int=5,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    r = m.n_factors
    n_plot = min(r, max_factors)
    F_mean = dropdims(mean(m.factor_draws, dims=1), dims=1)  # T_obs x r
    F_lo = dropdims(mapslices(x -> quantile(x, 0.16), m.factor_draws; dims=1), dims=1)
    F_hi = dropdims(mapslices(x -> quantile(x, 0.84), m.factor_draws; dims=1), dims=1)

    panels = _PanelSpec[]
    for j in 1:n_plot
        id = _next_plot_id("bfavar_f")
        fname = "Factor $j"
        T_obs = size(F_mean, 1)

        # Build data with mean + lo + hi
        rows = Vector{Pair{String,String}}[]
        for t in 1:T_obs
            push!(rows, [
                "x" => _json(t),
                "mean" => _json(F_mean[t, j]),
                "lo" => _json(F_lo[t, j]),
                "hi" => _json(F_hi[t, j])
            ])
        end
        data_json = _json_array_of_objects(rows)

        s_json = _series_json([fname], [_PLOT_COLORS[j]]; keys=["mean"])
        bands_json = "[{\"lo_key\":\"lo\",\"hi_key\":\"hi\",\"color\":\"$(_PLOT_COLORS[j])\",\"alpha\":$(_PLOT_CI_ALPHA)}]"
        js = _render_line_js(id, data_json, s_json;
                             bands_json=bands_json, xlabel="Period", ylabel="")
        push!(panels, _PanelSpec(id, "$fname (68% CI)", js))
    end

    if isempty(title)
        title = "Bayesian FAVAR (r=$(m.n_factors), p=$(m.p))"
    end

    p = _make_plot(panels; title=title, ncols=min(n_plot, 2),
                   note=_cap_note("factors", n_plot, r, "max_factors"))
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# StructuralDFM
# =============================================================================

const _SDFM_PLOT_VIEWS = (:irf, :impact, :fevd, :factors, :shocks, :loadings)

"""
    plot_result(m::StructuralDFM; view=:irf, var=nothing, ncols=0, title="", save_path=nothing)

Plot a fitted structural DFM.

# Views
- `:irf` — panel structural IRFs (default)
- `:impact` — impact heatmap (observables × shocks at horizon 0)
- `:fevd` — panel FEVD stacked areas
- `:factors` — estimated factor series
- `:shocks` — structural shock series
- `:loadings` — loading heatmap (variables × factors)
"""
function plot_result(m::StructuralDFM{T};
                     view::Symbol=:irf,
                     var::Union{Nothing,Int,String}=nothing,
                     ncols::Int=0,
                     max_vars::Int=12,
                     max_factors::Int=5,
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    view in _SDFM_PLOT_VIEWS || throw(ArgumentError(
        "Unknown view: :$view. Use $(_SDFM_PLOT_VIEWS)"))
    p = if view === :irf
        _plot_sdfm_irf(m; var=var, ncols=ncols, title=title)
    elseif view === :impact
        _plot_sdfm_impact(m; max_vars=max_vars, title=title)
    elseif view === :fevd
        _plot_sdfm_fevd(m; var=var, ncols=ncols, title=title)
    elseif view === :factors
        _plot_sdfm_factors(m; max_factors=max_factors, title=title)
    elseif view === :shocks
        _plot_sdfm_shocks(m; title=title)
    else
        _plot_sdfm_loadings(m; max_vars=max_vars, title=title)
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

function _plot_sdfm_irf(m::StructuralDFM; var, ncols::Int, title::String)
    H = size(m.structural_irf, 1)
    H < 1 && (H = 40)
    r = irf(m, H)
    plot_result(r; title=isempty(title) ? "Structural DFM IRFs" : title,
                var=var, ncols=ncols)
end

function _plot_sdfm_impact(m::StructuralDFM{T}; max_vars::Int, title::String) where {T}
    H = size(m.structural_irf, 1)
    H < 1 && (H = 1)
    impact = irf(m, max(H, 1)).values[1, :, :]   # N × q
    N, q = size(impact)
    shown = min(N, max_vars)
    rows = String[m.varnames[i] for i in 1:shown]
    cols = copy(m.shock_names)
    id = _next_plot_id("sdfm_imp")
    data_json = _matrix_heatmap_json(impact[1:shown, :], rows, cols)
    js = _render_heatmap_js(id, data_json, _json(rows), _json(cols);
                            scale=:diverging, midpoint=0,
                            xlabel="Shock", ylabel="Variable", tip_label="Impact")
    ptitle = _cap_title("Impact (h = 0)", shown, N)
    fig = isempty(title) ? "Structural DFM impact" : title
    _make_plot([_PanelSpec(id, ptitle, js)]; title=fig, ncols=1,
               note=_cap_note("variables", shown, N, "max_vars"))
end

function _plot_sdfm_fevd(m::StructuralDFM; var, ncols::Int, title::String)
    H = max(size(m.structural_irf, 1), 8)
    f = fevd(m, H; space=:panel)
    plot_result(f; var=var, ncols=ncols,
                title=isempty(title) ? "Structural DFM FEVD" : title)
end

function _plot_sdfm_factors(m::StructuralDFM; max_factors::Int, title::String)
    F = factors(m)
    T_obs, r = size(F)
    n_plot = min(r, max_factors)
    names = ["Factor $i" for i in 1:n_plot]
    id = _next_plot_id("sdfm_fac")
    data = _timeseries_data_json(F[:, 1:n_plot], names)
    s = _series_json(names, _palette_take(n_plot); keys=["v$i" for i in 1:n_plot])
    js = _render_line_js(id, data, s; xlabel="Period", ylabel="Factor")
    ptitle = _cap_title("Factors", n_plot, r)
    fig = isempty(title) ? "Structural DFM factors" : title
    _make_plot([_PanelSpec(id, ptitle, js)]; title=fig, ncols=1,
               note=_cap_note("factors", n_plot, r, "max_factors"))
end

function _plot_sdfm_shocks(m::StructuralDFM; title::String)
    ε = structural_shocks(m)
    names = copy(m.shock_names)
    q = length(names)
    id = _next_plot_id("sdfm_shk")
    data = _timeseries_data_json(ε, names)
    s = _series_json(names, _palette_take(q); keys=["v$i" for i in 1:q])
    refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js = _render_line_js(id, data, s; xlabel="Period", ylabel="Shock",
                         ref_lines_json=refs)
    fig = isempty(title) ? "Structural DFM shocks" : title
    _make_plot([_PanelSpec(id, "Structural shocks", js)]; title=fig, ncols=1)
end

function _plot_sdfm_loadings(m::StructuralDFM; max_vars::Int, title::String)
    Λ = loadings(m)
    N, r = size(Λ)
    shown = min(N, max_vars)
    rows = String[m.varnames[i] for i in 1:shown]
    cols = ["Factor $j" for j in 1:r]
    id = _next_plot_id("sdfm_load")
    data_json = _matrix_heatmap_json(Λ[1:shown, :], rows, cols)
    js = _render_heatmap_js(id, data_json, _json(rows), _json(cols);
                            scale=:diverging, midpoint=0,
                            xlabel="Factor", ylabel="Variable", tip_label="Loading")
    ptitle = _cap_title("Loadings", shown, N)
    fig = isempty(title) ? "Structural DFM loadings" : title
    _make_plot([_PanelSpec(id, ptitle, js)]; title=fig, ncols=1,
               note=_cap_note("variables", shown, N, "max_vars"))
end

# =============================================================================
# GeneralizedDynamicFactorModel
# =============================================================================

const _GDFM_PLOT_VIEWS = (:eigenvalues, :common, :variance)

"""
    plot_result(m::GeneralizedDynamicFactorModel; view=:eigenvalues, var=nothing, title="", save_path=nothing)

Plot a fitted generalized dynamic factor model.

# Views
- `:eigenvalues` — first `min(q+2, 6)` dynamic eigenvalues against frequency (log₁₀ scale)
- `:common` — overlay of `x_i` and the common component `χ_i` for `var=`
- `:variance` — per-variable common-variance share
"""
function plot_result(m::GeneralizedDynamicFactorModel{T};
                     view::Symbol=:eigenvalues,
                     var::Union{Nothing,Int,String}=nothing,
                     max_eigs::Int=6,
                     max_vars::Int=20,
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    view in _GDFM_PLOT_VIEWS || throw(ArgumentError(
        "Unknown view: :$view. Use $(_GDFM_PLOT_VIEWS)"))
    p = if view === :eigenvalues
        _plot_gdfm_eigenvalues(m; max_eigs=max_eigs, title=title)
    elseif view === :common
        _plot_gdfm_common(m; var=var, title=title)
    else
        _plot_gdfm_variance(m; max_vars=max_vars, title=title)
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

function _plot_gdfm_eigenvalues(m::GeneralizedDynamicFactorModel{T};
                                max_eigs::Int, title::String) where {T}
    dat = spectral_eigenvalue_plot_data(m)
    freq = dat.frequencies
    ev = dat.eigenvalues                      # N × n_freq
    n_eigs_total = size(ev, 1)
    n_plot = min(max(m.q + 2, 1), max_eigs, n_eigs_total)
    names = ["λ$i" for i in 1:n_plot]
    rows = Vector{Pair{String,String}}[]
    floor = T(1e-12)
    for j in 1:length(freq)
        row = Pair{String,String}["x" => _json(freq[j])]
        for i in 1:n_plot
            push!(row, "v$i" => _json(log10(max(ev[i, j], floor))))
        end
        push!(rows, row)
    end
    id = _next_plot_id("gdfm_eig")
    data = _json_array_of_objects(rows)
    s = _series_json(names, _palette_take(n_plot); keys=["v$i" for i in 1:n_plot])
    js = _render_line_js(id, data, s; xlabel="Frequency", ylabel="log₁₀ eigenvalue")
    ptitle = _cap_title("Dynamic eigenvalues", n_plot, n_eigs_total)
    fig = isempty(title) ? "GDFM dynamic eigenvalues" : title
    _make_plot([_PanelSpec(id, ptitle, js)]; title=fig, ncols=1,
               note=_cap_note("eigenvalues", n_plot, n_eigs_total, "max_eigs"))
end

function _plot_gdfm_common(m::GeneralizedDynamicFactorModel; var, title::String)
    i = _resolve_var(var, m.varnames, 1)
    names = [m.varnames[i], "Common"]
    Y = hcat(m.X[:, i], m.common_component[:, i])
    id = _next_plot_id("gdfm_com")
    data = _timeseries_data_json(Y, names)
    s = _series_json(names, _palette_take(2); keys=["v1", "v2"])
    js = _render_line_js(id, data, s; xlabel="Period", ylabel=m.varnames[i])
    fig = isempty(title) ? "GDFM common component" : title
    _make_plot([_PanelSpec(id, names[1], js)]; title=fig, ncols=1)
end

function _plot_gdfm_variance(m::GeneralizedDynamicFactorModel; max_vars::Int, title::String)
    share = common_variance_share(m)
    N = length(share)
    shown = min(N, max_vars)
    rows = Vector{Pair{String,String}}[]
    for i in 1:shown
        push!(rows, ["x" => _json(m.varnames[i]), "share" => _json(share[i])])
    end
    id = _next_plot_id("gdfm_var")
    data = _json_array_of_objects(rows)
    s = _series_json(["Common share"], [_PLOT_COLORS[1]]; keys=["share"])
    js = _render_bar_js(id, data, s; mode="grouped", ylabel="Share")
    ptitle = _cap_title("Common-variance share", shown, N)
    fig = isempty(title) ? "GDFM variance shares" : title
    _make_plot([_PanelSpec(id, ptitle, js)]; title=fig, ncols=1,
               note=_cap_note("variables", shown, N, "max_vars"))
end

# =============================================================================
# HASteadyState — Heterogeneous Agent wealth distribution
# =============================================================================

"""
    plot_result(ss::HASteadyState; view=:default, title="", save_path=nothing)

Plot heterogeneous agent steady state results.

# Views
- `:default` / `:distribution` — Wealth distribution histogram (marginal over income)
- `:lorenz` — Lorenz curve with 45-degree equality reference
- `:policy` — Policy functions (consumption and savings by income state)

# Example
```julia
p = plot_result(ss)                        # wealth distribution
p = plot_result(ss; view=:lorenz)          # Lorenz curve
p = plot_result(ss; view=:policy)          # policy functions
```
"""
function plot_result(ss::HASteadyState{T};
                     view::Symbol=:default,
                     max_bars::Int=60, max_states::Int=length(_PLOT_COLORS),
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    if view == :default || view == :distribution
        p = _plot_ha_distribution(ss; title=title, max_bars=max_bars)
    elseif view == :lorenz
        p = _plot_ha_lorenz(ss; title=title)
    elseif view == :policy
        p = _plot_ha_policy(ss; title=title, max_states=max_states)
    else
        throw(ArgumentError("Unknown view: $view. Use :distribution, :lorenz, or :policy"))
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
Wealth distribution histogram: marginal asset distribution (summed over income states)
plotted as a bar chart with asset grid on x-axis and density on y-axis.
"""
function _plot_ha_distribution(ss::HASteadyState{T}; title::String="",
                               max_bars::Int=60) where {T}
    a_grid = ss.grid.grids[1]
    n_a = ss.grid.n_points[1]
    n_e = ss.grid.n_income

    # Compute marginal asset distribution
    d_mat = reshape(vec(ss.distribution), n_a, n_e)
    d_asset = vec(sum(d_mat; dims=2))

    # Normalize
    total_mass = sum(d_asset)
    if total_mass > zero(T)
        d_asset ./= total_mass
    end

    # Aggregate the marginal mass into (at most 60) contiguous bins so NO grid
    # node's mass is dropped — bar heights are probability mass per bin and sum to
    # the total distribution mass (≈1). Point-sampling one node per bin would
    # discard the mass on skipped nodes (plotrule Anti-Pattern #7 / mass must be
    # conserved). When n_a ≤ 60 each bin holds one node, so heights equal the
    # per-node masses (values unchanged from the old path, now provably conserving).
    nbins = min(n_a, max_bars)
    bin_edges = round.(Int, range(0, n_a; length=nbins + 1))

    rows = Vector{Pair{String,String}}[]
    for b in 1:nbins
        lo = bin_edges[b] + 1
        hi = bin_edges[b + 1]
        hi < lo && continue
        bin_mass = sum(@view d_asset[lo:hi])
        # Label the bin by its right-edge asset value.
        label = _fmt_grid_label(a_grid[hi])
        push!(rows, [
            "x" => _json(label),
            "mass" => _json(bin_mass)
        ])
    end
    data_json = _json_array_of_objects(rows)

    id = _next_plot_id("ha_dist")
    s_json = _series_json(["Probability mass"], [_PLOT_COLORS[1]]; keys=["mass"])
    js = _render_bar_js(id, data_json, s_json;
                        mode="grouped", xlabel="Assets", ylabel="Probability mass")
    binned = nbins < n_a
    ptitle = binned ?
        "Marginal Wealth Distribution ($(n_a) grid nodes in $(nbins) bins)" :
        "Marginal Wealth Distribution"
    p1 = _PanelSpec(id, ptitle, js)

    if isempty(title)
        gini = _gini_coefficient(vec(ss.distribution), ss.grid)
        title = "Wealth Distribution (Gini = $(_fmt(gini; digits=3)))"
    end

    _make_plot([p1]; title=title, ncols=1)
end

"""Format asset grid value as a compact label string."""
function _fmt_grid_label(x::Real)
    ax = abs(x)
    if ax >= 1000
        return string(round(Int, x))
    elseif ax >= 1
        return string(round(x; digits=1))
    else
        return string(round(x; digits=2))
    end
end

"""
Lorenz curve: cumulative population share vs cumulative wealth share,
with the 45-degree equality line for reference.
"""
function _plot_ha_lorenz(ss::HASteadyState{T}; title::String="") where {T}
    a_grid = ss.grid.grids[1]
    n_a = ss.grid.n_points[1]
    n_e = ss.grid.n_income

    # Marginal asset distribution
    d_mat = reshape(vec(ss.distribution), n_a, n_e)
    d_asset = vec(sum(d_mat; dims=2))

    # Sort by asset level
    perm = sortperm(a_grid)
    a_sorted = a_grid[perm]
    d_sorted = d_asset[perm]

    # Normalize
    total_mass = sum(d_sorted)
    if total_mass > zero(T)
        d_sorted ./= total_mass
    end

    mean_wealth = dot(d_sorted, a_sorted)

    # Compute Lorenz curve points
    n_pts = length(a_sorted)
    cum_pop = zeros(T, n_pts + 1)
    cum_wealth = zeros(T, n_pts + 1)

    for i in 1:n_pts
        cum_pop[i + 1] = cum_pop[i] + d_sorted[i]
        cum_wealth[i + 1] = cum_wealth[i] + d_sorted[i] * a_sorted[i]
    end

    # Normalize wealth share by total wealth
    total_wealth = cum_wealth[end]
    if total_wealth > zero(T)
        cum_wealth ./= total_wealth
    end

    # Subsample for manageable plot size
    max_pts = 100
    step = max(1, div(n_pts + 1, max_pts))
    idxs = 1:step:(n_pts + 1)
    if last(idxs) != n_pts + 1
        idxs = vcat(collect(idxs), n_pts + 1)
    end

    rows = Vector{Pair{String,String}}[]
    for idx in idxs
        push!(rows, [
            "x" => _json(cum_pop[idx]),
            "lorenz" => _json(cum_wealth[idx]),
            "equality" => _json(cum_pop[idx])
        ])
    end
    data_json = _json_array_of_objects(rows)

    id = _next_plot_id("ha_lorenz")
    s_json = "[{\"name\":\"Lorenz Curve\",\"color\":\"$(_PLOT_COLORS[1])\",\"key\":\"lorenz\",\"dash\":\"\"}," *
             "{\"name\":\"Perfect Equality\",\"color\":\"#999999\",\"key\":\"equality\",\"dash\":\"6,3\"}]"

    js = _render_line_js(id, data_json, s_json;
                         xlabel="Cumulative Population Share",
                         ylabel="Cumulative Wealth Share")
    p1 = _PanelSpec(id, "Lorenz Curve", js)

    if isempty(title)
        gini = _gini_coefficient(vec(ss.distribution), ss.grid)
        title = "Lorenz Curve (Gini = $(_fmt(gini; digits=3)))"
    end

    _make_plot([p1]; title=title, ncols=1)
end

"""
Policy function plot: consumption and savings as functions of assets,
one line per income state. Two panels side by side.
"""
function _plot_ha_policy(ss::HASteadyState{T}; title::String="",
                         max_states::Int=length(_PLOT_COLORS)) where {T}
    a_grid = ss.grid.grids[1]
    n_a = ss.grid.n_points[1]
    n_e = ss.grid.n_income
    panels = _PanelSpec[]

    # Subsample grid points for cleaner plots
    max_pts = 80
    step = max(1, div(n_a, max_pts))
    idxs = 1:step:n_a
    if last(idxs) != n_a
        idxs = vcat(collect(idxs), n_a)
    end

    income_names = ["e$j" for j in 1:n_e]
    n_plot_e = min(n_e, max_states)

    for (pol_key, pol_label, pol_ylabel) in [
        (:consumption, "Consumption Policy", "c(a, e)"),
        (:savings, "Savings Policy", "a'(a, e)")
    ]
        haskey(ss.policies, pol_key) || continue
        pol = ss.policies[pol_key]

        id = _next_plot_id("ha_pol")
        rows = Vector{Pair{String,String}}[]
        for idx in idxs
            row = Pair{String,String}["x" => _json(a_grid[idx])]
            for j in 1:n_plot_e
                push!(row, "e$j" => _json(pol[idx, j]))
            end
            push!(rows, row)
        end
        data_json = _json_array_of_objects(rows)

        s_json = _series_json(income_names[1:n_plot_e],
                              _palette_take(n_plot_e);
                              keys=["e$j" for j in 1:n_plot_e])
        js = _render_line_js(id, data_json, s_json;
                             xlabel="Assets", ylabel=pol_ylabel)
        push!(panels, _PanelSpec(id, _cap_title(pol_label, n_plot_e, n_e), js))
    end

    if isempty(panels)
        throw(ArgumentError("No :consumption or :savings policy found in HASteadyState"))
    end

    if isempty(title)
        r_val = get(ss.prices, :r, NaN)
        title = "Policy Functions (r = $(_fmt(r_val; digits=4)))"
    end

    _make_plot(panels; title=title, ncols=min(length(panels), 2),
               note=_cap_note("income states", n_plot_e, n_e, "max_states"))
end

# =============================================================================
# StateSpaceModel (EV-37, #445)
# =============================================================================

"""
    plot_result(ss::StateSpaceModel; obs=1, state=nothing, conf_level=DEFAULT_CI_MACRO,
                title="", save_path=nothing)

Overlay the observed series (`obs`-th observable) with the filtered and smoothed level
of a fitted state-space model, plus a `conf_level` (`±z·√smoothed_cov`, default `0.68`,
`DEFAULT_CI_MACRO`) band around the smoothed path. `state` selects which state component
to overlay (defaults to the state that the `obs`-th observation loads on, or state 1).
"""
function plot_result(ss::StateSpaceModel{T}; obs::Int=1, state::Union{Int,Nothing}=nothing,
                     conf_level::Real=DEFAULT_CI_MACRO,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    lv = _validate_conf_level(conf_level)
    z_ci = T(_conf_z(lv))
    pct = _ci_pct(lv)
    isfitted(ss) || throw(ArgumentError("plot_result requires a fitted StateSpaceModel"))
    (1 <= obs <= ss.n_obs) || throw(ArgumentError("obs must be in 1:$(ss.n_obs)"))
    # Default state: the first state the obs-th observation loads on.
    st = state === nothing ? something(findfirst(!iszero, ss.Z[obs, :]), 1) : state
    (1 <= st <= ss.n_state) || throw(ArgumentError("state must be in 1:$(ss.n_state)"))

    z = ss.Z[obs, st]                                    # loading of obs on state st
    z = z == 0 ? one(T) : z
    rows = Vector{Pair{String,String}}[]
    for t in 1:ss.T_obs
        yv = ss.y[t, obs]
        filt = z * ss.filtered_state[t, st]
        smth = z * ss.smoothed_state[t, st]
        sd = z * sqrt(max(ss.smoothed_cov[st, st, t], zero(T)))
        push!(rows, Pair{String,String}[
            "x"    => _json(t),
            "obs"  => isnan(yv) ? "null" : _json(yv),
            "filt" => _json(filt),
            "smth" => _json(smth),
            "lo"   => _json(smth - z_ci * sd),
            "hi"   => _json(smth + z_ci * sd),
        ])
    end
    data_json = _json_array_of_objects(rows)
    id = _next_plot_id("ss_level")
    series = _series_json(["Observed", "Filtered", "Smoothed"],
                          [_PLOT_COLORS[1], _PLOT_COLORS[3], _PLOT_COLORS[2]];
                          keys=["obs", "filt", "smth"], dash=["", "4,3", ""])
    bands = "[{\"lo_key\":\"lo\",\"hi_key\":\"hi\",\"color\":\"$(_PLOT_COLORS[2])\",\"name\":\"$(pct)% CI\"}]"
    js = _render_line_js(id, data_json, series; bands_json=bands,
                         xlabel="Period", ylabel=ss.n_obs == 1 ? "Value" : "Series $obs")
    panel = _PanelSpec(id, "Filtered vs smoothed level ($(pct)% band)", js)
    isempty(title) && (title = "State-Space Model — level estimates")
    p = _make_plot([panel]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# QuantileFactorModel (QFM-02 / QFM-03)
# =============================================================================

const _QFM_PLOT_VIEWS = (:factors, :loadings, :variance, :scree)

# Internal helper: factors view for QuantileFactorModel (no docstring per visualization.md).
function _plot_qfm_factors(m::QuantileFactorModel{T};
                           var::Union{Int,String,Nothing}=nothing,
                           conf_level::Real=DEFAULT_CI_MACRO,
                           max_factors::Int=5,
                           ncols::Int=0,
                           title::String="") where {T}
    lv = _validate_conf_level(conf_level)
    z_ci = _conf_z(lv)
    pct = _ci_pct(lv)
    T_obs, r = size(m.factors)
    n_plot = min(r, max(1, max_factors))
    panels = _PanelSpec[]

    if m.se_factors !== nothing
        xs = collect(1:T_obs)
        for k in 1:n_plot
            id = _next_plot_id("qfm_f")
            est = Float64.(m.factors[:, k])
            se = Float64.(m.se_factors[:, k])
            lo = est .- z_ci .* se
            hi = est .+ z_ci .* se
            data_json = _keyed_rows_json(xs, ["est" => est, "lo" => lo, "hi" => hi])
            col_k = _PLOT_SERIES[mod1(k, length(_PLOT_SERIES))]
            s_json = _series_json(["Factor $k"], [col_k]; keys=["est"])
            bands = "[{\"lo_key\":\"lo\",\"hi_key\":\"hi\",\"color\":$(_json(col_k)),\"alpha\":$(_PLOT_CI_ALPHA),\"name\":\"$(pct)% CI\"}]"
            refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
            js = _render_line_js(id, data_json, s_json; bands_json=bands, ref_lines_json=refs,
                                 integer_x=true, xlabel="Period", ylabel="Factor $k")
            push!(panels, _PanelSpec(id, "Factor $k ($(pct)% CI)", js))
        end
    else
        id = _next_plot_id("qfm_fac")
        fac_names = ["Factor $k" for k in 1:n_plot]
        data_json = _timeseries_data_json(m.factors[:, 1:n_plot], fac_names)
        s_json = _series_json(fac_names, _colors_for(fac_names); keys=["v$k" for k in 1:n_plot])
        refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
        js = _render_line_js(id, data_json, s_json; ref_lines_json=refs,
                             integer_x=true, xlabel="Period", ylabel="Factor Value")
        push!(panels, _PanelSpec(id, _cap_title("Extracted Quantile Factors", n_plot, r), js))
    end

    if var !== nothing
        vi = _resolve_var(var, m.varnames)
        idv = _next_plot_id("qfm_fit")
        vname = m.varnames[vi]
        Y_cmp = hcat(m.X[:, vi], m.common_component[:, vi])
        snames = [vname, "Common (τ=$(_fmt(m.tau; digits=2)))"]
        data_v = _timeseries_data_json(Y_cmp, snames)
        s_v = _series_json(snames, [_PLOT_SERIES[1], _PLOT_SERIES[2]]; keys=["v1", "v2"], dash=["", "5,3"])
        js_v = _render_line_js(idv, data_v, s_v; integer_x=true, xlabel="Period", ylabel=vname)
        push!(panels, _PanelSpec(idv, "Observed vs Common: $(vname)", js_v))
    end

    ftitle = isempty(title) ?
        "Quantile Factor Model (τ=$(_fmt(m.tau; digits=3)), r=$(m.r), $(m.method))" : title
    ncols_use = ncols <= 0 ? (length(panels) == 1 ? 1 : min(length(panels), 2)) : ncols
    return _make_plot(panels; title=ftitle, ncols=ncols_use,
                      note=_cap_note("factors", n_plot, r, "max_factors"))
end

# Internal helper: loadings view for QuantileFactorModel.
function _plot_qfm_loadings(m::QuantileFactorModel{T};
                            var::Union{Int,String,Nothing}=nothing,
                            conf_level::Real=DEFAULT_CI_MACRO,
                            max_factors::Int=5,
                            max_vars::Int=20,
                            ncols::Int=0,
                            title::String="") where {T}
    lv = _validate_conf_level(conf_level)
    z_ci = _conf_z(lv)
    pct = _ci_pct(lv)
    N, r = size(m.loadings)
    panels = _PanelSpec[]
    note = ""

    if var !== nothing
        vi = _resolve_var(var, m.varnames)
        vname = m.varnames[vi]
        n_fac = min(r, max(1, max_factors))
        id = _next_plot_id("qfm_lv")
        if m.se_loadings !== nothing
            rows = String[]
            for k in 1:n_fac
                est = m.loadings[vi, k]
                se = m.se_loadings[vi, k]
                lo = isfinite(est) && isfinite(se) ? est - z_ci * se : est
                hi = isfinite(est) && isfinite(se) ? est + z_ci * se : est
                push!(rows, "{\"name\":$(_json("Factor $k")),\"effect\":$(_json(est)),\"ci_lo\":$(_json(lo)),\"ci_hi\":$(_json(hi))}")
            end
            js = _render_coef_plot_js(id, "[" * join(rows, ",\n") * "]";
                                      color=_PLOT_SERIES[1], ref_value=0,
                                      xlabel="Loading λ(τ)", ylabel="")
        else
            rows = [Pair{String,String}["x" => _json("Factor $k"), "val" => _json(m.loadings[vi, k])]
                    for k in 1:n_fac]
            s_json = _series_json(["Loading"], [_PLOT_SERIES[1]]; keys=["val"])
            js = _render_bar_js(id, _json_array_of_objects(rows), s_json;
                                mode="grouped", orientation="h", xlabel="Loading λ(τ)", ylabel="")
        end
        ci_tag = m.se_loadings !== nothing ? " ($(pct)% CI)" : ""
        push!(panels, _PanelSpec(id, _cap_title("Loadings: $(vname)$(ci_tag)", n_fac, r), js))
        note = _cap_note("factors", n_fac, r, "max_factors")
    else
        shown_v = min(N, max(1, max_vars))
        n_fac = min(r, max(1, max_factors))
        if m.se_loadings !== nothing
            for k in 1:n_fac
                id = _next_plot_id("qfm_lk")
                rows = String[]
                for i in 1:shown_v
                    est = m.loadings[i, k]
                    se = m.se_loadings[i, k]
                    lo = isfinite(est) && isfinite(se) ? est - z_ci * se : est
                    hi = isfinite(est) && isfinite(se) ? est + z_ci * se : est
                    push!(rows, "{\"name\":$(_json(m.varnames[i])),\"effect\":$(_json(est)),\"ci_lo\":$(_json(lo)),\"ci_hi\":$(_json(hi))}")
                end
                col_k = _PLOT_SERIES[mod1(k, length(_PLOT_SERIES))]
                js = _render_coef_plot_js(id, "[" * join(rows, ",\n") * "]";
                                          color=col_k, ref_value=0,
                                          xlabel="Loading λ_$k(τ)", ylabel="")
                push!(panels, _PanelSpec(id, _cap_title("Factor $k Loadings ($(pct)% CI)", shown_v, N), js))
            end
        else
            id = _next_plot_id("qfm_lbar")
            fac_names = ["Factor $k" for k in 1:n_fac]
            rows = Vector{Pair{String,String}}[]
            for i in 1:shown_v
                row = Pair{String,String}["x" => _json(m.varnames[i])]
                for k in 1:n_fac
                    push!(row, "f$k" => _json(m.loadings[i, k]))
                end
                push!(rows, row)
            end
            s_json = _series_json(fac_names, _colors_for(fac_names); keys=["f$k" for k in 1:n_fac])
            js = _render_bar_js(id, _json_array_of_objects(rows), s_json;
                                mode="grouped", orientation="h", xlabel="Loading λ(τ)", ylabel="")
            push!(panels, _PanelSpec(id, _cap_title("Factor Loadings", shown_v, N), js))
        end
        note_v = _cap_note("variables", shown_v, N, "max_vars")
        note_f = _cap_note("factors", n_fac, r, "max_factors")
        note = isempty(note_v) ? note_f : (isempty(note_f) ? note_v : note_v * " " * note_f)
    end

    ftitle = isempty(title) ?
        "Quantile Factor Loadings (τ=$(_fmt(m.tau; digits=3)), r=$(m.r))" : title
    ncols_use = ncols <= 0 ? (length(panels) == 1 ? 1 : min(length(panels), 2)) : ncols
    return _make_plot(panels; title=ftitle, ncols=ncols_use, note=note)
end

# Internal helper: variance / scree view for QuantileFactorModel.
function _plot_qfm_variance(m::QuantileFactorModel{T};
                            var::Union{Int,String,Nothing}=nothing,
                            max_vars::Int=20,
                            ncols::Int=0,
                            title::String="") where {T}
    N = length(m.pseudo_r2)
    r = length(m.sigma_sq)

    id1 = _next_plot_id("qfm_scree")
    rows1 = [Pair{String,String}["x" => _json("Factor $k"), "sig" => _json(m.sigma_sq[k])] for k in 1:r]
    s1 = _series_json(["σ²"], [_PLOT_SERIES[1]]; keys=["sig"])
    js1 = _render_bar_js(id1, _json_array_of_objects(rows1), s1;
                         mode="grouped", xlabel="Factor", ylabel="Factor Strength (σ²)")
    p1 = _PanelSpec(id1, "Factor Strength Scree", js1)

    id2 = _next_plot_id("qfm_pr2")
    var_idxs = var === nothing ? collect(1:min(N, max(1, max_vars))) : [_resolve_var(var, m.varnames)]
    rows2 = [Pair{String,String}["x" => _json(m.varnames[i]), "r2" => _json(m.pseudo_r2[i])] for i in var_idxs]
    s2 = _series_json(["Pseudo-R²"], [_PLOT_SERIES[2]]; keys=["r2"])
    js2 = _render_bar_js(id2, _json_array_of_objects(rows2), s2;
                         mode="grouped", orientation="h", xlabel="Pseudo-R²", ylabel="")
    p2_title = var === nothing ? _cap_title("Series Pseudo-R²", length(var_idxs), N) :
                                 "Pseudo-R²: $(m.varnames[var_idxs[1]])"
    p2 = _PanelSpec(id2, p2_title, js2)

    ftitle = isempty(title) ?
        "Quantile Factor Strength & Pseudo-R² (τ=$(_fmt(m.tau; digits=3)), r=$(m.r))" : title
    ncols_use = ncols <= 0 ? 2 : ncols
    note = var === nothing ? _cap_note("variables", length(var_idxs), N, "max_vars") : ""
    return _make_plot([p1, p2]; title=ftitle, ncols=ncols_use, note=note)
end

"""
    plot_result(m::QuantileFactorModel; view=:factors, var=nothing,
                conf_level=DEFAULT_CI_MACRO, max_factors=5,
                max_vars=20, ncols=0, title="", save_path=nothing)

Plot an estimated [`QuantileFactorModel`](@ref).

# Views
- `:factors` — time series of estimated quantile factors `m.factors` (with `conf_level`
  CI bands, default `0.68`, when `m.se_factors !== nothing`), plus observed vs common component for `var` if given.
- `:loadings` — horizontal bar or CI whisker plot of factor loadings `m.loadings` across
  variables (or across factors for a selected variable `var`).
- `:variance` / `:scree` — factor strength scree `m.sigma_sq` alongside per-series
  Koenker–Machado `m.pseudo_r2`.

# Keyword Arguments
- `view::Symbol = :factors` — exhibit view (`:factors`, `:loadings`, `:variance`, or `:scree`)
- `var::Union{Int,String,Nothing} = nothing` — optional variable index or name
- `conf_level::Real = DEFAULT_CI_MACRO` — confidence level in `(0, 1)` for SE bands (default `0.68`)
- `max_factors::Int = 5` — maximum number of factors displayed (C7 note if capped)
- `max_vars::Int = 20` — maximum number of panel series displayed (C7 note if capped)
- `ncols::Int = 0` — number of grid columns (`0` = auto)
- `title::String = ""` — custom figure title (`""` = auto-generated)
- `save_path::Union{String,Nothing} = nothing` — optional output HTML path
"""
function plot_result(m::QuantileFactorModel{T};
                     view::Symbol=:factors,
                     var::Union{Int,String,Nothing}=nothing,
                     conf_level::Real=DEFAULT_CI_MACRO,
                     max_factors::Int=5,
                     max_vars::Int=20,
                     ncols::Int=0,
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    lv = _validate_conf_level(conf_level)
    view in _QFM_PLOT_VIEWS || throw(ArgumentError(
        "Unknown view :$view for QuantileFactorModel. Valid views: $(_QFM_PLOT_VIEWS)"))

    p = if view === :factors
        _plot_qfm_factors(m; var=var, conf_level=lv, max_factors=max_factors, ncols=ncols, title=title)
    elseif view === :loadings
        _plot_qfm_loadings(m; var=var, conf_level=lv, max_factors=max_factors, max_vars=max_vars, ncols=ncols, title=title)
    else
        _plot_qfm_variance(m; var=var, max_vars=max_vars, ncols=ncols, title=title)
    end
    save_path !== nothing && save_plot(p, save_path)
    return p
end

# =============================================================================
# QuantileFactorGrid (QFM-05)
# =============================================================================

const _QFM_GRID_PLOT_VIEWS = (:factors, :loadings, :ic, :rank)

"""
    plot_result(g::QuantileFactorGrid; view=:factors, factor=1, var=1,
                conf_level=DEFAULT_CI_MACRO, max_taus=10,
                max_factors=5, ncols=0, title="", save_path=nothing)

Plot a multi-quantile [`QuantileFactorGrid`](@ref).

# Views
- `:factors` — overlay estimated quantile factor paths `F̂_factor(τ)` across `g.taus`.
- `:loadings` — `τ`-profile of estimated loadings across `g.taus` for variable `var`
  (with `conf_level` CI band, default `0.68`, on `factor` when standard errors were computed).
- `:ic` / `:rank` — quantile-specific factor count `r(τ)` (`r_by_tau`) and check-loss
  objective across `g.taus`.

# Keyword Arguments
- `view::Symbol = :factors` — exhibit view (`:factors`, `:loadings`, `:ic`, or `:rank`)
- `factor::Int = 1` — 1-based factor index for `:factors` (and CI band in `:loadings`)
- `var::Union{Int,String} = 1` — variable index or name for `:loadings`
- `conf_level::Real = DEFAULT_CI_MACRO` — confidence level in `(0, 1)` for `:loadings` SE band (default `0.68`)
- `max_taus::Int = 10` — maximum quantiles overlaid in `:factors` (C7 note if capped)
- `max_factors::Int = 5` — maximum factor curves drawn in `:loadings` (C7 note if capped)
- `ncols::Int = 0` — number of grid columns (`0` = auto)
- `title::String = ""` — custom figure title (`""` = auto-generated)
- `save_path::Union{String,Nothing} = nothing` — optional output HTML path
"""
function plot_result(g::QuantileFactorGrid{T};
                     view::Symbol=:factors,
                     factor::Int=1,
                     var::Union{Int,String}=1,
                     conf_level::Real=DEFAULT_CI_MACRO,
                     max_taus::Int=10,
                     max_factors::Int=5,
                     ncols::Int=0,
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    lv = _validate_conf_level(conf_level)
    z_ci = _conf_z(lv)
    pct = _ci_pct(lv)
    !isempty(g.models) || throw(ArgumentError("Cannot plot an empty QuantileFactorGrid"))
    view in _QFM_GRID_PLOT_VIEWS || throw(ArgumentError(
        "Unknown view :$view for QuantileFactorGrid. Valid views: $(_QFM_GRID_PLOT_VIEWS)"))
    r_max = maximum(g.r_by_tau)
    (1 <= factor <= r_max) || throw(ArgumentError("factor must be in 1:$r_max, got $factor"))
    vi = _resolve_var(var, g.varnames)

    perm = sortperm(g.taus)
    M = length(g.taus)

    p = if view === :factors
        idxs_avail = [j for j in perm if factor <= g.models[j].r]
        n_avail = length(idxs_avail)
        n_show = min(n_avail, max(1, max_taus))
        pick = n_show == n_avail ? idxs_avail :
               idxs_avail[unique(round.(Int, range(1, n_avail; length=n_show)))]
        T_obs = size(g.models[pick[1]].factors, 1)
        s_names = ["τ=$(_fmt(g.taus[j]; digits=2))" for j in pick]
        cols = ["t$k" => Float64.(g.models[j].factors[:, factor]) for (k, j) in enumerate(pick)]
        data_json = _keyed_rows_json(collect(1:T_obs), cols)
        s_json = _series_json(s_names, _colors_for(s_names); keys=["t$k" for k in 1:length(pick)])
        refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
        id = _next_plot_id("qgrid_f")
        js = _render_line_js(id, data_json, s_json; ref_lines_json=refs,
                             integer_x=true, xlabel="Period", ylabel="Factor $factor")
        ptitle = _cap_title("Factor $factor Across Quantiles", length(pick), n_avail)
        ftitle = isempty(title) ? "Quantile Factor Grid — Factor $factor ($(M) quantiles)" : title
        _make_plot([_PanelSpec(id, ptitle, js)]; title=ftitle,
                   ncols=(ncols <= 0 ? 1 : ncols),
                   note=_cap_note("quantiles", length(pick), n_avail, "max_taus"))
    elseif view === :loadings
        vname = g.varnames[vi]
        n_fac = min(r_max, max(1, max_factors))
        id = _next_plot_id("qgrid_l")
        if M == 1
            m1 = g.models[1]
            k_lim = min(m1.r, n_fac)
            rows = [Pair{String,String}["x" => _json("Factor $k"), "val" => _json(m1.loadings[vi, k])]
                    for k in 1:k_lim]
            s_json = _series_json(["Loading (τ=$(_fmt(m1.tau; digits=2)))"], [_PLOT_SERIES[1]]; keys=["val"])
            js = _render_bar_js(id, _json_array_of_objects(rows), s_json;
                                mode="grouped", orientation="h", xlabel="Loading λ(τ)", ylabel="")
        else
            taus_s = Float64[Float64(g.taus[j]) for j in perm]
            has_se = any(g.models[j].se_loadings !== nothing && factor <= g.models[j].r for j in perm)
            rows = Vector{Pair{String,String}}[]
            for (pos, j) in enumerate(perm)
                mj = g.models[j]
                row = Pair{String,String}["x" => _json(taus_s[pos])]
                for k in 1:n_fac
                    val = k <= mj.r ? Float64(mj.loadings[vi, k]) : NaN
                    push!(row, "f$k" => _json(val))
                end
                if has_se
                    if mj.se_loadings !== nothing && factor <= mj.r
                        est = Float64(mj.loadings[vi, factor])
                        se = Float64(mj.se_loadings[vi, factor])
                        push!(row, "lo" => _json(est - z_ci * se))
                        push!(row, "hi" => _json(est + z_ci * se))
                    else
                        push!(row, "lo" => "null")
                        push!(row, "hi" => "null")
                    end
                end
                push!(rows, row)
            end
            fac_names = ["Factor $k" for k in 1:n_fac]
            fac_cols = _colors_for(fac_names)
            s_json = _series_json(fac_names, fac_cols; keys=["f$k" for k in 1:n_fac])
            bands = has_se ?
                "[{\"lo_key\":\"lo\",\"hi_key\":\"hi\",\"color\":$(_json(fac_cols[min(factor, length(fac_cols))])),\"alpha\":$(_PLOT_CI_ALPHA),\"name\":\"Factor $factor $(pct)% CI\"}]" : "[]"
            refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
            js = _render_line_js(id, _json_array_of_objects(rows), s_json;
                                 bands_json=bands, ref_lines_json=refs,
                                 xlabel="Quantile τ", ylabel="Loading λ(τ)")
        end
        ptitle = _cap_title("Loading Profile: $(vname)", n_fac, r_max)
        ftitle = isempty(title) ? "Quantile Factor Loading Profile — $(vname)" : title
        _make_plot([_PanelSpec(id, ptitle, js)]; title=ftitle,
                   ncols=(ncols <= 0 ? 1 : ncols),
                   note=_cap_note("factors", n_fac, r_max, "max_factors"))
    else
        # :ic / :rank
        id1 = _next_plot_id("qgrid_r")
        rows1 = [Pair{String,String}["x" => _json("τ=$(_fmt(g.taus[j]; digits=2))"), "r" => _json(g.r_by_tau[j])]
                 for j in perm]
        s1 = _series_json(["Active Rank r(τ)"], [_PLOT_SERIES[1]]; keys=["r"])
        js1 = _render_bar_js(id1, _json_array_of_objects(rows1), s1;
                             mode="grouped", xlabel="Quantile τ", ylabel="Factors r(τ)")
        p1 = _PanelSpec(id1, "Selected Factor Rank r(τ)", js1)

        id2 = _next_plot_id("qgrid_obj")
        if M >= 2
            taus_s = Float64[Float64(g.taus[j]) for j in perm]
            objs_s = Float64[Float64(g.models[j].objective) for j in perm]
            data2 = _keyed_rows_json(taus_s, ["loss" => objs_s])
            s2 = _series_json(["Check Loss"], [_PLOT_SERIES[2]]; keys=["loss"])
            js2 = _render_line_js(id2, data2, s2; xlabel="Quantile τ", ylabel="Mean Check Loss")
        else
            rows2 = [Pair{String,String}["x" => _json("τ=$(_fmt(g.taus[1]; digits=2))"),
                                         "loss" => _json(g.models[1].objective)]]
            s2 = _series_json(["Check Loss"], [_PLOT_SERIES[2]]; keys=["loss"])
            js2 = _render_bar_js(id2, _json_array_of_objects(rows2), s2;
                                 mode="grouped", xlabel="Quantile τ", ylabel="Mean Check Loss")
        end
        p2 = _PanelSpec(id2, "Mean Check-Function Loss", js2)

        ftitle = isempty(title) ? "Quantile Factor Grid — Rank & Loss Across τ" : title
        _make_plot([p1, p2]; title=ftitle, ncols=(ncols <= 0 ? 2 : ncols))
    end

    save_path !== nothing && save_plot(p, save_path)
    return p
end

# =============================================================================
# QFMICResult (QFM-04)
# =============================================================================

const _QFM_IC_PLOT_VIEWS = (:ic, :scree, :loss)

"""
    plot_result(ic::QFMICResult; view=:ic, ncols=0, title="", save_path=nothing)

Plot quantile factor selection criteria from [`ic_criteria_qfm`](@ref).

# Views
- `:ic` — check-loss `IC1`, `IC2`, `IC3` curves across `k = 0:r_max` (with vertical
  reference lines at selected ranks), CDG rank-minimization scree `sigma_sq` vs
  threshold `P_NT`, and check-function `loss` path.
- `:scree` — factor strength scree `sigma_sq` vs `P_NT` and check-function `loss`.
- `:loss` — average check-function `loss` path over `k = 0:r_max`.

# Keyword Arguments
- `view::Symbol = :ic` — exhibit view (`:ic`, `:scree`, or `:loss`)
- `ncols::Int = 0` — number of grid columns (`0` = auto)
- `title::String = ""` — custom figure title (`""` = auto-generated)
- `save_path::Union{String,Nothing} = nothing` — optional output HTML path
"""
function plot_result(ic::QFMICResult{T};
                     view::Symbol=:ic,
                     ncols::Int=0,
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    view in _QFM_IC_PLOT_VIEWS || throw(ArgumentError(
        "Unknown view :$view for QFMICResult. Valid views: $(_QFM_IC_PLOT_VIEWS)"))

    ks_full = collect(0:ic.r_max)
    panels = _PanelSpec[]

    if view === :ic
        id1 = _next_plot_id("qfmic_ic")
        data1 = _keyed_rows_json(ks_full, [
            "ic1" => Float64.(ic.IC1),
            "ic2" => Float64.(ic.IC2),
            "ic3" => Float64.(ic.IC3),
        ])
        s1 = _series_json(["IC1 (r=$(ic.r_IC1))", "IC2 (r=$(ic.r_IC2))", "IC3 (r=$(ic.r_IC3))"],
                          [_PLOT_SERIES[1], _PLOT_SERIES[2], _PLOT_SERIES[3]];
                          keys=["ic1", "ic2", "ic3"], dash=["", "5,3", "2,2"])
        sel_rs = unique([ic.r_IC1, ic.r_IC2, ic.r_IC3])
        refs1 = "[" * join(["{\"value\":$(_json(rv)),\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"4,3\",\"axis\":\"x\"}"
                            for rv in sel_rs], ",") * "]"
        js1 = _render_line_js(id1, data1, s1; ref_lines_json=refs1,
                              integer_x=true, xlabel="Factors k", ylabel="IC Value")
        push!(panels, _PanelSpec(id1, "Check-Loss Information Criteria", js1))
    end

    if view === :ic || view === :scree
        id2 = _next_plot_id("qfmic_scree")
        ks_sig = collect(1:ic.r_max)
        rows2 = [Pair{String,String}["x" => _json(ks_sig[j]), "y" => _json(ic.sigma_sq[j])]
                 for j in eachindex(ks_sig)]
        refs2 = isfinite(ic.threshold_PNT) ?
            "[{\"value\":$(_json(Float64(ic.threshold_PNT))),\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"5,3\"}]" : "[]"
        js2 = _render_vbar_js(id2, _json_array_of_objects(rows2);
                              bar_color=_PLOT_SERIES[1], ref_lines_json=refs2,
                              tip_label="k =", integer_x=true,
                              xlabel="Factor k", ylabel="Factor Strength (σ²)")
        push!(panels, _PanelSpec(id2, "CDG Rank Scree (r_rank=$(ic.r_rank), P_NT=$(_fmt(ic.threshold_PNT; digits=3)))", js2))
    end

    id3 = _next_plot_id("qfmic_loss")
    data3 = _keyed_rows_json(ks_full, ["loss" => Float64.(ic.loss)])
    s3 = _series_json(["Check Loss"], [_PLOT_SERIES[4]]; keys=["loss"])
    js3 = _render_line_js(id3, data3, s3; integer_x=true, xlabel="Factors k", ylabel="Mean Check Loss")
    push!(panels, _PanelSpec(id3, "Check-Function Loss Path", js3))

    ftitle = isempty(title) ?
        "QFM Factor Selection (τ=$(_fmt(ic.tau; digits=3)), r_rank=$(ic.r_rank), r_IC2=$(ic.r_IC2))" : title
    ncols_use = ncols <= 0 ? (length(panels) == 1 ? 1 : min(length(panels), 3)) : ncols
    p = _make_plot(panels; title=ftitle, ncols=ncols_use)
    save_path !== nothing && save_plot(p, save_path)
    return p
end

# =============================================================================
# QFMSpanningResult (QFM-05)
# =============================================================================

"""
    plot_result(s::QFMSpanningResult; max_bars=24, ncols=0, title="", save_path=nothing)

Plot PCA-vs-QFM factor spanning diagnostics from [`test_qfm_spanning`](@ref),
comparing per-factor OLS ``R^2`` against Koenker–Machado quantile ``R^1(\\tau)``
and showing canonical correlations across quantiles.

# Keyword Arguments
- `max_bars::Int = 24` — maximum `(τ, factor)` rows shown in the R² / R¹ bar chart (C7 note if capped)
- `ncols::Int = 0` — number of grid columns (`0` = auto)
- `title::String = ""` — custom figure title (`""` = auto-generated)
- `save_path::Union{String,Nothing} = nothing` — optional output HTML path
"""
function plot_result(s::QFMSpanningResult{T};
                     max_bars::Int=24,
                     ncols::Int=0,
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    !isempty(s.taus) || throw(ArgumentError("Cannot plot an empty QFMSpanningResult"))
    perm = sortperm(s.taus)
    M = length(s.taus)

    all_rows = Vector{Pair{String,String}}[]
    for j in perm
        tau_lbl = "τ=$(_fmt(s.taus[j]; digits=2))"
        for k in 1:s.r_qfm[j]
            flag = s.unspanned[j][k] ? " *" : ""
            push!(all_rows, [
                "x" => _json("$(tau_lbl) F$k$(flag)"),
                "r2" => _json(s.ols_r2[j][k]),
                "r1" => _json(s.quantile_r1[j][k]),
            ])
        end
    end
    n_total = length(all_rows)
    n_show = min(n_total, max(1, max_bars))
    rows1 = all_rows[1:n_show]

    id1 = _next_plot_id("qspan_bar")
    s1 = _series_json(["OLS R²", "Quantile R¹"], [_PLOT_SERIES[1], _PLOT_SERIES[2]]; keys=["r2", "r1"])
    js1 = _render_bar_js(id1, _json_array_of_objects(rows1), s1;
                         mode="grouped", orientation="h", xlabel="R² / Pseudo-R¹", ylabel="")
    p1 = _PanelSpec(id1, _cap_title("Factor Spanning (* unspanned < $(_fmt(s.threshold; digits=2)))", n_show, n_total), js1)
    panels = _PanelSpec[p1]

    m_cc = maximum(length, s.cancor; init=0)
    if m_cc >= 1
        id2 = _next_plot_id("qspan_cc")
        if M >= 2
            taus_s = Float64[Float64(s.taus[j]) for j in perm]
            cc_rows = Vector{Pair{String,String}}[]
            for (pos, j) in enumerate(perm)
                row = Pair{String,String}["x" => _json(taus_s[pos])]
                ccj = s.cancor[j]
                for k in 1:m_cc
                    push!(row, "c$k" => _json(k <= length(ccj) ? Float64(ccj[k]) : NaN))
                end
                push!(cc_rows, row)
            end
            cc_names = ["Canonical Corr. $k" for k in 1:m_cc]
            s2 = _series_json(cc_names, _colors_for(cc_names); keys=["c$k" for k in 1:m_cc])
            js2 = _render_line_js(id2, _json_array_of_objects(cc_rows), s2;
                                  xlabel="Quantile τ", ylabel="Canonical Correlation")
        else
            cc1 = s.cancor[perm[1]]
            cc_rows = [Pair{String,String}["x" => _json("CC $k"), "cc" => _json(cc1[k])] for k in eachindex(cc1)]
            s2 = _series_json(["Canonical Corr."], [_PLOT_SERIES[3]]; keys=["cc"])
            js2 = _render_bar_js(id2, _json_array_of_objects(cc_rows), s2;
                                 mode="grouped", xlabel="Component", ylabel="Canonical Correlation")
        end
        push!(panels, _PanelSpec(id2, "Canonical Correlations (r_pca=$(s.r_pca))", js2))
    end

    ftitle = isempty(title) ?
        "PCA-vs-QFM Factor Spanning (r_pca=$(s.r_pca), threshold=$(_fmt(s.threshold; digits=2)))" : title
    ncols_use = ncols <= 0 ? min(length(panels), 2) : ncols
    p = _make_plot(panels; title=ftitle, ncols=ncols_use,
                   note=_cap_note("factor-quantile pairs", n_show, n_total, "max_bars"))
    save_path !== nothing && save_plot(p, save_path)
    return p
end

# =============================================================================
# QFMLoadingConstancyTest (QFM-05)
# =============================================================================

"""
    plot_result(t::QFMLoadingConstancyTest; ncols=0, title="", save_path=nothing)

Plot the bootstrap null distribution of the quantile loading constancy test
[`test_qfm_loading_constancy`](@ref) (`t.boot_stats`) with a vertical reference
line at the observed projection residual statistic `t.statistic` (or a summary
bar chart if `t.boot_stats` is empty).

# Keyword Arguments
- `ncols::Int = 0` — number of grid columns (`0` = auto)
- `title::String = ""` — custom figure title (`""` = auto-generated)
- `save_path::Union{String,Nothing} = nothing` — optional output HTML path
"""
function plot_result(t::QFMLoadingConstancyTest{T};
                     ncols::Int=0,
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    bs = Float64[Float64(v) for v in t.boot_stats if isfinite(v)]
    id = _next_plot_id("qconst")

    js = if !isempty(bs)
        bins_json = _histogram_bins(bs; density=true)
        has_spread = length(bs) >= 2 && Statistics.std(bs) > 1e-10
        dens_json = has_spread ? _kde_line_json(bs) : "[]"
        s_json = has_spread ?
            _series_json(["Bootstrap Null", "KDE"], [_PLOT_SERIES[1], _PLOT_SERIES[2]]) :
            _series_json(["Bootstrap Null"], [_PLOT_SERIES[1]])
        refs_json = isfinite(t.statistic) ?
            "[{\"value\":$(_json(Float64(t.statistic))),\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"5,3\",\"axis\":\"x\"}]" : "[]"
        _render_histogram_js(id, bins_json, s_json;
                             density_json=dens_json, ref_lines_json=refs_json,
                             xlabel="Projection Residual Norm²", ylabel="Density")
    else
        rows = [
            Pair{String,String}["x" => _json("Statistic S"), "val" => _json(t.statistic)],
            Pair{String,String}["x" => _json("p-value"), "val" => _json(t.pvalue)],
        ]
        s_json = _series_json(["Test Summary"], [_PLOT_SERIES[1]]; keys=["val"])
        _render_bar_js(id, _json_array_of_objects(rows), s_json;
                       mode="grouped", orientation="h", xlabel="Value", ylabel="")
    end

    ptitle = "S = $(_fmt(t.statistic; digits=4)), p = $(_fmt(t.pvalue; digits=4)) ($(t.bootstrap), B=$(t.n_boot))"
    ftitle = isempty(title) ?
        "QFM Loading Constancy Test (τ₁=$(_fmt(t.tau1; digits=2)) vs τ₂=$(_fmt(t.tau2; digits=2)))" : title
    p = _make_plot([_PanelSpec(id, ptitle, js)]; title=ftitle, ncols=(ncols <= 0 ? 1 : ncols))
    save_path !== nothing && save_plot(p, save_path)
    return p
end

# =============================================================================
# QuantileFactorPanelModel (QFM-06)
# =============================================================================

const _QFM_PANEL_PLOT_VIEWS = (:coef, :factors, :loadings)

"""
    plot_result(m::QuantileFactorPanelModel; view=:coef, conf_level=DEFAULT_CI_MICRO,
                max_factors=5, max_vars=20, ncols=0, title="", save_path=nothing)

Plot a panel quantile regression with interactive quantile factor effects
([`QuantileFactorPanelModel`](@ref)).

# Views
- `:coef` — horizontal dot-and-whisker coefficient plot (`m.beta` with confidence
  intervals from `StatsAPI.confint(m; level=conf_level)`), plus a unit-slope
  box-plot panel when `m.slope == :heterogeneous`.
- `:factors` — time series of extracted interactive quantile factors `m.factors`.
- `:loadings` — horizontal grouped bar chart of unit factor loadings `m.loadings`.

# Keyword Arguments
- `view::Symbol = :coef` — exhibit view (`:coef`, `:factors`, or `:loadings`)
- `conf_level::Real = DEFAULT_CI_MICRO` — confidence level in `(0, 1)` for `:coef` intervals (default `0.90`)
- `max_factors::Int = 5` — maximum number of factors shown in `:factors`/`:loadings` (C7 note if capped)
- `max_vars::Int = 20` — maximum number of cross-sectional units shown in `:loadings` (C7 note if capped)
- `ncols::Int = 0` — number of grid columns (`0` = auto)
- `title::String = ""` — custom figure title (`""` = auto-generated)
- `save_path::Union{String,Nothing} = nothing` — optional output HTML path
"""
function plot_result(m::QuantileFactorPanelModel{T};
                     view::Symbol=:coef,
                     conf_level::Real=DEFAULT_CI_MICRO,
                     max_factors::Int=5,
                     max_vars::Int=20,
                     ncols::Int=0,
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    view in _QFM_PANEL_PLOT_VIEWS || throw(ArgumentError(
        "Unknown view :$view for QuantileFactorPanelModel. Valid views: $(_QFM_PANEL_PLOT_VIEWS)"))
    lv = _validate_conf_level(conf_level)

    p = if view === :coef
        ci = StatsAPI.confint(m; level=lv)
        rows = String[]
        for j in eachindex(m.beta)
            push!(rows, "{\"name\":$(_json(m.varnames[j])),\"effect\":$(_json(m.beta[j])),\"ci_lo\":$(_json(ci[j, 1])),\"ci_hi\":$(_json(ci[j, 2]))}")
        end
        id1 = _next_plot_id("qpan_coef")
        js1 = _render_coef_plot_js(id1, "[" * join(rows, ",\n") * "]";
                                   color=_PLOT_SERIES[1], ref_value=0,
                                   xlabel="Coefficient (τ=$(_fmt(m.tau; digits=2)))", ylabel="")
        pct = _ci_pct(lv)
        slope_lbl = m.slope === :heterogeneous ? "Mean Group Slopes" : "Slope Coefficients"
        panels = _PanelSpec[_PanelSpec(id1, "$(slope_lbl) ($(pct)% CI)", js1)]

        if m.slope === :heterogeneous && size(m.beta_i, 1) >= 1 && size(m.beta_i, 2) == length(m.varnames)
            id2 = _next_plot_id("qpan_het")
            cols_i = [m.beta_i[:, j] for j in 1:size(m.beta_i, 2)]
            js2 = _render_box_js(id2, _boxes_json(m.varnames, cols_i);
                                 orientation=:h, xlabel="Unit Slope βᵢ(τ)", ylabel="")
            push!(panels, _PanelSpec(id2, "Unit-Specific Slope Distribution (N=$(m.n_units))", js2))
        end

        ftitle = isempty(title) ?
            "Interactive Quantile Factor Panel (τ=$(_fmt(m.tau; digits=2)), r=$(m.r), $(m.slope))" : title
        _make_plot(panels; title=ftitle, ncols=(ncols <= 0 ? length(panels) : ncols))
    elseif view === :factors
        T_obs, r = size(m.factors)
        n_fac = min(r, max(1, max_factors))
        fac_names = ["Factor $k" for k in 1:n_fac]
        id = _next_plot_id("qpan_fac")
        data_json = _timeseries_data_json(m.factors[:, 1:n_fac], fac_names)
        s_json = _series_json(fac_names, _colors_for(fac_names); keys=["v$k" for k in 1:n_fac])
        refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
        js = _render_line_js(id, data_json, s_json; ref_lines_json=refs,
                             integer_x=true, xlabel="Period", ylabel="Factor Value")
        ftitle = isempty(title) ?
            "Interactive Quantile Panel Factors (τ=$(_fmt(m.tau; digits=2)), r=$(m.r))" : title
        _make_plot([_PanelSpec(id, _cap_title("Extracted Panel Factors", n_fac, r), js)];
                   title=ftitle, ncols=(ncols <= 0 ? 1 : ncols),
                   note=_cap_note("factors", n_fac, r, "max_factors"))
    else
        # :loadings
        N, r = size(m.loadings)
        shown_u = min(N, max(1, max_vars))
        n_fac = min(r, max(1, max_factors))
        fac_names = ["Factor $k" for k in 1:n_fac]
        rows = Vector{Pair{String,String}}[]
        for i in 1:shown_u
            row = Pair{String,String}["x" => _json("Unit $i")]
            for k in 1:n_fac
                push!(row, "f$k" => _json(m.loadings[i, k]))
            end
            push!(rows, row)
        end
        id = _next_plot_id("qpan_load")
        s_json = _series_json(fac_names, _colors_for(fac_names); keys=["f$k" for k in 1:n_fac])
        js = _render_bar_js(id, _json_array_of_objects(rows), s_json;
                            mode="grouped", orientation="h", xlabel="Loading λᵢ(τ)", ylabel="")
        note_u = _cap_note("units", shown_u, N, "max_vars")
        note_f = _cap_note("factors", n_fac, r, "max_factors")
        note = isempty(note_u) ? note_f : (isempty(note_f) ? note_u : note_u * " " * note_f)
        ftitle = isempty(title) ?
            "Interactive Quantile Panel Loadings (τ=$(_fmt(m.tau; digits=2)), r=$(m.r))" : title
        _make_plot([_PanelSpec(id, _cap_title("Unit Factor Loadings", shown_u, N), js)];
                   title=ftitle, ncols=(ncols <= 0 ? 1 : ncols), note=note)
    end

    save_path !== nothing && save_plot(p, save_path)
    return p
end
