# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
plot_result methods for the nonlinear ARDL (NARDL, EV-09): cumulative dynamic
multipliers m⁺_h / m⁻_h with their bootstrap bands and the asymmetry (difference)
curve, one panel per asymmetric regressor.
"""

# =============================================================================
# NARDLMultipliers — one panel per asymmetric regressor
# =============================================================================

function _plot_nardl_multiplier_panel(mm::NARDLMultipliers{T}, i::Int) where {T}
    id = _next_plot_id("nardl_mult")
    has_ci = mm.nreps > 0
    rows = Vector{Pair{String,String}}[]
    for (h, hz) in enumerate(mm.horizons)
        row = Pair{String,String}[
            "x"    => _json(hz),
            "pos"  => _json(mm.m_pos[i, h]),
            "neg"  => _json(mm.m_neg[i, h]),
            "diff" => _json(mm.m_diff[i, h]),
        ]
        if has_ci
            push!(row, "plo" => _json(mm.m_pos_lo[i, h]))
            push!(row, "phi" => _json(mm.m_pos_hi[i, h]))
            push!(row, "nlo" => _json(mm.m_neg_lo[i, h]))
            push!(row, "nhi" => _json(mm.m_neg_hi[i, h]))
        end
        push!(rows, row)
    end
    data = _json_array_of_objects(rows)

    series = _series_json(["m⁺ (positive)", "m⁻ (negative)", "asymmetry m⁺−m⁻"],
                          [_PLOT_COLORS[1], _PLOT_COLORS[2], _PLOT_COLORS[3]];
                          keys=["pos", "neg", "diff"], dash=["", "", "6,3"])
    bands = has_ci ?
        "[{\"lo_key\":\"plo\",\"hi_key\":\"phi\",\"color\":\"$(_PLOT_COLORS[1])\"}," *
        "{\"lo_key\":\"nlo\",\"hi_key\":\"nhi\",\"color\":\"$(_PLOT_COLORS[2])\"}]" :
        "[]"
    refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"

    js = _render_line_js(id, data, series; bands_json=bands, ref_lines_json=refs,
                         xlabel="Horizon h", ylabel="Cumulative response of y")
    _PanelSpec(id, "Dynamic multipliers: $(mm.reg_names[i])", js)
end

"""
    plot_result(mm::NARDLMultipliers; view=:multipliers, title="", save_path=nothing)

Plot the NARDL cumulative dynamic multipliers. One panel per asymmetric regressor
shows `m⁺_h` and `m⁻_h` (with their bootstrap bands, if present) and the asymmetry
curve `m⁺_h − m⁻_h`, converging to θ⁺ / θ⁻ as `h` grows.
"""
function plot_result(mm::NARDLMultipliers{T}; view::Symbol=:multipliers,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    view == :multipliers ||
        throw(ArgumentError("view must be :multipliers for NARDLMultipliers; got :$view"))
    panels = [_plot_nardl_multiplier_panel(mm, i) for i in eachindex(mm.reg_names)]
    isempty(title) && (title = "NARDL Cumulative Dynamic Multipliers")
    ncols = length(panels) == 1 ? 1 : 2
    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(m::NARDLModel; view=:multipliers, H=24, bootstrap=false, title="",
                save_path=nothing, kwargs...)

Convenience: compute the cumulative dynamic multipliers of a fitted
[`NARDLModel`](@ref) out to horizon `H` and plot them. Defaults to `bootstrap=false`
for a fast preview; pass `bootstrap=true` (and an `rng`) for percentile bands.

- `view=:multipliers` (default) — the cumulative dynamic multipliers.
- `view=:diagnostics` — the shared four-panel residual diagnostics of the underlying
  ARDL fit (PLT-24).
"""
function plot_result(m::NARDLModel{T}; view::Symbol=:multipliers, H::Int=24,
                     bootstrap::Bool=false, acf_lags::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing, kwargs...) where {T}
    if view === :diagnostics
        resid = Float64[Float64(v) for v in residuals(m.ardl)]
        fitted = Float64[Float64(v) for v in m.ardl.fitted]
        panels = _residual_diagnostics_panels(resid, fitted; varname=m.yname, acf_lags=acf_lags)
        isempty(title) && (title = "NARDL Residual Diagnostics")
        p = _make_plot(panels; title=title, ncols=2)
        save_path !== nothing && save_plot(p, save_path)
        return p
    end
    view == :multipliers ||
        throw(ArgumentError("view must be :multipliers or :diagnostics for NARDLModel; got :$view"))
    mm = dynamic_multipliers(m, H; bootstrap=bootstrap, kwargs...)
    plot_result(mm; view=:multipliers, title=title, save_path=save_path)
end

# =============================================================================
# ARDL levels model / bounds test / long-run multipliers (#841 PR6)
# =============================================================================

"""
    plot_result(m::ARDLModel; view=:coef, title="", save_path=nothing, conf_level=0.95)

ARDL levels-model coefficients as a horizontal dot-and-whisker plot
(`β ± z·SE` at confidence level `conf_level`, default 0.95; SE from
`diag(vcov)`), with a zero reference line. Lag orders and case are stated in
the panel subtitle. Only `view=:coef` is defined.
"""
function plot_result(m::ARDLModel{T}; view::Symbol=:coef, title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     conf_level::Real=0.95) where {T}
    view === :coef ||
        throw(ArgumentError("unknown view :$view; valid views: :coef"))
    z = _conf_z(conf_level)
    panel = _coef_panel("ardl_coef", m.coefnames, m.coef, _diag_se(m.vcov);
                        z=z, ptitle="Coefficients (ARDL($(m.p), ($(join(m.q, ","))), " *
                        "case $(m.case), $(_ci_pct(conf_level))% CI)")
    ftitle = isempty(title) ? "ARDL Coefficients ($(m.yname))" : title
    p = _make_plot([panel]; title=ftitle, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(b::ARDLBoundsTest; title="", save_path=nothing)

Pesaran–Shin–Smith bounds test: the F and t statistics as grouped bars against
their I(0)/I(1) critical-value bands at the stored test `level`. A statistic
above the I(1) bound rejects no-cointegration; below I(0) fails to reject;
inside is inconclusive. Case, regressor count, and both decisions are stated
in the subtitle. Bands exist only at the stored level, so no `level` keyword
applies.
"""
function plot_result(b::ARDLBoundsTest{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    idx = findfirst(==(b.level), b.levels)
    if idx === nothing
        idx = argmin(abs.(b.levels .- b.level))
    end
    rows = [["x" => _json("F"), "stat" => _json(b.fstat),
             "lo" => _json(b.f_lower[idx]), "hi" => _json(b.f_upper[idx])],
            ["x" => _json("t"), "stat" => _json(b.tstat),
             "lo" => _json(b.t_lower[idx]), "hi" => _json(b.t_upper[idx])]]
    data_json = _json_array_of_objects(rows)
    s_json = _series_json(["Statistic", "I(0)", "I(1)"],
                          [_PLOT_SERIES[1], _PLOT_SERIES[2], _PLOT_ALERT];
                          keys=["stat", "lo", "hi"])
    id = _next_plot_id("ardlbounds")
    js = _render_bar_js(id, data_json, s_json; mode="grouped", orientation="v",
                        xlabel="Statistic", ylabel="Value")
    isempty(title) && (title = "ARDL Bounds Test (case $(b.case))")
    ptitle = "k=$(b.k), level=$(b.level): F $(b.f_decision), t $(b.t_decision)"
    p = _make_plot([_PanelSpec(id, ptitle, js)]; title=title)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(lr::ARDLLongRun; title="", save_path=nothing, conf_level=0.95)

ARDL long-run multipliers as a horizontal dot-and-whisker plot
(`θ ± z·SE` at confidence level `conf_level`, default 0.95), with a zero
reference line.
"""
function plot_result(lr::ARDLLongRun{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     conf_level::Real=0.95) where {T}
    z = _conf_z(conf_level)
    panel = _coef_panel("ardl_lr", lr.varnames, lr.theta, lr.se;
                        z=z, ptitle="Long-Run Multipliers ($(_ci_pct(conf_level))% CI)")
    ftitle = isempty(title) ? "ARDL Long-Run Multipliers" : title
    p = _make_plot([panel]; title=ftitle, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(s::NARDLSymmetryTest; title="", save_path=nothing, level=5)

NARDL long/short-run symmetry Wald tests: per-regressor χ² and F p-values as
bars against `α` (default 5%). Long-run (`θ⁺ = θ⁻`) and short-run (cumulative
dynamic multipliers) symmetry share the figure; positive/negative long-run
multipliers are stated in the subtitle alongside the rejection count.
"""
function plot_result(s::NARDLSymmetryTest{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     level::Int=5) where {T}
    labels = String[]
    pvs = T[]
    for (i, nm) in enumerate(s.reg_names)
        push!(labels, "LR χ²: $nm"); push!(pvs, s.lr_p_chi2[i])
        push!(labels, "LR F: $nm"); push!(pvs, s.lr_p_f[i])
        push!(labels, "SR χ²: $nm"); push!(pvs, s.sr_p_chi2[i])
        push!(labels, "SR F: $nm"); push!(pvs, s.sr_p_f[i])
    end
    th = join(["$nm: $(_fmt(s.theta_pos[i]; digits=3))/$(_fmt(s.theta_neg[i]; digits=3))"
               for (i, nm) in enumerate(s.reg_names)], ", ")
    _pvalue_bar_plot("nardlsym", labels, pvs, "NARDL Symmetry Tests",
        "θ⁺/θ⁻: $(th)";
        title=title, save_path=save_path, level=level)
end
