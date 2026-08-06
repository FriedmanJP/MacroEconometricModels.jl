# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
plot_result methods for the policy-counterfactual result types (CF-22, #402):
PolicyCounterfactual, OPPResult, OPPSequence, CounterfactualMoments,
SpanningDiagnostic, CounterfactualHistory, ForecastSufficiency.
"""

const _CF_ZERO_REF = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"

# date-label tick map [{v, label}] for integer x positions
_cf_tick_json(labels::Vector{String}) =
    "[" * join(("{\"v\":$(i),\"label\":$(_json(labels[i]))}" for i in eachindex(labels)), ",") * "]"

"""
    plot_result(r::PolicyCounterfactual; vars=:all, ncols=0, title="",
                save_path=nothing, spanned_tol=0.05)

Baseline (dashed) vs counterfactual (solid, with band when draws were
propagated) per outcome/instrument. When the rule is NOT enforceable within
the shock span (`rel_residual > spanned_tol`) an implementation-error panel is
appended automatically — the honesty signal is visual too.
"""
function plot_result(r::PolicyCounterfactual{T};
                     vars::Union{Symbol,Vector{Symbol}}=:all,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     spanned_tol::Real=0.05) where {T}
    sel = vars === :all ? vcat(r.outcomes, r.instruments) : vars
    panels = _PanelSpec[]
    for sym in sel
        i = findfirst(==(sym), r.outcomes)
        k = i === nothing ? findfirst(==(sym), r.instruments) : nothing
        (i === nothing && k === nothing) && throw(ArgumentError(
            "variable :$sym not found among $(vcat(r.outcomes, r.instruments))"))
        base = i === nothing ? r.z_base[k] : r.x_base[i]
        cf = i === nothing ? r.z_cf[k] : r.x_cf[i]
        bands = i === nothing ? r.z_bands : r.x_bands
        bi = i === nothing ? k : i
        id = _next_plot_id("cfpath")
        cols = Pair{String,Vector{Float64}}["base" => Vector{Float64}(base),
                                            "cf" => Vector{Float64}(cf)]
        bjson = "[]"
        if bands !== nothing
            push!(cols, "lo" => Vector{Float64}(bands[bi][:, 1]))
            push!(cols, "hi" => Vector{Float64}(bands[bi][:, end]))
            bjson = "[{\"lo_key\":\"lo\",\"hi_key\":\"hi\",\"color\":\"$(_PLOT_COLORS[1])\",\"alpha\":$(_PLOT_CI_ALPHA)}]"
        end
        data = _keyed_rows_json(1:r.H, cols)
        s_json = _series_json(["Baseline", "Counterfactual"],
                              [_PLOT_COLORS[2], _PLOT_COLORS[1]];
                              keys=["base", "cf"], dash=["5,3", ""])
        js = _render_line_js(id, data, s_json; bands_json=bjson,
                             ref_lines_json=_CF_ZERO_REF, integer_x=true,
                             xlabel="Horizon", ylabel="Path")
        push!(panels, _PanelSpec(id, string(sym), js))
    end
    if r.rel_residual > spanned_tol && length(r.error_path) > 0
        id = _next_plot_id("cferr")
        data = _keyed_rows_json(1:length(r.error_path),
                                ["err" => Vector{Float64}(r.error_path)])
        s_json = _series_json(["Implementation error"], [_PLOT_ALERT]; keys=["err"])
        js = _render_line_js(id, data, s_json; ref_lines_json=_CF_ZERO_REF,
                             integer_x=true, xlabel="Stacked index",
                             ylabel="Rule violation")
        push!(panels, _PanelSpec(id,
            "Implementation error (rel. residual = $(round(r.rel_residual, sigdigits=3)))", js))
    end
    isempty(title) && (title = "Policy counterfactual: $(r.rule_name)")
    p = _make_plot(panels; title=title, ncols=ncols <= 0 ? min(2, length(panels)) : ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(r::OPPResult; view=:delta, ncols=0, title="", save_path=nothing)

`view = :delta`: the OPP components with their widest credible whiskers
(bands from `estimate_opp`; the subtitle carries the reversed band polarity).
`view = :paths`: announced vs perturbed objective-gap paths (and instrument
paths when a recommendation is available).
"""
function plot_result(r::OPPResult{T}; view::Symbol=:delta, ncols::Int=0,
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    view in (:delta, :paths) || throw(ArgumentError(
        "view: expected :delta or :paths, got :$view"))
    panels = _PanelSpec[]
    if view == :delta
        n_s = length(r.delta)
        id = _next_plot_id("oppdelta")
        if r.bands === nothing
            data = _keyed_rows_json([string(l) for l in r.shock_labels],
                                    ["delta" => Vector{Float64}(r.delta)])
            s_json = _series_json(["δ*"], [_PLOT_COLORS[1]]; keys=["delta"])
            js = _render_bar_js(id, data, s_json; mode="grouped",
                                xlabel="Shock direction", ylabel="δ")
            push!(panels, _PanelSpec(id, "OPP δ* (plug-in)", js))
        else
            lmax = maximum(keys(r.bands))
            data = _whisker_data_json(collect(1:n_s), Vector{Float64}(r.delta),
                                      Vector{Float64}(r.bands[lmax][:, 1]),
                                      Vector{Float64}(r.bands[lmax][:, 2]), 0;
                                      mark_reference=false)
            js = _render_whisker_js(id, data; point_label="δ (median)",
                                    ref_lines_json=_CF_ZERO_REF,
                                    xlabel="Shock direction", ylabel="δ")
            push!(panels, _PanelSpec(id,
                "OPP δ* with $(round(Int, 100 * lmax))% band — rejection at LOWER levels is the conservative choice", js))
        end
        isempty(title) && (title = "Optimal policy perturbation")
    else
        for (j, sym) in enumerate(r.outcomes)
            id = _next_plot_id("opppath")
            data = _keyed_rows_json(1:r.H, ["base" => Vector{Float64}(r.Y_base[j]),
                                            "opp" => Vector{Float64}(r.Y_opp[j])])
            s_json = _series_json(["Announced", "Perturbed"],
                                  [_PLOT_COLORS[2], _PLOT_COLORS[1]];
                                  keys=["base", "opp"], dash=["5,3", ""])
            js = _render_line_js(id, data, s_json; ref_lines_json=_CF_ZERO_REF,
                                 integer_x=true, xlabel="Horizon", ylabel="Gap")
            push!(panels, _PanelSpec(id, string(sym), js))
        end
        if r.P_base !== nothing
            for (k, sym) in enumerate(r.instruments)
                id = _next_plot_id("opppath")
                data = _keyed_rows_json(1:r.H,
                                        ["base" => Vector{Float64}(r.P_base[k]),
                                         "opp" => Vector{Float64}(r.P_opp[k])])
                s_json = _series_json(["Announced", "Recommended"],
                                      [_PLOT_COLORS[2], _PLOT_COLORS[1]];
                                      keys=["base", "opp"], dash=["5,3", ""])
                js = _render_line_js(id, data, s_json; integer_x=true,
                                     xlabel="Horizon", ylabel="Instrument")
                push!(panels, _PanelSpec(id, "$(sym) (instrument)", js))
            end
        end
        isempty(title) && (title = "OPP paths: announced vs perturbed")
    end
    p = _make_plot(panels; title=title, ncols=ncols <= 0 ? min(2, length(panels)) : ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(r::OPPSequence; view=:fan, shock=1, ncols=0, title="",
                save_path=nothing)

`view = :fan`: median OPP over decision dates with nested credible fans
(BM Fig.-1 style; requires simulated bands). `view = :decomposition`: δ vs
the time-consistent δ with the news/preference/aging revision parts.
`shock` selects the shock dimension.
"""
function plot_result(r::OPPSequence{T}; view::Symbol=:fan, shock::Int=1,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    view in (:fan, :decomposition) || throw(ArgumentError(
        "view: expected :fan or :decomposition, got :$view"))
    n_s, n_d = size(r.delta)
    1 <= shock <= n_s || throw(ArgumentError(
        "shock: expected 1..$n_s, got $shock"))
    ticks = _cf_tick_json(r.dates)
    panels = _PanelSpec[]
    if view == :fan
        if r.bands === nothing
            id = _next_plot_id("oppseq")
            data = _keyed_rows_json(1:n_d, ["delta" => Vector{Float64}(r.delta[shock, :])])
            s_json = _series_json(["δ"], [_PLOT_COLORS[1]]; keys=["delta"])
            js = _render_line_js(id, data, s_json; ref_lines_json=_CF_ZERO_REF,
                                 x_ticks_json=ticks, xlabel="Date", ylabel="δ")
            push!(panels, _PanelSpec(id, r.shock_labels[shock], js))
        else
            lv = sort(collect(keys(r.bands)))
            qlevels = vcat([(1 - l) / 2 for l in reverse(lv)], [(1 + l) / 2 for l in lv])
            Q = Matrix{Float64}(undef, n_d, 2 * length(lv))
            for (j, l) in enumerate(reverse(lv))
                Q[:, j] = r.bands[l][shock, 1, :]
            end
            for (j, l) in enumerate(lv)
                Q[:, length(lv)+j] = r.bands[l][shock, 2, :]
            end
            id = _next_plot_id("oppseq")
            data = _fan_data_json(Q, qlevels, Vector{Float64}(r.delta[shock, :]))
            fan = _fan_bands_json(qlevels)
            js = _render_fan_js(id, data, fan; median_key="med",
                                central_label="δ (median)",
                                ref_lines_json=_CF_ZERO_REF,
                                xlabel="Date", ylabel="δ")
            push!(panels, _PanelSpec(id,
                "$(r.shock_labels[shock]) — rejection at LOWER levels is the conservative choice", js))
        end
        isempty(title) && (title = "OPP sequence: $(r.loss_name)")
    else
        id = _next_plot_id("oppseq")
        data = _keyed_rows_json(1:n_d,
                                ["delta" => Vector{Float64}(r.delta[shock, :]),
                                 "tc" => Vector{Float64}(r.delta_tc[shock, :]),
                                 "news" => Vector{Float64}(r.news_part[shock, :]),
                                 "pref" => Vector{Float64}(r.pref_part[shock, :]),
                                 "aging" => Vector{Float64}(r.aging_part[shock, :])])
        s_json = _series_json(["δ", "δ (time-consistent)", "news", "preference", "aging"],
                              [_PLOT_COLORS[1], _PLOT_COLORS[2], _PLOT_COLORS[3],
                               _PLOT_COLORS[4], _PLOT_COLORS[5]];
                              keys=["delta", "tc", "news", "pref", "aging"],
                              dash=["", "5,3", "2,2", "2,2", "2,2"])
        js = _render_line_js(id, data, s_json; ref_lines_json=_CF_ZERO_REF,
                             x_ticks_json=ticks, xlabel="Date", ylabel="δ")
        push!(panels, _PanelSpec(id, "$(r.shock_labels[shock]) revision decomposition", js))
        isempty(title) && (title = "OPP time-consistency decomposition: $(r.loss_name)")
    end
    p = _make_plot(panels; title=title, ncols=ncols <= 0 ? 1 : ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(r::CounterfactualMoments; view=:sd, ncols=0, title="",
                save_path=nothing)

`view = :sd`: grouped bars of baseline vs counterfactual standard deviations
(band whiskers reported in `report()` when present). `view = :corr`: baseline
and counterfactual correlation heatmaps.
"""
function plot_result(r::CounterfactualMoments{T}; view::Symbol=:sd, ncols::Int=0,
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    view in (:sd, :corr) || throw(ArgumentError(
        "view: expected :sd or :corr, got :$view"))
    labels = String.(r.varnames)
    panels = _PanelSpec[]
    if view == :sd
        id = _next_plot_id("cfmom")
        data = _keyed_rows_json(labels, ["base" => Vector{Float64}(r.sd_base),
                                         "cf" => Vector{Float64}(r.sd_cf)])
        s_json = _series_json(["Baseline", "Counterfactual"],
                              [_PLOT_COLORS[2], _PLOT_COLORS[1]]; keys=["base", "cf"])
        js = _render_bar_js(id, data, s_json; mode="grouped",
                            xlabel="Variable", ylabel="Std. dev.")
        push!(panels, _PanelSpec(id, "Unconditional volatility", js))
        isempty(title) && (title = "Second-moment counterfactual: $(r.policy_name)")
    else
        for (nm, C) in (("Baseline", r.corr_base), ("Counterfactual", r.corr_cf))
            id = _next_plot_id("cfcorr")
            data = _matrix_heat_json(Matrix{Float64}(C), labels, labels)
            js = _render_heatmap_js(id, data, _json(labels), _json(labels);
                                    tip_label="Correlation", scale=:diverging,
                                    color_domain=[-1.0, 1.0])
            push!(panels, _PanelSpec(id, "$nm correlations", js))
        end
        isempty(title) && (title = "Correlation structure: $(r.policy_name)")
    end
    p = _make_plot(panels; title=title, ncols=ncols <= 0 ? length(panels) : ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(r::SpanningDiagnostic; ncols=0, title="", save_path=nothing)

Per-outcome overlay of the empirics-only and model-extrapolated
counterfactual paths — the two lines the diagnostic is about. The figure
title carries the loading gauge.
"""
function plot_result(r::SpanningDiagnostic{T}; ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    panels = _PanelSpec[]
    for (i, sym) in enumerate(r.outcomes)
        id = _next_plot_id("cfspan")
        H = length(r.x_cf_emp[i])
        data = _keyed_rows_json(1:H, ["emp" => Vector{Float64}(r.x_cf_emp[i]),
                                      "full" => Vector{Float64}(r.x_cf_full[i])])
        s_json = _series_json(["Empirics only", "Model extrapolated"],
                              [_PLOT_COLORS[1], _PLOT_COLORS[2]];
                              keys=["emp", "full"], dash=["", "5,3"])
        js = _render_line_js(id, data, s_json; ref_lines_json=_CF_ZERO_REF,
                             integer_x=true, xlabel="Horizon", ylabel="Path")
        push!(panels, _PanelSpec(id,
            "$(sym) (rel. gap = $(round(r.gap_rel[i], sigdigits=3)))", js))
    end
    if isempty(title)
        verdict = r.spanned ? "spanned" : "NOT spanned"
        title = "Spanning diagnostic: $verdict (loading inside span = $(round(r.loading_inside, sigdigits=3)))"
    end
    p = _make_plot(panels; title=title, ncols=ncols <= 0 ? min(2, length(panels)) : ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(r::CounterfactualHistory; vars=:all, ncols=0, title="",
                save_path=nothing)

Realized (solid) vs counterfactual (dashed, with band when draws were
propagated) paths over the window's calendar labels.
"""
function plot_result(r::CounterfactualHistory{T};
                     vars::Union{Symbol,Vector{Symbol}}=:all,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    sel = vars === :all ? r.varnames : vars
    ticks = _cf_tick_json(r.dates)
    n_d = length(r.dates)
    panels = _PanelSpec[]
    for sym in sel
        v = findfirst(==(sym), r.varnames)
        v === nothing && throw(ArgumentError(
            "variable :$sym not found among $(r.varnames)"))
        id = _next_plot_id("cfhist")
        cols = Pair{String,Vector{Float64}}["real" => Vector{Float64}(r.realized[:, v]),
                                            "cf" => Vector{Float64}(r.cf[:, v])]
        bjson = "[]"
        if r.cf_bands !== nothing
            push!(cols, "lo" => Vector{Float64}(r.cf_bands[:, v, 1]))
            push!(cols, "hi" => Vector{Float64}(r.cf_bands[:, v, end]))
            bjson = "[{\"lo_key\":\"lo\",\"hi_key\":\"hi\",\"color\":\"$(_PLOT_COLORS[1])\",\"alpha\":$(_PLOT_CI_ALPHA)}]"
        end
        data = _keyed_rows_json(1:n_d, cols)
        s_json = _series_json(["Realized", "Counterfactual"],
                              [_PLOT_COLORS[2], _PLOT_COLORS[1]];
                              keys=["real", "cf"], dash=["", "5,3"])
        js = _render_line_js(id, data, s_json; bands_json=bjson,
                             ref_lines_json=_CF_ZERO_REF, x_ticks_json=ticks,
                             xlabel="Date", ylabel="Level")
        push!(panels, _PanelSpec(id, string(sym), js))
    end
    isempty(title) && (title = "Historical counterfactual: $(r.policy_name)")
    p = _make_plot(panels; title=title, ncols=ncols <= 0 ? min(2, length(panels)) : ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(r::ForecastSufficiency; ncols=0, title="", save_path=nothing)

Forecast-error-variance ratios (Wold / full information) over the horizon,
one line per observable, with the sufficiency reference at 1.
"""
function plot_result(r::ForecastSufficiency{T}; ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    id = _next_plot_id("cfsuff")
    n_obs = length(r.observables)
    cols = Pair{String,Vector{Float64}}["v$j" => Vector{Float64}(r.fev_ratio[:, j])
                                        for j in 1:n_obs]
    data = _keyed_rows_json(1:r.H, cols)
    s_json = _series_json(String.(r.observables),
                          [_PLOT_COLORS[mod1(j, length(_PLOT_COLORS))] for j in 1:n_obs];
                          keys=["v$j" for j in 1:n_obs])
    refs = "[{\"value\":1,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js = _render_line_js(id, data, s_json; ref_lines_json=refs, integer_x=true,
                         xlabel="Horizon", ylabel="FEV ratio (Wold / full info)")
    inv = r.invertible ? "invertible" : "non-invertible"
    panels = [_PanelSpec(id, "Forecast sufficiency ($inv)", js)]
    isempty(title) && (title = "Forecast-sufficiency laboratory")
    p = _make_plot(panels; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end
