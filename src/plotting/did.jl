# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
plot_result methods for DiD types: DIDResult, EventStudyLP, BaconDecomposition.
"""

# NOTE: `_render_scatter_js` was relocated to render.jl (plotrule A1 — renderers
# live only in render.jl) in the PLT plotting overhaul (PLT-19). The call sites in
# this file (BaconDecomposition) use it unchanged.

# =============================================================================
# DIDResult — Event Study Coefficient Plot
# =============================================================================

"""
    plot_result(did::DIDResult; title="", save_path=nothing)

Plot DiD event study coefficients with confidence bands.

Displays ATT coefficients by event time with CI bands, a vertical dashed
line at treatment time (event time = 0), a horizontal zero reference line,
and the reference period marker.
"""
function plot_result(did::DIDResult{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    n_evt = length(did.event_times)
    id = _next_plot_id("did_es")

    # Build data JSON
    rows = Vector{Pair{String,String}}[]
    for i in 1:n_evt
        push!(rows, [
            "x" => _json(did.event_times[i]),
            "att" => _json(did.att[i]),
            "ci_lo" => _json(did.ci_lower[i]),
            "ci_hi" => _json(did.ci_upper[i]),
            "zero" => "0"
        ])
    end
    data_json = _json_array_of_objects(rows)

    s_json = _series_json(["ATT"], [_PLOT_COLORS[1]]; keys=["att"])
    bands = "[{\"lo_key\":\"ci_lo\",\"hi_key\":\"ci_hi\",\"color\":\"$(_PLOT_COLORS[1])\",\"alpha\":$(_PLOT_CI_ALPHA)}]"

    # Reference lines: horizontal zero + vertical treatment line at event time 0
    # (the line renderer's axis:"x" ref-line option — plotrule A4, no scale-clone).
    refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}," *
           "{\"value\":0,\"color\":\"#d62728\",\"dash\":\"6,3\",\"axis\":\"x\"}]"

    js = _render_line_js(id, data_json, s_json;
                         bands_json=bands, ref_lines_json=refs,
                         xlabel="Event Time", ylabel="ATT")

    ptitle = "Event Study"
    panel = _PanelSpec(id, ptitle, js)

    if isempty(title)
        method_str = did.method == :twfe ? "TWFE" :
                     did.method == :callaway_santanna ? "Callaway-Sant'Anna" :
                     did.method == :sun_abraham ? "Sun-Abraham" :
                     did.method == :bjs ? "BJS" :
                     did.method == :did_multiplegt ? "dCDH" :
                     string(did.method)
        title = "DiD Event Study: $(did.outcome_var) ($method_str)"
    end

    p = _make_plot([panel]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# EventStudyLP — LP-based Event Study Plot
# =============================================================================

"""
    plot_result(eslp::EventStudyLP; title="", save_path=nothing)

Plot LP-based event study dynamic treatment effects with confidence bands.

Same style as DIDResult but uses coefficients from LP regressions.
Title includes "(LP-DiD)" if clean_controls is true, else "(Event Study LP)".
"""
function plot_result(eslp::EventStudyLP{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    n_evt = length(eslp.event_times)
    id = _next_plot_id("eslp")

    # Build data JSON
    rows = Vector{Pair{String,String}}[]
    for i in 1:n_evt
        push!(rows, [
            "x" => _json(eslp.event_times[i]),
            "coef" => _json(eslp.coefficients[i]),
            "ci_lo" => _json(eslp.ci_lower[i]),
            "ci_hi" => _json(eslp.ci_upper[i]),
            "zero" => "0"
        ])
    end
    data_json = _json_array_of_objects(rows)

    label = eslp.clean_controls ? "LP-DiD" : "LP"
    s_json = _series_json([label], [_PLOT_COLORS[1]]; keys=["coef"])
    bands = "[{\"lo_key\":\"ci_lo\",\"hi_key\":\"ci_hi\",\"color\":\"$(_PLOT_COLORS[1])\",\"alpha\":$(_PLOT_CI_ALPHA)}]"
    # horizontal zero + vertical treatment line at event time 0 (axis:"x" ref-line
    # option — plotrule A4, no scale-clone).
    refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}," *
           "{\"value\":0,\"color\":\"#d62728\",\"dash\":\"6,3\",\"axis\":\"x\"}]"

    js = _render_line_js(id, data_json, s_json;
                         bands_json=bands, ref_lines_json=refs,
                         xlabel="Event Time", ylabel="Coefficient")

    method_label = eslp.clean_controls ? "LP-DiD" : "Event Study LP"
    ptitle = "Dynamic Treatment Effects ($method_label)"
    panel = _PanelSpec(id, ptitle, js)

    if isempty(title)
        title = "$(eslp.outcome_var) ($(method_label))"
    end

    p = _make_plot([panel]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

function plot_result(r::LPDiDResult{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    eslp = EventStudyLP{T}(r.coefficients, r.se, r.ci_lower, r.ci_upper,
                           r.event_times, r.reference_period,
                           [zeros(T,1,1) for _ in r.event_times],
                           [zeros(T,0,1) for _ in r.event_times],
                           r.vcov, r.nobs_per_horizon,
                           r.outcome_var, r.treatment_var,
                           r.T_obs, r.n_groups, r.ylags, r.pre_window, r.post_window,
                           true, r.cluster, r.conf_level, r.data)
    plot_result(eslp; title=isempty(title) ? "LP-DiD (Dube et al. 2025)" : title,
                save_path=save_path)
end

# =============================================================================
# BaconDecomposition — Scatter Plot
# =============================================================================

"""
    plot_result(bd::BaconDecomposition; title="", save_path=nothing)

Plot Bacon decomposition as a scatter plot.

X-axis = 2x2 DiD estimate, Y-axis = weight, colored by comparison type.
Horizontal dashed line at the overall TWFE estimate.
"""
function plot_result(bd::BaconDecomposition{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    n = length(bd.estimates)
    id = _next_plot_id("bacon")

    # Map comparison types to display names
    type_names = Dict{Symbol,String}(
        :earlier_vs_later => "Earlier vs Later",
        :later_vs_earlier => "Later vs Earlier",
        :treated_vs_untreated => "Treated vs Untreated"
    )

    # Build data JSON: {x: estimate, y: weight, group: type_name}
    rows = String[]
    unique_types = unique(bd.comparison_type)
    for i in 1:n
        gname = get(type_names, bd.comparison_type[i], string(bd.comparison_type[i]))
        push!(rows, "{\"x\":$(_json(bd.estimates[i])),\"y\":$(_json(bd.weights[i])),\"group\":$(_json(gname))}")
    end
    data_json = "[" * join(rows, ",\n") * "]"

    # Build groups JSON
    group_colors = String[]
    group_names_list = String[]
    for (gi, gt) in enumerate(unique_types)
        gname = get(type_names, gt, string(gt))
        push!(group_names_list, gname)
        push!(group_colors, _PLOT_COLORS[mod1(gi, length(_PLOT_COLORS))])
    end
    groups_parts = String[]
    for (gi, gname) in enumerate(group_names_list)
        push!(groups_parts, "{\"name\":$(_json(gname)),\"color\":$(_json(group_colors[gi]))}")
    end
    groups_json = "[" * join(groups_parts, ",") * "]"

    # Reference line at overall ATT (horizontal)
    refs = "[{\"value\":$(_json(bd.overall_att)),\"color\":\"#d62728\",\"dash\":\"6,3\",\"axis\":\"x\"}]"

    js = _render_scatter_js(id, data_json, groups_json;
                            ref_lines_json=refs,
                            xlabel="2x2 DiD Estimate", ylabel="Weight")

    ptitle = "Estimate vs Weight"
    panel = _PanelSpec(id, ptitle, js)

    if isempty(title)
        title = "Bacon Decomposition (TWFE = $(_fmt(bd.overall_att; digits=3)))"
    end

    p = _make_plot([panel]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# HonestDiDResult — Dual CI Band Plot
# =============================================================================

"""
    plot_result(hd::HonestDiDResult; title="", save_path=nothing)

Plot HonestDiD sensitivity analysis with dual CI bands.

Shows original confidence intervals (narrow) and robust confidence intervals
(wide, accounting for parallel trends violations), with breakdown value
annotated.
"""
function plot_result(hd::HonestDiDResult{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    n_evt = length(hd.post_event_times)
    id = _next_plot_id("honest_did")

    # Build data JSON
    rows = Vector{Pair{String,String}}[]
    for i in 1:n_evt
        push!(rows, [
            "x" => _json(hd.post_event_times[i]),
            "att" => _json(hd.post_att[i]),
            "orig_lo" => _json(hd.original_ci_lower[i]),
            "orig_hi" => _json(hd.original_ci_upper[i]),
            "robust_lo" => _json(hd.robust_ci_lower[i]),
            "robust_hi" => _json(hd.robust_ci_upper[i]),
            "zero" => "0"
        ])
    end
    data_json = _json_array_of_objects(rows)

    s_json = _series_json(["ATT"], [_PLOT_COLORS[1]]; keys=["att"])
    bands = "[{\"lo_key\":\"robust_lo\",\"hi_key\":\"robust_hi\",\"color\":\"$(_PLOT_COLORS[2])\",\"alpha\":0.15}," *
            "{\"lo_key\":\"orig_lo\",\"hi_key\":\"orig_hi\",\"color\":\"$(_PLOT_COLORS[1])\",\"alpha\":$(_PLOT_CI_ALPHA)}]"
    refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"

    js = _render_line_js(id, data_json, s_json;
                         bands_json=bands, ref_lines_json=refs,
                         xlabel="Event Time", ylabel="ATT")

    ptitle = hd.restriction == :sd ?
        "Honest DiD Sensitivity (\u0394^SD, M = $(_fmt(hd.M; digits=3)))" :
        "Honest DiD Sensitivity (\u0394^RM, M\u0305 = $(_fmt(hd.Mbar; digits=3)))"
    panel = _PanelSpec(id, ptitle, js)

    if isempty(title)
        title = "Honest DiD: Robust CI (breakdown = $(_fmt(hd.breakdown_value; digits=3)))"
    end

    p = _make_plot([panel]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end
