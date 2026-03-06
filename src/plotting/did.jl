# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
plot_result methods for DiD types: DIDResult, EventStudyLP, BaconDecomposition.
"""

# =============================================================================
# Scatter Chart Renderer
# =============================================================================

"""
Generate D3.js code for a scatter plot with color-coded groups and optional
horizontal reference lines.

- `id`: SVG container element ID
- `data_json`: JSON array of {x, y, group} data points
- `groups_json`: JSON array of {name, color} group configs
- `ref_lines_json`: JSON array of {value, color, dash, axis} reference lines
  (axis = "x" or "y", default "y")
- `xlabel`, `ylabel`: axis labels
"""
function _render_scatter_js(id::String, data_json::String, groups_json::String;
                            ref_lines_json::String="[]",
                            xlabel::String="", ylabel::String="")
    """
(function() {
    const data = $(data_json);
    const groups = $(groups_json);
    const refLines = $(ref_lines_json);

    const container = d3.select('#$(id)');
    const W = Math.max(container.node().clientWidth - 24, 280);
    const margin = {top:10, right:15, bottom:35, left:55};
    const w = W - margin.left - margin.right;
    const h = Math.min(w * 0.6, 250);

    const svg = container.append('svg').attr('width', W).attr('height', h + margin.top + margin.bottom);
    const g = svg.append('g').attr('transform', 'translate('+margin.left+','+margin.top+')');

    // Compute domains
    const xVals = data.map(d => d.x).filter(v => v !== null);
    const yVals = data.map(d => d.y).filter(v => v !== null);
    refLines.forEach(r => {
        if(r.axis === 'x') xVals.push(r.value);
        else yVals.push(r.value);
    });

    const xExt = d3.extent(xVals);
    const xPad = (xExt[1] - xExt[0]) * 0.08 || 1;
    const x = d3.scaleLinear().domain([xExt[0] - xPad, xExt[1] + xPad]).range([0, w]);

    const yExt = d3.extent(yVals);
    const yPad = (yExt[1] - yExt[0]) * 0.08 || 0.01;
    const y = d3.scaleLinear().domain([yExt[0] - yPad, yExt[1] + yPad]).range([h, 0]);

    // Grid
    g.append('g').attr('class','grid').call(d3.axisLeft(y).tickSize(-w).tickFormat(''));

    // Reference lines
    refLines.forEach(r => {
        if(r.axis === 'x') {
            g.append('line').attr('x1', x(r.value)).attr('x2', x(r.value))
                .attr('y1', 0).attr('y2', h)
                .attr('stroke', r.color || '#999').attr('stroke-width', 1)
                .attr('stroke-dasharray', r.dash || '4,3');
        } else {
            g.append('line').attr('x1', 0).attr('x2', w)
                .attr('y1', y(r.value)).attr('y2', y(r.value))
                .attr('stroke', r.color || '#999').attr('stroke-width', 1)
                .attr('stroke-dasharray', r.dash || '4,3');
        }
    });

    // Build color map from groups
    const colorMap = {};
    groups.forEach(gr => { colorMap[gr.name] = gr.color; });

    // Scatter points
    g.selectAll('circle').data(data).join('circle')
        .attr('cx', d => x(d.x))
        .attr('cy', d => y(d.y))
        .attr('r', 5)
        .attr('fill', d => colorMap[d.group] || '#999')
        .attr('opacity', 0.8)
        .attr('stroke', '#fff')
        .attr('stroke-width', 0.5)
        .on('mouseover', function(evt, d) {
            d3.select(this).attr('r', 7).attr('opacity', 1);
            showTip(evt, '<b>'+d.group+'</b><br>x: '+fmt(d.x)+'<br>y: '+fmt(d.y));
        })
        .on('mouseout', function() {
            d3.select(this).attr('r', 5).attr('opacity', 0.8);
            hideTip();
        });

    // Axes
    g.append('g').attr('class','axis').attr('transform','translate(0,'+h+')')
        .call(d3.axisBottom(x).ticks(8));
    g.append('g').attr('class','axis').call(d3.axisLeft(y).ticks(6));

    if('$(xlabel)') g.append('text').attr('x',w/2).attr('y',h+30).attr('text-anchor','middle')
        .attr('font-size','11px').attr('fill','#666').text('$(xlabel)');
    if('$(ylabel)') g.append('text').attr('transform','rotate(-90)')
        .attr('x',-h/2).attr('y',-42).attr('text-anchor','middle')
        .attr('font-size','11px').attr('fill','#666').text('$(ylabel)');

    // Legend
    if(groups.length > 1) {
        const leg = g.append('g').attr('class','legend').attr('transform','translate(5,-5)');
        groups.forEach((gr, i) => {
            const gi = leg.append('g').attr('transform','translate('+(i*130)+',0)');
            gi.append('circle').attr('cx',6).attr('cy',0).attr('r',5)
                .attr('fill',gr.color).attr('opacity',0.8);
            gi.append('text').attr('x',16).attr('y',4).attr('font-size','10px')
                .attr('fill','#555').text(gr.name);
        });
    }
})();
"""
end

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

    # Reference lines: horizontal zero + vertical treatment time (x=0)
    refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"

    js = _render_line_js(id, data_json, s_json;
                         bands_json=bands, ref_lines_json=refs,
                         xlabel="Event Time", ylabel="ATT")

    # Add vertical dashed line at x=0 (treatment time) via custom JS appended
    # after the line chart renders
    vline_js = """
(function() {
    const container = d3.select('#$(id)');
    const svgEl = container.select('svg');
    const gEl = svgEl.select('g');
    const W = +svgEl.attr('width');
    const margin = {top:10, right:15, bottom:35, left:55};
    const w = W - margin.left - margin.right;
    const xVals = $(_json(collect(did.event_times)));
    const x = d3.scaleLinear().domain(d3.extent(xVals)).range([0,w]);
    gEl.append('line')
        .attr('x1', x(0)).attr('x2', x(0))
        .attr('y1', 0).attr('y2', gEl.select('.grid').node().getBBox().height)
        .attr('stroke', '#d62728').attr('stroke-width', 1)
        .attr('stroke-dasharray', '6,3');
})();
"""

    ptitle = "Event Study"
    panel = _PanelSpec(id, ptitle, js * "\n" * vline_js)

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
    refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"

    js = _render_line_js(id, data_json, s_json;
                         bands_json=bands, ref_lines_json=refs,
                         xlabel="Event Time", ylabel="Coefficient")

    # Vertical dashed line at x=0
    vline_js = """
(function() {
    const container = d3.select('#$(id)');
    const svgEl = container.select('svg');
    const gEl = svgEl.select('g');
    const W = +svgEl.attr('width');
    const margin = {top:10, right:15, bottom:35, left:55};
    const w = W - margin.left - margin.right;
    const xVals = $(_json(collect(eslp.event_times)));
    const x = d3.scaleLinear().domain(d3.extent(xVals)).range([0,w]);
    gEl.append('line')
        .attr('x1', x(0)).attr('x2', x(0))
        .attr('y1', 0).attr('y2', gEl.select('.grid').node().getBBox().height)
        .attr('stroke', '#d62728').attr('stroke-width', 1)
        .attr('stroke-dasharray', '6,3');
})();
"""

    method_label = eslp.clean_controls ? "LP-DiD" : "Event Study LP"
    ptitle = "Dynamic Treatment Effects ($method_label)"
    panel = _PanelSpec(id, ptitle, js * "\n" * vline_js)

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
        title = "Bacon Decomposition (TWFE = $(_json(bd.overall_att)))"
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

    ptitle = "HonestDiD Sensitivity (M\u0305 = $(_json(hd.Mbar)))"
    panel = _PanelSpec(id, ptitle, js)

    if isempty(title)
        title = "HonestDiD: Robust CI (breakdown = $(_json(hd.breakdown_value)))"
    end

    p = _make_plot([panel]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end
