# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

"""
plot_result methods for cross-sectional regression types: RegModel, LogitModel,
ProbitModel, MarginalEffects.
"""

# =============================================================================
# RegModel — OLS / WLS / IV Regression Diagnostics
# =============================================================================

"""
    plot_result(m::RegModel{T}; title="", save_path=nothing)

Plot OLS/WLS/IV regression diagnostics: residuals vs fitted, residual histogram,
and Q-Q plot.
"""
function plot_result(m::RegModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    resid = m.residuals
    fitted_vals = m.fitted
    n = length(resid)

    # Panel 1: Residuals vs Fitted (scatter)
    id1 = _next_plot_id("reg_rvf")
    scatter_rows = String[]
    for i in 1:n
        push!(scatter_rows, "{\"x\":$(_json(fitted_vals[i])),\"y\":$(_json(resid[i])),\"group\":\"Residuals\"}")
    end
    data1 = "[" * join(scatter_rows, ",\n") * "]"
    groups1 = "[{\"name\":\"Residuals\",\"color\":\"$(_PLOT_COLORS[1])\"}]"
    refs1 = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js1 = _render_scatter_js(id1, data1, groups1;
                             ref_lines_json=refs1,
                             xlabel="Fitted Values", ylabel="Residuals")
    p1 = _PanelSpec(id1, "Residuals vs Fitted", js1)

    # Panel 2: Residual Histogram (bar)
    id2 = _next_plot_id("reg_hist")
    n_bins = 20
    rmin = minimum(resid)
    rmax = maximum(resid)
    bin_width = (rmax - rmin) / n_bins
    if bin_width <= zero(T)
        bin_width = one(T)
    end
    bin_counts = zeros(Int, n_bins)
    for r in resid
        idx = clamp(floor(Int, (r - rmin) / bin_width) + 1, 1, n_bins)
        bin_counts[idx] += 1
    end
    hist_rows = Vector{Pair{String,String}}[]
    for i in 1:n_bins
        lo = rmin + (i - 1) * bin_width
        hi = rmin + i * bin_width
        label = string(round((lo + hi) / 2; digits=2))
        push!(hist_rows, ["x" => _json(label), "s1" => _json(bin_counts[i])])
    end
    data2 = _json_array_of_objects(hist_rows)
    s2 = _series_json(["Count"], [_PLOT_COLORS[2]]; keys=["s1"])
    js2 = _render_bar_js(id2, data2, s2; mode="stacked", ylabel="Frequency")
    p2 = _PanelSpec(id2, "Residual Histogram", js2)

    # Panel 3: Q-Q Plot (scatter)
    id3 = _next_plot_id("reg_qq")
    sorted_resid = sort(resid)
    qq_rows = String[]
    for i in 1:n
        theoretical = quantile(Normal(), (i - T(0.5)) / n)
        push!(qq_rows, "{\"x\":$(_json(theoretical)),\"y\":$(_json(sorted_resid[i])),\"group\":\"Q-Q\"}")
    end
    data3 = "[" * join(qq_rows, ",\n") * "]"
    groups3 = "[{\"name\":\"Q-Q\",\"color\":\"$(_PLOT_COLORS[3])\"}]"

    # 45-degree reference line: from min to max of theoretical quantiles
    q_min = quantile(Normal(), T(0.5) / n)
    q_max = quantile(Normal(), (n - T(0.5)) / n)
    refs3 = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"

    # Build scatter with a 45-degree line overlay
    js3_scatter = _render_scatter_js(id3, data3, groups3;
                                     ref_lines_json=refs3,
                                     xlabel="Theoretical Quantiles",
                                     ylabel="Sample Quantiles")

    # Append 45-degree line via custom D3 snippet
    resid_std = std(resid)
    resid_mean = mean(resid)
    js3_line = """
(function() {
    const container = d3.select('#$(id3)');
    const svgEl = container.select('svg');
    const gEl = svgEl.select('g');
    const W = +svgEl.attr('width');
    const margin = {top:10, right:15, bottom:35, left:55};
    const w = W - margin.left - margin.right;
    const h = Math.min(w * 0.6, 250);

    const qMin = $(_json(q_min));
    const qMax = $(_json(q_max));
    const rMean = $(_json(resid_mean));
    const rStd = $(_json(resid_std));

    // x domain matches scatter
    const data = $(data3);
    const xVals = data.map(d => d.x);
    const yVals = data.map(d => d.y);
    yVals.push(0);
    const xExt = d3.extent(xVals);
    const xPad = (xExt[1] - xExt[0]) * 0.08 || 1;
    const x = d3.scaleLinear().domain([xExt[0] - xPad, xExt[1] + xPad]).range([0, w]);
    const yExt = d3.extent(yVals);
    const yPad = (yExt[1] - yExt[0]) * 0.08 || 0.01;
    const y = d3.scaleLinear().domain([yExt[0] - yPad, yExt[1] + yPad]).range([h, 0]);

    // 45-degree line: sample quantile = mean + std * theoretical quantile
    const x1 = qMin, x2 = qMax;
    const y1 = rMean + rStd * x1;
    const y2 = rMean + rStd * x2;
    gEl.append('line')
        .attr('x1', x(x1)).attr('x2', x(x2))
        .attr('y1', y(y1)).attr('y2', y(y2))
        .attr('stroke', '#d62728').attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '6,3');
})();
"""
    js3 = js3_scatter * "\n" * js3_line
    p3 = _PanelSpec(id3, "Normal Q-Q Plot", js3)

    if isempty(title)
        method_str = m.method == :ols ? "OLS" : m.method == :wls ? "WLS" : "IV/2SLS"
        title = "$method_str Regression Diagnostics"
    end

    p = _make_plot([p1, p2, p3]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# Binary Choice Shared Helper
# =============================================================================

"""
Generate 2-panel binary choice model diagnostic plot:
1. Sorted predicted probabilities colored by outcome
2. Distribution of predicted probabilities by outcome (grouped bar)
"""
function _plot_binary_choice(y::AbstractVector{T}, fitted_probs::AbstractVector{T},
                              model_name::String; title::String="",
                              save_path::Union{String,Nothing}=nothing) where {T}
    n = length(y)

    # Panel 1: Sorted predicted probabilities colored by outcome
    id1 = _next_plot_id("bc_sort")
    order = sortperm(fitted_probs)
    scatter_rows = String[]
    for (rank, idx) in enumerate(order)
        grp = y[idx] > T(0.5) ? "y = 1" : "y = 0"
        push!(scatter_rows, "{\"x\":$(_json(rank)),\"y\":$(_json(fitted_probs[idx])),\"group\":$(_json(grp))}")
    end
    data1 = "[" * join(scatter_rows, ",\n") * "]"
    groups1 = "[{\"name\":\"y = 1\",\"color\":\"$(_PLOT_COLORS[1])\"},{\"name\":\"y = 0\",\"color\":\"$(_PLOT_COLORS[4])\"}]"
    refs1 = "[{\"value\":0.5,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js1 = _render_scatter_js(id1, data1, groups1;
                             ref_lines_json=refs1,
                             xlabel="Observation (sorted by P\u0302)",
                             ylabel="Predicted Probability")
    p1 = _PanelSpec(id1, "Sorted Predicted Probabilities", js1)

    # Panel 2: Distribution by outcome (grouped bar)
    id2 = _next_plot_id("bc_dist")
    n_bins = 10
    bin_counts_0 = zeros(Int, n_bins)
    bin_counts_1 = zeros(Int, n_bins)
    for i in 1:n
        # Map probability to bin [1, n_bins]
        b = clamp(floor(Int, fitted_probs[i] * n_bins) + 1, 1, n_bins)
        if y[i] > T(0.5)
            bin_counts_1[b] += 1
        else
            bin_counts_0[b] += 1
        end
    end
    bar_rows = Vector{Pair{String,String}}[]
    for b in 1:n_bins
        lo = (b - 1) / n_bins
        hi = b / n_bins
        label = string(round((lo + hi) / 2; digits=2))
        push!(bar_rows, ["x" => _json(label), "s0" => _json(bin_counts_0[b]), "s1" => _json(bin_counts_1[b])])
    end
    data2 = _json_array_of_objects(bar_rows)
    s2 = _series_json(["y = 0", "y = 1"], [_PLOT_COLORS[4], _PLOT_COLORS[1]]; keys=["s0", "s1"])
    js2 = _render_bar_js(id2, data2, s2; mode="grouped",
                         xlabel="Predicted Probability", ylabel="Count")
    p2 = _PanelSpec(id2, "Distribution by Outcome", js2)

    if isempty(title)
        title = "$model_name Model Diagnostics"
    end

    p = _make_plot([p1, p2]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# LogitModel
# =============================================================================

"""
    plot_result(m::LogitModel{T}; title="", save_path=nothing)

Plot logit model diagnostics: sorted predicted probabilities by outcome and
distribution of predictions by outcome group.
"""
function plot_result(m::LogitModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    _plot_binary_choice(m.y, m.fitted, "Logit"; title=title, save_path=save_path)
end

# =============================================================================
# ProbitModel
# =============================================================================

"""
    plot_result(m::ProbitModel{T}; title="", save_path=nothing)

Plot probit model diagnostics: sorted predicted probabilities by outcome and
distribution of predictions by outcome group.
"""
function plot_result(m::ProbitModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    _plot_binary_choice(m.y, m.fitted, "Probit"; title=title, save_path=save_path)
end

# =============================================================================
# Coefficient Plot Renderer (for MarginalEffects)
# =============================================================================

"""
Generate D3.js code for a horizontal coefficient plot with CI error bars.

- `id`: SVG container element ID
- `data_json`: JSON array of {name, effect, ci_lo, ci_hi} objects
- `xlabel`, `ylabel`: axis labels
"""
function _render_coef_plot_js(id::String, data_json::String;
                               xlabel::String="", ylabel::String="")
    """
(function() {
    const data = $(data_json);

    const container = d3.select('#$(id)');
    const W = Math.max(container.node().clientWidth - 24, 280);
    const margin = {top:10, right:15, bottom:35, left:100};
    const w = W - margin.left - margin.right;
    const h = Math.max(data.length * 28, 120);

    const svg = container.append('svg').attr('width', W).attr('height', h + margin.top + margin.bottom);
    const g = svg.append('g').attr('transform', 'translate('+margin.left+','+margin.top+')');

    // Scales
    const allX = [];
    data.forEach(d => { allX.push(d.effect, d.ci_lo, d.ci_hi); });
    allX.push(0);
    const xExt = d3.extent(allX);
    const xPad = (xExt[1] - xExt[0]) * 0.12 || 0.1;
    const x = d3.scaleLinear().domain([xExt[0] - xPad, xExt[1] + xPad]).range([0, w]);
    const y = d3.scaleBand().domain(data.map(d => d.name)).range([0, h]).padding(0.3);

    // Grid
    g.append('g').attr('class','grid')
        .call(d3.axisBottom(x).tickSize(h).tickFormat(''))
        .attr('transform','translate(0,0)');

    // Zero reference line
    g.append('line')
        .attr('x1', x(0)).attr('x2', x(0))
        .attr('y1', 0).attr('y2', h)
        .attr('stroke', '#d62728').attr('stroke-width', 1)
        .attr('stroke-dasharray', '6,3');

    // CI whiskers (horizontal lines)
    g.selectAll('.ci-line').data(data).join('line')
        .attr('class', 'ci-line')
        .attr('x1', d => x(d.ci_lo))
        .attr('x2', d => x(d.ci_hi))
        .attr('y1', d => y(d.name) + y.bandwidth()/2)
        .attr('y2', d => y(d.name) + y.bandwidth()/2)
        .attr('stroke', '$(_PLOT_COLORS[1])')
        .attr('stroke-width', 1.5);

    // CI caps (vertical lines at ends)
    const capH = y.bandwidth() * 0.4;
    g.selectAll('.ci-cap-lo').data(data).join('line')
        .attr('class', 'ci-cap-lo')
        .attr('x1', d => x(d.ci_lo)).attr('x2', d => x(d.ci_lo))
        .attr('y1', d => y(d.name) + y.bandwidth()/2 - capH/2)
        .attr('y2', d => y(d.name) + y.bandwidth()/2 + capH/2)
        .attr('stroke', '$(_PLOT_COLORS[1])').attr('stroke-width', 1.5);
    g.selectAll('.ci-cap-hi').data(data).join('line')
        .attr('class', 'ci-cap-hi')
        .attr('x1', d => x(d.ci_hi)).attr('x2', d => x(d.ci_hi))
        .attr('y1', d => y(d.name) + y.bandwidth()/2 - capH/2)
        .attr('y2', d => y(d.name) + y.bandwidth()/2 + capH/2)
        .attr('stroke', '$(_PLOT_COLORS[1])').attr('stroke-width', 1.5);

    // Effect bars (horizontal from 0 to effect)
    g.selectAll('.effect-bar').data(data).join('rect')
        .attr('class', 'effect-bar')
        .attr('x', d => x(Math.min(0, d.effect)))
        .attr('y', d => y(d.name) + y.bandwidth() * 0.15)
        .attr('width', d => Math.abs(x(d.effect) - x(0)))
        .attr('height', y.bandwidth() * 0.7)
        .attr('fill', '$(_PLOT_COLORS[1])')
        .attr('opacity', 0.7);

    // Effect dots (circles at effect value)
    g.selectAll('.effect-dot').data(data).join('circle')
        .attr('class', 'effect-dot')
        .attr('cx', d => x(d.effect))
        .attr('cy', d => y(d.name) + y.bandwidth()/2)
        .attr('r', 4)
        .attr('fill', '$(_PLOT_COLORS[1])')
        .attr('stroke', '#fff')
        .attr('stroke-width', 1);

    // Axes
    g.append('g').attr('class','axis').attr('transform','translate(0,'+h+')')
        .call(d3.axisBottom(x).ticks(8));
    g.append('g').attr('class','axis')
        .call(d3.axisLeft(y));

    if('$(xlabel)') g.append('text').attr('x',w/2).attr('y',h+30).attr('text-anchor','middle')
        .attr('font-size','11px').attr('fill','#666').text('$(xlabel)');
    if('$(ylabel)') g.append('text').attr('transform','rotate(-90)')
        .attr('x',-h/2).attr('y',-85).attr('text-anchor','middle')
        .attr('font-size','11px').attr('fill','#666').text('$(ylabel)');

    // Tooltip
    svg.append('rect').attr('width',W).attr('height',h+margin.top+margin.bottom)
        .attr('fill','none').attr('pointer-events','all')
        .on('mousemove', function(evt) {
            const [mx, my] = d3.pointer(evt, g.node());
            const names = data.map(d => d.name);
            const bandY = y.step();
            const idx = Math.min(Math.floor(my / bandY), data.length-1);
            const d = data[Math.max(0,idx)];
            if(!d) return;
            showTip(evt, '<b>'+d.name+'</b><br>Effect: '+fmt(d.effect)+'<br>CI: ['+fmt(d.ci_lo)+', '+fmt(d.ci_hi)+']');
        })
        .on('mouseout', hideTip);
})();
"""
end

# =============================================================================
# MarginalEffects
# =============================================================================

"""
    plot_result(me::MarginalEffects{T}; title="", save_path=nothing)

Plot marginal effects as a horizontal coefficient plot with CI error bars.
"""
function plot_result(me::MarginalEffects{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    id = _next_plot_id("me_coef")

    # Build data JSON
    rows = String[]
    for i in 1:length(me.effects)
        push!(rows, "{\"name\":$(_json(me.varnames[i])),\"effect\":$(_json(me.effects[i])),\"ci_lo\":$(_json(me.ci_lower[i])),\"ci_hi\":$(_json(me.ci_upper[i]))}")
    end
    data_json = "[" * join(rows, ",\n") * "]"

    pct = round(Int, me.conf_level * 100)
    js = _render_coef_plot_js(id, data_json;
                               xlabel="Effect Size",
                               ylabel="")

    type_str = me.type == :ame ? "Average Marginal Effects" :
               me.type == :mem ? "Marginal Effects at Mean" :
               "Marginal Effects at Representative"
    ptitle = "$type_str ($(pct)% CI)"
    panel = _PanelSpec(id, ptitle, js)

    if isempty(title)
        title = type_str
    end

    p = _make_plot([panel]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end
