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
# ARCHModel
# =============================================================================

"""
    plot_result(m::ARCHModel; title="", save_path=nothing)

Plot ARCH model diagnostics: returns, conditional volatility, standardized residuals.
"""
function plot_result(m::ARCHModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    _plot_volatility_diagnostics(m.y, m.conditional_variance, "ARCH($(m.q))";
                                  title=title, save_path=save_path)
end

# =============================================================================
# GARCHModel
# =============================================================================

"""
    plot_result(m::GARCHModel; title="", save_path=nothing)

Plot GARCH model diagnostics.
"""
function plot_result(m::GARCHModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    _plot_volatility_diagnostics(m.y, m.conditional_variance, "GARCH($(m.p),$(m.q))";
                                  title=title, save_path=save_path)
end

# =============================================================================
# EGARCHModel
# =============================================================================

"""
    plot_result(m::EGARCHModel; title="", save_path=nothing)

Plot EGARCH model diagnostics.
"""
function plot_result(m::EGARCHModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    _plot_volatility_diagnostics(m.y, m.conditional_variance, "EGARCH($(m.p),$(m.q))";
                                  title=title, save_path=save_path)
end

# =============================================================================
# GJRGARCHModel
# =============================================================================

"""
    plot_result(m::GJRGARCHModel; title="", save_path=nothing)

Plot GJR-GARCH model diagnostics.
"""
function plot_result(m::GJRGARCHModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    _plot_volatility_diagnostics(m.y, m.conditional_variance, "GJR-GARCH($(m.p),$(m.q))";
                                  title=title, save_path=save_path)
end

# =============================================================================
# SVModel
# =============================================================================

"""
    plot_result(m::SVModel; title="", save_path=nothing)

Plot SV model diagnostics with posterior quantile bands on volatility.
"""
function plot_result(m::SVModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
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
"""
function plot_result(fm::FactorModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    # Panel 1: Scree plot (eigenvalues as bar chart)
    id1 = _next_plot_id("fm_scree")
    n_eig = min(length(fm.eigenvalues), 10)
    rows1 = Vector{Pair{String,String}}[]
    for i in 1:n_eig
        push!(rows1, ["x" => _json("PC $i"), "eig" => _json(fm.eigenvalues[i])])
    end
    data1 = _json_array_of_objects(rows1)
    s1 = _series_json(["Eigenvalue"], [_PLOT_COLORS[1]]; keys=["eig"])
    js1 = _render_bar_js(id1, data1, s1; mode="grouped", ylabel="Eigenvalue")
    p1 = _PanelSpec(id1, "Scree Plot", js1)

    # Panel 2: Factor series
    id2 = _next_plot_id("fm_fac")
    T_obs, r = size(fm.factors)
    n_plot = min(r, 5)
    fac_names = ["Factor $i" for i in 1:n_plot]
    fac_colors = _PLOT_COLORS[1:n_plot]
    data2 = _timeseries_data_json(fm.factors[:, 1:n_plot], fac_names)
    s2 = _series_json(fac_names, fac_colors; keys=["v$i" for i in 1:n_plot])
    js2 = _render_line_js(id2, data2, s2; xlabel="Period", ylabel="Factor Value")
    p2 = _PanelSpec(id2, "Extracted Factors", js2)

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
"""
function plot_result(fm::DynamicFactorModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    # Panel 1: Scree
    id1 = _next_plot_id("dfm_scree")
    n_eig = min(length(fm.eigenvalues), 10)
    rows1 = Vector{Pair{String,String}}[]
    for i in 1:n_eig
        push!(rows1, ["x" => _json("PC $i"), "eig" => _json(fm.eigenvalues[i])])
    end
    data1 = _json_array_of_objects(rows1)
    s1 = _series_json(["Eigenvalue"], [_PLOT_COLORS[1]]; keys=["eig"])
    js1 = _render_bar_js(id1, data1, s1; mode="grouped", ylabel="Eigenvalue")
    p1 = _PanelSpec(id1, "Scree Plot", js1)

    # Panel 2: Factor series
    id2 = _next_plot_id("dfm_fac")
    T_obs, r = size(fm.factors)
    n_plot = min(r, 5)
    fac_names = ["Factor $i" for i in 1:n_plot]
    data2 = _timeseries_data_json(fm.factors[:, 1:n_plot], fac_names)
    s2 = _series_json(fac_names, _PLOT_COLORS[1:n_plot]; keys=["v$i" for i in 1:n_plot])
    js2 = _render_line_js(id2, data2, s2; xlabel="Period", ylabel="Factor Value")
    p2 = _PanelSpec(id2, "Extracted Factors (VAR($( fm.p)))", js2)

    if isempty(title)
        title = "Dynamic Factor Model (r=$(fm.r), p=$(fm.p))"
    end

    p = _make_plot([p1, p2]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# TimeSeriesData
# =============================================================================

"""
    plot_result(d::TimeSeriesData; vars=nothing, ncols=0, title="", save_path=nothing)

Plot time series data: one panel per variable (capped at 12).

- `vars`: Variable indices or names to plot. `nothing` = all (up to 12).
"""
function plot_result(d::TimeSeriesData{T};
                     vars::Union{Vector,Nothing}=nothing,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    if vars === nothing
        idxs = collect(1:min(d.n_vars, 12))
    else
        idxs = [v isa String ? _resolve_var(v, d.varnames) : v for v in vars]
    end

    panels = _PanelSpec[]
    for vi in idxs
        id = _next_plot_id("ts")
        ptitle = d.varnames[vi]

        rows = Vector{Pair{String,String}}[]
        for t in 1:d.T_obs
            push!(rows, [
                "x" => _json(d.time_index[t]),
                "v1" => _json(d.data[t, vi])
            ])
        end
        data_json = _json_array_of_objects(rows)
        s_json = _series_json([d.varnames[vi]], [_PLOT_COLORS[mod1(vi, length(_PLOT_COLORS))]];
                              keys=["v1"])

        js = _render_line_js(id, data_json, s_json; xlabel="Time", ylabel="")
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        title = isempty(desc(d)) ? "Time Series Data" : desc(d)
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# PanelData
# =============================================================================

"""
    plot_result(d::PanelData; vars=nothing, ncols=0, title="", save_path=nothing)

Plot panel data: one panel per variable, with lines for each group.
"""
function plot_result(d::PanelData{T};
                     vars::Union{Vector,Nothing}=nothing,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    if vars === nothing
        var_idxs = collect(1:min(d.n_vars, 6))
    else
        var_idxs = [v isa String ? _resolve_var(v, d.varnames) : v for v in vars]
    end

    # Group data by group_id
    unique_groups = sort(unique(d.group_id))
    n_groups_plot = min(length(unique_groups), 10)
    groups_to_plot = unique_groups[1:n_groups_plot]

    panels = _PanelSpec[]
    for vi in var_idxs
        id = _next_plot_id("pd")
        ptitle = d.varnames[vi]

        # Build data with one column per group
        # Get unique time points
        unique_times = sort(unique(d.time_id))
        rows = Vector{Pair{String,String}}[]
        for t in unique_times
            row = Pair{String,String}["x" => _json(t)]
            for (gi, gid) in enumerate(groups_to_plot)
                mask = (d.group_id .== gid) .& (d.time_id .== t)
                idx = findfirst(mask)
                val = idx !== nothing ? d.data[idx, vi] : NaN
                push!(row, "g$gi" => _json(val))
            end
            push!(rows, row)
        end
        data_json = _json_array_of_objects(rows)

        group_names = [gi <= length(d.group_names) ? d.group_names[gi] : "Group $gid"
                       for (gi, gid) in enumerate(groups_to_plot)]
        s_json = _series_json(group_names, _PLOT_COLORS[1:n_groups_plot];
                              keys=["g$i" for i in 1:n_groups_plot])

        js = _render_line_js(id, data_json, s_json; xlabel="Time", ylabel="")
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        title = isempty(desc(d)) ? "Panel Data" : desc(d)
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# OccBinIRF
# =============================================================================

"""
    plot_result(oirf::OccBinIRF; title="", save_path=nothing)

Plot OccBin IRF comparison: linear (dashed) vs piecewise-linear (solid) with
shaded binding-period rectangles for each variable.
"""
function plot_result(oirf::OccBinIRF{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    H, n_vars = size(oirf.piecewise)
    panels = _PanelSpec[]

    for j in 1:n_vars
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
                "bind" => _json(binding[h] ? 1 : 0)
            ])
        end
        data_json = _json_array_of_objects(rows)

        js = _render_occbin_panel_js(id, data_json, "Horizon", oirf.varnames[j])
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        title = "OccBin IRF — Shock: $(oirf.shock_name)"
    end

    p = _make_plot(panels; title=title, ncols=min(n_vars, 3))
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# OccBinSolution
# =============================================================================

"""
    plot_result(sol::OccBinSolution; title="", save_path=nothing)

Plot OccBin solution comparison: linear (dashed) vs piecewise-linear (solid) with
shaded binding-period rectangles for each variable.
"""
function plot_result(sol::OccBinSolution{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    nperiods, n_vars = size(sol.piecewise_path)
    panels = _PanelSpec[]

    for j in 1:n_vars
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
                "bind" => _json(binding[h] ? 1 : 0)
            ])
        end
        data_json = _json_array_of_objects(rows)

        js = _render_occbin_panel_js(id, data_json, "Period", sol.varnames[j])
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        title = "OccBin Piecewise-Linear Solution"
    end

    p = _make_plot(panels; title=title, ncols=min(n_vars, 3))
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# Shared OccBin Panel Renderer
# =============================================================================

"""
Render a single OccBin panel with linear (dashed) vs piecewise (solid) lines,
shaded binding-period rectangles, and zero reference line.

Data format: [{h, lin, pw, bind}, ...] where bind is 0 or 1.
"""
function _render_occbin_panel_js(id::String, data_json::String,
                                  xlabel::String, ylabel::String)
    lin_color = _PLOT_COLORS[3]   # green for linear
    pw_color = _PLOT_COLORS[1]    # blue for piecewise
    bind_color = "#d62728"        # red for binding region
    """
(function() {
    const data_$(id) = $(data_json);
    const container_$(id) = d3.select('#$(id)');
    const W_$(id) = Math.max(container_$(id).node().clientWidth - 24, 280);
    const margin_$(id) = {top:10, right:15, bottom:35, left:55};
    const w_$(id) = W_$(id) - margin_$(id).left - margin_$(id).right;
    const h_$(id) = Math.min(w_$(id) * 0.6, 250);

    const svg_$(id) = container_$(id).append('svg')
        .attr('width', W_$(id))
        .attr('height', h_$(id) + margin_$(id).top + margin_$(id).bottom);
    const g_$(id) = svg_$(id).append('g')
        .attr('transform', 'translate('+margin_$(id).left+','+margin_$(id).top+')');

    // Compute domains
    const xVals_$(id) = data_$(id).map(d => d.h);
    const allY_$(id) = [];
    data_$(id).forEach(d => {
        if(d.lin!==null) allY_$(id).push(d.lin);
        if(d.pw!==null) allY_$(id).push(d.pw);
    });
    allY_$(id).push(0);

    const x_$(id) = d3.scaleLinear().domain(d3.extent(xVals_$(id))).range([0, w_$(id)]);
    const yExt_$(id) = d3.extent(allY_$(id));
    const yPad_$(id) = (yExt_$(id)[1] - yExt_$(id)[0]) * 0.08 || 0.1;
    const y_$(id) = d3.scaleLinear()
        .domain([yExt_$(id)[0] - yPad_$(id), yExt_$(id)[1] + yPad_$(id)])
        .range([h_$(id), 0]);

    // Grid
    g_$(id).append('g').attr('class','grid')
        .call(d3.axisLeft(y_$(id)).tickSize(-w_$(id)).tickFormat(''));

    // Binding-period shaded rectangles
    const xStep_$(id) = w_$(id) / Math.max(xVals_$(id).length - 1, 1);
    data_$(id).forEach((d, i) => {
        if(d.bind === 1) {
            const x0_$(id) = x_$(id)(d.h) - xStep_$(id) * 0.5;
            const x1_$(id) = x_$(id)(d.h) + xStep_$(id) * 0.5;
            g_$(id).append('rect')
                .attr('x', Math.max(0, x0_$(id)))
                .attr('y', 0)
                .attr('width', Math.min(x1_$(id), w_$(id)) - Math.max(0, x0_$(id)))
                .attr('height', h_$(id))
                .attr('fill', '$(bind_color)')
                .attr('opacity', 0.08);
        }
    });

    // Zero reference line
    g_$(id).append('line')
        .attr('x1', 0).attr('x2', w_$(id))
        .attr('y1', y_$(id)(0)).attr('y2', y_$(id)(0))
        .attr('stroke', '#999').attr('stroke-width', 1)
        .attr('stroke-dasharray', '4,3');

    // Linear line (dashed)
    const linLine_$(id) = d3.line().x(d => x_$(id)(d.h)).y(d => y_$(id)(d.lin))
        .defined(d => d.lin !== null);
    g_$(id).append('path').datum(data_$(id)).attr('d', linLine_$(id))
        .attr('fill', 'none').attr('stroke', '$(lin_color)')
        .attr('stroke-width', 1.8).attr('stroke-dasharray', '6,3');

    // Piecewise line (solid)
    const pwLine_$(id) = d3.line().x(d => x_$(id)(d.h)).y(d => y_$(id)(d.pw))
        .defined(d => d.pw !== null);
    g_$(id).append('path').datum(data_$(id)).attr('d', pwLine_$(id))
        .attr('fill', 'none').attr('stroke', '$(pw_color)')
        .attr('stroke-width', 1.8);

    // Axes
    g_$(id).append('g').attr('class','axis')
        .attr('transform', 'translate(0,'+h_$(id)+')')
        .call(d3.axisBottom(x_$(id)).ticks(Math.min(xVals_$(id).length, 8)));
    g_$(id).append('g').attr('class','axis')
        .call(d3.axisLeft(y_$(id)).ticks(6));

    if('$(xlabel)') g_$(id).append('text')
        .attr('x', w_$(id)/2).attr('y', h_$(id)+30)
        .attr('text-anchor','middle').attr('font-size','11px')
        .attr('fill','#666').text('$(xlabel)');
    if('$(ylabel)') g_$(id).append('text')
        .attr('transform','rotate(-90)')
        .attr('x', -h_$(id)/2).attr('y', -42)
        .attr('text-anchor','middle').attr('font-size','11px')
        .attr('fill','#666').text('$(ylabel)');

    // Legend
    const leg_$(id) = g_$(id).append('g').attr('class','legend')
        .attr('transform','translate(5,-5)');
    // Linear (dashed)
    const g1_$(id) = leg_$(id).append('g').attr('transform','translate(0,0)');
    g1_$(id).append('line').attr('x1',0).attr('x2',16).attr('y1',0).attr('y2',0)
        .attr('stroke','$(lin_color)').attr('stroke-width',2)
        .attr('stroke-dasharray','6,3');
    g1_$(id).append('text').attr('x',20).attr('y',4)
        .attr('font-size','10px').attr('fill','#555').text('Linear');
    // Piecewise (solid)
    const g2_$(id) = leg_$(id).append('g').attr('transform','translate(80,0)');
    g2_$(id).append('line').attr('x1',0).attr('x2',16).attr('y1',0).attr('y2',0)
        .attr('stroke','$(pw_color)').attr('stroke-width',2);
    g2_$(id).append('text').attr('x',20).attr('y',4)
        .attr('font-size','10px').attr('fill','#555').text('Piecewise');
    // Binding
    const g3_$(id) = leg_$(id).append('g').attr('transform','translate(175,0)');
    g3_$(id).append('rect').attr('width',12).attr('height',10).attr('y',-5)
        .attr('fill','$(bind_color)').attr('opacity',0.15);
    g3_$(id).append('text').attr('x',16).attr('y',4)
        .attr('font-size','10px').attr('fill','#555').text('Binding');

    // Tooltip
    svg_$(id).append('rect')
        .attr('width', W_$(id))
        .attr('height', h_$(id) + margin_$(id).top + margin_$(id).bottom)
        .attr('fill','none').attr('pointer-events','all')
        .on('mousemove', function(evt) {
            const [mx] = d3.pointer(evt, g_$(id).node());
            const x0 = x_$(id).invert(mx);
            const idx = d3.minIndex(data_$(id), d => Math.abs(d.h - x0));
            const d = data_$(id)[idx];
            let html = '<b>h='+d.h+'</b>';
            if(d.lin!==null) html += '<br>Linear: '+fmt(d.lin);
            if(d.pw!==null) html += '<br>Piecewise: '+fmt(d.pw);
            if(d.bind===1) html += '<br><span style="color:$(bind_color)">Binding</span>';
            showTip(evt, html);
        })
        .on('mouseout', hideTip);
})();
"""
end
