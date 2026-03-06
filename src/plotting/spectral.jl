# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
plot_result methods for spectral analysis types: ACFResult, SpectralDensityResult,
CrossSpectrumResult, TransferFunctionResult.
"""

# =============================================================================
# Bar Chart Renderer (vertical bars from baseline=0)
# =============================================================================

"""
Generate D3.js code for a vertical bar chart with bars drawn from baseline=0,
plus optional horizontal reference lines (e.g., CI bounds).

- `id`: SVG container element ID
- `data_json`: JSON array of {x, y} data points
- `bar_color`: bar fill color
- `ref_lines_json`: JSON array of {value, color, dash} horizontal lines
- `xlabel`, `ylabel`: axis labels
"""
function _render_vbar_js(id::String, data_json::String;
                         bar_color::String=_PLOT_COLORS[1],
                         ref_lines_json::String="[]",
                         xlabel::String="", ylabel::String="")
    """
(function() {
    const data = $(data_json);
    const refLines = $(ref_lines_json);

    const container = d3.select('#$(id)');
    const W = Math.max(container.node().clientWidth - 24, 280);
    const margin = {top:10, right:15, bottom:35, left:55};
    const w = W - margin.left - margin.right;
    const h = Math.min(w * 0.6, 250);

    const svg = container.append('svg').attr('width', W).attr('height', h + margin.top + margin.bottom);
    const g = svg.append('g').attr('transform', 'translate('+margin.left+','+margin.top+')');

    const xVals = data.map(d => d.x);
    const yVals = data.map(d => d.y);
    refLines.forEach(r => yVals.push(r.value));
    yVals.push(0);

    const xExt = d3.extent(xVals);
    const xPad = xVals.length > 1 ? (xVals[1] - xVals[0]) * 0.5 : 0.5;
    const x = d3.scaleLinear().domain([xExt[0] - xPad, xExt[1] + xPad]).range([0, w]);

    const yExt = d3.extent(yVals);
    const yPad = (yExt[1] - yExt[0]) * 0.08 || 0.1;
    const y = d3.scaleLinear().domain([yExt[0] - yPad, yExt[1] + yPad]).range([h, 0]);

    // Grid
    g.append('g').attr('class','grid').call(d3.axisLeft(y).tickSize(-w).tickFormat(''));

    // Zero line
    g.append('line').attr('x1',0).attr('x2',w)
        .attr('y1',y(0)).attr('y2',y(0))
        .attr('stroke','#333').attr('stroke-width',0.8);

    // Reference lines (CI bounds)
    refLines.forEach(r => {
        g.append('line').attr('x1',0).attr('x2',w)
            .attr('y1',y(r.value)).attr('y2',y(r.value))
            .attr('stroke',r.color||'#d62728').attr('stroke-width',1)
            .attr('stroke-dasharray',r.dash||'5,4');
    });

    // Bars
    const barW = Math.max(1, Math.min(6, w / data.length * 0.6));
    data.forEach(d => {
        const yTop = d.y >= 0 ? y(d.y) : y(0);
        const yBot = d.y >= 0 ? y(0) : y(d.y);
        g.append('rect')
            .attr('x', x(d.x) - barW/2)
            .attr('y', yTop)
            .attr('width', barW)
            .attr('height', Math.max(0.5, yBot - yTop))
            .attr('fill', '$(bar_color)')
            .attr('opacity', 0.85);
    });

    // Axes
    g.append('g').attr('class','axis').attr('transform','translate(0,'+h+')')
        .call(d3.axisBottom(x).ticks(Math.min(xVals.length, 10)));
    g.append('g').attr('class','axis').call(d3.axisLeft(y).ticks(6));

    if('$(xlabel)') g.append('text').attr('x',w/2).attr('y',h+30).attr('text-anchor','middle')
        .attr('font-size','11px').attr('fill','#666').text('$(xlabel)');
    if('$(ylabel)') g.append('text').attr('transform','rotate(-90)')
        .attr('x',-h/2).attr('y',-42).attr('text-anchor','middle')
        .attr('font-size','11px').attr('fill','#666').text('$(ylabel)');

    // Tooltip
    svg.append('rect').attr('width',W).attr('height',h+margin.top+margin.bottom)
        .attr('fill','none').attr('pointer-events','all')
        .on('mousemove', function(evt) {
            const [mx] = d3.pointer(evt, g.node());
            const x0 = x.invert(mx);
            const idx = d3.minIndex(data, d => Math.abs(d.x - x0));
            const d = data[idx];
            showTip(evt, '<b>Lag '+d.x+'</b><br>Value: '+fmt(d.y));
        })
        .on('mouseout', hideTip);
})();
"""
end

# =============================================================================
# ACF Data Helpers
# =============================================================================

"""Convert ACF/PACF lag values to JSON [{x, y}, ...]."""
function _acf_bar_data_json(lags::AbstractVector{Int}, vals::AbstractVector{T}) where {T}
    rows = Vector{Pair{String,String}}[]
    for i in eachindex(lags)
        push!(rows, ["x" => _json(lags[i]), "y" => _json(vals[i])])
    end
    _json_array_of_objects(rows)
end

# =============================================================================
# ACFResult
# =============================================================================

"""
    plot_result(r::ACFResult; title="", save_path=nothing)

Plot ACF and PACF as side-by-side bar charts with ±CI dashed lines.
For CCF results (when `r.ccf` is populated), a single cross-correlation bar chart.
"""
function plot_result(r::ACFResult{T};
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    ci_refs = "[{\"value\":$(_json(r.ci)),\"color\":\"#d62728\",\"dash\":\"5,4\"},{\"value\":$(_json(-r.ci)),\"color\":\"#d62728\",\"dash\":\"5,4\"}]"

    if r.ccf !== nothing
        # CCF plot — single panel
        id = _next_plot_id("ccf")
        data_json = _acf_bar_data_json(r.lags, r.ccf)
        js = _render_vbar_js(id, data_json;
                             bar_color=_PLOT_COLORS[1],
                             ref_lines_json=ci_refs,
                             xlabel="Lag", ylabel="CCF")
        panels = [_PanelSpec(id, "Cross-Correlation Function", js)]
        if isempty(title)
            title = "Cross-Correlation Function (n=$(r.nobs))"
        end
        p = _make_plot(panels; title=title, ncols=1)
    else
        # ACF + PACF side-by-side
        panels = _PanelSpec[]

        # ACF panel
        has_acf = any(!iszero, r.acf)
        if has_acf
            id_acf = _next_plot_id("acf")
            data_acf = _acf_bar_data_json(r.lags, r.acf)
            js_acf = _render_vbar_js(id_acf, data_acf;
                                     bar_color=_PLOT_COLORS[1],
                                     ref_lines_json=ci_refs,
                                     xlabel="Lag", ylabel="ACF")
            push!(panels, _PanelSpec(id_acf, "Autocorrelation", js_acf))
        end

        # PACF panel
        has_pacf = any(!iszero, r.pacf)
        if has_pacf
            id_pacf = _next_plot_id("pacf")
            data_pacf = _acf_bar_data_json(r.lags, r.pacf)
            js_pacf = _render_vbar_js(id_pacf, data_pacf;
                                      bar_color=_PLOT_COLORS[2],
                                      ref_lines_json=ci_refs,
                                      xlabel="Lag", ylabel="PACF")
            push!(panels, _PanelSpec(id_pacf, "Partial Autocorrelation", js_pacf))
        end

        # Fallback if neither populated (shouldn't happen)
        if isempty(panels)
            id_acf = _next_plot_id("acf")
            data_acf = _acf_bar_data_json(r.lags, r.acf)
            js_acf = _render_vbar_js(id_acf, data_acf;
                                     bar_color=_PLOT_COLORS[1],
                                     ref_lines_json=ci_refs,
                                     xlabel="Lag", ylabel="ACF")
            push!(panels, _PanelSpec(id_acf, "Autocorrelation", js_acf))
        end

        if isempty(title)
            title = "ACF / PACF (n=$(r.nobs))"
        end
        ncols = length(panels) == 2 ? 2 : 1
        p = _make_plot(panels; title=title, ncols=ncols)
    end

    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# Spectral Density Data Helpers
# =============================================================================

"""Convert spectral density to JSON [{x, density, ci_lo, ci_hi}, ...]."""
function _spectral_data_json(freq::AbstractVector{T}, density::AbstractVector{T},
                             ci_lo::AbstractVector{T}, ci_hi::AbstractVector{T};
                             log_scale::Bool=true) where {T}
    rows = Vector{Pair{String,String}}[]
    for i in eachindex(freq)
        d_val = log_scale ? log10(max(density[i], T(1e-30))) : density[i]
        lo_val = log_scale ? log10(max(ci_lo[i], T(1e-30))) : ci_lo[i]
        hi_val = log_scale ? log10(max(ci_hi[i], T(1e-30))) : ci_hi[i]
        push!(rows, [
            "x" => _json(freq[i]),
            "density" => _json(d_val),
            "ci_lo" => _json(lo_val),
            "ci_hi" => _json(hi_val)
        ])
    end
    _json_array_of_objects(rows)
end

# =============================================================================
# SpectralDensityResult
# =============================================================================

"""
    plot_result(r::SpectralDensityResult; title="", save_path=nothing)

Plot log-spectral density line with CI shading band.
"""
function plot_result(r::SpectralDensityResult{T};
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    id = _next_plot_id("spec")
    data_json = _spectral_data_json(r.freq, r.density, r.ci_lower, r.ci_upper)

    s_json = _series_json(["Log Spectral Density"], [_PLOT_COLORS[1]]; keys=["density"])
    bands = "[{\"lo_key\":\"ci_lo\",\"hi_key\":\"ci_hi\",\"color\":\"$(_PLOT_COLORS[1])\",\"alpha\":$(_PLOT_CI_ALPHA)}]"

    js = _render_line_js(id, data_json, s_json;
                         bands_json=bands,
                         xlabel="Frequency (radians)", ylabel="log₁₀ Spectral Density")
    panels = [_PanelSpec(id, "Spectral Density ($(r.method))", js)]

    if isempty(title)
        title = "Spectral Density Estimate ($(r.method), n=$(r.nobs))"
    end

    p = _make_plot(panels; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# Cross-Spectrum Data Helpers
# =============================================================================

"""Convert cross-spectrum fields to JSON for coherence + phase panels."""
function _cross_spectrum_data_json(freq::AbstractVector{T},
                                   vals::AbstractVector{T};
                                   key::String="y") where {T}
    rows = Vector{Pair{String,String}}[]
    for i in eachindex(freq)
        push!(rows, ["x" => _json(freq[i]), key => _json(vals[i])])
    end
    _json_array_of_objects(rows)
end

# =============================================================================
# CrossSpectrumResult
# =============================================================================

"""
    plot_result(r::CrossSpectrumResult; title="", save_path=nothing)

Plot cross-spectrum: 2-panel with coherence line and phase line.
"""
function plot_result(r::CrossSpectrumResult{T};
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    panels = _PanelSpec[]

    # Panel 1: Coherence
    id_coh = _next_plot_id("coh")
    data_coh = _cross_spectrum_data_json(r.freq, r.coherence; key="coh")
    s_coh = _series_json(["Squared Coherence"], [_PLOT_COLORS[1]]; keys=["coh"])
    js_coh = _render_line_js(id_coh, data_coh, s_coh;
                             xlabel="Frequency (radians)", ylabel="Coherence")
    push!(panels, _PanelSpec(id_coh, "Squared Coherence", js_coh))

    # Panel 2: Phase
    id_ph = _next_plot_id("phase")
    data_ph = _cross_spectrum_data_json(r.freq, r.phase; key="phase")
    s_ph = _series_json(["Phase"], [_PLOT_COLORS[2]]; keys=["phase"])
    refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js_ph = _render_line_js(id_ph, data_ph, s_ph;
                            ref_lines_json=refs,
                            xlabel="Frequency (radians)", ylabel="Phase (radians)")
    push!(panels, _PanelSpec(id_ph, "Phase Spectrum", js_ph))

    if isempty(title)
        title = "Cross-Spectral Analysis (n=$(r.nobs))"
    end

    p = _make_plot(panels; title=title, ncols=2)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# TransferFunctionResult
# =============================================================================

"""
    plot_result(r::TransferFunctionResult; title="", save_path=nothing)

Plot transfer function: 2-panel with gain line and phase line.
"""
function plot_result(r::TransferFunctionResult{T};
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    panels = _PanelSpec[]

    # Panel 1: Gain
    id_gain = _next_plot_id("tfgain")
    data_gain = _cross_spectrum_data_json(r.freq, r.gain; key="gain")
    s_gain = _series_json(["Gain"], [_PLOT_COLORS[1]]; keys=["gain"])
    js_gain = _render_line_js(id_gain, data_gain, s_gain;
                              xlabel="Frequency (radians)", ylabel="Gain")
    push!(panels, _PanelSpec(id_gain, "Gain (Amplitude)", js_gain))

    # Panel 2: Phase
    id_ph = _next_plot_id("tfphase")
    data_ph = _cross_spectrum_data_json(r.freq, r.phase; key="phase")
    s_ph = _series_json(["Phase"], [_PLOT_COLORS[2]]; keys=["phase"])
    refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js_ph = _render_line_js(id_ph, data_ph, s_ph;
                            ref_lines_json=refs,
                            xlabel="Frequency (radians)", ylabel="Phase (radians)")
    push!(panels, _PanelSpec(id_ph, "Phase Shift", js_ph))

    if isempty(title)
        title = "Transfer Function — $(r.filter) Filter"
    end

    p = _make_plot(panels; title=title, ncols=2)
    save_path !== nothing && save_plot(p, save_path)
    p
end
