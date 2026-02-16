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
D3.js rendering engine: CSS, HTML skeleton, chart renderers (line, area, bar),
composition, save/display/show.
"""

# =============================================================================
# CSS
# =============================================================================

function _render_css(ncols::Int)
    col_width = ncols <= 1 ? "100%" : "$(round(Int, 100 / ncols))%"
    """
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        font-family: $(_PLOT_FONT);
        background: #fff;
        color: #333;
        padding: 20px;
    }
    .figure-title {
        font-size: 18px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 16px;
        color: #222;
    }
    .figure-source {
        font-size: 11px;
        color: #888;
        text-align: center;
        margin-top: 8px;
    }
    .panel-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        justify-content: center;
    }
    .panel {
        width: calc($col_width - 16px);
        min-width: 300px;
        background: #fff;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 12px;
    }
    .panel-title {
        font-size: 13px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 8px;
        color: #444;
    }
    .axis text { font-size: 11px; fill: #666; }
    .axis line, .axis path { stroke: #ccc; }
    .grid line { stroke: #f0f0f0; stroke-dasharray: 2,2; }
    .grid .domain { stroke: none; }
    .tooltip {
        position: absolute;
        background: rgba(0,0,0,0.8);
        color: #fff;
        padding: 6px 10px;
        border-radius: 4px;
        font-size: 11px;
        pointer-events: none;
        opacity: 0;
        transition: opacity 0.15s;
        z-index: 1000;
    }
    .legend { font-size: 11px; }
    .legend rect { rx: 2; }
    svg { overflow: visible; }
    """
end

# =============================================================================
# HTML Skeleton
# =============================================================================

function _render_html(; title::String, css::String, body::String, scripts::String)
    """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$(title)</title>
<style>$(css)</style>
</head>
<body>
$(body)
<div class="tooltip" id="tooltip"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>
$(_render_js_core())
$(scripts)
</script>
</body>
</html>"""
end

# =============================================================================
# Shared D3.js Core
# =============================================================================

function _render_js_core()
    """
// Shared utilities
const tooltip = d3.select('#tooltip');
function fmt(v) {
    if (v === null || v === undefined) return '';
    return Math.abs(v) >= 1000 ? v.toFixed(1) : (Math.abs(v) >= 1 ? v.toFixed(3) : v.toFixed(4));
}
function showTip(evt, html) {
    tooltip.html(html).style('opacity',1)
        .style('left',(evt.pageX+12)+'px').style('top',(evt.pageY-28)+'px');
}
function hideTip() { tooltip.style('opacity',0); }
"""
end

# =============================================================================
# Line Chart Renderer
# =============================================================================

"""
Generate D3.js code for a line chart with optional CI bands and reference lines.

- `id`: SVG container element ID
- `data_json`: JSON array of data points
- `series_json`: JSON array of {name, color, dash, key} series configs
- `bands_json`: JSON array of {lo_key, hi_key, color, alpha} band configs
- `ref_lines_json`: JSON array of {value, color, dash} horizontal reference lines
- `xlabel`, `ylabel`: axis labels
"""
function _render_line_js(id::String, data_json::String, series_json::String;
                         bands_json::String="[]", ref_lines_json::String="[]",
                         xlabel::String="", ylabel::String="")
    """
(function() {
    const data = $(data_json);
    const series = $(series_json);
    const bands = $(bands_json);
    const refLines = $(ref_lines_json);

    const container = d3.select('#$(id)');
    const W = Math.max(container.node().clientWidth - 24, 280);
    const margin = {top:10, right:15, bottom:35, left:55};
    const w = W - margin.left - margin.right;
    const h = Math.min(w * 0.6, 250);

    const svg = container.append('svg').attr('width', W).attr('height', h + margin.top + margin.bottom);
    const g = svg.append('g').attr('transform', 'translate('+margin.left+','+margin.top+')');

    // Compute domains
    const xVals = data.map(d => d.x);
    const allYVals = [];
    series.forEach(s => data.forEach(d => { if(d[s.key]!==null) allYVals.push(d[s.key]); }));
    bands.forEach(b => data.forEach(d => {
        if(d[b.lo_key]!==null) allYVals.push(d[b.lo_key]);
        if(d[b.hi_key]!==null) allYVals.push(d[b.hi_key]);
    }));
    refLines.forEach(r => allYVals.push(r.value));

    const x = d3.scaleLinear().domain(d3.extent(xVals)).range([0,w]);
    const yExt = d3.extent(allYVals);
    const yPad = (yExt[1]-yExt[0])*0.08 || 1;
    const y = d3.scaleLinear().domain([yExt[0]-yPad, yExt[1]+yPad]).range([h,0]);

    // Grid
    g.append('g').attr('class','grid').call(d3.axisLeft(y).tickSize(-w).tickFormat(''));

    // Bands
    bands.forEach(b => {
        const area = d3.area()
            .x(d => x(d.x))
            .y0(d => y(d[b.lo_key]!==null ? d[b.lo_key] : 0))
            .y1(d => y(d[b.hi_key]!==null ? d[b.hi_key] : 0))
            .defined(d => d[b.lo_key]!==null && d[b.hi_key]!==null);
        g.append('path').datum(data).attr('d',area)
            .attr('fill',b.color).attr('opacity',b.alpha||0.15);
    });

    // Reference lines
    refLines.forEach(r => {
        g.append('line').attr('x1',0).attr('x2',w)
            .attr('y1',y(r.value)).attr('y2',y(r.value))
            .attr('stroke',r.color||'#999').attr('stroke-width',1)
            .attr('stroke-dasharray',r.dash||'4,3');
    });

    // Lines
    series.forEach(s => {
        const line = d3.line().x(d=>x(d.x)).y(d=>y(d[s.key]))
            .defined(d=>d[s.key]!==null);
        g.append('path').datum(data).attr('d',line)
            .attr('fill','none').attr('stroke',s.color).attr('stroke-width',1.8)
            .attr('stroke-dasharray',s.dash||'');
    });

    // Axes
    g.append('g').attr('class','axis').attr('transform','translate(0,'+h+')')
        .call(d3.axisBottom(x).ticks(Math.min(xVals.length,8)));
    g.append('g').attr('class','axis').call(d3.axisLeft(y).ticks(6));

    if('$(xlabel)') g.append('text').attr('x',w/2).attr('y',h+30).attr('text-anchor','middle')
        .attr('font-size','11px').attr('fill','#666').text('$(xlabel)');
    if('$(ylabel)') g.append('text').attr('transform','rotate(-90)')
        .attr('x',-h/2).attr('y',-42).attr('text-anchor','middle')
        .attr('font-size','11px').attr('fill','#666').text('$(ylabel)');

    // Legend
    if(series.length > 1) {
        const leg = g.append('g').attr('class','legend').attr('transform','translate(5,-5)');
        series.forEach((s,i) => {
            const gi = leg.append('g').attr('transform','translate('+(i*100)+',0)');
            gi.append('line').attr('x1',0).attr('x2',16).attr('y1',0).attr('y2',0)
                .attr('stroke',s.color).attr('stroke-width',2)
                .attr('stroke-dasharray',s.dash||'');
            gi.append('text').attr('x',20).attr('y',4).attr('font-size','10px')
                .attr('fill','#555').text(s.name);
        });
    }

    // Tooltip overlay
    svg.append('rect').attr('width',W).attr('height',h+margin.top+margin.bottom)
        .attr('fill','none').attr('pointer-events','all')
        .on('mousemove', function(evt) {
            const [mx] = d3.pointer(evt, g.node());
            const x0 = x.invert(mx);
            const idx = d3.minIndex(data, d => Math.abs(d.x - x0));
            const d = data[idx];
            let html = '<b>x='+fmt(d.x)+'</b>';
            series.forEach(s => { if(d[s.key]!==null) html += '<br>'+s.name+': '+fmt(d[s.key]); });
            showTip(evt, html);
        })
        .on('mouseout', hideTip);
})();
"""
end

# =============================================================================
# Stacked Area Chart Renderer
# =============================================================================

"""
Generate D3.js code for a stacked area chart (proportions 0–1).
"""
function _render_area_js(id::String, data_json::String, series_json::String;
                         xlabel::String="", ylabel::String="Proportion")
    """
(function() {
    const data = $(data_json);
    const series = $(series_json);

    const container = d3.select('#$(id)');
    const W = Math.max(container.node().clientWidth - 24, 280);
    const margin = {top:10, right:15, bottom:35, left:55};
    const w = W - margin.left - margin.right;
    const h = Math.min(w * 0.6, 250);

    const svg = container.append('svg').attr('width', W).attr('height', h + margin.top + margin.bottom);
    const g = svg.append('g').attr('transform', 'translate('+margin.left+','+margin.top+')');

    const keys = series.map(s => s.key);
    const x = d3.scaleLinear().domain(d3.extent(data, d=>d.x)).range([0,w]);
    const y = d3.scaleLinear().domain([0,1]).range([h,0]);

    const stack = d3.stack().keys(keys).order(d3.stackOrderNone).offset(d3.stackOffsetNone);
    const stacked = stack(data);

    const area = d3.area()
        .x(d => x(d.data.x))
        .y0(d => y(d[0]))
        .y1(d => y(d[1]));

    g.selectAll('.layer').data(stacked).join('path')
        .attr('class','layer')
        .attr('d', area)
        .attr('fill', (d,i) => series[i].color)
        .attr('opacity', 0.85);

    g.append('g').attr('class','grid').call(d3.axisLeft(y).tickSize(-w).tickFormat(''));
    g.append('g').attr('class','axis').attr('transform','translate(0,'+h+')')
        .call(d3.axisBottom(x).ticks(Math.min(data.length,8)));
    g.append('g').attr('class','axis').call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.0%')));

    if('$(xlabel)') g.append('text').attr('x',w/2).attr('y',h+30).attr('text-anchor','middle')
        .attr('font-size','11px').attr('fill','#666').text('$(xlabel)');
    if('$(ylabel)') g.append('text').attr('transform','rotate(-90)')
        .attr('x',-h/2).attr('y',-42).attr('text-anchor','middle')
        .attr('font-size','11px').attr('fill','#666').text('$(ylabel)');

    // Legend
    const leg = g.append('g').attr('class','legend').attr('transform','translate(5,-5)');
    series.forEach((s,i) => {
        const gi = leg.append('g').attr('transform','translate('+(i*90)+',0)');
        gi.append('rect').attr('width',12).attr('height',12).attr('y',-6).attr('fill',s.color).attr('opacity',0.85);
        gi.append('text').attr('x',16).attr('y',4).attr('font-size','10px').attr('fill','#555').text(s.name);
    });

    // Tooltip
    svg.append('rect').attr('width',W).attr('height',h+margin.top+margin.bottom)
        .attr('fill','none').attr('pointer-events','all')
        .on('mousemove', function(evt) {
            const [mx] = d3.pointer(evt, g.node());
            const x0 = x.invert(mx);
            const idx = d3.minIndex(data, d => Math.abs(d.x - x0));
            const d = data[idx];
            let html = '<b>h='+d.x+'</b>';
            series.forEach(s => { html += '<br>'+s.name+': '+(d[s.key]*100).toFixed(1)+'%'; });
            showTip(evt, html);
        })
        .on('mouseout', hideTip);
})();
"""
end

# =============================================================================
# Bar Chart Renderer
# =============================================================================

"""
Generate D3.js code for a stacked or grouped bar chart.
- `mode`: "stacked" or "grouped"
"""
function _render_bar_js(id::String, data_json::String, series_json::String;
                        mode::String="stacked", xlabel::String="", ylabel::String="")
    """
(function() {
    const data = $(data_json);
    const series = $(series_json);
    const mode = '$(mode)';

    const container = d3.select('#$(id)');
    const W = Math.max(container.node().clientWidth - 24, 280);
    const margin = {top:10, right:15, bottom:35, left:55};
    const w = W - margin.left - margin.right;
    const h = Math.min(w * 0.6, 250);

    const svg = container.append('svg').attr('width', W).attr('height', h + margin.top + margin.bottom);
    const g = svg.append('g').attr('transform', 'translate('+margin.left+','+margin.top+')');

    const keys = series.map(s => s.key);
    const x = d3.scaleBand().domain(data.map(d=>d.x)).range([0,w]).padding(0.15);

    let yMin = 0, yMax = 0;
    if(mode === 'stacked') {
        const stack = d3.stack().keys(keys).order(d3.stackOrderNone).offset(d3.stackOffsetDiverging);
        const stacked = stack(data);
        stacked.forEach(layer => layer.forEach(d => {
            yMin = Math.min(yMin, d[0], d[1]);
            yMax = Math.max(yMax, d[0], d[1]);
        }));
        const yPad = (yMax - yMin) * 0.08 || 1;
        const y = d3.scaleLinear().domain([yMin - yPad, yMax + yPad]).range([h, 0]);

        g.append('g').attr('class','grid').call(d3.axisLeft(y).tickSize(-w).tickFormat(''));
        g.selectAll('.layer').data(stacked).join('g').attr('class','layer')
            .attr('fill', (d,i) => series[i].color)
            .selectAll('rect').data(d => d).join('rect')
            .attr('x', d => x(d.data.x))
            .attr('y', d => y(Math.max(d[0], d[1])))
            .attr('height', d => Math.abs(y(d[0]) - y(d[1])))
            .attr('width', x.bandwidth());

        // Zero line
        g.append('line').attr('x1',0).attr('x2',w)
            .attr('y1',y(0)).attr('y2',y(0))
            .attr('stroke','#333').attr('stroke-width',0.8);

        g.append('g').attr('class','axis').attr('transform','translate(0,'+h+')')
            .call(d3.axisBottom(x).tickValues(x.domain().filter((d,i) =>
                i % Math.max(1,Math.floor(data.length/8)) === 0)));
        g.append('g').attr('class','axis').call(d3.axisLeft(y).ticks(6));
    } else {
        // Grouped
        data.forEach(d => keys.forEach(k => {
            yMin = Math.min(yMin, d[k]||0);
            yMax = Math.max(yMax, d[k]||0);
        }));
        const yPad = (yMax - yMin) * 0.08 || 1;
        const y = d3.scaleLinear().domain([yMin - yPad, yMax + yPad]).range([h, 0]);
        const x1 = d3.scaleBand().domain(keys).range([0, x.bandwidth()]).padding(0.05);

        g.append('g').attr('class','grid').call(d3.axisLeft(y).tickSize(-w).tickFormat(''));
        g.selectAll('.group').data(data).join('g').attr('class','group')
            .attr('transform', d => 'translate('+x(d.x)+',0)')
            .selectAll('rect').data(d => keys.map(k => ({key:k, val:d[k]||0})))
            .join('rect')
            .attr('x', d => x1(d.key))
            .attr('y', d => y(Math.max(0, d.val)))
            .attr('height', d => Math.abs(y(0) - y(d.val)))
            .attr('width', x1.bandwidth())
            .attr('fill', d => series.find(s=>s.key===d.key).color);

        g.append('line').attr('x1',0).attr('x2',w)
            .attr('y1',y(0)).attr('y2',y(0))
            .attr('stroke','#333').attr('stroke-width',0.8);

        g.append('g').attr('class','axis').attr('transform','translate(0,'+h+')')
            .call(d3.axisBottom(x).tickValues(x.domain().filter((d,i) =>
                i % Math.max(1,Math.floor(data.length/8)) === 0)));
        g.append('g').attr('class','axis').call(d3.axisLeft(y).ticks(6));
    }

    if('$(xlabel)') g.append('text').attr('x',w/2).attr('y',h+30).attr('text-anchor','middle')
        .attr('font-size','11px').attr('fill','#666').text('$(xlabel)');
    if('$(ylabel)') g.append('text').attr('transform','rotate(-90)')
        .attr('x',-h/2).attr('y',-42).attr('text-anchor','middle')
        .attr('font-size','11px').attr('fill','#666').text('$(ylabel)');

    // Legend
    if(series.length > 1) {
        const leg = g.append('g').attr('class','legend').attr('transform','translate(5,-5)');
        series.forEach((s,i) => {
            const gi = leg.append('g').attr('transform','translate('+(i*90)+',0)');
            gi.append('rect').attr('width',12).attr('height',12).attr('y',-6).attr('fill',s.color);
            gi.append('text').attr('x',16).attr('y',4).attr('font-size','10px').attr('fill','#555').text(s.name);
        });
    }

    // Tooltip
    svg.append('rect').attr('width',W).attr('height',h+margin.top+margin.bottom)
        .attr('fill','none').attr('pointer-events','all')
        .on('mousemove', function(evt) {
            const [mx] = d3.pointer(evt, g.node());
            const bands = x.domain();
            const idx = Math.min(Math.floor(mx / (w / bands.length)), bands.length-1);
            const d = data[Math.max(0,idx)];
            if(!d) return;
            let html = '<b>'+d.x+'</b>';
            series.forEach(s => { if(d[s.key]!==undefined) html += '<br>'+s.name+': '+fmt(d[s.key]); });
            showTip(evt, html);
        })
        .on('mouseout', hideTip);
})();
"""
end

# =============================================================================
# Panel Body Rendering
# =============================================================================

function _render_body_single(panel::_PanelSpec; title::String="")
    html = ""
    if !isempty(title)
        html *= "<div class=\"figure-title\">$(title)</div>\n"
    end
    html *= """<div class="panel">
<div class="panel-title">$(panel.title)</div>
<div id="$(panel.id)"></div>
</div>"""
    html
end

function _render_body_figure(panels::Vector{_PanelSpec}; title::String="",
                             source::String="", note::String="")
    html = ""
    if !isempty(title)
        html *= "<div class=\"figure-title\">$(title)</div>\n"
    end
    html *= "<div class=\"panel-grid\">\n"
    for p in panels
        html *= """<div class="panel">
<div class="panel-title">$(p.title)</div>
<div id="$(p.id)"></div>
</div>\n"""
    end
    html *= "</div>\n"
    if !isempty(source)
        html *= "<div class=\"figure-source\">$(source)</div>\n"
    end
    if !isempty(note)
        html *= "<div class=\"figure-source\">$(note)</div>\n"
    end
    html
end

# =============================================================================
# Composition
# =============================================================================

"""
Compose multiple panels into a single PlotOutput HTML document.
"""
function _make_plot(panels::Vector{_PanelSpec}; title::String="",
                    ncols::Int=0, source::String="", note::String="")
    # Auto-determine columns
    if ncols <= 0
        ncols = length(panels) <= 2 ? 1 : min(length(panels), 3)
    end

    css = _render_css(ncols)
    if length(panels) == 1
        body = _render_body_single(panels[1]; title=title)
    else
        body = _render_body_figure(panels; title=title, source=source, note=note)
    end
    scripts = join([p.js for p in panels], "\n")

    html = _render_html(; title=isempty(title) ? "MacroEconometricModels Plot" : title,
                         css=css, body=body, scripts=scripts)
    PlotOutput(html)
end

# =============================================================================
# Save / Display / Show
# =============================================================================

"""
    save_plot(p::PlotOutput, path::String)

Write the HTML visualization to a file.

# Example
```julia
p = plot_result(irf_result)
save_plot(p, "irf_plot.html")
```
"""
function save_plot(p::PlotOutput, path::String)
    open(path, "w") do f
        write(f, p.html)
    end
    path
end

"""
    display_plot(p::PlotOutput)

Open the HTML visualization in the system default browser.
"""
function display_plot(p::PlotOutput)
    tmpfile = tempname() * ".html"
    save_plot(p, tmpfile)
    if Sys.isapple()
        run(`open $tmpfile`)
    elseif Sys.islinux()
        run(`xdg-open $tmpfile`)
    elseif Sys.iswindows()
        run(`cmd /c start $tmpfile`)
    end
    tmpfile
end

function Base.show(io::IO, ::MIME"text/html", p::PlotOutput)
    print(io, p.html)
end

function Base.show(io::IO, ::MIME"text/plain", p::PlotOutput)
    n = length(p.html)
    print(io, "PlotOutput($(n) bytes HTML with inline D3.js)")
end

function Base.show(io::IO, p::PlotOutput)
    show(io, MIME"text/plain"(), p)
end
