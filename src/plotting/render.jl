# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

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
<title>$(_esc_html(title))</title>
<style>$(css)</style>
</head>
<body>
$(body)
<script>$(_d3_source())</script>
<script>
$(_render_js_core())
$(scripts)
</script>
</body>
</html>"""
end

# Embeddable HTML fragment — no <!DOCTYPE>/<html>/<head>/<body>, so two plots can be
# dropped into one notebook/Documenter page without nesting whole documents (A11).
# D3 is inlined per fragment (the UMD build reassigns window.d3 idempotently); the
# shared JS core is guarded so re-inclusion on the same page is a runtime no-op.
function _render_fragment(; css::String, body::String, scripts::String)
    """<style>$(css)</style>
$(body)
<script>$(_d3_source())</script>
<script>
$(_render_js_core())
$(scripts)
</script>"""
end

# =============================================================================
# Shared D3.js Core
# =============================================================================

function _render_js_core()
    """
// Shared, embed-safe core. Guarded so two PlotOutputs on one page don't collide
// (A11); the tooltip is a single lazily-created node on document.body, NOT a fixed
// global id, so concatenated fragments share it instead of fighting over it.
if (typeof window.__mem_core === 'undefined') {
    window.__mem_core = {
        tip: d3.select('body').append('div').attr('class','tooltip'),
        fmt: function fmt(v) {
            if (v === null || v === undefined) return '';
            return Math.abs(v) >= 1000 ? v.toFixed(1) : (Math.abs(v) >= 1 ? v.toFixed(3) : v.toFixed(4));
        },
        showTip: function showTip(evt, html) {
            window.__mem_core.tip.html(html).style('opacity',1)
                .style('left',(evt.pageX+12)+'px').style('top',(evt.pageY-28)+'px');
        },
        hideTip: function hideTip() { window.__mem_core.tip.style('opacity',0); }
    };
}
// var aliases are redeclaration-safe across concatenated fragments (unlike const),
// so renderer call sites keep using fmt()/showTip()/hideTip() unchanged.
var fmt = window.__mem_core.fmt;
var showTip = window.__mem_core.showTip;
var hideTip = window.__mem_core.hideTip;
"""
end

# Build the axis-label <text> appends. Labels are emitted as JSON string literals
# (A8 JS-string sink) and omitted entirely, Julia-side, when empty — so no raw user
# text is ever interpolated into quoted JS. `g`/`w`/`h` are the D3 group/width/height
# var names in the caller's scope; `yl_y` is the y-axis-label y offset.
function _axis_labels_js(xlabel::AbstractString, ylabel::AbstractString;
                         g::String="g", w::String="w", h::String="h",
                         yl_y::String="-42")
    s = ""
    if !isempty(xlabel)
        s *= "    $(g).append('text').attr('x',$(w)/2).attr('y',$(h)+30).attr('text-anchor','middle')" *
             ".attr('font-size','11px').attr('fill','#666').text($(_json(xlabel)));\n"
    end
    if !isempty(ylabel)
        s *= "    $(g).append('text').attr('transform','rotate(-90)').attr('x',-$(h)/2).attr('y',$(yl_y))" *
             ".attr('text-anchor','middle').attr('font-size','11px').attr('fill','#666').text($(_json(ylabel)));\n"
    end
    s
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
- `ref_lines_json`: JSON array of {value, color, dash, axis} reference lines
  (`axis`: "y" (default) = horizontal at y(value); "x" = vertical at x(value))
- `x_ticks_json`: `"null"` (integer/auto ticks) or a JSON array of {v, label}
  drawn only at real data x-values (PLT-08 date axes)
- `regions_json`: JSON array of {x0, x1, color, alpha} shaded x-ranges
  (e.g. recession bands), drawn behind the series
- `xlabel`, `ylabel`: axis labels
"""
function _render_line_js(id::String, data_json::String, series_json::String;
                         bands_json::String="[]", ref_lines_json::String="[]",
                         x_ticks_json::String="null", regions_json::String="[]",
                         xlabel::String="", ylabel::String="")
    # Emit the x-axis call. When x_ticks_json == "null" this is byte-identical to
    # the historical integer-tick axis; otherwise ticks/labels come only from the
    # supplied data x-values (no fractional/date-less ticks). (PLT-08)
    xaxis_js = if x_ticks_json == "null"
        "g.append('g').attr('class','axis').attr('transform','translate(0,'+h+')')\n" *
        "        .call(d3.axisBottom(x).ticks(Math.min(xVals.length,8)));"
    else
        "const _xt = $(x_ticks_json);\n" *
        "    const _xtm = new Map(_xt.map(t => [t.v, t.label]));\n" *
        "    const _xtv = _xt.map(t => t.v);\n" *
        "    const _xtk = Math.max(1, Math.ceil(_xtv.length / 8));\n" *
        "    g.append('g').attr('class','axis').attr('transform','translate(0,'+h+')')\n" *
        "        .call(d3.axisBottom(x).tickValues(_xtv.filter((_,i) => i % _xtk === 0)).tickFormat(v => { const l = _xtm.get(v); return l === undefined ? '' : l; }));"
    end
    """
(function() {
    const data = $(data_json);
    const series = $(series_json);
    const bands = $(bands_json);
    const refLines = $(ref_lines_json);
    const regions = $(regions_json);

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
    refLines.forEach(r => { if((r.axis||'y') !== 'x') allYVals.push(r.value); });

    const x = d3.scaleLinear().domain(d3.extent(xVals)).range([0,w]);
    const yExt = d3.extent(allYVals);
    const yPad = (yExt[1]-yExt[0])*0.08 || 1;
    const y = d3.scaleLinear().domain([yExt[0]-yPad, yExt[1]+yPad]).range([h,0]);

    // Grid
    g.append('g').attr('class','grid').call(d3.axisLeft(y).tickSize(-w).tickFormat(''));

    // Shaded x-regions (e.g. recession bands) — drawn behind series
    regions.forEach(rg => {
        g.append('rect').attr('x', x(rg.x0)).attr('y', 0)
            .attr('width', Math.max(0, x(rg.x1) - x(rg.x0))).attr('height', h)
            .attr('fill', rg.color || '#888').attr('opacity', rg.alpha || 0.12);
    });

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

    // Reference lines (axis:"y" → horizontal at y(value); axis:"x" → vertical at x(value))
    refLines.forEach(r => {
        if((r.axis||'y') === 'x') {
            g.append('line').attr('x1',x(r.value)).attr('x2',x(r.value))
                .attr('y1',0).attr('y2',h)
                .attr('stroke',r.color||'#999').attr('stroke-width',1)
                .attr('stroke-dasharray',r.dash||'4,3');
        } else {
            g.append('line').attr('x1',0).attr('x2',w)
                .attr('y1',y(r.value)).attr('y2',y(r.value))
                .attr('stroke',r.color||'#999').attr('stroke-width',1)
                .attr('stroke-dasharray',r.dash||'4,3');
        }
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
    $(xaxis_js)
    g.append('g').attr('class','axis').call(d3.axisLeft(y).ticks(6));

$(_axis_labels_js(xlabel, ylabel))

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

- `x_ticks_json`: `"null"` (integer/auto ticks) or a JSON array of {v, label}
  drawn only at real data x-values (PLT-08 date axes).
"""
function _render_area_js(id::String, data_json::String, series_json::String;
                         x_ticks_json::String="null",
                         xlabel::String="", ylabel::String="Proportion")
    xaxis_js = if x_ticks_json == "null"
        "g.append('g').attr('class','axis').attr('transform','translate(0,'+h+')')\n" *
        "        .call(d3.axisBottom(x).ticks(Math.min(data.length,8)));"
    else
        "const _xt = $(x_ticks_json);\n" *
        "    const _xtm = new Map(_xt.map(t => [t.v, t.label]));\n" *
        "    const _xtv = _xt.map(t => t.v);\n" *
        "    const _xtk = Math.max(1, Math.ceil(_xtv.length / 8));\n" *
        "    g.append('g').attr('class','axis').attr('transform','translate(0,'+h+')')\n" *
        "        .call(d3.axisBottom(x).tickValues(_xtv.filter((_,i) => i % _xtk === 0)).tickFormat(v => { const l = _xtm.get(v); return l === undefined ? '' : l; }));"
    end
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
    $(xaxis_js)
    g.append('g').attr('class','axis').call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.0%')));

$(_axis_labels_js(xlabel, ylabel))

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
- `orientation`: "v" (vertical, categories on x-band) or "h" (horizontal, named
  categories on the y-band with values on a horizontal linear/log x-axis). Use "h"
  for named entities (news impacts, coefficients) so labels read horizontally.
- `logscale`: log-scale the value axis (log-y when vertical, log-x when horizontal);
  intended for all-positive magnitudes (e.g. singular values). No zero line when set.
"""
function _render_bar_js(id::String, data_json::String, series_json::String;
                        mode::String="stacked", orientation::String="v",
                        logscale::Bool=false, xlabel::String="", ylabel::String="")
    """
(function() {
    const data = $(data_json);
    const series = $(series_json);
    const mode = '$(mode)';
    const orientation = '$(orientation)';
    const logscale = $(logscale ? "true" : "false");

    const container = d3.select('#$(id)');
    const W = Math.max(container.node().clientWidth - 24, 280);
    const keys = series.map(s => s.key);

    if(orientation === 'h') {
        // ---- Horizontal: named categories on the y-band, values on x -------
        const margin = {top:10, right:20, bottom:35, left:120};
        const w = W - margin.left - margin.right;
        const rowStep = 26;
        const h = Math.max(data.length * rowStep, 60);

        const svg = container.append('svg').attr('width', W).attr('height', h + margin.top + margin.bottom);
        const g = svg.append('g').attr('transform', 'translate('+margin.left+','+margin.top+')');

        const y = d3.scaleBand().domain(data.map(d=>d.x)).range([0,h]).padding(0.15);

        let vMin = 0, vMax = 0;
        let stacked = null;
        if(mode === 'stacked') {
            const stack = d3.stack().keys(keys).order(d3.stackOrderNone).offset(d3.stackOffsetDiverging);
            stacked = stack(data);
            stacked.forEach(layer => layer.forEach(d => { vMin = Math.min(vMin,d[0],d[1]); vMax = Math.max(vMax,d[0],d[1]); }));
        } else {
            data.forEach(d => keys.forEach(k => { vMin = Math.min(vMin,d[k]||0); vMax = Math.max(vMax,d[k]||0); }));
        }

        let x;
        if(logscale) {
            const lo = vMax > 0 ? (vMin > 0 ? vMin : vMax/1e6) : 1e-6;
            x = d3.scaleLog().domain([lo, vMax > 0 ? vMax : 1]).range([0,w]).clamp(true);
        } else {
            const vPad = (vMax - vMin) * 0.08 || 1;
            x = d3.scaleLinear().domain([vMin - (vMin<0?vPad:0), vMax + vPad]).range([0,w]);
        }
        const barBase = logscale ? x.range()[0] : x(0);
        const barX = v => logscale ? Math.min(barBase, x(v)) : Math.min(x(0), x(v));
        const barW = v => logscale ? Math.abs(x(v) - barBase) : Math.abs(x(v) - x(0));

        g.append('g').attr('class','grid').attr('transform','translate(0,'+h+')')
            .call(d3.axisBottom(x).tickSize(-h).tickFormat(''));

        if(mode === 'stacked') {
            g.selectAll('.layer').data(stacked).join('g').attr('class','layer')
                .attr('fill', (d,i) => series[i].color)
                .selectAll('rect').data(d => d).join('rect')
                .attr('y', d => y(d.data.x))
                .attr('x', d => x(Math.min(d[0], d[1])))
                .attr('width', d => Math.abs(x(d[0]) - x(d[1])))
                .attr('height', y.bandwidth());
        } else {
            const y1 = d3.scaleBand().domain(keys).range([0, y.bandwidth()]).padding(0.05);
            g.selectAll('.group').data(data).join('g').attr('class','group')
                .attr('transform', d => 'translate(0,'+y(d.x)+')')
                .selectAll('rect').data(d => keys.map(k => ({key:k, val:d[k]||0})))
                .join('rect')
                .attr('y', d => y1(d.key))
                .attr('x', d => barX(d.val))
                .attr('width', d => barW(d.val))
                .attr('height', y1.bandwidth())
                .attr('fill', d => series.find(s=>s.key===d.key).color);
        }

        // Zero line (vertical) — omitted on a log scale (no zero)
        if(!logscale) {
            g.append('line').attr('x1',x(0)).attr('x2',x(0)).attr('y1',0).attr('y2',h)
                .attr('stroke','#333').attr('stroke-width',0.8);
        }

        g.append('g').attr('class','axis').attr('transform','translate(0,'+h+')')
            .call(d3.axisBottom(x).ticks(6));
        g.append('g').attr('class','axis')
            .call(d3.axisLeft(y).tickFormat(d => d.length > 16 ? d.slice(0,14)+'..' : d));

$(_axis_labels_js(xlabel, ylabel; yl_y="-(margin.left-12)"))

        // Legend
        if(series.length > 1) {
            const leg = g.append('g').attr('class','legend').attr('transform','translate(5,-5)');
            series.forEach((s,i) => {
                const gi = leg.append('g').attr('transform','translate('+(i*90)+',0)');
                gi.append('rect').attr('width',12).attr('height',12).attr('y',-6).attr('fill',s.color);
                gi.append('text').attr('x',16).attr('y',4).attr('font-size','10px').attr('fill','#555').text(s.name);
            });
        }

        // Tooltip (per y-band)
        svg.append('rect').attr('width',W).attr('height',h+margin.top+margin.bottom)
            .attr('fill','none').attr('pointer-events','all')
            .on('mousemove', function(evt) {
                const [,my] = d3.pointer(evt, g.node());
                const cats = y.domain();
                const idx = Math.min(Math.max(0,Math.floor(my / (h / cats.length))), cats.length-1);
                const d = data[idx];
                if(!d) return;
                let html = '<b>'+d.x+'</b>';
                series.forEach(s => { if(d[s.key]!==undefined) html += '<br>'+s.name+': '+fmt(d[s.key]); });
                showTip(evt, html);
            })
            .on('mouseout', hideTip);
        return;
    }

    // ---- Vertical (default): categories on the x-band, values on y ---------
    const margin = {top:10, right:15, bottom:35, left:55};
    const w = W - margin.left - margin.right;
    const h = Math.min(w * 0.6, 250);

    const svg = container.append('svg').attr('width', W).attr('height', h + margin.top + margin.bottom);
    const g = svg.append('g').attr('transform', 'translate('+margin.left+','+margin.top+')');

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
        const y = logscale ?
            d3.scaleLog().domain([yMax>0?(yMin>0?yMin:yMax/1e6):1e-6, yMax>0?yMax:1]).range([h,0]).clamp(true) :
            d3.scaleLinear().domain([yMin - yPad, yMax + yPad]).range([h, 0]);

        g.append('g').attr('class','grid').call(d3.axisLeft(y).tickSize(-w).tickFormat(''));
        g.selectAll('.layer').data(stacked).join('g').attr('class','layer')
            .attr('fill', (d,i) => series[i].color)
            .selectAll('rect').data(d => d).join('rect')
            .attr('x', d => x(d.data.x))
            .attr('y', d => y(Math.max(d[0], d[1])))
            .attr('height', d => Math.abs(y(d[0]) - y(d[1])))
            .attr('width', x.bandwidth());

        // Zero line
        if(!logscale) {
            g.append('line').attr('x1',0).attr('x2',w)
                .attr('y1',y(0)).attr('y2',y(0))
                .attr('stroke','#333').attr('stroke-width',0.8);
        }

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
        const y = logscale ?
            d3.scaleLog().domain([yMax>0?(yMin>0?yMin:yMax/1e6):1e-6, yMax>0?yMax:1]).range([h,0]).clamp(true) :
            d3.scaleLinear().domain([yMin - yPad, yMax + yPad]).range([h, 0]);
        const yBase = logscale ? y.range()[0] : y(0);
        const x1 = d3.scaleBand().domain(keys).range([0, x.bandwidth()]).padding(0.05);

        g.append('g').attr('class','grid').call(d3.axisLeft(y).tickSize(-w).tickFormat(''));
        g.selectAll('.group').data(data).join('g').attr('class','group')
            .attr('transform', d => 'translate('+x(d.x)+',0)')
            .selectAll('rect').data(d => keys.map(k => ({key:k, val:d[k]||0})))
            .join('rect')
            .attr('x', d => x1(d.key))
            .attr('y', d => logscale ? (d.val>0?y(d.val):h) : y(Math.max(0, d.val)))
            .attr('height', d => logscale ? Math.max(0,yBase-y(Math.max(d.val,y.domain()[0]))) : Math.abs(y(0) - y(d.val)))
            .attr('width', x1.bandwidth())
            .attr('fill', d => series.find(s=>s.key===d.key).color);

        if(!logscale) {
            g.append('line').attr('x1',0).attr('x2',w)
                .attr('y1',y(0)).attr('y2',y(0))
                .attr('stroke','#333').attr('stroke-width',0.8);
        }

        g.append('g').attr('class','axis').attr('transform','translate(0,'+h+')')
            .call(d3.axisBottom(x).tickValues(x.domain().filter((d,i) =>
                i % Math.max(1,Math.floor(data.length/8)) === 0)));
        g.append('g').attr('class','axis').call(d3.axisLeft(y).ticks(6));
    }

$(_axis_labels_js(xlabel, ylabel))

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
# Heatmap Chart Renderer
# =============================================================================

"""
Generate D3.js code for a color-coded heatmap matrix.
- `data_json`: JSON array of {x, y, v} objects (v can be null for missing)
- `row_labels_json`: JSON array of row label strings
- `col_labels_json`: JSON array of column label strings
- `color_domain`: [min, max] for the diverging color scale
"""
function _render_heatmap_js(id::String, data_json::String,
                            row_labels_json::String, col_labels_json::String;
                            xlabel::String="", ylabel::String="",
                            color_domain::Vector{<:Real}=[-3, 3])
    cmin, cmax = color_domain[1], color_domain[2]
    """
(function() {
    const data = $(data_json);
    const rowLabels = $(row_labels_json);
    const colLabels = $(col_labels_json);

    const container = d3.select('#$(id)');
    const W = Math.max(container.node().clientWidth - 24, 280);
    const maxLabelW = 120;
    const margin = {top:10, right:15, bottom:35, left: maxLabelW};
    const w = W - margin.left - margin.right;
    const cellH = Math.max(Math.min(18, 250 / rowLabels.length), 8);
    const h = cellH * rowLabels.length;

    const svg = container.append('svg').attr('width', W).attr('height', h + margin.top + margin.bottom);
    const g = svg.append('g').attr('transform', 'translate('+margin.left+','+margin.top+')');

    const x = d3.scaleBand().domain(colLabels).range([0, w]).padding(0.02);
    const y = d3.scaleBand().domain(rowLabels).range([0, h]).padding(0.02);

    const color = d3.scaleSequential(t => d3.interpolateRdBu(1 - t))
        .domain([$(cmin), $(cmax)]);
    const grey = '#d9d9d9';

    g.selectAll('rect.cell').data(data).join('rect').attr('class','cell')
        .attr('x', d => x(d.x))
        .attr('y', d => y(d.y))
        .attr('width', x.bandwidth())
        .attr('height', y.bandwidth())
        .attr('fill', d => d.v === null ? grey : color(Math.max($(cmin), Math.min($(cmax), d.v))))
        .attr('rx', 1)
        .on('mouseover', function(evt, d) {
            const val = d.v === null ? 'Missing' : fmt(d.v);
            showTip(evt, '<b>'+d.y+'</b><br>Period '+d.x+': '+val);
        })
        .on('mouseout', hideTip);

    // Axes
    g.append('g').attr('class','axis').attr('transform','translate(0,'+h+')')
        .call(d3.axisBottom(x).tickValues(colLabels.filter((d,i) =>
            i % Math.max(1, Math.floor(colLabels.length/10)) === 0)));
    g.append('g').attr('class','axis')
        .call(d3.axisLeft(y).tickFormat(d => d.length > 20 ? d.slice(0,18)+'..' : d));

$(_axis_labels_js(xlabel, ylabel; yl_y="-maxLabelW+10"))
})();
"""
end

# =============================================================================
# Panel Body Rendering
# =============================================================================

function _render_body_single(panel::_PanelSpec; title::String="")
    html = ""
    if !isempty(title)
        html *= "<div class=\"figure-title\">$(_esc_html(title))</div>\n"
    end
    html *= """<div class="panel">
<div class="panel-title">$(_esc_html(panel.title))</div>
<div id="$(panel.id)"></div>
</div>"""
    html
end

function _render_body_figure(panels::Vector{_PanelSpec}; title::String="",
                             source::String="", note::String="")
    html = ""
    if !isempty(title)
        html *= "<div class=\"figure-title\">$(_esc_html(title))</div>\n"
    end
    html *= "<div class=\"panel-grid\">\n"
    for p in panels
        html *= """<div class="panel">
<div class="panel-title">$(_esc_html(p.title))</div>
<div id="$(p.id)"></div>
</div>\n"""
    end
    html *= "</div>\n"
    if !isempty(source)
        html *= "<div class=\"figure-source\">$(_esc_html(source))</div>\n"
    end
    if !isempty(note)
        html *= "<div class=\"figure-source\">$(_esc_html(note))</div>\n"
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

    doc_title = isempty(title) ? "MacroEconometricModels Plot" : title
    html = _render_html(; title=doc_title, css=css, body=body, scripts=scripts)
    fragment = _render_fragment(; css=css, body=body, scripts=scripts)
    PlotOutput(html, fragment)
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

# Embeddable fragment for notebook/Documenter pages (A11); full standalone document
# is kept for save_plot/display_plot. Directly-constructed outputs (empty fragment)
# fall back to the full html.
function Base.show(io::IO, ::MIME"text/html", p::PlotOutput)
    print(io, isempty(p.fragment) ? p.html : p.fragment)
end

function Base.show(io::IO, ::MIME"text/plain", p::PlotOutput)
    n = length(p.html)
    print(io, "PlotOutput($(n) bytes, self-contained inline D3.js)")
end

function Base.show(io::IO, p::PlotOutput)
    show(io, MIME"text/plain"(), p)
end
