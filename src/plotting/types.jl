# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Core types and JSON/HTML-escaping utilities for self-contained plotting with
vendored (inline) D3.js.
"""

# =============================================================================
# Output Types
# =============================================================================

"""
    PlotOutput

Self-contained HTML visualization with **vendored (inline) D3.js** — it renders
offline, with no CDN dependency.

# Fields
- `html::String`: complete standalone HTML document (used by `save_plot` /
  `display_plot`), starting with `<!DOCTYPE html>`.
- `fragment::String`: embeddable HTML fragment (no `<!DOCTYPE>`/`<html>`/`<head>`/
  `<body>`), emitted by `show(::MIME"text/html")` so multiple plots can be embedded
  in one notebook/Documenter page without nesting whole documents. Empty for
  directly-constructed outputs (falls back to `html` on show).

# Usage
```julia
p = plot_result(irf_result)
save_plot(p, "irf.html")       # save to file
display_plot(p)                 # open in browser
```
"""
struct PlotOutput
    html::String
    fragment::String
end

# Back-compatible single-argument constructor: no embeddable fragment (show falls
# back to the full document). Keeps `PlotOutput(html)` call sites working.
PlotOutput(html::AbstractString) = PlotOutput(String(html), "")

"""Internal panel specification for multi-panel figures."""
struct _PanelSpec
    id::String
    title::String
    js::String
end

# =============================================================================
# Theme Constants
# =============================================================================

const _PLOT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"
]

const _PLOT_FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif"

const _PLOT_CI_ALPHA = 0.15

"""
    _palette(i) -> String

Categorical color for the 1-based index `i`, cycling through `_PLOT_COLORS` via
`mod1`. An index beyond the palette length wraps rather than throwing — the safe
accessor to use instead of `_PLOT_COLORS[i]` when `i` can exceed the palette length
(plotrule Color: "slicing `_PLOT_COLORS[1:n]` with `n > 20` is an error").
"""
_palette(i::Int)::String = _PLOT_COLORS[mod1(i, length(_PLOT_COLORS))]

"""
    _palette_take(n) -> Vector{String}

`n` categorical colors, cycling through `_PLOT_COLORS`; correct for any `n ≥ 0`.
Drop-in replacement for the unguarded `_PLOT_COLORS[1:n]` slice (which throws a
`BoundsError` the moment `n > length(_PLOT_COLORS)`).
"""
_palette_take(n::Int)::Vector{String} = String[_palette(i) for i in 1:n]

# =============================================================================
# Color system by role (plotrule Color; PLT-13)
# =============================================================================
#
# The single categorical palette is split by role so that (a) the reserved
# alert/reference red is never spent on an ordinary series, (b) sequential vs
# diverging matrix scales are chosen by data sign, and (c) color↔entity mapping is
# stable across every panel of a figure and across a filtered replot.
#
# CVD validation (plotrule Color: "validated for color-vision deficiency once per
# change … record the result in this file"): recorded in docs/plotrule.md.

"""
    _PLOT_ALERT

The single reserved alert / reference hue (`#d62728`). Used for CI bounds,
treatment / zero reference lines, binding-region shading — never as an ordinary
series color (plotrule Color: "red is the alert/reference color").
"""
const _PLOT_ALERT = "#d62728"

"""
    _PLOT_SERIES

The categorical **series** palette, with the reserved alert reds (`#d62728` and its
desaturated twin `#ff9896`) excluded so no ordinary series is drawn in a hue that
collides with a red reference element (plotrule Color). CVD-validated; see
docs/plotrule.md. Assign hues in fixed order and keep them entity-stable via
`_color_map` / `_color_for`.
"""
const _PLOT_SERIES = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"
]

"""
    _PLOT_SERIES_DARK

Dark-surface-validated series variants (NOT an automatic inversion) selected when
the plot renders on a dark background (plotrule Theming; PLT-14). Same length /
order as `_PLOT_SERIES` so an entity keeps its slot across themes.
"""
const _PLOT_SERIES_DARK = [
    "#4e9bd1", "#ffa64d", "#54c254", "#b18ed6",
    "#b07d6f", "#ef9ad6", "#a3a3a3", "#d4d551", "#4fd4e5",
    "#c9dcef", "#ffd3a3", "#c0e8b3", "#dccbeb",
    "#dcc7c0", "#fbd4e8", "#e0e0e0", "#ececa8", "#c6ecf2"
]

"""`_PLOT_SEQUENTIAL` — single-hue D3 interpolator NAME for **nonnegative** matrices
(Leontief inverse, multipliers): light→dark, no meaningless midpoint (plotrule
Color / Heatmaps)."""
const _PLOT_SEQUENTIAL = "Blues"

"""`_PLOT_DIVERGING` — diverging D3 interpolator NAME for **signed** data with a
meaningful midpoint (z-scores, contributions, signed covariances)."""
const _PLOT_DIVERGING = "RdBu"

"""
    _color_map(names) -> Dict{String,String}

Assign `_PLOT_SERIES` colors in **first-seen order** over the figure's full ordered
name list, so entity *j* keeps its color in every panel and a filtered replot that
drops some entities does NOT repaint the survivors (plotrule Color: entity-stable).
Duplicate names collapse to one slot; the palette cycles for more than
`length(_PLOT_SERIES)` distinct names.
"""
function _color_map(names::AbstractVector{<:AbstractString})
    m = Dict{String,String}()
    i = 0
    for nm in names
        s = String(nm)
        haskey(m, s) && continue
        i += 1
        m[s] = _PLOT_SERIES[mod1(i, length(_PLOT_SERIES))]
    end
    m
end

"""
    _colors_for(names) -> Vector{String}

The per-entity series colors for `names`, resolved through `_color_map` (so the
returned vector is entity-stable and collision-free in first-seen order). Drop-in
replacement for a positional `_palette_take(length(names))` slice that additionally
survives filtering.
"""
function _colors_for(names::AbstractVector{<:AbstractString})
    m = _color_map(names)
    String[m[String(n)] for n in names]
end

"""
    _color_for(name) -> String

Deterministic cross-figure color for a single `name` (hash-based), so the same
named entity reads the same color in a *separate* figure where no full name list is
available. Within one figure `_color_map` (order-based, collision-free) wins.
"""
_color_for(name::AbstractString)::String =
    _PLOT_SERIES[mod1(1 + Int(abs(hash(String(name))) % length(_PLOT_SERIES)), length(_PLOT_SERIES))]

# Global counter for unique SVG IDs (thread-safe via Ref)
const _plot_counter = Ref(0)
function _next_plot_id(prefix::String)
    _plot_counter[] += 1
    "$(prefix)_$(_plot_counter[])"
end

# =============================================================================
# Vendored D3.js (self-contained output — plotrule A12)
# =============================================================================

# Cache the vendored D3 blob in a Ref, populated LAZILY on first use — never at
# precompile/module-init time, since const-Ref caches filled during precompilation
# do not survive into the running session.
const _D3_SOURCE = Ref{String}("")

"""
    _d3_source() -> String

Return the vendored D3.js v7.8.5 UMD source (read once from
`src/plotting/assets/d3.v7.min.js`, then cached). Inlined into every `PlotOutput`
so plots render offline (plotrule A12).
"""
function _d3_source()
    if isempty(_D3_SOURCE[])
        _D3_SOURCE[] = read(joinpath(@__DIR__, "assets", "d3.v7.min.js"), String)
    end
    _D3_SOURCE[]
end

# =============================================================================
# HTML / JSON escaping (plotrule A7/A8)
# =============================================================================

"""
    _esc_html(s) -> String

Escape a user string for the **HTML-text sink** (panel/figure titles, `<title>`,
figure source/note). `&` first. See plotrule A8.
"""
_esc_html(s::AbstractString) = replace(string(s),
    "&" => "&amp;", "<" => "&lt;", ">" => "&gt;", "\"" => "&quot;", "'" => "&#39;")

# =============================================================================
# Minimal JSON Serializer
# =============================================================================

# String serialization for the JSON / JS-string-literal sink (plotrule A7). Covers
# \\ " \n \r \t, all control chars < U+0020, and — because output is embedded in a
# <script> block — `<` (as <, which neutralizes </script> and <!-- while
# decoding back to `<` as displayed text) plus the JS line separators U+2028/U+2029.
function _json(x::AbstractString)
    io = IOBuffer()
    print(io, '"')
    for c in x
        if     c == '\\'      print(io, "\\\\")
        elseif c == '"'       print(io, "\\\"")
        elseif c == '\n'      print(io, "\\n")
        elseif c == '\r'      print(io, "\\r")
        elseif c == '\t'      print(io, "\\t")
        elseif c == '<'       print(io, "\\u003c")
        elseif c == '\u2028'  print(io, "\\u2028")
        elseif c == '\u2029'  print(io, "\\u2029")
        elseif c < ' '   print(io, "\\u", lpad(string(UInt16(c), base=16), 4, '0'))
        else                  print(io, c)
        end
    end
    print(io, '"')
    String(take!(io))
end

function _json(x::Number)
    (isnan(x) || isinf(x)) && return "null"
    string(x)
end

_json(::Nothing) = "null"
_json(::Missing) = "null"
_json(x::Bool) = x ? "true" : "false"
_json(x::Symbol) = _json(string(x))
# Calendar values → ISO-8601 quoted string (plotrule A7; PLT-08 date axes).
_json(x::Dates.Date) = _json(string(x))
_json(x::Dates.DateTime) = _json(string(x))
_json(x::AbstractVector) = "[" * join([_json(v) for v in x], ",") * "]"

function _json_obj(pairs::Vector{Pair{String,String}})
    "{" * join(["\"$(k)\":$(v)" for (k, v) in pairs], ",") * "}"
end

function _json_array_of_objects(rows::Vector{Vector{Pair{String,String}}})
    "[" * join([_json_obj(row) for row in rows], ",\n") * "]"
end
