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
_json(x::AbstractVector) = "[" * join([_json(v) for v in x], ",") * "]"

function _json_obj(pairs::Vector{Pair{String,String}})
    "{" * join(["\"$(k)\":$(v)" for (k, v) in pairs], ",") * "}"
end

function _json_array_of_objects(rows::Vector{Vector{Pair{String,String}}})
    "[" * join([_json_obj(row) for row in rows], ",\n") * "]"
end
