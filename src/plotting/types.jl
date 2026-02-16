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
Core types and JSON utilities for inline D3.js plotting.
"""

# =============================================================================
# Output Types
# =============================================================================

"""
    PlotOutput

Self-contained HTML document with inline D3.js visualization.

# Fields
- `html::String`: Complete HTML document string

# Usage
```julia
p = plot_result(irf_result)
save_plot(p, "irf.html")       # save to file
display_plot(p)                 # open in browser
```
"""
struct PlotOutput
    html::String
end

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
# Minimal JSON Serializer
# =============================================================================

_json(x::AbstractString) = "\"" * replace(replace(replace(string(x),
    "\\" => "\\\\"), "\"" => "\\\""), "\n" => "\\n") * "\""

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
