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
Shared PrettyTables formatting utilities for publication-quality display.

Provides a unified borderless table format, backend switching (text/LaTeX/HTML),
and common formatting helpers used across all show methods in the package.
"""

using PrettyTables

# =============================================================================
# Display Backend Configuration
# =============================================================================

# Global display backend (:text, :latex, :html)
const _DISPLAY_BACKEND = Ref{Symbol}(:text)

"""
    set_display_backend(backend::Symbol)

Set the PrettyTables output backend. Options: `:text` (default), `:latex`, `:html`.

# Examples
```julia
set_display_backend(:latex)   # all show() methods now emit LaTeX
set_display_backend(:html)    # switch to HTML tables
set_display_backend(:text)    # back to terminal-friendly text
```
"""
function set_display_backend(backend::Symbol)
    backend ∈ (:text, :latex, :html) || throw(ArgumentError("backend must be :text, :latex, or :html, got :$backend"))
    _DISPLAY_BACKEND[] = backend
end

"""
    get_display_backend() -> Symbol

Return the current PrettyTables display backend (`:text`, `:latex`, or `:html`).
"""
get_display_backend() = _DISPLAY_BACKEND[]

# Shared borderless text table format (Stata-style)
const _TEXT_TABLE_FORMAT = TextTableFormat(
    borders = text_table_borders__borderless,
    horizontal_line_after_column_labels = true
)

"""
    _pretty_table(io::IO, data; kwargs...)

Central PrettyTables wrapper that respects the global display backend.

For `:text` backend, applies `_TEXT_TABLE_FORMAT` automatically.
For `:latex` and `:html` backends, omits text-only formatting options.
"""
function _pretty_table(io::IO, data; kwargs...)
    be = _DISPLAY_BACKEND[]
    if be == :text
        pretty_table(io, data; backend = :text, table_format = _TEXT_TABLE_FORMAT, kwargs...)
    elseif be == :latex
        pretty_table(io, data; backend = :latex, kwargs...)
    else
        pretty_table(io, data; backend = :html, kwargs...)
    end
end

# =============================================================================
# Formatting Helpers
# =============================================================================

_fmt(x::Real; digits::Int=4) = round(x, digits=digits)
_fmt_pct(x::Real; digits::Int=1) = string(round(x * 100, digits=digits), "%")

function _format_pvalue(pval::Real)
    pval < 0.001 && return "<0.001"
    pval > 0.999 && return ">0.999"
    return string(round(pval, digits=4))
end

function _significance_stars(pvalue::Real)
    pvalue < 0.01 && return "***"
    pvalue < 0.05 && return "**"
    pvalue < 0.10 && return "*"
    return ""
end

"""Select representative horizons for display."""
function _select_horizons(H::Int)
    H <= 5 && return collect(1:H)
    H <= 12 && return [1, 4, 8, H]
    H <= 24 && return [1, 4, 8, 12, H]
    return [1, 4, 8, 12, 24, H]
end

"""
    _coef_table(io, title, names, coefs, se; dist=:z, dof_r=0, level=0.95)

Publication-quality 7-column coefficient table (Stata/EViews style).

Columns: Name | Coef. | Std.Err. | z/t | P>|z/t| | [95% CI lower | CI upper] | stars
"""
function _coef_table(io::IO, title::String, names::Vector{String},
                     coefs::Vector{T}, se::Vector{T};
                     dist::Symbol=:z, dof_r::Int=0, level::Real=0.95) where {T}
    n = length(names)
    alpha = 1 - level
    z_crit = dist == :z ? T(quantile(Normal(), 1 - alpha/2)) :
                          T(quantile(TDist(dof_r), 1 - alpha/2))
    stat_label = dist == :z ? "z" : "t"
    ci_pct = round(Int, 100 * level)

    data = Matrix{Any}(undef, n, 8)
    for i in 1:n
        est = coefs[i]
        se_i = se[i]
        stat = se_i > 0 ? est / se_i : T(NaN)
        pval = if isnan(stat)
            T(NaN)
        elseif dist == :z
            T(2) * (one(T) - cdf(Normal(), abs(stat)))
        else
            T(2) * (one(T) - cdf(TDist(dof_r), abs(stat)))
        end
        ci_lo = est - z_crit * se_i
        ci_hi = est + z_crit * se_i
        stars = isnan(pval) ? "" : _significance_stars(pval)
        data[i, 1] = names[i]
        data[i, 2] = _fmt(est)
        data[i, 3] = _fmt(se_i)
        data[i, 4] = isnan(stat) ? "—" : string(_fmt(stat))
        data[i, 5] = isnan(pval) ? "—" : _format_pvalue(pval)
        data[i, 6] = _fmt(ci_lo)
        data[i, 7] = _fmt(ci_hi)
        data[i, 8] = stars
    end

    _pretty_table(io, data;
        title = title,
        column_labels = ["", "Coef.", "Std.Err.", stat_label, "P>|$stat_label|", "[$ci_pct%", "CI]", ""],
        alignment = [:l, :r, :r, :r, :r, :r, :r, :l],
    )
end

"""Print significance legend footer."""
function _sig_legend(io::IO)
    _pretty_table(io, Any["Significance" "*** p<0.01, ** p<0.05, * p<0.10"];
        column_labels=["",""], alignment=[:l,:l])
end

"""Print a labeled matrix as a PrettyTables table."""
function _matrix_table(io::IO, M::AbstractMatrix, title::String;
                       row_labels=nothing, col_labels=nothing, digits::Int=4)
    n, m = size(M)
    row_labels = something(row_labels, ["[$i]" for i in 1:n])
    col_labels = something(col_labels, ["[$j]" for j in 1:m])
    data = Matrix{Any}(undef, n, m + 1)
    for i in 1:n
        data[i, 1] = row_labels[i]
        for j in 1:m
            data[i, j+1] = round(M[i, j], digits=digits)
        end
    end
    _pretty_table(io, data;
        title = title,
        column_labels = vcat([""], col_labels),
        alignment = vcat([:l], fill(:r, m))
    )
end
