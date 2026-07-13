# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Shared PrettyTables formatting utilities for publication-quality display.

Provides a unified borderless table format, backend switching (text/LaTeX/HTML),
and common formatting helpers used across all show methods in the package.
"""

using PrettyTables
using ScopedValues: ScopedValue
import ScopedValues

# =============================================================================
# Display Backend Configuration
# =============================================================================

# Display backend (:text, :latex, :html). Hybrid design (#249, finding G-08):
#   _DISPLAY_BACKEND_DEFAULT — process-wide default, set by `set_display_backend`
#                              (mutating; preserves the historical top-level API).
#   _DISPLAY_BACKEND         — per-task scoped override set by `with_display_backend`;
#                              `nothing` ⇒ fall back to the process default.
# `get_display_backend` reads the scoped value FIRST, so two concurrent tasks each
# under their own `with_display_backend` scope never collide (a `ScopedValue` is
# task-inherited; the old single `Ref` was shared and order-dependent).
const _DISPLAY_BACKEND_DEFAULT = Ref{Symbol}(:text)
const _DISPLAY_BACKEND = ScopedValue{Union{Nothing,Symbol}}(nothing)

"""
    set_display_backend(backend::Symbol)

Set the process-wide default PrettyTables output backend. Options: `:text` (default),
`:latex`, `:html`. For a task-local, concurrency-safe override use
[`with_display_backend`](@ref) instead.

# Examples
```julia
set_display_backend(:latex)   # all show() methods now emit LaTeX
set_display_backend(:html)    # switch to HTML tables
set_display_backend(:text)    # back to terminal-friendly text
```
"""
function set_display_backend(backend::Symbol)
    backend ∈ (:text, :latex, :html) || throw(ArgumentError("backend must be :text, :latex, or :html, got :$backend"))
    _DISPLAY_BACKEND_DEFAULT[] = backend
end

"""
    get_display_backend() -> Symbol

Return the PrettyTables display backend in effect (`:text`, `:latex`, or `:html`):
the task-scoped override from an enclosing [`with_display_backend`](@ref) if one is
active, otherwise the process default set by [`set_display_backend`](@ref).
"""
get_display_backend() = something(_DISPLAY_BACKEND[], _DISPLAY_BACKEND_DEFAULT[])

"""
    with_display_backend(f, backend::Symbol)

Run `f()` with the PrettyTables display backend set to `backend` for the current task
(and any tasks it spawns) only, restoring the previous setting on exit. Unlike
[`set_display_backend`](@ref), this is concurrency-safe: two tasks may render under
different backends simultaneously without interfering with one another.

# Examples
```julia
latex_str = with_display_backend(:latex) do
    sprint(show, model)   # emits LaTeX regardless of the process default
end
```
"""
function with_display_backend(f, backend::Symbol)
    backend ∈ (:text, :latex, :html) || throw(ArgumentError("backend must be :text, :latex, or :html, got :$backend"))
    ScopedValues.with(f, _DISPLAY_BACKEND => backend)
end

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
    kw = Dict{Symbol,Any}(kwargs)
    # Suppress the column-label row entirely when every label is empty (spec/stationarity/
    # note blocks pass column_labels=["",""]); otherwise PrettyTables prints a stray empty
    # header row plus its separator rule → multi-blank-line bands after every title. Only
    # fully-empty label vectors are suppressed — _coef_table (["","Coef.",…]) and
    # _matrix_table (vcat([""],labels)) keep their real headers. (S7/T162)
    if !haskey(kw, :show_column_labels) && haskey(kw, :column_labels)
        cl = kw[:column_labels]
        if !isempty(cl) && all(isequal(""), cl)
            kw[:show_column_labels] = false
        end
    end
    be = get_display_backend()
    if be == :text
        # Disable PrettyTables v3 fit-to-display cropping so significance/CI columns and
        # note rows are never silently dropped in non-TTY output (files, pipes, Documenter,
        # CI logs). A publication table must never truncate its own stars column. (S1/T161)
        pretty_table(io, data; backend = :text, table_format = _TEXT_TABLE_FORMAT,
                     fit_table_in_display_horizontally = false,
                     fit_table_in_display_vertically = false, kw...)
    elseif be == :latex
        pretty_table(io, data; backend = :latex, kw...)
    else
        pretty_table(io, data; backend = :html, kw...)
    end
end

# =============================================================================
# Formatting Helpers
# =============================================================================

"""
    _fmt(x; digits=4) -> String

Fixed-decimal string formatter for table cells (S2/T163). Always returns a `String`
with exactly `digits` decimals so a column's decimal points align (Stata/EViews style),
normalizes `-0.0`→`0.0`, and falls back to `%.3g` scientific notation when `|x|` is
below `5e-5` (would otherwise print as all-zeros) or `≥ 1e6` (would print an unreadable
integer run). `NaN`/`Inf` render as `"NaN"`/`"Inf"`/`"-Inf"`. Routing every table cell
through this one formatter removes the ragged decimals, `-0.0` leaks, and collapsed/raw
exponential numbers that `round()` produced.
"""
function _fmt(x::Real; digits::Int=4)
    isnan(x) && return "NaN"
    isinf(x) && return x > 0 ? "Inf" : "-Inf"
    xf = float(x)
    xf == 0 && (xf = zero(xf))                      # -0.0 -> +0.0 (exact zero)
    ax = abs(xf)
    if xf != 0 && (ax < 5e-5 || ax >= 1e6)
        return Printf.format(Printf.Format("%.3g"), xf)   # e.g. 4.73e+114, 1.23e-06
    end
    s = Printf.format(Printf.Format("%.$(digits)f"), xf)
    # Residual "-0.00.." guard: sub-threshold-but-nonzero values rounded at digits<4 can
    # still render a signed all-zero string; strip the leading minus in that case.
    if startswith(s, "-") && all(c -> c == '0' || c == '.', @view s[nextind(s, 1):end])
        s = s[nextind(s, 1):end]
    end
    return s
end
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

# =============================================================================
# Shared display conventions (S8/T166)
# =============================================================================

const _LABELS = Dict{Symbol,String}(
    :hc0 => "HC0 (robust)", :hc1 => "HC1 (robust)", :hc2 => "HC2 (robust)", :hc3 => "HC3 (robust)",
    :ols => "OLS", :cluster => "Cluster-robust", :newey_west => "Newey–West (HAC)", :robust => "Robust (QMLE)",
    :css => "CSS", :mle => "MLE", :css_mle => "CSS-MLE",
    :normal => "Normal", :direct => "Direct (NIW)", :gibbs => "Gibbs", :minnesota => "Minnesota",
    :ar1 => "AR(1)", :iid => "i.i.d.", :none => "None",
    :jarque_bera => "Jarque–Bera", :doornik_hansen => "Doornik–Hansen",
    :mardia_skewness => "Mardia skewness", :mardia_kurtosis => "Mardia kurtosis",
    :henze_zirkler => "Henze–Zirkler",
)

"""
    _label(s) -> String

Map a publication Symbol/String to a human-readable display name (S8/T166). Unknown
symbols are prettified (title-case, underscores→spaces) rather than dumped raw.
"""
_label(s::Symbol) = get(_LABELS, s) do
    join(uppercasefirst.(split(String(s), '_')), " ")
end
_label(s::AbstractString) = _label(Symbol(s))

"""Render a Bool as Yes/No for publication tables (S8/T166)."""
_yesno(b::Bool) = b ? "Yes" : "No"

# Single canonical p-value formatter name (S8/T166); _format_pvalue is the implementation.
const _fmt_pvalue = _format_pvalue

# Intercept display convention (S8/T166): show "(Intercept)" everywhere. Internal tokens
# ("const"/"_cons") stay intact for margins/detection logic — map only at render time.
const _INTERCEPT_LABEL = "(Intercept)"
_display_intercept(name::AbstractString) =
    name in ("const", "_cons", "Intercept (c)", "(Intercept)") ? _INTERCEPT_LABEL : name

"""Select representative horizons for display.

`unique` drops the duplicated endpoint when `H` coincides with a fixed anchor (e.g. `H=8`
gave `[1,4,8,8]` → a doubled row in forecast/IRF tables). `unique` preserves
first-occurrence order and never reorders. (B2/T165) NOTE follow-up: for `H` in `6:7` the
anchor `8` exceeds `H`; that pre-existing quirk is out of T165 scope."""
function _select_horizons(H::Int)
    H <= 5 && return collect(1:H)
    hs = H <= 12 ? [1, 4, 8, H] :
         H <= 24 ? [1, 4, 8, 12, H] :
                   [1, 4, 8, 12, 24, H]
    return unique(hs)
end

"""
    _coef_table(io, title, names, coefs, se; dist=:z, dof_r=0, level=0.95)

Publication-quality 7-column coefficient table (Stata/EViews style).

Columns: Name | Coef. | Std.Err. | z/t | P>|z/t| | [95% CI lower | CI upper] | stars
"""
function _coef_table(io::IO, title::String, names::Vector{String},
                     coefs::Vector{T}, se::Vector{T};
                     dist::Symbol=:z, dof_r::Int=0, level::Real=0.95,
                     ref_rows::Union{Nothing,AbstractVector{Int}}=nothing,
                     coef_label::String="Coef.") where {T}
    n = length(names)
    alpha = 1 - level
    z_crit = dist == :z ? T(quantile(Normal(), 1 - alpha/2)) :
                          T(quantile(TDist(dof_r), 1 - alpha/2))
    stat_label = dist == :z ? "z" : "t"
    ci_pct = round(Int, 100 * level)
    tol = sqrt(eps(T))

    data = Matrix{Any}(undef, n, 8)
    for i in 1:n
        est = coefs[i]
        se_i = se[i]
        is_ref = ref_rows !== nothing && i in ref_rows
        # Guard against "significant zero" rows (S6/T164): a row is degenerate when it is a
        # reference period, when its z-ratio would be astronomically large (se is dust
        # relative to |est|, i.e. only z ≳ 1/√eps), when both est and se are ~0, or when
        # either is non-finite. Degenerate rows must never print a computed stat/p/stars.
        degen = is_ref ||
                se_i <= tol * max(abs(est), one(T)) ||
                (abs(est) <= tol && se_i <= tol) ||
                !isfinite(est) || !isfinite(se_i)
        if degen
            data[i, 1] = is_ref ? names[i] * " (ref)" : names[i]
            data[i, 2] = is_ref ? "—" : _fmt(est)   # ref rows carry no estimate; keep raw dust otherwise
            data[i, 3] = is_ref ? "—" : _fmt(se_i)
            data[i, 4] = "—"        # stat
            data[i, 5] = "—"        # p-value
            data[i, 6] = "—"        # CI lower (meaningless here)
            data[i, 7] = "—"        # CI upper
            data[i, 8] = ""         # no stars
        else
            stat = est / se_i
            pval = dist == :z ? T(2) * (one(T) - cdf(Normal(), abs(stat))) :
                                T(2) * (one(T) - cdf(TDist(dof_r), abs(stat)))
            data[i, 1] = names[i]
            data[i, 2] = _fmt(est)
            data[i, 3] = _fmt(se_i)
            data[i, 4] = string(_fmt(stat))
            data[i, 5] = _format_pvalue(pval)
            data[i, 6] = _fmt(est - z_crit * se_i)
            data[i, 7] = _fmt(est + z_crit * se_i)
            data[i, 8] = _significance_stars(pval)
        end
    end

    _pretty_table(io, data;
        title = title,
        column_labels = ["", coef_label, "Std.Err.", stat_label, "P>|$stat_label|", "[$ci_pct%", "CI]", ""],
        alignment = [:l, :r, :r, :r, :r, :r, :r, :l],
    )
end

"""Print the significance legend as a plain text line — never a table, so it cannot
inherit the empty-header band or the horizontal-crop truncation. (S7/T162)"""
function _sig_legend(io::IO)
    println(io, "Significance: *** p<0.01, ** p<0.05, * p<0.10")
end

"""Emit a one-line degenerate-fit warning when any coefficient is non-finite or exploded
(`|coef| > 1e10`) — a symptom of perfect separation or severe collinearity — so a report
can never silently certify such a fit as "Converged Yes". Plain text, not a table. The
zero-SE symptom is deliberately NOT a trigger (it false-fires on legitimately
boundary-pinned GMM/SMM parameters); the exploded/non-finite coefficient is the reliable
signal. (S6/T164)"""
function _degenerate_fit_banner(io::IO, coefs::AbstractVector)
    if any(!isfinite, coefs) || any(c -> abs(c) > 1e10, coefs)
        println(io, "WARNING: degenerate fit — non-finite or exploded coefficients " *
                    "(possible perfect separation / collinearity); standard errors unreliable.")
    end
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
            data[i, j+1] = _fmt(M[i, j]; digits=digits)   # share the fixed-decimal convention (S2/T163)
        end
    end
    _pretty_table(io, data;
        title = title,
        column_labels = vcat([""], col_labels),
        alignment = vcat([:l], fill(:r, m))
    )
end
