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
Display methods for DSGE model specifications.

Provides text, LaTeX, and HTML renderers for `DSGESpec{T}`, with recursive
expression converters that handle time-indexed variables, expectation operators,
Greek letter mapping, and operator precedence.
"""

# =============================================================================
# Precedence constants for expression rendering
# =============================================================================

const _PREC_EQ      = 10    # equation-level (=)
const _PREC_ADDSUB  = 50    # addition, subtraction
const _PREC_MULDIV  = 60    # multiplication, division
const _PREC_UNARY   = 70    # unary minus
const _PREC_POWER   = 80    # exponentiation
const _PREC_ATOM    = 100   # atoms (numbers, symbols, fractions)

# =============================================================================
# Greek letter -> LaTeX mapping
# =============================================================================

const _GREEK_LATEX = Dict{String,String}(
    "α" => "\\alpha",
    "β" => "\\beta",
    "γ" => "\\gamma",
    "δ" => "\\delta",
    "ε" => "\\varepsilon",
    "ζ" => "\\zeta",
    "η" => "\\eta",
    "θ" => "\\theta",
    "ι" => "\\iota",
    "κ" => "\\kappa",
    "λ" => "\\lambda",
    "μ" => "\\mu",
    "ν" => "\\nu",
    "ξ" => "\\xi",
    "π" => "\\pi",
    "ρ" => "\\rho",
    "σ" => "\\sigma",
    "τ" => "\\tau",
    "υ" => "\\upsilon",
    "φ" => "\\phi",
    "χ" => "\\chi",
    "ψ" => "\\psi",
    "ω" => "\\omega",
    "Α" => "A",
    "Β" => "B",
    "Γ" => "\\Gamma",
    "Δ" => "\\Delta",
    "Ε" => "E",
    "Ζ" => "Z",
    "Η" => "H",
    "Θ" => "\\Theta",
    "Ι" => "I",
    "Κ" => "K",
    "Λ" => "\\Lambda",
    "Μ" => "M",
    "Ν" => "N",
    "Ξ" => "\\Xi",
    "Π" => "\\Pi",
    "Ρ" => "P",
    "Σ" => "\\Sigma",
    "Τ" => "T",
    "Υ" => "\\Upsilon",
    "Φ" => "\\Phi",
    "Χ" => "X",
    "Ψ" => "\\Psi",
    "Ω" => "\\Omega",
)

# =============================================================================
# Symbol -> LaTeX converter
# =============================================================================

"""
    _sym_to_latex(sym::Symbol) -> String

Convert a Julia symbol to LaTeX representation.

Handles Greek letters, subscripted names (e.g., `φ_π` -> `\\phi_{\\pi}`),
and plain ASCII names. Multi-character subscripts are wrapped in `{}`.
"""
function _sym_to_latex(sym::Symbol)
    s = string(sym)

    # Split on underscore: base_sub
    parts = split(s, "_"; limit=2)
    base = parts[1]
    sub = length(parts) > 1 ? parts[2] : ""

    # Convert base to LaTeX
    base_latex = get(_GREEK_LATEX, base, base)

    if isempty(sub)
        return base_latex
    end

    # Convert subscript to LaTeX (may itself be a Greek letter)
    sub_latex = get(_GREEK_LATEX, sub, sub)

    # Wrap in braces
    return base_latex * "_{" * sub_latex * "}"
end

# =============================================================================
# Number formatting for display
# =============================================================================

"""
    _format_num_display(x::Number) -> String

Format a number for equation display, dropping trailing `.0` for integers.
"""
function _format_num_display(x::Number)
    if x isa Integer
        return string(x)
    end
    # Check if float is effectively an integer
    if isfinite(x) && x == floor(x) && abs(x) < 1e15
        return string(Int(x))
    end
    return string(x)
end

# =============================================================================
# Time-reference renderers
# =============================================================================

"""
    _time_offset(ex) -> Int

Extract the time offset from a ref subscript expression.
Returns 0 for `:t`, +k for `:(t + k)`, -k for `:(t - k)`.
"""
function _time_offset(ex)
    ex === :t && return 0
    if ex isa Expr && ex.head == :call && length(ex.args) == 3 && ex.args[2] === :t
        op = ex.args[1]
        k = ex.args[3]
        k isa Number || return 0
        op === :(+) && return Int(k)
        op === :(-) && return -Int(k)
    end
    return 0
end

"""
    _ref_to_text(ex::Expr, endog::Vector{Symbol}) -> String

Render a `var[t±k]` reference as text.

Forward-looking endogenous variables (offset > 0) are wrapped in `E_t[...]`.
Time subscripts: `y_t`, `y_{t-1}`, `y_{t+1}`.
Subscripted variable names like `ε_d[t]` render as `ε_{d,t}`.
"""
function _ref_to_text(ex::Expr, endog::Vector{Symbol})
    @assert ex.head == :ref && length(ex.args) == 2
    varname = ex.args[1]
    offset = _time_offset(ex.args[2])

    vstr = string(varname)

    # Build time subscript
    tsub = if offset == 0
        "t"
    elseif offset > 0
        "t+$offset"
    else
        "t$offset"  # e.g., "t-1"
    end

    # Handle subscripted variable names: merge subscripts
    # e.g., ε_d with time t -> ε_{d,t}
    base, sub = _split_var_name(vstr)
    if !isempty(sub)
        display_str = base * "_{" * sub * "," * tsub * "}"
    elseif offset == 0
        display_str = vstr * "_t"
    else
        display_str = vstr * "_{" * tsub * "}"
    end

    # Wrap forward-looking endogenous in E_t[...]
    if offset > 0 && varname isa Symbol && varname in endog
        return "E_t[" * display_str * "]"
    end
    return display_str
end

"""
    _ref_to_latex(ex::Expr, endog::Vector{Symbol}) -> String

Render a `var[t±k]` reference as LaTeX.

Forward-looking endogenous variables are wrapped in `\\mathbb{E}_t[...]`.
Uses Greek letter mapping for variable names.
"""
function _ref_to_latex(ex::Expr, endog::Vector{Symbol})
    @assert ex.head == :ref && length(ex.args) == 2
    varname = ex.args[1]
    offset = _time_offset(ex.args[2])

    vstr = string(varname)

    # Build time subscript
    tsub = if offset == 0
        "t"
    elseif offset > 0
        "t+$offset"
    else
        "t$offset"
    end

    # Split variable name on underscore for combined subscript
    base, sub = _split_var_name(vstr)
    base_latex = get(_GREEK_LATEX, base, base)

    if !isempty(sub)
        sub_latex = get(_GREEK_LATEX, sub, sub)
        display_str = base_latex * "_{" * sub_latex * "," * tsub * "}"
    else
        display_str = base_latex * "_{" * tsub * "}"
    end

    if offset > 0 && varname isa Symbol && varname in endog
        return "\\mathbb{E}_t\\left[" * display_str * "\\right]"
    end
    return display_str
end

"""
    _split_var_name(s::String) -> (base::String, sub::String)

Split a variable name on underscore into base and subscript parts.
Returns `("ε", "d")` for `"ε_d"`, or `("y", "")` for `"y"`.
"""
function _split_var_name(s::String)
    parts = split(s, "_"; limit=2)
    base = parts[1]
    sub = length(parts) > 1 ? parts[2] : ""
    return (base, sub)
end

# =============================================================================
# Recursive expression -> text converter
# =============================================================================

"""
    _expr_to_text(ex, endog, exog, params) -> String

Convert a Julia `Expr` (DSGE equation) to a human-readable text string.
Handles arithmetic operators, time-indexed variables, and operator precedence.
"""
function _expr_to_text(ex, endog::Vector{Symbol}, exog::Vector{Symbol}, params::Vector{Symbol})
    str, _ = _expr_to_text_impl(ex, endog, exog, params)
    return str
end

"""
    _expr_to_text_impl(ex, endog, exog, params) -> (String, Int)

Internal recursive converter returning `(text, precedence)`.
"""
function _expr_to_text_impl(ex, endog::Vector{Symbol}, exog::Vector{Symbol}, params::Vector{Symbol})
    # --- Atoms ---
    if ex isa Number
        return (_format_num_display(ex), _PREC_ATOM)
    end
    if ex isa Symbol
        return (string(ex), _PREC_ATOM)
    end

    # Not an Expr — fallback
    if !(ex isa Expr)
        return (string(ex), _PREC_ATOM)
    end

    # --- :ref — time-indexed variable ---
    if ex.head == :ref && length(ex.args) == 2 && ex.args[1] isa Symbol
        varname = ex.args[1]
        all_vars = vcat(endog, exog)
        if varname in all_vars
            return (_ref_to_text(ex, endog), _PREC_ATOM)
        end
        # Unknown ref — render as symbol[subscript]
        return (string(varname) * "[" * string(ex.args[2]) * "]", _PREC_ATOM)
    end

    # --- :call — function/operator ---
    if ex.head == :call && length(ex.args) >= 2
        op = ex.args[1]

        # Addition: +(a, b, c, ...)
        if op === :(+) && length(ex.args) >= 3
            parts = String[]
            for i in 2:length(ex.args)
                s, p = _expr_to_text_impl(ex.args[i], endog, exog, params)
                if i > 2
                    # Check if subexpression starts with minus sign
                    if startswith(s, "-")
                        push!(parts, " - " * lstrip(s[2:end]))
                    else
                        push!(parts, " + " * s)
                    end
                else
                    push!(parts, s)
                end
            end
            return (join(parts, ""), _PREC_ADDSUB)
        end

        # Unary minus: -(a)
        if op === :(-) && length(ex.args) == 2
            s, p = _expr_to_text_impl(ex.args[2], endog, exog, params)
            if p <= _PREC_ADDSUB
                return ("-(" * s * ")", _PREC_UNARY)
            end
            return ("-" * s, _PREC_UNARY)
        end

        # Binary minus: -(a, b)
        if op === :(-) && length(ex.args) == 3
            ls, lp = _expr_to_text_impl(ex.args[2], endog, exog, params)
            rs, rp = _expr_to_text_impl(ex.args[3], endog, exog, params)
            # Wrap rhs if it has lower or equal precedence (to preserve grouping)
            if rp <= _PREC_ADDSUB
                rs = "(" * rs * ")"
            end
            return (ls * " - " * rs, _PREC_ADDSUB)
        end

        # Multiplication: *(a, b, c, ...)
        if op === :(*) && length(ex.args) >= 3
            parts = String[]
            for i in 2:length(ex.args)
                s, p = _expr_to_text_impl(ex.args[i], endog, exog, params)
                if p < _PREC_MULDIV
                    s = "(" * s * ")"
                end
                push!(parts, s)
            end
            return (join(parts, " "), _PREC_MULDIV)
        end

        # Division: /(a, b)
        if op === :(/) && length(ex.args) == 3
            ns, np = _expr_to_text_impl(ex.args[2], endog, exog, params)
            ds, dp = _expr_to_text_impl(ex.args[3], endog, exog, params)
            return ("(" * ns * "/" * ds * ")", _PREC_ATOM)
        end

        # Power: ^(a, b)
        if op === :(^) && length(ex.args) == 3
            bs, bp = _expr_to_text_impl(ex.args[2], endog, exog, params)
            es, ep = _expr_to_text_impl(ex.args[3], endog, exog, params)
            if bp < _PREC_POWER
                bs = "(" * bs * ")"
            end
            # Wrap exponent in parens if it's not atomic (avoid K^α - 1 ambiguity)
            if ep < _PREC_ATOM
                es = "(" * es * ")"
            end
            return (bs * "^" * es, _PREC_POWER)
        end

        # Generic function call: f(args...)
        fname = string(op)
        arg_strs = String[]
        for i in 2:length(ex.args)
            s, _ = _expr_to_text_impl(ex.args[i], endog, exog, params)
            push!(arg_strs, s)
        end
        return (fname * "(" * join(arg_strs, ", ") * ")", _PREC_ATOM)
    end

    # Fallback
    return (string(ex), _PREC_ATOM)
end

# =============================================================================
# Recursive expression -> LaTeX converter
# =============================================================================

"""
    _expr_to_latex(ex, endog, exog, params) -> String

Convert a Julia `Expr` (DSGE equation) to a LaTeX math string.
Handles arithmetic operators, time-indexed variables, Greek letters,
fractions, and operator precedence.
"""
function _expr_to_latex(ex, endog::Vector{Symbol}, exog::Vector{Symbol}, params::Vector{Symbol})
    str, _ = _expr_to_latex_impl(ex, endog, exog, params)
    return str
end

"""
    _expr_to_latex_impl(ex, endog, exog, params) -> (String, Int)

Internal recursive converter returning `(latex_string, precedence)`.
"""
function _expr_to_latex_impl(ex, endog::Vector{Symbol}, exog::Vector{Symbol}, params::Vector{Symbol})
    # --- Atoms ---
    if ex isa Number
        return (_format_num_display(ex), _PREC_ATOM)
    end
    if ex isa Symbol
        return (_sym_to_latex(ex), _PREC_ATOM)
    end

    # Not an Expr — fallback
    if !(ex isa Expr)
        return (string(ex), _PREC_ATOM)
    end

    # --- :ref — time-indexed variable ---
    if ex.head == :ref && length(ex.args) == 2 && ex.args[1] isa Symbol
        varname = ex.args[1]
        all_vars = vcat(endog, exog)
        if varname in all_vars
            return (_ref_to_latex(ex, endog), _PREC_ATOM)
        end
        # Unknown ref
        return (_sym_to_latex(varname) * "_{" * string(ex.args[2]) * "}", _PREC_ATOM)
    end

    # --- :call — function/operator ---
    if ex.head == :call && length(ex.args) >= 2
        op = ex.args[1]

        # Addition: +(a, b, c, ...)
        if op === :(+) && length(ex.args) >= 3
            parts = String[]
            for i in 2:length(ex.args)
                s, p = _expr_to_latex_impl(ex.args[i], endog, exog, params)
                if i > 2
                    if startswith(s, "-")
                        push!(parts, " - " * lstrip(s[2:end]))
                    else
                        push!(parts, " + " * s)
                    end
                else
                    push!(parts, s)
                end
            end
            return (join(parts, ""), _PREC_ADDSUB)
        end

        # Unary minus: -(a)
        if op === :(-) && length(ex.args) == 2
            s, p = _expr_to_latex_impl(ex.args[2], endog, exog, params)
            if p <= _PREC_ADDSUB
                return ("-\\left(" * s * "\\right)", _PREC_UNARY)
            end
            return ("-" * s, _PREC_UNARY)
        end

        # Binary minus: -(a, b)
        if op === :(-) && length(ex.args) == 3
            ls, lp = _expr_to_latex_impl(ex.args[2], endog, exog, params)
            rs, rp = _expr_to_latex_impl(ex.args[3], endog, exog, params)
            if rp <= _PREC_ADDSUB
                rs = "\\left(" * rs * "\\right)"
            end
            return (ls * " - " * rs, _PREC_ADDSUB)
        end

        # Multiplication: *(a, b, c, ...)
        if op === :(*) && length(ex.args) >= 3
            parts = String[]
            for i in 2:length(ex.args)
                s, p = _expr_to_latex_impl(ex.args[i], endog, exog, params)
                if p < _PREC_MULDIV
                    s = "\\left(" * s * "\\right)"
                end
                push!(parts, s)
            end
            return (join(parts, " \\, "), _PREC_MULDIV)
        end

        # Division: /(a, b) -> \frac{num}{den}
        if op === :(/) && length(ex.args) == 3
            ns, _ = _expr_to_latex_impl(ex.args[2], endog, exog, params)
            ds, _ = _expr_to_latex_impl(ex.args[3], endog, exog, params)
            return ("\\frac{" * ns * "}{" * ds * "}", _PREC_ATOM)
        end

        # Power: ^(a, b) -> base^{exp}
        if op === :(^) && length(ex.args) == 3
            bs, bp = _expr_to_latex_impl(ex.args[2], endog, exog, params)
            es, _ = _expr_to_latex_impl(ex.args[3], endog, exog, params)
            if bp < _PREC_POWER
                bs = "\\left(" * bs * "\\right)"
            end
            return (bs * "^{" * es * "}", _PREC_POWER)
        end

        # Generic function call: \mathrm{f}\left(args\right)
        fname = string(op)
        # Common math functions get special LaTeX treatment
        latex_fname = if fname in ("log", "exp", "sin", "cos", "tan", "sqrt")
            "\\" * fname
        else
            "\\mathrm{" * fname * "}"
        end
        arg_strs = String[]
        for i in 2:length(ex.args)
            s, _ = _expr_to_latex_impl(ex.args[i], endog, exog, params)
            push!(arg_strs, s)
        end
        return (latex_fname * "\\left(" * join(arg_strs, ", ") * "\\right)", _PREC_ATOM)
    end

    # Fallback
    return (string(ex), _PREC_ATOM)
end

# =============================================================================
# Equation display helpers
# =============================================================================

"""
    _equation_to_display(eq::Expr, endog, exog, params; mode=:text) -> String

Convert a stored residual equation (LHS - RHS form) back to `LHS = RHS` display.

If the top-level expression is a binary `-` (i.e., `call(:-, A, B)`), it is displayed
as `A = B`. Otherwise, the full expression is shown as `expr = 0`.
"""
function _equation_to_display(eq::Expr, endog::Vector{Symbol}, exog::Vector{Symbol},
                               params::Vector{Symbol}; mode::Symbol=:text)
    converter = mode == :latex ? _expr_to_latex : _expr_to_text

    # Check if top-level is binary subtraction: LHS - RHS
    if eq.head == :call && length(eq.args) == 3 && eq.args[1] === :(-)
        lhs_str = converter(eq.args[2], endog, exog, params)
        rhs_str = converter(eq.args[3], endog, exog, params)
        return lhs_str * " = " * rhs_str
    end

    # Otherwise display as expr = 0
    return converter(eq, endog, exog, params) * " = 0"
end

# =============================================================================
# Steady state display helpers
# =============================================================================

"""
    _steady_state_text(endog::Vector{Symbol}, ss::Vector{T}) -> String

Format steady state values as text with Unicode combining macron.
"""
function _steady_state_text(endog::Vector{Symbol}, ss::AbstractVector)
    lines = String[]
    for (i, v) in enumerate(endog)
        vstr = string(v)
        # Unicode combining macron: char + \u0304
        ss_name = vstr * "\u0304"
        val = ss[i]
        val_str = _format_num_display(isfinite(val) ? round(val; digits=6) : val)
        push!(lines, "  " * ss_name * " = " * val_str)
    end
    return join(lines, "\n")
end

"""
    _steady_state_latex(endog::Vector{Symbol}, ss::AbstractVector) -> String

Format steady state values as LaTeX with `\\bar{}`.
"""
function _steady_state_latex(endog::Vector{Symbol}, ss::AbstractVector)
    lines = String[]
    for (i, v) in enumerate(endog)
        latex_name = _sym_to_latex(v)
        val = ss[i]
        val_str = _format_num_display(isfinite(val) ? round(val; digits=6) : val)
        push!(lines, "\\bar{" * latex_name * "} &= " * val_str * " \\\\")
    end
    return join(lines, "\n")
end

# =============================================================================
# Base.show dispatcher for DSGESpec
# =============================================================================

function Base.show(io::IO, spec::DSGESpec{T}) where {T}
    backend = get_display_backend()
    if backend == :latex
        _show_dsge_latex(io, spec)
    elseif backend == :html
        _show_dsge_html(io, spec)
    else
        _show_dsge_text(io, spec)
    end
end

# =============================================================================
# Text renderer
# =============================================================================

"""
    _show_dsge_text(io, spec)

Text-mode display for `DSGESpec`: header, calibration, numbered equations,
and steady state (if computed).
"""
function _show_dsge_text(io::IO, spec::DSGESpec{T}) where {T}
    # --- Header ---
    println(io, "DSGE Model Specification")
    println(io, repeat("=", 50))
    println(io, "  Endogenous variables:  ", spec.n_endog,
            "  (", join(string.(spec.endog), ", "), ")")
    println(io, "  Exogenous shocks:      ", spec.n_exog,
            "  (", join(string.(spec.exog), ", "), ")")
    println(io, "  Parameters:            ", spec.n_params)
    println(io, "  Equations:             ", length(spec.equations))
    println(io, "  Forward-looking:       ", spec.n_expect)
    println(io)

    # --- Calibration ---
    println(io, "Calibration")
    println(io, repeat("-", 50))
    for p in spec.params
        val = get(spec.param_values, p, missing)
        println(io, "  ", string(p), " = ", val isa Missing ? "?" : _format_num_display(val))
    end
    println(io)

    # --- Equations ---
    println(io, "Model Equations")
    println(io, repeat("-", 50))
    for (i, eq) in enumerate(spec.equations)
        eq_str = _equation_to_display(eq, spec.endog, spec.exog, spec.params; mode=:text)
        println(io, "  ($i)  ", eq_str)
    end

    # --- Steady state ---
    if !isempty(spec.steady_state)
        println(io)
        println(io, "Steady State")
        println(io, repeat("-", 50))
        println(io, _steady_state_text(spec.endog, spec.steady_state))
    end
end

# =============================================================================
# LaTeX renderer
# =============================================================================

"""
    _show_dsge_latex(io, spec)

LaTeX-mode display for `DSGESpec`: model equations in `align` environment,
calibration in `tabular`, and steady state with `\\bar{}` notation.
"""
function _show_dsge_latex(io::IO, spec::DSGESpec{T}) where {T}
    # --- Header comment ---
    println(io, "% DSGE Model Specification")
    println(io, "% Endogenous: ", join(string.(spec.endog), ", "))
    println(io, "% Exogenous: ", join(string.(spec.exog), ", "))
    println(io)

    # --- Equations ---
    println(io, "\\begin{align}")
    for (i, eq) in enumerate(spec.equations)
        eq_str = _equation_to_display(eq, spec.endog, spec.exog, spec.params; mode=:latex)
        label = "\\label{eq:dsge_$i}"
        trailing = i < length(spec.equations) ? " \\\\" : ""
        println(io, "  ", eq_str, " ", label, trailing)
    end
    println(io, "\\end{align}")
    println(io)

    # --- Calibration ---
    println(io, "\\begin{tabular}{ll}")
    println(io, "\\hline")
    println(io, "Parameter & Value \\\\")
    println(io, "\\hline")
    for p in spec.params
        val = get(spec.param_values, p, missing)
        val_str = val isa Missing ? "?" : _format_num_display(val)
        println(io, "\$", _sym_to_latex(p), "\$ & ", val_str, " \\\\")
    end
    println(io, "\\hline")
    println(io, "\\end{tabular}")

    # --- Steady state ---
    if !isempty(spec.steady_state)
        println(io)
        println(io, "\\begin{align}")
        print(io, _steady_state_latex(spec.endog, spec.steady_state))
        println(io)
        println(io, "\\end{align}")
    end
end

# =============================================================================
# HTML + MathJax renderer
# =============================================================================

"""
    _show_dsge_html(io, spec)

HTML-mode display for `DSGESpec`: MathJax-rendered equations, HTML table
for calibration, and steady state values.
"""
function _show_dsge_html(io::IO, spec::DSGESpec{T}) where {T}
    # --- Header ---
    println(io, "<div class=\"dsge-spec\">")
    println(io, "<h3>DSGE Model Specification</h3>")
    println(io, "<p>Endogenous: ", join(string.(spec.endog), ", "),
            " | Exogenous: ", join(string.(spec.exog), ", "),
            " | Parameters: ", spec.n_params,
            " | Forward-looking: ", spec.n_expect, "</p>")

    # --- Equations (MathJax) ---
    println(io, "<h4>Model Equations</h4>")
    println(io, "\$\$\\begin{align}")
    for (i, eq) in enumerate(spec.equations)
        eq_str = _equation_to_display(eq, spec.endog, spec.exog, spec.params; mode=:latex)
        tag = "\\tag{" * string(i) * "}"
        trailing = i < length(spec.equations) ? " \\\\" : ""
        println(io, "  ", eq_str, " ", tag, trailing)
    end
    println(io, "\\end{align}\$\$")

    # --- Calibration (HTML table) ---
    println(io, "<h4>Calibration</h4>")
    println(io, "<table>")
    println(io, "<tr><th>Parameter</th><th>Value</th></tr>")
    for p in spec.params
        val = get(spec.param_values, p, missing)
        val_str = val isa Missing ? "?" : _format_num_display(val)
        latex_p = _sym_to_latex(p)
        println(io, "<tr><td>\\(", latex_p, "\\)</td><td>", val_str, "</td></tr>")
    end
    println(io, "</table>")

    # --- Steady state ---
    if !isempty(spec.steady_state)
        println(io, "<h4>Steady State</h4>")
        println(io, "\$\$\\begin{align}")
        print(io, _steady_state_latex(spec.endog, spec.steady_state))
        println(io)
        println(io, "\\end{align}\$\$")
    end

    println(io, "</div>")
end
