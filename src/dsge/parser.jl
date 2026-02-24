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
    @dsge begin ... end

Parse a DSGE model specification block into a [`DSGESpec{Float64}`](@ref).

## Declaration syntax

```julia
spec = @dsge begin
    parameters: ρ = 0.9, σ = 0.01
    endogenous: C, K, Y, A
    exogenous: ε_A

    C[t] + K[t] = (1-δ)*K[t-1] + K[t-1]^α
    A[t] = ρ * A[t-1] + σ * ε_A[t]
end
```

Time references: `var[t]` (current), `var[t-1]` (lagged), `var[t+1]` (lead).
Expectations operator: `E[t](expr)` is stripped under rational expectations.

Returns a `DSGESpec{Float64}` with callable residual functions `f(y_t, y_lag, y_lead, ε, θ) → scalar`.
"""
macro dsge(block)
    block.head == :block || error("@dsge requires a begin...end block")
    _dsge_impl(block)
end

# =============================================================================
# Top-level implementation (called at macro-expansion time)
# =============================================================================

function _dsge_impl(block::Expr)
    params = Symbol[]
    param_defaults = Dict{Symbol,Any}()
    endog = Symbol[]
    exog = Symbol[]
    raw_equations = Expr[]
    ss_body = nothing  # steady_state block body (Expr or nothing)

    stmts = filter(a -> !(a isa LineNumberNode), block.args)

    for stmt in stmts
        label = _detect_declaration(stmt)
        if label === :parameters
            _extract_parameters!(stmt, params, param_defaults)
        elseif label === :endogenous
            append!(endog, _extract_names(stmt))
        elseif label === :exogenous
            append!(exog, _extract_names(stmt))
        elseif label === :steady_state
            # Single-line: steady_state: [expr]
            # AST: (call : steady_state <body_expr>)
            ss_body = stmt.args[3]  # the expression after the colon
        elseif label === nothing
            # Check for multi-line: steady_state = begin...end
            # AST: (= :steady_state (block ...))
            if stmt isa Expr && stmt.head == :(=) && stmt.args[1] === :steady_state
                ss_body = stmt.args[2]
            elseif stmt isa Expr && stmt.head == :(=)
                push!(raw_equations, stmt)
            else
                error("@dsge: unrecognized statement: $stmt")
            end
        end
    end

    isempty(params) && error("@dsge: no parameters declared")
    isempty(endog) && error("@dsge: no endogenous variables declared")
    isempty(exog) && error("@dsge: no exogenous variables declared")
    length(raw_equations) != length(endog) &&
        error("@dsge: expected $(length(endog)) equations (one per endogenous variable), got $(length(raw_equations))")

    # ── Augmentation for deep lags, deep leads, and news shocks (#54) ──
    original_endog = copy(endog)
    original_raw_equations = deepcopy(raw_equations)

    # Scan offsets on residual-form equations (so we see all var[t±k] refs)
    scan_eqs = Expr[_strip_expectation_operator(_equation_to_residual(eq)) for eq in raw_equations]
    offsets = _scan_offsets(scan_eqs, endog, exog)

    # Determine if augmentation is needed
    has_deep_offset = any(v -> v.max_lag > 1 || v.max_lead > 1, values(offsets))
    has_exog_lag = false
    for v in exog
        if haskey(offsets, v) && offsets[v].max_lag > 0
            has_exog_lag = true
            break
        end
    end
    needs_augmentation = has_deep_offset || has_exog_lag

    if needs_augmentation
        aux_endog, aux_equations, sub_map = _generate_augmentation(offsets, endog, exog)

        # Apply substitutions to raw user equations
        for i in eachindex(raw_equations)
            raw_equations[i] = _apply_augmentation_subs(raw_equations[i], sub_map)
        end

        # Extend endogenous variables and append auxiliary equations
        append!(endog, aux_endog)
        append!(raw_equations, aux_equations)

        # Revalidate counts after augmentation
        length(raw_equations) != length(endog) &&
            error("@dsge: augmentation error — $(length(endog)) endogenous but $(length(raw_equations)) equations")
    end

    aug_flag = needs_augmentation
    max_lag_val = maximum((get(offsets, v, (max_lag=0, max_lead=0)).max_lag for v in vcat(original_endog, exog)); init=1)
    max_lag_val = max(max_lag_val, 1)
    max_lead_val = maximum((get(offsets, v, (max_lag=0, max_lead=0)).max_lead for v in original_endog); init=1)
    max_lead_val = max(max_lead_val, 1)

    # ── Classify forward-looking equations (on augmented set) ──
    forward_indices = Int[]
    for (i, eq) in enumerate(raw_equations)
        if _has_forward_looking(eq, endog, exog)
            push!(forward_indices, i)
        end
    end
    n_expect = length(forward_indices)

    # Build residual functions and cleaned equation expressions
    residual_fn_exprs = Expr[]
    equation_exprs = Expr[]
    for eq in raw_equations
        residual_ex = _equation_to_residual(eq)
        # Strip E[t](...) operators
        residual_ex = _strip_expectation_operator(residual_ex)
        # Substitute var[t±k] → vector indexing
        subst_ex = _substitute_vars(residual_ex, endog, exog, params)
        push!(equation_exprs, residual_ex)

        # Build a closure expression:
        # (y_t, y_lag, y_lead, ε, θ) -> <substituted expression>
        fn_expr = Expr(:->, Expr(:tuple, :_y_t_, :_y_lag_, :_y_lead_, :_ε_, :_θ_), subst_ex)
        push!(residual_fn_exprs, fn_expr)
    end

    # ── Build original equations (pre-augmentation residual form) for display ──
    orig_eq_exprs = Expr[]
    for eq in original_raw_equations
        resid = _equation_to_residual(eq)
        resid = _strip_expectation_operator(resid)
        push!(orig_eq_exprs, resid)
    end

    # Build the constructor call as a quoted expression
    param_vals_expr = Expr(:call, :Dict,
        [Expr(:call, :(=>), QuoteNode(p), param_defaults[p]) for p in params]...)

    endog_expr = Expr(:vect, [QuoteNode(s) for s in endog]...)
    exog_expr = Expr(:vect, [QuoteNode(s) for s in exog]...)
    params_expr = Expr(:vect, [QuoteNode(s) for s in params]...)

    fwd_expr = Expr(:vect, forward_indices...)

    # Wrap equation expressions as QuoteNodes so they store as Expr
    eq_vec_expr = Expr(:ref, :Expr, [QuoteNode(eq) for eq in equation_exprs]...)

    # Build the vector of residual functions
    fn_vec_expr = Expr(:ref, :Function, residual_fn_exprs...)

    # Build original_endog and original_equations expressions
    original_endog_expr = Expr(:vect, [QuoteNode(s) for s in original_endog]...)
    orig_eq_vec_expr = Expr(:ref, :Expr, [QuoteNode(eq) for eq in orig_eq_exprs]...)

    # Build ss_fn expression if steady_state block was provided
    ss_fn_expr = if ss_body !== nothing
        # Build: (_ss_θ_) -> begin <param unpacking>; <ss_body> end
        param_unpack = [:($(p) = _ss_θ_[$(QuoteNode(p))]) for p in params]
        if ss_body isa Expr && ss_body.head == :block
            # Multi-line: insert param unpacking at the start of the block
            inner = filter(a -> !(a isa LineNumberNode), ss_body.args)
            body = Expr(:block, param_unpack..., inner...)
        else
            # Single-line: wrap in block with param unpacking
            body = Expr(:block, param_unpack..., ss_body)
        end
        Expr(:->, :_ss_θ_, body)
    else
        :nothing
    end

    result = quote
        DSGESpec{Float64}(
            $endog_expr, $exog_expr, $params_expr,
            $param_vals_expr,
            $eq_vec_expr,
            $fn_vec_expr,
            $n_expect, $fwd_expr, Float64[], $ss_fn_expr;
            original_endog=$original_endog_expr,
            original_equations=$orig_eq_vec_expr,
            augmented=$aug_flag,
            max_lag=$max_lag_val,
            max_lead=$max_lead_val
        )
    end

    return esc(result)
end

# =============================================================================
# Declaration detection and extraction
# =============================================================================

"""
    _detect_declaration(stmt) → :parameters | :endogenous | :exogenous | nothing

Detect whether a statement is a declaration line (parameters:, endogenous:, exogenous:).

Julia parses `label: name` as `(call : label name)` and
`label: name = val` as `(= (call : label name) ...)`.
Multi-name `label: a, b, c` becomes `(tuple (call : label a) b c)`.
"""
function _detect_declaration(stmt)
    if stmt isa Expr
        # Case 1: `label: name = value` or `label: name = v1, name2 = v2, ...`
        # Parsed as (= (call : label name) ...)
        if stmt.head == :(=) && stmt.args[1] isa Expr &&
           stmt.args[1].head == :call && length(stmt.args[1].args) >= 3 &&
           stmt.args[1].args[1] === :(:)
            return stmt.args[1].args[2]
        end

        # Case 2: `label: name` (single, no value)
        # Parsed as (call : label name)
        if stmt.head == :call && length(stmt.args) >= 3 && stmt.args[1] === :(:)
            return stmt.args[2]
        end

        # Case 3: `label: name1, name2, ...` (multiple, no values)
        # Parsed as (tuple (call : label name1) name2 ...)
        if stmt.head == :tuple && length(stmt.args) >= 1 &&
           stmt.args[1] isa Expr && stmt.args[1].head == :call &&
           length(stmt.args[1].args) >= 3 && stmt.args[1].args[1] === :(:)
            return stmt.args[1].args[2]
        end
    end
    return nothing
end

"""
    _extract_parameters!(stmt, params, param_defaults)

Extract parameter names and default values from a `parameters: ...` declaration.

Handles three parsing patterns:
- Single: `parameters: ρ = 0.9` → `(= (call : parameters ρ) (block _ 0.9))`
- Multi: `parameters: ρ = 0.9, σ = 0.01` → nested `=` chain with `tuple` nodes
"""
function _extract_parameters!(stmt::Expr, params::Vector{Symbol}, defaults::Dict{Symbol,Any})
    # stmt.head must be :(=)
    # LHS: (call : parameters first_name)
    first_name = stmt.args[1].args[3]::Symbol
    rhs = stmt.args[2]

    # Unwrap block wrapper if present: (block LineNumberNode value)
    if rhs isa Expr && rhs.head == :block
        inner_stmts = filter(a -> !(a isa LineNumberNode), rhs.args)
        length(inner_stmts) == 1 || error("@dsge: malformed parameter declaration")
        rhs = inner_stmts[1]
    end

    # Now rhs is either:
    #   - a literal value (single param case): 0.9
    #   - a nested (= (tuple val next_name) ...) chain (multi param case)
    _collect_param_chain!(first_name, rhs, params, defaults)
    return nothing
end

"""
    _collect_param_chain!(name, rhs, params, defaults)

Recursively collect (name, value) pairs from the nested AST.

For `parameters: ρ = 0.9, σ = 0.01`:
  - name = :ρ, rhs = (= (tuple 0.9 σ) 0.01)
  - Extract ρ = 0.9 from (tuple 0.9 σ), then recurse with name = :σ, rhs = 0.01

For `parameters: ρ = 0.9` (single):
  - name = :ρ, rhs = 0.9
"""
function _collect_param_chain!(name::Symbol, rhs, params::Vector{Symbol}, defaults::Dict{Symbol,Any})
    if rhs isa Expr && rhs.head == :(=)
        # Multi-param: rhs = (= (tuple prev_value next_name) rest)
        tuple_part = rhs.args[1]
        rest = rhs.args[2]
        if tuple_part isa Expr && tuple_part.head == :tuple && length(tuple_part.args) == 2
            value = tuple_part.args[1]
            next_name = tuple_part.args[2]::Symbol
            push!(params, name)
            defaults[name] = value
            _collect_param_chain!(next_name, rest, params, defaults)
        else
            error("@dsge: cannot parse parameter declaration for $name")
        end
    else
        # Terminal case: rhs is just a value
        push!(params, name)
        defaults[name] = rhs
    end
end

"""
    _extract_names(stmt) → Vector{Symbol}

Extract variable names from an `endogenous:` or `exogenous:` declaration.

Handles:
- Single: `endogenous: y` → `(call : endogenous y)` → [:y]
- Multi: `endogenous: y, k, a` → `(tuple (call : endogenous y) k a)` → [:y, :k, :a]
"""
function _extract_names(stmt::Expr)
    names = Symbol[]
    if stmt.head == :call && length(stmt.args) >= 3 && stmt.args[1] === :(:)
        # Single variable: (call : label name)
        push!(names, stmt.args[3]::Symbol)
    elseif stmt.head == :tuple
        # Multiple: (tuple (call : label name1) name2 ...)
        first_call = stmt.args[1]
        push!(names, first_call.args[3]::Symbol)
        for i in 2:length(stmt.args)
            push!(names, stmt.args[i]::Symbol)
        end
    else
        error("@dsge: cannot extract names from: $stmt")
    end
    return names
end

# =============================================================================
# Time-index parsing
# =============================================================================

"""
    _parse_time_index(ex) → Int

Parse time index from a `ref` subscript expression.
- `t` → 0
- `(call + t 1)` → 1
- `(call - t 1)` → -1
"""
function _parse_time_index(ex)
    if ex === :t
        return 0
    elseif ex isa Expr && ex.head == :call && length(ex.args) == 3 && ex.args[2] === :t
        op = ex.args[1]
        offset = ex.args[3]
        if op === :(+)
            return Int(offset)
        elseif op === :(-)
            return -Int(offset)
        end
    end
    error("@dsge: unrecognized time index expression: $ex")
end

"""
    _is_time_ref(ex, varset) → Bool

Check if `ex` is a `var[t±k]` reference where `var ∈ varset`.
"""
function _is_time_ref(ex, varset::Vector{Symbol})
    ex isa Expr && ex.head == :ref && length(ex.args) == 2 &&
        ex.args[1] isa Symbol && ex.args[1] ∈ varset
end

"""
    _is_expectation_operator(ex) → Bool

Check if `ex` is `E[t](...)` — a function call with `E[t]` as callee.
Parsed as `(call (ref E t) args...)`.
"""
function _is_expectation_operator(ex)
    ex isa Expr && ex.head == :call && length(ex.args) >= 2 &&
        ex.args[1] isa Expr && ex.args[1].head == :ref &&
        ex.args[1].args[1] === :E
end

# =============================================================================
# Forward-looking detection
# =============================================================================

"""
    _has_forward_looking(eq, endog, exog) → Bool

Check if equation `eq` contains any `[t+1]` terms (endogenous forward-looking)
or `E[t](...)` operator.
"""
function _has_forward_looking(eq::Expr, endog::Vector{Symbol}, exog::Vector{Symbol})
    found = Ref(false)
    _walk_expr(eq) do ex
        # Check for E[t](...) operator
        if _is_expectation_operator(ex)
            found[] = true
            return
        end
        # Check for var[t+1] where var is endogenous
        if _is_time_ref(ex, endog)
            idx = _parse_time_index(ex.args[2])
            if idx > 0
                found[] = true
            end
        end
    end
    return found[]
end

"""
    _walk_expr(f, ex)

Recursively walk expression tree, calling `f(node)` on every `Expr` node.
"""
function _walk_expr(f::Function, ex)
    if ex isa Expr
        f(ex)
        for a in ex.args
            _walk_expr(f, a)
        end
    end
end

# =============================================================================
# Equation transformation
# =============================================================================

"""
    _equation_to_residual(eq) → Expr

Transform `LHS = RHS` to `LHS - (RHS)`, creating a residual expression.
"""
function _equation_to_residual(eq::Expr)
    eq.head == :(=) || error("@dsge: equation must be LHS = RHS, got: $eq")
    lhs = eq.args[1]
    rhs = eq.args[2]
    # Unwrap block wrapper if present
    if rhs isa Expr && rhs.head == :block
        inner = filter(a -> !(a isa LineNumberNode), rhs.args)
        length(inner) == 1 || error("@dsge: malformed equation RHS")
        rhs = inner[1]
    end
    return Expr(:call, :(-), lhs, rhs)
end

"""
    _strip_expectation_operator(ex) → Expr

Recursively replace `E[t](inner_expr)` with just `inner_expr`.
Under rational expectations linearization, the expectation operator is stripped.
"""
function _strip_expectation_operator(ex)
    if !(ex isa Expr)
        return ex
    end
    # If this is E[t](arg), return the stripped arg
    if _is_expectation_operator(ex)
        # E[t](arg) — the arg is ex.args[2]
        return _strip_expectation_operator(ex.args[2])
    end
    # Recurse into children
    new_args = [_strip_expectation_operator(a) for a in ex.args]
    return Expr(ex.head, new_args...)
end

# =============================================================================
# Variable substitution
# =============================================================================

"""
    _substitute_vars(ex, endog, exog, params) → Expr

Recursively replace:
- `var[t]` → `_y_t_[i]` where `i` = index of `var` in `endog`
- `var[t-1]` → `_y_lag_[i]`
- `var[t+1]` → `_y_lead_[i]`
- `shock[t]` → `_ε_[j]` where `j` = index of `shock` in `exog`
- bare parameter symbols → `_θ_[QuoteNode(name)]`
"""
function _substitute_vars(ex, endog::Vector{Symbol}, exog::Vector{Symbol}, params::Vector{Symbol})
    if ex isa Expr
        # Time-indexed endogenous variable: var[t±k]
        if ex.head == :ref && length(ex.args) == 2 && ex.args[1] isa Symbol
            varname = ex.args[1]::Symbol
            if varname ∈ endog
                idx = findfirst(==(varname), endog)
                offset = _parse_time_index(ex.args[2])
                if offset == 0
                    return Expr(:ref, :_y_t_, idx)
                elseif offset < 0
                    return Expr(:ref, :_y_lag_, idx)
                else  # offset > 0
                    return Expr(:ref, :_y_lead_, idx)
                end
            elseif varname ∈ exog
                jdx = findfirst(==(varname), exog)
                # Shocks should only be [t]
                offset = _parse_time_index(ex.args[2])
                offset == 0 || error("@dsge: exogenous shock $varname can only be indexed at [t], got offset $offset")
                return Expr(:ref, :_ε_, jdx)
            end
            # Not a known variable — could be some other indexing; leave as is
        end

        # Recurse into children
        new_args = Any[]
        for a in ex.args
            push!(new_args, _substitute_vars(a, endog, exog, params))
        end
        return Expr(ex.head, new_args...)

    elseif ex isa Symbol
        # Bare parameter symbol → θ[:name]
        if ex ∈ params
            return Expr(:ref, :_θ_, QuoteNode(ex))
        end
        # Other symbols (operators like +, -, *, ^, numeric constants, etc.) pass through
        return ex
    else
        # Literal values (numbers, etc.)
        return ex
    end
end

# =============================================================================
# Augmentation functions for deep lags, deep leads, and news shocks (#54)
# =============================================================================

"""
    _scan_offsets(equations, endog, exog)

Walk equation expression trees and record the maximum lag and lead offset
for each endogenous/exogenous variable referenced as `var[t±k]`.
Returns `Dict{Symbol, @NamedTuple{max_lag::Int, max_lead::Int}}`.
"""
function _scan_offsets(equations::Vector{Expr}, endog::Vector{Symbol}, exog::Vector{Symbol})
    offsets = Dict{Symbol, @NamedTuple{max_lag::Int, max_lead::Int}}()
    all_vars = vcat(endog, exog)
    for eq in equations
        _walk_expr(eq) do ex
            if ex isa Expr && ex.head == :ref && length(ex.args) == 2 && ex.args[1] isa Symbol
                varname = ex.args[1]::Symbol
                if varname ∈ all_vars
                    offset = _parse_time_index(ex.args[2])
                    prev = get(offsets, varname, (max_lag=0, max_lead=0))
                    lag = max(prev.max_lag, offset < 0 ? -offset : 0)
                    lead = max(prev.max_lead, offset > 0 ? offset : 0)
                    offsets[varname] = (max_lag=lag, max_lead=lead)
                end
            end
        end
    end
    offsets
end

"""
    _generate_augmentation(offsets, endog, exog)

Generate auxiliary endogenous variables and identity equations for deep lags,
deep leads, and exogenous news shocks.

Returns `(aux_endog, aux_equations, sub_map)` where:
- `aux_endog::Vector{Symbol}` — new auxiliary variable names
- `aux_equations::Vector{Expr}` — identity equations in `LHS = RHS` form
- `sub_map::Dict{Tuple{Symbol,Int}, Tuple{Symbol,Int}}` — substitution map
"""
function _generate_augmentation(offsets::Dict{Symbol, @NamedTuple{max_lag::Int, max_lead::Int}},
                                 endog::Vector{Symbol}, exog::Vector{Symbol})
    aux_endog = Symbol[]
    aux_equations = Expr[]
    sub_map = Dict{Tuple{Symbol,Int}, Tuple{Symbol,Int}}()

    for (var, info) in offsets
        if var ∈ endog
            # --- Deep endogenous lags: var[t-k] where k > 1 ---
            if info.max_lag > 1
                for j in 1:(info.max_lag - 1)
                    aux_name = Symbol("__lag_", var, "_", j)
                    push!(aux_endog, aux_name)
                    if j == 1
                        # __lag_var_1[t] = var[t-1]
                        eq = Expr(:(=),
                            Expr(:ref, aux_name, :t),
                            Expr(:ref, var, Expr(:call, :(-), :t, 1)))
                        push!(aux_equations, eq)
                    else
                        # __lag_var_j[t] = __lag_var_{j-1}[t-1]
                        prev_name = Symbol("__lag_", var, "_", j - 1)
                        eq = Expr(:(=),
                            Expr(:ref, aux_name, :t),
                            Expr(:ref, prev_name, Expr(:call, :(-), :t, 1)))
                        push!(aux_equations, eq)
                    end
                    # Sub: (var, -(j+1)) → (__lag_var_j, -1)
                    sub_map[(var, -(j + 1))] = (aux_name, -1)
                end
            end

            # --- Deep endogenous leads: var[t+k] where k > 1 ---
            if info.max_lead > 1
                for j in 1:(info.max_lead - 1)
                    aux_name = Symbol("__fwd_", var, "_", j)
                    push!(aux_endog, aux_name)
                    if j == 1
                        # __fwd_var_1[t] = var[t+1]
                        eq = Expr(:(=),
                            Expr(:ref, aux_name, :t),
                            Expr(:ref, var, Expr(:call, :(+), :t, 1)))
                        push!(aux_equations, eq)
                    else
                        # __fwd_var_j[t] = __fwd_var_{j-1}[t+1]
                        prev_name = Symbol("__fwd_", var, "_", j - 1)
                        eq = Expr(:(=),
                            Expr(:ref, aux_name, :t),
                            Expr(:ref, prev_name, Expr(:call, :(+), :t, 1)))
                        push!(aux_equations, eq)
                    end
                    # Sub: (var, j+1) → (__fwd_var_j, 1)
                    sub_map[(var, j + 1)] = (aux_name, 1)
                end
            end

        elseif var ∈ exog
            # --- Exogenous news shocks: ε[t-k] where k > 0 ---
            if info.max_lag > 0
                for j in 1:info.max_lag
                    aux_name = Symbol("__news_", var, "_", j)
                    push!(aux_endog, aux_name)
                    if j == 1
                        # __news_ε_1[t] = ε[t]
                        eq = Expr(:(=),
                            Expr(:ref, aux_name, :t),
                            Expr(:ref, var, :t))
                        push!(aux_equations, eq)
                    else
                        # __news_ε_j[t] = __news_ε_{j-1}[t-1]
                        prev_name = Symbol("__news_", var, "_", j - 1)
                        eq = Expr(:(=),
                            Expr(:ref, aux_name, :t),
                            Expr(:ref, prev_name, Expr(:call, :(-), :t, 1)))
                        push!(aux_equations, eq)
                    end
                    # Sub: (ε, -j) → (__news_ε_j, -1)
                    sub_map[(var, -j)] = (aux_name, -1)
                end
            end
        end
    end

    return aux_endog, aux_equations, sub_map
end

"""
    _apply_augmentation_subs(ex, sub_map)

Walk an expression tree and replace `var[t±k]` references according to `sub_map`.
When `(varname, offset)` is found in `sub_map`, replaces with `(new_var, new_offset)`.
"""
function _apply_augmentation_subs(ex, sub_map::Dict{Tuple{Symbol,Int}, Tuple{Symbol,Int}})
    if ex isa Expr
        if ex.head == :ref && length(ex.args) == 2 && ex.args[1] isa Symbol
            varname = ex.args[1]::Symbol
            offset = try
                _parse_time_index(ex.args[2])
            catch
                nothing
            end
            if offset !== nothing
                key = (varname, offset)
                if haskey(sub_map, key)
                    new_var, new_offset = sub_map[key]
                    if new_offset == 0
                        new_time = :t
                    elseif new_offset > 0
                        new_time = Expr(:call, :(+), :t, new_offset)
                    else
                        new_time = Expr(:call, :(-), :t, -new_offset)
                    end
                    return Expr(:ref, new_var, new_time)
                end
            end
        end
        # Recurse into children
        new_args = Any[_apply_augmentation_subs(a, sub_map) for a in ex.args]
        return Expr(ex.head, new_args...)
    else
        return ex
    end
end
