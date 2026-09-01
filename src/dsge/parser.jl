# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
    @dsge begin ... end

Parse a DSGE model specification block into a [`ModelSpec{Float64,NoAgents}`](@ref).

## Declaration syntax

```julia
spec = @dsge begin
    parameters: ρ = 0.9, σ = 0.01
    endogenous: C, K, Y, A
    exogenous: ε_A

    C[t] + K[t] = (1-δ)*K[t-1] + K[t-1]^α
    A[t] = A[t-1]^ρ * exp(σ * ε_A[t])
end
```

Time references: `var[t]` (current), `var[t-1]` (lagged), `var[t+1]` (lead).
A lead `x[t+k]` *is* the rational expectation of `x`. The `E[t](...)` operator
is rejected.

Equations may be named (`euler: β * C[t+1] / C[t] = 1 / R[t]`). Unlabeled
`var[t] = ...` uses `var` as both the name and the defining variable.

Occasionally binding constraints attach a `:binding` regime to the equation
whose `defines` is the constrained variable:

```julia
    constraint: i[t] >= 0
    taylor:     i[t] = φπ * π[t]
```

If no equation (or several) define `i`, write `constraint: taylor = i[t] >= 0`.

Optional Bellman declarations for [`vfi_solver`](@ref):

```julia
    utility: log(C)
    beta: β
    controls: C
```

`transition` and `control_bounds` are inferred when omitted (`next_state=:residual`
when control FOCs can be dropped). Compound rewards such as
`utility: C^(1-σ)/(1-σ)` are stored as expressions and compiled by
[`vfi_solver`](@ref). `utility` / `beta` / `controls` are read from the spec
when the corresponding keyword is omitted.

`clock:` (`discrete` / `continuous`) and `horizon:` (`infinite` / `finite` /
`ages` / `perpetual_youth`) set [`ModelIR`](@ref) flags. Extra keys on
`horizon:` (`J=`, `retire=`, `survival=`, `earnings=`, …) and the
`discrete:` / `absorbing:` option lists are stored as IR declarations; they
do not compile [`ContinuousHouseholdSystem`](@ref), [`LifeCycleSystem`](@ref),
or [`DCEGMSystem`](@ref). Use `to_spec` on the family constructor.

Returns a `ModelSpec{Float64,NoAgents}` with callable residual functions
`f(y_t, y_lag, y_lead, ε, θ) → scalar`.
"""
macro dsge(block)
    block.head == :block || error("@dsge requires a begin...end block")
    _dsge_impl(block)
end

# Reserved declaration labels. Unknown `name: LHS = RHS` is a named equation.
const _RESERVED_DSGE_LABELS = (
    :parameters, :endogenous, :exogenous, :linear, :steady_state,
    :utility, :beta, :controls, :varnames, :clock, :horizon,
    :constraint, :discrete, :absorbing,
)

"""One `@dsge constraint:` line, resolved onto a defining equation after the scan."""
struct _ConstraintDecl
    variable::Symbol
    bound::Float64
    direction::Symbol
    expr::Expr
    on::Union{Nothing,Symbol}
end

# =============================================================================
# Top-level implementation (called at macro-expansion time)
# =============================================================================

function _dsge_impl(block::Expr)
    params = Symbol[]
    param_defaults = Dict{Symbol,Any}()
    endog = Symbol[]
    exog = Symbol[]
    raw_eq_stmts = Tuple{Union{Nothing,Symbol},Expr}[]
    ss_body = nothing  # steady_state block body (Expr or nothing)
    is_linear = false  # linear: true declaration
    bellman_util_ex = nothing
    bellman_beta_ex = nothing
    bellman_cons_ex = nothing
    bellman_ctrl_ex = Symbol[]
    user_varnames = nothing
    clock_val = :discrete
    horizon_val = :infinite
    extra_decls = IRDecl[]
    constraint_decls = _ConstraintDecl[]

    stmts = filter(a -> !(a isa LineNumberNode), block.args)

    # Check for heterogeneous agent declarations — delegate to HA parser
    if _has_ha_declarations(stmts)
        return _parse_ha_dsge(block)
    end

    for stmt in stmts
        label = _detect_declaration(stmt)
        if label !== nothing && label in _RESERVED_DSGE_LABELS &&
           stmt isa Expr && stmt.head == :(=) &&
           stmt.args[1] isa Expr && stmt.args[1].head == :call &&
           length(stmt.args[1].args) >= 3
            third = stmt.args[1].args[3]
            if third isa Expr && (third.head === :ref || third.head === :call)
                error("@dsge: '$label' is a reserved declaration label and cannot name an equation; rename the equation.")
            end
        end
        if label === :parameters
            _extract_parameters!(stmt, params, param_defaults)
        elseif label === :endogenous
            append!(endog, _extract_names(stmt))
        elseif label === :exogenous
            append!(exog, _extract_names(stmt))
        elseif label === :linear
            is_linear = _extract_linear_value(stmt)
        elseif label === :steady_state
            ss_body = stmt.args[3]
        elseif label === :utility
            u_rhs = stmt.args[3]
            if u_rhs isa Expr && u_rhs.head == :call && length(u_rhs.args) == 2 &&
               u_rhs.args[1] isa Symbol && u_rhs.args[2] isa Symbol
                bellman_util_ex = u_rhs.args[1]          # log(C) → :log
                bellman_cons_ex = u_rhs.args[2]
            elseif u_rhs isa Expr
                bellman_util_ex = QuoteNode(u_rhs)
            else
                bellman_util_ex = u_rhs
            end
        elseif label === :beta
            bellman_beta_ex = stmt.args[3]
        elseif label === :controls
            bellman_ctrl_ex = _extract_names(stmt)
        elseif label === :varnames
            user_varnames = _extract_varnames(stmt)
        elseif label === :clock
            clock_val = _extract_clock_horizon_flag(stmt, :clock)
            push!(extra_decls, IRDecl(:clock, nothing, stmt))
        elseif label === :horizon
            horizon_val = _extract_clock_horizon_flag(stmt, :horizon)
            push!(extra_decls, IRDecl(:horizon, nothing, stmt))
        elseif label === :constraint
            append!(constraint_decls, _extract_constraint_decls(stmt))
            push!(extra_decls, IRDecl(:constraint, nothing, stmt))
        elseif label === :discrete || label === :absorbing
            push!(extra_decls, IRDecl(label, nothing, stmt))
        elseif label !== nothing
            # Unknown `name: LHS = RHS` is a named equation
            stmt isa Expr && stmt.head == :(=) ||
                error("@dsge: unrecognized declaration :$label")
            push!(raw_eq_stmts, (label, stmt))
        elseif stmt isa Expr && stmt.head == :(=) && stmt.args[1] === :steady_state
            ss_body = stmt.args[2]
        elseif stmt isa Expr && stmt.head == :(=)
            push!(raw_eq_stmts, (nothing, stmt))
        else
            error("@dsge: unrecognized statement: $stmt")
        end
    end

    isempty(params) && error("@dsge: no parameters declared")
    isempty(endog) && error("@dsge: no endogenous variables declared")
    isempty(exog) && error("@dsge: no exogenous variables declared")

    if bellman_util_ex isa QuoteNode && bellman_util_ex.value isa Expr &&
       bellman_cons_ex === nothing
        candidates = Symbol[]
        _walk_expr(bellman_util_ex.value) do node
            for a in node.args
                if a isa Symbol && (a in endog || a in bellman_ctrl_ex)
                    push!(candidates, a)
                end
            end
        end
        unique!(candidates)
        if length(candidates) == 1
            bellman_cons_ex = candidates[1]
        else
            error("@dsge: utility: expression references $(candidates) — " *
                  "need exactly one consumption symbol (or write utility: log(C))")
        end
    end
    if bellman_cons_ex !== nothing &&
       !(bellman_cons_ex in endog) && !(bellman_cons_ex in bellman_ctrl_ex)
        error("@dsge: utility: consumption :$bellman_cons_ex is not an " *
              "endogenous or control variable (endog = $endog)")
    end

    raw_equations = Expr[]
    eq_names = Symbol[]
    eq_defines = Union{Nothing,Symbol}[]
    used_names = Set{Symbol}()
    anon = 0
    for (given_name, stmt) in raw_eq_stmts
        if given_name === nothing
            eq = stmt
            lhs = stmt.args[1]
            def = _lhs_defines(lhs, endog)
            name = def === nothing ? (anon += 1; Symbol("eq_", anon)) : def
            if name in used_names
                # Keep `defines` so constraint: can report the ambiguity (MSR-23).
                anon += 1
                name = Symbol("eq_", anon)
            end
        else
            lhs = stmt.args[1].args[3]
            rhs = _unwrap_eq_rhs(stmt.args[2])
            eq = Expr(:(=), lhs, rhs)
            def = _lhs_defines(lhs, endog)
            name = given_name
            name in used_names && error("@dsge: duplicate equation name :$name")
        end
        push!(used_names, name)
        push!(eq_names, name)
        push!(eq_defines, def)
        push!(raw_equations, eq)
    end

    length(raw_equations) != length(endog) &&
        error("@dsge: expected $(length(endog)) equations (one per endogenous variable), got $(length(raw_equations))")

    for eq in raw_equations
        _reject_expectation_operator(eq)
    end

    # ── Augmentation for deep lags, deep leads, and news shocks (#54) ──
    original_endog = copy(endog)
    original_raw_equations = deepcopy(raw_equations)
    original_eq_names = copy(eq_names)
    original_eq_defines = copy(eq_defines)

    scan_eqs = Expr[_equation_to_residual(eq) for eq in raw_equations]
    offsets = _scan_offsets(scan_eqs, endog, exog)

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

        for i in eachindex(raw_equations)
            raw_equations[i] = _apply_augmentation_subs(raw_equations[i], sub_map)
        end

        append!(endog, aux_endog)
        append!(raw_equations, aux_equations)
        for aux_eq in aux_equations
            aux_name = aux_eq.args[1]
            aux_sym = aux_name isa Expr && aux_name.head == :ref ? aux_name.args[1] : aux_name
            aux_sym isa Symbol || error("@dsge: augmentation produced unnamed identity")
            push!(eq_names, aux_sym)
            push!(eq_defines, aux_sym)
        end

        length(raw_equations) != length(endog) &&
            error("@dsge: augmentation error — $(length(endog)) endogenous but $(length(raw_equations)) equations")
    end

    aug_flag = needs_augmentation
    max_lag_val = maximum((get(offsets, v, (max_lag=0, max_lead=0)).max_lag for v in vcat(original_endog, exog)); init=1)
    max_lag_val = max(max_lag_val, 1)
    max_lead_val = maximum((get(offsets, v, (max_lag=0, max_lead=0)).max_lead for v in original_endog); init=1)
    max_lead_val = max(max_lead_val, 1)

    forward_indices = Int[]
    for (i, eq) in enumerate(raw_equations)
        if _has_forward_looking(eq, endog, exog)
            push!(forward_indices, i)
        end
    end
    n_expect = length(forward_indices)

    # `_substitute_vars` / `_equation_to_residual` live in ir.jl (shared with
    # load-time recompile). The macro still emits closures inline — no eval here.
    residual_fn_exprs = Expr[]
    residual_exprs = Expr[]
    timings = TimingInfo[]
    for (i, eq) in enumerate(raw_equations)
        residual_ex = _equation_to_residual(eq)
        subst_ex = _substitute_vars(residual_ex, endog, exog, params)
        push!(residual_exprs, residual_ex)
        fn_expr = Expr(:->, Expr(:tuple, :_y_t_, :_y_lag_, :_y_lead_, :_ε_, :_θ_), subst_ex)
        push!(residual_fn_exprs, fn_expr)
        push!(timings, _equation_timing(residual_ex, endog))
    end

    orig_residual_exprs = Expr[_equation_to_residual(eq) for eq in original_raw_equations]
    orig_timings = TimingInfo[_equation_timing(ex, original_endog) for ex in orig_residual_exprs]

    param_vals_expr = Expr(:call, :Dict,
        [Expr(:call, :(=>), QuoteNode(p), param_defaults[p]) for p in params]...)

    endog_expr = Expr(:vect, [QuoteNode(s) for s in endog]...)
    exog_expr = Expr(:vect, [QuoteNode(s) for s in exog]...)
    params_expr = Expr(:vect, [QuoteNode(s) for s in params]...)
    fwd_expr = Expr(:vect, forward_indices...)
    original_endog_expr = Expr(:vect, [QuoteNode(s) for s in original_endog]...)

    binding_regimes = _constraint_binding_exprs(constraint_decls, eq_names, eq_defines, endog)

    function _named_eq_expr(name, def, residual_ex, fn_expr, timing, binding_ex=nothing)
        def_ex = def === nothing ? :nothing : QuoteNode(def)
        timing_ex = :($(TimingInfo)($(timing.max_lag), $(timing.max_lead), $(timing.has_lead)))
        if binding_ex === nothing
            :(NamedEquation($(QuoteNode(name)), $def_ex, $(QuoteNode(residual_ex)), $fn_expr;
                            timing=$timing_ex))
        else
            :(NamedEquation($(QuoteNode(name)), $def_ex, $(QuoteNode(residual_ex)), $fn_expr;
                            timing=$timing_ex,
                            regimes=Dict{Symbol,NamedEquation}(:binding => $binding_ex)))
        end
    end

    eq_vec_expr = Expr(:vect, (_named_eq_expr(eq_names[i], eq_defines[i], residual_exprs[i],
                                             residual_fn_exprs[i], timings[i], binding_regimes[i])
                               for i in eachindex(raw_equations))...)
    orig_eq_vec_expr = Expr(:vect, (_named_eq_expr(original_eq_names[i], original_eq_defines[i],
                                                   orig_residual_exprs[i], :(identity),
                                                   orig_timings[i],
                                                   i <= length(binding_regimes) ? binding_regimes[i] : nothing)
                                    for i in eachindex(original_raw_equations))...)
    fn_vec_expr = Expr(:ref, :Function, residual_fn_exprs...)

    ss_fn_expr = if ss_body !== nothing
        param_unpack = [:($(p) = _ss_θ_[$(QuoteNode(p))]) for p in params]
        if ss_body isa Expr && ss_body.head == :block
            inner = filter(a -> !(a isa LineNumberNode), ss_body.args)
            body = Expr(:block, param_unpack..., inner...)
        else
            body = Expr(:block, param_unpack..., ss_body)
        end
        Expr(:->, :_ss_θ_, body)
    elseif is_linear
        n_endog_val = length(endog)
        :((_ss_θ_) -> zeros($n_endog_val))
    else
        :nothing
    end

    decls = IRDecl[
        IRDecl(:parameters, nothing, copy(params)),
        IRDecl(:endogenous, nothing, copy(original_endog)),
        IRDecl(:exogenous, nothing, copy(exog)),
    ]
    append!(decls, extra_decls)
    ir_eqs = IREquation[IREquation(original_eq_names[i], original_eq_defines[i],
                                   original_raw_equations[i].args[1],
                                   _unwrap_eq_rhs(original_raw_equations[i].args[2]))
                        for i in eachindex(original_raw_equations)]
    ir = ModelIR(clock_val, horizon_val, decls, ir_eqs)

    vnames = if user_varnames === nothing
        nothing
    else
        extra = [string(s) for s in endog[length(original_endog)+1:end]]
        vcat(user_varnames, extra)
    end
    varnames_expr = vnames === nothing ? :nothing : Expr(:vect, vnames...)

    result = quote
        let _eqs = $eq_vec_expr
            ModelSpec{Float64}(
                $endog_expr, $exog_expr, $params_expr,
                $param_vals_expr,
                _eqs,
                $fn_vec_expr,
                $n_expect, $fwd_expr, Float64[], $ss_fn_expr;
                original_endog=$original_endog_expr,
                original_equations=$orig_eq_vec_expr,
                augmented=$aug_flag,
                max_lag=$max_lag_val,
                max_lead=$max_lead_val,
                linear=$is_linear,
                bellman_utility=$(bellman_util_ex === nothing ? :nothing : bellman_util_ex),
                bellman_beta=$(bellman_beta_ex === nothing ? :nothing :
                    (bellman_beta_ex isa Symbol ? QuoteNode(bellman_beta_ex) : bellman_beta_ex)),
                bellman_consumption=$(bellman_cons_ex === nothing ? :nothing : QuoteNode(bellman_cons_ex)),
                bellman_controls=$(Expr(:vect, (QuoteNode(s) for s in bellman_ctrl_ex)...)),
                agents=NamedTuple(),
                ir=$(ir),
                varnames=$varnames_expr,
            )
        end
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

        # Case 1b: `horizon: ages, J = 60, retire = 45`
        # Parsed as (= (tuple (call : horizon ages) J) ...)
        if stmt.head == :(=) && stmt.args[1] isa Expr && stmt.args[1].head == :tuple
            first = stmt.args[1].args[1]
            if first isa Expr && first.head == :call && length(first.args) >= 3 &&
               first.args[1] === :(:) && first.args[2] in (:clock, :horizon)
                return first.args[2]
            end
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

        # Case 4: `constraint: var[t] >= bound` — comparison binds tighter than `:`,
        # so this is `(>= (constraint: var[t]) bound)`, not `(constraint: (>= var[t] bound))`.
        if _is_constraint_comparison(stmt)
            return :constraint
        end
        if stmt.head == :tuple && length(stmt.args) >= 1 && _is_constraint_comparison(stmt.args[1])
            return :constraint
        end
    end
    return nothing
end

"""
    _is_constraint_comparison(stmt) → Bool

`constraint: var[t] >= bound` parses as `(>= (constraint: var[t]) bound)`.
"""
function _is_constraint_comparison(stmt)
    stmt isa Expr && stmt.head == :call && length(stmt.args) == 3 || return false
    (stmt.args[1] === :(>=) || stmt.args[1] === :(<=)) || return false
    lhs = stmt.args[2]
    return lhs isa Expr && lhs.head == :call && length(lhs.args) >= 3 &&
           lhs.args[1] === :(:) && lhs.args[2] === :constraint
end

"""
    _extract_constraint_decls(stmt) → Vector{_ConstraintDecl}

Parse `constraint: i[t] >= 0`, `constraint: i[t] >= 0, c[t] >= 0`, or
`constraint: taylor = i[t] >= 0` (explicit equation name).
"""
function _extract_constraint_decls(stmt::Expr)
    out = _ConstraintDecl[]
    if _is_constraint_comparison(stmt)
        push!(out, _constraint_from_labeled_cmp(stmt, nothing))
    elseif stmt.head == :tuple
        first = true
        for arg in stmt.args
            if first && _is_constraint_comparison(arg)
                push!(out, _constraint_from_labeled_cmp(arg, nothing))
            elseif arg isa Expr && arg.head == :call && length(arg.args) == 3 &&
                   (arg.args[1] === :(>=) || arg.args[1] === :(<=))
                push!(out, _constraint_from_cmp(arg, nothing))
            else
                error("@dsge: cannot parse constraint in list: $arg")
            end
            first = false
        end
    elseif stmt.head == :(=)
        on = stmt.args[1]
        on isa Expr && on.head == :call && length(on.args) >= 3 &&
            on.args[1] === :(:) && on.args[2] === :constraint ||
            error("@dsge: cannot parse constraint declaration: $stmt")
        eqname = on.args[3]
        eqname isa Symbol || error("@dsge: `constraint: <eqname> = var[t] >= bound` " *
                                   "needs an equation name, got $eqname. " *
                                   "For the default mapping write `constraint: var[t] >= bound`.")
        rhs = _unwrap_eq_rhs(stmt.args[2])
        rhs isa Expr && rhs.head == :call && length(rhs.args) == 3 &&
            (rhs.args[1] === :(>=) || rhs.args[1] === :(<=)) ||
            error("@dsge: `constraint: $eqname = ...` must be `var[t] >= bound` or `var[t] <= bound`")
        push!(out, _constraint_from_cmp(rhs, eqname))
    else
        error("@dsge: cannot parse constraint declaration: $stmt")
    end
    return out
end

function _constraint_from_labeled_cmp(stmt::Expr, on::Union{Nothing,Symbol})
    op = stmt.args[1]
    lhs_inner = stmt.args[2].args[3]
    rhs = stmt.args[3]
    cmp = Expr(:call, op, lhs_inner, rhs)
    return _constraint_from_cmp(cmp, on)
end

function _constraint_from_cmp(cmp::Expr, on::Union{Nothing,Symbol})
    op = cmp.args[1]
    var = _constraint_lhs_var(cmp.args[2])
    bound = Float64(_eval_bound(cmp.args[3], Float64))
    dir = op === :(>=) ? :geq : :leq
    return _ConstraintDecl(var, bound, dir, cmp, on)
end

function _constraint_lhs_var(lhs)
    if lhs isa Expr && lhs.head == :ref && length(lhs.args) == 2 &&
       lhs.args[1] isa Symbol && lhs.args[2] === :t
        return lhs.args[1]::Symbol
    end
    error("@dsge: constraint LHS must be var[t], got: $lhs")
end

"""
    _resolve_constraint_equation(var, on, eq_names, eq_defines) → Int

Index of the equation that a `constraint:` replaces. `on` is the explicit
equation name from `constraint: taylor = i[t] >= 0`.
"""
function _resolve_constraint_equation(var::Symbol, on::Union{Nothing,Symbol},
                                      eq_names::Vector{Symbol},
                                      eq_defines::Vector{<:Union{Nothing,Symbol}})
    if on !== nothing
        idx = findfirst(==(on), eq_names)
        idx === nothing && error("@dsge: constraint on :$on: no equation named :$on. " *
                                 "Write `constraint: $on = $var[t] >= bound` with a declared name.")
        return idx
    end
    idxs = findall(d -> d === var, eq_defines)
    if length(idxs) == 1
        return idxs[1]
    elseif isempty(idxs)
        error("@dsge: no equation defines :$var. Name the defining equation " *
              "(e.g. `taylor: $var[t] = ...`) or write `constraint: <eqname> = $var[t] >= bound`.")
    else
        names = eq_names[idxs]
        error("@dsge: $(length(idxs)) equations define :$var ($(join(names, ", "))). " *
              "Disambiguate with `constraint: <eqname> = $var[t] >= bound`.")
    end
end

"""Quoted `:binding` NamedEquation per compiled equation, or `nothing`."""
function _constraint_binding_exprs(constraint_decls::Vector{_ConstraintDecl},
                                   eq_names::Vector{Symbol},
                                   eq_defines::Vector{<:Union{Nothing,Symbol}},
                                   endog::Vector{Symbol})
    binding = Vector{Union{Nothing,Expr}}(nothing, length(eq_names))
    used = Dict{Int,Symbol}()
    for c in constraint_decls
        idx = _resolve_constraint_equation(c.variable, c.on, eq_names, eq_defines)
        if haskey(used, idx)
            error("@dsge: constraints on :$(used[idx]) and :$(c.variable) both replace " *
                  "equation :$(eq_names[idx])")
        end
        used[idx] = c.variable
        var_idx = findfirst(==(c.variable), endog)
        var_idx === nothing && error("@dsge: constraint variable :$(c.variable) is not endogenous")
        bind_fn = :((y_t, y_lag, y_lead, epsilon, theta) -> y_t[$var_idx] - $(c.bound))
        bind_expr = Expr(:(=), Expr(:ref, c.variable, :t), c.bound)
        binding[idx] = :(NamedEquation($(QuoteNode(eq_names[idx])), $(QuoteNode(c.variable)),
                                       $(QuoteNode(bind_expr)), $bind_fn;
                                       timing=$(TimingInfo)()))
    end
    return binding
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
    _extract_linear_value(stmt) → Bool

Extract boolean value from a `linear: true` or `linear: false` declaration.
"""
function _extract_linear_value(stmt::Expr)
    # stmt is (call : linear true) for `linear: true`
    if stmt.head == :call && length(stmt.args) >= 3 && stmt.args[1] === :(:)
        val = stmt.args[3]
        val === true && return true
        val === false && return false
        error("@dsge: linear declaration must be `linear: true` or `linear: false`, got: $val")
    end
    error("@dsge: cannot parse linear declaration: $stmt")
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
# `_parse_time_index` lives in ir.jl (shared with `_substitute_vars` / recompile).

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

Check if equation `eq` contains any endogenous `[t+k]` lead, `k > 0`.
"""
function _has_forward_looking(eq::Expr, endog::Vector{Symbol}, exog::Vector{Symbol})
    found = Ref(false)
    _walk_expr(eq) do ex
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
    _lhs_defines(lhs, endog) → Union{Nothing,Symbol}

If `lhs` is `var[t]` or a bare endogenous `var`, that variable is the defining
target of the equation.
"""
function _lhs_defines(lhs, endog::Vector{Symbol})
    if lhs isa Expr && lhs.head == :ref && lhs.args[1] isa Symbol
        v = lhs.args[1]::Symbol
        v in endog || return nothing
        try
            return _parse_time_index(lhs.args[2]) == 0 ? v : nothing
        catch
            return nothing
        end
    elseif lhs isa Symbol && lhs in endog
        return lhs
    end
    return nothing
end

_unwrap_eq_rhs(rhs) = rhs
function _unwrap_eq_rhs(rhs::Expr)
    if rhs.head == :block
        inner = filter(a -> !(a isa LineNumberNode), rhs.args)
        length(inner) == 1 || return rhs
        return inner[1]
    end
    return rhs
end

function _extract_varnames(stmt::Expr)
    raw = Any[]
    if stmt.head == :call && length(stmt.args) >= 3 && stmt.args[1] === :(:)
        push!(raw, stmt.args[3])
    elseif stmt.head == :tuple
        first_call = stmt.args[1]
        push!(raw, first_call.args[3])
        for i in 2:length(stmt.args)
            push!(raw, stmt.args[i])
        end
    else
        error("@dsge: cannot extract varnames from: $stmt")
    end
    names = String[]
    for v in raw
        if v isa String
            push!(names, v)
        elseif v isa Symbol
            push!(names, string(v))
        else
            error("@dsge: varnames entries must be symbols or strings, got $v")
        end
    end
    return names
end

const _VALID_CLOCK = (:discrete, :continuous)
const _VALID_HORIZON = (:infinite, :finite, :ages, :perpetual_youth)

function _validate_clock_horizon(label::Symbol, val::Symbol)
    allowed = label === :clock ? _VALID_CLOCK : _VALID_HORIZON
    val in allowed || error("@dsge: $label must be $(join(allowed, ", ")), got :$val")
    return val
end

"""
    _extract_clock_horizon_flag(stmt, label) → Symbol

`clock: continuous` / `horizon: ages` set the [`ModelIR`](@ref) flag.
Compound `horizon: ages, J = 60, retire = 45` (and `survival=` /
`earnings=`) still returns only the flag; the raw statement is stored as
an `IRDecl` for a later compile.
"""
function _extract_clock_horizon_flag(stmt::Expr, label::Symbol)
    if stmt.head == :call && length(stmt.args) >= 3 && stmt.args[1] === :(:)
        val = stmt.args[3]
        val isa Symbol || error("@dsge: $label declaration must be a symbol, got: $val")
        return _validate_clock_horizon(label, val)
    end
    # `horizon: ages, J = 60, retire = 45` → (= (tuple (call : horizon ages) J) ...)
    if stmt.head == :(=) && stmt.args[1] isa Expr && stmt.args[1].head == :tuple
        first = stmt.args[1].args[1]
        if first isa Expr && first.head == :call && length(first.args) >= 3 &&
           first.args[1] === :(:) && first.args[2] === label
            val = first.args[3]
            val isa Symbol || error("@dsge: $label declaration must be a symbol, got: $val")
            return _validate_clock_horizon(label, val)
        end
    end
    error("@dsge: cannot parse $label declaration: $stmt")
end

function _equation_timing(eq::Expr, endog::Vector{Symbol})
    max_lag = 0
    max_lead = 0
    _walk_expr(eq) do ex
        if _is_time_ref(ex, endog)
            idx = _parse_time_index(ex.args[2])
            if idx < 0
                max_lag = max(max_lag, -idx)
            elseif idx > 0
                max_lead = max(max_lead, idx)
            end
        end
    end
    TimingInfo(max_lag, max_lead, max_lead > 0)
end

"""
    _reject_expectation_operator(ex)

Error if `ex` contains the removed `E[t](...)` call form. A bare endogenous
`E[t]` (employment, endowment) is a time reference, not this operator.
"""
function _reject_expectation_operator(ex)
    _walk_expr(ex) do node
        if _is_expectation_operator(node)
            error("@dsge: E[t](...) was removed; write the lead directly (x[t+1] is E_t x_{t+1})")
        end
    end
    return nothing
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

# `_equation_to_residual` and `_substitute_vars` live in ir.jl so `@dsge` and
# ModelSpec load-time recompile share one substitution. The macro still emits
# closures inline (no `eval` at expansion).

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
