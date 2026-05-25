# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Parser extensions for heterogeneous agent DSGE models.

Detects `heterogeneous:`, `idiosyncratic:`, and `aggregation:` declarations
within a `@dsge begin...end` block and constructs an `HADSGESpec{Float64}`
instead of a `DSGESpec{Float64}`.

# Supported syntax

```julia
@dsge begin
    parameters: alpha = 0.36, beta_hh = 0.99, delta = 0.025, rho_z = 0.95, sigma_z = 0.007
    endogenous: Y, K, r, w, Z
    exogenous: eps_Z

    heterogeneous: a in [0.0, 200.0], n_grid = 100, utility = log, discount = beta_hh, borrowing = 0.0

    idiosyncratic: e ~ Rouwenhorst(0.966, 0.5, 5)

    aggregation: K = sum(a)

    Y[t] = Z[t] * K[t-1]^alpha
    r[t] = alpha * Z[t] * K[t-1]^(alpha-1) - delta
    w[t] = (1 - alpha) * Z[t] * K[t-1]^alpha
    Z[t] = rho_z * Z[t-1] + sigma_z * eps_Z[t]
end
```

# References
- Auclert, A., Bardóczy, B., Rognlie, M., & Straub, L. (2021). Using the
  sequence-space Jacobian to solve and estimate heterogeneous-agent models.
  *Econometrica*, 89(5), 2375–2408.
"""

# =============================================================================
# _detect_ha_declaration — detect heterogeneous/idiosyncratic/aggregation
# =============================================================================

"""
    _detect_ha_declaration(stmt) → :heterogeneous | :idiosyncratic | :aggregation | nothing

Detect HA-specific declaration labels that the standard `_detect_declaration`
does not handle due to non-standard AST nesting.

AST patterns:
- `heterogeneous: a in [lb, ub], ...` → `(= (tuple (call in (call : heterogeneous a) [lb,ub]) ...) ...)`
- `idiosyncratic: e ~ Rouwenhorst(...)` → `(call ~ (call : idiosyncratic e) (call Rouwenhorst ...))`
- `aggregation: K = sum(a)` → `(= (call : aggregation K) ...)` (handled by standard _detect_declaration)
"""
function _detect_ha_declaration(stmt)
    stmt isa Expr || return nothing

    # Case 1: heterogeneous: a in [lb, ub], ...
    # AST: (= (tuple (call in (call : heterogeneous a) [lb,ub]) n_grid) ...)
    if stmt.head == :(=) && stmt.args[1] isa Expr && stmt.args[1].head == :tuple
        tuple_expr = stmt.args[1]
        if length(tuple_expr.args) >= 1 && tuple_expr.args[1] isa Expr &&
           tuple_expr.args[1].head == :call && length(tuple_expr.args[1].args) >= 2 &&
           tuple_expr.args[1].args[1] === :in
            # Check for (call : heterogeneous a) inside the `in` call
            in_lhs = tuple_expr.args[1].args[2]
            if in_lhs isa Expr && in_lhs.head == :call &&
               length(in_lhs.args) >= 3 && in_lhs.args[1] === :(:) &&
               in_lhs.args[2] === :heterogeneous
                return :heterogeneous
            end
        end
    end

    # Case 2: idiosyncratic: e ~ Rouwenhorst(...)
    # AST: (call ~ (call : idiosyncratic e) (call Rouwenhorst ...))
    if stmt.head == :call && length(stmt.args) >= 3 && stmt.args[1] === :(~)
        lhs = stmt.args[2]
        if lhs isa Expr && lhs.head == :call && length(lhs.args) >= 3 &&
           lhs.args[1] === :(:) && lhs.args[2] === :idiosyncratic
            return :idiosyncratic
        end
    end

    # Case 3: aggregation: K = sum(a) — already detected by _detect_declaration
    # but we check explicitly here too for completeness
    label = _detect_declaration(stmt)
    if label === :aggregation
        return :aggregation
    end

    return nothing
end

# =============================================================================
# _has_ha_declarations — quick scan for heterogeneous blocks
# =============================================================================

"""
    _has_ha_declarations(stmts) → Bool

Scan filtered statements for any HA-specific declarations.
"""
function _has_ha_declarations(stmts)
    for stmt in stmts
        d = _detect_ha_declaration(stmt)
        if d === :heterogeneous || d === :idiosyncratic || d === :aggregation
            return true
        end
    end
    return false
end

# =============================================================================
# _parse_heterogeneous! — extract grid/utility/discount/borrowing
# =============================================================================

"""
    _parse_heterogeneous!(stmt, het_info)

Parse the `heterogeneous: a in [lb, ub], n_grid = N, utility = log, discount = beta, borrowing = lb`
declaration and populate `het_info` dict.

Extracts:
- `:asset_name` — Symbol (e.g., :a)
- `:asset_min`, `:asset_max` — Float64 bounds
- `:n_grid` — Int grid size
- `:utility` — Symbol (e.g., :log, :crra)
- `:discount` — Symbol or Float64
- `:borrowing` — Float64 lower bound
"""
function _parse_heterogeneous!(stmt::Expr, het_info::Dict{Symbol,Any})
    # stmt.head == :(=)
    # stmt.args[1] = (tuple (call in (call : heterogeneous a) [lb,ub]) n_grid)
    # stmt.args[2] = (= (tuple 100 utility) (= (tuple log discount) (= (tuple beta borrowing) 0.0)))

    tuple_expr = stmt.args[1]
    in_call = tuple_expr.args[1]  # (call in (call : heterogeneous a) [lb,ub])

    # Extract asset name
    colon_call = in_call.args[2]  # (call : heterogeneous a)
    het_info[:asset_name] = colon_call.args[3]::Symbol

    # Extract bounds from [lb, ub]
    bounds_expr = in_call.args[3]  # (vect lb ub)
    if bounds_expr isa Expr && bounds_expr.head == :vect && length(bounds_expr.args) == 2
        het_info[:asset_min] = Float64(bounds_expr.args[1])
        het_info[:asset_max] = Float64(bounds_expr.args[2])
    else
        error("@dsge heterogeneous: expected [lb, ub] bounds, got: $bounds_expr")
    end

    # Remaining key=value pairs: n_grid, utility, discount, borrowing
    # These follow the same nested (= (tuple val next_key) ...) pattern as parameters
    remaining_keys = Symbol[]
    remaining_vals = Any[]

    # The second element of the tuple is n_grid (the first key after the comma)
    if length(tuple_expr.args) >= 2
        first_key = tuple_expr.args[2]::Symbol
        # The value chain is in stmt.args[2]
        _collect_het_kv!(first_key, stmt.args[2], remaining_keys, remaining_vals)
    end

    # Process collected key-value pairs
    for (k, v) in zip(remaining_keys, remaining_vals)
        if k === :n_grid
            het_info[:n_grid] = Int(v)
        elseif k === :utility
            het_info[:utility] = v isa Symbol ? v : Symbol(v)
        elseif k === :discount
            het_info[:discount] = v  # could be Symbol or Float64
        elseif k === :borrowing
            het_info[:borrowing] = Float64(v)
        else
            error("@dsge heterogeneous: unknown key '$k'")
        end
    end

    # Defaults
    if !haskey(het_info, :n_grid)
        het_info[:n_grid] = 200
    end
    if !haskey(het_info, :utility)
        het_info[:utility] = :log
    end
    if !haskey(het_info, :borrowing)
        het_info[:borrowing] = 0.0
    end

    return nothing
end

"""
    _collect_het_kv!(name, rhs, keys, vals)

Recursively collect (key, value) pairs from the nested AST, reusing the same
parsing pattern as `_collect_param_chain!`.
"""
function _collect_het_kv!(name::Symbol, rhs, keys::Vector{Symbol}, vals::Vector{Any})
    # Unwrap block wrapper
    if rhs isa Expr && rhs.head == :block
        inner = filter(a -> !(a isa LineNumberNode), rhs.args)
        if length(inner) == 1
            rhs = inner[1]
        end
    end

    if rhs isa Expr && rhs.head == :(=)
        tuple_part = rhs.args[1]
        rest = rhs.args[2]
        if tuple_part isa Expr && tuple_part.head == :tuple && length(tuple_part.args) == 2
            value = tuple_part.args[1]
            next_name = tuple_part.args[2]::Symbol
            push!(keys, name)
            push!(vals, value)
            _collect_het_kv!(next_name, rest, keys, vals)
        else
            # Terminal assignment: name = value
            push!(keys, name)
            push!(vals, rhs)
        end
    else
        push!(keys, name)
        push!(vals, rhs)
    end
end

# =============================================================================
# _parse_idiosyncratic! — extract income process specification
# =============================================================================

"""
    _parse_idiosyncratic!(stmt, idio_info)

Parse `idiosyncratic: e ~ Rouwenhorst(rho, sigma, n)` or
`idiosyncratic: e ~ Tauchen(rho, sigma, n)`.

Extracts:
- `:shock_name` — Symbol (e.g., :e)
- `:method` — Symbol (:Rouwenhorst or :Tauchen)
- `:rho`, `:sigma`, `:n_states` — process parameters
"""
function _parse_idiosyncratic!(stmt::Expr, idio_info::Dict{Symbol,Any})
    # AST: (call ~ (call : idiosyncratic e) (call Rouwenhorst 0.966 0.5 5))
    lhs = stmt.args[2]  # (call : idiosyncratic e)
    rhs = stmt.args[3]  # (call Rouwenhorst 0.966 0.5 5)

    idio_info[:shock_name] = lhs.args[3]::Symbol

    if rhs isa Expr && rhs.head == :call && length(rhs.args) >= 4
        method_name = rhs.args[1]
        idio_info[:method] = method_name isa Symbol ? method_name : Symbol(method_name)
        idio_info[:rho] = Float64(rhs.args[2])
        idio_info[:sigma] = Float64(rhs.args[3])
        idio_info[:n_states] = Int(rhs.args[4])
    else
        error("@dsge idiosyncratic: expected Method(rho, sigma, n_states), got: $rhs")
    end

    return nothing
end

# =============================================================================
# _parse_aggregation! — extract aggregation mapping
# =============================================================================

"""
    _parse_aggregation!(stmt, agg_info)

Parse `aggregation: K = sum(a)`.

Extracts:
- `:target_var` — Symbol (e.g., :K)
- `:source_var` — Symbol (e.g., :a)
- `:agg_fn` — Symbol (e.g., :sum)
"""
function _parse_aggregation!(stmt::Expr, agg_info::Dict{Symbol,Any})
    # AST: (= (call : aggregation K) (block _ (call sum a)))
    lhs = stmt.args[1]  # (call : aggregation K)
    rhs = stmt.args[2]  # (block _ (call sum a))

    agg_info[:target_var] = lhs.args[3]::Symbol

    # Unwrap block wrapper
    if rhs isa Expr && rhs.head == :block
        inner = filter(a -> !(a isa LineNumberNode), rhs.args)
        if length(inner) == 1
            rhs = inner[1]
        end
    end

    if rhs isa Expr && rhs.head == :call && length(rhs.args) >= 2
        agg_info[:agg_fn] = rhs.args[1]::Symbol
        agg_info[:source_var] = rhs.args[2]::Symbol
    else
        error("@dsge aggregation: expected fn(var), got: $rhs")
    end

    return nothing
end

# =============================================================================
# _parse_ha_dsge — main HA-DSGE parser
# =============================================================================

"""
    _parse_ha_dsge(block) → Expr

Parse a `@dsge begin...end` block containing heterogeneous agent declarations
and return a quoted expression that constructs an `HADSGESpec{Float64}`.

Separates the block into:
1. Standard declarations (parameters, endogenous, exogenous) and equations
2. HA-specific declarations (heterogeneous, idiosyncratic, aggregation)

Constructs the aggregate `DSGESpec` via `_minimal_agg_spec` (since the
individual household problem is specified declaratively, not as full residual
functions), then builds the `HADSGESpec` with the appropriate grid, income
process, individual problem, and aggregation.
"""
function _parse_ha_dsge(block::Expr)
    stmts = filter(a -> !(a isa LineNumberNode), block.args)

    # Collect standard declarations
    params = Symbol[]
    param_defaults = Dict{Symbol,Any}()
    endog = Symbol[]
    exog = Symbol[]
    raw_equations = Expr[]
    ss_body = nothing

    # Collect HA declarations
    het_info = Dict{Symbol,Any}()
    idio_info = Dict{Symbol,Any}()
    agg_info = Dict{Symbol,Any}()

    for stmt in stmts
        # Check HA-specific declarations first
        ha_label = _detect_ha_declaration(stmt)
        if ha_label === :heterogeneous
            _parse_heterogeneous!(stmt, het_info)
            continue
        elseif ha_label === :idiosyncratic
            _parse_idiosyncratic!(stmt, idio_info)
            continue
        elseif ha_label === :aggregation
            _parse_aggregation!(stmt, agg_info)
            continue
        end

        # Standard declarations
        label = _detect_declaration(stmt)
        if label === :parameters
            _extract_parameters!(stmt, params, param_defaults)
        elseif label === :endogenous
            append!(endog, _extract_names(stmt))
        elseif label === :exogenous
            append!(exog, _extract_names(stmt))
        elseif label === :steady_state
            ss_body = stmt.args[3]
        elseif label === nothing
            if stmt isa Expr && stmt.head == :(=) && stmt.args[1] === :steady_state
                ss_body = stmt.args[2]
            elseif stmt isa Expr && stmt.head == :(=)
                push!(raw_equations, stmt)
            else
                error("@dsge (HA): unrecognized statement: $stmt")
            end
        end
    end

    # Validate
    isempty(params) && error("@dsge (HA): no parameters declared")
    isempty(endog) && error("@dsge (HA): no endogenous variables declared")
    isempty(exog) && error("@dsge (HA): no exogenous variables declared")
    isempty(het_info) && error("@dsge (HA): no heterogeneous block found")
    isempty(idio_info) && error("@dsge (HA): no idiosyncratic shocks declared")
    isempty(agg_info) && error("@dsge (HA): no aggregation rule declared")

    # Extract parsed values
    asset_name = het_info[:asset_name]
    asset_min = het_info[:asset_min]
    asset_max = het_info[:asset_max]
    n_grid = het_info[:n_grid]
    utility_type = het_info[:utility]
    discount_sym = het_info[:discount]
    borrowing_lb = het_info[:borrowing]

    shock_name = idio_info[:shock_name]
    income_method = idio_info[:method]
    income_rho = idio_info[:rho]
    income_sigma = idio_info[:sigma]
    income_n = idio_info[:n_states]

    agg_target = agg_info[:target_var]
    agg_source = agg_info[:source_var]

    # Resolve discount factor: either a parameter name or a literal
    discount_val = if discount_sym isa Symbol && haskey(param_defaults, discount_sym)
        param_defaults[discount_sym]
    elseif discount_sym isa Number
        discount_sym
    else
        error("@dsge (HA): discount factor '$discount_sym' not found in parameters")
    end

    # Build het_params from all parameter defaults
    param_pairs = [Expr(:call, :(=>), QuoteNode(p), param_defaults[p]) for p in params]

    # Determine CRRA sigma from utility type
    sigma_c_val = if utility_type === :log
        1.0
    else
        # Default to 1.0 (log utility) if not recognized
        1.0
    end

    # Build the constructor expression
    result = quote
        # Income process
        local _income_ = if $(QuoteNode(income_method)) === :Rouwenhorst
            rouwenhorst($income_rho, $income_sigma, $income_n)
        elseif $(QuoteNode(income_method)) === :Tauchen
            tauchen($income_rho, $income_sigma, $income_n)
        else
            error("Unknown income discretization method: " * string($(QuoteNode(income_method))))
        end

        # Grid
        local _grid_ = HAGrid(; assets=($asset_min, $asset_max, $n_grid),
                                income_states=$income_n)

        # Utility functions (CRRA with sigma_c)
        local _u_, _up_, _upi_ = MacroEconometricModels._crra_utility($sigma_c_val)

        # Budget function: c + a' = (1+r)*a + w*e
        local _budget_fn_ = MacroEconometricModels._ks_budget

        # Individual problem
        local _individual_ = IndividualProblem{Float64}(
            _u_, _up_, _upi_, Float64($discount_val), _budget_fn_,
            [Float64($borrowing_lb)], nothing, 1
        )

        # Aggregate spec (lightweight placeholder)
        local _agg_param_vals_ = Dict{Symbol,Float64}($(param_pairs...))
        local _alpha_ = get(_agg_param_vals_, :alpha, 0.36)
        local _delta_ = get(_agg_param_vals_, :delta, 0.025)
        local _rho_z_ = get(_agg_param_vals_, :rho_z, 0.95)
        local _sigma_z_ = get(_agg_param_vals_, :sigma_z, 0.007)
        local _agg_spec_ = MacroEconometricModels._minimal_agg_spec(;
            alpha=_alpha_, delta=_delta_, rho_z=_rho_z_, sigma_z=_sigma_z_)

        # Aggregation mapping
        local _aggregation_ = Pair{Symbol,Function}[
            $(QuoteNode(agg_target)) => MacroEconometricModels._agg_var1
        ]

        # Het params
        local _het_params_ = Dict{Symbol,Float64}($(param_pairs...))
        if !haskey(_het_params_, :Z)
            _het_params_[:Z] = 1.0
        end
        if !haskey(_het_params_, :L)
            _het_params_[:L] = 1.0
        end

        HADSGESpec{Float64}(_agg_spec_, _individual_, _income_, _grid_,
                             _aggregation_, _het_params_)
    end

    return esc(result)
end

# =============================================================================
# solve(::HADSGESpec) — dispatch for HA-DSGE models
# =============================================================================

"""
    solve(spec::HADSGESpec{T}; method=:ssj, ss=nothing, kwargs...) → HADSGESolution{T} | KrusellSmithSolution{T}

Solve a heterogeneous agent DSGE model.

Computes the stationary equilibrium (if not supplied), then applies the
chosen solution method.

# Methods
- `:ssj` — Sequence-Space Jacobian (Auclert et al. 2021). Default.
- `:reiter` — Reiter (2009) linearization with distribution reduction.
- `:krusell_smith` — Krusell & Smith (1998) simulation with PLM regression.

# Arguments
- `spec::HADSGESpec{T}` — model specification
- `method::Symbol` — solution method (default `:ssj`)
- `ss::Union{Nothing,HASteadyState{T}}` — precomputed steady state; if `nothing`, computed automatically

# Keyword Arguments (passed to steady state and solver)
- `K_init`, `r_bounds`, `max_iter`, `tol`, `verbose` — steady-state kwargs
- `T_horizon`, `n_reduced` — SSJ/Reiter kwargs
- `T_sim`, `T_burn`, `max_outer`, `rho_z`, `sigma_z` — Krusell-Smith kwargs

# Returns
- `HADSGESolution{T}` for `:ssj` and `:reiter`
- `KrusellSmithSolution{T}` for `:krusell_smith`

# References
- Auclert, A., Bardóczy, B., Rognlie, M., & Straub, L. (2021). Using the
  sequence-space Jacobian to solve and estimate heterogeneous-agent models.
  *Econometrica*, 89(5), 2375–2408.
- Reiter, M. (2009). Solving heterogeneous-agent models by projection and
  perturbation. *Journal of Economic Dynamics and Control*, 33(3), 649–665.
- Krusell, P., & Smith, A. A. (1998). Income and wealth heterogeneity in the
  macroeconomy. *Journal of Political Economy*, 106(5), 867–896.
"""
function solve(spec::HADSGESpec{T}; method::Symbol=:ssj,
               ss::Union{Nothing,HASteadyState{T}}=nothing,
               kwargs...) where {T<:AbstractFloat}
    # Compute steady state if not supplied
    if ss === nothing
        # Extract steady-state relevant kwargs
        ss_keys = (:K_init, :r_bounds, :max_iter, :tol, :verbose, :price_fn)
        ss_kwargs = Dict{Symbol,Any}()
        for k in ss_keys
            if haskey(kwargs, k)
                ss_kwargs[k] = kwargs[k]
            end
        end
        ss = compute_steady_state(spec; ss_kwargs...)
    end

    if method === :ssj
        # Extract SSJ-specific kwargs
        T_horizon = get(kwargs, :T_horizon, 300)
        n_reduced = get(kwargs, :n_reduced, 30)
        return _ssj_solve(spec, ss; T_horizon=T_horizon, n_reduced=n_reduced)

    elseif method === :reiter
        n_reduced = get(kwargs, :n_reduced, 30)
        G1, impact, n_red, explained = _reiter_linearize(
            ss, spec.individual, spec.grid, spec.income; n_reduced=n_reduced
        )

        # Build a minimal DSGESolution and HADSGESolution from Reiter output
        n_sys = size(G1, 1)
        endog_names = [Symbol("x_$i") for i in 1:n_sys]
        exog_names = [:epsilon]
        dummy_spec_inner = DSGESpec{T}(
            endog_names, exog_names, Symbol[], Dict{Symbol,T}(),
            [:(0 + 0) for _ in 1:n_sys],
            [((yt, yl, yle, eps, th) -> zero(T)) for _ in 1:n_sys],
            0, Int[], zeros(T, n_sys), nothing
        )
        C_sol = zeros(T, n_sys)
        eigenvalues = eigvals(G1)
        eu = [1, 1]
        Gamma0 = Matrix{T}(I, n_sys, n_sys)
        linear = LinearDSGE{T}(Gamma0, copy(G1), zeros(T, n_sys), copy(impact),
                                zeros(T, n_sys, 0), dummy_spec_inner)
        dsge_sol = DSGESolution{T}(G1, impact, C_sol, eu, :reiter, eigenvalues,
                                    dummy_spec_inner, linear)
        reduction_basis = Matrix{T}(I, n_red, n_red)
        return HADSGESolution{T}(ss, dsge_sol, :reiter, spec, reduction_basis,
                                  spec.grid.total_individual_states, n_red,
                                  explained, nothing)

    elseif method === :krusell_smith
        # Extract KS-specific kwargs
        T_sim = get(kwargs, :T_sim, 11000)
        T_burn = get(kwargs, :T_burn, 1000)
        max_outer = get(kwargs, :max_outer, 20)
        rho_z = get(kwargs, :rho_z, 0.95)
        sigma_z = get(kwargs, :sigma_z, 0.007)

        return _krusell_smith_solve(ss, spec.individual, spec.grid, spec.income,
                                     _default_cobb_douglas_price_fn,
                                     spec.het_params;
                                     T_sim=T_sim, T_burn=T_burn,
                                     max_outer=max_outer,
                                     rho_z=rho_z, sigma_z=sigma_z)
    else
        error("Unknown HA-DSGE method: :$method. Use :ssj, :reiter, or :krusell_smith.")
    end
end
