# DSGE News Shocks & Arbitrary Lag/Lead Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the DSGE module to support arbitrary lag/lead depths (`y[t-k]`, `y[t+k]`) and news/anticipated shocks (`ε[t-k]`) via automatic state-space augmentation at parse time.

**Architecture:** The `@dsge` macro scans equations for deep offsets, generates auxiliary variables + identity equations to reduce everything to standard lag-1/lead-1 form, then compiles residual functions on the augmented system. Solvers (gensys, BK) work unchanged on the larger system. Output functions filter to show only original user variables.

**Tech Stack:** Julia macros (metaprogramming), expression tree walking, existing DSGE pipeline.

---

### Task 1: Add Augmentation Metadata Fields to DSGESpec

**Files:**
- Modify: `src/dsge/types.jl:50-79` (DSGESpec struct + constructor)
- Modify: `src/dsge/steady_state.jl:96-102` (_update_steady_state)
- Modify: `src/dsge/occbin.jl:171-173` (_derive_alternative_regime DSGESpec constructor)
- Test: `test/dsge/test_dsge.jl`

**Context:** `DSGESpec{T}` is an immutable struct with an inner constructor. It's reconstructed in `_update_steady_state` and `_derive_alternative_regime`. All constructor calls must pass the new fields.

**Step 1: Add fields to DSGESpec struct**

In `src/dsge/types.jl`, add 7 new fields to the struct (after `ss_fn`) and update the inner constructor to accept them with backward-compatible defaults:

```julia
struct DSGESpec{T<:AbstractFloat}
    endog::Vector{Symbol}
    exog::Vector{Symbol}
    params::Vector{Symbol}
    param_values::Dict{Symbol,T}
    equations::Vector{Expr}
    residual_fns::Vector{Function}
    n_endog::Int
    n_exog::Int
    n_params::Int
    n_expect::Int
    forward_indices::Vector{Int}
    steady_state::Vector{T}
    varnames::Vector{String}
    ss_fn::Union{Nothing, Function}
    # ── augmentation metadata (issue #54) ──
    original_endog::Vector{Symbol}
    original_equations::Vector{Expr}
    n_original_endog::Int
    n_original_eq::Int
    augmented::Bool
    max_lag::Int
    max_lead::Int

    function DSGESpec{T}(endog, exog, params, param_values, equations, residual_fns,
                         n_expect, forward_indices, steady_state,
                         ss_fn::Union{Nothing, Function}=nothing;
                         original_endog::Vector{Symbol}=endog,
                         original_equations::Vector{Expr}=equations,
                         augmented::Bool=false,
                         max_lag::Int=1,
                         max_lead::Int=1) where {T<:AbstractFloat}
        n_endog = length(endog)
        n_exog = length(exog)
        n_params = length(params)
        @assert length(equations) == n_endog "Need as many equations as endogenous variables"
        @assert length(residual_fns) == n_endog
        @assert length(forward_indices) == n_expect
        varnames = [string(s) for s in endog]
        n_original_endog = length(original_endog)
        n_original_eq = length(original_equations)
        new{T}(endog, exog, params, param_values, equations, residual_fns,
               n_endog, n_exog, n_params, n_expect, forward_indices, steady_state, varnames, ss_fn,
               original_endog, original_equations, n_original_endog, n_original_eq,
               augmented, max_lag, max_lead)
    end
end
```

**Step 2: Add `_original_var_indices` utility**

Add this function immediately after the DSGESpec struct in `src/dsge/types.jl`:

```julia
"""
    _original_var_indices(spec::DSGESpec) → Vector{Int}

Return indices of the user's original endogenous variables in the (possibly augmented) state vector.
For non-augmented models, returns `1:n_endog`.
"""
function _original_var_indices(spec::DSGESpec)
    if !spec.augmented
        return collect(1:spec.n_endog)
    end
    return [findfirst(==(v), spec.endog) for v in spec.original_endog]
end
```

**Step 3: Update `_update_steady_state`**

In `src/dsge/steady_state.jl`, update `_update_steady_state` to pass through the new keyword arguments:

```julia
function _update_steady_state(spec::DSGESpec{T}, y_ss::Vector{T}) where {T}
    DSGESpec{T}(
        spec.endog, spec.exog, spec.params, spec.param_values,
        spec.equations, spec.residual_fns,
        spec.n_expect, spec.forward_indices, y_ss, spec.ss_fn;
        original_endog=spec.original_endog,
        original_equations=spec.original_equations,
        augmented=spec.augmented,
        max_lag=spec.max_lag,
        max_lead=spec.max_lead
    )
end
```

**Step 4: Update `_derive_alternative_regime` in occbin.jl**

In `src/dsge/occbin.jl`, update the DSGESpec constructor call at line 171-173:

```julia
    DSGESpec{T}(spec.endog, spec.exog, spec.params, spec.param_values,
                new_equations, new_residual_fns,
                n_expect_new, new_forward_indices, spec.steady_state, spec.ss_fn;
                original_endog=spec.original_endog,
                original_equations=spec.original_equations,
                augmented=spec.augmented,
                max_lag=spec.max_lag,
                max_lead=spec.max_lead)
```

**Step 5: Run tests to verify backward compatibility**

Run: `julia --project=. -e 'using Test, MacroEconometricModels; import MacroEconometricModels: set_display_backend, PlotOutput; include("test/dsge/test_dsge.jl")'`
Expected: All 451 existing tests pass (no behavioral change for lag-1/lead-1 models).

**Step 6: Commit**

```bash
git add src/dsge/types.jl src/dsge/steady_state.jl src/dsge/occbin.jl
git commit -m "refactor(dsge): add augmentation metadata fields to DSGESpec (#54)"
```

---

### Task 2: Implement Augmentation Scanner and Generator

**Files:**
- Modify: `src/dsge/parser.jl` (add 3 new functions before `_dsge_impl`)
- Test: `test/dsge/test_dsge.jl`

**Context:** These functions operate on raw `Expr` trees from the `@dsge` macro. They scan equations for deep offsets, then generate auxiliary variables, identity equations, and a substitution map.

**Step 1: Write tests for scanner and generator**

Add to `test/dsge/test_dsge.jl` before the closing `end`:

```julia
@testset "Augmentation: _scan_offsets (#54)" begin
    endog = [:y, :K]
    exog = [:ε, :ε_A]

    # Simple lag-1: no deep offsets
    eq1 = :(y[t] - (K[t-1] + ε[t]))
    offsets = MacroEconometricModels._scan_offsets([eq1], endog, exog)
    @test offsets[:K] == (max_lag=1, max_lead=0)
    @test offsets[:y] == (max_lag=0, max_lead=0)
    @test !haskey(offsets, :ε)  # exog tracked separately
    @test offsets[:ε] == (max_lag=0, max_lead=0) || !haskey(offsets, :ε)

    # Deep endogenous lag
    eq2 = :(y[t] - (K[t-1] + K[t-3] + ε[t]))
    offsets2 = MacroEconometricModels._scan_offsets([eq2], endog, exog)
    @test offsets2[:K] == (max_lag=3, max_lead=0)

    # News shock
    eq3 = :(y[t] - (ε_A[t] + ε_A[t-8]))
    offsets3 = MacroEconometricModels._scan_offsets([eq3], endog, exog)
    @test offsets3[:ε_A] == (max_lag=8, max_lead=0)

    # Deep lead
    eq4 = :(y[t] - K[t+2])
    offsets4 = MacroEconometricModels._scan_offsets([eq4], endog, exog)
    @test offsets4[:K] == (max_lag=0, max_lead=2)
end

@testset "Augmentation: _generate_augmentation (#54)" begin
    endog = [:y, :K]
    exog = [:ε_A]

    # News shock: ε_A[t-3]
    offsets = Dict(:y => (max_lag=0, max_lead=0),
                   :K => (max_lag=1, max_lead=0),
                   :ε_A => (max_lag=3, max_lead=0))
    aug = MacroEconometricModels._generate_augmentation(offsets, endog, exog)
    @test length(aug.aux_endog) == 3  # 3 news auxiliaries for ε_A with lag 3
    @test length(aug.aux_equations) == 3
    @test haskey(aug.sub_map, (:ε_A, -3))
    @test haskey(aug.sub_map, (:ε_A, -2))
    @test haskey(aug.sub_map, (:ε_A, -1))

    # Endogenous deep lag: K[t-3]
    offsets2 = Dict(:y => (max_lag=0, max_lead=0),
                    :K => (max_lag=3, max_lead=0),
                    :ε_A => (max_lag=0, max_lead=0))
    aug2 = MacroEconometricModels._generate_augmentation(offsets2, endog, exog)
    @test length(aug2.aux_endog) == 2  # K_lag1, K_lag2 (lag-1 is standard, aux for 2 and 3)
    @test haskey(aug2.sub_map, (:K, -2))
    @test haskey(aug2.sub_map, (:K, -3))
end
```

**Step 2: Run tests to verify they fail**

Run: `julia --project=. -e 'using Test, MacroEconometricModels; import MacroEconometricModels: set_display_backend, PlotOutput; include("test/dsge/test_dsge.jl")'`
Expected: FAIL — `_scan_offsets` not defined.

**Step 3: Implement `_scan_offsets`**

Add to `src/dsge/parser.jl` BEFORE the `_dsge_impl` function (around line 46):

```julia
# =============================================================================
# State-space augmentation for arbitrary lag/lead depths (issue #54)
# =============================================================================

"""
    _scan_offsets(equations, endog, exog) → Dict{Symbol, NamedTuple}

Scan equation expressions for the maximum lag and lead depth of each variable.
Returns a Dict mapping variable name → `(max_lag=Int, max_lead=Int)`.
Tracks both endogenous and exogenous variables.
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
```

**Step 4: Implement `_generate_augmentation`**

Add immediately after `_scan_offsets`:

```julia
"""
    _generate_augmentation(offsets, endog, exog) → NamedTuple

Generate auxiliary variables, identity equations, and substitution map for
state-space augmentation.

Returns:
- `aux_endog::Vector{Symbol}` — new endogenous variables to append
- `aux_equations::Vector{Expr}` — identity equations in `LHS = RHS` form
- `sub_map::Dict{Tuple{Symbol,Int}, Tuple{Symbol,Int}}` — `(var, offset)` → `(new_var, new_offset)`
"""
function _generate_augmentation(offsets::Dict, endog::Vector{Symbol}, exog::Vector{Symbol})
    aux_endog = Symbol[]
    aux_equations = Expr[]
    sub_map = Dict{Tuple{Symbol,Int}, Tuple{Symbol,Int}}()

    # --- Endogenous lags: var[t-k] for k > 1 ---
    for var in endog
        info = get(offsets, var, (max_lag=0, max_lead=0))
        info.max_lag <= 1 && continue

        # Create auxiliaries: __lag_var_1 through __lag_var_{k-1}
        # __lag_var_1[t] = var[t-1]
        # __lag_var_j[t] = __lag_var_{j-1}[t-1]  for j >= 2
        # Substitution: var[t-m] → __lag_var_{m-1}[t-1]  for m >= 2
        for j in 1:(info.max_lag - 1)
            aux_name = Symbol("__lag_", var, "_", j)
            push!(aux_endog, aux_name)

            if j == 1
                # __lag_var_1[t] = var[t-1]
                eq = Expr(:(=), Expr(:ref, aux_name, :t),
                                Expr(:ref, var, Expr(:call, :-, :t, 1)))
            else
                prev_aux = Symbol("__lag_", var, "_", j - 1)
                # __lag_var_j[t] = __lag_var_{j-1}[t-1]
                eq = Expr(:(=), Expr(:ref, aux_name, :t),
                                Expr(:ref, prev_aux, Expr(:call, :-, :t, 1)))
            end
            push!(aux_equations, eq)

            # var[t-(j+1)] → aux_name[t-1]
            sub_map[(var, -(j + 1))] = (aux_name, -1)
        end
    end

    # --- Endogenous leads: var[t+k] for k > 1 ---
    for var in endog
        info = get(offsets, var, (max_lag=0, max_lead=0))
        info.max_lead <= 1 && continue

        for j in 1:(info.max_lead - 1)
            aux_name = Symbol("__fwd_", var, "_", j)
            push!(aux_endog, aux_name)

            if j == 1
                # __fwd_var_1[t] = var[t+1]
                eq = Expr(:(=), Expr(:ref, aux_name, :t),
                                Expr(:ref, var, Expr(:call, :+, :t, 1)))
            else
                prev_aux = Symbol("__fwd_", var, "_", j - 1)
                # __fwd_var_j[t] = __fwd_var_{j-1}[t+1]
                eq = Expr(:(=), Expr(:ref, aux_name, :t),
                                Expr(:ref, prev_aux, Expr(:call, :+, :t, 1)))
            end
            push!(aux_equations, eq)

            # var[t+(j+1)] → aux_name[t+1]
            sub_map[(var, j + 1)] = (aux_name, 1)
        end
    end

    # --- Exogenous news shocks: ε[t-k] for k > 0 ---
    for shock in exog
        info = get(offsets, shock, (max_lag=0, max_lead=0))
        info.max_lag <= 0 && continue

        for j in 1:info.max_lag
            aux_name = Symbol("__news_", shock, "_", j)
            push!(aux_endog, aux_name)

            if j == 1
                # __news_ε_1[t] = ε[t]  (identity: captures current shock)
                eq = Expr(:(=), Expr(:ref, aux_name, :t),
                                Expr(:ref, shock, :t))
            else
                prev_aux = Symbol("__news_", shock, "_", j - 1)
                # __news_ε_j[t] = __news_ε_{j-1}[t-1]
                eq = Expr(:(=), Expr(:ref, aux_name, :t),
                                Expr(:ref, prev_aux, Expr(:call, :-, :t, 1)))
            end
            push!(aux_equations, eq)

            # ε[t-j] → __news_ε_j[t-1]
            sub_map[(shock, -j)] = (aux_name, -1)
        end
    end

    (aux_endog=aux_endog, aux_equations=aux_equations, sub_map=sub_map)
end
```

**Step 5: Implement `_apply_augmentation_subs`**

Add immediately after `_generate_augmentation`:

```julia
"""
    _apply_augmentation_subs(ex, sub_map, endog, exog) → Expr

Walk expression tree and replace variable references according to the substitution map.
`sub_map` maps `(varname, offset)` → `(new_varname, new_offset)`.
"""
function _apply_augmentation_subs(ex, sub_map::Dict{Tuple{Symbol,Int}, Tuple{Symbol,Int}},
                                   endog::Vector{Symbol}, exog::Vector{Symbol})
    if ex isa Expr
        # Check for var[t±k] reference
        if ex.head == :ref && length(ex.args) == 2 && ex.args[1] isa Symbol
            varname = ex.args[1]::Symbol
            all_vars = vcat(endog, exog)
            if varname ∈ all_vars
                offset = _parse_time_index(ex.args[2])
                key = (varname, offset)
                if haskey(sub_map, key)
                    new_var, new_offset = sub_map[key]
                    if new_offset == 0
                        return Expr(:ref, new_var, :t)
                    elseif new_offset > 0
                        return Expr(:ref, new_var, Expr(:call, :+, :t, new_offset))
                    else
                        return Expr(:ref, new_var, Expr(:call, :-, :t, -new_offset))
                    end
                end
            end
        end
        # Recurse into children
        new_args = [_apply_augmentation_subs(a, sub_map, endog, exog) for a in ex.args]
        return Expr(ex.head, new_args...)
    else
        return ex
    end
end
```

**Step 6: Run tests**

Run: `julia --project=. -e 'using Test, MacroEconometricModels; import MacroEconometricModels: set_display_backend, PlotOutput; include("test/dsge/test_dsge.jl")'`
Expected: Scanner and generator tests pass. All 451 existing tests still pass.

**Step 7: Commit**

```bash
git add src/dsge/parser.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): add augmentation scanner and generator (#54)"
```

---

### Task 3: Wire Augmentation into @dsge Macro

**Files:**
- Modify: `src/dsge/parser.jl:51-162` (`_dsge_impl` function)
- Test: `test/dsge/test_dsge.jl`

**Context:** The `@dsge` macro implementation (`_dsge_impl`) currently processes equations directly. We insert an augmentation step between equation parsing (line 84) and forward-looking detection (line 92). After augmentation, the extended `endog` and `equations` lists are used for all subsequent steps.

**Step 1: Write integration test**

Add to `test/dsge/test_dsge.jl`:

```julia
@testset "Augmentation: @dsge with news shock (#54)" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ_0 = 0.01, σ_3 = 0.007
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ_0 * ε[t] + σ_3 * ε[t-3]
    end
    @test spec.augmented == true
    @test spec.n_original_endog == 1
    @test spec.original_endog == [:y]
    @test spec.n_endog == 4  # y + 3 news auxiliaries
    @test spec.max_lag == 3
    @test length(spec.original_equations) == 1
    @test length(spec.equations) == 4  # 1 user + 3 identity
end

@testset "Augmentation: @dsge with deep endogenous lag (#54)" begin
    spec = @dsge begin
        parameters: a1 = 0.5, a2 = 0.3, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = a1 * y[t-1] + a2 * y[t-2] + σ * ε[t]
    end
    @test spec.augmented == true
    @test spec.n_original_endog == 1
    @test spec.n_endog == 2  # y + 1 lag auxiliary
    @test spec.max_lag == 2
end

@testset "Augmentation: backward compat — no augmentation needed (#54)" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    @test spec.augmented == false
    @test spec.n_original_endog == spec.n_endog
    @test spec.original_endog == spec.endog
    @test spec.max_lag == 1
    @test spec.max_lead == 1
end
```

**Step 2: Modify `_dsge_impl` to insert augmentation**

Replace `_dsge_impl` in `src/dsge/parser.jl` (lines 51-162). The key changes are:

1. After parsing declarations + raw_equations (line 84), add augmentation logic
2. Store `original_endog` and `original_equations` before augmentation modifies them
3. Pass new keyword arguments to the DSGESpec constructor

The modified function:

```julia
function _dsge_impl(block::Expr)
    params = Symbol[]
    param_defaults = Dict{Symbol,Any}()
    endog = Symbol[]
    exog = Symbol[]
    raw_equations = Expr[]
    ss_body = nothing

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
            ss_body = stmt.args[3]
        elseif label === nothing
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

    # ── State-space augmentation (issue #54) ──
    original_endog = copy(endog)
    offsets = _scan_offsets([_equation_to_residual(eq) for eq in raw_equations],
                            endog, exog)

    needs_aug = any(v -> v.max_lag > 1 || v.max_lead > 1,
                    (get(offsets, s, (max_lag=0, max_lead=0)) for s in endog)) ||
                any(v -> v.max_lag > 0,
                    (get(offsets, s, (max_lag=0, max_lead=0)) for s in exog))

    aug_flag = needs_aug
    max_lag_val = 1
    max_lead_val = 1

    if needs_aug
        aug = _generate_augmentation(offsets, endog, exog)

        # Apply substitutions to user equations
        for i in eachindex(raw_equations)
            raw_equations[i] = _apply_augmentation_subs(raw_equations[i], aug.sub_map, endog, exog)
        end

        # Extend endogenous variables and append identity equations
        append!(endog, aug.aux_endog)
        append!(raw_equations, aug.aux_equations)

        # Compute max_lag / max_lead from offsets
        all_lags = [get(offsets, s, (max_lag=0, max_lead=0)).max_lag for s in vcat(original_endog, exog)]
        all_leads = [get(offsets, s, (max_lag=0, max_lead=0)).max_lead for s in original_endog]
        max_lag_val = maximum(all_lags; init=1)
        max_lead_val = maximum(all_leads; init=1)
    end

    # Classify forward-looking equations (on augmented set)
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
    original_equation_exprs = Expr[]  # user equations before augmentation subs

    for (i, eq) in enumerate(raw_equations)
        residual_ex = _equation_to_residual(eq)
        residual_ex = _strip_expectation_operator(residual_ex)
        subst_ex = _substitute_vars(residual_ex, endog, exog, params)
        push!(equation_exprs, residual_ex)

        fn_expr = Expr(:->, Expr(:tuple, :_y_t_, :_y_lag_, :_y_lead_, :_ε_, :_θ_), subst_ex)
        push!(residual_fn_exprs, fn_expr)
    end

    # Store original (pre-augmentation) equations for display
    # These are the first len(original_endog) equations in residual form
    # but we need them BEFORE substitution was applied.
    # We stored original_endog before augmentation. Re-derive original equations:
    # Actually, we need the original raw_equations before _apply_augmentation_subs.
    # Since we modified them in-place above, we need to store them beforehand.
    # This is handled by passing original_equations separately (see below).

    # Build the constructor call
    param_vals_expr = Expr(:call, :Dict,
        [Expr(:call, :(=>), QuoteNode(p), param_defaults[p]) for p in params]...)

    endog_expr = Expr(:vect, [QuoteNode(s) for s in endog]...)
    exog_expr = Expr(:vect, [QuoteNode(s) for s in exog]...)
    params_expr = Expr(:vect, [QuoteNode(s) for s in params]...)
    original_endog_expr = Expr(:vect, [QuoteNode(s) for s in original_endog]...)

    fwd_expr = Expr(:vect, forward_indices...)

    eq_vec_expr = Expr(:ref, :Expr, [QuoteNode(eq) for eq in equation_exprs]...)
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
            original_equations=Expr[$(QuoteNode.(equation_exprs[1:length(original_endog)])...)],
            augmented=$aug_flag,
            max_lag=$max_lag_val,
            max_lead=$max_lead_val
        )
    end

    return esc(result)
end
```

**Important implementation note:** The `original_equations` field stores the first `n_original_endog` equation Exprs from `equation_exprs`. These are the user's equations in residual form AFTER augmentation substitutions (e.g., `ε[t-8]` → `__news_ε_8[t-1]`). For display of the original equations, we need the PRE-substitution versions. To fix this, save the pre-substitution equations BEFORE calling `_apply_augmentation_subs`:

Insert this line BEFORE the `if needs_aug` block:
```julia
    original_raw_equations = deepcopy(raw_equations)
```

Then in the `if needs_aug` block, after building `equation_exprs`, compute original equations from `original_raw_equations`:
```julia
    # Store original equations (pre-augmentation) in residual form for display
    original_eq_exprs = Expr[]
    for eq in original_raw_equations
        resid = _equation_to_residual(eq)
        resid = _strip_expectation_operator(resid)
        push!(original_eq_exprs, resid)
    end
```

And pass `original_eq_exprs` instead of `equation_exprs[1:length(original_endog)]` when `needs_aug`:
```julia
    original_equations_final = needs_aug ? original_eq_exprs : equation_exprs
```

Use `original_equations_final` in the constructor:
```julia
    original_equations=Expr[$(QuoteNode.(original_equations_final)...)]
```

**Step 3: Handle ss_fn with augmentation**

When the model is augmented, the `ss_fn` returns values for the ORIGINAL variables only. The augmented steady state needs to pad:
- Endogenous lag auxiliaries: same as original variable
- News shock auxiliaries: zeros

This padding should happen in `compute_steady_state`. But for now, since `ss_fn` is optional and the numerical solver works on the full augmented system, just ensure `ss_fn` returns a vector of length `n_endog` (augmented). The simplest approach: if augmented and ss_fn provided, the ss_fn should return original values + the solver fills in auxiliaries. We can handle this in a follow-up. For now, `ss_fn` support with augmented models is deferred.

**Step 4: Run integration tests**

Run: `julia --project=. -e 'using Test, MacroEconometricModels; import MacroEconometricModels: set_display_backend, PlotOutput; include("test/dsge/test_dsge.jl")'`
Expected: All new augmentation tests pass. All 451 existing tests still pass.

**Step 5: Commit**

```bash
git add src/dsge/parser.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): wire state-space augmentation into @dsge macro (#54)"
```

---

### Task 4: Output Filtering in simulate, irf, fevd

**Files:**
- Modify: `src/dsge/simulation.jl:41-66` (simulate)
- Modify: `src/dsge/simulation.jl:84-108` (irf)
- Modify: `src/dsge/simulation.jl:120-145` (fevd)
- Test: `test/dsge/test_dsge.jl`

**Context:** After augmentation, `sol.G1` is `n_aug × n_aug` and `sol.impact` is `n_aug × n_exog`. The simulation runs on the full augmented system, but output should only contain the original user variables.

**Step 1: Write tests for output filtering**

Add to `test/dsge/test_dsge.jl`:

```julia
@testset "Augmentation: solve and simulate AR(2) (#54)" begin
    spec = @dsge begin
        parameters: a1 = 0.5, a2 = 0.3, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = a1 * y[t-1] + a2 * y[t-2] + σ * ε[t]
        steady_state = [0.0, 0.0]  # y=0, __lag_y_1=0
    end
    spec = compute_steady_state(spec)
    linear = linearize(spec)
    sol = solve(linear; method=:gensys)

    @test is_determined(sol)

    # IRF should only show original variable 'y'
    ir = irf(sol, 20)
    @test length(ir.variables) == 1
    @test ir.variables == ["y"]
    @test size(ir.values, 2) == 1  # only 1 variable

    # Verify AR(2) dynamics: IRF at h=1 should equal σ (impact)
    @test abs(ir.values[1, 1, 1] - 1.0) < 0.1  # σ * 1 unit shock

    # FEVD should only show original variable
    fv = fevd(sol, 20)
    @test length(fv.variables) == 1

    # Simulate should return only original variable
    sim = simulate(sol, 100; shock_draws=zeros(100, 1))
    @test size(sim, 2) == 1  # only 'y' column
end

@testset "Augmentation: news shock IRF timing (#54)" begin
    spec = @dsge begin
        parameters: σ_0 = 1.0, σ_3 = 0.5
        endogenous: y
        exogenous: ε
        y[t] = σ_0 * ε[t] + σ_3 * ε[t-3]
        steady_state = [0.0, 0.0, 0.0, 0.0]  # y + 3 news aux
    end
    spec = compute_steady_state(spec)
    linear = linearize(spec)
    sol = solve(linear; method=:gensys)

    @test is_determined(sol)

    ir = irf(sol, 10)
    @test length(ir.variables) == 1  # only 'y'

    # At h=1: immediate impact σ_0 = 1.0
    @test abs(ir.values[1, 1, 1] - 1.0) < 0.1
    # At h=4: delayed news impact σ_3 = 0.5 (shock at t hits y at t+3 via news chain)
    @test abs(ir.values[4, 1, 1] - 0.5) < 0.1
end
```

**Step 2: Modify `simulate` for output filtering**

In `src/dsge/simulation.jl`, modify the `simulate` function. After computing `levels` on the full augmented system, filter to original variables:

```julia
function simulate(sol::DSGESolution{T}, T_periods::Int;
                  shock_draws::Union{Nothing,AbstractMatrix}=nothing,
                  rng=Random.default_rng()) where {T<:AbstractFloat}
    n = nvars(sol)
    n_e = nshocks(sol)
    y_ss = sol.spec.steady_state

    if shock_draws !== nothing
        @assert size(shock_draws) == (T_periods, n_e) "shock_draws must be ($T_periods, $n_e)"
        e = T.(shock_draws)
    else
        e = randn(rng, T, T_periods, n_e)
    end

    dev = zeros(T, T_periods, n)
    for t in 1:T_periods
        y_prev = t == 1 ? zeros(T, n) : dev[t-1, :]
        dev[t, :] = sol.G1 * y_prev + sol.impact * e[t, :] + sol.C_sol
    end

    levels = dev .+ y_ss'

    # Filter to original variables if augmented
    if sol.spec.augmented
        orig_idx = _original_var_indices(sol.spec)
        return levels[:, orig_idx]
    end
    return levels
end
```

**Step 3: Modify `irf` for output filtering**

In `src/dsge/simulation.jl`, modify the `irf` function:

```julia
function irf(sol::DSGESolution{T}, horizon::Int;
             ci_type::Symbol=:none, kwargs...) where {T<:AbstractFloat}
    n = nvars(sol)
    n_e = nshocks(sol)

    point_irf = zeros(T, horizon, n, n_e)
    G1_power = Matrix{T}(I, n, n)

    for h in 1:horizon
        for j in 1:n_e
            point_irf[h, :, j] = G1_power * sol.impact[:, j]
        end
        G1_power = G1_power * sol.G1
    end

    # Filter to original variables if augmented
    if sol.spec.augmented
        orig_idx = _original_var_indices(sol.spec)
        point_irf = point_irf[:, orig_idx, :]
        var_names = [string(s) for s in sol.spec.original_endog]
        n_out = length(orig_idx)
    else
        var_names = sol.spec.varnames
        n_out = n
    end
    shock_names = [string(s) for s in sol.spec.exog]

    ci_lower = zeros(T, horizon, n_out, n_e)
    ci_upper = zeros(T, horizon, n_out, n_e)

    ImpulseResponse{T}(point_irf, ci_lower, ci_upper, horizon,
                        var_names, shock_names, ci_type)
end
```

**Step 4: Modify `fevd` for output filtering**

In `src/dsge/simulation.jl`, modify the `fevd` function. The simplest approach: since `irf` already filters, just update the variable names:

```julia
function fevd(sol::DSGESolution{T}, horizon::Int) where {T<:AbstractFloat}
    irf_result = irf(sol, horizon)
    n_vars = length(irf_result.variables)
    n_e = nshocks(sol)

    decomp = zeros(T, n_vars, n_e, horizon)
    props  = zeros(T, n_vars, n_e, horizon)

    @inbounds for h in 1:horizon
        for i in 1:n_vars
            total = zero(T)
            for j in 1:n_e
                prev = h == 1 ? zero(T) : decomp[i, j, h-1]
                decomp[i, j, h] = prev + irf_result.values[h, i, j]^2
                total += decomp[i, j, h]
            end
            total > 0 && (props[i, :, h] = decomp[i, :, h] ./ total)
        end
    end

    var_names = irf_result.variables
    shock_names = irf_result.shocks

    FEVD{T}(decomp, props, var_names, shock_names)
end
```

**Step 5: Run tests**

Run: `julia --project=. -e 'using Test, MacroEconometricModels; import MacroEconometricModels: set_display_backend, PlotOutput; include("test/dsge/test_dsge.jl")'`
Expected: AR(2) and news shock tests pass. All existing tests still pass.

**Step 6: Commit**

```bash
git add src/dsge/simulation.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): filter augmented output in simulate/irf/fevd (#54)"
```

---

### Task 5: Output Filtering in Perfect Foresight and OccBin

**Files:**
- Modify: `src/dsge/perfect_foresight.jl:103-107`
- Modify: `src/dsge/occbin.jl` (4 OccBinSolution constructor calls + 2 OccBinIRF calls)
- Test: `test/dsge/test_dsge.jl`

**Context:** Perfect foresight returns `PerfectForesightPath` with full augmented `path` matrix. OccBin returns `OccBinSolution` with `varnames` from `spec.varnames`. Both need to filter to original variables.

**Step 1: Modify `perfect_foresight` output**

In `src/dsge/perfect_foresight.jl`, at line 107, filter the path and deviations:

```julia
    path_full = reshape(copy(x), n, T_periods)'
    deviations_full = path_full .- y_ss'

    # Filter to original variables if augmented
    if spec.augmented
        orig_idx = _original_var_indices(spec)
        path = Matrix{FT}(path_full[:, orig_idx])
        deviations = Matrix{FT}(deviations_full[:, orig_idx])
    else
        path = Matrix{FT}(path_full)
        deviations = Matrix{FT}(deviations_full)
    end

    PerfectForesightPath{FT}(path, deviations, converged, iter, spec)
```

**Step 2: Modify OccBin `varnames`**

In `src/dsge/occbin.jl`, at each `OccBinSolution` constructor call, use original varnames:

```julia
    varnames = spec.augmented ? [string(s) for s in spec.original_endog] : spec.varnames
```

And filter `linear_path`, `pw_path`, `steady_state` to original variable columns:

For the single-constraint `occbin_solve` (~line 608-612):
```julia
    orig_idx = _original_var_indices(spec)
    OccBinSolution{T}(
        linear_path[:, orig_idx], pw_path[:, orig_idx], spec.steady_state[orig_idx],
        regime_history, converged, iterations,
        spec, spec.augmented ? [string(s) for s in spec.original_endog] : spec.varnames
    )
```

Apply the same pattern to all 4 OccBinSolution constructor calls and the 2 OccBinIRF calls.

**Step 3: Run tests**

Run: `julia --project=. -e 'using Test, MacroEconometricModels; import MacroEconometricModels: set_display_backend, PlotOutput; include("test/dsge/test_dsge.jl")'`
Expected: All tests pass including existing PF and OccBin tests.

**Step 4: Commit**

```bash
git add src/dsge/perfect_foresight.jl src/dsge/occbin.jl
git commit -m "feat(dsge): filter augmented output in perfect_foresight and occbin (#54)"
```

---

### Task 6: Display Updates

**Files:**
- Modify: `src/dsge/display.jl:627-664` (_show_dsge_text)
- Modify: `src/dsge/display.jl:676-715` (_show_dsge_latex)
- Modify: `src/dsge/display.jl:727-end` (_show_dsge_html)
- Test: `test/dsge/test_dsge.jl`

**Context:** Display functions currently iterate over `spec.equations` and `spec.endog`. After augmentation, these include auxiliary variables and identity equations. Display should show only the original user equations using `spec.original_equations` and `spec.original_endog`, and note when augmentation is active.

**Step 1: Modify `_show_dsge_text`**

In `src/dsge/display.jl`, update the text display function. Key changes:
- Use `spec.original_endog` for header variable list
- Use `spec.n_original_endog` for variable count
- Use `spec.original_equations` for equation rendering
- Add augmentation status line if augmented

```julia
function _show_dsge_text(io::IO, spec::DSGESpec{T}) where {T}
    # --- Header ---
    println(io, "DSGE Model Specification")
    println(io, repeat("=", 50))
    disp_endog = spec.augmented ? spec.original_endog : spec.endog
    n_disp = length(disp_endog)
    println(io, "  Endogenous variables:  ", n_disp,
            "  (", join(string.(disp_endog), ", "), ")")
    println(io, "  Exogenous shocks:      ", spec.n_exog,
            "  (", join(string.(spec.exog), ", "), ")")
    println(io, "  Parameters:            ", spec.n_params)
    disp_eq = spec.augmented ? spec.original_equations : spec.equations
    println(io, "  Equations:             ", length(disp_eq))
    println(io, "  Forward-looking:       ", spec.n_expect)
    if spec.augmented
        println(io, "  Augmented state dim:   ", spec.n_endog,
                "  (max lag: ", spec.max_lag, ", max lead: ", spec.max_lead, ")")
    end
    println(io)

    # --- Calibration (unchanged) ---
    println(io, "Calibration")
    println(io, repeat("-", 50))
    for p in spec.params
        val = get(spec.param_values, p, missing)
        println(io, "  ", string(p), " = ", val isa Missing ? "?" : _format_num_display(val))
    end
    println(io)

    # --- Equations (original only) ---
    println(io, "Model Equations")
    println(io, repeat("-", 50))
    for (i, eq) in enumerate(disp_eq)
        eq_str = _equation_to_display(eq, spec.original_endog, spec.exog, spec.params; mode=:text)
        println(io, "  ($i)  ", eq_str)
    end

    # --- Steady state (original variables only) ---
    if !isempty(spec.steady_state)
        println(io)
        println(io, "Steady State")
        println(io, repeat("-", 50))
        if spec.augmented
            orig_idx = _original_var_indices(spec)
            println(io, _steady_state_text(spec.original_endog, spec.steady_state[orig_idx]))
        else
            println(io, _steady_state_text(spec.endog, spec.steady_state))
        end
    end
end
```

**Step 2: Apply same pattern to `_show_dsge_latex` and `_show_dsge_html`**

Same logic: use `original_endog`, `original_equations`, filter steady state. For LaTeX equations in HTML mode, pass `spec.original_endog` to `_equation_to_display`.

**Step 3: Write display test**

Add to `test/dsge/test_dsge.jl`:

```julia
@testset "Augmentation: display shows original equations (#54)" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ_0 = 0.01, σ_3 = 0.007
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ_0 * ε[t] + σ_3 * ε[t-3]
    end

    io = IOBuffer()
    show(io, spec)
    output = String(take!(io))

    # Should show original equation count (1), not augmented (4)
    @test occursin("Equations:             1", output)
    # Should show original variable (y), not auxiliaries
    @test occursin("(y)", output)
    @test !occursin("__news", output)
    # Should show augmented state dim
    @test occursin("Augmented state dim", output)
    # Should show ε[t-3] in original equation (not __news substitution)
    @test occursin("ε", output)
end
```

**Step 4: Run tests**

Run: `julia --project=. -e 'using Test, MacroEconometricModels; import MacroEconometricModels: set_display_backend, PlotOutput; include("test/dsge/test_dsge.jl")'`
Expected: All tests pass.

**Step 5: Commit**

```bash
git add src/dsge/display.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): display original equations for augmented models (#54)"
```

---

### Task 7: Full Integration Tests

**Files:**
- Modify: `test/dsge/test_dsge.jl`

**Context:** Comprehensive tests covering the complete pipeline: parse → augment → steady state → linearize → solve → simulate/IRF/FEVD → display → estimation for various augmentation scenarios.

**Step 1: Write comprehensive test suite**

Add to `test/dsge/test_dsge.jl`:

```julia
@testset "News Shocks: Full Pipeline (#54)" begin
    # Beaudry-Portier style: technology with news component
    spec = @dsge begin
        parameters: ρ = 0.9, σ_0 = 0.01, σ_4 = 0.007
        endogenous: A, Y
        exogenous: ε_A
        A[t] = ρ * A[t-1] + σ_0 * ε_A[t] + σ_4 * ε_A[t-4]
        Y[t] = A[t]
        steady_state = begin
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # A, Y + 4 news aux
        end
    end
    spec = compute_steady_state(spec)
    linear = linearize(spec)
    sol = solve(linear; method=:gensys)

    @test is_determined(sol)
    @test sol.spec.augmented
    @test sol.spec.n_original_endog == 2

    # IRF: only A and Y shown
    ir = irf(sol, 20)
    @test length(ir.variables) == 2
    @test ir.variables == ["A", "Y"]

    # News shock timing: impact at h=1 from σ_0, delayed at h=5 from σ_4
    # (ε at t=0 contributes σ_0 immediately, ε at t=-4 that arrives at t=0 was drawn 4 periods ago)
    @test abs(ir.values[1, 1, 1]) > 0  # immediate impact on A

    # FEVD
    fv = fevd(sol, 20)
    @test length(fv.variables) == 2

    # Simulate
    sim = simulate(sol, 50; shock_draws=zeros(50, 1))
    @test size(sim, 2) == 2

    # Display
    io = IOBuffer()
    show(io, spec)
    output = String(take!(io))
    @test occursin("Augmented state dim", output)
    @test !occursin("__news", output)
end

@testset "Higher-Order Lead: y[t+2] (#54)" begin
    spec = @dsge begin
        parameters: a = 0.3, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = a * y[t+2] + σ * ε[t]
        steady_state = [0.0, 0.0]  # y + 1 fwd auxiliary
    end
    spec = compute_steady_state(spec)
    @test spec.augmented
    @test spec.max_lead == 2
    @test spec.n_endog == 2

    linear = linearize(spec)
    sol = solve(linear; method=:gensys)
    ir = irf(sol, 10)
    @test length(ir.variables) == 1
    @test ir.variables == ["y"]
end

@testset "Mixed: deep lag + news shock (#54)" begin
    spec = @dsge begin
        parameters: a1 = 0.4, a2 = 0.2, σ_0 = 1.0, σ_2 = 0.5
        endogenous: y
        exogenous: ε
        y[t] = a1 * y[t-1] + a2 * y[t-2] + σ_0 * ε[t] + σ_2 * ε[t-2]
        steady_state = begin
            zeros(4)  # y + 1 lag aux + 2 news aux
        end
    end
    spec = compute_steady_state(spec)
    @test spec.augmented
    @test spec.n_original_endog == 1
    @test spec.n_endog == 4  # y + __lag_y_1 + __news_ε_1 + __news_ε_2

    linear = linearize(spec)
    sol = solve(linear; method=:gensys)
    @test is_determined(sol)

    ir = irf(sol, 20)
    @test length(ir.variables) == 1
end

@testset "Augmentation: analytical_moments with news (#54)" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ_0 = 0.01, σ_2 = 0.005
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ_0 * ε[t] + σ_2 * ε[t-2]
        steady_state = [0.0, 0.0, 0.0]  # y + 2 news aux
    end
    spec = compute_steady_state(spec)
    sol = solve(linearize(spec); method=:gensys)

    # analytical_moments should work (uses Lyapunov on augmented system)
    moments = analytical_moments(sol)
    @test haskey(moments, :variance)
    @test size(moments[:variance]) == (3, 3)  # augmented system
end
```

**Step 2: Run full test suite**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: All tests pass including the new ones.

**Step 3: Commit**

```bash
git add test/dsge/test_dsge.jl
git commit -m "test(dsge): add comprehensive news shock and augmentation tests (#54)"
```

---

## Execution Notes

**Dependency order:** Tasks 1 → 2 → 3 → 4,5,6 (parallel) → 7

**Key risk:** The `@dsge` macro operates at Julia's macro expansion time. The augmentation functions (`_scan_offsets`, `_generate_augmentation`, `_apply_augmentation_subs`) must be pure functions on `Expr` trees — no runtime state. They are called inside `_dsge_impl` which runs at compile time.

**Backward compatibility:** For models with max_lag ≤ 1 and max_lead ≤ 1 and no exogenous lags, `augmented=false` and all behavior is identical to current code.

**Steady state with augmentation:** The numerical solver works on the full augmented system. Identity equations enforce correct relationships at steady state (lag auxiliaries = original var, news auxiliaries = 0). The `ss_fn` analytical function is NOT supported for augmented models in this implementation — it would need to return augmented-length vectors. Users should use numerical SS or provide augmented-length steady state vectors.
