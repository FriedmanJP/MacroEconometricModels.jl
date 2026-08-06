# Counterfactual module — model-implied news-shock menus (CF-07, #387)
#
# The model-implied sufficient statistic is the SQUARE causal-effect map: for
# each outcome/instrument an H×H matrix whose column s is the response to a
# policy shock announced at date 1 that hits the rule at date s (a news
# shock). With this menu, rule counterfactuals are exact (kernel :exact path).
#
# IMPLEMENTATION ROUTE (deviation from both options sketched in #387, with
# reason): the issue's route (a) — textual substitution + rebuild through the
# parser — is impossible at runtime: `@dsge` compiles `residual_fns` at macro
# expansion, and closures rebuilt via runtime `eval` hit the world-age problem
# inside the same `policy_news_matrix` call. Route (b) — raw Γ0/Γ1/Ψ/Π
# surgery — bypasses tested machinery. Instead we COMPOSE the existing
# compiled residual functions: every old equation is wrapped so the policy
# disturbance becomes ε_pol[t] + q₁[t−1], and a shared Laseen–Svensson news
# pipeline is appended:
#
#     q_i[t] = q_{i+1}[t−1] + ν_i[t]   (i = 1…J−1),    q_J[t] = ν_J[t]
#
# A unit impulse in ν_s at t = 1 travels down the registers and reaches the
# rule at t = s + 1 — exactly menu column s + 1 (column 1 is the unanticipated
# shock ε itself). One pipeline is shared by ALL news horizons, so the state
# dimension grows LINEARLY by J = H−1 (the parser's own ε[t−k] shift registers
# would grow by H(H−1)/2). Cost: one linear solve of dimension n + H − 1.

# Wrap an old residual so the policy disturbance is ε_pol[t] + q₁[t−1].
# Applied uniformly to EVERY old equation: replacing the process ε by the
# anticipated-augmented process everywhere it enters (including any parser
# `__news_*` seed registers from user-written ε[t−k] terms) is the correct
# semantics.
function _cf_wrap_policy_residual(fn::Function, n_old::Int, ex_old::Int, pol_idx::Int)
    return (y_t, y_lag, y_lead, eps_vec, theta) -> begin
        eps_mod = [j == pol_idx ? eps_vec[j] + y_lag[n_old+1] : eps_vec[j] for j in 1:ex_old]
        fn(view(y_t, 1:n_old), view(y_lag, 1:n_old), view(y_lead, 1:n_old), eps_mod, theta)
    end
end

# Residual of pipeline register i (of max_news): q_i[t] − q_{i+1}[t−1] − ν_i[t].
# nu_pos = 0 when this register has no shock feeding it (chunked menus).
function _cf_register_residual(n_old::Int, i::Int, max_news::Int, nu_pos::Int)
    return (y_t, y_lag, y_lead, eps_vec, theta) -> begin
        r = y_t[n_old+i]
        i < max_news && (r -= y_lag[n_old+i+1])
        nu_pos > 0 && (r -= eps_vec[nu_pos])
        r
    end
end

# Display-only Expr rewrite: ε_pol[...] -> (ε_pol[...] + q₁[t−1]).
function _cf_subst_policy_expr(ex, policy_shock::Symbol, addend::Expr)
    ex isa Expr || return ex
    if ex.head == :ref && ex.args[1] == policy_shock
        return Expr(:call, :+, ex, addend)
    end
    return Expr(ex.head, map(a -> _cf_subst_policy_expr(a, policy_shock, addend), ex.args)...)
end

"""
    _augment_policy_news(spec, policy_shock, max_news; active=1:max_news) -> DSGESpec

Append the shared news pipeline (`max_news` registers, one news shock per
horizon in `active`) to `spec`, composing the existing compiled residual
functions — no re-parsing, no runtime `eval`. Registers are zero in steady
state; `ss_fn` is wrapped accordingly; all parse flags propagate (the
`_respec` lesson, audit E-07).
"""
function _augment_policy_news(spec::DSGESpec{T}, policy_shock::Symbol, max_news::Int;
                              active::AbstractVector{Int}=1:max_news) where {T<:AbstractFloat}
    pol_idx = findfirst(==(policy_shock), spec.exog)
    pol_idx === nothing && throw(ArgumentError(
        "policy_shock :$policy_shock not found in spec.exog = $(spec.exog)"))
    max_news >= 1 || throw(ArgumentError(
        "max_news: expected >= 1, got $max_news"))
    all(i -> 1 <= i <= max_news, active) || throw(ArgumentError(
        "active: news horizons must lie in 1:$max_news"))

    n_old = spec.n_endog
    ex_old = spec.n_exog
    q_syms = [Symbol("__cf_news_q", i) for i in 1:max_news]
    nu_syms = [Symbol("__cf_news_", policy_shock, "_", i) for i in active]
    for s in q_syms
        s in spec.endog && throw(ArgumentError(
            "augmentation name collision: $s already exists in the spec"))
    end

    new_endog = vcat(spec.endog, q_syms)
    new_exog = vcat(spec.exog, nu_syms)
    nu_pos = Dict{Int,Int}(i => ex_old + r for (r, i) in enumerate(active))

    new_fns = Function[_cf_wrap_policy_residual(fn, n_old, ex_old, pol_idx)
                       for fn in spec.residual_fns]
    for i in 1:max_news
        push!(new_fns, _cf_register_residual(n_old, i, max_news, get(nu_pos, i, 0)))
    end

    # Display equations: substitute in the old ones, append the registers.
    q1lag = Expr(:ref, q_syms[1], :(t - 1))
    new_eqs = Expr[_cf_subst_policy_expr(eq, policy_shock, q1lag) for eq in spec.equations]
    for i in 1:max_news
        parts = Any[]
        i < max_news && push!(parts, Expr(:ref, q_syms[i+1], :(t - 1)))
        haskey(nu_pos, i) && push!(parts, Expr(:ref, Symbol("__cf_news_", policy_shock, "_", i), :t))
        rhs = isempty(parts) ? 0 : (length(parts) == 1 ? parts[1] : Expr(:call, :+, parts...))
        push!(new_eqs, Expr(:call, :-, Expr(:ref, q_syms[i], :t), rhs))
    end

    ss = isempty(spec.steady_state) ? T[] : vcat(spec.steady_state, zeros(T, max_news))
    old_ss_fn = spec.ss_fn
    new_ss_fn = old_ss_fn === nothing ? nothing :
                (theta -> vcat(old_ss_fn(theta), zeros(max_news)))

    DSGESpec{T}(new_endog, new_exog, spec.params, spec.param_values, new_eqs, new_fns,
                spec.n_expect, spec.forward_indices, ss, new_ss_fn;
                original_endog=(spec.augmented ? spec.original_endog : spec.endog),
                original_equations=spec.original_equations,
                augmented=true,
                max_lag=spec.max_lag,
                max_lead=spec.max_lead,
                linear=spec.linear)
end

"""
    policy_news_matrix(spec::DSGESpec, policy_shock, outcomes, instruments=[];
                       H=100, solver=:gensys, chunk=0) -> PolicyCausalEffects

Assemble the model-implied **square** news menu `Θ_ν` (McKay–Wolf Prop. 1;
CMW §2.2): for each mapped outcome/instrument an `H × H` matrix whose column
`s` is the response to the policy shock announced at date 1 and hitting the
rule at date `s` (column 1 = the unanticipated shock). Returns a square
[`PolicyCausalEffects`](@ref) (`n_s = H`, `source = :dsge`) with which rule
counterfactuals are exact (kernel `:exact` path).

- `outcomes` / `instruments`: `Pair{Symbol,Symbol}` maps from module symbol to
  model variable (an original endogenous of the spec).
- `solver`: linear solvers only (`:gensys`, `:klein`, `:blanchard_kahn`) —
  nonlinear news menus are a research topic, not a package feature.
- `chunk > 0` builds the menu in chunks of `chunk` news horizons
  (augment–solve–extract per chunk). The shared pipeline keeps the state
  dimension at `n + H − 1` regardless, so chunking only bounds the number of
  shock columns per solve; the default monolithic path is one solve.

**Cost**: state dimension grows by `H − 1` (shared Laseen–Svensson pipeline —
linear in `H`, not the O(H²) of per-horizon shift registers); the dominant
cost is one QZ decomposition of dimension `n + H − 1`.

**Rule immateriality**: `Θ_ν` itself DIFFERS across baseline closure rules,
but counterfactuals constructed from it are identical — the prevailing rule
need not be known; it must merely render the system determinate (McKay–Wolf;
CMW). The cross-rule invariance is asserted end-to-end in the CF-23 oracle
suite.
"""
function policy_news_matrix(spec::DSGESpec{T}, policy_shock::Symbol,
                            outcomes::AbstractVector{<:Pair{Symbol,Symbol}},
                            instruments::AbstractVector{<:Pair{Symbol,Symbol}}=Pair{Symbol,Symbol}[];
                            H::Int=100, solver::Symbol=:gensys,
                            chunk::Int=0) where {T<:AbstractFloat}
    H >= 1 || throw(ArgumentError("H: expected H >= 1, got $H"))
    solver in (:gensys, :klein, :blanchard_kahn) || throw(ArgumentError(
        "policy_news_matrix supports the linear solvers :gensys/:klein/:blanchard_kahn only (nonlinear news menus are out of scope), got :$solver"))
    isempty(outcomes) && throw(ArgumentError("outcomes: expected at least one outcome"))

    base_vars = spec.augmented ? spec.original_endog : spec.endog
    out_syms = Symbol[first(p) for p in outcomes]
    ins_syms = Symbol[first(p) for p in instruments]
    model_vars = vcat([last(p) for p in outcomes], [last(p) for p in instruments])
    for v in model_vars
        v in base_vars || throw(ArgumentError(
            "model variable :$v not found among the spec's original endogenous $(base_vars)"))
    end

    J = H - 1
    n_vars = length(model_vars)
    menu = [Matrix{T}(undef, H, H) for _ in 1:n_vars]

    ranges = J == 0 ? UnitRange{Int}[] :
             (chunk <= 0 ? [1:J] : [i:min(i + chunk - 1, J) for i in 1:chunk:J])

    function _extract_col!(ir, model_var::Symbol, shock_name::String, col::Int, v::Int)
        vi = findfirst(==(string(model_var)), ir.variables)
        si = findfirst(==(shock_name), ir.shocks)
        (vi === nothing || si === nothing) && throw(ArgumentError(
            "internal: could not locate $(model_var)/$(shock_name) in the augmented IRF"))
        menu[v][:, col] = ir.values[1:H, vi, si]
        return nothing
    end

    _check_eu(sol) = sol.eu == [1, 1] || throw(ArgumentError(
        "the augmented system is not determinate (eu = $(sol.eu)); the baseline closure rule must induce determinacy — close the model with ANY determinacy-inducing rule; the closure is immaterial for the counterfactual (McKay–Wolf Prop. 1, CMW §2.2)"))

    col1_done = false
    if isempty(ranges)
        sol = solve(spec; method=solver)
        _check_eu(sol)
        ir = irf(sol, H)
        for (v, mv) in enumerate(model_vars)
            _extract_col!(ir, mv, string(policy_shock), 1, v)
        end
        col1_done = true
    end
    for rg in ranges
        spec_aug = _augment_policy_news(spec, policy_shock, last(rg); active=collect(rg))
        sol = solve(spec_aug; method=solver)
        _check_eu(sol)
        ir = irf(sol, H)
        if !col1_done
            for (v, mv) in enumerate(model_vars)
                _extract_col!(ir, mv, string(policy_shock), 1, v)
            end
            col1_done = true
        end
        for i in rg
            sname = string(Symbol("__cf_news_", policy_shock, "_", i))
            for (v, mv) in enumerate(model_vars)
                _extract_col!(ir, mv, sname, i + 1, v)
            end
        end
    end

    n_x = length(out_syms)
    labels = [s == 1 ? "$(policy_shock) (s=0)" : "$(policy_shock) news (s=$(s-1))" for s in 1:H]
    PolicyCausalEffects{T}(out_syms, ins_syms,
                           menu[1:n_x], menu[n_x+1:end],
                           nothing, nothing, H, labels, :dsge)
end
