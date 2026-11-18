# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Automated observable-set invertibility search ranking candidate observable combinations
by Poor Man's invertibility / Komunjer–Ng transmission zeros (`ABCD-02`) and Sims (2012)
shock recoverability (`ABCD-04`), with a greedy / beam-search fallback for high-dimensional
models and `Tables.jl` / `DataFrame` support.

References:
- Fernández-Villaverde, J., Rubio-Ramírez, J. F., Sargent, T. J., & Watson, M. W. (2007).
  ABCs (and Ds) of Understanding VARs. *American Economic Review*, 97(3), p. 1024.
- Giannone, D., & Reichlin, L. (2006). Does information help recovering structural shocks
  from past observations? *Journal of the European Economic Association*, 4(2–3), 455–465.
- Komunjer, I., & Ng, S. (2011). Dynamic Identification of Dynamic Stochastic General
  Equilibrium Models. *Econometrica*, 79(6), 1995–2032.
- Sims, E. R. (2012). News, non-invertibility, and structural VARs.
  *Advances in Econometrics*, 28, 81–135.
"""

# =============================================================================
# ObservableSearchResult{T}
# =============================================================================

"""
    ObservableSearchResult{T<:AbstractFloat}

Ranked table of candidate observable subsets evaluated by [`poor_mans_invertibility`](@ref)
and [`shock_recoverability`](@ref) (`R²_j(0)`), implementing the `Tables.jl` row interface
so `DataFrame(result)` works directly.

Subsets are sorted best-first by `(is_invertible, min_R2, mean_R2, -rho, -cond_D)`:

```math
\\min_{j \\in \\mathcal{J}} R_j^2(0) = \\min_{j \\in \\mathcal{J}} \\mathrm{Var}\\!\\left(\\mathbb{E}[w_{j,t} \\mid y^t]\\right), \\qquad
\\rho = \\max_i |\\lambda_i(M)|
```

where:
- ``\\mathcal{J}`` is the index set of target shocks (`target_shocks`)
- ``R_j^2(0)`` is contemporaneous recoverability of structural shock ``w_{j,t}``
- ``M`` is the state-error feedback matrix ``A - B D^{-1} C`` (or finite Rosenbrock transmission zeros)
- ``\\rho`` is the spectral radius / maximum transmission-zero modulus

# Fields
- `observables::Vector{Vector{Symbol}}` — candidate observable subsets (sorted best-first)
- `is_invertible::Vector{Bool}` — invertibility indicator per subset
- `rho::Vector{T}` — maximum modulus of eigenvalues of `M` or finite transmission zeros (`Inf`/`NaN` if singular/underdetermined)
- `min_R2::Vector{T}` — minimum contemporaneous `R²_j(0)` across `target_shocks`
- `mean_R2::Vector{T}` — mean contemporaneous `R²_j(0)` across `target_shocks`
- `cond_D::Vector{T}` — condition number of impact submatrix `D` (`Inf` when rank-deficient)
- `status::Vector{Symbol}` — invertibility regime classification symbol per subset
- `k::Int` — target number of observables per subset
- `n_evaluated::Int` — number of size-`k` subsets evaluated
- `total_combinations::Int` — total combinatorial search space `binomial(length(candidates) - length(must_include), k - length(must_include))`
- `used_greedy::Bool` — `true` when `total_combinations > max_combinations` triggered beam/greedy fallback
- `target_shocks::Vector{Int}` — structural shock indices scored by `min_R2` and `mean_R2`
- `shock_names::Vector{String}` — names of the scored structural shocks
"""
struct ObservableSearchResult{T<:AbstractFloat}
    observables::Vector{Vector{Symbol}}
    is_invertible::Vector{Bool}
    rho::Vector{T}
    min_R2::Vector{T}
    mean_R2::Vector{T}
    cond_D::Vector{T}
    status::Vector{Symbol}
    k::Int
    n_evaluated::Int
    total_combinations::Int
    used_greedy::Bool
    target_shocks::Vector{Int}
    shock_names::Vector{String}

    function ObservableSearchResult{T}(
        observables::AbstractVector,
        is_invertible::AbstractVector,
        rho::AbstractVector,
        min_R2::AbstractVector,
        mean_R2::AbstractVector,
        cond_D::AbstractVector,
        status::AbstractVector,
        k::Int,
        n_evaluated::Int,
        total_combinations::Int,
        used_greedy::Bool,
        target_shocks::AbstractVector,
        shock_names::AbstractVector,
    ) where {T<:AbstractFloat}
        k >= 1 || throw(ArgumentError("k must be >= 1; got $k"))
        n_evaluated >= 0 || throw(ArgumentError("n_evaluated must be >= 0; got $n_evaluated"))
        total_combinations >= 0 || throw(ArgumentError("total_combinations must be >= 0; got $total_combinations"))

        obs_v = Vector{Symbol}[Symbol.(o) for o in observables]
        inv_v = Bool[Bool(x) for x in is_invertible]
        rho_v = Vector{T}(rho)
        min_v = Vector{T}(min_R2)
        mean_v = Vector{T}(mean_R2)
        cond_v = Vector{T}(cond_D)
        st_v = Symbol[Symbol(s) for s in status]
        ts_v = Int[Int(i) for i in target_shocks]
        sn_v = String[string(s) for s in shock_names]

        n = length(obs_v)
        n == n_evaluated || throw(ArgumentError(
            "length(observables) ($n) must equal n_evaluated ($n_evaluated)"))
        length(inv_v) == n || throw(ArgumentError("is_invertible length mismatch"))
        length(rho_v) == n || throw(ArgumentError("rho length mismatch"))
        length(min_v) == n || throw(ArgumentError("min_R2 length mismatch"))
        length(mean_v) == n || throw(ArgumentError("mean_R2 length mismatch"))
        length(cond_v) == n || throw(ArgumentError("cond_D length mismatch"))
        length(st_v) == n || throw(ArgumentError("status length mismatch"))
        length(ts_v) == length(sn_v) || throw(ArgumentError(
            "target_shocks and shock_names must have equal length"))

        new{T}(obs_v, inv_v, rho_v, min_v, mean_v, cond_v, st_v,
               k, n_evaluated, total_combinations, used_greedy, ts_v, sn_v)
    end
end

is_invertible(r::ObservableSearchResult) = any(r.is_invertible)

# =============================================================================
# Tables.jl Row Source Interface
# =============================================================================

Tables.istable(::Type{<:ObservableSearchResult}) = true
Tables.rowaccess(::Type{<:ObservableSearchResult}) = true

function Tables.schema(r::ObservableSearchResult{T}) where {T}
    return Tables.Schema(
        (:observables, :is_invertible, :rho, :min_R2, :mean_R2, :cond_D, :status),
        (Vector{Symbol}, Bool, T, T, T, T, Symbol),
    )
end

function Tables.rows(r::ObservableSearchResult{T}) where {T}
    return [
        (
            observables   = copy(r.observables[i]),
            is_invertible = r.is_invertible[i],
            rho           = r.rho[i],
            min_R2        = r.min_R2[i],
            mean_R2       = r.mean_R2[i],
            cond_D        = r.cond_D[i],
            status        = r.status[i],
        )
        for i in eachindex(r.observables)
    ]
end

# =============================================================================
# Display & Reporting
# =============================================================================

function Base.show(io::IO, res::ObservableSearchResult{T}) where {T}
    n_inv = count(res.is_invertible)
    best_obs = isempty(res.observables) ? "—" : join(string.(res.observables[1]), ", ")
    best_r2 = isempty(res.min_R2) ? "—" : _fmt(res.min_R2[1]; digits=6)
    best_rho = isempty(res.rho) ? "—" : _fmt(res.rho[1]; digits=6)

    spec_data = Any[
        "Subset size (k)"          res.k;
        "Evaluated subsets"        res.n_evaluated;
        "Total combinations"       res.total_combinations;
        "Search mode"              res.used_greedy ? "Greedy / beam" : "Exhaustive";
        "Invertible subsets found" n_inv;
        "Target shocks"            isempty(res.shock_names) ? "—" : join(res.shock_names, ", ");
        "Best observable set"      best_obs;
        "Best min R²(0)"           best_r2;
        "Best ρ(M)"                best_rho;
    ]
    _pretty_table(io, spec_data;
        title = "Observable-Set Invertibility Search",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    return nothing
end

"""
    report([io::IO], res::ObservableSearchResult; max_rows=25)

Print the summary and ranked candidate observable table of an [`ObservableSearchResult`](@ref).
"""
function report(io::IO, res::ObservableSearchResult{T}; max_rows::Int=25) where {T}
    show(io, res)
    n_show = min(length(res.observables), max_rows)
    if n_show > 0
        println(io)
        tbl = Matrix{Any}(undef, n_show, 8)
        for i in 1:n_show
            tbl[i, 1] = i
            tbl[i, 2] = join(string.(res.observables[i]), ", ")
            tbl[i, 3] = _yesno(res.is_invertible[i])
            tbl[i, 4] = _fmt(res.rho[i]; digits=6)
            tbl[i, 5] = _fmt(res.min_R2[i]; digits=6)
            tbl[i, 6] = _fmt(res.mean_R2[i]; digits=6)
            tbl[i, 7] = _fmt(res.cond_D[i]; digits=4)
            tbl[i, 8] = string(res.status[i])
        end
        _pretty_table(io, tbl;
            title = "Ranked Observable Subsets (Top $n_show of $(length(res.observables)))",
            column_labels = ["Rank", "Observables", "Inv.", "ρ", "min R²(0)", "mean R²(0)", "cond(D)", "Status"],
            alignment = [:r, :l, :c, :r, :r, :r, :r, :l],
        )
    end
    return nothing
end

report(res::ObservableSearchResult; max_rows::Int=25) = report(stdout, res; max_rows=max_rows)

# =============================================================================
# Internal Helpers (_abcd_srch_*)
# =============================================================================

_abcd_srch_default_nshocks(sol::Union{DSGESolution,PerturbationSolution,HADSGESolution,ABCDSystem}) =
    nshocks(sol)

_abcd_srch_default_nshocks(ss::StateSpaceModel) = nshocks(abcd_representation(ss))

_abcd_srch_default_candidates(sol::Union{DSGESolution,PerturbationSolution}) =
    copy(sol.spec.augmented ? sol.spec.original_endog : sol.spec.endog)

function _abcd_srch_default_candidates(sol::HADSGESolution)
    agg = _ha_obs_names(sol)
    isempty(agg) && return copy(sol.linear_solution.spec.endog)
    return Symbol.(agg)
end

_abcd_srch_default_candidates(ss::StateSpaceModel) =
    Symbol[Symbol("obs_$i") for i in 1:ss.n_obs]

_abcd_srch_default_candidates(sys::ABCDSystem) = Symbol.(sys.obs_names)

function _abcd_srch_extract_base(
    sol::Union{DSGESolution,PerturbationSolution},
    cand_syms::Vector{Symbol};
    measurement_error=nothing,
    abcd_tol::Real=1e-10,
)
    return abcd_representation(sol, cand_syms;
                               measurement_error=measurement_error,
                               reduction=:predetermined,
                               tol=abcd_tol)
end

function _abcd_srch_extract_base(
    sol::HADSGESolution,
    cand_syms::Vector{Symbol};
    measurement_error=nothing,
    abcd_tol::Real=1e-10,
)
    return abcd_representation(sol, cand_syms;
                               measurement_error=measurement_error,
                               reduction=:none,
                               tol=abcd_tol)
end

function _abcd_srch_extract_base(
    ss::StateSpaceModel,
    cand_syms::Vector{Symbol};
    measurement_error=nothing,
    abcd_tol::Real=1e-10,
)
    return abcd_representation(ss, cand_syms; reduction=:none, tol=abcd_tol)
end

function _abcd_srch_extract_base(
    sys::ABCDSystem{T},
    cand_syms::Vector{Symbol};
    measurement_error=nothing,
    abcd_tol::Real=1e-10,
) where {T<:AbstractFloat}
    base = _abcd_var_subset(sys, cand_syms)
    if measurement_error !== nothing
        B_aug, D_aug, sn_aug = _augment_abcd_measurement_error(
            base.B, base.D, base.shock_names, base.obs_names, measurement_error)
        return ABCDSystem{T}(base.A, B_aug, base.C, D_aug, base.c_state, base.d_obs,
                             base.state_names, base.obs_names, sn_aug)
    end
    return base
end

function _abcd_srch_resolve_shocks(shock_names::Vector{String}, target_shocks)
    m = length(shock_names)
    m >= 1 || throw(ArgumentError("Model has 0 structural shocks"))
    sh_vec = collect(target_shocks)
    isempty(sh_vec) && throw(ArgumentError("target_shocks cannot be empty"))

    if all(x -> x isa Integer, sh_vec)
        idx = Int[Int(x) for x in sh_vec]
        for i in idx
            (1 <= i <= m) || throw(BoundsError(1:m, i))
        end
        return unique(idx)
    elseif all(x -> x isa Symbol || x isa AbstractString, sh_vec)
        sh_syms = Symbol.(shock_names)
        idx = Int[]
        for s in Symbol.(sh_vec)
            pos = findfirst(==(s), sh_syms)
            pos === nothing && throw(ArgumentError(
                "Target shock :$s not found in available shocks: $sh_syms"))
            push!(idx, pos)
        end
        return unique(idx)
    else
        throw(ArgumentError(
            "target_shocks must be integer indices, Symbols, or Strings; got $(typeof(target_shocks))"))
    end
end

function _abcd_srch_safe_binomial(n::Int, r::Int)
    (r < 0 || r > n) && return 0
    val = binomial(big(n), big(r))
    return val > big(typemax(Int)) ? typemax(Int) : Int(val)
end

function _abcd_srch_combinations(items::Vector{Symbol}, r::Int)
    out = Vector{Vector{Symbol}}()
    r == 0 && (push!(out, Symbol[]); return out)
    n = length(items)
    r > n && return out
    buf = Vector{Symbol}(undef, r)
    function _rec(start::Int, depth::Int)
        if depth > r
            push!(out, copy(buf))
            return
        end
        limit = n - (r - depth)
        for i in start:limit
            buf[depth] = items[i]
            _rec(i + 1, depth + 1)
        end
    end
    _rec(1, 1)
    return out
end

function _abcd_srch_cond(D::Matrix{T}) where {T<:AbstractFloat}
    isempty(D) && return T(Inf)
    WT = promote_type(T, Float64)
    svs = svdvals(Matrix{WT}(D))
    isempty(svs) && return T(Inf)
    s1 = svs[1]
    s_last = svs[end]
    thresh = max(WT(max(size(D)...)) * eps(WT) * WT(10), WT(1e-14)) * s1
    (s1 <= zero(WT) || s_last <= thresh) && return T(Inf)
    return T(s1 / s_last)
end

function _abcd_srch_eval_subset(
    abcd_full::ABCDSystem{T},
    cand_pos::Dict{Symbol,Int},
    sub_syms::Vector{Symbol},
    target_idx::Vector{Int};
    reduction::Symbol=:minimal,
    abcd_tol::Real=1e-10,
    tol::Real=1e-8,
    rcond_tol::Real=1e-12,
) where {T<:AbstractFloat}
    obs_idx = Int[cand_pos[s] for s in sub_syms]
    C_sub = Matrix{T}(abcd_full.C[obs_idx, :])
    D_sub = Matrix{T}(abcd_full.D[obs_idx, :])
    d_sub = Vector{T}(abcd_full.d_obs[obs_idx])
    on_sub = Vector{String}(abcd_full.obs_names[obs_idx])

    if reduction === :minimal
        A_m, B_m, C_m, sn_m, U_tot = _abcd_minimal_realization(
            abcd_full.A, abcd_full.B, C_sub, abcd_full.state_names; tol=abcd_tol)
        c_m = Vector{T}(U_tot' * abcd_full.c_state)
    else
        A_m = abcd_full.A
        B_m = abcd_full.B
        C_m = C_sub
        sn_m = abcd_full.state_names
        c_m = abcd_full.c_state
    end

    sys_sub = ABCDSystem{T}(A_m, B_m, C_m, D_sub, c_m, d_sub,
                            sn_m, on_sub, abcd_full.shock_names)
    inv_res = poor_mans_invertibility(sys_sub; tol=tol, max_var_lags=0)
    cD = _abcd_srch_cond(D_sub)

    r2_vec = try
        shock_recoverability(sys_sub; max_lead=0, rcond_tol=rcond_tol).r2_contemp
    catch e
        if e isa ArgumentError && occursin("DARE solver failed", e.msg)
            # Unit-circle transmission zero or borderline undetectability fallback
            try
                n_s = nstates(sys_sub)
                sys_reg = ABCDSystem{T}(
                    n_s == 0 ? sys_sub.A : sys_sub.A * T(1 - 1e-6),
                    sys_sub.B * T(1 - 1e-6),
                    sys_sub.C,
                    sys_sub.D,
                    sys_sub.c_state,
                    sys_sub.d_obs,
                    sys_sub.state_names,
                    sys_sub.obs_names,
                    sys_sub.shock_names,
                )
                shock_recoverability(sys_reg; max_lead=0, rcond_tol=rcond_tol).r2_contemp
            catch
                zeros(T, nshocks(sys_sub))
            end
        else
            rethrow()
        end
    end

    r2_t = r2_vec[target_idx]
    min_r2 = T(minimum(r2_t))
    mean_r2 = T(mean(r2_t))

    return (
        observables   = sub_syms,
        is_invertible = inv_res.invertible,
        rho           = inv_res.max_modulus,
        min_R2        = min_r2,
        mean_R2       = mean_r2,
        cond_D        = cD,
        status        = inv_res.status,
    )
end

function _abcd_srch_sort_key(row)
    rho = row.rho
    rho_class = isnan(rho) ? 2 : (isinf(rho) ? 1 : 0)
    rho_val = (isnan(rho) || isinf(rho)) ? zero(typeof(rho)) : rho
    cD = isnan(row.cond_D) ? typeof(row.cond_D)(Inf) : row.cond_D
    return (
        !row.is_invertible,
        -row.min_R2,
        -row.mean_R2,
        rho_class,
        rho_val,
        cD,
    )
end

function _abcd_srch_canonical_key(syms::Vector{Symbol}, cand_pos::Dict{Symbol,Int})
    return Tuple(sort!(Int[cand_pos[s] for s in syms]))
end

# =============================================================================
# Public API: search_invertible_observables
# =============================================================================

"""
    search_invertible_observables(sol, k::Int=nshocks(sol); candidates=..., must_include=Symbol[], target_shocks=1:nshocks(sol), max_combinations=2000, ...) -> ObservableSearchResult

Search over `k`-element subsets of candidate observables containing `must_include` and rank
them by invertibility ([`poor_mans_invertibility`](@ref)) and contemporaneous structural
shock recoverability ([`shock_recoverability`](@ref)):

```math
\\text{Rank by } \\left(\\text{is\\_invertible},\\; \\min_{j \\in \\mathcal{J}} R_j^2(0),\\;
\\frac{1}{|\\mathcal{J}|} \\sum_{j \\in \\mathcal{J}} R_j^2(0),\\; -\\rho,\\; -\\kappa(D)\\right)
```

where:
- ``\\mathcal{J}`` is the target shock index set (`target_shocks`)
- ``R_j^2(0)`` is contemporaneous shock recoverability from the steady-state innovations representation
- ``\\rho`` is the spectral radius of ``A - B D^{-1} C`` (or maximum modulus of finite transmission zeros)
- ``\\kappa(D)`` is the condition number of the observation impact submatrix ``D``

When the total number of subsets exceeds `max_combinations`, the search switches from
exhaustive enumeration to a budgeted beam search with single-swap local refinement.
The returned [`ObservableSearchResult`](@ref) implements the `Tables.jl` row interface so
`DataFrame(result)` works out of the box.

# Keyword Arguments
- `candidates`: vector of candidate observable symbols (defaults to model observables)
- `must_include::AbstractVector{Symbol}=Symbol[]`: observables that must appear in every candidate subset
- `target_shocks=1:nshocks(sol)`: shock indices or names scored by `min_R2` and `mean_R2`
- `max_combinations::Int=2000`: maximum combinations for exhaustive enumeration before switching to greedy/beam search
- `measurement_error=nothing`: optional measurement-error SDs or covariance forwarded to [`abcd_representation`](@ref)
- `reduction::Symbol=:minimal`: state-space reduction mode (`:minimal` removes unobservable states per subset; `:predetermined` or `:none` skips staircase deflation)
- `abcd_tol::Real=1e-10`: tolerance for minimal realization deflation
- `tol::Real=1e-8`: unit-circle tolerance for [`poor_mans_invertibility`](@ref)
- `rcond_tol::Real=1e-12`: conditioning tolerance for [`shock_recoverability`](@ref)

# References
- Fernández-Villaverde, J., Rubio-Ramírez, J. F., Sargent, T. J., & Watson, M. W. (2007).
  ABCs (and Ds) of Understanding VARs. *American Economic Review*, 97(3), p. 1024.
- Giannone, D., & Reichlin, L. (2006). Does information help recovering structural shocks
  from past observations? *Journal of the European Economic Association*, 4(2–3), 455–465.
"""
function search_invertible_observables(
    sol::Union{DSGESolution,PerturbationSolution,HADSGESolution,StateSpaceModel,ABCDSystem},
    k::Int=_abcd_srch_default_nshocks(sol);
    candidates=_abcd_srch_default_candidates(sol),
    must_include::AbstractVector=Symbol[],
    target_shocks=1:_abcd_srch_default_nshocks(sol),
    max_combinations::Int=2000,
    measurement_error=nothing,
    reduction::Symbol=:minimal,
    abcd_tol::Real=1e-10,
    tol::Real=1e-8,
    rcond_tol::Real=1e-12,
)
    k >= 1 || throw(ArgumentError("Subset size k must be >= 1; got $k"))
    max_combinations >= 1 || throw(ArgumentError("max_combinations must be >= 1; got $max_combinations"))
    tol >= 0 || throw(ArgumentError("tol must be non-negative; got $tol"))
    rcond_tol > 0 || throw(ArgumentError("rcond_tol must be positive; got $rcond_tol"))
    _validate_abcd_reduction(reduction, (:minimal, :predetermined, :none))

    cand_syms = unique(Symbol.(collect(candidates)))
    isempty(cand_syms) && throw(ArgumentError("candidates cannot be empty"))
    must_syms = unique(Symbol.(collect(must_include)))

    for s in must_syms
        s in cand_syms || throw(ArgumentError(
            "must_include observable :$s not found in candidates: $cand_syms"))
    end
    length(must_syms) <= k || throw(ArgumentError(
        "must_include has $(length(must_syms)) variables, which exceeds k=$k"))
    length(cand_syms) >= k || throw(ArgumentError(
        "Only $(length(cand_syms)) candidates available for subset size k=$k"))

    abcd_full = _abcd_srch_extract_base(sol, cand_syms;
                                        measurement_error=measurement_error,
                                        abcd_tol=abcd_tol)
    T = eltype(abcd_full.A)
    target_idx = _abcd_srch_resolve_shocks(abcd_full.shock_names, target_shocks)
    target_shock_names = Vector{String}(abcd_full.shock_names[target_idx])

    cand_pos = Dict{Symbol,Int}(s => i for (i, s) in enumerate(cand_syms))
    remaining = Symbol[s for s in cand_syms if !(s in must_syms)]
    r = k - length(must_syms)
    total_combos = _abcd_srch_safe_binomial(length(remaining), r)

    eval_fn(sub) = _abcd_srch_eval_subset(
        abcd_full, cand_pos, sub, target_idx;
        reduction=reduction, abcd_tol=abcd_tol, tol=tol, rcond_tol=rcond_tol,
    )

    RowT = NamedTuple{
        (:observables, :is_invertible, :rho, :min_R2, :mean_R2, :cond_D, :status),
        Tuple{Vector{Symbol}, Bool, T, T, T, T, Symbol},
    }
    rows = RowT[]

    if total_combos <= max_combinations
        used_greedy = false
        for combo in _abcd_srch_combinations(remaining, r)
            sub = vcat(must_syms, combo)
            push!(rows, eval_fn(sub))
        end
    else
        used_greedy = true
        seen = Set{Tuple{Vararg{Int}}}()
        beam = Vector{Symbol}[copy(must_syms)]

        if r == 0
            push!(rows, eval_fn(copy(must_syms)))
        else
            # Beam search across steps 1..r with budget strictly <= max_combinations
            per_step_budget = max(1, div(max_combinations, max(r, 1)))
            for step in 1:r
                step_rows = RowT[]
                step_seen = Set{Tuple{Vararg{Int}}}()
                is_final = (step == r)
                step_limit = is_final ? max(1, max_combinations - length(rows)) : per_step_budget

                for base in beam
                    for cand in remaining
                        cand in base && continue
                        child = vcat(base, [cand])
                        ck = _abcd_srch_canonical_key(child, cand_pos)
                        ck in step_seen && continue
                        push!(step_seen, ck)
                        row = eval_fn(child)
                        push!(step_rows, row)
                        if is_final && !(ck in seen)
                            push!(seen, ck)
                            push!(rows, row)
                        end
                        length(step_rows) >= step_limit && break
                    end
                    length(step_rows) >= step_limit && break
                end

                sort!(step_rows; by=_abcd_srch_sort_key)
                if !is_final
                    next_width = max(1, min(length(step_rows),
                        div(per_step_budget, max(length(remaining) - step, 1))))
                    beam = Vector{Symbol}[step_rows[i].observables for i in 1:next_width]
                end
            end

            # 1-swap local refinement around best size-k candidate within remaining budget
            if !isempty(rows) && length(rows) < max_combinations
                sort!(rows; by=_abcd_srch_sort_key)
                best_sub = rows[1].observables
                must_set = Set(must_syms)
                swap_positions = [i for i in eachindex(best_sub) if !(best_sub[i] in must_set)]
                for pos in swap_positions
                    for cand in remaining
                        cand in best_sub && continue
                        length(rows) >= max_combinations && break
                        trial = copy(best_sub)
                        trial[pos] = cand
                        ck = _abcd_srch_canonical_key(trial, cand_pos)
                        ck in seen && continue
                        push!(seen, ck)
                        push!(rows, eval_fn(trial))
                    end
                    length(rows) >= max_combinations && break
                end
            end
        end
    end

    sort!(rows; by=_abcd_srch_sort_key)

    return ObservableSearchResult{T}(
        Vector{Symbol}[row.observables for row in rows],
        Bool[row.is_invertible for row in rows],
        T[row.rho for row in rows],
        T[row.min_R2 for row in rows],
        T[row.mean_R2 for row in rows],
        T[row.cond_D for row in rows],
        Symbol[row.status for row in rows],
        k,
        length(rows),
        total_combos,
        used_greedy,
        target_idx,
        target_shock_names,
    )
end
