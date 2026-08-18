# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Sequence-space block composition (DAG) and second-order sequence-space Jacobians.

The first-order household Jacobian (`_ssj_jacobian`, the fake-news algorithm of
Auclert, Bardóczy, Rognlie & Straub 2021) prices *one* heterogeneous-agent block.
The value of the sequence-space method, though, is **composition**: a model is a
directed acyclic graph (DAG) of blocks — heterogeneous-agent blocks (`HetBlock`,
`MitBlock`) and equation blocks (`SimpleBlock`) — whose general-equilibrium
Jacobian follows from the implicit function theorem applied along a topological
ordering.

This file provides

1. `SimpleBlock` / `HetBlock` — block types exposing sequence-space Jacobians,
2. `combine_blocks` — DAG assembly with topological sort and cycle detection,
3. `ssj_jacobian` — forward accumulation of total derivatives along the DAG, plus
   the general-equilibrium system `H_U`, `H_Z` for arbitrary unknowns/targets/shocks,
4. `ssj_irf` — first-order impulse responses `dU = −H_U⁻¹ H_Z dZ`, and
   **second-order** impulse responses (`order=2`) via directional contraction of
   the second-order sequence-space Jacobian tensors.

# References
- Auclert, A., Bardóczy, B., Rognlie, M., & Straub, L. (2021). Using the
  sequence-space Jacobian to solve and estimate heterogeneous-agent models.
  *Econometrica*, 89(5), 2375–2408.
- Auclert, A., Bardóczy, B., & Rognlie, M. (2023). MPCs, MPEs, and multipliers:
  a trilemma for New Keynesian models. *Review of Economics and Statistics*,
  105(3), 700–712. (Sequence-space block algebra.)
- Bhandari, A., Bourany, T., Evans, D., & Golosov, M. (2023). A perturbational
  approach for approximating heterogeneous-agent models. NBER WP 31744.
  (Higher-order sequence-space perturbation.)
"""

using LinearAlgebra

# =============================================================================
# Block types
# =============================================================================

"""
    AbstractSSJBlock

Supertype of sequence-space blocks. A block maps input sequences to output
sequences and can report (i) its steady-state input/output levels, (ii) its
first-order sequence-space Jacobians `∂O_t/∂I_s`, and (iii) a *nonlinear*
evaluation along a full input path (used for the second-order solution and for
market-clearing residual diagnostics).

Concrete subtypes: [`SimpleBlock`](@ref), [`HetBlock`](@ref), [`MitBlock`](@ref).
"""
abstract type AbstractSSJBlock end

"""
    SimpleBlock{T,F} <: AbstractSSJBlock

An equation block: a smooth, time-invariant function of its inputs at a fixed set
of leads and lags,

    (O¹_t, …, O^m_t) = f(I¹_{t-l₁}, …, I^k_{t-l_p}),

evaluated pointwise in `t`. Its sequence-space Jacobian is the constant partial
derivative `∂f_o/∂I_{i,l}` placed on the `l`-th diagonal, so the block Jacobian is
a (possibly banded) Toeplitz matrix — no simulation required.

Use `SimpleBlock` for production functions, Taylor rules, market-clearing
residuals, budget identities, and any other equation-representable relation.

# Constructor

    SimpleBlock(f; inputs, outputs, ss_inputs, lags=Dict(), name=:simple)

- `f` — `f(x::AbstractVector) -> AbstractVector` returning one value per element of
  `outputs`. `x` is ordered by [`ssj_arg_order`](@ref): inputs in the order given,
  and within each input its lags in ascending order. `f` must be generic in the
  element type (it is differentiated with `ForwardDiff`), so annotate arguments
  with `Real`/`AbstractVector` rather than `Float64`.
- `inputs::Vector{Symbol}` — input variable names.
- `outputs::Vector{Symbol}` — output variable names produced by this block.
- `ss_inputs::Dict{Symbol,T}` — steady-state level of every input.
- `lags::Dict{Symbol,Vector{Int}}` — lead/lag structure. `l > 0` means `I_{t-l}`
  (a lag), `l < 0` means `I_{t+|l|}` (a lead). Inputs absent from `lags` enter
  contemporaneously (`[0]`). An input may enter at several leads/lags.
- `name::Symbol` — block label used in displays and error messages.

Steady-state outputs are computed at construction by evaluating `f` at
`ss_inputs`; they are available as `block.ss_outputs`.

# Example — Cobb-Douglas firm (Krusell-Smith)

```julia
firm = SimpleBlock(
    x -> begin
        K_lag, Z, L = x[1], x[2], x[3]
        r = alpha * Z * (K_lag / L)^(alpha - 1) - delta
        w = (1 - alpha) * Z * (K_lag / L)^alpha
        Y = Z * K_lag^alpha * L^(1 - alpha)
        [r, w, Y]
    end;
    inputs  = [:K, :Z, :L],
    outputs = [:r, :w, :Y],
    lags    = Dict(:K => [1]),          # capital enters with a one-period lag
    ss_inputs = Dict(:K => K_ss, :Z => 1.0, :L => 1.0),
    name = :firm)
```
"""
struct SimpleBlock{T<:AbstractFloat,F} <: AbstractSSJBlock
    name::Symbol
    inputs::Vector{Symbol}
    outputs::Vector{Symbol}
    lags::Dict{Symbol,Vector{Int}}
    f::F
    ss_inputs::Dict{Symbol,T}
    ss_outputs::Dict{Symbol,T}
    arg_order::Vector{Tuple{Symbol,Int}}
end

"""
    ssj_arg_order(block::SimpleBlock) -> Vector{Tuple{Symbol,Int}}

The `(input, lag)` pairs, in the order in which `SimpleBlock`'s function `f`
receives them: inputs in declaration order, and within each input its lags in
ascending order (leads, which are negative, first).
"""
ssj_arg_order(b::SimpleBlock) = b.arg_order

function SimpleBlock(f;
                     inputs::AbstractVector{Symbol},
                     outputs::AbstractVector{Symbol},
                     ss_inputs::AbstractDict{Symbol,T},
                     lags::AbstractDict{Symbol,<:AbstractVector{Int}}=Dict{Symbol,Vector{Int}}(),
                     name::Symbol=:simple) where {T<:AbstractFloat}
    inputs = collect(Symbol, inputs)
    outputs = collect(Symbol, outputs)
    isempty(inputs) && throw(ArgumentError("SimpleBlock :$name has no inputs"))
    isempty(outputs) && throw(ArgumentError("SimpleBlock :$name has no outputs"))
    length(unique(inputs)) == length(inputs) ||
        throw(ArgumentError("SimpleBlock :$name has duplicate inputs: $inputs"))
    length(unique(outputs)) == length(outputs) ||
        throw(ArgumentError("SimpleBlock :$name has duplicate outputs: $outputs"))
    for i in inputs
        haskey(ss_inputs, i) || throw(ArgumentError(
            "SimpleBlock :$name is missing the steady-state level of input :$i " *
            "(supply it in `ss_inputs`)"))
    end
    for k in keys(lags)
        k in inputs || throw(ArgumentError(
            "SimpleBlock :$name declares lags for :$k, which is not an input"))
    end

    lag_map = Dict{Symbol,Vector{Int}}()
    arg_order = Tuple{Symbol,Int}[]
    for i in inputs
        li = sort(unique(collect(Int, get(lags, i, Int[0]))))
        lag_map[i] = li
        for l in li
            push!(arg_order, (i, l))
        end
    end

    x_ss = T[ss_inputs[i] for (i, _) in arg_order]
    y_ss = collect(f(x_ss))
    length(y_ss) == length(outputs) || throw(ArgumentError(
        "SimpleBlock :$name declares $(length(outputs)) outputs but `f` returned " *
        "$(length(y_ss)) values"))
    ss_out = Dict{Symbol,T}(o => T(y_ss[k]) for (k, o) in enumerate(outputs))

    return SimpleBlock{T,typeof(f)}(name, inputs, outputs, lag_map, f,
                                    Dict{Symbol,T}(ss_inputs), ss_out, arg_order)
end

# Individual-outcome families the fake-news aggregation understands.
const _HETBLOCK_OUTPUT_KIND = Dict{Symbol,Symbol}(
    :C => :consumption, :c => :consumption, :consumption => :consumption,
    :K => :savings, :A => :savings, :B => :savings,
    :assets => :savings, :a => :savings, :savings => :savings,
    :N => :labor, :n => :labor, :hours => :labor,
    :labor => :labor, :L => :labor, :efficiency_labor => :labor,
)

"""
    HetBlock{T} <: AbstractSSJBlock

A heterogeneous-agent block: the household problem of an incomplete-markets model,
mapping aggregate price sequences (`:r`, `:w`, …) to aggregate outcome sequences
(`:A`/`:K`/`:B` for assets, `:C` for consumption).

Its sequence-space Jacobian is the **fake-news** Jacobian of Auclert, Bardóczy,
Rognlie & Straub (2021) — dense, with non-zero anticipation entries `J[t,s] ≠ 0`
for `t < s`. Its nonlinear evaluation along an input path is a backward pass of the
endogenous grid method from the terminal steady state followed by a forward pass
of the Young (2010) histogram from the initial stationary distribution.

# Constructors

    HetBlock(spec::ModelSpec, ss::HASteadyState; inputs, outputs, name=:household, dx=1e-4)
    HetBlock(ss::HASteadyState, individual, grid, income; inputs, outputs, name=:household, dx=1e-4)

Non-household populations (`LifeCycleSystem`, `ContinuousHouseholdSystem`)
construct a [`MitBlock`](@ref) from the same `HetBlock(spec, ss)` call.
`DCEGMSystem` is rejected (G-11: discrete-choice kinks).

- `inputs::Vector{Symbol}` — prices the household responds to; each must be a key
  of `ss.prices` (default `[:r, :w]`).
- `outputs::Vector{Symbol}` — aggregates the block produces. Asset aggregates
  (`:A`, `:K`, `:B`, `:assets`, `:a`, `:savings`) aggregate the savings policy;
  consumption aggregates (`:C`, `:c`, `:consumption`) aggregate the consumption
  policy. Any other name is rejected rather than silently treated as an asset.
- `dx::Real` — finite-difference step used inside the fake-news backward pass.
  The default `1e-4` matches `_ssj_jacobian`. Benchmarked against a converged
  central finite difference of the *nonlinear* transition path on Krusell-Smith,
  the resulting Jacobian is accurate to about `8e-5` in relative terms at
  `dx=1e-4` and about `2e-7` at `dx=1e-5`; the gap is finite-difference
  truncation across the borrowing-constraint kink, not error in the fake-news
  recursion. Pass `dx=1e-5` when the Jacobian feeds an estimation likelihood.

# Restrictions

One- or two-asset grids. The GE close in `_ssj_solve` uses a one-unknown
capital DAG; two-asset models keep the `_ssj_jacobian` helper rather than
silently dropping `:two_asset_hank`.
"""
struct HetBlock{T<:AbstractFloat} <: AbstractSSJBlock
    name::Symbol
    inputs::Vector{Symbol}
    outputs::Vector{Symbol}
    steady_state::HASteadyState{T}
    individual::IndividualProblem{T}
    grid::HAGrid{T}
    income::IncomeProcess{T}
    dx::T
    ss_inputs::Dict{Symbol,T}
    ss_outputs::Dict{Symbol,T}
end

function HetBlock(ss::HASteadyState{T}, individual::IndividualProblem{T},
                  grid::HAGrid{T}, income::IncomeProcess{T};
                  inputs::AbstractVector{Symbol}=Symbol[:r, :w],
                  outputs::AbstractVector{Symbol}=Symbol[:A],
                  name::Symbol=:household,
                  dx::Real=T(1e-4)) where {T<:AbstractFloat}
    inputs = collect(Symbol, inputs)
    outputs = collect(Symbol, outputs)
    grid.n_dims in (1, 2) || throw(ArgumentError(
        "HetBlock :$name requires a one- or two-asset grid (got n_dims=$(grid.n_dims))"))
    individual.n_asset_dims == grid.n_dims || throw(ArgumentError(
        "HetBlock :$name: individual n_asset_dims must match the grid"))
    isempty(inputs) && throw(ArgumentError("HetBlock :$name has no inputs"))
    isempty(outputs) && throw(ArgumentError("HetBlock :$name has no outputs"))
    for i in inputs
        haskey(ss.prices, i) || throw(ArgumentError(
            "HetBlock :$name input :$i is not a steady-state price " *
            "(available: $(sort(collect(keys(ss.prices)))))"))
    end
    for o in outputs
        haskey(_HETBLOCK_OUTPUT_KIND, o) || throw(ArgumentError(
            "HetBlock :$name output :$o is not a recognized household aggregate. " *
            "Use an asset aggregate (:A, :K, :B, :assets, :a, :savings), a " *
            "consumption aggregate (:C, :c, :consumption), or — for an " *
            "endogenous-labor model — a labor aggregate (:N, :n, :hours for mean " *
            "hours, :L for efficiency units)."))
    end

    c_ss = ss.policies[:consumption]
    a_ss = ss.policies[:savings]
    D_ss = _normalized_distribution(ss)
    n_ss = any(_ssj_needs_labor, outputs) ?
           labor_policy(individual, grid, income, ss.prices, c_ss) : nothing
    ss_out = Dict{Symbol,T}(o => dot(_ssj_outcome_vector(o, c_ss, a_ss, n_ss, income), D_ss)
                            for o in outputs)
    ss_in = Dict{Symbol,T}(i => ss.prices[i] for i in inputs)

    return HetBlock{T}(name, inputs, outputs, ss, individual, grid, income,
                       T(dx), ss_in, ss_out)
end

function HetBlock(spec::ModelSpec{T}, ss::HASteadyState{T};
                  inputs::AbstractVector{Symbol}=Symbol[:r, :w],
                  outputs::AbstractVector{Symbol}=Symbol[:A],
                  name::Symbol=:household,
                  dx::Real=T(1e-4)) where {T<:AbstractFloat}
    return HetBlock(ss, _hh(spec).individual, _hh(spec).grid, _hh(spec).income;
                    inputs=inputs, outputs=outputs, name=name, dx=dx)
end

"""
    MitBlock{T,E} <: AbstractSSJBlock

Sequence-space block for a non-household [`AbstractAgentSystem`](@ref)
(life-cycle OLG or one-asset continuous time). Maps aggregate price sequences
to aggregate outcomes by a **partial-equilibrium** MIT path (backward policy
sweep, forward histogram / KFE) rather than the fake-news algorithm.

The first-order Jacobian is a one-sided finite difference of that PE path.
`DCEGMSystem` is not accepted — discrete-choice kinks make a sequence-space
Jacobian ill-defined (G-11); use [`dcegm_mit`](@ref).

# Constructors

    MitBlock(spec::ModelSpec, ss::LifeCycleSteadyState; inputs, outputs, name, dx)
    MitBlock(spec::ModelSpec, ss::CTSteadyState; inputs, outputs, name, dx, dt)
    HetBlock(spec, ss)   # same, when `ss` is a life-cycle or CT steady state

`inputs` default to `[:r, :w]`. `outputs` are the same asset / consumption
names as [`HetBlock`](@ref) (`:A`/`:K`/`:C`, …). Labor aggregates are not
produced (both families have exogenous labor).
"""
struct MitBlock{T<:AbstractFloat,E} <: AbstractSSJBlock
    name::Symbol
    inputs::Vector{Symbol}
    outputs::Vector{Symbol}
    dx::T
    ss_inputs::Dict{Symbol,T}
    ss_outputs::Dict{Symbol,T}
    evaluate::E
end

export MitBlock

function MitBlock(evaluate, ::Type{T};
                  inputs::AbstractVector{Symbol},
                  outputs::AbstractVector{Symbol},
                  ss_inputs::AbstractDict{Symbol,T},
                  ss_outputs::Union{Nothing,AbstractDict{Symbol,T}}=nothing,
                  name::Symbol=:mit,
                  dx::Real=T(1e-4)) where {T<:AbstractFloat}
    inputs = collect(Symbol, inputs)
    outputs = collect(Symbol, outputs)
    isempty(inputs) && throw(ArgumentError("MitBlock :$name has no inputs"))
    isempty(outputs) && throw(ArgumentError("MitBlock :$name has no outputs"))
    for i in inputs
        haskey(ss_inputs, i) || throw(ArgumentError(
            "MitBlock :$name is missing the steady-state level of input :$i"))
        i in (:r, :w) || throw(ArgumentError(
            "MitBlock :$name input :$i is not a supported price (use :r and/or :w)"))
    end
    for o in outputs
        kind = get(_HETBLOCK_OUTPUT_KIND, o, nothing)
        kind === nothing && throw(ArgumentError(
            "MitBlock :$name output :$o is not a recognized aggregate. " *
            "Use an asset aggregate (:A, :K, :B, :assets, :a, :savings) or a " *
            "consumption aggregate (:C, :c, :consumption)."))
        kind === :labor && throw(ArgumentError(
            "MitBlock :$name does not produce labor aggregates (exogenous labor)"))
    end
    ss_in = Dict{Symbol,T}(ss_inputs)
    ss_out = if ss_outputs === nothing
        Th0 = 2
        flat = Dict{Symbol,Vector{T}}(i => fill(ss_in[i], Th0) for i in inputs)
        y0 = evaluate(flat, Th0)
        Dict{Symbol,T}(o => T(y0[o][1]) for o in outputs)
    else
        so = Dict{Symbol,T}(ss_outputs)
        for o in outputs
            haskey(so, o) || throw(ArgumentError(
                "MitBlock :$name is missing the steady-state level of output :$o"))
        end
        so
    end
    return MitBlock{T,typeof(evaluate)}(name, inputs, outputs, T(dx), ss_in, ss_out, evaluate)
end

"`HetBlock` / `MitBlock` fallback: only HA / life-cycle / one-asset CT compose."
function HetBlock(spec::ModelSpec, ss; kwargs...)
    throw(ArgumentError(
        "HetBlock(spec, ss) supports HouseholdSystem+HASteadyState, " *
        "LifeCycleSystem+LifeCycleSteadyState, and " *
        "ContinuousHouseholdSystem+CTSteadyState. " *
        "DCEGM is MIT-only (G-11): discrete-choice kinks make fake-news SSJ " *
        "ill-defined; use dcegm_mit."))
end

function MitBlock(spec::ModelSpec, ss; kwargs...)
    throw(ArgumentError(
        "MitBlock(spec, ss) supports LifeCycleSystem+LifeCycleSteadyState and " *
        "ContinuousHouseholdSystem+CTSteadyState. " *
        "DCEGM is MIT-only (G-11): discrete-choice kinks make a sequence-space " *
        "Jacobian ill-defined; use dcegm_mit."))
end

# Disambiguate MitBlock(evaluate, ::Type{T}) vs MitBlock(spec, ss).
# Without this, MitBlock(::ModelSpec, ::Type{<:AbstractFloat}) is ambiguous
# and Aqua.test_ambiguities fails on the LTS empirical job.
function MitBlock(spec::ModelSpec, ::Type{T}; kwargs...) where {T<:AbstractFloat}
    MitBlock(spec, nothing; kwargs...)
end

"Stationary distribution of `ss` as a normalized `N`-vector (column-major, income slowest)."
function _normalized_distribution(ss::HASteadyState{T}) where {T<:AbstractFloat}
    D = vec(copy(ss.distribution))
    s = sum(D)
    s > zero(T) && (D ./= s)
    return D
end

# =============================================================================
# SSJModel — the DAG
# =============================================================================

"""
    SSJModel{T}

A directed acyclic graph of sequence-space blocks, produced by
[`combine_blocks`](@ref).

Fields:
- `name::Symbol` — model label
- `blocks::Vector{AbstractSSJBlock}` — blocks in **topological order**
- `exogenous::Vector{Symbol}` — variables no block produces (unknowns, shocks,
  and calibrated constants), in first-appearance order
- `endogenous::Vector{Symbol}` — variables produced by some block, in evaluation order
- `ss_values::Dict{Symbol,T}` — steady-state level of every variable in the DAG
"""
struct SSJModel{T<:AbstractFloat}
    name::Symbol
    blocks::Vector{AbstractSSJBlock}
    exogenous::Vector{Symbol}
    endogenous::Vector{Symbol}
    ss_values::Dict{Symbol,T}
end

_block_eltype(::SimpleBlock{T}) where {T} = T
_block_eltype(::HetBlock{T}) where {T} = T
_block_eltype(::MitBlock{T}) where {T} = T

"""
    combine_blocks(blocks...; name=:ssj_model, ss_tol=1e-6) -> SSJModel

Assemble sequence-space `blocks` into a DAG.

Blocks are linked by variable name: block `B` depends on block `A` whenever an
output of `A` is an input of `B`. The result is topologically sorted (Kahn's
algorithm, ties broken by the order in which blocks were supplied, so the ordering
is deterministic). A cycle among blocks raises an error naming the blocks
involved — in the sequence-space method a feedback loop is closed through
*unknowns*, which are exogenous to the DAG and solved for by
[`ssj_jacobian`](@ref)/[`ssj_irf`](@ref), so the block graph itself must be acyclic.

# Consistency checks
- Two blocks producing the same variable is an error (the DAG would be ambiguous).
- Where a block consumes a variable another block produces, the two steady-state
  levels must agree to within `ss_tol` in relative terms; a mismatch emits a
  warning naming the variable and both values. An inconsistent steady state
  silently invalidates every Jacobian built on it, so this check is on by default.

# Example

```julia
model = combine_blocks(firm, household, mkt_clearing; name=:krusell_smith)
```
"""
function combine_blocks(blocks::AbstractSSJBlock...;
                        name::Symbol=:ssj_model, ss_tol::Real=1e-6)
    bs = collect(AbstractSSJBlock, blocks)
    isempty(bs) && throw(ArgumentError("combine_blocks requires at least one block"))
    T = _block_eltype(bs[1])
    for b in bs
        _block_eltype(b) === T || throw(ArgumentError(
            "combine_blocks: blocks mix element types ($T and $(_block_eltype(b))); " *
            "all blocks must share one floating-point type"))
    end

    # ── producer map + duplicate-output detection ────────────────────────────
    producer = Dict{Symbol,Int}()
    for (j, b) in enumerate(bs)
        for o in b.outputs
            if haskey(producer, o)
                throw(ArgumentError(
                    "combine_blocks: variable :$o is produced by both block " *
                    ":$(bs[producer[o]].name) and block :$(b.name)"))
            end
            producer[o] = j
        end
    end
    # ── topological sort (Kahn, deterministic tie-break by original index) ───
    n = length(bs)
    indeg = zeros(Int, n)
    children = [Int[] for _ in 1:n]
    for (j, b) in enumerate(bs)
        parents = Set{Int}()
        for i in b.inputs
            haskey(producer, i) || continue
            p = producer[i]
            p == j && throw(ArgumentError(
                "combine_blocks: block :$(b.name) consumes its own output :$i"))
            push!(parents, p)
        end
        for p in parents
            push!(children[p], j)
            indeg[j] += 1
        end
    end

    order = Int[]
    ready = sort([j for j in 1:n if indeg[j] == 0])
    while !isempty(ready)
        j = popfirst!(ready)
        push!(order, j)
        for k in children[j]
            indeg[k] -= 1
            if indeg[k] == 0
                insert!(ready, searchsortedfirst(ready, k), k)
            end
        end
    end
    if length(order) < n
        stuck = [string(bs[j].name) for j in 1:n if indeg[j] > 0]
        throw(ArgumentError(
            "combine_blocks: the block graph has a cycle involving " *
            join(stuck, ", ") * ". Break the loop by declaring the looping " *
            "variable an unknown (it is then exogenous to the DAG and solved for " *
            "by ssj_jacobian/ssj_irf)."))
    end
    sorted = bs[order]

    # ── variable bookkeeping + steady-state consistency ──────────────────────
    ss_values = Dict{Symbol,T}()
    endogenous = Symbol[]
    exogenous = Symbol[]
    tol = T(ss_tol)
    for b in sorted
        for i in b.inputs
            v = b.ss_inputs[i]
            if haskey(ss_values, i)
                ref = ss_values[i]
                scale = max(abs(ref), abs(v), one(T))
                if abs(ref - v) > tol * scale
                    @warn "combine_blocks: inconsistent steady state for :$i" produced=ref consumed=v block=b.name
                end
            else
                ss_values[i] = v
                haskey(producer, i) || push!(exogenous, i)
            end
        end
        for o in b.outputs
            ss_values[o] = b.ss_outputs[o]
            push!(endogenous, o)
        end
    end

    return SSJModel{T}(name, sorted, exogenous, endogenous, ss_values)
end

# =============================================================================
# Block Jacobians
# =============================================================================

"""
    block_jacobian(block, T_horizon) -> Dict{Tuple{Symbol,Symbol},Matrix}

First-order sequence-space Jacobians of one block: `d[(output, input)][t, s]` is
`∂O_t/∂I_s`. Pairs whose derivative is identically zero are omitted.

For a [`SimpleBlock`](@ref) the derivatives are constants obtained by
`ForwardDiff` at the steady state and laid on the corresponding lead/lag
diagonals. For a [`HetBlock`](@ref) each pair is the fake-news Jacobian
(`_ssj_jacobian`). For a [`MitBlock`](@ref) each pair is a one-sided finite
difference of the partial-equilibrium MIT path.
"""
function block_jacobian(b::SimpleBlock{T}, T_horizon::Int) where {T<:AbstractFloat}
    Th = T_horizon
    x_ss = T[b.ss_inputs[i] for (i, _) in b.arg_order]
    Jf = ForwardDiff.jacobian(x -> collect(b.f(x)), x_ss)
    out = Dict{Tuple{Symbol,Symbol},Matrix{T}}()
    for (io, o) in enumerate(b.outputs), i in b.inputs
        M = zeros(T, Th, Th)
        nonzero = false
        for (k, (iname, l)) in enumerate(b.arg_order)
            iname === i || continue
            c = T(Jf[io, k])
            iszero(c) && continue
            nonzero = true
            _add_shift!(M, c, l)
        end
        nonzero && (out[(o, i)] = M)
    end
    return out
end

function block_jacobian(b::HetBlock{T}, T_horizon::Int) where {T<:AbstractFloat}
    out = Dict{Tuple{Symbol,Symbol},Matrix{T}}()
    for o in b.outputs, i in b.inputs
        out[(o, i)] = _ssj_jacobian(b.steady_state, b.individual, b.grid, b.income,
                                    i, o; T_horizon=T_horizon, dx=b.dx)
    end
    return out
end

function block_jacobian(b::MitBlock{T}, T_horizon::Int) where {T<:AbstractFloat}
    Th = T_horizon
    base_paths = Dict{Symbol,Vector{T}}(i => fill(b.ss_inputs[i], Th) for i in b.inputs)
    base = b.evaluate(base_paths, Th)
    out = Dict{Tuple{Symbol,Symbol},Matrix{T}}()
    for o in b.outputs, i in b.inputs
        out[(o, i)] = zeros(T, Th, Th)
    end
    invdx = one(T) / b.dx
    for i in b.inputs, s in 1:Th
        paths = Dict{Symbol,Vector{T}}(j => copy(base_paths[j]) for j in b.inputs)
        paths[i][s] += b.dx
        yp = b.evaluate(paths, Th)
        for o in b.outputs
            @inbounds for t in 1:Th
                out[(o, i)][t, s] = (yp[o][t] - base[o][t]) * invdx
            end
        end
    end
    return out
end

"""
    _add_shift!(M, c, l)

Add `c` to the `l`-th diagonal of `M`, i.e. `M[t, t-l] += c` for every in-range
`t`. `l > 0` is a lag (subdiagonal), `l < 0` a lead (superdiagonal). Entries that
fall outside `1:Th` are dropped — the standard truncation assumption that the
economy sits at its steady state before date 1 and after date `Th`.
"""
function _add_shift!(M::AbstractMatrix{T}, c::T, l::Int) where {T<:AbstractFloat}
    Th = size(M, 1)
    @inbounds for t in 1:Th
        s = t - l
        (1 <= s <= Th) && (M[t, s] += c)
    end
    return M
end

# =============================================================================
# Nonlinear evaluation along a path
# =============================================================================

"""
    _block_evaluate(block, input_paths, T_horizon) -> Dict{Symbol,Vector}

Nonlinear evaluation of one block along full **level** input paths. Values outside
`1:T_horizon` are taken to be at the steady state.
"""
function _block_evaluate(b::SimpleBlock{T}, input_paths::Dict{Symbol,Vector{T}},
                         T_horizon::Int) where {T<:AbstractFloat}
    Th = T_horizon
    out = Dict{Symbol,Vector{T}}(o => zeros(T, Th) for o in b.outputs)
    x = zeros(T, length(b.arg_order))
    @inbounds for t in 1:Th
        for (k, (i, l)) in enumerate(b.arg_order)
            s = t - l
            x[k] = (1 <= s <= Th) ? input_paths[i][s] : b.ss_inputs[i]
        end
        y = b.f(x)
        for (io, o) in enumerate(b.outputs)
            out[o][t] = T(y[io])
        end
    end
    return out
end

function _block_evaluate(b::HetBlock{T}, input_paths::Dict{Symbol,Vector{T}},
                         T_horizon::Int) where {T<:AbstractFloat}
    Th = T_horizon
    ss = b.steady_state
    c_ss = ss.policies[:consumption]
    D = _normalized_distribution(ss)

    # ── Backward pass: policies from the terminal steady state back to date 1 ─
    c_store = Vector{Matrix{T}}(undef, Th)
    a_store = Vector{Matrix{T}}(undef, Th)
    c_next = c_ss
    prices_t = copy(ss.prices)
    for t in Th:-1:1
        for i in b.inputs
            prices_t[i] = input_paths[i][t]
        end
        c_now, a_now = _egm_backward_step(b.individual, b.grid, b.income,
                                          prices_t, c_next)
        c_store[t] = c_now
        a_store[t] = a_now
        c_next = c_now
    end

    # ── Forward pass: distribution from the initial stationary distribution ──
    out = Dict{Symbol,Vector{T}}(o => zeros(T, Th) for o in b.outputs)
    for t in 1:Th
        n_t = nothing
        if any(_ssj_needs_labor, b.outputs)
            # Hours respond to the wage, so they must be evaluated at date t's
            # prices, not the steady-state ones.
            p_t = copy(ss.prices)
            for i in b.inputs
                p_t[i] = input_paths[i][t]
            end
            n_t = labor_policy(b.individual, b.grid, b.income, p_t, c_store[t])
        end
        for o in b.outputs
            out[o][t] = dot(_ssj_outcome_vector(o, c_store[t], a_store[t], n_t, b.income), D)
        end
        if t < Th
            D = _build_transition_matrix(a_store[t], b.grid, b.income) * D
        end
    end
    return out
end

function _block_evaluate(b::MitBlock{T}, input_paths::Dict{Symbol,Vector{T}},
                         T_horizon::Int) where {T<:AbstractFloat}
    return b.evaluate(input_paths, T_horizon)
end

"""
    _dag_evaluate(model, exog_paths, T_horizon) -> Dict{Symbol,Vector}

Nonlinear evaluation of the whole DAG: propagate **level** paths for the exogenous
variables through the blocks in topological order and return level paths for every
variable in the model. Exogenous variables absent from `exog_paths` are held at
their steady-state levels.
"""
function _dag_evaluate(model::SSJModel{T}, exog_paths::Dict{Symbol,Vector{T}},
                       T_horizon::Int) where {T<:AbstractFloat}
    Th = T_horizon
    paths = Dict{Symbol,Vector{T}}()
    for x in model.exogenous
        paths[x] = haskey(exog_paths, x) ? copy(exog_paths[x]) :
                   fill(model.ss_values[x], Th)
    end
    for b in model.blocks
        inp = Dict{Symbol,Vector{T}}()
        for i in b.inputs
            inp[i] = haskey(paths, i) ? paths[i] : fill(b.ss_inputs[i], Th)
        end
        merge!(paths, _block_evaluate(b, inp, Th))
    end
    return paths
end

# =============================================================================
# General-equilibrium Jacobian
# =============================================================================

"""
    SSJGEJacobian{T}

General-equilibrium sequence-space Jacobian of an [`SSJModel`](@ref), produced by
[`ssj_jacobian`](@ref).

Fields:
- `model::SSJModel{T}` — the DAG
- `unknowns::Vector{Symbol}` / `targets::Vector{Symbol}` / `shocks::Vector{Symbol}`
- `T_horizon::Int` — truncation horizon
- `H_U::Matrix{T}` — `(n_targets·T) × (n_unknowns·T)` derivative of the targets
  with respect to the unknowns
- `H_Z::Matrix{T}` — `(n_targets·T) × (n_shocks·T)` derivative of the targets with
  respect to the shocks
- `curlyJ::Dict{Symbol,Dict{Symbol,Matrix{T}}}` — total derivative of every
  variable in the DAG with respect to every unknown and shock
- `H_U_fact::Factorization{T}` — cached LU factorization of `H_U`
"""
struct SSJGEJacobian{T<:AbstractFloat}
    model::SSJModel{T}
    unknowns::Vector{Symbol}
    targets::Vector{Symbol}
    shocks::Vector{Symbol}
    T_horizon::Int
    H_U::Matrix{T}
    H_Z::Matrix{T}
    curlyJ::Dict{Symbol,Dict{Symbol,Matrix{T}}}
    H_U_fact::Factorization{T}
end

"""
    ssj_jacobian(model; unknowns, targets, shocks, T_horizon=300) -> SSJGEJacobian

Assemble the general-equilibrium sequence-space Jacobian of a DAG.

Each block's Jacobians are accumulated forward along the topological order, giving
the **total** derivative `𝒥[v][x] = dv/dx` of every variable `v` with respect to
every exogenous driver `x ∈ unknowns ∪ shocks`. Stacking the rows belonging to
`targets` gives the equilibrium system

    H_U · dU + H_Z · dZ = 0,   dU = −H_U⁻¹ H_Z dZ,

the sequence-space analogue of the implicit function theorem. `H_U` is factorized
once and reused by [`ssj_irf`](@ref).

# Arguments
- `model::SSJModel` — from [`combine_blocks`](@ref)
- `unknowns::Vector{Symbol}` — endogenous aggregates solved for; each must be
  *exogenous to the DAG* (produced by no block)
- `targets::Vector{Symbol}` — equilibrium conditions that must equal their
  steady-state value; each must be produced by some block. `length(targets)` must
  equal `length(unknowns)`
- `shocks::Vector{Symbol}` — exogenous drivers, also exogenous to the DAG
- `T_horizon::Int` — truncation horizon `T` (default 300)
- `target_tol::Real` — targets are equilibrium *residuals*, so each must vanish at
  the linearization point. A steady-state target level above `target_tol` in
  absolute value emits a warning naming the target and its level (default `1e-6`;
  set `Inf` to silence when a target is deliberately centred on a nonzero
  constant). A non-clearing steady state invalidates the whole linearization, and
  nothing downstream can detect it.

# Returns
[`SSJGEJacobian{T}`](@ref).

# Example

```julia
gej = ssj_jacobian(model; unknowns=[:K], targets=[:asset_mkt], shocks=[:Z],
                   T_horizon=100)
```
"""
function ssj_jacobian(model::SSJModel{T};
                      unknowns::AbstractVector{Symbol},
                      targets::AbstractVector{Symbol},
                      shocks::AbstractVector{Symbol},
                      T_horizon::Int=300,
                      target_tol::Real=1e-6) where {T<:AbstractFloat}
    U = collect(Symbol, unknowns)
    G = collect(Symbol, targets)
    Z = collect(Symbol, shocks)
    Th = T_horizon
    Th >= 2 || throw(ArgumentError("T_horizon must be at least 2 (got $Th)"))
    length(U) == length(G) || throw(ArgumentError(
        "ssj_jacobian needs as many targets as unknowns (got $(length(U)) " *
        "unknowns and $(length(G)) targets)"))
    isempty(U) && throw(ArgumentError("ssj_jacobian requires at least one unknown"))
    for u in U
        u in model.exogenous || throw(ArgumentError(
            "unknown :$u is produced by a block; unknowns must be exogenous to the " *
            "DAG (model exogenous variables: $(model.exogenous))"))
    end
    for z in Z
        z in model.exogenous || throw(ArgumentError(
            "shock :$z is produced by a block; shocks must be exogenous to the DAG " *
            "(model exogenous variables: $(model.exogenous))"))
        z in U && throw(ArgumentError("variable :$z is declared both unknown and shock"))
    end
    for g in G
        g in model.endogenous || throw(ArgumentError(
            "target :$g is not produced by any block (model outputs: $(model.endogenous))"))
        # Targets are equilibrium residuals: they must vanish at the linearization
        # point, or every Jacobian below is taken around a point that does not clear.
        lvl = model.ss_values[g]
        if isfinite(target_tol) && abs(lvl) > T(target_tol)
            @warn "ssj_jacobian: target :$g does not vanish in steady state; the " *
                  "model is being linearized around a point that does not clear" level=lvl tol=target_tol
        end
    end

    exog = vcat(U, Z)
    Id = Matrix{T}(I, Th, Th)
    curlyJ = Dict{Symbol,Dict{Symbol,Matrix{T}}}()
    for x in exog
        curlyJ[x] = Dict{Symbol,Matrix{T}}(x => copy(Id))
    end

    for b in model.blocks
        Jb = block_jacobian(b, Th)
        for o in b.outputs
            acc = Dict{Symbol,Matrix{T}}()
            for i in b.inputs
                haskey(Jb, (o, i)) || continue
                haskey(curlyJ, i) || continue          # input is a calibrated constant
                Joi = Jb[(o, i)]
                for (x, M) in curlyJ[i]
                    if haskey(acc, x)
                        mul!(acc[x], Joi, M, one(T), one(T))
                    else
                        acc[x] = Joi * M
                    end
                end
            end
            curlyJ[o] = acc
        end
    end

    nt, nu, nz = length(G), length(U), length(Z)
    H_U = zeros(T, nt * Th, nu * Th)
    H_Z = zeros(T, nt * Th, nz * Th)
    for (it, g) in enumerate(G)
        rows = ((it - 1) * Th + 1):(it * Th)
        gj = curlyJ[g]
        for (iu, u) in enumerate(U)
            haskey(gj, u) || continue
            H_U[rows, ((iu - 1) * Th + 1):(iu * Th)] = gj[u]
        end
        for (iz, z) in enumerate(Z)
            haskey(gj, z) || continue
            H_Z[rows, ((iz - 1) * Th + 1):(iz * Th)] = gj[z]
        end
    end

    fact = lu(H_U; check=false)
    issuccess(fact) || error(
        "ssj_jacobian: the clearing Jacobian H_U is singular. Check that every " *
        "target actually responds to the unknowns (targets=$G, unknowns=$U) and " *
        "that the DAG closes the model.")

    return SSJGEJacobian{T}(model, U, G, Z, Th, H_U, H_Z, curlyJ, fact)
end

# =============================================================================
# Impulse responses
# =============================================================================

"""
    SSJImpulseResponse{T}

Impulse response of an [`SSJModel`](@ref) to an aggregate shock, produced by
[`ssj_irf`](@ref).

Fields:
- `paths::Dict{Symbol,Vector{T}}` — total deviation from steady state of every
  variable in the DAG (first order if `order == 1`, first plus second order if
  `order == 2`)
- `first_order::Dict{Symbol,Vector{T}}` — the first-order component alone
- `correction::Dict{Symbol,Vector{T}}` — the second-order component (empty when
  `order == 1`)
- `shocks::Dict{Symbol,Vector{T}}` — the shock paths supplied
- `unknowns`/`targets` — names carried over from the Jacobian
- `order::Int` — approximation order
- `T_horizon::Int`
- `target_residual::Dict{Symbol,T}` — `max_t |target_t − target_ss|` obtained by
  evaluating the DAG **nonlinearly** at the returned paths, or `NaN` when the
  diagnostic was switched off. This is the honest accuracy measure: it falls from
  `O(σ²)` at first order to `O(σ³)` at second order.
"""
struct SSJImpulseResponse{T<:AbstractFloat}
    paths::Dict{Symbol,Vector{T}}
    first_order::Dict{Symbol,Vector{T}}
    correction::Dict{Symbol,Vector{T}}
    shocks::Dict{Symbol,Vector{T}}
    unknowns::Vector{Symbol}
    targets::Vector{Symbol}
    order::Int
    T_horizon::Int
    target_residual::Dict{Symbol,T}
end

"Total first-order response of every DAG variable to unknown path `du` and shock path `z`."
function _propagate(gej::SSJGEJacobian{T}, du::AbstractVector{T},
                    z::AbstractVector{T}) where {T<:AbstractFloat}
    Th = gej.T_horizon
    driver = Dict{Symbol,Vector{T}}()
    for (iu, u) in enumerate(gej.unknowns)
        driver[u] = du[((iu - 1) * Th + 1):(iu * Th)]
    end
    for (iz, s) in enumerate(gej.shocks)
        driver[s] = z[((iz - 1) * Th + 1):(iz * Th)]
    end
    out = Dict{Symbol,Vector{T}}()
    for (v, jx) in gej.curlyJ
        acc = zeros(T, Th)
        for (x, M) in jx
            haskey(driver, x) || continue
            mul!(acc, M, driver[x], one(T), one(T))
        end
        out[v] = acc
    end
    # Exogenous variables the DAG never touches still deviate by their own path.
    for (x, p) in driver
        haskey(out, x) || (out[x] = copy(p))
    end
    return out
end

"""
    ssj_irf(gej, dZ; order=1, fd_step=1.0, residual=true) -> SSJImpulseResponse

Impulse response of the general-equilibrium sequence-space system to the shock
paths `dZ` (a `Dict` of *deviations* from steady state, one length-`T` vector per
shock; omitted shocks are held at zero).

# First order (`order = 1`)

Solves `dU = −H_U⁻¹ H_Z dZ` and propagates `(dU, dZ)` through the accumulated
total derivatives `𝒥`, giving the linear response of every variable in the DAG.

# Second order (`order = 2`)

Expanding the equilibrium condition `H(U, Z) = 0` to second order in the shock
size gives

    H_U dU¹ + H_Z dZ = 0,
    H_U dU² + ½ D²H[v, v] = 0,     v = (dU¹, dZ),

so `dU² = −½ H_U⁻¹ D²H[v, v]`. The directional second derivative `D²H[v, v]` is
the second-order sequence-space Jacobian tensor **contracted with the first-order
solution path**, which is all the second-order solution ever needs — it is obtained
here from a central difference of two *nonlinear* DAG evaluations,

    D²v[v, v] ≈ (v(x* + h·v) + v(x* − h·v) − 2·v*) / h²,

for every variable at once. This avoids ever materializing the `T³` tensors (at
`T = 300` a single tensor is 2.7·10⁷ entries per input pair) while returning the
identical contraction. The second-order path of variable `v` is then
`𝒥_v·dU² + ½ D²v[v, v]`; by construction the targets are zero to second order.

Because `dU²` scales as the square of the shock size, the second-order solution
collapses onto the first-order one as the shock shrinks.

# Arguments
- `gej::SSJGEJacobian` — from [`ssj_jacobian`](@ref)
- `dZ::Dict{Symbol,<:AbstractVector}` — shock deviation paths, length `T_horizon`
- `order::Int` — `1` (default) or `2`
- `fd_step::Real` — scaling `h` of the central difference used for the second-order
  contraction (default `1.0`, i.e. differencing at the actual shock size). Reduce
  it if the shock is large enough to push the household problem far off its
  steady state; raise it if the shock is so small that the `O(σ²)` term is lost in
  roundoff.
- `residual::Bool` — evaluate the DAG nonlinearly at the returned paths and report
  the maximum absolute target deviation (default `true`; costs one extra nonlinear
  pass per het block).

# Cost

`order=1` needs no nonlinear pass (one more with `residual=true`); `order=2` needs
three (`x* ± h·v` and the steady-state baseline), four with `residual=true`. One
nonlinear pass over a het block costs about the same as one column-sweep of its
fake-news Jacobian.

# Returns
[`SSJImpulseResponse{T}`](@ref).

# Example

```julia
dZ = Dict(:Z => [0.01 * 0.9^(t - 1) for t in 1:100])
irf1 = ssj_irf(gej, dZ)                # first order
irf2 = ssj_irf(gej, dZ; order=2)       # + second-order risk/nonlinearity correction
irf2.target_residual[:asset_mkt] < irf1.target_residual[:asset_mkt]   # true
```
"""
function ssj_irf(gej::SSJGEJacobian{T}, dZ::AbstractDict{Symbol,<:AbstractVector};
                 order::Int=1, fd_step::Real=1.0,
                 residual::Bool=true) where {T<:AbstractFloat}
    Th = gej.T_horizon
    order in (1, 2) || throw(ArgumentError("ssj_irf supports order 1 or 2 (got $order)"))
    fd_step > 0 || throw(ArgumentError("fd_step must be positive (got $fd_step)"))
    for k in keys(dZ)
        k in gej.shocks || throw(ArgumentError(
            "ssj_irf: :$k is not a declared shock (declared: $(gej.shocks))"))
        length(dZ[k]) == Th || throw(ArgumentError(
            "ssj_irf: shock path :$k has length $(length(dZ[k])), expected $Th"))
    end

    shock_paths = Dict{Symbol,Vector{T}}()
    for s in gej.shocks
        shock_paths[s] = haskey(dZ, s) ? T.(collect(dZ[s])) : zeros(T, Th)
    end
    # `reduce(...; init)` keeps the stacked path typed when there are no shocks —
    # a bare `vcat()` returns `Vector{Any}`, which breaks the H_Z product.
    z = reduce(vcat, (shock_paths[s] for s in gej.shocks); init=zeros(T, 0))

    du1 = -(gej.H_U_fact \ (gej.H_Z * z))
    first_order = _propagate(gej, du1, z)

    model = gej.model
    ss = model.ss_values
    correction = Dict{Symbol,Vector{T}}()
    total = first_order

    if order == 2
        h = T(fd_step)
        dev = Dict{Symbol,Vector{T}}()
        for u in gej.unknowns
            dev[u] = first_order[u]
        end
        for (s, p) in shock_paths
            dev[s] = p
        end
        plus = _dag_evaluate(model, Dict{Symbol,Vector{T}}(
            x => ss[x] .+ h .* p for (x, p) in dev), Th)
        minus = _dag_evaluate(model, Dict{Symbol,Vector{T}}(
            x => ss[x] .- h .* p for (x, p) in dev), Th)
        # Baseline of the second difference is the DAG evaluated *nonlinearly* on the
        # steady-state path, not the flat steady-state level: a het block started from
        # a distribution that is only approximately stationary drifts by the steady
        # state's own convergence tolerance, and that drift is common to `plus` and
        # `minus`, so it cancels here and would otherwise bias D².
        base = _dag_evaluate(model, Dict{Symbol,Vector{T}}(), Th)

        # D²v[v,v] for every variable, from the central second difference.
        D2 = Dict{Symbol,Vector{T}}()
        for (v, pv) in plus
            (haskey(minus, v) && haskey(base, v)) || continue
            D2[v] = (pv .+ minus[v] .- 2 .* base[v]) ./ (h * h)
        end

        d2H = reduce(vcat, (D2[g] for g in gej.targets); init=zeros(T, 0))
        du2 = -(gej.H_U_fact \ (T(0.5) .* d2H))
        prop2 = _propagate(gej, du2, zeros(T, length(z)))

        correction = Dict{Symbol,Vector{T}}()
        for (v, p) in prop2
            correction[v] = haskey(D2, v) ? p .+ T(0.5) .* D2[v] : copy(p)
        end
        total = Dict{Symbol,Vector{T}}(v => p .+ get(correction, v, zeros(T, Th))
                                       for (v, p) in first_order)
    end

    # ── Nonlinear market-clearing residual at the returned paths ─────────────
    resid = Dict{Symbol,T}(g => T(NaN) for g in gej.targets)
    if residual
        exog_levels = Dict{Symbol,Vector{T}}()
        for u in gej.unknowns
            exog_levels[u] = ss[u] .+ total[u]
        end
        for (s, p) in shock_paths
            exog_levels[s] = ss[s] .+ p
        end
        nl = _dag_evaluate(model, exog_levels, Th)
        for g in gej.targets
            resid[g] = maximum(abs, nl[g] .- ss[g])
        end
    end

    return SSJImpulseResponse{T}(total, first_order, correction, shock_paths,
                                 copy(gej.unknowns), copy(gej.targets), order, Th,
                                 resid)
end

ssj_irf(gej::SSJGEJacobian{T}, shock::Symbol, path::AbstractVector; kwargs...) where {T} =
    ssj_irf(gej, Dict{Symbol,Vector{T}}(shock => T.(collect(path))); kwargs...)

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, b::SimpleBlock)
    print(io, "SimpleBlock(:", b.name, "): ", join(string.(b.inputs), ", "),
          " → ", join(string.(b.outputs), ", "))
end

function Base.show(io::IO, b::HetBlock)
    print(io, "HetBlock(:", b.name, "): ", join(string.(b.inputs), ", "),
          " → ", join(string.(b.outputs), ", "))
end

function Base.show(io::IO, b::MitBlock)
    print(io, "MitBlock(:", b.name, "): ", join(string.(b.inputs), ", "),
          " → ", join(string.(b.outputs), ", "))
end

function Base.show(io::IO, m::SSJModel)
    print(io, "SSJModel(:", m.name, "): ", length(m.blocks), " blocks, ",
          length(m.exogenous), " exogenous, ", length(m.endogenous), " endogenous")
end

function Base.show(io::IO, g::SSJGEJacobian{T}) where {T}
    print(io, "SSJGEJacobian{$T}: unknowns=", g.unknowns, ", targets=", g.targets,
          ", shocks=", g.shocks, ", T=", g.T_horizon)
end

function Base.show(io::IO, r::SSJImpulseResponse{T}) where {T}
    print(io, "SSJImpulseResponse{$T}: order=", r.order, ", T=", r.T_horizon,
          ", ", length(r.paths), " variables")
end

"""
    report(model::SSJModel)

Print the block graph in topological order with each block's inputs, outputs, and
type, followed by the model's exogenous variables and their steady-state levels.
"""
function report(model::SSJModel{T}) where {T}
    io = stdout
    n = length(model.blocks)
    data = Matrix{Any}(undef, n, 4)
    for (i, b) in enumerate(model.blocks)
        data[i, 1] = i
        data[i, 2] = string(b.name)
        data[i, 3] = b isa HetBlock ? "HetBlock" :
                     b isa MitBlock ? "MitBlock" : "SimpleBlock"
        data[i, 4] = join(string.(b.inputs), ", ") * " → " * join(string.(b.outputs), ", ")
    end
    _pretty_table(io, data;
        title="Sequence-Space DAG: $(model.name)",
        column_labels=["#", "Block", "Type", "Inputs → Outputs"],
        alignment=[:r, :l, :l, :l])

    exo = model.exogenous
    if !isempty(exo)
        ex_data = Matrix{Any}(undef, length(exo), 2)
        for (i, x) in enumerate(exo)
            ex_data[i, 1] = string(x)
            ex_data[i, 2] = _fmt(model.ss_values[x]; digits=6)
        end
        _pretty_table(io, ex_data;
            title="Exogenous Variables (steady state)",
            column_labels=["Variable", "Level"],
            alignment=[:l, :r])
    end
    return nothing
end

"""
    report(gej::SSJGEJacobian)

Print the general-equilibrium system dimensions, the unknowns/targets/shocks, and
the norm of each `(target, unknown)` and `(target, shock)` Jacobian block.
"""
function report(gej::SSJGEJacobian{T}) where {T}
    io = stdout
    Th = gej.T_horizon
    head = Any[
        "Model"           string(gej.model.name);
        "Horizon T"       Th;
        "Unknowns"        join(string.(gej.unknowns), ", ");
        "Targets"         join(string.(gej.targets), ", ");
        "Shocks"          isempty(gej.shocks) ? "—" : join(string.(gej.shocks), ", ");
        "size(H_U)"       string(size(gej.H_U, 1), "×", size(gej.H_U, 2));
        "size(H_Z)"       string(size(gej.H_Z, 1), "×", size(gej.H_Z, 2))
    ]
    _pretty_table(io, head;
        title="Sequence-Space GE Jacobian",
        column_labels=["", "Value"],
        alignment=[:l, :r])

    drivers = vcat(gej.unknowns, gej.shocks)
    if !isempty(drivers)
        data = Matrix{Any}(undef, length(gej.targets) * length(drivers), 4)
        row = 0
        for g in gej.targets, x in drivers
            row += 1
            M = get(gej.curlyJ[g], x, nothing)
            data[row, 1] = string(g)
            data[row, 2] = string(x)
            data[row, 3] = x in gej.unknowns ? "unknown" : "shock"
            data[row, 4] = M === nothing ? "0" : _fmt(opnorm(M, 1); digits=6)
        end
        _pretty_table(io, data;
            title="Target Sensitivities (‖∂target/∂driver‖₁)",
            column_labels=["Target", "Driver", "Kind", "Norm"],
            alignment=[:l, :l, :l, :r])
    end
    return nothing
end

"""
    report(r::SSJImpulseResponse)

Print the approximation order, the peak response of each unknown, the relative
size of the second-order correction, and the nonlinear market-clearing residual.
"""
function report(r::SSJImpulseResponse{T}) where {T}
    io = stdout
    head = Any[
        "Order"           r.order;
        "Horizon T"       r.T_horizon;
        "Unknowns"        join(string.(r.unknowns), ", ");
        "Targets"         join(string.(r.targets), ", ")
    ]
    _pretty_table(io, head;
        title="Sequence-Space Impulse Response",
        column_labels=["", "Value"],
        alignment=[:l, :r])

    vars = sort(collect(keys(r.paths)); by=string)
    data = Matrix{Any}(undef, length(vars), 4)
    for (i, v) in enumerate(vars)
        p = r.paths[v]
        f = r.first_order[v]
        c = get(r.correction, v, nothing)
        data[i, 1] = string(v)
        data[i, 2] = _fmt(p[1]; digits=6)
        data[i, 3] = _fmt(maximum(abs, p); digits=6)
        data[i, 4] = if c === nothing
            "—"
        else
            den = maximum(abs, f)
            den > zero(T) ? _fmt(maximum(abs, c) / den; digits=6) : "—"
        end
    end
    _pretty_table(io, data;
        title="Responses",
        column_labels=["Variable", "Impact", "Peak |·|", "2nd/1st"],
        alignment=[:l, :r, :r, :r])

    res_data = Matrix{Any}(undef, length(r.targets), 2)
    for (i, g) in enumerate(r.targets)
        res_data[i, 1] = string(g)
        res_data[i, 2] = _fmt(r.target_residual[g]; digits=6)
    end
    _pretty_table(io, res_data;
        title="Nonlinear Target Residual (max |deviation|)",
        column_labels=["Target", "Residual"],
        alignment=[:l, :r])
    return nothing
end
