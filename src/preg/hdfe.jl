# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
High-dimensional fixed-effect (HDFE) absorption by the method of alternating
projections (Guimarães & Portugal 2010; Correia 2016 `reghdfe`; Gaure 2013 `lfe`).

Given `D` categorical dimensions, the within transformation for the full dummy
design `D` is the orthogonal projection `M = I - D(D'D)⁻D'`. Materializing `D`
is infeasible once the level counts are large (worker × firm, firm × year ×
product). Alternating projections compute `M v` by cycling the *single-dimension*
demeaning operators `M_d = I - P_d`, each of which costs one O(n) pass:

```
v ← M_D ⋯ M_2 M_1 v      (repeat until v stops moving)
```

By von Neumann–Halperin the cycle converges to the projection onto
`⋂_d range(M_d) = range(M)`, i.e. exactly the multi-way within transformation,
without ever forming a dummy.
"""

using LinearAlgebra, Statistics

# =============================================================================
# Level coding and group → index maps
# =============================================================================

"""
    _hdfe_codes(ids) -> (codes, n_levels)

Dense-rank an arbitrary id vector into consecutive codes `1:G` in order of first
appearance. `NaN` is rejected up front: `Dict` keys compare with `isequal`, under
which `isequal(NaN, NaN)` is `true`, so a missing-coded `NaN` would otherwise be
silently absorbed as a legitimate level.
"""
function _hdfe_codes(ids::AbstractVector)
    if eltype(ids) <: AbstractFloat
        any(isnan, ids) && throw(ArgumentError(
            "absorb dimension contains NaN — fixed-effect levels must be fully observed"))
    end
    lookup = Dict{eltype(ids),Int}()
    codes = Vector{Int}(undef, length(ids))
    g = 0
    @inbounds for i in eachindex(ids)
        v = ids[i]
        c = get(lookup, v, 0)
        if c == 0
            g += 1
            lookup[v] = g
            c = g
        end
        codes[i] = c
    end
    codes, g
end

"""
    _hdfe_index_sets(codes, G) -> Vector{Vector{Int}}

Group → observation-indices map for dense codes `1:G`, built in two O(n) passes
(count, then fill). Same contract as `_group_index_map` — buckets fill in
ascending index order — specialized to dense integer codes so the map is a
`Vector` of exactly-sized `Vector{Int}` rather than a `Dict`. Built once and
reused by every sweep of the projection loop, which is the whole point: the inner
loop must not rescan the id vector.
"""
function _hdfe_index_sets(codes::Vector{Int}, G::Int)
    counts = zeros(Int, G)
    @inbounds for c in codes
        counts[c] += 1
    end
    sets = [Vector{Int}(undef, counts[g]) for g in 1:G]
    fill!(counts, 0)
    @inbounds for i in eachindex(codes)
        c = codes[i]
        counts[c] += 1
        sets[c][counts[c]] = i
    end
    sets
end

# =============================================================================
# One alternating-projections sweep
# =============================================================================

"""
    _hdfe_sweep!(V, idxsets, buf)

One full cycle `M_D ⋯ M_1` applied in place to every column of `V`: for each
dimension, subtract each level's mean. `buf` is a scratch vector of length
`size(V, 2)` holding the current level's column means.
"""
function _hdfe_sweep!(V::Matrix{T}, idxsets::Vector{Vector{Vector{Int}}},
                      buf::Vector{T}) where {T<:AbstractFloat}
    ncol = size(V, 2)
    @inbounds for sets in idxsets           # each FE dimension
        for idx in sets                     # each level of that dimension
            m = length(idx)
            m == 0 && continue
            for j in 1:ncol
                s = zero(T)
                for i in idx
                    s += V[i, j]
                end
                buf[j] = s / m
            end
            for j in 1:ncol
                b = buf[j]
                for i in idx
                    V[i, j] -= b
                end
            end
        end
    end
    V
end

# =============================================================================
# Degrees of freedom absorbed
# =============================================================================

"""Union-find root with path compression."""
function _hdfe_find(parent::Vector{Int}, x::Int)
    r = x
    @inbounds while parent[r] != r
        r = parent[r]
    end
    @inbounds while parent[x] != r
        nxt = parent[x]
        parent[x] = r
        x = nxt
    end
    r
end

"""
    _hdfe_components(codes1, G1, codes2, G2) -> Int

Number of connected components of the bipartite graph whose vertices are the
levels of the first two FE dimensions and whose edges are the observed
`(level₁, level₂)` pairs (Abowd, Creecy & Kramarz 2002 "mobility groups").

The two-way dummy design has rank `G₁ + G₂ - C`: each component contributes
exactly one collinearity (its own level-shift between the two dimensions), not
one overall.
"""
function _hdfe_components(codes1::Vector{Int}, G1::Int, codes2::Vector{Int}, G2::Int)
    parent = collect(1:(G1 + G2))
    @inbounds for i in eachindex(codes1)
        a = _hdfe_find(parent, codes1[i])
        b = _hdfe_find(parent, G1 + codes2[i])
        a != b && (parent[a] = b)
    end
    roots = Set{Int}()
    @inbounds for v in 1:(G1 + G2)
        push!(roots, _hdfe_find(parent, v))
    end
    length(roots)
end

"""
    _hdfe_dof(codes, levels) -> (n_absorbed, n_components, marginal)

Count the fixed-effect parameters absorbed by the within transformation, i.e. the
rank of the dummy design.

- **One dimension** — `G₁` exactly.
- **Two dimensions** — `G₁ + G₂ - C` exactly, where `C` is the number of
  connected components (`_hdfe_components`). On a balanced panel `C = 1`, giving
  the familiar `N + T - 1`.
- **Three or more** — `G₁ + G₂ - C + Σ_{d≥3}(G_d - 1)`. No closed form for the
  rank exists beyond two dimensions, so each further dimension is charged one
  collinearity (with the constant, already absorbed). This is an **upper bound**
  on the true rank: any additional collinearity among dimensions 3+ is missed, so
  `n_absorbed` is never understated and the resulting dof is never overstated —
  the small-sample correction errs conservative (SEs too large, not too small).

`marginal[d]` is dimension `d`'s own contribution to the total, attributed in
absorption order. The total is order-invariant for `D ≤ 2` (`C` is symmetric in
the two dimensions); for `D ≥ 3` both the total and the split depend on which
dimensions are listed first. Estimated **coefficients** are unaffected either
way — they depend only on the range of the dummy design, not the bookkeeping.
"""
function _hdfe_dof(codes::Vector{Vector{Int}}, levels::Vector{Int})
    D = length(levels)
    D == 0 && return (n_absorbed=0, n_components=0, marginal=Int[])
    D == 1 && return (n_absorbed=levels[1], n_components=1, marginal=[levels[1]])

    C = _hdfe_components(codes[1], levels[1], codes[2], levels[2])
    marginal = zeros(Int, D)
    marginal[1] = levels[1]
    marginal[2] = levels[2] - C
    for d in 3:D
        marginal[d] = levels[d] - 1
    end
    (n_absorbed=sum(marginal), n_components=C, marginal=marginal)
end

"""
    _hdfe_nested_in(codes, G, cluster_codes) -> Bool

`true` when every level of the FE dimension lies entirely inside one cluster
(the FE is *nested* within the clustering dimension).

Nested fixed effects must not be charged against the cluster-robust dof: the
`G/(G-1)` cluster factor already accounts for them. Entity FE clustered on entity
— the default panel setup — is the canonical case, and skipping this check is the
classic way to get HDFE standard errors wrong.
"""
function _hdfe_nested_in(codes::Vector{Int}, G::Int, cluster_codes::Vector{Int})
    first_seen = zeros(Int, G)
    @inbounds for i in eachindex(codes)
        g = codes[i]
        if first_seen[g] == 0
            first_seen[g] = cluster_codes[i]
        elseif first_seen[g] != cluster_codes[i]
            return false
        end
    end
    true
end

# =============================================================================
# absorb_fe — public entry point
# =============================================================================

"""
    absorb_fe(y, X, fe_groups; tol=1e-8, maxiter=1000, accel=true)

Absorb `D` high-dimensional fixed-effect dimensions from `y` and `X` by the
**method of alternating projections** (Guimarães & Portugal 2010; Correia 2016),
without ever forming the dummy design matrix.

Each dimension's demeaning operator `M_d = I - P_d` is an orthogonal projection;
cycling them converges to the projection onto the intersection of their ranges,
which is exactly the multi-way within transformation. OLS of the returned `y` on
the returned `X` is therefore the within-all-FE estimator.

# Arguments
- `y::AbstractVector` — outcome (length `n`)
- `X::AbstractMatrix` — regressors (`n × k`); may be `n × 0`
- `fe_groups::AbstractVector{<:AbstractVector}` — one id vector of length `n` per
  FE dimension. Ids may be of any type (integers, floats, strings); they are
  dense-ranked internally.

# Keyword Arguments
- `tol::Real=1e-8` — convergence tolerance on the relative movement of `[y X]`
  per iteration, measured against the norm of the column-standardized data
- `maxiter::Int=1000` — maximum iterations (an iteration is one sweep without
  acceleration, two with)
- `accel::Bool=true` — Irons–Tuck (vector Aitken Δ²) extrapolation

# Returns
A `NamedTuple`:

| Field | Type | Description |
|---|---|---|
| `y` | `Vector{T}` | residualized outcome |
| `X` | `Matrix{T}` | residualized regressors |
| `n_absorbed` | `Int` | absorbed FE parameters (rank of the dummy design) |
| `n_levels` | `Vector{Int}` | levels per dimension |
| `n_components` | `Int` | connected components of the first two dimensions |
| `marginal` | `Vector{Int}` | per-dimension contribution to `n_absorbed` |
| `converged` | `Bool` | tolerance reached before `maxiter` |
| `iterations` | `Int` | iterations run |
| `sweeps` | `Int` | total demeaning sweeps |
| `change` | `T` | final relative movement |

# Acceleration
Plain alternating projections converge linearly at a rate set by the angle
between the dummy subspaces, which is punishing when the dimensions are weakly
connected (sparse worker–firm mobility). With `accel=true`, each iteration takes
two sweeps `y = T(x)`, `z = T(y)` and extrapolates

```math
x_{new} = z - \\frac{\\langle z-y,\\; z-2y+x\\rangle}{\\langle z-2y+x,\\; z-2y+x\\rangle}\\,(z-y)
```

which is exact for a single geometric mode.

Two properties make this safe. First, the extrapolation weights on `(x, y, z)`
sum to one, so `x_new` stays in the affine set `x₀ + span(D)` — the accelerated
iterate is still `x₀` minus something in the fixed-effect span, i.e. still a
*valid* residualization no matter how badly it overshoots. Only orthogonality to
`D` (the convergence criterion) is ever at stake. Second, the true answer `M x₀`
is the minimum-norm point of that affine set, so `‖·‖` is an exact merit
function: the extrapolated point is accepted only when its norm does not exceed
the un-accelerated one. Acceleration can therefore never make an iterate worse.

# Examples
```julia
n = 500
firm  = rand(1:40, n)
year  = rand(1:10, n)
X = randn(n, 2)
y = X * [1.5, -0.8] + randn(n)
a = absorb_fe(y, X, [firm, year])
beta = a.X \\ a.y            # within-firm-and-year estimator
a.n_absorbed                 # 40 + 10 - (components)
```

# References
- Guimarães, P. & Portugal, P. (2010). A simple feasible procedure to fit models
  with high-dimensional fixed effects. *The Stata Journal* 10(4), 628-649.
- Correia, S. (2016). *A Feasible Estimator for Linear Models with Multi-Way
  Fixed Effects*. `reghdfe`.
- Gaure, S. (2013). OLS with multiple high dimensional category variables.
  *Computational Statistics & Data Analysis* 66, 8-18.
- Abowd, J. M., Creecy, R. H. & Kramarz, F. (2002). Computing person and firm
  effects using linked longitudinal employer-employee data. *Cornell/US Census
  Bureau Technical Paper* TP-2002-06.
"""
function absorb_fe(y::AbstractVector{T}, X::AbstractMatrix{T},
                   fe_groups::AbstractVector{<:AbstractVector};
                   tol::Real=1e-8, maxiter::Int=1000,
                   accel::Bool=true) where {T<:AbstractFloat}
    n = length(y)
    size(X, 1) == n || throw(DimensionMismatch(
        "X has $(size(X, 1)) rows but y has $n elements"))
    isempty(fe_groups) && throw(ArgumentError(
        "absorb_fe needs at least one fixed-effect dimension"))
    maxiter >= 1 || throw(ArgumentError("maxiter must be >= 1, got $maxiter"))
    tol > 0 || throw(ArgumentError("tol must be positive, got $tol"))

    # ---- Dense-rank each dimension, build its group → index map once ----
    D = length(fe_groups)
    codes = Vector{Vector{Int}}(undef, D)
    levels = Vector{Int}(undef, D)
    idxsets = Vector{Vector{Vector{Int}}}(undef, D)
    for d in 1:D
        length(fe_groups[d]) == n || throw(DimensionMismatch(
            "FE dimension $d has $(length(fe_groups[d])) ids but y has $n elements"))
        c, G = _hdfe_codes(fe_groups[d])
        G == n && throw(ArgumentError(
            "FE dimension $d has one level per observation ($n levels) — absorbing it " *
            "would remove all variation. Pass a categorical dimension, not a continuous one."))
        codes[d] = c
        levels[d] = G
        idxsets[d] = _hdfe_index_sets(c, G)
    end

    # ---- Column-standardize so `tol` is scale-free ----
    # Demeaning acts on each column independently, so scaling commutes exactly
    # with absorption; only the (joint) extrapolation coefficient sees it.
    k = size(X, 2)
    V = Matrix{T}(undef, n, k + 1)
    @views V[:, 1] .= y
    @views V[:, 2:end] .= X
    scales = ones(T, k + 1)
    for j in 1:(k + 1)
        nj = norm(@view V[:, j])
        if nj > zero(T) && isfinite(nj)
            scales[j] = nj
            @views V[:, j] ./= nj
        end
    end

    ref = max(norm(V), sqrt(eps(T)))
    buf = Vector{T}(undef, k + 1)
    tolT = T(tol)

    converged = false
    iters = 0
    sweeps = 0
    change = T(Inf)

    Vprev = similar(V)
    Vy = accel ? similar(V) : V          # only allocated when accelerating

    for iter in 1:maxiter
        iters = iter
        copyto!(Vprev, V)
        _hdfe_sweep!(V, idxsets, buf); sweeps += 1      # V = T(x)

        # Convergence is measured on the FIRST sweep, ‖T(x) - x‖: that is exactly
        # the movement the projections still find, i.e. the residual
        # non-orthogonality to the dummy span. Measuring the post-extrapolation
        # step instead would only bound this quantity through its square.
        delta = zero(T)
        @inbounds for i in eachindex(V)
            d = V[i] - Vprev[i]
            delta += d * d
        end
        change = sqrt(delta) / ref

        if accel
            copyto!(Vy, V)
            _hdfe_sweep!(V, idxsets, buf); sweeps += 1  # V = T(y) = z

            # Irons-Tuck: Δ = z - y, Δ² = z - 2y + x
            den = zero(T)
            num = zero(T)
            @inbounds for i in eachindex(V)
                d1 = V[i] - Vy[i]
                d2 = V[i] - 2 * Vy[i] + Vprev[i]
                num += d1 * d2
                den += d2 * d2
            end
            if den > zero(T) && isfinite(num) && isfinite(den)
                c = num / den
                # Merit function: `M x₀` is the minimum-norm point of the affine
                # set x₀ + span(D), which every iterate (accelerated or not)
                # belongs to. Accept the extrapolation only if it lowers the norm.
                cand_sq = zero(T)
                @inbounds for i in eachindex(V)
                    v = V[i] - c * (V[i] - Vy[i])
                    cand_sq += v * v
                end
                if isfinite(cand_sq) && cand_sq <= dot(V, V)
                    @inbounds for i in eachindex(V)
                        V[i] -= c * (V[i] - Vy[i])
                    end
                end
            end
        end

        if change <= tolT
            converged = true
            break
        end
    end

    # ---- Undo the standardization ----
    for j in 1:(k + 1)
        @views V[:, j] .*= scales[j]
    end

    dof = _hdfe_dof(codes, levels)

    (y = V[:, 1],
     X = V[:, 2:end],
     n_absorbed = dof.n_absorbed,
     n_levels = levels,
     n_components = dof.n_components,
     marginal = dof.marginal,
     converged = converged,
     iterations = iters,
     sweeps = sweeps,
     change = change)
end

# Promote mixed / non-float inputs to a common float type.
function absorb_fe(y::AbstractVector, X::AbstractMatrix,
                   fe_groups::AbstractVector{<:AbstractVector}; kwargs...)
    T = float(promote_type(eltype(y), eltype(X)))
    absorb_fe(Vector{T}(y), Matrix{T}(X), fe_groups; kwargs...)
end

# =============================================================================
# PanelData plumbing
# =============================================================================

"""
    _hdfe_dimension(pd, dim) -> Vector

Resolve an `absorb` dimension name against a `PanelData` container. A matching
variable column wins; otherwise the reserved panel-index aliases apply:
`:entity`/`:id`/`:unit`/`:group` → entity ids, `:time`/`:period` → time ids,
`:cohort` → cohort ids.
"""
function _hdfe_dimension(pd::PanelData{T}, dim::Symbol) where {T}
    idx = findfirst(==(String(dim)), pd.varnames)
    if idx !== nothing
        col = pd.data[:, idx]
        any(isnan, col) && throw(ArgumentError(
            "absorb dimension :$dim contains NaN — fixed-effect levels must be fully observed"))
        return col
    end
    dim in (:entity, :id, :unit, :group) && return pd.group_id
    dim in (:time, :period) && return pd.time_id
    if dim === :cohort
        pd.cohort_id === nothing && throw(ArgumentError(
            "absorb=:cohort requires a cohort-indexed panel — build it with " *
            "xtset(df, :group, :time; cohort=:col)"))
        return pd.cohort_id
    end
    throw(ArgumentError(
        "absorb dimension :$dim not found. Available variables: $(pd.varnames). " *
        "Reserved names: :entity (:id/:unit/:group), :time (:period), :cohort"))
end

"""
    _hdfe_cluster_absorbed(fe_ids, marginal, clusters) -> Int

Absorbed parameters charged against the **cluster-robust** dof: the sum of the
marginal contributions of the FE dimensions *not* nested within the clustering
variable. Entity FE clustered on entity contributes 0, reproducing the standard
one-way within-estimator correction exactly.
"""
function _hdfe_cluster_absorbed(fe_ids::AbstractVector{<:AbstractVector},
                                marginal::Vector{Int}, clusters::AbstractVector)
    ccodes, _ = _hdfe_codes(clusters)
    total = 0
    for (d, ids) in enumerate(fe_ids)
        c, G = _hdfe_codes(ids)
        _hdfe_nested_in(c, G, ccodes) || (total += marginal[d])
    end
    total
end
