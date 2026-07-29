# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# MacroEconometricModels.jl — Adaptive (curvature-equidistributed) HA asset grids
#
# The Young (2010) histogram and the EGM policy are both piecewise linear on the asset grid,
# so their approximation error on a cell of width h scales like h²·|p''|. Equidistributing
# q = |p''|^{1/2} therefore equalizes the *error* per cell rather than the width per cell —
# the classical de Boor equidistribution principle.
#
# Two details are what make it work on an actual asset grid, and both were measured against a
# high-resolution (n = 1600) Krusell-Smith reference:
#
#  1. The uniform component must be blended as a NORMALIZED measure, not as an additive floor
#     inside the monitor. An asset grid spans [0, a_max] with a_max chosen so the ergodic set
#     fits — on the shipped Krusell-Smith calibration 99% of the mass sits below a = 214 of a
#     1000-wide domain. A constant floor is a uniform density over that whole domain, so it
#     hands ~78% of the nodes to the empty tail. Writing M = (1-λ)/L + λ·q/∫q makes λ exactly
#     the share of nodes placed by curvature.
#
#  2. The monitor must be capped. The stationary distribution has an ATOM at the borrowing
#     constraint, and a histogram atom in a cell of width w reports |p''| ~ 1/w² — a
#     discretization artifact, not density curvature. Uncapped it packs the whole grid into
#     the bottom 3e-4 of the domain and is 17x WORSE than the shipped `:geometric` grid
#     (|Δr| = 2.9e-4 vs 2.0e-5); capped at 3x the median it is BETTER (1.7e-5).
#
# References:
#   de Boor (1973), Good approximation by splines with variable knots II
#   Huang & Russell (2011), Adaptive Moving Mesh Methods
#   Brumm & Scheidegger (2017), Using adaptive sparse grids to solve high-dimensional
#     dynamic models, Econometrica 85(5)

# =============================================================================
# Internal helpers
# =============================================================================

# Trapezoidal cell widths attached to each node of a (possibly nonuniform) grid.
function _node_widths(x::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(x)
    w = zeros(T, n)
    w[1] = (x[2] - x[1]) / 2
    @inbounds for i in 2:(n - 1)
        w[i] = (x[i + 1] - x[i - 1]) / 2
    end
    w[n] = (x[n] - x[n - 1]) / 2
    return w
end

# Three-point second derivative on a nonuniform grid:
#   f''(x_i) ≈ 2[h₁ f_{i+1} − (h₁+h₂) f_i + h₂ f_{i−1}] / (h₁ h₂ (h₁+h₂)),
# which collapses to the usual (f_{i+1} − 2f_i + f_{i−1})/h² when h₁ = h₂ = h.
# Endpoints copy their neighbour (a one-sided estimate would be noisier than it is worth
# for a monitor function).
function _second_derivative_nonuniform(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(x)
    d2 = zeros(T, n)
    @inbounds for i in 2:(n - 1)
        h1 = x[i] - x[i - 1]
        h2 = x[i + 1] - x[i]
        d2[i] = 2 * (h1 * y[i + 1] - (h1 + h2) * y[i] + h2 * y[i - 1]) / (h1 * h2 * (h1 + h2))
    end
    d2[1] = d2[2]
    d2[n] = d2[n - 1]
    return d2
end

# One pass of a [1/4, 1/2, 1/4] filter. Smoothing the MONITOR (not the density) is what keeps
# histogram sampling noise from dictating node placement.
function _smooth3(v::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(v)
    n < 3 && return collect(T, v)
    out = zeros(T, n)
    out[1] = (v[1] + v[2]) / 2
    @inbounds for i in 2:(n - 1)
        out[i] = (v[i - 1] + 2 * v[i] + v[i + 1]) / 4
    end
    out[n] = (v[n - 1] + v[n]) / 2
    return out
end

# Marginal mass over asset dimension `dim`, from a distribution stored either flat or shaped.
function _asset_marginal(distribution::AbstractArray, grid::HAGrid{T}, dim::Int) where {T<:AbstractFloat}
    dims = vcat(grid.n_points, grid.n_income)
    expected = prod(dims)
    length(distribution) == expected || throw(ArgumentError(
        "adapt_ha_grid: distribution has $(length(distribution)) entries but the grid " *
        "implies $expected (n_points = $(grid.n_points), n_income = $(grid.n_income))"))
    d = reshape(collect(T, vec(distribution)), dims...)
    others = Tuple(k for k in 1:length(dims) if k != dim)
    m = vec(sum(d; dims=others))
    total = sum(m)
    total > zero(T) || throw(ArgumentError("adapt_ha_grid: the distribution has zero total mass"))
    return m ./ total
end

# =============================================================================
# Adaptive asset grid
# =============================================================================

"""
    adaptive_asset_grid(nodes, mass; n=length(nodes), curvature=0.9, monitor_cap=3.0,
                        smoothing=2, is_density=false) → Vector{T}

Re-place `n` asset grid points on `[nodes[1], nodes[end]]` so that they concentrate where the
stationary density has curvature.

Both the Young (2010) histogram and the EGM policy are piecewise linear on the asset grid, so
their error on a cell of width `h` scales like `h²·|p''|`; equidistributing
``q = |p''|^{1/2}`` equalizes the *error* per cell instead of the *width* per cell
(de Boor 1973). The monitor blends that with a uniform measure,

```math
M(a) = (1 - \\lambda)\\,\\frac{1}{a_{\\max} - a_{\\min}} + \\lambda\\,\\frac{q(a)}{\\int q},
```

and the new nodes are placed at equal increments of ``\\int M``. Both components integrate to
1 over the domain, so ``\\lambda`` = `curvature` is **exactly the share of nodes allocated by
curvature** — the remaining `1 - curvature` share is spread uniformly. This normalization
matters: an asset grid is mostly empty tail (on the shipped Krusell-Smith calibration 99% of
the mass sits below `a = 214` of a 1000-wide domain), and an additive floor would hand ~78%
of the nodes to that tail.

# Arguments
- `nodes::AbstractVector{T}` — the current (strictly increasing) asset grid
- `mass::AbstractVector{T}` — probability **mass** at each node, i.e. the Young-histogram
  marginal. Pass `is_density=true` to supply a density instead.

# Keyword Arguments
- `n::Int=length(nodes)` — number of points in the new grid
- `curvature::Real=0.9` — share `λ ∈ [0,1]` of nodes placed by curvature. `curvature=0`
  returns an exactly **uniform** grid (the regression baseline); `1` is pure de Boor
  equidistribution, which leaves no floor in flat regions and can open very wide tail cells.
- `monitor_cap::Real=3.0` — cap on `q` at this multiple of its positive median; `Inf`
  disables. **Do not disable it on a model with a borrowing constraint**: the stationary
  distribution has an atom there, and a histogram atom in a cell of width `w` reports
  `|p''| ~ 1/w²` — an artifact, not curvature. Uncapped it packs the grid into the bottom
  `3e-4` of the domain.
- `smoothing::Int=2` — passes of a `[1/4, 1/2, 1/4]` filter applied to `q`, so histogram
  sampling noise does not dictate node placement
- `is_density::Bool=false` — treat `mass` as a density already (skip the division by the
  trapezoidal cell widths)

# Returns
A strictly increasing `Vector{T}` of length `n` with the same endpoints as `nodes`.

# Examples
```julia
grid_new = adaptive_asset_grid(ss.grid.grids[1], vec(sum(ss.distribution; dims=2)))
```

See also [`adapt_ha_grid`](@ref), [`HAGrid`](@ref).
"""
function adaptive_asset_grid(nodes::AbstractVector{T}, mass::AbstractVector{T};
                             n::Int=length(nodes),
                             curvature::Real=0.9,
                             monitor_cap::Real=3.0,
                             smoothing::Int=2,
                             is_density::Bool=false) where {T<:AbstractFloat}
    n_old = length(nodes)
    n_old >= 3 || throw(ArgumentError("adaptive_asset_grid: need at least 3 input nodes, got $n_old"))
    length(mass) == n_old || throw(ArgumentError(
        "adaptive_asset_grid: mass has $(length(mass)) entries but nodes has $n_old"))
    n >= 3 || throw(ArgumentError("adaptive_asset_grid: need at least 3 output nodes, got $n"))
    0 <= curvature <= 1 || throw(ArgumentError(
        "adaptive_asset_grid: curvature is a share and must lie in [0, 1], got $curvature"))
    monitor_cap > 0 || throw(ArgumentError("adaptive_asset_grid: monitor_cap must be > 0, got $monitor_cap"))
    smoothing >= 0 || throw(ArgumentError("adaptive_asset_grid: smoothing must be >= 0, got $smoothing"))
    all(diff(nodes) .> 0) || throw(ArgumentError("adaptive_asset_grid: nodes must be strictly increasing"))

    a_min = T(nodes[1])
    a_max = T(nodes[end])
    L = a_max - a_min
    x = collect(T, nodes)
    w = _node_widths(x)

    # Mass → density. On a nonuniform grid the histogram mass is already integrated over the
    # cell, so dividing by the cell width is what makes `p` comparable across the domain.
    p = is_density ? collect(T, mass) : collect(T, mass) ./ w

    # de Boor monitor q = |p''|^{1/2}, capped against the borrowing-constraint atom.
    q = sqrt.(abs.(_second_derivative_nonuniform(x, p)))
    if isfinite(monitor_cap)
        pos = filter(>(zero(T)), q)
        if !isempty(pos)
            thr = T(monitor_cap) * median(pos)
            thr > zero(T) && (q = min.(q, thr))
        end
    end
    for _ in 1:smoothing
        q = _smooth3(q)
    end

    Iq = sum(q .* w)
    lam = T(curvature)
    mon = if Iq > zero(T) && isfinite(Iq) && lam > 0
        (one(T) - lam) / L .+ lam .* q ./ Iq
    else
        fill(one(T) / L, n_old)
    end
    all(isfinite, mon) || (mon = fill(one(T) / L, n_old))

    # Cumulative ∫M (trapezoid). M > 0 wherever λ < 1 ⇒ C is strictly increasing.
    C = zeros(T, n_old)
    @inbounds for i in 2:n_old
        C[i] = C[i - 1] + (mon[i] + mon[i - 1]) / 2 * (x[i] - x[i - 1])
    end
    total = C[n_old]
    total > zero(T) || return collect(range(a_min, a_max; length=n))

    out = zeros(T, n)
    out[1] = a_min
    out[n] = a_max
    i = 1
    @inbounds for j in 2:(n - 1)
        target = total * T(j - 1) / T(n - 1)
        while i < n_old - 1 && C[i + 1] < target
            i += 1
        end
        dC = C[i + 1] - C[i]
        frac = dC > zero(T) ? (target - C[i]) / dC : zero(T)
        out[j] = x[i] + frac * (x[i + 1] - x[i])
    end

    # Roundoff in the inversion can tie adjacent nodes; nudge them apart so the grid stays
    # strictly increasing (a tied node would make the Young transition weights singular).
    @inbounds for j in 2:(n - 1)
        out[j] = max(out[j], nextfloat(out[j - 1]))
    end
    out[n] = a_max
    out[n] > out[n - 1] || (out[n - 1] = prevfloat(a_max))

    return out
end

"""
    adapt_ha_grid(grid::HAGrid{T}, distribution; n_points=grid.n_points,
                  curvature=0.9, monitor_cap=3.0, smoothing=2) → HAGrid{T}
    adapt_ha_grid(spec::HADSGESpec{T}, ss::HASteadyState{T}; kwargs...) → HADSGESpec{T}

Rebuild an HA grid with nodes equidistributed by the curvature of the stationary density.

The bounds, labels and number of income states are preserved, so the borrowing constraint and
the aggregation functions carry over unchanged. Only the *placement* of the asset nodes moves.

The two-argument `HADSGESpec`/`HASteadyState` method returns a new specification carrying the
adapted grid; re-run [`compute_steady_state`](@ref) on it to solve the model on the new grid.

Measured on `load_ha_example(:krusell_smith)` at `n_a = 200` against an `n_a = 1600`
`:geometric` reference (`r = 0.00772022`, `K = 42.39574`), one adaptation round improves on
the shipped grid: `|Δr|` falls from `2.01e-5` to `1.68e-5` and `|ΔK|/K` from `9.62e-4` to
`8.05e-4`. The gain is modest because `:geometric` is already tuned for this density shape;
the method earns its keep when the density shape is not known in advance.

# Keyword Arguments
- `n_points` — nodes per asset dimension in the new grid (default: unchanged)
- `curvature::Real=0.9` — share of nodes placed by curvature; `0` returns a uniform grid
- `monitor_cap::Real=3.0` — cap on the monitor against the borrowing-constraint atom
- `smoothing::Int=2` — smoothing passes on the monitor function

# Examples
```julia
ss    = compute_steady_state(spec)
spec2 = adapt_ha_grid(spec, ss)
ss2   = compute_steady_state(spec2)     # solved on the curvature-adapted grid
```

See also [`adaptive_asset_grid`](@ref), [`ha_grid_diagnostics`](@ref).
"""
function adapt_ha_grid(grid::HAGrid{T}, distribution::AbstractArray;
                       n_points=grid.n_points,
                       curvature::Real=0.9,
                       monitor_cap::Real=3.0,
                       smoothing::Int=2) where {T<:AbstractFloat}
    np = collect(Int, n_points)
    length(np) == grid.n_dims || throw(ArgumentError(
        "adapt_ha_grid: n_points must have $(grid.n_dims) entries, got $(length(np))"))

    new_grids = Vector{Vector{T}}(undef, grid.n_dims)
    for d in 1:grid.n_dims
        marg = _asset_marginal(distribution, grid, d)
        new_grids[d] = adaptive_asset_grid(grid.grids[d], marg;
                                           n=np[d], curvature=curvature,
                                           monitor_cap=monitor_cap, smoothing=smoothing)
    end

    return HAGrid{T}(new_grids, np, grid.n_dims, grid.n_income, grid.bounds, grid.labels)
end

function adapt_ha_grid(spec::HADSGESpec{T}, ss::HASteadyState{T}; kwargs...) where {T<:AbstractFloat}
    new_grid = adapt_ha_grid(spec.grid, ss.distribution; kwargs...)
    return HADSGESpec{T}(spec.aggregate_spec, spec.individual, spec.income, new_grid,
                         spec.aggregation, spec.het_params;
                         model=spec.model, distribution=spec.distribution)
end
