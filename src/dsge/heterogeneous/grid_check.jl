# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Asset-grid adequacy diagnostics for heterogeneous agent steady states.

The Young (2010) transition matrix clamps the savings policy into the asset grid
(`_build_transition_matrix`). Clamping conserves *mass* exactly, so no
distributional check can see it — but it destroys *assets*, so the market a
truncated model clears is not the market it claims to solve. These diagnostics
make that discrepancy visible.
"""

# =============================================================================
# HAGridDiagnostics — grid adequacy of a one-asset HA steady state
# =============================================================================

"""
    HAGridDiagnostics{T}

Diagnostics on whether a one-asset HA steady state's asset grid is wide enough
for its stationary distribution.

Fields:
- `a_min::T`, `a_max::T`, `n_a::Int`, `n_e::Int` — grid geometry
- `ceiling_mass::T` — stationary mass at the top asset node
- `floor_mass::T` — stationary mass at the bottom asset node
- `n_cells_above::Int` — cells whose savings policy exceeds `a_max`
- `n_cells_below::Int` — cells whose savings policy falls below `a_min`
- `truncation_flux_up::T` — `∫ max(a′ − a_max, 0) dμ`, assets **destroyed** per
  period by the upper clamp
- `truncation_flux_down::T` — `∫ max(a_min − a′, 0) dμ`, assets **created** per
  period by the lower clamp
- `assets_held::T` — `∫ a dμ`, the aggregate the steady-state bisection clears on
- `assets_desired::T` — `∫ a′ dμ`, the aggregate the sequence-space household
  block reports
- `clearing_residual::T` — `assets_desired − assets_held`
- `relative_residual::T` — `clearing_residual / max(|assets_held|, 1)`
- `max_savings::T` — `max a′` over all cells
- `ceiling_mass_tol::T`, `residual_tol::T` — thresholds used for `adequate`
- `adequate::Bool` — grid is wide enough on both margins

# Exact identity

For a **stationary** Young histogram the aggregates are linked by

    ∫ a′ dμ − ∫ a dμ  =  ∫ max(a′ − a_max, 0) dμ − ∫ max(a_min − a′, 0) dμ

i.e. `clearing_residual == truncation_flux_up − truncation_flux_down`. This is
an identity, not an approximation: stationarity says next period's holdings
integrate to `∫ a dμ`, and the only wedge between those and the desired `∫ a′ dμ`
is what the clamp removed. A non-zero residual therefore *is* truncation.

See also [`ha_grid_diagnostics`](@ref).
"""
struct HAGridDiagnostics{T<:AbstractFloat}
    a_min::T
    a_max::T
    n_a::Int
    n_e::Int
    ceiling_mass::T
    floor_mass::T
    n_cells_above::Int
    n_cells_below::Int
    truncation_flux_up::T
    truncation_flux_down::T
    assets_held::T
    assets_desired::T
    clearing_residual::T
    relative_residual::T
    max_savings::T
    ceiling_mass_tol::T
    residual_tol::T
    adequate::Bool
end

"""
    _ha_grid_diagnostics(a_policy, dist, grid; ceiling_mass_tol=1e-6, residual_tol=1e-6)

Kernel computing [`HAGridDiagnostics`](@ref) from a savings policy, a
distribution and a one-asset grid. Never mutates its arguments — the
distribution is normalized on a copy.
"""
function _ha_grid_diagnostics(a_policy::AbstractMatrix{T}, dist::AbstractArray{T},
                              grid::HAGrid{T};
                              ceiling_mass_tol::Real=T(1e-6),
                              residual_tol::Real=T(1e-6)) where {T<:AbstractFloat}
    grid.n_dims == 1 || throw(ArgumentError(
        "ha_grid_diagnostics: one-asset grids only (got n_dims = $(grid.n_dims))."))

    a_grid = grid.grids[1]
    n_a = length(a_grid)
    n_e = div(length(dist), n_a)
    size(a_policy) == (n_a, n_e) || throw(DimensionMismatch(
        "ha_grid_diagnostics: savings policy is $(size(a_policy)), expected ($n_a, $n_e)."))

    d = reshape(collect(vec(dist)), n_a, n_e)
    total = sum(d)
    total > zero(T) || throw(ArgumentError("ha_grid_diagnostics: distribution sums to $total."))
    d ./= total

    a_lo, a_hi = a_grid[1], a_grid[end]
    tol_node = sqrt(eps(T)) * max(one(T), abs(a_hi))

    ceiling_mass = zero(T); floor_mass = zero(T)
    flux_up = zero(T); flux_down = zero(T)
    held = zero(T); desired = zero(T)
    n_above = 0; n_below = 0
    max_sav = typemin(T)

    @inbounds for j in 1:n_e
        ceiling_mass += d[n_a, j]
        floor_mass   += d[1, j]
        for i in 1:n_a
            w  = d[i, j]
            ap = a_policy[i, j]
            held    += a_grid[i] * w
            desired += ap * w
            ap > max_sav && (max_sav = ap)
            if ap > a_hi + tol_node
                n_above += 1
                flux_up += (ap - a_hi) * w
            elseif ap < a_lo - tol_node
                n_below += 1
                flux_down += (a_lo - ap) * w
            end
        end
    end

    residual = desired - held
    rel = residual / max(abs(held), one(T))
    cm_tol = T(ceiling_mass_tol)
    r_tol  = T(residual_tol)
    adequate = ceiling_mass <= cm_tol && abs(rel) <= r_tol && n_below == 0

    return HAGridDiagnostics{T}(a_lo, a_hi, n_a, n_e, ceiling_mass, floor_mass,
                                n_above, n_below, flux_up, flux_down,
                                held, desired, residual, rel, max_sav,
                                cm_tol, r_tol, adequate)
end

"""
    _ha_grid_diagnostics(b_policy, dist, grid; ...) → HAGridDiagnostics

Liquid-grid adequacy of a two-asset histogram. The struct still describes one
asset dimension: here that dimension is liquid `b` (the borrowing constraint).
Illiquid truncation is reported separately on `ss.aggregates[:A_policy]`.
"""
function _ha_grid_diagnostics(b_policy::AbstractArray{T,3}, dist::AbstractArray{T},
                              grid::HAGrid{T};
                              ceiling_mass_tol::Real=T(1e-6),
                              residual_tol::Real=T(1e-6)) where {T<:AbstractFloat}
    grid.n_dims == 2 || throw(ArgumentError(
        "ha_grid_diagnostics: 3-D policy requires a two-asset grid"))
    b_grid = grid.grids[1]
    n_b = length(b_grid)
    n_a = grid.n_points[2]
    n_e = grid.n_income
    size(b_policy) == (n_b, n_a, n_e) || throw(DimensionMismatch(
        "ha_grid_diagnostics: liquid policy is $(size(b_policy)), expected ($n_b, $n_a, $n_e)."))
    d = collect(vec(dist))
    length(d) == n_b * n_a * n_e || throw(DimensionMismatch(
        "ha_grid_diagnostics: distribution length $(length(d)) ≠ $n_b×$n_a×$n_e"))
    total = sum(d)
    total > zero(T) || throw(ArgumentError("ha_grid_diagnostics: distribution sums to $total."))
    d ./= total

    b_lo, b_hi = b_grid[1], b_grid[end]
    tol_node = sqrt(eps(T)) * max(one(T), abs(b_hi))
    ceiling_mass = zero(T); floor_mass = zero(T)
    flux_up = zero(T); flux_down = zero(T)
    held = zero(T); desired = zero(T)
    n_above = 0; n_below = 0
    max_sav = typemin(T)

    @inbounds for je in 1:n_e, ia in 1:n_a, ib in 1:n_b
        w = d[_ha_state_index(ib, ia, je, n_b, n_a)]
        bp = b_policy[ib, ia, je]
        ib == n_b && (ceiling_mass += w)
        ib == 1 && (floor_mass += w)
        held += b_grid[ib] * w
        desired += bp * w
        bp > max_sav && (max_sav = bp)
        if bp > b_hi + tol_node
            n_above += 1
            flux_up += (bp - b_hi) * w
        elseif bp < b_lo - tol_node
            n_below += 1
            flux_down += (b_lo - bp) * w
        end
    end
    residual = desired - held
    rel = residual / max(abs(held), one(T))
    cm_tol = T(ceiling_mass_tol)
    r_tol = T(residual_tol)
    adequate = ceiling_mass <= cm_tol && abs(rel) <= r_tol && n_below == 0
    return HAGridDiagnostics{T}(b_lo, b_hi, n_b, n_e, ceiling_mass, floor_mass,
                                n_above, n_below, flux_up, flux_down,
                                held, desired, residual, rel, max_sav,
                                cm_tol, r_tol, adequate)
end

"""
    ha_grid_diagnostics(ss::HASteadyState; ceiling_mass_tol=1e-6, residual_tol=1e-6)
        → HAGridDiagnostics

Check whether the asset grid of a solved one-asset HA steady state is wide
enough for its own stationary distribution.

An HA steady state can be *exactly* stationary and still fail to clear its asset
market: the Young (2010) transition clamps the savings policy at the top grid
node, which conserves mass but destroys assets. `excess_demand` cannot see this,
because it is measured on the already-clamped aggregate `∫ a dμ`. This function
compares that against the aggregate the policy actually implies, `∫ a′ dμ`.

# Example

```julia
ss = compute_steady_state(load_ha_example(:krusell_smith))
d  = ha_grid_diagnostics(ss)
d.adequate            # true — the grid is wide enough
d.ceiling_mass        # stationary mass pinned at a_max
d.clearing_residual   # ∫a′dμ − ∫a dμ; non-zero ⟺ the grid truncates
```

Raise `a_max` (and prefer `grid_type=:geometric`, whose bottom spacing does not
degrade as the ceiling rises) if `adequate` is `false`.

See also [`HAGridDiagnostics`](@ref), [`compute_steady_state`](@ref).
"""
function ha_grid_diagnostics(ss::HASteadyState{T}; kwargs...) where {T<:AbstractFloat}
    pol = ss.grid.n_dims == 2 ? ss.policies[:liquid_savings] : ss.policies[:savings]
    return _ha_grid_diagnostics(pol, ss.distribution, ss.grid; kwargs...)
end

"""
    _check_grid_adequacy(d::HAGridDiagnostics, mode::Symbol; context="") → d

Emit the grid-adequacy verdict. `mode` is `:none` (silent), `:warn` or `:error`.
Returns `d` unchanged so it can be used inline.
"""
function _check_grid_adequacy(d::HAGridDiagnostics{T}, mode::Symbol;
                              context::AbstractString="") where {T<:AbstractFloat}
    mode === :none && return d
    mode in (:warn, :error) || throw(ArgumentError(
        "grid_check must be :none, :warn or :error; got :$mode"))
    d.adequate && return d

    prefix = isempty(context) ? "" : "$context: "

    if d.ceiling_mass > d.ceiling_mass_tol || abs(d.relative_residual) > d.residual_tol
        msg = "$(prefix)asset grid truncates the stationary distribution. " *
              "$(round(100 * d.ceiling_mass; sigdigits=4))% of mass sits at the top " *
              "node a_max = $(d.a_max) and $(d.n_cells_above) cell(s) want to save " *
              "beyond it (max a' = $(round(d.max_savings; sigdigits=6))). The Young " *
              "transition clamps them, destroying " *
              "$(round(d.truncation_flux_up; sigdigits=4)) units of assets per period: " *
              "households hold ∫a dμ = $(round(d.assets_held; sigdigits=8)) but their " *
              "policy implies ∫a' dμ = $(round(d.assets_desired; sigdigits=8)), a " *
              "residual of $(round(d.clearing_residual; sigdigits=4)) " *
              "($(round(100 * d.relative_residual; sigdigits=4))%). The asset market " *
              "does NOT clear at this steady state, and `excess_demand` cannot see it " *
              "because it is measured on the clamped aggregate. Raise a_max — e.g. " *
              "HAGrid(; assets=($(d.a_min), $(2 * d.a_max), $(d.n_a)), " *
              "grid_type=:geometric) — or silence this with grid_check=:none."
        mode === :error ? throw(ArgumentError(msg)) : @warn msg maxlog = 1
    end

    if d.n_cells_below > 0
        msg = "$(prefix)savings policy falls below the grid floor a_min = $(d.a_min) in " *
              "$(d.n_cells_below) cell(s). The Young transition clamps them upward, " *
              "CREATING $(round(d.truncation_flux_down; sigdigits=4)) units of assets " *
              "per period out of nothing. The grid lower bound must equal the borrowing " *
              "constraint."
        mode === :error ? throw(ArgumentError(msg)) : @warn msg maxlog = 1
    end

    return d
end

function Base.show(io::IO, d::HAGridDiagnostics{T}) where {T}
    print(io, "HAGridDiagnostics{$T}: a ∈ [$(d.a_min), $(d.a_max)], ",
              "ceiling mass=$(_fmt(100 * d.ceiling_mass; digits=4))%, ",
              "residual=$(_fmt(d.clearing_residual; digits=6)), ",
              d.adequate ? "adequate" : "TRUNCATED")
end
