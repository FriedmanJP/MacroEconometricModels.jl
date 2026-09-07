# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# MacroEconometricModels.jl — Smolyak grid + Chebyshev-collocation interpolant for VFI
#
# Private building blocks for Smolyak value-function iteration (issues #817, #819,
# parent #620 §2). Nothing here is wired into `vfi_solver` yet: the tensor-only
# solver path is untouched until #819.
#
# Design (per #817):
# - Reuse `src/dsge/projection.jl` (`_smolyak_admissible_levels`,
#   `_smolyak_grid_from_levels`, `_chebyshev_basis_multi`, `_scale_to_unit`) with the
#   SAME helper calls as PFI's Smolyak path (`src/dsge/pfi.jl:260-285`), so node sets
#   are identical for the same bounds/μ. No combination-technique solve weights:
#   collocation refits `coeffs = basis_matrix \ Vvals` per VFI iteration, matching
#   the PFI precedent.
# - The basis factorization is built ONCE per solve (`VFISmolyakCache`) and reused
#   every iteration; the value interpolant is rebuilt immutably per outer VFI
#   iteration BEFORE the threaded node loop (never mutated inside it).
# - Out-of-box evaluation clamps + applies the same linear penalty as
#   `_vfi_multilinear_scalar` — never silent extrapolation, never `NaN`/throw.
#
# References:
#   Judd-Maliar-Maliar-Valero (2014), Smolyak Method for Nonlinear Dynamic Models
#   Krueger-Kubler (2004), Computing Equilibrium in OLG Models with Production

"""
    VFISmolyakCache{T}

Read-only Smolyak collocation cache for VFI, built ONCE per solve by
`_vfi_build_smolyak_grid` and read-only thereafter.

Fields:
- `nodes` — `N × nx` physical collocation nodes
- `multi_indices` — `N × nx` unisolvent Chebyshev multi-index set
- `F` — LU factorization of the `N × N` collocation basis matrix, reused for every
  per-iteration `coeffs = F \\ Vvals` refit
- `levels` — `n_blocks × nx` admissible level set (fills
  `ProjectionSolution.smolyak_levels` when wired in)
- `state_bounds` — `nx × 2` physical state box
"""
struct VFISmolyakCache{T<:AbstractFloat}
    nodes::Matrix{T}
    multi_indices::Matrix{Int}
    F::LU{T,Matrix{T}}
    levels::Matrix{Int}
    state_bounds::Matrix{T}
end

"""
    _vfi_build_smolyak_grid(state_bounds, nx, smolyak_mu) -> VFISmolyakCache

Build the Smolyak collocation cache for VFI over `state_bounds` (`nx × 2`).
`smolyak_mu` is a scalar (isotropic `|l|₁ ≤ μ`) or an `nx`-vector (anisotropic
`Σ l_k/μ_k ≤ 1`); both go through `_smolyak_level_vector` /
`_smolyak_admissible_levels` / `_smolyak_grid_from_levels`, i.e. the identical
construction PFI uses, so node sets match PFI's for the same bounds/μ.

Default μ=2 for VFI: PFI's μ=3 default is too rich per-iteration for VFI's
refit-every-sweep cost (N(4,3)=137 vs N(4,2)=41 nodes).

Note the d=1 degeneracy: Smolyak with `nx == 1` gives `N == n_basis`
(1, 3, 5, 9, 17 nodes for μ=0..4) — benign but wasteful next to a plain tensor
row, so callers route `nx == 1` to the tensor path.
"""
function _vfi_build_smolyak_grid(state_bounds::AbstractMatrix{T}, nx::Integer,
                                 smolyak_mu) where {T<:AbstractFloat}
    nx >= 1 || throw(ArgumentError("_vfi_build_smolyak_grid: nx must be ≥ 1, got $nx"))
    size(state_bounds, 1) == nx || throw(ArgumentError(
        "_vfi_build_smolyak_grid: state_bounds has $(size(state_bounds, 1)) rows, nx = $nx"))
    mu_vec = _smolyak_level_vector(Int(nx), smolyak_mu)
    level_list = _smolyak_admissible_levels(mu_vec)
    nodes_unit, multi_indices = _smolyak_grid_from_levels(level_list)
    nodes_phys = Matrix{T}(_scale_from_unit(Matrix{Float64}(nodes_unit),
                                            Matrix{Float64}(state_bounds)))
    basis_matrix = Matrix{T}(_chebyshev_basis_multi(nodes_unit, multi_indices))
    size(basis_matrix, 1) == size(basis_matrix, 2) || throw(ArgumentError(
        "_vfi_build_smolyak_grid: collocation system is not square " *
        "($(size(basis_matrix, 1)) nodes × $(size(basis_matrix, 2)) basis)"))
    F = lu(basis_matrix)
    levels = zeros(Int, length(level_list), Int(nx))
    for (i, l) in enumerate(level_list)
        levels[i, :] = l
    end
    return VFISmolyakCache{T}(nodes_phys, multi_indices, F, levels,
                              Matrix{T}(state_bounds))
end

"""
    VSmolyakInterpolant{T}

Immutable Chebyshev-collocation interpolant of the value function on a Smolyak
grid. Rebuilt once per outer VFI iteration via `build_V_interpolant` BEFORE the
threaded node loop; calling it is pure (locals only — no mutation, memo, or RNG),
so sharing one interpolant across threads is safe.

Out-of-box evaluation clamps to the state box and subtracts the same linear
penalty `_vfi_multilinear_scalar` applies (`50 × distance/span` per dimension),
mirroring `_pfi_compute_expectations` clamping: no `NaN`, no throw.
"""
struct VSmolyakInterpolant{T<:AbstractFloat}
    state_bounds::Matrix{T}
    multi_indices::Matrix{Int}
    coeffs::Vector{T}
end

"""
    build_V_interpolant(cache::VFISmolyakCache{T}, Vvals::AbstractVector) -> VSmolyakInterpolant{T}

Collocation refit `coeffs = cache.F \\ Vvals` reusing the cached factorization
(PFI precedent `src/dsge/pfi.jl:260-285`). `Vvals` holds the value function over
the cache nodes in order.
"""
function build_V_interpolant(cache::VFISmolyakCache{T},
                             Vvals::AbstractVector) where {T<:AbstractFloat}
    n = size(cache.nodes, 1)
    length(Vvals) == n || throw(ArgumentError(
        "build_V_interpolant: got $(length(Vvals)) values for $n Smolyak nodes"))
    coeffs = Vector{T}(cache.F \ Vector{T}(Vvals))
    return VSmolyakInterpolant{T}(cache.state_bounds, cache.multi_indices, coeffs)
end

function (itp::VSmolyakInterpolant{T})(s::AbstractVector) where {T<:AbstractFloat}
    nx = size(itp.state_bounds, 1)
    length(s) == nx || throw(ArgumentError(
        "VSmolyakInterpolant: got $(length(s)) states for an $nx-state grid"))
    xc = Vector{T}(undef, nx)
    pen = zero(T)
    @inbounds for d in 1:nx
        lo = itp.state_bounds[d, 1]
        hi = itp.state_bounds[d, 2]
        xd = T(s[d])
        if xd < lo
            span = hi - lo
            pen += T(50) * (lo - xd) / max(span, eps(T))
            xc[d] = lo
        elseif xd > hi
            span = hi - lo
            pen += T(50) * (xd - hi) / max(span, eps(T))
            xc[d] = hi
        else
            xc[d] = xd
        end
    end
    z = _scale_to_unit(xc, itp.state_bounds)
    z = clamp.(z, T(-1), T(1))
    B = _chebyshev_basis_multi(reshape(Vector{T}(z), 1, nx), itp.multi_indices)
    return dot(@view(B[1, :]), itp.coeffs) - pen
end
