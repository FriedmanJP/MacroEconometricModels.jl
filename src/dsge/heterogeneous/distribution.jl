# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Non-stochastic histogram method for tracking the wealth distribution
(Young 2010).

Constructs a sparse transition matrix Λ such that D_{t+1} = Λ D_t, where D is
the distribution over (asset, income) states.  The lottery-based assignment
ensures that off-grid policy values are split between adjacent grid points
while preserving total mass.

# References
- Young, E. R. (2010). Solving the incomplete markets model with aggregate
  uncertainty using the Krusell–Smith algorithm and non-stochastic simulations.
  *Journal of Economic Dynamics and Control*, 34(1), 36–41.
"""

# =============================================================================
# _build_transition_matrix — Young (2010) lottery on the asset grid
# =============================================================================

"""
    _build_transition_matrix(a_policy, grid, income) → SparseMatrixCSC{T}

Build the sparse transition matrix Λ for the joint (asset, income) distribution
using the non-stochastic histogram method of Young (2010).

For each source state (a_i, e_j) the savings policy a' = `a_policy[i,j]` is
bracketed on the asset grid.  Lottery weights split mass between the two
enclosing grid points, and the income transition matrix distributes mass across
future income states.

The matrix Λ is N × N where N = n_a × n_e.  Index convention:
(j-1) * n_a + i maps to asset point i and income state j.
Columns of Λ sum to 1.

# Arguments
- `a_policy::Matrix{T}` — savings policy, n_a × n_e
- `grid::HAGrid{T}` — one-asset grid
- `income::IncomeProcess{T}` — idiosyncratic income process (row-stochastic)
"""
function _build_transition_matrix(a_policy::Matrix{T}, grid::HAGrid{T},
                                   income::IncomeProcess{T}) where {T<:AbstractFloat}
    a_grid = grid.grids[1]
    n_a = length(a_grid)
    n_e = length(income.states)
    N = n_a * n_e
    Pi = income.transition  # row-stochastic: Pi[j, jp] = P(e' = e_jp | e = e_j)

    # Pre-allocate COO vectors — upper bound on nonzeros: 2 * n_e * n_a * n_e
    max_nnz = 2 * n_e * N
    rows = Vector{Int}(undef, max_nnz)
    cols = Vector{Int}(undef, max_nnz)
    vals = Vector{T}(undef, max_nnz)
    count = 0

    @inbounds for j in 1:n_e
        for i in 1:n_a
            col_idx = (j - 1) * n_a + i
            a_prime = a_policy[i, j]

            # Clamp to grid bounds
            a_prime = clamp(a_prime, a_grid[1], a_grid[end])

            # Find bracket: a_grid[k] <= a_prime <= a_grid[k+1]
            k = searchsortedfirst(a_grid, a_prime) - 1
            k = clamp(k, 1, n_a - 1)

            # Lottery weights
            denom = a_grid[k + 1] - a_grid[k]
            if denom < T(1e-15)
                # Degenerate interval — all mass to lower point
                w_lo = one(T)
            else
                w_lo = (a_grid[k + 1] - a_prime) / denom
            end
            w_lo = clamp(w_lo, zero(T), one(T))
            w_hi = one(T) - w_lo

            # Distribute across future income states
            for jp in 1:n_e
                p_trans = Pi[j, jp]
                if p_trans < T(1e-20)
                    continue
                end

                # Lower bracket point
                if w_lo > T(1e-20)
                    count += 1
                    rows[count] = (jp - 1) * n_a + k
                    cols[count] = col_idx
                    vals[count] = w_lo * p_trans
                end

                # Upper bracket point
                if w_hi > T(1e-20)
                    count += 1
                    rows[count] = (jp - 1) * n_a + k + 1
                    cols[count] = col_idx
                    vals[count] = w_hi * p_trans
                end
            end
        end
    end

    # Trim to actual count and build sparse matrix (duplicates are summed)
    resize!(rows, count)
    resize!(cols, count)
    resize!(vals, count)

    Lambda = sparse(rows, cols, vals, N, N)
    return Lambda
end

# =============================================================================
# _stationary_dist_young — power iteration on the transition matrix
# =============================================================================

"""
    _stationary_dist_young(Lambda; max_iter=10000, tol=1e-12) → (Vector{T}, Bool)

Compute the stationary distribution d* satisfying d* = Λ d* via power
iteration.  Returns the distribution (non-negative, sums to 1) and a `converged`
flag (`true` iff ‖d_{t+1} − d_t‖_∞ met `tol` before exhausting `max_iter`; #240/H-17).

# Arguments
- `Lambda::SparseMatrixCSC{T}` — transition matrix (columns sum to 1)
- `max_iter::Int` — maximum number of iterations (default 10 000)
- `tol` — convergence tolerance on ‖d_{t+1} − d_t‖_∞ (default 1e-12)
"""
function _stationary_dist_young(Lambda::SparseMatrixCSC{T};
                                 max_iter::Int=10_000,
                                 tol::Real=1e-12) where {T<:AbstractFloat}
    N = size(Lambda, 1)
    tol_T = T(tol)

    # The stationary distribution is the RIGHT eigenvector of the column-stochastic
    # transition Λ for eigenvalue 1 (Λ d = d, d ≥ 0, Σd = 1). Solve it in ONE sparse
    # LU solve instead of thousands of power-iteration mat-vecs (#242): (I − Λ) is
    # singular, so replace one equation with the mass normalization Σg = 1. (NOTE:
    # transposing to (I − Λ')g = 0 gives the LEFT eigenvector = the all-ones vector
    # ⇒ the uniform distribution, which is WRONG for a column-stochastic Λ.)
    local g
    solved = false
    try
        A = spdiagm(0 => ones(T, N)) - Lambda
        A[N, :] .= one(T)                      # replace last row with Σg = 1
        b = zeros(T, N); b[N] = one(T)
        g = A \ b
        solved = all(isfinite, g) && sum(g) > zero(T)
    catch
        solved = false
    end

    if solved
        @inbounds for i in eachindex(g)         # project onto the simplex
            g[i] < zero(T) && (g[i] = zero(T))
        end
        g ./= sum(g)
        return g, true
    end

    # Fallback: power iteration (robustness guard if the LU solve fails).
    d = fill(one(T) / N, N)
    for _ in 1:max_iter
        d_new = Lambda * d
        s = sum(d_new)
        s > zero(T) && (d_new ./= s)
        if maximum(abs.(d_new .- d)) < tol_T
            return d_new, true
        end
        d = d_new
    end
    d ./= sum(d)
    return d, false
end

# =============================================================================
# _forward_iterate — single-step distribution update
# =============================================================================

"""
    _forward_iterate(Lambda, d_old) → d_new

Advance the distribution one period: d_new = Λ d_old, then normalize to sum
to 1.

# Arguments
- `Lambda::SparseMatrixCSC{T}` — transition matrix (columns sum to 1)
- `d_old::Vector{T}` — current distribution (sums to 1)
"""
function _forward_iterate(Lambda::SparseMatrixCSC{T},
                           d_old::Vector{T}) where {T<:AbstractFloat}
    d_new = Lambda * d_old
    s = sum(d_new)
    if s > zero(T)
        d_new ./= s
    end
    return d_new
end

# =============================================================================
# _aggregate — integrate a variable over the distribution
# =============================================================================

"""
    _aggregate(d, grid; var_index=1) → scalar

Compute the aggregate (mean) of asset dimension `var_index` under the
distribution `d`:

    X = Σ_j Σ_i  grid[var_index][i] × d[(j-1)*n_a + i]

# Arguments
- `d::Vector{T}` — distribution over (asset, income) states (length n_a × n_e)
- `grid::HAGrid{T}` — grid (one-asset)
- `var_index::Int` — which grid dimension to integrate (default 1)
"""
function _aggregate(d::Vector{T}, grid::HAGrid{T};
                     var_index::Int=1) where {T<:AbstractFloat}
    a_grid = grid.grids[var_index]
    n_a = length(a_grid)
    N = length(d)
    n_e = div(N, n_a)

    agg = zero(T)
    @inbounds for j in 1:n_e
        offset = (j - 1) * n_a
        for i in 1:n_a
            agg += a_grid[i] * d[offset + i]
        end
    end
    return agg
end
