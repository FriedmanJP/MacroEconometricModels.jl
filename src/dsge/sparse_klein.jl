# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# MacroEconometricModels.jl — sparse/structured linear DSGE solution ([T270])
#
# The linear solvers reach the stable solvent `G` of
#
#     f_lead·G² + f_0·G + f_1 = 0
#
# through a dense complex QZ of the `2n × 2n` companion pencil. That is `O(n³)` with a large
# constant and `O(n²)` memory, and it ignores the fact that each equation of a medium-large
# model touches only a handful of variables: a 400-sector system has `n = 800` with a density
# of 0.2%, and its companion QZ takes ~8 s.
#
# This file adds a path that never forms a QZ decomposition. Newton's method on the quadratic
# gives, for an iterate `G_k` with residual `R_k = f_lead·G_k² + f_0·G_k + f_1`,
#
#     (f_lead·G_k + f_0)·ΔG + f_lead·ΔG·G_k = −R_k
#
# — a generalized Sylvester equation `A·X + B·X·C = D` — solved matrix-free by GMRES. Newton
# converges quadratically: 7 iterations on the benchmark below, independent of `n`.
#
# ## Where the speedup actually comes from
#
# Not from `O(nnz)` matvecs. The solvent `G` is dense even when the model is sparse, so
# `A = f_lead·G + f_0` is dense and the matvec `A·X + f_lead·(X·G)` is dominated by dense
# `O(n³)` matrix products. The win is that those are a handful of BLAS-3 REAL matmuls, whereas
# the dense route runs a COMPLEX QZ on the `2n × 2n` companion pencil — far more flops per
# dimension and no BLAS-3 efficiency. Sparsity helps second-order, by making the `f_0·X` and
# `f_lead·(·)` products cheap.
#
# Because the cost is set by how quickly GMRES converges, the speedup is **model-dependent**.
# Measured end-to-end against the dense core on a multi-sector benchmark:
#
# | n | density | speedup |
# |---|---|---|
# | 100 | 0.017 | 0.21× (slower — routed to dense) |
# | 400 | 0.004 | 1.15× |
# | 800 | 0.002 | 1.67-1.84× |
# | 1600 | 0.001 | 2.68× |
#
# On a FULLY DENSE model of the same size the advantage is 1.8× at `n = 400` but gone by
# `n = 800`, which is why the routing heuristic requires sparsity as well as size.
#
# It does **not** make the determinacy verdict cheaper: `eu` still comes from the Sims (2002)
# rank test ([T267]), which decomposes the `(n + k)` canonical pencil and is ~22% of the dense
# cost. The figures above are end-to-end and already include it; the solvent step alone looks
# better than the model as a whole gets.
#
# ## Why this is safe
#
# Newton finds *a* solvent, not necessarily the stable one, and its basin depends on the
# starting point (`G = 0` here). So the result is only accepted when it is certified: the
# residual must be small AND `G` must be stable (`max|eig(G)| < div`). Otherwise the dense
# companion-QZ core runs and its answer is used. The determinacy verdict is never taken from
# this path. A model that the sparse route cannot solve is therefore slower, never wrong.
#
# References:
#   Klein, P. (2000). Using the generalized Schur form to solve a multivariate linear rational
#     expectations model. Journal of Economic Dynamics and Control 24(10), 1405-1423.
#   Kamenik, O. (2005). Solving SDGE models: a new algorithm for the Sylvester equation.
#     Computational Economics 25(1-2), 167-187.

using SparseArrays

"""
    _sparse_density(mats...) → Float64

Fraction of structurally nonzero entries across the supplied matrices. Used by the routing
heuristic; returns `1.0` for empty input so a degenerate model never looks sparse.
"""
function _sparse_density(mats...)
    total = 0
    nz = 0
    for A in mats
        total += length(A)
        nz += count(!iszero, A)
    end
    return total == 0 ? 1.0 : nz / total
end

"""
    _should_use_sparse_klein(f_0, f_1, f_lead; min_n=400, max_density=0.05) → Bool

Routing heuristic for the `:auto` sparse path: take it only when the model is both **large
enough** for the dense QZ to hurt and **sparse enough** for the matrix-free matvecs to pay off.

The thresholds come from the benchmark in this file's header, not from taste: below `n ≈ 400`
the dense companion QZ is as fast or faster, and a dense operator makes the GMRES matvec
`O(n³)` per iteration, which is strictly worse than decomposing once.
"""
function _should_use_sparse_klein(f_0::AbstractMatrix, f_1::AbstractMatrix,
                                  f_lead::AbstractMatrix;
                                  min_n::Int=400, max_density::Real=0.05)
    n = size(f_0, 1)
    n >= min_n || return false
    return _sparse_density(f_0, f_1, f_lead) <= max_density
end

"""
    _newton_solvent(f_0, f_1, f_lead; div, tol=1e-11, maxiter=50, gmres_tol=1e-10) → (G, info)

Stable solvent of `f_lead·G² + f_0·G + f_1 = 0` by Newton's method, each step a matrix-free
GMRES solve of the generalized Sylvester equation

```
(f_lead·G_k + f_0)·ΔG + f_lead·ΔG·G_k = −R_k
```

with the operator applied through sparse products, so no dense `n × n` factorization is ever
formed. Starts from `G = 0`.

Returns `(G, info)` with `info` carrying `:ok`, `:residual`, `:iterations`, `:spectral_radius`
and `:reason`. `G` is `nothing` unless the solve **certifies**: the relative residual must
reach `tol` and the spectral radius must be below `div`, because Newton converges to whichever
solvent its basin contains and only the stable one is the model's solution. The caller falls
back to the dense core when it does not certify.
"""
function _newton_solvent(f_0::AbstractMatrix{T}, f_1::AbstractMatrix{T},
                         f_lead::AbstractMatrix{T};
                         div::Real=1.0 + 1e-8, tol::Real=1e-11, maxiter::Int=50,
                         gmres_tol::Real=1e-10) where {T<:AbstractFloat}
    n = size(f_0, 1)
    n == 0 && return (zeros(T, 0, 0),
                      (ok=true, residual=zero(T), iterations=0, spectral_radius=zero(T),
                       reason=:empty))

    f0s = sparse(f_0)
    f1s = sparse(f_1)
    fls = sparse(f_lead)
    scale = max(norm(f_1, Inf), norm(f_0, Inf), one(T))

    G = zeros(T, n, n)
    resid = T(Inf)
    iters = 0
    for it in 1:maxiter
        iters = it
        R = fls * (G * G) + f0s * G + f1s
        resid = norm(R, Inf) / scale
        resid <= tol && break
        all(isfinite, R) || return (nothing, (ok=false, residual=T(Inf), iterations=it,
                                              spectral_radius=T(Inf), reason=:diverged))

        A = fls * G + f0s          # sparse + sparse*dense ⇒ dense, but never factorized
        matvec! = function (y::AbstractVector, x::AbstractVector)
            X = reshape(x, n, n)
            copyto!(y, vec(A * X + fls * (X * G)))
            return y
        end
        step = _gmres_solve!(matvec!, -vec(R), n * n;
                             gmres_tol=T(gmres_tol), gmres_max_outer=30)
        all(isfinite, step) || return (nothing, (ok=false, residual=T(Inf), iterations=it,
                                                 spectral_radius=T(Inf), reason=:diverged))
        G .+= reshape(step, n, n)
    end

    if !(resid <= tol)
        return (nothing, (ok=false, residual=resid, iterations=iters,
                          spectral_radius=T(NaN), reason=:not_converged))
    end
    # Newton finds A solvent; only the stable one is the model's solution.
    rho = maximum(abs, eigvals(G); init=zero(T))
    rho < div || return (nothing, (ok=false, residual=resid, iterations=iters,
                                   spectral_radius=rho, reason=:unstable_solvent))
    return (G, (ok=true, residual=resid, iterations=iters, spectral_radius=rho,
                reason=:converged))
end
