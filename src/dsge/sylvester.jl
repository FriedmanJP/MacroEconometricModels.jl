# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# MacroEconometricModels.jl — Generalized-Schur (Bartels-Stewart / Kamenik) matrix-equation solvers
#
# Two equations dominate the DSGE solver stack:
#
#   1. the generalized Sylvester equation  `A·X + B·X·C^{⊗d} = D`, which delivers the order-`d`
#      perturbation coefficient tensors (`d = 2, 3`), and
#   2. the discrete Lyapunov equation `P = A·P·A' + Q`, which delivers unconditional covariances
#      for analytical moments, Kalman initialization, and the pruned state space.
#
# Both were previously solved either by vectorizing into an `n·nv^d` dense system — `O(n³·nv^{3d})`
# work on an `(n·nv^d)²` matrix that exhausts memory well before `n = 50` — or by an iterative
# method (GMRES for the Sylvester, doubling for the Lyapunov). The generalized-Schur approach
# used here decomposes ONCE and then back-substitutes through the resulting triangular structure,
# which is `O(n³ + n²·nv^d + d²·n·nv^{d+1})` and, being direct, carries no iteration tolerance.
#
# The two equations end up with OPPOSITE defaults, and the reason is measurement rather than
# taste. For the Sylvester equation the direct method wins outright — it is orders of magnitude
# faster than both alternatives and is the only one that runs at all once `n·nv^d` passes ~25k.
# For the Lyapunov equation it does not: doubling is a handful of BLAS-3 products against a
# complex Schur decomposition plus a non-BLAS column sweep, and it measures 5-10x faster at equal
# accuracy. So `_dlyap` keeps doubling in front and holds Bartels-Stewart in reserve for the
# cases doubling's ABSOLUTE convergence test mishandles. See `_dlyap` for the numbers.
#
# ## Why complex Schur
#
# Kamenik (2005) works in the REAL Schur form, where a complex-conjugate eigenvalue pair leaves a
# 2×2 block on the diagonal. Those blocks have to be carried through the whole recursion, and at
# order `d` they couple `2^d` columns at a time — the bulk of the algorithm's complexity, and the
# bulk of the places it can go wrong. Taking the COMPLEX Schur form instead makes both factors
# strictly upper triangular, so every solve is a clean scalar back-substitution and the 2×2 logic
# disappears entirely. The cost is complex arithmetic (~4× the flops of the real form) on an
# `O(n³)` decomposition that is not the bottleneck; the benefit is that the block bookkeeping
# cannot be got wrong. The solution of a real system is real, so the imaginary part is discarded
# at the end — and the residual check below is what certifies that discarding it was legitimate.
#
# References:
#   Kamenik, O. (2005). Solving SDGE models: A new algorithm for the Sylvester equation.
#     Computational Economics 25(1-2), 167-187.
#   Bartels, R. H. & Stewart, G. W. (1972). Solution of the matrix equation AX + XB = C.
#     Communications of the ACM 15(9), 820-826.
#   Barraud, A. Y. (1977). A numerical algorithm to solve A'XA - X = Q.
#     IEEE Transactions on Automatic Control 22(5), 883-885.
#   Golub, G. H., Nash, S. & Van Loan, C. (1979). A Hessenberg-Schur method for the problem
#     AX + XB = C. IEEE Transactions on Automatic Control 24(6), 909-913.

using LinearAlgebra

# =============================================================================
# Shared triangular kernel
# =============================================================================

"""
    _tri_pencil_solve!(x, S, Tm, alpha, rhs) → smallest |pivot|

Solve `(S + α·Tm)·x = rhs` in place for **upper-triangular** `S` and `Tm`, by back substitution.
This is the base case of the Sylvester recursion.

Returns the smallest `|S[i,i] + α·Tm[i,i]|` encountered. That quantity IS the solvability
condition: the equation is singular exactly when some generalized eigenvalue of the pencil
`(S, Tm)` equals `−α`, so the caller can report a near-singular problem instead of returning a
plausible-looking but meaningless solution.

`x` and `rhs` may alias: `rhs[i]` is consumed before `x[i]` is written, and entries above `i`
are read only from `x`.
"""
@inline function _tri_pencil_solve!(x::AbstractVector{C}, S::AbstractMatrix{C},
                                    Tm::AbstractMatrix{C}, alpha::C,
                                    rhs::AbstractVector{C}) where {C<:Complex}
    n = length(x)
    small = typemax(real(C))
    @inbounds for i in n:-1:1
        acc = rhs[i]
        for k in (i + 1):n
            acc -= (S[i, k] + alpha * Tm[i, k]) * x[k]
        end
        piv = S[i, i] + alpha * Tm[i, i]
        ap = abs(piv)
        ap < small && (small = ap)
        x[i] = acc / piv
    end
    return small
end

"""
    _tri_shift_solve!(x, S, alpha, rhs) → smallest |pivot|

Solve `(I + α·S)·x = rhs` in place for an **upper-triangular** `S`. The Lyapunov column sweep
bottoms out here, on `(I − conj(s_jj)·S)·p̃_j = r_j`. Same aliasing contract and same
smallest-pivot return as `_tri_pencil_solve!`; kept separate so the identity term costs
no arithmetic.
"""
@inline function _tri_shift_solve!(x::AbstractVector{C}, S::AbstractMatrix{C},
                                   alpha::C, rhs::AbstractVector{C}) where {C<:Complex}
    n = length(x)
    small = typemax(real(C))
    @inbounds for i in n:-1:1
        acc = rhs[i]
        for k in (i + 1):n
            acc -= alpha * S[i, k] * x[k]
        end
        piv = one(C) + alpha * S[i, i]
        ap = abs(piv)
        ap < small && (small = ap)
        x[i] = acc / piv
    end
    return small
end

# =============================================================================
# Generalized Sylvester: A·X + B·X·C^{⊗d} = D
# =============================================================================

"""
    _sylv_triangular!(X, D, S, Tm, Tc, alpha, d, nv, piv_min)

Recursive back-substitution for the **transformed** equation `S·X + α·Tm·X·Tc^{⊗d} = D`, where
all three of `S`, `Tm`, `Tc` are upper triangular. Writes the solution into `X` (`n × nv^d`);
`D` is consumed.

## The recursion

Split the `nv^d` columns into `nv` blocks of `nv^{d-1}` (Kronecker convention: the FIRST index is
slowest-varying, matching `kron(A, B)[:, (ja-1)·nv + jb]`). With `P = Tc^{⊗(d-1)}` and
`Tc^{⊗d} = Tc ⊗ P`, block `m` of the equation reads

```
S·X_m + α·Tc[m,m]·Tm·X_m·P  =  D_m − α·Tm·(Σ_{l<m} Tc[l,m]·X_l)·P
```

because upper-triangularity of `Tc` kills every `l > m`. That is the SAME equation one order
down, with `α ← α·Tc[m,m]` — hence the recursion, bottoming out at `d = 0` on a single column and
the triangular pencil solve `(S + α·Tm)·x = r`. Blocks are visited in increasing `m`, so every
`X_l` appearing on the right is already known.

The `Σ_{l<m}` term is accumulated by scattering forward (each solved `X_m` is added into the
accumulators of all later blocks) rather than re-summed per block, and `·P` is applied through
`_apply_kron_power` so the `nv^{d-1} × nv^{d-1}` operator is never formed.
"""
function _sylv_triangular!(X::AbstractMatrix{C}, D::AbstractMatrix{C},
                           S::AbstractMatrix{C}, Tm::AbstractMatrix{C}, Tc::AbstractMatrix{C},
                           alpha::C, d::Int, nv::Int,
                           piv_min::Base.RefValue{R}) where {C<:Complex,R<:Real}
    if d == 0
        p = _tri_pencil_solve!(view(X, :, 1), S, Tm, alpha, view(D, :, 1))
        p < piv_min[] && (piv_min[] = p)
        return X
    end

    n = size(X, 1)
    blk = nv^(d - 1)
    # Acc[m] accumulates Σ_{l<m} Tc[l,m]·X_l. Allocated lazily: a diagonal Tc (the common case
    # when C is already triangular) never touches an accumulator at all.
    Acc = Vector{Union{Nothing,Matrix{C}}}(nothing, nv)

    for m in 1:nv
        cols = ((m - 1) * blk + 1):(m * blk)
        Xm = view(X, :, cols)
        Rm = Matrix{C}(view(D, :, cols))
        Am = Acc[m]
        if Am !== nothing
            # Rm ← Rm − α·Tm·(Am·Tc^{⊗(d-1)})
            W = _apply_kron_power(Am, Tc, d - 1)
            mul!(Rm, Tm, W, -alpha, one(C))
            Acc[m] = nothing        # release before recursing; peak stays O(n·nv^d)
        end

        _sylv_triangular!(Xm, Rm, S, Tm, Tc, alpha * Tc[m, m], d - 1, nv, piv_min)

        @inbounds for mm in (m + 1):nv
            t = Tc[m, mm]
            iszero(t) && continue
            Amm = Acc[mm]
            if Amm === nothing
                Amm = zeros(C, n, blk)
                Acc[mm] = Amm
            end
            @. Amm += t * Xm
        end
    end
    return X
end

"""
    _sylvester_residual(A, B, C, X, D, d) → relative residual

Relative Frobenius residual of `A·X + B·X·C^{⊗d} − D`, computed matrix-free through
`_apply_kron_power`. Used to certify every solve produced here ([T145]): a direct method
still fails on a near-singular pencil, and it must say so rather than return a plausible-looking
tensor.
"""
function _sylvester_residual(A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T},
                             X::AbstractMatrix{T}, D::AbstractMatrix{T}, d::Int) where {T<:Real}
    R = A * X + B * _apply_kron_power(X, C, d) - D
    dn = norm(D)
    return dn > 0 ? norm(R) / dn : norm(R)
end

"""
    _kamenik_sylvester(A, B, C, D, d) → (X, info)

Solve the generalized Sylvester equation

```math
A X + B X \\, C^{\\otimes d} = D
```

for `X` (`n × nv^d`), where `A, B` are `n × n` and `C` is `nv × nv`, by the generalized-Schur
method of Kamenik (2005). At `d = 1` this is the ordinary generalized Sylvester equation
`A·X + B·X·C = D`.

Returns `(X, info)` where `info` is a `NamedTuple` with `:ok`, `:residual`, `:min_pivot` and
`:reason`. `X` is `nothing` when `info.ok` is `false`; the caller decides whether to fall back.
This never throws on a numerically hopeless problem — it reports.

## Method

1. Generalized Schur (QZ) of the pencil: `A = Q·S·Zᴴ`, `B = Q·Tm·Zᴴ` with `S`, `Tm` upper
   triangular. Working on the PENCIL rather than on `A⁻¹B` is what lets a **singular `A`** through
   — and `f_c` is singular for any model with a purely static or purely forward-looking equation,
   even though the full Sylvester operator stays invertible.
2. Complex Schur of the other operator: `C = Z_c·T_c·Z_cᴴ`, `T_c` upper triangular.
3. Change variables to `Ỹ = Zᴴ·X·Z_c^{⊗d}`. Since `(Z_cᴴ)^{⊗d}·C^{⊗d} = T_c^{⊗d}·(Z_cᴴ)^{⊗d}`,
   the equation becomes `S·Ỹ + Tm·Ỹ·T_c^{⊗d} = D̃` with `D̃ = Qᴴ·D·Z_c^{⊗d}`, in which ALL
   THREE operators are triangular.
4. Back-substitute (`_sylv_triangular!`).
5. Transform back: `X = Z·Ỹ·(Z_cᴴ)^{⊗d}`.

The Kronecker powers in steps 3 and 5 are applied through `_apply_kron_power`, so the
`nv^d × nv^d` operator is never materialized at any point.

The equation is solvable exactly when no generalized eigenvalue `−S[i,i]/Tm[i,i]` of the pencil
coincides with a `d`-fold product of eigenvalues of `C`; `info.min_pivot` (scaled by the pencil
norm) measures the distance to that failure.

## Cost

`O(n³ + nv³)` for the two decompositions, then `O(n²·nv^d)` triangular solves and
`O(d²·n·nv^{d+1})` Kronecker applications — against `O(n³·nv^{3d})` time and `O(n²·nv^{2d})`
memory for the vectorized dense solve. For a model with `n = 60` states at order 3 the dense
operator alone would not fit in memory.
"""
function _kamenik_sylvester(A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T},
                            D::AbstractMatrix{T}, d::Int;
                            pivot_tol::Real=1e-12) where {T<:AbstractFloat}
    n = size(A, 1)
    nv = size(C, 1)
    size(A) == (n, n) || throw(ArgumentError("A must be square, got $(size(A))"))
    size(B) == (n, n) || throw(ArgumentError("B must be $n×$n, got $(size(B))"))
    size(C) == (nv, nv) || throw(ArgumentError("C must be square, got $(size(C))"))
    d >= 0 || throw(ArgumentError("d must be non-negative, got $d"))
    size(D) == (n, nv^d) || throw(ArgumentError(
        "D must be $n×$(nv^d) for d=$d, got $(size(D))"))

    fail(reason) = (nothing, (ok=false, residual=T(Inf), min_pivot=zero(T), reason=reason))
    (n == 0 || nv^d == 0) && return (zeros(T, n, nv^d),
                                     (ok=true, residual=zero(T), min_pivot=zero(T), reason=:empty))

    CT = Complex{T}

    # Steps 1-2 — QZ of the (A, B) pencil and Schur of C; all factors upper triangular.
    local S, Tm, Q, Z, Tc, Zc
    try
        F = LinearAlgebra.schur(Matrix{CT}(A), Matrix{CT}(B))
        S, Tm, Q, Z = F.S, F.T, F.Q, F.Z
        CF = LinearAlgebra.schur(Matrix{CT}(C))
        Tc, Zc = CF.T, CF.Z
    catch err
        err isa LinearAlgebra.LAPACKException || err isa LinearAlgebra.SingularException ||
            rethrow()
        return fail(:decomposition_failed)
    end

    # Step 3 — D̃ = Qᴴ·D·Z_c^{⊗d}
    Dt = _apply_kron_power(Q' * Matrix{CT}(D), Zc, d)

    # Step 4 — triangular back-substitution
    Xt = zeros(CT, n, nv^d)
    piv_min = Ref(typemax(T))
    _sylv_triangular!(Xt, Dt, S, Tm, Tc, one(CT), d, nv, piv_min)
    # A vanishing pivot IS the singular case, so report that rather than the non-finite entries
    # it produces downstream. Measured relative to the pencil scale, since Q and Z are unitary
    # and hence ‖S‖ ~ ‖A‖, ‖Tm‖ ~ ‖B‖ — an absolute threshold would be units-dependent.
    scale = max(opnorm(S, 1), opnorm(Tm, 1), one(T))
    piv_min[] > pivot_tol * scale || return (nothing, (ok=false, residual=T(Inf),
                                                       min_pivot=piv_min[], reason=:near_singular))
    all(isfinite, Xt) || return fail(:nonfinite)

    # Step 5 — X = Z·Ỹ·(Z_cᴴ)^{⊗d}; the original system is real, so the imaginary part is roundoff.
    Xc = Z * _apply_kron_power(Xt, Matrix(Zc'), d)
    X = real.(Xc)

    resid = _sylvester_residual(Matrix{T}(A), Matrix{T}(B), Matrix{T}(C), X, Matrix{T}(D), d)
    isfinite(resid) || return fail(:nonfinite)   # keep the contract: !ok ⟹ X === nothing
    return (X, (ok=true, residual=resid, min_pivot=piv_min[], reason=:converged))
end

# =============================================================================
# Discrete Lyapunov: P = A·P·A' + Q
# =============================================================================

"""
    _bartels_stewart_dlyap(A, Qm) → (P, info)

Solve the discrete Lyapunov equation `P = A·P·A' + Q` by the Bartels-Stewart / Barraud method:
complex-Schur `A = Z·S·Zᴴ`, solve the triangular equation `P̃ = S·P̃·Sᴴ + Q̃` one column at a
time, transform back.

Returns `(P, info)` with `info` carrying `:ok`, `:residual`, `:min_pivot` and `:reason`; `P` is
`nothing` when the solve fails. Direct — no iteration count, no convergence tolerance.

## Column sweep

With `W = S·P̃` and `Sᴴ[k,j] = conj(S[j,k])` nonzero only for `k ≥ j`, column `j` of
`P̃ = S·P̃·Sᴴ + Q̃` separates into

```
(I − conj(s_jj)·S) · p̃_j  =  q̃_j + Σ_{k>j} conj(S[j,k])·w_k
```

so sweeping `j = n, n−1, …, 1` has every `w_k` on the right already available. Each column is one
triangular solve, giving `O(n³)` overall.

The pivot `1 − conj(s_jj)·s_ii` vanishes exactly when `λ_i·conj(λ_j) = 1`, the classical
solvability condition; for a stable `A` (`|λ| < 1`) it is bounded away from zero, and
`info.min_pivot` reports how close the problem came.
"""
function _bartels_stewart_dlyap(A::AbstractMatrix{T}, Qm::AbstractMatrix{T};
                                pivot_tol::Real=1e-12) where {T<:AbstractFloat}
    n = size(A, 1)
    size(A) == (n, n) || throw(ArgumentError("A must be square, got $(size(A))"))
    size(Qm) == (n, n) || throw(ArgumentError("Q must be $n×$n, got $(size(Qm))"))
    n == 0 && return (zeros(T, 0, 0), (ok=true, residual=zero(T), min_pivot=zero(T), reason=:empty))

    CT = Complex{T}
    F = LinearAlgebra.schur(Matrix{CT}(A))
    S, Z = F.T, F.Z
    Qt = Z' * Matrix{CT}(Qm) * Z

    P = zeros(CT, n, n)
    W = zeros(CT, n, n)         # W[:, k] = S·P̃[:, k], reused by every earlier column
    rhs = Vector{CT}(undef, n)
    piv_min = typemax(T)

    @inbounds for j in n:-1:1
        copyto!(rhs, view(Qt, :, j))
        for k in (j + 1):n
            c = conj(S[j, k])
            iszero(c) && continue
            axpy!(c, view(W, :, k), rhs)
        end
        p = _tri_shift_solve!(view(P, :, j), S, -conj(S[j, j]), rhs)
        p < piv_min && (piv_min = p)
        mul!(view(W, :, j), S, view(P, :, j))
    end
    all(isfinite, P) || return (nothing, (ok=false, residual=T(Inf), min_pivot=piv_min,
                                          reason=:nonfinite))

    Pc = Z * P * Z'
    Pr = real.(Pc)
    Pr = (Pr + Pr') / 2         # the solution of a symmetric problem is symmetric

    R = A * Pr * A' + Qm - Pr
    qn = norm(Qm)
    resid = qn > 0 ? norm(R) / qn : norm(R)
    ok = isfinite(resid) && piv_min > pivot_tol
    return (Pr, (ok=ok, residual=resid, min_pivot=piv_min,
                 reason=ok ? :converged : :near_singular))
end

"""
    _dlyap(A, Qm; tol=1e-8, warn_label="") → Matrix

Solve `P = A·P·A' + Q`, certifying the result: the doubling iteration
[`_dlyap_doubling`](@ref) runs first, and `_bartels_stewart_dlyap` takes over when
doubling's relative residual misses `tol`.

## Why doubling leads

The reverse order looks more natural — Bartels-Stewart is direct, doubling is iterative — but
measurement says otherwise. Doubling is a handful of `O(n³)` BLAS-3 matrix products and
converges quadratically (`ρ(A_k) = ρ^{2^k}`), whereas Bartels-Stewart pays for a COMPLEX Schur
decomposition plus a column sweep that is not BLAS-3. At `n = 400` doubling takes ~80 ms against
~750 ms, and both land at a relative residual near `1e-14`; the same ordering holds from `n = 50`
up. So doubling is the fast path, and the direct method is what catches the cases doubling
cannot: its convergence test `max|B_{k+1} − B_k| < 1e-12` is ABSOLUTE, so it can stop early when
`‖Q‖` is small or grind through all 500 squarings when `‖Q‖` is large or `ρ(A) → 1`.

The gate is the **measured residual**, not a spectral-radius heuristic: the two methods degrade
for different reasons and no single threshold on `ρ(A)` separates them. Checking costs two
`O(n³)` products against doubling's ~15.

Both are `O(n³)`-class and replace the `O(n⁶)` dense `(I − A⊗A)⁻¹` solve this function
supplanted, which formed an `n²×n²` matrix — 11.9 GB at `n = 200`.

If BOTH methods miss `tol`, the more accurate result is returned together with a warning;
nothing is ever returned silently unconverged ([T145]).
"""
function _dlyap(A::AbstractMatrix{T}, Qm::AbstractMatrix{T};
                tol::Real=1e-8, warn_label::AbstractString="") where {T<:AbstractFloat}
    qn = norm(Qm)
    relresid(P) = begin
        r = norm(A * P * A' + Qm - P)
        qn > 0 ? r / qn : r
    end

    # Doubling's own non-convergence warning is premature here — the Bartels-Stewart fallback
    # below may well rescue it — so hold it back and report once, at the end, if both fail.
    Pd = _suppress_warnings() do
        _dlyap_doubling(A, Qm)
    end
    rd = all(isfinite, Pd) ? relresid(Pd) : T(Inf)
    rd <= tol && return Pd

    P, info = _bartels_stewart_dlyap(A, Qm)
    (P !== nothing && info.ok && info.residual <= tol) && return P

    # Neither certified: keep the better one and say so. A diverging doubling iteration returns
    # Inf/NaN, so compare on finiteness first — `info.residual < NaN` is false and would
    # otherwise hand back the garbage.
    bs_ok = P !== nothing && isfinite(info.residual)
    best_is_bs = bs_ok && !(isfinite(rd) && rd < info.residual)
    lbl = isempty(warn_label) ? "" : " ($warn_label)"
    @warn "Discrete Lyapunov solve did not reach tol=$tol$lbl: doubling relative residual " *
          "$rd, Bartels-Stewart relative residual " *
          "$(P === nothing ? "failed ($(info.reason))" : string(info.residual)). " *
          "Returning the more accurate of the two; the unconditional covariance may be " *
          "inaccurate (near-unit-root transition?)." maxlog = 1
    return best_is_bs ? P : Pd
end
