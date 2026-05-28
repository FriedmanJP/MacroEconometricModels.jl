# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
    _solve_qz_quadratic(f_0, f_1, f_lead, f_ε; div=1.0 + 1e-8)
        → (G, impact, eigenvalues, n_stable, eu, residual)

Solve the quadratic matrix equation `f_lead·G² + f_0·G + f_1 = 0` for the unique stable
solvent `G` (n×n) via the QZ (generalized Schur) decomposition of its companion pencil, and
recover the shock impact `M = -(f_0 + f_lead·G)⁻¹·f_ε`.

This is the correct Klein (2000) / Blanchard-Kahn (1980) treatment of a linear rational-
expectations model `f_lead·E_t[y_{t+1}] + f_0·y_t + f_1·y_{t-1} + f_ε·ε_t = 0`, valid for
models with forward-looking variables, lags, and static equations alike.

Companion pencil `L·x = λ·M·x` with `x = [a; λ·a]` and `(f_lead·λ² + f_0·λ + f_1)·a = 0`:

    L = [ 0     I    ]      M = [ I   0      ]
        [ -f_1  -f_0 ]          [ 0   f_lead ]

Determinacy is read off the companion's stable-root count `n_stable` (roots with `|λ| < div`):
`n_stable == n` → determinate `[1,1]`; `> n` → indeterminate `[1,0]`; `< n` → no stable
solution `[0,0]`. `residual = ‖f_lead·G² + f_0·G + f_1‖∞` is a convention-independent
self-check on the recovered `G`.

The stable solvent is recovered as `G = Z_b · Z_t⁻¹`, where `[Z_t; Z_b]` are the top/bottom
n-row blocks of the first `n` (stable) columns of the ordered right Schur vectors `Z`.

The returned `eigenvalues` are the UNORDERED companion generalized eigenvalues (diagnostic
only); do NOT reorder them — wrappers report `eigvals(G1)` separately.
"""
function _solve_qz_quadratic(f_0::AbstractMatrix{T}, f_1::AbstractMatrix{T},
        f_lead::AbstractMatrix{T}, f_ε::AbstractMatrix{T};
        div::Real=1.0 + 1e-8) where {T<:AbstractFloat}
    f0 = Matrix{T}(f_0); f1 = Matrix{T}(f_1); flead = Matrix{T}(f_lead); fε = Matrix{T}(f_ε)
    n = size(f0, 1)
    N = 2n
    Z0 = zeros(T, n, n)
    In = Matrix{T}(I, n, n)

    # Companion pencil
    L = [Z0    In;
         -f1   -f0]
    M = [In    Z0;
         Z0    flead]

    F = schur(complex(L), complex(M))
    λ = F.values                                   # 2n generalized eigenvalues (Inf where β≈0)

    stable_select = BitVector(abs.(λ) .< T(div))
    n_stable = count(stable_select)

    eu = n_stable == n ? [1, 1] : (n_stable > n ? [1, 0] : [0, 0])

    G = zeros(T, n, n)
    if n_stable >= n
        # For n_stable > n (indeterminate) this returns one representative stable solvent;
        # callers treat eu[2] == 0 as indeterminate.
        Fo = ordschur(F, stable_select)
        Zt = Fo.Z[1:n, 1:n]
        Zb = Fo.Z[n+1:N, 1:n]
        if rank(Zt) == n
            G = real((Zt' \ Zb')')                 # G = Zb·Zt⁻¹ via backslash
        else
            eu = [eu[1], 0]
        end
    end

    A = f0 + flead * G
    impact = try
        Matrix{T}(-(A \ fε))
    catch
        fill(T(NaN), n, size(fε, 2))
    end

    residual = maximum(abs.(flead * G * G + f0 * G + f1); init = zero(T))

    (G=G, impact=impact, eigenvalues=Vector{ComplexF64}(λ),
     n_stable=n_stable, eu=eu, residual=residual)
end
