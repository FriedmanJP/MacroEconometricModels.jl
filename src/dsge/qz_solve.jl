# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
    _place_divhat(mags, div, cluster_tol) -> divhat

Determinacy boundary for stable-root counting. `mags` are the moduli of the generalized
eigenvalues. Returns the caller's nominal `div` unchanged, but emits a WARNING when MULTIPLE
eigenvalues cluster within `cluster_tol` of the unit circle — the case where the
stable/unstable split is numerically delicate and the count can flip under tiny perturbations.

The nominal `div = 1 + 1e-8` is the Sims (2002) convention that treats a lone unit root as
non-explosive (stable); deliberately reclassifying every `|λ| = 1` as unstable would break
legitimate unit-root models (e.g. a random-walk technology, or a knife-edge ρ+β=1 forward
model), so the resolution is left to the caller: pass an explicit `div` (e.g. `div < 1`) to
`_solve_qz_quadratic`/`gensys`/`klein`/`blanchard_kahn` to force a strict split, and read the
warning to know when it matters.

This is the SINGLE boundary helper shared by `gensys()` and `_solve_qz_quadratic`, so every
determinacy path agrees on the threshold and the warning.
"""
function _place_divhat(mags::AbstractVector{<:Real}, div::Real, cluster_tol::Real)
    Tf = float(promote_type(eltype(mags), typeof(div), typeof(cluster_tol)))
    dv = Tf(div)
    finite = filter(isfinite, mags)
    isempty(finite) && return dv
    near_unit = count(m -> abs(m - one(Tf)) <= Tf(cluster_tol), finite)
    near_unit >= 2 && @warn "DSGE determinacy: $near_unit generalized eigenvalues cluster " *
        "within cluster_tol=$cluster_tol of |λ|=1; the stable/unstable split is numerically " *
        "delicate — pass an explicit `div` to force a deterministic classification."
    dv
end

# =============================================================================
# Sims (2002) existence / uniqueness rank conditions
# =============================================================================
#
# The determinacy verdict `eu` used to be a Blanchard-Kahn ROOT COUNT: compare the number of
# stable generalized eigenvalues against the number of states. That count is a theorem only
# under a rank condition that root-counting never checks, and it misclassifies exactly the cases
# where the rank condition bites — redundant expectational errors, shocks that load on the
# unstable block in a direction the errors cannot absorb, and other degenerate configurations.
# Sims (2002) states existence and uniqueness directly as rank conditions on the QZ-decomposed
# system, and that is what is computed here ([T267]).

"""
    _orth_basis(A, rtol) → (U_r, V_r, rank)

Orthonormal bases for the COLUMN space (`U_r`) and ROW space (`V_r`) of `A`, together with the
numerical rank, taken from the singular vectors whose singular values clear a **scale-relative**
threshold `rtol · σ_max` (`rtol < 0` selects the default `max(size(A)) · eps`). The threshold is
relative because an absolute cutoff would make the determinacy verdict depend on the units the
model happens to be written in.
"""
function _orth_basis(A::AbstractMatrix, rtol::Real)
    m, k = size(A)
    if m == 0 || k == 0
        C = eltype(A)
        return (zeros(C, m, 0), zeros(C, k, 0), 0)
    end
    F = svd(A)
    smax = maximum(F.S; init=zero(eltype(F.S)))
    tol = (rtol < 0 ? maximum(size(A)) * eps(Float64) : Float64(rtol)) * smax
    r = count(s -> s > tol, F.S)
    return (F.U[:, 1:r], F.V[:, 1:r], r)
end

"""
    _sims_rank_eu(Qp, nstab, Psi, Pi; rank_rtol=1e-8) → NamedTuple

Sims (2002) existence/uniqueness verdict from an ORDERED generalized Schur decomposition, with
the stable block first. `Qp` is `Q'` from `schur(Γ0, Γ1)` (so `Q'Γ0 Z = S`), `nstab` the number
of stable generalized eigenvalues, and `Psi`/`Pi` the shock and expectational-error loadings of
`Γ0 s_t = Γ1 s_{t-1} + Ψ ε_t + Π η_t`.

Premultiplying the system by `Q'` splits it into a stable block (rows `1:nstab`) and an unstable
block (rows `nstab+1:end`). The unstable block must be killed for the solution to be bounded,
which turns into two conditions:

**Existence** — for every shock realization there must exist an expectational error that offsets
it on the unstable block, i.e. `Q₂Ψ v = -Q₂Π η` must be solvable for every `v`:

```
col span(Q₂Ψ) ⊆ col span(Q₂Π)
```

Sims's own `gensys.m` tests the stronger `rank(Q₂Π) ≥ n_unstable` (full row rank), which is
sufficient but not necessary — it declares non-existence for models where `Q₂Ψ` happens to lie
in a deficient span. Both are reported: `existence` is the span condition, `full_row_rank` the
sufficient one.

**Uniqueness** — no expectational-error direction may move the stable block while being left
free by the unstable block:

```
row span(Q₁Π) ⊆ row span(Q₂Π)
```

The singular values of the residual `V₁ − V₂V₂ᴴV₁` (with `V₁`, `V₂` orthonormal bases of the two
row spaces) are the **sines of the principal angles** between them, so they live in `[0, 1]` and
`rank_rtol` is a genuine angle cutoff rather than a scale-dependent guess: a truly loose
direction registers a singular value near 1, orders of magnitude above any sane threshold.

Returns `(eu, n_loose, rank_Q2Pi, rank_Q1Pi, existence_residual, full_row_rank)`, where
`eu = [existence, uniqueness]` with `1 = yes`.
"""
function _sims_rank_eu(Qp::AbstractMatrix, nstab::Int, Psi::AbstractMatrix,
                       Pi::AbstractMatrix; rank_rtol::Real=1e-8)
    N = size(Qp, 1)
    nunstab = N - nstab
    Q1 = Qp[1:nstab, :]
    Q2 = Qp[nstab+1:N, :]
    Q2Pi = Q2 * Pi
    Q2Psi = Q2 * Psi
    Q1Pi = Q1 * Pi

    U2, V2, r2 = _orth_basis(Q2Pi, rank_rtol)

    # --- Existence: is every column of Q₂Ψ in the column span of Q₂Π? ---
    # A zero `nz` means the unstable block carries no shock at all, so there is nothing to
    # offset and existence is automatic (this also covers nunstab == 0, where Q₂Ψ is empty).
    nz = norm(Q2Psi)
    exist_resid = nz > 0 ? norm(Q2Psi - U2 * (U2' * Q2Psi)) / nz : 0.0
    exists = exist_resid <= rank_rtol

    # --- Uniqueness: is the row span of Q₁Π inside the row span of Q₂Π? ---
    _, V1, r1 = _orth_basis(Q1Pi, rank_rtol)
    n_loose = if r1 == 0
        0                          # no expectational error reaches the stable block
    elseif r2 == 0
        r1                         # ... and none of them is pinned down
    else
        loose = V1 - V2 * (V2' * V1)
        # Singular values here are sines of principal angles ∈ [0,1]; no rescaling needed.
        sv = svdvals(loose)
        count(s -> s > rank_rtol, sv)
    end

    return (eu=[exists ? 1 : 0, n_loose == 0 ? 1 : 0],
            n_loose=n_loose, rank_Q2Pi=r2, rank_Q1Pi=r1,
            existence_residual=exist_resid, full_row_rank=(r2 >= nunstab))
end

"""
    _sims_augmented_system(f_0, f_1, f_lead, f_ε; lead_rtol=1e-10) → (Γ0, Γ1, Ψ, Π, fwd)

Rewrite `f_lead·E_t[y_{t+1}] + f_0·y_t + f_1·y_{t-1} + f_ε·ε_t = 0` in the Sims canonical form
`Γ0 s_t = Γ1 s_{t-1} + Ψ ε_t + Π η_t`, which is what the rank conditions are stated over.

With `ξ_t = E_t[y_{t+1}]` restricted to the variables that ACTUALLY appear with a lead
(`fwd = {j : f_lead[:,j] ≠ 0}`, `k = |fwd|`) and `s_t = [y_t; ξ_t]`:

```
Γ0 = [ f_0     f_lead[:,fwd] ]   Γ1 = [ -f_1   0  ]   Ψ = [ -f_ε ]   Π = [ 0  ]
     [ E_fwd   0             ]        [  0     I  ]       [  0   ]       [ I_k ]
```

where the second block row is the definition `y_t[fwd] = ξ_{t-1} + η_t`.

Restricting to `fwd` is not an optimization — it is required for correctness. Giving every
variable an expectational error would hand the uniqueness test `k = n` free errors, most of them
attached to no lead at all, and those spurious errors register as loose directions: the model
would be declared indeterminate purely because of how the system was written down ([T124]).

The pencil `(Γ1, Γ0)` has the same finite generalized eigenvalues as the companion pencil of the
quadratic, since `det(Γ1 − μΓ0) = ±det(f_lead·μ² + f_0·μ + f_1)`.
"""
function _sims_augmented_system(f_0::AbstractMatrix{T}, f_1::AbstractMatrix{T},
                                f_lead::AbstractMatrix{T}, f_ε::AbstractMatrix{T};
                                lead_rtol::Real=1e-10) where {T<:AbstractFloat}
    n = size(f_0, 1)
    scale = maximum(abs, f_lead; init=zero(T))
    tol = T(lead_rtol) * max(one(T), scale)
    fwd = [j for j in 1:n if maximum(abs, view(f_lead, :, j); init=zero(T)) > tol]
    k = length(fwd)
    n_eps = size(f_ε, 2)

    Gamma0 = zeros(T, n + k, n + k)
    Gamma0[1:n, 1:n] = f_0
    Gamma1 = zeros(T, n + k, n + k)
    Gamma1[1:n, 1:n] = -f_1
    Psi = zeros(T, n + k, n_eps)
    Psi[1:n, :] = -f_ε
    Pi = zeros(T, n + k, k)
    @inbounds for (i, j) in enumerate(fwd)
        Gamma0[1:n, n+i] = view(f_lead, :, j)
        Gamma0[n+i, j] = one(T)          # E_fwd selection row
        Gamma1[n+i, n+i] = one(T)        # ξ_{t-1}
        Pi[n+i, i] = one(T)              # η_t
    end
    return (Gamma0, Gamma1, Psi, Pi, fwd)
end

"""
    _sims_existence_uniqueness(f_0, f_1, f_lead, f_ε; divhat, rank_rtol=1e-8) → NamedTuple

Sims (2002) determinacy verdict for the quadratic system `f_lead·G² + f_0·G + f_1 = 0`.
Builds the Sims canonical form ([`_sims_augmented_system`](@ref)), orders its QZ by the
stable/unstable split at `divhat`, and applies the rank conditions ([`_sims_rank_eu`](@ref)).

`divhat` is passed in already placed by [`_place_divhat`](@ref) so the caller's near-unit-root
warning fires exactly once.
"""
function _sims_existence_uniqueness(f_0::AbstractMatrix{T}, f_1::AbstractMatrix{T},
                                    f_lead::AbstractMatrix{T}, f_ε::AbstractMatrix{T};
                                    divhat::Real, rank_rtol::Real=1e-8) where {T<:AbstractFloat}
    Gamma0, Gamma1, Psi, Pi, fwd = _sims_augmented_system(f_0, f_1, f_lead, f_ε)
    N = size(Gamma0, 1)
    N == 0 && return (eu=[1, 1], n_loose=0, rank_Q2Pi=0, rank_Q1Pi=0,
                      existence_residual=0.0, full_row_rank=true, nstab=0, n_expect=0)

    # REAL QZ, and of the pencil in the order (Γ1, Γ0). Two deliberate choices, both for speed
    # on what is a per-draw call inside Bayesian estimation:
    #   * (Γ1, Γ0) makes `F.values` the transition eigenvalues of Γ0⁻¹Γ1 directly, with no
    #     reciprocal and no special-casing of infinite roots.
    #   * the rank conditions need only the ORDERED `Q'` — never a triangular solve — so the
    #     2×2 blocks of the real Schur form are harmless, and a real QZ is ~3-4× cheaper than
    #     the complex one. A conjugate pair shares a modulus, so `|μ| < divhat` selects both
    #     or neither and `ordschur` never has to split a block.
    F = LinearAlgebra.schur(Matrix{T}(Gamma1), Matrix{T}(Gamma0))
    stable = BitVector(abs.(F.values) .< divhat)
    Fo = LinearAlgebra.ordschur(F, stable)
    nstab = count(stable)

    res = _sims_rank_eu(Fo.Q', nstab, Psi, Pi; rank_rtol=rank_rtol)
    return (res..., nstab=nstab, n_expect=length(fwd))
end

"""
    _solve_qz_quadratic(f_0, f_1, f_lead, f_ε; div=1.0 + 1e-8, cluster_tol=1e-6)
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

Determinacy `eu` is the **Sims (2002) existence/uniqueness rank verdict**
([`_sims_existence_uniqueness`](@ref)), not a root count. The Blanchard-Kahn count
(`n_stable == n` → `[1,1]`; `> n` → `[1,0]`; `< n` → `[0,0]`) is still computed and returned as
`eu_count` for diagnostics, and the two agree away from the degenerate cases that root-counting
gets wrong. `residual = ‖f_lead·G² + f_0·G + f_1‖∞` is a convention-independent self-check on
the recovered `G`.

The stable solvent is recovered as `G = Z_b · Z_t⁻¹`, where `[Z_t; Z_b]` are the top/bottom
n-row blocks of the first `n` (stable) columns of the ordered right Schur vectors `Z`.

The returned `eigenvalues` are the UNORDERED companion generalized eigenvalues (diagnostic
only); do NOT reorder them — wrappers report `eigvals(G1)` separately.
"""
function _solve_qz_quadratic(f_0::AbstractMatrix{T}, f_1::AbstractMatrix{T},
        f_lead::AbstractMatrix{T}, f_ε::AbstractMatrix{T};
        div::Real=1.0 + 1e-8, cluster_tol::Real=1e-6,
        rank_rtol::Real=1e-8, sparse::Union{Bool,Symbol}=:auto) where {T<:AbstractFloat}
    f0 = Matrix{T}(f_0); f1 = Matrix{T}(f_1); flead = Matrix{T}(f_lead); fε = Matrix{T}(f_ε)
    n = size(f0, 1)

    # Sparse/structured route ([T270]): Newton on the quadratic with matrix-free GMRES, which
    # skips the dense companion QZ entirely. It is only taken when it certifies (small residual
    # AND a stable solvent), so a model it cannot handle falls through to the dense core below
    # and is never answered wrongly. `eu` is NOT taken from here — it always comes from the
    # Sims rank test, so the verdict is identical on either route.
    sparse in (true, false, :auto) || throw(ArgumentError(
        "sparse must be true, false, or :auto; got $sparse"))
    use_sparse = sparse === true ? true :
                 sparse === false ? false : _should_use_sparse_klein(f0, f1, flead)
    if use_sparse
        G_sp, sp_info = _newton_solvent(f0, f1, flead; div=div)
        if G_sp !== nothing
            return _qz_quadratic_finish(f0, f1, flead, fε, G_sp, div, cluster_tol,
                                        rank_rtol, sp_info)
        end
        sparse === true && @warn "Sparse Klein did not certify ($(sp_info.reason), relative " *
            "residual $(sp_info.residual)); falling back to the dense companion-QZ core." maxlog = 1
    end
    N = 2n
    Z0 = zeros(T, n, n)
    In = Matrix{T}(I, n, n)

    # Companion pencil
    L = [Z0    In;
         -f1   -f0]
    M = [In    Z0;
         Z0    flead]

    F = schur(complex(L), complex(M))
    λ = F.values                                   # 2n generalized eigenvalues (NaN for infinite roots from singular M)

    mags = abs.(λ)
    divhat = _place_divhat(mags, div, cluster_tol) # exact |λ|=1 → unstable; standard models unchanged
    stable_select = BitVector(mags .< T(divhat))
    n_stable = count(stable_select)

    # Blanchard-Kahn root count — retained as a diagnostic, no longer the verdict ([T267]).
    eu_count = n_stable == n ? [1, 1] : (n_stable > n ? [1, 0] : [0, 0])

    # Sims (2002) rank conditions on the canonical form. `divhat` is passed through so the
    # near-unit-root warning above is not emitted a second time for the same spectrum.
    sims = _sims_existence_uniqueness(f0, f1, flead, fε; divhat=divhat, rank_rtol=rank_rtol)
    eu = copy(sims.eu)

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
    else
        # Fewer stable roots than states: no stable solvent exists at all, whatever the rank
        # conditions say about the canonical form.
        eu = [0, 0]
    end

    A = f0 + flead * G
    impact = try
        Matrix{T}(-(A \ fε))
    catch
        fill(T(NaN), n, size(fε, 2))
    end

    residual = maximum(abs.(flead * G * G + f0 * G + f1); init = zero(T))

    (G=G, impact=impact, eigenvalues=Vector{ComplexF64}(λ),
     n_stable=n_stable, eu=eu, residual=residual,
     eu_count=eu_count, sims=sims, sparse=nothing)
end

"""
    _qz_quadratic_finish(f0, f1, flead, fε, G, div, cluster_tol, rank_rtol, sparse_info) → NamedTuple

Assemble the `_solve_qz_quadratic` return value from an already-computed solvent `G`, used by
the sparse route ([T270]).

The determinacy verdict and the diagnostic eigenvalues still come from the same places as on
the dense route — `eu` from the Sims (2002) rank test, the eigenvalues from `G` itself — so the
two routes agree on everything but how `G` was found. `n_stable` is reported as the number of
stable eigenvalues of `G` (which is `n` by construction, since an unstable solvent is rejected),
and `eu_count` is the Blanchard-Kahn reading of that.
"""
function _qz_quadratic_finish(f0::Matrix{T}, f1::Matrix{T}, flead::Matrix{T}, fε::Matrix{T},
                              G::Matrix{T}, div::Real, cluster_tol::Real, rank_rtol::Real,
                              sparse_info) where {T<:AbstractFloat}
    n = size(f0, 1)
    divhat = _place_divhat(abs.(eigvals(G)), div, cluster_tol)
    sims = _sims_existence_uniqueness(f0, f1, flead, fε; divhat=divhat, rank_rtol=rank_rtol)

    A = f0 + flead * G
    impact = try
        Matrix{T}(-(A \ fε))
    catch
        fill(T(NaN), n, size(fε, 2))
    end
    residual = maximum(abs.(flead * G * G + f0 * G + f1); init = zero(T))
    eigs = Vector{ComplexF64}(eigvals(G))
    n_stable = count(<(divhat), abs.(eigs))

    return (G=G, impact=impact, eigenvalues=eigs, n_stable=n_stable,
            eu=copy(sims.eu), residual=residual,
            eu_count=(n_stable == n ? [1, 1] : [1, 0]), sims=sims, sparse=sparse_info)
end

"""
    _gensys_qz(spec, ld) -> (; G, impact, eu, …)

Route a linearized DSGE through the companion-QZ core ([`_solve_qz_quadratic`]). `gensys()`
counts determinacy from the (Γ0,Γ1) pencil, which folds the lead Jacobian into Π and drops it,
so its `eu`/`G1`/`impact` are wrong for any forward-looking model. This recovers the raw
Jacobians (`f_0 = Γ0`, `f_1 = -Γ1`, `f_lead` directly from `spec`, `f_ε = -Ψ`) and solves the
correct companion pencil — mirroring the split already used in `solve(:gensys)`. Use `.G` (not
`.G1`), `.impact`, `.eu`.
"""
_gensys_qz(spec, ld) = _solve_qz_quadratic(ld.Gamma0, -ld.Gamma1,
    _dsge_jacobian(spec, spec.steady_state, :lead), -ld.Psi)
