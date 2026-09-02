# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Structural VECM (SVEC) identification: King–Plosser–Stock–Watson (1991)
permanent/transitory decomposition with long-run restrictions via β⊥
(Lütkepohl 2005, §9.3; Gonzalo–Ng 2001).
"""

using LinearAlgebra

# =============================================================================
# Result
# =============================================================================

"""
    SVECResult{T} <: AbstractAnalysisResult

Structural VECM identification. Permanent shocks occupy the leading
`n_permanent = n − r` columns of `B0`; transitory shocks have zero long-run
impact (`Ξ B₀` has `r` zero columns).

# Fields
- `B0`: contemporaneous impact ``u_t = B_0 \\varepsilon_t``
- `Q`: rotation ``Q = L^{-1} B_0`` with ``L = \\mathrm{chol}(\\hat\\Sigma)``
- `Xi`: long-run impact of reduced-form shocks (Granger representation)
- `n_permanent`: number of common trends ``n - r``
- `vecm`: the estimated [`VECMModel`](@ref)
- `identification`: [`IdentificationStatus`](@ref)
"""
struct SVECResult{T<:AbstractFloat} <: AbstractAnalysisResult
    B0::Matrix{T}
    Q::Matrix{T}
    Xi::Matrix{T}
    n_permanent::Int
    vecm::VECMModel{T}
    identification::IdentificationStatus
end

function Base.show(io::IO, r::SVECResult{T}) where {T}
    n = size(r.B0, 1)
    spec = Any[
        "Variables"          n;
        "Cointegrating rank" n - r.n_permanent;
        "Permanent shocks"   r.n_permanent;
        "Transitory shocks"  n - r.n_permanent;
        "Identification"     String(r.identification.status);
    ]
    _pretty_table(io, spec;
        title = "Structural VECM (SVEC)",
        column_labels = ["", ""],
        alignment = [:l, :r])
    names = r.vecm.varnames
    snames = vcat(["P$i" for i in 1:r.n_permanent],
                  ["T$i" for i in 1:(n - r.n_permanent)])
    _matrix_table(io, r.B0, "Impact B₀"; row_labels=names, col_labels=snames)
    _matrix_table(io, r.Xi * r.B0, "Long-run ΞB₀"; row_labels=names, col_labels=snames)
    return nothing
end

# =============================================================================
# Granger long-run matrix
# =============================================================================

"""Ψ = I − Σᵢ Γᵢ from the VECM short-run lag polynomial."""
function _svec_psi(vecm::VECMModel{T}) where {T<:AbstractFloat}
    n = nvars(vecm)
    Psi = Matrix{T}(I, n, n)
    @inbounds for G in vecm.Gamma
        Psi .-= G
    end
    Psi
end

"""
Long-run impact ``Ξ = β_⊥ (α_⊥' Ψ β_⊥)^{-1} α_⊥'`` (Johansen 1991;
Lütkepohl 2005, eq. 9.2.13). Rank ``n - r``.
"""
function _svec_xi(alpha::AbstractMatrix{T}, beta::AbstractMatrix{T},
                  Psi::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = size(alpha, 1)
    r = size(alpha, 2)
    n_perm = n - r
    n_perm == 0 && return zeros(T, n, n)
    r == 0 && return Matrix{T}(robust_inv(Psi; silent=true))
    aperp = nullspace(Matrix{T}(alpha'))
    bperp = nullspace(Matrix{T}(beta'))
    size(aperp, 2) == n_perm || throw(ArgumentError(
        "α has rank $(n - size(aperp, 2)); expected cointegrating rank $r"))
    size(bperp, 2) == n_perm || throw(ArgumentError(
        "β has rank $(n - size(bperp, 2)); expected cointegrating rank $r"))
    Mid = aperp' * Psi * bperp
    bperp * (robust_inv(Matrix{T}(Mid); silent=true) * aperp')
end

# =============================================================================
# KPSW closed form
# =============================================================================

function _svec_sign_normalize!(B0::Matrix{T}, Xi::AbstractMatrix{T},
                               n_perm::Int) where {T<:AbstractFloat}
    n = size(B0, 1)
    lr = Xi * B0
    @inbounds for j in 1:n_perm
        pivot = n_perm == 1 ? lr[1, j] : lr[min(j, n), j]
        pivot < 0 && (B0[:, j] .*= -one(T))
    end
    @inbounds for j in (n_perm + 1):n
        B0[j, j] < 0 && (B0[:, j] .*= -one(T))
    end
    B0
end

"""Just-identified KPSW B₀: lower-triangular ΞB₀ on the permanent block and
Cholesky on the transitory block (Gonzalo–Ng orthogonalisation + rotation)."""
function _svec_kpsw_B0(alpha::AbstractMatrix{T}, beta::AbstractMatrix{T},
                       Sigma::AbstractMatrix{T}, Xi::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = size(Sigma, 1)
    r = size(alpha, 2)
    n_perm = n - r
    Σ = Matrix{T}(Sigma)

    if n_perm == 0
        B0 = Matrix{T}(safe_cholesky(Σ))
        return _svec_sign_normalize!(B0, Xi, n_perm)
    end
    if r == 0
        V = Xi * Σ * Xi'
        D = Matrix{T}(safe_cholesky(Matrix{T}((V + V') / 2)))
        B0 = Matrix{T}(robust_inv(Xi; silent=true) * D)
        return _svec_sign_normalize!(B0, Xi, n_perm)
    end

    aperp = nullspace(Matrix{T}(alpha'))
    G = vcat(aperp', Matrix{T}(beta'))
    GΣG = G * Σ * G'
    Pchol = Matrix{T}(safe_cholesky(Matrix{T}((GΣG + GΣG') / 2)))
    B0 = Matrix{T}(robust_inv(G; silent=true) * Pchol)

    if n_perm >= 2
        C_perm = Xi * B0[:, 1:n_perm]
        M = C_perm[1:n_perm, :]
        qrM = qr(M')
        Qrot = Matrix{T}(qrM.Q)
        Rlower = M * Qrot
        @inbounds for j in 1:n_perm
            Rlower[j, j] < 0 && (Qrot[:, j] .*= -one(T))
        end
        B0[:, 1:n_perm] = B0[:, 1:n_perm] * Qrot
    end
    _svec_sign_normalize!(B0, Xi, n_perm)
end

function _svec_kpsw_pattern(n::Int, n_perm::Int, ::Type{T}=Float64) where {T<:AbstractFloat}
    n >= 1 || throw(ArgumentError("n must be positive"))
    (0 <= n_perm <= n) || throw(ArgumentError("n_permanent must be in 0:$n"))
    A = Matrix{T}(I, n, n)
    B = fill(T(NaN), n, n)
    lr = fill(T(NaN), n, n)
    @inbounds for j in (n_perm + 1):n, i in 1:n
        lr[i, j] = zero(T)
    end
    @inbounds for j in 1:n_perm, i in 1:(j - 1)
        lr[i, j] = zero(T)
    end
    @inbounds for j in (n_perm + 1):n, i in (n_perm + 1):(j - 1)
        B[i, j] = zero(T)
    end
    SVARPattern(A, B; long_run=lr)
end

function _svec_default_status(n::Int, n_perm::Int)
    r = n - n_perm
    orders = zeros(Int, n)
    @inbounds for j in 1:n_perm
        orders[j] = j - 1
    end
    @inbounds for k in 1:r
        orders[n_perm + k] = n_perm + (k - 1)
    end
    IdentificationStatus(:exact, copy(orders), orders, 0)
end

"""Zero the transitory columns of ``Ξ B_0`` (KPSW: those shocks have no long-run impact)."""
function _svec_force_transitory_zeros!(lr::AbstractMatrix{T}, n_perm::Int) where {T}
    n = size(lr, 1)
    @inbounds for j in (n_perm + 1):n, i in 1:n
        lr[i, j] = zero(T)
    end
    lr
end

function _svec_has_transitory_zeros(lr, n_perm::Int)
    lr === nothing && return false
    n = size(lr, 1)
    @inbounds for j in (n_perm + 1):n, i in 1:n
        v = lr[i, j]
        (isnan(v) || !iszero(v)) && return false
    end
    true
end

function _svec_resolve_pattern(n::Int, n_perm::Int, pattern,
                               long_run_zeros, short_run_zeros, ::Type{T}) where {T}
    if pattern !== nothing
        pattern isa SVARPattern || throw(ArgumentError("pattern must be an SVARPattern"))
        pat = _ab_promote(pattern, T)
        # A complete pattern that already encodes the transitory-zero columns
        # of ΞB₀ is left unchanged; otherwise merge those KPSW PT zeros in.
        if _svec_has_transitory_zeros(pat.long_run, n_perm)
            return pat
        end
        lr = pat.long_run === nothing ? fill(T(NaN), n, n) : copy(pat.long_run)
        _svec_force_transitory_zeros!(lr, n_perm)
        return SVARPattern(pat.A, pat.B; long_run=lr)
    end
    pat0 = _svec_kpsw_pattern(n, n_perm, T)
    A = copy(pat0.A)
    B = copy(pat0.B)
    lr = copy(pat0.long_run)
    if short_run_zeros !== nothing
        size(short_run_zeros) == (n, n) || throw(ArgumentError(
            "short_run_zeros must be $n×$n, got $(size(short_run_zeros))"))
        B = Matrix{T}(short_run_zeros)
    end
    if long_run_zeros !== nothing
        size(long_run_zeros) == (n, n) || throw(ArgumentError(
            "long_run_zeros must be $n×$n, got $(size(long_run_zeros))"))
        user = Matrix{T}(long_run_zeros)
        # Overlay user zeros onto a long-run matrix that already has the
        # transitory columns of ΞB₀ at zero (do not drop the PT restrictions).
        lr = fill(T(NaN), n, n)
        _svec_force_transitory_zeros!(lr, n_perm)
        @inbounds for j in 1:n, i in 1:n
            v = user[i, j]
            isnan(v) || (lr[i, j] = v)
        end
        _svec_force_transitory_zeros!(lr, n_perm)
    end
    SVARPattern(A, B; long_run=lr)
end

# =============================================================================
# Public API
# =============================================================================

"""
    identify_svec(vecm; long_run_zeros=nothing, short_run_zeros=nothing,
                  pattern=nothing) -> SVECResult

Identify a structural VECM (King, Plosser, Stock and Watson 1991; Lütkepohl
2005, §9.3). With cointegrating rank ``r`` there are ``n-r`` permanent and
``r`` transitory shocks. The long-run impact of reduced-form shocks is

```math
Ξ = β_⊥ (α_⊥' Ψ β_⊥)^{-1} α_⊥', \\qquad Ψ = I - \\sum_{i=1}^{p-1} Γ_i,
```

so ``Ξ B_0`` has rank ``n-r`` and ``r`` columns that are identically zero.

The default is the Gonzalo–Ng closed form ``G = [α_⊥'; β']``,
``B_0 = G^{-1} \\mathrm{chol}(G Σ G')``, with a QR rotation of the permanent
block so that ``Ξ B_0`` is lower triangular (just-identified KPSW). Custom
zeros are imposed by `long_run_zeros` / `short_run_zeros` (`NaN` = free, a
finite number = fixed) or by a full [`SVARPattern`](@ref) and estimated by
[`estimate_svar`](@ref) as Lütkepohl's B-model (``A = I``, zeros on ``B_0``
and ``Ξ B_0``). The two numerical routes need not match. User zeros are merged
with the KPSW transitory-zero columns of ``Ξ B_0``; a complete `pattern` that
already encodes those columns is left unchanged.
"""
function identify_svec(vecm::VECMModel{T};
                       long_run_zeros=nothing,
                       short_run_zeros=nothing,
                       pattern=nothing,
                       n_starts::Int=5,
                       max_iter::Int=400,
                       rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    n = nvars(vecm)
    r = vecm.rank
    n_perm = n - r
    Psi = _svec_psi(vecm)
    Xi = _svec_xi(vecm.alpha, vecm.beta, Psi)
    Sigma = Matrix{T}(vecm.Sigma)
    use_default = pattern === nothing && long_run_zeros === nothing &&
                  short_run_zeros === nothing

    if use_default
        B0 = _svec_kpsw_B0(vecm.alpha, vecm.beta, Sigma, Xi)
        L = safe_cholesky(Sigma)
        Q = L \ B0
        return SVECResult{T}(Matrix{T}(B0), Matrix{T}(Q), Matrix{T}(Xi),
                             n_perm, vecm, _svec_default_status(n, n_perm))
    end

    pat = _svec_resolve_pattern(n, n_perm, pattern, long_run_zeros,
                                short_run_zeros, T)
    var = to_var(vecm)
    svar = estimate_svar(var, pat; n_starts=n_starts, max_iter=max_iter,
                         rng=rng, long_run_matrix=Xi)
    B0 = svar.A \ svar.B
    SVECResult{T}(Matrix{T}(B0), Matrix{T}(svar.Q), Matrix{T}(Xi),
                  n_perm, vecm, svar.identification)
end

# =============================================================================
# Permanent / transitory decomposition of the levels
# =============================================================================

function _svec_gonzalo_ng_levels(vecm::VECMModel{T}) where {T<:AbstractFloat}
    Y = vecm.Y
    n = nvars(vecm)
    r = vecm.rank
    r == 0 && return copy(Y), zeros(T, size(Y, 1), n)
    r == n && return zeros(T, size(Y, 1), n), copy(Y)
    aperp = nullspace(Matrix{T}(vecm.alpha'))
    bperp = nullspace(Matrix{T}(vecm.beta'))
    Mid = aperp' * bperp
    Proj = bperp * (robust_inv(Matrix{T}(Mid); silent=true) * aperp')
    P = Y * Proj'
    (P, Y - P)
end

function _svec_kpsw_levels(vecm::VECMModel{T}) where {T<:AbstractFloat}
    Y = vecm.Y
    U = vecm.U
    n = nvars(vecm)
    T_obs = size(Y, 1)
    T_eff = size(U, 1)
    offset = T_obs - T_eff
    Xi = _svec_xi(vecm.alpha, vecm.beta, _svec_psi(vecm))
    P = Matrix{T}(Y)
    if T_eff > 0
        cumu = cumsum(U, dims=1)
        P_eff = cumu * Xi'
        y0 = offset >= 1 ? Y[offset, :] : zeros(T, n)
        @inbounds for t in 1:T_eff
            P[offset + t, :] = y0 .+ P_eff[t, :]
        end
    end
    (P, Y - P)
end

"""
    permanent_transitory(vecm; method=:gonzalo_ng) -> NamedTuple

Decompose the VECM levels into permanent and transitory components.

- `:gonzalo_ng` (Gonzalo and Ng 2001):
  ``P_t = β_⊥ (α_⊥' β_⊥)^{-1} α_⊥' y_t``, ``T_t = y_t - P_t``.
  The two components sum to the data; ``T_t`` is a linear combination of the
  cointegrating relations and is stationary.
- `:kpsw`: Beveridge–Nelson common trends ``Ξ \\sum_{s} u_s``.

Returns `(permanent, transitory, method, n_permanent)` with `T_obs × n` matrices.
"""
function permanent_transitory(vecm::VECMModel{T};
                              method::Symbol=:gonzalo_ng) where {T<:AbstractFloat}
    method ∈ (:gonzalo_ng, :kpsw) || throw(ArgumentError(
        "method must be :gonzalo_ng or :kpsw, got :$method"))
    P, Tr = method === :gonzalo_ng ? _svec_gonzalo_ng_levels(vecm) :
                                     _svec_kpsw_levels(vecm)
    (permanent=P, transitory=Tr, method=method, n_permanent=nvars(vecm) - vecm.rank)
end
