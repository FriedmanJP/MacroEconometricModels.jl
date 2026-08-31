# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Shared helper functions for statistical SVAR identification via higher moments (Lewis 2025).

These utilities are used across non-Gaussianity (ICA, ML) and heteroskedasticity-based
identification methods:
- `_whiten`: PCA-based pre-whitening
- `_givens_to_orthogonal` / `_orthogonal_to_givens`: rotation parameterization
- `_ica_to_svar`: convert ICA unmixing to structural form
- `_eigendecomposition_id`: heteroskedasticity-based identification

References:
- Lewis, D. J. (2025). "Identification based on higher moments in macroeconometrics."
"""

using LinearAlgebra, Statistics

"""Pre-whiten data via PCA: Z = W_white * U' such that Cov(Z) = I."""
function _whiten(U::Matrix{T}) where {T<:AbstractFloat}
    mu = mean(U, dims=1)
    Uc = U .- mu
    Sigma = Symmetric(Uc' * Uc / size(Uc, 1))
    E = eigen(Sigma)
    idx = sortperm(E.values, rev=true)
    vals = E.values[idx]
    vecs = E.vectors[:, idx]

    # Only keep components with positive eigenvalues
    k = sum(vals .> eps(T) * maximum(vals) * 100)
    D_inv_sqrt = Diagonal(T(1) ./ sqrt.(vals[1:k]))
    W_white = D_inv_sqrt * vecs[:, 1:k]'
    dewhiten = vecs[:, 1:k] * Diagonal(sqrt.(vals[1:k]))

    Z = Matrix{T}((W_white * Uc')')  # T × k
    (Z, Matrix{T}(W_white), Matrix{T}(dewhiten))
end

# =============================================================================
# Givens Rotation Parameterization
# =============================================================================

"""Convert n(n-1)/2 Givens angles to n × n orthogonal matrix."""
function _givens_to_orthogonal(angles::AbstractVector{T}, n::Int) where {T<:AbstractFloat}
    Q = Matrix{T}(I, n, n)
    idx = 1
    for i in 1:n-1
        for j in (i+1):n
            c, s = cos(angles[idx]), sin(angles[idx])
            G = Matrix{T}(I, n, n)
            G[i, i], G[j, j] = c, c
            G[i, j], G[j, i] = -s, s
            Q = Q * G
            idx += 1
        end
    end
    Q
end

"""Extract n(n-1)/2 Givens angles from orthogonal matrix (approximate)."""
function _orthogonal_to_givens(Q::AbstractMatrix{T}, n::Int) where {T<:AbstractFloat}
    n_angles = n * (n - 1) ÷ 2
    angles = zeros(T, n_angles)
    R = copy(Q)
    idx = n_angles
    for i in (n-1):-1:1
        for j in n:-1:(i+1)
            angles[idx] = atan(R[j, i], R[i, i])
            c, s = cos(angles[idx]), sin(angles[idx])
            G = Matrix{T}(I, n, n)
            G[i, i], G[j, j] = c, c
            G[i, j], G[j, i] = s, -s
            R = G * R
            idx -= 1
        end
    end
    angles
end
"""Convert ICA unmixing matrix to SVAR representation: B₀, Q, shocks."""
function _ica_to_svar(W_ica::Matrix{T}, model::VARModel{T}) where {T<:AbstractFloat}
    n = nvars(model)
    L = safe_cholesky(model.Sigma)

    # Full unmixing: W_full * u_t = ε_t, so B₀ = W_full⁻¹
    # From whitened: W_ica * W_white * u_t = ε_t
    # W_full = W_ica * W_white (if Z = W_white * U')
    # But we want B₀ = L * Q where Q is orthogonal

    # Compute B₀ = W_full⁻¹
    B0_raw = robust_inv(W_ica)

    # Extract Q: Q = L⁻¹ B₀
    L_inv = robust_inv(Matrix(L))
    Q_raw = L_inv * B0_raw

    # Enforce orthogonality via polar decomposition
    F = svd(Q_raw)
    Q = F.U * F.Vt

    # Recompute B₀ from L and Q for consistency
    B0 = Matrix(L) * Q

    # Structural shocks
    shocks = (robust_inv(B0) * model.U')'

    # Normalize: make diagonal of B₀ positive (sign convention)
    for j in 1:n
        if B0[j, j] < 0
            B0[:, j] *= -one(T)
            Q[:, j] *= -one(T)
            shocks[:, j] *= -one(T)
        end
    end

    (B0, Q, shocks)
end

"""
Identify B₀ from two covariance matrices via the symmetric generalized eigenproblem.

Given Σ₁, Σ₂:
  L₁ = chol(Σ₁)
  L₁⁻¹ Σ₂ L₁⁻ᵀ = W Λ W'
  B₀ = L₁ W

Eigenvalues Λ are sorted ascending. Columns of Q = W are unit-norm and B₀ is
signed so that its diagonal is positive. Identification requires distinct
eigenvalues; a relative gap below 1e-8 throws `IdentificationError`.

Returns `(B₀, Q, Λ)`.
"""
function _eigendecomposition_id(Sigma1::Matrix{T}, Sigma2::Matrix{T}) where {T<:AbstractFloat}
    n = size(Sigma1, 1)
    L1 = safe_cholesky(Sigma1)
    L1M = Matrix(L1)
    M = Symmetric(L1M \ Sigma2 / L1M')
    E = eigen(M)
    λ = real.(E.values)
    Q = real.(E.vectors)
    idx = sortperm(λ)
    λ = λ[idx]
    Q = Q[:, idx]
    for j in 1:n
        nrm = norm(Q[:, j])
        nrm > zero(T) && (Q[:, j] ./= nrm)
        if dot(Q[:, j], Q[:, j]) > 0 && Q[j, j] < 0
            Q[:, j] .*= -one(T)
        end
    end
    B0 = L1M * Q
    for j in 1:n
        if B0[j, j] < 0
            B0[:, j] .*= -one(T)
            Q[:, j] .*= -one(T)
        end
    end
    gap = n == 1 ? one(T) : minimum(abs(λ[i] - λ[j]) for i in 1:n, j in 1:n if i != j) / max(maximum(abs.(λ)), eps(T))
    gap < T(1e-8) && throw(IdentificationError(
        "heteroskedastic identification requires distinct λ; relative gap=$gap"))
    (B0, Q, λ)
end
