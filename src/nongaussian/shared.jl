# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

"""
Shared helper functions for non-Gaussian SVAR identification.

These utilities are used across the ICA, ML, and heteroskedastic
identification methods:
- `_whiten`: PCA-based pre-whitening
- `_givens_to_orthogonal` / `_orthogonal_to_givens`: rotation parameterization
- `_ica_to_svar`: convert ICA unmixing to structural form
- `_eigendecomposition_id`: heteroskedasticity-based identification
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
Identify B₀ from two covariance matrices via eigendecomposition.

Given Σ₁, Σ₂:
  Σ₁⁻¹ Σ₂ has eigendecomposition V D V⁻¹
  B₀ = Σ₁^{1/2} V (normalized so B₀ B₀' = Σ₁)

Returns (B₀, Λ) where Λ = diag(D) are relative variance ratios.
Identification requires distinct eigenvalues.
"""
function _eigendecomposition_id(Sigma1::Matrix{T}, Sigma2::Matrix{T}) where {T<:AbstractFloat}
    n = size(Sigma1, 1)
    S1_inv = robust_inv(Sigma1)
    M = S1_inv * Sigma2

    E = eigen(M)
    D = real.(E.values)
    V = real.(E.vectors)

    # Sort by eigenvalue magnitude for consistent ordering
    idx = sortperm(D)
    D = D[idx]
    V = V[:, idx]

    # B₀ = chol(Σ₁) * V, normalized so columns have unit norm
    L1 = safe_cholesky(Sigma1)
    B0 = Matrix(L1) * V

    # Normalize columns
    for j in 1:n
        B0[:, j] /= norm(B0[:, j])
    end

    # Scale so B₀ B₀' ≈ Σ₁
    # B₀ = L₁ * Q where Q is orthogonal
    Q_raw = robust_inv(Matrix(L1)) * B0
    F = svd(Q_raw)
    Q = F.U * F.Vt
    B0 = Matrix(L1) * Q

    Lambda = D

    # Sign convention: positive diagonal
    for j in 1:n
        if B0[j, j] < 0
            B0[:, j] *= -one(T)
            Q[:, j] *= -one(T)
        end
    end

    (B0, Q, Lambda)
end
