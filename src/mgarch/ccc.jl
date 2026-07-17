# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
CCC-GARCH (Bollerslev 1990) estimation, plus the shared building blocks reused by the
DCC and BEKK estimators: standardized-residual extraction, correlation-from-covariance
normalization, and the joint multivariate-Gaussian quasi log-likelihood.
"""

# =============================================================================
# Shared building blocks
# =============================================================================

"""
    _mgarch_validate(Y) -> (Ymat, T, n)

Coerce and validate a multivariate return matrix (T×n, T ≥ 2, n ≥ 2, finite).
"""
function _mgarch_validate(Y::AbstractMatrix{T}) where {T<:AbstractFloat}
    Ymat = Matrix{T}(Y)
    Tn, n = size(Ymat)
    n >= 2 || throw(ArgumentError("multivariate GARCH requires at least 2 series (got n=$n)"))
    Tn >= 2 || throw(ArgumentError("need at least 2 observations (got T=$Tn)"))
    all(isfinite, Ymat) || throw(ArgumentError("Y contains non-finite values"))
    Ymat, Tn, n
end

"""
    _fit_margins(Y, p, q) -> (margins, mu, D, z, resid)

Fit a univariate GARCH(p,q) to each column via `estimate_garch` (REUSED — no univariate
GARCH is re-implemented here). Returns the margin models, the per-series means `mu`,
the conditional standard-deviation matrix `D` (T×n, `Dₜ = √hₜ`), the standardized
residuals `z` (T×n), and the raw residuals `resid` (T×n, `εₜ = yₜ - μ`).
"""
function _fit_margins(Y::Matrix{T}, p::Int, q::Int) where {T}
    Tn, n = size(Y)
    margins = Vector{GARCHModel{T}}(undef, n)
    mu = Vector{T}(undef, n)
    D = Matrix{T}(undef, Tn, n)
    z = Matrix{T}(undef, Tn, n)
    resid = Matrix{T}(undef, Tn, n)
    for i in 1:n
        g = estimate_garch(view(Y, :, i), p, q)
        margins[i] = g
        mu[i] = g.mu
        @inbounds for t in 1:Tn
            D[t, i] = sqrt(g.conditional_variance[t])
            z[t, i] = g.standardized_residuals[t]
            resid[t, i] = g.residuals[t]
        end
    end
    margins, mu, D, z, resid
end

"""
    _uncentered_Q(z) -> Matrix

Correlation-targeting matrix `Q̄ = (1/T) Σₜ zₜz'ₜ` (uncentered second moment of the
standardized residuals). This is the DCC targeting identity.
"""
function _uncentered_Q(z::AbstractMatrix{T}) where {T}
    Tn = size(z, 1)
    Q = (z' * z) ./ T(Tn)
    Matrix{T}((Q + Q') ./ 2)  # symmetrize against round-off
end

"""
    _corr_from_cov(Q) -> Matrix

Normalize a covariance-like matrix to a correlation matrix `diag(Q)^{-1/2} Q diag(Q)^{-1/2}`
(unit diagonal). Clamps off-diagonals to [-1, 1] to defend against round-off.
"""
function _corr_from_cov(Q::AbstractMatrix{T}) where {T}
    n = size(Q, 1)
    d = Vector{T}(undef, n)
    @inbounds for i in 1:n
        d[i] = sqrt(max(Q[i, i], eps(T)))
    end
    R = Matrix{T}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        r = Q[i, j] / (d[i] * d[j])
        R[i, j] = i == j ? one(T) : clamp(r, -one(T), one(T))
    end
    R
end

"""
    _mgarch_loglik(resid, H) -> T

Joint multivariate-Gaussian (quasi) log-likelihood
`ℓ = -½ Σₜ [ n·log(2π) + log|Hₜ| + εₜ' Hₜ⁻¹ εₜ ]`.
`resid` is T×n raw residuals; `H` is the n×n×T covariance path.
"""
function _mgarch_loglik(resid::AbstractMatrix{T}, H::Array{T,3}) where {T}
    Tn, n = size(resid)
    c = n * log(T(2π))
    ll = zero(T)
    @inbounds for t in 1:Tn
        Ht = @view H[:, :, t]
        L = safe_cholesky(Matrix{T}(Ht))   # lower-triangular factor, L L' = Hₜ
        logdet_H = T(2) * sum(log, diag(L))
        e = Vector{T}(@view resid[t, :])
        u = L \ e                          # L⁻¹ εₜ  ⇒  εₜ' Hₜ⁻¹ εₜ = ‖u‖²
        quad = dot(u, u)
        ll += -T(0.5) * (c + logdet_H + quad)
    end
    ll
end

"""
    _build_H(D, R) -> Array{T,3}

Assemble the covariance path from a conditional-sd matrix `D` (T×n) and either a constant
correlation `R::Matrix` (CCC/BEKK) or a time-varying `R::Array{T,3}` (DCC):
`Hₜ = Dₜ Rₜ Dₜ` with `Dₜ = diag(D[t, :])`.
"""
function _build_H(D::AbstractMatrix{T}, R::Matrix{T}) where {T}
    Tn, n = size(D)
    H = Array{T,3}(undef, n, n, Tn)
    @inbounds for t in 1:Tn
        for j in 1:n, i in 1:n
            H[i, j, t] = D[t, i] * R[i, j] * D[t, j]
        end
    end
    H
end

function _build_H(D::AbstractMatrix{T}, R::Array{T,3}) where {T}
    Tn, n = size(D)
    H = Array{T,3}(undef, n, n, Tn)
    @inbounds for t in 1:Tn
        for j in 1:n, i in 1:n
            H[i, j, t] = D[t, i] * R[i, j, t] * D[t, j]
        end
    end
    H
end

# =============================================================================
# CCC-GARCH (Bollerslev 1990)
# =============================================================================

"""
    estimate_ccc(Y; p=1, q=1) -> MGARCHModel

Constant Conditional Correlation GARCH (Bollerslev 1990). Fits a univariate GARCH(p,q)
to each of the `n` columns of `Y` (T×n) — REUSING [`estimate_garch`](@ref) for the margins
— then holds the standardized-residual correlation constant:

- ``D_t = \\mathrm{diag}(\\sigma_{1t}, …, \\sigma_{nt})`` from the margins,
- ``z_t = D_t^{-1}\\varepsilon_t`` standardized residuals,
- ``R = \\mathrm{corr}`` of `z` (constant), and
- ``H_t = D_t R D_t``.

Returns an [`MGARCHModel`](@ref) with `kind = :ccc`. The estimator is fully two-step: each
margin is estimated by univariate QMLE, and `R` is the closed-form standardized-residual
correlation, so there is no second-stage optimization.

# Arguments
- `Y`: T×n matrix of returns (columns are series).
- `p`, `q`: GARCH / ARCH orders for every margin (default 1, 1).

# Example
```julia
m = estimate_ccc(Y)
Σ = covariances(m)      # n×n×T
R = correlations(m)     # constant correlation broadcast across t
```
"""
function estimate_ccc(Y::AbstractMatrix{T}; p::Int=1, q::Int=1) where {T<:AbstractFloat}
    Ymat, Tn, n = _mgarch_validate(Y)
    margins, mu, D, z, resid = _fit_margins(Ymat, p, q)

    Qbar = _uncentered_Q(z)
    R = _corr_from_cov(Qbar)         # constant correlation, unit diagonal
    H = _build_H(D, R)

    loglik = _mgarch_loglik(resid, H)
    k = sum(StatsAPI.dof(g) for g in margins) + n * (n - 1) ÷ 2
    aic_val, bic_val = _compute_aic_bic(loglik, k, Tn)
    converged = all(g -> g.converged, margins)

    MGARCHModel{T}(Ymat, mu, margins, H, R, R, T[], String[],
                   fill(T(NaN), 0, 0), loglik, aic_val, bic_val,
                   :ccc, :none, :none, converged, n)
end

estimate_ccc(Y::AbstractMatrix; kwargs...) = estimate_ccc(Float64.(Y); kwargs...)
