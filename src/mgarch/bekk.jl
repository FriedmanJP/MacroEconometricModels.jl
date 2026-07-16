# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Scalar and diagonal BEKK(1,1) (Engle & Kroner 1995) with covariance (variance) targeting.
The constant matrix `C` is fixed so the model's unconditional covariance equals the sample
covariance `Σ̄`; only the news (A) and persistence (B) parameters are estimated by one-step
QMLE. Targeting keeps the recursion positive semidefinite and the estimator stable.
"""

import Optim

# =============================================================================
# Per-observation Gaussian contributions (type-generic, ForwardDiff-friendly)
# =============================================================================

# ℓ_t = -½ ( n·log(2π) + log|H_t| + ε_t' H_t⁻¹ ε_t ), using Symmetric factorization so
# ForwardDiff can differentiate through logdet / solve. Returns a length-T vector.
function _mgarch_gauss_contribs(resid::AbstractMatrix{T}, H::AbstractArray{S,3}) where {S,T}
    Tn, n = size(resid)
    c = n * log(S(2π))
    out = Vector{S}(undef, Tn)
    for t in 1:Tn
        Ht = Symmetric(H[:, :, t])
        e = S.(@view resid[t, :])
        out[t] = -S(0.5) * (c + logdet(Ht) + dot(e, Ht \ e))
    end
    out
end

# =============================================================================
# BEKK covariance recursions (variance targeting)
# =============================================================================

# Scalar: H_t = (1-a-b)Σ̄ + a ε_{t-1}ε_{t-1}' + b H_{t-1},  H_1 = Σ̄.
function _bekk_scalar_H(resid::AbstractMatrix{T}, a::S, b::S,
                        Sigbar::AbstractMatrix{T}) where {S,T}
    Tn, n = size(resid)
    H = Array{S,3}(undef, n, n, Tn)
    Ht = Matrix{S}(Sigbar)
    om = (one(S) - a - b)
    Sb = S.(Sigbar)
    for t in 1:Tn
        H[:, :, t] .= Ht
        e = S.(@view resid[t, :])
        Ht = om .* Sb .+ a .* (e * e') .+ b .* Ht
        Ht = (Ht + Ht') ./ 2
    end
    H
end

# Diagonal: H_t = C̃ + A ε_{t-1}ε_{t-1}' A + B H_{t-1} B,  A=diag(a), B=diag(b),
# C̃ = Σ̄ ∘ (11' - aa' - bb') so the unconditional covariance is Σ̄.
function _bekk_diag_H(resid::AbstractMatrix{T}, avec::AbstractVector{S},
                      bvec::AbstractVector{S}, Sigbar::AbstractMatrix{T}) where {S,T}
    Tn, n = size(resid)
    Sb = S.(Sigbar)
    Ctil = Sb .* (ones(S, n, n) .- avec * avec' .- bvec * bvec')
    Ctil = (Ctil + Ctil') ./ 2
    H = Array{S,3}(undef, n, n, Tn)
    Ht = Matrix{S}(Sb)
    for t in 1:Tn
        H[:, :, t] .= Ht
        e = S.(@view resid[t, :])
        A_e = avec .* e
        Ht = Ctil .+ (A_e * A_e') .+ (bvec * bvec') .* Ht
        Ht = (Ht + Ht') ./ 2
    end
    H
end

# H from the natural parameter vector (scalar [a,b]; diagonal [a₁…aₙ,b₁…bₙ]).
function _bekk_H_from_natural(natural::AbstractVector{S}, resid::AbstractMatrix{T},
                              Sigbar::AbstractMatrix{T}, kind::Symbol, n::Int) where {S,T}
    if kind === :scalar
        return _bekk_scalar_H(resid, natural[1], natural[2], Sigbar)
    else
        return _bekk_diag_H(resid, natural[1:n], natural[n+1:2n], Sigbar)
    end
end

_bekk_contribs(natural, resid, Sigbar, kind, n) =
    _mgarch_gauss_contribs(resid, _bekk_H_from_natural(natural, resid, Sigbar, kind, n))

_bekk_negloglik(natural, resid, Sigbar, kind, n) =
    -sum(_bekk_contribs(natural, resid, Sigbar, kind, n))

# --- parameter maps (θ-space → natural, enforcing stationarity) ---
# Scalar: a=sλ, b=s(1-λ), s=σ(θ₁)∈(0,1) is a+b, λ=σ(θ₂).
function _bekk_scalar_ab(theta::AbstractVector{T}) where {T}
    s = one(T) / (one(T) + exp(-theta[1]))
    lam = one(T) / (one(T) + exp(-theta[2]))
    T[s * lam, s * (one(T) - lam)]
end

# Diagonal: per asset aᵢ=√(sᵢλᵢ), bᵢ=√(sᵢ(1-λᵢ)) ⇒ aᵢ²+bᵢ²=sᵢ<1 (sufficient for stability).
function _bekk_diag_ab(theta::AbstractVector{T}, n::Int) where {T}
    a = Vector{T}(undef, n); b = Vector{T}(undef, n)
    @inbounds for i in 1:n
        s = one(T) / (one(T) + exp(-theta[i]))
        lam = one(T) / (one(T) + exp(-theta[n+i]))
        a[i] = sqrt(s * lam)
        b[i] = sqrt(s * (one(T) - lam))
    end
    vcat(a, b)
end

# =============================================================================
# BEKK estimation
# =============================================================================

"""
    estimate_bekk(Y; kind=:scalar) -> MGARCHModel

Scalar or diagonal BEKK(1,1) (Engle & Kroner 1995) with covariance targeting.

- `kind = :scalar` (default): ``H_t = (1-a-b)\\bar\\Sigma + a\\,\\varepsilon_{t-1}\\varepsilon_{t-1}' + b\\,H_{t-1}``,
  scalars `a, b ≥ 0`, `a+b < 1`.
- `kind = :diagonal`: ``H_t = \\tilde C + A\\varepsilon_{t-1}\\varepsilon_{t-1}'A + B H_{t-1} B``,
  `A = diag(a)`, `B = diag(b)`, ``\\tilde C = \\bar\\Sigma \\odot (\\mathbf{1}\\mathbf{1}' - aa' - bb')``.

In both cases the intercept is fixed by variance targeting so the unconditional covariance
equals the sample covariance `Σ̄` of the demeaned data; only the news/persistence parameters
are estimated by one-step Gaussian QMLE. BEKK models the covariance directly, so there are
no univariate margins (`m.margins` is empty). Every `Hₜ` on the fitted path is symmetric PSD.

Returns an [`MGARCHModel`](@ref) with `kind = :bekk`, constant correlation stored as `Rbar`
(the unconditional correlation), and the QML sandwich covariance of the estimated parameters
cached in `param_vcov`.

# Example
```julia
m = estimate_bekk(Y)             # scalar
Σ = covariances(m)               # n×n×T conditional covariances
md = estimate_bekk(Y; kind=:diagonal)
```
"""
function estimate_bekk(Y::AbstractMatrix{T}; kind::Symbol=:scalar) where {T<:AbstractFloat}
    kind in (:scalar, :diagonal) ||
        throw(ArgumentError("kind must be :scalar or :diagonal, got :$kind"))
    Ymat, Tn, n = _mgarch_validate(Y)

    mu = vec(mean(Ymat; dims=1))
    resid = Ymat .- mu'
    Sigbar = Matrix{T}(cov(resid; corrected=false))
    Sigbar = (Sigbar + Sigbar') ./ 2

    if kind === :scalar
        theta0 = _dcc_theta0(T(0.05), T(0.90))  # a≈0.05, b≈0.90
        obj = theta -> _bekk_negloglik(_bekk_scalar_ab(theta), resid, Sigbar, :scalar, n)
        res1 = Optim.optimize(obj, theta0, Optim.NelderMead(),
                              Optim.Options(iterations=3000, show_trace=false))
        res = Optim.optimize(obj, Optim.minimizer(res1), Optim.NelderMead(),
                             Optim.Options(iterations=3000, show_trace=false))
        natural = _bekk_scalar_ab(Optim.minimizer(res))
        pnames = ["a", "b"]
    else
        theta0 = vcat(fill(_dcc_theta0(T(0.05), T(0.90))[1], n),
                      fill(_dcc_theta0(T(0.05), T(0.90))[2], n))
        obj = theta -> _bekk_negloglik(_bekk_diag_ab(theta, n), resid, Sigbar, :diagonal, n)
        res1 = Optim.optimize(obj, theta0, Optim.NelderMead(),
                              Optim.Options(iterations=5000, show_trace=false))
        res = Optim.optimize(obj, Optim.minimizer(res1), Optim.NelderMead(),
                             Optim.Options(iterations=5000, show_trace=false))
        natural = _bekk_diag_ab(Optim.minimizer(res), n)
        pnames = vcat(["a$i" for i in 1:n], ["b$i" for i in 1:n])
    end

    Hgen = _bekk_H_from_natural(natural, resid, Sigbar, kind, n)
    H = Array{T,3}(Hgen)
    loglik = _mgarch_loglik(resid, H)

    k = n + length(natural)  # n means (targeted C absorbs the rest) + news/persistence
    aic_val, bic_val = _compute_aic_bic(loglik, k, Tn)

    param_vcov = try
        Hmat = _numerical_hessian(x -> _bekk_negloglik(x, resid, Sigbar, kind, n), natural)
        S = ForwardDiff.jacobian(x -> _bekk_contribs(x, resid, Sigbar, kind, n), natural)
        V = Matrix{T}(_qmle_sandwich_cov(Hmat, S))
        (V + V') ./ 2
    catch
        fill(T(NaN), length(natural), length(natural))
    end

    Rbar = _corr_from_cov(Sigbar)
    converged = Optim.converged(res)

    MGARCHModel{T}(Ymat, mu, Vector{GARCHModel{T}}(), H, Rbar, Rbar, natural,
                   pnames, param_vcov, loglik, aic_val, bic_val,
                   :bekk, :none, kind, converged, n)
end

estimate_bekk(Y::AbstractMatrix; kwargs...) = estimate_bekk(Float64.(Y); kwargs...)
