# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
DCC-GARCH (Engle 2002) with optional Aielli (2013) cDCC correction. Two-step QMLE: the
univariate GARCH margins (REUSED from `estimate_garch`) are estimated first and held
fixed; the correlation dynamics `(a, b)` are then estimated by maximizing the second-stage
correlation quasi-likelihood, conditioning on the fixed standardized residuals.
"""

import Optim

# =============================================================================
# DCC correlation recursion
# =============================================================================

# Logit-simplex map θ ∈ ℝ² → (a, b) with a,b ≥ 0 and a+b < 1.
# s = σ(θ₁) is the persistence a+b ∈ (0,1); λ = σ(θ₂) splits it: a = sλ, b = s(1-λ).
function _dcc_ab(theta::AbstractVector{T}) where {T}
    s = one(T) / (one(T) + exp(-theta[1]))
    lam = one(T) / (one(T) + exp(-theta[2]))
    a = s * lam
    b = s * (one(T) - lam)
    a, b
end

function _dcc_theta0(a::T, b::T) where {T}
    s = clamp(a + b, T(1e-4), one(T) - T(1e-4))
    lam = clamp(a / s, T(1e-4), one(T) - T(1e-4))
    [log(s / (one(T) - s)), log(lam / (one(T) - lam))]
end

"""
    _dcc_R_path(z, a, b, Qbar; correction=:none) -> Array{T,3}

Time-varying correlation path. With standard targeting (`correction=:none`):

``Q_t = (1-a-b)\\bar{Q} + a\\,z_{t-1}z_{t-1}' + b\\,Q_{t-1}``,
``R_t = \\mathrm{diag}(Q_t)^{-1/2} Q_t \\mathrm{diag}(Q_t)^{-1/2}``,

with `Q_1 = Q̄`. The Aielli (2013) cDCC correction replaces the outer product with
`q*_{t-1} q*_{t-1}'` where `q*_t = diag(Q_t)^{1/2} z_t`, removing the standard-DCC
targeting bias in the correlation intercept.
"""
function _dcc_R_path(z::AbstractMatrix{T}, a::T, b::T, Qbar::AbstractMatrix{T};
                     correction::Symbol=:none) where {T}
    Tn, n = size(z)
    R = Array{T,3}(undef, n, n, Tn)
    Q = Matrix{T}(Qbar)
    om = (one(T) - a - b)
    @inbounds for t in 1:Tn
        Rt = _corr_from_cov(Q)
        R[:, :, t] .= Rt
        zt = @view z[t, :]
        if correction === :aielli
            qs = Vector{T}(undef, n)
            for i in 1:n
                qs[i] = sqrt(max(Q[i, i], eps(T))) * zt[i]
            end
            outer = qs * qs'
        else
            outer = Vector{T}(zt) * Vector{T}(zt)'
        end
        Q = om .* Qbar .+ a .* outer .+ b .* Q
        Q = (Q + Q') ./ 2
    end
    R
end

# Second-stage correlation quasi-log-likelihood contributions (Engle 2002, step 2):
# ℓ_t = -½ ( log|R_t| + z_t' R_t⁻¹ z_t ).  Sum is the objective; per-t vector feeds the
# QML sandwich outer-product-of-scores.  Type-generic in `ab` for ForwardDiff.
function _dcc_corr_contribs(ab::AbstractVector{S}, z::AbstractMatrix{T},
                            Qbar::AbstractMatrix{T}, correction::Symbol) where {S,T}
    Tn, n = size(z)
    a = ab[1]; b = ab[2]
    om = (one(S) - a - b)
    Q = Matrix{S}(Qbar)
    out = Vector{S}(undef, Tn)
    for t in 1:Tn
        Rt = _corr_from_cov(Q)
        zt = S.(@view z[t, :])
        Rsym = Symmetric(Rt)
        ld = logdet(Rsym)
        quad = dot(zt, Rsym \ zt)
        out[t] = -S(0.5) * (ld + quad)
        if correction === :aielli
            qs = [sqrt(max(Q[i, i], eps(S))) * zt[i] for i in 1:n]
            outer = qs * qs'
        else
            outer = zt * zt'
        end
        Q = om .* Qbar .+ a .* outer .+ b .* Q
        Q = (Q + Q') ./ 2
    end
    out
end

_dcc_neg_corr_loglik(ab, z, Qbar, correction) = -sum(_dcc_corr_contribs(ab, z, Qbar, correction))

# =============================================================================
# DCC estimation
# =============================================================================

"""
    estimate_dcc(Y; p=1, q=1, correction=:none) -> MGARCHModel

Dynamic Conditional Correlation GARCH (Engle 2002) via two-step QMLE.

**Step 1** — fit a univariate GARCH(p,q) to each column of `Y` (T×n) with
[`estimate_garch`](@ref) and collect standardized residuals `zₜ`.

**Step 2** — correlation targeting `Q̄ = (1/T) Σₜ zₜz'ₜ`; estimate `(a,b)` by maximizing
the correlation quasi-likelihood

``\\ell_c(a,b) = -\\tfrac12 \\sum_t \\big( \\log|R_t| + z_t' R_t^{-1} z_t \\big)``

subject to `a,b ≥ 0` and `a+b < 1` (enforced by a logit-simplex reparametrization). The
margins are held fixed inside the `(a,b)` objective — they are NOT re-estimated.

`correction`:
- `:none` (default) — standard DCC (Engle 2002).
- `:aielli` — cDCC (Aielli 2013), replacing `zₜz'ₜ` with `q*ₜq*'ₜ`,
  `q*ₜ = diag(Qₜ)^{1/2} zₜ`, which removes the standard-DCC intercept-targeting bias.
  (Targeting still uses the sample `Q̄` of `z`; the exact Aielli fixed-point targeting is a
  documented refinement — see `estimate_dcc` notes.)

Returns an [`MGARCHModel`](@ref) with `kind = :dcc`, a time-varying `R::Array{T,3}`, and
the second-stage QML sandwich covariance of `(a,b)` cached in `param_vcov`. Setting
`a = b = 0` reduces the model exactly to CCC.

# Example
```julia
m = estimate_dcc(Y)
a, b = coef(m)
Rt = correlations(m)     # n×n×T time-varying correlations
```
"""
function estimate_dcc(Y::AbstractMatrix{T}; p::Int=1, q::Int=1,
                      correction::Symbol=:none) where {T<:AbstractFloat}
    correction in (:none, :aielli) ||
        throw(ArgumentError("correction must be :none or :aielli, got :$correction"))
    Ymat, Tn, n = _mgarch_validate(Y)
    margins, mu, D, z, resid = _fit_margins(Ymat, p, q)

    Qbar = _uncentered_Q(z)

    obj = theta -> _dcc_neg_corr_loglik(collect(_dcc_ab(theta)), z, Qbar, correction)
    theta0 = _dcc_theta0(T(0.03), T(0.95))
    res1 = Optim.optimize(obj, theta0, Optim.NelderMead(),
                          Optim.Options(iterations=2000, show_trace=false))
    res = Optim.optimize(obj, Optim.minimizer(res1), Optim.NelderMead(),
                         Optim.Options(iterations=2000, show_trace=false))
    a, b = _dcc_ab(Optim.minimizer(res))
    ab = T[a, b]

    R = _dcc_R_path(z, a, b, Qbar; correction=correction)
    H = _build_H(D, R)
    loglik = _mgarch_loglik(resid, H)

    k = sum(StatsAPI.dof(g) for g in margins) + 2
    aic_val, bic_val = _compute_aic_bic(loglik, k, Tn)

    # Second-stage QML sandwich covariance of (a,b), evaluated at the optimum.
    param_vcov = try
        Hmat = _numerical_hessian(x -> _dcc_neg_corr_loglik(x, z, Qbar, correction), ab)
        S = ForwardDiff.jacobian(x -> _dcc_corr_contribs(x, z, Qbar, correction), ab)
        V = Matrix{T}(_qmle_sandwich_cov(Hmat, S))
        (V + V') ./ 2
    catch
        fill(T(NaN), 2, 2)
    end

    converged = Optim.converged(res) && all(g -> g.converged, margins)

    MGARCHModel{T}(Ymat, mu, margins, H, R, _corr_from_cov(Qbar), ab,
                   ["a", "b"], param_vcov, loglik, aic_val, bic_val,
                   :dcc, correction, :none, converged, n)
end

estimate_dcc(Y::AbstractMatrix; kwargs...) = estimate_dcc(Float64.(Y); kwargs...)
