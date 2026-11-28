# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

"""
Non-Gaussian maximum likelihood SVAR identification.

Parametric identification via non-Gaussianity (Lewis 2025, Section 4).
Estimates B₀ and distribution parameters jointly by maximizing the log-likelihood
under non-Gaussian shock distributions: Student-t, symmetric or asymmetric mixture
of normals, PML (Pearson Type IV), skew-normal, or a unified dispatcher.

Optimization uses multi-start `Optim.LBFGS()` with `ForwardDiff` gradients (warm-started
from JADE, FastICA, and random orthogonal rotations), and covariance matrices support
`ForwardDiff` Hessian (`:hessian`), GMR (2017) Huber–White sandwich (`:sandwich`),
outer product of gradients (`:opg`), and automatic selection (`:auto`) with full
Jacobian delta-method `B₀` standard errors.

References:
- Lewis, D. J. (2025). "Identification based on higher moments in macroeconometrics."
- Lanne, M., Meitz, M. & Saikkonen, P. (2017). "Identification and estimation of non-Gaussian SVAR."
- Gourieroux, C., Monfort, A. & Renne, J.-P. (2017). "Statistical inference for independent component analysis."
- Gourieroux, C., Monfort, A. & Renne, J.-P. (2020). "Identification and estimation in non-fundamental structural VARMA models."
- Lanne, M. & Lütkepohl, H. (2010). "Structural Vector Autoregressions With Nonnormal Residuals."
- Herwartz, H. (2018). "Hodges-Lehmann detection of structural shocks."
- Azzalini, A. (1985). "A class of distributions which includes the normal ones."
"""

using LinearAlgebra, Statistics, Distributions, Random
import Optim

# =============================================================================
# Result Type
# =============================================================================

"""
    NonGaussianMLResult{T} <: AbstractNonGaussianSVAR

Result from non-Gaussian maximum likelihood SVAR identification.

Fields:
- `B0::Matrix{T}` — structural impact matrix (n × n)
- `Q::Matrix{T}` — rotation matrix
- `shocks::Matrix{T}` — structural shocks (T_eff × n)
- `distribution::Symbol` — `:student_t`, `:mixture_normal`, `:mixture_normal_asym`, `:pml`, `:skew_normal`
- `loglik::T` — log-likelihood at MLE
- `loglik_gaussian::T` — Gaussian log-likelihood (for LR test)
- `dist_params::Dict{Symbol, Any}` — distribution parameters and `:vcov_type` metadata
- `vcov::Matrix{T}` — asymptotic covariance of (angles, distribution parameters)
- `se::Matrix{T}` — delta-method standard errors for B₀
- `converged::Bool`
- `iterations::Int`
- `aic::T`
- `bic::T`
- `shock_names::Vector{String}` — labels for the n structural shocks
"""
struct NonGaussianMLResult{T<:AbstractFloat} <: AbstractNonGaussianSVAR
    B0::Matrix{T}
    Q::Matrix{T}
    shocks::Matrix{T}
    distribution::Symbol
    loglik::T
    loglik_gaussian::T
    dist_params::Dict{Symbol, Any}
    vcov::Matrix{T}
    se::Matrix{T}
    converged::Bool
    iterations::Int
    aic::T
    bic::T
    shock_names::Vector{String}
end

function NonGaussianMLResult{T}(B0, Q, shocks, distribution, loglik, loglik_gaussian,
                                 dist_params, vcov, se, converged, iterations,
                                 aic, bic) where {T<:AbstractFloat}
    n = size(B0, 2)
    NonGaussianMLResult{T}(B0, Q, shocks, distribution, loglik, loglik_gaussian,
                            dist_params, vcov, se, converged, iterations, aic, bic,
                            _default_shock_names(n))
end

function NonGaussianMLResult(B0, Q, shocks, distribution, loglik, loglik_gaussian,
                              dist_params, vcov, se, converged, iterations, aic, bic)
    T = eltype(B0)
    NonGaussianMLResult{T}(B0, Q, shocks, distribution, loglik, loglik_gaussian,
                            dist_params, vcov, se, converged, iterations, aic, bic)
end

function Base.show(io::IO, r::NonGaussianMLResult{T}) where {T}
    n = size(r.B0, 1)
    lr_stat = T(2) * (r.loglik - r.loglik_gaussian)
    lr_stat = max(lr_stat, zero(T))
    # Degrees of freedom: number of additional distribution parameters (n · p_dist)
    n_dist_params = if haskey(r.dist_params, :all_params)
        length(r.dist_params[:all_params])
    else
        _n_dist_params(r.distribution) * n
    end
    lr_df = max(n_dist_params, 1)
    lr_pval = lr_stat > 0 ? T(1) - cdf(Chisq(lr_df), lr_stat) : one(T)
    lr_stars = _significance_stars(lr_pval)

    spec = Any[
        "Distribution"   string(r.distribution);
        "Variables"      n;
        "Log-likelihood" _fmt(r.loglik; digits=4);
        "Log-lik (Gaussian)" _fmt(r.loglik_gaussian; digits=4);
        "AIC"            _fmt(r.aic; digits=2);
        "BIC"            _fmt(r.bic; digits=2);
        "Converged"      r.converged ? "Yes" : "No";
        "Iterations"     r.iterations
    ]
    _pretty_table(io, spec;
        title = "Non-Gaussian ML Identification Result",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    # LR test against Gaussian
    lr_data = Any[
        "LR statistic" string(_fmt(lr_stat; digits=2), " ", lr_stars);
        "P-value"      _format_pvalue(lr_pval);
        "H₀"           "Gaussian shocks (standard SVAR)"
    ]
    _pretty_table(io, lr_data;
        title = "LR Test: Non-Gaussian vs. Gaussian",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    _show_B0_ses(io, r.B0, r.se)
end

# =============================================================================
# Log-pdf Functions (Real-generic for ForwardDiff.Dual compatibility)
# =============================================================================

"""Numerically stable and ForwardDiff-safe logistic sigmoid."""
@inline function _stable_sigmoid(u::T) where {T<:Real}
    if u >= zero(T)
        one(T) / (one(T) + exp(-u))
    else
        e = exp(u)
        e / (one(T) + e)
    end
end

"""Student-t log-pdf (standardized to unit variance)."""
function _student_t_logpdf(x::Tx, nu::Tn) where {Tx<:Real, Tn<:Real}
    T = float(promote_type(Tx, Tn))
    nu_safe = max(T(nu), T(2.01))
    s = sqrt(nu_safe / (nu_safe - T(2)))
    z = T(x) * s
    logpdf(TDist(nu_safe), z) + log(s)
end

"""Asymmetric 2-component Gaussian mixture log-pdf: p*N(μ₁,σ₁²) + (1-p)*N(μ₂,σ₂²)."""
function _mixture_normal_logpdf(x::Tx, p_mix::Tp, mu1::Tm1, sigma1::Ts1,
                                mu2::Tm2, sigma2::Ts2) where {Tx<:Real, Tp<:Real, Tm1<:Real, Ts1<:Real, Tm2<:Real, Ts2<:Real}
    T = float(promote_type(Tx, Tp, Tm1, Ts1, Tm2, Ts2))
    p = T(p_mix)
    xt = T(x)
    l1 = log(p) + logpdf(Normal(T(mu1), T(sigma1)), xt)
    l2 = log(one(T) - p) + logpdf(Normal(T(mu2), T(sigma2)), xt)
    m = max(l1, l2)
    m + log(exp(l1 - m) + exp(l2 - m))
end

"""Zero-mean 2-component Gaussian mixture log-pdf: p*N(0,σ₁²) + (1-p)*N(0,σ₂²)."""
function _mixture_normal_logpdf(x::Tx, p_mix::Tp, sigma1::Ts1,
                                sigma2::Ts2) where {Tx<:Real, Tp<:Real, Ts1<:Real, Ts2<:Real}
    z = zero(float(promote_type(Tx, Tp, Ts1, Ts2)))
    _mixture_normal_logpdf(x, p_mix, z, sigma1, z, sigma2)
end

"""Skew-normal log-pdf: f(x) = 2 φ(x) Φ(αx)."""
function _skew_normal_logpdf(x::Tx, alpha::Ta) where {Tx<:Real, Ta<:Real}
    T = float(promote_type(Tx, Ta))
    xt = T(x)
    Tf = typeof(ForwardDiff.value(zero(T)))
    d0 = Normal(zero(Tf), one(Tf))
    T(log(T(2)) + logpdf(d0, xt) + logcdf(d0, T(alpha) * xt))
end

"""Real part of log Γ(m + i y) using SpecialFunctions for IEEE floats."""
@inline function _re_loggamma(m::T, y::T) where {T<:Base.IEEEFloat}
    real(SpecialFunctions.loggamma(complex(m, y)))
end

"""
Real-arithmetic recurrence + 6-term Stirling series for Re log Γ(m + i y).
Enables arbitrary-order ForwardDiff differentiation without Complex{Dual}.
"""
function _re_loggamma(m::T, y::T) where {T<:Real}
    acc = zero(T)
    x = m
    while x < 16
        acc -= T(0.5) * log(x * x + y * y)
        x += one(T)
    end
    r2 = x * x + y * y
    log_r = T(0.5) * log(r2)
    theta = atan(y, x)
    stirling = (x - T(0.5)) * log_r - y * theta - x + T(0.5) * log(T(2π))
    inv_r2 = one(T) / r2
    u = x * inv_r2
    v = -y * inv_r2
    u2 = u * u - v * v
    v2 = T(2) * u * v
    p_re, p_im = u, v
    stirling += p_re / T(12)
    p_re, p_im = p_re * u2 - p_im * v2, p_re * v2 + p_im * u2
    stirling -= p_re / T(360)
    p_re, p_im = p_re * u2 - p_im * v2, p_re * v2 + p_im * u2
    stirling += p_re / T(1260)
    p_re, p_im = p_re * u2 - p_im * v2, p_re * v2 + p_im * u2
    stirling -= p_re / T(1680)
    p_re, p_im = p_re * u2 - p_im * v2, p_re * v2 + p_im * u2
    stirling += p_re / T(1188)
    p_re, p_im = p_re * u2 - p_im * v2, p_re * v2 + p_im * u2
    stirling -= p_re * T(691) / T(360360)
    acc + stirling
end

@inline function _re_loggamma(m::T, y::T) where {T<:Integer}
    _re_loggamma(float(m), float(y))
end

@inline function _re_loggamma(m::Real, y::Real)
    _re_loggamma(promote(float(m), float(y))...)
end

"""
Genuine Pearson Type IV log-pdf, standardized to zero mean and unit variance (#566).

    f(x) = k·[1 + z²]^{−m}·exp(−ν·arctan(z)),   z = (x − λ)/a
    log k = (2m−2)·log 2 + 2·Re log Γ(m + iν/2) − log π − log a − log Γ(2m−1)

(Heinrich 2004 parameterization) with scale/location pinned via r = 2(m−1):

    a = r·√((r−1)/(r² + ν²)),   λ = a·ν/r

so the density integrates to 1 with mean 0 and variance 1 for every skewness ν ∈ ℝ
and tail exponent m > 2.
"""
function _pearson_iv_logpdf(x::Tx, skew::Ts, m::Tm) where {Tx<:Real, Ts<:Real, Tm<:Real}
    T = float(promote_type(Tx, Ts, Tm))
    m_safe = max(T(m), T(2.05))                 # m > 2 ⟺ the variance exists
    sk = T(skew)
    r = T(2) * (m_safe - one(T))
    a = r * sqrt((r - one(T)) / (r^2 + sk^2))
    lam = a * sk / r
    z = (T(x) - lam) / a
    logk = (T(2) * m_safe - T(2)) * log(T(2)) +
           T(2) * _re_loggamma(m_safe, sk / T(2)) -
           log(T(pi)) - log(a) - SpecialFunctions.loggamma(T(2) * m_safe - one(T))
    logk - m_safe * log1p(z^2) - sk * atan(z)
end

# =============================================================================
# Gaussian Reference Log-Likelihood
# =============================================================================

"""Compute Gaussian log-likelihood for a VAR model (for LR test comparison)."""
function _gaussian_loglik(model::VARModel{T}) where {T<:AbstractFloat}
    T_obs = size(model.U, 1)
    n = nvars(model)
    ld = logdet_safe(model.Sigma)
    -T_obs / 2 * (n * log(T(2π)) + ld) - T_obs / 2 * n  # trace term = n under MLE
end

# =============================================================================
# Unified Non-Gaussian Log-Likelihood
# =============================================================================

@inline function _unpack_shock_mixture(u1::Tθ, u2::Tθ) where {Tθ<:Real}
    eps_p = Tθ(1e-4)
    eps_v = Tθ(1e-4)
    p_mix = eps_p + (one(Tθ) - Tθ(2) * eps_p) * _stable_sigmoid(u1)
    w = eps_v + (one(Tθ) - Tθ(2) * eps_v) * _stable_sigmoid(u2)
    sigma1 = sqrt(w / p_mix)
    sigma2 = sqrt((one(Tθ) - w) / (one(Tθ) - p_mix))
    (p_mix, sigma1, sigma2)
end

@inline function _unpack_shock_mixture_asym(u1::Tθ, u2::Tθ, u3::Tθ) where {Tθ<:Real}
    eps_p = Tθ(1e-4)
    eps_b = Tθ(1e-4)
    eps_v = Tθ(1e-4)
    p_mix = eps_p + (one(Tθ) - Tθ(2) * eps_p) * _stable_sigmoid(u1)
    r_mu = (one(Tθ) - eps_b) * tanh(u2)
    mu1 = sqrt((one(Tθ) - p_mix) / p_mix) * r_mu
    mu2 = -sqrt(p_mix / (one(Tθ) - p_mix)) * r_mu
    B = one(Tθ) - r_mu * r_mu
    w = eps_v + (one(Tθ) - Tθ(2) * eps_v) * _stable_sigmoid(u3)
    sigma1 = sqrt(w * B / p_mix)
    sigma2 = sqrt((one(Tθ) - w) * B / (one(Tθ) - p_mix))
    (p_mix, mu1, sigma1, mu2, sigma2)
end

@inline _ng_primal_float(x::AbstractFloat) = x
@inline _ng_primal_float(x::ForwardDiff.Dual) = _ng_primal_float(ForwardDiff.value(x))

"""Add shock `j`'s log-pdf contributions across all `T_obs` observations into `ell` (or return sum)."""
function _nongaussian_shock_loglik_1d!(ell::Union{Nothing,AbstractVector},
                                       s::AbstractVector{Ts},
                                       dp_j::AbstractVector{Tθ},
                                       distribution::Symbol) where {Ts<:Real, Tθ<:Real}
    Tacc = ell === nothing ? float(promote_type(Ts, Tθ)) : eltype(ell)
    T_obs = length(s)
    ll = zero(Tacc)
    if distribution === :student_t
        nu_safe = max(Tacc(Tθ(2.01) + exp(clamp(dp_j[1], Tθ(-30), Tθ(12)))), Tacc(2.01))
        half_nup1 = (nu_safe + one(Tacc)) / Tacc(2)
        nm2 = nu_safe - Tacc(2)
        inv_nm2 = one(Tacc) / nm2
        logc = SpecialFunctions.loggamma(half_nup1) -
               SpecialFunctions.loggamma(nu_safe / Tacc(2)) -
               Tacc(0.5) * log(Tacc(pi) * nm2)
        @inbounds for t in 1:T_obs
            xt = Tacc(s[t])
            v = logc - half_nup1 * log1p(xt * xt * inv_nm2)
            if ell === nothing
                ll += v
            else
                ell[t] += v
            end
        end
    elseif distribution === :mixture_normal
        p_mix, sigma1, sigma2 = _unpack_shock_mixture(dp_j[1], dp_j[2])
        p_t = Tacc(p_mix); s1 = Tacc(sigma1); s2 = Tacc(sigma2)
        half_log2pi = Tacc(0.5) * log(Tacc(2π))
        c1 = log(p_t) - log(s1) - half_log2pi
        c2 = log(one(Tacc) - p_t) - log(s2) - half_log2pi
        inv_2v1 = Tacc(0.5) / (s1 * s1)
        inv_2v2 = Tacc(0.5) / (s2 * s2)
        @inbounds for t in 1:T_obs
            xt = Tacc(s[t])
            x2 = xt * xt
            l1 = c1 - x2 * inv_2v1
            l2 = c2 - x2 * inv_2v2
            m = max(l1, l2)
            v = m + log(exp(l1 - m) + exp(l2 - m))
            if ell === nothing
                ll += v
            else
                ell[t] += v
            end
        end
    elseif distribution === :mixture_normal_asym
        p_mix, mu1, sigma1, mu2, sigma2 = _unpack_shock_mixture_asym(
            dp_j[1], dp_j[2], dp_j[3])
        p_t = Tacc(p_mix); m1 = Tacc(mu1); s1 = Tacc(sigma1); m2 = Tacc(mu2); s2 = Tacc(sigma2)
        half_log2pi = Tacc(0.5) * log(Tacc(2π))
        c1 = log(p_t) - log(s1) - half_log2pi
        c2 = log(one(Tacc) - p_t) - log(s2) - half_log2pi
        inv_s1 = one(Tacc) / s1
        inv_s2 = one(Tacc) / s2
        @inbounds for t in 1:T_obs
            xt = Tacc(s[t])
            z1 = (xt - m1) * inv_s1
            z2 = (xt - m2) * inv_s2
            l1 = c1 - Tacc(0.5) * z1 * z1
            l2 = c2 - Tacc(0.5) * z2 * z2
            m = max(l1, l2)
            v = m + log(exp(l1 - m) + exp(l2 - m))
            if ell === nothing
                ll += v
            else
                ell[t] += v
            end
        end
    elseif distribution === :pml
        sk = Tacc(dp_j[1])
        m_safe = max(Tacc(Tθ(2.05) + exp(clamp(dp_j[2], Tθ(-30), Tθ(9.2)))), Tacc(2.05))
        r = Tacc(2) * (m_safe - one(Tacc))
        a = r * sqrt((r - one(Tacc)) / (r * r + sk * sk))
        lam = a * sk / r
        inv_a = one(Tacc) / a
        logk = (Tacc(2) * m_safe - Tacc(2)) * log(Tacc(2)) +
               Tacc(2) * _re_loggamma(m_safe, sk / Tacc(2)) -
               log(Tacc(pi)) - log(a) - SpecialFunctions.loggamma(Tacc(2) * m_safe - one(Tacc))
        @inbounds for t in 1:T_obs
            z = (Tacc(s[t]) - lam) * inv_a
            v = logk - m_safe * log1p(z * z) - sk * atan(z)
            if ell === nothing
                ll += v
            else
                ell[t] += v
            end
        end
    elseif distribution === :skew_normal
        alpha = Tacc(dp_j[1])
        Tf = typeof(_ng_primal_float(zero(Tacc)))
        d0 = Normal(zero(Tf), one(Tf))
        c_sn = log(Tacc(2)) - Tacc(0.5) * log(Tacc(2π))
        @inbounds for t in 1:T_obs
            xt = Tacc(s[t])
            v = c_sn - Tacc(0.5) * xt * xt + logcdf(d0, alpha * xt)
            if ell === nothing
                ll += v
            else
                ell[t] += v
            end
        end
    else
        throw(ArgumentError("Unknown distribution: $distribution"))
    end
    ll
end

"""1D shock log-likelihood contribution across all T_obs observations."""
function _nongaussian_shock_loglik_1d(s::AbstractVector{Ts},
                                      dp_j::AbstractVector{Tθ},
                                      distribution::Symbol) where {Ts<:Real, Tθ<:Real}
    Tacc = float(promote_type(Ts, Tθ))
    _nongaussian_shock_loglik_1d!(nothing, s, dp_j, distribution)::Tacc
end

"""Vector of per-observation log-likelihood contributions `[ℓ_1(θ), ..., ℓ_T(θ)]`."""
function _nongaussian_obs_loglik_vec(params::AbstractVector{Tθ},
                                     Z_chol::AbstractMatrix{T},
                                     logdet_inv::T,
                                     n::Int,
                                     distribution::Symbol) where {Tθ<:Real, T<:AbstractFloat}
    angles, dp = _unpack_nongaussian_params(params, n, distribution)
    Q = n == 1 ? Matrix{Tθ}(I, 1, 1) : _givens_to_orthogonal(angles, n)
    shocks = Z_chol * Q
    T_obs = size(Z_chol, 1)
    k_dp = _n_dist_params(distribution)
    Tacc = float(promote_type(eltype(shocks), Tθ))
    ell = fill(Tacc(logdet_inv), T_obs)
    @inbounds for j in 1:n
        idx = (j - 1) * k_dp
        _nongaussian_shock_loglik_1d!(ell, @view(shocks[:, j]),
                                      @view(dp[(idx + 1):(idx + k_dp)]),
                                      distribution)
    end
    ell
end

"""Log-likelihood evaluated directly from orthogonal rotation Q and distribution params dp."""
function _nongaussian_loglik_from_Q(Q::AbstractMatrix{TQ},
                                    dp::AbstractVector{Tθ},
                                    Z_chol::AbstractMatrix{T},
                                    logdet_inv::T,
                                    n::Int,
                                    distribution::Symbol) where {TQ<:Real, Tθ<:Real, T<:AbstractFloat}
    shocks = Z_chol * Q
    T_obs = size(Z_chol, 1)
    k_dp = _n_dist_params(distribution)
    Tacc = promote_type(eltype(shocks), Tθ)
    loglik = Tacc(T_obs) * Tacc(logdet_inv)
    @inbounds for j in 1:n
        idx = (j - 1) * k_dp
        loglik += _nongaussian_shock_loglik_1d(@view(shocks[:, j]),
                                               @view(dp[(idx + 1):(idx + k_dp)]),
                                               distribution)
    end
    loglik
end

"""Log-likelihood evaluated directly on pre-whitened residuals Z_chol = (L⁻¹ U')'."""
function _nongaussian_loglik_from_whitened(params::AbstractVector{Tθ},
                                           Z_chol::AbstractMatrix{T},
                                           logdet_inv::T,
                                           n::Int,
                                           distribution::Symbol) where {Tθ<:Real, T<:AbstractFloat}
    angles, dp = _unpack_nongaussian_params(params, n, distribution)
    Q = n == 1 ? Matrix{Tθ}(I, 1, 1) : _givens_to_orthogonal(angles, n)
    _nongaussian_loglik_from_Q(Q, dp, Z_chol, logdet_inv, n, distribution)
end

"""Single-observation log-likelihood contribution ℓ_t(θ) for per-obs score matrix G_t."""
function _nongaussian_obs_loglik(params::AbstractVector{Tθ},
                                  z_t::AbstractVector{T},
                                  logdet_inv::T,
                                  n::Int,
                                  distribution::Symbol) where {Tθ<:Real, T<:AbstractFloat}
    angles, dp = _unpack_nongaussian_params(params, n, distribution)
    Q = n == 1 ? Matrix{Tθ}(I, 1, 1) : _givens_to_orthogonal(angles, n)
    eps_t = Q' * z_t

    ll = Tθ(logdet_inv)
    if distribution === :student_t
        @inbounds for j in 1:n
            nu = Tθ(2.01) + exp(clamp(dp[j], Tθ(-30), Tθ(12)))
            ll += _student_t_logpdf(eps_t[j], nu)
        end
    elseif distribution === :mixture_normal
        @inbounds for j in 1:n
            idx = (j - 1) * 2
            p_mix, sigma1, sigma2 = _unpack_shock_mixture(dp[idx + 1], dp[idx + 2])
            ll += _mixture_normal_logpdf(eps_t[j], p_mix, sigma1, sigma2)
        end
    elseif distribution === :mixture_normal_asym
        @inbounds for j in 1:n
            idx = (j - 1) * 3
            p_mix, mu1, sigma1, mu2, sigma2 = _unpack_shock_mixture_asym(
                dp[idx + 1], dp[idx + 2], dp[idx + 3])
            ll += _mixture_normal_logpdf(eps_t[j], p_mix, mu1, sigma1, mu2, sigma2)
        end
    elseif distribution === :pml
        @inbounds for j in 1:n
            idx = (j - 1) * 2
            skew = dp[idx + 1]
            m = Tθ(2.05) + exp(clamp(dp[idx + 2], Tθ(-30), Tθ(9.2)))
            ll += _pearson_iv_logpdf(eps_t[j], skew, m)
        end
    elseif distribution === :skew_normal
        @inbounds for j in 1:n
            ll += _skew_normal_logpdf(eps_t[j], dp[j])
        end
    else
        throw(ArgumentError("Unknown distribution: $distribution"))
    end
    ll
end

"""
Full non-Gaussian log-likelihood:
ℓ(θ) = Σ_t [log|det(B₀⁻¹)| + Σ_j log f_j(ε_{j,t}; θ_j)]
where ε_t = B₀⁻¹ u_t.
"""
function _nongaussian_loglik(angles::AbstractVector{T1},
                             dist_params_vec::AbstractVector{T2},
                             U::AbstractMatrix{T},
                             L::LowerTriangular{T,<:AbstractMatrix{T}},
                             n::Int;
                             distribution::Symbol=:student_t) where {T1<:Real, T2<:Real, T<:AbstractFloat}
    Z_chol = Matrix{T}(transpose(L \ transpose(U)))
    logdet_inv = -sum(i -> log(max(abs(L[i, i]), eps(T))), 1:n)
    Tθ = promote_type(T1, T2)
    params = vcat(Vector{Tθ}(angles), Vector{Tθ}(dist_params_vec))
    _nongaussian_loglik_from_whitened(params, Z_chol, logdet_inv, n, distribution)
end

# =============================================================================
# Parameter Packing/Unpacking
# =============================================================================

"""Pack rotation angles and distribution parameters into single vector."""
function _pack_nongaussian_params(angles::AbstractVector{T1},
                                  dist_params::AbstractVector{T2}) where {T1, T2}
    T = promote_type(T1, T2)
    vcat(Vector{T}(angles), Vector{T}(dist_params))
end

"""Unpack single vector into rotation angles and distribution parameters."""
function _unpack_nongaussian_params(params::AbstractVector, n::Int, distribution::Symbol)
    n_angles = n * (n - 1) ÷ 2
    angles = params[1:n_angles]
    dist_params = params[n_angles+1:end]
    (angles, dist_params)
end

"""Determine number of distribution parameters per shock."""
function _n_dist_params(distribution::Symbol)
    distribution === :student_t ? 1 :
    distribution === :mixture_normal ? 2 :
    distribution === :mixture_normal_asym ? 3 :
    distribution === :pml ? 2 :
    distribution === :skew_normal ? 1 :
    throw(ArgumentError("Unknown distribution: $distribution"))
end

# =============================================================================
# Hessian, Sandwich Covariance, and Delta-Method Standard Errors
# =============================================================================

"""Compute numerical Hessian via central finite differences (optionally parallelized across upper-triangular pairs; #948)."""
function _numerical_hessian(f::Function, x::Vector{T};
                            eps_step::T=T(1e-5),
                            threaded::Bool=false) where {T<:AbstractFloat}
    n = length(x)
    H = zeros(T, n, n)
    pairs = Tuple{Int,Int}[(i, j) for i in 1:n for j in i:n]
    vals = Vector{T}(undef, length(pairs))
    inv_denom = one(T) / (4 * eps_step^2)

    function _eval_pair(idx::Int)
        i, j = pairs[idx]
        xpp = copy(x); xpp[i] += eps_step; xpp[j] += eps_step
        xpm = copy(x); xpm[i] += eps_step; xpm[j] -= eps_step
        xmp = copy(x); xmp[i] -= eps_step; xmp[j] += eps_step
        xmm = copy(x); xmm[i] -= eps_step; xmm[j] -= eps_step
        vals[idx] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) * inv_denom
    end

    if threaded && Threads.nthreads() > 1 && length(pairs) > 1
        _with_single_blas_thread() do
            Threads.@threads :dynamic for idx in eachindex(pairs)
                _eval_pair(idx)
            end
        end
    else
        for idx in eachindex(pairs)
            _eval_pair(idx)
        end
    end

    @inbounds for (idx, (i, j)) in enumerate(pairs)
        H[i, j] = vals[idx]
        H[j, i] = vals[idx]
    end
    H
end

@inline function _resolve_vcov_type(vcov_type::Symbol, distribution::Symbol)
    vcov_type in (:auto, :hessian, :sandwich, :opg) ||
        throw(ArgumentError("vcov_type must be :auto, :hessian, :sandwich, or :opg (got :$vcov_type)"))
    if vcov_type === :auto
        return distribution in (:pml, :mixture_normal, :mixture_normal_asym) ? :sandwich : :hessian
    end
    vcov_type
end

"""
Compute asymptotic parameter covariance (`:hessian`, `:sandwich`, `:opg`, or `:auto`)
via `ForwardDiff` Hessian / per-observation scores `G_t`, and propagate angle covariance
to `B₀` via full Jacobian delta-method (`_delta_B0_se`).
"""
function _nongaussian_vcov(params::AbstractVector{T},
                           U::AbstractMatrix{T},
                           L::LowerTriangular{T,<:AbstractMatrix{T}},
                           n::Int,
                           distribution::Symbol;
                           vcov_type::Symbol=:auto) where {T<:AbstractFloat}
    eff_type = _resolve_vcov_type(vcov_type, distribution)
    n_angles = n * (n - 1) ÷ 2
    n_dp = _n_dist_params(distribution) * n
    n_total = n_angles + n_dp
    T_obs = size(U, 1)
    p_vec = Vector{T}(params)

    Z_chol = Matrix{T}(transpose(L \ transpose(U)))
    logdet_inv = -sum(i -> log(max(abs(L[i, i]), eps(T))), 1:n)
    neg_ll = p -> -_nongaussian_loglik_from_whitened(p, Z_chol, logdet_inv, n, distribution)

    H_inv = if eff_type in (:hessian, :sandwich)
        try
            H_raw = Matrix{T}(ForwardDiff.hessian(neg_ll, p_vec))
            _spd_inv(H_raw)
        catch
            zeros(T, n_total, n_total)
        end
    else
        zeros(T, n_total, n_total)
    end

    J_opg = if eff_type in (:sandwich, :opg)
        obs_vec_fn = p -> _nongaussian_obs_loglik_vec(p, Z_chol, logdet_inv, n, distribution)
        G_t = Matrix{T}(ForwardDiff.jacobian(obs_vec_fn, p_vec))
        J = G_t' * G_t
        Matrix{T}((J + J') / 2)
    else
        zeros(T, n_total, n_total)
    end

    vcov_full = if eff_type === :hessian
        H_inv
    elseif eff_type === :sandwich
        V = H_inv * J_opg * H_inv
        Matrix{T}((V + V') / 2)
    else  # :opg
        try
            _spd_inv(J_opg)
        catch
            zeros(T, n_total, n_total)
        end
    end

    if n_angles == 0
        return (vcov_full, zeros(T, n, n))
    end

    angles_opt = Vector{T}(p_vec[1:n_angles])
    vcov_angles = Matrix{T}(vcov_full[1:n_angles, 1:n_angles])
    L_mat = Matrix{T}(L)
    se_B0, _ = _delta_B0_se(angles_opt, vcov_angles,
                            a -> L_mat * _givens_to_orthogonal(a, n), n)

    (vcov_full, se_B0)
end

# =============================================================================
# Core Estimation (Multi-Start L-BFGS + ForwardDiff)
# =============================================================================

function _q_to_so_givens(Q_raw::AbstractMatrix{T}, n::Int) where {T<:AbstractFloat}
    n <= 1 && return T[]
    Q_so = Matrix{T}(Q_raw)
    if det(Q_so) < zero(T)
        Q_so[:, n] .*= -one(T)
    end
    _orthogonal_to_givens(Q_so, n)
end

function _default_dist_start(::Type{T}, n::Int, distribution::Symbol, s::Int) where {T<:AbstractFloat}
    if distribution === :student_t
        return fill(isodd(s) ? log(T(5.0) - T(2.01)) : log(T(3.5) - T(2.01)), n)
    elseif distribution === :mixture_normal
        return repeat(isodd(s) ? T[-1.1, 0.85] : T[1.1, -0.85], n)
    elseif distribution === :mixture_normal_asym
        if s == 1
            return repeat(T[-1.5, 0.6, 0.5], n)
        elseif s == 2
            return repeat(T[-1.5, -0.6, 0.5], n)
        elseif s == 3
            return repeat(T[1.1, 0.5, -0.5], n)
        elseif s == 4
            return repeat(T[-1.0, -0.6, -0.5], n)
        else
            return repeat(isodd(s) ? T[0.0, 0.5, 0.8] : T[0.0, -0.5, -0.8], n)
        end
    elseif distribution === :pml
        return repeat([isodd(s) ? T(0.3) : T(-0.3), log(T(5.0) - T(2.05))], n)
    elseif distribution === :skew_normal
        return fill(isodd(s) ? T(0.5) : T(-0.5), n)
    end
    throw(ArgumentError("Unknown distribution: $distribution"))
end

"""Internal: estimate non-Gaussian ML SVAR for a given distribution."""
function _estimate_nongaussian_ml(model::VARModel{T}, distribution::Symbol;
                                  max_iter::Int=500, tol::Real=T(1e-6),
                                  dist_init::Union{Nothing, AbstractVector{<:Real}}=nothing,
                                  vcov_type::Symbol=:auto,
                                  n_starts::Int=5,
                                  rng::AbstractRNG=Random.default_rng(),
                                  seed::Union{Integer,Nothing}=nothing) where {T<:AbstractFloat}
    n_starts >= 1 || throw(ArgumentError("n_starts must be >= 1 (got $n_starts)"))
    eff_vcov_type = _resolve_vcov_type(vcov_type, distribution)
    rng = _resolve_repro_rng(rng, seed)

    n = nvars(model)
    L = safe_cholesky(model.Sigma)
    Z_chol = Matrix{T}(transpose(L \ transpose(model.U)))
    logdet_inv = -sum(i -> log(max(abs(L[i, i]), eps(T))), 1:n)
    T_obs = size(model.U, 1)

    n_angles = n * (n - 1) ÷ 2
    n_dp = _n_dist_params(distribution)
    n_dp_total = n_dp * n
    tol_T = T(tol)

    # Build multi-start angle initializations:
    # Start 1: Cholesky (0 angles); Start 2: JADE warm-start;
    # Start 3: FastICA warm-start; Starts 4..n_starts: Haar random SO(n)
    angle_starts = Vector{Vector{T}}(undef, n_starts)
    angle_starts[1] = zeros(T, n_angles)
    if n_starts >= 2
        angle_starts[2] = if n >= 2
            try
                _q_to_so_givens(identify_jade(model).Q, n)
            catch
                _q_to_so_givens(generate_Q(n, T; rng=rng), n)
            end
        else
            T[]
        end
    end
    if n_starts >= 3
        angle_starts[3] = if n >= 2
            try
                _q_to_so_givens(identify_fastica(model; rng=rng).Q, n)
            catch
                _q_to_so_givens(generate_Q(n, T; rng=rng), n)
            end
        else
            T[]
        end
    end
    for s in 4:n_starts
        angle_starts[s] = n >= 2 ? _q_to_so_givens(generate_Q(n, T; rng=rng), n) : T[]
    end

    dist_starts = Vector{Vector{T}}(undef, n_starts)
    for s in 1:n_starts
        if dist_init !== nothing
            d0 = Vector{T}(dist_init)
            length(d0) == n_dp_total || throw(ArgumentError(
                "dist_init must have length $n_dp_total (got $(length(d0)))"))
            dist_starts[s] = s == 1 ? d0 : (d0 .+ T(0.1) .* randn(rng, T, n_dp_total))
        else
            dist_starts[s] = _default_dist_start(T, n, distribution, s)
        end
    end

    neg_ll = p -> -_nongaussian_loglik_from_whitened(p, Z_chol, logdet_inv, n, distribution)

    best_nll = T(Inf)
    best_params = _pack_nongaussian_params(angle_starts[1], dist_starts[1])
    best_conv = false
    best_iters = 0

    for s in 1:n_starts
        p0 = _pack_nongaussian_params(angle_starts[s], dist_starts[s])
        res = try
            cfg = ForwardDiff.GradientConfig(neg_ll, p0)
            g! = (G, x) -> ForwardDiff.gradient!(G, neg_ll, x, cfg)
            opts = Optim.Options(iterations=max_iter, g_tol=tol_T,
                                 f_reltol=(tol_T < T(1e-6) ? zero(T) : T(1e-12)),
                                 allow_f_increases=true, show_trace=false)
            Optim.optimize(neg_ll, g!, p0, Optim.LBFGS(), opts)
        catch
            try
                Optim.optimize(neg_ll, p0, Optim.NelderMead(),
                               Optim.Options(iterations=max_iter, g_tol=tol_T,
                                             show_trace=false))
            catch
                nothing
            end
        end
        res === nothing && continue
        val = T(Optim.minimum(res))
        if isfinite(val) && (!isfinite(best_nll) || val < best_nll - T(1e-10))
            best_nll = val
            best_params = Vector{T}(Optim.minimizer(res))
            best_conv = Optim.converged(res)
            best_iters = Optim.iterations(res)
        end
    end

    if !isfinite(best_nll)
        best_nll = T(neg_ll(best_params))
    end

    angles_opt, dp_opt = _unpack_nongaussian_params(best_params, n, distribution)
    Q_raw = n == 1 ? Matrix{T}(I, 1, 1) : _givens_to_orthogonal(angles_opt, n)
    B0_raw = Matrix{T}(L) * Q_raw
    shocks_raw = Z_chol * Q_raw

    loglik = -best_nll
    loglik_g = _gaussian_loglik(model)

    # Normalize signs so diag(B0) >= 0
    s_norm = [B0_raw[j, j] < zero(T) ? -one(T) : one(T) for j in 1:n]
    B0 = copy(B0_raw)
    Q = copy(Q_raw)
    shocks = copy(shocks_raw)
    @inbounds for j in 1:n
        if s_norm[j] < zero(T)
            B0[:, j] .*= -one(T)
            Q[:, j] .*= -one(T)
            shocks[:, j] .*= -one(T)
        end
    end

    # Compute vcov in the exact SO(n) chart of _Q_givens_angles(Q, n)
    params_vcov = if n <= 1
        best_params
    else
        Qp = copy(Q)
        if det(Qp) < zero(T)
            Qp[:, n] .*= -one(T)
        end
        s_qp = [dot(@view(Qp[:, j]), @view(Q_raw[:, j])) < zero(T) ? -one(T) : one(T) for j in 1:n]
        angles_qp = _orthogonal_to_givens(Qp, n)
        dp_qp = Vector{T}(dp_opt)
        if distribution === :skew_normal
            @inbounds for j in 1:n
                s_qp[j] < zero(T) && (dp_qp[j] = -dp_qp[j])
            end
        elseif distribution === :pml
            @inbounds for j in 1:n
                s_qp[j] < zero(T) && (dp_qp[(j - 1) * 2 + 1] = -dp_qp[(j - 1) * 2 + 1])
            end
        elseif distribution === :mixture_normal_asym
            @inbounds for j in 1:n
                s_qp[j] < zero(T) && (dp_qp[(j - 1) * 3 + 2] = -dp_qp[(j - 1) * 3 + 2])
            end
        end
        _pack_nongaussian_params(angles_qp, dp_qp)
    end

    vcov_mat, se_mat = _nongaussian_vcov(params_vcov, model.U, L, n, distribution;
                                         vcov_type=vcov_type)

    # Align vcov rows/cols for shock n's asymmetric parameter when det(Q) < 0
    # (params_vcov negated shock n relative to result.Q and result.dist_params)
    if n > 1 && det(Q) < zero(T)
        n_angles = n * (n - 1) ÷ 2
        k_flip = if distribution === :skew_normal
            n_angles + n
        elseif distribution === :pml
            n_angles + (n - 1) * 2 + 1
        elseif distribution === :mixture_normal_asym
            n_angles + (n - 1) * 3 + 2
        else
            0
        end
        if k_flip > 0 && k_flip <= size(vcov_mat, 1) && k_flip <= size(vcov_mat, 2)
            vcov_mat[:, k_flip] .*= -one(T)
            vcov_mat[k_flip, :] .*= -one(T)
        end
    end

    # Distribution parameters dict (consistent with sign-normalized shocks)
    dp_dict = Dict{Symbol, Any}(:vcov_type => eff_vcov_type)
    if distribution === :student_t
        dp_dict[:nu] = [T(2.01) + exp(clamp(dp_opt[j], T(-30), T(12))) for j in 1:n]
        dp_dict[:all_params] = copy(dp_dict[:nu])
    elseif distribution === :mixture_normal
        p_mix_vals = Vector{T}(undef, n)
        sigma1_vals = Vector{T}(undef, n)
        sigma2_vals = Vector{T}(undef, n)
        for j in 1:n
            idx = (j - 1) * 2
            pm, s1, s2 = _unpack_shock_mixture(dp_opt[idx + 1], dp_opt[idx + 2])
            p_mix_vals[j] = pm
            sigma1_vals[j] = s1
            sigma2_vals[j] = s2
        end
        dp_dict[:p_mix] = p_mix_vals
        dp_dict[:sigma1] = sigma1_vals
        dp_dict[:sigma2] = sigma2_vals
        dp_dict[:all_params] = vcat(p_mix_vals, sigma1_vals)
    elseif distribution === :mixture_normal_asym
        p_mix_vals = Vector{T}(undef, n)
        mu1_vals = Vector{T}(undef, n)
        sigma1_vals = Vector{T}(undef, n)
        mu2_vals = Vector{T}(undef, n)
        sigma2_vals = Vector{T}(undef, n)
        for j in 1:n
            idx = (j - 1) * 3
            pm, m1, s1, m2, s2 = _unpack_shock_mixture_asym(
                dp_opt[idx + 1], dp_opt[idx + 2], dp_opt[idx + 3])
            p_mix_vals[j] = pm
            mu1_vals[j] = s_norm[j] * m1
            sigma1_vals[j] = s1
            mu2_vals[j] = s_norm[j] * m2
            sigma2_vals[j] = s2
        end
        dp_dict[:p_mix] = p_mix_vals
        dp_dict[:mu1] = mu1_vals
        dp_dict[:sigma1] = sigma1_vals
        dp_dict[:mu2] = mu2_vals
        dp_dict[:sigma2] = sigma2_vals
        dp_dict[:all_params] = vcat(p_mix_vals, mu1_vals, sigma1_vals)
    elseif distribution === :pml
        # Pearson IV: :kappa holds skewness ν, :nu the tail exponent m
        dp_dict[:kappa] = [s_norm[j] * dp_opt[(j - 1) * 2 + 1] for j in 1:n]
        dp_dict[:nu] = [T(2.05) + exp(clamp(dp_opt[(j - 1) * 2 + 2], T(-30), T(9.2))) for j in 1:n]
        dp_dict[:all_params] = vcat(dp_dict[:kappa], dp_dict[:nu])
    elseif distribution === :skew_normal
        dp_dict[:alpha] = [s_norm[j] * dp_opt[j] for j in 1:n]
        dp_dict[:all_params] = copy(dp_dict[:alpha])
    end

    n_params = length(best_params)
    aic_val = -T(2) * loglik + T(2) * n_params
    bic_val = -T(2) * loglik + log(T(T_obs)) * n_params

    NonGaussianMLResult{T}(B0, Q, shocks, distribution, loglik, loglik_g, dp_dict,
                           vcov_mat, se_mat, best_conv, best_iters, aic_val, bic_val)
end

# =============================================================================
# Public API
# =============================================================================

"""
    identify_student_t(model::VARModel; max_iter=500, tol=1e-6, vcov_type=:auto,
                       n_starts=5, rng=Random.default_rng(), seed=nothing) -> NonGaussianMLResult

Identify SVAR assuming Student-t distributed structural shocks via multi-start `LBFGS()`
with `ForwardDiff` gradients.

Each shock εⱼ ~ t(νⱼ) (standardized to unit variance). Identification is achieved
when at most one νⱼ = ∞ (Gaussian). `vcov_type` controls asymptotic covariance
(`:auto` selects `:hessian`; `:sandwich` and `:opg` are also supported).

**Reference**: Lanne, Meitz & Saikkonen (2017)
"""
function identify_student_t(model::VARModel{T}; max_iter::Int=500,
                            tol::Real=T(1e-6), vcov_type::Symbol=:auto,
                            n_starts::Int=5,
                            rng::AbstractRNG=Random.default_rng(),
                            seed::Union{Integer,Nothing}=nothing) where {T<:AbstractFloat}
    _estimate_nongaussian_ml(model, :student_t; max_iter=max_iter, tol=tol,
                             vcov_type=vcov_type, n_starts=n_starts,
                             rng=rng, seed=seed)
end

"""
    identify_mixture_normal(model::VARModel; n_components=2, asymmetric=false,
                            max_iter=500, tol=1e-6, vcov_type=:auto, n_starts=5,
                            rng=Random.default_rng(), seed=nothing) -> NonGaussianMLResult

Identify SVAR assuming two-component Gaussian mixture structural shocks via multi-start
`LBFGS()` with `ForwardDiff` gradients.

- `asymmetric=false` (default): zero-mean mixture εⱼ ~ pⱼ N(0, σ₁ⱼ²) + (1-pⱼ) N(0, σ₂ⱼ²)
  with unit variance (`:mixture_normal`, Lanne & Lütkepohl 2010).
- `asymmetric=true`: non-zero-mean mixture εⱼ ~ pⱼ N(μ₁ⱼ, σ₁ⱼ²) + (1-pⱼ) N(μ₂ⱼ, σ₂ⱼ²)
  with zero mean and unit variance (`:mixture_normal_asym`, Gouriéroux, Monfort & Renne 2020).

`vcov_type=:auto` selects the Gouriéroux, Monfort & Renne (2017) Huber–White sandwich
covariance; `:hessian` and `:opg` are also supported.

**References**: Lanne & Lütkepohl (2010); Gouriéroux, Monfort & Renne (2017, 2020)
"""
function identify_mixture_normal(model::VARModel{T}; n_components::Int=2,
                                 asymmetric::Bool=false,
                                 max_iter::Int=500, tol::Real=T(1e-6),
                                 vcov_type::Symbol=:auto, n_starts::Int=5,
                                 rng::AbstractRNG=Random.default_rng(),
                                 seed::Union{Integer,Nothing}=nothing) where {T<:AbstractFloat}
    n_components == 2 || throw(ArgumentError(
        "identify_mixture_normal currently supports only n_components=2 (got $n_components)"))
    dist = asymmetric ? :mixture_normal_asym : :mixture_normal
    _estimate_nongaussian_ml(model, dist; max_iter=max_iter, tol=tol,
                             vcov_type=vcov_type, n_starts=n_starts,
                             rng=rng, seed=seed)
end

"""
    identify_pml(model::VARModel; max_iter=500, tol=1e-6, vcov_type=:auto,
                 n_starts=5, rng=Random.default_rng(), seed=nothing) -> NonGaussianMLResult

Identify SVAR by pseudo-maximum likelihood with Pearson Type IV structural shocks
via multi-start `LBFGS()` with `ForwardDiff` gradients (genuine normalized density since
#566; the `:pml` name is kept for API stability).

Allows both skewness and excess kurtosis in the structural shocks: per shock,
`dist_params[:kappa]` is the Pearson-IV skewness parameter ν (sign flips the skew
direction; 0 = symmetric) and `dist_params[:nu]` the tail exponent m (> 2; at ν = 0
the density is the unit-variance scaled Student t with 2m−1 df). `vcov_type=:auto`
selects the Gouriéroux, Monfort & Renne (2017) Huber–White sandwich covariance.

**References**: Pearson (1895); Heinrich (2004); Gouriéroux, Monfort & Renne (2017);
Herwartz (2018).
"""
function identify_pml(model::VARModel{T}; max_iter::Int=500,
                      tol::Real=T(1e-6), vcov_type::Symbol=:auto,
                      n_starts::Int=5,
                      rng::AbstractRNG=Random.default_rng(),
                      seed::Union{Integer,Nothing}=nothing) where {T<:AbstractFloat}
    _estimate_nongaussian_ml(model, :pml; max_iter=max_iter, tol=tol,
                             vcov_type=vcov_type, n_starts=n_starts,
                             rng=rng, seed=seed)
end

"""
    identify_skew_normal(model::VARModel; max_iter=500, tol=1e-6, vcov_type=:auto,
                         n_starts=5, rng=Random.default_rng(), seed=nothing) -> NonGaussianMLResult

Identify SVAR assuming skew-normal distributed structural shocks via multi-start
`LBFGS()` with `ForwardDiff` gradients.

Each shock εⱼ has pdf f(x) = 2 φ(x) Φ(αⱼ x), where αⱼ controls skewness.
`vcov_type=:auto` selects `:hessian`; `:sandwich` and `:opg` are also supported.

**Reference**: Azzalini (1985)
"""
function identify_skew_normal(model::VARModel{T}; max_iter::Int=500,
                              tol::Real=T(1e-6), vcov_type::Symbol=:auto,
                              n_starts::Int=5,
                              rng::AbstractRNG=Random.default_rng(),
                              seed::Union{Integer,Nothing}=nothing) where {T<:AbstractFloat}
    _estimate_nongaussian_ml(model, :skew_normal; max_iter=max_iter, tol=tol,
                             vcov_type=vcov_type, n_starts=n_starts,
                             rng=rng, seed=seed)
end

"""
    identify_nongaussian_ml(model::VARModel; distribution=:student_t, asymmetric=false,
                            max_iter=500, tol=1e-6, vcov_type=:auto, n_starts=5,
                            rng=Random.default_rng(), seed=nothing) -> NonGaussianMLResult

Unified non-Gaussian ML SVAR identification dispatcher using multi-start `LBFGS()`
with `ForwardDiff` gradients.

Supported distributions:
- `:student_t` — independent Student-t shocks (Lanne, Meitz & Saikkonen 2017)
- `:mixture_normal` — zero-mean mixture of two normals (Lanne & Lütkepohl 2010), or asymmetric when `asymmetric=true`
- `:mixture_normal_asym` — asymmetric 2-component Gaussian mixture (Gouriéroux, Monfort & Renne 2020)
- `:pml` — Pearson Type IV pseudo-ML (Heinrich 2004; Gouriéroux, Monfort & Renne 2017; Herwartz 2018)
- `:skew_normal` — skew-normal (Azzalini 1985)
"""
function identify_nongaussian_ml(model::VARModel{T}; distribution::Symbol=:student_t,
                                 asymmetric::Bool=false,
                                 max_iter::Int=500, tol::Real=T(1e-6),
                                 vcov_type::Symbol=:auto, n_starts::Int=5,
                                 rng::AbstractRNG=Random.default_rng(),
                                 seed::Union{Integer,Nothing}=nothing) where {T<:AbstractFloat}
    dist = (distribution === :mixture_normal && asymmetric) ? :mixture_normal_asym : distribution
    _estimate_nongaussian_ml(model, dist; max_iter=max_iter, tol=tol,
                             vcov_type=vcov_type, n_starts=n_starts,
                             rng=rng, seed=seed)
end
