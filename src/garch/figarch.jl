# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
FIGARCH / FIEGARCH estimation (fractionally-integrated volatility, EV-14, #422).

- FIGARCH(p,d,q) — Baillie, Bollerslev & Mikkelsen (1996). ARCH(∞) representation
  σ²ₜ = ω* + Σ λᵢ ε²_{t−i}, with λ-weights built from EV-13's `_frac_diff_weights`
  ((1−L)ᵈ) convolved with φ(L)/(1−β(L)) and truncated at `truncation` lags.
- FIEGARCH — Bollerslev & Mikkelsen (1996). Log-variance MA(∞)
  ln σ²ₜ = ω + Σ ψⱼ g(z_{t−1−j}), g(z)=θz+γ(|z|−E|z|), ψ from `_frac_diff_weights(−d,·)`.

Reuses the shared volatility helpers (`_volatility_negloglik`, `_compute_aic_bic`,
`_sanitize_init_params`, `_numerical_hessian`, `_qmle_sandwich_cov`, `robust_inv`)
and EV-13's `_frac_diff_weights` (imported — not re-derived). The QMLE score matrix
is built by finite differences (the fractional-weight recursion is not
ForwardDiff-safe), then plugged into the existing Bollerslev–Wooldridge sandwich.
"""

import Optim

# =============================================================================
# Transform helpers (log / logit) and a finite-difference score jacobian
# =============================================================================

_logistic(x) = inv(one(x) + exp(-x))              # ℝ → (0,1)
_logit(v) = log(v / (one(v) - v))                 # (0,1) → ℝ

"""
    _fd_jacobian(f, x; step) -> Matrix

Central finite-difference Jacobian of the vector-valued `f` at `x` (rows = elements
of `f(x)`, columns = elements of `x`). Used to assemble the QMLE per-observation
score matrix `S` for the Bollerslev–Wooldridge sandwich, because the fractional
`_frac_diff_weights` recursion is not ForwardDiff-differentiable.
"""
function _fd_jacobian(f, x::Vector{T}; step::T=T(1e-5)) where {T}
    k = length(x)
    f0 = f(x)
    J = Matrix{T}(undef, length(f0), k)
    @inbounds for j in 1:k
        xp = copy(x); xp[j] += step
        xm = copy(x); xm[j] -= step
        J[:, j] = (f(xp) .- f(xm)) ./ (2step)
    end
    J
end

# =============================================================================
# FIGARCH ARCH(∞) weights and filter
# =============================================================================

"""
    _figarch_lambda(d, phi, beta, K) -> Vector

Truncated ARCH(∞) weights `[λ₁,…,λ_K]` of a FIGARCH(p,d,q), where

    λ(L) = 1 − (1−β(L))⁻¹ φ(L)(1−L)ᵈ,   φ(L)=1−φ₁L−…, β(L)=β₁L+….

Built by convolving the `(1−L)ᵈ` fractional-difference weights (EV-13's
`_frac_diff_weights`, so `δ₀=1, δ_k=δ_{k−1}(k−1−d)/k`) with `φ(L)`, then dividing by
`(1−β(L))` via the recursion `c_k = g_k + Σβ_j c_{k−j}` (`g_k = δ_k − Σφ_j δ_{k−j}`).
Since `λ_i = −c_i` and `c₀=1`, `λ₀=0` (dropped). At `d=0`, `q=p=1`, φ, β this
reduces to `λ_i = (φ−β)β^{i−1}` — the GARCH(1,1) ARCH(∞) weights with `α=φ−β`.
"""
function _figarch_lambda(d::T, phi::AbstractVector{T}, beta::AbstractVector{T}, K::Int) where {T<:AbstractFloat}
    q = length(phi)
    p = length(beta)
    delta = _frac_diff_weights(d, K)            # δ₀..δ_K  (indexed delta[k+1])
    g = Vector{T}(undef, K + 1)
    @inbounds for k in 0:K
        gk = delta[k+1]
        for j in 1:q
            k - j >= 0 && (gk -= phi[j] * delta[k-j+1])
        end
        g[k+1] = gk
    end
    c = Vector{T}(undef, K + 1)
    @inbounds for k in 0:K
        ck = g[k+1]
        for j in 1:p
            k - j >= 0 && (ck += beta[j] * c[k-j+1])
        end
        c[k+1] = ck
    end
    lambda = Vector{T}(undef, K)
    @inbounds for i in 1:K
        lambda[i] = -c[i+1]
    end
    lambda
end

"""
    _figarch_nonneg_count(lambda; warn=true) -> Int

Count the negative entries of the truncated ARCH(∞) weights `lambda` (below
`−√eps`), i.e. violations of the Baillie–Bollerslev–Mikkelsen non-negativity
conditions. When `warn=true` and any are negative, emit a `@warn` (never throws) —
the FIGARCH variance process is only guaranteed positive when all `λᵢ ≥ 0`.
"""
function _figarch_nonneg_count(lambda::AbstractVector{T}; warn::Bool=true) where {T}
    n_neg = count(<(-sqrt(eps(T))), lambda)
    if warn && n_neg > 0
        @warn "FIGARCH: $n_neg of $(length(lambda)) truncated λ-weights are negative — " *
              "Baillie–Bollerslev–Mikkelsen non-negativity conditions violated; " *
              "conditional variances may not be guaranteed positive."
    end
    n_neg
end

"""
    _figarch_filter(omega_star, lambda, resid_sq, backcast) -> Vector

Conditional variances of the truncated FIGARCH ARCH(∞):
`σ²ₜ = ω* + Σ_{i=1}^{K} λᵢ ê²_{t−i}` with `ê²_{t−i} = resid_sq[t−i]` if `t−i ≥ 1`
else `backcast` (sample second moment of the residuals, matching `rugarch`). Floored
at `eps(T)`.
"""
function _figarch_filter(omega_star::T, lambda::AbstractVector{T},
                         resid_sq::AbstractVector{T}, backcast::T) where {T}
    n = length(resid_sq)
    K = length(lambda)
    h = Vector{T}(undef, n)
    @inbounds for t in 1:n
        s = omega_star
        for i in 1:K
            e2 = t - i >= 1 ? resid_sq[t-i] : backcast
            s += lambda[i] * e2
        end
        h[t] = max(s, eps(T))
    end
    h
end

# =============================================================================
# FIGARCH (negative) log-likelihood
# =============================================================================

# optimization-space params: [μ, log ω, logit φ₁..φq, logit β₁..βp, logit d]
function _figarch_unpack(params::AbstractVector{T}, p::Int, q::Int) where {T}
    mu = params[1]
    omega = exp(params[2])
    phi = _logistic.(params[3:2+q])
    beta = _logistic.(params[3+q:2+q+p])
    d = _logistic(params[3+q+p])
    mu, omega, phi, beta, d
end

function _figarch_sigma2(params::AbstractVector{T}, y::AbstractVector{T},
                         p::Int, q::Int, K::Int) where {T}
    mu, omega, phi, beta, d = _figarch_unpack(params, p, q)
    sb = sum(beta)
    resid = y .- mu
    rsq = resid .^ 2
    backcast = mean(rsq)
    lambda = _figarch_lambda(d, phi, beta, K)
    omega_star = omega / (one(T) - sb)
    _figarch_filter(omega_star, lambda, rsq, backcast), rsq
end

function _figarch_negloglik(params::AbstractVector{T}, y::AbstractVector{T},
                            p::Int, q::Int, K::Int) where {T}
    _, _, _, beta, _ = _figarch_unpack(params, p, q)
    sum(beta) >= one(T) && return T(1e10)
    h, rsq = _figarch_sigma2(params, y, p, q, K)
    _volatility_negloglik(h, rsq, length(y))
end

function _figarch_loglik_contribs(params::AbstractVector{T}, y::AbstractVector{T},
                                  p::Int, q::Int, K::Int) where {T}
    h, rsq = _figarch_sigma2(params, y, p, q, K)
    _volatility_loglik_contribs(h, rsq)
end

# =============================================================================
# FIGARCH estimation
# =============================================================================

"""
    estimate_figarch(r; p=1, q=1, d0=0.4, truncation=1000, dist=:normal) -> FIGARCHModel

Estimate a FIGARCH(p,d,q) model (Baillie, Bollerslev & Mikkelsen 1996) by Gaussian
QMLE. The conditional variance is the truncated ARCH(∞)

```math
\\sigma^2_t = \\frac{\\omega}{1-\\beta(1)} + \\Big[1 - (1-\\beta(L))^{-1}\\phi(L)(1-L)^d\\Big]\\varepsilon^2_t
           = \\omega^* + \\sum_{i=1}^{K} \\lambda_i \\varepsilon^2_{t-i},
```

with the λ-weights built from EV-13's `(1-L)^d` fractional-difference weights and
truncated at `truncation` lags `K`. The memory parameter `d ∈ (0,1)` yields the
hyperbolic decay `λ_i ∝ i^{-1-d}`.

# Keywords
- `p`, `q`: `β(L)` (GARCH) and `φ(L)` (ARCH) polynomial orders (default 1, 1).
- `d0`: initial value for `d ∈ (0,1)` (default 0.4).
- `truncation`: ARCH(∞) truncation lag `K` (default 1000). The weights decay only
  ``\\propto i^{-1-d}``, so **do not silently lower it** for small `d`.
- `dist`: innovation distribution — only `:normal` (Gaussian QMLE) is supported.

Standard errors use the Bollerslev–Wooldridge (1992) QML sandwich with a
delta-method back-transform. Non-negativity of the truncated λ-weights (the BBM
inequality conditions) is checked after fitting; a violation is **warned** (never
thrown) and the count stored in `m.n_neg_lambda`.

# Example
```julia
m = estimate_figarch(r; truncation=500)
report(m)
m.d            # long-memory parameter
```
"""
function estimate_figarch(r::AbstractVector{T}; p::Int=1, q::Int=1, d0::Real=0.4,
                          truncation::Int=1000, dist::Symbol=:normal) where {T<:AbstractFloat}
    _validate_data(r, "r")
    _validate_volatility_inputs(r, p, q)
    dist === :normal || throw(ArgumentError("estimate_figarch only supports dist=:normal (Gaussian QMLE), got :$dist"))
    truncation >= 1 || throw(ArgumentError("truncation must be ≥ 1, got $truncation"))
    (0 < d0 < 1) || throw(ArgumentError("d0 must be in (0,1), got $d0"))
    y = Vector{T}(r)
    n = length(y)
    K = min(truncation, n - 1)              # cannot use more lags than available edge

    # --- initial parameters (optimization space) --------------------------------
    mu0 = mean(y)
    phi0 = fill(T(0.3) / q, q)
    beta0 = fill(T(0.5) / p, p)
    var0 = var(y .- mu0; corrected=false)
    omega0 = max(var0 * (one(T) - sum(beta0)) * T(0.1), eps(T))
    params0 = _sanitize_init_params(vcat(mu0, log(omega0),
                                         _logit.(phi0), _logit.(beta0), _logit(T(d0))))

    obj = params -> _figarch_negloglik(params, y, p, q, K)
    res1 = Optim.optimize(obj, params0, Optim.NelderMead(),
                          Optim.Options(iterations=3000, show_trace=false))
    res = Optim.optimize(obj, Optim.minimizer(res1), Optim.LBFGS(),
                         Optim.Options(iterations=1000, g_tol=T(1e-8),
                                       f_reltol=T(1e-11), show_trace=false))
    converged = Optim.converged(res) || Optim.converged(res1)

    p_opt = Optim.minimizer(res)
    mu, omega, phi, beta, d = _figarch_unpack(p_opt, p, q)
    lambda = _figarch_lambda(d, phi, beta, K)
    omega_star = omega / (one(T) - sum(beta))

    resid = y .- mu
    rsq = resid .^ 2
    h = _figarch_filter(omega_star, lambda, rsq, mean(rsq))
    z = resid ./ sqrt.(h)

    loglik = -Optim.minimum(res)
    k = 3 + q + p
    aic_val, bic_val = _compute_aic_bic(loglik, k, n)

    # BBM non-negativity: warn (do not throw) on negative truncated λ-weights
    n_neg = _figarch_nonneg_count(lambda)

    # Bollerslev–Wooldridge QMLE sandwich (optimization space), finite-difference score
    param_vcov = try
        H = _numerical_hessian(obj, p_opt)
        S = _fd_jacobian(θ -> _figarch_loglik_contribs(θ, y, p, q, K), p_opt)
        Matrix{T}(_qmle_sandwich_cov(H, S))
    catch
        fill(T(NaN), k, k)
    end

    FIGARCHModel{T}(y, p, q, mu, omega, phi, beta, d, lambda, h, z, resid, fill(mu, n),
                    loglik, aic_val, bic_val, K, n_neg, :qmle, converged,
                    Optim.iterations(res), param_vcov)
end

estimate_figarch(r::AbstractVector; kwargs...) = estimate_figarch(Float64.(r); kwargs...)

"""
    StatsAPI.stderror(m::FIGARCHModel{T}; cov_type=:robust) -> Vector{T}

Standard errors for `coef(m) = [μ, ω, φ₁…φq, β₁…βp, d]` via the delta method on the
optimization→natural transform (`ω=exp·`, `φ/β/d=logistic·`). `cov_type`: `:robust`
(default) is the cached Bollerslev–Wooldridge QMLE sandwich `H⁻¹(S'S)H⁻¹`;
`:hessian` is inverse observed information.
"""
function StatsAPI.stderror(m::FIGARCHModel{T}; cov_type::Symbol=:robust) where {T}
    p, q = m.p, m.q
    k = 3 + q + p
    cov_type in (:robust, :qmle, :sandwich, :bw, :hessian, :opg_hessian) ||
        throw(ArgumentError("cov_type must be :robust or :hessian, got :$cov_type"))

    p_opt = vcat(m.mu, log(m.omega), _logit.(m.phi), _logit.(m.beta), _logit(m.d))
    C_opt = if cov_type in (:robust, :qmle, :sandwich, :bw) && all(isfinite, m.param_vcov)
        m.param_vcov
    else
        obj = params -> _figarch_negloglik(params, m.y, p, q, m.truncation)
        try
            H = _numerical_hessian(obj, p_opt)
            if cov_type in (:robust, :qmle, :sandwich, :bw)
                S = _fd_jacobian(θ -> _figarch_loglik_contribs(θ, m.y, p, q, m.truncation), p_opt)
                _qmle_sandwich_cov(H, S)
            else
                robust_inv(H)
            end
        catch
            return fill(T(NaN), k)
        end
    end

    dse = sqrt.(max.(diag(C_opt), zero(T)))       # SEs in optimization space
    # elementwise |∂natural/∂transformed|
    jac = Vector{T}(undef, k)
    jac[1] = one(T)                                # μ identity
    jac[2] = m.omega                               # ω = exp(·)
    for i in 1:q
        jac[2+i] = m.phi[i] * (one(T) - m.phi[i])  # logistic'
    end
    for j in 1:p
        jac[2+q+j] = m.beta[j] * (one(T) - m.beta[j])
    end
    jac[3+q+p] = m.d * (one(T) - m.d)
    jac .* dse
end

"""Standard error of the fractional-integration parameter `d`."""
d_stderror(m::FIGARCHModel) = stderror(m)[end]

# =============================================================================
# FIEGARCH log-variance MA(∞) weights and filter
# =============================================================================

"""
    _fiegarch_psi(d, phi, beta, K) -> Vector

Truncated MA(∞) weights `[ψ₀,…,ψ_K]` of the FIEGARCH log-variance filter
`(1−β(L))⁻¹ φ(L)(1−L)^{−d}`. Uses the **negative-order** fractional weights
`_frac_diff_weights(−d, K)` for `(1−L)^{−d}`, convolved with `φ(L)=1−φ₁L−…` then
divided by `(1−β(L))` (recursion `ψ_k = h_k + Σβ_j ψ_{k−j}`, `ψ₀=1`).
"""
function _fiegarch_psi(d::T, phi::AbstractVector{T}, beta::AbstractVector{T}, K::Int) where {T<:AbstractFloat}
    q = length(phi)
    p = length(beta)
    delta = _frac_diff_weights(-d, K)          # (1−L)^{−d} weights
    hh = Vector{T}(undef, K + 1)
    @inbounds for k in 0:K
        hk = delta[k+1]
        for j in 1:q
            k - j >= 0 && (hk -= phi[j] * delta[k-j+1])
        end
        hh[k+1] = hk
    end
    psi = Vector{T}(undef, K + 1)
    @inbounds for k in 0:K
        pk = hh[k+1]
        for j in 1:p
            k - j >= 0 && (pk += beta[j] * psi[k-j+1])
        end
        psi[k+1] = pk
    end
    psi
end

"""
    _fiegarch_filter(omega, theta, gamma, psi, resid) -> (h, z, log_h)

FIEGARCH conditional variances via the truncated log-variance recursion
`ln σ²ₜ = ω + Σ_{j=0}^{K} ψⱼ g(z_{t−1−j})`, `g(z)=θz+γ(|z|−E|z|)`,
`E|z|=√(2/π)`, pre-sample `g(z)=0`. `z_t = ε_t/σ_t` is computed sequentially. The
log-variance is clamped to `[−50, 50]` for numerical safety.
"""
function _fiegarch_filter(omega::T, theta::T, gamma::T, psi::AbstractVector{T},
                          resid::AbstractVector{T}) where {T}
    n = length(resid)
    K = length(psi) - 1
    E_abs_z = sqrt(T(2) / T(π))
    h = Vector{T}(undef, n)
    z = Vector{T}(undef, n)
    gz = Vector{T}(undef, n)      # news g(z_t)
    @inbounds for t in 1:n
        s = omega
        for j in 0:K
            idx = t - 1 - j
            idx >= 1 && (s += psi[j+1] * gz[idx])
        end
        s = clamp(s, T(-50), T(50))
        h[t] = exp(s)
        z[t] = resid[t] / sqrt(h[t])
        gz[t] = theta * z[t] + gamma * (abs(z[t]) - E_abs_z)
    end
    h, z, log.(h)
end

# =============================================================================
# FIEGARCH (negative) log-likelihood
# =============================================================================

# optimization-space params: [μ, ω, θ, γ, logit φ₁..φq, logit β₁..βp, logit d]
function _fiegarch_unpack(params::AbstractVector{T}, p::Int, q::Int) where {T}
    mu = params[1]
    omega = params[2]
    theta = params[3]
    gamma = params[4]
    phi = _logistic.(params[5:4+q])
    beta = _logistic.(params[5+q:4+q+p])
    d = _logistic(params[5+q+p])
    mu, omega, theta, gamma, phi, beta, d
end

function _fiegarch_sigma2(params::AbstractVector{T}, y::AbstractVector{T},
                          p::Int, q::Int, K::Int) where {T}
    mu, omega, theta, gamma, phi, beta, d = _fiegarch_unpack(params, p, q)
    resid = y .- mu
    psi = _fiegarch_psi(d, phi, beta, K)
    h, _, _ = _fiegarch_filter(omega, theta, gamma, psi, resid)
    h, resid .^ 2
end

function _fiegarch_negloglik(params::AbstractVector{T}, y::AbstractVector{T},
                             p::Int, q::Int, K::Int) where {T}
    h, rsq = _fiegarch_sigma2(params, y, p, q, K)
    _volatility_negloglik(h, rsq, length(y))
end

function _fiegarch_loglik_contribs(params::AbstractVector{T}, y::AbstractVector{T},
                                   p::Int, q::Int, K::Int) where {T}
    h, rsq = _fiegarch_sigma2(params, y, p, q, K)
    _volatility_loglik_contribs(h, rsq)
end

# =============================================================================
# FIEGARCH estimation
# =============================================================================

"""
    estimate_fiegarch(r; p=1, q=1, d0=0.4, truncation=1000, dist=:normal) -> FIEGARCHModel

Estimate a FIEGARCH model (Bollerslev & Mikkelsen 1996) by Gaussian QMLE. The log
conditional variance follows the long-memory MA(∞)

```math
\\ln \\sigma^2_t = \\omega + (1-\\beta(L))^{-1}\\phi(L)(1-L)^{-d}\\, g(z_{t-1})
              = \\omega + \\sum_{j\\ge 0} \\psi_j\\, g(z_{t-1-j}),
```

with news function `g(z) = θz + γ(|z| − E|z|)` (θ the sign/leverage term, γ the
magnitude term) and MA(∞) weights `ψ` built from the negative-order fractional
weights `(1-L)^{-d}` (EV-13's `_frac_diff_weights(-d, ·)`). Because the log variance
is unconstrained there is **no positivity constraint**; `d ∈ (0,1)` is
logit-transformed and `truncation` fixes the MA(∞) lag.

# Example
```julia
m = estimate_fiegarch(r; truncation=500)
report(m)
m.gamma        # magnitude / leverage coefficient
```
"""
function estimate_fiegarch(r::AbstractVector{T}; p::Int=1, q::Int=1, d0::Real=0.4,
                           truncation::Int=1000, dist::Symbol=:normal) where {T<:AbstractFloat}
    _validate_data(r, "r")
    _validate_volatility_inputs(r, p, q)
    dist === :normal || throw(ArgumentError("estimate_fiegarch only supports dist=:normal (Gaussian QMLE), got :$dist"))
    truncation >= 1 || throw(ArgumentError("truncation must be ≥ 1, got $truncation"))
    (0 < d0 < 1) || throw(ArgumentError("d0 must be in (0,1), got $d0"))
    y = Vector{T}(r)
    n = length(y)
    K = min(truncation, n - 1)

    mu0 = mean(y)
    var0 = var(y .- mu0; corrected=false)
    omega0 = log(max(var0, eps(T)))
    theta0 = T(-0.05)
    gamma0 = T(0.1)
    phi0 = fill(T(0.3) / q, q)
    beta0 = fill(T(0.5) / p, p)
    params0 = _sanitize_init_params(vcat(mu0, omega0, theta0, gamma0,
                                         _logit.(phi0), _logit.(beta0), _logit(T(d0))))

    obj = params -> _fiegarch_negloglik(params, y, p, q, K)
    res1 = Optim.optimize(obj, params0, Optim.NelderMead(),
                          Optim.Options(iterations=3000, show_trace=false))
    res = Optim.optimize(obj, Optim.minimizer(res1), Optim.LBFGS(),
                         Optim.Options(iterations=1000, g_tol=T(1e-8),
                                       f_reltol=T(1e-11), show_trace=false))
    converged = Optim.converged(res) || Optim.converged(res1)

    p_opt = Optim.minimizer(res)
    mu, omega, theta, gamma, phi, beta, d = _fiegarch_unpack(p_opt, p, q)
    psi = _fiegarch_psi(d, phi, beta, K)
    resid = y .- mu
    h, z, _ = _fiegarch_filter(omega, theta, gamma, psi, resid)

    loglik = -Optim.minimum(res)
    k = 5 + q + p
    aic_val, bic_val = _compute_aic_bic(loglik, k, n)

    param_vcov = try
        H = _numerical_hessian(obj, p_opt)
        S = _fd_jacobian(θ -> _fiegarch_loglik_contribs(θ, y, p, q, K), p_opt)
        Matrix{T}(_qmle_sandwich_cov(H, S))
    catch
        fill(T(NaN), k, k)
    end

    FIEGARCHModel{T}(y, p, q, mu, omega, theta, gamma, phi, beta, d, psi, h, z, resid,
                     fill(mu, n), loglik, aic_val, bic_val, K, :qmle, converged,
                     Optim.iterations(res), param_vcov)
end

estimate_fiegarch(r::AbstractVector; kwargs...) = estimate_fiegarch(Float64.(r); kwargs...)

"""
    StatsAPI.stderror(m::FIEGARCHModel{T}; cov_type=:robust) -> Vector{T}

Standard errors for `coef(m) = [μ, ω, θ, γ, φ₁…φq, β₁…βp, d]`. `μ, ω, θ, γ` are
untransformed; `φ, β, d` use the logistic delta method. `cov_type`: `:robust`
(default) is the cached Bollerslev–Wooldridge QMLE sandwich; `:hessian` is inverse
observed information.
"""
function StatsAPI.stderror(m::FIEGARCHModel{T}; cov_type::Symbol=:robust) where {T}
    p, q = m.p, m.q
    k = 5 + q + p
    cov_type in (:robust, :qmle, :sandwich, :bw, :hessian, :opg_hessian) ||
        throw(ArgumentError("cov_type must be :robust or :hessian, got :$cov_type"))

    p_opt = vcat(m.mu, m.omega, m.theta, m.gamma, _logit.(m.phi), _logit.(m.beta), _logit(m.d))
    C_opt = if cov_type in (:robust, :qmle, :sandwich, :bw) && all(isfinite, m.param_vcov)
        m.param_vcov
    else
        obj = params -> _fiegarch_negloglik(params, m.y, p, q, m.truncation)
        try
            H = _numerical_hessian(obj, p_opt)
            if cov_type in (:robust, :qmle, :sandwich, :bw)
                S = _fd_jacobian(θ -> _fiegarch_loglik_contribs(θ, m.y, p, q, m.truncation), p_opt)
                _qmle_sandwich_cov(H, S)
            else
                robust_inv(H)
            end
        catch
            return fill(T(NaN), k)
        end
    end

    dse = sqrt.(max.(diag(C_opt), zero(T)))
    jac = ones(T, k)                               # μ, ω, θ, γ identity
    for i in 1:q
        jac[4+i] = m.phi[i] * (one(T) - m.phi[i])
    end
    for j in 1:p
        jac[4+q+j] = m.beta[j] * (one(T) - m.beta[j])
    end
    jac[5+q+p] = m.d * (one(T) - m.d)
    jac .* dse
end

"""Standard error of the fractional-integration parameter `d`."""
d_stderror(m::FIEGARCHModel) = stderror(m)[end]

# =============================================================================
# Forecasting (simulation, reusing the shared VolatilityForecast builder)
# =============================================================================

"""
    forecast(m::FIGARCHModel, h; conf_level=0.95, n_sim=10000) -> VolatilityForecast

Multi-step variance forecasts by iterating the truncated ARCH(∞) recursion. Future
`ε²` are simulated (`ε = σz`, `z ~ N(0,1)`); each simulated path feeds its squared
innovations back into `σ²_{t} = ω* + Σ λ_i ε²_{t-i}`. Point forecasts are path
means; CI bounds are empirical quantiles.
"""
function forecast(m::FIGARCHModel{T}, h::Int; conf_level::Real=0.95, n_sim::Int=10000,
                  rng::AbstractRNG=Random.default_rng()) where {T}
    h < 1 && throw(ArgumentError("Forecast horizon must be ≥ 1"))
    conf_level = T(conf_level)
    K = length(m.lambda)
    omega_star = m.omega / (one(T) - sum(m.beta))
    backcast = mean(m.residuals .^ 2)
    hist = m.residuals .^ 2                       # realized ε² history
    paths = Matrix{T}(undef, n_sim, h)
    for s in 1:n_sim
        buf = copy(hist)
        for t in 1:h
            ht = omega_star
            L = length(buf)
            @inbounds for i in 1:K
                e2 = L - i + 1 >= 1 ? buf[L-i+1] : backcast
                ht += m.lambda[i] * e2
            end
            ht = max(ht, eps(T))
            paths[s, t] = ht
            push!(buf, ht * randn(rng, T)^2)
        end
    end
    _build_volatility_forecast(paths, h, conf_level, :figarch)
end

"""
    forecast(m::FIEGARCHModel, h; conf_level=0.95, n_sim=10000) -> VolatilityForecast

Multi-step variance forecasts by iterating the truncated log-variance MA(∞); future
standardized innovations `z ~ N(0,1)` are simulated and fed through the news
function `g(z)`.
"""
function forecast(m::FIEGARCHModel{T}, h::Int; conf_level::Real=0.95, n_sim::Int=10000,
                  rng::AbstractRNG=Random.default_rng()) where {T}
    h < 1 && throw(ArgumentError("Forecast horizon must be ≥ 1"))
    conf_level = T(conf_level)
    K = length(m.psi) - 1
    E_abs_z = sqrt(T(2) / T(π))
    # in-sample news g(z_t) history
    gz_hist = m.theta .* m.standardized_residuals .+
              m.gamma .* (abs.(m.standardized_residuals) .- E_abs_z)
    paths = Matrix{T}(undef, n_sim, h)
    for s in 1:n_sim
        gbuf = copy(gz_hist)
        for t in 1:h
            lg = m.omega
            L = length(gbuf)
            @inbounds for j in 0:K
                idx = L - j
                idx >= 1 && (lg += m.psi[j+1] * gbuf[idx])
            end
            lg = clamp(lg, T(-50), T(50))
            ht = exp(lg)
            paths[s, t] = ht
            zt = randn(rng, T)
            push!(gbuf, m.theta * zt + m.gamma * (abs(zt) - E_abs_z))
        end
    end
    _build_volatility_forecast(paths, h, conf_level, :fiegarch)
end

StatsAPI.predict(m::FIGARCHModel, h::Int; kwargs...) = forecast(m, h; kwargs...).forecast
StatsAPI.predict(m::FIEGARCHModel, h::Int; kwargs...) = forecast(m, h; kwargs...).forecast

# =============================================================================
# News impact curve
# =============================================================================

"""
    news_impact_curve(m::FIGARCHModel; range=(-3,3), n_points=200)

FIGARCH news impact curve: how a shock `ε_{t-1}` maps to `σ²_t`, holding all deeper
lagged `ε²` at a **reference conditional variance** `σ̄²` (the sample second moment
of the residuals). With `σ²_t = ω* + λ₁ε²_{t-1} + Σ_{i≥2} λ_i σ̄²`, the curve is the
symmetric parabola `NIC(ε) = C + λ₁ε²`, `C = ω* + σ̄²Σ_{i≥2}λ_i`.
Returns `(shocks, variance)`.
"""
function news_impact_curve(m::FIGARCHModel{T}; range::Tuple{Real,Real}=(-3.0, 3.0), n_points::Int=200) where {T}
    sigma2_bar = mean(m.residuals .^ 2)
    sigma = sqrt(sigma2_bar)
    shocks = collect(LinRange(T(range[1]) * sigma, T(range[2]) * sigma, n_points))
    omega_star = m.omega / (one(T) - sum(m.beta))
    tail = zero(T)
    @inbounds for i in 2:length(m.lambda)
        tail += m.lambda[i] * sigma2_bar
    end
    C = omega_star + tail
    lam1 = m.lambda[1]
    variance = map(e -> max(C + lam1 * e^2, eps(T)), shocks)
    (shocks=shocks, variance=variance)
end

"""
    news_impact_curve(m::FIEGARCHModel; range=(-3,3), n_points=200)

FIEGARCH news impact curve (asymmetric, log form): `ε_{t-1}` enters through the
standardized shock `z = ε/σ̄` and the news function `g(z)=θz+γ(|z|−E|z|)` with unit
MA weight `ψ₀`, deeper lags held at the reference log-variance. Returns
`(shocks, variance)`. `θ ≠ 0` makes the curve asymmetric in the sign of the shock.
"""
function news_impact_curve(m::FIEGARCHModel{T}; range::Tuple{Real,Real}=(-3.0, 3.0), n_points::Int=200) where {T}
    sigma = sqrt(mean(m.residuals .^ 2))
    shocks = collect(LinRange(T(range[1]) * sigma, T(range[2]) * sigma, n_points))
    E_abs_z = sqrt(T(2) / T(π))
    base = m.omega                                   # deeper-lag news ≈ 0 at reference
    variance = map(shocks) do e
        z = e / sigma
        g = m.theta * z + m.gamma * (abs(z) - E_abs_z)
        exp(clamp(base + m.psi[1] * g, T(-50), T(50)))
    end
    (shocks=shocks, variance=variance)
end

# =============================================================================
# Display
# =============================================================================

function _show_figarch(io::IO, m::FIGARCHModel{T}) where {T}
    se = try
        stderror(m)
    catch
        fill(T(NaN), 3 + m.q + m.p)
    end
    names = String["μ (mean)", "ω (intercept)"]
    vals = T[m.mu, m.omega]
    for i in 1:m.q; push!(names, "φ[$i]"); push!(vals, m.phi[i]); end
    for j in 1:m.p; push!(names, "β[$j]"); push!(vals, m.beta[j]); end
    push!(names, "d (frac. int.)"); push!(vals, m.d)
    _coef_table(io, "FIGARCH($(m.p),d,$(m.q)) Model  [ARCH(∞) truncation K=$(m.truncation)]",
                names, vals, se; dist=:z)

    fit_data = Any[
        "Observations"       length(m.y);
        "Log-likelihood"     _fmt(m.loglik; digits=4);
        "AIC"                _fmt(m.aic; digits=4);
        "BIC"                _fmt(m.bic; digits=4);
        "Memory d"           _fmt(m.d);
        "Truncation K"       m.truncation;
        "Neg. λ-weights"     m.n_neg_lambda;
        "Converged"          _yesno(m.converged)
    ]
    _pretty_table(io, fit_data; column_labels=["Fit", "Value"], alignment=[:l, :r])
    m.n_neg_lambda > 0 && println(io,
        "Note: $(m.n_neg_lambda) negative λ-weight(s) — BBM non-negativity conditions violated.")
    _sig_legend(io)
end

function _show_fiegarch(io::IO, m::FIEGARCHModel{T}) where {T}
    se = try
        stderror(m)
    catch
        fill(T(NaN), 5 + m.q + m.p)
    end
    names = String["μ (mean)", "ω (intercept)", "θ (sign)", "γ (magnitude)"]
    vals = T[m.mu, m.omega, m.theta, m.gamma]
    for i in 1:m.q; push!(names, "φ[$i]"); push!(vals, m.phi[i]); end
    for j in 1:m.p; push!(names, "β[$j]"); push!(vals, m.beta[j]); end
    push!(names, "d (frac. int.)"); push!(vals, m.d)
    _coef_table(io, "FIEGARCH($(m.p),d,$(m.q)) Model  [MA(∞) truncation K=$(m.truncation)]",
                names, vals, se; dist=:z)

    fit_data = Any[
        "Observations"       length(m.y);
        "Log-likelihood"     _fmt(m.loglik; digits=4);
        "AIC"                _fmt(m.aic; digits=4);
        "BIC"                _fmt(m.bic; digits=4);
        "Memory d"           _fmt(m.d);
        "Truncation K"       m.truncation;
        "Converged"          _yesno(m.converged)
    ]
    _pretty_table(io, fit_data; column_labels=["Fit", "Value"], alignment=[:l, :r])
    _sig_legend(io)
end
