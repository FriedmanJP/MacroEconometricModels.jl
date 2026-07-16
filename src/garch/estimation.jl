# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
GARCH(p,q), EGARCH(p,q), and GJR-GARCH(p,q) estimation via MLE.
"""

import Optim

# =============================================================================
# GARCH Filter
# =============================================================================

"""
    _garch_filter(omega, alpha, beta, eps_sq)

Compute GARCH conditional variances:
h_t = ω + Σα_i ε²_{t-i} + Σβ_j h_{t-j}
"""
function _garch_filter(omega::T, alpha::Vector{T}, beta::Vector{T}, eps_sq::Vector{T}) where {T}
    n = length(eps_sq)
    q = length(alpha)
    p = length(beta)
    h = Vector{T}(undef, n)
    backcast = mean(eps_sq)

    @inbounds for t in 1:n
        h[t] = omega
        for i in 1:q
            eps2_lag = t - i >= 1 ? eps_sq[t-i] : backcast
            h[t] += alpha[i] * eps2_lag
        end
        for j in 1:p
            h_lag = t - j >= 1 ? h[t-j] : backcast
            h[t] += beta[j] * h_lag
        end
        h[t] = max(h[t], eps(T))
    end
    h
end

# =============================================================================
# EGARCH Filter
# =============================================================================

"""
    _egarch_filter(omega, alpha, gamma, beta, resid, backcast_logh)

Compute EGARCH conditional variances via log(h_t) recursion:
log(h_t) = ω + Σα_i(|z_{t-i}| - E|z|) + Σγ_i z_{t-i} + Σβ_j log(h_{t-j})
where z_t = ε_t / σ_t and E|z| = √(2/π) for Gaussian.
"""
function _egarch_filter(omega::T, alpha::Vector{T}, gamma::Vector{T},
                         beta::Vector{T}, resid::Vector{T}, backcast_logh::T) where {T}
    n = length(resid)
    q = length(alpha)
    p = length(beta)
    h = Vector{T}(undef, n)
    log_h = Vector{T}(undef, n)
    z = Vector{T}(undef, n)
    E_abs_z = sqrt(T(2) / T(π))  # E[|z|] for standard normal

    @inbounds for t in 1:n
        log_h[t] = omega
        for i in 1:q
            if t - i >= 1
                zt = z[t-i]
            else
                zt = zero(T)
            end
            log_h[t] += alpha[i] * (abs(zt) - E_abs_z) + gamma[i] * zt
        end
        for j in 1:p
            lh_lag = t - j >= 1 ? log_h[t-j] : backcast_logh
            log_h[t] += beta[j] * lh_lag
        end
        # Clamp to prevent overflow/underflow
        log_h[t] = clamp(log_h[t], T(-50), T(50))
        h[t] = exp(log_h[t])
        z[t] = resid[t] / sqrt(h[t])
    end
    h, z, log_h
end

# =============================================================================
# GJR-GARCH Filter
# =============================================================================

"""
    _gjr_garch_filter(omega, alpha, gamma, beta, resid)

Compute GJR-GARCH conditional variances:
h_t = ω + Σ(α_i + γ_i I(ε_{t-i}<0)) ε²_{t-i} + Σβ_j h_{t-j}
"""
function _gjr_garch_filter(omega::T, alpha::Vector{T}, gamma::Vector{T},
                            beta::Vector{T}, resid::Vector{T}) where {T}
    n = length(resid)
    q = length(alpha)
    p = length(beta)
    h = Vector{T}(undef, n)
    resid_sq = resid .^ 2
    backcast = mean(resid_sq)
    floor_val = eps(T)

    @inbounds for t in 1:n
        h[t] = omega
        for i in 1:q
            if t - i >= 1
                e2 = resid_sq[t-i]
                indicator = resid[t-i] < zero(T) ? one(T) : zero(T)
            else
                e2 = backcast
                indicator = T(0.5)  # Expected value for backcast
            end
            h[t] += (alpha[i] + gamma[i] * indicator) * e2
        end
        for j in 1:p
            h_lag = t - j >= 1 ? h[t-j] : backcast
            h[t] += beta[j] * h_lag
        end
        h[t] = max(h[t], floor_val)
    end
    h
end

# =============================================================================
# Negative Log-Likelihoods
# =============================================================================

function _garch_negloglik(params::Vector{T}, y::Vector{T}, p::Int, q::Int) where {T}
    n = length(y)
    # Unpack: mu, log(omega), log(alpha_1..q), log(beta_1..p)
    mu = params[1]
    omega = exp(params[2])
    alpha = exp.(params[3:2+q])
    beta = exp.(params[3+q:2+q+p])

    # Stationarity check
    sum(alpha) + sum(beta) >= one(T) && return T(1e10)

    resid = y .- mu
    resid_sq = resid .^ 2
    h = _garch_filter(omega, alpha, beta, resid_sq)
    _volatility_negloglik(h, resid_sq, n)
end

function _egarch_negloglik(params::Vector{T}, y::Vector{T}, p::Int, q::Int) where {T}
    n = length(y)
    # Unpack: mu, omega, alpha_1..q, gamma_1..q, beta_1..p (all unconstrained)
    mu = params[1]
    omega = params[2]
    alpha = params[3:2+q]
    gamma = params[3+q:2+2q]
    beta = params[3+2q:2+2q+p]

    # Stationarity of log-variance check
    sum(abs.(beta)) >= one(T) && return T(1e10)

    resid = y .- mu
    backcast_logh = log(var(resid; corrected=false))
    h, _, _ = _egarch_filter(omega, alpha, gamma, beta, resid, backcast_logh)
    resid_sq = resid .^ 2
    _volatility_negloglik(h, resid_sq, n)
end

function _gjr_negloglik(params::Vector{T}, y::Vector{T}, p::Int, q::Int) where {T}
    n = length(y)
    # Unpack: mu, log(omega), log(alpha_1..q), log(gamma_1..q), log(beta_1..p)
    mu = params[1]
    omega = exp(params[2])
    alpha = exp.(params[3:2+q])
    gamma = exp.(params[3+q:2+2q])
    beta = exp.(params[3+2q:2+2q+p])

    # Stationarity check: α + γ/2 + β < 1
    sum(alpha) + sum(gamma) / 2 + sum(beta) >= one(T) && return T(1e10)

    resid = y .- mu
    h = _gjr_garch_filter(omega, alpha, gamma, beta, resid)
    resid_sq = resid .^ 2
    _volatility_negloglik(h, resid_sq, n)
end

# =============================================================================
# Per-observation log-likelihood contributions (for QMLE sandwich scores)
# =============================================================================
# Mirror the negloglik unpack + filter WITHOUT the scalar stationarity penalty guard
# (evaluated at the stationary MLE, so the guard is inactive) and return the vector of
# per-observation Gaussian log-lik contributions. ForwardDiff.jacobian of these gives the
# n×k score matrix S; the filters are generic `where {T}` so Duals propagate.

function _garch_loglik_contribs(params, y, p::Int, q::Int)
    mu = params[1]
    omega = exp(params[2])
    alpha = exp.(params[3:2+q])
    beta = exp.(params[3+q:2+q+p])
    resid = y .- mu
    rsq = resid .^ 2
    h = _garch_filter(omega, alpha, beta, rsq)
    _volatility_loglik_contribs(h, rsq)
end

function _egarch_loglik_contribs(params, y, p::Int, q::Int)
    mu = params[1]
    omega = params[2]
    alpha = params[3:2+q]
    gamma = params[3+q:2+2q]
    beta = params[3+2q:2+2q+p]
    resid = y .- mu
    backcast_logh = log(var(resid; corrected=false))
    h, _, _ = _egarch_filter(omega, alpha, gamma, beta, resid, backcast_logh)
    rsq = resid .^ 2
    _volatility_loglik_contribs(h, rsq)
end

function _gjr_loglik_contribs(params, y, p::Int, q::Int)
    mu = params[1]
    omega = exp(params[2])
    alpha = exp.(params[3:2+q])
    gamma = exp.(params[3+q:2+2q])
    beta = exp.(params[3+2q:2+2q+p])
    resid = y .- mu
    h = _gjr_garch_filter(omega, alpha, gamma, beta, resid)
    rsq = resid .^ 2
    _volatility_loglik_contribs(h, rsq)
end

# =============================================================================
# GARCH Estimation
# =============================================================================

"""
    estimate_garch(y, p=1, q=1; method=:mle) -> GARCHModel

Estimate GARCH(p,q) model via Maximum Likelihood (Bollerslev 1986).

σ²ₜ = ω + α₁ε²ₜ₋₁ + ... + αqε²ₜ₋q + β₁σ²ₜ₋₁ + ... + βpσ²ₜ₋p

# Arguments
- `y`: Time series vector
- `p`: GARCH order (default 1)
- `q`: ARCH order (default 1)
- `method`: Estimation method (currently only `:mle`)

# Example
```julia
model = estimate_garch(y, 1, 1)
println("Persistence: ", persistence(model))
```
"""
function estimate_garch(y::AbstractVector{T}, p::Int=1, q::Int=1; method::Symbol=:mle) where {T<:AbstractFloat}
    _validate_data(y, "y")
    _validate_volatility_inputs(y, p, q)
    y_vec = Vector{T}(y)
    n = length(y_vec)

    # Initial parameters
    mu_init = mean(y_vec)
    var_init = var(y_vec .- mu_init; corrected=false)
    omega_init = var_init * T(0.05)
    alpha_init = fill(T(0.05), q)
    beta_init = fill(T(0.85) / p, p)

    params_init = _sanitize_init_params(vcat(mu_init, log(omega_init), log.(alpha_init), log.(beta_init)))

    # Two-stage optimization
    obj = params -> _garch_negloglik(params, y_vec, p, q)
    result1 = Optim.optimize(obj, params_init, Optim.NelderMead(),
        Optim.Options(iterations=2000, show_trace=false))
    result = Optim.optimize(obj, Optim.minimizer(result1), Optim.LBFGS(),
        Optim.Options(iterations=1000, g_tol=T(1e-8), show_trace=false))

    params_opt = Optim.minimizer(result)
    mu = params_opt[1]
    omega = exp(params_opt[2])
    alpha = exp.(params_opt[3:2+q])
    beta = exp.(params_opt[3+q:2+q+p])

    resid = y_vec .- mu
    resid_sq = resid .^ 2
    h = _garch_filter(omega, alpha, beta, resid_sq)
    z = resid ./ sqrt.(h)

    loglik = -Optim.minimum(result)
    k = 2 + q + p
    aic_val, bic_val = _compute_aic_bic(loglik, k, n)

    # Cache the QMLE sandwich covariance (optimization space) once; stderror
    # reads it instead of recomputing the numerical Hessian on every call.
    # Evaluated at the RECONSTRUCTED parameter vector (log∘exp of the stored
    # fields) — exactly what the per-call stderror recompute used before.
    params_cache = vcat(mu, log(omega), log.(alpha), log.(beta))
    param_vcov = try
        H = _numerical_hessian(obj, params_cache)
        S = ForwardDiff.jacobian(θ -> _garch_loglik_contribs(θ, y_vec, p, q), params_cache)
        Matrix{T}(_qmle_sandwich_cov(H, S))
    catch
        fill(T(NaN), k, k)
    end

    GARCHModel(y_vec, p, q, mu, omega, alpha, beta, h, z, resid, fill(mu, n),
               loglik, aic_val, bic_val, method, Optim.converged(result), Optim.iterations(result),
               param_vcov)
end

estimate_garch(y::AbstractVector, p::Int=1, q::Int=1; kwargs...) = estimate_garch(Float64.(y), p, q; kwargs...)

"""
    StatsAPI.stderror(m::GARCHModel{T}; cov_type=:robust) -> Vector{T}

Standard errors for a GARCH model via the numerical observed-information Hessian +
delta method. Returns an SE vector matching `coef(m)` = [μ, ω, α₁..αq, β₁..βp].

`cov_type`:
- `:robust` (default, aliases `:qmle`/`:sandwich`/`:bw`) — Bollerslev–Wooldridge (1992)
  QMLE sandwich `H⁻¹(S'S)H⁻¹`, consistent under non-Gaussian (fat-tailed) innovations.
- `:hessian` (alias `:opg_hessian`) — inverse observed information `H⁻¹` (valid only
  under correct Gaussian specification).
"""
function StatsAPI.stderror(m::GARCHModel{T}; cov_type::Symbol=:robust) where {T}
    p, q = m.p, m.q
    n_params = 2 + q + p

    cov_type in (:robust, :qmle, :sandwich, :bw, :hessian, :opg_hessian) ||
        throw(ArgumentError("cov_type must be :robust or :hessian, got :$cov_type"))

    C_opt = if cov_type in (:robust, :qmle, :sandwich, :bw) && all(isfinite, m.param_vcov)
        m.param_vcov  # cached at estimation time
    else
        # cache miss (hand-built model, estimation-time failure) or :hessian request
        params_opt = vcat(m.mu, log(m.omega), log.(m.alpha), log.(m.beta))
        obj = params -> _garch_negloglik(params, m.y, p, q)
        try
            H = _numerical_hessian(obj, params_opt)
            if cov_type in (:robust, :qmle, :sandwich, :bw)
                S = ForwardDiff.jacobian(θ -> _garch_loglik_contribs(θ, m.y, p, q), params_opt)
                _qmle_sandwich_cov(H, S)
            else
                robust_inv(H)
            end
        catch
            return fill(T(NaN), n_params)
        end
    end

    se = Vector{T}(undef, n_params)
    se[1] = sqrt(max(C_opt[1, 1], zero(T)))             # mu
    se[2] = m.omega * sqrt(max(C_opt[2, 2], zero(T)))   # omega
    for i in 1:q
        se[2+i] = m.alpha[i] * sqrt(max(C_opt[2+i, 2+i], zero(T)))
    end
    for j in 1:p
        se[2+q+j] = m.beta[j] * sqrt(max(C_opt[2+q+j, 2+q+j], zero(T)))
    end
    se
end

# =============================================================================
# EGARCH Estimation
# =============================================================================

"""
    estimate_egarch(y, p=1, q=1; method=:mle) -> EGARCHModel

Estimate EGARCH(p,q) model via Maximum Likelihood (Nelson 1991).

log(σ²ₜ) = ω + Σαᵢ(|zₜ₋ᵢ| - E|z|) + Σγᵢzₜ₋ᵢ + Σβⱼlog(σ²ₜ₋ⱼ)

The γ parameters capture leverage effects (typically γ < 0 for equities).

# Arguments
- `y`: Time series vector
- `p`: GARCH order (default 1)
- `q`: ARCH order (default 1)
- `method`: Estimation method (currently only `:mle`)

# Example
```julia
model = estimate_egarch(y, 1, 1)
println("Leverage: ", model.gamma[1])
```
"""
function estimate_egarch(y::AbstractVector{T}, p::Int=1, q::Int=1; method::Symbol=:mle) where {T<:AbstractFloat}
    _validate_data(y, "y")
    _validate_volatility_inputs(y, p, q)
    y_vec = Vector{T}(y)
    n = length(y_vec)

    mu_init = mean(y_vec)
    var_init = var(y_vec .- mu_init; corrected=false)
    omega_init = log(var_init) * (one(T) - T(0.9))
    alpha_init = fill(T(0.1), q)
    gamma_init = fill(T(-0.05), q)
    beta_init = fill(T(0.9) / p, p)

    params_init = _sanitize_init_params(vcat(mu_init, omega_init, alpha_init, gamma_init, beta_init))

    obj = params -> _egarch_negloglik(params, y_vec, p, q)
    result1 = Optim.optimize(obj, params_init, Optim.NelderMead(),
        Optim.Options(iterations=2000, show_trace=false))
    result = Optim.optimize(obj, Optim.minimizer(result1), Optim.LBFGS(),
        Optim.Options(iterations=1000, g_tol=T(1e-8), show_trace=false))

    params_opt = Optim.minimizer(result)
    mu = params_opt[1]
    omega = params_opt[2]
    alpha = params_opt[3:2+q]
    gamma = params_opt[3+q:2+2q]
    beta = params_opt[3+2q:2+2q+p]

    resid = y_vec .- mu
    backcast_logh = log(var(resid; corrected=false))
    h, z, _ = _egarch_filter(omega, alpha, gamma, beta, resid, backcast_logh)

    loglik = -Optim.minimum(result)
    k = 2 + 2q + p
    aic_val, bic_val = _compute_aic_bic(loglik, k, n)

    # Cache evaluated at the reconstructed parameter vector (== stored fields;
    # EGARCH params are untransformed) — matches the per-call recompute exactly
    params_cache = vcat(mu, omega, alpha, gamma, beta)
    param_vcov = try
        H = _numerical_hessian(obj, params_cache)
        S = ForwardDiff.jacobian(θ -> _egarch_loglik_contribs(θ, y_vec, p, q), params_cache)
        Matrix{T}(_qmle_sandwich_cov(H, S))
    catch
        fill(T(NaN), k, k)
    end

    EGARCHModel(y_vec, p, q, mu, omega, alpha, gamma, beta, h, z, resid, fill(mu, n),
                loglik, aic_val, bic_val, method, Optim.converged(result), Optim.iterations(result),
                param_vcov)
end

estimate_egarch(y::AbstractVector, p::Int=1, q::Int=1; kwargs...) = estimate_egarch(Float64.(y), p, q; kwargs...)

"""
    StatsAPI.stderror(m::EGARCHModel{T}; cov_type=:robust) -> Vector{T}

Standard errors for an EGARCH model via the numerical observed-information Hessian.
EGARCH params are unconstrained — no delta method needed. Returns an SE vector matching
`coef(m)` = [μ, ω, α₁..αq, γ₁..γq, β₁..βp]. `cov_type`: `:robust` (default) uses the
Bollerslev–Wooldridge QMLE sandwich `H⁻¹(S'S)H⁻¹`; `:hessian` uses inverse information.
"""
function StatsAPI.stderror(m::EGARCHModel{T}; cov_type::Symbol=:robust) where {T}
    p, q = m.p, m.q
    n_params = 2 + 2q + p

    cov_type in (:robust, :qmle, :sandwich, :bw, :hessian, :opg_hessian) ||
        throw(ArgumentError("cov_type must be :robust or :hessian, got :$cov_type"))

    C_opt = if cov_type in (:robust, :qmle, :sandwich, :bw) && all(isfinite, m.param_vcov)
        m.param_vcov  # cached at estimation time
    else
        # EGARCH parameters are all unconstrained in optimization space
        params_opt = vcat(m.mu, m.omega, m.alpha, m.gamma, m.beta)
        obj = params -> _egarch_negloglik(params, m.y, p, q)
        try
            H = _numerical_hessian(obj, params_opt)
            if cov_type in (:robust, :qmle, :sandwich, :bw)
                S = ForwardDiff.jacobian(θ -> _egarch_loglik_contribs(θ, m.y, p, q), params_opt)
                _qmle_sandwich_cov(H, S)
            else
                robust_inv(H)
            end
        catch
            return fill(T(NaN), n_params)
        end
    end

    se = sqrt.(max.(diag(C_opt), zero(T)))
    se
end

# =============================================================================
# GJR-GARCH Estimation
# =============================================================================

"""
    estimate_gjr_garch(y, p=1, q=1; method=:mle) -> GJRGARCHModel

Estimate GJR-GARCH(p,q) model via Maximum Likelihood (Glosten, Jagannathan & Runkle 1993).

σ²ₜ = ω + Σ(αᵢ + γᵢI(εₜ₋ᵢ<0))ε²ₜ₋ᵢ + Σβⱼσ²ₜ₋ⱼ

γᵢ > 0 captures the asymmetric (leverage) effect.

# Arguments
- `y`: Time series vector
- `p`: GARCH order (default 1)
- `q`: ARCH order (default 1)
- `method`: Estimation method (currently only `:mle`)

# Example
```julia
model = estimate_gjr_garch(y, 1, 1)
println("Asymmetry: ", model.gamma[1])
```
"""
function estimate_gjr_garch(y::AbstractVector{T}, p::Int=1, q::Int=1; method::Symbol=:mle) where {T<:AbstractFloat}
    _validate_data(y, "y")
    _validate_volatility_inputs(y, p, q)
    y_vec = Vector{T}(y)
    n = length(y_vec)

    mu_init = mean(y_vec)
    var_init = var(y_vec .- mu_init; corrected=false)
    omega_init = var_init * T(0.05)
    alpha_init = fill(T(0.03), q)
    gamma_init = fill(T(0.04), q)
    beta_init = fill(T(0.85) / p, p)

    params_init = _sanitize_init_params(vcat(mu_init, log(omega_init), log.(alpha_init), log.(gamma_init), log.(beta_init)))

    obj = params -> _gjr_negloglik(params, y_vec, p, q)
    result1 = Optim.optimize(obj, params_init, Optim.NelderMead(),
        Optim.Options(iterations=2000, show_trace=false))
    result = Optim.optimize(obj, Optim.minimizer(result1), Optim.LBFGS(),
        Optim.Options(iterations=1000, g_tol=T(1e-8), show_trace=false))

    params_opt = Optim.minimizer(result)
    mu = params_opt[1]
    omega = exp(params_opt[2])
    alpha = exp.(params_opt[3:2+q])
    gamma = exp.(params_opt[3+q:2+2q])
    beta = exp.(params_opt[3+2q:2+2q+p])

    resid = y_vec .- mu
    h = _gjr_garch_filter(omega, alpha, gamma, beta, resid)
    z = resid ./ sqrt.(h)

    loglik = -Optim.minimum(result)
    k = 2 + 2q + p
    aic_val, bic_val = _compute_aic_bic(loglik, k, n)

    # Cache evaluated at the reconstructed (log∘exp) parameter vector — exactly
    # what the per-call stderror recompute used before
    params_cache = vcat(mu, log(omega), log.(alpha), log.(gamma), log.(beta))
    param_vcov = try
        H = _numerical_hessian(obj, params_cache)
        S = ForwardDiff.jacobian(θ -> _gjr_loglik_contribs(θ, y_vec, p, q), params_cache)
        Matrix{T}(_qmle_sandwich_cov(H, S))
    catch
        fill(T(NaN), k, k)
    end

    GJRGARCHModel(y_vec, p, q, mu, omega, alpha, gamma, beta, h, z, resid, fill(mu, n),
                  loglik, aic_val, bic_val, method, Optim.converged(result), Optim.iterations(result),
                  param_vcov)
end

estimate_gjr_garch(y::AbstractVector, p::Int=1, q::Int=1; kwargs...) = estimate_gjr_garch(Float64.(y), p, q; kwargs...)

"""
    StatsAPI.stderror(m::GJRGARCHModel{T}; cov_type=:robust) -> Vector{T}

Standard errors for a GJR-GARCH model via the numerical observed-information Hessian +
delta method. Returns an SE vector matching `coef(m)` = [μ, ω, α₁..αq, γ₁..γq, β₁..βp].
`cov_type`: `:robust` (default) uses the Bollerslev–Wooldridge QMLE sandwich
`H⁻¹(S'S)H⁻¹`; `:hessian` uses inverse observed information.
"""
function StatsAPI.stderror(m::GJRGARCHModel{T}; cov_type::Symbol=:robust) where {T}
    p, q = m.p, m.q
    n_params = 2 + 2q + p

    cov_type in (:robust, :qmle, :sandwich, :bw, :hessian, :opg_hessian) ||
        throw(ArgumentError("cov_type must be :robust or :hessian, got :$cov_type"))

    C_opt = if cov_type in (:robust, :qmle, :sandwich, :bw) && all(isfinite, m.param_vcov)
        m.param_vcov  # cached at estimation time
    else
        params_opt = vcat(m.mu, log(m.omega), log.(m.alpha), log.(m.gamma), log.(m.beta))
        obj = params -> _gjr_negloglik(params, m.y, p, q)
        try
            H = _numerical_hessian(obj, params_opt)
            if cov_type in (:robust, :qmle, :sandwich, :bw)
                S = ForwardDiff.jacobian(θ -> _gjr_loglik_contribs(θ, m.y, p, q), params_opt)
                _qmle_sandwich_cov(H, S)
            else
                robust_inv(H)
            end
        catch
            return fill(T(NaN), n_params)
        end
    end

    se = Vector{T}(undef, n_params)
    se[1] = sqrt(max(C_opt[1, 1], zero(T)))                # mu (untransformed)
    se[2] = m.omega * sqrt(max(C_opt[2, 2], zero(T)))      # omega = exp(log_omega)
    for i in 1:q
        se[2+i] = m.alpha[i] * sqrt(max(C_opt[2+i, 2+i], zero(T)))      # alpha
    end
    for i in 1:q
        se[2+q+i] = m.gamma[i] * sqrt(max(C_opt[2+q+i, 2+q+i], zero(T))) # gamma
    end
    for j in 1:p
        se[2+2q+j] = m.beta[j] * sqrt(max(C_opt[2+2q+j, 2+2q+j], zero(T)))  # beta
    end
    se
end

# =============================================================================
# IGARCH / Component-GARCH / APARCH estimation (EV-15, #423)
# =============================================================================
# Extends the GARCH scaffold: reuses `_garch_filter`, the two-stage NelderMead→LBFGS
# optimizer, the shared `_volatility_negloglik`/`_volatility_loglik_contribs`
# likelihood pieces, `_numerical_hessian`, and the cached Bollerslev–Wooldridge
# `_qmle_sandwich_cov`. Standard errors are obtained by a ForwardDiff delta-method
# Jacobian of the reconstructed coefficient vector w.r.t. the free transform params.

# --- small transform helpers ---
_igarch_logistic(x) = inv(one(x) + exp(-x))

"""Multinomial-logit simplex: `theta` (length m-1) → weights `w` (length m, wᵢ>0, Σw=1)."""
function _simplex_from_logits(theta, m::Int)
    E = eltype(theta)
    denom = one(E)
    @inbounds for j in eachindex(theta)
        denom += exp(theta[j])
    end
    w = Vector{E}(undef, m)
    @inbounds for i in 1:(m - 1)
        w[i] = exp(theta[i]) / denom
    end
    w[m] = one(E) / denom
    w
end

# =============================================================================
# IGARCH
# =============================================================================

# free params: [mu, log(omega), θ₁..θ_{q+p-1}]  (simplex over the q+p ARCH/GARCH weights)
function _igarch_unpack(params, p::Int, q::Int)
    mu = params[1]
    omega = exp(params[2])
    w = _simplex_from_logits(params[3:end], q + p)
    alpha = w[1:q]
    beta = w[q+1:q+p]
    mu, omega, alpha, beta
end

function _igarch_negloglik(params, y, p::Int, q::Int)
    n = length(y)
    mu, omega, alpha, beta = _igarch_unpack(params, p, q)
    resid = y .- mu
    rsq = resid .^ 2
    h = _garch_filter(omega, alpha, beta, rsq)
    _volatility_negloglik(h, rsq, n)
end

function _igarch_loglik_contribs(params, y, p::Int, q::Int)
    mu, omega, alpha, beta = _igarch_unpack(params, p, q)
    resid = y .- mu
    rsq = resid .^ 2
    h = _garch_filter(omega, alpha, beta, rsq)
    _volatility_loglik_contribs(h, rsq)
end

_igarch_coef_from_free(params, p::Int, q::Int) =
    (mp = _igarch_unpack(params, p, q); vcat(mp[1], mp[2], mp[3], mp[4]))

"""
    estimate_igarch(y, p=1, q=1; method=:mle) -> IGARCHModel

Estimate an Integrated GARCH(p,q) model (Engle & Bollerslev 1986) imposing
`Σα + Σβ = 1` exactly (persistence = 1, divergent unconditional variance). The
`q+p` ARCH/GARCH weights are parameterized through a multinomial-logit simplex, so
`q+p-1` weights are free.

# Example
```julia
m = estimate_igarch(y, 1, 1)
persistence(m)  # == 1.0 exactly
```
"""
function estimate_igarch(y::AbstractVector{T}, p::Int=1, q::Int=1; method::Symbol=:mle) where {T<:AbstractFloat}
    _validate_data(y, "y")
    _validate_volatility_inputs(y, p, q)
    p < 1 && throw(ArgumentError("IGARCH requires GARCH order p ≥ 1, got $p"))
    y_vec = Vector{T}(y)
    n = length(y_vec)

    mu_init = mean(y_vec)
    var_init = var(y_vec .- mu_init; corrected=false)
    omega_init = var_init * T(0.02)
    # start near α≈0.1, remaining mass on β (unit-sum simplex)
    m_w = q + p
    w0 = fill(T(0.1) / q, q)
    append!(w0, fill((one(T) - sum(w0)) / p, p))
    # invert simplex: θ_i = log(w_i / w_last)
    theta0 = [log(w0[i] / w0[end]) for i in 1:(m_w - 1)]
    params_init = _sanitize_init_params(vcat(mu_init, log(omega_init), theta0))

    obj = params -> _igarch_negloglik(params, y_vec, p, q)
    result1 = Optim.optimize(obj, params_init, Optim.NelderMead(),
        Optim.Options(iterations=2000, show_trace=false))
    result = Optim.optimize(obj, Optim.minimizer(result1), Optim.LBFGS(),
        Optim.Options(iterations=1000, g_tol=T(1e-8), show_trace=false))

    params_opt = Optim.minimizer(result)
    mu, omega, alpha, beta = _igarch_unpack(params_opt, p, q)
    alpha = Vector{T}(alpha); beta = Vector{T}(beta)

    resid = y_vec .- mu
    rsq = resid .^ 2
    h = _garch_filter(omega, alpha, beta, rsq)
    z = resid ./ sqrt.(h)

    loglik = -Optim.minimum(result)
    k = 1 + q + p
    aic_val, bic_val = _compute_aic_bic(loglik, k, n)

    param_vcov = try
        H = _numerical_hessian(obj, params_opt)
        S = ForwardDiff.jacobian(θ -> _igarch_loglik_contribs(θ, y_vec, p, q), params_opt)
        Matrix{T}(_qmle_sandwich_cov(H, S))
    catch
        fill(T(NaN), k, k)
    end

    IGARCHModel{T}(y_vec, p, q, mu, omega, alpha, beta, h, z, resid, fill(mu, n),
                   loglik, aic_val, bic_val, method, Optim.converged(result),
                   Optim.iterations(result), param_vcov)
end

estimate_igarch(y::AbstractVector, p::Int=1, q::Int=1; kwargs...) = estimate_igarch(Float64.(y), p, q; kwargs...)

"""
    StatsAPI.stderror(m::IGARCHModel{T}; cov_type=:robust) -> Vector{T}

Delta-method standard errors matching `coef(m) = [μ, ω, α₁…αq, β₁…βp]`. The
constrained (`Σα+Σβ=1`) coefficients are nonlinear functions of the free simplex
logits, so the full Jacobian of the reconstructed coefficient vector is applied to
the free-space QMLE covariance. `cov_type`: `:robust` (Bollerslev–Wooldridge
sandwich) or `:hessian` (inverse observed information).
"""
function StatsAPI.stderror(m::IGARCHModel{T}; cov_type::Symbol=:robust) where {T}
    p, q = m.p, m.q
    ncoef = 2 + q + p
    _volatility_delta_se(m, ncoef, cov_type,
        free -> _igarch_coef_from_free(free, p, q),
        () -> _igarch_reconstruct_free(m),
        obj_free -> _igarch_negloglik(obj_free, m.y, p, q),
        free -> _igarch_loglik_contribs(free, m.y, p, q))
end

# reconstruct the free-param vector from stored fields (inverse simplex)
function _igarch_reconstruct_free(m::IGARCHModel{T}) where {T}
    w = vcat(m.alpha, m.beta)
    theta = [log(w[i] / w[end]) for i in 1:(m.q + m.p - 1)]
    vcat(m.mu, log(m.omega), theta)
end

# =============================================================================
# Shared delta-method SE for the EV-15 constrained/transformed models
# =============================================================================
"""
    _volatility_delta_se(m, ncoef, cov_type, coef_map, free_reconstruct, negll, contribs)

Standard errors for a volatility model whose stored coefficients are a nonlinear
map `coef_map(free)` of the free optimization parameters. Uses the cached
`param_vcov` (free space) for `:robust`, recomputes the inverse Hessian for
`:hessian`, and propagates via the ForwardDiff Jacobian `J` of `coef_map`:
`Cov(coef) = J V J'`.
"""
function _volatility_delta_se(m, ncoef::Int, cov_type::Symbol,
                              coef_map, free_reconstruct, negll, contribs)
    T = eltype(m.y)
    cov_type in (:robust, :qmle, :sandwich, :bw, :hessian, :opg_hessian) ||
        throw(ArgumentError("cov_type must be :robust or :hessian, got :$cov_type"))
    free = free_reconstruct()
    V = if cov_type in (:robust, :qmle, :sandwich, :bw) && all(isfinite, m.param_vcov)
        m.param_vcov
    else
        try
            H = _numerical_hessian(negll, free)
            if cov_type in (:robust, :qmle, :sandwich, :bw)
                S = ForwardDiff.jacobian(contribs, free)
                _qmle_sandwich_cov(H, S)
            else
                robust_inv(H)
            end
        catch
            return fill(T(NaN), ncoef)
        end
    end
    J = try
        ForwardDiff.jacobian(coef_map, free)
    catch
        return fill(T(NaN), ncoef)
    end
    C = J * V * J'
    T[sqrt(max(C[i, i], zero(T))) for i in 1:ncoef]
end

# =============================================================================
# Component-GARCH (Engle–Lee 1999)
# =============================================================================

"""
    _cgarch_filter(mu_omega, rho, phi, alpha, beta, rsq, backcast) -> (h, q)

Component-GARCH(1,1) recursion returning total variance `h=σ²` and permanent `q`.
"""
function _cgarch_filter(omega, rho, phi, alpha, beta, rsq, backcast)
    n = length(rsq)
    E = promote_type(eltype(rsq), typeof(omega))
    h = Vector{E}(undef, n)
    qc = Vector{E}(undef, n)
    floorv = eps(float(real(E)))
    @inbounds for t in 1:n
        e2_lag = t > 1 ? rsq[t-1] : backcast
        h_lag  = t > 1 ? h[t-1] : backcast
        q_lag  = t > 1 ? qc[t-1] : omega
        qt = omega + rho * (q_lag - omega) + phi * (e2_lag - h_lag)
        ht = qt + alpha * (e2_lag - q_lag) + beta * (h_lag - q_lag)
        qc[t] = max(qt, floorv)
        h[t]  = max(ht, floorv)
    end
    h, qc
end

# free params: [mu, log(omega), logit(rho), log(phi), log(alpha), log(beta)]
function _cgarch_unpack(params)
    mu    = params[1]
    omega = exp(params[2])
    rho   = _igarch_logistic(params[3])
    phi   = exp(params[4])
    alpha = exp(params[5])
    beta  = exp(params[6])
    mu, omega, rho, phi, alpha, beta
end

function _cgarch_negloglik(params, y)
    n = length(y)
    mu, omega, rho, phi, alpha, beta = _cgarch_unpack(params)
    E = eltype(params)
    # Engle–Lee identification / non-negativity guards
    (alpha + beta >= rho) && return E(1e10)        # trend must be more persistent
    (alpha + beta >= one(E)) && return E(1e10)     # transitory stationarity
    (phi >= beta) && return E(1e10)                # non-negativity (0 ≤ φ < β)
    resid = y .- mu
    rsq = resid .^ 2
    backcast = sum(rsq) / n
    h, _ = _cgarch_filter(omega, rho, phi, alpha, beta, rsq, backcast)
    _volatility_negloglik(h, rsq, n)
end

function _cgarch_loglik_contribs(params, y)
    n = length(y)
    mu, omega, rho, phi, alpha, beta = _cgarch_unpack(params)
    resid = y .- mu
    rsq = resid .^ 2
    backcast = sum(rsq) / n
    h, _ = _cgarch_filter(omega, rho, phi, alpha, beta, rsq, backcast)
    _volatility_loglik_contribs(h, rsq)
end

_cgarch_coef_from_free(params) = collect(_cgarch_unpack(params))

"""
    estimate_cgarch(y; method=:mle) -> CGARCHModel

Estimate a Component GARCH(1,1) model (Engle & Lee 1999) splitting conditional
variance into a persistent long-run component `q` (reverting to `ω` with
persistence `ρ`) and a transitory component with persistence `α+β`. Identification
requires `ρ > α+β`, enforced during optimization.

# Example
```julia
m = estimate_cgarch(y)
comp = component_variances(m)   # (permanent, transitory, total)
```
"""
function estimate_cgarch(y::AbstractVector{T}; method::Symbol=:mle) where {T<:AbstractFloat}
    _validate_data(y, "y")
    _validate_volatility_inputs(y, 1, 1)
    y_vec = Vector{T}(y)
    n = length(y_vec)

    mu_init = mean(y_vec)
    var_init = var(y_vec .- mu_init; corrected=false)
    # ρ≈0.99, φ≈0.05, α≈0.05, β≈0.85  (ρ ≫ α+β, φ<β)
    params_init = _sanitize_init_params(T[mu_init, log(var_init),
        log(T(0.99) / (one(T) - T(0.99))), log(T(0.05)), log(T(0.05)), log(T(0.85))])

    obj = params -> _cgarch_negloglik(params, y_vec)
    result1 = Optim.optimize(obj, params_init, Optim.NelderMead(),
        Optim.Options(iterations=3000, show_trace=false))
    result = Optim.optimize(obj, Optim.minimizer(result1), Optim.LBFGS(),
        Optim.Options(iterations=1000, g_tol=T(1e-8), show_trace=false))

    params_opt = Optim.minimizer(result)
    mu, omega, rho, phi, alpha, beta = _cgarch_unpack(params_opt)

    resid = y_vec .- mu
    rsq = resid .^ 2
    backcast = sum(rsq) / n
    h, qperm = _cgarch_filter(omega, rho, phi, alpha, beta, rsq, backcast)
    h = Vector{T}(h); qperm = Vector{T}(qperm)
    transitory = h .- qperm
    z = resid ./ sqrt.(h)

    loglik = -Optim.minimum(result)
    k = 6
    aic_val, bic_val = _compute_aic_bic(loglik, k, n)

    param_vcov = try
        H = _numerical_hessian(obj, params_opt)
        S = ForwardDiff.jacobian(θ -> _cgarch_loglik_contribs(θ, y_vec), params_opt)
        Matrix{T}(_qmle_sandwich_cov(H, S))
    catch
        fill(T(NaN), k, k)
    end

    CGARCHModel{T}(y_vec, T(mu), T(omega), T(rho), T(phi), T(alpha), T(beta),
                   qperm, transitory, h, z, resid, fill(T(mu), n),
                   loglik, aic_val, bic_val, method, Optim.converged(result),
                   Optim.iterations(result), param_vcov)
end

estimate_cgarch(y::AbstractVector; kwargs...) = estimate_cgarch(Float64.(y); kwargs...)

function StatsAPI.stderror(m::CGARCHModel{T}; cov_type::Symbol=:robust) where {T}
    _volatility_delta_se(m, 6, cov_type,
        _cgarch_coef_from_free,
        () -> _cgarch_reconstruct_free(m),
        free -> _cgarch_negloglik(free, m.y),
        free -> _cgarch_loglik_contribs(free, m.y))
end

_cgarch_reconstruct_free(m::CGARCHModel{T}) where {T} =
    T[m.mu, log(m.omega), log(m.rho / (one(T) - m.rho)), log(m.phi), log(m.alpha), log(m.beta)]

# =============================================================================
# APARCH (Ding–Granger–Engle 1993)
# =============================================================================

"""
    _aparch_filter(omega, alpha, gamma, beta, delta, resid) -> (h, s)

APARCH recursion in `s = σ^δ`: `sₜ = ω + Σαᵢ(|εₜ₋ᵢ|−γᵢεₜ₋ᵢ)^δ + Σβⱼ sₜ₋ⱼ`, then
`h = σ² = s^{2/δ}`. With `δ=2, γ=0` this reduces bit-for-bit to `_garch_filter`.
"""
function _aparch_filter(omega, alpha, gamma, beta, delta, resid)
    n = length(resid)
    q = length(alpha)
    p = length(beta)
    E = promote_type(eltype(resid), typeof(omega), typeof(delta))
    s = Vector{E}(undef, n)
    bc = zero(E)
    @inbounds for t in 1:n
        bc += abs(resid[t])^delta
    end
    bc /= n
    floorv = eps(float(real(E)))
    @inbounds for t in 1:n
        st = omega
        for i in 1:q
            if t - i >= 1
                e = resid[t-i]
                st += alpha[i] * (abs(e) - gamma[i] * e)^delta
            else
                st += alpha[i] * bc
            end
        end
        for j in 1:p
            slag = t - j >= 1 ? s[t-j] : bc
            st += beta[j] * slag
        end
        s[t] = max(st, floorv)
    end
    h = s .^ (2 / delta)
    h, s
end

# Ding–Granger–Engle persistence moment E(|z|−γz)^δ for z~N(0,1), by Gauss–Hermite.
function _aparch_kappa(gamma::Real, delta::Real; n::Int=48)
    x, w = _gauss_hermite_nodes_weights(n)   # ∫ f(x) e^{-x²} dx
    acc = 0.0
    inv_sqrt_pi = 1 / sqrt(π)
    for k in 1:n
        z = sqrt(2.0) * x[k]
        base = abs(z) - gamma * z
        acc += w[k] * base^delta
    end
    inv_sqrt_pi * acc
end

# free params layout: [mu, log(omega), log(alpha)_q, (atanh(gamma)_q?), log(beta)_p, (log(delta)?)]
struct _APARCHLayout
    q::Int
    p::Int
    fix_delta::Union{Nothing,Float64}
    fix_gamma::Union{Nothing,Float64}
end

function _aparch_unpack(params, lay::_APARCHLayout)
    E = eltype(params)
    q, p = lay.q, lay.p
    idx = 2
    mu = params[1]
    omega = exp(params[idx]); idx += 1
    alpha = [exp(params[idx + i - 1]) for i in 1:q]; idx += q
    if lay.fix_gamma === nothing
        gamma = [tanh(params[idx + i - 1]) for i in 1:q]; idx += q
    else
        gamma = fill(E(lay.fix_gamma), q)
    end
    beta = [exp(params[idx + j - 1]) for j in 1:p]; idx += p
    if lay.fix_delta === nothing
        delta = exp(params[idx])
    else
        delta = E(lay.fix_delta)
    end
    mu, omega, alpha, gamma, beta, delta
end

function _aparch_negloglik(params, y, lay::_APARCHLayout)
    n = length(y)
    mu, omega, alpha, gamma, beta, delta = _aparch_unpack(params, lay)
    E = eltype(params)
    (sum(alpha) + sum(beta) >= one(E)) && return E(1e10)
    resid = y .- mu
    h, _ = _aparch_filter(omega, alpha, gamma, beta, delta, resid)
    rsq = resid .^ 2
    _volatility_negloglik(h, rsq, n)
end

function _aparch_loglik_contribs(params, y, lay::_APARCHLayout)
    mu, omega, alpha, gamma, beta, delta = _aparch_unpack(params, lay)
    resid = y .- mu
    h, _ = _aparch_filter(omega, alpha, gamma, beta, delta, resid)
    rsq = resid .^ 2
    _volatility_loglik_contribs(h, rsq)
end

function _aparch_coef_from_free(params, lay::_APARCHLayout)
    mu, omega, alpha, gamma, beta, delta = _aparch_unpack(params, lay)
    vcat(mu, omega, alpha, gamma, beta, delta)
end

"""
    estimate_aparch(y, p=1, q=1; fix_delta=nothing, fix_gamma=nothing, method=:mle) -> APARCHModel

Estimate an Asymmetric Power ARCH(p,q) model (Ding, Granger & Engle 1993):

    σₜ^δ = ω + Σᵢ αᵢ(|εₜ₋ᵢ| − γᵢεₜ₋ᵢ)^δ + Σⱼ βⱼ σₜ₋ⱼ^δ

with `δ > 0` and leverage `γᵢ ∈ (−1,1)`. Pin parameters to recover nested models:
`fix_delta=2.0, fix_gamma=0.0` ≡ GARCH; `fix_delta=2.0` (γ free) ≡ GJR-GARCH;
`fix_delta=1.0` (γ free) ≡ Zakoïan TARCH.

# Keyword arguments
- `fix_delta`: pin the power `δ` (default free)
- `fix_gamma`: pin the leverage `γ` for all lags (default free)

# Example
```julia
m = estimate_aparch(y, 1, 1)                 # full APARCH
g = estimate_aparch(y, 1, 1; fix_delta=2.0, fix_gamma=0.0)  # ≡ GARCH(1,1)
```
"""
function estimate_aparch(y::AbstractVector{T}, p::Int=1, q::Int=1;
                         fix_delta::Union{Nothing,Real}=nothing,
                         fix_gamma::Union{Nothing,Real}=nothing,
                         method::Symbol=:mle) where {T<:AbstractFloat}
    _validate_data(y, "y")
    _validate_volatility_inputs(y, p, q)
    fix_delta !== nothing && fix_delta <= 0 && throw(ArgumentError("fix_delta must be > 0"))
    fix_gamma !== nothing && abs(fix_gamma) >= 1 && throw(ArgumentError("fix_gamma must be in (-1, 1)"))
    y_vec = Vector{T}(y)
    n = length(y_vec)

    lay = _APARCHLayout(q, p,
        fix_delta === nothing ? nothing : Float64(fix_delta),
        fix_gamma === nothing ? nothing : Float64(fix_gamma))

    mu_init = mean(y_vec)
    var_init = var(y_vec .- mu_init; corrected=false)
    d0 = fix_delta === nothing ? T(1.5) : T(fix_delta)
    omega_init = (var_init^(d0 / 2)) * T(0.05)
    init = vcat(mu_init, log(omega_init), log.(fill(T(0.05), q)))
    fix_gamma === nothing && (init = vcat(init, fill(T(0.0), q)))   # atanh(0)=0
    init = vcat(init, log.(fill(T(0.85) / p, p)))
    fix_delta === nothing && (init = vcat(init, log(d0)))
    params_init = _sanitize_init_params(Vector{T}(init))

    obj = params -> _aparch_negloglik(params, y_vec, lay)
    result1 = Optim.optimize(obj, params_init, Optim.NelderMead(),
        Optim.Options(iterations=4000, show_trace=false))
    result = Optim.optimize(obj, Optim.minimizer(result1), Optim.LBFGS(),
        Optim.Options(iterations=1500, g_tol=T(1e-8), show_trace=false))

    params_opt = Optim.minimizer(result)
    mu, omega, alpha, gamma, beta, delta = _aparch_unpack(params_opt, lay)
    alpha = Vector{T}(alpha); gamma = Vector{T}(gamma); beta = Vector{T}(beta)

    resid = y_vec .- mu
    h, s = _aparch_filter(omega, alpha, gamma, beta, delta, resid)
    h = Vector{T}(h); s = Vector{T}(s)
    z = resid ./ sqrt.(h)

    loglik = -Optim.minimum(result)
    n_params = length(params_opt)
    aic_val, bic_val = _compute_aic_bic(loglik, n_params, n)

    param_vcov = try
        H = _numerical_hessian(obj, params_opt)
        S = ForwardDiff.jacobian(θ -> _aparch_loglik_contribs(θ, y_vec, lay), params_opt)
        Matrix{T}(_qmle_sandwich_cov(H, S))
    catch
        fill(T(NaN), n_params, n_params)
    end

    APARCHModel{T}(y_vec, p, q, T(mu), T(omega), alpha, gamma, beta, T(delta),
                   s, h, z, resid, fill(T(mu), n), loglik, aic_val, bic_val,
                   fix_delta !== nothing, fix_gamma !== nothing, n_params,
                   method, Optim.converged(result), Optim.iterations(result), param_vcov)
end

estimate_aparch(y::AbstractVector, p::Int=1, q::Int=1; kwargs...) = estimate_aparch(Float64.(y), p, q; kwargs...)

function StatsAPI.stderror(m::APARCHModel{T}; cov_type::Symbol=:robust) where {T}
    lay = _APARCHLayout(m.q, m.p,
        m.fixed_delta ? Float64(m.delta) : nothing,
        m.fixed_gamma ? Float64(m.gamma[1]) : nothing)
    ncoef = 3 + 2m.q + m.p
    _volatility_delta_se(m, ncoef, cov_type,
        free -> _aparch_coef_from_free(free, lay),
        () -> _aparch_reconstruct_free(m, lay),
        free -> _aparch_negloglik(free, m.y, lay),
        free -> _aparch_loglik_contribs(free, m.y, lay))
end

function _aparch_reconstruct_free(m::APARCHModel{T}, lay::_APARCHLayout) where {T}
    v = T[m.mu, log(m.omega)]
    append!(v, log.(m.alpha))
    lay.fix_gamma === nothing && append!(v, atanh.(m.gamma))
    append!(v, log.(m.beta))
    lay.fix_delta === nothing && push!(v, log(m.delta))
    v
end
