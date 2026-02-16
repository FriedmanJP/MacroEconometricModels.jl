# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <wookyung9207@gmail.com>
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
    _egarch_filter(omega, alpha, gamma, beta, eps, backcast_logh)

Compute EGARCH conditional variances via log(h_t) recursion:
log(h_t) = ω + Σα_i(|z_{t-i}| - E|z|) + Σγ_i z_{t-i} + Σβ_j log(h_{t-j})
where z_t = ε_t / σ_t and E|z| = √(2/π) for Gaussian.
"""
function _egarch_filter(omega::T, alpha::Vector{T}, gamma::Vector{T},
                         beta::Vector{T}, eps::Vector{T}, backcast_logh::T) where {T}
    n = length(eps)
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
        z[t] = eps[t] / sqrt(h[t])
    end
    h, z, log_h
end

# =============================================================================
# GJR-GARCH Filter
# =============================================================================

"""
    _gjr_garch_filter(omega, alpha, gamma, beta, eps)

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

    eps = y .- mu
    eps_sq = eps .^ 2
    h = _garch_filter(omega, alpha, beta, eps_sq)
    _volatility_negloglik(h, eps_sq, n)
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

    eps = y .- mu
    backcast_logh = log(var(eps; corrected=false))
    h, _, _ = _egarch_filter(omega, alpha, gamma, beta, eps, backcast_logh)
    eps_sq = eps .^ 2
    _volatility_negloglik(h, eps_sq, n)
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

    eps = y_vec .- mu
    eps_sq = eps .^ 2
    h = _garch_filter(omega, alpha, beta, eps_sq)
    z = eps ./ sqrt.(h)

    loglik = -Optim.minimum(result)
    k = 2 + q + p
    aic_val, bic_val = _compute_aic_bic(loglik, k, n)

    GARCHModel(y_vec, p, q, mu, omega, alpha, beta, h, z, eps, fill(mu, n),
               loglik, aic_val, bic_val, method, Optim.converged(result), Optim.iterations(result))
end

estimate_garch(y::AbstractVector, p::Int=1, q::Int=1; kwargs...) = estimate_garch(Float64.(y), p, q; kwargs...)

"""
    StatsAPI.stderror(m::GARCHModel{T}) -> Vector{T}

MLE standard errors for GARCH model via numerical Hessian + delta method.
Returns SE vector matching `coef(m)` = [μ, ω, α₁..αq, β₁..βp].
"""
function StatsAPI.stderror(m::GARCHModel{T}) where {T}
    p, q = m.p, m.q
    params_opt = vcat(m.mu, log(m.omega), log.(m.alpha), log.(m.beta))

    obj = params -> _garch_negloglik(params, m.y, p, q)
    H = _numerical_hessian(obj, params_opt)

    C_opt = try; robust_inv(H); catch; return fill(T(NaN), 2 + q + p); end

    se = Vector{T}(undef, 2 + q + p)
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

    eps = y_vec .- mu
    backcast_logh = log(var(eps; corrected=false))
    h, z, _ = _egarch_filter(omega, alpha, gamma, beta, eps, backcast_logh)

    loglik = -Optim.minimum(result)
    k = 2 + 2q + p
    aic_val, bic_val = _compute_aic_bic(loglik, k, n)

    EGARCHModel(y_vec, p, q, mu, omega, alpha, gamma, beta, h, z, eps, fill(mu, n),
                loglik, aic_val, bic_val, method, Optim.converged(result), Optim.iterations(result))
end

estimate_egarch(y::AbstractVector, p::Int=1, q::Int=1; kwargs...) = estimate_egarch(Float64.(y), p, q; kwargs...)

"""
    StatsAPI.stderror(m::EGARCHModel{T}) -> Vector{T}

MLE standard errors for EGARCH model via numerical Hessian.
EGARCH params are unconstrained — no delta method needed.
Returns SE vector matching `coef(m)` = [μ, ω, α₁..αq, γ₁..γq, β₁..βp].
"""
function StatsAPI.stderror(m::EGARCHModel{T}) where {T}
    p, q = m.p, m.q
    # EGARCH parameters are all unconstrained in optimization space
    params_opt = vcat(m.mu, m.omega, m.alpha, m.gamma, m.beta)

    obj = params -> _egarch_negloglik(params, m.y, p, q)
    H = _numerical_hessian(obj, params_opt)

    n_params = 2 + 2q + p
    C_opt = try; robust_inv(H); catch; return fill(T(NaN), n_params); end

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

    GJRGARCHModel(y_vec, p, q, mu, omega, alpha, gamma, beta, h, z, resid, fill(mu, n),
                  loglik, aic_val, bic_val, method, Optim.converged(result), Optim.iterations(result))
end

estimate_gjr_garch(y::AbstractVector, p::Int=1, q::Int=1; kwargs...) = estimate_gjr_garch(Float64.(y), p, q; kwargs...)

"""
    StatsAPI.stderror(m::GJRGARCHModel{T}) -> Vector{T}

MLE standard errors for GJR-GARCH model via numerical Hessian + delta method.
Returns SE vector matching `coef(m)` = [μ, ω, α₁..αq, γ₁..γq, β₁..βp].
"""
function StatsAPI.stderror(m::GJRGARCHModel{T}) where {T}
    p, q = m.p, m.q
    params_opt = vcat(m.mu, log(m.omega), log.(m.alpha), log.(m.gamma), log.(m.beta))

    obj = params -> _gjr_negloglik(params, m.y, p, q)
    H = _numerical_hessian(obj, params_opt)

    n_params = 2 + 2q + p
    C_opt = try; robust_inv(H); catch; return fill(T(NaN), n_params); end

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
