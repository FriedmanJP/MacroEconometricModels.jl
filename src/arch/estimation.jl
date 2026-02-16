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
ARCH(q) estimation via MLE (Engle 1982).

Shared helpers for volatility model estimation are defined here and
reused by GARCH, EGARCH, GJR-GARCH, and SV modules.
"""

import Optim

# =============================================================================
# Shared Helpers
# =============================================================================

"""Validate inputs for volatility model estimation."""
function _validate_volatility_inputs(y::AbstractVector, p::Int, q::Int)
    n = length(y)
    n < 20 && throw(ArgumentError("Need at least 20 observations for volatility models, got $n"))
    q < 1 && throw(ArgumentError("ARCH order q must be ≥ 1, got $q"))
    p < 0 && throw(ArgumentError("GARCH order p must be ≥ 0, got $p"))
end

"""Sanitize initial parameters for NelderMead: clamp to finite range, replace NaN."""
function _sanitize_init_params(params::Vector{T}) where {T}
    out = copy(params)
    @inbounds for i in eachindex(out)
        if !isfinite(out[i])
            out[i] = zero(T)
        else
            out[i] = clamp(out[i], T(-30), T(30))
        end
    end
    out
end

"""
    _volatility_negloglik(h, eps_sq, n)

Gaussian log-likelihood for conditional variance models:
ℓ = -n/2 log(2π) - 1/2 Σ[log(hₜ) + ε²ₜ/hₜ]

Returns negative log-likelihood (for minimization).
"""
function _volatility_negloglik(h::Vector{T}, eps_sq::Vector{T}, n::Int) where {T}
    ll = -T(n) / 2 * log(T(2π))
    @inbounds for t in 1:n
        ll -= (log(h[t]) + eps_sq[t] / h[t]) / 2
    end
    -ll
end

# =============================================================================
# ARCH Filter
# =============================================================================

"""
    _arch_filter(omega, alpha, eps_sq)

Compute ARCH conditional variances: h_t = ω + Σα_i ε²_{t-i}
"""
function _arch_filter(omega::T, alpha::Vector{T}, eps_sq::Vector{T}) where {T}
    n = length(eps_sq)
    q = length(alpha)
    h = Vector{T}(undef, n)
    backcast = mean(eps_sq)

    @inbounds for t in 1:n
        h[t] = omega
        for i in 1:q
            eps2_lag = t - i >= 1 ? eps_sq[t-i] : backcast
            h[t] += alpha[i] * eps2_lag
        end
        h[t] = max(h[t], eps(T))
    end
    h
end

# =============================================================================
# ARCH Negative Log-Likelihood
# =============================================================================

function _arch_negloglik(params::Vector{T}, y::Vector{T}, q::Int) where {T}
    n = length(y)

    # Unpack: mu, log(omega), log(alpha_1), ..., log(alpha_q)
    mu = params[1]
    omega = exp(params[2])
    alpha = exp.(params[3:2+q])

    # Stationarity check
    sum(alpha) >= one(T) && return T(1e10)

    eps = y .- mu
    eps_sq = eps .^ 2
    h = _arch_filter(omega, alpha, eps_sq)

    _volatility_negloglik(h, eps_sq, n)
end

# =============================================================================
# ARCH Estimation
# =============================================================================

"""
    estimate_arch(y, q; method=:mle) -> ARCHModel

Estimate ARCH(q) model via Maximum Likelihood.

σ²ₜ = ω + α₁ε²ₜ₋₁ + ... + αqε²ₜ₋q

Uses two-stage optimization: NelderMead initialization → LBFGS refinement.

# Arguments
- `y`: Time series vector
- `q`: ARCH order (≥ 1)
- `method`: Estimation method (currently only `:mle`)

# Returns
`ARCHModel` with estimated parameters and conditional variances.

# Example
```julia
y = randn(500)
model = estimate_arch(y, 1)
println("ω = ", model.omega, ", α₁ = ", model.alpha[1])
```
"""
function estimate_arch(y::AbstractVector{T}, q::Int; method::Symbol=:mle) where {T<:AbstractFloat}
    _validate_volatility_inputs(y, 0, q)
    y_vec = Vector{T}(y)
    n = length(y_vec)

    # Initial parameters (log-transformed for positivity)
    mu_init = mean(y_vec)
    eps_init = y_vec .- mu_init
    var_init = var(eps_init; corrected=false)
    omega_init = var_init * T(0.05)
    alpha_init = fill(T(0.9) / q, q)

    params_init = _sanitize_init_params(vcat(mu_init, log(omega_init), log.(alpha_init)))

    # Stage 1: NelderMead
    obj = p -> _arch_negloglik(p, y_vec, q)
    result1 = Optim.optimize(obj, params_init, Optim.NelderMead(),
        Optim.Options(iterations=1000, show_trace=false))

    # Stage 2: LBFGS refinement
    result = Optim.optimize(obj, Optim.minimizer(result1), Optim.LBFGS(),
        Optim.Options(iterations=500, g_tol=T(1e-8), show_trace=false))

    params_opt = Optim.minimizer(result)
    mu = params_opt[1]
    omega = exp(params_opt[2])
    alpha = exp.(params_opt[3:2+q])

    eps = y_vec .- mu
    eps_sq = eps .^ 2
    h = _arch_filter(omega, alpha, eps_sq)
    z = eps ./ sqrt.(h)

    negll = Optim.minimum(result)
    loglik = -negll
    k = 2 + q  # mu + omega + q alphas
    aic_val, bic_val = _compute_aic_bic(loglik, k, n)

    converged = Optim.converged(result)
    iterations = Optim.iterations(result)

    ARCHModel(y_vec, q, mu, omega, alpha, h, z, eps, fill(mu, n), loglik,
              aic_val, bic_val, method, converged, iterations)
end

estimate_arch(y::AbstractVector, q::Int; kwargs...) = estimate_arch(Float64.(y), q; kwargs...)

# =============================================================================
# Standard Errors for ARCH
# =============================================================================

"""
    StatsAPI.stderror(m::ARCHModel{T}) -> Vector{T}

MLE standard errors for ARCH model via numerical Hessian.
Returns SE vector matching `coef(m)` = [μ, ω, α₁, ..., αq].
"""
function StatsAPI.stderror(m::ARCHModel{T}) where {T}
    q = m.q
    # Reconstruct optimization-space parameters: [mu, log(omega), log(alpha)...]
    params_opt = vcat(m.mu, log(m.omega), log.(m.alpha))

    obj = p -> _arch_negloglik(p, m.y, q)
    H = _numerical_hessian(obj, params_opt)

    # Invert Hessian for covariance in optimization space
    C_opt = try
        robust_inv(H)
    catch
        return fill(T(NaN), 2 + q)
    end

    # Delta method: SE(exp(x)) = exp(x) * SE(x)
    se = Vector{T}(undef, 2 + q)
    se[1] = sqrt(max(C_opt[1, 1], zero(T)))             # mu (untransformed)
    se[2] = m.omega * sqrt(max(C_opt[2, 2], zero(T)))   # omega = exp(log_omega)
    for i in 1:q
        se[2+i] = m.alpha[i] * sqrt(max(C_opt[2+i, 2+i], zero(T)))  # alpha_i = exp(log_alpha_i)
    end
    se
end
