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
Stochastic Volatility model estimation via Kim-Shephard-Chib (1998) Gibbs sampler
with Omori et al. (2007) 10-component mixture approximation.

Three variants:
1. Basic SV (Taylor 1986)
2. SV with leverage (correlated innovations)
3. SV with Student-t errors
"""

# =============================================================================
# Omori et al. (2007) 10-component mixture for log(χ²₁)
# =============================================================================

# Mixture weights, means, variances from Omori et al. (2007), Table 1
const _KSC_WEIGHTS = [
    0.00609, 0.04775, 0.13057, 0.20674, 0.22715,
    0.18842, 0.12047, 0.05591, 0.01575, 0.00115
]

const _KSC_MEANS = [
    1.92677, 1.34744, 0.73504, 0.02266, -0.85173,
    -1.97278, -3.46788, -5.55246, -8.68384, -14.65000
]

const _KSC_VARIANCES = [
    0.11265, 0.17788, 0.26768, 0.40611, 0.62699,
    0.98583, 1.57469, 2.54498, 4.16591, 7.33342
]

# =============================================================================
# Core KSC Gibbs Steps
# =============================================================================

"""Draw mixture indicators s_t | h_t, y* from discrete posterior."""
function _ksc_draw_indicators(y_star::Vector{T}, h::Vector{T}) where {T}
    n = length(y_star)
    s = Vector{Int}(undef, n)
    log_probs = Vector{T}(undef, 10)

    for t in 1:n
        resid = y_star[t] - h[t]
        for k in 1:10
            log_probs[k] = log(T(_KSC_WEIGHTS[k])) -
                           T(0.5) * log(T(_KSC_VARIANCES[k])) -
                           T(0.5) * (resid - T(_KSC_MEANS[k]))^2 / T(_KSC_VARIANCES[k])
        end
        # Normalize in log space
        max_lp = maximum(log_probs)
        probs = exp.(log_probs .- max_lp)
        probs ./= sum(probs)

        # Sample from categorical
        u = rand(T)
        cumprob = zero(T)
        s[t] = 10
        for k in 1:10
            cumprob += probs[k]
            if u <= cumprob
                s[t] = k
                break
            end
        end
    end
    s
end

"""
Forward-Filtering Backward-Sampling (FFBS) for latent log-volatilities.

Given the linear Gaussian state-space (conditional on mixture indicators):
    y*_t = h_t + ξ_t,    ξ_t ~ N(m_{s_t}, v²_{s_t})
    h_t = μ + φ(h_{t-1} - μ) + σ_η η_t

Returns drawn h_{1:T}.
"""
function _ksc_ffbs(y_star::Vector{T}, s::Vector{Int},
                   mu::T, phi::T, sigma_eta::T) where {T}
    n = length(y_star)
    sigma2_eta = sigma_eta^2

    # Forward filtering: compute filtered means and variances
    h_filt = Vector{T}(undef, n)
    P_filt = Vector{T}(undef, n)

    # Initialize from stationary distribution
    h_pred = mu
    P_pred = sigma2_eta / max(one(T) - phi^2, T(1e-8))

    for t in 1:n
        m_st = T(_KSC_MEANS[s[t]])
        v_st = T(_KSC_VARIANCES[s[t]])

        # Update step
        innov = y_star[t] - m_st - h_pred
        F_t = P_pred + v_st
        K_t = P_pred / F_t  # Kalman gain

        h_filt[t] = h_pred + K_t * innov
        P_filt[t] = P_pred * (one(T) - K_t)

        # Predict step (for next t)
        if t < n
            h_pred = mu + phi * (h_filt[t] - mu)
            P_pred = phi^2 * P_filt[t] + sigma2_eta
        end
    end

    # Backward sampling
    h = Vector{T}(undef, n)
    h[n] = h_filt[n] + sqrt(max(P_filt[n], T(1e-12))) * randn(T)

    for t in (n-1):-1:1
        # Smoother gain
        h_pred_tp1 = mu + phi * (h_filt[t] - mu)
        P_pred_tp1 = phi^2 * P_filt[t] + sigma2_eta
        J_t = phi * P_filt[t] / P_pred_tp1

        h_smooth = h_filt[t] + J_t * (h[t+1] - h_pred_tp1)
        P_smooth = P_filt[t] - J_t^2 * P_pred_tp1
        P_smooth = max(P_smooth, T(1e-12))

        h[t] = h_smooth + sqrt(P_smooth) * randn(T)
    end

    h
end

"""
Draw parameters (μ, φ, σ_η) | h_{1:T} via conjugate updates and MH.

- σ_η²: Inverse-Gamma conjugate update
- (μ, φ): Joint Metropolis-Hastings with stationarity constraint |φ| < 1
"""
function _ksc_draw_params(h::Vector{T}, mu_curr::T, phi_curr::T, sigma_eta_curr::T) where {T}
    n = length(h)

    # --- Draw σ_η² from IG posterior ---
    # Prior: σ_η² ~ IG(a0, b0) with a0=2.5, b0=0.025 (weakly informative)
    a0 = T(2.5)
    b0 = T(0.025)
    ss = zero(T)
    for t in 2:n
        resid = h[t] - mu_curr - phi_curr * (h[t-1] - mu_curr)
        ss += resid^2
    end
    a_post = a0 + T(n - 1) / T(2)
    b_post = b0 + ss / T(2)
    sigma2_eta_new = b_post / rand(Gamma(a_post))  # IG draw via reciprocal of Gamma
    sigma_eta_new = sqrt(sigma2_eta_new)

    # --- Draw (μ, φ) via MH ---
    # Proposal: random walk on (μ, logit_phi)
    logit_phi_curr = log((one(T) + phi_curr) / (one(T) - phi_curr + T(1e-10)))
    mu_prop = mu_curr + T(0.1) * randn(T)
    logit_phi_prop = logit_phi_curr + T(0.1) * randn(T)
    phi_prop = (T(2) / (one(T) + exp(-logit_phi_prop))) - one(T)

    # Check stationarity
    if abs(phi_prop) < one(T)
        # Log-likelihood of h | μ, φ, σ_η
        function _loglik_h(mu_val::T, phi_val::T, se::T) where {T}
            se2 = se^2
            ll = zero(T)
            # Stationary initial distribution
            P0 = se2 / max(one(T) - phi_val^2, T(1e-8))
            ll += -T(0.5) * log(P0) - T(0.5) * (h[1] - mu_val)^2 / P0
            for t in 2:n
                resid_t = h[t] - mu_val - phi_val * (h[t-1] - mu_val)
                ll += -T(0.5) * log(se2) - T(0.5) * resid_t^2 / se2
            end
            ll
        end

        ll_curr = _loglik_h(mu_curr, phi_curr, sigma_eta_new)
        ll_prop = _loglik_h(mu_prop, phi_prop, sigma_eta_new)

        # Prior: μ ~ N(0, 10²), φ via Beta(20, 1.5) on (φ+1)/2
        log_prior_curr = -T(0.5) * mu_curr^2 / T(100) +
                         T(19) * log(max((one(T) + phi_curr) / T(2), T(1e-10))) +
                         T(0.5) * log(max((one(T) - phi_curr) / T(2), T(1e-10)))
        log_prior_prop = -T(0.5) * mu_prop^2 / T(100) +
                         T(19) * log(max((one(T) + phi_prop) / T(2), T(1e-10))) +
                         T(0.5) * log(max((one(T) - phi_prop) / T(2), T(1e-10)))

        # Jacobian for logit transform
        log_jac_curr = log(max((one(T) - phi_curr^2) / T(2), T(1e-10)))
        log_jac_prop = log(max((one(T) - phi_prop^2) / T(2), T(1e-10)))

        log_alpha = (ll_prop + log_prior_prop + log_jac_prop) -
                    (ll_curr + log_prior_curr + log_jac_curr)

        if log(rand(T)) < log_alpha
            return (mu_prop, phi_prop, sigma_eta_new)
        end
    end

    (mu_curr, phi_curr, sigma_eta_new)
end

# =============================================================================
# Student-t variant: draw scale mixture variables
# =============================================================================

"""Draw scale mixture variables λ_t | y_t, h_t, ν for Student-t SV."""
function _ksc_draw_lambda(y::Vector{T}, h::Vector{T}, nu::T) where {T}
    n = length(y)
    lambda = Vector{T}(undef, n)
    for t in 1:n
        a = (nu + one(T)) / T(2)
        b = (nu + y[t]^2 * exp(-h[t])) / T(2)
        lambda[t] = rand(Gamma(a)) / b  # Gamma → 1/IG
    end
    lambda
end

"""Draw degrees of freedom ν | λ_{1:T} via MH with log-normal proposal."""
function _ksc_draw_nu(lambda::Vector{T}, nu_curr::T) where {T}
    n = length(lambda)

    # Proposal
    nu_prop = exp(log(nu_curr) + T(0.1) * randn(T))
    nu_prop < T(2.01) && return nu_curr

    # Log-likelihood of λ | ν
    function _loglik_nu(nu::T)
        a = nu / T(2)
        ll = n * (a * log(a) - loggamma(a)) + (a - one(T)) * sum(log, lambda) - a * sum(lambda)
        ll
    end

    ll_curr = _loglik_nu(nu_curr)
    ll_prop = _loglik_nu(nu_prop)

    # Prior: ν ~ Exponential(0.1) truncated to (2, ∞)
    log_prior_curr = -T(0.1) * nu_curr
    log_prior_prop = -T(0.1) * nu_prop

    # Log-normal proposal Jacobian
    log_jac = log(nu_prop) - log(nu_curr)

    log_alpha = (ll_prop + log_prior_prop + log_jac) - (ll_curr + log_prior_curr)
    if log(rand(T)) < log_alpha
        return nu_prop
    end
    nu_curr
end

# =============================================================================
# Leverage variant helpers
# =============================================================================

"""
FFBS for leverage SV: h_t depends on z_{t-1} = y_{t-1}/σ_{t-1}.

    y*_t = h_t + ξ_t
    h_t = μ + φ(h_{t-1} - μ) + ρ σ_η z_{t-1} + σ_η √(1-ρ²) η_t
"""
function _ksc_ffbs_leverage(y_star::Vector{T}, y::Vector{T}, s::Vector{Int},
                            mu::T, phi::T, sigma_eta::T, rho::T) where {T}
    n = length(y_star)
    sigma2_eta = sigma_eta^2
    sigma_eta_cond = sigma_eta * sqrt(max(one(T) - rho^2, T(1e-8)))
    sigma2_cond = sigma_eta_cond^2

    h_filt = Vector{T}(undef, n)
    P_filt = Vector{T}(undef, n)

    # Initialize
    h_pred = mu
    P_pred = sigma2_eta / max(one(T) - phi^2, T(1e-8))

    for t in 1:n
        m_st = T(_KSC_MEANS[s[t]])
        v_st = T(_KSC_VARIANCES[s[t]])

        innov = y_star[t] - m_st - h_pred
        F_t = P_pred + v_st
        K_t = P_pred / F_t

        h_filt[t] = h_pred + K_t * innov
        P_filt[t] = P_pred * (one(T) - K_t)

        if t < n
            z_t = y[t] * exp(-h_filt[t] / T(2))
            h_pred = mu + phi * (h_filt[t] - mu) + rho * sigma_eta * z_t
            P_pred = phi^2 * P_filt[t] + sigma2_cond
        end
    end

    # Backward sampling
    h = Vector{T}(undef, n)
    h[n] = h_filt[n] + sqrt(max(P_filt[n], T(1e-12))) * randn(T)

    for t in (n-1):-1:1
        z_t = y[t] * exp(-h_filt[t] / T(2))
        h_pred_tp1 = mu + phi * (h_filt[t] - mu) + rho * sigma_eta * z_t
        P_pred_tp1 = phi^2 * P_filt[t] + sigma2_cond
        J_t = phi * P_filt[t] / P_pred_tp1

        h_smooth = h_filt[t] + J_t * (h[t+1] - h_pred_tp1)
        P_smooth = P_filt[t] - J_t^2 * P_pred_tp1
        P_smooth = max(P_smooth, T(1e-12))

        h[t] = h_smooth + sqrt(P_smooth) * randn(T)
    end

    h
end

"""Draw ρ | h, y, μ, φ, σ_η via MH."""
function _ksc_draw_rho(h::Vector{T}, y::Vector{T},
                       mu::T, phi::T, sigma_eta::T, rho_curr::T) where {T}
    n = length(h)

    # Proposal: truncated random walk on atanh(ρ)
    atanh_curr = atanh(clamp(rho_curr, T(-0.999), T(0.999)))
    atanh_prop = atanh_curr + T(0.1) * randn(T)
    rho_prop = tanh(atanh_prop)

    function _loglik_rho(rho_val::T)
        sigma_cond = sigma_eta * sqrt(max(one(T) - rho_val^2, T(1e-8)))
        ll = zero(T)
        for t in 2:n
            z_prev = y[t-1] * exp(-h[t-1] / T(2))
            h_mean = mu + phi * (h[t-1] - mu) + rho_val * sigma_eta * z_prev
            resid_t = h[t] - h_mean
            ll += -T(0.5) * log(sigma_cond^2) - T(0.5) * resid_t^2 / sigma_cond^2
        end
        ll
    end

    ll_curr = _loglik_rho(rho_curr)
    ll_prop = _loglik_rho(rho_prop)

    # Prior: Uniform(-1, 1) — flat, so no prior ratio
    # Jacobian: d(atanh)/dρ = 1/(1-ρ²), so log|Jac| adjustment
    log_jac_curr = -log(max(one(T) - rho_curr^2, T(1e-10)))
    log_jac_prop = -log(max(one(T) - rho_prop^2, T(1e-10)))

    log_alpha = (ll_prop + log_jac_prop) - (ll_curr + log_jac_curr)
    if log(rand(T)) < log_alpha
        return rho_prop
    end
    rho_curr
end

# =============================================================================
# Main Estimation Function
# =============================================================================

"""
    estimate_sv(y; n_samples=2000, burnin=1000,
                dist=:normal, leverage=false,
                quantile_levels=[0.025, 0.5, 0.975]) -> SVModel

Estimate a Stochastic Volatility model via Kim-Shephard-Chib (1998) Gibbs sampler
with Omori et al. (2007) 10-component mixture approximation.

# Model
    yₜ = exp(hₜ/2) εₜ,       εₜ ~ N(0,1)
    hₜ = μ + φ(hₜ₋₁ - μ) + σ_η ηₜ,  ηₜ ~ N(0,1)

# Arguments
- `y`: Time series vector
- `n_samples`: Number of posterior samples to keep (default 2000)
- `burnin`: Number of burn-in iterations to discard (default 1000)
- `dist`: Error distribution (`:normal` or `:studentt`)
- `leverage`: Whether to include leverage effect (correlated innovations)
- `quantile_levels`: Quantile levels for volatility posterior

# Example
```julia
y = randn(200) .* exp.(cumsum(0.1 .* randn(200)) ./ 2)
model = estimate_sv(y; n_samples=1000)
println("φ = ", mean(model.phi_post))
```
"""
function estimate_sv(y::AbstractVector{T};
                     n_samples::Int=2000, burnin::Int=1000,
                     dist::Symbol=:normal, leverage::Bool=false,
                     quantile_levels::Vector{<:Real}=[0.025, 0.5, 0.975]) where {T<:AbstractFloat}
    n = length(y)
    n < 20 && throw(ArgumentError("Need at least 20 observations for SV model, got $n"))
    y_vec = Vector{T}(y)
    ql = T.(quantile_levels)

    # Offset constant for log transformation (avoid log(0))
    c = T(1e-5)

    # Transform: y* = log(y² + c)
    y_star = log.(y_vec.^2 .+ c)

    # Initialize parameters
    mu = T(mean(y_star))
    phi = T(0.95)
    sigma_eta = T(0.2)
    h = copy(y_star)  # Initialize h at y*
    rho = zero(T)     # leverage parameter
    nu = T(10.0)      # degrees of freedom for Student-t
    lambda = ones(T, n)  # scale mixture variables for Student-t

    # Storage
    total_iters = burnin + n_samples
    mu_draws = Vector{T}(undef, n_samples)
    phi_draws = Vector{T}(undef, n_samples)
    sigma_eta_draws = Vector{T}(undef, n_samples)
    h_draws = Matrix{T}(undef, n_samples, n)

    draw_idx = 0

    for iter in 1:total_iters
        # For Student-t: modify y_star to account for scale mixture
        if dist == :studentt
            y_star_eff = log.(y_vec.^2 ./ lambda .+ c)
        else
            y_star_eff = y_star
        end

        # Step 1: Draw mixture indicators s_t | h_t, y*
        s = _ksc_draw_indicators(y_star_eff, h)

        # Step 2: Draw h_{1:T} via FFBS
        if leverage
            h = _ksc_ffbs_leverage(y_star_eff, y_vec, s, mu, phi, sigma_eta, rho)
        else
            h = _ksc_ffbs(y_star_eff, s, mu, phi, sigma_eta)
        end

        # Step 3: Draw (μ, φ, σ_η)
        mu, phi, sigma_eta = _ksc_draw_params(h, mu, phi, sigma_eta)

        # Step 4 (leverage): Draw ρ
        if leverage
            rho = _ksc_draw_rho(h, y_vec, mu, phi, sigma_eta, rho)
        end

        # Step 5 (Student-t): Draw λ_t and ν
        if dist == :studentt
            lambda = _ksc_draw_lambda(y_vec, h, nu)
            nu = _ksc_draw_nu(lambda, nu)
        end

        # Store post-burnin
        if iter > burnin
            draw_idx += 1
            mu_draws[draw_idx] = mu
            phi_draws[draw_idx] = phi
            sigma_eta_draws[draw_idx] = sigma_eta
            h_draws[draw_idx, :] = h
        end
    end

    # Compute volatility posterior summaries
    vol_mean = Vector{T}(undef, n)
    vol_quantiles = Matrix{T}(undef, n, length(ql))

    for t in 1:n
        vol_draws_t = exp.(@view h_draws[:, t])
        vol_mean[t] = mean(vol_draws_t)
        @inbounds for (j, q) in enumerate(ql)
            vol_quantiles[t, j] = quantile(vol_draws_t, q)
        end
    end

    SVModel(y_vec, h_draws, mu_draws, phi_draws, sigma_eta_draws,
            vol_mean, vol_quantiles, ql, dist, leverage, n_samples)
end

estimate_sv(y::AbstractVector; kwargs...) = estimate_sv(Float64.(y); kwargs...)
