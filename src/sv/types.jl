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
Type definitions and StatsAPI interface for Stochastic Volatility models.
"""

# =============================================================================
# SV Model Type
# =============================================================================

"""
    SVModel{T} <: AbstractVolatilityModel

Stochastic Volatility model (Taylor 1986), estimated via Kim-Shephard-Chib (1998) Gibbs sampler:
    yₜ = exp(hₜ/2) εₜ,       εₜ ~ N(0,1)
    hₜ = μ + φ(hₜ₋₁ - μ) + σ_η ηₜ,  ηₜ ~ N(0,1)

# Fields
- `y::Vector{T}`: Original data
- `h_draws::Matrix{T}`: Posterior draws of latent log-volatilities (n_samples × n_obs)
- `mu_post::Vector{T}`: Posterior draws of μ (log-variance level)
- `phi_post::Vector{T}`: Posterior draws of φ (persistence)
- `sigma_eta_post::Vector{T}`: Posterior draws of σ_η (volatility of volatility)
- `volatility_mean::Vector{T}`: Posterior mean of exp(hₜ) at each time t
- `volatility_quantiles::Matrix{T}`: Quantiles of exp(hₜ) (T × n_quantiles)
- `quantile_levels::Vector{T}`: Quantile levels (e.g., [0.025, 0.5, 0.975])
- `dist::Symbol`: Error distribution (:normal or :studentt)
- `leverage::Bool`: Whether leverage effect was estimated
- `n_samples::Int`: Number of posterior samples
"""
struct SVModel{T<:AbstractFloat} <: AbstractVolatilityModel
    y::Vector{T}
    h_draws::Matrix{T}          # n_samples × n_obs
    mu_post::Vector{T}
    phi_post::Vector{T}
    sigma_eta_post::Vector{T}
    volatility_mean::Vector{T}
    volatility_quantiles::Matrix{T}
    quantile_levels::Vector{T}
    dist::Symbol
    leverage::Bool
    n_samples::Int
end

# Accessors needed by _show_volatility_model
# SVModel doesn't have omega/alpha/beta, so we define mu/omega for display compatibility
function persistence(m::SVModel{T}) where {T}
    mean(m.phi_post)
end

function unconditional_variance(m::SVModel{T}) where {T}
    mu = mean(m.mu_post)
    exp(mu)
end

# =============================================================================
# StatsAPI Interface
# =============================================================================

"""Number of observations."""
StatsAPI.nobs(m::SVModel) = length(m.y)

"""Posterior mean coefficients `[μ, φ, σ_η]`."""
StatsAPI.coef(m::SVModel) = [mean(m.mu_post), mean(m.phi_post), mean(m.sigma_eta_post)]

"""Standardized residuals ``y_t / \\sqrt{\\hat{\\sigma}^2_t}``."""
StatsAPI.residuals(m::SVModel) = m.y ./ sqrt.(m.volatility_mean)

"""Posterior mean volatility series ``\\hat{\\sigma}^2_t``."""
StatsAPI.predict(m::SVModel) = m.volatility_mean

"""`false` — SV models are nonlinear."""
StatsAPI.islinear(::SVModel) = false

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, m::SVModel{T}) where {T}
    mu_m = mean(m.mu_post)
    phi_m = mean(m.phi_post)
    se_m = mean(m.sigma_eta_post)

    mu_s = std(m.mu_post)
    phi_s = std(m.phi_post)
    se_s = std(m.sigma_eta_post)

    mu_q = quantile(m.mu_post, [0.025, 0.5, 0.975])
    phi_q = quantile(m.phi_post, [0.025, 0.5, 0.975])
    se_q = quantile(m.sigma_eta_post, [0.025, 0.5, 0.975])

    title = "Stochastic Volatility Model"
    m.leverage && (title *= " (with leverage)")
    m.dist == :studentt && (title *= " (Student-t)")

    data = Any[
        "μ"   _fmt(mu_m)  _fmt(mu_s)  _fmt(mu_q[1])  _fmt(mu_q[2])  _fmt(mu_q[3]);
        "φ"   _fmt(phi_m) _fmt(phi_s) _fmt(phi_q[1])  _fmt(phi_q[2])  _fmt(phi_q[3]);
        "σ_η" _fmt(se_m)  _fmt(se_s)  _fmt(se_q[1])  _fmt(se_q[2])  _fmt(se_q[3])
    ]

    _pretty_table(io, data;
        title = title,
        column_labels = ["Parameter", "Mean", "Std", "2.5%", "50%", "97.5%"],
        alignment = [:l, :r, :r, :r, :r, :r],
    )

    info_data = Any[
        "Observations"     nobs(m);
        "Posterior samples" m.n_samples;
        "Distribution"     string(m.dist);
        "Leverage"         m.leverage ? "Yes" : "No";
        "Persistence (φ̄)" _fmt(mean(m.phi_post));
        "Unconditional σ²" _fmt(unconditional_variance(m))
    ]
    _pretty_table(io, info_data;
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
end
