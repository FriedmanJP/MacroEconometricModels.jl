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
Forecasting functions for AR, MA, ARMA, and ARIMA models.

Implements:
- Multi-step ahead forecasting
- Forecast standard errors via ψ-weights
- Confidence intervals
"""

using LinearAlgebra, Distributions

# =============================================================================
# Shared Helpers
# =============================================================================

"""
    _confidence_band(forecasts, se, conf_level)

Compute symmetric confidence interval bounds from forecasts, standard errors,
and a confidence level.
"""
function _confidence_band(forecasts::Vector{T}, se::Vector{T}, conf_level::T) where {T}
    z = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    (forecasts .- z .* se, forecasts .+ z .* se)
end

# =============================================================================
# ψ-Weights (MA Representation Coefficients)
# =============================================================================

"""
    _compute_psi_weights(phi, theta, h) -> Vector{T}

Compute ψ-weights for the MA(∞) representation of an ARMA process.

The ARMA(p,q) process can be written as:
yₜ = μ + Σⱼ₌₀^∞ ψⱼ εₜ₋ⱼ

where ψ₀ = 1 and ψⱼ follows the recursion:
ψⱼ = φ₁ψⱼ₋₁ + ... + φₚψⱼ₋ₚ + θⱼ

Returns [ψ₁, ψ₂, ..., ψₕ] (excludes ψ₀ = 1).
"""
function _compute_psi_weights(phi::Vector{T}, theta::Vector{T}, h::Int) where {T<:AbstractFloat}
    p, q = length(phi), length(theta)
    psi = zeros(T, h)

    @inbounds for j in 1:h
        # AR contribution
        ar_part = zero(T)
        for i in 1:min(p, j)
            psi_prev = j - i == 0 ? one(T) : psi[j-i]
            ar_part += phi[i] * psi_prev
        end

        # MA contribution
        ma_part = j <= q ? theta[j] : zero(T)

        psi[j] = ar_part + ma_part
    end

    psi
end

# =============================================================================
# Forecast Variance
# =============================================================================

"""
    _forecast_variance(sigma2, psi, h) -> Vector{T}

Compute h-step ahead forecast variance.

Var(eₜ₊ₕ) = σ² (1 + ψ₁² + ψ₂² + ... + ψₕ₋₁²)
"""
function _forecast_variance(sigma2::T, psi::Vector{T}, h::Int) where {T<:AbstractFloat}
    var_fc = zeros(T, h)
    var_fc[1] = sigma2

    cumsum_psi_sq = zero(T)
    @inbounds for j in 2:h
        cumsum_psi_sq += psi[j-1]^2
        var_fc[j] = sigma2 * (1 + cumsum_psi_sq)
    end

    var_fc
end

# =============================================================================
# Unified ARMA Forecasting
# =============================================================================

"""
    _forecast_arma(y, resid, c, phi, theta, sigma2, h, conf_level) -> ARIMAForecast

Unified point forecast + CI computation for any ARMA(p,q) model.
AR models pass `theta=T[]`, MA models pass `phi=T[]`.
"""
function _forecast_arma(y::Vector{T}, resid::Vector{T}, c::T,
                        phi::Vector{T}, theta::Vector{T},
                        sigma2::T, h::Int, conf_level::T) where {T<:AbstractFloat}
    p, q = length(phi), length(theta)
    n = length(y)

    # Point forecasts via recursion
    forecasts = zeros(T, h)
    y_ext = vcat(y, zeros(T, h))
    eps_ext = vcat(resid, zeros(T, h))

    @inbounds for j in 1:h
        y_hat = c
        for i in 1:p
            y_hat += phi[i] * y_ext[n + j - i]
        end
        for i in 1:q
            idx = n + j - i
            if idx >= 1 && idx <= n
                y_hat += theta[i] * eps_ext[idx]
            end
        end
        forecasts[j] = y_hat
        y_ext[n + j] = y_hat
    end

    # ψ-weights, variance, and confidence bands
    psi = _compute_psi_weights(phi, theta, h)
    var_fc = _forecast_variance(sigma2, psi, h)
    se = sqrt.(var_fc)
    ci_lower, ci_upper = _confidence_band(forecasts, se, conf_level)

    ARIMAForecast(forecasts, ci_lower, ci_upper, se, h, conf_level)
end

# =============================================================================
# Public Forecast Methods (thin wrappers)
# =============================================================================

"""
    forecast(model::ARModel, h; conf_level=0.95) -> ARIMAForecast

Compute h-step ahead forecasts with confidence intervals for AR model.
"""
function forecast(model::ARModel{T}, h::Int; conf_level::T=T(0.95)) where {T<:AbstractFloat}
    h < 1 && throw(ArgumentError("Forecast horizon h must be positive"))
    _forecast_arma(model.y, model.residuals, model.c, model.phi, T[], model.sigma2, h, conf_level)
end

"""
    forecast(model::MAModel, h; conf_level=0.95) -> ARIMAForecast

Compute h-step ahead forecasts with confidence intervals for MA model.
"""
function forecast(model::MAModel{T}, h::Int; conf_level::T=T(0.95)) where {T<:AbstractFloat}
    h < 1 && throw(ArgumentError("Forecast horizon h must be positive"))
    _forecast_arma(model.y, model.residuals, model.c, T[], model.theta, model.sigma2, h, conf_level)
end

"""
    forecast(model::ARMAModel, h; conf_level=0.95) -> ARIMAForecast

Compute h-step ahead forecasts with confidence intervals for ARMA model.
"""
function forecast(model::ARMAModel{T}, h::Int; conf_level::T=T(0.95)) where {T<:AbstractFloat}
    h < 1 && throw(ArgumentError("Forecast horizon h must be positive"))
    _forecast_arma(model.y, model.residuals, model.c, model.phi, model.theta, model.sigma2, h, conf_level)
end

"""
    forecast(model::ARIMAModel, h; conf_level=0.95) -> ARIMAForecast

Compute h-step ahead forecasts with confidence intervals for ARIMA model.
Forecasts are computed on the differenced series and then integrated back
to the original scale.
"""
function forecast(model::ARIMAModel{T}, h::Int; conf_level::T=T(0.95)) where {T<:AbstractFloat}
    h < 1 && throw(ArgumentError("Forecast horizon h must be positive"))

    fc_diff = _forecast_arma(model.y_diff, model.residuals, model.c,
                              model.phi, model.theta, model.sigma2, h, conf_level)
    model.d == 0 && return fc_diff

    forecasts = _integrate_forecasts(model.y, fc_diff.forecast, model.d)
    ci_lower = _integrate_forecasts(model.y, fc_diff.ci_lower, model.d)
    ci_upper = _integrate_forecasts(model.y, fc_diff.ci_upper, model.d)
    se = _integrate_se(fc_diff.se, model.d)

    ARIMAForecast(forecasts, ci_lower, ci_upper, se, h, conf_level)
end

"""
    _integrate_forecasts(y, fc_diff, d) -> Vector{T}

Integrate d-differenced forecasts back to original scale.
"""
function _integrate_forecasts(y::Vector{T}, fc_diff::Vector{T}, d::Int) where {T<:AbstractFloat}
    d == 0 && return fc_diff

    h = length(fc_diff)
    n = length(y)

    # Build chain of cumulative sums
    fc = copy(fc_diff)

    for _ in 1:d
        # Get last value from original series (after accounting for previous integrations)
        # For d=1: yₜ₊ₕ = yₜ + Σⱼ₌₁ʰ Δyₜ₊ⱼ
        fc = cumsum(fc)
    end

    # Add the reference levels
    if d == 1
        fc .+= y[end]
    elseif d == 2
        # For d=2, need to account for both level and trend
        fc .+= y[end]
        # Add linear trend from last two observations
        trend = y[end] - y[end-1]
        fc .+= trend .* (1:h)
    else
        # General case: use polynomial extrapolation
        for i in 1:d
            fc .+= y[end - d + i] * binomial(d, i-1)
        end
    end

    fc
end

"""
    _integrate_se(se_diff, d) -> Vector{T}

Approximate standard errors after integration.

For d-fold integration, the variance grows roughly as h^d.
This is a conservative approximation.
"""
function _integrate_se(se_diff::Vector{T}, d::Int) where {T<:AbstractFloat}
    d == 0 && return se_diff

    h = length(se_diff)
    se = copy(se_diff)

    # Approximate: for d=1, Var(cumsum) ≈ Σ Var (grows with h)
    # Conservative multiplier based on horizon
    for _ in 1:d
        cumvar = cumsum(se .^ 2)
        se = sqrt.(cumvar)
    end

    se
end

# =============================================================================
# Convenience Function for StatsAPI
# =============================================================================

"""
    predict(model::AbstractARIMAModel, h::Int) -> Vector{T}

Return h-step ahead point forecasts (without confidence intervals).
"""
function StatsAPI.predict(model::AbstractARIMAModel, h::Int)
    fc = forecast(model, h)
    fc.forecast
end
