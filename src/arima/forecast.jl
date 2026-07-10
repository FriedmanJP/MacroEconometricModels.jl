# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

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

"""
    _expand_ar_with_differencing(phi, d) -> Vector{T}

Expand the AR operator with `d` unit roots: φ*(L) = φ(L)(1-L)^d, of order `p+d`. Both `phi`
and the result are in the recursion convention (yₜ = Σ φᵢ yₜ₋ᵢ, characteristic polynomial
a(L) = 1 - Σ φᵢ Lⁱ). Used to compute forecast-error variances on the original (nondifferenced)
scale for ARIMA(d≥1) models.
"""
function _expand_ar_with_differencing(phi::Vector{T}, d::Int) where {T<:AbstractFloat}
    d == 0 && return copy(phi)
    p = length(phi)
    a = zeros(T, p + 1); a[1] = one(T)
    for i in 1:p
        a[i+1] = -phi[i]
    end
    b = T[binomial(d, k) * (-1)^k for k in 0:d]     # (1-L)^d coefficients
    ab = zeros(T, p + d + 1)
    for i in 0:p, k in 0:d
        ab[i+k+1] += a[i+1] * b[k+1]
    end
    T[-ab[j] for j in 2:(p+d+1)]                     # recursion-convention φ* = -ab[2:end]
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
function forecast(model::ARModel{T}, h::Int; conf_level::Real=0.95) where {T<:AbstractFloat}
    conf_level = T(conf_level)
    h < 1 && throw(ArgumentError("Forecast horizon h must be positive"))
    _forecast_arma(model.y, model.residuals, model.c, model.phi, T[], model.sigma2, h, conf_level)
end

"""
    forecast(model::MAModel, h; conf_level=0.95) -> ARIMAForecast

Compute h-step ahead forecasts with confidence intervals for MA model.
"""
function forecast(model::MAModel{T}, h::Int; conf_level::Real=0.95) where {T<:AbstractFloat}
    conf_level = T(conf_level)
    h < 1 && throw(ArgumentError("Forecast horizon h must be positive"))
    _forecast_arma(model.y, model.residuals, model.c, T[], model.theta, model.sigma2, h, conf_level)
end

"""
    forecast(model::ARMAModel, h; conf_level=0.95) -> ARIMAForecast

Compute h-step ahead forecasts with confidence intervals for ARMA model.
"""
function forecast(model::ARMAModel{T}, h::Int; conf_level::Real=0.95) where {T<:AbstractFloat}
    conf_level = T(conf_level)
    h < 1 && throw(ArgumentError("Forecast horizon h must be positive"))
    _forecast_arma(model.y, model.residuals, model.c, model.phi, model.theta, model.sigma2, h, conf_level)
end

"""
    forecast(model::ARIMAModel, h; conf_level=0.95) -> ARIMAForecast

Compute h-step ahead forecasts with confidence intervals for ARIMA model.
Forecasts are computed on the differenced series and then integrated back
to the original scale.
"""
function forecast(model::ARIMAModel{T}, h::Int; conf_level::Real=0.95) where {T<:AbstractFloat}
    conf_level = T(conf_level)
    h < 1 && throw(ArgumentError("Forecast horizon h must be positive"))

    fc_diff = _forecast_arma(model.y_diff, model.residuals, model.c,
                              model.phi, model.theta, model.sigma2, h, conf_level)
    model.d == 0 && return fc_diff

    forecasts = _integrate_forecasts(model.y, fc_diff.forecast, model.d)
    # Forecast-error variance on the ORIGINAL scale from the ψ-weights of the nondifferenced
    # operator φ*(L) = φ(L)(1-L)^d — the old code integrated the differenced CI arrays and
    # reported sqrt(cumsum(se_diff.^2)) separately, so ci ≠ forecast ± z·se and neither was
    # the correct variance (it dropped the ψ cross-terms of the full operator).
    phi_star = _expand_ar_with_differencing(model.phi, model.d)
    psi_full = _compute_psi_weights(phi_star, model.theta, h)
    se = sqrt.(_forecast_variance(model.sigma2, psi_full, h))
    ci_lower, ci_upper = _confidence_band(forecasts, se, conf_level)

    ARIMAForecast(forecasts, ci_lower, ci_upper, se, h, conf_level)
end

"""
    _integrate_forecasts(y, fc_diff, d) -> Vector{T}

Integrate d-differenced forecasts back to original scale.
"""
function _integrate_forecasts(y::Vector{T}, fc_diff::Vector{T}, d::Int) where {T<:AbstractFloat}
    d == 0 && return fc_diff

    # Invert d-fold differencing by successive integration. The forecast of the
    # (d-k)-th difference is the running sum of the forecast of the (d-k+1)-th
    # difference, seeded by the last observed (d-k)-th difference:
    #   f[Δ^{d-k} y] = cumsum(f[Δ^{d-k+1} y]) + (Δ^{d-k} y)[end].
    # This reproduces the d=1 (+y[end]) and d=2 (level + linear trend) closed forms
    # and, unlike the previous binomial boundary term, is exact for d ≥ 3 (a degree-d
    # polynomial trend is continued exactly).
    fc = copy(fc_diff)                         # forecast of Δ^d y
    for k in 1:d
        m = d - k                              # restore the level of the m-th difference
        fc = cumsum(fc)
        fc .+= _last_difference(y, m)          # (Δ^m y)[end]
    end
    fc
end

"""`(Δ^m y)[end]` — last value of the m-th backward difference of `y`."""
function _last_difference(y::AbstractVector{T}, m::Int) where {T<:AbstractFloat}
    s = zero(T)
    for i in 0:m
        s += T((-1)^i * binomial(m, i)) * y[end - i]
    end
    s
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
