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
# ARMA Forecasting
# =============================================================================

"""
    forecast(model::ARModel, h; conf_level=0.95) -> ARIMAForecast

Compute h-step ahead forecasts with confidence intervals for AR model.

# Arguments
- `model`: Fitted ARModel
- `h`: Forecast horizon
- `conf_level`: Confidence level for intervals (default 0.95)

# Returns
`ARIMAForecast` with point forecasts, confidence intervals, and standard errors.
"""
function forecast(model::ARModel{T}, h::Int; conf_level::T=T(0.95)) where {T<:AbstractFloat}
    h < 1 && throw(ArgumentError("Forecast horizon h must be positive"))

    y = model.y
    p = model.p
    n = length(y)

    # Point forecasts
    forecasts = zeros(T, h)
    y_ext = vcat(y, zeros(T, h))  # Extended series for recursive forecasting

    @inbounds for j in 1:h
        y_hat = model.c
        for i in 1:p
            y_hat += model.phi[i] * y_ext[n + j - i]
        end
        forecasts[j] = y_hat
        y_ext[n + j] = y_hat
    end

    # ψ-weights for AR model (no MA terms)
    psi = _compute_psi_weights(model.phi, T[], h)

    # Forecast variance and standard errors
    var_fc = _forecast_variance(model.sigma2, psi, h)
    se = sqrt.(var_fc)

    ci_lower, ci_upper = _confidence_band(forecasts, se, conf_level)

    ARIMAForecast(forecasts, ci_lower, ci_upper, se, h, conf_level)
end

"""
    forecast(model::MAModel, h; conf_level=0.95) -> ARIMAForecast

Compute h-step ahead forecasts with confidence intervals for MA model.
"""
function forecast(model::MAModel{T}, h::Int; conf_level::T=T(0.95)) where {T<:AbstractFloat}
    h < 1 && throw(ArgumentError("Forecast horizon h must be positive"))

    q = model.q
    residuals = model.residuals
    n = length(residuals)

    # Point forecasts
    # For MA(q), forecasts beyond q steps are just the mean
    forecasts = zeros(T, h)

    @inbounds for j in 1:h
        y_hat = model.c
        for i in 1:min(q, n + j - 1)
            if j - i >= 1
                # Future residuals are zero (best linear predictor)
                continue
            else
                # Past residuals
                idx = n + j - i
                if idx >= 1 && idx <= n
                    y_hat += model.theta[i] * residuals[idx]
                end
            end
        end
        forecasts[j] = y_hat
    end

    # ψ-weights for pure MA model
    psi = model.theta[1:min(q, h)]
    if length(psi) < h
        psi = vcat(psi, zeros(T, h - length(psi)))
    end

    # Forecast variance
    var_fc = _forecast_variance(model.sigma2, psi, h)
    se = sqrt.(var_fc)

    ci_lower, ci_upper = _confidence_band(forecasts, se, conf_level)

    ARIMAForecast(forecasts, ci_lower, ci_upper, se, h, conf_level)
end

"""
    forecast(model::ARMAModel, h; conf_level=0.95) -> ARIMAForecast

Compute h-step ahead forecasts with confidence intervals for ARMA model.

# Arguments
- `model`: Fitted ARMAModel
- `h`: Forecast horizon
- `conf_level`: Confidence level for intervals (default 0.95)

# Returns
`ARIMAForecast` with point forecasts, confidence intervals, and standard errors.
"""
function forecast(model::ARMAModel{T}, h::Int; conf_level::T=T(0.95)) where {T<:AbstractFloat}
    h < 1 && throw(ArgumentError("Forecast horizon h must be positive"))

    y = model.y
    p, q = model.p, model.q
    residuals = model.residuals
    n = length(y)

    # Point forecasts via recursion
    forecasts = zeros(T, h)
    y_ext = vcat(y, zeros(T, h))
    eps_ext = vcat(residuals, zeros(T, h))  # Future residuals = 0

    @inbounds for j in 1:h
        y_hat = model.c

        # AR component
        for i in 1:p
            y_hat += model.phi[i] * y_ext[n + j - i]
        end

        # MA component (only for past residuals, future = 0)
        for i in 1:q
            idx = n + j - i
            if idx >= 1 && idx <= n
                y_hat += model.theta[i] * eps_ext[idx]
            end
        end

        forecasts[j] = y_hat
        y_ext[n + j] = y_hat
    end

    # ψ-weights
    psi = _compute_psi_weights(model.phi, model.theta, h)

    # Forecast variance
    var_fc = _forecast_variance(model.sigma2, psi, h)
    se = sqrt.(var_fc)

    ci_lower, ci_upper = _confidence_band(forecasts, se, conf_level)

    ARIMAForecast(forecasts, ci_lower, ci_upper, se, h, conf_level)
end

"""
    forecast(model::ARIMAModel, h; conf_level=0.95) -> ARIMAForecast

Compute h-step ahead forecasts with confidence intervals for ARIMA model.

Forecasts are computed on the differenced series and then integrated back
to the original scale.

# Arguments
- `model`: Fitted ARIMAModel
- `h`: Forecast horizon
- `conf_level`: Confidence level for intervals (default 0.95)

# Returns
`ARIMAForecast` with point forecasts (on original scale), confidence intervals, and standard errors.
"""
function forecast(model::ARIMAModel{T}, h::Int; conf_level::T=T(0.95)) where {T<:AbstractFloat}
    h < 1 && throw(ArgumentError("Forecast horizon h must be positive"))

    d = model.d
    y = model.y
    y_diff = model.y_diff
    n = length(y)
    n_diff = length(y_diff)

    # Build ARMA model for forecasting on differenced series
    arma_model = ARMAModel(y_diff, model.p, model.q, model.c, model.phi, model.theta,
                           model.sigma2, model.residuals, model.fitted, model.loglik,
                           model.aic, model.bic, model.method, model.converged, model.iterations)

    # Forecast differenced series
    fc_diff = forecast(arma_model, h; conf_level=conf_level)

    # Integrate forecasts back to original scale
    if d == 0
        return fc_diff
    end

    # Recursive integration
    forecasts = _integrate_forecasts(y, fc_diff.forecast, d)
    ci_lower = _integrate_forecasts(y, fc_diff.ci_lower, d)
    ci_upper = _integrate_forecasts(y, fc_diff.ci_upper, d)

    # Standard errors need adjustment for integration
    # For d=1: Var(yₜ₊ₕ - yₜ) = Var(Σⱼ₌₁ʰ Δyₜ₊ⱼ)
    # This is a conservative approximation
    se = _integrate_se(fc_diff.se, d)

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
