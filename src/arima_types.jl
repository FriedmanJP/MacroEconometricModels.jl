"""
Type definitions for AR, MA, ARMA, and ARIMA models.
"""

using LinearAlgebra, Statistics

# =============================================================================
# Abstract Type
# =============================================================================

"""
    AbstractARIMAModel <: StatsAPI.RegressionModel

Abstract supertype for all univariate ARIMA-class models.
"""
abstract type AbstractARIMAModel <: StatsAPI.RegressionModel end

# =============================================================================
# AR Model
# =============================================================================

"""
    ARModel{T} <: AbstractARIMAModel

Autoregressive AR(p) model: yₜ = c + φ₁yₜ₋₁ + ... + φₚyₜ₋ₚ + εₜ

# Fields
- `y::Vector{T}`: Original data
- `p::Int`: AR order
- `c::T`: Intercept
- `phi::Vector{T}`: AR coefficients [φ₁, ..., φₚ]
- `sigma2::T`: Innovation variance
- `residuals::Vector{T}`: Estimated residuals
- `fitted::Vector{T}`: Fitted values
- `loglik::T`: Log-likelihood
- `aic::T`: Akaike Information Criterion
- `bic::T`: Bayesian Information Criterion
- `method::Symbol`: Estimation method (:ols, :mle)
- `converged::Bool`: Whether optimization converged
- `iterations::Int`: Number of iterations (0 for OLS)
"""
struct ARModel{T<:AbstractFloat} <: AbstractARIMAModel
    y::Vector{T}
    p::Int
    c::T
    phi::Vector{T}
    sigma2::T
    residuals::Vector{T}
    fitted::Vector{T}
    loglik::T
    aic::T
    bic::T
    method::Symbol
    converged::Bool
    iterations::Int
end

# =============================================================================
# MA Model
# =============================================================================

"""
    MAModel{T} <: AbstractARIMAModel

Moving average MA(q) model: yₜ = c + εₜ + θ₁εₜ₋₁ + ... + θqεₜ₋q

# Fields
- `y::Vector{T}`: Original data
- `q::Int`: MA order
- `c::T`: Intercept
- `theta::Vector{T}`: MA coefficients [θ₁, ..., θq]
- `sigma2::T`: Innovation variance
- `residuals::Vector{T}`: Estimated residuals
- `fitted::Vector{T}`: Fitted values
- `loglik::T`: Log-likelihood
- `aic::T`: Akaike Information Criterion
- `bic::T`: Bayesian Information Criterion
- `method::Symbol`: Estimation method (:css, :mle, :css_mle)
- `converged::Bool`: Whether optimization converged
- `iterations::Int`: Number of iterations
"""
struct MAModel{T<:AbstractFloat} <: AbstractARIMAModel
    y::Vector{T}
    q::Int
    c::T
    theta::Vector{T}
    sigma2::T
    residuals::Vector{T}
    fitted::Vector{T}
    loglik::T
    aic::T
    bic::T
    method::Symbol
    converged::Bool
    iterations::Int
end

# =============================================================================
# ARMA Model
# =============================================================================

"""
    ARMAModel{T} <: AbstractARIMAModel

Autoregressive moving average ARMA(p,q) model:
yₜ = c + φ₁yₜ₋₁ + ... + φₚyₜ₋ₚ + εₜ + θ₁εₜ₋₁ + ... + θqεₜ₋q

# Fields
- `y::Vector{T}`: Original data
- `p::Int`: AR order
- `q::Int`: MA order
- `c::T`: Intercept
- `phi::Vector{T}`: AR coefficients [φ₁, ..., φₚ]
- `theta::Vector{T}`: MA coefficients [θ₁, ..., θq]
- `sigma2::T`: Innovation variance
- `residuals::Vector{T}`: Estimated residuals
- `fitted::Vector{T}`: Fitted values
- `loglik::T`: Log-likelihood
- `aic::T`: Akaike Information Criterion
- `bic::T`: Bayesian Information Criterion
- `method::Symbol`: Estimation method (:css, :mle, :css_mle)
- `converged::Bool`: Whether optimization converged
- `iterations::Int`: Number of iterations
"""
struct ARMAModel{T<:AbstractFloat} <: AbstractARIMAModel
    y::Vector{T}
    p::Int
    q::Int
    c::T
    phi::Vector{T}
    theta::Vector{T}
    sigma2::T
    residuals::Vector{T}
    fitted::Vector{T}
    loglik::T
    aic::T
    bic::T
    method::Symbol
    converged::Bool
    iterations::Int
end

# =============================================================================
# ARIMA Model
# =============================================================================

"""
    ARIMAModel{T} <: AbstractARIMAModel

Autoregressive integrated moving average ARIMA(p,d,q) model.
The model is fit to the d-times differenced series as ARMA(p,q).

# Fields
- `y::Vector{T}`: Original (undifferenced) data
- `y_diff::Vector{T}`: Differenced series
- `p::Int`: AR order
- `d::Int`: Integration order (number of differences)
- `q::Int`: MA order
- `c::T`: Intercept (on differenced series)
- `phi::Vector{T}`: AR coefficients
- `theta::Vector{T}`: MA coefficients
- `sigma2::T`: Innovation variance
- `residuals::Vector{T}`: Estimated residuals
- `fitted::Vector{T}`: Fitted values (on differenced series)
- `loglik::T`: Log-likelihood
- `aic::T`: Akaike Information Criterion
- `bic::T`: Bayesian Information Criterion
- `method::Symbol`: Estimation method
- `converged::Bool`: Whether optimization converged
- `iterations::Int`: Number of iterations
"""
struct ARIMAModel{T<:AbstractFloat} <: AbstractARIMAModel
    y::Vector{T}
    y_diff::Vector{T}
    p::Int
    d::Int
    q::Int
    c::T
    phi::Vector{T}
    theta::Vector{T}
    sigma2::T
    residuals::Vector{T}
    fitted::Vector{T}
    loglik::T
    aic::T
    bic::T
    method::Symbol
    converged::Bool
    iterations::Int
end

# =============================================================================
# Forecast Result
# =============================================================================

"""
    ARIMAForecast{T}

Forecast result from an ARIMA-class model.

# Fields
- `forecast::Vector{T}`: Point forecasts
- `ci_lower::Vector{T}`: Lower confidence interval bound
- `ci_upper::Vector{T}`: Upper confidence interval bound
- `se::Vector{T}`: Standard errors of forecasts
- `horizon::Int`: Forecast horizon
- `conf_level::T`: Confidence level (e.g., 0.95)
"""
struct ARIMAForecast{T<:AbstractFloat}
    forecast::Vector{T}
    ci_lower::Vector{T}
    ci_upper::Vector{T}
    se::Vector{T}
    horizon::Int
    conf_level::T
end

# =============================================================================
# Order Selection Result
# =============================================================================

"""
    ARIMAOrderSelection{T}

Result from automatic ARIMA order selection.

# Fields
- `best_p_aic::Int`: Best AR order by AIC
- `best_q_aic::Int`: Best MA order by AIC
- `best_p_bic::Int`: Best AR order by BIC
- `best_q_bic::Int`: Best MA order by BIC
- `aic_matrix::Matrix{T}`: AIC values for all (p,q) combinations
- `bic_matrix::Matrix{T}`: BIC values for all (p,q) combinations
- `best_model_aic::AbstractARIMAModel`: Fitted model with best AIC
- `best_model_bic::AbstractARIMAModel`: Fitted model with best BIC
"""
struct ARIMAOrderSelection{T<:AbstractFloat, M<:AbstractARIMAModel}
    best_p_aic::Int
    best_q_aic::Int
    best_p_bic::Int
    best_q_bic::Int
    aic_matrix::Matrix{T}
    bic_matrix::Matrix{T}
    best_model_aic::M
    best_model_bic::M
end

# =============================================================================
# Type Accessors
# =============================================================================

"""Return AR order p."""
ar_order(m::ARModel) = m.p
ar_order(m::MAModel) = 0
ar_order(m::ARMAModel) = m.p
ar_order(m::ARIMAModel) = m.p

"""Return MA order q."""
ma_order(m::ARModel) = 0
ma_order(m::MAModel) = m.q
ma_order(m::ARMAModel) = m.q
ma_order(m::ARIMAModel) = m.q

"""Return integration order d."""
diff_order(m::ARModel) = 0
diff_order(m::MAModel) = 0
diff_order(m::ARMAModel) = 0
diff_order(m::ARIMAModel) = m.d

# =============================================================================
# StatsAPI Interface
# =============================================================================

# Number of observations
StatsAPI.nobs(m::AbstractARIMAModel) = length(m.y)

# Coefficients
StatsAPI.coef(m::ARModel) = vcat(m.c, m.phi)
StatsAPI.coef(m::MAModel) = vcat(m.c, m.theta)
StatsAPI.coef(m::ARMAModel) = vcat(m.c, m.phi, m.theta)
StatsAPI.coef(m::ARIMAModel) = vcat(m.c, m.phi, m.theta)

# Residuals
StatsAPI.residuals(m::AbstractARIMAModel) = m.residuals

# Fitted values
StatsAPI.predict(m::AbstractARIMAModel) = m.fitted

# Log-likelihood and information criteria
StatsAPI.loglikelihood(m::AbstractARIMAModel) = m.loglik
StatsAPI.aic(m::AbstractARIMAModel) = m.aic
StatsAPI.bic(m::AbstractARIMAModel) = m.bic

# Degrees of freedom
StatsAPI.dof(m::AbstractARIMAModel) = ar_order(m) + ma_order(m) + 2

# Residual degrees of freedom
StatsAPI.dof_residual(m::AbstractARIMAModel) = length(m.residuals) - dof(m) + 1

# R-squared (proportion of variance explained)
function StatsAPI.r2(m::AbstractARIMAModel)
    y_centered = m.y[end-length(m.residuals)+1:end] .- mean(m.y[end-length(m.residuals)+1:end])
    ss_tot = sum(abs2, y_centered)
    ss_res = sum(abs2, m.residuals)
    max(1 - ss_res / ss_tot, zero(eltype(m.residuals)))
end

# Model is linear
StatsAPI.islinear(::AbstractARIMAModel) = true

# =============================================================================
# Display Methods
# =============================================================================

function _show_arima_model(io::IO, header::String, m::AbstractARIMAModel;
                           phi::Vector=Float64[], theta::Vector=Float64[])
    println(io, header)
    println(io, "  Intercept: $(round(m.c, digits=4))")
    !isempty(phi) && println(io, "  AR coefficients: $(round.(phi, digits=4))")
    !isempty(theta) && println(io, "  MA coefficients: $(round.(theta, digits=4))")
    println(io, "  σ²: $(round(m.sigma2, digits=4))")
    println(io, "  Log-likelihood: $(round(m.loglik, digits=2))")
    println(io, "  AIC: $(round(m.aic, digits=2)), BIC: $(round(m.bic, digits=2))")
    print(io, "  Method: $(m.method), Converged: $(m.converged)")
end

Base.show(io::IO, m::ARModel) = _show_arima_model(io, "AR($(m.p)) Model", m; phi=m.phi)
Base.show(io::IO, m::MAModel) = _show_arima_model(io, "MA($(m.q)) Model", m; theta=m.theta)
Base.show(io::IO, m::ARMAModel) = _show_arima_model(io, "ARMA($(m.p),$(m.q)) Model", m; phi=m.phi, theta=m.theta)
Base.show(io::IO, m::ARIMAModel) = _show_arima_model(io, "ARIMA($(m.p),$(m.d),$(m.q)) Model", m; phi=m.phi, theta=m.theta)

function Base.show(io::IO, f::ARIMAForecast)
    println(io, "ARIMA Forecast (h=$(f.horizon), $(round(100*f.conf_level, digits=0))% CI)")
    for i in 1:min(5, f.horizon)
        println(io, "  h=$i: $(round(f.forecast[i], digits=4)) [$(round(f.ci_lower[i], digits=4)), $(round(f.ci_upper[i], digits=4))]")
    end
    f.horizon > 5 && print(io, "  ... ($(f.horizon - 5) more)")
end

function Base.show(io::IO, r::ARIMAOrderSelection)
    println(io, "ARIMA Order Selection")
    println(io, "  Best by AIC: p=$(r.best_p_aic), q=$(r.best_q_aic) (AIC=$(round(r.aic_matrix[r.best_p_aic+1, r.best_q_aic+1], digits=2)))")
    print(io, "  Best by BIC: p=$(r.best_p_bic), q=$(r.best_q_bic) (BIC=$(round(r.bic_matrix[r.best_p_bic+1, r.best_q_bic+1], digits=2)))")
end
