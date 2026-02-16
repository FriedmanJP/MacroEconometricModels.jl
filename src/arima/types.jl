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
Type definitions for AR, MA, ARMA, and ARIMA models.
"""

using LinearAlgebra, Statistics

# =============================================================================
# Abstract Type
# =============================================================================

"""
    AbstractARIMAModel{T<:AbstractFloat} <: StatsAPI.RegressionModel

Abstract supertype for all univariate ARIMA-class models, parametric on
the floating-point element type `T`.
"""
abstract type AbstractARIMAModel{T<:AbstractFloat} <: StatsAPI.RegressionModel end

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
struct ARModel{T<:AbstractFloat} <: AbstractARIMAModel{T}
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
struct MAModel{T<:AbstractFloat} <: AbstractARIMAModel{T}
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
struct ARMAModel{T<:AbstractFloat} <: AbstractARIMAModel{T}
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
struct ARIMAModel{T<:AbstractFloat} <: AbstractARIMAModel{T}
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
struct ARIMAForecast{T<:AbstractFloat} <: AbstractForecastResult{T}
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

# Variance-covariance matrix (diagonal from stderror)
function StatsAPI.vcov(m::AbstractARIMAModel)
    se = stderror(m)
    Diagonal(se .^ 2)
end

# Confidence intervals
function StatsAPI.confint(m::AbstractARIMAModel; level::Real=0.95)
    T = eltype(m.y)
    se = stderror(m)
    z = T(quantile(Normal(), (1 + level) / 2))
    c = coef(m)
    hcat(c .- z .* se, c .+ z .* se)
end

# Model is linear
StatsAPI.islinear(::AbstractARIMAModel) = true

# =============================================================================
# Display Methods
# =============================================================================

function _show_arima_model(io::IO, header::String, m::AbstractARIMAModel;
                           phi::Vector=Float64[], theta::Vector=Float64[])
    # Build parameter names and values
    param_names = String["Intercept (c)"]
    param_vals = eltype(m.y)[m.c]
    for (i, p) in enumerate(phi)
        push!(param_names, "φ[$i]")
        push!(param_vals, p)
    end
    for (i, t) in enumerate(theta)
        push!(param_names, "θ[$i]")
        push!(param_vals, t)
    end

    # Compute standard errors
    T_float = eltype(m.y)
    se = try
        stderror(m)
    catch
        fill(T_float(NaN), length(param_vals))
    end
    # Pad SE if shorter (shouldn't happen, but defensive)
    while length(se) < length(param_vals)
        push!(se, T_float(NaN))
    end

    # Publication-quality coefficient table
    _coef_table(io, header, param_names, param_vals, se; dist=:z)

    # σ² row (separate — not a regression coefficient)
    sigma_data = Any["σ²" _fmt(m.sigma2)]
    _pretty_table(io, sigma_data;
        column_labels = ["", "Estimate"],
        alignment = [:l, :r],
    )

    # Fit statistics table
    n_obs = length(m.y)
    r2_val = r2(m)
    fit_data = Any[
        "Observations"   n_obs;
        "Log-likelihood" _fmt(m.loglik; digits=4);
        "AIC"            _fmt(m.aic; digits=4);
        "BIC"            _fmt(m.bic; digits=4);
        "R²"             _fmt(r2_val);
        "S.E. of regression" _fmt(sqrt(m.sigma2));
        "Method"         string(m.method);
        "Converged"      m.converged ? "Yes" : "No"
    ]
    _pretty_table(io, fit_data;
        column_labels = ["Fit", "Value"],
        alignment = [:l, :r],
    )

    _sig_legend(io)
end

Base.show(io::IO, m::ARModel) = _show_arima_model(io, "AR($(m.p)) Model", m; phi=m.phi)
Base.show(io::IO, m::MAModel) = _show_arima_model(io, "MA($(m.q)) Model", m; theta=m.theta)
Base.show(io::IO, m::ARMAModel) = _show_arima_model(io, "ARMA($(m.p),$(m.q)) Model", m; phi=m.phi, theta=m.theta)
Base.show(io::IO, m::ARIMAModel) = _show_arima_model(io, "ARIMA($(m.p),$(m.d),$(m.q)) Model", m; phi=m.phi, theta=m.theta)

function Base.show(io::IO, f::ARIMAForecast)
    h = f.horizon
    ci_pct = round(Int, 100 * f.conf_level)
    n_show = min(10, h)
    nrows = h > n_show ? n_show + 1 : n_show
    data = Matrix{Any}(undef, nrows, 5)
    for i in 1:n_show
        data[i, 1] = i
        data[i, 2] = _fmt(f.forecast[i])
        data[i, 3] = _fmt(f.se[i])
        data[i, 4] = _fmt(f.ci_lower[i])
        data[i, 5] = _fmt(f.ci_upper[i])
    end
    if h > n_show
        data[nrows, 1] = "..."
        data[nrows, 2] = "($(h - n_show) more)"
        data[nrows, 3] = ""
        data[nrows, 4] = ""
        data[nrows, 5] = ""
    end
    _pretty_table(io, data;
        title = "ARIMA Forecast (h=$h, $(ci_pct)% CI)",
        column_labels = ["h", "Forecast", "Std. Err.", "Lower", "Upper"],
        alignment = [:r, :r, :r, :r, :r],
    )
end

function Base.show(io::IO, r::ARIMAOrderSelection)
    aic_val = _fmt(r.aic_matrix[r.best_p_aic+1, r.best_q_aic+1]; digits=2)
    bic_val = _fmt(r.bic_matrix[r.best_p_bic+1, r.best_q_bic+1]; digits=2)
    data = Any[
        "AIC" r.best_p_aic r.best_q_aic aic_val;
        "BIC" r.best_p_bic r.best_q_bic bic_val
    ]
    _pretty_table(io, data;
        title = "ARIMA Order Selection",
        column_labels = ["Criterion", "p", "q", "Value"],
        alignment = [:l, :r, :r, :r],
    )
end
