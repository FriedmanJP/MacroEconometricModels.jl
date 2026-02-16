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
Type definitions for Local Projection methods.

Implements types for:
- Core LP estimation (Jordà 2005)
- LP with IV (Stock & Watson 2018)
- Smooth LP with B-splines (Barnichon & Brownlees 2019)
- State-dependent LP (Auerbach & Gorodnichenko 2013)
- Propensity score LP (Angrist et al. 2018)

Note: GMM types are defined in gmm.jl
Note: Covariance estimator types are defined in covariance_estimators.jl
"""

using LinearAlgebra, StatsAPI

# =============================================================================
# Abstract Types
# =============================================================================

"""Abstract supertype for Local Projection models."""
abstract type AbstractLPModel <: StatsAPI.RegressionModel end

"""Abstract supertype for LP impulse response results."""
abstract type AbstractLPImpulseResponse <: AbstractImpulseResponse end

# Note: AbstractCovarianceEstimator and its subtypes (NeweyWestEstimator, WhiteEstimator,
# DriscollKraayEstimator) are now defined in covariance_estimators.jl

# =============================================================================
# Core LP Model Type (Jordà 2005)
# =============================================================================

"""
    LPModel{T} <: AbstractLPModel

Local Projection model estimated via OLS with robust standard errors (Jordà 2005).

The LP regression for horizon h:
    y_{t+h} = α_h + β_h * shock_t + Γ_h * controls_t + ε_{t+h}

Fields:
- Y: Response data matrix (T_obs × n_vars)
- shock_var: Index of shock variable in Y
- response_vars: Indices of response variables (default: all)
- horizon: Maximum IRF horizon H
- lags: Number of control lags included
- B: Vector of coefficient matrices, one per horizon h=0,...,H
- residuals: Vector of residual matrices per horizon
- vcov: Vector of robust covariance matrices per horizon
- T_eff: Effective sample sizes per horizon
- cov_estimator: Covariance estimator used
"""
struct LPModel{T<:AbstractFloat} <: AbstractLPModel
    Y::Matrix{T}
    shock_var::Int
    response_vars::Vector{Int}
    horizon::Int
    lags::Int
    B::Vector{Matrix{T}}
    residuals::Vector{Matrix{T}}
    vcov::Vector{Matrix{T}}
    T_eff::Vector{Int}
    cov_estimator::AbstractCovarianceEstimator

    function LPModel(Y::Matrix{T}, shock_var::Int, response_vars::Vector{Int},
                     horizon::Int, lags::Int, B::Vector{Matrix{T}},
                     residuals::Vector{Matrix{T}}, vcov::Vector{Matrix{T}},
                     T_eff::Vector{Int}, cov_estimator::AbstractCovarianceEstimator) where {T<:AbstractFloat}
        n = size(Y, 2)
        @assert 1 <= shock_var <= n "shock_var must be in 1:$n"
        @assert all(1 .<= response_vars .<= n) "response_vars must be in 1:$n"
        @assert horizon >= 0 "horizon must be non-negative"
        @assert lags >= 0 "lags must be non-negative"
        @assert length(B) == horizon + 1 "B must have H+1 elements"
        @assert length(residuals) == horizon + 1 "residuals must have H+1 elements"
        @assert length(vcov) == horizon + 1 "vcov must have H+1 elements"
        @assert length(T_eff) == horizon + 1 "T_eff must have H+1 elements"
        new{T}(Y, shock_var, response_vars, horizon, lags, B, residuals, vcov, T_eff, cov_estimator)
    end
end

# Convenience constructor with type promotion
function LPModel(Y::AbstractMatrix, shock_var::Int, response_vars::Vector{Int},
                 horizon::Int, lags::Int, B::Vector{<:AbstractMatrix},
                 residuals::Vector{<:AbstractMatrix}, vcov::Vector{<:AbstractMatrix},
                 T_eff::Vector{Int}, cov_estimator::AbstractCovarianceEstimator)
    T = promote_type(eltype(Y), eltype(first(B)))
    LPModel(Matrix{T}(Y), shock_var, response_vars, horizon, lags,
            [Matrix{T}(b) for b in B], [Matrix{T}(r) for r in residuals],
            [Matrix{T}(v) for v in vcov], T_eff, cov_estimator)
end

# Accessors
nvars(model::LPModel) = size(model.Y, 2)
nlags(model::LPModel) = model.lags
nhorizons(model::LPModel) = model.horizon + 1
nresponse(model::LPModel) = length(model.response_vars)

# =============================================================================
# LP Impulse Response Type
# =============================================================================

"""
    LPImpulseResponse{T} <: AbstractLPImpulseResponse

LP-based impulse response function with confidence intervals from robust standard errors.

Fields:
- values: Point estimates (H+1 × n_response)
- ci_lower: Lower CI bounds
- ci_upper: Upper CI bounds
- se: Standard errors
- horizon: Maximum horizon
- response_vars: Names of response variables
- shock_var: Name of shock variable
- cov_type: Covariance estimator type
- conf_level: Confidence level used
"""
struct LPImpulseResponse{T<:AbstractFloat} <: AbstractLPImpulseResponse
    values::Matrix{T}
    ci_lower::Matrix{T}
    ci_upper::Matrix{T}
    se::Matrix{T}
    horizon::Int
    response_vars::Vector{String}
    shock_var::String
    cov_type::Symbol
    conf_level::T

    function LPImpulseResponse{T}(values::Matrix{T}, ci_lower::Matrix{T}, ci_upper::Matrix{T},
                                   se::Matrix{T}, horizon::Int, response_vars::Vector{String},
                                   shock_var::String, cov_type::Symbol, conf_level::T) where {T<:AbstractFloat}
        @assert size(values) == size(ci_lower) == size(ci_upper) == size(se)
        @assert size(values, 1) == horizon + 1
        @assert length(response_vars) == size(values, 2)
        @assert 0 < conf_level < 1
        new{T}(values, ci_lower, ci_upper, se, horizon, response_vars, shock_var, cov_type, conf_level)
    end
end

# =============================================================================

# LP variant types (LP-IV, Smooth, State-Dependent, Propensity) — extracted to types_variants.jl
include("types_variants.jl")

# =============================================================================
# Structural LP Type (Plagborg-Møller & Wolf 2021)
# =============================================================================

"""
    StructuralLP{T} <: AbstractFrequentistResult

Structural Local Projection result combining VAR-based identification with LP estimation.

Estimates multi-shock IRFs by computing orthogonalized structural shocks from a VAR model
and using them as regressors in LP regressions (Plagborg-Møller & Wolf 2021).

Fields:
- `irf`: 3D impulse responses (H × n × n) — reuses `ImpulseResponse{T}`
- `structural_shocks`: Structural shocks (T_eff × n)
- `var_model`: Underlying VAR model used for identification
- `Q`: Rotation/identification matrix
- `method`: Identification method used (:cholesky, :sign, :long_run, :fastica, etc.)
- `lags`: Number of LP control lags
- `cov_type`: HAC estimator type
- `se`: Standard errors (H × n × n)
- `lp_models`: Individual LP model per shock
"""
struct StructuralLP{T<:AbstractFloat} <: AbstractFrequentistResult
    irf::ImpulseResponse{T}
    structural_shocks::Matrix{T}
    var_model::VARModel{T}
    Q::Matrix{T}
    method::Symbol
    lags::Int
    cov_type::Symbol
    se::Array{T,3}
    lp_models::Vector{LPModel{T}}
end

# Accessors
nvars(slp::StructuralLP) = nvars(slp.var_model)

# =============================================================================
# LP Forecast Type
# =============================================================================

"""
    LPForecast{T}

Direct multi-step LP forecast result.

Each horizon h uses its own regression coefficients directly (no recursion),
producing ŷ_{T+h} = α_h + β_h·shock_h + Γ_h·controls_T.

Fields:
- `forecasts`: Point forecasts (H × n_response)
- `ci_lower`: Lower CI bounds (H × n_response)
- `ci_upper`: Upper CI bounds (H × n_response)
- `se`: Standard errors (H × n_response)
- `horizon`: Maximum forecast horizon
- `response_vars`: Response variable indices
- `shock_var`: Shock variable index
- `shock_path`: Assumed shock trajectory
- `conf_level`: Confidence level
- `ci_method`: CI method (:analytical, :bootstrap, :none)
"""
struct LPForecast{T<:AbstractFloat} <: AbstractForecastResult{T}
    forecasts::Matrix{T}
    ci_lower::Matrix{T}
    ci_upper::Matrix{T}
    se::Matrix{T}
    horizon::Int
    response_vars::Vector{Int}
    shock_var::Int
    shock_path::Vector{T}
    conf_level::T
    ci_method::Symbol

    function LPForecast(forecasts::Matrix{T}, ci_lower::Matrix{T}, ci_upper::Matrix{T},
                        se::Matrix{T}, horizon::Int, response_vars::Vector{Int},
                        shock_var::Int, shock_path::Vector{T}, conf_level::T,
                        ci_method::Symbol) where {T<:AbstractFloat}
        @assert size(forecasts) == size(ci_lower) == size(ci_upper) == size(se)
        @assert size(forecasts, 1) == horizon
        @assert size(forecasts, 2) == length(response_vars)
        @assert length(shock_path) == horizon
        @assert 0 < conf_level < 1
        @assert ci_method ∈ (:analytical, :bootstrap, :none)
        new{T}(forecasts, ci_lower, ci_upper, se, horizon, response_vars,
               shock_var, shock_path, conf_level, ci_method)
    end
end

# =============================================================================
# LP-FEVD Type (Gorodnichenko & Lee 2019)
# =============================================================================

"""
    LPFEVD{T} <: AbstractFEVD

LP-based Forecast Error Variance Decomposition (Gorodnichenko & Lee 2019).

Uses R²-based estimator: regress estimated LP forecast errors on identified
structural shocks to measure the share of forecast error variance attributable
to each shock. Includes VAR-based bootstrap bias correction and CIs.

# Fields
- `proportions`: Raw FEVD estimates (n × n × H), `[i,j,h]` = share of
  variable i's h-step forecast error variance due to shock j
- `bias_corrected`: Bias-corrected FEVD (n × n × H)
- `se`: Bootstrap standard errors (n × n × H)
- `ci_lower`: Lower CI bounds (n × n × H)
- `ci_upper`: Upper CI bounds (n × n × H)
- `method`: Estimator (:r2, :lp_a, :lp_b)
- `horizon`: Maximum FEVD horizon
- `n_boot`: Number of bootstrap replications used
- `conf_level`: Confidence level for CIs
- `bias_correction`: Whether bias correction was applied

# Reference
Gorodnichenko, Y. & Lee, B. (2019). "Forecast Error Variance Decompositions
with Local Projections." *JBES*, 38(4), 921–933.
"""
struct LPFEVD{T<:AbstractFloat} <: AbstractFEVD
    proportions::Array{T,3}
    bias_corrected::Array{T,3}
    se::Array{T,3}
    ci_lower::Array{T,3}
    ci_upper::Array{T,3}
    method::Symbol
    horizon::Int
    n_boot::Int
    conf_level::T
    bias_correction::Bool
end

# =============================================================================
# StatsAPI Interface for LP Models
# =============================================================================

StatsAPI.coef(model::LPModel) = model.B
StatsAPI.coef(model::LPModel, h::Int) = model.B[h + 1]
StatsAPI.residuals(model::LPModel) = model.residuals
StatsAPI.residuals(model::LPModel, h::Int) = model.residuals[h + 1]
StatsAPI.vcov(model::LPModel) = model.vcov
StatsAPI.vcov(model::LPModel, h::Int) = model.vcov[h + 1]
StatsAPI.nobs(model::LPModel) = size(model.Y, 1)
StatsAPI.nobs(model::LPModel, h::Int) = model.T_eff[h + 1]
StatsAPI.dof(model::LPModel) = sum(length(b) for b in model.B)
StatsAPI.islinear(::LPModel) = true

# For LPIVModel
StatsAPI.coef(model::LPIVModel) = model.B
StatsAPI.coef(model::LPIVModel, h::Int) = model.B[h + 1]
StatsAPI.residuals(model::LPIVModel) = model.residuals
StatsAPI.vcov(model::LPIVModel) = model.vcov
StatsAPI.nobs(model::LPIVModel) = size(model.Y, 1)
StatsAPI.islinear(::LPIVModel) = true

# For SmoothLPModel
StatsAPI.coef(model::SmoothLPModel) = model.theta
StatsAPI.residuals(model::SmoothLPModel) = model.residuals
StatsAPI.vcov(model::SmoothLPModel) = model.vcov_theta
StatsAPI.nobs(model::SmoothLPModel) = size(model.Y, 1)
StatsAPI.islinear(::SmoothLPModel) = true

# For StateLPModel
StatsAPI.residuals(model::StateLPModel) = model.residuals
StatsAPI.residuals(model::StateLPModel, h::Int) = model.residuals[h + 1]
StatsAPI.nobs(model::StateLPModel) = size(model.Y, 1)
StatsAPI.islinear(::StateLPModel) = true

# For PropensityLPModel
StatsAPI.coef(model::PropensityLPModel) = model.B
StatsAPI.residuals(model::PropensityLPModel) = model.residuals
StatsAPI.vcov(model::PropensityLPModel) = model.vcov
StatsAPI.nobs(model::PropensityLPModel) = size(model.Y, 1)
StatsAPI.islinear(::PropensityLPModel) = true

