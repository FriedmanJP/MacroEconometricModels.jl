# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
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

# LP-IV Model Type (Stock & Watson 2018)
# =============================================================================

"""
    LPIVModel{T} <: AbstractLPModel

Local Projection with Instrumental Variables (Stock & Watson 2018).
Uses 2SLS estimation at each horizon.

Fields:
- Y: Response data matrix
- shock_var: Index of endogenous shock variable
- response_vars: Indices of response variables
- instruments: Instrument matrix (T × n_instruments)
- horizon: Maximum horizon
- lags: Number of control lags
- B: 2SLS coefficient matrices per horizon
- residuals: Residuals per horizon
- vcov: Robust covariance matrices per horizon
- first_stage_F: First-stage F-statistics per horizon (for weak IV test)
- first_stage_coef: First-stage coefficients per horizon
- T_eff: Effective sample sizes
- cov_estimator: Covariance estimator used
"""
struct LPIVModel{T<:AbstractFloat} <: AbstractLPModel
    Y::Matrix{T}
    shock_var::Int
    response_vars::Vector{Int}
    instruments::Matrix{T}
    horizon::Int
    lags::Int
    B::Vector{Matrix{T}}
    residuals::Vector{Matrix{T}}
    vcov::Vector{Matrix{T}}
    first_stage_F::Vector{T}
    first_stage_coef::Vector{Vector{T}}
    T_eff::Vector{Int}
    cov_estimator::AbstractCovarianceEstimator

    function LPIVModel{T}(Y::Matrix{T}, shock_var::Int, response_vars::Vector{Int},
                          instruments::Matrix{T}, horizon::Int, lags::Int,
                          B::Vector{Matrix{T}}, residuals::Vector{Matrix{T}},
                          vcov::Vector{Matrix{T}}, first_stage_F::Vector{T},
                          first_stage_coef::Vector{Vector{T}}, T_eff::Vector{Int},
                          cov_estimator::AbstractCovarianceEstimator) where {T<:AbstractFloat}
        n = size(Y, 2)
        @assert 1 <= shock_var <= n "shock_var must be in 1:$n"
        @assert all(1 .<= response_vars .<= n) "response_vars must be in 1:$n"
        @assert size(instruments, 1) == size(Y, 1) "instruments must have same T as Y"
        @assert size(instruments, 2) >= 1 "need at least one instrument"
        @assert length(first_stage_F) == horizon + 1
        new{T}(Y, shock_var, response_vars, instruments, horizon, lags, B, residuals,
               vcov, first_stage_F, first_stage_coef, T_eff, cov_estimator)
    end
end

n_instruments(model::LPIVModel) = size(model.instruments, 2)

# =============================================================================
# B-Spline Basis Type
# =============================================================================

"""
    BSplineBasis{T} <: Any

B-spline basis for smooth LP (Barnichon & Brownlees 2019).

Fields:
- degree: Spline degree (typically 3 for cubic)
- n_interior_knots: Number of interior knots
- knots: Full knot vector including boundary knots
- basis_matrix: Precomputed basis matrix at horizon points (H+1 × n_basis)
- horizons: Horizon points where basis is evaluated
"""
struct BSplineBasis{T<:AbstractFloat}
    degree::Int
    n_interior_knots::Int
    knots::Vector{T}
    basis_matrix::Matrix{T}
    horizons::Vector{Int}

    function BSplineBasis{T}(degree::Int, n_interior_knots::Int, knots::Vector{T},
                             basis_matrix::Matrix{T}, horizons::Vector{Int}) where {T<:AbstractFloat}
        @assert degree >= 0 "degree must be non-negative"
        @assert n_interior_knots >= 0 "n_interior_knots must be non-negative"
        n_basis = n_interior_knots + degree + 1
        @assert size(basis_matrix, 2) == n_basis "basis_matrix columns must equal n_basis"
        @assert size(basis_matrix, 1) == length(horizons) "basis_matrix rows must equal length(horizons)"
        new{T}(degree, n_interior_knots, knots, basis_matrix, horizons)
    end
end

n_basis(basis::BSplineBasis) = basis.n_interior_knots + basis.degree + 1

# =============================================================================
# Smooth LP Model Type (Barnichon & Brownlees 2019)
# =============================================================================

"""
    SmoothLPModel{T} <: AbstractLPModel

Smooth Local Projection with B-spline basis (Barnichon & Brownlees 2019).

The IRF is parameterized as: β(h) = Σ_j θ_j B_j(h)
where B_j are B-spline basis functions.

Fields:
- Y: Response data matrix
- shock_var: Shock variable index
- response_vars: Response variable indices
- horizon: Maximum horizon
- lags: Number of control lags
- spline_basis: B-spline basis configuration
- theta: Spline coefficients (n_basis × n_response)
- vcov_theta: Covariance of theta (vectorized)
- lambda: Smoothing penalty parameter
- irf_values: Smoothed IRF point estimates (H+1 × n_response)
- irf_se: Standard errors of smoothed IRF
- residuals: Pooled residuals
- T_eff: Effective sample size
- cov_estimator: Covariance estimator used
"""
struct SmoothLPModel{T<:AbstractFloat} <: AbstractLPModel
    Y::Matrix{T}
    shock_var::Int
    response_vars::Vector{Int}
    horizon::Int
    lags::Int
    spline_basis::BSplineBasis{T}
    theta::Matrix{T}
    vcov_theta::Matrix{T}
    lambda::T
    irf_values::Matrix{T}
    irf_se::Matrix{T}
    residuals::Matrix{T}
    T_eff::Int
    cov_estimator::AbstractCovarianceEstimator

    function SmoothLPModel{T}(Y::Matrix{T}, shock_var::Int, response_vars::Vector{Int},
                              horizon::Int, lags::Int, spline_basis::BSplineBasis{T},
                              theta::Matrix{T}, vcov_theta::Matrix{T}, lambda::T,
                              irf_values::Matrix{T}, irf_se::Matrix{T}, residuals::Matrix{T},
                              T_eff::Int, cov_estimator::AbstractCovarianceEstimator) where {T<:AbstractFloat}
        n = size(Y, 2)
        @assert 1 <= shock_var <= n
        @assert all(1 .<= response_vars .<= n)
        @assert lambda >= 0 "lambda must be non-negative"
        @assert size(theta, 1) == n_basis(spline_basis)
        @assert size(theta, 2) == length(response_vars)
        new{T}(Y, shock_var, response_vars, horizon, lags, spline_basis, theta,
               vcov_theta, lambda, irf_values, irf_se, residuals, T_eff, cov_estimator)
    end
end

# =============================================================================
# State Transition Type
# =============================================================================

"""
    StateTransition{T} <: Any

Smooth state transition function for state-dependent LP.

F(z_t) = exp(-γ(z_t - c)) / (1 + exp(-γ(z_t - c)))

Fields:
- state_var: State variable values (standardized)
- gamma: Transition smoothness parameter (higher = sharper)
- threshold: Transition threshold c
- method: Transition function type (:logistic, :exponential, :indicator)
- F_values: Precomputed transition function values
"""
struct StateTransition{T<:AbstractFloat}
    state_var::Vector{T}
    gamma::T
    threshold::T
    method::Symbol
    F_values::Vector{T}

    function StateTransition(state_var::Vector{T}, gamma::T, threshold::T,
                             method::Symbol=:logistic) where {T<:AbstractFloat}
        @assert gamma > 0 "gamma must be positive"
        method ∉ (:logistic, :exponential, :indicator) &&
            throw(ArgumentError("method must be :logistic, :exponential, or :indicator"))

        # Compute F values
        F_values = if method == :logistic
            @. exp(-gamma * (state_var - threshold)) / (1 + exp(-gamma * (state_var - threshold)))
        elseif method == :exponential
            @. 1 - exp(-gamma * (state_var - threshold)^2)
        else  # :indicator
            T.(state_var .>= threshold)
        end

        new{T}(state_var, gamma, threshold, method, F_values)
    end
end

# =============================================================================
# State-Dependent LP Model Type (Auerbach & Gorodnichenko 2013)
# =============================================================================

"""
    StateLPModel{T} <: AbstractLPModel

State-dependent Local Projection (Auerbach & Gorodnichenko 2013).

Model: y_{t+h} = F(z_t)[α_E + β_E * shock_t + ...] + (1-F(z_t))[α_R + β_R * shock_t + ...]

F(z) is a smooth transition function, typically logistic.
State E = expansion (high z), State R = recession (low z).

Fields:
- Y: Response data matrix
- shock_var: Shock variable index
- response_vars: Response variable indices
- horizon: Maximum horizon
- lags: Number of control lags
- state: StateTransition configuration
- B_expansion: Coefficients in expansion state (per horizon)
- B_recession: Coefficients in recession state (per horizon)
- residuals: Residuals per horizon
- vcov_expansion: Covariance in expansion (per horizon)
- vcov_recession: Covariance in recession (per horizon)
- vcov_diff: Covariance of difference (per horizon)
- T_eff: Effective sample sizes
- cov_estimator: Covariance estimator used
"""
struct StateLPModel{T<:AbstractFloat} <: AbstractLPModel
    Y::Matrix{T}
    shock_var::Int
    response_vars::Vector{Int}
    horizon::Int
    lags::Int
    state::StateTransition{T}
    B_expansion::Vector{Matrix{T}}
    B_recession::Vector{Matrix{T}}
    residuals::Vector{Matrix{T}}
    vcov_expansion::Vector{Matrix{T}}
    vcov_recession::Vector{Matrix{T}}
    vcov_diff::Vector{Matrix{T}}
    T_eff::Vector{Int}
    cov_estimator::AbstractCovarianceEstimator

    function StateLPModel{T}(Y::Matrix{T}, shock_var::Int, response_vars::Vector{Int},
                             horizon::Int, lags::Int, state::StateTransition{T},
                             B_expansion::Vector{Matrix{T}}, B_recession::Vector{Matrix{T}},
                             residuals::Vector{Matrix{T}}, vcov_expansion::Vector{Matrix{T}},
                             vcov_recession::Vector{Matrix{T}}, vcov_diff::Vector{Matrix{T}},
                             T_eff::Vector{Int}, cov_estimator::AbstractCovarianceEstimator) where {T<:AbstractFloat}
        n = size(Y, 2)
        @assert 1 <= shock_var <= n
        @assert all(1 .<= response_vars .<= n)
        @assert length(state.state_var) == size(Y, 1) "state_var must have same length as Y"
        @assert length(B_expansion) == horizon + 1
        @assert length(B_recession) == horizon + 1
        new{T}(Y, shock_var, response_vars, horizon, lags, state, B_expansion,
               B_recession, residuals, vcov_expansion, vcov_recession, vcov_diff,
               T_eff, cov_estimator)
    end
end

# =============================================================================
# Propensity Score Configuration
# =============================================================================

"""
    PropensityScoreConfig{T} <: Any

Configuration for propensity score estimation and IPW.

Fields:
- method: Propensity model (:logit, :probit)
- trimming: (lower, upper) bounds for propensity scores
- normalize: Normalize weights to sum to 1 within groups
"""
struct PropensityScoreConfig{T<:AbstractFloat}
    method::Symbol
    trimming::Tuple{T,T}
    normalize::Bool

    function PropensityScoreConfig{T}(method::Symbol, trimming::Tuple{T,T},
                                       normalize::Bool) where {T<:AbstractFloat}
        method ∉ (:logit, :probit) && throw(ArgumentError("method must be :logit or :probit"))
        @assert 0 <= trimming[1] < trimming[2] <= 1 "trimming must be in [0,1]"
        new{T}(method, trimming, normalize)
    end
end

function PropensityScoreConfig(; method::Symbol=:logit,
                                 trimming::Tuple{<:Real,<:Real}=(0.01, 0.99),
                                 normalize::Bool=true)
    PropensityScoreConfig{Float64}(method, (Float64(trimming[1]), Float64(trimming[2])), normalize)
end

# =============================================================================
# Propensity Score LP Model Type (Angrist et al. 2018)
# =============================================================================

"""
    PropensityLPModel{T} <: AbstractLPModel

Local Projection with Inverse Propensity Weighting (Angrist et al. 2018).

Estimates Average Treatment Effect (ATE) at each horizon using IPW.

Fields:
- Y: Response data matrix
- treatment: Binary treatment indicator vector
- response_vars: Response variable indices
- covariates: Covariate matrix for propensity model
- horizon: Maximum horizon
- propensity_scores: Estimated propensity scores P(D=1|X)
- ipw_weights: Inverse propensity weights
- B: IPW-weighted regression coefficients per horizon
- residuals: Residuals per horizon
- vcov: Robust covariance matrices per horizon
- ate: Average treatment effects per horizon (for each response var)
- ate_se: Standard errors of ATE
- config: Propensity score configuration
- T_eff: Effective sample sizes
- cov_estimator: Covariance estimator used
"""
struct PropensityLPModel{T<:AbstractFloat} <: AbstractLPModel
    Y::Matrix{T}
    treatment::Vector{Bool}
    response_vars::Vector{Int}
    covariates::Matrix{T}
    horizon::Int
    propensity_scores::Vector{T}
    ipw_weights::Vector{T}
    B::Vector{Matrix{T}}
    residuals::Vector{Matrix{T}}
    vcov::Vector{Matrix{T}}
    ate::Matrix{T}  # (H+1) × n_response
    ate_se::Matrix{T}
    config::PropensityScoreConfig{T}
    T_eff::Vector{Int}
    cov_estimator::AbstractCovarianceEstimator

    function PropensityLPModel{T}(Y::Matrix{T}, treatment::AbstractVector{Bool}, response_vars::Vector{Int},
                                  covariates::Matrix{T}, horizon::Int, propensity_scores::Vector{T},
                                  ipw_weights::Vector{T}, B::Vector{Matrix{T}},
                                  residuals::Vector{Matrix{T}}, vcov::Vector{Matrix{T}},
                                  ate::Matrix{T}, ate_se::Matrix{T}, config::PropensityScoreConfig{T},
                                  T_eff::Vector{Int}, cov_estimator::AbstractCovarianceEstimator) where {T<:AbstractFloat}
        n = size(Y, 2)
        @assert length(treatment) == size(Y, 1)
        @assert all(1 .<= response_vars .<= n)
        @assert size(covariates, 1) == size(Y, 1)
        @assert length(propensity_scores) == size(Y, 1)
        @assert all(0 .<= propensity_scores .<= 1)
        @assert size(ate, 1) == horizon + 1
        @assert size(ate, 2) == length(response_vars)
        new{T}(Y, collect(Bool, treatment), response_vars, covariates, horizon, propensity_scores,
               ipw_weights, B, residuals, vcov, ate, ate_se, config, T_eff, cov_estimator)
    end
end

n_treated(model::PropensityLPModel) = sum(model.treatment)
n_control(model::PropensityLPModel) = sum(.!model.treatment)

