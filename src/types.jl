"""
Type hierarchy for MacroEconometricModels.jl - core abstract and concrete types.
"""

using StatsAPI, LinearAlgebra

# =============================================================================
# Abstract Types
# =============================================================================

"""Abstract supertype for Vector Autoregression models."""
abstract type AbstractVARModel <: StatsAPI.RegressionModel end

"""Abstract supertype for impulse response function results."""
abstract type AbstractImpulseResponse end

"""Abstract supertype for forecast error variance decomposition results."""
abstract type AbstractFEVD end

"""Abstract supertype for Bayesian prior specifications."""
abstract type AbstractPrior end

"""Abstract supertype for factor models (static and dynamic)."""
abstract type AbstractFactorModel <: StatsAPI.StatisticalModel end

# =============================================================================
# VAR Models
# =============================================================================

"""
    VARModel{T} <: AbstractVARModel

VAR model estimated via OLS.

Fields: Y (data), p (lags), B (coefficients), U (residuals), Sigma (covariance), aic, bic, hqic.
"""
struct VARModel{T<:AbstractFloat} <: AbstractVARModel
    Y::Matrix{T}
    p::Int
    B::Matrix{T}
    U::Matrix{T}
    Sigma::Matrix{T}
    aic::T
    bic::T
    hqic::T

    function VARModel(Y::Matrix{T}, p::Int, B::Matrix{T}, U::Matrix{T},
                      Sigma::Matrix{T}, aic::T, bic::T, hqic::T) where {T<:AbstractFloat}
        n = size(Y, 2)
        @assert size(B, 1) == 1 + n*p && size(B, 2) == n "B dimensions mismatch"
        @assert size(Sigma) == (n, n) "Sigma must be n × n"
        new{T}(Y, p, B, U, Sigma, aic, bic, hqic)
    end
end

# Convenience constructor with type promotion
function VARModel(Y::AbstractMatrix, p::Int, B::AbstractMatrix, U::AbstractMatrix,
                  Sigma::AbstractMatrix, aic::Real, bic::Real, hqic::Real)
    T = promote_type(eltype(Y), eltype(B), eltype(U), eltype(Sigma), typeof(aic))
    VARModel(Matrix{T}(Y), p, Matrix{T}(B), Matrix{T}(U), Matrix{T}(Sigma),
             T(aic), T(bic), T(hqic))
end

# Accessors
nvars(model::VARModel) = size(model.Y, 2)
nlags(model::VARModel) = model.p
ncoefs(model::VARModel) = 1 + nvars(model) * model.p
effective_nobs(model::VARModel) = size(model.Y, 1) - model.p

# =============================================================================
# Impulse Response Functions
# =============================================================================

"""
    ImpulseResponse{T} <: AbstractImpulseResponse

IRF results with optional confidence intervals.

Fields: values (H×n×n), ci_lower, ci_upper, horizon, variables, shocks, ci_type.
"""
struct ImpulseResponse{T<:AbstractFloat} <: AbstractImpulseResponse
    values::Array{T,3}
    ci_lower::Array{T,3}
    ci_upper::Array{T,3}
    horizon::Int
    variables::Vector{String}
    shocks::Vector{String}
    ci_type::Symbol
end

"""
    BayesianImpulseResponse{T} <: AbstractImpulseResponse

Bayesian IRF with posterior quantiles.

Fields: quantiles (H×n×n×q), mean (H×n×n), horizon, variables, shocks, quantile_levels.
"""
struct BayesianImpulseResponse{T<:AbstractFloat} <: AbstractImpulseResponse
    quantiles::Array{T,4}
    mean::Array{T,3}
    horizon::Int
    variables::Vector{String}
    shocks::Vector{String}
    quantile_levels::Vector{T}
end

# =============================================================================
# FEVD
# =============================================================================

"""FEVD results: decomposition (n×n×H) and proportions."""
struct FEVD{T<:AbstractFloat} <: AbstractFEVD
    decomposition::Array{T,3}
    proportions::Array{T,3}
end

"""Bayesian FEVD with posterior quantiles."""
struct BayesianFEVD{T<:AbstractFloat} <: AbstractFEVD
    quantiles::Array{T,4}
    mean::Array{T,3}
    horizon::Int
    variables::Vector{String}
    shocks::Vector{String}
    quantile_levels::Vector{T}
end

# =============================================================================
# Priors
# =============================================================================

"""
    MinnesotaHyperparameters{T} <: AbstractPrior

Minnesota prior hyperparameters: tau (tightness), decay, lambda (sum-of-coef),
mu (co-persistence), omega (covariance).
"""
struct MinnesotaHyperparameters{T<:AbstractFloat} <: AbstractPrior
    tau::T
    decay::T
    lambda::T
    mu::T
    omega::T
end

function MinnesotaHyperparameters(; tau::Real=3.0, decay::Real=0.5,
                                   lambda::Real=5.0, mu::Real=2.0, omega::Real=2.0)
    T = promote_type(typeof(tau), typeof(decay), typeof(lambda), typeof(mu), typeof(omega))
    MinnesotaHyperparameters{T}(T(tau), T(decay), T(lambda), T(mu), T(omega))
end

# =============================================================================
# Factor Models
# =============================================================================

"""
    FactorModel{T} <: AbstractFactorModel

Static factor model via PCA: Xₜ = Λ Fₜ + eₜ.

Fields: X, factors, loadings, eigenvalues, explained_variance, cumulative_variance, r, standardized.
"""
struct FactorModel{T<:AbstractFloat} <: AbstractFactorModel
    X::Matrix{T}
    factors::Matrix{T}
    loadings::Matrix{T}
    eigenvalues::Vector{T}
    explained_variance::Vector{T}
    cumulative_variance::Vector{T}
    r::Int
    standardized::Bool
end

"""
    DynamicFactorModel{T} <: AbstractFactorModel

Dynamic factor model: Xₜ = Λ Fₜ + eₜ, Fₜ = Σᵢ Aᵢ Fₜ₋ᵢ + ηₜ.

Fields: X, factors, loadings, A (VAR coefficients), factor_residuals, Sigma_eta, Sigma_e,
eigenvalues, explained_variance, cumulative_variance, r, p, method, standardized,
converged, iterations, loglik.
"""
struct DynamicFactorModel{T<:AbstractFloat} <: AbstractFactorModel
    X::Matrix{T}
    factors::Matrix{T}
    loadings::Matrix{T}
    A::Vector{Matrix{T}}
    factor_residuals::Matrix{T}
    Sigma_eta::Matrix{T}
    Sigma_e::Matrix{T}
    eigenvalues::Vector{T}
    explained_variance::Vector{T}
    cumulative_variance::Vector{T}
    r::Int
    p::Int
    method::Symbol
    standardized::Bool
    converged::Bool
    iterations::Int
    loglik::T
end

"""
    GeneralizedDynamicFactorModel{T} <: AbstractFactorModel

GDFM with frequency-dependent loadings: Xₜ = χₜ + ξₜ.

Fields: X, factors, common_component, idiosyncratic, loadings_spectral,
spectral_density_X, spectral_density_chi, eigenvalues_spectral, frequencies,
q (dynamic factors), r (static factors), bandwidth, kernel, standardized, variance_explained.
"""
struct GeneralizedDynamicFactorModel{T<:AbstractFloat} <: AbstractFactorModel
    X::Matrix{T}
    factors::Matrix{T}
    common_component::Matrix{T}
    idiosyncratic::Matrix{T}
    loadings_spectral::Array{Complex{T},3}
    spectral_density_X::Array{Complex{T},3}
    spectral_density_chi::Array{Complex{T},3}
    eigenvalues_spectral::Matrix{T}
    frequencies::Vector{T}
    q::Int
    r::Int
    bandwidth::Int
    kernel::Symbol
    standardized::Bool
    variance_explained::Vector{T}
end
