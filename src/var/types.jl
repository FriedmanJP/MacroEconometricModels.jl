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
Concrete type definitions for VAR models, IRF, FEVD, and priors.
"""

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

function Base.show(io::IO, m::VARModel{T}) where {T}
    n = nvars(m)
    spec = Any[
        "Variables"    n;
        "Lags"         m.p;
        "Observations" size(m.Y, 1);
        "AIC"          _fmt(m.aic; digits=2);
        "BIC"          _fmt(m.bic; digits=2);
        "HQIC"         _fmt(m.hqic; digits=2)
    ]
    _pretty_table(io, spec;
        title = "VAR($(m.p)) Model",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
end

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
