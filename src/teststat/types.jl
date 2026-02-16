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
Abstract type and result structs for unit root and stationarity tests.
"""

using StatsAPI

# =============================================================================
# Abstract Type
# =============================================================================

"""Abstract supertype for all unit root test results."""
abstract type AbstractUnitRootTest <: StatsAPI.HypothesisTest end

# =============================================================================
# Result Types
# =============================================================================

"""
    ADFResult{T} <: AbstractUnitRootTest

Augmented Dickey-Fuller test result.

Fields:
- `statistic`: ADF test statistic (t-ratio on γ)
- `pvalue`: Approximate p-value (MacKinnon 1994, 2010)
- `lags`: Number of augmenting lags used
- `regression`: Regression specification (:none, :constant, :trend)
- `critical_values`: Critical values at 1%, 5%, 10% levels
- `nobs`: Effective number of observations
"""
struct ADFResult{T<:AbstractFloat} <: AbstractUnitRootTest
    statistic::T
    pvalue::T
    lags::Int
    regression::Symbol
    critical_values::Dict{Int,T}
    nobs::Int
end

"""
    KPSSResult{T} <: AbstractUnitRootTest

KPSS stationarity test result.

Fields:
- `statistic`: KPSS test statistic
- `pvalue`: Approximate p-value
- `regression`: Regression specification (:constant, :trend)
- `critical_values`: Critical values at 1%, 5%, 10% levels
- `bandwidth`: Bartlett kernel bandwidth used
- `nobs`: Number of observations
"""
struct KPSSResult{T<:AbstractFloat} <: AbstractUnitRootTest
    statistic::T
    pvalue::T
    regression::Symbol
    critical_values::Dict{Int,T}
    bandwidth::Int
    nobs::Int
end

"""
    PPResult{T} <: AbstractUnitRootTest

Phillips-Perron test result.

Fields:
- `statistic`: PP test statistic (Zt or Zα)
- `pvalue`: Approximate p-value
- `regression`: Regression specification (:none, :constant, :trend)
- `critical_values`: Critical values at 1%, 5%, 10% levels
- `bandwidth`: Newey-West bandwidth used
- `nobs`: Effective number of observations
"""
struct PPResult{T<:AbstractFloat} <: AbstractUnitRootTest
    statistic::T
    pvalue::T
    regression::Symbol
    critical_values::Dict{Int,T}
    bandwidth::Int
    nobs::Int
end

"""
    ZAResult{T} <: AbstractUnitRootTest

Zivot-Andrews structural break unit root test result.

Fields:
- `statistic`: Minimum t-statistic across all break points
- `pvalue`: Approximate p-value
- `break_index`: Index of estimated structural break
- `break_fraction`: Break point as fraction of sample
- `regression`: Break specification (:constant, :trend, :both)
- `critical_values`: Critical values at 1%, 5%, 10% levels
- `lags`: Number of augmenting lags
- `nobs`: Effective number of observations
"""
struct ZAResult{T<:AbstractFloat} <: AbstractUnitRootTest
    statistic::T
    pvalue::T
    break_index::Int
    break_fraction::T
    regression::Symbol
    critical_values::Dict{Int,T}
    lags::Int
    nobs::Int
end

"""
    NgPerronResult{T} <: AbstractUnitRootTest

Ng-Perron unit root test result (MZα, MZt, MSB, MPT).

Fields:
- `MZa`: Modified Zα statistic
- `MZt`: Modified Zt statistic
- `MSB`: Modified Sargan-Bhargava statistic
- `MPT`: Modified Point-optimal statistic
- `regression`: Regression specification (:constant, :trend)
- `critical_values`: Dict mapping statistic name to critical values
- `nobs`: Effective number of observations
"""
struct NgPerronResult{T<:AbstractFloat} <: AbstractUnitRootTest
    MZa::T
    MZt::T
    MSB::T
    MPT::T
    regression::Symbol
    critical_values::Dict{Symbol,Dict{Int,T}}
    nobs::Int
end

"""
    JohansenResult{T} <: AbstractUnitRootTest

Johansen cointegration test result.

Fields:
- `trace_stats`: Trace test statistics for each rank
- `trace_pvalues`: P-values for trace tests
- `max_eigen_stats`: Maximum eigenvalue test statistics
- `max_eigen_pvalues`: P-values for max eigenvalue tests
- `rank`: Estimated cointegration rank (at 5% level)
- `eigenvectors`: Cointegrating vectors (β), columns are vectors
- `adjustment`: Adjustment coefficients (α)
- `eigenvalues`: Eigenvalues from reduced rank regression
- `critical_values_trace`: Critical values for trace test (rows: ranks, cols: 10%, 5%, 1%)
- `critical_values_max`: Critical values for max eigenvalue test
- `deterministic`: Deterministic specification
- `lags`: Number of lags in VECM
- `nobs`: Effective number of observations
"""
struct JohansenResult{T<:AbstractFloat} <: AbstractUnitRootTest
    trace_stats::Vector{T}
    trace_pvalues::Vector{T}
    max_eigen_stats::Vector{T}
    max_eigen_pvalues::Vector{T}
    rank::Int
    eigenvectors::Matrix{T}
    adjustment::Matrix{T}
    eigenvalues::Vector{T}
    critical_values_trace::Matrix{T}
    critical_values_max::Matrix{T}
    deterministic::Symbol
    lags::Int
    nobs::Int
end

"""
    VARStationarityResult{T}

VAR model stationarity check result.

Fields:
- `is_stationary`: true if all eigenvalues have modulus < 1
- `eigenvalues`: Eigenvalues of companion matrix (may be real or complex)
- `max_modulus`: Maximum eigenvalue modulus
- `companion_matrix`: The companion form matrix F
"""
struct VARStationarityResult{T<:AbstractFloat, E<:Union{T, Complex{T}}}
    is_stationary::Bool
    eigenvalues::Vector{E}
    max_modulus::T
    companion_matrix::Matrix{T}
end

# =============================================================================
# StatsAPI Interface for Unit Root Tests
# =============================================================================

# Common interface for all unit root tests
StatsAPI.nobs(r::ADFResult) = r.nobs
StatsAPI.nobs(r::KPSSResult) = r.nobs
StatsAPI.nobs(r::PPResult) = r.nobs
StatsAPI.nobs(r::ZAResult) = r.nobs
StatsAPI.nobs(r::NgPerronResult) = r.nobs
StatsAPI.nobs(r::JohansenResult) = r.nobs

StatsAPI.dof(r::ADFResult) = r.lags + (r.regression == :none ? 1 : r.regression == :constant ? 2 : 3)
StatsAPI.dof(r::KPSSResult) = r.regression == :constant ? 1 : 2
StatsAPI.dof(r::PPResult) = r.regression == :none ? 1 : r.regression == :constant ? 2 : 3
StatsAPI.dof(r::ZAResult) = r.lags + (r.regression == :constant ? 4 : r.regression == :trend ? 4 : 5)
StatsAPI.dof(r::NgPerronResult) = r.regression == :constant ? 1 : 2
StatsAPI.dof(r::JohansenResult) = r.lags

# pvalue - already stored in struct
StatsAPI.pvalue(r::ADFResult) = r.pvalue
StatsAPI.pvalue(r::KPSSResult) = r.pvalue
StatsAPI.pvalue(r::PPResult) = r.pvalue
StatsAPI.pvalue(r::ZAResult) = r.pvalue
# For NgPerron, return MZt p-value as primary (most commonly used)
StatsAPI.pvalue(r::NgPerronResult) = _ngperron_pvalue(r.MZt, r.regression, :MZt)
# For Johansen, return minimum trace p-value
StatsAPI.pvalue(r::JohansenResult) = minimum(r.trace_pvalues)
