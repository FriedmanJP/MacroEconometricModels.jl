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
# Structural Break Test Types
# =============================================================================

"""
    AndrewsResult{T} <: AbstractUnitRootTest

Andrews (1993) / Andrews-Ploberger (1994) structural break test result.

Tests for a single unknown structural break point in a linear regression
by computing sup-Wald, exp-Wald, or mean-Wald statistics over a trimmed
range of candidate break dates.

Fields:
- `statistic`: Test statistic (sup-Wald, exp-Wald, or mean-Wald)
- `pvalue`: Approximate p-value from Hansen (1997) critical values
- `break_index`: Index of estimated break date (for sup-Wald)
- `break_fraction`: Break date as fraction of sample
- `test_type`: Type of test (:supwald, :expwald, :meanwald)
- `critical_values`: Critical values at 1%, 5%, 10% levels
- `stat_sequence`: Full sequence of Wald statistics across candidate breaks
- `trimming`: Trimming fraction (e.g., 0.15)
- `nobs`: Number of observations
- `n_params`: Number of parameters tested for instability
"""
struct AndrewsResult{T<:AbstractFloat} <: AbstractUnitRootTest
    statistic::T
    pvalue::T
    break_index::Int
    break_fraction::T
    test_type::Symbol
    critical_values::Dict{Int,T}
    stat_sequence::Vector{T}
    trimming::T
    nobs::Int
    n_params::Int
end

"""
    BaiPerronResult{T} <: AbstractUnitRootTest

Bai-Perron (1998, 2003) multiple structural break test result.

Tests for multiple unknown structural break points using sequential and
simultaneous procedures, with BIC/LWZ information criteria for break
number selection.

Fields:
- `n_breaks`: Estimated number of breaks
- `break_dates`: Estimated break date indices
- `break_cis`: Confidence intervals for break dates as (lower, upper) tuples
- `regime_coefs`: Coefficient estimates for each regime
- `regime_ses`: Standard errors for each regime
- `supf_stats`: sup-F(l) statistics for l = 1, ..., max_breaks
- `supf_pvalues`: P-values for sup-F tests
- `sequential_stats`: Sequential sup-F(l+1|l) statistics
- `sequential_pvalues`: P-values for sequential tests
- `bic_values`: BIC for each number of breaks (0, 1, ..., max_breaks)
- `lwz_values`: LWZ (modified Schwarz) for each number of breaks
- `trimming`: Trimming fraction (e.g., 0.15)
- `nobs`: Number of observations
"""
struct BaiPerronResult{T<:AbstractFloat} <: AbstractUnitRootTest
    n_breaks::Int
    break_dates::Vector{Int}
    break_cis::Vector{Tuple{Int,Int}}
    regime_coefs::Vector{Vector{T}}
    regime_ses::Vector{Vector{T}}
    supf_stats::Vector{T}
    supf_pvalues::Vector{T}
    sequential_stats::Vector{T}
    sequential_pvalues::Vector{T}
    bic_values::Vector{T}
    lwz_values::Vector{T}
    trimming::T
    nobs::Int
end

# =============================================================================
# Panel Unit Root Test Types
# =============================================================================

"""
    PANICResult{T} <: AbstractUnitRootTest

Bai-Ng (2004, 2010) PANIC (Panel Analysis of Nonstationarity in
Idiosyncratic and Common components) test result.

Decomposes panel data into common factors and idiosyncratic components
via principal components, then tests each for unit roots separately.

Fields:
- `factor_adf_stats`: ADF statistics for each estimated common factor
- `factor_adf_pvalues`: P-values for factor ADF tests
- `pooled_statistic`: Pooled test statistic for idiosyncratic components
- `pooled_pvalue`: P-value for pooled test
- `individual_stats`: Individual unit ADF statistics on defactored data
- `individual_pvalues`: Individual unit p-values
- `n_factors`: Number of common factors used
- `method`: Pooling method (:pooled, :modified)
- `nobs`: Time dimension (T)
- `n_units`: Cross-section dimension (N)
"""
struct PANICResult{T<:AbstractFloat} <: AbstractUnitRootTest
    factor_adf_stats::Vector{T}
    factor_adf_pvalues::Vector{T}
    pooled_statistic::T
    pooled_pvalue::T
    individual_stats::Vector{T}
    individual_pvalues::Vector{T}
    n_factors::Int
    method::Symbol
    nobs::Int
    n_units::Int
end

"""
    PesaranCIPSResult{T} <: AbstractUnitRootTest

Pesaran (2007) CIPS (Cross-sectionally Augmented IPS) panel unit root
test result.

Augments individual ADF regressions with cross-section averages to account
for cross-sectional dependence. The CIPS statistic is the average of
individual CADF statistics.

Fields:
- `cips_statistic`: Cross-sectionally augmented IPS statistic (average of CADF)
- `pvalue`: Approximate p-value from Pesaran (2007) critical value tables
- `individual_cadf_stats`: CADF statistics for each cross-section unit
- `critical_values`: Critical values at 1%, 5%, 10% levels
- `lags`: Number of augmenting lags
- `deterministic`: Deterministic specification (:none, :constant, :trend)
- `nobs`: Time dimension (T)
- `n_units`: Cross-section dimension (N)
"""
struct PesaranCIPSResult{T<:AbstractFloat} <: AbstractUnitRootTest
    cips_statistic::T
    pvalue::T
    individual_cadf_stats::Vector{T}
    critical_values::Dict{Int,T}
    lags::Int
    deterministic::Symbol
    nobs::Int
    n_units::Int
end

"""
    MoonPerronResult{T} <: AbstractUnitRootTest

Moon-Perron (2004) panel unit root test result.

Uses factor-adjusted t-statistics for testing the unit root null in
panels with cross-sectional dependence. Reports two test statistics
(t_a and t_b) with different bias corrections.

Fields:
- `t_a_statistic`: First modified t-statistic (t_a^*)
- `t_b_statistic`: Second modified t-statistic (t_b^*)
- `pvalue_a`: P-value for t_a^* (standard normal under H0)
- `pvalue_b`: P-value for t_b^* (standard normal under H0)
- `n_factors`: Number of common factors estimated
- `nobs`: Time dimension (T)
- `n_units`: Cross-section dimension (N)
"""
struct MoonPerronResult{T<:AbstractFloat} <: AbstractUnitRootTest
    t_a_statistic::T
    t_b_statistic::T
    pvalue_a::T
    pvalue_b::T
    n_factors::Int
    nobs::Int
    n_units::Int
end

"""
    FactorBreakResult{T} <: AbstractUnitRootTest

Structural break test for factor models.

Tests for structural instability in factor loadings or the number of
factors, following Breitung-Eickmeier (2011) or Chen-Dolado-Gonzalo (2014).

Fields:
- `statistic`: Test statistic
- `pvalue`: P-value
- `break_date`: Estimated break date index (nothing if no break detected)
- `method`: Test method (:breitung_eickmeier, :chen_dolado_gonzalo)
- `n_factors`: Number of factors in the model
- `nobs`: Time dimension (T)
- `n_vars`: Number of observed variables
"""
struct FactorBreakResult{T<:AbstractFloat} <: AbstractUnitRootTest
    statistic::T
    pvalue::T
    break_date::Union{Int, Nothing}
    method::Symbol
    n_factors::Int
    nobs::Int
    n_vars::Int
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
StatsAPI.nobs(r::AndrewsResult) = r.nobs
StatsAPI.nobs(r::BaiPerronResult) = r.nobs
StatsAPI.nobs(r::PANICResult) = r.nobs
StatsAPI.nobs(r::PesaranCIPSResult) = r.nobs
StatsAPI.nobs(r::MoonPerronResult) = r.nobs
StatsAPI.nobs(r::FactorBreakResult) = r.nobs

StatsAPI.dof(r::ADFResult) = r.lags + (r.regression == :none ? 1 : r.regression == :constant ? 2 : 3)
StatsAPI.dof(r::KPSSResult) = r.regression == :constant ? 1 : 2
StatsAPI.dof(r::PPResult) = r.regression == :none ? 1 : r.regression == :constant ? 2 : 3
StatsAPI.dof(r::ZAResult) = r.lags + (r.regression == :constant ? 4 : r.regression == :trend ? 4 : 5)
StatsAPI.dof(r::NgPerronResult) = r.regression == :constant ? 1 : 2
StatsAPI.dof(r::JohansenResult) = r.lags
StatsAPI.dof(r::AndrewsResult) = r.n_params
StatsAPI.dof(r::BaiPerronResult) = r.n_breaks
StatsAPI.dof(r::PANICResult) = r.n_factors
StatsAPI.dof(r::PesaranCIPSResult) = r.lags + (r.deterministic == :none ? 1 : r.deterministic == :constant ? 2 : 3)
StatsAPI.dof(r::MoonPerronResult) = r.n_factors
StatsAPI.dof(r::FactorBreakResult) = r.n_factors

# pvalue - already stored in struct
StatsAPI.pvalue(r::ADFResult) = r.pvalue
StatsAPI.pvalue(r::KPSSResult) = r.pvalue
StatsAPI.pvalue(r::PPResult) = r.pvalue
StatsAPI.pvalue(r::ZAResult) = r.pvalue
# For NgPerron, return MZt p-value as primary (most commonly used)
StatsAPI.pvalue(r::NgPerronResult) = _ngperron_pvalue(r.MZt, r.regression, :MZt)
# For Johansen, return minimum trace p-value
StatsAPI.pvalue(r::JohansenResult) = minimum(r.trace_pvalues)
# Structural break tests
StatsAPI.pvalue(r::AndrewsResult) = r.pvalue
# For BaiPerron, return minimum of sup-F p-values (most significant break)
StatsAPI.pvalue(r::BaiPerronResult) = isempty(r.supf_pvalues) ? one(eltype(r.supf_stats)) : minimum(r.supf_pvalues)
# Panel unit root tests
StatsAPI.pvalue(r::PANICResult) = r.pooled_pvalue
StatsAPI.pvalue(r::PesaranCIPSResult) = r.pvalue
# For MoonPerron, return t_a p-value as primary
StatsAPI.pvalue(r::MoonPerronResult) = r.pvalue_a
StatsAPI.pvalue(r::FactorBreakResult) = r.pvalue
