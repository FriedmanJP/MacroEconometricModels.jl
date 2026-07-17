# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

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

"""
    FourierADFResult{T} <: AbstractUnitRootTest

Fourier ADF unit root test result (Enders & Lee 2012).
"""
struct FourierADFResult{T<:AbstractFloat} <: AbstractUnitRootTest
    statistic::T
    pvalue::T
    frequency::Int
    f_statistic::T
    f_pvalue::T
    lags::Int
    regression::Symbol
    critical_values::Dict{Int,T}
    f_critical_values::Dict{Int,T}
    nobs::Int
end

"""
    FourierKPSSResult{T} <: AbstractUnitRootTest

Fourier KPSS stationarity test result (Becker, Enders & Lee 2006).
"""
struct FourierKPSSResult{T<:AbstractFloat} <: AbstractUnitRootTest
    statistic::T
    pvalue::T
    frequency::Int
    f_statistic::T
    f_pvalue::T
    regression::Symbol
    critical_values::Dict{Int,T}
    f_critical_values::Dict{Int,T}
    bandwidth::Int
    nobs::Int
end

"""
    DFGLSResult{T} <: AbstractUnitRootTest

DF-GLS unit root test result (Elliott, Rothenberg & Stock 1996).
Also reports ERS Pt statistic and Ng-Perron MGLS statistics.
"""
struct DFGLSResult{T<:AbstractFloat} <: AbstractUnitRootTest
    statistic::T
    pvalue::T
    pt_statistic::T
    pt_pvalue::T
    MZa::T
    MZt::T
    MSB::T
    MPT::T
    lags::Int
    regression::Symbol
    critical_values::Dict{Int,T}
    pt_critical_values::Dict{Int,T}
    mgls_critical_values::Dict{Symbol,Dict{Int,T}}
    nobs::Int
end

"""
    LMUnitRootResult{T} <: AbstractUnitRootTest

LM unit root test result (Schmidt-Phillips 1992; Lee-Strazicich 2003, 2013).
"""
struct LMUnitRootResult{T<:AbstractFloat} <: AbstractUnitRootTest
    statistic::T
    pvalue::T
    breaks::Int
    break_dates::Vector{Int}
    break_fractions::Vector{T}
    lags::Int
    regression::Symbol
    critical_values::Dict{Int,T}
    nobs::Int
end

"""
    ADF2BreakResult{T} <: AbstractUnitRootTest

ADF unit root test with two structural breaks (Narayan & Popp 2010).
"""
struct ADF2BreakResult{T<:AbstractFloat} <: AbstractUnitRootTest
    statistic::T
    pvalue::T
    break1::Int
    break2::Int
    break1_fraction::T
    break2_fraction::T
    lags::Int
    model::Symbol
    critical_values::Dict{Int,T}
    nobs::Int
end

"""
    GregoryHansenResult{T} <: AbstractUnitRootTest

Gregory-Hansen cointegration test with structural break (Gregory & Hansen 1996).
"""
struct GregoryHansenResult{T<:AbstractFloat} <: AbstractUnitRootTest
    adf_statistic::T
    adf_pvalue::T
    zt_statistic::T
    zt_pvalue::T
    za_statistic::T
    za_pvalue::T
    adf_break::Int
    zt_break::Int
    za_break::Int
    model::Symbol
    n_regressors::Int
    adf_critical_values::Dict{Int,T}
    za_critical_values::Dict{Int,T}
    nobs::Int
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

# New unit root test types
StatsAPI.nobs(r::FourierADFResult) = r.nobs
StatsAPI.nobs(r::FourierKPSSResult) = r.nobs
StatsAPI.nobs(r::DFGLSResult) = r.nobs
StatsAPI.nobs(r::LMUnitRootResult) = r.nobs
StatsAPI.nobs(r::ADF2BreakResult) = r.nobs
StatsAPI.nobs(r::GregoryHansenResult) = r.nobs

StatsAPI.dof(r::FourierADFResult) = r.lags + 2 + (r.regression == :trend ? 1 : 0)
StatsAPI.dof(r::FourierKPSSResult) = r.regression == :constant ? 3 : 4
StatsAPI.dof(r::DFGLSResult) = r.lags + (r.regression == :constant ? 1 : 2)
StatsAPI.dof(r::LMUnitRootResult) = r.lags + r.breaks * 2
StatsAPI.dof(r::ADF2BreakResult) = r.lags + (r.model == :level ? 4 : 8)
StatsAPI.dof(r::GregoryHansenResult) = r.n_regressors

StatsAPI.pvalue(r::FourierADFResult) = r.pvalue
StatsAPI.pvalue(r::FourierKPSSResult) = r.pvalue
StatsAPI.pvalue(r::DFGLSResult) = r.pvalue
StatsAPI.pvalue(r::LMUnitRootResult) = r.pvalue
StatsAPI.pvalue(r::ADF2BreakResult) = r.pvalue
StatsAPI.pvalue(r::GregoryHansenResult) = r.adf_pvalue

# =============================================================================
# First-Generation Panel Unit Root Tests (EV-20, #428)
# LLC / IPS / Breitung / Fisher / Hadri. Cross-sectional-independence battery
# forming EViews' "Panel Unit Root Test" dialog. Result structs below; the
# estimator functions live in src/teststat/{llc,ips,breitung_panel,fisher_panel,
# hadri}.jl and their show methods in src/teststat/show.jl (append-only blocks).
# LLC/IPS/Breitung/Fisher share the UNIT-ROOT null; Hadri flips it (STATIONARITY
# null) and is RIGHT-tailed.
# =============================================================================

"""
    LLCResult{T} <: AbstractUnitRootTest

Levin-Lin-Chu (2002) pooled panel unit root test result (unit-root null).

The bias-adjusted statistic `t*_δ` is asymptotically N(0,1); very negative
values reject H0 (panel unit root). See [`llc_test`](@ref).

Fields:
- `statistic`: bias-adjusted `t*_δ` (N(0,1) under H0)
- `pvalue`: left-tailed normal p-value
- `t_unadjusted`: unadjusted pooled t-statistic `t_δ`
- `delta`: pooled OLS coefficient `δ̂` (≈ ρ̂ − 1)
- `S_N`: mean long-run/short-run standard-deviation ratio S̄_N
- `mu_star`, `sigma_star`: LLC (2002) Table 2 mean/std adjustments at T̃
- `T_tilde`: adjusted per-unit sample T̃ = T − p̄ − 1
- `lags`: per-unit ADF augmentation lags
- `deterministic`: `:none`, `:constant`, or `:trend`
- `nobs`: time dimension T
- `n_units`: cross-section dimension N
"""
struct LLCResult{T<:AbstractFloat} <: AbstractUnitRootTest
    statistic::T
    pvalue::T
    t_unadjusted::T
    delta::T
    S_N::T
    mu_star::T
    sigma_star::T
    T_tilde::T
    lags::Vector{Int}
    deterministic::Symbol
    nobs::Int
    n_units::Int
end

"""
    IPSResult{T} <: AbstractUnitRootTest

Im-Pesaran-Shin (2003) `W_tbar` panel unit root test result (unit-root null).

Averages the per-unit ADF t-statistics and standardizes with the finite-sample
moments of IPS (2003, Table 3). `W_tbar` is asymptotically N(0,1); very negative
values reject H0. See [`ips_test`](@ref).

Fields:
- `statistic`: standardized `W_tbar` (N(0,1) under H0)
- `pvalue`: left-tailed normal p-value
- `tbar`: mean of the per-unit ADF t-statistics
- `individual_t`: per-unit ADF t-statistics
- `E_mean`, `V_mean`: averaged Table-3 E[t_iT] / Var[t_iT]
- `lags`: per-unit ADF augmentation lags
- `deterministic`: `:constant` or `:trend`
- `nobs`: time dimension T
- `n_units`: cross-section dimension N
"""
struct IPSResult{T<:AbstractFloat} <: AbstractUnitRootTest
    statistic::T
    pvalue::T
    tbar::T
    individual_t::Vector{T}
    E_mean::T
    V_mean::T
    lags::Vector{Int}
    deterministic::Symbol
    nobs::Int
    n_units::Int
end

"""
    BreitungPanelResult{T} <: AbstractUnitRootTest

Breitung (2000) pooled panel unit root test result (unit-root null).

The `λ` statistic uses a bias-free forward-orthogonal-deviations construction, so
it needs no moment table — it is asymptotically N(0,1) under H0, with very
negative values rejecting. Named `breitung_panel_test`/`BreitungPanelResult` to
avoid collision with the unrelated Breitung-Eickmeier factor break test. See
[`breitung_panel_test`](@ref).

Fields:
- `statistic`: Breitung `λ` (N(0,1) under H0)
- `pvalue`: left-tailed normal p-value
- `lags`: prewhitening lag order p
- `deterministic`: `:none`, `:constant`, or `:trend`
- `nobs`: time dimension T
- `n_units`: cross-section dimension N
"""
struct BreitungPanelResult{T<:AbstractFloat} <: AbstractUnitRootTest
    statistic::T
    pvalue::T
    lags::Int
    deterministic::Symbol
    nobs::Int
    n_units::Int
end

"""
    FisherPanelResult{T} <: AbstractUnitRootTest

Fisher-type (Maddala-Wu 1999 / Choi 2001) combination panel unit root test
result (unit-root null). Combines per-unit ADF/PP p-values four ways.

Fields:
- `statistic`, `pvalue`: the combination selected by `combine` (primary)
- `P`, `P_pvalue`: Maddala-Wu `P = −2Σln p_i ~ χ²(2N)` (upper-tailed)
- `Z`, `Z_pvalue`: Choi inverse-normal `Z = N^{-1/2}Σ Φ⁻¹(p_i) ~ N(0,1)`
- `Lstar`, `Lstar_pvalue`: Choi logit `L* ~ t(5N+4)`
- `Pm`, `Pm_pvalue`: Choi modified `Pm ~ N(0,1)` (upper-tailed)
- `individual_pvalues`: per-unit p-values
- `base`: `:adf` or `:pp`
- `combine`: `:mw`, `:choi`, `:logit`, or `:pm`
- `nobs`: time dimension T
- `n_units`: cross-section dimension N
"""
struct FisherPanelResult{T<:AbstractFloat} <: AbstractUnitRootTest
    statistic::T
    pvalue::T
    P::T
    P_pvalue::T
    Z::T
    Z_pvalue::T
    Lstar::T
    Lstar_pvalue::T
    Pm::T
    Pm_pvalue::T
    individual_pvalues::Vector{T}
    base::Symbol
    combine::Symbol
    nobs::Int
    n_units::Int
end

"""
    HadriResult{T} <: AbstractUnitRootTest

Hadri (2000) LM panel STATIONARITY test result. Unlike the other four
first-generation tests, H0 is *stationarity* (trend/level stationary) and the
statistic is RIGHT-tailed: very positive `Z` rejects. See [`hadri_test`](@ref).

Fields:
- `statistic`: standardized `Z = √N(LM̄ − ξ)/ζ` (N(0,1) under H0)
- `pvalue`: right-tailed normal p-value
- `LM`: mean per-unit LM statistic LM̄
- `xi`, `zeta`: Hadri (2000) standardization constants (ξ, ζ = √ζ²)
- `hetero`: whether per-unit (heteroskedastic) σ̂² was used
- `deterministic`: `:constant` or `:trend`
- `nobs`: time dimension T
- `n_units`: cross-section dimension N
"""
struct HadriResult{T<:AbstractFloat} <: AbstractUnitRootTest
    statistic::T
    pvalue::T
    LM::T
    xi::T
    zeta::T
    hetero::Bool
    deterministic::Symbol
    nobs::Int
    n_units::Int
end

# --- StatsAPI interface (EV-20) ---
StatsAPI.nobs(r::LLCResult) = r.nobs
StatsAPI.nobs(r::IPSResult) = r.nobs
StatsAPI.nobs(r::BreitungPanelResult) = r.nobs
StatsAPI.nobs(r::FisherPanelResult) = r.nobs
StatsAPI.nobs(r::HadriResult) = r.nobs

StatsAPI.dof(r::LLCResult) = r.n_units
StatsAPI.dof(r::IPSResult) = r.n_units
StatsAPI.dof(r::BreitungPanelResult) = r.n_units
StatsAPI.dof(r::FisherPanelResult) = 2 * r.n_units
StatsAPI.dof(r::HadriResult) = r.n_units

StatsAPI.pvalue(r::LLCResult) = r.pvalue
StatsAPI.pvalue(r::IPSResult) = r.pvalue
StatsAPI.pvalue(r::BreitungPanelResult) = r.pvalue
StatsAPI.pvalue(r::FisherPanelResult) = r.pvalue
StatsAPI.pvalue(r::HadriResult) = r.pvalue

# =============================================================================
# Panel Cointegration Tests (EV-21, #429)
# Pedroni / Kao / Westerlund / Fisher-Johansen. All H0: no cointegration.
# =============================================================================

"""
    PedroniResult{T} <: AbstractUnitRootTest

Pedroni (1999, 2004) residual-based panel cointegration test result. Seven
statistics; **`panel-v` (index 1) is right-tailed**, the other six are
left-tailed. See [`pedroni_test`](@ref).

Fields:
- `names`: the seven statistic names, in order
- `raw`: raw (empirical) statistics
- `statistics`: standardized `(raw − μ√N)/√v` (N(0,1) under H0)
- `pvalues`: per-statistic p-values (right-tailed for `panel-v`, else left)
- `mu`, `v`: Pedroni (1999) Table 2 (μ, v) moments for this (trend, k)
- `trend`: `:none`, `:constant`, or `:trend`
- `n_regressors`: number of regressors k
- `bandwidth`: Newey-West bandwidth used for the residual LRVs
- `adf_lags`: augmentation order of the parametric statistics
- `nobs`: time dimension T
- `n_units`: cross-section dimension N
"""
struct PedroniResult{T<:AbstractFloat} <: AbstractUnitRootTest
    names::Vector{String}
    raw::Vector{T}
    statistics::Vector{T}
    pvalues::Vector{T}
    mu::Vector{T}
    v::Vector{T}
    trend::Symbol
    n_regressors::Int
    bandwidth::Int
    adf_lags::Int
    nobs::Int
    n_units::Int
end

"""
    KaoResult{T} <: AbstractUnitRootTest

Kao (1999) residual-based panel cointegration test result (homogeneous
cointegrating vector). Five DF-type statistics, all N(0,1) and left-tailed. See
[`kao_test`](@ref).

Fields:
- `names`: statistic names (`DFrho`, `DFt`, `DFrho_star`, `DFt_star`, `ADF`)
- `statistics`: standardized statistics (N(0,1) under H0)
- `pvalues`: left-tailed normal p-values
- `rho`: pooled AR(1) coefficient ρ̂ of the residuals
- `t_rho`: DF t-statistic for H0: ρ = 1
- `t_adf`: ADF t-statistic for H0: ρ = 1
- `sigma_v2`, `omega_v2`: short-run σ̂²_ν and long-run ω̂²_ν conditional variances
- `lags`: ADF lag order p
- `kernel_lags`: Bartlett bandwidth for ω̂²_ν
- `n_regressors`: number of regressors k
- `nobs`: time dimension T
- `n_units`: cross-section dimension N
"""
struct KaoResult{T<:AbstractFloat} <: AbstractUnitRootTest
    names::Vector{String}
    statistics::Vector{T}
    pvalues::Vector{T}
    rho::T
    t_rho::T
    t_adf::T
    sigma_v2::T
    omega_v2::T
    lags::Int
    kernel_lags::Int
    n_regressors::Int
    nobs::Int
    n_units::Int
end

"""
    WesterlundResult{T} <: AbstractUnitRootTest

Westerlund (2007) ECM panel cointegration test result. Four statistics (`Gt`,
`Ga`, `Pt`, `Pa`), all N(0,1) and left-tailed. See [`westerlund_test`](@ref).

Fields:
- `names`: `["Gt", "Ga", "Pt", "Pa"]`
- `raw`: raw statistics
- `statistics`: standardized Z-scores (N(0,1) under H0)
- `pvalues`: asymptotic left-tailed p-values
- `bootstrap_pvalues`: seeded-bootstrap p-values (`NaN` if `bootstrap == 0`)
- `trend`: `:none`, `:constant`, or `:trend`
- `n_regressors`: number of regressors k
- `lags`, `leads`: short-run lag/lead orders p, q
- `lrwindow`: Bartlett window for the long-run variances
- `bootstrap`: number of bootstrap replications (0 ⇒ none)
- `seed`: bootstrap RNG seed
- `nobs`: time dimension T
- `n_units`: cross-section dimension N
"""
struct WesterlundResult{T<:AbstractFloat} <: AbstractUnitRootTest
    names::Vector{String}
    raw::Vector{T}
    statistics::Vector{T}
    pvalues::Vector{T}
    bootstrap_pvalues::Vector{T}
    trend::Symbol
    n_regressors::Int
    lags::Int
    leads::Int
    lrwindow::Int
    bootstrap::Int
    seed::Int
    nobs::Int
    n_units::Int
end

"""
    FisherJohansenResult{T} <: AbstractUnitRootTest

Fisher-type (Maddala-Wu / Choi) combination of per-unit Johansen cointegration
tests. Combined trace and max-eigenvalue statistics per rank hypothesis. See
[`fisher_johansen_test`](@ref).

Fields:
- `ranks`: rank hypotheses tested (`0, 1, …, n-1`)
- `trace_statistics`, `trace_pvalues`: combined trace statistic and p-value per rank
- `max_statistics`, `max_pvalues`: combined max-eigenvalue statistic and p-value per rank
- `individual_trace_pvalues`, `individual_max_pvalues`: N×n per-unit p-values
- `combine`: `:mw` (χ²) or `:choi` (Z)
- `deterministic`: passed to the per-unit Johansen tests
- `lags`: per-unit VAR lag order
- `rank`: estimated cointegration rank (first non-rejected combined trace test)
- `n_units`: cross-section dimension N
- `n_series`: number of series n
"""
struct FisherJohansenResult{T<:AbstractFloat} <: AbstractUnitRootTest
    ranks::Vector{Int}
    trace_statistics::Vector{T}
    trace_pvalues::Vector{T}
    max_statistics::Vector{T}
    max_pvalues::Vector{T}
    individual_trace_pvalues::Matrix{T}
    individual_max_pvalues::Matrix{T}
    combine::Symbol
    deterministic::Symbol
    lags::Int
    rank::Int
    n_units::Int
    n_series::Int
end

# =============================================================================
# Dumitrescu-Hurlin panel Granger non-causality test (EV-24, #432)
# =============================================================================

"""
    DumitrescuHurlinResult{T} <: AbstractUnitRootTest

Result from the Dumitrescu-Hurlin (2012) heterogeneous-panel Granger
non-causality test. See [`dh_causality_test`](@ref).

Fields:
- `Wbar`: average individual Wald statistic `W̄ = N⁻¹ Σ W_i` (χ²(p) convention)
- `Zbar`, `Zbar_pvalue`: asymptotic standardized statistic `Z̄` (right-tailed) and p-value
- `Ztilde`, `Ztilde_pvalue`: small-`T` standardized statistic `Z̃` (right-tailed) and p-value
- `W_i`: per-unit Wald statistics (retained units only)
- `p`: lag order
- `N`: number of units retained (satisfying `T_i > 2p+5`)
- `nobs`: representative effective regression sample (mean `T_i`)
- `n_skipped`: units dropped for insufficient observations
- `bootstrap`: number of block-bootstrap replications (0 if none)
- `seed`: bootstrap RNG seed
- `bootstrap_pvalue`: CSD-robust bootstrap p-value on `Z̄` (`NaN` if `bootstrap == 0`)
- `cause`, `effect`: variable names (`x` Granger-causes `y`?)
"""
struct DumitrescuHurlinResult{T<:AbstractFloat} <: AbstractUnitRootTest
    Wbar::T
    Zbar::T
    Zbar_pvalue::T
    Ztilde::T
    Ztilde_pvalue::T
    W_i::Vector{T}
    p::Int
    N::Int
    nobs::Int
    n_skipped::Int
    bootstrap::Int
    seed::Int
    bootstrap_pvalue::T
    cause::Symbol
    effect::Symbol
end

# --- StatsAPI interface (EV-21) ---
StatsAPI.nobs(r::PedroniResult) = r.nobs
StatsAPI.nobs(r::KaoResult) = r.nobs
StatsAPI.nobs(r::WesterlundResult) = r.nobs
StatsAPI.dof(r::PedroniResult) = r.n_units
StatsAPI.dof(r::KaoResult) = r.n_units
StatsAPI.dof(r::WesterlundResult) = r.n_units
StatsAPI.dof(r::FisherJohansenResult) = 2 * r.n_units
# Primary p-value: the most significant (smallest) across the reported statistics.
StatsAPI.pvalue(r::PedroniResult) = minimum(r.pvalues)
StatsAPI.pvalue(r::KaoResult) = minimum(r.pvalues)
StatsAPI.pvalue(r::WesterlundResult) = minimum(r.pvalues)
StatsAPI.pvalue(r::FisherJohansenResult) = r.trace_pvalues[1]

# =============================================================================
# Residual-based / parameter-stability cointegration tests (EV-11)
# =============================================================================

"""
    EngleGrangerResult{T} <: AbstractUnitRootTest

Engle–Granger (1987) two-step residual-based cointegration test. `H₀`: no cointegration.

# Fields
- `statistic::T` — residual ADF `t`-statistic (`t` on `ρ` in `Δû_t = ρû_{t-1} + …`)
- `pvalue::T` — MacKinnon (1996/2010) cointegration-surface asymptotic p-value (`N = k+1`)
- `lags::Int` — augmenting lags used (or selected by IC)
- `regression::Symbol` — deterministic case (`:none`/`:constant`/`:trend`)
- `k::Int` — number of `I(1)` regressors
- `N::Int` — number of `I(1)` series (`= k+1`)
- `nobs::Int` — number of level observations
"""
struct EngleGrangerResult{T<:AbstractFloat} <: AbstractUnitRootTest
    statistic::T
    pvalue::T
    lags::Int
    regression::Symbol
    k::Int
    N::Int
    nobs::Int
end

"""
    PhillipsOuliarisResult{T} <: AbstractUnitRootTest

Phillips–Ouliaris (1990) residual-based cointegration test. `H₀`: no cointegration.
Reports the `t`-ratio `Ẑ_t` (`statistic`/`pvalue`) and the normalized-bias `Ẑ_α`
(`z_alpha`/`z_alpha_pvalue`).

# Fields
- `statistic::T` — `Ẑ_t` (primary; studentized PP-style statistic)
- `pvalue::T` — `Ẑ_t` MacKinnon cointegration-surface p-value (`N = k+1`)
- `z_alpha::T` — `Ẑ_α` normalized-bias statistic
- `z_alpha_pvalue::T` — `Ẑ_α` p-value (Monte-Carlo `PO_ZA_CV` bracketing interpolation)
- `regression::Symbol` — deterministic case
- `kernel::Symbol` — HAC kernel used for the residual long-run variance
- `bandwidth::T` — resolved truncation-lag bandwidth
- `k::Int`, `N::Int`, `nobs::Int`
"""
struct PhillipsOuliarisResult{T<:AbstractFloat} <: AbstractUnitRootTest
    statistic::T
    pvalue::T
    z_alpha::T
    z_alpha_pvalue::T
    regression::Symbol
    kernel::Symbol
    bandwidth::T
    k::Int
    N::Int
    nobs::Int
end

"""
    HansenInstabilityResult{T} <: AbstractUnitRootTest

Hansen (1992) `L_c` parameter-instability test for a cointegrating regression.
`H₀`: cointegration with stable coefficients (large `L_c` ⇒ reject).

# Fields
- `statistic::T` — `L_c` statistic
- `pvalue::T` — bracketing p-value from Monte-Carlo `HANSEN_LC_CV`
- `regression::Symbol` — deterministic case
- `trend::Symbol` — the underlying `CointRegModel` trend (`:none`/`:const`/`:linear`)
- `nparam::Int` — number of regression parameters `p = d+k`
- `k::Int` — number of `I(1)` regressors
- `nobs::Int`
"""
struct HansenInstabilityResult{T<:AbstractFloat} <: AbstractUnitRootTest
    statistic::T
    pvalue::T
    regression::Symbol
    trend::Symbol
    nparam::Int
    k::Int
    nobs::Int
end

"""
    ParkAddedResult{T} <: AbstractUnitRootTest

Park (1990) `H(p, q)` added-superfluous-trends test. `H₀`: genuine cointegration
(`I(0)` errors); large `H` ⇒ reject in favour of a spurious regression.

# Fields
- `statistic::T` — `H(p, q)` Wald statistic (`~ χ²(q_add)` under `H₀`)
- `pvalue::T` — asymptotic `χ²(q_add)` upper-tail p-value
- `q_add::Int` — number of superfluous trends added (degrees of freedom)
- `base_order::Int` — highest deterministic-trend order already present (`p`)
- `regression::Symbol` — deterministic case
- `trend::Symbol` — underlying `CointRegModel` trend
- `k::Int`, `nobs::Int`
"""
struct ParkAddedResult{T<:AbstractFloat} <: AbstractUnitRootTest
    statistic::T
    pvalue::T
    q_add::Int
    base_order::Int
    regression::Symbol
    trend::Symbol
    k::Int
    nobs::Int
end

# --- StatsAPI interface (EV-11) ---
StatsAPI.nobs(r::EngleGrangerResult) = r.nobs
StatsAPI.nobs(r::PhillipsOuliarisResult) = r.nobs
StatsAPI.nobs(r::HansenInstabilityResult) = r.nobs
StatsAPI.nobs(r::ParkAddedResult) = r.nobs
StatsAPI.pvalue(r::EngleGrangerResult) = r.pvalue
StatsAPI.pvalue(r::PhillipsOuliarisResult) = r.pvalue
StatsAPI.pvalue(r::HansenInstabilityResult) = r.pvalue
StatsAPI.pvalue(r::ParkAddedResult) = r.pvalue
# --- StatsAPI interface (EV-24) ---
StatsAPI.nobs(r::DumitrescuHurlinResult) = r.nobs
StatsAPI.dof(r::DumitrescuHurlinResult) = r.N
# Primary p-value: the small-T Z̃ (recommended for finite panels).
StatsAPI.pvalue(r::DumitrescuHurlinResult) = r.Ztilde_pvalue

# =============================================================================
# EDF goodness-of-fit battery (EV-26, #434)
# KS / Lilliefors / Cramér–von Mises / Anderson–Darling / Watson against a
# specified or ML-estimated continuous distribution. Right-tailed: large
# statistic ⇒ reject the null of good fit. See [`edf_test`](@ref).
# =============================================================================

"""
    EDFTestResult{T} <: AbstractUnitRootTest

Result from the empirical-distribution-function goodness-of-fit battery
[`edf_test`](@ref). `H₀`: the data follow `dist`; `H₁`: they do not.

Fields:
- `test`: EDF statistic — `:ks`, `:lilliefors`, `:cvm`, `:ad`, `:watson`
- `dist`: null family (`:normal`, `:exponential`, `:logistic`, `:gumbel`,
  `:gamma`, `:weibull`, `:chisq`)
- `params`: `:estimate` (ML-fit θ) or `:specified`
- `statistic`: the value compared to the critical values (modified statistic for
  the estimated-normal AD/CvM/Watson routes; raw EDF statistic otherwise)
- `raw_statistic`: the unmodified EDF statistic
- `pvalue`: p-value (`NaN` when no published null table exists for the requested
  estimated-parameter family)
- `nobs`: number of observations
- `theta`: fitted or specified distribution parameters
- `critical_values`: `Dict` of `1/5/10 (%) => critical value` (empty when
  unavailable)
- `case`: human-readable label of the null-distribution case used
"""
struct EDFTestResult{T<:AbstractFloat} <: AbstractUnitRootTest
    test::Symbol
    dist::Symbol
    params::Symbol
    statistic::T
    raw_statistic::T
    pvalue::T
    nobs::Int
    theta::Vector{T}
    critical_values::Dict{Int,T}
    case::String
end

# --- StatsAPI interface (EV-26) ---
StatsAPI.nobs(r::EDFTestResult) = r.nobs
StatsAPI.pvalue(r::EDFTestResult) = r.pvalue

# =============================================================================
# Variance-ratio / random-walk tests (EV-27, #435)
# =============================================================================

"""
    VarianceRatioResult{T} <: AbstractUnitRootTest

Result from [`variance_ratio_test`](@ref): Lo–MacKinlay (1988) overlapping variance
ratios with Chow–Denning (1993) joint testing, optional Wright (2000) rank/sign
statistics and Kim (2006) wild-bootstrap p-values. `H₀`: the level series is a
random walk (`VR(q) = 1` ∀ `q`).

# Fields
- `q::Vector{Int}` — aggregation values tested
- `vr::Vector{T}` — `VR(q)` (one per `q`)
- `z::Vector{T}` — homoskedastic `Z(q)` statistics (asymptotically `N(0,1)`)
- `z_star::Vector{T}` — heteroskedasticity-robust `Z*(q)` statistics
- `z_pvalue`, `z_star_pvalue::Vector{T}` — per-`q` two-sided asymptotic p-values
- `cd_stat`, `cd_pvalue::T` — Chow–Denning `max_q |Z(q)|` and its SMM-complement p-value
- `cd_star_stat`, `cd_star_pvalue::T` — robust Chow–Denning `max_q |Z*(q)|` + p-value
- `method::Symbol` — `:lomackinlay` or `:wright`
- `robust::Bool` — whether the robust branch is the reported primary
- `wright::Bool` — whether Wright statistics were computed
- `R1`, `R2`, `S1::Vector{T}` — Wright rank (`R1`/`R2`) and sign (`S1`) statistics (empty unless `wright`)
- `R1_pvalue`, `R2_pvalue`, `S1_pvalue::Vector{T}` — Wright simulated-null two-sided p-values
- `bootstrap::Int` — number of Kim (2006) wild-bootstrap replications (`0` = none)
- `boot_weights::Symbol` — `:rademacher` or `:normal`
- `seed::Int` — wild-bootstrap RNG seed
- `z_star_boot_pvalue::Vector{T}` — per-`q` wild-bootstrap p-values (empty if none)
- `cd_boot_pvalue::T` — Chow–Denning wild-bootstrap p-value (`NaN` if none)
- `nobs::Int` — number of level observations
"""
struct VarianceRatioResult{T<:AbstractFloat} <: AbstractUnitRootTest
    q::Vector{Int}
    vr::Vector{T}
    z::Vector{T}
    z_star::Vector{T}
    z_pvalue::Vector{T}
    z_star_pvalue::Vector{T}
    cd_stat::T
    cd_pvalue::T
    cd_star_stat::T
    cd_star_pvalue::T
    method::Symbol
    robust::Bool
    wright::Bool
    R1::Vector{T}
    R2::Vector{T}
    S1::Vector{T}
    R1_pvalue::Vector{T}
    R2_pvalue::Vector{T}
    S1_pvalue::Vector{T}
    bootstrap::Int
    boot_weights::Symbol
    seed::Int
    z_star_boot_pvalue::Vector{T}
    cd_boot_pvalue::T
    nobs::Int
end

# --- StatsAPI interface (EV-27) ---
StatsAPI.nobs(r::VarianceRatioResult) = r.nobs
StatsAPI.dof(r::VarianceRatioResult) = length(r.q)
# Primary p-value: the reported Chow–Denning joint test (robust unless robust=false).
StatsAPI.pvalue(r::VarianceRatioResult) = r.robust ? r.cd_star_pvalue : r.cd_pvalue

# =============================================================================
# BDS independence test (Brock-Dechert-Scheinkman-LeBaron 1996) — EV-28 (#436)
# =============================================================================

"""
    BDSResult{T} <: AbstractUnitRootTest

Result of the Brock-Dechert-Scheinkman-LeBaron (BDS) test for iid/independence.
The statistic `w_m = √T (C_m(ε) − C_1(ε)^m) / σ_m(ε) → N(0,1)` is reported for
each embedding dimension `m` and distance threshold `ε` (a table with one row per
`(m, ε)` pair). Two-sided p-values (departures from iid in either direction reject).

Fields:
- `m::Vector{Int}` — embedding dimensions tested (rows of the statistic table).
- `eps::Vector{T}` — distance thresholds `ε` (columns; `ε = eps_frac · sd`).
- `eps_frac::Vector{T}` — the `ε`-multipliers of the sample sd.
- `sd::T` — sample standard deviation of the input series.
- `statistic::Matrix{T}` — `w_m` statistics, `length(m) × length(eps)`.
- `pvalue::Matrix{T}` — asymptotic two-sided N(0,1) p-values, same shape.
- `boot_pvalue::Matrix{T}` — permutation-bootstrap p-values (`NaN` if `bootstrap==0`).
- `C::Matrix{T}` — correlation integrals `C_m(ε)`, same shape (diagnostic).
- `nobs::Int` — series length `T`.
- `small_sample::Bool` — `true` when `T < 200` (asymptotic distribution unreliable).
- `bootstrap::Int` — number of permutation replications (0 = none).
- `seed::Int` — RNG seed used for the bootstrap.
"""
struct BDSResult{T<:AbstractFloat} <: AbstractUnitRootTest
    m::Vector{Int}
    eps::Vector{T}
    eps_frac::Vector{T}
    sd::T
    statistic::Matrix{T}
    pvalue::Matrix{T}
    boot_pvalue::Matrix{T}
    C::Matrix{T}
    nobs::Int
    small_sample::Bool
    bootstrap::Int
    seed::Int
end

# --- StatsAPI interface (EV-28) ---
StatsAPI.nobs(r::BDSResult) = r.nobs
# Primary p-value: the most significant cell of the (m, ε) table.
StatsAPI.pvalue(r::BDSResult) = minimum(r.pvalue)
# Equality-of-distribution + rank-correlation "Basic statistics" battery
# (EV-34, #442). Both result types subtype StatsAPI.HypothesisTest directly.
# =============================================================================

"""
    EqualityTestResult{T} <: StatsAPI.HypothesisTest

Result of a grouped equality-of-distribution test (`equality_test`, `ttest`,
`anova_test`).

Fields:
- `test_name::Symbol` — e.g. `:two_sample_t`, `:welch_t`, `:anova`, `:welch_anova`,
  `:mann_whitney`, `:wilcoxon_signed_rank`, `:kruskal_wallis`, `:van_der_waerden`,
  `:median_chisq`, `:variance_f`, `:bartlett`, `:levene`, `:brown_forsythe`,
  `:siegel_tukey`
- `statistic::T` — the reported statistic (t / F / U / V / H / χ² / rank sum)
- `pvalue::T` — two-sided p-value
- `df1::T` — degrees of freedom (numerator df for F; χ²/t df; fractional for Welch)
- `df2::T` — denominator df for F tests, `NaN` otherwise
- `n_groups::Int` — number of groups
- `group_sizes::Vector{Int}` — per-group sample sizes
- `exact::Bool` — whether the exact null (vs an asymptotic approximation) was used
- `detail::String` — method annotation (pooled/Welch, tie-corrected, …)
"""
struct EqualityTestResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest
    test_name::Symbol
    statistic::T
    pvalue::T
    df1::T
    df2::T
    n_groups::Int
    group_sizes::Vector{Int}
    exact::Bool
    detail::String
end

"""
    CorTestResult{T} <: StatsAPI.HypothesisTest

Result of a rank/association test (`cor_test`).

Fields:
- `method::Symbol` — `:pearson`, `:spearman`, or `:kendall`
- `estimate::T` — the coefficient (`r`, `ρ`, or `τ_a`/`τ_b`)
- `statistic::T` — test statistic (`t`, `S`, `z`, or concordant count `T`)
- `pvalue::T` — two-sided p-value
- `n::Int` — number of paired observations
- `df::T` — `n − 2` for Pearson/Spearman, `NaN` for Kendall
- `ci_lower::T`, `ci_upper::T` — Fisher-z CI (Pearson only; `NaN` otherwise)
- `exact::Bool` — whether the exact null was used
- `detail::String` — method annotation
"""
struct CorTestResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest
    method::Symbol
    estimate::T
    statistic::T
    pvalue::T
    n::Int
    df::T
    ci_lower::T
    ci_upper::T
    exact::Bool
    detail::String
end

# --- StatsAPI interface (EV-34) ---
StatsAPI.nobs(r::EqualityTestResult) = sum(r.group_sizes)
StatsAPI.pvalue(r::EqualityTestResult) = r.pvalue
StatsAPI.dof(r::EqualityTestResult) = r.df1
StatsAPI.nobs(r::CorTestResult) = r.n
StatsAPI.pvalue(r::CorTestResult) = r.pvalue

# =============================================================================
# Explosive / rational-bubble detection (EV-30, #438)
# SADF (Phillips-Wu-Yu 2011) / GSADF (Phillips-Shi-Yu 2015). RIGHT-TAILED:
# rejection of the unit-root null (in favour of a mildly explosive root) is
# `statistic > critical_value`, using UPPER null quantiles. Estimator functions
# live in src/teststat/bubble.jl; show in src/teststat/show.jl; plot in
# src/plotting/teststat.jl.
# =============================================================================

"""
    BubbleResult{T} <: AbstractUnitRootTest

Result of a sup-ADF explosive-behaviour test ([`sadf_test`](@ref) /
[`gsadf_test`](@ref)). **Right-tailed**: reject the unit-root null for a large
statistic using upper simulated critical values.

Fields:
- `kind::Symbol` — `:sadf` (fixed-start) or `:gsadf` (double-sup)
- `statistic::T` — the sup statistic (SADF or GSADF)
- `pvalue::T` — right-tailed MC p-value (fraction of null draws ≥ statistic)
- `critical_values::Dict{Int,T}` — simulated upper CVs at percent keys `10/5/1`
- `bsadf::Vector{T}` — the r₂-indexed date-stamping sequence (backward sup-ADF
  for `:gsadf`; fixed-start recursive ADF for `:sadf`)
- `cv_seq::Vector{T}` — per-r₂ 95% critical-value sequence (same length as `bsadf`)
- `r2_index::Vector{Int}` — the level index of `y` for each sequence position
- `episodes::Vector{Tuple{Int,Int}}` — stamped bubble `(start, end)` index pairs
- `r0::T` — minimum window fraction actually used
- `adflag::Int` — augmenting-lag order of the window ADF regressions
- `cv_method::Symbol` — `:asymptotic` or `:wildboot`
- `mc_reps::Int` — null Monte-Carlo replications
- `nobs::Int` — sample length T
"""
struct BubbleResult{T<:AbstractFloat} <: AbstractUnitRootTest
    kind::Symbol
    statistic::T
    pvalue::T
    critical_values::Dict{Int,T}
    bsadf::Vector{T}
    cv_seq::Vector{T}
    r2_index::Vector{Int}
    episodes::Vector{Tuple{Int,Int}}
    r0::T
    adflag::Int
    cv_method::Symbol
    mc_reps::Int
    nobs::Int
end

# --- StatsAPI interface (EV-30) ---
StatsAPI.nobs(r::BubbleResult) = r.nobs
StatsAPI.pvalue(r::BubbleResult) = r.pvalue
StatsAPI.dof(r::BubbleResult) = r.adflag + 2
# Seasonal unit roots (HEGY) + ERS point-optimal test (EV-29, #437)
# HEGYResult renders a frequency-by-frequency table (zero / Nyquist / harmonic
# pairs). ERSResult is the standalone point-optimal P_T (the same statistic that
# also lives inside DFGLSResult.pt_statistic). Estimators in teststat/hegy.jl and
# teststat/dfgls.jl; show methods in teststat/show.jl.
# =============================================================================

"""
    HEGYResult{T} <: AbstractUnitRootTest

Hylleberg-Engle-Granger-Yoo (1990, quarterly) / Beaulieu-Miron (1993, monthly)
seasonal unit-root test result. H₀ at each frequency: a unit root is present.

Fields:
- `frequency::Int` — data periodicity (`4` quarterly, `12` monthly)
- `deterministic::Symbol` — deterministic case (e.g. `:const_trend_seas`)
- `lags::Int` — number of augmenting lags of Δ_s y
- `pi_coefs::Vector{T}` — the π coefficients (`π₁, π₂, …`)
- `t_zero::T` — t(π₁): zero-frequency statistic (left-tailed)
- `t_nyquist::T` — t(π₂): Nyquist (π) statistic (left-tailed)
- `t_zero_cv::Dict{Int,T}`, `t_nyquist_cv::Dict{Int,T}` — (1%,5%,10%) CVs
- `pair_freqs::Vector{T}` — angular frequencies of the harmonic pairs (radians)
- `pair_F::Vector{T}` — joint F statistic for each conjugate pair (right-tailed)
- `pair_F_cv::Dict{Int,T}` — common (1%,5%,10%) paired-F CVs
- `F_seasonal::T` — joint F over all seasonal frequencies (Nyquist + pairs)
- `F_all::T` — joint F over all frequencies (zero + seasonal)
- `nobs::Int` — effective observations in the auxiliary regression
"""
struct HEGYResult{T<:AbstractFloat} <: AbstractUnitRootTest
    frequency::Int
    deterministic::Symbol
    lags::Int
    pi_coefs::Vector{T}
    t_zero::T
    t_nyquist::T
    t_zero_cv::Dict{Int,T}
    t_nyquist_cv::Dict{Int,T}
    pair_freqs::Vector{T}
    pair_F::Vector{T}
    pair_F_cv::Dict{Int,T}
    F_seasonal::T
    F_all::T
    nobs::Int
end

"""
    ERSResult{T} <: AbstractUnitRootTest

Elliott-Rothenberg-Stock (1996) feasible point-optimal `P_T` test result.
Small `P_T` rejects the unit-root null.

Fields:
- `P_T::T` — feasible point-optimal statistic
- `pvalue::T` — interpolated p-value
- `regression::Symbol` — `:constant` (demean) or `:trend` (detrend)
- `critical_values::Dict{Int,T}` — (1%,5%,10%) CVs (ERS 1996, Table 1)
- `nobs::Int` — number of observations
"""
struct ERSResult{T<:AbstractFloat} <: AbstractUnitRootTest
    P_T::T
    pvalue::T
    regression::Symbol
    critical_values::Dict{Int,T}
    nobs::Int
end

# --- StatsAPI interface (EV-29) ---
StatsAPI.nobs(r::HEGYResult) = r.nobs
# Primary p-value: interpolate the zero-frequency t on its own CVs (left tail).
StatsAPI.pvalue(r::HEGYResult) = _ers_pt_pvalue(r.t_zero, r.t_zero_cv)
StatsAPI.dof(r::HEGYResult) = r.lags + length(r.pi_coefs)
StatsAPI.nobs(r::ERSResult) = r.nobs
StatsAPI.pvalue(r::ERSResult) = r.pvalue
StatsAPI.dof(r::ERSResult) = r.regression == :constant ? 1 : 2
