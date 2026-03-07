# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Result types for spectral analysis: ACF/PACF, spectral density, cross-spectrum,
and transfer function analysis.
"""

# =============================================================================
# ACF / PACF / CCF Result
# =============================================================================

"""
    ACFResult{T} <: AbstractAnalysisResult

Result from autocorrelation, partial autocorrelation, or cross-correlation analysis.

# Fields
- `lags::Vector{Int}` — lag indices
- `acf::Vector{T}` — autocorrelation values
- `pacf::Vector{T}` — partial autocorrelation values
- `ci::T` — confidence interval half-width (±1.96/√n by default)
- `ccf::Union{Nothing,Vector{T}}` — cross-correlation values (for `ccf()`)
- `q_stats::Vector{T}` — cumulative Ljung-Box Q-statistics
- `q_pvalues::Vector{T}` — p-values of Q-statistics
- `nobs::Int` — number of observations
"""
struct ACFResult{T<:AbstractFloat} <: AbstractAnalysisResult
    lags::Vector{Int}
    acf::Vector{T}
    pacf::Vector{T}
    ci::T
    ccf::Union{Nothing,Vector{T}}
    q_stats::Vector{T}
    q_pvalues::Vector{T}
    nobs::Int
end

# =============================================================================
# Spectral Density Result
# =============================================================================

"""
    SpectralDensityResult{T} <: AbstractAnalysisResult

Result from spectral density estimation (periodogram, Welch, smoothed, AR).

# Fields
- `freq::Vector{T}` — frequency grid [0, π]
- `density::Vector{T}` — estimated spectral density
- `ci_lower::Vector{T}` — lower confidence bound
- `ci_upper::Vector{T}` — upper confidence bound
- `method::Symbol` — estimation method (:periodogram, :welch, :smoothed, :ar)
- `bandwidth::T` — effective bandwidth
- `nobs::Int` — number of observations
"""
struct SpectralDensityResult{T<:AbstractFloat} <: AbstractAnalysisResult
    freq::Vector{T}
    density::Vector{T}
    ci_lower::Vector{T}
    ci_upper::Vector{T}
    method::Symbol
    bandwidth::T
    nobs::Int
end

# =============================================================================
# Cross-Spectrum Result
# =============================================================================

"""
    CrossSpectrumResult{T} <: AbstractAnalysisResult

Result from cross-spectral analysis between two series.

# Fields
- `freq::Vector{T}` — frequency grid [0, π]
- `co_spectrum::Vector{T}` — co-spectrum (real part of cross-spectral density)
- `quad_spectrum::Vector{T}` — quadrature spectrum (negative imaginary part)
- `coherence::Vector{T}` — squared coherence ∈ [0, 1]
- `phase::Vector{T}` — phase spectrum (radians)
- `gain::Vector{T}` — gain (amplitude ratio)
- `nobs::Int` — number of observations
"""
struct CrossSpectrumResult{T<:AbstractFloat} <: AbstractAnalysisResult
    freq::Vector{T}
    co_spectrum::Vector{T}
    quad_spectrum::Vector{T}
    coherence::Vector{T}
    phase::Vector{T}
    gain::Vector{T}
    nobs::Int
end

# =============================================================================
# Transfer Function Result
# =============================================================================

"""
    TransferFunctionResult{T} <: AbstractAnalysisResult

Result from transfer function (frequency response) analysis of a filter.

# Fields
- `freq::Vector{T}` — frequency grid [0, π]
- `gain::Vector{T}` — gain (amplitude) at each frequency
- `phase::Vector{T}` — phase shift (radians) at each frequency
- `filter::Symbol` — filter type (:hp, :bk, :hamilton, :ideal_bandpass)
"""
struct TransferFunctionResult{T<:AbstractFloat} <: AbstractAnalysisResult
    freq::Vector{T}
    gain::Vector{T}
    phase::Vector{T}
    filter::Symbol
end
