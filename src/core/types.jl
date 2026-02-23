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
Type hierarchy for MacroEconometricModels.jl - core abstract types.
"""

using StatsAPI, LinearAlgebra

# =============================================================================
# Abstract Types - Data Containers
# =============================================================================

"""
    AbstractMacroData

Abstract supertype for all MacroEconometricModels data containers.
Subtypes: `TimeSeriesData`, `PanelData`, `CrossSectionData`.
"""
abstract type AbstractMacroData end

# =============================================================================
# Abstract Types - Base Analysis Results
# =============================================================================

"""
    AbstractAnalysisResult

Abstract supertype for all innovation accounting and structural analysis results.
Provides a unified interface for accessing results from various methods (IRF, FEVD, HD).

Subtypes should implement:
- `point_estimate(result)` - return point estimate
- `has_uncertainty(result)` - return true if uncertainty bounds available
- `uncertainty_bounds(result)` - return (lower, upper) bounds if available
"""
abstract type AbstractAnalysisResult end

"""
    AbstractFrequentistResult <: AbstractAnalysisResult

Frequentist analysis results with point estimates and optional confidence intervals.
"""
abstract type AbstractFrequentistResult <: AbstractAnalysisResult end

"""
    AbstractBayesianResult <: AbstractAnalysisResult

Bayesian analysis results with posterior quantiles and means.
"""
abstract type AbstractBayesianResult <: AbstractAnalysisResult end

# =============================================================================
# Abstract Types - Model Types
# =============================================================================

"""Abstract supertype for Vector Autoregression models."""
abstract type AbstractVARModel <: StatsAPI.RegressionModel end

"""Abstract supertype for Bayesian prior specifications."""
abstract type AbstractPrior end

"""Abstract supertype for factor models (static and dynamic)."""
abstract type AbstractFactorModel <: StatsAPI.StatisticalModel end

"""Abstract supertype for multivariate normality test results."""
abstract type AbstractNormalityTest <: StatsAPI.HypothesisTest end

"""Abstract supertype for non-Gaussian SVAR identification results."""
abstract type AbstractNonGaussianSVAR end

"""Abstract supertype for univariate volatility models (ARCH/GARCH/SV)."""
abstract type AbstractVolatilityModel <: StatsAPI.RegressionModel end

"""Abstract supertype for trend-cycle decomposition filter results."""
abstract type AbstractFilterResult end

"""Abstract supertype for nowcasting models (DFM, BVAR, Bridge)."""
abstract type AbstractNowcastModel <: StatsAPI.StatisticalModel end

"""Abstract supertype for DSGE models."""
abstract type AbstractDSGEModel <: StatsAPI.StatisticalModel end

"""
    AbstractForecastResult{T<:AbstractFloat}

Abstract supertype for all forecast result types. Subtypes:
`ARIMAForecast`, `VolatilityForecast`, `LPForecast`, `VECMForecast`, `FactorForecast`.

All subtypes have at least a `horizon::Int` field.
"""
abstract type AbstractForecastResult{T<:AbstractFloat} end

# =============================================================================
# Forecast Accessor Functions
# =============================================================================

"""
    point_forecast(f::AbstractForecastResult)

Return the point forecast values (Vector or Matrix).

Most subtypes store the point forecast in a `.forecast` field; this default
accessor returns that field.  Overrides exist for `VECMForecast` (`.levels`)
and `FactorForecast` (`.observables`).
"""
point_forecast(f::AbstractForecastResult) = f.forecast

"""
    lower_bound(f::AbstractForecastResult)

Return the lower confidence interval bound.

Most subtypes store this in `.ci_lower`; an override exists for
`FactorForecast` (`.observables_lower`).
"""
lower_bound(f::AbstractForecastResult) = f.ci_lower

"""
    upper_bound(f::AbstractForecastResult)

Return the upper confidence interval bound.

Most subtypes store this in `.ci_upper`; an override exists for
`FactorForecast` (`.observables_upper`).
"""
upper_bound(f::AbstractForecastResult) = f.ci_upper

"""
    forecast_horizon(f::AbstractForecastResult)

Return the forecast horizon (number of steps ahead).
"""
forecast_horizon(f::AbstractForecastResult) = f.horizon

# =============================================================================
# Abstract Types - Analysis Result Types
# =============================================================================

"""Abstract supertype for impulse response function results."""
abstract type AbstractImpulseResponse <: AbstractAnalysisResult end

"""Abstract supertype for forecast error variance decomposition results."""
abstract type AbstractFEVD <: AbstractAnalysisResult end
