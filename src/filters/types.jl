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
Type definitions for time series filter results.

All filter result types inherit from `AbstractFilterResult` (defined in `core/types.jl`).
Provides unified accessors `trend()` and `cycle()` for trend-cycle decomposition results.
"""

# =============================================================================
# HP Filter Result
# =============================================================================

"""
    HPFilterResult{T} <: AbstractFilterResult

Result of the Hodrick-Prescott filter (Hodrick & Prescott 1997).

# Fields
- `trend::Vector{T}`: Estimated trend component (length T_obs)
- `cycle::Vector{T}`: Cyclical component y - trend (length T_obs)
- `lambda::T`: Smoothing parameter used
- `T_obs::Int`: Number of observations
"""
struct HPFilterResult{T<:AbstractFloat} <: AbstractFilterResult
    trend::Vector{T}
    cycle::Vector{T}
    lambda::T
    T_obs::Int
end

# =============================================================================
# Hamilton Filter Result
# =============================================================================

"""
    HamiltonFilterResult{T} <: AbstractFilterResult

Result of the Hamilton (2018) regression filter.

Regresses \$y_{t+h}\$ on \$[1, y_t, y_{t-1}, \\ldots, y_{t-p+1}]\$.
Residuals are the cyclical component; fitted values are the trend.

# Fields
- `trend::Vector{T}`: Fitted values (length T_obs - h - p + 1)
- `cycle::Vector{T}`: Residuals (length T_obs - h - p + 1)
- `beta::Vector{T}`: OLS coefficients [intercept; lag coefficients]
- `h::Int`: Forecast horizon
- `p::Int`: Number of lags
- `T_obs::Int`: Original series length
- `valid_range::UnitRange{Int}`: Indices into original series where results are valid
"""
struct HamiltonFilterResult{T<:AbstractFloat} <: AbstractFilterResult
    trend::Vector{T}
    cycle::Vector{T}
    beta::Vector{T}
    h::Int
    p::Int
    T_obs::Int
    valid_range::UnitRange{Int}
end

# =============================================================================
# Beveridge-Nelson Decomposition Result
# =============================================================================

"""
    BeveridgeNelsonResult{T} <: AbstractFilterResult

Result of the Beveridge-Nelson (1981) trend-cycle decomposition.

Decomposes a unit-root process into a permanent (random walk + drift) component
and a transitory (stationary) component using the ARIMA representation.

# Fields
- `permanent::Vector{T}`: Permanent (trend) component
- `transitory::Vector{T}`: Transitory (cycle) component
- `drift::T`: Estimated drift (mean of first differences)
- `long_run_multiplier::T`: Long-run multiplier psi(1) = 1 + sum(psi weights)
- `arima_order::Tuple{Int,Int,Int}`: (p, d, q) order used
- `T_obs::Int`: Number of observations
"""
struct BeveridgeNelsonResult{T<:AbstractFloat} <: AbstractFilterResult
    permanent::Vector{T}
    transitory::Vector{T}
    drift::T
    long_run_multiplier::T
    arima_order::Tuple{Int,Int,Int}
    T_obs::Int
end

# =============================================================================
# Baxter-King Band-Pass Filter Result
# =============================================================================

"""
    BaxterKingResult{T} <: AbstractFilterResult

Result of the Baxter-King (1999) band-pass filter.

Isolates cyclical fluctuations in a specified frequency band while removing
both low-frequency trend and high-frequency noise.

# Fields
- `cycle::Vector{T}`: Band-pass filtered component (length T_obs - 2K)
- `trend::Vector{T}`: Residual trend y - cycle (length T_obs - 2K)
- `weights::Vector{T}`: Symmetric filter weights [a_0, a_1, ..., a_K]
- `pl::Int`: Lower period bound (e.g., 6 quarters)
- `pu::Int`: Upper period bound (e.g., 32 quarters)
- `K::Int`: Truncation length (observations lost at each end)
- `T_obs::Int`: Original series length
- `valid_range::UnitRange{Int}`: Indices into original series where results are valid
"""
struct BaxterKingResult{T<:AbstractFloat} <: AbstractFilterResult
    cycle::Vector{T}
    trend::Vector{T}
    weights::Vector{T}
    pl::Int
    pu::Int
    K::Int
    T_obs::Int
    valid_range::UnitRange{Int}
end

# =============================================================================
# Boosted HP Filter Result
# =============================================================================

"""
    BoostedHPResult{T} <: AbstractFilterResult

Result of the boosted HP filter (Phillips & Shi 2021).

Iteratively re-applies the HP filter to the cyclical component until a stopping
criterion is met, improving trend estimation by removing remaining unit root behavior.

# Fields
- `trend::Vector{T}`: Final trend estimate (length T_obs)
- `cycle::Vector{T}`: Final cyclical component (length T_obs)
- `lambda::T`: Smoothing parameter used
- `iterations::Int`: Number of boosting iterations performed
- `stopping::Symbol`: Stopping criterion used (:ADF, :BIC, or :fixed)
- `bic_path::Vector{T}`: BIC value at each iteration
- `adf_pvalues::Vector{T}`: ADF p-values at each iteration
- `T_obs::Int`: Number of observations
"""
struct BoostedHPResult{T<:AbstractFloat} <: AbstractFilterResult
    trend::Vector{T}
    cycle::Vector{T}
    lambda::T
    iterations::Int
    stopping::Symbol
    bic_path::Vector{T}
    adf_pvalues::Vector{T}
    T_obs::Int
end

# =============================================================================
# Unified Accessors
# =============================================================================

"""
    trend(result::AbstractFilterResult) -> Vector

Return the trend component from a filter result.
For `BeveridgeNelsonResult`, returns the permanent component.
"""
trend(r::HPFilterResult) = r.trend
trend(r::HamiltonFilterResult) = r.trend
trend(r::BeveridgeNelsonResult) = r.permanent
trend(r::BaxterKingResult) = r.trend
trend(r::BoostedHPResult) = r.trend

"""
    cycle(result::AbstractFilterResult) -> Vector

Return the cyclical component from a filter result.
For `BeveridgeNelsonResult`, returns the transitory component.
"""
cycle(r::HPFilterResult) = r.cycle
cycle(r::HamiltonFilterResult) = r.cycle
cycle(r::BeveridgeNelsonResult) = r.transitory
cycle(r::BaxterKingResult) = r.cycle
cycle(r::BoostedHPResult) = r.cycle

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, r::HPFilterResult)
    data = Any[
        "Observations"  r.T_obs;
        "Lambda"        _fmt(r.lambda; digits=1);
        "Trend mean"    _fmt(mean(r.trend));
        "Cycle std"     _fmt(std(r.cycle))
    ]
    _pretty_table(io, data;
        title = "Hodrick-Prescott Filter",
        column_labels = ["", "Value"],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, r::HamiltonFilterResult)
    data = Any[
        "Observations (original)"  r.T_obs;
        "Observations (effective)" length(r.cycle);
        "Horizon (h)"              r.h;
        "Lags (p)"                 r.p;
        "Valid range"              "$(r.valid_range.start):$(r.valid_range.stop)";
        "Cycle std"                _fmt(std(r.cycle))
    ]
    _pretty_table(io, data;
        title = "Hamilton (2018) Filter",
        column_labels = ["", "Value"],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, r::BeveridgeNelsonResult)
    p, d, q = r.arima_order
    data = Any[
        "Observations"         r.T_obs;
        "ARIMA order"          "($p,$d,$q)";
        "Drift"                _fmt(r.drift);
        "Long-run multiplier"  _fmt(r.long_run_multiplier);
        "Transitory std"       _fmt(std(r.transitory))
    ]
    _pretty_table(io, data;
        title = "Beveridge-Nelson Decomposition",
        column_labels = ["", "Value"],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, r::BaxterKingResult)
    data = Any[
        "Observations (original)"  r.T_obs;
        "Observations (filtered)"  length(r.cycle);
        "Period band"              "[$(r.pl), $(r.pu)]";
        "Truncation (K)"           r.K;
        "Obs. lost"                2 * r.K;
        "Valid range"              "$(r.valid_range.start):$(r.valid_range.stop)";
        "Cycle std"                _fmt(std(r.cycle))
    ]
    _pretty_table(io, data;
        title = "Baxter-King Band-Pass Filter",
        column_labels = ["", "Value"],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, r::BoostedHPResult)
    data = Any[
        "Observations"  r.T_obs;
        "Lambda"        _fmt(r.lambda; digits=1);
        "Iterations"    r.iterations;
        "Stopping"      string(r.stopping);
        "Cycle std"     _fmt(std(r.cycle))
    ]
    _pretty_table(io, data;
        title = "Boosted HP Filter (Phillips & Shi 2021)",
        column_labels = ["", "Value"],
        alignment = [:l, :r],
    )
end
