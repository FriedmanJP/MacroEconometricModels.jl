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
Helper functions for unit root tests: critical values, p-values, bandwidth, regression matrices.
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# Critical Value & P-value Functions
# =============================================================================

"""Compute ADF critical values using MacKinnon response surface."""
function adf_critical_values(regression::Symbol, nobs::Int, T::Type=Float64)
    coefs = MACKINNON_ADF_COEFS[regression]
    Dict{Int,T}(
        level => T(c[1] + c[2]/nobs + c[3]/nobs^2)
        for (level, c) in coefs
    )
end

"""Approximate p-value for ADF test using MacKinnon (1994) interpolation."""
function adf_pvalue(stat::T, regression::Symbol, nobs::Int) where {T<:AbstractFloat}
    # Get critical values at standard levels
    cv = adf_critical_values(regression, nobs, T)

    # Simple interpolation between critical values
    if stat <= cv[1]
        return T(0.001)  # Below 1% critical value
    elseif stat <= cv[5]
        # Interpolate between 1% and 5%
        return T(0.01 + 0.04 * (stat - cv[1]) / (cv[5] - cv[1]))
    elseif stat <= cv[10]
        # Interpolate between 5% and 10%
        return T(0.05 + 0.05 * (stat - cv[5]) / (cv[10] - cv[5]))
    else
        # Above 10% critical value - use normal approximation for large values
        # The ADF statistic converges to standard normal under stationarity
        return T(min(1.0, 0.10 + 0.90 * (1 - cdf(Normal(), -stat))))
    end
end

"""Approximate p-value for KPSS test."""
function kpss_pvalue(stat::T, regression::Symbol) where {T<:AbstractFloat}
    cv = KPSS_CRITICAL_VALUES[regression]

    if stat >= cv[1]
        return T(0.01)
    elseif stat >= cv[5]
        return T(0.01 + 0.04 * (cv[1] - stat) / (cv[1] - cv[5]))
    elseif stat >= cv[10]
        return T(0.05 + 0.05 * (cv[5] - stat) / (cv[5] - cv[10]))
    else
        return T(0.10 + 0.40 * (cv[10] - stat) / cv[10])
    end
end

"""Approximate p-value for Zivot-Andrews test."""
function za_pvalue(stat::T, regression::Symbol) where {T<:AbstractFloat}
    cv = ZA_CRITICAL_VALUES[regression]

    if stat <= cv[1]
        return T(0.01)
    elseif stat <= cv[5]
        return T(0.01 + 0.04 * (stat - cv[1]) / (cv[5] - cv[1]))
    elseif stat <= cv[10]
        return T(0.05 + 0.05 * (stat - cv[5]) / (cv[10] - cv[5]))
    else
        return T(min(1.0, 0.10 + 0.30 * (stat - cv[10]) / abs(cv[10])))
    end
end

"""Approximate p-value for Ng-Perron tests."""
function _ngperron_pvalue(stat::T, regression::Symbol, test::Symbol) where {T<:AbstractFloat}
    cv = NGPERRON_CRITICAL_VALUES[regression][test]

    # For MZa, MZt: more negative = reject
    # For MSB: smaller = reject
    # For MPT: smaller = reject
    if test in (:MZa, :MZt)
        if stat <= cv[1]
            return T(0.01)
        elseif stat <= cv[5]
            return T(0.01 + 0.04 * (stat - cv[1]) / (cv[5] - cv[1]))
        elseif stat <= cv[10]
            return T(0.05 + 0.05 * (stat - cv[5]) / (cv[10] - cv[5]))
        else
            return T(min(1.0, 0.10 + 0.30 * (stat - cv[10]) / abs(cv[10])))
        end
    else  # MSB, MPT
        if stat <= cv[1]
            return T(0.01)
        elseif stat <= cv[5]
            return T(0.01 + 0.04 * (stat - cv[1]) / (cv[5] - cv[1]))
        elseif stat <= cv[10]
            return T(0.05 + 0.05 * (stat - cv[5]) / (cv[10] - cv[5]))
        else
            return T(min(1.0, 0.10 + 0.30 * (stat - cv[10]) / cv[10]))
        end
    end
end

# =============================================================================
# ADF Lag Selection & Regression Matrix
# =============================================================================

"""Compute optimal lag length for ADF test using information criterion."""
function adf_select_lags(y::AbstractVector{T}, max_lags::Int, regression::Symbol,
                         criterion::Symbol) where {T<:AbstractFloat}
    n = length(y)
    dy = diff(y)
    y_lag = y[1:end-1]

    best_ic = T(Inf)
    best_lag = 0

    for p in 0:max_lags
        nobs_eff = n - 1 - p
        nobs_eff < 10 && continue

        # Build regression matrix
        X = _build_adf_matrix(y, dy, p, regression)
        Y = dy[(p+1):end]

        k = size(X, 2)
        nobs_eff = length(Y)

        # OLS estimation
        XtX = X'X
        det(XtX) ≈ 0 && continue
        B = XtX \ (X'Y)
        resid = Y - X * B
        sse = sum(resid.^2)
        sigma2 = sse / (nobs_eff - k)

        # Information criterion
        ll = -nobs_eff/2 * (log(2π) + log(sigma2) + 1)
        ic = if criterion == :aic
            -2ll + 2k
        elseif criterion == :bic
            -2ll + k * log(nobs_eff)
        else  # :hqic
            -2ll + 2k * log(log(nobs_eff))
        end

        if ic < best_ic
            best_ic = ic
            best_lag = p
        end
    end

    best_lag
end

"""Build ADF regression matrix."""
function _build_adf_matrix(y::AbstractVector{T}, dy::AbstractVector{T},
                           lags::Int, regression::Symbol) where {T<:AbstractFloat}
    n = length(dy)
    nobs = n - lags

    # y_{t-1} column
    y_lag = y[(lags+1):(n)]

    # Lagged differences
    if lags > 0
        dy_lags = Matrix{T}(undef, nobs, lags)
        for j in 1:lags
            dy_lags[:, j] = dy[(lags+1-j):(n-j)]
        end
    end

    # Build design matrix based on regression type
    if regression == :none
        X = lags > 0 ? hcat(y_lag, dy_lags) : reshape(y_lag, :, 1)
    elseif regression == :constant
        ones_col = ones(T, nobs)
        X = lags > 0 ? hcat(ones_col, y_lag, dy_lags) : hcat(ones_col, y_lag)
    else  # :trend
        ones_col = ones(T, nobs)
        trend = T.(1:nobs)
        X = lags > 0 ? hcat(ones_col, trend, y_lag, dy_lags) : hcat(ones_col, trend, y_lag)
    end

    X
end

# =============================================================================
# Long-run Variance Estimation
# =============================================================================

"""Compute Newey-West bandwidth using Andrews (1991) AR(1) rule."""
function _nw_bandwidth(resid::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(resid)
    # AR(1) approximation for bandwidth
    rho = cor(resid[1:end-1], resid[2:end])
    rho = clamp(rho, -0.99, 0.99)
    # Andrews (1991) optimal bandwidth for Bartlett kernel
    bw = floor(Int, 1.1447 * (4 * rho^2 / (1 - rho^2)^2 * n)^(1/3))
    max(1, min(bw, n - 1))
end

"""Compute long-run variance using Bartlett kernel."""
function _long_run_variance(resid::AbstractVector{T}, bandwidth::Int) where {T<:AbstractFloat}
    n = length(resid)
    gamma0 = var(resid; corrected=false)

    lrv = gamma0
    for j in 1:bandwidth
        weight = 1 - j / (bandwidth + 1)  # Bartlett kernel
        gamma_j = sum(resid[1:end-j] .* resid[1+j:end]) / n
        lrv += 2 * weight * gamma_j
    end

    lrv
end

# =============================================================================
# Regression Name Helper
# =============================================================================

"""Helper function to format regression specification name."""
function _regression_name(regression::Symbol)
    if regression == :none
        return "None"
    elseif regression == :constant
        return "Constant"
    elseif regression == :trend
        return "Constant + Trend"
    elseif regression == :both
        return "Constant + Trend"
    else
        return string(regression)
    end
end
