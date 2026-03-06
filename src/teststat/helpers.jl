# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Helper functions for unit root tests: critical values, p-values, bandwidth, regression matrices.
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# Critical Value & P-value Functions
# =============================================================================

"""Compute ADF critical values using response surface (Cheung & Lai 1995)."""
function adf_critical_values(regression::Symbol, nobs::Int, lags::Int=0, ::Type{TF}=Float64) where {TF<:AbstractFloat}
    coefs = MACKINNON_ADF_COEFS[regression]
    Dict{Int,TF}(
        level => TF(c[1] + c[2]/nobs + c[3]/nobs^2 + c[4]*(lags/nobs) + c[5]*(lags/nobs)^2)
        for (level, c) in coefs
    )
end

"""Approximate p-value for ADF test using MacKinnon (1994) interpolation."""
function adf_pvalue(stat::T, regression::Symbol, nobs::Int, lags::Int=0) where {T<:AbstractFloat}
    # Get critical values at standard levels
    cv = adf_critical_values(regression, nobs, lags, T)

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

# =============================================================================
# Critical Value Helpers for New Tests
# =============================================================================

"""Compute DF-GLS critical values via response surface."""
function _dfgls_critical_values(regression::Symbol, nobs::Int, lags::Int, ::Type{TF}=Float64) where {TF<:AbstractFloat}
    coefs = DFGLS_RSF_COEFS[regression]
    invT = 1.0 / nobs
    pT = lags / nobs
    Dict{Int,TF}(
        level => TF(c[1] + c[2]*invT + c[3]*invT^2 + c[4]*invT^3 + c[5]*invT^4 +
                     c[6]*pT + c[7]*pT^2 + c[8]*pT^3 + c[9]*pT^4)
        for (level, c) in coefs
    )
end

"""Compute LM unit root critical values via response surface."""
function _lm_unitroot_critical_values(breaks::Int, nobs::Int, lags::Int, ::Type{TF}=Float64) where {TF<:AbstractFloat}
    if breaks == 2
        return Dict{Int,TF}(k => TF(v) for (k, v) in LM_2BREAK_A_CV)
    end
    if breaks == 1
        return Dict{Int,TF}(k => TF(v) for (k, v) in LM_1BREAK_A_CV)
    end
    coefs = LM_UNITROOT_RSF[0]
    invT = 1.0 / nobs
    pT = lags / nobs
    Dict{Int,TF}(
        level => TF(c[1] + c[2]*invT + c[3]*invT^2 + c[4]*pT + c[5]*pT^2)
        for (level, c) in coefs
    )
end

"""Get sample bracket for Fourier critical values."""
function _fourier_sample_bracket(n::Int)
    n <= 150 ? 1 : n <= 349 ? 2 : n <= 500 ? 3 : 4
end

"""Get Narayan-Popp critical values based on sample size."""
function _narayan_popp_cv(model::Symbol, n::Int, ::Type{TF}=Float64) where {TF<:AbstractFloat}
    table = NARAYAN_POPP_CV[model]
    key = n <= 50 ? 50 : n <= 200 ? 200 : n <= 400 ? 400 : 999
    Dict{Int,TF}(k => TF(v) for (k, v) in table[key])
end

"""Get ERS Pt critical values by interpolating sample size."""
function _ers_pt_critical_values(regression::Symbol, nobs::Int, ::Type{TF}=Float64) where {TF<:AbstractFloat}
    table = ERS_PT_CV[regression]
    key = nobs <= 50 ? 50 : nobs <= 100 ? 100 : nobs <= 200 ? 200 : 500
    Dict{Int,TF}(k => TF(v) for (k, v) in table[key])
end
