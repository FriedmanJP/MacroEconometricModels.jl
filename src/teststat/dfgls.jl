# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
DF-GLS unit root test with ERS point optimal statistic and MGLS statistics.
"""

using LinearAlgebra, Statistics

"""
    dfgls_test(y; regression=:constant, lags=:aic, max_lags=nothing) -> DFGLSResult

DF-GLS unit root test (Elliott, Rothenberg & Stock 1996).

Tests H₀: unit root against H₁: stationary using GLS-detrended data.
Reports DF-GLS τ statistic, ERS Pt statistic, and MGLS statistics.

# Arguments
- `y`: Time series vector
- `regression`: `:constant` or `:trend`
- `lags`: Number of lags, or `:aic`/`:bic` for automatic selection
- `max_lags`: Maximum lags for IC selection

# References
- Elliott, G., Rothenberg, T. J., & Stock, J. H. (1996). Efficient tests for an
  autoregressive unit root. Econometrica, 64(4), 813-836.
- Ng, S., & Perron, P. (2001). Lag length selection and the construction of
  unit root tests with good size and power. Econometrica, 69(6), 1519-1554.
"""
function dfgls_test(y::AbstractVector{T};
                     regression::Symbol=:constant,
                     lags::Union{Int,Symbol}=:aic,
                     max_lags::Union{Int,Nothing}=nothing) where {T<:AbstractFloat}
    regression ∈ (:constant, :trend) ||
        throw(ArgumentError("regression must be :constant or :trend"))
    n = length(y)
    n < 30 && throw(ArgumentError("Need at least 30 observations, got $n"))

    max_p = isnothing(max_lags) ? floor(Int, 12 * (n / 100)^0.25) : max_lags

    # GLS detrending (ERS 1996)
    c_bar = regression == :constant ? T(-7) : T(-13.5)
    alpha = one(T) + c_bar / n

    # Quasi-difference y
    y_qd = copy(y)
    y_qd[2:end] = y[2:end] - alpha * y[1:end-1]

    # Deterministic regressors: original and quasi-differenced
    if regression == :constant
        Z_orig = reshape(ones(T, n), :, 1)
        z_qd = ones(T, n)
        z_qd[2:end] .= 1 - alpha
        Z_qd = reshape(z_qd, :, 1)
    else
        Z_orig = hcat(ones(T, n), T.(1:n))
        z1 = ones(T, n)
        z1[2:end] .= 1 - alpha
        z2 = T.(1:n)
        z2[2:end] = z2[2:end] - alpha * z2[1:end-1]
        Z_qd = hcat(z1, z2)
    end

    # GLS coefficients from QD regression, applied to original Z
    delta = Z_qd \ y_qd
    y_d = y - Z_orig * delta

    # ADF on detrended series (no deterministic terms)
    dy_d = diff(y_d)

    best_ic = T(Inf)
    best_stat = T(NaN)
    best_p = 0

    p_range = lags isa Int ? [lags] : collect(0:max_p)

    for p in p_range
        nobs = n - 1 - p
        nobs < 10 && continue

        Y = dy_d[(p+1):end]
        y_lag = y_d[(p+1):(n-1)]

        if p > 0
            dy_lags = Matrix{T}(undef, nobs, p)
            for j in 1:p
                dy_lags[:, j] = dy_d[(p+1-j):(n-1-j)]
            end
            X = hcat(y_lag, dy_lags)
        else
            X = reshape(y_lag, :, 1)
        end

        XtX = X'X
        cond(XtX) > 1e12 && continue

        B = XtX \ (X'Y)
        resid_vec = Y - X * B
        sigma2 = sum(resid_vec.^2) / (nobs - size(X, 2))

        if lags isa Symbol
            log_ssr = log(sum(resid_vec.^2) / nobs)
            k_params = size(X, 2)
            ic = lags == :aic ? log_ssr + 2 * k_params / nobs : log_ssr + log(nobs) * k_params / nobs

            if ic < best_ic
                best_ic = ic
                se = sqrt.(max.(sigma2 * diag(inv(XtX)), zero(T)))
                best_stat = se[1] > zero(T) ? B[1] / se[1] : T(NaN)
                best_p = p
            end
        else
            se = sqrt.(max.(sigma2 * diag(inv(XtX)), zero(T)))
            best_stat = se[1] > zero(T) ? B[1] / se[1] : T(NaN)
            best_p = p
        end
    end

    # ERS Pt statistic
    ssr_qd = sum((y_qd - Z_qd * delta).^2)
    delta_ols = Z_orig \ y
    ssr_ols = sum((y - Z_orig * delta_ols).^2)

    # Spectral density at zero via AR approximation on first differences of detrended series
    k_ar = floor(Int, 4 * (n / 100)^(2 / 9))
    dy_d_full = diff(y_d)

    if k_ar > 0 && length(dy_d_full) > k_ar + 5
        Y_ar = dy_d_full[(k_ar+1):end]
        X_ar = hcat([dy_d_full[(k_ar+1-j):(end-j)] for j in 1:k_ar]...)
        rho_ar = X_ar \ Y_ar
        resid_ar = Y_ar - X_ar * rho_ar
        s2 = var(resid_ar; corrected=true)
        ar_sum = 1 - sum(rho_ar)
        f00 = s2 / ar_sum^2
    else
        f00 = var(dy_d_full; corrected=true)
    end

    pt_stat = (ssr_qd - alpha * ssr_ols) / f00

    # MGLS statistics (Ng-Perron on GLS-detrended data)
    sum_yd2 = sum(y_d[1:end-1].^2) / n^2
    T_term = y_d[n]^2 / n

    MZa = (T_term - f00) / (2 * sum_yd2)
    MSB_val = sqrt(sum_yd2 / f00)
    MZt = MZa * MSB_val

    if regression == :constant
        MPT_val = (c_bar^2 * sum_yd2 + T_term) / f00
    else
        MPT_val = (c_bar^2 * sum_yd2 + (1 - c_bar) * T_term) / f00
    end

    nobs = n - 1 - best_p

    # Critical values
    cv = _dfgls_critical_values(regression, nobs, best_p, T)
    pt_cv = _ers_pt_critical_values(regression, n, T)

    # MGLS critical values (reuse Ng-Perron CVs)
    mgls_cv = Dict{Symbol,Dict{Int,T}}()
    np_cv = NGPERRON_CRITICAL_VALUES[regression]
    mgls_cv[:MZa] = Dict{Int,T}(k => T(v) for (k, v) in np_cv[:MZa])
    mgls_cv[:MZt] = Dict{Int,T}(k => T(v) for (k, v) in np_cv[:MZt])
    mgls_cv[:MSB] = Dict{Int,T}(k => T(v) for (k, v) in np_cv[:MSB])
    mgls_cv[:MPT] = Dict{Int,T}(k => T(v) for (k, v) in np_cv[:MPT])

    # P-value for DF-GLS (reject when stat is very negative)
    pval = if best_stat <= cv[1]; T(0.001)
    elseif best_stat <= cv[5]; T(0.01) + (best_stat - cv[1]) / (cv[5] - cv[1]) * T(0.04)
    elseif best_stat <= cv[10]; T(0.05) + (best_stat - cv[5]) / (cv[10] - cv[5]) * T(0.05)
    else; T(0.20)
    end

    # Pt p-value (reject when Pt is small)
    pt_pval = if pt_stat <= pt_cv[1]; T(0.001)
    elseif pt_stat <= pt_cv[5]; T(0.01) + (pt_stat - pt_cv[1]) / (pt_cv[5] - pt_cv[1]) * T(0.04)
    elseif pt_stat <= pt_cv[10]; T(0.05) + (pt_stat - pt_cv[5]) / (pt_cv[10] - pt_cv[5]) * T(0.05)
    else; T(0.20)
    end

    DFGLSResult(best_stat, pval, pt_stat, pt_pval, MZa, MZt, MSB_val, MPT_val,
                best_p, regression, cv, pt_cv, mgls_cv, nobs)
end

dfgls_test(y::AbstractVector; kwargs...) = dfgls_test(Float64.(y); kwargs...)
