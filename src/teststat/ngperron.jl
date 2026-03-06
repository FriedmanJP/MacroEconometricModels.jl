# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Ng-Perron unit root tests with GLS detrending (MZα, MZt, MSB, MPT).
"""

using LinearAlgebra, Statistics

"""
    ngperron_test(y; regression=:constant) -> NgPerronResult

Ng-Perron unit root tests with GLS detrending (MZα, MZt, MSB, MPT).

Tests H₀: y has a unit root against H₁: y is stationary.
These tests have better size properties than ADF/PP in small samples.

# Arguments
- `y`: Time series vector
- `regression`: :constant (default) or :trend

# Returns
`NgPerronResult` containing MZα, MZt, MSB, MPT statistics and critical values.

# Example
```julia
y = cumsum(randn(100))
result = ngperron_test(y)
# Check if MZt rejects at 5%
result.MZt < result.critical_values[:MZt][5]
```

# References
- Ng, S., & Perron, P. (2001). Lag length selection and the construction of
  unit root tests with good size and power. Econometrica, 69(6), 1519-1554.
"""
function ngperron_test(y::AbstractVector{T};
                       regression::Symbol=:constant) where {T<:AbstractFloat}

    regression ∈ (:constant, :trend) ||
        throw(ArgumentError("regression must be :constant or :trend"))

    n = length(y)
    n < 20 && throw(ArgumentError("Time series too short (n=$n), need at least 20 observations"))

    # GLS detrending parameter
    c_bar = regression == :constant ? T(-7.0) : T(-13.5)
    alpha = 1 + c_bar / n

    # Construct quasi-differenced data
    y_qd = copy(y)
    y_qd[2:end] = y[2:end] - alpha * y[1:end-1]

    # Deterministic regressors: ORIGINAL (for detrending) and quasi-differenced (for GLS)
    if regression == :constant
        Z_orig = ones(T, n)
        z_qd = ones(T, n)
        z_qd[2:end] .= 1 - alpha
        Z_qd = reshape(z_qd, :, 1)
        Z_orig_mat = reshape(Z_orig, :, 1)
    else  # :trend
        Z_orig = hcat(ones(T, n), T.(1:n))
        z1 = ones(T, n)
        z1[2:end] .= 1 - alpha
        z2 = T.(1:n)
        z2[2:end] = z2[2:end] - alpha * z2[1:end-1]
        Z_qd = hcat(z1, z2)
        Z_orig_mat = Z_orig
    end

    # GLS detrending: coefficients from QD regression, applied to original Z
    delta = Z_qd \ y_qd
    y_d = y - Z_orig_mat * delta

    # Compute statistics
    # Autoregressive spectral density estimate at frequency zero
    # Ng & Perron (2001, Table 1): fit AR(k) to FIRST DIFFERENCES of GLS-detrended series
    k = floor(Int, 4 * (n / 100)^(2/9))  # MAIC bandwidth

    # AR(k) regression on first differences of detrended series for spectral density
    dy_d = diff(y_d)
    if k > 0 && length(dy_d) > k + 5
        Y_ar = dy_d[(k+1):end]
        X_ar = hcat([dy_d[(k+1-j):(end-j)] for j in 1:k]...)
        rho_ar = X_ar \ Y_ar
        resid_ar = Y_ar - X_ar * rho_ar
        s2 = var(resid_ar; corrected=true)
        ar_sum = 1 - sum(rho_ar)
        s2_ar = s2 / ar_sum^2
    else
        s2_ar = var(dy_d; corrected=true)
    end

    # Components for statistics
    sum_yd2 = sum(y_d[1:end-1].^2) / n^2
    T_term = y_d[end]^2 / n

    # MZα statistic
    MZa = (T_term - s2_ar) / (2 * sum_yd2)

    # MSB statistic
    MSB = sqrt(sum_yd2 / s2_ar)

    # MZt statistic
    MZt = MZa * MSB

    # MPT statistic
    if regression == :constant
        MPT = (c_bar^2 * sum_yd2 + T_term) / s2_ar
    else
        MPT = (c_bar^2 * sum_yd2 + (1 - c_bar) * T_term) / s2_ar
    end

    # Critical values
    cv = Dict{Symbol,Dict{Int,T}}(
        stat => Dict{Int,T}(k => T(v) for (k, v) in vals)
        for (stat, vals) in NGPERRON_CRITICAL_VALUES[regression]
    )

    NgPerronResult(MZa, MZt, MSB, MPT, regression, cv, n)
end

ngperron_test(y::AbstractVector; kwargs...) = ngperron_test(Float64.(y); kwargs...)
