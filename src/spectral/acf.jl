# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Autocorrelation (ACF), partial autocorrelation (PACF), and cross-correlation (CCF)
functions with Ljung-Box Q-statistics.

Implements:
- `acf(y; lags)` — sample autocorrelation function
- `pacf(y; lags, method)` — sample partial autocorrelation via Levinson-Durbin or OLS
- `acf_pacf(y; lags)` — both ACF and PACF in one call
- `ccf(x, y; lags)` — cross-correlation function

References:
- Box, G. E. P. & Jenkins, G. M. (1976). Time Series Analysis.
- Ljung, G. M. & Box, G. E. P. (1978). "On a Measure of Lack of Fit in Time Series Models."
- Brockwell, P. J. & Davis, R. A. (1991). Time Series: Theory and Methods.
"""

# =============================================================================
# Helpers
# =============================================================================

"""Levinson-Durbin recursion for PACF from ACF values."""
function _pacf_levinson(rho::AbstractVector{T}, maxlag::Int) where {T<:AbstractFloat}
    pacf_vals = zeros(T, maxlag)
    length(rho) < maxlag && return pacf_vals

    # Lag 1
    pacf_vals[1] = rho[1]
    phi = zeros(T, maxlag)
    phi[1] = rho[1]

    for k in 2:maxlag
        # Numerator: rho[k] - sum phi[j]*rho[k-j]
        num = rho[k]
        for j in 1:(k-1)
            num -= phi[j] * rho[abs(k - j)]
        end
        # Denominator: 1 - sum phi[j]*rho[j]
        den = one(T)
        for j in 1:(k-1)
            den -= phi[j] * rho[j]
        end
        abs(den) < T(1e-15) && break
        pacf_vals[k] = num / den

        # Update phi
        phi_new = zeros(T, maxlag)
        phi_new[k] = pacf_vals[k]
        for j in 1:(k-1)
            phi_new[j] = phi[j] - pacf_vals[k] * phi[k - j]
        end
        phi .= phi_new
    end
    pacf_vals
end

"""OLS-based PACF: regress y_t on y_{t-1},...,y_{t-k}, take last coefficient."""
function _pacf_ols(y::AbstractVector{T}, maxlag::Int) where {T<:AbstractFloat}
    n = length(y)
    pacf_vals = zeros(T, maxlag)

    for k in 1:maxlag
        neff = n - k
        neff < k + 1 && break

        # Build design matrix
        X = ones(T, neff, k + 1)  # intercept + k lags
        for j in 1:k
            for t in 1:neff
                X[t, j + 1] = y[t + k - j]
            end
        end
        yy = y[(k+1):n]

        # OLS
        XtX = X' * X
        Xty = X' * yy
        beta = XtX \ Xty
        pacf_vals[k] = beta[end]  # last lag coefficient
    end
    pacf_vals
end

"""Cumulative Ljung-Box Q-statistics from ACF values."""
function _ljung_box_cumulative(rho::AbstractVector{T}, n::Int, maxlag::Int) where {T<:AbstractFloat}
    q_stats = zeros(T, maxlag)
    q_pvalues = ones(T, maxlag)

    cumQ = zero(T)
    for k in 1:maxlag
        cumQ += rho[k]^2 / (n - k)
        q_stats[k] = n * (n + 2) * cumQ
        # Q ~ chi2(k) under H0
        q_pvalues[k] = k > 0 ? ccdf(Chisq(k), q_stats[k]) : one(T)
    end
    q_stats, q_pvalues
end

"""Return a zero ACFResult when data is degenerate (e.g., constant series)."""
function _zero_acf_result(::Type{T}, lags::Vector{Int}, n::Int;
                          ccf_vals::Union{Nothing,Vector{T}}=nothing) where {T<:AbstractFloat}
    maxlag = length(lags)
    ci = T(1.96) / sqrt(T(n))
    ACFResult{T}(lags, zeros(T, maxlag), zeros(T, maxlag), ci,
                 ccf_vals, zeros(T, maxlag), ones(T, maxlag), n)
end

# =============================================================================
# ACF
# =============================================================================

"""
    acf(y::AbstractVector; lags::Int=0, conf_level::Real=0.95) -> ACFResult

Compute the sample autocorrelation function.

# Arguments
- `y` — time series vector
- `lags` — maximum lag (default: min(n-1, 10*log10(n)))
- `conf_level` — confidence level for the white-noise band (default: 0.95)

# Returns
`ACFResult` with ACF values, empty PACF, Ljung-Box Q-stats, and p-values.

# Example
```julia
result = acf(randn(200), lags=20)
result.acf          # autocorrelation at lags 1:20
result.q_pvalues    # Ljung-Box p-values
```
"""
function acf(y::AbstractVector{T}; lags::Int=0, conf_level::Real=0.95) where {T<:AbstractFloat}
    n = length(y)
    n < 3 && throw(ArgumentError("Need at least 3 observations, got $n"))

    maxlag = lags > 0 ? min(lags, n - 1) : min(n - 1, round(Int, 10 * log10(n)))
    maxlag = max(1, maxlag)
    lag_idx = collect(1:maxlag)

    # Sample ACF via biased estimator (Bartlett)
    ymean = mean(y)
    gamma0 = sum((y[t] - ymean)^2 for t in 1:n) / n
    gamma0 < T(1e-15) && return _zero_acf_result(T, lag_idx, n)

    rho = zeros(T, maxlag)
    for k in 1:maxlag
        rho[k] = sum((y[t] - ymean) * (y[t-k] - ymean) for t in (k+1):n) / n / gamma0
    end

    z_val = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    ci = z_val / sqrt(T(n))

    q_stats, q_pvalues = _ljung_box_cumulative(rho, n, maxlag)

    ACFResult{T}(lag_idx, rho, zeros(T, maxlag), ci, nothing, q_stats, q_pvalues, n)
end

# Non-Float64 fallback
acf(y::AbstractVector{<:Real}; kwargs...) = acf(Float64.(y); kwargs...)

# =============================================================================
# PACF
# =============================================================================

"""
    pacf(y::AbstractVector; lags::Int=0, method::Symbol=:levinson, conf_level::Real=0.95) -> ACFResult

Compute the sample partial autocorrelation function.

# Arguments
- `y` — time series vector
- `lags` — maximum lag (default: min(n-1, 10*log10(n)))
- `method` — `:levinson` (Levinson-Durbin, default) or `:ols`
- `conf_level` — confidence level for the white-noise band (default: 0.95)

# Returns
`ACFResult` with PACF values, empty ACF, and no Q-stats.
"""
function pacf(y::AbstractVector{T}; lags::Int=0, method::Symbol=:levinson,
              conf_level::Real=0.95) where {T<:AbstractFloat}
    n = length(y)
    n < 3 && throw(ArgumentError("Need at least 3 observations, got $n"))
    method in (:levinson, :ols) || throw(ArgumentError("method must be :levinson or :ols"))

    maxlag = lags > 0 ? min(lags, n - 1) : min(n - 1, round(Int, 10 * log10(n)))
    maxlag = max(1, maxlag)
    lag_idx = collect(1:maxlag)

    z_val = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    ci = z_val / sqrt(T(n))

    if method == :levinson
        # Need ACF first
        ymean = mean(y)
        gamma0 = sum((y[t] - ymean)^2 for t in 1:n) / n
        gamma0 < T(1e-15) && return _zero_acf_result(T, lag_idx, n)

        rho = zeros(T, maxlag)
        for k in 1:maxlag
            rho[k] = sum((y[t] - ymean) * (y[t-k] - ymean) for t in (k+1):n) / n / gamma0
        end
        pacf_vals = _pacf_levinson(rho, maxlag)
    else
        pacf_vals = _pacf_ols(y, maxlag)
    end

    ACFResult{T}(lag_idx, zeros(T, maxlag), pacf_vals, ci, nothing,
                 zeros(T, maxlag), ones(T, maxlag), n)
end

# Non-Float64 fallback
pacf(y::AbstractVector{<:Real}; kwargs...) = pacf(Float64.(y); kwargs...)

# =============================================================================
# ACF + PACF combined
# =============================================================================

"""
    acf_pacf(y::AbstractVector; lags::Int=0, method::Symbol=:levinson, conf_level::Real=0.95) -> ACFResult

Compute both ACF and PACF in a single pass.

# Arguments
Same as `acf()` / `pacf()`.

# Returns
`ACFResult` with both `acf` and `pacf` fields populated, plus Ljung-Box Q-stats.
"""
function acf_pacf(y::AbstractVector{T}; lags::Int=0, method::Symbol=:levinson,
                  conf_level::Real=0.95) where {T<:AbstractFloat}
    n = length(y)
    n < 3 && throw(ArgumentError("Need at least 3 observations, got $n"))
    method in (:levinson, :ols) || throw(ArgumentError("method must be :levinson or :ols"))

    maxlag = lags > 0 ? min(lags, n - 1) : min(n - 1, round(Int, 10 * log10(n)))
    maxlag = max(1, maxlag)
    lag_idx = collect(1:maxlag)

    # ACF
    ymean = mean(y)
    gamma0 = sum((y[t] - ymean)^2 for t in 1:n) / n
    gamma0 < T(1e-15) && return _zero_acf_result(T, lag_idx, n)

    rho = zeros(T, maxlag)
    for k in 1:maxlag
        rho[k] = sum((y[t] - ymean) * (y[t-k] - ymean) for t in (k+1):n) / n / gamma0
    end

    # PACF
    pacf_vals = method == :levinson ? _pacf_levinson(rho, maxlag) : _pacf_ols(y, maxlag)

    z_val = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    ci = z_val / sqrt(T(n))

    q_stats, q_pvalues = _ljung_box_cumulative(rho, n, maxlag)

    ACFResult{T}(lag_idx, rho, pacf_vals, ci, nothing, q_stats, q_pvalues, n)
end

# Non-Float64 fallback
acf_pacf(y::AbstractVector{<:Real}; kwargs...) = acf_pacf(Float64.(y); kwargs...)

# =============================================================================
# CCF
# =============================================================================

"""
    ccf(x::AbstractVector, y::AbstractVector; lags::Int=0, conf_level::Real=0.95) -> ACFResult

Compute the cross-correlation function between `x` and `y`.

Returns CCF(k) = Cor(x_{t+k}, y_t) for k = -lags:lags, stored in the `ccf` field.
Positive lags mean `x` leads `y`.

# Arguments
- `x`, `y` — time series vectors (same length)
- `lags` — maximum lag (default: min(n-1, 10*log10(n)))
- `conf_level` — confidence level (default: 0.95)

# Returns
`ACFResult` with `ccf` field populated. The `lags` field runs from `-lags` to `+lags`.
"""
function ccf(x::AbstractVector{T}, y::AbstractVector{T};
             lags::Int=0, conf_level::Real=0.95) where {T<:AbstractFloat}
    n = length(x)
    n == length(y) || throw(DimensionMismatch("x and y must have the same length"))
    n < 3 && throw(ArgumentError("Need at least 3 observations, got $n"))

    maxlag = lags > 0 ? min(lags, n - 1) : min(n - 1, round(Int, 10 * log10(n)))
    maxlag = max(1, maxlag)
    lag_idx = collect(-maxlag:maxlag)

    xmean, ymean = mean(x), mean(y)
    sx = sqrt(sum((x[t] - xmean)^2 for t in 1:n) / n)
    sy = sqrt(sum((y[t] - ymean)^2 for t in 1:n) / n)

    (sx < T(1e-15) || sy < T(1e-15)) && return _zero_acf_result(T, lag_idx, n;
        ccf_vals=zeros(T, 2 * maxlag + 1))

    ccf_vals = zeros(T, 2 * maxlag + 1)
    for k in -maxlag:maxlag
        s = zero(T)
        for t in max(1, 1 - k):min(n, n - k)
            s += (x[t + k] - xmean) * (y[t] - ymean)
        end
        ccf_vals[k + maxlag + 1] = s / (n * sx * sy)
    end

    z_val = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    ci = z_val / sqrt(T(n))

    ACFResult{T}(lag_idx, zeros(T, length(lag_idx)), zeros(T, length(lag_idx)),
                 ci, ccf_vals, zeros(T, length(lag_idx)), ones(T, length(lag_idx)), n)
end

# Non-Float64 fallbacks
ccf(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}; kwargs...) =
    ccf(Float64.(x), Float64.(y); kwargs...)
ccf(x::AbstractVector{T}, y::AbstractVector{<:Real}; kwargs...) where {T<:AbstractFloat} =
    ccf(x, T.(y); kwargs...)
ccf(x::AbstractVector{<:Real}, y::AbstractVector{T}; kwargs...) where {T<:AbstractFloat} =
    ccf(T.(x), y; kwargs...)
