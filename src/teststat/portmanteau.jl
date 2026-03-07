# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Portmanteau tests for serial correlation:
- Ljung-Box Q test (Ljung & Box 1978)
- Box-Pierce Q test (Box & Pierce 1970)
- Durbin-Watson test (Durbin & Watson 1950)

References:
- Ljung, G. M. & Box, G. E. P. (1978). "On a Measure of Lack of Fit."
- Box, G. E. P. & Pierce, D. A. (1970). "Distribution of Residual Autocorrelations."
- Durbin, J. & Watson, G. S. (1950, 1951). "Testing for Serial Correlation."
"""

# =============================================================================
# Result Types
# =============================================================================

"""
    LjungBoxResult{T} <: StatsAPI.HypothesisTest

Result from the Ljung-Box Q test for serial correlation.

# Fields
- `statistic::T` — Q-statistic
- `pvalue::T` — p-value (χ² distribution)
- `df::Int` — degrees of freedom
- `lags::Int` — number of lags tested
- `nobs::Int` — number of observations
"""
struct LjungBoxResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest
    statistic::T
    pvalue::T
    df::Int
    lags::Int
    nobs::Int
end

"""
    BoxPierceResult{T} <: StatsAPI.HypothesisTest

Result from the Box-Pierce Q test for serial correlation.

# Fields
- `statistic::T` — Q₀-statistic
- `pvalue::T` — p-value (χ² distribution)
- `df::Int` — degrees of freedom
- `lags::Int` — number of lags tested
- `nobs::Int` — number of observations
"""
struct BoxPierceResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest
    statistic::T
    pvalue::T
    df::Int
    lags::Int
    nobs::Int
end

"""
    DurbinWatsonResult{T} <: StatsAPI.HypothesisTest

Result from the Durbin-Watson test for first-order serial correlation.

# Fields
- `statistic::T` — DW statistic ∈ [0, 4]
- `pvalue::T` — approximate p-value
- `nobs::Int` — number of observations
"""
struct DurbinWatsonResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest
    statistic::T
    pvalue::T
    nobs::Int
end

# StatsAPI interface
StatsAPI.nobs(r::LjungBoxResult) = r.nobs
StatsAPI.nobs(r::BoxPierceResult) = r.nobs
StatsAPI.nobs(r::DurbinWatsonResult) = r.nobs
StatsAPI.pvalue(r::LjungBoxResult) = r.pvalue
StatsAPI.pvalue(r::BoxPierceResult) = r.pvalue
StatsAPI.pvalue(r::DurbinWatsonResult) = r.pvalue
StatsAPI.dof(r::LjungBoxResult) = r.df
StatsAPI.dof(r::BoxPierceResult) = r.df

# =============================================================================
# Ljung-Box Q Test
# =============================================================================

"""
    ljung_box_test(y::AbstractVector; lags::Int=0, fitdf::Int=0) -> LjungBoxResult

Ljung-Box Q test for serial correlation (H₀: no autocorrelation up to lag `lags`).

# Arguments
- `y` — time series or residuals
- `lags` — number of lags (default: min(n-1, 10*log10(n)))
- `fitdf` — number of fitted parameters to subtract from DOF (e.g., p+q for ARMA(p,q))

# Returns
`LjungBoxResult` with Q-statistic and p-value.
"""
function ljung_box_test(y::AbstractVector{T}; lags::Int=0, fitdf::Int=0) where {T<:AbstractFloat}
    n = length(y)
    n < 3 && throw(ArgumentError("Need at least 3 observations"))

    maxlag = lags > 0 ? min(lags, n - 1) : min(n - 1, round(Int, 10 * log10(n)))
    maxlag = max(1, maxlag)

    # ACF
    ymean = mean(y)
    gamma0 = sum((y[t] - ymean)^2 for t in 1:n) / n
    gamma0 < T(1e-15) && return LjungBoxResult{T}(zero(T), one(T), max(1, maxlag - fitdf), maxlag, n)

    Q = zero(T)
    for k in 1:maxlag
        rho_k = sum((y[t] - ymean) * (y[t-k] - ymean) for t in (k+1):n) / (n * gamma0)
        Q += rho_k^2 / (n - k)
    end
    Q *= n * (n + 2)

    df = max(1, maxlag - fitdf)
    pval = ccdf(Chisq(df), Q)

    LjungBoxResult{T}(Q, T(pval), df, maxlag, n)
end

# Non-Float64 fallback
ljung_box_test(y::AbstractVector{<:Real}; kwargs...) = ljung_box_test(Float64.(y); kwargs...)

# =============================================================================
# Box-Pierce Q Test
# =============================================================================

"""
    box_pierce_test(y::AbstractVector; lags::Int=0, fitdf::Int=0) -> BoxPierceResult

Box-Pierce Q₀ test for serial correlation. Simpler (unweighted) version of Ljung-Box.

# Arguments
Same as `ljung_box_test`.
"""
function box_pierce_test(y::AbstractVector{T}; lags::Int=0, fitdf::Int=0) where {T<:AbstractFloat}
    n = length(y)
    n < 3 && throw(ArgumentError("Need at least 3 observations"))

    maxlag = lags > 0 ? min(lags, n - 1) : min(n - 1, round(Int, 10 * log10(n)))
    maxlag = max(1, maxlag)

    ymean = mean(y)
    gamma0 = sum((y[t] - ymean)^2 for t in 1:n) / n
    gamma0 < T(1e-15) && return BoxPierceResult{T}(zero(T), one(T), max(1, maxlag - fitdf), maxlag, n)

    Q0 = zero(T)
    for k in 1:maxlag
        rho_k = sum((y[t] - ymean) * (y[t-k] - ymean) for t in (k+1):n) / (n * gamma0)
        Q0 += rho_k^2
    end
    Q0 *= n

    df = max(1, maxlag - fitdf)
    pval = ccdf(Chisq(df), Q0)

    BoxPierceResult{T}(Q0, T(pval), df, maxlag, n)
end

# Non-Float64 fallback
box_pierce_test(y::AbstractVector{<:Real}; kwargs...) = box_pierce_test(Float64.(y); kwargs...)

# =============================================================================
# Durbin-Watson Test
# =============================================================================

"""
    durbin_watson_test(resid::AbstractVector) -> DurbinWatsonResult

Durbin-Watson test for first-order autocorrelation in residuals.

DW ≈ 2(1 - ρ̂₁). Values near 2 indicate no autocorrelation;
near 0 indicates positive autocorrelation; near 4 indicates negative.

# Arguments
- `resid` — regression residuals

# Returns
`DurbinWatsonResult` with DW statistic and approximate p-value.
"""
function durbin_watson_test(resid::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(resid)
    n < 3 && throw(ArgumentError("Need at least 3 observations"))

    num = sum((resid[t] - resid[t-1])^2 for t in 2:n)
    den = sum(resid[t]^2 for t in 1:n)
    dw = den > T(1e-15) ? num / den : T(2.0)

    # Approximate p-value using normal approximation
    # E[DW] ≈ 2, Var[DW] ≈ 4/n for large n
    z = (dw - T(2.0)) / sqrt(T(4.0) / n)
    pval = T(2 * ccdf(Normal(), abs(z)))  # two-sided

    DurbinWatsonResult{T}(dw, pval, n)
end

# Non-Float64 fallback
durbin_watson_test(resid::AbstractVector{<:Real}) = durbin_watson_test(Float64.(resid))
