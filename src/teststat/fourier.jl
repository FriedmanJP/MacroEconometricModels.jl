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
Fourier ADF and Fourier KPSS unit root tests with flexible Fourier form.
"""

# =============================================================================
# Fourier ADF Test (Enders & Lee 2012)
# =============================================================================

"""
    fourier_adf_test(y; regression=:constant, fmax=3, lags=:aic, max_lags=nothing, trim=0.15) -> FourierADFResult

Fourier ADF unit root test with flexible Fourier form (Enders & Lee 2012).

Tests H0: unit root against H1: stationary with smooth structural breaks
captured by low-frequency Fourier components.

# Arguments
- `y`: Time series vector
- `regression`: `:constant` or `:trend`
- `fmax`: Maximum Fourier frequency (default 3, max 5)
- `lags`: Number of augmenting lags, or `:aic`/`:bic` for automatic selection
- `max_lags`: Maximum lags for IC selection (default: floor(12*(T/100)^0.25))
- `trim`: Trimming fraction (default 0.15)

# References
- Enders, W., & Lee, J. (2012). A unit root test using a Fourier series to
  approximate smooth breaks. Oxford Bulletin of Economics and Statistics, 74(4), 574-599.
"""
function fourier_adf_test(y::AbstractVector{T};
                           regression::Symbol=:constant,
                           fmax::Int=3,
                           lags::Union{Int,Symbol}=:aic,
                           max_lags::Union{Int,Nothing}=nothing,
                           trim::Real=0.15) where {T<:AbstractFloat}
    regression in (:constant, :trend) || throw(ArgumentError("regression must be :constant or :trend"))
    1 <= fmax <= 5 || throw(ArgumentError("fmax must be between 1 and 5"))

    n = length(y)
    n < 50 && throw(ArgumentError("Need at least 50 observations, got $n"))

    max_p = isnothing(max_lags) ? floor(Int, 12 * (n / 100)^0.25) : max_lags
    dy = diff(y)

    # Baseline SSR without Fourier terms (for F-test)
    ssr_no_fourier, _, p_nof = _fourier_adf_regression(y, dy, n, max_p, lags, regression, 0, T)

    # Find optimal k by minimum SSR across Fourier frequencies
    best_k = 1
    best_ssr = T(Inf)
    best_stat = T(Inf)
    best_p = 0

    for k in 1:fmax
        ssr, stat, p_used = _fourier_adf_regression(y, dy, n, max_p, lags, regression, k, T)
        if ssr < best_ssr
            best_ssr = ssr
            best_k = k
            best_stat = stat
            best_p = p_used
        end
    end

    # Effective sample size and number of regressors
    nobs = n - 1 - best_p
    k_base = regression == :trend ? 3 : 2  # const + [trend] + y_{t-1}
    k_fourier = k_base + 2 + best_p        # + sin + cos + lags

    # F-test for joint significance of Fourier terms
    f_stat = ((ssr_no_fourier - best_ssr) / 2) / (best_ssr / (nobs - k_fourier))

    # Critical values from tables
    bracket = _fourier_sample_bracket(n)
    cv = Dict{Int,T}(level => T(v) for (level, v) in FOURIER_ADF_CV[regression][bracket][best_k])
    f_cv = Dict{Int,T}(level => T(v) for (level, v) in FOURIER_F_CV[regression][bracket])

    pval = _interp_fourier_adf_pvalue(best_stat, cv)
    f_pval = _interp_f_pvalue(f_stat, f_cv)

    FourierADFResult(best_stat, pval, best_k, f_stat, f_pval, best_p, regression, cv, f_cv, nobs)
end

"""ADF regression with optional Fourier terms; returns (SSR, t-stat on gamma, lags used)."""
function _fourier_adf_regression(y::AbstractVector, dy::AbstractVector, n::Int,
                                  max_p::Int, lags_spec::Union{Int,Symbol},
                                  regression::Symbol, k::Int, ::Type{T}) where {T}
    # Determine lag to use
    p_use = if lags_spec isa Int
        lags_spec
    else
        _select_fourier_adf_lags(y, dy, n, max_p, lags_spec, regression, k, T)
    end

    nobs = n - 1 - p_use
    Y = dy[(p_use+1):end]
    X = _build_fourier_adf_X(y, dy, n, p_use, regression, k, T)

    B = (X'X) \ (X'Y)
    resid_vec = Y - X * B
    ssr = sum(resid_vec.^2)

    sigma2 = ssr / (nobs - size(X, 2))
    se = sqrt.(sigma2 * diag(inv(X'X)))
    gamma_idx = regression == :trend ? 3 : 2
    t_stat = B[gamma_idx] / se[gamma_idx]

    (ssr, t_stat, p_use)
end

"""Select lag order for Fourier ADF regression using information criteria."""
function _select_fourier_adf_lags(y::AbstractVector, dy::AbstractVector, n::Int,
                                   max_p::Int, criterion::Symbol,
                                   regression::Symbol, k::Int, ::Type{T}) where {T}
    best_ic = T(Inf)
    best_p = 0

    for p in 0:max_p
        nobs = n - 1 - p
        nobs < 10 && continue

        Y = dy[(p+1):end]
        X = _build_fourier_adf_X(y, dy, n, p, regression, k, T)
        size(X, 1) != nobs && continue

        ncols = size(X, 2)
        ncols >= nobs && continue

        XtX = X'X
        det(XtX) == 0 && continue
        B = XtX \ (X'Y)
        resid_vec = Y - X * B
        ssr = sum(resid_vec.^2)

        sigma2 = ssr / (nobs - ncols)
        sigma2 <= 0 && continue
        ll = -nobs / 2 * (log(2 * T(pi)) + log(sigma2) + 1)
        ic = if criterion == :aic
            -2ll + 2ncols
        else  # :bic
            -2ll + ncols * log(nobs)
        end

        if ic < best_ic
            best_ic = ic
            best_p = p
        end
    end

    best_p
end

"""Build design matrix for Fourier ADF regression."""
function _build_fourier_adf_X(y::AbstractVector, dy::AbstractVector, n::Int,
                               p::Int, regression::Symbol, k::Int, ::Type{T}) where {T}
    nobs = n - 1 - p
    t_vec = T.((p+1):(n-1))  # time indices for the regression sample

    # Deterministic terms
    cols = Vector{Vector{T}}()
    push!(cols, ones(T, nobs))  # constant
    if regression == :trend
        push!(cols, t_vec)
    end

    # Lagged level y_{t-1}
    push!(cols, T.(y[(p+1):(n-1)]))

    # Fourier terms
    if k > 0
        push!(cols, sin.(2 * T(pi) * k .* t_vec ./ n))
        push!(cols, cos.(2 * T(pi) * k .* t_vec ./ n))
    end

    # Lagged differences
    for j in 1:p
        push!(cols, T.(dy[(p+1-j):(n-1-j)]))
    end

    hcat(cols...)
end

# =============================================================================
# Fourier KPSS Test (Becker, Enders & Lee 2006)
# =============================================================================

"""
    fourier_kpss_test(y; regression=:constant, fmax=3, bandwidth=nothing) -> FourierKPSSResult

Fourier KPSS stationarity test (Becker, Enders & Lee 2006).

Tests H0: stationary (with Fourier terms) against H1: unit root.

# Arguments
- `y`: Time series vector
- `regression`: `:constant` or `:trend`
- `fmax`: Maximum Fourier frequency (default 3, max 3 for KPSS tables)
- `bandwidth`: Bartlett kernel bandwidth (default: Newey-West)

# References
- Becker, R., Enders, W., & Lee, J. (2006). A stationarity test in the presence
  of an unknown number of smooth breaks. Journal of Time Series Analysis, 27(3), 381-409.
"""
function fourier_kpss_test(y::AbstractVector{T};
                            regression::Symbol=:constant,
                            fmax::Int=3,
                            bandwidth::Union{Int,Nothing}=nothing) where {T<:AbstractFloat}
    regression in (:constant, :trend) || throw(ArgumentError("regression must be :constant or :trend"))

    n = length(y)
    n < 50 && throw(ArgumentError("Need at least 50 observations, got $n"))

    bw = isnothing(bandwidth) ? floor(Int, 4 * (n / 100)^0.25) : bandwidth
    fmax_use = min(fmax, 3)  # KPSS tables only go to k=3

    # Baseline SSR without Fourier terms
    t_vec = T.(1:n)
    X0 = regression == :constant ? ones(T, n, 1) : hcat(ones(T, n), t_vec)
    B0 = X0 \ y
    resid0 = y - X0 * B0
    ssr_no_fourier = sum(resid0.^2)

    # Find optimal k by minimum SSR
    best_k = 1
    best_ssr = T(Inf)

    for k in 1:fmax_use
        X = if regression == :constant
            hcat(ones(T, n), sin.(2 * T(pi) * k .* t_vec ./ n), cos.(2 * T(pi) * k .* t_vec ./ n))
        else
            hcat(ones(T, n), t_vec, sin.(2 * T(pi) * k .* t_vec ./ n), cos.(2 * T(pi) * k .* t_vec ./ n))
        end
        B = X \ y
        resid_vec = y - X * B
        ssr = sum(resid_vec.^2)

        if ssr < best_ssr
            best_ssr = ssr
            best_k = k
        end
    end

    # Compute KPSS statistic at optimal k
    X = if regression == :constant
        hcat(ones(T, n), sin.(2 * T(pi) * best_k .* t_vec ./ n), cos.(2 * T(pi) * best_k .* t_vec ./ n))
    else
        hcat(ones(T, n), t_vec, sin.(2 * T(pi) * best_k .* t_vec ./ n), cos.(2 * T(pi) * best_k .* t_vec ./ n))
    end
    B = X \ y
    resid_vec = y - X * B

    # Partial sum of residuals
    S = cumsum(resid_vec)

    # Long-run variance (Bartlett kernel)
    gamma0 = sum(resid_vec.^2) / n
    lrv = gamma0
    for j in 1:bw
        w = 1 - j / (bw + 1)
        gamma_j = sum(resid_vec[j+1:n] .* resid_vec[1:n-j]) / n
        lrv += 2 * w * gamma_j
    end

    kpss_stat = sum(S.^2) / (n^2 * lrv)

    # F-test for joint significance of Fourier terms
    k_fourier = size(X, 2)
    f_stat = ((ssr_no_fourier - best_ssr) / 2) / (best_ssr / (n - k_fourier))

    # Critical values
    bracket = min(_fourier_sample_bracket(n), 2)  # KPSS tables only have 2 brackets
    cv_table = FOURIER_KPSS_CV[regression][bracket]
    best_k_cv = min(best_k, length(cv_table))
    cv = Dict{Int,T}(level => T(v) for (level, v) in cv_table[best_k_cv])
    f_bracket = min(bracket, length(FOURIER_F_CV[regression]))
    f_cv = Dict{Int,T}(level => T(v) for (level, v) in FOURIER_F_CV[regression][f_bracket])

    # P-value (KPSS: reject when stat is LARGE)
    pval = if kpss_stat >= cv[1]; T(0.001)
    elseif kpss_stat >= cv[5]; T(0.01) + (cv[1] - kpss_stat) / (cv[1] - cv[5]) * T(0.04)
    elseif kpss_stat >= cv[10]; T(0.05) + (cv[5] - kpss_stat) / (cv[5] - cv[10]) * T(0.05)
    else; T(0.20)
    end

    f_pval = _interp_f_pvalue(f_stat, f_cv)

    FourierKPSSResult(kpss_stat, pval, best_k, f_stat, f_pval, regression, cv, f_cv, bw, n)
end

# =============================================================================
# P-value Interpolation Helpers
# =============================================================================

function _interp_fourier_adf_pvalue(stat::T, cv::Dict{Int,T}) where {T}
    if stat <= cv[1]; return T(0.001)
    elseif stat <= cv[5]; return T(0.01) + (stat - cv[1]) / (cv[5] - cv[1]) * T(0.04)
    elseif stat <= cv[10]; return T(0.05) + (stat - cv[5]) / (cv[10] - cv[5]) * T(0.05)
    else; return T(0.20)
    end
end

function _interp_f_pvalue(stat::T, cv::Dict{Int,T}) where {T}
    if stat >= cv[1]; return T(0.001)
    elseif stat >= cv[5]; return T(0.01) + (cv[1] - stat) / (cv[1] - cv[5]) * T(0.04)
    elseif stat >= cv[10]; return T(0.05) + (cv[5] - stat) / (cv[5] - cv[10]) * T(0.05)
    else; return T(0.20)
    end
end

# =============================================================================
# Float64 Fallbacks
# =============================================================================

fourier_adf_test(y::AbstractVector; kwargs...) = fourier_adf_test(Float64.(y); kwargs...)
fourier_kpss_test(y::AbstractVector; kwargs...) = fourier_kpss_test(Float64.(y); kwargs...)
