# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Johansen cointegration test for VAR systems.
"""

using LinearAlgebra


# =============================================================================
# Deterministic Specification & Critical Values
# =============================================================================

function _parse_deterministic(d::Union{Symbol,Int})
    if d == 1 || d === :none
        return 1, :none
    elseif d == 2 || d === :constant || d === :restricted_constant ||
           d === :constant_restricted
        return 2, :constant
    elseif d == 3 || d === :unrestricted_constant ||
           d === :constant_unrestricted || d === :drift
        return 3, :unrestricted_constant
    elseif d == 4 || d === :trend || d === :restricted_trend ||
           d === :trend_restricted
        return 4, :trend
    elseif d == 5 || d === :unrestricted_trend || d === :trend_unrestricted
        return 5, :unrestricted_trend
    else
        throw(ArgumentError(
            "deterministic must be :none, :constant, :unrestricted_constant, " *
            ":trend, or :unrestricted_trend (got $d)"
        ))
    end
end

# Case 3 (:unrestricted_constant) — Osterwald-Lenum (1992, Table 1)
const _JOHANSEN_TRACE_CV_CASE3 = Dict(
    1 => [2.69, 3.84, 6.51],
    2 => [13.33, 15.41, 20.04],
    3 => [26.79, 29.68, 35.65],
    4 => [43.95, 47.21, 54.46],
    5 => [64.84, 68.52, 76.07],
    6 => [85.18, 90.39, 104.20],
    7 => [118.99, 124.25, 136.06],
    8 => [151.38, 157.11, 168.92],
    9 => [186.54, 192.89, 206.95],
    10 => [224.63, 231.26, 247.18],
)

const _JOHANSEN_MAX_CV_CASE3 = Dict(
    1 => [2.69, 3.84, 6.51],
    2 => [12.07, 14.07, 18.63],
    3 => [18.60, 20.97, 25.52],
    4 => [24.73, 27.07, 32.24],
    5 => [30.67, 33.46, 38.77],
    6 => [36.25, 39.43, 44.59],
    7 => [42.06, 45.28, 51.30],
    8 => [48.43, 51.42, 57.07],
    9 => [54.01, 57.12, 62.80],
    10 => [59.00, 62.81, 68.83],
)

# Case 5 (:unrestricted_trend) — Osterwald-Lenum (1992, Table 2)
const _JOHANSEN_TRACE_CV_CASE5 = Dict(
    1 => [2.71, 3.84, 6.63],
    2 => [16.14, 18.30, 22.81],
    3 => [32.15, 35.03, 40.87],
    4 => [51.60, 55.08, 62.01],
    5 => [75.00, 79.10, 87.17],
    6 => [102.14, 106.92, 116.31],
    7 => [132.80, 138.45, 149.20],
    8 => [167.03, 173.30, 185.12],
    9 => [204.81, 211.85, 224.96],
    10 => [245.98, 253.94, 268.42],
)

const _JOHANSEN_MAX_CV_CASE5 = Dict(
    1 => [2.71, 3.84, 6.63],
    2 => [14.07, 16.15, 20.85],
    3 => [20.25, 22.76, 27.84],
    4 => [26.41, 28.84, 34.02],
    5 => [32.40, 35.08, 40.35],
    6 => [38.30, 41.15, 46.80],
    7 => [44.20, 47.18, 53.05],
    8 => [50.15, 53.25, 59.20],
    9 => [56.00, 59.25, 65.40],
    10 => [61.80, 65.20, 71.50],
)

"""
    _johansen_moments(Y, p, deterministic) -> NamedTuple

Compute concentrated product-moment matrices S00, S11, S01 and concentrated
residuals R0, R1 for Johansen (1991, 1995) Cases 1 through 5.
"""
function _johansen_moments(Y::AbstractMatrix{T}, p::Int,
                           deterministic::Union{Symbol,Int}) where {T<:AbstractFloat}
    case, _ = _parse_deterministic(deterministic)
    T_obs = size(Y, 1)
    dY = diff(Y; dims=1)
    Y_lag = Y[p:end-1, :]
    T_eff = T_obs - p

    dY_lags = if p > 1
        hcat([dY[(p-j):(end-j), :] for j in 1:(p-1)]...)
    else
        Matrix{T}(undef, T_eff, 0)
    end
    dY_eff = dY[p:end, :]

    trend = T.(1:T_eff)
    const_col = ones(T, T_eff)

    if case == 1
        Z = dY_lags
        Y_lag_aug = Y_lag
    elseif case == 2
        Z = dY_lags
        Y_lag_aug = hcat(Y_lag, const_col)
    elseif case == 3
        Z = isempty(dY_lags) ? reshape(const_col, :, 1) : hcat(const_col, dY_lags)
        Y_lag_aug = Y_lag
    elseif case == 4
        Z = isempty(dY_lags) ? reshape(const_col, :, 1) : hcat(const_col, dY_lags)
        Y_lag_aug = hcat(Y_lag, trend)
    else  # case == 5
        Z_det = hcat(const_col, trend)
        Z = isempty(dY_lags) ? Z_det : hcat(Z_det, dY_lags)
        Y_lag_aug = Y_lag
    end

    if size(Z, 2) > 0
        R0 = dY_eff - Z * (Z \ dY_eff)
        R1 = Y_lag_aug - Z * (Z \ Y_lag_aug)
    else
        R0 = copy(dY_eff)
        R1 = copy(Y_lag_aug)
    end

    S00 = (R0' * R0) / T_eff
    S11 = (R1' * R1) / T_eff
    S01 = (R0' * R1) / T_eff

    return (; S00, S11, S01, Y_lag, Y_lag_aug, dY_lags, dY_eff, T_eff, R0, R1, Z, case)
end

"""
    _johansen_pvalue(stat, m, deterministic, test) -> p

Asymptotic p-value for the Johansen trace or maximum-eigenvalue statistic via
the Doornik (1998) gamma approximation: the asymptotic distribution for
`m = n - r` common trends under the null is approximated by
`Gamma(mean^2/var, var/mean)` with `mean` and `var` from case-specific response
surfaces, reproducing the MacKinnon-Haug-Michelis (1999) numerical
distributions to simulation accuracy.

Deterministic cases map to Doornik/gretl rows: `:none` -> no constant,
`:constant` -> restricted constant, `:trend` -> restricted trend (matching the
Osterwald-Lenum critical-value tables used for the rank decision).
"""
function _johansen_pvalue(stat::T, m::Int, deterministic::Union{Symbol,Int},
                          test::Symbol) where {T<:AbstractFloat}
    case, _ = _parse_deterministic(deterministic)
    x = (Float64(m)^2, Float64(m), 1.0,
         m == 1 ? 1.0 : 0.0, m == 2 ? 1.0 : 0.0, sqrt(Float64(m)))
    local mn::Float64, vr::Float64
    if test == :trace
        mn = sum(x[i] * JOHANSEN_TRACE_M_COEF[case, i] for i in 1:6)
        vr = sum(x[i] * JOHANSEN_TRACE_V_COEF[case, i] for i in 1:6)
    else
        mn = sum(x[i+1] * JOHANSEN_MAXEV_M_COEF[case, i] for i in 1:5)
        vr = sum(x[i+1] * JOHANSEN_MAXEV_V_COEF[case, i] for i in 1:5)
    end
    (isfinite(stat) && stat > zero(T)) || return one(T)
    p = 1.0 - cdf(Gamma(mn^2 / vr, vr / mn), Float64(stat))
    return clamp(T(p), zero(T), one(T))
end

"""
    johansen_test(Y, p; deterministic=:constant) -> JohansenResult

Johansen cointegration test for VAR system.

Tests for the number of cointegrating relationships among variables using
trace and maximum eigenvalue tests.

# Arguments
- `Y`: Data matrix (T × n)
- `p`: Number of lags in the VECM representation
- `deterministic`: Specification for deterministic terms
  - :none - No deterministic terms (Case 1)
  - :constant - Constant in cointegrating relation (Case 2, default)
  - :unrestricted_constant - Unrestricted constant in differences (Case 3)
  - :trend - Linear trend in cointegrating relation (Case 4)
  - :unrestricted_trend - Unrestricted linear trend in differences (Case 5)

# Returns
`JohansenResult` containing trace and max-eigenvalue statistics, cointegrating
vectors, adjustment coefficients, and estimated rank.
"""
function johansen_test(Y::AbstractMatrix{T}, p::Int;
                       deterministic::Union{Symbol,Int}=:constant,
                       significance::Real=0.05) where {T<:AbstractFloat}
    case, det_sym = _parse_deterministic(deterministic)

    T_obs, n = size(Y)
    T_obs < n + p + 10 && throw(ArgumentError(
        "Not enough observations for Johansen test"
    ))
    p < 1 && throw(ArgumentError("Number of lags p must be at least 1"))

    # Compute concentrated moments matching Johansen (1991, 1995) cases
    mom = _johansen_moments(Y, p, det_sym)
    T_eff = mom.T_eff
    Y_lag_aug = mom.Y_lag_aug
    S00 = mom.S00
    S11 = mom.S11
    S01 = mom.S01
    S10 = S01'

    # Solve generalized eigenvalue problem: |λS₁₁ - S₁₀S₀₀⁻¹S₀₁| = 0
    S00_inv = robust_inv(S00)
    A = S11 \ (S10 * S00_inv * S01)

    # Eigendecomposition
    eig = eigen(A)
    idx = sortperm(real.(eig.values), rev=true)
    eigenvalues_all = real.(eig.values[idx])
    eigenvectors_all = real.(eig.vectors[:, idx])

    # Canonical eigenvalues (top n)
    eigenvalues = clamp.(eigenvalues_all[1:n], 0, 1 - eps(T))

    # Test statistics
    trace_stats = Vector{T}(undef, n)
    max_eigen_stats = Vector{T}(undef, n)

    for r in 0:(n-1)
        trace_stats[r+1] = -T_eff * sum(log.(1 .- eigenvalues[(r+1):n]))
        max_eigen_stats[r+1] = -T_eff * log(1 - eigenvalues[r+1])
    end

    # Select critical values based on deterministic specification
    cv_trace_tbl, cv_max_tbl = if case == 1
        JOHANSEN_TRACE_CV_NONE, JOHANSEN_MAX_CV_NONE
    elseif case == 2
        JOHANSEN_TRACE_CV_CONSTANT, JOHANSEN_MAX_CV_CONSTANT
    elseif case == 3
        _JOHANSEN_TRACE_CV_CASE3, _JOHANSEN_MAX_CV_CASE3
    elseif case == 4
        JOHANSEN_TRACE_CV_TREND, JOHANSEN_MAX_CV_TREND
    else  # case == 5
        _JOHANSEN_TRACE_CV_CASE5, _JOHANSEN_MAX_CV_CASE5
    end

    cv_trace = Matrix{T}(undef, n, 3)
    cv_max = Matrix{T}(undef, n, 3)

    for r in 0:(n-1)
        n_minus_r = n - r
        if haskey(cv_trace_tbl, n_minus_r)
            cv_trace[r+1, :] = T.(cv_trace_tbl[n_minus_r])
            cv_max[r+1, :] = T.(cv_max_tbl[n_minus_r])
        else
            m_ct = n_minus_r
            x = (Float64(m_ct)^2, Float64(m_ct), 1.0,
                 m_ct == 1 ? 1.0 : 0.0, m_ct == 2 ? 1.0 : 0.0, sqrt(Float64(m_ct)))
            mn_tr = sum(x[i] * JOHANSEN_TRACE_M_COEF[case, i] for i in 1:6)
            vr_tr = sum(x[i] * JOHANSEN_TRACE_V_COEF[case, i] for i in 1:6)
            g_tr = Gamma(mn_tr^2 / vr_tr, vr_tr / mn_tr)
            cv_trace[r+1, :] = T.(quantile(g_tr, [0.90, 0.95, 0.99]))

            mn_mx = sum(x[i+1] * JOHANSEN_MAXEV_M_COEF[case, i] for i in 1:5)
            vr_mx = sum(x[i+1] * JOHANSEN_MAXEV_V_COEF[case, i] for i in 1:5)
            g_mx = Gamma(mn_mx^2 / vr_mx, vr_mx / mn_mx)
            cv_max[r+1, :] = T.(quantile(g_mx, [0.90, 0.95, 0.99]))
        end
    end

    trace_pvalues = Vector{T}(undef, n)
    max_pvalues = Vector{T}(undef, n)

    for r in 1:n
        m_ct = n - (r - 1)
        trace_pvalues[r] = _johansen_pvalue(trace_stats[r], m_ct, det_sym, :trace)
        max_pvalues[r] = _johansen_pvalue(max_eigen_stats[r], m_ct, det_sym, :max)
    end

    rank = _rank_from_trace(trace_stats, cv_trace, significance)

    r_eff = max(1, rank)
    beta_aug = eigenvectors_all[:, 1:r_eff]
    beta_norm = beta_aug * robust_inv(Matrix{T}(beta_aug[1:r_eff, :]))
    beta = Matrix{T}(beta_norm[1:n, :])
    alpha = Matrix{T}(S01 * beta_norm * robust_inv(Matrix{T}(beta_norm' * S11 * beta_norm)))

    return JohansenResult(
        trace_stats, trace_pvalues,
        max_eigen_stats, max_pvalues,
        rank, beta, alpha, eigenvalues,
        cv_trace, cv_max,
        det_sym, p, T_eff,
    )
end

johansen_test(Y::AbstractMatrix, p::Int; kwargs...) = johansen_test(Float64.(Y), p; kwargs...)

# Estimated cointegration rank = number of leading trace-test rejections (B1/T171).
# Rejecting H₀: rank ≤ r (trace stat > CV) implies rank ≥ r+1. Single source shared by
# johansen_test and estimate_vecm(rank=:auto) so the reported rank never disagrees.
function _rank_from_trace(trace_stats::AbstractVector, cv_trace::AbstractMatrix, significance::Real)
    cv_col = significance <= 0.01 ? 3 : significance <= 0.05 ? 2 : 1
    r = 0
    for i in 0:(length(trace_stats) - 1)
        if trace_stats[i+1] > cv_trace[i+1, cv_col]
            r = i + 1
        else
            break
        end
    end
    r
end

_select_rank_trace(joh::JohansenResult, significance::Real) =
    _rank_from_trace(joh.trace_stats, joh.critical_values_trace, significance)
