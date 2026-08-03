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

"""
    _mackinnon_pvalue(stat, regression) -> T

MacKinnon (1996) response-surface asymptotic p-value for the ADF/DF τ statistic (N=1).
Returns `p = Φ(P(τ))` where the polynomial link `P` is a quadratic for `τ ≤ τ*` and a
cubic above it, saturating to 0/1 outside the tabulated `[τ_min, τ_max]` range. The
Normal CDF here is the response-surface LINK, not a tail on the raw statistic (the
Dickey-Fuller limiting distribution is not Normal).
"""
function _mackinnon_pvalue(stat::T, regression::Symbol) where {T<:AbstractFloat}
    haskey(MACKINNON_PVAL_SMALLP, regression) ||
        throw(ArgumentError("regression must be :none, :constant, or :trend; got :$regression"))
    stat > MACKINNON_PVAL_TAUMAX[regression] && return one(T)
    stat < MACKINNON_PVAL_TAUMIN[regression] && return zero(T)
    poly = if stat <= MACKINNON_PVAL_TAUSTAR[regression]
        c = MACKINNON_PVAL_SMALLP[regression]
        c[1] + c[2] * stat + c[3] * stat^2
    else
        c = MACKINNON_PVAL_LARGEP[regression]
        c[1] + c[2] * stat + c[3] * stat^2 + c[4] * stat^3
    end
    T(cdf(Normal(), poly))
end

"""MacKinnon (1996) response-surface asymptotic p-value for the ADF/DF τ statistic
(also used by Phillips-Perron — same limiting distribution). `nobs`/`lags` are accepted
for call-site compatibility but unused: the surface is asymptotic (N=1)."""
function adf_pvalue(stat::T, regression::Symbol, nobs::Int, lags::Int=0) where {T<:AbstractFloat}
    _mackinnon_pvalue(stat, regression)
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

"""
Compute the optimal ADF augmentation lag by information criterion.

All candidate lags are scored on a SINGLE fixed sample of `(n-1)-max_lags` observations
(Ng & Perron 1995), so the AIC/BIC/HQIC are comparable across candidates — scoring each
`p` on its own `(n-1)-p` sample (dropping `p` leading rows) compares criteria across
different sample sizes and biases the choice toward too-few lags. The final ADF statistic
is then computed by `adf_test` on the selected lag's natural sample (statsmodels
`adfuller` convention).
"""
function adf_select_lags(y::AbstractVector{T}, max_lags::Int, regression::Symbol,
                         criterion::Symbol) where {T<:AbstractFloat}
    n = length(y)
    dy = diff(y)

    # Coefficient count for a candidate with p lagged differences (matches _build_adf_matrix)
    base = regression == :none ? 1 : regression == :constant ? 2 : 3
    ncoef(p) = base + p

    # Cap max_lags so the LARGEST model keeps ≥1 residual df on the fixed sample
    while max_lags > 0 && (n - 1 - max_lags) - ncoef(max_lags) < 1
        max_lags -= 1
    end
    nobs_fixed = n - 1 - max_lags
    nobs_fixed < 10 && return 0   # degenerate short series ⇒ no augmentation

    # Fixed dependent rows: dy observations (max_lags+1 : n-1), identical span for every p
    Y = dy[(max_lags+1):end]

    best_ic = T(Inf)
    best_lag = 0
    for p in 0:max_lags
        # Right-aligned design has (n-1)-p rows ending at dy[n-1]; drop the leading
        # (max_lags-p) rows so every candidate uses the same time span as Y.
        Xfull = _build_adf_matrix(y, dy, p, regression)
        ndrop = max_lags - p
        X = @view Xfull[(ndrop+1):end, :]

        k = size(X, 2)
        XtX = X'X
        det(XtX) ≈ 0 && continue
        B = XtX \ (X'Y)
        resid = Y - X * B
        sse = sum(abs2, resid)
        sigma2 = sse / (nobs_fixed - k)
        (!isfinite(sigma2) || sigma2 <= zero(T)) && continue

        ll = -nobs_fixed / 2 * (log(2π) + log(sigma2) + 1)
        ic = if criterion == :aic
            -2ll + 2k
        elseif criterion == :bic
            -2ll + k * log(nobs_fixed)
        else  # :hqic
            -2ll + 2k * log(log(nobs_fixed))
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

"""
    _interp_cv_row(table, Tgrid, n) -> NTuple{4,Float64}

Interpolate a simulated critical-value table (rows = `Tgrid`, columns =
1%, 2.5%, 5%, 10%) at sample size `n`, linearly in 1/T. Outside the grid the
end rows are used unchanged rather than extrapolated.
"""
function _interp_cv_row(table::Matrix{Float64}, Tgrid::NTuple{G,Int}, n::Int) where {G}
    row(i) = (table[i,1], table[i,2], table[i,3], table[i,4])
    n <= Tgrid[1] && return row(1)
    n >= Tgrid[G] && return row(G)
    j = 1
    while j < G - 1 && n > Tgrid[j+1]
        j += 1
    end
    x, x0, x1 = 1/n, 1/Tgrid[j], 1/Tgrid[j+1]
    w = (x - x1) / (x0 - x1)                      # w = 1 at Tgrid[j], 0 at Tgrid[j+1]
    a, b = row(j), row(j+1)
    ntuple(i -> w * a[i] + (1 - w) * b[i], 4)
end

"""Build the 1%/5%/10% dict carried by a unit-root result from a 4-quantile row."""
_cv_dict(row::NTuple{4,Float64}, ::Type{TF}) where {TF<:AbstractFloat} =
    Dict{Int,TF}(1 => TF(row[1]), 5 => TF(row[3]), 10 => TF(row[4]))

"""
    _lm_unitroot_cv_row(breaks, n, regression) -> NTuple{4,Float64}

Simulated (1%, 2.5%, 5%, 10%) critical values of the minimised LM statistic for
a series of length `n`, interpolated in 1/T. Every (breaks, regression) pair has
its own table: the no-break `:level` design is Schmidt-Phillips intercept-only,
`:both` adds the trend, and the break designs are Lee-Strazicich Models A and C.
See `LM_UNITROOT_SIM_CV` for how the tables were generated and what they are
conditional on.
"""
function _lm_unitroot_cv_row(breaks::Int, n::Int, regression::Symbol=:level)
    key = (breaks, regression == :both ? :both : :level)
    _interp_cv_row(LM_UNITROOT_SIM_CV[key], BREAK_TEST_SIM_T, n)
end

"""Compute LM unit root 1%/5%/10% critical values (see `_lm_unitroot_cv_row`)."""
_lm_unitroot_critical_values(breaks::Int, n::Int, regression::Symbol=:level,
                             ::Type{TF}=Float64) where {TF<:AbstractFloat} =
    _cv_dict(_lm_unitroot_cv_row(breaks, n, regression), TF)

"""
    _break_test_pvalue(stat, row) -> T

Piecewise-linear p-value for a minimised break-search statistic from its
simulated (1%, 2.5%, 5%, 10%) quantile row. Only the left tail is tabulated, so
the result saturates at 0.001 below the 1% point and is reported as 0.20 above
the 10% point.
"""
function _break_test_pvalue(stat::T, row::NTuple{4,Float64}) where {T<:AbstractFloat}
    c1, c25, c5, c10 = T(row[1]), T(row[2]), T(row[3]), T(row[4])
    seg(lo, hi, plo, phi) = plo + (stat - lo) / max(hi - lo, eps(T)) * (phi - plo)
    stat <= c1 && return T(0.001)
    stat <= c25 && return seg(c1, c25, T(0.01), T(0.025))
    stat <= c5 && return seg(c25, c5, T(0.025), T(0.05))
    stat <= c10 && return seg(c5, c10, T(0.05), T(0.10))
    return T(0.20)
end

"""Get sample bracket for Fourier critical values."""
function _fourier_sample_bracket(n::Int)
    n <= 150 ? 1 : n <= 349 ? 2 : n <= 500 ? 3 : 4
end

"""
    _adf_2break_cv_row(model, n) -> NTuple{4,Float64}

Simulated (1%, 2.5%, 5%, 10%) critical values of the minimised two-break ADF
statistic for a series of length `n`, interpolated in 1/T. See
`ADF_2BREAK_SIM_CV` for the generation parameters.
"""
_adf_2break_cv_row(model::Symbol, n::Int) =
    _interp_cv_row(ADF_2BREAK_SIM_CV[model == :both ? :both : :level],
                   BREAK_TEST_SIM_T, n)

"""Compute two-break ADF 1%/5%/10% critical values (see `_adf_2break_cv_row`)."""
_adf_2break_cv(model::Symbol, n::Int, ::Type{TF}=Float64) where {TF<:AbstractFloat} =
    _cv_dict(_adf_2break_cv_row(model, n), TF)

"""Get ERS Pt critical values by interpolating sample size."""
function _ers_pt_critical_values(regression::Symbol, nobs::Int, ::Type{TF}=Float64) where {TF<:AbstractFloat}
    table = ERS_PT_CV[regression]
    key = nobs <= 50 ? 50 : nobs <= 100 ? 100 : nobs <= 200 ? 200 : 500
    Dict{Int,TF}(k => TF(v) for (k, v) in table[key])
end

"""
Get HEGY seasonal unit-root critical values (EV-29). Returns a tuple
`(t_zero_cv, t_nyquist_cv, pair_F_cv)` of (1%,5%,10%) dicts for the given
`frequency` (4 or 12) and `deterministic` case. Published (HEGY 1990 quarterly /
Beaulieu-Miron 1993 monthly), not live-verified — see `critical_values.jl`.
"""
function _hegy_critical_values(frequency::Int, deterministic::Symbol,
                               ::Type{TF}=Float64) where {TF<:AbstractFloat}
    table = frequency == 4 ? HEGY_CV_QUARTERLY : HEGY_CV_MONTHLY
    # Fall back to the fullest deterministic case if an exotic one is requested.
    case = haskey(table, deterministic) ? deterministic : :const_trend_seas
    block = table[case]
    tz = Dict{Int,TF}(k => TF(v) for (k, v) in block[:t_zero])
    tn = Dict{Int,TF}(k => TF(v) for (k, v) in block[:t_nyquist])
    pf = Dict{Int,TF}(k => TF(v) for (k, v) in block[:pair_F])
    return (tz, tn, pf)
end
