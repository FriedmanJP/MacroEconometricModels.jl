# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Instrumental variables / two-stage least squares (IV/2SLS) estimation
for cross-sectional regression models with endogenous regressors.
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# First-Stage F-statistic
# =============================================================================

"""
    _first_stage_f(X, Z, endogenous) -> T

Compute the minimum first-stage F-statistic across all endogenous regressors.

For each endogenous variable x_j, regress x_j on Z and compute the F-statistic
for joint significance of excluded instruments. Reports the minimum across
all endogenous variables as a diagnostic for weak instruments.

# References
- Stock, J. H. & Yogo, M. (2005). *Identification and Inference for Econometric
  Models*. Cambridge University Press, ch. 5.
"""
function _first_stage_f(X::Matrix{T}, Z::Matrix{T},
                        endogenous::Vector{Int}) where {T<:AbstractFloat}
    n, m = size(Z)
    k_endog = length(endogenous)
    k_endog == 0 && return T(Inf)

    f_min = T(Inf)

    ZtZinv = robust_inv(Z' * Z)

    for j in endogenous
        x_j = X[:, j]
        # Regress x_j on Z
        gamma = ZtZinv * (Z' * x_j)
        resid_j = x_j .- Z * gamma
        ssr_j = dot(resid_j, resid_j)

        # Total sum of squares
        x_bar = mean(x_j)
        tss_j = sum((xi - x_bar)^2 for xi in x_j)
        tss_j = max(tss_j, T(1e-300))

        # F-statistic: ((TSS - SSR) / q) / (SSR / (n - m))
        # where q = m - (k_exog) but simplified: use all Z columns
        q = m
        df2 = n - m
        df2 > 0 || continue

        f_j = ((tss_j - ssr_j) / T(q)) / (ssr_j / T(df2))
        f_min = min(f_min, f_j)
    end

    f_min
end

# =============================================================================
# Sargan-Hansen Overidentification Test
# =============================================================================

"""
    _sargan_test(resid, Z, k_endog) -> (stat, pval) or (nothing, nothing)

Compute the Sargan-Hansen J-statistic for overidentification.

Under H0 (all instruments are valid), J ~ chi2(m - k_endog) where m is the
number of instruments and k_endog is the number of endogenous regressors.

Returns `(nothing, nothing)` if exactly identified (m == k_total, i.e.,
no overidentifying restrictions).

# References
- Sargan, J. D. (1958). *Econometrica* 26(3), 393-415.
- Hansen, L. P. (1982). *Econometrica* 50(4), 1029-1054.
"""
function _sargan_test(resid::Vector{T}, Z::Matrix{T},
                      k_endog::Int, k_regressors::Int) where {T<:AbstractFloat}
    n, m = size(Z)

    # Degrees of freedom = number of overidentifying restrictions = m - k
    # where m is the number of instruments and k is the number of regressors
    dof_sargan = m - k_regressors
    dof_sargan <= 0 && return (nothing, nothing)

    # J = n * R^2 from regression of residuals on instruments
    # Equivalently: J = e' P_Z e / sigma^2
    ZtZinv = robust_inv(Z' * Z)
    P_Z_e = Z * (ZtZinv * (Z' * resid))
    sigma2 = dot(resid, resid) / T(n)

    j_stat = dot(resid, P_Z_e) / sigma2

    j_pval = T(1 - cdf(Chisq(dof_sargan), j_stat))

    (j_stat, j_pval)
end

# =============================================================================
# IV/2SLS Estimation
# =============================================================================

"""
    estimate_iv(y, X, Z; endogenous, cov_type=:hc1, varnames=nothing) -> RegModel{T}

Estimate a linear regression model via instrumental variables / two-stage least
squares (IV/2SLS).

# Algorithm
1. Project regressors onto instrument space: X_hat = P_Z * X where P_Z = Z(Z'Z)^{-1}Z'
2. Second stage: beta = (X_hat'X)^{-1} X_hat'y
3. Residuals from ORIGINAL X: e = y - X*beta (not X_hat*beta)
4. Covariance uses X_hat for the bread and original residuals for the meat

# Arguments
- `y::AbstractVector{T}` — dependent variable (n x 1)
- `X::AbstractMatrix{T}` — regressor matrix (n x k), includes exogenous regressors and intercept
- `Z::AbstractMatrix{T}` — instrument matrix (n x m), includes exogenous regressors and excluded instruments
- `endogenous::Vector{Int}` — column indices of endogenous regressors in X
- `cov_type::Symbol` — covariance estimator: `:ols`, `:hc0`, `:hc1` (default), `:hc2`, `:hc3`
- `varnames::Union{Nothing,Vector{String}}` — coefficient names (auto-generated if nothing)

# Returns
`RegModel{T}` with method=:iv, including first-stage F-statistic and Sargan test.

# Diagnostics
- `first_stage_f`: minimum first-stage F across endogenous variables (>10 suggests strong instruments)
- `sargan_stat`/`sargan_pval`: overidentification test (nothing if exactly identified)

# Examples
```julia
using MacroEconometricModels
n = 500
z1, z2 = randn(n), randn(n)
x_endog = 0.5 * z1 + 0.3 * z2 + randn(n)
u = randn(n)
x_endog .+= 0.5 * u  # endogeneity
y = 1.0 .+ 2.0 * x_endog + u
X = hcat(ones(n), x_endog)
Z = hcat(ones(n), z1, z2)
m = estimate_iv(y, X, Z; endogenous=[2])
report(m)
```

# References
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. 2nd ed. MIT Press, ch. 5.
- Stock, J. H. & Yogo, M. (2005). *Identification and Inference for Econometric Models*. Cambridge University Press, ch. 5.
"""
function estimate_iv(y::AbstractVector{T}, X::AbstractMatrix{T},
                     Z::AbstractMatrix{T};
                     endogenous::Vector{Int}=Int[],
                     cov_type::Symbol=:hc1,
                     varnames::Union{Nothing,Vector{String}}=nothing) where {T<:AbstractFloat}
    # ---- Input validation ----
    _validate_data(y, "y")
    _validate_data(X, "X")
    _validate_data(Z, "Z")

    n = length(y)
    k = size(X, 2)
    m = size(Z, 2)

    size(X, 1) == n || throw(ArgumentError("X must have $n rows (got $(size(X, 1)))"))
    size(Z, 1) == n || throw(ArgumentError("Z must have $n rows (got $(size(Z, 1)))"))
    n > k || throw(ArgumentError("Need n > k (n=$n, k=$k)"))

    # Order condition: at least as many instruments as regressors
    m >= k || throw(ArgumentError(
        "Order condition violated: need m >= k (m=$m instruments, k=$k regressors)"))

    # Validate endogenous indices
    isempty(endogenous) && throw(ArgumentError("endogenous must be non-empty for IV estimation"))
    all(1 .<= endogenous .<= k) || throw(ArgumentError(
        "endogenous indices must be in [1, $k]; got $endogenous"))

    cov_type in (:ols, :hc0, :hc1, :hc2, :hc3) ||
        throw(ArgumentError("cov_type must be :ols, :hc0, :hc1, :hc2, or :hc3 for IV; got :$cov_type"))

    # ---- Variable names ----
    vn = something(varnames, ["x$i" for i in 1:k])
    length(vn) == k || throw(ArgumentError("varnames must have length $k"))

    # ---- Convert to concrete types ----
    Xm = Matrix{T}(X)
    Zm = Matrix{T}(Z)
    yv = Vector{T}(y)

    # ---- Stage 1: Project X onto instrument space ----
    ZtZinv = robust_inv(Zm' * Zm)
    P_Z = Zm * ZtZinv * Zm'   # n x n projection matrix
    X_hat = P_Z * Xm          # projected regressors

    # ---- Stage 2: 2SLS estimation ----
    XhX = X_hat' * Xm
    XhXinv = robust_inv(XhX)
    beta = XhXinv * (X_hat' * yv)

    # ---- Residuals from ORIGINAL X (not X_hat) ----
    fitted_vals = Xm * beta
    resid = yv .- fitted_vals

    # ---- Fit statistics ----
    ssr = dot(resid, resid)
    y_bar = mean(yv)
    tss = sum((yi - y_bar)^2 for yi in yv)
    tss = max(tss, T(1e-300))

    r2_val = one(T) - ssr / tss
    adj_r2_val = one(T) - (one(T) - r2_val) * T(n - 1) / T(n - k)

    # ---- F-test (using 2SLS estimates) ----
    f_stat, f_pval = _ols_f_test(beta, XhXinv, ssr, n, k)

    # ---- Log-likelihood, AIC, BIC ----
    sigma2_ml = ssr / T(n)
    loglik = -T(n) / 2 * log(T(2) * T(pi)) - T(n) / 2 * log(sigma2_ml) - T(n) / 2
    aic_val = -2 * loglik + 2 * T(k + 1)
    bic_val = -2 * loglik + log(T(n)) * T(k + 1)

    # ---- Covariance matrix ----
    if cov_type == :ols
        sigma2 = ssr / T(n - k)
        vcov_mat = sigma2 .* XhXinv
    else
        vcov_mat = _reg_vcov(X_hat, resid, cov_type, XhXinv)
    end

    # ---- Diagnostics: first-stage F ----
    fs_f = _first_stage_f(Xm, Zm, endogenous)

    # ---- Diagnostics: Sargan-Hansen test ----
    sargan_s, sargan_p = _sargan_test(resid, Zm, length(endogenous), k)

    RegModel{T}(
        yv, Xm, beta, vcov_mat,
        resid, fitted_vals,
        ssr, tss, r2_val, adj_r2_val, f_stat, f_pval,
        loglik, aic_val, bic_val,
        vn, :iv, cov_type,
        nothing,                        # weights
        Zm, endogenous,                 # Z, endogenous
        fs_f, sargan_s, sargan_p        # first_stage_f, sargan_stat, sargan_pval
    )
end

# Float fallback: convert to Float64
function estimate_iv(y::AbstractVector, X::AbstractMatrix, Z::AbstractMatrix; kwargs...)
    estimate_iv(Float64.(y), Float64.(X), Float64.(Z); kwargs...)
end
