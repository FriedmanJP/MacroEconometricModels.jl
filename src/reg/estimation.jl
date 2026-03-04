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
OLS and WLS estimation for cross-sectional linear regression models.
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# F-test Helper
# =============================================================================

"""
    _ols_f_test(beta, XtXinv, ssr, n, k) -> (f_stat, f_pval)

Joint significance F-test: H0: beta_2 = ... = beta_k = 0 (all slopes zero).

Uses the Wald form: F = (R beta)' [R V R']^{-1} (R beta) / q,
where R selects all non-intercept coefficients and q = k - 1.
"""
function _ols_f_test(beta::Vector{T}, XtXinv::Matrix{T},
                     ssr::T, n::Int, k::Int) where {T<:AbstractFloat}
    q = k - 1  # number of restrictions (all slopes = 0)
    q < 1 && return (zero(T), one(T))  # intercept-only model

    # Classical F-test: F = (ESS/q) / (SSR/(n-k))
    # where ESS = TSS - SSR, but we compute via Wald form for robustness
    sigma2 = ssr / T(n - k)

    # Select slope coefficients (indices 2:k, assuming first is intercept)
    R_beta = beta[2:k]
    R_V_R = sigma2 .* XtXinv[2:k, 2:k]

    f_stat = try
        dot(R_beta, robust_inv(R_V_R) * R_beta) / T(q)
    catch
        zero(T)
    end

    f_pval = if f_stat > zero(T) && isfinite(f_stat)
        df1 = q
        df2 = n - k
        df2 > 0 ? T(1 - cdf(FDist(df1, df2), f_stat)) : one(T)
    else
        one(T)
    end

    (f_stat, f_pval)
end

# =============================================================================
# OLS / WLS Estimation
# =============================================================================

"""
    estimate_reg(y, X; cov_type=:hc1, weights=nothing, varnames=nothing, clusters=nothing) -> RegModel{T}

Estimate a linear regression model via OLS or WLS.

OLS: beta = (X'X)^{-1} X'y
WLS: beta = (X'WX)^{-1} X'Wy where W = diag(weights)

# Arguments
- `y::AbstractVector{T}` — dependent variable (n x 1)
- `X::AbstractMatrix{T}` — regressor matrix (n x k), should include a constant column
- `cov_type::Symbol` — covariance estimator: `:ols`, `:hc0`, `:hc1` (default), `:hc2`, `:hc3`, `:cluster`
- `weights::Union{Nothing,AbstractVector}` — WLS weights (nothing for OLS)
- `varnames::Union{Nothing,Vector{String}}` — coefficient names (auto-generated if nothing)
- `clusters::Union{Nothing,AbstractVector}` — cluster assignments (required for `:cluster`)

# Returns
`RegModel{T}` with estimated coefficients, robust covariance matrix, fit statistics.

# Examples
```julia
using MacroEconometricModels
n = 200
X = hcat(ones(n), randn(n, 2))
beta_true = [1.0, 2.0, -0.5]
y = X * beta_true + 0.5 * randn(n)
m = estimate_reg(y, X)
report(m)
```

# References
- Greene, W. H. (2018). *Econometric Analysis*. 8th ed. Pearson.
- White, H. (1980). *Econometrica* 48(4), 817-838.
"""
function estimate_reg(y::AbstractVector{T}, X::AbstractMatrix{T};
                      cov_type::Symbol=:hc1,
                      weights::Union{Nothing,AbstractVector}=nothing,
                      varnames::Union{Nothing,Vector{String}}=nothing,
                      clusters::Union{Nothing,AbstractVector}=nothing) where {T<:AbstractFloat}
    # ---- Input validation ----
    _validate_data(y, "y")
    _validate_data(X, "X")

    n = length(y)
    k = size(X, 2)
    size(X, 1) == n || throw(ArgumentError("X must have $n rows (got $(size(X, 1)))"))
    n > k || throw(ArgumentError("Need n > k (n=$n, k=$k)"))

    cov_type in (:ols, :hc0, :hc1, :hc2, :hc3, :cluster) ||
        throw(ArgumentError("cov_type must be :ols, :hc0, :hc1, :hc2, :hc3, or :cluster; got :$cov_type"))

    if cov_type == :cluster
        clusters === nothing && throw(ArgumentError("clusters required for :cluster cov_type"))
        length(clusters) == n || throw(ArgumentError("clusters must have length $n"))
    end

    if weights !== nothing
        length(weights) == n || throw(ArgumentError("weights must have length $n"))
        all(w -> w > zero(T), weights) || throw(ArgumentError("All weights must be positive"))
    end

    # ---- Variable names ----
    vn = something(varnames, ["x$i" for i in 1:k])
    length(vn) == k || throw(ArgumentError("varnames must have length $k"))

    # ---- Estimation ----
    method = weights === nothing ? :ols : :wls

    if method == :wls
        sqrtW = Diagonal(sqrt.(Vector{T}(weights)))
        Xw = sqrtW * Matrix{T}(X)
        yw = sqrtW * Vector{T}(y)
        XtXinv = robust_inv(Xw' * Xw)
        beta = XtXinv * (Xw' * yw)
    else
        Xm = Matrix{T}(X)
        XtXinv = robust_inv(Xm' * Xm)
        beta = XtXinv * (Xm' * Vector{T}(y))
    end

    # ---- Residuals and fitted values ----
    fitted_vals = Vector{T}(X * beta)
    resid = Vector{T}(y) .- fitted_vals

    # ---- Fit statistics ----
    ssr = dot(resid, resid)
    y_bar = mean(y)
    tss = sum((yi - y_bar)^2 for yi in y)
    tss = max(tss, T(1e-300))  # avoid division by zero for constant y

    r2_val = one(T) - ssr / tss
    adj_r2_val = one(T) - (one(T) - r2_val) * T(n - 1) / T(n - k)

    # ---- F-test ----
    f_stat, f_pval = _ols_f_test(beta, XtXinv, ssr, n, k)

    # ---- Log-likelihood, AIC, BIC ----
    sigma2_ml = ssr / T(n)
    loglik = -T(n) / 2 * log(T(2) * T(pi)) - T(n) / 2 * log(sigma2_ml) - T(n) / 2
    aic_val = -2 * loglik + 2 * T(k + 1)   # +1 for sigma^2
    bic_val = -2 * loglik + log(T(n)) * T(k + 1)

    # ---- Covariance matrix ----
    # For WLS, use the original X and residuals for sandwich estimators
    X_for_vcov = Matrix{T}(X)
    if method == :wls
        # For HC estimators under WLS, use the transformed residuals and design
        # but report the original X residuals for interpretation
        XtXinv_vcov = robust_inv(X_for_vcov' * Diagonal(Vector{T}(weights)) * X_for_vcov)
        vcov_mat = _reg_vcov(X_for_vcov, resid, cov_type, XtXinv_vcov; clusters=clusters)
    else
        vcov_mat = _reg_vcov(X_for_vcov, resid, cov_type, XtXinv; clusters=clusters)
    end

    # ---- Weights ----
    w = weights === nothing ? nothing : Vector{T}(weights)

    RegModel{T}(
        Vector{T}(y), Matrix{T}(X), beta, vcov_mat,
        resid, fitted_vals,
        ssr, tss, r2_val, adj_r2_val, f_stat, f_pval,
        loglik, aic_val, bic_val,
        vn, method, cov_type, w,
        nothing, nothing,     # Z, endogenous (not IV)
        nothing, nothing, nothing  # first_stage_f, sargan_stat, sargan_pval
    )
end

# Float fallback: convert AbstractVector + AbstractMatrix to Float64
function estimate_reg(y::AbstractVector, X::AbstractMatrix; kwargs...)
    estimate_reg(Float64.(y), Float64.(X); kwargs...)
end
