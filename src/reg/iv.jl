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
    _partial_out(A, W) -> residuals of A on W

Residualize each column of `A` on the included-exogenous matrix `W` (M_W·A). Returns a
copy of `A` when `W` has no columns.
"""
function _partial_out(A::AbstractMatrix{T}, W::AbstractMatrix{T}) where {T<:AbstractFloat}
    size(W, 2) == 0 && return Matrix{T}(A)
    A .- W * (robust_inv(W' * W) * (W' * A))
end
_partial_out(a::AbstractVector{T}, W::AbstractMatrix{T}) where {T<:AbstractFloat} =
    vec(_partial_out(reshape(a, :, 1), W))

"""
    _first_stage_f(X, Z, endogenous) -> T

Minimum EXCLUDED-instrument partial first-stage F across the endogenous regressors.

For each endogenous x_j, partial out the included exogenous W = X[:, non-endogenous] and test
the joint significance of the excluded instruments:
    F_j = ((SSR_restricted − SSR_unrestricted) / q) / (SSR_unrestricted / (n − m)),
with SSR_restricted from x_j ~ W, SSR_unrestricted from x_j ~ Z (all m instruments), and
q = m − #W = number of excluded instruments. This is the correct weak-instrument diagnostic;
the previous overall-F (all of Z about the grand mean, q = m) mechanically inflated as the
included controls W predicted x_j.

# References
- Stock, J. H. & Yogo, M. (2005). *Identification and Inference for Econometric Models*, ch. 5.
"""
function _first_stage_f(X::Matrix{T}, Z::Matrix{T},
                        endogenous::Vector{Int}) where {T<:AbstractFloat}
    n, m = size(Z)
    k = size(X, 2)
    length(endogenous) == 0 && return T(Inf)

    incl = setdiff(1:k, endogenous)
    W = X[:, incl]
    n_incl = length(incl)
    q = m - n_incl                 # number of EXCLUDED instruments
    df2 = n - m
    (q <= 0 || df2 <= 0) && return T(NaN)

    ZtZinv = robust_inv(Z' * Z)
    f_min = T(Inf)
    for j in endogenous
        x_j = X[:, j]
        # Unrestricted: x_j on ALL instruments Z
        resid_u = x_j .- Z * (ZtZinv * (Z' * x_j))
        ssr_u = dot(resid_u, resid_u)
        # Restricted: x_j on the included exogenous W only
        resid_r = n_incl == 0 ? x_j : _partial_out(x_j, W)
        ssr_r = dot(resid_r, resid_r)

        f_j = ((ssr_r - ssr_u) / T(q)) / (ssr_u / T(df2))
        f_min = min(f_min, f_j)
    end
    f_min
end

"""
    _cragg_donald_f(X, Z, endogenous) -> T

Cragg-Donald (1993) weak-instrument F: the minimum generalized eigenvalue of the concentration
matrix, scaled by the number of excluded instruments q. Generalizes the partial first-stage F
to multiple endogenous regressors under homoskedasticity; for a single endogenous regressor it
equals `_first_stage_f`. Returns `NaN` if under-identified.
"""
function _cragg_donald_f(X::Matrix{T}, Z::Matrix{T},
                         endogenous::Vector{Int}) where {T<:AbstractFloat}
    n, m = size(Z); k = size(X, 2); k_endog = length(endogenous)
    k_endog == 0 && return T(NaN)
    incl = setdiff(1:k, endogenous)
    W = X[:, incl]; Xen = X[:, endogenous]
    q = m - length(incl); df2 = n - m
    (q < k_endog || df2 <= 0) && return T(NaN)

    Xt = _partial_out(Xen, W)                       # M_W · X_endog
    Zt = _partial_out(Z, W)                         # M_W · Z (rank q, so Zt'Zt is rank-deficient)
    PztXt = Zt * (robust_inv(Zt' * Zt; silent=true) * (Zt' * Xt))
    ESS = Matrix{T}(Xt' * PztXt)                     # explained SS (= SSR_r − SSR_u)
    Xhat = Z * (robust_inv(Z' * Z) * (Z' * Xen))
    Vr = Xen .- Xhat
    Sigma = Matrix{T}((Vr' * Vr) ./ T(df2))          # first-stage residual covariance
    lambda = eigvals(Symmetric(ESS), Symmetric(Sigma))
    minimum(real.(lambda)) / T(q)
end

"""
    _kleibergen_paap_f(X, Z, endogenous; cov_type=:hc1) -> T

Kleibergen-Paap (2006) rk Wald F: the heteroskedasticity-robust weak-instrument F. For a single
endogenous regressor this is exactly the HC-robust first-stage F (reduces to Cragg-Donald under
homoskedasticity); with several endogenous regressors the per-regressor minimum robust F is
reported. Returns `NaN` if under-identified.
"""
function _kleibergen_paap_f(X::Matrix{T}, Z::Matrix{T}, endogenous::Vector{Int};
                            cov_type::Symbol=:hc1) where {T<:AbstractFloat}
    n, m = size(Z); k = size(X, 2)
    length(endogenous) == 0 && return T(NaN)
    incl = setdiff(1:k, endogenous)
    W = X[:, incl]
    q = m - length(incl); df2 = n - m
    (q <= 0 || df2 <= 0) && return T(NaN)

    Zt = _partial_out(Z, W)
    U, S, _ = svd(Zt)
    tol = maximum(S) * n * eps(T)
    r = count(>(tol), S)
    r == 0 && return T(NaN)
    Zb = Matrix{T}(U[:, 1:r])                        # orthonormal basis of excluded space
    f_min = T(Inf)
    for j in endogenous
        xt = _partial_out(X[:, j], W)                # M_W · x_j
        gamma = Zb' * xt                             # Zb'Zb = I ⇒ OLS coefs
        ehat = xt .- Zb * gamma
        Vg = zeros(T, r, r)                           # robust cov: Σ_i Zb_i ê_i² Zb_iᵀ
        @inbounds for i in 1:n
            w2 = ehat[i]^2
            for a in 1:r, b in 1:r
                Vg[a, b] += Zb[i, a] * Zb[i, b] * w2
            end
        end
        cov_type == :hc1 && (Vg .*= T(n) / T(n - m))
        wald = dot(gamma, robust_inv(Symmetric(Vg)) * gamma)
        f_min = min(f_min, wald / T(r))
    end
    f_min
end

# Stock & Yogo (2005) Table 5.2 — 10% maximal-size 2SLS critical values, one endogenous
# regressor, by number of excluded instruments q. Returns `nothing` outside the tabulated range.
const _STOCK_YOGO_10PCT_1ENDOG = Dict(1 => 16.38, 2 => 19.93, 3 => 22.30, 4 => 24.58, 5 => 26.87)

function _stock_yogo_cv(n_endog::Int, q::Int)
    n_endog == 1 || return nothing
    get(_STOCK_YOGO_10PCT_1ENDOG, q, nothing)
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
                      k_endog::Int, k_regressors::Int, cov_type::Symbol) where {T<:AbstractFloat}
    n, m = size(Z)

    # Degrees of freedom = number of overidentifying restrictions = m - k
    # where m is the number of instruments and k is the number of regressors
    dof_sargan = m - k_regressors
    dof_sargan <= 0 && return (nothing, nothing)

    j_stat = if cov_type == :ols
        # Classical Sargan (homoskedastic): J = e'P_Z e / σ̂²  (= n·R² of e on Z)
        ZtZinv = robust_inv(Z' * Z)
        P_Z_e = Z * (ZtZinv * (Z' * resid))
        sigma2 = dot(resid, resid) / T(n)
        dot(resid, P_Z_e) / sigma2
    else
        # Robust Hansen J (HC0 moment-covariance meat, matching Stata ivreg2):
        # J = g' Ω̂⁻¹ g,  g = Z'e (length m),  Ω̂ = Σᵢ eᵢ² ZᵢZᵢ' = Z' diag(e²) Z.
        # No n/(n−m) or hc1 scaling — the small-sample factors cancel in N·ḡ'Ŵ⁻¹ḡ.
        g = Z' * resid
        S = Z' * Diagonal(resid .^ 2) * Z
        dot(g, robust_inv(S) * g)
    end

    j_pval = T(1 - cdf(Chisq(dof_sargan), max(j_stat, zero(T))))
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

    # ---- Weak-instrument diagnostics ----
    fs_f = _first_stage_f(Xm, Zm, endogenous)                       # excluded-instrument partial F
    cd_f = _cragg_donald_f(Xm, Zm, endogenous)                     # Cragg-Donald F
    kp_cov = cov_type == :ols ? :hc1 : cov_type
    kp_f = _kleibergen_paap_f(Xm, Zm, endogenous; cov_type=kp_cov)  # Kleibergen-Paap rk Wald F
    q_excl = m - (k - length(endogenous))
    sy = _stock_yogo_cv(length(endogenous), q_excl)                # Stock-Yogo 10% critical value
    cd_val = (cd_f === nothing || isnan(cd_f)) ? nothing : cd_f
    kp_val = (kp_f === nothing || isnan(kp_f)) ? nothing : kp_f
    sy_val = sy === nothing ? nothing : T(sy)

    # ---- Diagnostics: Sargan-Hansen test ----
    sargan_s, sargan_p = _sargan_test(resid, Zm, length(endogenous), k, cov_type)

    RegModel{T}(
        yv, Xm, beta, vcov_mat,
        resid, fitted_vals,
        ssr, tss, r2_val, adj_r2_val, f_stat, f_pval,
        loglik, aic_val, bic_val,
        vn, :iv, cov_type,
        nothing,                        # weights
        Zm, endogenous,                 # Z, endogenous
        fs_f, sargan_s, sargan_p,       # first_stage_f, sargan_stat, sargan_pval
        cd_val, kp_val, sy_val          # cragg_donald_f, kleibergen_paap_f, stock_yogo_10pct
    )
end

# Float fallback: convert to Float64
function estimate_iv(y::AbstractVector, X::AbstractMatrix, Z::AbstractMatrix; kwargs...)
    estimate_iv(Float64.(y), Float64.(X), Float64.(Z); kwargs...)
end
