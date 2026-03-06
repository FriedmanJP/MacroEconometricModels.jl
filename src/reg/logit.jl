# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Binary logistic regression estimated via iteratively reweighted least squares (IRLS).
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# Deviance Residuals
# =============================================================================

"""
    _deviance_residuals(y, mu) -> Vector{T}

Compute deviance residuals for a binary response model.

    d_i = sign(y_i - mu_i) * sqrt(2 * |y_i * log(mu_i / y_i) + (1 - y_i) * log((1 - mu_i) / (1 - y_i))|)

For y_i = 0: d_i = -sqrt(2 * |log(1 - mu_i)|)
For y_i = 1: d_i = +sqrt(2 * |log(mu_i)|)
"""
function _deviance_residuals(y::Vector{T}, mu::Vector{T}) where {T<:AbstractFloat}
    n = length(y)
    d = Vector{T}(undef, n)
    @inbounds for i in 1:n
        mu_i = clamp(mu[i], T(1e-10), one(T) - T(1e-10))
        if y[i] == one(T)
            # d_i = sqrt(2 * (-log(mu_i)))
            d[i] = sqrt(2 * (-log(mu_i)))
        elseif y[i] == zero(T)
            # d_i = -sqrt(2 * (-log(1 - mu_i)))
            d[i] = -sqrt(2 * (-log(one(T) - mu_i)))
        else
            # General case (shouldn't occur for strict 0/1)
            term1 = y[i] > zero(T) ? y[i] * log(mu_i / y[i]) : zero(T)
            term2 = (one(T) - y[i]) > zero(T) ? (one(T) - y[i]) * log((one(T) - mu_i) / (one(T) - y[i])) : zero(T)
            d[i] = sign(y[i] - mu_i) * sqrt(abs(2 * (term1 + term2)))
        end
    end
    d
end

# =============================================================================
# IRLS for Logit
# =============================================================================

"""
    _irls_logit(y, X; maxiter=100, tol=1e-8) -> (beta, mu, w, loglik, converged, iterations)

Iteratively reweighted least squares (Fisher scoring) for logistic regression.

Uses the canonical logit link: mu = 1 / (1 + exp(-X*beta)).
Working response: z = eta + (y - mu) / w, where w = mu * (1 - mu).

# Returns
- `beta::Vector{T}` — estimated coefficients
- `mu::Vector{T}` — fitted probabilities
- `w::Vector{T}` — final IRLS weights mu*(1-mu)
- `loglik::T` — maximized log-likelihood
- `converged::Bool` — whether convergence criterion was met
- `iterations::Int` — number of iterations performed
"""
function _irls_logit(y::Vector{T}, X::Matrix{T};
                     maxiter::Int=100, tol::T=T(1e-8)) where {T<:AbstractFloat}
    n, k = size(X)

    # Initialize at zero
    beta = zeros(T, k)
    loglik_old = T(-Inf)
    converged = false
    iter = 0

    mu = Vector{T}(undef, n)
    w = Vector{T}(undef, n)

    for it in 1:maxiter
        iter = it

        # Linear predictor
        eta = X * beta

        # Mean response: logistic function, clamped for numerical safety
        @inbounds for i in 1:n
            mu[i] = clamp(one(T) / (one(T) + exp(-eta[i])), T(1e-10), one(T) - T(1e-10))
        end

        # Log-likelihood
        loglik_new = zero(T)
        @inbounds for i in 1:n
            loglik_new += y[i] * log(mu[i]) + (one(T) - y[i]) * log(one(T) - mu[i])
        end

        # Check convergence: |loglik_new - loglik_old| < tol * (|loglik_old| + 1)
        if abs(loglik_new - loglik_old) < tol * (abs(loglik_old) + one(T))
            converged = true
            loglik_old = loglik_new
            break
        end
        loglik_old = loglik_new

        # IRLS weights and working response
        @inbounds for i in 1:n
            w[i] = mu[i] * (one(T) - mu[i])
            w[i] = max(w[i], T(1e-10))  # floor for numerical stability
        end

        # Weighted least squares: beta = (X'WX)^{-1} X'Wz
        # where z = eta + (y - mu) / w
        W = Diagonal(w)
        XtWX = X' * W * X
        XtWX_inv = robust_inv(XtWX)

        # Working response
        z = eta .+ (y .- mu) ./ w
        beta = XtWX_inv * (X' * W * z)
    end

    (beta, mu, w, loglik_old, converged, iter)
end

# =============================================================================
# Logit Estimation
# =============================================================================

"""
    estimate_logit(y, X; cov_type=:ols, varnames=nothing, clusters=nothing, maxiter=100, tol=1e-8) -> LogitModel{T}

Estimate a binary logistic regression model via maximum likelihood (IRLS/Fisher scoring).

# Algorithm
Uses iteratively reweighted least squares with the canonical logit link
mu = 1/(1+exp(-X*beta)). Converges when the change in log-likelihood is below
`tol * (|loglik| + 1)`.

# Arguments
- `y::AbstractVector{T}` — binary dependent variable (0/1), n x 1
- `X::AbstractMatrix{T}` — regressor matrix (n x k), should include a constant column
- `cov_type::Symbol` — covariance estimator: `:ols` (information matrix), `:hc0`, `:hc1`, `:hc2`, `:hc3`, `:cluster`
- `varnames::Union{Nothing,Vector{String}}` — coefficient names (auto-generated if nothing)
- `clusters::Union{Nothing,AbstractVector}` — cluster assignments (required for `:cluster`)
- `maxiter::Int` — maximum IRLS iterations (default 100)
- `tol` — convergence tolerance (default 1e-8)

# Returns
`LogitModel{T}` with estimated coefficients, McFadden pseudo-R-squared, deviance residuals.

# Examples
```julia
using MacroEconometricModels, Random
rng = MersenneTwister(42)
n = 500
X = hcat(ones(n), randn(rng, n, 2))
beta_true = [0.0, 1.5, -1.0]
p = 1 ./ (1 .+ exp.(-X * beta_true))
y = Float64.(rand(rng, n) .< p)
m = estimate_logit(y, X; varnames=["const", "x1", "x2"])
report(m)
```

# References
- McCullagh, P. & Nelder, J. A. (1989). *Generalized Linear Models*. Chapman & Hall.
- Agresti, A. (2002). *Categorical Data Analysis*. 2nd ed. Wiley.
"""
function estimate_logit(y::AbstractVector{T}, X::AbstractMatrix{T};
                        cov_type::Symbol=:ols,
                        varnames::Union{Nothing,Vector{String}}=nothing,
                        clusters::Union{Nothing,AbstractVector}=nothing,
                        maxiter::Int=100,
                        tol::T=T(1e-8)) where {T<:AbstractFloat}
    # ---- Input validation ----
    _validate_data(y, "y")
    _validate_data(X, "X")

    n = length(y)
    k = size(X, 2)
    size(X, 1) == n || throw(ArgumentError("X must have $n rows (got $(size(X, 1)))"))
    n > k || throw(ArgumentError("Need n > k (n=$n, k=$k)"))

    # Check binary response
    yv = Vector{T}(y)
    all(yi -> yi == zero(T) || yi == one(T), yv) ||
        throw(ArgumentError("y must be binary (0/1) for logit estimation"))

    cov_type in (:ols, :hc0, :hc1, :hc2, :hc3, :cluster) ||
        throw(ArgumentError("cov_type must be :ols, :hc0, :hc1, :hc2, :hc3, or :cluster; got :$cov_type"))

    if cov_type == :cluster
        clusters === nothing && throw(ArgumentError("clusters required for :cluster cov_type"))
        length(clusters) == n || throw(ArgumentError("clusters must have length $n"))
    end

    # ---- Variable names ----
    vn = something(varnames, ["x$i" for i in 1:k])
    length(vn) == k || throw(ArgumentError("varnames must have length $k"))

    Xm = Matrix{T}(X)

    # ---- IRLS estimation ----
    beta, mu, w, loglik_val, converged, iterations = _irls_logit(yv, Xm;
                                                                  maxiter=maxiter, tol=tol)

    # ---- Null model log-likelihood ----
    p_bar = mean(yv)
    p_bar = clamp(p_bar, T(1e-10), one(T) - T(1e-10))
    loglik_null = T(n) * (p_bar * log(p_bar) + (one(T) - p_bar) * log(one(T) - p_bar))

    # ---- McFadden pseudo R-squared ----
    pseudo_r2 = one(T) - loglik_val / loglik_null

    # ---- AIC / BIC ----
    aic_val = -2 * loglik_val + 2 * T(k)
    bic_val = -2 * loglik_val + log(T(n)) * T(k)

    # ---- Covariance matrix ----
    W = Diagonal(w)
    XtWX = Xm' * W * Xm
    info_inv = robust_inv(XtWX)

    if cov_type == :ols
        vcov_mat = info_inv
    else
        # Sandwich: V = info_inv * S * info_inv
        # where S uses score residuals (y - mu) as the "residuals"
        score_resid = yv .- mu
        vcov_mat = _reg_vcov(Xm, score_resid, cov_type, info_inv; clusters=clusters)
    end

    # ---- Deviance residuals ----
    dev_resid = _deviance_residuals(yv, mu)

    LogitModel{T}(
        yv, Xm, beta, vcov_mat,
        dev_resid, mu,
        loglik_val, loglik_null, pseudo_r2,
        aic_val, bic_val,
        vn, converged, iterations, cov_type
    )
end

# Float fallback
function estimate_logit(y::AbstractVector, X::AbstractMatrix; kwargs...)
    estimate_logit(Float64.(y), Float64.(X); kwargs...)
end
