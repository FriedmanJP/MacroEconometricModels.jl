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
Binary probit regression estimated via iteratively reweighted least squares (IRLS).
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# IRLS for Probit
# =============================================================================

"""
    _irls_probit(y, X; maxiter=100, tol=1e-8) -> (beta, mu, w, loglik, converged, iterations)

Iteratively reweighted least squares (Fisher scoring) for probit regression.

Uses the probit link: mu = Phi(X*beta), where Phi is the standard normal CDF.
Fisher scoring weights: w = phi(eta)^2 / (mu * (1 - mu)).
Working response: z = eta + (y - mu) / phi(eta).

# Returns
- `beta::Vector{T}` — estimated coefficients
- `mu::Vector{T}` — fitted probabilities Phi(X*beta)
- `w::Vector{T}` — final Fisher scoring weights
- `loglik::T` — maximized log-likelihood
- `converged::Bool` — whether convergence criterion was met
- `iterations::Int` — number of iterations performed
"""
function _irls_probit(y::Vector{T}, X::Matrix{T};
                      maxiter::Int=100, tol::T=T(1e-8)) where {T<:AbstractFloat}
    n, k = size(X)
    d = Normal(zero(T), one(T))

    # Initialize at zero
    beta = zeros(T, k)
    loglik_old = T(-Inf)
    converged = false
    iter = 0

    mu = Vector{T}(undef, n)
    phi_eta = Vector{T}(undef, n)
    w = Vector{T}(undef, n)

    for it in 1:maxiter
        iter = it

        # Linear predictor
        eta = X * beta

        # Mean response: Phi(eta), clamped for numerical safety
        @inbounds for i in 1:n
            mu[i] = clamp(T(cdf(d, eta[i])), T(1e-10), one(T) - T(1e-10))
            phi_eta[i] = max(T(pdf(d, eta[i])), T(1e-10))
        end

        # Log-likelihood
        loglik_new = zero(T)
        @inbounds for i in 1:n
            loglik_new += y[i] * log(mu[i]) + (one(T) - y[i]) * log(one(T) - mu[i])
        end

        # Check convergence
        if abs(loglik_new - loglik_old) < tol * (abs(loglik_old) + one(T))
            converged = true
            loglik_old = loglik_new
            break
        end
        loglik_old = loglik_new

        # Fisher scoring weights: w_i = phi(eta_i)^2 / (mu_i * (1 - mu_i))
        @inbounds for i in 1:n
            denom = mu[i] * (one(T) - mu[i])
            denom = max(denom, T(1e-10))
            w[i] = phi_eta[i]^2 / denom
            w[i] = max(w[i], T(1e-10))
        end

        # Weighted least squares: beta = (X'WX)^{-1} X'Wz
        # Working response: z = eta + (y - mu) / phi(eta)
        W = Diagonal(w)
        XtWX = X' * W * X
        XtWX_inv = robust_inv(XtWX)

        z = eta .+ (y .- mu) ./ phi_eta
        beta = XtWX_inv * (X' * W * z)
    end

    (beta, mu, w, loglik_old, converged, iter)
end

# =============================================================================
# Probit Estimation
# =============================================================================

"""
    estimate_probit(y, X; cov_type=:ols, varnames=nothing, clusters=nothing, maxiter=100, tol=1e-8) -> ProbitModel{T}

Estimate a binary probit regression model via maximum likelihood (IRLS/Fisher scoring).

# Algorithm
Uses iteratively reweighted least squares with the probit link
mu = Phi(X*beta), where Phi is the standard normal CDF. Converges when the change
in log-likelihood is below `tol * (|loglik| + 1)`.

# Arguments
- `y::AbstractVector{T}` — binary dependent variable (0/1), n x 1
- `X::AbstractMatrix{T}` — regressor matrix (n x k), should include a constant column
- `cov_type::Symbol` — covariance estimator: `:ols` (information matrix), `:hc0`, `:hc1`, `:hc2`, `:hc3`, `:cluster`
- `varnames::Union{Nothing,Vector{String}}` — coefficient names (auto-generated if nothing)
- `clusters::Union{Nothing,AbstractVector}` — cluster assignments (required for `:cluster`)
- `maxiter::Int` — maximum IRLS iterations (default 100)
- `tol` — convergence tolerance (default 1e-8)

# Returns
`ProbitModel{T}` with estimated coefficients, McFadden pseudo-R-squared, deviance residuals.

# Examples
```julia
using MacroEconometricModels, Random
rng = MersenneTwister(42)
n = 500
X = hcat(ones(n), randn(rng, n, 2))
beta_true = [0.0, 1.0, -0.8]
using Distributions
p = cdf.(Normal(), X * beta_true)
y = Float64.(rand(rng, n) .< p)
m = estimate_probit(y, X; varnames=["const", "x1", "x2"])
report(m)
```

# References
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. 2nd ed. MIT Press, ch. 15.
- Cameron, A. C. & Trivedi, P. K. (2005). *Microeconometrics*. Cambridge University Press, ch. 14.
"""
function estimate_probit(y::AbstractVector{T}, X::AbstractMatrix{T};
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
        throw(ArgumentError("y must be binary (0/1) for probit estimation"))

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
    beta, mu, w, loglik_val, converged, iterations = _irls_probit(yv, Xm;
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
        score_resid = yv .- mu
        vcov_mat = _reg_vcov(Xm, score_resid, cov_type, info_inv; clusters=clusters)
    end

    # ---- Deviance residuals (same formula as logit) ----
    dev_resid = _deviance_residuals(yv, mu)

    ProbitModel{T}(
        yv, Xm, beta, vcov_mat,
        dev_resid, mu,
        loglik_val, loglik_null, pseudo_r2,
        aic_val, bic_val,
        vn, converged, iterations, cov_type
    )
end

# Float fallback
function estimate_probit(y::AbstractVector, X::AbstractMatrix; kwargs...)
    estimate_probit(Float64.(y), Float64.(X); kwargs...)
end
