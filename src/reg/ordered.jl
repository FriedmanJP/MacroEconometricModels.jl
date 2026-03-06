# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Ordered logit and probit regression estimated via Newton-Raphson MLE.

Implements the cumulative link model:

    P(y <= j | x) = F(alpha_j - x' beta)

where F is the logistic CDF (logit) or standard normal CDF (probit),
alpha_1 < alpha_2 < ... < alpha_{J-1} are the cutpoints (thresholds),
and beta are the slope coefficients.

No intercept should appear in X -- it is absorbed by the cutpoints.
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# Type Definitions
# =============================================================================

"""
    OrderedLogitModel{T} <: StatsAPI.RegressionModel

Ordered logistic regression model estimated via maximum likelihood.

# Fields
- `y::Vector{Int}` -- dependent variable (remapped to 1:J)
- `X::Matrix{T}` -- regressor matrix (no intercept)
- `beta::Vector{T}` -- slope coefficients (K)
- `cutpoints::Vector{T}` -- cutpoints/thresholds (J-1)
- `vcov_mat::Matrix{T}` -- joint vcov of [beta; cutpoints] (K+J-1 x K+J-1)
- `fitted::Matrix{T}` -- predicted probabilities (n x J)
- `loglik::T` -- maximized log-likelihood
- `loglik_null::T` -- null model log-likelihood (cutpoints only)
- `pseudo_r2::T` -- McFadden's pseudo R-squared
- `aic::T` -- Akaike information criterion
- `bic::T` -- Bayesian information criterion
- `varnames::Vector{String}` -- coefficient names
- `categories::Vector` -- original category values
- `converged::Bool` -- whether optimization converged
- `iterations::Int` -- number of iterations performed
- `cov_type::Symbol` -- covariance estimator

# References
- McCullagh, P. (1980). *JRSS B* 42(2), 109-142.
- Agresti, A. (2010). *Analysis of Ordinal Categorical Data*. 2nd ed. Wiley.
"""
struct OrderedLogitModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    y::Vector{Int}
    X::Matrix{T}
    beta::Vector{T}
    cutpoints::Vector{T}
    vcov_mat::Matrix{T}
    fitted::Matrix{T}
    loglik::T
    loglik_null::T
    pseudo_r2::T
    aic::T
    bic::T
    varnames::Vector{String}
    categories::Vector
    converged::Bool
    iterations::Int
    cov_type::Symbol
end

"""
    OrderedProbitModel{T} <: StatsAPI.RegressionModel

Ordered probit regression model estimated via maximum likelihood.

Same fields as `OrderedLogitModel{T}`, using the standard normal CDF
as the link function.

# References
- McCullagh, P. (1980). *JRSS B* 42(2), 109-142.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. 2nd ed. MIT Press.
"""
struct OrderedProbitModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    y::Vector{Int}
    X::Matrix{T}
    beta::Vector{T}
    cutpoints::Vector{T}
    vcov_mat::Matrix{T}
    fitted::Matrix{T}
    loglik::T
    loglik_null::T
    pseudo_r2::T
    aic::T
    bic::T
    varnames::Vector{String}
    categories::Vector
    converged::Bool
    iterations::Int
    cov_type::Symbol
end

const OrderedModel{T} = Union{OrderedLogitModel{T}, OrderedProbitModel{T}}

# =============================================================================
# Link Functions
# =============================================================================

# Logistic CDF and PDF
_logistic_cdf(::Type{T}, x::T) where {T<:AbstractFloat} = one(T) / (one(T) + exp(-x))
_logistic_pdf(::Type{T}, x::T) where {T<:AbstractFloat} = begin
    p = _logistic_cdf(T, x)
    p * (one(T) - p)
end

# Normal CDF and PDF
_normal_cdf(::Type{T}, x::T) where {T<:AbstractFloat} = T(cdf(Normal(zero(T), one(T)), x))
_normal_pdf(::Type{T}, x::T) where {T<:AbstractFloat} = T(pdf(Normal(zero(T), one(T)), x))

# =============================================================================
# Category Probabilities
# =============================================================================

"""
    _ordered_probs(alpha, xb, J, F_cdf) -> Vector{T}

Compute P(y = j | x) for j = 1, ..., J given cutpoints alpha (J-1),
linear predictor xb = x'beta, and CDF function F_cdf.

P(y = j) = F(alpha_j - xb) - F(alpha_{j-1} - xb)
with F(alpha_0 - xb) = 0 and F(alpha_J - xb) = 1.
"""
function _ordered_probs(alpha::Vector{T}, xb::T, J::Int,
                        F_cdf::Function) where {T<:AbstractFloat}
    probs = Vector{T}(undef, J)
    eps_floor = T(1e-15)
    # P(y = 1) = F(alpha_1 - xb)
    F_prev = zero(T)
    @inbounds for j in 1:(J-1)
        F_cur = F_cdf(T, alpha[j] - xb)
        probs[j] = max(F_cur - F_prev, eps_floor)
        F_prev = F_cur
    end
    # P(y = J) = 1 - F(alpha_{J-1} - xb)
    @inbounds probs[J] = max(one(T) - F_prev, eps_floor)
    probs
end

# =============================================================================
# Log-Likelihood, Gradient, and Hessian
# =============================================================================

"""
    _ordered_loglik_score_hessian(y, X, beta, alpha, J, F_cdf, F_pdf)

Compute log-likelihood, score (gradient), and Hessian for ordered model.
Parameter vector is theta = [beta; alpha].
"""
function _ordered_loglik_score_hessian(
        y::Vector{Int}, X::Matrix{T}, beta::Vector{T}, alpha::Vector{T},
        J::Int, F_cdf::Function, F_pdf::Function) where {T<:AbstractFloat}
    n, K = size(X)
    Jm1 = J - 1
    P = K + Jm1  # total parameters

    loglik = zero(T)
    score = zeros(T, P)
    H = zeros(T, P, P)

    eps_floor = T(1e-15)

    @inbounds for i in 1:n
        xi = @view X[i, :]
        xb = dot(xi, beta)
        j = y[i]

        # Cumulative probabilities at boundaries
        # F_upper = F(alpha_j - xb), F_lower = F(alpha_{j-1} - xb)
        if j == 1
            F_lower = zero(T)
            f_lower = zero(T)
            F_upper = F_cdf(T, alpha[1] - xb)
            f_upper = F_pdf(T, alpha[1] - xb)
        elseif j == J
            F_lower = F_cdf(T, alpha[Jm1] - xb)
            f_lower = F_pdf(T, alpha[Jm1] - xb)
            F_upper = one(T)
            f_upper = zero(T)
        else
            F_lower = F_cdf(T, alpha[j-1] - xb)
            f_lower = F_pdf(T, alpha[j-1] - xb)
            F_upper = F_cdf(T, alpha[j] - xb)
            f_upper = F_pdf(T, alpha[j] - xb)
        end

        p_ij = max(F_upper - F_lower, eps_floor)
        loglik += log(p_ij)

        # Score contributions
        # d log p / d beta = -(f_upper - f_lower) / p * x  (note the sign from chain rule)
        # d log p / d alpha_j = f_upper / p  (if j < J, for the upper boundary)
        # d log p / d alpha_{j-1} = -f_lower / p  (if j > 1, for the lower boundary)
        dp_dbeta = -(f_upper - f_lower)  # dp/d(xb) = -(f_upper - f_lower)
        score_beta_factor = dp_dbeta / p_ij

        # Score for beta: d logp / d beta = score_beta_factor * x
        for k in 1:K
            score[k] += score_beta_factor * xi[k]
        end

        # Score for cutpoints
        if j < J
            # d logp / d alpha_j = f_upper / p_ij
            score[K + j] += f_upper / p_ij
        end
        if j > 1
            # d logp / d alpha_{j-1} = -f_lower / p_ij
            score[K + j - 1] += -f_lower / p_ij
        end
    end

    # Use BHHH (outer product of gradients) approximation for Hessian
    # Recompute per-observation scores for BHHH
    H .= zero(T)
    scores_i = zeros(T, P)
    @inbounds for i in 1:n
        xi = @view X[i, :]
        xb = dot(xi, beta)
        j = y[i]

        if j == 1
            F_lower = zero(T)
            f_lower = zero(T)
            F_upper = F_cdf(T, alpha[1] - xb)
            f_upper = F_pdf(T, alpha[1] - xb)
        elseif j == J
            F_lower = F_cdf(T, alpha[Jm1] - xb)
            f_lower = F_pdf(T, alpha[Jm1] - xb)
            F_upper = one(T)
            f_upper = zero(T)
        else
            F_lower = F_cdf(T, alpha[j-1] - xb)
            f_lower = F_pdf(T, alpha[j-1] - xb)
            F_upper = F_cdf(T, alpha[j] - xb)
            f_upper = F_pdf(T, alpha[j] - xb)
        end

        p_ij = max(F_upper - F_lower, eps_floor)

        scores_i .= zero(T)

        dp_dbeta = -(f_upper - f_lower)
        sbf = dp_dbeta / p_ij
        for k in 1:K
            scores_i[k] = sbf * xi[k]
        end
        if j < J
            scores_i[K + j] = f_upper / p_ij
        end
        if j > 1
            scores_i[K + j - 1] += -f_lower / p_ij
        end

        # Outer product: H -= s_i * s_i' (negative because we maximize)
        for a in 1:P
            for b in 1:P
                H[a, b] -= scores_i[a] * scores_i[b]
            end
        end
    end

    (loglik, score, H)
end

# =============================================================================
# Newton-Raphson Estimation
# =============================================================================

"""
    _nr_ordered(y, X, J, F_cdf, F_pdf; maxiter=200, tol=1e-8)

Newton-Raphson optimization for ordered logit/probit.
Returns (beta, alpha, loglik, converged, iterations).
"""
function _nr_ordered(y::Vector{Int}, X::Matrix{T}, J::Int,
                     F_cdf::Function, F_pdf::Function;
                     maxiter::Int=200, tol::T=T(1e-8)) where {T<:AbstractFloat}
    n, K = size(X)
    Jm1 = J - 1
    P = K + Jm1

    # Initialize beta at zero, cutpoints evenly spaced
    beta = zeros(T, K)
    alpha = collect(range(T(-1), T(1), length=Jm1))

    loglik_old = T(-Inf)
    converged = false
    iter = 0

    for it in 1:maxiter
        iter = it
        loglik_val, score, H = _ordered_loglik_score_hessian(
            y, X, beta, alpha, J, F_cdf, F_pdf)

        # Check convergence
        if abs(loglik_val - loglik_old) < tol * (abs(loglik_old) + one(T))
            converged = true
            loglik_old = loglik_val
            break
        end
        loglik_old = loglik_val

        # Newton step: theta_new = theta_old - H^{-1} * score
        H_inv = robust_inv(Hermitian(H))
        H_inv = Matrix{T}(H_inv)
        delta = H_inv * score

        # Update parameters
        beta .-= delta[1:K]
        alpha .-= delta[K+1:P]

        # Enforce cutpoint ordering
        for j in 2:Jm1
            if alpha[j] <= alpha[j-1]
                alpha[j] = alpha[j-1] + T(1e-4)
            end
        end
    end

    (beta, alpha, loglik_old, converged, iter)
end

# =============================================================================
# Null Model Log-Likelihood
# =============================================================================

"""
    _ordered_null_loglik(y, J)

Log-likelihood of the null model (cutpoints only, no covariates).
P(y = j) = n_j / n for each category.
"""
function _ordered_null_loglik(y::Vector{Int}, J::Int, ::Type{T}) where {T<:AbstractFloat}
    n = length(y)
    ll = zero(T)
    for j in 1:J
        n_j = count(==(j), y)
        if n_j > 0
            ll += T(n_j) * log(T(n_j) / T(n))
        end
    end
    ll
end

# =============================================================================
# Score Matrix (for sandwich covariance)
# =============================================================================

"""
    _ordered_score_matrix(y, X, beta, alpha, J, F_cdf, F_pdf) -> Matrix{T}

Compute n x P matrix of per-observation score vectors.
"""
function _ordered_score_matrix(
        y::Vector{Int}, X::Matrix{T}, beta::Vector{T}, alpha::Vector{T},
        J::Int, F_cdf::Function, F_pdf::Function) where {T<:AbstractFloat}
    n, K = size(X)
    Jm1 = J - 1
    P = K + Jm1
    eps_floor = T(1e-15)

    S = zeros(T, n, P)

    @inbounds for i in 1:n
        xi = @view X[i, :]
        xb = dot(xi, beta)
        j = y[i]

        if j == 1
            f_lower = zero(T)
            F_upper = F_cdf(T, alpha[1] - xb)
            f_upper = F_pdf(T, alpha[1] - xb)
            F_lower = zero(T)
        elseif j == J
            F_lower = F_cdf(T, alpha[Jm1] - xb)
            f_lower = F_pdf(T, alpha[Jm1] - xb)
            f_upper = zero(T)
            F_upper = one(T)
        else
            F_lower = F_cdf(T, alpha[j-1] - xb)
            f_lower = F_pdf(T, alpha[j-1] - xb)
            F_upper = F_cdf(T, alpha[j] - xb)
            f_upper = F_pdf(T, alpha[j] - xb)
        end

        p_ij = max(F_upper - F_lower, eps_floor)

        dp_dbeta = -(f_upper - f_lower)
        sbf = dp_dbeta / p_ij
        for k in 1:K
            S[i, k] = sbf * xi[k]
        end
        if j < J
            S[i, K + j] += f_upper / p_ij
        end
        if j > 1
            S[i, K + j - 1] += -f_lower / p_ij
        end
    end

    S
end

# =============================================================================
# Estimation Functions
# =============================================================================

"""
    estimate_ologit(y, X; cov_type=:ols, varnames=nothing, clusters=nothing, maxiter=200, tol=1e-8) -> OrderedLogitModel{T}

Estimate an ordered logistic regression model via maximum likelihood (Newton-Raphson).

# Model
Cumulative link model: P(y <= j | x) = Logistic(alpha_j - x' beta).
X should NOT include an intercept column -- it is absorbed into cutpoints.

# Arguments
- `y::AbstractVector` -- ordinal dependent variable (will be remapped to 1:J)
- `X::AbstractMatrix{T}` -- regressor matrix (n x K, no intercept)
- `cov_type::Symbol` -- covariance estimator: `:ols` (MLE), `:hc1` (sandwich)
- `varnames::Union{Nothing,Vector{String}}` -- coefficient names (auto-generated if nothing)
- `clusters::Union{Nothing,AbstractVector}` -- cluster assignments (for `:cluster`)
- `maxiter::Int` -- maximum Newton-Raphson iterations (default 200)
- `tol` -- convergence tolerance (default 1e-8)

# Returns
`OrderedLogitModel{T}` with estimated coefficients, cutpoints, and joint vcov.

# Examples
```julia
using MacroEconometricModels, Random, Distributions
rng = MersenneTwister(42)
n = 1000
X = randn(rng, n, 2)
xb = X * [1.0, -0.5]
p = 1 ./ (1 .+ exp.(-([0.0 1.5]' .- xb)))
u = rand(rng, n)
y = [u[i] < p[1,i] ? 1 : u[i] < p[2,i] ? 2 : 3 for i in 1:n]
m = estimate_ologit(y, X; varnames=["x1", "x2"])
report(m)
```

# References
- McCullagh, P. (1980). *JRSS B* 42(2), 109-142.
- Agresti, A. (2010). *Analysis of Ordinal Categorical Data*. 2nd ed. Wiley.
"""
function estimate_ologit(y::AbstractVector, X::AbstractMatrix{T};
                         cov_type::Symbol=:ols,
                         varnames::Union{Nothing,Vector{String}}=nothing,
                         clusters::Union{Nothing,AbstractVector}=nothing,
                         maxiter::Int=200,
                         tol::T=T(1e-8)) where {T<:AbstractFloat}
    _estimate_ordered(y, X, :logit; cov_type=cov_type, varnames=varnames,
                      clusters=clusters, maxiter=maxiter, tol=tol)
end

"""
    estimate_oprobit(y, X; cov_type=:ols, varnames=nothing, clusters=nothing, maxiter=200, tol=1e-8) -> OrderedProbitModel{T}

Estimate an ordered probit regression model via maximum likelihood (Newton-Raphson).

# Model
Cumulative link model: P(y <= j | x) = Phi(alpha_j - x' beta).
X should NOT include an intercept column -- it is absorbed into cutpoints.

# Arguments
Same as `estimate_ologit`.

# Returns
`OrderedProbitModel{T}` with estimated coefficients, cutpoints, and joint vcov.

# Examples
```julia
using MacroEconometricModels, Random, Distributions
rng = MersenneTwister(42)
n = 1000
X = randn(rng, n, 2)
xb = X * [0.8, -0.5]
d = Normal()
p = cdf.(d, [0.0 1.0]' .- xb)
u = rand(rng, n)
y = [u[i] < p[1,i] ? 1 : u[i] < p[2,i] ? 2 : 3 for i in 1:n]
m = estimate_oprobit(y, X; varnames=["x1", "x2"])
report(m)
```

# References
- McCullagh, P. (1980). *JRSS B* 42(2), 109-142.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. 2nd ed. MIT Press.
"""
function estimate_oprobit(y::AbstractVector, X::AbstractMatrix{T};
                          cov_type::Symbol=:ols,
                          varnames::Union{Nothing,Vector{String}}=nothing,
                          clusters::Union{Nothing,AbstractVector}=nothing,
                          maxiter::Int=200,
                          tol::T=T(1e-8)) where {T<:AbstractFloat}
    _estimate_ordered(y, X, :probit; cov_type=cov_type, varnames=varnames,
                      clusters=clusters, maxiter=maxiter, tol=tol)
end

"""Internal: common estimation logic for ordered logit/probit."""
function _estimate_ordered(y::AbstractVector, X::AbstractMatrix{T}, link::Symbol;
                           cov_type::Symbol=:ols,
                           varnames::Union{Nothing,Vector{String}}=nothing,
                           clusters::Union{Nothing,AbstractVector}=nothing,
                           maxiter::Int=200,
                           tol::T=T(1e-8)) where {T<:AbstractFloat}
    # ---- Input validation ----
    _validate_data(X, "X")

    n = length(y)
    K = size(X, 2)
    size(X, 1) == n || throw(ArgumentError("X must have $n rows (got $(size(X, 1)))"))

    cov_type in (:ols, :hc0, :hc1, :hc2, :hc3, :cluster) ||
        throw(ArgumentError("cov_type must be :ols, :hc0, :hc1, :hc2, :hc3, or :cluster; got :$cov_type"))

    if cov_type == :cluster
        clusters === nothing && throw(ArgumentError("clusters required for :cluster cov_type"))
        length(clusters) == n || throw(ArgumentError("clusters must have length $n"))
    end

    # ---- Category remapping ----
    cats = sort(unique(y))
    J = length(cats)
    J >= 3 || throw(ArgumentError("Need at least 3 categories for ordered model (got $J)"))
    n > K + J - 1 || throw(ArgumentError("Need n > K + J - 1 (n=$n, K=$K, J=$J)"))

    cat_map = Dict(cats[j] => j for j in 1:J)
    yint = [cat_map[yi] for yi in y]

    # ---- Variable names ----
    vn = something(varnames, ["x$i" for i in 1:K])
    length(vn) == K || throw(ArgumentError("varnames must have length $K"))

    Xm = Matrix{T}(X)

    # ---- Select link functions ----
    F_cdf, F_pdf = if link == :logit
        (_logistic_cdf, _logistic_pdf)
    else
        (_normal_cdf, _normal_pdf)
    end

    # ---- Newton-Raphson estimation ----
    beta, alpha, loglik_val, converged, iterations = _nr_ordered(
        yint, Xm, J, F_cdf, F_pdf; maxiter=maxiter, tol=tol)

    # ---- Null model log-likelihood ----
    loglik_null = _ordered_null_loglik(yint, J, T)

    # ---- McFadden pseudo R-squared ----
    pseudo_r2 = one(T) - loglik_val / loglik_null

    # ---- AIC / BIC ----
    P = K + J - 1
    aic_val = -2 * loglik_val + 2 * T(P)
    bic_val = -2 * loglik_val + log(T(n)) * T(P)

    # ---- Covariance matrix ----
    if cov_type == :ols
        # Classical MLE: V = -H^{-1} (BHHH approximation)
        _, _, H = _ordered_loglik_score_hessian(
            yint, Xm, beta, alpha, J, F_cdf, F_pdf)
        vcov_mat = Matrix{T}(robust_inv(Hermitian(-H)))
    else
        # Sandwich estimator: V = H^{-1} S H^{-1}
        # where S = sum s_i s_i' (with HC adjustments)
        _, _, H = _ordered_loglik_score_hessian(
            yint, Xm, beta, alpha, J, F_cdf, F_pdf)
        H_inv = Matrix{T}(robust_inv(Hermitian(-H)))

        S_mat = _ordered_score_matrix(yint, Xm, beta, alpha, J, F_cdf, F_pdf)

        if cov_type == :cluster
            # Cluster-robust
            unique_clusters = unique(clusters)
            G = length(unique_clusters)
            B = zeros(T, P, P)
            for g in unique_clusters
                idx = findall(==(g), clusters)
                sg = vec(sum(S_mat[idx, :], dims=1))
                B .+= sg * sg'
            end
            correction = T(G) / T(G - 1) * T(n - 1) / T(n - P)
            B .*= correction
            vcov_mat = H_inv * B * H_inv
        else
            # HC variants
            B = zeros(T, P, P)
            for i in 1:n
                si = @view S_mat[i, :]
                omega = if cov_type == :hc0
                    one(T)
                elseif cov_type == :hc1
                    one(T)
                else
                    one(T)  # HC2/HC3 not well-defined for MLE; treat as HC0
                end
                B .+= omega .* (si * si')
            end
            if cov_type == :hc1
                B .*= T(n) / T(n - P)
            end
            vcov_mat = H_inv * B * H_inv
        end
    end

    # ---- Fitted probabilities ----
    fitted_probs = Matrix{T}(undef, n, J)
    @inbounds for i in 1:n
        xb = dot(@view(Xm[i, :]), beta)
        fitted_probs[i, :] .= _ordered_probs(alpha, xb, J, F_cdf)
    end

    # ---- Construct result ----
    if link == :logit
        OrderedLogitModel{T}(
            yint, Xm, beta, alpha, vcov_mat, fitted_probs,
            loglik_val, loglik_null, pseudo_r2, aic_val, bic_val,
            vn, collect(cats), converged, iterations, cov_type
        )
    else
        OrderedProbitModel{T}(
            yint, Xm, beta, alpha, vcov_mat, fitted_probs,
            loglik_val, loglik_null, pseudo_r2, aic_val, bic_val,
            vn, collect(cats), converged, iterations, cov_type
        )
    end
end

# Float fallback
function estimate_ologit(y::AbstractVector, X::AbstractMatrix; kwargs...)
    estimate_ologit(y, Matrix{Float64}(X); kwargs...)
end

function estimate_oprobit(y::AbstractVector, X::AbstractMatrix; kwargs...)
    estimate_oprobit(y, Matrix{Float64}(X); kwargs...)
end

# =============================================================================
# StatsAPI Interface
# =============================================================================

for MT in (:OrderedLogitModel, :OrderedProbitModel)
    @eval begin
        StatsAPI.coef(m::$MT) = m.beta
        StatsAPI.vcov(m::$MT) = m.vcov_mat
        StatsAPI.nobs(m::$MT) = length(m.y)
        StatsAPI.dof(m::$MT) = length(m.beta) + length(m.cutpoints)
        StatsAPI.dof_residual(m::$MT) = length(m.y) - dof(m)
        StatsAPI.loglikelihood(m::$MT) = m.loglik
        StatsAPI.aic(m::$MT) = m.aic
        StatsAPI.bic(m::$MT) = m.bic
        StatsAPI.islinear(::$MT) = false
        StatsAPI.predict(m::$MT) = m.fitted

        function StatsAPI.stderror(m::$MT{T}) where {T}
            sqrt.(max.(diag(m.vcov_mat), zero(T)))
        end

        function StatsAPI.confint(m::$MT{T}; level::Real=0.95) where {T}
            se_all = stderror(m)
            K = length(m.beta)
            # Return CI for all parameters [beta; cutpoints]
            theta = vcat(m.beta, m.cutpoints)
            crit = T(quantile(Normal(), 1 - (1 - level) / 2))
            hcat(theta .- crit .* se_all, theta .+ crit .* se_all)
        end
    end
end

# =============================================================================
# Predict (out-of-sample)
# =============================================================================

"""
    StatsAPI.predict(m::OrderedLogitModel{T}, X_new::AbstractMatrix) -> Matrix{T}

Predict category probabilities for new data from an ordered logit model.

Returns an n_new x J probability matrix where each row sums to 1.
"""
function StatsAPI.predict(m::OrderedLogitModel{T}, X_new::AbstractMatrix) where {T<:AbstractFloat}
    _predict_ordered(m, X_new, _logistic_cdf)
end

"""
    StatsAPI.predict(m::OrderedProbitModel{T}, X_new::AbstractMatrix) -> Matrix{T}

Predict category probabilities for new data from an ordered probit model.

Returns an n_new x J probability matrix where each row sums to 1.
"""
function StatsAPI.predict(m::OrderedProbitModel{T}, X_new::AbstractMatrix) where {T<:AbstractFloat}
    _predict_ordered(m, X_new, _normal_cdf)
end

function _predict_ordered(m, X_new::AbstractMatrix, F_cdf::Function)
    T_type = eltype(m.beta)
    K = length(m.beta)
    size(X_new, 2) == K ||
        throw(ArgumentError("X_new must have $K columns (got $(size(X_new, 2)))"))
    Xm = Matrix{T_type}(X_new)
    n_new = size(Xm, 1)
    J = length(m.cutpoints) + 1
    probs = Matrix{T_type}(undef, n_new, J)
    @inbounds for i in 1:n_new
        xb = dot(@view(Xm[i, :]), m.beta)
        probs[i, :] .= _ordered_probs(m.cutpoints, xb, J, F_cdf)
    end
    probs
end

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, m::OrderedLogitModel{T}) where {T}
    _show_ordered(io, m, "Ordered Logit")
end

function Base.show(io::IO, m::OrderedProbitModel{T}) where {T}
    _show_ordered(io, m, "Ordered Probit")
end

function _show_ordered(io::IO, m, model_name::String)
    n = nobs(m)
    K = length(m.beta)
    J = length(m.cutpoints) + 1
    p = dof(m)

    spec = Any[
        "Model"         model_name;
        "Observations"  n;
        "Covariates"    K;
        "Categories"    J;
        "Parameters"    p;
        "Log-lik."      _fmt(m.loglik; digits=2);
        "Log-lik. null" _fmt(m.loglik_null; digits=2);
        "Pseudo R-sq."  _fmt(m.pseudo_r2);
        "AIC"           _fmt(m.aic; digits=2);
        "BIC"           _fmt(m.bic; digits=2);
        "Converged"     m.converged ? "Yes" : "No";
        "Iterations"    m.iterations
    ]
    _pretty_table(io, spec;
        title = "$model_name Regression",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    se_all = stderror(m)
    se_beta = se_all[1:K]
    se_cut = se_all[K+1:end]

    # Slope coefficients
    _coef_table(io, "Coefficients", m.varnames, m.beta, se_beta; dist=:z)

    # Cutpoints
    cut_names = ["cut$(j)" for j in 1:length(m.cutpoints)]
    _coef_table(io, "Cutpoints", cut_names, m.cutpoints, se_cut; dist=:z)

    _sig_legend(io)
end

# report dispatches
report(m::OrderedLogitModel) = show(stdout, m)
report(m::OrderedProbitModel) = show(stdout, m)
