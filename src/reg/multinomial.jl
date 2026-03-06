# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Multinomial logit regression estimated via Newton-Raphson MLE.

Implements the multinomial logit model:

    P(y = j | x) = exp(x' beta_j) / sum_k exp(x' beta_k)

with beta_1 = 0 (base category normalization).

Beta is a K x (J-1) matrix; coef(m) returns vec(beta) of length K(J-1).
"""


# =============================================================================
# Type Definition
# =============================================================================

"""
    MultinomialLogitModel{T} <: StatsAPI.RegressionModel

Multinomial logistic regression model estimated via maximum likelihood.

# Fields
- `y::Vector{Int}` -- dependent variable (remapped to 1:J)
- `X::Matrix{T}` -- regressor matrix (n x K, with intercept if desired)
- `beta::Matrix{T}` -- coefficient matrix (K x (J-1)), base category = 1
- `vcov_mat::Matrix{T}` -- vcov of vec(beta), K(J-1) x K(J-1)
- `fitted::Matrix{T}` -- predicted probabilities (n x J)
- `loglik::T` -- maximized log-likelihood
- `loglik_null::T` -- null model log-likelihood: n * log(1/J)
- `pseudo_r2::T` -- McFadden's pseudo R-squared
- `aic::T` -- Akaike information criterion
- `bic::T` -- Bayesian information criterion
- `varnames::Vector{String}` -- coefficient names
- `categories::Vector` -- original category values
- `converged::Bool` -- whether optimization converged
- `iterations::Int` -- number of iterations performed
- `cov_type::Symbol` -- covariance estimator

# References
- McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior.
  In P. Zarembka (Ed.), *Frontiers in Econometrics* (pp. 105-142). Academic Press.
- Train, K. E. (2009). *Discrete Choice Methods with Simulation*. 2nd ed. Cambridge Univ. Press.
"""
struct MultinomialLogitModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    y::Vector{Int}
    X::Matrix{T}
    beta::Matrix{T}        # K x (J-1)
    vcov_mat::Matrix{T}    # K(J-1) x K(J-1)
    fitted::Matrix{T}      # n x J
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

# =============================================================================
# Softmax with log-sum-exp trick
# =============================================================================

"""
    _mlogit_probs(xi, beta, J) -> Vector{T}

Compute P(y = j | x_i) for j = 1, ..., J using softmax with log-sum-exp trick.
beta is K x (J-1); base category (j=1) has beta_1 = 0.
"""
function _mlogit_probs(xi::AbstractVector{T}, beta::Matrix{T}, J::Int) where {T<:AbstractFloat}
    K = length(xi)
    Jm1 = J - 1
    # Linear predictors: v[1] = 0 (base), v[j] = x' beta_j for j=2..J
    v = zeros(T, J)
    @inbounds for j in 1:Jm1
        s = zero(T)
        for k in 1:K
            s += xi[k] * beta[k, j]
        end
        v[j+1] = s  # category j+1 corresponds to column j of beta
    end
    # Log-sum-exp
    v_max = maximum(v)
    denom = zero(T)
    @inbounds for j in 1:J
        denom += exp(v[j] - v_max)
    end
    log_denom = v_max + log(denom)
    probs = Vector{T}(undef, J)
    eps_floor = T(1e-15)
    @inbounds for j in 1:J
        probs[j] = max(exp(v[j] - log_denom), eps_floor)
    end
    probs
end

# =============================================================================
# Log-Likelihood, Score, and Hessian
# =============================================================================

"""
    _mlogit_loglik_score_hessian(y, X, beta, J) -> (loglik, score, H)

Compute log-likelihood, score vector (K(J-1)), and analytical Hessian
for multinomial logit. The Hessian is the expected (negative) information matrix:
H = -sum_i X_i' W_i X_i, where W_i has diagonal p_j(1-p_j), off-diagonal -p_j*p_k.
"""
function _mlogit_loglik_score_hessian(
        y::Vector{Int}, X::Matrix{T}, beta::Matrix{T},
        J::Int) where {T<:AbstractFloat}
    n, K = size(X)
    Jm1 = J - 1
    P = K * Jm1  # total parameters

    loglik = zero(T)
    score = zeros(T, P)
    H = zeros(T, P, P)

    @inbounds for i in 1:n
        xi = @view X[i, :]
        probs = _mlogit_probs(xi, beta, J)
        j = y[i]

        loglik += log(probs[j])

        # Score: d logL / d beta_m = (1(y_i = m+1) - p_{m+1}) * x_i  for m=1..J-1
        # vec(beta) is ordered as [beta_1; beta_2; ... ; beta_{J-1}] where each beta_m is length K
        for m in 1:Jm1
            indicator = (j == m + 1) ? one(T) : zero(T)
            resid_m = indicator - probs[m + 1]
            offset_m = (m - 1) * K
            for k in 1:K
                score[offset_m + k] += resid_m * xi[k]
            end
        end

        # Analytical Hessian: H_{m,l} block = -sum_i x_i x_i' * w_{m,l}
        # where w_{m,m} = p_{m+1}(1 - p_{m+1}), w_{m,l} = -p_{m+1}*p_{l+1}
        for m in 1:Jm1
            offset_m = (m - 1) * K
            for l in m:Jm1
                offset_l = (l - 1) * K
                if m == l
                    w_ml = -probs[m + 1] * (one(T) - probs[m + 1])
                else
                    w_ml = probs[m + 1] * probs[l + 1]
                end
                for a in 1:K
                    for b in 1:K
                        val = w_ml * xi[a] * xi[b]
                        H[offset_m + a, offset_l + b] += val
                        if m != l
                            H[offset_l + b, offset_m + a] += val
                        end
                    end
                end
            end
        end
    end

    (loglik, score, H)
end

# =============================================================================
# Score Matrix (for sandwich covariance)
# =============================================================================

"""
    _mlogit_score_matrix(y, X, beta, J) -> Matrix{T}

Compute n x P matrix of per-observation score vectors.
"""
function _mlogit_score_matrix(
        y::Vector{Int}, X::Matrix{T}, beta::Matrix{T},
        J::Int) where {T<:AbstractFloat}
    n, K = size(X)
    Jm1 = J - 1
    P = K * Jm1

    S = zeros(T, n, P)

    @inbounds for i in 1:n
        xi = @view X[i, :]
        probs = _mlogit_probs(xi, beta, J)
        j = y[i]

        for m in 1:Jm1
            indicator = (j == m + 1) ? one(T) : zero(T)
            resid_m = indicator - probs[m + 1]
            offset_m = (m - 1) * K
            for k in 1:K
                S[i, offset_m + k] = resid_m * xi[k]
            end
        end
    end

    S
end

# =============================================================================
# Newton-Raphson Estimation
# =============================================================================

"""
    _nr_mlogit(y, X, J; maxiter=200, tol=1e-8)

Newton-Raphson optimization for multinomial logit.
Returns (beta, loglik, converged, iterations).
"""
function _nr_mlogit(y::Vector{Int}, X::Matrix{T}, J::Int;
                    maxiter::Int=200, tol::T=T(1e-8)) where {T<:AbstractFloat}
    n, K = size(X)
    Jm1 = J - 1
    P = K * Jm1

    # Initialize beta at zero
    beta = zeros(T, K, Jm1)

    loglik_old = T(-Inf)
    converged = false
    iter = 0

    for it in 1:maxiter
        iter = it
        loglik_val, score, H = _mlogit_loglik_score_hessian(y, X, beta, J)

        # Check convergence
        if abs(loglik_val - loglik_old) < tol * (abs(loglik_old) + one(T))
            converged = true
            loglik_old = loglik_val
            break
        end
        loglik_old = loglik_val

        # Newton step: theta_new = theta_old - H^{-1} * score
        # H is negative semi-definite, so -H is PSD
        H_inv = Matrix{T}(robust_inv(Hermitian(-H)))
        delta = H_inv * score

        # Update: beta_vec += delta (since H is negative, -H^{-1} * score = H_inv * score)
        beta_vec = vec(beta)
        beta_vec .+= delta
        beta .= reshape(beta_vec, K, Jm1)
    end

    (beta, loglik_old, converged, iter)
end

# =============================================================================
# Estimation Function
# =============================================================================

"""
    estimate_mlogit(y, X; cov_type=:ols, varnames=nothing, clusters=nothing, maxiter=200, tol=1e-8) -> MultinomialLogitModel{T}

Estimate a multinomial logistic regression model via maximum likelihood (Newton-Raphson).

# Model
Multinomial logit: P(y = j | x) = exp(x' beta_j) / sum_k exp(x' beta_k),
with beta_1 = 0 (base category).

# Arguments
- `y::AbstractVector` -- categorical dependent variable (will be remapped to 1:J)
- `X::AbstractMatrix{T}` -- regressor matrix (n x K)
- `cov_type::Symbol` -- covariance estimator: `:ols` (MLE), `:hc0`, `:hc1` (sandwich), `:cluster`
- `varnames::Union{Nothing,Vector{String}}` -- coefficient names (auto-generated if nothing)
- `clusters::Union{Nothing,AbstractVector}` -- cluster assignments (for `:cluster`)
- `maxiter::Int` -- maximum Newton-Raphson iterations (default 200)
- `tol` -- convergence tolerance (default 1e-8)

# Returns
`MultinomialLogitModel{T}` with estimated coefficients and joint vcov.

# Examples
```julia
using MacroEconometricModels, Random
rng = MersenneTwister(42)
n = 1000
X = [ones(n) randn(rng, n, 2)]
beta_true = [0.5 -0.3; 1.0 -0.5; -0.5 0.8]  # K=3 x (J-1)=2
V = X * beta_true
P = exp.(V) ./ (1 .+ sum(exp.(V), dims=2))
# ... generate y from probabilities
m = estimate_mlogit(y, X; varnames=["const", "x1", "x2"])
report(m)
```

# References
- McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior.
- Train, K. E. (2009). *Discrete Choice Methods with Simulation*. 2nd ed. Cambridge Univ. Press.
"""
function estimate_mlogit(y::AbstractVector, X::AbstractMatrix{T};
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

    cov_type in (:ols, :hc0, :hc1, :cluster) ||
        throw(ArgumentError("cov_type must be :ols, :hc0, :hc1, or :cluster; got :$cov_type"))

    if cov_type == :cluster
        clusters === nothing && throw(ArgumentError("clusters required for :cluster cov_type"))
        length(clusters) == n || throw(ArgumentError("clusters must have length $n"))
    end

    # ---- Category remapping ----
    cats = sort(unique(y))
    J = length(cats)
    J >= 3 || throw(ArgumentError("Need at least 3 categories for multinomial model (got $J)"))
    Jm1 = J - 1
    P = K * Jm1
    n > P || throw(ArgumentError("Need n > K*(J-1) (n=$n, K=$K, J=$J)"))

    cat_map = Dict(cats[j] => j for j in 1:J)
    yint = [cat_map[yi] for yi in y]

    # ---- Variable names ----
    vn = something(varnames, ["x$i" for i in 1:K])
    length(vn) == K || throw(ArgumentError("varnames must have length $K"))

    Xm = Matrix{T}(X)

    # ---- Newton-Raphson estimation ----
    beta, loglik_val, converged, iterations = _nr_mlogit(
        yint, Xm, J; maxiter=maxiter, tol=tol)

    # ---- Null model log-likelihood ----
    loglik_null = T(n) * log(one(T) / T(J))

    # ---- McFadden pseudo R-squared ----
    pseudo_r2 = one(T) - loglik_val / loglik_null

    # ---- AIC / BIC ----
    aic_val = -2 * loglik_val + 2 * T(P)
    bic_val = -2 * loglik_val + log(T(n)) * T(P)

    # ---- Covariance matrix ----
    _, _, H = _mlogit_loglik_score_hessian(yint, Xm, beta, J)

    if cov_type == :ols
        # Classical MLE: V = (-H)^{-1}
        vcov_mat = Matrix{T}(robust_inv(Hermitian(-H)))
    else
        # Sandwich estimator: V = (-H)^{-1} B (-H)^{-1}
        H_inv = Matrix{T}(robust_inv(Hermitian(-H)))

        S_mat = _mlogit_score_matrix(yint, Xm, beta, J)

        if cov_type == :cluster
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
            B = S_mat' * S_mat
            if cov_type == :hc1
                B .*= T(n) / T(n - P)
            end
            vcov_mat = H_inv * B * H_inv
        end
    end

    # ---- Fitted probabilities ----
    fitted_probs = Matrix{T}(undef, n, J)
    @inbounds for i in 1:n
        fitted_probs[i, :] .= _mlogit_probs(@view(Xm[i, :]), beta, J)
    end

    # ---- Construct result ----
    MultinomialLogitModel{T}(
        yint, Xm, beta, vcov_mat, fitted_probs,
        loglik_val, loglik_null, pseudo_r2, aic_val, bic_val,
        vn, collect(cats), converged, iterations, cov_type
    )
end

# Float64 fallback
function estimate_mlogit(y::AbstractVector, X::AbstractMatrix; kwargs...)
    estimate_mlogit(y, Matrix{Float64}(X); kwargs...)
end

# =============================================================================
# StatsAPI Interface
# =============================================================================

StatsAPI.coef(m::MultinomialLogitModel) = vec(m.beta)
StatsAPI.vcov(m::MultinomialLogitModel) = m.vcov_mat
StatsAPI.nobs(m::MultinomialLogitModel) = length(m.y)
StatsAPI.dof(m::MultinomialLogitModel) = length(vec(m.beta))
StatsAPI.dof_residual(m::MultinomialLogitModel) = length(m.y) - dof(m)
StatsAPI.loglikelihood(m::MultinomialLogitModel) = m.loglik
StatsAPI.aic(m::MultinomialLogitModel) = m.aic
StatsAPI.bic(m::MultinomialLogitModel) = m.bic
StatsAPI.islinear(::MultinomialLogitModel) = false
StatsAPI.predict(m::MultinomialLogitModel) = m.fitted

function StatsAPI.stderror(m::MultinomialLogitModel{T}) where {T}
    sqrt.(max.(diag(m.vcov_mat), zero(T)))
end

function StatsAPI.confint(m::MultinomialLogitModel{T}; level::Real=0.95) where {T}
    se_all = stderror(m)
    theta = vec(m.beta)
    crit = T(quantile(Normal(), 1 - (1 - level) / 2))
    hcat(theta .- crit .* se_all, theta .+ crit .* se_all)
end

# =============================================================================
# Predict (out-of-sample)
# =============================================================================

"""
    StatsAPI.predict(m::MultinomialLogitModel{T}, X_new::AbstractMatrix) -> Matrix{T}

Predict category probabilities for new data from a multinomial logit model.

Returns an n_new x J probability matrix where each row sums to 1.
"""
function StatsAPI.predict(m::MultinomialLogitModel{T}, X_new::AbstractMatrix) where {T<:AbstractFloat}
    K = size(m.X, 2)
    size(X_new, 2) == K ||
        throw(ArgumentError("X_new must have $K columns (got $(size(X_new, 2)))"))
    Xm = Matrix{T}(X_new)
    n_new = size(Xm, 1)
    J = size(m.fitted, 2)
    probs = Matrix{T}(undef, n_new, J)
    @inbounds for i in 1:n_new
        probs[i, :] .= _mlogit_probs(@view(Xm[i, :]), m.beta, J)
    end
    probs
end

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, m::MultinomialLogitModel{T}) where {T}
    n = nobs(m)
    K = size(m.X, 2)
    J = size(m.fitted, 2)
    p = dof(m)

    spec = Any[
        "Model"         "Multinomial Logit";
        "Observations"  n;
        "Covariates"    K;
        "Categories"    J;
        "Base category" string(m.categories[1]);
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
        title = "Multinomial Logit Regression",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    se_all = stderror(m)
    Jm1 = J - 1

    # Per-alternative coefficient blocks
    for j in 1:Jm1
        offset = (j - 1) * K
        beta_j = m.beta[:, j]
        se_j = se_all[offset+1:offset+K]
        cat_label = string(m.categories[j + 1])
        _coef_table(io, "Alternative $cat_label (vs $(m.categories[1]))",
                    m.varnames, beta_j, se_j; dist=:z)
    end

    _sig_legend(io)
end

# report dispatch
report(m::MultinomialLogitModel) = show(stdout, m)
