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

# Density derivatives f'(z) (needed for the analytic observed-information Hessian)
# Logistic: f(z)=F(z)(1-F(z)) ⇒ f'(z)=f(z)(1-2F(z)). Normal: f'(z)=-z·φ(z).
_logistic_pdf_deriv(::Type{T}, x::T) where {T<:AbstractFloat} = begin
    F = _logistic_cdf(T, x)
    f = F * (one(T) - F)
    f * (one(T) - 2 * F)
end
_normal_pdf_deriv(::Type{T}, x::T) where {T<:AbstractFloat} = -x * _normal_pdf(T, x)

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
    _ordered_loglik_score_hessian(y, X, beta, alpha, J, F_cdf, F_pdf, F_dpdf)

Compute log-likelihood, score (gradient), and the analytic observed-information
Hessian for the ordered model. Parameter vector is theta = [beta; alpha].
`F_dpdf` is the density derivative f'(z) used for the Hessian curvature terms.
"""
function _ordered_loglik_score_hessian(
        y::Vector{Int}, X::Matrix{T}, beta::Vector{T}, alpha::Vector{T},
        J::Int, F_cdf::Function, F_pdf::Function, F_dpdf::Function) where {T<:AbstractFloat}
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

    # Analytic observed-information Hessian (NOT BHHH). For ℓ=log p,
    #   ∂²ℓ/∂θ_a∂θ_b = (∂²p/∂θ_a∂θ_b)/p − s_a s_b,
    # where the −s_a s_b term is BHHH and the curvature term (∂²p/∂θ²)/p is what BHHH drops.
    # With z_u=α_j−x'β, z_l=α_{j-1}−x'β and f'(z)=F_dpdf(z), the nonzero curvature blocks are
    #   ββ: (f'_u−f'_l)/p·xxᵀ, β·α_j: −f'_u/p·x, β·α_{j-1}: +f'_l/p·x,
    #   α_jα_j: f'_u/p, α_{j-1}α_{j-1}: −f'_l/p, α_jα_{j-1}: 0 (that cross is pure −s_a s_b).
    H .= zero(T)
    scores_i = zeros(T, P)
    @inbounds for i in 1:n
        xi = @view X[i, :]
        xb = dot(xi, beta)
        j = y[i]

        if j == 1
            F_lower = zero(T); f_lower = zero(T); fp_lower = zero(T)
            zu = alpha[1] - xb
            F_upper = F_cdf(T, zu); f_upper = F_pdf(T, zu); fp_upper = F_dpdf(T, zu)
        elseif j == J
            zl = alpha[Jm1] - xb
            F_lower = F_cdf(T, zl); f_lower = F_pdf(T, zl); fp_lower = F_dpdf(T, zl)
            F_upper = one(T); f_upper = zero(T); fp_upper = zero(T)
        else
            zl = alpha[j-1] - xb; zu = alpha[j] - xb
            F_lower = F_cdf(T, zl); f_lower = F_pdf(T, zl); fp_lower = F_dpdf(T, zl)
            F_upper = F_cdf(T, zu); f_upper = F_pdf(T, zu); fp_upper = F_dpdf(T, zu)
        end

        p_ij = max(F_upper - F_lower, eps_floor)

        scores_i .= zero(T)
        sbf = -(f_upper - f_lower) / p_ij
        for k in 1:K
            scores_i[k] = sbf * xi[k]
        end
        (j < J) && (scores_i[K + j] = f_upper / p_ij)
        (j > 1) && (scores_i[K + j - 1] += -f_lower / p_ij)

        # −s_i s_iᵀ (BHHH part) over all parameter pairs
        for a in 1:P, b in 1:P
            H[a, b] -= scores_i[a] * scores_i[b]
        end

        # + curvature (∂²p/∂θ²)/p over the nonzero blocks
        cbb = (fp_upper - fp_lower) / p_ij            # ββ scalar
        for a in 1:K, b in 1:K
            H[a, b] += cbb * xi[a] * xi[b]
        end
        if j < J
            au = K + j
            cbu = -fp_upper / p_ij                    # β·α_j
            for k in 1:K
                H[k, au] += cbu * xi[k]
                H[au, k] += cbu * xi[k]
            end
            H[au, au] += fp_upper / p_ij              # α_jα_j
        end
        if j > 1
            al = K + j - 1
            cbl = fp_lower / p_ij                      # β·α_{j-1}
            for k in 1:K
                H[k, al] += cbl * xi[k]
                H[al, k] += cbl * xi[k]
            end
            H[al, al] += -fp_lower / p_ij             # α_{j-1}α_{j-1}
        end
    end

    (loglik, score, H)
end

# =============================================================================
# Newton-Raphson Estimation
# =============================================================================

"""
    _nr_ordered(y, X, J, F_cdf, F_pdf, F_dpdf; maxiter=200, tol=1e-8)

Newton-Raphson optimization for ordered logit/probit (true Newton via analytic Hessian).
Returns (beta, alpha, loglik, converged, iterations).
"""
function _nr_ordered(y::Vector{Int}, X::Matrix{T}, J::Int,
                     F_cdf::Function, F_pdf::Function, F_dpdf::Function;
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
            y, X, beta, alpha, J, F_cdf, F_pdf, F_dpdf)

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
- `cov_type::Symbol` -- covariance estimator: `:ols` (MLE information), `:hc0`, `:hc1`
  (sandwich), or `:cluster` (requires `clusters`)
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
Same as `estimate_ologit` (including `cov_type ∈ (:ols, :hc0, :hc1, :cluster)`).

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

    cov_type in (:ols, :hc0, :hc1, :cluster) ||
        throw(ArgumentError("cov_type must be :ols, :hc0, :hc1, or :cluster; got :$cov_type"))

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
    F_cdf, F_pdf, F_dpdf = if link == :logit
        (_logistic_cdf, _logistic_pdf, _logistic_pdf_deriv)
    else
        (_normal_cdf, _normal_pdf, _normal_pdf_deriv)
    end

    # ---- Newton-Raphson estimation ----
    beta, alpha, loglik_val, converged, iterations = _nr_ordered(
        yint, Xm, J, F_cdf, F_pdf, F_dpdf; maxiter=maxiter, tol=tol)

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
        # Classical MLE: V = (-H)^{-1} = observed-information inverse (Stata vce(oim))
        _, _, H = _ordered_loglik_score_hessian(
            yint, Xm, beta, alpha, J, F_cdf, F_pdf, F_dpdf)
        vcov_mat = Matrix{T}(robust_inv(Hermitian(-H)))
    else
        # Sandwich estimator: V = (-H)^{-1} S (-H)^{-1}, bread = observed information
        _, _, H = _ordered_loglik_score_hessian(
            yint, Xm, beta, alpha, J, F_cdf, F_pdf, F_dpdf)
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
            B = S_mat' * S_mat  # outer product of scores
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
# Residuals for multi-category responses (#507)
#
# Shared by the ordered models here and by MultinomialLogitModel (multinomial.jl
# is included after this file). A K-category response has K residuals per
# observation, so these return an n x K matrix rather than the length-n vector
# the binary models return.
# =============================================================================

"""
    _category_residuals(y, P, kind) -> Matrix{T}

Residual matrix for a discrete response with `K` categories, given the observed
category codes `y` (values in `1:K`) and the fitted probability matrix `P` (`n x K`).

With the indicator `dᵢⱼ = 1{yᵢ = j}`:

- `:response` — `rᵢⱼ = dᵢⱼ - P̂ᵢⱼ`. Rows sum to exactly zero.
- `:pearson`  — `rᵢⱼ / sqrt(P̂ᵢⱼ(1 - P̂ᵢⱼ))`, the variance-standardized form.
- `:deviance` — `sign(rᵢⱼ)·sqrt(2 dᵢⱼ log(dᵢⱼ / P̂ᵢⱼ))`, which is nonzero only in the
  observed cell and whose total sum of squares is the model deviance `-2·loglik`.
"""
function _category_residuals(y::Vector{Int}, P::Matrix{T}, kind::Symbol) where {T<:AbstractFloat}
    kind in (:response, :pearson, :deviance) ||
        throw(ArgumentError("kind must be :response, :pearson, or :deviance; got :$kind"))
    n, K = size(P)
    length(y) == n || throw(ArgumentError("y and the fitted matrix disagree on n"))
    R = Matrix{T}(undef, n, K)
    @inbounds for i in 1:n, j in 1:K
        d = y[i] == j ? one(T) : zero(T)
        p = clamp(P[i, j], T(1e-15), one(T) - T(1e-15))
        r = d - p
        R[i, j] = if kind === :response
            r
        elseif kind === :pearson
            r / sqrt(p * (one(T) - p))
        else
            # 0·log0 = 0, so every unobserved cell contributes nothing.
            d > zero(T) ? sign(r) * sqrt(2 * d * log(d / p)) : zero(T)
        end
    end
    R
end

for MT in (:OrderedLogitModel, :OrderedProbitModel)
    @eval begin
        """
            residuals(m::$($MT); kind=:response) -> Matrix{T}

        Residual matrix (`n x K`, one column per outcome category) for an ordered model.

        Unlike [`LogitModel`](@ref)/[`ProbitModel`](@ref), which return a length-`n` vector
        of deviance residuals, a `K`-category response has `K` residuals per observation, so
        this returns a matrix. For the length-`n` analogue of the binary score residual — the
        quantity score and LM specification tests are built on — use
        [`generalized_residuals`](@ref).

        `kind` selects `:response` (default, `dᵢⱼ - P̂ᵢⱼ`, rows summing to zero), `:pearson`,
        or `:deviance` (sum of squares equals `-2·loglik`).
        """
        StatsAPI.residuals(m::$MT{T}; kind::Symbol=:response) where {T} =
            _category_residuals(m.y, m.fitted, kind)
    end
end

"""
    generalized_residuals(m::OrderedLogitModel) -> Vector{T}
    generalized_residuals(m::OrderedProbitModel) -> Vector{T}

Generalized residuals for an ordered model (Chesher & Irish 1987; Gourieroux, Monfort,
Renault & Trognon 1987): the length-`n` vector

```math
e_i = \\frac{f(c_{j-1} - x_i'\\beta) - f(c_j - x_i'\\beta)}{P(y_i = j \\mid x_i)},
\\qquad j = y_i,
```

with `c₀ = -∞`, `c_K = +∞`, and `f` the logistic or standard-normal density. Equivalently
`eᵢ = ∂ℓᵢ/∂(x_i'β)`, the score of the observation's log-likelihood with respect to its
index, which for the probit case is exactly `E[εᵢ | yᵢ, xᵢ]`.

This is the quantity that makes outer-product-of-gradients LM specification tests work, and
it is the length-`n` analogue of the binary models' score residual: on a two-category fit it
reduces exactly to `yᵢ - p̂ᵢ`.

For the per-category residual matrix, see [`residuals`](@ref).

# References
- Chesher, A. & Irish, M. (1987). *Journal of Econometrics* 34(1-2), 33-61.
- Gourieroux, C., Monfort, A., Renault, E. & Trognon, A. (1987). *Journal of Econometrics*
  34(1-2), 5-32.
"""
function generalized_residuals(m::OrderedLogitModel{T}) where {T<:AbstractFloat}
    _ordered_gen_resid(m, _logistic_pdf)
end

function generalized_residuals(m::OrderedProbitModel{T}) where {T<:AbstractFloat}
    _ordered_gen_resid(m, _normal_pdf)
end

function _ordered_gen_resid(m, F_pdf::Function)
    T = eltype(m.beta)
    n = length(m.y)
    J = length(m.cutpoints) + 1
    xb = m.X * m.beta
    e = Vector{T}(undef, n)
    @inbounds for i in 1:n
        j = m.y[i]
        # f(-Inf) = f(+Inf) = 0, so the boundary categories drop one term.
        f_lo = j == 1 ? zero(T) : F_pdf(T, m.cutpoints[j-1] - xb[i])
        f_hi = j == J ? zero(T) : F_pdf(T, m.cutpoints[j] - xb[i])
        p = max(m.fitted[i, j], T(1e-15))
        e[i] = (f_lo - f_hi) / p
    end
    e
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

    _degenerate_fit_banner(io, m.beta)
    _sig_legend(io)
end

# report dispatches
report(m::OrderedLogitModel) = show(stdout, m)
report(m::OrderedProbitModel) = show(stdout, m)

# =============================================================================
# Marginal Effects (AME) for Ordered Models
# =============================================================================

"""
    marginal_effects(m::OrderedLogitModel{T}) -> NamedTuple

Compute average marginal effects (AME) for an ordered logit model.

Returns a K x J matrix of AMEs where element (k,j) is the average
marginal effect of variable k on P(y = j).

Formula: dP(y=j)/dx_k = [f(alpha_{j-1} - x'beta) - f(alpha_j - x'beta)] * beta_k
where f is the logistic PDF, alpha_0 = -Inf, alpha_J = +Inf.

Key property: AMEs sum to zero across categories for each variable.

# Returns
Named tuple with fields:
- `effects::Matrix{T}` -- K x J matrix of AMEs
- `se::Matrix{T}` -- delta-method standard errors (same shape)
- `varnames::Vector{String}` -- variable names
- `categories::Vector` -- category labels

# References
- Cameron, A. C. & Trivedi, P. K. (2005). *Microeconometrics*. Cambridge University Press, ch. 15.
- Greene, W. H. (2012). *Econometric Analysis*. 7th ed. Prentice Hall.
"""
function marginal_effects(m::OrderedLogitModel{T}) where {T<:AbstractFloat}
    _ordered_marginal_effects(m, _logistic_pdf)
end

"""
    marginal_effects(m::OrderedProbitModel{T}) -> NamedTuple

Compute average marginal effects (AME) for an ordered probit model.

Same structure as `marginal_effects(::OrderedLogitModel)` but using the
standard normal PDF.

# References
- Cameron, A. C. & Trivedi, P. K. (2005). *Microeconometrics*. Cambridge University Press, ch. 15.
- Greene, W. H. (2012). *Econometric Analysis*. 7th ed. Prentice Hall.
"""
function marginal_effects(m::OrderedProbitModel{T}) where {T<:AbstractFloat}
    _ordered_marginal_effects(m, _normal_pdf)
end

"""Internal: compute AME + delta-method SEs for ordered logit/probit."""
function _ordered_marginal_effects(m, F_pdf::Function)
    T_type = eltype(m.beta)
    K = length(m.beta)
    J = length(m.cutpoints) + 1
    alpha = m.cutpoints
    beta = m.beta

    ame = _ordered_ame_core(m.X, beta, alpha, F_pdf)

    # Finite-difference Jacobian of vec(AME) w.r.t. θ = [β; α]
    n_theta = K + length(alpha)
    G = zeros(T_type, K * J, n_theta)
    h = sqrt(eps(T_type))
    ame0 = vec(ame)
    theta0 = vcat(beta, alpha)
    for p in 1:n_theta
        theta = copy(theta0)
        step = max(abs(theta[p]), one(T_type)) * h
        theta[p] += step
        ame_p = vec(_ordered_ame_core(m.X, theta[1:K], theta[K+1:end], F_pdf))
        G[:, p] .= (ame_p .- ame0) ./ step
    end
    V = m.vcov_mat
    se = if size(V, 1) >= n_theta
        V_ame = G * V[1:n_theta, 1:n_theta] * G'
        reshape(sqrt.(max.(diag(V_ame), zero(T_type))), K, J)
    else
        fill(T_type(NaN), K, J)
    end

    (effects=ame, se=se, varnames=copy(m.varnames), categories=copy(m.categories))
end

function _ordered_ame_core(X::Matrix{T}, beta::AbstractVector{S}, alpha::AbstractVector{S},
                           F_pdf::Function) where {T,S}
    n = size(X, 1)
    K = length(beta)
    J = length(alpha) + 1
    ame = zeros(S, K, J)
    @inbounds for i in 1:n
        xi = @view X[i, :]
        xb = dot(xi, beta)
        for j in 1:J
            f_lower = j == 1 ? zero(S) : F_pdf(S, alpha[j-1] - xb)
            f_upper = j == J ? zero(S) : F_pdf(S, alpha[j] - xb)
            factor = f_lower - f_upper
            for k in 1:K
                ame[k, j] += factor * beta[k]
            end
        end
    end
    ame ./= S(n)
    ame
end

# =============================================================================
# Brant Test (Proportional Odds / Parallel Regression Assumption)
# =============================================================================

"""
    brant_test(m::OrderedLogitModel{T}) -> NamedTuple

Brant test of the proportional odds (parallel regression) assumption for
an ordered logit model (Brant 1990).

For each cutpoint j = 1,...,J-1, fits a binary logit (y ≤ j vs y > j).
Under H0 (proportional odds) the binary-logit slope vectors are equal:

    β̂₁ = β̂₂ = ⋯ = β̂_{J−1}

Contrasts are formed against the last binary logit,

    d_j = β̂_j − β̂_{J−1},  j = 1,…,J−2

with the **joint** score covariance of the binary logits (shared sample —
not the independence approximation). Overall Wald χ² has K·(J−2) degrees of
freedom; per-variable tests have (J−2) df each.

Note: the binary slopes are **not** compared to the pooled ordered-logit
`m.beta` — under H0 the binary MLEs need not equal the ologit MLE (different
likelihoods), only each other.

# Returns
Named tuple with fields:
- `statistic::T` -- overall Wald test statistic
- `pvalue::T` -- overall p-value (chi-squared)
- `df::Int` -- degrees of freedom K*(J-2)
- `per_variable::Vector{T}` -- per-variable p-values (length K)
- `binary_coefs::Matrix{T}` -- K x (J-1) matrix of binary logit coefficients

# References
- Brant, R. (1990). Assessing proportionality in the proportional odds model
  for ordinal logistic regression. *Biometrics* 46, 1171-1178.
"""
function brant_test(m::OrderedLogitModel{T}) where {T<:AbstractFloat}
    n = length(m.y)
    K = length(m.beta)
    J = length(m.cutpoints) + 1
    Jm1 = J - 1

    Jm1 >= 2 || throw(ArgumentError("Brant test requires at least 3 categories (J >= 3)"))

    # Fit J-1 binary logits: y <= j vs y > j
    X_bin = hcat(ones(T, n), m.X)  # n x (K+1)

    binary_coefs = Matrix{T}(undef, K, Jm1)
    # Per-observation scores for the FULL [1 X] parameter vector (K+1 × n per cutpoint).
    # The binary logits carry an intercept, so the slope covariance is the slope block of
    # the (K+1)-dimensional sandwich — dropping the intercept before inverting the bread
    # leaves the statistic location-dependent (#547).
    scores = Vector{Matrix{T}}(undef, Jm1)
    breads = Vector{Matrix{T}}(undef, Jm1)

    for j in 1:Jm1
        y_bin = T.(m.y .<= j)
        mj = estimate_logit(y_bin, X_bin; maxiter=200)
        binary_coefs[:, j] = mj.beta[2:end]

        # Score contribution s_i = x_i (y_i − μ_i); bread = (X'WX)^{-1}
        mu = mj.fitted
        resid = y_bin .- mu
        sc = Matrix{T}(undef, K + 1, n)
        @inbounds for i in 1:n
            for k in 1:(K + 1)
                sc[k, i] = X_bin[i, k] * resid[i]
            end
        end
        scores[j] = sc
        w = mu .* (one(T) .- mu)
        XtWX = X_bin' * (X_bin .* w)
        breads[j] = Matrix{T}(robust_inv(Hermitian((XtWX .+ XtWX') ./ 2)))
    end

    # Joint covariance Cov(β̂_a, β̂_b) = Bread_a · (Σ_i s_{a,i} s_{b,i}') · Bread_b,
    # formed in full dimension and then restricted to the slope block.
    function _joint_cov(a::Int, b::Int)
        meat = zeros(T, K + 1, K + 1)
        sa = scores[a]; sb = scores[b]
        @inbounds for i in 1:n
            meat .+= sa[:, i] * sb[:, i]'
        end
        (breads[a] * meat * breads[b])[2:end, 2:end]
    end

    # Contrasts: β̂_j − β̂_{J−1} for j = 1,...,J-2 (equality of binary slopes)
    n_contrasts = Jm1 - 1
    d_stack = Vector{T}(undef, K * n_contrasts)
    V_stack = zeros(T, K * n_contrasts, K * n_contrasts)
    ref = Jm1
    for j in 1:n_contrasts
        offset = (j - 1) * K
        d_stack[offset+1:offset+K] = binary_coefs[:, j] - binary_coefs[:, ref]
        for j2 in 1:n_contrasts
            offset2 = (j2 - 1) * K
            # Var(β_j − β_ref, β_{j2} − β_ref) = V_jj2 − V_j,ref − V_ref,j2 + V_ref,ref
            V_stack[offset+1:offset+K, offset2+1:offset2+K] =
                _joint_cov(j, j2) - _joint_cov(j, ref) -
                _joint_cov(ref, j2) + _joint_cov(ref, ref)
        end
    end
    V_stack = (V_stack .+ V_stack') ./ 2
    V_stack_inv = Matrix{T}(robust_inv(Hermitian(V_stack)))
    stat_overall = max(dot(d_stack, V_stack_inv * d_stack), zero(T))
    df_overall = K * n_contrasts
    pval_overall = T(1 - cdf(Chisq(df_overall), stat_overall))

    # Per-variable tests: β_{k,1} = ⋯ = β_{k,J−1}
    per_var_pvals = Vector{T}(undef, K)
    df_per_var = n_contrasts
    for k in 1:K
        d_k = Vector{T}(undef, n_contrasts)
        V_k = zeros(T, n_contrasts, n_contrasts)
        for j in 1:n_contrasts
            d_k[j] = binary_coefs[k, j] - binary_coefs[k, ref]
            for j2 in 1:n_contrasts
                V_k[j, j2] = (_joint_cov(j, j2) - _joint_cov(j, ref) -
                              _joint_cov(ref, j2) + _joint_cov(ref, ref))[k, k]
            end
        end
        V_k = (V_k .+ V_k') ./ 2
        V_k_inv = Matrix{T}(robust_inv(Hermitian(V_k)))
        stat_k = max(dot(d_k, V_k_inv * d_k), zero(T))
        per_var_pvals[k] = T(1 - cdf(Chisq(df_per_var), stat_k))
    end

    (statistic=stat_overall, pvalue=pval_overall, df=df_overall,
     per_variable=per_var_pvals, binary_coefs=binary_coefs)
end
