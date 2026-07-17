# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Censored (Tobit) and truncated-normal regression by maximum likelihood.

Type-I Tobit (Tobin 1958) is fit for the normal case in the globally concave Olsen (1978)
reparameterization δ = β/σ, γ = 1/σ (optimized as `(δ, log γ)`), then back-transformed to
`(β, σ)` with delta-method standard errors. Logistic and extreme-value Tobit variants are fit
directly in `(β, log σ)`. Truncated regression uses the Hausman–Wise (1977) truncated-normal
log-likelihood. Marginal effects follow the McDonald–Moffitt (1980) decomposition.

The truncated-normal moment helpers `_mills` (inverse Mills ratio) and `_lambda` (two-sided
truncation hazard) are shared with the Heckman selection model (EV-18, `heckman.jl`).
"""

using LinearAlgebra, Statistics, Distributions, StatsAPI

# =============================================================================
# Shared truncated-normal moment helpers (reused by heckman.jl, EV-18)
# =============================================================================

"""
    _mills(a) -> φ(a) / Φ(a)

Inverse Mills ratio (lower-tail): the hazard `φ(a)/Φ(a)` of the standard normal, where `φ`
is the pdf and `Φ` the cdf. Equals `E[z | z > -a]` for `z ~ N(0,1)` and is the standard
selection-correction term in the Heckman (1979) model. Evaluated via `logpdf`/`logcdf` so it
stays finite far into the left tail (`Φ(a) → 0`) instead of overflowing to `0/0`.
"""
function _mills(a::T) where {T<:AbstractFloat}
    N = Normal(zero(T), one(T))
    exp(logpdf(N, a) - logcdf(N, a))
end
_mills(a::Real) = _mills(float(a))

"""
    _lambda(a, b) -> (φ(a) - φ(b)) / (Φ(b) - Φ(a))

Two-sided truncated-normal hazard. For `z ~ N(0,1)` truncated to `(a, b)`,
`E[z | a < z < b] = (φ(a) - φ(b)) / (Φ(b) - Φ(a))`. With `b = +∞` this reduces to the
lower-truncation term `φ(a)/Φ(-a) = _mills(-a)`; with `a = -∞` to `-φ(b)/Φ(b) = -_mills(b)`.
Used for the mean of a doubly truncated normal in Tobit conditional marginal effects and by
the Heckman model. The denominator is floored at `√eps` to avoid division by zero under heavy
truncation.
"""
function _lambda(a::T, b::T) where {T<:AbstractFloat}
    N = Normal(zero(T), one(T))
    denom = cdf(N, b) - cdf(N, a)
    denom = max(denom, sqrt(eps(T)))
    (pdf(N, a) - pdf(N, b)) / denom
end
_lambda(a::Real, b::Real) = (p = promote(float(a), float(b)); _lambda(p[1], p[2]))

# =============================================================================
# TobitModel — Type-I censored regression
# =============================================================================

"""
    TobitModel{T} <: StatsAPI.RegressionModel

Type-I Tobit (censored) regression estimated by maximum likelihood. See [`estimate_tobit`](@ref).

# Fields
- `y`, `X` — response and regressor matrix (include an intercept column in `X`).
- `beta::Vector{T}` — slope coefficients on the latent index `x'β`.
- `sigma::T` — latent-error standard deviation.
- `vcov_mat::Matrix{T}` — joint `(k+1)×(k+1)` covariance of `(β, σ)` (σ last); `stderror`
  returns the first `k` (the β block).
- `sigma_se::T` — delta-method standard error of `σ`.
- `residuals`, `fitted` — `y - Xβ` and the latent index `Xβ`.
- `loglik`, `aic`, `bic` — maximized log-likelihood and information criteria (params = k+1).
- `lower::T`, `upper::T` — censoring thresholds (`-Inf`/`Inf` for one-sided).
- `n_censored_left::Int`, `n_censored_right::Int` — censored-observation counts.
- `dist::Symbol` — latent error law: `:normal` (default, Olsen-reparameterized), `:logistic`,
  or `:extreme_value`.
- `varnames::Vector{String}`, `method::Symbol` (`:tobit`), `converged::Bool`.

# References
- Tobin, J. (1958). *Econometrica* 26(1), 24-36.
- Olsen, R. J. (1978). *Econometrica* 46(5), 1211-1215.
- McDonald, J. F. & Moffitt, R. A. (1980). *Review of Economics and Statistics* 62(2), 318-321.
"""
struct TobitModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    y::Vector{T}
    X::Matrix{T}
    beta::Vector{T}
    sigma::T
    vcov_mat::Matrix{T}
    sigma_se::T
    residuals::Vector{T}
    fitted::Vector{T}
    loglik::T
    aic::T
    bic::T
    lower::T
    upper::T
    n_censored_left::Int
    n_censored_right::Int
    dist::Symbol
    varnames::Vector{String}
    method::Symbol
    converged::Bool
end

"""
    TruncRegModel{T} <: StatsAPI.RegressionModel

Truncated-normal regression estimated by maximum likelihood (Hausman & Wise 1977). Every
observation is assumed drawn from `N(x'β, σ²)` truncated to `(lower, upper)`; there is no
censored contribution. See [`estimate_truncreg`](@ref).

# Fields
Mirror [`TobitModel`](@ref) except there are no censoring counts (`n_truncated::Int` records
how many sample points lie inside the truncation region — i.e. `nobs`). `method` is `:truncreg`.

# References
- Hausman, J. A. & Wise, D. A. (1977). *Econometrica* 45(4), 919-938.
"""
struct TruncRegModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    y::Vector{T}
    X::Matrix{T}
    beta::Vector{T}
    sigma::T
    vcov_mat::Matrix{T}
    sigma_se::T
    residuals::Vector{T}
    fitted::Vector{T}
    loglik::T
    aic::T
    bic::T
    lower::T
    upper::T
    n_truncated::Int
    dist::Symbol
    varnames::Vector{String}
    method::Symbol
    converged::Bool
end

# =============================================================================
# Log-likelihoods
# =============================================================================

# Standardized latent-error distribution for a given `dist` symbol.
# `T` may be a ForwardDiff.Dual (hence `<:Real`, not `<:AbstractFloat`) during autodiff.
function _tobit_stddist(dist::Symbol, ::Type{T}) where {T<:Real}
    dist === :normal        ? Normal(zero(T), one(T)) :
    dist === :logistic      ? Logistic(zero(T), one(T)) :
    dist === :extreme_value ? Gumbel(zero(T), one(T)) :
    throw(ArgumentError("dist must be :normal, :logistic, or :extreme_value; got :$dist"))
end

# Tobit log-likelihood at (β, σ) for arbitrary lower/upper and standardized error law `D`.
# Uncensored: log[(1/σ) f((y-x'β)/σ)];  left-censored (y≤lower): log F((lower-x'β)/σ);
# right-censored (y≥upper): log[1 - F((upper-x'β)/σ)] = logccdf.
function _tobit_loglik(beta::AbstractVector{S}, sigma::S, y::Vector{T}, X::Matrix{T},
                       lower::T, upper::T, D) where {S,T<:AbstractFloat}
    n = length(y)
    ll = zero(S)
    logsig = log(sigma)
    @inbounds for i in 1:n
        xb = zero(S)
        for j in 1:length(beta)
            xb += X[i, j] * beta[j]
        end
        yi = y[i]
        if isfinite(lower) && yi <= lower
            ll += logcdf(D, (S(lower) - xb) / sigma)
        elseif isfinite(upper) && yi >= upper
            ll += logccdf(D, (S(upper) - xb) / sigma)
        else
            r = (S(yi) - xb) / sigma
            ll += logpdf(D, r) - logsig
        end
    end
    ll
end

# Normal-Tobit negative log-likelihood in the Olsen (1978) space params = [δ; η], γ = exp(η),
# β = δ/γ = δ·σ, σ = 1/γ. Globally concave in (δ, γ); we optimize (δ, log γ).
function _tobit_negll_olsen(params::AbstractVector{S}, y::Vector{T}, X::Matrix{T},
                            lower::T, upper::T) where {S,T<:AbstractFloat}
    k = size(X, 2)
    delta = @view params[1:k]
    eta = params[k+1]                # η = log γ
    gamma = exp(eta)                 # γ = 1/σ > 0
    N = Normal(zero(S), one(S))
    n = length(y)
    ll = zero(S)
    @inbounds for i in 1:n
        xd = zero(S)
        for j in 1:k
            xd += X[i, j] * delta[j]
        end
        yi = y[i]
        if isfinite(lower) && yi <= lower
            ll += logcdf(N, gamma * S(lower) - xd)
        elseif isfinite(upper) && yi >= upper
            ll += logccdf(N, gamma * S(upper) - xd)
        else
            # log[(1/σ)φ((y-x'β)/σ)] = log γ + log φ(γy - x'δ) = η + logφ(γy - x'δ)
            ll += eta + logpdf(N, gamma * S(yi) - xd)
        end
    end
    -ll
end

# Direct (β, log σ) negative log-likelihood for logistic / extreme-value Tobit.
function _tobit_negll_direct(params::AbstractVector{S}, y::Vector{T}, X::Matrix{T},
                             lower::T, upper::T, dist::Symbol) where {S,T<:AbstractFloat}
    k = size(X, 2)
    beta = @view params[1:k]
    sigma = exp(params[k+1])
    D = _tobit_stddist(dist, S)
    -_tobit_loglik(beta, sigma, y, X, lower, upper, D)
end

# Truncated-normal (Hausman–Wise) negative log-likelihood in (β, log σ).
# ℓ_i = log[(1/σ)φ(r_i)] - log[Φ((U-x'β)/σ) - Φ((L-x'β)/σ)].
function _truncreg_negll(params::AbstractVector{S}, y::Vector{T}, X::Matrix{T},
                         lower::T, upper::T) where {S,T<:AbstractFloat}
    k = size(X, 2)
    beta = @view params[1:k]
    sigma = exp(params[k+1])
    N = Normal(zero(S), one(S))
    logsig = log(sigma)
    n = length(y)
    ll = zero(S)
    @inbounds for i in 1:n
        xb = zero(S)
        for j in 1:k
            xb += X[i, j] * beta[j]
        end
        r = (S(y[i]) - xb) / sigma
        # log-denominator = log P(lower < y* < upper): use tail-stable forms one-sided.
        if !isfinite(upper)
            logdenom = logccdf(N, (S(lower) - xb) / sigma)          # 1 - Φ(a)
        elseif !isfinite(lower)
            logdenom = logcdf(N, (S(upper) - xb) / sigma)           # Φ(b)
        else
            logdenom = log(cdf(N, (S(upper) - xb) / sigma) - cdf(N, (S(lower) - xb) / sigma))
        end
        ll += logpdf(N, r) - logsig - logdenom
    end
    -ll
end

# =============================================================================
# estimate_tobit
# =============================================================================

"""
    estimate_tobit(y, X; lower=0.0, upper=Inf, dist=:normal, varnames=nothing,
                   maxiter=1000, tol=1e-10) -> TobitModel{T}

Estimate a Type-I Tobit (censored) regression by maximum likelihood.

Observations with `y ≤ lower` are treated as left-censored, `y ≥ upper` as right-censored, and
the remainder as uncensored. Supports one-sided (`lower=0, upper=Inf`, the classic Tobit),
upper-only (`lower=-Inf`), and two-sided (both finite) censoring.

# Algorithm
For `dist=:normal` the likelihood is maximized in the globally concave Olsen (1978)
reparameterization `δ = β/σ`, `γ = 1/σ` (optimized as `(δ, log γ)` via `Optim.LBFGS` with
forward-mode autodiff), then back-transformed to `β = δ/γ`, `σ = 1/γ`. Standard errors come
from the delta method applied to the inverse observed-information matrix. `dist=:logistic` and
`dist=:extreme_value` skip the Olsen step and optimize `(β, log σ)` directly.

# Arguments
- `y::AbstractVector` — response (converted to `Float64` internally if needed).
- `X::AbstractMatrix` — `n × k` regressors; include a constant column for an intercept.
- `lower`, `upper` — censoring thresholds.
- `dist::Symbol` — `:normal` (default), `:logistic`, or `:extreme_value`.
- `varnames` — coefficient names (auto-generated if `nothing`).

# Returns
[`TobitModel`](@ref). Use [`marginal_effects`](@ref) for the McDonald–Moffitt decomposition.

# Examples
```julia
using MacroEconometricModels
ystar = 0.4 .+ randn(200)
y = max.(ystar, 0.0)                    # left-censored at 0
X = hcat(ones(200), randn(200))
m = estimate_tobit(y, X; lower=0.0, varnames=["const", "x1"])
report(m)
```

# References
- Tobin, J. (1958). *Econometrica* 26(1), 24-36.
- Olsen, R. J. (1978). *Econometrica* 46(5), 1211-1215.
"""
function estimate_tobit(y::AbstractVector{T}, X::AbstractMatrix{T};
                        lower::Real=0.0, upper::Real=Inf, dist::Symbol=:normal,
                        varnames::Union{Nothing,Vector{String}}=nothing,
                        maxiter::Int=1000, tol::T=T(1e-10)) where {T<:AbstractFloat}
    _validate_data(y, "y")
    _validate_data(X, "X")
    n = length(y)
    k = size(X, 2)
    size(X, 1) == n || throw(ArgumentError("X must have $n rows (got $(size(X, 1)))"))
    n > k || throw(ArgumentError("Need n > k (n=$n, k=$k)"))
    L = T(lower); U = T(upper)
    L < U || throw(ArgumentError("lower ($L) must be < upper ($U)"))
    dist in (:normal, :logistic, :extreme_value) ||
        throw(ArgumentError("dist must be :normal, :logistic, or :extreme_value; got :$dist"))

    yv = Vector{T}(y)
    Xm = Matrix{T}(X)
    vn = something(varnames, ["x$i" for i in 1:k])
    length(vn) == k || throw(ArgumentError("varnames must have length $k"))

    n_left  = isfinite(L) ? count(<=(L), yv) : 0
    n_right = isfinite(U) ? count(>=(U), yv) : 0

    # ---- Starting values from OLS ----
    beta0 = Xm \ yv
    resid0 = yv .- Xm * beta0
    sigma0 = max(std(resid0), sqrt(eps(T)))

    # ---- Optimize ----
    opts = Optim.Options(iterations=maxiter, g_tol=T(1e-11))
    if dist === :normal
        # Olsen space: params = [δ; η], δ = β/σ, η = log(1/σ)
        p0 = vcat(beta0 ./ sigma0, log(one(T) / sigma0))
        obj = p -> _tobit_negll_olsen(p, yv, Xm, L, U)
        g! = (G, p) -> ForwardDiff.gradient!(G, obj, p)
        res = Optim.optimize(obj, g!, p0, Optim.LBFGS(), opts)
        p̂ = Optim.minimizer(res)
        δ̂ = p̂[1:k]; η̂ = p̂[k+1]; γ̂ = exp(η̂)
        σ̂ = one(T) / γ̂
        β̂ = δ̂ ./ γ̂
        # Observed information in (δ, η) space → covariance
        H = ForwardDiff.hessian(obj, p̂)
        Vp = Matrix{T}(robust_inv(H))
        # Jacobian J = ∂(β, σ)/∂(δ, η):  β_j = δ_j e^{-η},  σ = e^{-η}
        #   ∂β_j/∂δ_l = (l==j) e^{-η} = (l==j) σ ;  ∂β_j/∂η = -δ_j e^{-η} = -β_j
        #   ∂σ/∂δ_l   = 0                          ;  ∂σ/∂η   = -e^{-η}   = -σ
        J = zeros(T, k + 1, k + 1)
        for j in 1:k
            J[j, j] = σ̂
            J[j, k+1] = -β̂[j]
        end
        J[k+1, k+1] = -σ̂
        V = Symmetric(J * Vp * J')
    else
        # Direct (β, log σ) space
        p0 = vcat(beta0, log(sigma0))
        obj = p -> _tobit_negll_direct(p, yv, Xm, L, U, dist)
        g! = (G, p) -> ForwardDiff.gradient!(G, obj, p)
        res = Optim.optimize(obj, g!, p0, Optim.LBFGS(), opts)
        p̂ = Optim.minimizer(res)
        β̂ = p̂[1:k]; σ̂ = exp(p̂[k+1])
        H = ForwardDiff.hessian(obj, p̂)
        Vp = Matrix{T}(robust_inv(H))
        # J: β identity, σ = e^{logσ} ⇒ ∂σ/∂logσ = σ
        J = Matrix{T}(I, k + 1, k + 1)
        J[k+1, k+1] = σ̂
        V = Symmetric(J * Vp * J')
    end

    Vfull = Matrix{T}(V)
    converged = Optim.converged(res)
    fitted = Xm * β̂
    resid = yv .- fitted

    D = _tobit_stddist(dist, T)
    loglik = _tobit_loglik(β̂, σ̂, yv, Xm, L, U, D)
    npar = k + 1
    aic = -2 * loglik + 2 * T(npar)
    bic = -2 * loglik + log(T(n)) * T(npar)
    sigma_se = sqrt(max(Vfull[k+1, k+1], zero(T)))

    TobitModel{T}(yv, Xm, β̂, σ̂, Vfull, sigma_se, resid, fitted,
                  loglik, aic, bic, L, U, n_left, n_right, dist, vn, :tobit, converged)
end

estimate_tobit(y::AbstractVector, X::AbstractMatrix; kwargs...) =
    estimate_tobit(Float64.(y), Float64.(X); kwargs...)

# =============================================================================
# estimate_truncreg
# =============================================================================

"""
    estimate_truncreg(y, X; lower=0.0, upper=Inf, varnames=nothing,
                      maxiter=1000, tol=1e-10) -> TruncRegModel{T}

Estimate a truncated-normal regression by maximum likelihood (Hausman & Wise 1977). Unlike
Tobit, the sample is *truncated*: only observations with `lower < y < upper` are observed, and
there is no censored mass. All `y` must lie strictly inside `(lower, upper)` or an error is
thrown.

Optimizes `(β, log σ)` with `Optim.LBFGS` and forward-mode autodiff; delta-method SEs.

# Examples
```julia
using MacroEconometricModels
ystar = 0.4 .+ randn(500); X = hcat(ones(500), randn(500))
keep = ystar .> 0
m = estimate_truncreg(ystar[keep], X[keep, :]; lower=0.0, varnames=["const", "x1"])
report(m)
```

# References
- Hausman, J. A. & Wise, D. A. (1977). *Econometrica* 45(4), 919-938.
"""
function estimate_truncreg(y::AbstractVector{T}, X::AbstractMatrix{T};
                           lower::Real=0.0, upper::Real=Inf,
                           varnames::Union{Nothing,Vector{String}}=nothing,
                           maxiter::Int=1000, tol::T=T(1e-10)) where {T<:AbstractFloat}
    _validate_data(y, "y")
    _validate_data(X, "X")
    n = length(y)
    k = size(X, 2)
    size(X, 1) == n || throw(ArgumentError("X must have $n rows (got $(size(X, 1)))"))
    n > k || throw(ArgumentError("Need n > k (n=$n, k=$k)"))
    L = T(lower); U = T(upper)
    L < U || throw(ArgumentError("lower ($L) must be < upper ($U)"))

    yv = Vector{T}(y)
    all(yi -> L < yi < U, yv) ||
        throw(ArgumentError("truncated regression requires every y strictly inside (lower, upper)"))

    Xm = Matrix{T}(X)
    vn = something(varnames, ["x$i" for i in 1:k])
    length(vn) == k || throw(ArgumentError("varnames must have length $k"))

    beta0 = Xm \ yv
    resid0 = yv .- Xm * beta0
    sigma0 = max(std(resid0), sqrt(eps(T)))

    p0 = vcat(beta0, log(sigma0))
    obj = p -> _truncreg_negll(p, yv, Xm, L, U)
    g! = (G, p) -> ForwardDiff.gradient!(G, obj, p)
    opts = Optim.Options(iterations=maxiter, g_tol=T(1e-11))
    res = Optim.optimize(obj, g!, p0, Optim.LBFGS(), opts)
    p̂ = Optim.minimizer(res)
    β̂ = p̂[1:k]; σ̂ = exp(p̂[k+1])

    H = ForwardDiff.hessian(obj, p̂)
    Vp = Matrix{T}(robust_inv(H))
    J = Matrix{T}(I, k + 1, k + 1)
    J[k+1, k+1] = σ̂
    Vfull = Matrix{T}(Symmetric(J * Vp * J'))

    converged = Optim.converged(res)
    fitted = Xm * β̂
    resid = yv .- fitted
    loglik = -Optim.minimum(res)
    npar = k + 1
    aic = -2 * loglik + 2 * T(npar)
    bic = -2 * loglik + log(T(n)) * T(npar)
    sigma_se = sqrt(max(Vfull[k+1, k+1], zero(T)))

    TruncRegModel{T}(yv, Xm, β̂, σ̂, Vfull, sigma_se, resid, fitted,
                     loglik, aic, bic, L, U, n, :normal, vn, :truncreg, converged)
end

estimate_truncreg(y::AbstractVector, X::AbstractMatrix; kwargs...) =
    estimate_truncreg(Float64.(y), Float64.(X); kwargs...)

# =============================================================================
# McDonald–Moffitt marginal effects for Tobit
# =============================================================================

# Per-observation McDonald–Moffitt "scale" factors as a function of the latent index η=x'β and
# σ, for a given effect `which`. Elementary functions only (ForwardDiff-friendly). Returns the
# scalar s such that ∂effect/∂x_j = s · β_j (continuous regressors).
function _mm_scale(η::S, sigma::S, L::T, U::T, which::Symbol) where {S,T<:AbstractFloat}
    N = Normal(zero(S), one(S))
    a = isfinite(L) ? (S(L) - η) / sigma : S(-Inf)
    b = isfinite(U) ? (S(U) - η) / sigma : S(Inf)
    Φa = isfinite(a) ? cdf(N, a) : zero(S)
    Φb = isfinite(b) ? cdf(N, b) : one(S)
    φa = isfinite(a) ? pdf(N, a) : zero(S)
    φb = isfinite(b) ? pdf(N, b) : zero(S)
    P = Φb - Φa                                  # P(uncensored)
    if which === :unconditional
        return P                                  # ∂E[y]/∂x_j = P · β_j
    elseif which === :probability
        return (φa - φb) / sigma                  # ∂P(uncensored)/∂x_j = (φa-φb)/σ · β_j
    elseif which === :conditional
        Pc = max(P, sqrt(eps(S)))
        num = φa - φb
        aφa = isfinite(a) ? a * φa : zero(S)
        bφb = isfinite(b) ? b * φb : zero(S)
        # d/dη E[y | L<y*<U] = 1 + [ (aφa - bφb)·P - num² ] / P²
        return one(S) + ((aφa - bφb) * Pc - num^2) / Pc^2
    else
        throw(ArgumentError("which must be :unconditional, :conditional, or :probability; got :$which"))
    end
end

# AME effect vector as a function of θ = [β; σ]; averaged over the sample.
function _tobit_ame(theta::AbstractVector{S}, X::Matrix{T}, L::T, U::T,
                    which::Symbol) where {S,T<:AbstractFloat}
    k = size(X, 2)
    beta = @view theta[1:k]
    sigma = theta[k+1]
    n = size(X, 1)
    acc = zero(S)
    @inbounds for i in 1:n
        η = zero(S)
        for j in 1:k
            η += X[i, j] * beta[j]
        end
        acc += _mm_scale(η, sigma, L, U, which)
    end
    s̄ = acc / S(n)
    [s̄ * beta[j] for j in 1:k]
end

"""
    marginal_effects(m::TobitModel; which=:unconditional, conf_level=0.95) -> MarginalEffects{T}

Average marginal effects for a Tobit model via the McDonald–Moffitt (1980) decomposition, with
delta-method standard errors (Jacobian through `(β, σ)` by forward-mode autodiff).

`which` selects the effect:
- `:unconditional` (default) — `∂E[y]/∂xⱼ = β̄ⱼ · mean_i[Φ(bᵢ) − Φ(aᵢ)]`, the total effect on
  the observed (censored) outcome. For classic left-censoring at 0 this is `Φ(x'β/σ)·βⱼ`.
- `:conditional` — `∂E[y | lower<y<upper]/∂xⱼ`, the effect on the uncensored subpopulation mean.
- `:probability` — `∂P(lower<y<upper)/∂xⱼ = (1/σ)[φ(aᵢ)−φ(bᵢ)]·βⱼ`; for left-censoring at 0,
  `(1/σ)φ(x'β/σ)·βⱼ`.

with `aᵢ=(lower−x'ᵢβ)/σ`, `bᵢ=(upper−x'ᵢβ)/σ`. Intercept rows carry `NaN` (no marginal effect).

# References
- McDonald, J. F. & Moffitt, R. A. (1980). *Review of Economics and Statistics* 62(2), 318-321.
- Greene, W. H. (2018). *Econometric Analysis*, 8th ed. Pearson, ch. 19.
"""
function marginal_effects(m::TobitModel{T}; which::Symbol=:unconditional,
                          conf_level::Real=0.95) where {T<:AbstractFloat}
    which in (:unconditional, :conditional, :probability) ||
        throw(ArgumentError("which must be :unconditional, :conditional, or :probability; got :$which"))
    k = length(m.beta)
    theta = vcat(m.beta, m.sigma)
    g = θ -> _tobit_ame(θ, m.X, m.lower, m.upper, which)
    me = g(theta)
    G = ForwardDiff.jacobian(g, theta)            # k × (k+1)
    var_me = G * m.vcov_mat * G'
    se = sqrt.(max.(diag(var_me), zero(T)))

    kinds = _me_column_kinds(m.X)
    z_stat = Vector{T}(undef, k)
    p_values = Vector{T}(undef, k)
    @inbounds for j in 1:k
        z_stat[j] = se[j] > zero(T) ? me[j] / se[j] : zero(T)
        p_values[j] = T(2 * (1 - cdf(Normal(), abs(z_stat[j]))))
    end
    z_crit = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    ci_lower = me .- z_crit .* se
    ci_upper = me .+ z_crit .* se
    @inbounds for j in 1:k
        if kinds[j] == :intercept
            me[j] = T(NaN); se[j] = T(NaN); z_stat[j] = T(NaN)
            p_values[j] = T(NaN); ci_lower[j] = T(NaN); ci_upper[j] = T(NaN)
        end
    end
    MarginalEffects{T}(me, se, z_stat, p_values, ci_lower, ci_upper,
                       copy(m.varnames), :ame, T(conf_level))
end

# =============================================================================
# StatsAPI interface
# =============================================================================

for M in (:TobitModel, :TruncRegModel)
    @eval begin
        StatsAPI.coef(m::$M) = m.beta
        StatsAPI.vcov(m::$M) = m.vcov_mat[1:length(m.beta), 1:length(m.beta)]
        StatsAPI.residuals(m::$M) = m.residuals
        StatsAPI.predict(m::$M) = m.fitted
        StatsAPI.nobs(m::$M) = length(m.y)
        StatsAPI.dof(m::$M) = length(m.beta) + 1
        StatsAPI.dof_residual(m::$M) = length(m.y) - length(m.beta) - 1
        StatsAPI.loglikelihood(m::$M) = m.loglik
        StatsAPI.aic(m::$M) = m.aic
        StatsAPI.bic(m::$M) = m.bic
        StatsAPI.islinear(::$M) = false
        StatsAPI.stderror(m::$M) = sqrt.(max.(diag(m.vcov_mat)[1:length(m.beta)], zero(eltype(m.beta))))
    end
end

function StatsAPI.confint(m::Union{TobitModel{T},TruncRegModel{T}}; level::Real=0.95) where {T}
    se = stderror(m)
    crit = T(quantile(Normal(), 1 - (1 - level) / 2))
    hcat(m.beta .- crit .* se, m.beta .+ crit .* se)
end

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, m::TobitModel{T}) where {T}
    dist_str = m.dist === :normal ? "Normal" :
               m.dist === :logistic ? "Logistic" : "Extreme-value"
    lo = isfinite(m.lower) ? _fmt(m.lower) : "-Inf"
    up = isfinite(m.upper) ? _fmt(m.upper) : "Inf"
    spec = Any[
        "Model"           "Tobit ($dist_str)";
        "Observations"    nobs(m);
        "Covariates"      length(m.beta);
        "Censoring"       "[$lo, $up]";
        "Left-censored"   m.n_censored_left;
        "Right-censored"  m.n_censored_right;
        "Uncensored"      nobs(m) - m.n_censored_left - m.n_censored_right;
        "sigma"           _fmt(m.sigma);
        "Log-lik."        _fmt(m.loglik; digits=2);
        "AIC"             _fmt(m.aic; digits=2);
        "BIC"             _fmt(m.bic; digits=2);
        "Converged"       m.converged ? "Yes" : "No"
    ]
    _pretty_table(io, spec; title="Tobit Regression",
                  column_labels=["Specification", ""], alignment=[:l, :r])
    _coef_table(io, "Coefficients", m.varnames, m.beta, stderror(m); dist=:z)
    _coef_table(io, "Scale", ["sigma"], [m.sigma], [m.sigma_se]; dist=:z)
    _sig_legend(io)
end

function Base.show(io::IO, m::TruncRegModel{T}) where {T}
    lo = isfinite(m.lower) ? _fmt(m.lower) : "-Inf"
    up = isfinite(m.upper) ? _fmt(m.upper) : "Inf"
    spec = Any[
        "Model"         "Truncated regression (Normal)";
        "Observations"  nobs(m);
        "Covariates"    length(m.beta);
        "Truncation"    "($lo, $up)";
        "sigma"         _fmt(m.sigma);
        "Log-lik."      _fmt(m.loglik; digits=2);
        "AIC"           _fmt(m.aic; digits=2);
        "BIC"           _fmt(m.bic; digits=2);
        "Converged"     m.converged ? "Yes" : "No"
    ]
    _pretty_table(io, spec; title="Truncated Regression",
                  column_labels=["Specification", ""], alignment=[:l, :r])
    _coef_table(io, "Coefficients", m.varnames, m.beta, stderror(m); dist=:z)
    _coef_table(io, "Scale", ["sigma"], [m.sigma], [m.sigma_se]; dist=:z)
    _sig_legend(io)
end
