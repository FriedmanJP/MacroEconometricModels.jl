# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Count-data regression: Poisson and Negative-Binomial-2 (EV-19, #427).

Poisson is fit by iteratively reweighted least squares on the canonical log link and
reported by default with the Gourieroux–Monfort–Trognon (1984) pseudo-ML sandwich, which
is consistent for `β` whenever the conditional mean is correct — equidispersion is *not*
required. NegBin2 (Cameron & Trivedi 1986) adds the quadratic variance `μ + αμ²` and is fit
jointly in `(β, log α)`. `dispersion_test` implements the Cameron & Trivedi (1990) auxiliary
regression that discriminates between the two.

Zero-inflated and hurdle variants are out of scope; neither is stubbed here.
"""

using LinearAlgebra, Statistics, Distributions, StatsAPI
import SpecialFunctions

# =============================================================================
# Types
# =============================================================================

"""
    PoissonModel{T} <: StatsAPI.RegressionModel

Poisson regression on the log link, `E[y | x] = exp(x'β + offset)`. See [`estimate_poisson`](@ref).

# Fields
- `y`, `X` — response and regressor matrix (include an intercept column in `X`).
- `beta::Vector{T}` — coefficients on the log-mean index.
- `vcov_mat::Matrix{T}` — coefficient covariance under `cov_type`.
- `residuals::Vector{T}` — deviance residuals (`sign(y-μ)·√dᵢ`), matching the `LogitModel`
  convention. Response residuals are `y - fitted`; Pearson residuals `(y-μ)/√μ`.
- `fitted::Vector{T}` — conditional means `μᵢ`.
- `offset::Union{Nothing,Vector{T}}` — log-exposure entering the index with coefficient 1.
- `loglik`, `loglik_null` — maximized and intercept-only log-likelihoods (both include the
  `-log(y!)` term, so they are comparable with R's `logLik(glm(..., family=poisson))`).
- `pseudo_r2::T` — McFadden `1 - loglik/loglik_null`.
- `deviance`, `null_deviance` — `2Σ[y log(y/μ) - (y-μ)]` at the fit and at the null.
- `aic`, `bic` — information criteria (`k` parameters).
- `varnames::Vector{String}`, `converged::Bool`, `iterations::Int`, `cov_type::Symbol`.

# References
- Gourieroux, C., Monfort, A. & Trognon, A. (1984). *Econometrica* 52(3), 701-720.
- Cameron, A. C. & Trivedi, P. K. (2013). *Regression Analysis of Count Data*. 2nd ed. CUP.
"""
struct PoissonModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    y::Vector{T}
    X::Matrix{T}
    beta::Vector{T}
    vcov_mat::Matrix{T}
    residuals::Vector{T}
    fitted::Vector{T}
    offset::Union{Nothing,Vector{T}}
    loglik::T
    loglik_null::T
    pseudo_r2::T
    deviance::T
    null_deviance::T
    aic::T
    bic::T
    varnames::Vector{String}
    converged::Bool
    iterations::Int
    cov_type::Symbol
end

"""
    NegBinModel{T} <: StatsAPI.RegressionModel

Negative-Binomial-2 regression: `E[y|x] = μ = exp(x'β + offset)`, `Var[y|x] = μ + αμ²`.
See [`estimate_nbreg`](@ref).

# Fields
Mirror [`PoissonModel`](@ref), plus

- `alpha::T` — overdispersion parameter. R's `MASS::glm.nb` reports `theta = 1/alpha`.
- `alpha_se::T` — delta-method standard error of `α` from the joint `(β, log α)` Hessian.
- `vcov_mat::Matrix{T}` — the joint `(k+1)×(k+1)` covariance of `(β, α)` with `α` last;
  `vcov`/`stderror` return the `β` block.

`α → 0` recovers the Poisson model.

# References
- Cameron, A. C. & Trivedi, P. K. (1986). *Journal of Applied Econometrics* 1(1), 29-53.
- Lawless, J. F. (1987). *Canadian Journal of Statistics* 15(3), 209-225.
"""
struct NegBinModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    y::Vector{T}
    X::Matrix{T}
    beta::Vector{T}
    alpha::T
    vcov_mat::Matrix{T}
    alpha_se::T
    residuals::Vector{T}
    fitted::Vector{T}
    offset::Union{Nothing,Vector{T}}
    loglik::T
    loglik_null::T
    pseudo_r2::T
    deviance::T
    null_deviance::T
    aic::T
    bic::T
    varnames::Vector{String}
    converged::Bool
    iterations::Int
    cov_type::Symbol
end

"""
    DispersionTest{T}

Cameron & Trivedi (1990) overdispersion test result. See [`dispersion_test`](@ref).

Carries both auxiliary-regression variants: `nb2` tests `Var = μ + αμ²` (the NegBin2
alternative) and `nb1` tests `Var = (1+α)μ`. Each field is a `NamedTuple`
`(alpha, se, t_stat, p_value)`. A significantly positive `α̂` rejects equidispersion in
favour of overdispersion, i.e. prefer [`estimate_nbreg`](@ref) over [`estimate_poisson`](@ref).
"""
struct DispersionTest{T<:AbstractFloat}
    nb2::NamedTuple{(:alpha, :se, :t_stat, :p_value),NTuple{4,T}}
    nb1::NamedTuple{(:alpha, :se, :t_stat, :p_value),NTuple{4,T}}
    n::Int
end

# =============================================================================
# Likelihood / deviance kernels
# =============================================================================

# log Γ(y+1) summed over the sample. Constant in β, but R's `logLik` includes it and
# comparability with an external oracle is worth the flops.
function _log_factorial_sum(y::Vector{T}) where {T<:AbstractFloat}
    s = zero(T)
    @inbounds for yi in y
        s += SpecialFunctions.loggamma(yi + one(T))
    end
    s
end

"""Poisson log-likelihood `Σ[y log μ - μ - log Γ(y+1)]`."""
function _poisson_loglik(y::Vector{T}, mu::Vector{T}) where {T<:AbstractFloat}
    ll = zero(T)
    @inbounds for i in eachindex(y)
        ll += y[i] * log(mu[i]) - mu[i] - SpecialFunctions.loggamma(y[i] + one(T))
    end
    ll
end

"""Poisson deviance `2Σ[y log(y/μ) - (y-μ)]`; the `y=0` term is `2μ`."""
function _poisson_deviance(y::Vector{T}, mu::Vector{T}) where {T<:AbstractFloat}
    d = zero(T)
    @inbounds for i in eachindex(y)
        yi = y[i]
        d += yi > zero(T) ? 2 * (yi * log(yi / mu[i]) - (yi - mu[i])) : 2 * mu[i]
    end
    d
end

"""Signed-root deviance residuals for the Poisson/NegBin2 fit."""
function _count_dev_residuals(y::Vector{T}, mu::Vector{T}, alpha::T) where {T<:AbstractFloat}
    n = length(y)
    r = Vector{T}(undef, n)
    @inbounds for i in 1:n
        yi = y[i]; mi = mu[i]
        if alpha <= zero(T)
            di = yi > zero(T) ? 2 * (yi * log(yi / mi) - (yi - mi)) : 2 * mi
        else
            th = one(T) / alpha
            t1 = yi > zero(T) ? yi * log(yi / mi) : zero(T)
            di = 2 * (t1 - (yi + th) * log((yi + th) / (mi + th)))
        end
        r[i] = sign(yi - mi) * sqrt(max(di, zero(T)))
    end
    r
end

"""
NegBin2 log-likelihood in the `θ = 1/α` parameterization:
`Σ[logΓ(y+θ) - logΓ(θ) - logΓ(y+1) + θ log(θ/(θ+μ)) + y log(μ/(θ+μ))]`.
Written generically in the parameter type so `ForwardDiff` can differentiate it.
"""
function _nb2_loglik(beta::AbstractVector{S}, alpha::S, y::Vector{T}, X::Matrix{T},
                     off::Vector{T}) where {S,T<:AbstractFloat}
    n, k = size(X)
    theta = one(S) / alpha
    lgt = SpecialFunctions.loggamma(theta)
    ll = zero(S)
    @inbounds for i in 1:n
        eta = S(off[i])
        for j in 1:k
            eta += X[i, j] * beta[j]
        end
        # exp overflows above ~709; the fit is meaningless long before that, but the
        # optimizer must not be handed an Inf while it probes.
        eta = min(eta, S(700))
        mu = exp(eta)
        yi = S(y[i])
        ll += SpecialFunctions.loggamma(yi + theta) - lgt -
              SpecialFunctions.loggamma(yi + one(S)) +
              theta * (log(theta) - log(theta + mu)) +
              yi * (eta - log(theta + mu))
    end
    ll
end

# Negative log-likelihood over p = [β; log α], for Optim.
function _nb2_negll(p::AbstractVector{S}, y::Vector{T}, X::Matrix{T},
                    off::Vector{T}) where {S,T<:AbstractFloat}
    k = size(X, 2)
    beta = @view p[1:k]
    # α is bounded away from 0 so 1/α and logΓ(1/α) stay finite: at α = 1e-10 the NegBin2
    # is numerically Poisson already.
    alpha = max(exp(p[k+1]), S(1e-10))
    -_nb2_loglik(beta, alpha, y, X, off)
end

# =============================================================================
# IRLS for the Poisson log link
# =============================================================================

"""
    _irls_poisson(y, X, off; maxiter=100, tol=1e-10) -> (beta, mu, w, loglik, converged, iterations)

Fisher scoring for Poisson regression on the log link. Working weights `w = μ`, working
response `z = (η - offset) + (y - μ)/μ`, so each step solves `β = (X'WX)⁻¹X'Wz`.

Started from the GLM convention `μ⁰ = y + 0.1` (which is finite even for all-zero cells).
"""
function _irls_poisson(y::Vector{T}, X::Matrix{T}, off::Vector{T};
                       maxiter::Int=100, tol::T=T(1e-10)) where {T<:AbstractFloat}
    n, k = size(X)
    mu = y .+ T(0.1)
    eta = log.(mu)
    beta = zeros(T, k)
    w = Vector{T}(undef, n)
    loglik_old = T(-Inf)
    converged = false
    iter = 0

    for it in 1:maxiter
        iter = it
        @inbounds for i in 1:n
            w[i] = max(mu[i], T(1e-10))
        end
        z = (eta .- off) .+ (y .- mu) ./ w
        W = Diagonal(w)
        beta = robust_inv(X' * W * X) * (X' * W * z)

        eta = X * beta .+ off
        @inbounds for i in 1:n
            eta[i] = min(eta[i], T(700))
            mu[i] = max(exp(eta[i]), T(1e-300))
        end

        loglik_new = _poisson_loglik(y, mu)
        if abs(loglik_new - loglik_old) < tol * (abs(loglik_old) + one(T))
            loglik_old = loglik_new
            converged = true
            break
        end
        loglik_old = loglik_new
    end

    # Return the information weights evaluated AT β̂, not the working weights left over
    # from the previous step. R's `summary.glm` reports the stale ones (it reuses the last
    # IRLS QR), which shifts its standard errors by ~1e-6 relative at this tolerance.
    @inbounds for i in 1:n
        w[i] = max(mu[i], T(1e-10))
    end

    (beta, mu, w, loglik_old, converged, iter)
end

# Shared input handling for both estimators.
function _count_prepare(y::AbstractVector{T}, X::AbstractMatrix{T},
                        offset, exposure,
                        varnames::Union{Nothing,Vector{String}}) where {T<:AbstractFloat}
    _validate_data(y, "y")
    _validate_data(X, "X")
    n = length(y)
    k = size(X, 2)
    size(X, 1) == n || throw(ArgumentError("X must have $n rows (got $(size(X, 1)))"))
    n > k || throw(ArgumentError("Need n > k (n=$n, k=$k)"))

    yv = Vector{T}(y)
    any(<(zero(T)), yv) &&
        throw(ArgumentError("count response y must be nonnegative; found a negative value"))
    # Frequency weights are legitimate counts, so the test is integrality, not `isa Integer`.
    tol_int = sqrt(eps(T))
    bad = findfirst(v -> abs(v - round(v)) > tol_int, yv)
    bad === nothing ||
        throw(ArgumentError("count response y must be integer-valued; y[$bad] = $(yv[bad])"))

    offset === nothing || exposure === nothing ||
        throw(ArgumentError("supply at most one of `offset` and `exposure` (exposure enters as offset = log(exposure))"))
    off = if exposure !== nothing
        length(exposure) == n || throw(ArgumentError("exposure must have length $n"))
        any(<=(zero(T)), exposure) &&
            throw(ArgumentError("exposure must be strictly positive"))
        log.(Vector{T}(exposure))
    elseif offset !== nothing
        length(offset) == n || throw(ArgumentError("offset must have length $n"))
        Vector{T}(offset)
    else
        nothing
    end

    vn = something(varnames, ["x$i" for i in 1:k])
    length(vn) == k || throw(ArgumentError("varnames must have length $k"))

    (yv, Matrix{T}(X), off, vn, n, k)
end

# =============================================================================
# Poisson
# =============================================================================

"""
    estimate_poisson(y, X; offset=nothing, exposure=nothing, cov_type=:robust,
                     varnames=nothing, clusters=nothing, maxiter=100, tol=1e-10) -> PoissonModel{T}

Poisson regression on the log link, `E[y | x] = exp(x'β + offset)`, by IRLS (Fisher scoring).

# Covariance
`cov_type` defaults to **`:robust`** — the Gourieroux–Monfort–Trognon (1984) pseudo-ML
sandwich `A⁻¹BA⁻¹` with `A = X'diag(μ)X` and `B = X'diag((y-μ)²)X`. The Poisson QMLE is
consistent for `β` under a correct conditional mean *regardless of equidispersion*, but the
naive information-matrix errors are only valid under `Var = μ`. Overdispersed data with
`:mle` standard errors badly overstate precision; the robust default is deliberate.

Other options: `:mle` (inverse information, R's `summary(glm(...))` errors), `:hc1`, `:hc2`,
`:hc3` (finite-sample variants of the sandwich), `:cluster` (needs `clusters`).

# Arguments
- `y` — nonnegative, integer-valued counts. Non-integer or negative values raise.
- `X` — `n × k` regressors; include a constant column for an intercept.
- `offset` — added to the index with coefficient 1.
- `exposure` — strictly positive exposure; enters as `offset = log(exposure)`. Supply at
  most one of `offset` / `exposure`.

# Returns
[`PoissonModel`](@ref). See [`incidence_rate_ratio`](@ref), [`marginal_effects`](@ref) and
[`dispersion_test`](@ref).

# Examples
```julia
using MacroEconometricModels
X = hcat(ones(200), randn(200))
mu = exp.(0.5 .+ 0.4 .* X[:, 2])
y = Float64.(rand.(Poisson.(mu)))
m = estimate_poisson(y, X; varnames=["const", "x1"])
report(m)
```

# References
- Gourieroux, C., Monfort, A. & Trognon, A. (1984). *Econometrica* 52(3), 701-720.
- Cameron, A. C. & Trivedi, P. K. (2013). *Regression Analysis of Count Data*. 2nd ed. CUP.
"""
function estimate_poisson(y::AbstractVector{T}, X::AbstractMatrix{T};
                          offset::Union{Nothing,AbstractVector}=nothing,
                          exposure::Union{Nothing,AbstractVector}=nothing,
                          cov_type::Symbol=:robust,
                          varnames::Union{Nothing,Vector{String}}=nothing,
                          clusters::Union{Nothing,AbstractVector}=nothing,
                          maxiter::Int=100,
                          tol::T=T(1e-10)) where {T<:AbstractFloat}
    yv, Xm, off, vn, n, k = _count_prepare(y, X, offset, exposure, varnames)

    cov_type in (:robust, :mle, :hc0, :hc1, :hc2, :hc3, :cluster) ||
        throw(ArgumentError("cov_type must be :robust, :mle, :hc0, :hc1, :hc2, :hc3, or :cluster; got :$cov_type"))
    if cov_type == :cluster
        clusters === nothing && throw(ArgumentError("clusters required for :cluster cov_type"))
        length(clusters) == n || throw(ArgumentError("clusters must have length $n"))
    end

    offv = off === nothing ? zeros(T, n) : off

    beta, mu, w, loglik_val, converged, iterations =
        _irls_poisson(yv, Xm, offv; maxiter=maxiter, tol=tol)
    converged || @warn "Poisson IRLS did not converge in $maxiter iterations; results are unreliable"

    # ---- Null model (intercept only, same offset) ----
    _, mu0, _, loglik_null, _, _ = _irls_poisson(yv, ones(T, n, 1), offv;
                                                 maxiter=maxiter, tol=tol)

    pseudo_r2 = loglik_null == zero(T) ? T(NaN) : one(T) - loglik_val / loglik_null
    dev = _poisson_deviance(yv, mu)
    null_dev = _poisson_deviance(yv, mu0)
    aic_val = -2 * loglik_val + 2 * T(k)
    bic_val = -2 * loglik_val + log(T(n)) * T(k)

    info_inv = Matrix{T}(robust_inv(Xm' * Diagonal(w) * Xm))
    vcov_mat = if cov_type == :mle
        info_inv
    else
        # Score residuals y - μ with the IRLS weights: :robust is the HC0-form sandwich,
        # which is exactly the GMT pseudo-ML variance.
        ct = cov_type == :robust ? :hc0 : cov_type
        _reg_vcov(Xm, yv .- mu, ct, info_inv; clusters=clusters, weights=w)
    end

    PoissonModel{T}(yv, Xm, beta, Matrix{T}(vcov_mat),
                    _count_dev_residuals(yv, mu, zero(T)), mu, off,
                    loglik_val, loglik_null, pseudo_r2, dev, null_dev,
                    aic_val, bic_val, vn, converged, iterations, cov_type)
end

estimate_poisson(y::AbstractVector, X::AbstractMatrix; kwargs...) =
    estimate_poisson(Float64.(y), Float64.(X); kwargs...)

# =============================================================================
# Negative Binomial 2
# =============================================================================

"""
    estimate_nbreg(y, X; offset=nothing, exposure=nothing, varnames=nothing,
                   maxiter=1000, tol=1e-10) -> NegBinModel{T}

Negative-Binomial-2 regression: `E[y|x] = μ = exp(x'β + offset)`, `Var[y|x] = μ + αμ²`.

# Algorithm
The log-likelihood is maximized jointly in `(β, log α)` — the log reparameterization keeps
`α > 0` without a constrained solver — by `Optim.LBFGS` with forward-mode autodiff, started
from the Poisson fit for `β` and from the Cameron–Trivedi moment estimate for `α`. The
`(β, α)` covariance is the delta-method transform of the inverse observed information.

`α → 0` is the Poisson limit; on equidispersed data the optimizer drives `α` to the floor
and the coefficients coincide with [`estimate_poisson`](@ref).

# Note
R's `MASS::glm.nb` reports `theta = 1/alpha`. Convert before comparing.

# Returns
[`NegBinModel`](@ref).

# References
- Cameron, A. C. & Trivedi, P. K. (1986). *Journal of Applied Econometrics* 1(1), 29-53.
- Lawless, J. F. (1987). *Canadian Journal of Statistics* 15(3), 209-225.
"""
function estimate_nbreg(y::AbstractVector{T}, X::AbstractMatrix{T};
                        offset::Union{Nothing,AbstractVector}=nothing,
                        exposure::Union{Nothing,AbstractVector}=nothing,
                        varnames::Union{Nothing,Vector{String}}=nothing,
                        maxiter::Int=1000,
                        tol::T=T(1e-10)) where {T<:AbstractFloat}
    yv, Xm, off, vn, n, k = _count_prepare(y, X, offset, exposure, varnames)
    offv = off === nothing ? zeros(T, n) : off

    # ---- Starting values: Poisson β, Cameron–Trivedi moment α ----
    beta0, mu0, _, _, _, _ = _irls_poisson(yv, Xm, offv; maxiter=100, tol=T(1e-10))
    a_mom = _ct_moment_alpha(yv, mu0)
    alpha0 = clamp(a_mom, T(1e-3), T(10))

    obj = p -> _nb2_negll(p, yv, Xm, offv)
    g! = (G, p) -> ForwardDiff.gradient!(G, obj, p)
    p0 = vcat(beta0, log(alpha0))
    res = Optim.optimize(obj, g!, p0, Optim.LBFGS(),
                         Optim.Options(iterations=maxiter, g_tol=tol))
    p̂ = Optim.minimizer(res)
    beta = Vector{T}(p̂[1:k])
    alpha = max(exp(p̂[k+1]), T(1e-10))
    converged = Optim.converged(res)
    converged || @warn "NegBin2 optimization did not converge; results are unreliable"

    # Covariance in (β, log α), then delta to (β, α): ∂α/∂logα = α.
    H = ForwardDiff.hessian(obj, p̂)
    Vp = Matrix{T}(robust_inv(H))
    J = Matrix{T}(I, k + 1, k + 1)
    J[k+1, k+1] = alpha
    V = Matrix{T}(Symmetric(J * Vp * J'))
    alpha_se = sqrt(max(V[k+1, k+1], zero(T)))

    eta = Xm * beta .+ offv
    mu = exp.(min.(eta, T(700)))
    loglik_val = _nb2_loglik(beta, alpha, yv, Xm, offv)

    # ---- Null model: intercept-only NegBin2, same offset ----
    ones_n = ones(T, n, 1)
    obj0 = p -> _nb2_negll(p, yv, ones_n, offv)
    g0! = (G, p) -> ForwardDiff.gradient!(G, obj0, p)
    res0 = Optim.optimize(obj0, g0!, [log(max(mean(yv), T(1e-6))), log(alpha0)],
                          Optim.LBFGS(), Optim.Options(iterations=maxiter, g_tol=tol))
    loglik_null = -Optim.minimum(res0)

    pseudo_r2 = loglik_null == zero(T) ? T(NaN) : one(T) - loglik_val / loglik_null
    npar = k + 1
    aic_val = -2 * loglik_val + 2 * T(npar)
    bic_val = -2 * loglik_val + log(T(n)) * T(npar)

    # Deviance and null deviance share the fitted α — a deviance is only interpretable
    # against a null measured on the same dispersion scale, which is also the GLM
    # convention R's `glm.nb` follows. `loglik_null`, by contrast, re-estimates α, because
    # the McFadden pseudo-R² compares two separately maximized likelihoods.
    # The null mean for the deviance is the intercept-only fit at the SAME α (with no
    # offset this is just ȳ, independent of α; with one it is not).
    objf = b -> -_nb2_loglik(b, alpha, yv, ones_n, offv)
    resf = Optim.optimize(objf, [log(max(mean(yv), T(1e-6)))], Optim.LBFGS(),
                          Optim.Options(iterations=maxiter, g_tol=tol))
    mu_null = exp.(Optim.minimizer(resf)[1] .+ offv)
    dev = _nb2_deviance(yv, mu, alpha)
    null_dev = _nb2_deviance(yv, mu_null, alpha)

    NegBinModel{T}(yv, Xm, beta, alpha, V, alpha_se,
                   _count_dev_residuals(yv, mu, alpha), mu, off,
                   loglik_val, loglik_null, pseudo_r2, dev, null_dev,
                   aic_val, bic_val, vn, converged, Optim.iterations(res), :mle)
end

estimate_nbreg(y::AbstractVector, X::AbstractMatrix; kwargs...) =
    estimate_nbreg(Float64.(y), Float64.(X); kwargs...)

"""NegBin2 deviance `2Σ[y log(y/μ) - (y+θ) log((y+θ)/(μ+θ))]`, `θ = 1/α`."""
function _nb2_deviance(y::Vector{T}, mu::Vector{T}, alpha::T) where {T<:AbstractFloat}
    th = one(T) / alpha
    d = zero(T)
    @inbounds for i in eachindex(y)
        yi = y[i]
        t1 = yi > zero(T) ? yi * log(yi / mu[i]) : zero(T)
        d += 2 * (t1 - (yi + th) * log((yi + th) / (mu[i] + th)))
    end
    d
end

# =============================================================================
# Overdispersion test — Cameron & Trivedi (1990)
# =============================================================================

# OLS slope of z on mu through the origin: the NB2 moment estimator of alpha.
function _ct_moment_alpha(y::Vector{T}, mu::Vector{T}) where {T<:AbstractFloat}
    num = zero(T); den = zero(T)
    @inbounds for i in eachindex(y)
        z = ((y[i] - mu[i])^2 - y[i]) / mu[i]
        num += z * mu[i]
        den += mu[i]^2
    end
    den > zero(T) ? num / den : zero(T)
end

"""
    dispersion_test(m::PoissonModel) -> DispersionTest{T}

Cameron & Trivedi (1990) test of Poisson equidispersion against a specified
overdispersion alternative, via the auxiliary OLS regression of

    zᵢ = [(yᵢ - μ̂ᵢ)² - yᵢ] / μ̂ᵢ

on a single regressor. Two alternatives are reported:

- `nb2` — `z` on `μ̂` through the origin, testing `Var = μ + αμ²` (the [`estimate_nbreg`](@ref)
  alternative).
- `nb1` — `z` on a constant, testing `Var = (1+α)μ`.

`H₀: α = 0` (equidispersion) is a one-sided alternative in practice: a significantly
**positive** `α̂` calls for NegBin2, while `α̂ < 0` indicates underdispersion, which neither
Poisson nor NegBin2 can represent. The reported `p_value` is two-sided; halve it for the
one-sided reading.

The `t` statistics are asymptotically standard normal under `H₀` and are computed with
ordinary (non-robust) OLS errors, matching the reference implementation.

# References
- Cameron, A. C. & Trivedi, P. K. (1990). *Journal of Econometrics* 46(3), 347-364.
"""
function dispersion_test(m::PoissonModel{T}) where {T<:AbstractFloat}
    y = m.y
    mu = m.fitted
    n = length(y)
    z = [((y[i] - mu[i])^2 - y[i]) / mu[i] for i in 1:n]

    # NB2: regression through the origin on mu.
    Sxx = sum(abs2, mu)
    a2 = Sxx > zero(T) ? dot(z, mu) / Sxx : zero(T)
    r2 = z .- a2 .* mu
    s2 = dot(r2, r2) / T(n - 1)
    se2 = sqrt(s2 / Sxx)
    t2 = se2 > zero(T) ? a2 / se2 : zero(T)

    # NB1: regression on a constant.
    a1 = mean(z)
    r1 = z .- a1
    s1 = dot(r1, r1) / T(n - 1)
    se1 = sqrt(s1 / T(n))
    t1 = se1 > zero(T) ? a1 / se1 : zero(T)

    pv(t) = T(2 * ccdf(Normal(), abs(t)))
    DispersionTest{T}(
        (alpha=a2, se=se2, t_stat=t2, p_value=pv(t2)),
        (alpha=a1, se=se1, t_stat=t1, p_value=pv(t1)),
        n)
end

# =============================================================================
# Incidence-rate ratios
# =============================================================================

"""
    incidence_rate_ratio(m; conf_level=0.95) -> OddsRatio{T}

Incidence-rate ratios `exp(βⱼ)` for a count model — the multiplicative effect of a one-unit
increase in `xⱼ` on the conditional mean — with delta-method standard errors
`SE(IRR) = IRR · SE(β)` and confidence intervals formed on the log scale.

Mirrors [`odds_ratio`](@ref) for binary models and returns the same [`OddsRatio`](@ref)
container. Accepts [`PoissonModel`](@ref) and [`NegBinModel`](@ref).
"""
function incidence_rate_ratio(m::Union{PoissonModel{T},NegBinModel{T}};
                              conf_level::Real=0.95) where {T<:AbstractFloat}
    beta = m.beta
    se_beta = stderror(m)
    irr = exp.(beta)
    z_crit = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    OddsRatio{T}(irr, irr .* se_beta,
                 exp.(beta .- z_crit .* se_beta),
                 exp.(beta .+ z_crit .* se_beta),
                 copy(m.varnames), T(conf_level))
end

# =============================================================================
# Marginal effects
# =============================================================================

# AME vector for a log-link count mean. Continuous columns use the derivative
# β_j·mean(μ); binary {0,1} columns use the discrete change, the same convention the
# binary-choice and Tobit marginal effects already follow.
function _count_ame(beta::AbstractVector{S}, X::Matrix{T}, off::Vector{T},
                    kinds::Vector{Symbol}) where {S,T<:AbstractFloat}
    n, k = size(X)
    acc = zeros(S, k)
    @inbounds for i in 1:n
        eta = S(off[i])
        for j in 1:k
            eta += X[i, j] * beta[j]
        end
        mu = exp(min(eta, S(700)))
        for j in 1:k
            if kinds[j] == :continuous
                acc[j] += mu * beta[j]
            elseif kinds[j] == :binary
                acc[j] += exp(min(eta + (one(S) - X[i, j]) * beta[j], S(700))) -
                          exp(min(eta - X[i, j] * beta[j], S(700)))
            end
        end
    end
    acc ./ S(n)
end

"""
    marginal_effects(m::PoissonModel; conf_level=0.95) -> MarginalEffects{T}
    marginal_effects(m::NegBinModel; conf_level=0.95) -> MarginalEffects{T}

Average marginal effects on the conditional mean, with delta-method standard errors.

Because `E[y|x] = exp(x'β + offset)` for both models, the effects depend only on `β` — the
NegBin2 dispersion `α` does not enter the mean, so only the `β` block of the covariance is
used.

- Continuous `xⱼ`: `∂E[y]/∂xⱼ = βⱼ·μᵢ`, averaged over the sample.
- Binary `{0,1}` `xⱼ`: the discrete change `E[y | xⱼ=1] - E[y | xⱼ=0]`, averaged — the same
  convention as [`marginal_effects`](@ref) for binary-choice models.
- The intercept column carries `NaN` (no marginal effect).
"""
function marginal_effects(m::Union{PoissonModel{T},NegBinModel{T}};
                          conf_level::Real=0.95) where {T<:AbstractFloat}
    k = length(m.beta)
    off = m.offset === nothing ? zeros(T, size(m.X, 1)) : m.offset
    kinds = _me_column_kinds(m.X)
    g = b -> _count_ame(b, m.X, off, kinds)
    me = g(m.beta)
    G = ForwardDiff.jacobian(g, m.beta)             # k × k
    Vb = m.vcov_mat[1:k, 1:k]
    se = sqrt.(max.(diag(G * Vb * G'), zero(T)))

    z_stat = Vector{T}(undef, k)
    p_values = Vector{T}(undef, k)
    @inbounds for j in 1:k
        z_stat[j] = se[j] > zero(T) ? me[j] / se[j] : zero(T)
        p_values[j] = T(2 * ccdf(Normal(), abs(z_stat[j])))
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

for M in (:PoissonModel, :NegBinModel)
    @eval begin
        StatsAPI.coef(m::$M) = m.beta
        StatsAPI.residuals(m::$M) = m.residuals
        StatsAPI.predict(m::$M) = m.fitted
        StatsAPI.fitted(m::$M) = m.fitted
        StatsAPI.nobs(m::$M) = length(m.y)
        StatsAPI.loglikelihood(m::$M) = m.loglik
        StatsAPI.nullloglikelihood(m::$M) = m.loglik_null
        StatsAPI.aic(m::$M) = m.aic
        StatsAPI.bic(m::$M) = m.bic
        StatsAPI.deviance(m::$M) = m.deviance
        StatsAPI.nulldeviance(m::$M) = m.null_deviance
        StatsAPI.islinear(::$M) = false
        StatsAPI.r2(m::$M) = m.pseudo_r2
    end
end

StatsAPI.vcov(m::PoissonModel) = m.vcov_mat
StatsAPI.stderror(m::PoissonModel{T}) where {T} =
    sqrt.(max.(diag(m.vcov_mat), zero(T)))
StatsAPI.dof(m::PoissonModel) = length(m.beta)
StatsAPI.dof_residual(m::PoissonModel) = length(m.y) - length(m.beta)

StatsAPI.vcov(m::NegBinModel) = m.vcov_mat[1:length(m.beta), 1:length(m.beta)]
StatsAPI.stderror(m::NegBinModel{T}) where {T} =
    sqrt.(max.(diag(m.vcov_mat)[1:length(m.beta)], zero(T)))
StatsAPI.dof(m::NegBinModel) = length(m.beta) + 1
StatsAPI.dof_residual(m::NegBinModel) = length(m.y) - length(m.beta) - 1

function StatsAPI.confint(m::Union{PoissonModel{T},NegBinModel{T}}; level::Real=0.95) where {T}
    se = stderror(m)
    crit = T(quantile(Normal(), 1 - (1 - level) / 2))
    hcat(m.beta .- crit .* se, m.beta .+ crit .* se)
end

"""
    predict(m::PoissonModel, Xnew; offset=nothing, exposure=nothing) -> Vector{T}
    predict(m::NegBinModel, Xnew; offset=nothing, exposure=nothing) -> Vector{T}

Conditional means `exp(Xnew·β + offset)` at new regressor values. `offset`/`exposure` follow
the estimator's convention; omitting both sets the offset to zero.
"""
function StatsAPI.predict(m::Union{PoissonModel{T},NegBinModel{T}}, Xnew::AbstractMatrix;
                          offset::Union{Nothing,AbstractVector}=nothing,
                          exposure::Union{Nothing,AbstractVector}=nothing) where {T}
    Xn = Matrix{T}(Xnew)
    size(Xn, 2) == length(m.beta) ||
        throw(ArgumentError("Xnew must have $(length(m.beta)) columns (got $(size(Xn, 2)))"))
    offset === nothing || exposure === nothing ||
        throw(ArgumentError("supply at most one of `offset` and `exposure`"))
    off = exposure !== nothing ? log.(Vector{T}(exposure)) :
          offset !== nothing ? Vector{T}(offset) : zeros(T, size(Xn, 1))
    length(off) == size(Xn, 1) || throw(ArgumentError("offset/exposure must have length $(size(Xn, 1))"))
    exp.(Xn * m.beta .+ off)
end

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, m::PoissonModel{T}) where {T}
    spec = Any[
        "Model"          "Poisson (log link)";
        "Observations"   nobs(m);
        "Covariates"     length(m.beta);
        "Offset"         (m.offset === nothing ? "None" : "Yes");
        "Log-lik."       _fmt(m.loglik; digits=2);
        "Null log-lik."  _fmt(m.loglik_null; digits=2);
        "Pseudo R-sq."   _fmt(m.pseudo_r2);
        "Deviance"       _fmt(m.deviance; digits=2);
        "Null deviance"  _fmt(m.null_deviance; digits=2);
        "AIC"            _fmt(m.aic; digits=2);
        "BIC"            _fmt(m.bic; digits=2);
        "Cov. type"      _label(m.cov_type);
        "Iterations"     m.iterations;
        "Converged"      _yesno(m.converged)
    ]
    _pretty_table(io, spec; title="Poisson Regression",
                  column_labels=["Specification", ""], alignment=[:l, :r])
    _coef_table(io, "Coefficients", m.varnames, m.beta, stderror(m); dist=:z)
    _sig_legend(io)
end

function Base.show(io::IO, m::NegBinModel{T}) where {T}
    spec = Any[
        "Model"          "Negative Binomial 2 (log link)";
        "Observations"   nobs(m);
        "Covariates"     length(m.beta);
        "Offset"         (m.offset === nothing ? "None" : "Yes");
        "alpha"          _fmt(m.alpha);
        "theta = 1/alpha" _fmt(one(T) / m.alpha);
        "Log-lik."       _fmt(m.loglik; digits=2);
        "Null log-lik."  _fmt(m.loglik_null; digits=2);
        "Pseudo R-sq."   _fmt(m.pseudo_r2);
        "Deviance"       _fmt(m.deviance; digits=2);
        "AIC"            _fmt(m.aic; digits=2);
        "BIC"            _fmt(m.bic; digits=2);
        "Converged"      _yesno(m.converged)
    ]
    _pretty_table(io, spec; title="Negative Binomial Regression",
                  column_labels=["Specification", ""], alignment=[:l, :r])
    _coef_table(io, "Coefficients", m.varnames, m.beta, stderror(m); dist=:z)
    _coef_table(io, "Dispersion", ["alpha"], [m.alpha], [m.alpha_se]; dist=:z)
    _sig_legend(io)
end

function Base.show(io::IO, r::DispersionTest{T}) where {T}
    rows = Any[
        "H0"                    "Equidispersion (alpha = 0)";
        "Observations"          r.n;
        "NB2  Var = mu+a*mu^2"  "";
        "  alpha"               _fmt(r.nb2.alpha);
        "  std. error"          _fmt(r.nb2.se);
        "  t"                   _fmt(r.nb2.t_stat; digits=3);
        "  p-value"             _format_pvalue(r.nb2.p_value);
        "NB1  Var = (1+a)*mu"   "";
        "  alpha"               _fmt(r.nb1.alpha);
        "  std. error"          _fmt(r.nb1.se);
        "  t"                   _fmt(r.nb1.t_stat; digits=3);
        "  p-value"             _format_pvalue(r.nb1.p_value)
    ]
    _pretty_table(io, rows; title="Cameron-Trivedi (1990) Overdispersion Test",
                  column_labels=["", ""], alignment=[:l, :r])
    verdict = r.nb2.p_value < T(0.05) ?
        (r.nb2.alpha > zero(T) ?
            "Reject equidispersion: overdispersion detected — prefer estimate_nbreg." :
            "Reject equidispersion: UNDERdispersion — neither Poisson nor NegBin2 fits the variance.") :
        "Fail to reject equidispersion — Poisson is adequate."
    println(io, verdict)
end
