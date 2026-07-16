# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Type definitions and StatsAPI interface for GARCH, EGARCH, and GJR-GARCH models.
"""

# =============================================================================
# GARCH Model Type
# =============================================================================

"""
    GARCHModel{T} <: AbstractVolatilityModel

GARCH(p,q) model (Bollerslev 1986):
σ²ₜ = ω + α₁ε²ₜ₋₁ + ... + αqε²ₜ₋q + β₁σ²ₜ₋₁ + ... + βpσ²ₜ₋p

# Fields
- `y::Vector{T}`: Original data
- `p::Int`: GARCH order (lagged variances)
- `q::Int`: ARCH order (lagged squared residuals)
- `mu::T`: Mean (intercept)
- `omega::T`: Variance intercept (ω > 0)
- `alpha::Vector{T}`: ARCH coefficients [α₁, ..., αq]
- `beta::Vector{T}`: GARCH coefficients [β₁, ..., βp]
- `conditional_variance::Vector{T}`: Estimated conditional variances σ²ₜ
- `standardized_residuals::Vector{T}`: Standardized residuals zₜ = εₜ/σₜ
- `residuals::Vector{T}`: Raw residuals εₜ = yₜ - μ
- `fitted::Vector{T}`: Fitted values (mean)
- `loglik::T`: Log-likelihood
- `aic::T`: Akaike Information Criterion
- `bic::T`: Bayesian Information Criterion
- `method::Symbol`: Estimation method
- `converged::Bool`: Whether optimization converged
- `iterations::Int`: Number of iterations
- `param_vcov::Matrix{T}`: Cached QMLE sandwich covariance in optimization (transform) space
"""
struct GARCHModel{T<:AbstractFloat} <: AbstractVolatilityModel
    y::Vector{T}
    p::Int
    q::Int
    mu::T
    omega::T
    alpha::Vector{T}
    beta::Vector{T}
    conditional_variance::Vector{T}
    standardized_residuals::Vector{T}
    residuals::Vector{T}
    fitted::Vector{T}
    loglik::T
    aic::T
    bic::T
    method::Symbol
    converged::Bool
    iterations::Int
    param_vcov::Matrix{T}
end

# Backward-compatible constructor (no cached covariance): stderror recomputes on demand
function GARCHModel(y::Vector{T}, p::Int, q::Int, mu::T, omega::T,
                    alpha::Vector{T}, beta::Vector{T},
                    conditional_variance::Vector{T}, standardized_residuals::Vector{T},
                    residuals::Vector{T}, fitted::Vector{T},
                    loglik::T, aic::T, bic::T,
                    method::Symbol, converged::Bool, iterations::Int) where {T<:AbstractFloat}
    k = 2 + q + p
    GARCHModel{T}(y, p, q, mu, omega, alpha, beta, conditional_variance,
                  standardized_residuals, residuals, fitted, loglik, aic, bic,
                  method, converged, iterations, fill(T(NaN), k, k))
end

# =============================================================================
# EGARCH Model Type
# =============================================================================

"""
    EGARCHModel{T} <: AbstractVolatilityModel

EGARCH(p,q) model (Nelson 1991):
log(σ²ₜ) = ω + Σαᵢ(|zₜ₋ᵢ| - E|zₜ₋ᵢ|) + Σγᵢzₜ₋ᵢ + Σβⱼlog(σ²ₜ₋ⱼ)

The log specification ensures σ² > 0 without parameter constraints,
and γᵢ captures leverage effects (typically γ < 0).
"""
struct EGARCHModel{T<:AbstractFloat} <: AbstractVolatilityModel
    y::Vector{T}
    p::Int
    q::Int
    mu::T
    omega::T
    alpha::Vector{T}
    gamma::Vector{T}
    beta::Vector{T}
    conditional_variance::Vector{T}
    standardized_residuals::Vector{T}
    residuals::Vector{T}
    fitted::Vector{T}
    loglik::T
    aic::T
    bic::T
    method::Symbol
    converged::Bool
    iterations::Int
    param_vcov::Matrix{T}
end

# Backward-compatible constructor (no cached covariance): stderror recomputes on demand
function EGARCHModel(y::Vector{T}, p::Int, q::Int, mu::T, omega::T,
                     alpha::Vector{T}, gamma::Vector{T}, beta::Vector{T},
                     conditional_variance::Vector{T}, standardized_residuals::Vector{T},
                     residuals::Vector{T}, fitted::Vector{T},
                     loglik::T, aic::T, bic::T,
                     method::Symbol, converged::Bool, iterations::Int) where {T<:AbstractFloat}
    k = 2 + 2q + p
    EGARCHModel{T}(y, p, q, mu, omega, alpha, gamma, beta, conditional_variance,
                   standardized_residuals, residuals, fitted, loglik, aic, bic,
                   method, converged, iterations, fill(T(NaN), k, k))
end

# =============================================================================
# GJR-GARCH Model Type
# =============================================================================

"""
    GJRGARCHModel{T} <: AbstractVolatilityModel

GJR-GARCH(p,q) model (Glosten, Jagannathan & Runkle 1993):
σ²ₜ = ω + Σ(αᵢ + γᵢI(εₜ₋ᵢ < 0))ε²ₜ₋ᵢ + Σβⱼσ²ₜ₋ⱼ

γᵢ > 0 means negative shocks increase variance more than positive shocks.
"""
struct GJRGARCHModel{T<:AbstractFloat} <: AbstractVolatilityModel
    y::Vector{T}
    p::Int
    q::Int
    mu::T
    omega::T
    alpha::Vector{T}
    gamma::Vector{T}
    beta::Vector{T}
    conditional_variance::Vector{T}
    standardized_residuals::Vector{T}
    residuals::Vector{T}
    fitted::Vector{T}
    loglik::T
    aic::T
    bic::T
    method::Symbol
    converged::Bool
    iterations::Int
    param_vcov::Matrix{T}
end

# Backward-compatible constructor (no cached covariance): stderror recomputes on demand
function GJRGARCHModel(y::Vector{T}, p::Int, q::Int, mu::T, omega::T,
                       alpha::Vector{T}, gamma::Vector{T}, beta::Vector{T},
                       conditional_variance::Vector{T}, standardized_residuals::Vector{T},
                       residuals::Vector{T}, fitted::Vector{T},
                       loglik::T, aic::T, bic::T,
                       method::Symbol, converged::Bool, iterations::Int) where {T<:AbstractFloat}
    k = 2 + 2q + p
    GJRGARCHModel{T}(y, p, q, mu, omega, alpha, gamma, beta, conditional_variance,
                     standardized_residuals, residuals, fitted, loglik, aic, bic,
                     method, converged, iterations, fill(T(NaN), k, k))
end

# =============================================================================
# GARCH-MIDAS Model Type (EV-02, #410)
# =============================================================================

"""
    GarchMidasModel{T} <: AbstractVolatilityModel

GARCH-MIDAS model (Engle, Ghysels & Sohn 2013) decomposing the conditional
variance of a high-frequency return `r_{i,t}` into a **short-run** unit-mean
GARCH(1,1) component `g_{i,t}` and a slowly-moving **long-run** component `τ_t`
driven by a MIDAS-weighted low-frequency covariate:

    σ²_{i,t} = τ_t · g_{i,t}
    g_{i,t}  = (1−α−β) + α (r_{i−1,t}−μ)²/τ_{i−1} + β g_{i−1,t}
    τ_t      = exp( m + θ · Σ_{k=1}^{K} φ_k(w) · X_{t−k} )

where `i` indexes the high-frequency observation, `t` its low-frequency block,
`X` is a macro level/volatility series (`rv=:macro`) or realized variance
(`rv=:realized`), and `φ_k(w)` are Beta MIDAS weights (`_midas_weights([1,w], K,
:beta2)`, monotone decaying, summing to 1). The short-run innovation is scaled by
`√τ` so `g` has unconditional mean 1 (τ carries the variance level).

# Fields
- `y::Vector{T}` — full high-frequency return series (untrimmed)
- `x_lf::Vector{T}` — low-frequency driver as supplied (empty for `rv=:realized`)
- `mu::T` — conditional mean μ
- `alpha::T` / `beta::T` — short-run ARCH/GARCH coefficients (α+β<1)
- `m_const::T` — long-run intercept m
- `theta::T` — MIDAS slope θ
- `w::T` — Beta weight shape (w>1 ⇒ monotone decaying)
- `weights::Vector{T}` — realized weight curve φ(ŵ), length `K` (sums to 1)
- `tau::Vector{T}` — long-run component per retained HF obs
- `g::Vector{T}` — short-run component per retained HF obs (unit-mean)
- `conditional_variance::Vector{T}` — total σ² = τ·g per retained HF obs
- `ret_idx::Vector{Int}` — indices into `y` of the retained (non-ragged) HF obs
- `residuals::Vector{T}` — retained raw residuals r−μ
- `standardized_residuals::Vector{T}` — z = (r−μ)/√σ²
- `variance_ratio::T` — VR = Var(log τ)/Var(log σ²), long-run variation share
- `K::Int` — number of MIDAS lags
- `m_freq::Int` — high-to-low frequency ratio (HF obs per LF block)
- `n_blocks::Int` — number of complete low-frequency blocks
- `rv::Symbol` — `:macro` or `:realized`
- `span::Symbol` — `:fixed` (calendar-block τ) or `:rolling` (rolling-RV τ)
- `loglik::T` / `aic::T` / `bic::T`
- `method::Symbol` — estimation method (`:qmle`)
- `converged::Bool` / `iterations::Int`
- `param_vcov::Matrix{T}` — 6×6 QMLE sandwich covariance in optimization space,
  ordered `[μ, log α, log β, m, θ, w̃]`

# References
- Engle, Ghysels & Sohn (2013). *Review of Economics and Statistics* 95(3), 776-797.
"""
struct GarchMidasModel{T<:AbstractFloat} <: AbstractVolatilityModel
    y::Vector{T}
    x_lf::Vector{T}
    mu::T
    alpha::T
    beta::T
    m_const::T
    theta::T
    w::T
    weights::Vector{T}
    tau::Vector{T}
    g::Vector{T}
    conditional_variance::Vector{T}
    ret_idx::Vector{Int}
    residuals::Vector{T}
    standardized_residuals::Vector{T}
    variance_ratio::T
    K::Int
    m_freq::Int
    n_blocks::Int
    rv::Symbol
    span::Symbol
    loglik::T
    aic::T
    bic::T
    method::Symbol
    converged::Bool
    iterations::Int
    param_vcov::Matrix{T}
end

# =============================================================================
# FIGARCH / FIEGARCH Model Types (EV-14, #422)
# =============================================================================

"""
    FIGARCHModel{T} <: AbstractVolatilityModel

Fractionally-integrated GARCH(p,d,q) (Baillie, Bollerslev & Mikkelsen 1996). The
conditional variance follows an **ARCH(∞)** representation obtained by inverting
the fractional-difference operator:

    σ²ₜ = ω/(1−β(1)) + [1 − (1−β(L))⁻¹ φ(L)(1−L)ᵈ] ε²ₜ
        = ω* + Σ_{i≥1} λᵢ ε²_{t−i}

The λ-weights are built by convolving the `(1−L)ᵈ` fractional-difference weights
(EV-13's `_frac_diff_weights`) with `φ(L)` and the inverse of `(1−β(L))`, then
truncating at `truncation` lags. The memory parameter `d ∈ (0,1)` gives the
hyperbolic decay `λᵢ ∝ i^{−1−d}` characteristic of long-memory volatility. As
`d → 0`, FIGARCH(1,0,1) collapses to GARCH(1,1) with `α = φ − β`.

Non-negativity of the λ-weights (Baillie–Bollerslev–Mikkelsen inequality
conditions) guarantees a well-defined variance process; a violation is **warned**
(not thrown) and the count of negative truncated weights is stored in
`n_neg_lambda`.

# Fields
- `y::Vector{T}`: Return series `r`
- `p::Int` / `q::Int`: GARCH `β(L)` order and ARCH `φ(L)` order
- `mu::T`: Conditional mean μ
- `omega::T`: Variance intercept ω (> 0); the ARCH(∞) intercept is `ω/(1−β(1))`
- `phi::Vector{T}`: `φ(L)` coefficients `[φ₁,…,φq]`
- `beta::Vector{T}`: `β(L)` coefficients `[β₁,…,βp]`
- `d::T`: Fractional integration order `d ∈ (0,1)`
- `lambda::Vector{T}`: Truncated ARCH(∞) weights `[λ₁,…,λ_trunc]`
- `conditional_variance::Vector{T}`: Fitted σ²ₜ
- `standardized_residuals::Vector{T}`: zₜ = εₜ/σₜ
- `residuals::Vector{T}`: εₜ = yₜ − μ
- `fitted::Vector{T}`: Fitted mean (constant μ)
- `loglik::T` / `aic::T` / `bic::T`: Fit statistics
- `truncation::Int`: ARCH(∞) truncation lag
- `n_neg_lambda::Int`: Number of negative truncated λ-weights (BBM violation count)
- `method::Symbol`: Estimation method (`:qmle`)
- `converged::Bool` / `iterations::Int`
- `param_vcov::Matrix{T}`: Cached Bollerslev–Wooldridge QMLE sandwich covariance in
  optimization (transform) space, ordered `[μ, log ω, logit φ, logit β, logit d]`

# References
- Baillie, Bollerslev & Mikkelsen (1996). *Journal of Econometrics* 74(1), 3–30.
"""
struct FIGARCHModel{T<:AbstractFloat} <: AbstractVolatilityModel
    y::Vector{T}
    p::Int
    q::Int
    mu::T
    omega::T
    phi::Vector{T}
    beta::Vector{T}
    d::T
    lambda::Vector{T}
    conditional_variance::Vector{T}
    standardized_residuals::Vector{T}
    residuals::Vector{T}
    fitted::Vector{T}
    loglik::T
    aic::T
    bic::T
    truncation::Int
    n_neg_lambda::Int
    method::Symbol
    converged::Bool
    iterations::Int
    param_vcov::Matrix{T}
end

"""
    FIEGARCHModel{T} <: AbstractVolatilityModel

Fractionally-integrated EGARCH (Bollerslev & Mikkelsen 1996) — the log-variance
long-memory analogue of FIGARCH:

    ln σ²ₜ = ω + (1−β(L))⁻¹ φ(L)(1−L)^{−d} g(z_{t−1})
           = ω + Σ_{j≥0} ψⱼ g(z_{t−1−j})

where `g(z) = θz + γ(|z| − E|z|)` is the EGARCH news function (θ the sign/leverage
term, γ the magnitude term) and `E|z| = √(2/π)` under Gaussianity. Because the log
variance is unconstrained, there is **no positivity constraint** — long memory
enters through `(1−L)^{−d}` (note the negative exponent), whose MA(∞) weights are
built from `_frac_diff_weights(−d, truncation)` convolved with `φ(L)/(1−β(L))`.

# Fields
Same layout as [`FIGARCHModel`](@ref) plus the news-function parameters, with
`psi` the truncated MA(∞) weights and no `n_neg_lambda`:
- `theta::T`: EGARCH sign/leverage coefficient θ
- `gamma::T`: EGARCH magnitude coefficient γ
- `psi::Vector{T}`: Truncated MA(∞) weights `[ψ₀,…,ψ_{trunc}]`
- `param_vcov`: ordered `[μ, ω, θ, γ, logit φ, logit β, logit d]`

# References
- Bollerslev & Mikkelsen (1996). *Journal of Econometrics* 73(1), 151–184.
"""
struct FIEGARCHModel{T<:AbstractFloat} <: AbstractVolatilityModel
    y::Vector{T}
    p::Int
    q::Int
    mu::T
    omega::T
    theta::T
    gamma::T
    phi::Vector{T}
    beta::Vector{T}
    d::T
    psi::Vector{T}
    conditional_variance::Vector{T}
    standardized_residuals::Vector{T}
    residuals::Vector{T}
    fitted::Vector{T}
    loglik::T
    aic::T
    bic::T
    truncation::Int
    method::Symbol
    converged::Bool
    iterations::Int
    param_vcov::Matrix{T}
end

# =============================================================================
# Type Accessors
# =============================================================================

arch_order(m::GARCHModel) = m.q
arch_order(m::EGARCHModel) = m.q
arch_order(m::GJRGARCHModel) = m.q

"""Return GARCH order p."""
garch_order(m::GARCHModel) = m.p
garch_order(m::EGARCHModel) = m.p
garch_order(m::GJRGARCHModel) = m.p

persistence(m::GARCHModel) = sum(m.alpha) + sum(m.beta)
persistence(m::EGARCHModel) = sum(m.beta)
persistence(m::GJRGARCHModel) = sum(m.alpha) + sum(m.gamma) / 2 + sum(m.beta)

function halflife(m::Union{GARCHModel, EGARCHModel, GJRGARCHModel})
    p = persistence(m)
    p <= zero(p) && return Inf
    p >= one(p) && return Inf
    log(typeof(p)(0.5)) / log(p)
end

function unconditional_variance(m::GARCHModel)
    p = persistence(m)
    p >= one(p) && return typeof(m.omega)(Inf)
    m.omega / (one(p) - p)
end

function unconditional_variance(m::EGARCHModel{T}) where {T}
    # For EGARCH, unconditional log-variance = ω / (1 - Σβⱼ)
    sb = sum(m.beta)
    sb >= one(T) && return T(Inf)
    exp(m.omega / (one(T) - sb))
end

function unconditional_variance(m::GJRGARCHModel)
    p = persistence(m)
    p >= one(p) && return typeof(m.omega)(Inf)
    m.omega / (one(p) - p)
end

# =============================================================================
# StatsAPI Interface
# =============================================================================

"""Number of observations."""
StatsAPI.nobs(m::GARCHModel) = length(m.y)
"""Number of observations."""
StatsAPI.nobs(m::EGARCHModel) = length(m.y)
"""Number of observations."""
StatsAPI.nobs(m::GJRGARCHModel) = length(m.y)

"""Coefficient vector `[μ, ω, α₁, …, αq, β₁, …, βp]`."""
StatsAPI.coef(m::GARCHModel) = vcat(m.mu, m.omega, m.alpha, m.beta)
"""Coefficient vector `[μ, ω, α₁, …, αq, γ₁, …, γq, β₁, …, βp]`."""
StatsAPI.coef(m::EGARCHModel) = vcat(m.mu, m.omega, m.alpha, m.gamma, m.beta)
"""Coefficient vector `[μ, ω, α₁, …, αq, γ₁, …, γq, β₁, …, βp]`."""
StatsAPI.coef(m::GJRGARCHModel) = vcat(m.mu, m.omega, m.alpha, m.gamma, m.beta)

"""Raw residuals ``\\hat{\\varepsilon}_t``."""
StatsAPI.residuals(m::GARCHModel) = m.residuals
"""Raw residuals ``\\hat{\\varepsilon}_t``."""
StatsAPI.residuals(m::EGARCHModel) = m.residuals
"""Raw residuals ``\\hat{\\varepsilon}_t``."""
StatsAPI.residuals(m::GJRGARCHModel) = m.residuals

"""Conditional variance series ``\\hat{\\sigma}^2_t``."""
StatsAPI.predict(m::GARCHModel) = m.conditional_variance
"""Conditional variance series ``\\hat{\\sigma}^2_t``."""
StatsAPI.predict(m::EGARCHModel) = m.conditional_variance
"""Conditional variance series ``\\hat{\\sigma}^2_t``."""
StatsAPI.predict(m::GJRGARCHModel) = m.conditional_variance

"""Maximized log-likelihood."""
StatsAPI.loglikelihood(m::GARCHModel) = m.loglik
"""Maximized log-likelihood."""
StatsAPI.loglikelihood(m::EGARCHModel) = m.loglik
"""Maximized log-likelihood."""
StatsAPI.loglikelihood(m::GJRGARCHModel) = m.loglik

"""Akaike Information Criterion."""
StatsAPI.aic(m::GARCHModel) = m.aic
"""Akaike Information Criterion."""
StatsAPI.aic(m::EGARCHModel) = m.aic
"""Akaike Information Criterion."""
StatsAPI.aic(m::GJRGARCHModel) = m.aic

"""Bayesian Information Criterion."""
StatsAPI.bic(m::GARCHModel) = m.bic
"""Bayesian Information Criterion."""
StatsAPI.bic(m::EGARCHModel) = m.bic
"""Bayesian Information Criterion."""
StatsAPI.bic(m::GJRGARCHModel) = m.bic

"""Number of estimated parameters: `2 + q + p`."""
StatsAPI.dof(m::GARCHModel) = 2 + m.q + m.p        # mu + omega + q alphas + p betas
"""Number of estimated parameters: `2 + 2q + p`."""
StatsAPI.dof(m::EGARCHModel) = 2 + 2 * m.q + m.p   # mu + omega + q alphas + q gammas + p betas
"""Number of estimated parameters: `2 + 2q + p`."""
StatsAPI.dof(m::GJRGARCHModel) = 2 + 2 * m.q + m.p # mu + omega + q alphas + q gammas + p betas

"""`false` — GARCH models are nonlinear."""
StatsAPI.islinear(::GARCHModel) = false
"""`false` — EGARCH models are nonlinear."""
StatsAPI.islinear(::EGARCHModel) = false
"""`false` — GJR-GARCH models are nonlinear."""
StatsAPI.islinear(::GJRGARCHModel) = false

"""Residual degrees of freedom."""
StatsAPI.dof_residual(m::GARCHModel) = length(m.residuals) - dof(m)
"""Residual degrees of freedom."""
StatsAPI.dof_residual(m::EGARCHModel) = length(m.residuals) - dof(m)
"""Residual degrees of freedom."""
StatsAPI.dof_residual(m::GJRGARCHModel) = length(m.residuals) - dof(m)

# =============================================================================
# Display
# =============================================================================

Base.show(io::IO, m::GARCHModel) =
    _show_volatility_model(io, "GARCH($(m.p),$(m.q)) Model", m; alpha=m.alpha, beta=m.beta)

Base.show(io::IO, m::EGARCHModel) =
    _show_volatility_model(io, "EGARCH($(m.p),$(m.q)) Model", m; alpha=m.alpha, gamma=m.gamma, beta=m.beta)

Base.show(io::IO, m::GJRGARCHModel) =
    _show_volatility_model(io, "GJR-GARCH($(m.p),$(m.q)) Model", m; alpha=m.alpha, gamma=m.gamma, beta=m.beta)

# =============================================================================
# GARCH-MIDAS accessors / StatsAPI / display (EV-02, #410)
# =============================================================================

"""Short-run persistence α+β of a GARCH-MIDAS model."""
persistence(m::GarchMidasModel) = m.alpha + m.beta

function halflife(m::GarchMidasModel)
    p = persistence(m)
    (p <= zero(p) || p >= one(p)) && return Inf
    log(typeof(p)(0.5)) / log(p)
end

"""Number of retained (non-ragged) high-frequency observations."""
StatsAPI.nobs(m::GarchMidasModel) = length(m.ret_idx)
"""Coefficient vector `[μ, α, β, m, θ, w]`."""
StatsAPI.coef(m::GarchMidasModel) = [m.mu, m.alpha, m.beta, m.m_const, m.theta, m.w]
"""Retained raw residuals ``r_{i,t}-\\mu``."""
StatsAPI.residuals(m::GarchMidasModel) = m.residuals
"""Total conditional variance series ``\\hat{\\sigma}^2 = \\tau g``."""
StatsAPI.predict(m::GarchMidasModel) = m.conditional_variance
"""Maximized (quasi) log-likelihood."""
StatsAPI.loglikelihood(m::GarchMidasModel) = m.loglik
"""Akaike Information Criterion."""
StatsAPI.aic(m::GarchMidasModel) = m.aic
"""Bayesian Information Criterion."""
StatsAPI.bic(m::GarchMidasModel) = m.bic
"""Number of estimated parameters: 6 `[μ, α, β, m, θ, w]`."""
StatsAPI.dof(::GarchMidasModel) = 6
"""Residual degrees of freedom."""
StatsAPI.dof_residual(m::GarchMidasModel) = length(m.ret_idx) - 6
"""`false` — GARCH-MIDAS is nonlinear."""
StatsAPI.islinear(::GarchMidasModel) = false

Base.show(io::IO, m::GarchMidasModel) = _show_garch_midas(io, m)

# =============================================================================
# FIGARCH / FIEGARCH accessors / StatsAPI / display (EV-14, #422)
# =============================================================================

arch_order(m::FIGARCHModel) = m.q
arch_order(m::FIEGARCHModel) = m.q
garch_order(m::FIGARCHModel) = m.p
garch_order(m::FIEGARCHModel) = m.p

"""Long-memory (fractional integration) parameter `d`, governing hyperbolic decay."""
persistence(m::FIGARCHModel) = m.d
persistence(m::FIEGARCHModel) = m.d

"""Fractional-integration order `d` (long-memory parameter)."""
frac_order(m::FIGARCHModel) = m.d
frac_order(m::FIEGARCHModel) = m.d

StatsAPI.nobs(m::FIGARCHModel) = length(m.y)
StatsAPI.nobs(m::FIEGARCHModel) = length(m.y)

"""Coefficient vector `[μ, ω, φ₁…φq, β₁…βp, d]`."""
StatsAPI.coef(m::FIGARCHModel) = vcat(m.mu, m.omega, m.phi, m.beta, m.d)
"""Coefficient vector `[μ, ω, θ, γ, φ₁…φq, β₁…βp, d]`."""
StatsAPI.coef(m::FIEGARCHModel) = vcat(m.mu, m.omega, m.theta, m.gamma, m.phi, m.beta, m.d)

StatsAPI.residuals(m::FIGARCHModel) = m.residuals
StatsAPI.residuals(m::FIEGARCHModel) = m.residuals
StatsAPI.predict(m::FIGARCHModel) = m.conditional_variance
StatsAPI.predict(m::FIEGARCHModel) = m.conditional_variance
StatsAPI.loglikelihood(m::FIGARCHModel) = m.loglik
StatsAPI.loglikelihood(m::FIEGARCHModel) = m.loglik
StatsAPI.aic(m::FIGARCHModel) = m.aic
StatsAPI.aic(m::FIEGARCHModel) = m.aic
StatsAPI.bic(m::FIGARCHModel) = m.bic
StatsAPI.bic(m::FIEGARCHModel) = m.bic

"""Number of estimated parameters: `3 + q + p` (μ, ω, φ, β, d)."""
StatsAPI.dof(m::FIGARCHModel) = 3 + m.q + m.p
"""Number of estimated parameters: `5 + q + p` (μ, ω, θ, γ, φ, β, d)."""
StatsAPI.dof(m::FIEGARCHModel) = 5 + m.q + m.p
StatsAPI.islinear(::FIGARCHModel) = false
StatsAPI.islinear(::FIEGARCHModel) = false
StatsAPI.dof_residual(m::FIGARCHModel) = length(m.residuals) - dof(m)
StatsAPI.dof_residual(m::FIEGARCHModel) = length(m.residuals) - dof(m)

Base.show(io::IO, m::FIGARCHModel) = _show_figarch(io, m)
Base.show(io::IO, m::FIEGARCHModel) = _show_fiegarch(io, m)
