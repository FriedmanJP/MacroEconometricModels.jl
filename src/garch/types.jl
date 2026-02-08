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

# =============================================================================
# Display
# =============================================================================

Base.show(io::IO, m::GARCHModel) =
    _show_volatility_model(io, "GARCH($(m.p),$(m.q)) Model", m; alpha=m.alpha, beta=m.beta)

Base.show(io::IO, m::EGARCHModel) =
    _show_volatility_model(io, "EGARCH($(m.p),$(m.q)) Model", m; alpha=m.alpha, gamma=m.gamma, beta=m.beta)

Base.show(io::IO, m::GJRGARCHModel) =
    _show_volatility_model(io, "GJR-GARCH($(m.p),$(m.q)) Model", m; alpha=m.alpha, gamma=m.gamma, beta=m.beta)
