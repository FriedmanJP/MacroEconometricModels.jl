# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Type definitions for MIDAS (MIxed-DAta Sampling) regression models: a
low-frequency target regressed on many high-frequency lags through a
parsimonious weighting function (exp-Almon / Beta / polynomial Almon) or an
unrestricted U-MIDAS lag polynomial.
"""

using LinearAlgebra, Statistics, Distributions, StatsAPI

# =============================================================================
# MidasModel — restricted MIDAS / ADL-MIDAS / U-MIDAS
# =============================================================================

"""
    MidasModel{T} <: StatsAPI.RegressionModel

Mixed-data-sampling (MIDAS) regression of a low-frequency target on `K`
high-frequency lags aggregated through a weighting function `w(θ)`, optionally
augmented with `p_ar` autoregressive lags of the target (ADL-MIDAS).

The estimated equation is

    y_t = β₀ + β₁ · Σₖ wₖ(θ) x_{t,k} + Σⱼ ρⱼ y_{t−j} + u_t

where `x_{t,k}` is the `k`-th high-frequency lag (most-recent-first) within
low-frequency period `t`. For `weights_kind == :umidas` the weight function is
dropped and the `K` lags enter with unrestricted coefficients (plain OLS).

# Fields
- `y::Vector{T}` — aligned low-frequency target (retained rows only)
- `Xlags::Matrix{T}` — `n×K` stacked high-frequency lags (most-recent-first)
- `Wlin::Matrix{T}` — `n×(1+p_ar)` linear block `[1, y_{t-1}, …, y_{t-p_ar}]`
- `theta::Vector{T}` — weight-function parameters (empty for `:umidas`)
- `beta::Vector{T}` — linear coefficients `[β₀, β₁, ρ₁, …, ρ_{p_ar}]` (restricted)
  or `[β₀, b₁, …, b_K, ρ₁, …]` (`:umidas`)
- `vcov_mat::Matrix{T}` — variance-covariance of all free parameters `[β; θ]`
- `weights_kind::Symbol` — `:expalmon`, `:beta2`, `:beta3`, `:almon`, or `:umidas`
- `m::Int` — high-to-low frequency ratio (3 = monthly→quarterly, ≈66 = daily→quarterly)
- `K::Int` — number of high-frequency lags
- `p_ar::Int` — number of autoregressive lags (0 = plain MIDAS)
- `poly_degree::Int` — polynomial degree for `:almon`
- `h::Int` — direct forecast horizon the model targets (1 = nowcast)
- `w::Vector{T}` — realized weight curve `w(θ̂)`, length `K` (`:umidas` ⇒ raw lag coefs)
- `fitted::Vector{T}` — in-sample fitted values
- `residuals::Vector{T}` — in-sample residuals
- `ssr::T` — sum of squared residuals
- `sigma2::T` — residual variance `SSR/(n − p)`
- `r2::T` / `adj_r2::T` — (adjusted) R²
- `loglik::T` / `aic::T` / `bic::T` — Gaussian log-likelihood and information criteria
- `varnames::Vector{String}` — parameter names aligned with `[β; θ]`
- `converged::Bool` — NLS convergence flag (always `true` for `:umidas`)

# References
- Ghysels, Sinko & Valkanov (2007). *Econometric Reviews* 26(1), 53-90.
- Andreou, Ghysels & Kourtellos (2010). *JBES* 28(4), 493-524.
- Foroni, Marcellino & Schumacher (2015). *JRSS-A* 178(1), 57-82 (U-MIDAS).
"""
struct MidasModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    y::Vector{T}
    Xlags::Matrix{T}
    Wlin::Matrix{T}
    theta::Vector{T}
    beta::Vector{T}
    vcov_mat::Matrix{T}
    weights_kind::Symbol
    m::Int
    K::Int
    p_ar::Int
    poly_degree::Int
    h::Int
    w::Vector{T}
    fitted::Vector{T}
    residuals::Vector{T}
    ssr::T
    sigma2::T
    r2::T
    adj_r2::T
    loglik::T
    aic::T
    bic::T
    varnames::Vector{String}
    converged::Bool
end

# --- StatsAPI interface -------------------------------------------------------

StatsAPI.coef(m::MidasModel) = vcat(m.beta, m.theta)
StatsAPI.vcov(m::MidasModel) = m.vcov_mat
StatsAPI.residuals(m::MidasModel) = m.residuals
StatsAPI.predict(m::MidasModel) = m.fitted
StatsAPI.fitted(m::MidasModel) = m.fitted
StatsAPI.nobs(m::MidasModel) = length(m.y)
StatsAPI.dof(m::MidasModel) = length(m.beta) + length(m.theta)
StatsAPI.dof_residual(m::MidasModel) = length(m.y) - StatsAPI.dof(m)
StatsAPI.loglikelihood(m::MidasModel) = m.loglik
StatsAPI.aic(m::MidasModel) = m.aic
StatsAPI.bic(m::MidasModel) = m.bic
StatsAPI.r2(m::MidasModel) = m.r2
StatsAPI.islinear(::MidasModel) = false
StatsAPI.stderror(m::MidasModel) = sqrt.(abs.(diag(m.vcov_mat)))

"""
    midas_weights(m::MidasModel) -> Vector

Return the realized weight curve `w(θ̂)` (length `K`, most-recent-first). For
`:umidas` models this is the vector of unrestricted lag coefficients.
"""
midas_weights(m::MidasModel) = m.w

"""
    midas_weights(theta, K; kind=:expalmon) -> Vector

Evaluate a MIDAS weight function directly from parameters `theta`. Thin public
wrapper over the internal `_midas_weights`. Each supported `kind` returns a
length-`K` vector summing to 1.
"""
function midas_weights(theta::AbstractVector, K::Int; kind::Symbol=:expalmon)
    T = float(promote_type(eltype(theta), Float64))
    _midas_weights(convert(Vector{T}, collect(theta)), K, kind)
end

# =============================================================================
# MidasForecast — direct multi-step point forecast + NLS prediction interval
# =============================================================================

"""
    MidasForecast{T} <: AbstractForecastResult{T}

Direct `h`-step MIDAS forecast with a Gaussian NLS prediction interval.

# Fields
- `forecast::Vector{T}` — point forecast(s)
- `ci_lower::Vector{T}` / `ci_upper::Vector{T}` — prediction-interval bounds
- `se::Vector{T}` — prediction standard error(s)
- `horizon::Int` — direct forecast horizon
- `conf_level::T` — nominal coverage (e.g. 0.95)
"""
struct MidasForecast{T<:AbstractFloat} <: AbstractForecastResult{T}
    forecast::Vector{T}
    ci_lower::Vector{T}
    ci_upper::Vector{T}
    se::Vector{T}
    horizon::Int
    conf_level::T
end
