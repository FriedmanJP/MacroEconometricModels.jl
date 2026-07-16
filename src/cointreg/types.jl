# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Type definitions for single-equation cointegrating regression: Fully-Modified OLS
(Phillips–Hansen 1990), Canonical Cointegrating Regression (Park 1992), and Dynamic
OLS (Saikkonen 1991 / Stock–Watson 1993).
"""

using LinearAlgebra, Statistics, Distributions, StatsAPI

# =============================================================================
# CointRegModel — FMOLS / CCR / DOLS
# =============================================================================

"""
    CointRegModel{T} <: StatsAPI.RegressionModel

Single-equation cointegrating regression estimated by FMOLS, CCR, or DOLS. All three
estimators correct the OLS-on-levels regression `y_t = D_t'δ + x_t'β + u_t` (with `x_t`
an `I(1)` vector) for regressor endogeneity and error serial correlation to deliver the
asymptotically efficient, mixed-normal long-run coefficient vector `θ = (δ, β)`.

# Fields
- `y::Vector{T}` — dependent variable (levels)
- `X::Matrix{T}` — `T×k` stochastic (`I(1)`) regressor matrix (no deterministics)
- `method::Symbol` — `:fmols`, `:ccr`, or `:dols`
- `trend::Symbol` — deterministics: `:none`, `:const`, or `:linear`
- `kernel::Symbol` — HAC kernel used for the long-run covariance (`:bartlett`, `:parzen`, …)
- `bandwidth::T` — the resolved truncation-lag bandwidth actually used
- `coef::Vector{T}` — corrected long-run coefficients, ordered `[deterministics; stochastic]`
- `vcov::Matrix{T}` — long-run covariance matrix of `coef` (`(d+k)×(d+k)`)
- `residuals::Vector{T}` — regression residuals `y − [D X]·coef`
- `fitted::Vector{T}` — fitted values `[D X]·coef`
- `varnames::Vector{String}` — coefficient names (length `d+k`)
- `nobs::Int` — number of level observations `T`
- `leads::Int` — DOLS leads of `Δx` (0 for FMOLS/CCR)
- `lags::Int` — DOLS lags of `Δx` (0 for FMOLS/CCR)
- `Omega::Matrix{T}` — two-sided long-run covariance `Ω̂` of the stacked `(u, Δx)` process
- `Lambda::Matrix{T}` — one-sided long-run covariance `Λ̂ = Σ_{j≥0} Γ_j`, `Γ_j = T⁻¹ Σ_t ξ_t ξ_{t−j}'`
- `Sigma::Matrix{T}` — contemporaneous covariance `Σ̂ = Γ̂₀` of `(u, Δx)`
- `omega_uv::T` — conditional long-run variance `ω̂_{u·Δx} = Ω̂_{uu} − Ω̂_{uΔx}Ω̂_{ΔxΔx}⁻¹Ω̂_{Δxu}`
- `d::Int` — number of deterministic columns
- `k::Int` — number of stochastic regressors

# Block layout of `Omega`/`Lambda`/`Sigma`

The three long-run-covariance matrices are stored in the **stacked `(u, Δx)` ordering**
consumed by downstream stability/spurious tests (EV-11) and panel cointegration (EV-22):
row/column `1` is the equation residual `u`, and rows/columns `2:(k+1)` are `Δx`
(the first differences of the `k` stochastic regressors). `Λ̂` is the one-sided sum
`Σ_{j≥0} Γ_j` (**not** transposed), so `Λ̂ + Λ̂' − Γ̂₀ = Ω̂`.

# References
- Phillips, P. C. B. & Hansen, B. E. (1990). *Review of Economic Studies* 57(1), 99–125.
- Park, J. Y. (1992). *Econometrica* 60(1), 119–143.
- Saikkonen, P. (1991). *Econometric Theory* 7(1), 1–21.
- Stock, J. H. & Watson, M. W. (1993). *Econometrica* 61(4), 783–820.
"""
struct CointRegModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    y::Vector{T}
    X::Matrix{T}
    method::Symbol
    trend::Symbol
    kernel::Symbol
    bandwidth::T
    coef::Vector{T}
    vcov::Matrix{T}
    residuals::Vector{T}
    fitted::Vector{T}
    varnames::Vector{String}
    nobs::Int
    leads::Int
    lags::Int
    Omega::Matrix{T}
    Lambda::Matrix{T}
    Sigma::Matrix{T}
    omega_uv::T
    d::Int
    k::Int
end

# =============================================================================
# StatsAPI interface
# =============================================================================

StatsAPI.coef(m::CointRegModel) = m.coef
StatsAPI.vcov(m::CointRegModel) = m.vcov
StatsAPI.residuals(m::CointRegModel) = m.residuals
StatsAPI.predict(m::CointRegModel) = m.fitted
StatsAPI.fitted(m::CointRegModel) = m.fitted
StatsAPI.nobs(m::CointRegModel) = m.nobs
StatsAPI.dof(m::CointRegModel) = length(m.coef)
StatsAPI.dof_residual(m::CointRegModel) = m.nobs - m.d - m.k
StatsAPI.coefnames(m::CointRegModel) = m.varnames
StatsAPI.islinear(::CointRegModel) = true
StatsAPI.stderror(m::CointRegModel) = sqrt.(max.(diag(m.vcov), zero(eltype(m.vcov))))

function StatsAPI.confint(m::CointRegModel{T}; level::Real=0.95) where {T}
    se = stderror(m)
    df_r = dof_residual(m)
    z = T(quantile(TDist(df_r), 1 - (1 - level) / 2))
    return hcat(m.coef .- z .* se, m.coef .+ z .* se)
end

# =============================================================================
# Base.show — CointRegModel
# =============================================================================

function Base.show(io::IO, m::CointRegModel{T}) where {T}
    method_str = m.method == :fmols ? "FMOLS (Phillips–Hansen)" :
                 m.method == :ccr   ? "CCR (Park)" :
                                      "DOLS (Stock–Watson)"
    trend_str = m.trend == :none ? "none" : m.trend == :const ? "constant" : "constant + trend"

    spec = Any[
        "Method"        method_str;
        "Observations"  m.nobs;
        "Deterministics" trend_str;
        "Kernel"        _label(m.kernel);
        "Bandwidth"     _fmt(m.bandwidth; digits=3);
        "LR variance"   _fmt(m.omega_uv; digits=4)
    ]
    if m.method == :dols
        spec = vcat(spec, Any["Leads/Lags" "$(m.leads) / $(m.lags)"])
    end

    _pretty_table(io, spec;
        title = "Cointegrating Regression",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    _coef_table(io, "Long-run coefficients", m.varnames, m.coef, stderror(m);
        dist = :t, dof_r = dof_residual(m))
    _sig_legend(io)
end
