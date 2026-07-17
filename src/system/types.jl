# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Type definitions for multi-equation **systems** estimation (EV-35, #443):
seemingly-unrelated regressions (SUR, Zellner 1962) and three-stage least
squares (3SLS, Zellner & Theil 1962).
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# SURModel — seemingly-unrelated regressions (Zellner 1962)
# =============================================================================

"""
    SURModel{T<:AbstractFloat}

Seemingly-unrelated regressions (Zellner 1962) fitted by feasible GLS, optionally
iterated to the Gaussian MLE and/or subject to linear cross-equation restrictions
`R·vec(B) = r`. See [`estimate_sur`](@ref).

An `M`-equation system stacks `y = Xβ + u` with block-diagonal `X = blkdiag(X₁,…,X_M)`
and `Cov(u) = Σ ⊗ I_T`; the FGLS estimator is
`β̂ = (X'(Σ̂⁻¹⊗I)X)⁻¹ X'(Σ̂⁻¹⊗I)y` with classical covariance `(X'(Σ̂⁻¹⊗I)X)⁻¹`.
Efficiency gains over equation-by-equation OLS come from the cross-equation error
correlation; when every equation carries identical regressors the two coincide
exactly (Kruskal 1968).

# Fields
- `eqnames::Vector{String}` — equation labels (length `M`).
- `varnames::Vector{Vector{String}}` — per-equation coefficient names.
- `betas::Vector{Vector{T}}` — per-equation coefficient estimates.
- `ses::Vector{Vector{T}}` — per-equation coefficient standard errors.
- `vcov_mat::Matrix{T}` — full system covariance of the stacked coefficient vector.
- `Sigma::Matrix{T}` — `M×M` residual cross-covariance `Σ̂` (classical Zellner divisor `T`).
- `residuals::Vector{Vector{T}}` — per-equation FGLS residuals.
- `fitted::Vector{Vector{T}}` — per-equation fitted values.
- `nobs::Int` — observations per equation `T`.
- `det_sigma::T` — determinant of `Σ̂`.
- `mcelroy_r2::T` — McElroy (1977) system R².
- `loglik::T` — Gaussian system log-likelihood at `Σ̂`.
- `iterations::Int` — FGLS iterations (`1` for one-step; `>1` for iterated).
- `iterated::Bool`, `restricted::Bool` — variant flags.

# References
- Zellner, A. (1962). *Journal of the American Statistical Association* 57(298), 348-368.
- Kruskal, W. (1968). *Annals of Mathematical Statistics* 39(1), 70-75.
- McElroy, M. B. (1977). *Journal of Econometrics* 6(3), 381-387.
"""
struct SURModel{T<:AbstractFloat}
    eqnames::Vector{String}
    varnames::Vector{Vector{String}}
    betas::Vector{Vector{T}}
    ses::Vector{Vector{T}}
    vcov_mat::Matrix{T}
    Sigma::Matrix{T}
    residuals::Vector{Vector{T}}
    fitted::Vector{Vector{T}}
    nobs::Int
    det_sigma::T
    mcelroy_r2::T
    loglik::T
    iterations::Int
    iterated::Bool
    restricted::Bool
end

# =============================================================================
# ThreeSLSModel — three-stage least squares (Zellner & Theil 1962)
# =============================================================================

"""
    ThreeSLSModel{T<:AbstractFloat}

Three-stage least squares (Zellner & Theil 1962) for a simultaneous system:
each equation's regressors are first projected onto the instrument space
(`X̂ᵢ = P_Z Xᵢ`), the cross-equation residual covariance `Σ̂` is formed from
equation-by-equation 2SLS residuals, and the system GLS estimator
`β̂ = (X̂'(Σ̂⁻¹⊗I)X̂)⁻¹ X̂'(Σ̂⁻¹⊗I)y` combines instrumentation with the SUR
efficiency gain. See [`estimate_3sls`](@ref).

When the instrument set spans every regressor (`P_Z Xᵢ = Xᵢ`) the projection is a
no-op and 3SLS collapses to SUR; when every equation is exactly identified 3SLS
collapses to equation-by-equation 2SLS.

# Fields
- `eqnames::Vector{String}` — equation labels (length `M`).
- `varnames::Vector{Vector{String}}` — per-equation coefficient names.
- `betas::Vector{Vector{T}}` — per-equation coefficient estimates.
- `ses::Vector{Vector{T}}` — per-equation coefficient standard errors.
- `vcov_mat::Matrix{T}` — full system covariance of the stacked coefficient vector.
- `Sigma::Matrix{T}` — `M×M` residual cross-covariance from 2SLS residuals (divisor `T`).
- `residuals::Vector{Vector{T}}` — per-equation 3SLS residuals (from the ORIGINAL `Xᵢ`).
- `fitted::Vector{Vector{T}}` — per-equation fitted values.
- `nobs::Int` — observations per equation `T`.
- `det_sigma::T` — determinant of `Σ̂`.
- `mcelroy_r2::T` — McElroy (1977) system R².
- `n_instruments::Vector{Int}` — instrument-column count per equation.

# References
- Zellner, A. & Theil, H. (1962). *Econometrica* 30(1), 54-78.
- McElroy, M. B. (1977). *Journal of Econometrics* 6(3), 381-387.
"""
struct ThreeSLSModel{T<:AbstractFloat}
    eqnames::Vector{String}
    varnames::Vector{Vector{String}}
    betas::Vector{Vector{T}}
    ses::Vector{Vector{T}}
    vcov_mat::Matrix{T}
    Sigma::Matrix{T}
    residuals::Vector{Vector{T}}
    fitted::Vector{Vector{T}}
    nobs::Int
    det_sigma::T
    mcelroy_r2::T
    n_instruments::Vector{Int}
end

# =============================================================================
# Base.show — SURModel / ThreeSLSModel (one _coef_table per equation + footer)
# =============================================================================

function _system_footer(io::IO, m; title::String, extra=nothing)
    spec = Any[
        "Equations"    length(m.eqnames);
        "Obs./eq."     m.nobs;
        "det(Sigma)"   _fmt(m.det_sigma; digits=4);
        "McElroy R-sq" _fmt(m.mcelroy_r2)
    ]
    if extra !== nothing
        spec = vcat(spec, extra)
    end
    _pretty_table(io, spec;
        title = title,
        column_labels = ["System statistic", ""],
        alignment = [:l, :r])
end

function Base.show(io::IO, m::SURModel{T}) where {T}
    kind = m.iterated ? "Iterated SUR" : "SUR"
    m.restricted && (kind *= " (restricted)")
    extra = Any["Log-lik." _fmt(m.loglik; digits=2); "Iterations" m.iterations]
    _system_footer(io, m; title = "$kind — $(length(m.eqnames)) equations, Zellner (1962) FGLS",
        extra = extra)
    for j in eachindex(m.eqnames)
        _coef_table(io, "Equation: $(m.eqnames[j])", m.varnames[j], m.betas[j], m.ses[j];
            dist = :t, dof_r = max(m.nobs - length(m.betas[j]), 1))
    end
    _sig_legend(io)
end

function Base.show(io::IO, m::ThreeSLSModel{T}) where {T}
    extra = Any["Instruments/eq" join(m.n_instruments, ", ")]
    _system_footer(io, m; title = "3SLS — $(length(m.eqnames)) equations, Zellner-Theil (1962)",
        extra = extra)
    for j in eachindex(m.eqnames)
        _coef_table(io, "Equation: $(m.eqnames[j])", m.varnames[j], m.betas[j], m.ses[j];
            dist = :t, dof_r = max(m.nobs - length(m.betas[j]), 1))
    end
    _sig_legend(io)
end
