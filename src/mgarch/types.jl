# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Type definitions, accessors, StatsAPI interface, and display for multivariate GARCH
models — CCC (Bollerslev 1990), DCC/cDCC (Engle 2002 / Aielli 2013), and scalar/diagonal
BEKK(1,1) (Engle & Kroner 1995). EV-16 (#424).
"""

# =============================================================================
# MGARCHModel container
# =============================================================================

"""
    MGARCHModel{T} <: AbstractMGARCHModel

Fitted multivariate GARCH model producing a path of conditional covariance matrices
`Hₜ` (n×n) for a T×n return series.

# Fields
- `Y::Matrix{T}`: Data (T×n).
- `mu::Vector{T}`: Estimated per-series mean (length n).
- `margins::Vector{GARCHModel{T}}`: Univariate GARCH margin fits (empty for BEKK, which
  models the covariance directly).
- `H::Array{T,3}`: Conditional covariance matrices, `H[:,:,t]` is n×n (length-T path).
- `R::Union{Matrix{T},Array{T,3}}`: Conditional correlations — a constant `Matrix{T}`
  for CCC/BEKK, an `Array{T,3}` (`R[:,:,t]`) for DCC.
- `Rbar::Matrix{T}`: Unconditional / targeting correlation matrix.
- `params::Vector{T}`: Second-stage parameters (`[a,b]` for DCC / scalar BEKK,
  `[a₁…aₙ,b₁…bₙ]` for diagonal BEKK; empty for CCC).
- `param_names::Vector{String}`: Labels aligned with `params`.
- `param_vcov::Matrix{T}`: Cached QML sandwich covariance of `params` (second stage).
- `loglik::T`: Maximized joint Gaussian (quasi) log-likelihood.
- `aic::T`, `bic::T`: Information criteria (total parameter count = Σ margin dof + stage-2).
- `kind::Symbol`: `:ccc`, `:dcc`, or `:bekk`.
- `correction::Symbol`: DCC targeting correction — `:none` (standard) or `:aielli` (cDCC).
- `bekk_kind::Symbol`: `:scalar` or `:diagonal` (only meaningful when `kind == :bekk`).
- `converged::Bool`: Whether the second-stage optimizer converged.
- `n::Int`: Number of series.
"""
struct MGARCHModel{T<:AbstractFloat} <: AbstractMGARCHModel
    Y::Matrix{T}
    mu::Vector{T}
    margins::Vector{GARCHModel{T}}
    H::Array{T,3}
    R::Union{Matrix{T},Array{T,3}}
    Rbar::Matrix{T}
    params::Vector{T}
    param_names::Vector{String}
    param_vcov::Matrix{T}
    loglik::T
    aic::T
    bic::T
    kind::Symbol
    correction::Symbol
    bekk_kind::Symbol
    converged::Bool
    n::Int
end

# =============================================================================
# Accessors
# =============================================================================

"""
    covariances(m::MGARCHModel) -> Array{T,3}

Conditional covariance path `Hₜ` (n×n×T). `covariances(m)[:,:,t]` is the covariance at t.
"""
covariances(m::MGARCHModel) = m.H

"""
    correlations(m::MGARCHModel) -> Array{T,3}

Conditional correlation path `Rₜ` (n×n×T). Constant correlation models (CCC/BEKK) broadcast
their single correlation matrix across all t.
"""
function correlations(m::MGARCHModel{T}) where {T}
    Tn = size(m.H, 3)
    if m.R isa Array{T,3}
        return m.R
    else
        Rc = m.R::Matrix{T}
        out = Array{T,3}(undef, m.n, m.n, Tn)
        @inbounds for t in 1:Tn
            out[:, :, t] .= Rc
        end
        return out
    end
end

"""
    variances(m::MGARCHModel) -> Matrix{T}

Conditional variance series, T×n. Column `i` is the diagonal `Hₜ[i,i]` (the conditional
variance of series `i` over time).
"""
function variances(m::MGARCHModel{T}) where {T}
    Tn = size(m.H, 3)
    out = Matrix{T}(undef, Tn, m.n)
    @inbounds for t in 1:Tn, i in 1:m.n
        out[t, i] = m.H[i, i, t]
    end
    out
end

# =============================================================================
# StatsAPI interface
# =============================================================================

"""Number of time-series observations (rows of `Y`)."""
StatsAPI.nobs(m::MGARCHModel) = size(m.Y, 1)
"""Second-stage parameter vector (`[a,b]` for DCC/scalar BEKK; empty for CCC)."""
StatsAPI.coef(m::MGARCHModel) = m.params
"""Estimated conditional covariance path `Hₜ` (n×n×T)."""
StatsAPI.predict(m::MGARCHModel) = m.H
"""Maximized joint (quasi) log-likelihood."""
StatsAPI.loglikelihood(m::MGARCHModel) = m.loglik
"""Akaike Information Criterion."""
StatsAPI.aic(m::MGARCHModel) = m.aic
"""Bayesian Information Criterion."""
StatsAPI.bic(m::MGARCHModel) = m.bic
"""Total number of estimated parameters (Σ margin dof + second-stage params)."""
function StatsAPI.dof(m::MGARCHModel)
    mdof = isempty(m.margins) ? length(m.mu) : sum(StatsAPI.dof(g) for g in m.margins)
    mdof + length(m.params)
end
"""`false` — multivariate GARCH is a nonlinear model."""
StatsAPI.islinear(::MGARCHModel) = false

"""
    StatsAPI.stderror(m::MGARCHModel) -> Vector{T}

Second-stage QML sandwich standard errors for `params` (`[a,b]` etc.). Empty for CCC.
Margin standard errors are available via `stderror(m.margins[i])`.
"""
function StatsAPI.stderror(m::MGARCHModel{T}) where {T}
    isempty(m.params) && return T[]
    V = m.param_vcov
    (size(V, 1) == length(m.params) && all(isfinite, V)) || return fill(T(NaN), length(m.params))
    T[sqrt(max(V[i, i], zero(T))) for i in 1:length(m.params)]
end

# =============================================================================
# Display
# =============================================================================

Base.show(io::IO, m::MGARCHModel) = _show_mgarch(io, m)

function _show_mgarch(io::IO, m::MGARCHModel{T}) where {T}
    kindlabel = m.kind === :ccc ? "CCC (Bollerslev 1990)" :
                m.kind === :dcc ? (m.correction === :aielli ? "cDCC (Aielli 2013)" : "DCC (Engle 2002)") :
                "BEKK-$(m.bekk_kind) (Engle-Kroner 1995)"
    println(io, "Multivariate GARCH — $(kindlabel)")
    println(io, "  Series: $(m.n)   Observations: $(size(m.Y, 1))")

    # Margin summaries (CCC/DCC)
    if !isempty(m.margins)
        mnames = String[]
        mvals = T[]
        for (i, g) in enumerate(m.margins)
            push!(mnames, "series $i: α₁"); push!(mvals, g.alpha[1])
            push!(mnames, "series $i: β₁"); push!(mvals, g.beta[1])
        end
        mse = fill(T(NaN), length(mvals))
        _coef_table(io, "Margin GARCH(1,1) persistence", mnames, mvals, mse; dist=:z)
    end

    # Second-stage block
    if !isempty(m.params)
        se = try
            stderror(m)
        catch
            fill(T(NaN), length(m.params))
        end
        _coef_table(io, "Dynamics", m.param_names, m.params, se; dist=:z)
    end

    fit_data = Any[
        "Model"           string(m.kind);
        "Log-likelihood"  _fmt(m.loglik; digits=4);
        "AIC"             _fmt(m.aic; digits=4);
        "BIC"             _fmt(m.bic; digits=4);
        "Converged"       _yesno(m.converged)
    ]
    _pretty_table(io, fit_data; column_labels=["Fit", "Value"], alignment=[:l, :r])
    isempty(m.params) || _sig_legend(io)
end
