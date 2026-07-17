# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# Nonparametric regression & density — result types (EV-33, #441)
# =============================================================================

"""
    KernelDensity{T<:AbstractFloat} <: AbstractAnalysisResult

Kernel density estimate on an equally-spaced grid.

# Fields
- `x::Vector{T}`         : grid points (evaluation abscissae)
- `density::Vector{T}`   : estimated density `f̂(x)` on the grid
- `bandwidth::T`         : smoothing bandwidth `h`
- `kernel::Symbol`       : kernel (`:gaussian`/`:epanechnikov`/`:triangular`/`:uniform`)
- `bw_method::Symbol`    : bandwidth rule (`:silverman`/`:sj`/`:user`)
- `data::Vector{T}`      : original sample
- `nobs::Int`            : number of observations
"""
struct KernelDensity{T<:AbstractFloat} <: AbstractAnalysisResult
    x::Vector{T}
    density::Vector{T}
    bandwidth::T
    kernel::Symbol
    bw_method::Symbol
    data::Vector{T}
    nobs::Int
end

"""
    KernelRegression{T<:AbstractFloat} <: AbstractAnalysisResult

Nonparametric regression fit (Nadaraya–Watson / local-linear / local-polynomial).

The fit is evaluated at the sorted design points. `se` holds pointwise standard
errors from the effective-weight sandwich form `Var(m̂(x₀)) = σ̂²·‖ℓ(x₀)‖²`,
where `ℓ(x₀)` is the vector of effective weights (`m̂(x₀) = Σ ℓᵢ(x₀) yᵢ`).

# Fields
- `x::Vector{T}`        : sorted design points (evaluation grid = design points)
- `fitted::Vector{T}`   : `m̂(x)` at the design points
- `se::Vector{T}`       : pointwise standard errors of `m̂`
- `xdata::Vector{T}`    : original `x`, sorted
- `ydata::Vector{T}`    : original `y`, sorted by `x`
- `bandwidth::T`        : smoothing bandwidth `h`
- `method::Symbol`      : `:nw` (local constant), `:ll` (local linear), `:lp` (local polynomial)
- `degree::Int`         : local-polynomial degree (`:nw`⇒0, `:ll`⇒1, `:lp`⇒`degree`)
- `kernel::Symbol`      : kernel used for the weights
- `bw_method::Symbol`   : bandwidth rule (`:cv`/`:rot`/`:user`)
- `sigma2::T`           : residual-variance estimate `σ̂²`
- `nobs::Int`           : number of observations
"""
struct KernelRegression{T<:AbstractFloat} <: AbstractAnalysisResult
    x::Vector{T}
    fitted::Vector{T}
    se::Vector{T}
    xdata::Vector{T}
    ydata::Vector{T}
    bandwidth::T
    method::Symbol
    degree::Int
    kernel::Symbol
    bw_method::Symbol
    sigma2::T
    nobs::Int
end

"""
    LowessFit{T<:AbstractFloat} <: AbstractAnalysisResult

Cleveland (1979) LOWESS scatterplot smoother: tricube-weighted local-linear fit
with `iter` bisquare robustifying passes. Values are returned sorted by `x`.

# Fields
- `x::Vector{T}`       : sorted `x`
- `fitted::Vector{T}`  : smoothed `ŷ`, sorted by `x`
- `ydata::Vector{T}`   : original `y`, sorted by `x`
- `span::T`            : smoother span `f` (fraction of points in each window)
- `iter::Int`          : number of robustifying iterations
- `nobs::Int`          : number of observations
"""
struct LowessFit{T<:AbstractFloat} <: AbstractAnalysisResult
    x::Vector{T}
    fitted::Vector{T}
    ydata::Vector{T}
    span::T
    iter::Int
    nobs::Int
end
