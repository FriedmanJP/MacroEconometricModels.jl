# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Type definitions for the ARDL (autoregressive distributed lag) module
(`src/ardl/`).

This file defines the shared result types for the ARDL / bounds-testing path.
The module scaffold (this file + `estimation.jl` + `bounds.jl`) is created by
EV-08 (#416) and extended by later nonlinear-ARDL (NARDL, EV-09) and
pooled/mean-group panel-ARDL (PMG, EV-23) estimators. The design keeps the
coefficient-block bookkeeping (`ar_idx`, `x_idx`, `det_idx`) explicit so those
extensions can reuse `long_run` / `bounds_test` on a re-parameterised design.
"""

# =============================================================================
# ARDLLongRun — long-run multipliers with delta-method standard errors
# =============================================================================

"""
    ARDLLongRun{T<:AbstractFloat}

Long-run (level) coefficients of an [`ARDLModel`](@ref),

```math
\\hat\\theta_j = \\frac{\\sum_\\ell \\hat\\beta_{j\\ell}}{1 - \\sum_i \\hat\\varphi_i},
```

with delta-method standard errors from the OLS variance matrix. Returned by
[`long_run`](@ref); prints as a `_coef_table`-style block.

# Fields
- `theta::Vector{T}`: long-run multipliers (`k`-vector).
- `se::Vector{T}`: delta-method standard errors.
- `denom::T`: the common denominator `1 − Σφ̂` (the negative speed of adjustment).
- `varnames::Vector{String}`: regressor labels.
"""
struct ARDLLongRun{T<:AbstractFloat}
    theta::Vector{T}
    se::Vector{T}
    denom::T
    varnames::Vector{String}
end

# =============================================================================
# ARDLModel — single-equation ARDL(p, q₁…q_k) estimated by OLS on lagged levels
# =============================================================================

"""
    ARDLModel{T<:AbstractFloat}

Autoregressive distributed-lag model

```math
y_t = c + \\delta t + \\sum_{i=1}^{p} \\varphi_i\\, y_{t-i}
        + \\sum_{j=1}^{k} \\sum_{\\ell=0}^{q_j} \\beta_{j\\ell}\\, x_{j,t-\\ell} + u_t,
```

estimated by OLS on the lagged **levels** over the effective sample
`t = L+1 … N`, with `L = max(p, max_j q_j)`. The deterministic terms `c` and
`δ t` enter according to the Pesaran–Shin–Smith (2001) `case` (see
[`estimate_ardl`](@ref)). Long-run coefficients, the conditional error-correction
re-parameterisation, and the bounds test are recovered from the stored OLS
`coef` / `vcov` through the index-bookkeeping fields.

# Fields
- `y::Vector{T}`: dependent variable over the effective sample (`n`-vector).
- `X::Matrix{T}`: levels design matrix (`n × K`), column order
  `[deterministics; y lags; x₁ lags; …; x_k lags]`.
- `coef::Vector{T}`, `vcov::Matrix{T}`: OLS coefficients and `σ̂²(X'X)⁻¹`.
- `residuals::Vector{T}`, `fitted::Vector{T}`: OLS residuals / fitted values.
- `p::Int`: autoregressive order; `q::Vector{Int}`: distributed-lag orders per regressor (`k`-vector).
- `case::Int`: PSS (2001) deterministic case `∈ 1:5`.
- `trend::Symbol`: `:none`, `:const`, or `:trend` — the deterministics actually in `X`.
- `ssr::T`, `sigma2::T`: residual sum of squares and `σ̂² = ssr/(n−K)`.
- `loglik::T`, `aic::T`, `bic::T`: Gaussian log-likelihood and information criteria.
- `n::Int`, `K::Int`: effective sample size and number of coefficients.
- `ar_idx::Vector{Int}`: columns of `X` holding `y_{t-1..p}`.
- `x_idx::Vector{Vector{Int}}`: columns of `X` holding each regressor's `0..q_j` lags.
- `det_idx::Vector{Int}`: columns of `X` holding deterministics.
- `intercept_col::Int`, `trend_col::Int`: column of the intercept / trend (`0` if absent).
- `coefnames::Vector{String}`: coefficient labels aligned with `coef`.
- `xnames::Vector{String}`: regressor labels (`k`-vector); `yname::String`: dependent-variable label.
- `selected::Bool`: whether `(p, q)` were chosen by IC grid search; `ic::Symbol`: `:aic` / `:bic`.
- `longrun::Union{Nothing,ARDLLongRun{T}}`: cached long-run block.

# References
- Pesaran, M. H. & Shin, Y. (1999). *Econometrics and Economic Theory in the 20th Century*.
- Pesaran, M. H., Shin, Y. & Smith, R. J. (2001). *Journal of Applied Econometrics* 16, 289–326.
"""
mutable struct ARDLModel{T<:AbstractFloat}
    y::Vector{T}
    X::Matrix{T}
    coef::Vector{T}
    vcov::Matrix{T}
    residuals::Vector{T}
    fitted::Vector{T}
    p::Int
    q::Vector{Int}
    case::Int
    trend::Symbol
    ssr::T
    sigma2::T
    loglik::T
    aic::T
    bic::T
    n::Int
    K::Int
    ar_idx::Vector{Int}
    x_idx::Vector{Vector{Int}}
    det_idx::Vector{Int}
    intercept_col::Int
    trend_col::Int
    coefnames::Vector{String}
    xnames::Vector{String}
    yname::String
    selected::Bool
    ic::Symbol
    longrun::Union{Nothing,ARDLLongRun{T}}
end

# =============================================================================
# ARDLBoundsTest — Pesaran–Shin–Smith (2001) bounds test
# =============================================================================

"""
    ARDLBoundsTest{T<:AbstractFloat}

Result of the Pesaran–Shin–Smith (2001) bounds test for the existence of a
level relationship. The `F`-statistic is the joint (non-standard) Wald/F test
that **all** error-correction level terms are zero — the lagged dependent level
`y_{t-1}` and every lagged regressor level `x_{j,t-1}` (plus the restricted
intercept/trend in cases II/IV). The `t`-statistic is the (Dickey–Fuller-type)
`t`-ratio on the `y_{t-1}` level term.

Both statistics are compared **only** to the asymptotic I(0)/I(1) critical-value
bounds (PSS Tables CI(i)–CI(v) for `F`, CII(i)/CII(iii)/CII(v) for `t`); the
distributions are non-standard, so **no p-value is defined or reported**.

# Fields
- `fstat::T`: bounds `F`-statistic.
- `tstat::T`: bounds `t`-statistic on the lagged level of `y`.
- `k::Int`: number of distributed-lag regressors (indexes the CV tables).
- `case::Int`: PSS deterministic case `∈ 1:5`.
- `cv_source::Symbol`: critical-value source (`:pss`).
- `levels::Vector{T}`: significance levels of the tabulated bounds (`[0.10,0.05,0.025,0.01]`).
- `f_lower::Vector{T}`, `f_upper::Vector{T}`: I(0)/I(1) `F`-bounds at each level.
- `t_lower::Vector{T}`, `t_upper::Vector{T}`: I(0)/I(1) `t`-bounds (`NaN` when undefined for the case).
- `level::T`: significance level used for the reported decision.
- `f_decision::Symbol`: `:cointegrated`, `:not_cointegrated`, or `:inconclusive` (from `F`).
- `t_decision::Symbol`: same, from the `t`-statistic (`:undefined` for cases II/IV).
- `n::Int`: effective sample size.

# References
- Pesaran, M. H., Shin, Y. & Smith, R. J. (2001). *Journal of Applied Econometrics* 16, 289–326.
- Narayan, P. K. (2005). *Applied Economics* 37, 1979–1990.
"""
struct ARDLBoundsTest{T<:AbstractFloat}
    fstat::T
    tstat::T
    k::Int
    case::Int
    cv_source::Symbol
    levels::Vector{T}
    f_lower::Vector{T}
    f_upper::Vector{T}
    t_lower::Vector{T}
    t_upper::Vector{T}
    level::T
    f_decision::Symbol
    t_decision::Symbol
    n::Int
end
