# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-39 (#447): Forecast evaluation & combination — result types.
#
# A model-agnostic toolkit consumed through a duck-typed interface (plain
# AbstractVector of actuals + AbstractVector/Matrix of forecasts). No forecast
# struct is touched; a thin adapter can pull `.forecast`/`.point` out of the
# existing `*Forecast` types.

# =============================================================================
# Point-metric evaluation table
# =============================================================================

"""
    ForecastEvaluation{T}

Point-forecast accuracy table produced by [`forecast_evaluate`](@ref). Holds one
row of metrics per candidate model plus the Theil MSE bias/variance/covariance
decomposition (whose three proportions sum to one).

# Fields
- `models::Vector{String}` — model/column labels
- `metrics::Vector{String}` — metric names, in table order
- `values::Matrix{T}` — `n_models × n_metrics` metric values
- `decomp::Matrix{T}` — `n_models × 3` Theil MSE decomposition proportions
  `(bias, variance, covariance)`
- `n::Int` — number of evaluation observations
"""
struct ForecastEvaluation{T<:AbstractFloat}
    models::Vector{String}
    metrics::Vector{String}
    values::Matrix{T}
    decomp::Matrix{T}
    n::Int
end

# =============================================================================
# Diebold–Mariano (1995) equal-predictive-accuracy test
# =============================================================================

"""
    DMTestResult{T} <: StatsAPI.HypothesisTest

Diebold–Mariano (1995) test of equal predictive accuracy with the
Harvey–Leybourne–Newbold (1997) small-sample correction. Under `hln=true` the
statistic references the `t_{T−1}` distribution (matching R `forecast::dm.test`),
otherwise `N(0,1)`.

# Fields
- `statistic::T` — (corrected) DM statistic
- `pvalue::T`
- `dbar::T` — mean loss differential `d̄`
- `lrvar::T` — long-run variance `V̂` of `d_t` (truncated HAC, ÷T convention)
- `h::Int` — forecast horizon (sets the truncation lag `h−1`)
- `loss::Symbol` — `:se`, `:ad`, or `:custom`
- `hln::Bool` — HLN correction applied
- `alternative::Symbol` — `:two_sided`, `:less`, `:greater`
- `T_obs::Int`
"""
struct DMTestResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest
    statistic::T
    pvalue::T
    dbar::T
    lrvar::T
    h::Int
    loss::Symbol
    hln::Bool
    alternative::Symbol
    T_obs::Int
end

# =============================================================================
# Clark–West (2007) test for nested models
# =============================================================================

"""
    ClarkWestResult{T} <: StatsAPI.HypothesisTest

Clark–West (2007) adjusted-MSPE test for nested models. The Diebold–Mariano test
is invalid when one model nests the other; CW corrects the MSPE differential for
the noise introduced by estimating the extra parameters of the larger model.

# Fields
- `statistic::T` — one-sided CW statistic `mean(f̂)/HAC-SE`
- `pvalue::T` — one-sided (`greater`) p-value
- `fbar::T` — mean adjusted differential `mean(f̂)`
- `lrvar::T` — long-run variance of `f̂_t`
- `h::Int`
- `alternative::Symbol`
- `T_obs::Int`
"""
struct ClarkWestResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest
    statistic::T
    pvalue::T
    fbar::T
    lrvar::T
    h::Int
    alternative::Symbol
    T_obs::Int
end

# =============================================================================
# Mincer–Zarnowitz (1969) efficiency regression
# =============================================================================

"""
    MincerZarnowitzResult{T} <: StatsAPI.HypothesisTest

Mincer–Zarnowitz (1969) forecast-efficiency test. Regresses `actual = a + b·fc + u`
and jointly tests `(a, b) = (0, 1)` with a HAC (Newey–West) covariance. A weakly
efficient forecast implies `a = 0`, `b = 1`.

# Fields
- `a::T`, `b::T` — intercept and slope
- `se::Vector{T}` — HAC standard errors `[se_a, se_b]`
- `wald::T` — HAC Wald statistic (χ² with 2 df)
- `pvalue_wald::T`
- `fstat::T` — `wald/2`, referenced to `F(2, T−2)`
- `pvalue_f::T`
- `lags::Int` — Newey–West truncation lag
- `kernel::Symbol`
- `T_obs::Int`
"""
struct MincerZarnowitzResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest
    a::T
    b::T
    se::Vector{T}
    wald::T
    pvalue_wald::T
    fstat::T
    pvalue_f::T
    lags::Int
    kernel::Symbol
    T_obs::Int
end

# =============================================================================
# Harvey–Leybourne–Newbold (1998) forecast encompassing
# =============================================================================

"""
    ForecastEncompassingResult{T} <: StatsAPI.HypothesisTest

Regression-based forecast-encompassing test (Harvey, Leybourne & Newbold 1998).
Regresses `actual = a + b₁·fc1 + b₂·fc2 + u` with a HAC covariance and tests
`b₂ = 0`. Non-rejection means model 1 *encompasses* model 2 — model 2 adds no
useful information beyond model 1.

# Fields
- `b1::T`, `b2::T` — combination weights on the two forecasts
- `se_b2::T` — HAC standard error of `b₂`
- `tstat::T` — `b₂ / se_b2`
- `pvalue::T` — two-sided p-value (`t_{T−3}`)
- `lags::Int`
- `kernel::Symbol`
- `T_obs::Int`
"""
struct ForecastEncompassingResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest
    b1::T
    b2::T
    se_b2::T
    tstat::T
    pvalue::T
    lags::Int
    kernel::Symbol
    T_obs::Int
end

# =============================================================================
# Forecast combination
# =============================================================================

"""
    ForecastCombination{T}

Combined forecast produced by [`combine_forecasts`](@ref).

# Fields
- `weights::Vector{T}` — combination weights (Granger–Ramanathan weights may be
  negative and need not lie in `[0,1]`; they always sum to one)
- `combined::Vector{T}` — the combined forecast series `F·w`
- `method::Symbol` — `:equal`, `:bates_granger`, or `:granger_ramanathan`
- `mse::Vector{T}` — individual-model MSEs (relative to `actual`)
- `models::Vector{String}` — model labels
"""
struct ForecastCombination{T<:AbstractFloat}
    weights::Vector{T}
    combined::Vector{T}
    method::Symbol
    mse::Vector{T}
    models::Vector{String}
end
