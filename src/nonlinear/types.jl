# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Type definitions for nonlinear time series models (`src/nonlinear/`).

This file defines the shared result types for the nonlinear module. The module
scaffold (this file + `threshold.jl`) is created by EV-05 and extended by later
smooth-transition (STAR) and Markov-switching models. All concrete model types
subtype [`AbstractNonlinearTSModel`](@ref) (declared in `src/core/types.jl`).
"""

# =============================================================================
# Hansen (1996) linearity test result
# =============================================================================

"""
    HansenLinearityTest{T}

Result of Hansen's (1996) sup-LM / sup-Wald test of linearity against a
two-regime threshold alternative, with fixed-regressor-bootstrap p-values.

The null H₀: β₁ = β₂ (a linear model) is tested against the threshold
alternative. Because the threshold γ is unidentified under the null (the Davies
problem), the asymptotic distribution is nonstandard; p-values come from the
fixed-regressor bootstrap of Hansen (1996), not a χ² approximation.

# Fields
- `sup_lm::T`: sup over γ of the heteroskedasticity-robust LM statistic.
- `sup_wald::T`: sup over γ of the Wald (F-type) statistic `n·(S₀−S(γ))/S(γ)`.
- `pvalue_lm::T`: fixed-regressor-bootstrap p-value for `sup_lm`.
- `pvalue_wald::T`: fixed-regressor-bootstrap p-value for `sup_wald`.
- `gamma_sup::T`: threshold value attaining `sup_lm`.
- `reps::Int`: number of bootstrap replications.
- `trim::T`: trimming fraction used for the γ grid.
- `n_grid::Int`: number of threshold candidates searched.
"""
struct HansenLinearityTest{T<:AbstractFloat}
    sup_lm::T
    sup_wald::T
    pvalue_lm::T
    pvalue_wald::T
    gamma_sup::T
    reps::Int
    trim::T
    n_grid::Int
end

# =============================================================================
# Threshold / SETAR model
# =============================================================================

"""
    ThresholdModel{T} <: AbstractNonlinearTSModel

Two-regime threshold regression / self-exciting threshold autoregression (SETAR),
estimated by conditional least squares (Tong 1990; Hansen 2000).

The model is

```math
y_t = X_t'β_1·\\mathbf{1}\\{q_t ≤ γ\\} + X_t'β_2·\\mathbf{1}\\{q_t > γ\\} + u_t,
```

with the threshold `γ` estimated by grid search over the trimmed order statistics
of the threshold variable `q`, minimising the concentrated sum of squared
residuals `S(γ) = SSR₁(γ) + SSR₂(γ)`. For SETAR, `q_t = y_{t−d}` and
`X_t = [1, y_{t−1}, …, y_{t−p}]`.

# Fields
- `y::Vector{T}`: dependent variable (effective sample used in the regression).
- `X::Matrix{T}`: regressor matrix (`n × k`).
- `q::Vector{T}`: threshold variable (length `n`).
- `gamma::T`: estimated threshold γ̂.
- `gamma_ci::Tuple{T,T}`: Hansen (2000) LR-inversion confidence interval for γ.
- `gamma_ci_level::T`: confidence level of `gamma_ci` (e.g. 0.95).
- `beta1::Vector{T}`, `beta2::Vector{T}`: per-regime coefficients (`q ≤ γ̂` / `q > γ̂`).
- `se1::Vector{T}`, `se2::Vector{T}`: per-regime classical standard errors.
- `regime::Vector{Bool}`: regime-1 indicator `1{q_t ≤ γ̂}`.
- `ssr1::T`, `ssr2::T`, `ssr::T`: regime and total residual sums of squares.
- `sigma2::T`: pooled residual variance `S(γ̂)/n`.
- `residuals::Vector{T}`: pooled residuals.
- `n::Int`, `k::Int`: effective sample size and number of regressors per regime.
- `n1::Int`, `n2::Int`: per-regime observation counts.
- `p::Int`, `d::Int`: SETAR AR order and delay (`0` for a generic threshold model).
- `is_setar::Bool`: whether the model was fit via [`estimate_setar`](@ref).
- `aic::T`, `bic::T`: information criteria.
- `xnames::Vector{String}`: regressor labels.
- `qname::String`: label of the threshold variable.
- `trim::T`: trimming fraction.
- `linearity::Union{Nothing,HansenLinearityTest{T}}`: linearity-test result (if run).
"""
struct ThresholdModel{T<:AbstractFloat} <: AbstractNonlinearTSModel
    y::Vector{T}
    X::Matrix{T}
    q::Vector{T}
    gamma::T
    gamma_ci::Tuple{T,T}
    gamma_ci_level::T
    beta1::Vector{T}
    beta2::Vector{T}
    se1::Vector{T}
    se2::Vector{T}
    regime::Vector{Bool}
    ssr1::T
    ssr2::T
    ssr::T
    sigma2::T
    residuals::Vector{T}
    n::Int
    k::Int
    n1::Int
    n2::Int
    p::Int
    d::Int
    is_setar::Bool
    aic::T
    bic::T
    xnames::Vector{String}
    qname::String
    trim::T
    linearity::Union{Nothing,HansenLinearityTest{T}}
end

"""
    ThresholdForecast{T} <: AbstractForecastResult{T}

Bootstrap-simulation forecast from a SETAR [`ThresholdModel`](@ref).

# Fields
- `forecast::Vector{T}`: mean forecast path (length `horizon`).
- `ci_lower::Vector{T}`, `ci_upper::Vector{T}`: percentile bands.
- `se::Vector{T}`: standard deviation of the simulated paths at each horizon.
- `horizon::Int`: forecast horizon.
- `conf_level::T`: nominal band coverage (e.g. 0.90).
- `reps::Int`: number of simulated paths.
"""
struct ThresholdForecast{T<:AbstractFloat} <: AbstractForecastResult{T}
    forecast::Vector{T}
    ci_lower::Vector{T}
    ci_upper::Vector{T}
    se::Vector{T}
    horizon::Int
    conf_level::T
    reps::Int
end

# =============================================================================
# Hansen (2000) tabulated critical values for the threshold confidence interval
# =============================================================================

# Non-standard critical values for the likelihood-ratio statistic
# LR(γ) = n·(S(γ)−S(γ̂))/S(γ̂) under homoskedasticity (Hansen 2000, Table 1).
# They are the quantiles of the distribution with CDF (1 − exp(−x/2))²:
#   solving (1 − exp(−c/2))² = α gives c(.90)=5.94, c(.95)=7.35, c(.99)=10.59.
const HANSEN2000_CRIT = (var"0.90" = 5.94, var"0.95" = 7.35, var"0.99" = 10.59)

"""
    _hansen2000_crit(level::Real) -> Float64

Return the Hansen (2000, Table 1) critical value for the threshold-CI
likelihood-ratio statistic at confidence `level` (`0.90`, `0.95`, or `0.99`).
"""
function _hansen2000_crit(level::Real)
    if level ≈ 0.90
        return 5.94
    elseif level ≈ 0.95
        return 7.35
    elseif level ≈ 0.99
        return 10.59
    else
        throw(ArgumentError("Hansen (2000) critical values are tabulated only for " *
                            "level ∈ {0.90, 0.95, 0.99}; got $level."))
    end
end

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, m::ThresholdModel{T}) where {T}
    header = m.is_setar ? "SETAR(2; $(m.p), $(m.p)) Model  [delay d=$(m.d)]" :
                          "Two-Regime Threshold Regression"
    lvl = round(Int, 100 * m.gamma_ci_level)

    # Regime 1: q ≤ γ̂
    _coef_table(io, "$header — Regime 1: $(m.qname) ≤ $(_fmt(m.gamma)) (n=$(m.n1))",
                m.xnames, m.beta1, m.se1; dist=:t, dof_r=max(m.n1 - m.k, 1))
    # Regime 2: q > γ̂
    _coef_table(io, "Regime 2: $(m.qname) > $(_fmt(m.gamma)) (n=$(m.n2))",
                m.xnames, m.beta2, m.se2; dist=:t, dof_r=max(m.n2 - m.k, 1))

    fit_data = Any[
        "Threshold γ̂"          _fmt(m.gamma);
        "$(lvl)% CI for γ"      "[$(_fmt(m.gamma_ci[1])), $(_fmt(m.gamma_ci[2]))]";
        "Observations"         m.n;
        "Total SSR"            _fmt(m.ssr);
        "σ̂² (pooled)"          _fmt(m.sigma2);
        "AIC"                  _fmt(m.aic);
        "BIC"                  _fmt(m.bic)
    ]
    if m.linearity !== nothing
        lt = m.linearity
        fit_data = vcat(fit_data, Any[
            "Hansen sup-LM"        _fmt(lt.sup_lm);
            "  p-value (boot)"     _format_pvalue(lt.pvalue_lm);
            "Hansen sup-Wald"      _fmt(lt.sup_wald);
            "  p-value (boot)"     _format_pvalue(lt.pvalue_wald)
        ])
    end
    _pretty_table(io, fit_data; column_labels=["Threshold & Fit", "Value"],
                  alignment=[:l, :r])

    _show_note(io, "Hansen (2000) $(lvl)% CI inverts LR(γ)=n·(S(γ)−S(γ̂))/S(γ̂); " *
                   "c(.90)=5.94, c(.95)=7.35, c(.99)=10.59.")
    _sig_legend(io)
end

function Base.show(io::IO, f::ThresholdForecast{T}) where {T}
    ci_pct = round(Int, 100 * f.conf_level)
    n_show = min(10, f.horizon)
    data = Matrix{Any}(undef, n_show, 5)
    for i in 1:n_show
        data[i, 1] = i
        data[i, 2] = _fmt(f.forecast[i])
        data[i, 3] = _fmt(f.se[i])
        data[i, 4] = _fmt(f.ci_lower[i])
        data[i, 5] = _fmt(f.ci_upper[i])
    end
    _pretty_table(io, data;
        title = "SETAR Bootstrap Forecast ($(f.reps) paths)",
        column_labels = ["h", "Mean", "Std.Dev.", "$(ci_pct)% Lo", "$(ci_pct)% Hi"],
        alignment = [:r, :r, :r, :r, :r])
end

"""
    report(m::ThresholdModel)

Print a two-block regime coefficient summary with the estimated threshold, its
Hansen (2000) confidence interval, information criteria, and the Hansen (1996)
linearity test.
"""
report(m::ThresholdModel) = show(stdout, m)
report(io::IO, m::ThresholdModel) = show(io, m)
report(f::ThresholdForecast) = show(stdout, f)
report(io::IO, f::ThresholdForecast) = show(io, f)

# =============================================================================
# StatsAPI interface
# =============================================================================

StatsAPI.nobs(m::ThresholdModel) = m.n
StatsAPI.residuals(m::ThresholdModel) = m.residuals
StatsAPI.coef(m::ThresholdModel) = vcat(m.beta1, m.beta2)
StatsAPI.dof(m::ThresholdModel) = 2 * m.k + 1
StatsAPI.aic(m::ThresholdModel) = m.aic
StatsAPI.bic(m::ThresholdModel) = m.bic
