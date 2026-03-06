# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Type definitions for cross-sectional regression models: OLS/WLS, IV/2SLS,
Logit, Probit, and marginal effects.
"""

using LinearAlgebra, Statistics, Distributions, StatsAPI

# =============================================================================
# RegModel — OLS / WLS / IV
# =============================================================================

"""
    RegModel{T} <: StatsAPI.RegressionModel

Linear regression model estimated via OLS, WLS, or IV/2SLS.

# Fields
- `y::Vector{T}` — dependent variable
- `X::Matrix{T}` — regressor matrix (includes intercept if present)
- `beta::Vector{T}` — estimated coefficients
- `vcov_mat::Matrix{T}` — variance-covariance matrix of coefficients
- `residuals::Vector{T}` — OLS/WLS residuals
- `fitted::Vector{T}` — fitted values X * beta
- `ssr::T` — sum of squared residuals
- `tss::T` — total sum of squares (demeaned)
- `r2::T` — R-squared
- `adj_r2::T` — adjusted R-squared
- `f_stat::T` — F-statistic for joint significance
- `f_pval::T` — p-value of the F-test
- `loglik::T` — Gaussian log-likelihood
- `aic::T` — Akaike information criterion
- `bic::T` — Bayesian information criterion
- `varnames::Vector{String}` — coefficient names
- `method::Symbol` — estimation method (:ols, :wls, :iv)
- `cov_type::Symbol` — covariance estimator (:ols, :hc0, :hc1, :hc2, :hc3, :cluster)
- `weights::Union{Nothing,Vector{T}}` — WLS weights (nothing for OLS)
- `Z::Union{Nothing,Matrix{T}}` — instrument matrix (IV only)
- `endogenous::Union{Nothing,Vector{Int}}` — indices of endogenous regressors (IV only)
- `first_stage_f::Union{Nothing,T}` — first-stage F-statistic (IV only)
- `sargan_stat::Union{Nothing,T}` — Sargan overidentification statistic (IV only)
- `sargan_pval::Union{Nothing,T}` — Sargan test p-value (IV only)

# References
- White, H. (1980). *Econometrica* 48(4), 817-838.
- MacKinnon, J. G. & White, H. (1985). *JBES* 3(3), 305-314.
"""
struct RegModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    y::Vector{T}
    X::Matrix{T}
    beta::Vector{T}
    vcov_mat::Matrix{T}
    residuals::Vector{T}
    fitted::Vector{T}
    ssr::T
    tss::T
    r2::T
    adj_r2::T
    f_stat::T
    f_pval::T
    loglik::T
    aic::T
    bic::T
    varnames::Vector{String}
    method::Symbol
    cov_type::Symbol
    weights::Union{Nothing,Vector{T}}
    Z::Union{Nothing,Matrix{T}}
    endogenous::Union{Nothing,Vector{Int}}
    first_stage_f::Union{Nothing,T}
    sargan_stat::Union{Nothing,T}
    sargan_pval::Union{Nothing,T}
end

# =============================================================================
# LogitModel
# =============================================================================

"""
    LogitModel{T} <: StatsAPI.RegressionModel

Binary logistic regression model estimated via maximum likelihood (IRLS).

# Fields
- `y::Vector{T}` — binary dependent variable (0/1)
- `X::Matrix{T}` — regressor matrix
- `beta::Vector{T}` — estimated coefficients
- `vcov_mat::Matrix{T}` — variance-covariance matrix
- `residuals::Vector{T}` — deviance residuals
- `fitted::Vector{T}` — predicted probabilities P(y=1|X)
- `loglik::T` — maximized log-likelihood
- `loglik_null::T` — null model log-likelihood
- `pseudo_r2::T` — McFadden's pseudo R-squared
- `aic::T` — Akaike information criterion
- `bic::T` — Bayesian information criterion
- `varnames::Vector{String}` — coefficient names
- `converged::Bool` — whether IRLS converged
- `iterations::Int` — number of IRLS iterations
- `cov_type::Symbol` — covariance estimator (:ols, :hc1, etc.)

# References
- McCullagh, P. & Nelder, J. A. (1989). *Generalized Linear Models*. Chapman & Hall.
- Agresti, A. (2002). *Categorical Data Analysis*. 2nd ed. Wiley.
"""
struct LogitModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    y::Vector{T}
    X::Matrix{T}
    beta::Vector{T}
    vcov_mat::Matrix{T}
    residuals::Vector{T}
    fitted::Vector{T}
    loglik::T
    loglik_null::T
    pseudo_r2::T
    aic::T
    bic::T
    varnames::Vector{String}
    converged::Bool
    iterations::Int
    cov_type::Symbol
end

# =============================================================================
# ProbitModel
# =============================================================================

"""
    ProbitModel{T} <: StatsAPI.RegressionModel

Binary probit regression model estimated via maximum likelihood (IRLS).

# Fields
Same as `LogitModel{T}`, using the standard normal CDF as the link function.

# References
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. 2nd ed. MIT Press.
"""
struct ProbitModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    y::Vector{T}
    X::Matrix{T}
    beta::Vector{T}
    vcov_mat::Matrix{T}
    residuals::Vector{T}
    fitted::Vector{T}
    loglik::T
    loglik_null::T
    pseudo_r2::T
    aic::T
    bic::T
    varnames::Vector{String}
    converged::Bool
    iterations::Int
    cov_type::Symbol
end

# =============================================================================
# MarginalEffects
# =============================================================================

"""
    MarginalEffects{T}

Marginal effects computed from a Logit or Probit model.

# Fields
- `effects::Vector{T}` — marginal effects (AME, MEM, or MER)
- `se::Vector{T}` — delta-method standard errors
- `z_stat::Vector{T}` — z-statistics
- `p_values::Vector{T}` — two-sided p-values (normal distribution)
- `ci_lower::Vector{T}` — lower CI bounds
- `ci_upper::Vector{T}` — upper CI bounds
- `varnames::Vector{String}` — variable names
- `type::Symbol` — :ame (average), :mem (at-mean), or :mer (at-representative)
- `conf_level::T` — confidence level

# References
- Cameron, A. C. & Trivedi, P. K. (2005). *Microeconometrics*. Cambridge University Press.
"""
struct MarginalEffects{T<:AbstractFloat}
    effects::Vector{T}
    se::Vector{T}
    z_stat::Vector{T}
    p_values::Vector{T}
    ci_lower::Vector{T}
    ci_upper::Vector{T}
    varnames::Vector{String}
    type::Symbol
    conf_level::T
end

# =============================================================================
# StatsAPI Interface — RegModel
# =============================================================================

StatsAPI.coef(m::RegModel) = m.beta
StatsAPI.vcov(m::RegModel) = m.vcov_mat
StatsAPI.residuals(m::RegModel) = m.residuals
StatsAPI.predict(m::RegModel) = m.fitted
StatsAPI.nobs(m::RegModel) = length(m.y)
StatsAPI.dof(m::RegModel) = length(m.beta)
StatsAPI.dof_residual(m::RegModel) = length(m.y) - length(m.beta)
StatsAPI.loglikelihood(m::RegModel) = m.loglik
StatsAPI.aic(m::RegModel) = m.aic
StatsAPI.bic(m::RegModel) = m.bic
StatsAPI.islinear(::RegModel) = true
StatsAPI.r2(m::RegModel) = m.r2
StatsAPI.stderror(m::RegModel) = sqrt.(diag(m.vcov_mat))

"""Confidence intervals for RegModel coefficients (t-distribution)."""
function StatsAPI.confint(m::RegModel{T}; level::Real=0.95) where {T}
    se = stderror(m)
    df_r = dof_residual(m)
    crit = T(quantile(TDist(df_r), 1 - (1 - level) / 2))
    hcat(m.beta .- crit .* se, m.beta .+ crit .* se)
end

# =============================================================================
# StatsAPI Interface — LogitModel
# =============================================================================

StatsAPI.coef(m::LogitModel) = m.beta
StatsAPI.vcov(m::LogitModel) = m.vcov_mat
StatsAPI.residuals(m::LogitModel) = m.residuals
StatsAPI.predict(m::LogitModel) = m.fitted
StatsAPI.nobs(m::LogitModel) = length(m.y)
StatsAPI.dof(m::LogitModel) = length(m.beta)
StatsAPI.dof_residual(m::LogitModel) = length(m.y) - length(m.beta)
StatsAPI.loglikelihood(m::LogitModel) = m.loglik
StatsAPI.aic(m::LogitModel) = m.aic
StatsAPI.bic(m::LogitModel) = m.bic
StatsAPI.islinear(::LogitModel) = false
StatsAPI.stderror(m::LogitModel) = sqrt.(diag(m.vcov_mat))

"""Confidence intervals for LogitModel coefficients (normal distribution)."""
function StatsAPI.confint(m::LogitModel{T}; level::Real=0.95) where {T}
    se = stderror(m)
    crit = T(quantile(Normal(), 1 - (1 - level) / 2))
    hcat(m.beta .- crit .* se, m.beta .+ crit .* se)
end

# =============================================================================
# StatsAPI Interface — ProbitModel
# =============================================================================

StatsAPI.coef(m::ProbitModel) = m.beta
StatsAPI.vcov(m::ProbitModel) = m.vcov_mat
StatsAPI.residuals(m::ProbitModel) = m.residuals
StatsAPI.predict(m::ProbitModel) = m.fitted
StatsAPI.nobs(m::ProbitModel) = length(m.y)
StatsAPI.dof(m::ProbitModel) = length(m.beta)
StatsAPI.dof_residual(m::ProbitModel) = length(m.y) - length(m.beta)
StatsAPI.loglikelihood(m::ProbitModel) = m.loglik
StatsAPI.aic(m::ProbitModel) = m.aic
StatsAPI.bic(m::ProbitModel) = m.bic
StatsAPI.islinear(::ProbitModel) = false
StatsAPI.stderror(m::ProbitModel) = sqrt.(diag(m.vcov_mat))

"""Confidence intervals for ProbitModel coefficients (normal distribution)."""
function StatsAPI.confint(m::ProbitModel{T}; level::Real=0.95) where {T}
    se = stderror(m)
    crit = T(quantile(Normal(), 1 - (1 - level) / 2))
    hcat(m.beta .- crit .* se, m.beta .+ crit .* se)
end

# =============================================================================
# Base.show — RegModel
# =============================================================================

function Base.show(io::IO, m::RegModel{T}) where {T}
    n = nobs(m)
    k = dof(m)
    method_str = m.method == :ols ? "OLS" : m.method == :wls ? "WLS" : "IV/2SLS"

    spec = Any[
        "Method"       method_str;
        "Observations" n;
        "Covariates"   k;
        "R-squared"    _fmt(m.r2);
        "Adj. R-sq."   _fmt(m.adj_r2);
        "F-statistic"  _fmt(m.f_stat; digits=2);
        "F p-value"    _format_pvalue(m.f_pval);
        "AIC"          _fmt(m.aic; digits=2);
        "BIC"          _fmt(m.bic; digits=2);
        "Cov. type"    string(m.cov_type)
    ]

    if m.method == :iv && m.first_stage_f !== nothing
        spec = vcat(spec, Any[
            "1st-stage F" _fmt(m.first_stage_f; digits=2)
        ])
    end

    _pretty_table(io, spec;
        title = "$method_str Regression",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    _coef_table(io, "Coefficients", m.varnames, m.beta, stderror(m);
        dist = :t, dof_r = dof_residual(m))
    _sig_legend(io)
end

# =============================================================================
# Base.show — LogitModel
# =============================================================================

function Base.show(io::IO, m::LogitModel{T}) where {T}
    n = nobs(m)
    k = dof(m)

    spec = Any[
        "Model"         "Logit";
        "Observations"  n;
        "Covariates"    k;
        "Log-lik."      _fmt(m.loglik; digits=2);
        "Log-lik. null" _fmt(m.loglik_null; digits=2);
        "Pseudo R-sq."  _fmt(m.pseudo_r2);
        "AIC"           _fmt(m.aic; digits=2);
        "BIC"           _fmt(m.bic; digits=2);
        "Converged"     m.converged ? "Yes" : "No";
        "Iterations"    m.iterations
    ]
    _pretty_table(io, spec;
        title = "Logit Regression",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    _coef_table(io, "Coefficients", m.varnames, m.beta, stderror(m);
        dist = :z)
    _sig_legend(io)
end

# =============================================================================
# Base.show — ProbitModel
# =============================================================================

function Base.show(io::IO, m::ProbitModel{T}) where {T}
    n = nobs(m)
    k = dof(m)

    spec = Any[
        "Model"         "Probit";
        "Observations"  n;
        "Covariates"    k;
        "Log-lik."      _fmt(m.loglik; digits=2);
        "Log-lik. null" _fmt(m.loglik_null; digits=2);
        "Pseudo R-sq."  _fmt(m.pseudo_r2);
        "AIC"           _fmt(m.aic; digits=2);
        "BIC"           _fmt(m.bic; digits=2);
        "Converged"     m.converged ? "Yes" : "No";
        "Iterations"    m.iterations
    ]
    _pretty_table(io, spec;
        title = "Probit Regression",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    _coef_table(io, "Coefficients", m.varnames, m.beta, stderror(m);
        dist = :z)
    _sig_legend(io)
end

# =============================================================================
# Base.show — MarginalEffects
# =============================================================================

function Base.show(io::IO, me::MarginalEffects{T}) where {T}
    type_str = me.type == :ame ? "Average Marginal Effects" :
               me.type == :mem ? "Marginal Effects at Mean" :
               "Marginal Effects at Representative"

    _coef_table(io, type_str, me.varnames, me.effects, me.se;
        dist = :z, level = me.conf_level)
    _sig_legend(io)
end
