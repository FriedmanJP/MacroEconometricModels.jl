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
- `first_stage_f::Union{Nothing,T}` — minimum excluded-instrument partial F (IV only)
- `sargan_stat::Union{Nothing,T}` — Sargan overidentification statistic (IV only)
- `sargan_pval::Union{Nothing,T}` — Sargan test p-value (IV only)
- `cragg_donald_f::Union{Nothing,T}` — Cragg-Donald weak-instrument F (IV only)
- `kleibergen_paap_f::Union{Nothing,T}` — Kleibergen-Paap rk Wald F, robust (IV only)
- `stock_yogo_10pct::Union{Nothing,T}` — Stock-Yogo 10% maximal-size critical value (IV only)

# References
- White, H. (1980). *Econometrica* 48(4), 817-838.
- MacKinnon, J. G. & White, H. (1985). *JBES* 3(3), 305-314.
- Stock, J. H. & Yogo, M. (2005). *Identification and Inference for Econometric Models*, ch. 5.
- Kleibergen, F. & Paap, R. (2006). *Journal of Econometrics* 133(1), 97-126.
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
    cragg_donald_f::Union{Nothing,T}
    kleibergen_paap_f::Union{Nothing,T}
    stock_yogo_10pct::Union{Nothing,T}
end

# Back-compat outer constructor: legacy 24-arg positional calls (through `sargan_pval`)
# default the three weak-instrument diagnostic fields to `nothing`.
function RegModel{T}(y, X, beta, vcov_mat, residuals, fitted, ssr, tss, r2, adj_r2,
                     f_stat, f_pval, loglik, aic, bic, varnames, method, cov_type,
                     weights, Z, endogenous, first_stage_f, sargan_stat,
                     sargan_pval) where {T<:AbstractFloat}
    RegModel{T}(y, X, beta, vcov_mat, residuals, fitted, ssr, tss, r2, adj_r2,
                f_stat, f_pval, loglik, aic, bic, varnames, method, cov_type,
                weights, Z, endogenous, first_stage_f, sargan_stat, sargan_pval,
                nothing, nothing, nothing)
end

# =============================================================================
# PenalizedRegModel — ridge / LASSO / elastic net (EV-03, #411)
# =============================================================================

"""
    PenalizedRegModel{T} <: StatsAPI.RegressionModel

Penalized linear regression: ridge, LASSO, or elastic net (with optional adaptive-LASSO
weights and post-selection OLS refit). Coefficients are stored on the **natural** scale;
the intercept is fitted separately and never penalized. See [`estimate_elastic_net`](@ref),
[`estimate_lasso`](@ref), [`estimate_ridge`](@ref).

No standard errors, t-statistics, or p-values are reported: shrinkage estimators have no
valid closed-form inference, and post-LASSO SEs are trustworthy only under honest-inference
sparsity conditions this package does not verify.

# Fields
- `y`, `X` — response and regressors (no intercept column in `X`).
- `beta::Vector{T}` — natural-scale slope coefficients at the selected lambda.
- `beta0::T` — unpenalized intercept `ȳ − x̄'β`.
- `beta_std::Vector{T}` — standardized-scale coefficients at the selected lambda.
- `alpha::T` — elastic-net mixing (`1`=LASSO, `0`=ridge).
- `lambda::T` — selected penalty.
- `lambda_path::Vector{T}` — full (descending) lambda path.
- `coef_path::Matrix{T}` — `p × L` natural-scale coefficient path.
- `beta0_path::Vector{T}` — intercept along the path.
- `active_set::Vector{Int}` — indices of nonzero coefficients at the selected lambda.
- `df_path::Vector{T}`, `df_star::T` — degrees of freedom along the path / at selection.
- `fitted`, `residuals`, `ssr`, `tss`, `r2` — fit on the original scale.
- `loglik`, `aic`, `bic`, `ebic` — Gaussian information criteria at selection.
- `cv_mse`, `cv_se` — CV MSE curve and its standard error (`nothing` for IC/fixed selection).
- `lambda_min::T`, `lambda_1se::T` — CV-minimizing and 1-SE-rule lambdas.
- `select::Symbol` — `:cv`, `:aic`, `:bic`, `:ebic`, or `:fixed`.
- `cv::Symbol`, `nfolds::Int` — CV scheme and fold count.
- `adaptive::Bool`, `post::Bool`, `standardize::Bool` — variant flags.
- `xbar`, `xsd`, `ybar` — standardization stats (for prediction on new data).
- `varnames::Vector{String}` — regressor names.

# References
- Hoerl, A. E. & Kennard, R. W. (1970). *Technometrics* 12(1), 55-67.
- Tibshirani, R. (1996). *JRSS-B* 58(1), 267-288.
- Zou, H. & Hastie, T. (2005). *JRSS-B* 67(2), 301-320.
- Zou, H. (2006). *JASA* 101(476), 1418-1429.
- Friedman, J., Hastie, T. & Tibshirani, R. (2010). *J. Statistical Software* 33(1), 1-22.
- Belloni, A. & Chernozhukov, V. (2013). *Bernoulli* 19(2), 521-547.
"""
struct PenalizedRegModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    y::Vector{T}
    X::Matrix{T}
    beta::Vector{T}
    beta0::T
    beta_std::Vector{T}
    alpha::T
    lambda::T
    lambda_path::Vector{T}
    coef_path::Matrix{T}
    beta0_path::Vector{T}
    active_set::Vector{Int}
    df_path::Vector{T}
    df_star::T
    fitted::Vector{T}
    residuals::Vector{T}
    ssr::T
    tss::T
    r2::T
    loglik::T
    aic::T
    bic::T
    ebic::T
    cv_mse::Union{Nothing,Vector{T}}
    cv_se::Union{Nothing,Vector{T}}
    lambda_min::T
    lambda_1se::T
    select::Symbol
    cv::Symbol
    nfolds::Int
    adaptive::Bool
    post::Bool
    standardize::Bool
    xbar::Vector{T}
    xsd::Vector{T}
    ybar::T
    varnames::Vector{String}
end

function Base.show(io::IO, m::PenalizedRegModel{T}) where {T}
    kind = m.alpha == one(T) ? "LASSO" :
           m.alpha == zero(T) ? "Ridge" : "Elastic Net"
    m.adaptive && (kind = "Adaptive " * kind)
    m.post && (kind *= " + post-OLS")
    sel_str = m.select == :cv ? "CV ($(m.cv), $(m.nfolds)-fold)" :
              m.select == :fixed ? "fixed" : uppercase(string(m.select))

    spec = Any[
        "Method"        kind;
        "alpha"         _fmt(m.alpha; digits=3);
        "Observations"  length(m.y);
        "Regressors"    length(m.beta);
        "lambda"        _fmt(m.lambda; digits=5);
        "Selection"     sel_str;
        "Nonzero coef." length(m.active_set);
        "Eff. df"       _fmt(m.df_star; digits=2);
        "R-squared"     _fmt(m.r2);
        "AIC"           _fmt(m.aic; digits=2);
        "BIC"           _fmt(m.bic; digits=2)
    ]
    _pretty_table(io, spec;
        title = "$kind Regression",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    # Coefficient table (no SEs / p-values for shrinkage estimators): intercept + nonzero coefs.
    names = vcat("(Intercept)", m.varnames[m.active_set])
    vals = vcat(m.beta0, m.beta[m.active_set])
    data = Matrix{Any}(undef, length(names), 2)
    for i in eachindex(names)
        data[i, 1] = names[i]
        data[i, 2] = _fmt(vals[i])
    end
    _pretty_table(io, data;
        title = "Coefficients (nonzero)",
        column_labels = ["", "Coef."],
        alignment = [:l, :r],
    )
    println(io, "Note: penalized estimates carry no valid standard errors or p-values.")
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
        "Cov. type"    _label(m.cov_type)
    ]

    if m.method == :iv && m.first_stage_f !== nothing
        spec = vcat(spec, Any[
            "1st-stage F" _fmt(m.first_stage_f; digits=2)
        ])
        m.cragg_donald_f !== nothing && (spec = vcat(spec, Any[
            "Cragg-Donald F" _fmt(m.cragg_donald_f; digits=2)]))
        m.kleibergen_paap_f !== nothing && (spec = vcat(spec, Any[
            "Kleibergen-Paap F" _fmt(m.kleibergen_paap_f; digits=2)]))
        m.stock_yogo_10pct !== nothing && (spec = vcat(spec, Any[
            "Stock-Yogo 10%" _fmt(m.stock_yogo_10pct; digits=2)]))
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
    _degenerate_fit_banner(io, m.beta)
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
    _degenerate_fit_banner(io, m.beta)
    _sig_legend(io)
end

# =============================================================================
# Base.show — MarginalEffects
# =============================================================================

function Base.show(io::IO, me::MarginalEffects{T}) where {T}
    type_str = me.type == :ame ? "Average Marginal Effects" :
               me.type == :mem ? "Marginal Effects at Mean" :
               "Marginal Effects at Representative"

    # Intercept rows carry a NaN marginal effect (not defined) — omit them, like Stata,
    # and label the estimate column dy/dx rather than "Coef.". (S3/T167)
    keep = findall(isfinite, me.effects)
    _coef_table(io, type_str, me.varnames[keep], me.effects[keep], me.se[keep];
        dist = :z, level = me.conf_level, coef_label = "dy/dx")
    _sig_legend(io)
end

# =============================================================================
# OddsRatio — logit odds ratios with SEs (S3/T167)
# =============================================================================

"""
    OddsRatio{T}

Odds ratios for a logit model with delta-method standard errors and log-scale confidence
intervals. Fields mirror the former `odds_ratio` NamedTuple (`or`, `se`, `ci_lower`,
`ci_upper`, `varnames`) plus `conf_level`, so field access is unchanged.
"""
struct OddsRatio{T<:AbstractFloat}
    or::Vector{T}
    se::Vector{T}
    ci_lower::Vector{T}
    ci_upper::Vector{T}
    varnames::Vector{String}
    conf_level::T
end

function Base.show(io::IO, r::OddsRatio{T}) where {T}
    ci_pct = round(Int, 100 * r.conf_level)
    # omit the intercept row (its odds ratio is not interpretable)
    keep = findall(v -> _display_intercept(v) != _INTERCEPT_LABEL, r.varnames)
    isempty(keep) && (keep = collect(eachindex(r.varnames)))
    data = Matrix{Any}(undef, length(keep), 5)
    for (row, i) in enumerate(keep)
        data[row, 1] = r.varnames[i]
        data[row, 2] = _fmt(r.or[i])
        data[row, 3] = _fmt(r.se[i])
        data[row, 4] = _fmt(r.ci_lower[i])
        data[row, 5] = _fmt(r.ci_upper[i])
    end
    _pretty_table(io, data;
        title = "Odds Ratios",
        column_labels = ["", "Odds Ratio", "Std.Err.", "[$ci_pct%", "CI]"],
        alignment = [:l, :r, :r, :r, :r])
end
