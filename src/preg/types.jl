# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Panel regression types: PanelRegModel, PanelIVModel, PanelLogitModel,
PanelProbitModel, PanelTestResult, and their StatsAPI interfaces.
"""

# =============================================================================
# PanelRegModel — Linear panel (FE/RE/FD/Between/CRE)
# =============================================================================

"""
    PanelRegModel{T} <: StatsAPI.RegressionModel

Linear panel regression model estimated via Fixed Effects (FE), Random Effects (RE),
First-Difference (FD), Between, or Correlated Random Effects (CRE).

# Fields
- `beta::Vector{T}` — estimated coefficients
- `vcov_mat::Matrix{T}` — variance-covariance matrix of coefficients
- `residuals::Vector{T}` — residuals
- `fitted::Vector{T}` — fitted values
- `y::Vector{T}` — dependent variable
- `X::Matrix{T}` — regressor matrix
- `r2_within::T` — within R-squared
- `r2_between::T` — between R-squared
- `r2_overall::T` — overall R-squared
- `sigma_u::T` — between-group standard deviation
- `sigma_e::T` — within-group standard deviation
- `rho::T` — fraction of variance due to u_i: sigma_u^2 / (sigma_u^2 + sigma_e^2)
- `theta::Union{Nothing,T}` — quasi-demeaning parameter (RE only)
- `f_stat::T` — F-statistic for joint significance
- `f_pval::T` — p-value of the F-test
- `loglik::T` — log-likelihood
- `aic::T` — Akaike information criterion
- `bic::T` — Bayesian information criterion
- `varnames::Vector{String}` — coefficient names
- `method::Symbol` — :fe, :re, :fd, :between, :cre
- `twoway::Bool` — whether time fixed effects are included
- `cov_type::Symbol` — covariance estimator (:ols, :robust, :cluster, :driscoll_kraay)
- `n_obs::Int` — number of observations
- `n_groups::Int` — number of groups
- `n_periods_avg::T` — average number of periods per group
- `group_effects::Union{Nothing,Vector{T}}` — estimated group effects (FE only)
- `data::PanelData{T}` — original panel data

# References
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data*. 6th ed. Springer.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. 2nd ed. MIT Press.
"""
struct PanelRegModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    beta::Vector{T}
    vcov_mat::Matrix{T}
    residuals::Vector{T}
    fitted::Vector{T}
    y::Vector{T}
    X::Matrix{T}
    r2_within::T
    r2_between::T
    r2_overall::T
    sigma_u::T
    sigma_e::T
    rho::T
    theta::Union{Nothing,T}
    f_stat::T
    f_pval::T
    loglik::T
    aic::T
    bic::T
    varnames::Vector{String}
    method::Symbol
    twoway::Bool
    cov_type::Symbol
    n_obs::Int
    n_groups::Int
    n_periods_avg::T
    group_effects::Union{Nothing,Vector{T}}
    data::PanelData{T}
end

# =============================================================================
# PanelIVModel — Panel IV
# =============================================================================

"""
    PanelIVModel{T} <: StatsAPI.RegressionModel

Panel instrumental-variables regression model (FE-IV, RE-IV, FD-IV, Hausman-Taylor).

# Fields
- `beta::Vector{T}` — estimated coefficients
- `vcov_mat::Matrix{T}` — variance-covariance matrix
- `residuals::Vector{T}` — residuals
- `fitted::Vector{T}` — fitted values
- `y::Vector{T}` — dependent variable
- `X::Matrix{T}` — regressor matrix
- `Z::Matrix{T}` — instrument matrix
- `r2_within::T` — within R-squared
- `r2_between::T` — between R-squared
- `r2_overall::T` — overall R-squared
- `sigma_u::T` — between-group standard deviation
- `sigma_e::T` — within-group standard deviation
- `rho::T` — fraction of variance due to u_i
- `first_stage_f::T` — first-stage F-statistic
- `sargan_stat::Union{Nothing,T}` — Sargan overidentification statistic
- `sargan_pval::Union{Nothing,T}` — Sargan test p-value
- `varnames::Vector{String}` — coefficient names
- `endog_names::Vector{String}` — endogenous variable names
- `instrument_names::Vector{String}` — instrument names
- `method::Symbol` — :fe_iv, :re_iv, :fd_iv, :hausman_taylor
- `cov_type::Symbol` — covariance estimator
- `n_obs::Int` — number of observations
- `n_groups::Int` — number of groups
- `data::PanelData{T}` — original panel data

# References
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data*. 6th ed. Springer.
- Hausman, J. A. & Taylor, W. E. (1981). *Econometrica* 49(6), 1377-1398.
"""
struct PanelIVModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    beta::Vector{T}
    vcov_mat::Matrix{T}
    residuals::Vector{T}
    fitted::Vector{T}
    y::Vector{T}
    X::Matrix{T}
    Z::Matrix{T}
    r2_within::T
    r2_between::T
    r2_overall::T
    sigma_u::T
    sigma_e::T
    rho::T
    first_stage_f::T
    sargan_stat::Union{Nothing,T}
    sargan_pval::Union{Nothing,T}
    varnames::Vector{String}
    endog_names::Vector{String}
    instrument_names::Vector{String}
    method::Symbol
    cov_type::Symbol
    n_obs::Int
    n_groups::Int
    data::PanelData{T}
end

# =============================================================================
# PanelLogitModel — Panel Logit
# =============================================================================

"""
    PanelLogitModel{T} <: StatsAPI.RegressionModel

Panel logistic regression model (pooled, FE conditional, RE, CRE).

# Fields
- `beta::Vector{T}` — estimated coefficients
- `vcov_mat::Matrix{T}` — variance-covariance matrix
- `y::Vector{T}` — binary dependent variable (0/1)
- `X::Matrix{T}` — regressor matrix
- `fitted::Vector{T}` — predicted probabilities
- `loglik::T` — maximized log-likelihood
- `loglik_null::T` — null model log-likelihood
- `pseudo_r2::T` — McFadden's pseudo R-squared
- `aic::T` — Akaike information criterion
- `bic::T` — Bayesian information criterion
- `sigma_u::Union{Nothing,T}` — RE standard deviation (nothing for FE/pooled)
- `rho::Union{Nothing,T}` — fraction of variance due to u_i (RE only)
- `varnames::Vector{String}` — coefficient names
- `method::Symbol` — :pooled, :fe, :re, :cre
- `cov_type::Symbol` — covariance estimator
- `converged::Bool` — whether optimization converged
- `iterations::Int` — number of iterations
- `n_obs::Int` — number of observations
- `n_groups::Int` — number of groups
- `data::PanelData{T}` — original panel data

# References
- Chamberlain, G. (1980). *Review of Economic Studies* 47(1), 225-238.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. 2nd ed. MIT Press.
"""
struct PanelLogitModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    beta::Vector{T}
    vcov_mat::Matrix{T}
    y::Vector{T}
    X::Matrix{T}
    fitted::Vector{T}
    loglik::T
    loglik_null::T
    pseudo_r2::T
    aic::T
    bic::T
    sigma_u::Union{Nothing,T}
    rho::Union{Nothing,T}
    varnames::Vector{String}
    method::Symbol
    cov_type::Symbol
    converged::Bool
    iterations::Int
    n_obs::Int
    n_groups::Int
    data::PanelData{T}
end

# =============================================================================
# PanelProbitModel — Panel Probit (identical structure to PanelLogitModel)
# =============================================================================

"""
    PanelProbitModel{T} <: StatsAPI.RegressionModel

Panel probit regression model (pooled, FE conditional, RE, CRE).
Identical structure to `PanelLogitModel{T}`, using the standard normal CDF link.

# References
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. 2nd ed. MIT Press.
"""
struct PanelProbitModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    beta::Vector{T}
    vcov_mat::Matrix{T}
    y::Vector{T}
    X::Matrix{T}
    fitted::Vector{T}
    loglik::T
    loglik_null::T
    pseudo_r2::T
    aic::T
    bic::T
    sigma_u::Union{Nothing,T}
    rho::Union{Nothing,T}
    varnames::Vector{String}
    method::Symbol
    cov_type::Symbol
    converged::Bool
    iterations::Int
    n_obs::Int
    n_groups::Int
    data::PanelData{T}
end

# =============================================================================
# PanelTestResult — Specification tests
# =============================================================================

"""
    PanelTestResult{T} <: StatsAPI.HypothesisTest

Result from a panel specification test (Hausman, Breusch-Pagan, Pesaran CD, etc.).

# Fields
- `test_name::String` — e.g. "Hausman test"
- `statistic::T` — test statistic value
- `pvalue::T` — p-value
- `df::Union{Int,Tuple{Int,Int}}` — chi-squared df or (df1, df2) for F-tests
- `description::String` — human-readable description of the test result
"""
struct PanelTestResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest
    test_name::String
    statistic::T
    pvalue::T
    df::Union{Int,Tuple{Int,Int}}
    description::String
end

# =============================================================================
# StatsAPI Interface — PanelRegModel
# =============================================================================

StatsAPI.coef(m::PanelRegModel) = m.beta
StatsAPI.vcov(m::PanelRegModel) = m.vcov_mat
StatsAPI.residuals(m::PanelRegModel) = m.residuals
StatsAPI.predict(m::PanelRegModel) = m.fitted
StatsAPI.nobs(m::PanelRegModel) = m.n_obs
StatsAPI.dof(m::PanelRegModel) = length(m.beta)
StatsAPI.loglikelihood(m::PanelRegModel) = m.loglik
StatsAPI.aic(m::PanelRegModel) = m.aic
StatsAPI.bic(m::PanelRegModel) = m.bic
StatsAPI.islinear(::PanelRegModel) = true
StatsAPI.r2(m::PanelRegModel) = m.r2_within

function StatsAPI.dof_residual(m::PanelRegModel)
    m.n_obs - length(m.beta) - (m.method == :fe ? m.n_groups : 0)
end

function StatsAPI.stderror(m::PanelRegModel{T}) where {T}
    sqrt.(max.(diag(m.vcov_mat), zero(T)))
end

"""Confidence intervals for PanelRegModel coefficients (t-distribution)."""
function StatsAPI.confint(m::PanelRegModel{T}; level::Real=0.95) where {T}
    se = stderror(m)
    df_r = dof_residual(m)
    crit = T(quantile(TDist(df_r), 1 - (1 - level) / 2))
    hcat(m.beta .- crit .* se, m.beta .+ crit .* se)
end

# =============================================================================
# StatsAPI Interface — PanelIVModel
# =============================================================================

StatsAPI.coef(m::PanelIVModel) = m.beta
StatsAPI.vcov(m::PanelIVModel) = m.vcov_mat
StatsAPI.residuals(m::PanelIVModel) = m.residuals
StatsAPI.predict(m::PanelIVModel) = m.fitted
StatsAPI.nobs(m::PanelIVModel) = m.n_obs
StatsAPI.dof(m::PanelIVModel) = length(m.beta)
StatsAPI.islinear(::PanelIVModel) = true
StatsAPI.r2(m::PanelIVModel) = m.r2_within

function StatsAPI.dof_residual(m::PanelIVModel)
    m.n_obs - length(m.beta) - (m.method == :fe_iv ? m.n_groups : 0)
end

function StatsAPI.stderror(m::PanelIVModel{T}) where {T}
    sqrt.(max.(diag(m.vcov_mat), zero(T)))
end

"""Confidence intervals for PanelIVModel coefficients (t-distribution)."""
function StatsAPI.confint(m::PanelIVModel{T}; level::Real=0.95) where {T}
    se = stderror(m)
    df_r = dof_residual(m)
    crit = T(quantile(TDist(df_r), 1 - (1 - level) / 2))
    hcat(m.beta .- crit .* se, m.beta .+ crit .* se)
end

# =============================================================================
# StatsAPI Interface — PanelLogitModel
# =============================================================================

StatsAPI.coef(m::PanelLogitModel) = m.beta
StatsAPI.vcov(m::PanelLogitModel) = m.vcov_mat
StatsAPI.predict(m::PanelLogitModel) = m.fitted
StatsAPI.nobs(m::PanelLogitModel) = m.n_obs
StatsAPI.dof(m::PanelLogitModel) = length(m.beta)
StatsAPI.dof_residual(m::PanelLogitModel) = m.n_obs - length(m.beta)
StatsAPI.loglikelihood(m::PanelLogitModel) = m.loglik
StatsAPI.aic(m::PanelLogitModel) = m.aic
StatsAPI.bic(m::PanelLogitModel) = m.bic
StatsAPI.islinear(::PanelLogitModel) = false

function StatsAPI.stderror(m::PanelLogitModel{T}) where {T}
    sqrt.(max.(diag(m.vcov_mat), zero(T)))
end

"""Confidence intervals for PanelLogitModel coefficients (normal distribution)."""
function StatsAPI.confint(m::PanelLogitModel{T}; level::Real=0.95) where {T}
    se = stderror(m)
    crit = T(quantile(Normal(), 1 - (1 - level) / 2))
    hcat(m.beta .- crit .* se, m.beta .+ crit .* se)
end

# =============================================================================
# StatsAPI Interface — PanelProbitModel
# =============================================================================

StatsAPI.coef(m::PanelProbitModel) = m.beta
StatsAPI.vcov(m::PanelProbitModel) = m.vcov_mat
StatsAPI.predict(m::PanelProbitModel) = m.fitted
StatsAPI.nobs(m::PanelProbitModel) = m.n_obs
StatsAPI.dof(m::PanelProbitModel) = length(m.beta)
StatsAPI.dof_residual(m::PanelProbitModel) = m.n_obs - length(m.beta)
StatsAPI.loglikelihood(m::PanelProbitModel) = m.loglik
StatsAPI.aic(m::PanelProbitModel) = m.aic
StatsAPI.bic(m::PanelProbitModel) = m.bic
StatsAPI.islinear(::PanelProbitModel) = false

function StatsAPI.stderror(m::PanelProbitModel{T}) where {T}
    sqrt.(max.(diag(m.vcov_mat), zero(T)))
end

"""Confidence intervals for PanelProbitModel coefficients (normal distribution)."""
function StatsAPI.confint(m::PanelProbitModel{T}; level::Real=0.95) where {T}
    se = stderror(m)
    crit = T(quantile(Normal(), 1 - (1 - level) / 2))
    hcat(m.beta .- crit .* se, m.beta .+ crit .* se)
end

# =============================================================================
# Base.show — PanelRegModel
# =============================================================================

function Base.show(io::IO, m::PanelRegModel{T}) where {T}
    method_str = m.method == :fe ? "Fixed Effects" :
                 m.method == :re ? "Random Effects" :
                 m.method == :fd ? "First-Difference" :
                 m.method == :between ? "Between" :
                 m.method == :ab ? "Arellano-Bond" :
                 m.method == :bb ? "Blundell-Bond" : "Correlated RE"
    twoway_str = m.twoway ? " (Two-way)" : ""

    spec = Any[
        "Method"         method_str * twoway_str;
        "Observations"   m.n_obs;
        "Groups"         m.n_groups;
        "Avg. periods"   _fmt(m.n_periods_avg; digits=1);
        "R-sq. within"   _fmt(m.r2_within);
        "R-sq. between"  _fmt(m.r2_between);
        "R-sq. overall"  _fmt(m.r2_overall);
        "sigma_u"        _fmt(m.sigma_u);
        "sigma_e"        _fmt(m.sigma_e);
        "rho"            _fmt(m.rho);
        "F-statistic"    _fmt(m.f_stat; digits=2);
        "F p-value"      _format_pvalue(m.f_pval);
        "Cov. type"      string(m.cov_type)
    ]
    if m.theta !== nothing
        spec = vcat(spec, Any["theta" _fmt(m.theta)])
    end

    _pretty_table(io, spec;
        title = "Panel Regression — $method_str$twoway_str",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    _coef_table(io, "Coefficients", m.varnames, m.beta, stderror(m);
        dist = :t, dof_r = dof_residual(m))
    _sig_legend(io)
end

# =============================================================================
# Base.show — PanelIVModel
# =============================================================================

function Base.show(io::IO, m::PanelIVModel{T}) where {T}
    method_str = m.method == :fe_iv ? "FE-IV" :
                 m.method == :re_iv ? "RE-IV" :
                 m.method == :fd_iv ? "FD-IV" : "Hausman-Taylor"

    spec = Any[
        "Method"           method_str;
        "Observations"     m.n_obs;
        "Groups"           m.n_groups;
        "R-sq. within"     _fmt(m.r2_within);
        "R-sq. between"    _fmt(m.r2_between);
        "R-sq. overall"    _fmt(m.r2_overall);
        "sigma_u"          _fmt(m.sigma_u);
        "sigma_e"          _fmt(m.sigma_e);
        "rho"              _fmt(m.rho);
        "1st-stage F"      _fmt(m.first_stage_f; digits=2);
        "Endogenous"       join(m.endog_names, ", ");
        "Instruments"      join(m.instrument_names, ", ");
        "Cov. type"        string(m.cov_type)
    ]
    if m.sargan_stat !== nothing
        spec = vcat(spec, Any[
            "Sargan stat." _fmt(m.sargan_stat; digits=2);
            "Sargan p-val" _format_pvalue(m.sargan_pval)
        ])
    end

    _pretty_table(io, spec;
        title = "Panel IV Regression — $method_str",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    _coef_table(io, "Coefficients", m.varnames, m.beta, stderror(m);
        dist = :t, dof_r = dof_residual(m))
    _sig_legend(io)
end

# =============================================================================
# Base.show — PanelLogitModel
# =============================================================================

function Base.show(io::IO, m::PanelLogitModel{T}) where {T}
    method_str = m.method == :pooled ? "Pooled" :
                 m.method == :fe ? "Fixed Effects" :
                 m.method == :re ? "Random Effects" : "Correlated RE"

    spec = Any[
        "Model"          "Logit ($method_str)";
        "Observations"   m.n_obs;
        "Groups"         m.n_groups;
        "Log-lik."       _fmt(m.loglik; digits=2);
        "Log-lik. null"  _fmt(m.loglik_null; digits=2);
        "Pseudo R-sq."   _fmt(m.pseudo_r2);
        "AIC"            _fmt(m.aic; digits=2);
        "BIC"            _fmt(m.bic; digits=2);
        "Converged"      m.converged ? "Yes" : "No";
        "Iterations"     m.iterations
    ]
    if m.sigma_u !== nothing
        spec = vcat(spec, Any[
            "sigma_u" _fmt(m.sigma_u);
            "rho"     _fmt(m.rho)
        ])
    end

    _pretty_table(io, spec;
        title = "Panel Logit — $method_str",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    _coef_table(io, "Coefficients", m.varnames, m.beta, stderror(m);
        dist = :z)
    _sig_legend(io)
end

# =============================================================================
# Base.show — PanelProbitModel
# =============================================================================

function Base.show(io::IO, m::PanelProbitModel{T}) where {T}
    method_str = m.method == :pooled ? "Pooled" :
                 m.method == :fe ? "Fixed Effects" :
                 m.method == :re ? "Random Effects" : "Correlated RE"

    spec = Any[
        "Model"          "Probit ($method_str)";
        "Observations"   m.n_obs;
        "Groups"         m.n_groups;
        "Log-lik."       _fmt(m.loglik; digits=2);
        "Log-lik. null"  _fmt(m.loglik_null; digits=2);
        "Pseudo R-sq."   _fmt(m.pseudo_r2);
        "AIC"            _fmt(m.aic; digits=2);
        "BIC"            _fmt(m.bic; digits=2);
        "Converged"      m.converged ? "Yes" : "No";
        "Iterations"     m.iterations
    ]
    if m.sigma_u !== nothing
        spec = vcat(spec, Any[
            "sigma_u" _fmt(m.sigma_u);
            "rho"     _fmt(m.rho)
        ])
    end

    _pretty_table(io, spec;
        title = "Panel Probit — $method_str",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    _coef_table(io, "Coefficients", m.varnames, m.beta, stderror(m);
        dist = :z)
    _sig_legend(io)
end

# =============================================================================
# Base.show — PanelTestResult
# =============================================================================

function Base.show(io::IO, t::PanelTestResult{T}) where {T}
    df_str = t.df isa Tuple ? "($(t.df[1]), $(t.df[2]))" : string(t.df)
    data = Any[
        "Test"        t.test_name;
        "Statistic"   _fmt(t.statistic);
        "P-value"     _format_pvalue(t.pvalue);
        "DF"          df_str;
        "Result"      t.description
    ]
    _pretty_table(io, data;
        title = t.test_name,
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end
