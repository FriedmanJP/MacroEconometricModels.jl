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
- `kclass_k::Union{Nothing,T}` — k-class scalar `k` actually used (IV k-class only; `nothing` for
  the plain `:tsls`/`:2sls` path). `k=0` is OLS, `k=1` is 2SLS, `k=κ̂` is LIML,
  `k=κ̂−a/(n−m)` is Fuller.
- `kappa_hat::Union{Nothing,T}` — LIML minimum-eigenvalue `κ̂` (IV `:liml`/`:fuller` only;
  `nothing` otherwise). The implied Anderson (1949) LR overidentification statistic is `n·ln(κ̂)`.

# References
- White, H. (1980). *Econometrica* 48(4), 817-838.
- MacKinnon, J. G. & White, H. (1985). *JBES* 3(3), 305-314.
- Stock, J. H. & Yogo, M. (2005). *Identification and Inference for Econometric Models*, ch. 5.
- Kleibergen, F. & Paap, R. (2006). *Journal of Econometrics* 133(1), 97-126.
- Anderson, T. W. & Rubin, H. (1949). *Ann. Math. Statist.* 20(1), 46-63.
- Fuller, W. A. (1977). *Econometrica* 45(4), 939-953.
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
    kclass_k::Union{Nothing,T}          # k-class scalar (IV k-class only) — EV-36 (#444)
    kappa_hat::Union{Nothing,T}         # LIML minimum eigenvalue κ̂ — EV-36 (#444)
end

# Back-compat outer constructor: legacy 24-arg positional calls (through `sargan_pval`)
# default the weak-instrument diagnostic + k-class fields to `nothing`.
function RegModel{T}(y, X, beta, vcov_mat, residuals, fitted, ssr, tss, r2, adj_r2,
                     f_stat, f_pval, loglik, aic, bic, varnames, method, cov_type,
                     weights, Z, endogenous, first_stage_f, sargan_stat,
                     sargan_pval) where {T<:AbstractFloat}
    RegModel{T}(y, X, beta, vcov_mat, residuals, fitted, ssr, tss, r2, adj_r2,
                f_stat, f_pval, loglik, aic, bic, varnames, method, cov_type,
                weights, Z, endogenous, first_stage_f, sargan_stat, sargan_pval,
                nothing, nothing, nothing, nothing, nothing)
end

# Back-compat outer constructor: 27-arg positional calls (through `stock_yogo_10pct`,
# i.e. the pre-EV-36 IV constructor) default the two k-class fields to `nothing`.
function RegModel{T}(y, X, beta, vcov_mat, residuals, fitted, ssr, tss, r2, adj_r2,
                     f_stat, f_pval, loglik, aic, bic, varnames, method, cov_type,
                     weights, Z, endogenous, first_stage_f, sargan_stat, sargan_pval,
                     cragg_donald_f, kleibergen_paap_f, stock_yogo_10pct) where {T<:AbstractFloat}
    RegModel{T}(y, X, beta, vcov_mat, residuals, fitted, ssr, tss, r2, adj_r2,
                f_stat, f_pval, loglik, aic, bic, varnames, method, cov_type,
                weights, Z, endogenous, first_stage_f, sargan_stat, sargan_pval,
                cragg_donald_f, kleibergen_paap_f, stock_yogo_10pct, nothing, nothing)
end

# =============================================================================
# RobustRegModel — Huber/bisquare M-estimation + Yohai MM-estimation (EV-40, #448)
# =============================================================================

"""
    RobustRegModel{T} <: StatsAPI.RegressionModel

Outlier-resistant linear regression fitted by bounded-influence M-estimation (Huber or
Tukey bisquare, via iteratively reweighted least squares) or Yohai high-breakdown
MM-estimation (fast-S subsampling scale seed followed by a high-efficiency bisquare
M-step). See [`estimate_robust`](@ref).

M-estimation minimizes `Σ ρ((yᵢ − xᵢ'β)/ŝ)` for a bounded ρ, downweighting observations
with large scaled residuals `uᵢ = rᵢ/ŝ`. The final ψ-weights `wᵢ = ψ(uᵢ)/uᵢ` flag outliers:
points with `wᵢ ≈ 0` are effectively discarded.

# Fields
- `y::Vector{T}`, `X::Matrix{T}` — response and regressors (include an intercept column).
- `beta::Vector{T}` — robust coefficient estimates.
- `vcov_mat::Matrix{T}` — Huber–Ronchetti sandwich covariance of `beta`.
- `scale::T` — robust residual scale `ŝ`: normalized MAD (M-estimation) or the
  high-breakdown S-scale (MM-estimation).
- `weights::Vector{T}` — final ψ-weights `wᵢ = ψ(uᵢ)/uᵢ` (outlier flags; `≈0` ⇒ downweighted).
- `residuals::Vector{T}` — `yᵢ − xᵢ'β`.
- `fitted::Vector{T}` — `X·β`.
- `psi::Symbol` — influence function: `:huber` or `:bisquare`.
- `method::Symbol` — `:m` (M-estimation) or `:mm` (MM-estimation).
- `tuning::T` — the ψ tuning constant actually used (Huber `k`, bisquare `c`).
- `robust_r2::T` — robust R² `1 − Σρ(rᵢ/ŝ)/Σρ((yᵢ − median y)/ŝ)`.
- `varnames::Vector{String}` — coefficient names.
- `converged::Bool` — whether the IRLS M-step converged.
- `iterations::Int` — number of IRLS iterations in the (final) M-step.

# References
- Huber, P. J. (1964). *Annals of Mathematical Statistics* 35(1), 73-101.
- Huber, P. J. & Ronchetti, E. M. (2009). *Robust Statistics*. 2nd ed. Wiley.
- Yohai, V. J. (1987). *Annals of Statistics* 15(2), 642-656.
- Salibian-Barrera, M. & Yohai, V. J. (2006). *J. Computational and Graphical Statistics* 15(2), 414-427.
"""
struct RobustRegModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    y::Vector{T}
    X::Matrix{T}
    beta::Vector{T}
    vcov_mat::Matrix{T}
    scale::T
    weights::Vector{T}
    residuals::Vector{T}
    fitted::Vector{T}
    psi::Symbol
    method::Symbol
    tuning::T
    robust_r2::T
    varnames::Vector{String}
    converged::Bool
    iterations::Int
end

StatsAPI.coef(m::RobustRegModel) = m.beta
StatsAPI.vcov(m::RobustRegModel) = m.vcov_mat
StatsAPI.residuals(m::RobustRegModel) = m.residuals
StatsAPI.predict(m::RobustRegModel) = m.fitted
StatsAPI.nobs(m::RobustRegModel) = length(m.y)
StatsAPI.dof(m::RobustRegModel) = length(m.beta)
StatsAPI.dof_residual(m::RobustRegModel) = length(m.y) - length(m.beta)
StatsAPI.islinear(::RobustRegModel) = true
StatsAPI.stderror(m::RobustRegModel) = sqrt.(diag(m.vcov_mat))

"""Confidence intervals for RobustRegModel coefficients (t-distribution)."""
function StatsAPI.confint(m::RobustRegModel{T}; level::Real=0.95) where {T}
    se = stderror(m)
    df_r = dof_residual(m)
    crit = T(quantile(TDist(df_r), 1 - (1 - level) / 2))
    hcat(m.beta .- crit .* se, m.beta .+ crit .* se)
end

function Base.show(io::IO, m::RobustRegModel{T}) where {T}
    n = nobs(m)
    k = dof(m)
    psi_str = m.psi == :huber ? "Huber" : "Tukey bisquare"
    method_str = m.method == :mm ? "MM-estimation" : "M-estimation"
    # Downweighted-observation count: ψ-weight materially below one (Stata/rlm convention).
    n_down = count(w -> w < T(0.999), m.weights)
    n_zero = count(w -> w < T(1e-3), m.weights)

    spec = Any[
        "Method"          method_str;
        "Influence fn."   psi_str;
        "Tuning const."   _fmt(m.tuning; digits=3);
        "Observations"    n;
        "Covariates"      k;
        "Robust scale"    _fmt(m.scale; digits=4);
        "Robust R-sq."    _fmt(m.robust_r2);
        "Downweighted"    "$n_down of $n";
        "Rejected (w≈0)"  n_zero;
        "Converged"       m.converged ? "Yes" : "No";
        "Iterations"      m.iterations
    ]
    _pretty_table(io, spec;
        title = "Robust Regression ($method_str, $psi_str)",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    _coef_table(io, "Coefficients", m.varnames, m.beta, stderror(m);
        dist = :t, dof_r = dof_residual(m))
    _sig_legend(io)
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
# SelectionResult — variable selection (stepwise / best-subset / GETS) (EV-04, #412)
# =============================================================================

"""
    SelectionResult{T}

Result of automated regressor selection via [`select_variables`](@ref): stepwise
(forward/backward/bidirectional), exhaustive best-subset, or the LSE
general-to-specific (GETS) multi-path reduction.

# Fields
- `method::Symbol` — `:forward`, `:backward`, `:bidirectional`, `:best_subset`, or `:gets`.
- `criterion::Symbol` — move criterion (`:pvalue`, `:aic`, `:bic`).
- `selected::Vector{Int}` — selected column indices into the GUM (sorted; includes forced keeps).
- `keep::Vector{Int}` — forced-in column indices (detected intercepts ∪ user `keep`).
- `varnames::Vector{String}` — names of all GUM candidate columns.
- `path::Vector{Tuple{Symbol,Int,T}}` — ordered search moves `(action, column, criterion value)`
  where `action ∈ (:enter, :remove, :retain, :best_subset)`.
- `terminal_models::Vector{Vector{Int}}` — GETS diagnostic-passing terminal column sets (empty otherwise).
- `encompassing_f::Union{Nothing,T}` — parsimonious-encompassing F-statistic of the selection vs the GUM.
- `encompassing_pval::Union{Nothing,T}` — its p-value.
- `encompassing_df::Union{Nothing,Tuple{Int,Int}}` — `(q, n−k_gum)` F degrees of freedom.
- `final::RegModel{T}` — the refit selected model (post-selection SEs are conditional-on-selection).
- `n_gum::Int` — number of GUM candidate columns.

# References
- Hoover, K. D. & Perez, S. J. (1999). *Econometrics Journal* 2(2), 167-191.
- Hendry, D. F. & Krolzig, H.-M. (2005). *Economic Journal* 115(502), C32-C61.
- Pretis, F., Reade, J. J. & Sucarrat, G. (2018). *J. Statistical Software* 86(3).
"""
struct SelectionResult{T<:AbstractFloat}
    method::Symbol
    criterion::Symbol
    selected::Vector{Int}
    keep::Vector{Int}
    varnames::Vector{String}
    path::Vector{Tuple{Symbol,Int,T}}
    terminal_models::Vector{Vector{Int}}
    encompassing_f::Union{Nothing,T}
    encompassing_pval::Union{Nothing,T}
    encompassing_df::Union{Nothing,Tuple{Int,Int}}
    final::RegModel{T}
    n_gum::Int
end

function Base.show(io::IO, r::SelectionResult{T}) where {T}
    method_str = r.method === :forward ? "Forward stepwise" :
                 r.method === :backward ? "Backward stepwise" :
                 r.method === :bidirectional ? "Bidirectional stepwise" :
                 r.method === :best_subset ? "Best subset" : "General-to-Specific (GETS)"
    crit_str = uppercase(string(r.criterion))

    spec = Any[
        "Method"            method_str;
        "Criterion"         crit_str;
        "GUM regressors"    r.n_gum;
        "Selected"          length(r.selected);
        "Observations"      length(r.final.y)
    ]
    if r.method === :gets
        spec = vcat(spec, Any["Terminal models" length(r.terminal_models)])
    end
    if r.encompassing_f !== nothing
        spec = vcat(spec, Any[
            "Encompassing F"    _fmt(r.encompassing_f; digits=3);
            "Encompassing p"    _format_pvalue(r.encompassing_pval)])
    end
    _pretty_table(io, spec;
        title = "Variable Selection — $method_str",
        column_labels = ["Specification", ""],
        alignment = [:l, :r])

    # Search path (omit the trivial best-subset marker row).
    steps = filter(s -> s[1] !== :best_subset, r.path)
    if !isempty(steps)
        data = Matrix{Any}(undef, length(steps), 4)
        for (i, s) in enumerate(steps)
            act = s[1] === :enter ? "enter" : s[1] === :remove ? "remove" : "retain"
            data[i, 1] = i
            data[i, 2] = act
            data[i, 3] = r.varnames[s[2]]
            data[i, 4] = _fmt(s[3])
        end
        cval = r.criterion === :pvalue ? "p-value" : crit_str
        _pretty_table(io, data;
            title = "Search Path",
            column_labels = ["Step", "Action", "Variable", cval],
            alignment = [:r, :l, :l, :r])
    end

    println(io)
    println(io, "Selected model (post-selection SEs are conditional-on-selection):")
    show(io, r.final)
end

# StabilityResult / InfluenceStats — parameter-stability diagnostics (EV-32, #440)
# =============================================================================

"""
    StabilityResult{T} <: StatsAPI.HypothesisTest

Result of a recursive-residual parameter-stability test (Brown–Durbin–Evans 1975):
the CUSUM (`:cusum`) or CUSUM-of-squares (`:cusumsq`) path together with its
significance-band lines and a crossing flag. Produced by [`cusum_test`](@ref) and
[`cusumsq_test`](@ref).

# Fields
- `kind::Symbol` — `:cusum` or `:cusumsq`.
- `tindex::Vector{Int}` — observation index `t = k+1 … n` for each path point.
- `stat_path::Vector{T}` — the CUSUM statistic `W_t` (`:cusum`) or CUSUMSQ `S_t`
  (`:cusumsq`) at each `t`.
- `upper::Vector{T}` — upper significance-band line at each `t`.
- `lower::Vector{T}` — lower significance-band line at each `t`.
- `crossed::Bool` — whether the path breaches either band anywhere.
- `first_crossing::Union{Nothing,Int}` — observation index of the first breach
  (`nothing` if the path stays inside the bands).
- `level::T` — significance level of the bands (e.g. `0.05`).
- `recursive_resid::Vector{T}` — the standardized recursive residuals `w_t`.
- `n::Int`, `k::Int` — sample size and number of regressors.

# References
- Brown, R. L., Durbin, J. & Evans, J. M. (1975). *JRSS-B* 37(2), 149–192.
- Edgerton, D. & Wells, C. (1994). *Oxford Bull. Econ. Stat.* 56(3), 355–365.
"""
struct StabilityResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest
    kind::Symbol
    tindex::Vector{Int}
    stat_path::Vector{T}
    upper::Vector{T}
    lower::Vector{T}
    crossed::Bool
    first_crossing::Union{Nothing,Int}
    level::T
    recursive_resid::Vector{T}
    n::Int
    k::Int
end

"""
    InfluenceStats{T}

Observation-level regression influence and leverage statistics for a fitted
[`RegModel`](@ref), following Belsley, Kuh & Welsch (1980). Produced by
[`influence_stats`](@ref).

# Fields
- `hat::Vector{T}` — hat-matrix diagonals (leverage) `h_ii = x_i'(X'X)⁻¹x_i`.
- `student_internal::Vector{T}` — internally studentized residuals (R `rstandard`).
- `student_external::Vector{T}` — externally studentized residuals (R `rstudent`).
- `dffits::Vector{T}` — `DFFITS_i = t*_i √(h_ii/(1−h_ii))`.
- `cooksd::Vector{T}` — Cook's distance `D_i = (r_i²/k)·(h_ii/(1−h_ii))`.
- `dfbetas::Matrix{T}` — `n × k` DFBETAS (per-observation, per-coefficient).
- `sigma::T` — OLS residual standard error `√(SSR/(n−k))`.
- `high_leverage::Vector{Int}` — indices flagged `h_ii > 2k/n` (BKW rule of thumb).
- `influential::Vector{Int}` — indices flagged `|DFFITS_i| > 2√(k/n)`.
- `varnames::Vector{String}` — coefficient names (columns of `dfbetas`).
- `n::Int`, `k::Int` — sample size and number of regressors.

# References
- Belsley, D. A., Kuh, E. & Welsch, R. E. (1980). *Regression Diagnostics*. Wiley.
- Cook, R. D. (1977). *Technometrics* 19(1), 15–18.
"""
struct InfluenceStats{T<:AbstractFloat}
    hat::Vector{T}
    student_internal::Vector{T}
    student_external::Vector{T}
    dffits::Vector{T}
    cooksd::Vector{T}
    dfbetas::Matrix{T}
    sigma::T
    high_leverage::Vector{Int}
    influential::Vector{Int}
    varnames::Vector{String}
    n::Int
    k::Int
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
        # k-class family (EV-36): report the k-class scalar and, for LIML/Fuller, κ̂ plus
        # the Anderson (1949) LR overidentification statistic n·ln(κ̂) ~ χ²(m − k_reg).
        if m.kclass_k !== nothing
            spec = vcat(spec, Any["k-class k" _fmt(m.kclass_k; digits=4)])
        end
        if m.kappa_hat !== nothing
            spec = vcat(spec, Any["LIML κ̂" _fmt(m.kappa_hat; digits=4)])
            n_iv = size(m.Z, 2)
            dof_ar = n_iv - k
            if dof_ar > 0 && m.kappa_hat > zero(T)
                ar_stat = T(n) * log(m.kappa_hat)
                ar_p = T(1 - cdf(Chisq(dof_ar), max(ar_stat, zero(T))))
                spec = vcat(spec, Any["Anderson LR" _fmt(ar_stat; digits=2)])
                spec = vcat(spec, Any["Anderson LR p" _format_pvalue(ar_p)])
            end
        end
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

# =============================================================================
# HeckmanModel — sample-selection (incidental truncation) — EV-18 (#426)
# =============================================================================

"""
    HeckmanModel{T} <: StatsAPI.RegressionModel

Heckman (1979) sample-selection model for incidentally truncated data: the continuous outcome
`y` is observed only for the subsample where a binary selection indicator `d == 1`. Estimated by
[`estimate_heckman`](@ref) via either the two-step (Heckit) or full-information ML method.

# Fields
- `beta::Vector{T}` — outcome-equation coefficients (on `X`).
- `vcov_beta::Matrix{T}` — covariance of `beta`. For `:twostep`, the Greene-corrected two-step
  covariance (generated-regressor + selection-induced heteroskedasticity); for `:mle`, the
  delta-method block of the observed-information inverse.
- `outcome_names::Vector{String}` — outcome regressor names.
- `gamma::Vector{T}` — selection-equation (probit) coefficients (on `Z`).
- `vcov_gamma::Matrix{T}` — covariance of `gamma` (probit information inverse from step 1).
- `select_names::Vector{String}` — selection regressor names.
- `rho::T` — correlation between the outcome and selection errors.
- `sigma::T` — standard deviation of the outcome error.
- `lambda::T` — `rho * sigma`, the coefficient on the inverse-Mills ratio (two-step) or its
  MLE-implied value; the two-step `t`-test on `lambda` is the `H₀: no selection` test.
- `rho_se::T`, `sigma_se::T`, `lambda_se::T` — standard errors of `(rho, sigma, lambda)`
  (delta-method under `:mle`; the OLS/Greene SE of the Mills coefficient for `:twostep`, with
  `rho_se`/`sigma_se` set to `NaN` because the two-step does not deliver them directly).
- `mills::Vector{T}` — fitted inverse-Mills ratio `φ(z'γ̂)/Φ(z'γ̂)` for the selected observations.
- `method::Symbol` — `:twostep` or `:mle`.
- `loglik::T`, `aic::T`, `bic::T` — bivariate-normal log-likelihood and information criteria
  (`:mle`); for `:twostep` the profile log-likelihood evaluated at the two-step point estimates.
- `n_selected::Int`, `n_total::Int` — selected-subsample and full-sample counts.
- `y::Vector{T}`, `X::Matrix{T}` — outcome and outcome-regressors over the SELECTED subsample.
- `converged::Bool` — optimizer convergence (`true` for `:twostep`).

See also [`estimate_heckman`](@ref).
"""
struct HeckmanModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    beta::Vector{T}
    vcov_beta::Matrix{T}
    outcome_names::Vector{String}
    gamma::Vector{T}
    vcov_gamma::Matrix{T}
    select_names::Vector{String}
    rho::T
    sigma::T
    lambda::T
    rho_se::T
    sigma_se::T
    lambda_se::T
    mills::Vector{T}
    method::Symbol
    loglik::T
    aic::T
    bic::T
    n_selected::Int
    n_total::Int
    y::Vector{T}
    X::Matrix{T}
    converged::Bool
end
