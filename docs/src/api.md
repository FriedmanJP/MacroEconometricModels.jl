# [API Reference](@id api_page)

Complete API documentation for **MacroEconometricModels.jl**. Every export carries its docstring on exactly one of the twelve per-domain reference pages below. The quick-reference tables that follow group the exported functions by task and route each group to the narrative page that explains it.

The full API documentation is organized into the following per-domain pages:

- **[Data Management](@ref api_data)** — containers, datasets, cleaning, panel construction
- **[Univariate Models](@ref api_univariate)** — filters, ARIMA, volatility, spectral analysis
- **[Multivariate Models](@ref api_multivariate)** — VAR, VECM, BVAR, LP, factor models, FAVAR, innovation accounting
- **[Cross-Sectional Models](@ref api_cross_section)** — OLS/IV, logit/probit, ordered/multinomial
- **[Panel Models](@ref api_panel)** — panel VAR, panel regression, DiD, panel unit-root tests
- **[DSGE Models](@ref api_dsge)** — specification, solvers, estimation, constraints
- **[Structural & Statistical Identification](@ref api_structural)** — SVAR schemes, non-Gaussian/heteroskedastic identification
- **[GMM & SMM](@ref api_gmm)** — moment-based estimation
- **[Hypothesis Tests](@ref api_tests)** — unit root, breaks, comparison, portmanteau
- **[Nowcasting](@ref api_nowcasting)** — DFM/BVAR/bridge nowcasting and news
- **[Visualization](@ref api_visualization)** — `plot_result` dispatches
- **[Utilities & Display](@ref api_utilities)** — covariance estimators, output, references

```@docs
MacroEconometricModels.MacroEconometricModels
```

## Type Hierarchy

The abstract-type hierarchy is derived at build time by walking `subtypes()` from the package's top-level abstract roots, so it never drifts from the source:

```@eval
using MacroEconometricModels
import InteractiveUtils
import Markdown
const _M = MacroEconometricModels
_pkgabs = Set{Type}()
for n in names(_M; all=true)
    isdefined(_M, n) || continue
    v = getfield(_M, n)
    (v isa Type && isabstracttype(v) && parentmodule(v) === _M) && push!(_pkgabs, v)
end
_roots = sort!([t for t in _pkgabs if !(supertype(t) in _pkgabs)]; by=string)
_io = IOBuffer()
function _walk(io, t, indent)
    kids = sort!(InteractiveUtils.subtypes(t); by=string)
    for (i, k) in enumerate(kids)
        last = i == length(kids)
        println(io, indent, last ? "└── " : "├── ", nameof(k))
        _walk(io, k, indent * (last ? "    " : "│   "))
    end
end
for r in _roots
    sup = supertype(r)
    supstr = parentmodule(sup) === _M ? string(nameof(sup)) : string(sup)
    println(_io, nameof(r), " <: ", supstr)
    _walk(_io, r, "")
    println(_io)
end
Markdown.parse("```\n" * String(take!(_io)) * "```")
```

## Quick Reference Tables

Typed data containers, built-in datasets (FRED-MD, FRED-QD, Penn World Table), and data cleaning utilities. See [Data Management](@ref data_page) for theory and examples.

### Data Management

| Function | Description |
|----------|-------------|
| `TimeSeriesData(data; varnames, frequency, tcode)` | Typed time series container with metadata |
| `PanelData` / `CrossSectionData` | Panel and cross-section containers |
| `diagnose(d)` | Scan for NaN, Inf, constant columns |
| `fix(d; method=:listwise)` | Clean data (`:listwise`, `:interpolate`, `:mean`) |
| `validate_for_model(d, :var)` | Check dimensionality for model type |
| `apply_tcode(y, tcode)` | FRED transformation codes 1--7 |
| `inverse_tcode(y, tcode; x_prev)` | Undo FRED transformation |
| `apply_filter(d, :hp; component=:cycle)` | Apply time series filters per-variable |
| `describe_data(d)` | Per-variable summary statistics |
| `xtset(df, group_col, time_col)` | Stata-style panel construction |
| `group_data(pd, g)` | Extract single entity from panel |
| `to_matrix(d)` / `to_vector(d)` | Convert to raw matrix/vector |
| `desc(d)` / `vardesc(d, name)` | Dataset and per-variable descriptions |
| `set_desc!(d, text)` / `set_vardesc!(d, name, text)` | Set descriptions |
| `rename_vars!(d, old => new)` | Rename variables |
| `load_example(:fred_md)` / `load_example(:fred_qd)` / `load_example(:pwt)` / `load_example(:ddcg)` / `load_example(:mpdta)` / `load_example(:wiot)` | Load built-in datasets: FRED-MD, FRED-QD, PWT, DDCG (Acemoglu et al. 2019 → `PanelData`), mpdta (Callaway & Sant'Anna 2021 → `PanelData`), WIOT (WIOD → `IOData`) |

AR, MA, ARMA, and ARIMA model estimation with automatic order selection. See [ARIMA Models](@ref arima_page) for estimation methods, forecasting, and model selection.

### ARIMA Estimation Functions

| Function | Description |
|----------|-------------|
| `estimate_ar(y, p; method=:ols)` | AR(p) via OLS or MLE |
| `estimate_ma(y, q; method=:css_mle)` | MA(q) via CSS, MLE, or CSS-MLE |
| `estimate_arma(y, p, q; method=:css_mle)` | ARMA(p,q) via CSS, MLE, or CSS-MLE |
| `estimate_arima(y, p, d, q; method=:css_mle)` | ARIMA(p,d,q) via differencing + ARMA |
| `estimate_sarima(y, p, d, q, P, D, Q, s)` | Seasonal ARIMA |
| `forecast(model, h; conf_level=0.95)` | Multi-step forecasting with confidence intervals |
| `select_arima_order(y, max_p, max_q)` | Grid search for optimal ARMA order |
| `auto_arima(y)` / `auto_sarima(y)` | Automatic (seasonal) ARIMA order selection |
| `ic_table(y, max_p, max_q)` | Information criteria comparison table |
| `ar_order(m)` / `ma_order(m)` / `diff_order(m)` | Extract fitted orders |

Fractionally integrated ARMA and semiparametric estimation of the long-memory parameter ``d``. See [ARIMA Models](@ref arima_page).

### Long-Memory Models

| Function | Description |
|----------|-------------|
| `estimate_arfima(y, p, q; method=:css)` | ARFIMA(p,d,q) with estimated fractional order |
| `gph_test(y; m)` | Geweke-Porter-Hudak log-periodogram regression |
| `local_whittle(y; m)` | Local Whittle estimator of ``d`` |

Linear Gaussian state-space models estimated by the Kalman filter and smoother. See [State-Space Models](@ref statespace_page) for the `a1`/`P1`/`init_mode` semantics.

### State-Space Models

| Function | Description |
|----------|-------------|
| `local_level(y; init_mode=:kappa)` | Local level (random walk plus noise) model |
| `local_linear_trend(y; init_mode=:kappa)` | Local linear trend model |
| `estimate_statespace(ss, y)` | Run the filter/smoother on a built `StateSpaceModel` |
| `estimate_statespace(build, theta0, y; ...)` | Estimate free parameters by MLE from a builder function |
| `estimate_tvp_reg(y, X; intercept=true)` | Time-varying-parameter regression |

Threshold, smooth-transition, and Markov-switching regression for univariate series. See [Nonlinear Time Series](@ref nonlinear_page).

### Nonlinear Time Series

| Function | Description |
|----------|-------------|
| `estimate_threshold(y, X, q; trim=0.15)` | Hansen (2000) threshold regression |
| `estimate_setar(y, p, d=1; ...)` | Self-exciting threshold autoregression |
| `estimate_star(y, p; type=:auto)` | Smooth-transition AR (`:auto` selects LSTAR vs ESTAR) |
| `estimate_ms(y; ...)` / `estimate_ms_ar(y, p; k_regimes=2)` | Markov-switching regression / autoregression |
| `hansen_linearity_test(y, X, q; reps=1000)` | Hansen bootstrap test of linearity vs threshold |
| `star_linearity_test(y, p; s, d=1)` | Luukkonen-Saikkonen-Terasvirta LM linearity test |

Trend-cycle decomposition via HP, Hamilton, Beveridge-Nelson, Baxter-King, and boosted HP filters. See [Time Series Filters](@ref filters_page) for theory and comparisons.

### Time Series Filters

| Function | Description |
|----------|-------------|
| `hp_filter(y; lambda=1600.0)` | Hodrick-Prescott trend-cycle decomposition |
| `hamilton_filter(y; h=8, p=4)` | Hamilton (2018) regression filter |
| `beveridge_nelson(y; p=:auto, q=:auto)` | Beveridge-Nelson permanent/transitory decomposition |
| `baxter_king(y; pl=6, pu=32, K=12)` | Baxter-King band-pass filter |
| `boosted_hp(y; stopping=:BIC, lambda=1600.0)` | Boosted HP filter (Phillips & Shi 2021) |
| `x13_filter(y; frequency=12, method=:seats)` | X-13ARIMA-SEATS seasonal adjustment (see [X-13ARIMA-SEATS](@ref x13_page)) |
| `trend(result)` | Extract trend component from filter result |
| `cycle(result)` | Extract cyclical component from filter result |

VAR, VECM, BVAR, Local Projections, Factor Models, and Panel VAR estimation. See [VAR](@ref var_page), [VECM](@ref vecm_page), [BVAR](@ref bvar_page), [LP](@ref lp_page), [Factor Models](@ref factor_page), and [Panel VAR](@ref pvar_page) for theory and examples.

### Multivariate Estimation Functions

| Function | Description |
|----------|-------------|
| `estimate_var(Y, p)` | Estimate VAR(p) via OLS |
| `estimate_bvar(Y, p; ...)` | Estimate Bayesian VAR (conjugate NIW) |
| `estimate_lp(Y, shock_var, H; ...)` | Standard Local Projection |
| `estimate_lp_iv(Y, shock_var, Z, H; ...)` | LP with instrumental variables |
| `estimate_smooth_lp(Y, shock_var, H; ...)` | Smooth LP with B-splines |
| `estimate_state_lp(Y, shock_var, state_var, H; ...)` | State-dependent LP |
| `estimate_propensity_lp(Y, treatment, covariates, H; ...)` | LP with propensity scores |
| `doubly_robust_lp(Y, treatment, covariates, H; ...)` | Doubly robust LP estimator |
| `estimate_factors(X, r; ...)` | Static factor model via PCA |
| `estimate_dynamic_factors(X, r, p; ...)` | Dynamic factor model |
| `estimate_gdfm(X, q; ...)` | Generalized dynamic factor model |
| `estimate_pvar(pd, p; ...)` | Panel VAR via GMM (FD or System) |
| `estimate_pvar_feols(pd, p; ...)` | Panel VAR via Fixed-Effects OLS |
| `estimate_gmm(moment_fn, theta0, data; ...)` | GMM estimation |
| `estimate_lp_multi(Y, shock_vars, H; ...)` | LP for several shocks jointly |
| `estimate_lp_cholesky(Y, H; lags)` | Recursively identified LP |
| `estimate_lp_gmm(Y, shock_var, H; weighting)` | GMM-weighted LP |
| `structural_lp(Y, H; method=:cholesky, ...)` | Structural LP with multi-shock IRFs |
| `estimate_vecm(Y, p; rank=:auto, ...)` | Estimate VECM via Johansen MLE or Engle-Granger |
| `to_var(vecm)` | Convert VECM to VAR in levels |
| `conditional_forecast(model, conditions, h; ...)` | Forecast subject to conditioning paths |
| `forecast_condition(variable, horizon, value; sd)` | Build one conditioning restriction |
| `select_vecm_rank(Y, p; ...)` | Select cointegrating rank |
| `granger_causality_vecm(vecm, cause, effect)` | VECM Granger causality test |
| `forecast(vecm, h; ci_method=:none, ...)` | VECM forecast preserving cointegration |

Single-equation dynamic models in levels: ARDL with the Pesaran-Shin-Smith bounds test, its asymmetric NARDL extension, and the panel pooled mean group estimator. See [ARDL](@ref ardl_page).

### ARDL, NARDL, and PMG

| Function | Description |
|----------|-------------|
| `estimate_ardl(y, X; p, q, ic=:aic)` | ARDL(p,q) with optional automatic lag selection |
| `bounds_test(m; case=3, level=0.05)` | Pesaran-Shin-Smith (2001) bounds test for a level relationship |
| `estimate_nardl(y, X; asymmetric)` | Nonlinear (asymmetric) ARDL |
| `dynamic_multipliers(m, H; bootstrap=false)` | NARDL cumulative dynamic multipliers |
| `estimate_pmg(pd, y, xs...; method=:pmg)` | Pooled mean group / mean group / dynamic FE panel ARDL |

Single-equation cointegrating-vector estimators that correct the OLS second-order bias. See [Cointegrating Regression](@ref cointreg_page).

### Cointegrating Regression

| Function | Description |
|----------|-------------|
| `estimate_cointreg(y, X; method=:fmols)` | FMOLS, CCR, or DOLS cointegrating regression |
| `estimate_xtcointreg(pd, y, xs...; pooling)` | Panel FMOLS/CCR/DOLS (grouped or pooled) |

Seemingly unrelated regressions and three-stage least squares for systems of linear equations. See [VAR](@ref var_page) for the multivariate context.

### Systems of Equations

| Function | Description |
|----------|-------------|
| `estimate_sur(eqs; iterate=false)` | Zellner SUR (feasible GLS, optionally iterated) |
| `estimate_3sls(eqs, Z; instruments)` | Three-stage least squares for simultaneous systems |

Bayesian VARs whose coefficients or volatilities move over time, and VARs mixing monthly and quarterly observables. See [BVAR](@ref bvar_page).

### Time-Varying and Mixed-Frequency VAR

| Function | Description |
|----------|-------------|
| `estimate_tvpvar(Y, p; tvp=true, sv=true)` | Primiceri TVP-VAR with stochastic volatility |
| `volatility_path(post)` | Posterior stochastic-volatility paths |
| `estimate_mfvar(data, p; freq_ratio=3)` | Mixed-frequency VAR with latent monthly states |
| `latent_path(post)` | Posterior paths of the latent high-frequency series |
| `optimize_hyperparameters_glp(Y, p; ...)` | Giannone-Lenza-Primiceri hierarchical prior selection |

Impulse response functions, forecast error variance decomposition, historical decomposition, six restriction-based identification schemes, proxy / AB-model / max-share / SVEC, Giacomini–Kitagawa robust Bayes, and 15 statistical identification methods (5 ICA, 4 non-Gaussian ML plus dispatcher, GMM, 4 heteroskedasticity). The `method=` keyword accepts twenty-five symbols. See [Innovation Accounting](@ref innovation_accounting_page), [Structural Identification](@ref structural_identification_page), and [Statistical Identification](@ref nongaussian_page).

### Structural Analysis Functions

| Function | Description |
|----------|-------------|
| `irf(model, H; ...)` | Compute impulse response functions |
| `irf(model, H; bootstrap=:wild)` | Wild / moving-block residual bootstrap bands |
| `irf(model, H; bias_correct=true)` | Kilian (1998) bias-corrected bootstrap bands |
| `fevd(model, H; ...)` | Forecast error variance decomposition |
| `generalized_fevd(model, H)` | Pesaran-Shin (1998) generalized FEVD — order-invariant, no orthogonalization |
| `identify_cholesky(model)` | Cholesky identification |
| `identify_sign(model; ...)` | Sign restriction identification |
| `identify_long_run(model)` | Blanchard-Quah identification |
| `identify_narrative(model; ...)` | Narrative sign restrictions |
| `identify_arias(model, restrictions, H; ...)` | Arias et al. (2018) sign + zero restrictions |
| `identify_uhlig(model, restrictions, H; ...)` | Mountford-Uhlig (2009) penalty function sign + zero restrictions |
| `identify_fastica(model; ...)` | FastICA SVAR identification |
| `identify_jade(model; ...)` | JADE SVAR identification |
| `identify_sobi(model; ...)` | SOBI SVAR identification |
| `identify_dcov(model; ...)` | Distance covariance SVAR identification |
| `identify_hsic(model; ...)` | HSIC SVAR identification |
| `identify_student_t(model; ...)` | Student-t ML SVAR identification |
| `identify_mixture_normal(model; ...)` | Mixture-normal ML SVAR identification |
| `identify_pml(model; ...)` | Pseudo-ML SVAR identification |
| `identify_skew_normal(model; ...)` | Skew-normal ML SVAR identification |
| `identify_nongaussian_ml(model; ...)` | Unified non-Gaussian ML dispatcher |
| `identify_gmm_moments(model; ...)` | Coskewness / cokurtosis GMM SVAR identification |
| `identify_proxy(model, Z; ...)` | External-instrument (proxy) SVAR identification |
| `estimate_svar(model, pattern; ...)` | Amisano–Giannini AB-model ML |
| `identify_max_share(model; target=...)` | Max-share / news-shock identification |
| `identify_svec(vecm; ...)` | Structural VECM (KPSW / Gonzalo–Ng) |
| `identify_robust_bayes(post, r, H; ...)` | Giacomini–Kitagawa robust Bayes |
| `identify_markov_switching(model; ...)` | Markov-switching SVAR identification |
| `identify_garch(model; ...)` | GARCH SVAR identification |
| `identify_smooth_transition(model, s; ...)` | Smooth-transition SVAR identification |
| `identify_external_volatility(model, regime)` | External volatility SVAR identification |
| `pvar_oirf(model, H)` | Panel VAR orthogonalized IRF (Cholesky) |
| `pvar_girf(model, H)` | Panel VAR generalized IRF (Pesaran & Shin 1998) |
| `pvar_fevd(model, H)` | Panel VAR forecast error variance decomposition |
| `pvar_stability(model)` | Panel VAR eigenvalue stability check |
| `pvar_bootstrap_irf(model, H; ...)` | Panel VAR bootstrap IRF confidence intervals |
| `lp_fevd(slp, H; method=:r2, ...)` | LP-FEVD (Gorodnichenko & Lee 2019) |
| `cumulative_irf(lp_irfs)` | Cumulative IRF from LP impulse response |
| `historical_decomposition(slp)` | Historical decomposition from structural LP |
| `contribution(hd, shock)` / `total_shock_contribution(hd)` | Per-shock and total contributions from a decomposition |
| `verify_decomposition(hd)` | Check that the shock contributions reproduce the data |

Direct multi-step forecasting from Local Projection models. See [Local Projections](@ref lp_page) for estimation details.

### LP Forecasting Functions

| Function | Description |
|----------|-------------|
| `forecast(lp, shock_path; ...)` | Direct multi-step LP forecast |
| `forecast(slp, shock_idx, shock_path; ...)` | Structural LP conditional forecast |

Augmented Dickey-Fuller, KPSS, Phillips-Perron, Zivot-Andrews, Ng-Perron, and Johansen cointegration tests. See [Hypothesis Tests](@ref tests_page) for interpretation and examples.

### Unit Root Test Functions

| Function | Description |
|----------|-------------|
| `adf_test(y; ...)` | Augmented Dickey-Fuller unit root test |
| `kpss_test(y; ...)` | KPSS stationarity test |
| `pp_test(y; ...)` | Phillips-Perron unit root test |
| `za_test(y; ...)` | Zivot-Andrews structural break test |
| `ngperron_test(y; ...)` | Ng-Perron unit root tests (MZα, MZt, MSB, MPT) |
| `johansen_test(Y, p; ...)` | Johansen cointegration test |
| `is_stationary(model)` | Check VAR model stationarity |
| `unit_root_summary(y; ...)` | Run multiple tests with summary |
| `test_all_variables(Y; ...)` | Apply test to all columns |

Likelihood ratio (LR) and Lagrange multiplier (LM/score) tests for comparing nested models across ARIMA, VAR, and GARCH families. See [Hypothesis Tests](@ref tests_page).

### Model Comparison Tests

| Function | Description |
|----------|-------------|
| `lr_test(m1, m2)` | Likelihood ratio test for nested models |
| `lm_test(m1, m2)` | Lagrange multiplier (score) test for nested models |

Pairwise and block Wald tests for Granger causality in VAR models. See [Hypothesis Tests](@ref tests_page) for details.

### Granger Causality Tests

| Function | Description |
|----------|-------------|
| `granger_test(model, cause, effect)` | Pairwise or block Granger causality test |
| `granger_test_all(model)` | All-pairs pairwise Granger causality matrix |

Convenience functions for extracting impulse responses from fitted LP models. See [Local Projections](@ref lp_page).

### LP IRF Extraction

| Function | Description |
|----------|-------------|
| `lp_irf(model; ...)` | Extract IRF from LPModel |
| `lp_irf(model; ci_type=:bootstrap)` | Fixed-design wild/block bootstrap bands for LP |
| `lp_iv_irf(model; ...)` | Extract IRF from LPIVModel |
| `smooth_lp_irf(model; ...)` | Extract smoothed IRF |
| `state_irf(model; ...)` | Extract state-dependent IRFs |
| `propensity_irf(model; ...)` | Extract ATE impulse response |

Static PCA, Dynamic Factor, and Generalized Dynamic Factor model estimation, forecasting, and selection criteria. See [Factor Models](@ref factor_page).

### Factor Model Functions

| Function | Description |
|----------|-------------|
| `estimate_factors(X, r; ...)` | Estimate r-factor model |
| `estimate_dynamic_factors(X, r, p; ...)` | Dynamic factor model |
| `estimate_gdfm(X, q; ...)` | Generalized dynamic factor model |
| `forecast(fm, h; p=1, ci_method=:none)` | Static FM forecast (fits VAR(p) on factors) |
| `forecast(dfm, h; ci_method=:none)` | DFM forecast (`:none/:theoretical/:bootstrap/:simulation`) |
| `forecast(gdfm, h; ci_method=:none)` | GDFM forecast (`method=:ar` or `:one_sided`/`:spectral`) |
| `ic_criteria(X, r_max)` | Bai-Ng information criteria |
| `ic_criteria_dynamic(X, max_r, max_p)` | DFM factor/lag selection |
| `ic_criteria_gdfm(X, max_q)` | GDFM eigenvalue-ratio / 90% variance heuristic |
| `hallin_liska(X, q_max)` | Hallin–Liška (2007) IC with second stability plateau |
| `bai_ng_q(X, r)` | Bai–Ng (2007) residual-covariance rank statistics |
| `amengual_watson_q(X, r, p)` | Amengual–Watson (2007) Bai–Ng IC on lagged-factor residuals |
| `scree_plot_data(model)` | Data for scree plot |
| `is_stationary(dfm)` | Check DFM factor VAR stationarity |
| `common_variance_share(gdfm)` | GDFM common variance share per variable |
| `predict(fm)` | Fitted values (all factor model types) |
| `residuals(fm)` | Idiosyncratic residuals (all factor model types) |
| `r2(fm)` | Per-variable ``R^2`` (all factor model types) |
| `nobs(fm)` | Number of observations |
| `dof(fm)` | Degrees of freedom |
| `loglikelihood(dfm)` | Log-likelihood (DFM only) |
| `aic(dfm)` / `bic(dfm)` | Information criteria (DFM only) |

Bayesian prior optimization, instrument strength tests, and Panel VAR specification tests. See [BVAR](@ref bvar_page) and [Panel VAR](@ref pvar_page).

### Diagnostic Functions

| Function | Description |
|----------|-------------|
| `optimize_hyperparameters(Y, p; ...)` | Optimize Minnesota prior (τ only) |
| `optimize_hyperparameters_full(Y, p; ...)` | Joint optimization over (τ, λ, μ) (BGR 2010) |
| `posterior_mean_model(post; ...)` | VARModel from posterior mean |
| `posterior_median_model(post; ...)` | VARModel from posterior median |
| `weak_instrument_test(model; ...)` | Test for weak instruments |
| `montiel_olea_pflueger_f(model)` | Montiel Olea-Pflueger effective first-stage F |
| `lp_iv_ar_band(model; level=0.95)` | Anderson-Rubin weak-instrument-robust LP-IV band |
| `compare_var_lp(Y, H; lags)` | Compare VAR and LP impulse responses |
| `compare_smooth_lp(Y, shock_var, H; lambda)` | Compare smooth-LP fits across ``\lambda`` |
| `cross_validate_lambda(Y, shock_var, H; lambda_grid)` | Cross-validate the smooth-LP penalty |
| `sargan_test(model, h)` | Overidentification test |
| `test_regime_difference(model; ...)` | Test regime differences |
| `propensity_diagnostics(model)` | Propensity score diagnostics |
| `pvar_hansen_j(model)` | Hansen J-test for Panel VAR |
| `pvar_mmsc(model)` | Andrews-Lu MMSC for Panel VAR |
| `pvar_lag_selection(pd, max_p; ...)` | Panel VAR lag order selection |
| `j_test(model)` | Hansen J-test for GMM |
| `gmm_summary(model)` | Summary statistics for GMM |

Multivariate normality tests for VAR residuals. See [Statistical Identification](@ref nongaussian_page) for using these as pre-tests for ICA/ML identification.

### Normality Test Functions

| Function | Description |
|----------|-------------|
| `jarque_bera_test(model; method=:multivariate)` | Multivariate Jarque-Bera test |
| `mardia_test(model; type=:both)` | Mardia skewness/kurtosis tests |
| `doornik_hansen_test(model)` | Doornik-Hansen omnibus test |
| `henze_zirkler_test(model)` | Henze-Zirkler characteristic function test |
| `normality_test_suite(model)` | Run all normality tests |

Diagnostic tests for non-Gaussian SVAR identification validity. See [Statistical Identification](@ref nongaussian_page).

### Identifiability Test Functions

| Function | Description |
|----------|-------------|
| `test_shock_gaussianity(result)` | Test non-Gaussianity of recovered shocks |
| `test_gaussian_vs_nongaussian(model; ...)` | LR test: Gaussian vs non-Gaussian |
| `test_shock_independence(result; ...)` | Test independence of recovered shocks |
| `test_label_stability(model; ...)` | Column-label match fraction (no p-value) |
| `test_lambda_distinct(result; ...)` | Wald test that relative variances differ |
| `test_gaussian_shock_count(result)` | Holm-adjusted count of Gaussian shocks |
| `test_identification_strength(model; ...)` | Deprecated wrapper; see `test_label_stability` / `test_lambda_distinct` / `test_gaussian_shock_count` |
| `test_overidentification(model, result; ...)` | Nested LR / ``RVR'`` Wald overidentification test |

ARCH, GARCH, EGARCH, GJR-GARCH, and Stochastic Volatility estimation, forecasting, and diagnostics. See [Volatility Models](@ref volatility_page).

### Volatility Model Functions

| Function | Description |
|----------|-------------|
| `estimate_arch(y, q)` | ARCH(q) via MLE |
| `estimate_garch(y, p, q)` | GARCH(p,q) via MLE |
| `estimate_egarch(y, p, q)` | EGARCH(p,q) via MLE |
| `estimate_gjr_garch(y, p, q)` | GJR-GARCH(p,q) via MLE |
| `estimate_igarch(y; ...)` | Integrated GARCH (unit persistence imposed) |
| `estimate_cgarch(y; ...)` | Component GARCH (permanent + transitory variance) |
| `estimate_aparch(y; ...)` | Asymmetric power ARCH |
| `estimate_figarch(r; p, q, truncation=1000)` | Fractionally integrated GARCH |
| `estimate_fiegarch(r; p, q, truncation=1000)` | Fractionally integrated EGARCH |
| `estimate_garch_midas(r, x_lf; K, m_freq)` | GARCH-MIDAS long/short-run components |
| `estimate_sv(y; variant, ...)` | Stochastic Volatility via KSC Gibbs |
| `forecast(vol_model, h)` | Volatility forecast with simulation CIs |
| `arch_lm_test(y_or_model, q)` | ARCH-LM test for conditional heteroskedasticity |
| `ljung_box_squared(z_or_model, K)` | Ljung-Box test on squared residuals |
| `sign_bias_test(z_or_model)` | Engle-Ng sign and size bias tests |
| `nyblom_test(m)` | Nyblom parameter-constancy test |
| `component_variances(m)` | Permanent/transitory split of a `CGARCHModel` |
| `news_impact_curve(model)` | News impact curve (GARCH family) |
| `persistence(model)` | Persistence measure |
| `halflife(model)` | Volatility half-life |
| `unconditional_variance(model)` | Unconditional variance |
| `arch_order(model)` | ARCH order ``q`` |
| `garch_order(model)` | GARCH order ``p`` |
| `predict(m)` | Conditional variance series ``\hat{\sigma}^2_t`` |
| `residuals(m)` | Raw residuals (ARCH/GARCH) or standardized (SV) |
| `coef(m)` | Coefficient vector |
| `nobs(m)` | Number of observations |
| `loglikelihood(m)` | Maximized log-likelihood (ARCH/GARCH) |
| `aic(m)` / `bic(m)` | Information criteria (ARCH/GARCH) |
| `dof(m)` | Number of estimated parameters |

Constant-correlation, dynamic-correlation, and BEKK models for the conditional covariance of a vector of returns. See [Volatility Models](@ref volatility_page).

### Multivariate GARCH

| Function | Description |
|----------|-------------|
| `estimate_ccc(Y; p=1, q=1)` | Bollerslev constant conditional correlation |
| `estimate_dcc(Y; p=1, q=1)` | Engle dynamic conditional correlation |
| `estimate_bekk(Y; kind=:scalar)` | BEKK (`:scalar` or `:diagonal`) |
| `forecast(mgarch, h)` | Conditional covariance forecast |

Regressions that mix a low-frequency dependent variable with high-frequency regressors through a parametric lag polynomial. See [MIDAS Regression](@ref midas_page).

### MIDAS Regression

| Function | Description |
|----------|-------------|
| `estimate_midas(y_lf, X_hf; m, K, weights=:expalmon)` | MIDAS regression with exponential-Almon or beta weights |
| `midas_weights(m)` | Fitted high-frequency weighting curve |

Accuracy metrics, equal-predictive-ability tests, and forecast combination. See [Forecast Evaluation](@ref forecast_evaluation_page).

### Forecast Evaluation

| Function | Description |
|----------|-------------|
| `forecast_evaluate(actual, fc; ...)` | RMSE/MAE/MAPE/MASE/Theil accuracy table |
| `diebold_mariano(e1, e2; h=1, loss=:se)` | Diebold-Mariano equal-predictive-ability test (HLN correction on by default) |
| `clark_west(e_small, e_big, f_adj; h)` | Clark-West test for nested models |
| `mincer_zarnowitz(actual, fc; lags)` | Mincer-Zarnowitz unbiasedness regression |
| `forecast_encompassing(actual, fc1, fc2)` | Forecast encompassing test |
| `combine_forecasts(F, actual; method=:equal)` | Combine competing forecasts (equal, inverse-MSE, OLS weights) |

Mixed-frequency nowcasting via DFM, BVAR, and bridge equations with news decomposition. See [Nowcasting](@ref nowcast_page) for theory and examples.

### Nowcasting Functions

| Function | Description |
|----------|-------------|
| `nowcast_dfm(Y, nM, nQ; r=2, p=1, ...)` | DFM nowcasting via EM + Kalman smoother (Banbura & Modugno 2014) |
| `nowcast_bvar(Y, nM, nQ; lags=5, ...)` | Large BVAR nowcasting with GLP priors (Cimadomo et al. 2022) |
| `nowcast_bridge(Y, nM, nQ; lagM=1, ...)` | Bridge equation combination nowcasting (Banbura et al. 2023) |
| `nowcast(model)` | Extract current-quarter nowcast and next-quarter forecast |
| `forecast(dfm_or_bvar, h; ...)` | Multi-step ahead forecast from nowcasting model |
| `nowcast_news(X_new, X_old, dfm, t; ...)` | News decomposition: attribute revision to data releases |
| `balance_panel(d; r=2, method=:dfm)` | Fill NaN in TimeSeriesData/PanelData via DFM |

Publication-quality tables, display backend switching, and bibliographic references. See individual section pages for usage examples.

### Display and Output Functions

| Function | Description |
|----------|-------------|
| `set_display_backend(sym)` | Switch output format (`:text`/`:latex`/`:html`) |
| `get_display_backend()` | Current display backend |
| `report(result)` | Print comprehensive summary |
| `table(result, ...)` | Extract results as matrix |
| `print_table([io], result, ...)` | Print formatted table |
| `long_table(result)` | Tidy/long `DataFrame` of an array-valued result |
| `write_csv(result, path)` | Export a result (coefficient table or `long_table`) to CSV |
| `set_log_level(level)` | Set the global minimum log level |
| `with_min_level(f, level)` | Run `f()` with a scoped minimum log level |
| `capture_manifest(; seed)` | Capture a reproducibility manifest (seed, threads, versions, git) |
| `reproduce(result)` | Re-run a randomized result from its seed; returns a `ReproReport` |
| `save_model(model, path)` | Persist a fitted model, data container, or named bundle to a versioned container |
| `load_model(path)` | Reconstruct a saved model (or a `Dict` of objects for a bundle) |
| `model_info(path)` | Read the file header (`note`, versions, type tags) without reconstructing the payload |
| `refs(model; format=...)` | Bibliographic references |
| `refs(io, :method; format=...)` | References by method name |

HAC (Newey-West), heteroskedasticity-robust (White), and panel-robust (Driscoll-Kraay) covariance estimators.

### Covariance Functions

| Function | Description |
|----------|-------------|
| `newey_west(X, residuals; ...)` | Newey-West HAC estimator |
| `white_vcov(X, residuals; ...)` | White heteroskedasticity-robust |
| `driscoll_kraay(X, residuals; ...)` | Driscoll-Kraay panel-robust |
| `long_run_variance(x; ...)` | Long-run variance estimate |
| `long_run_covariance(X; ...)` | Long-run covariance matrix |
| `optimal_bandwidth_nw(residuals)` | Automatic bandwidth selection |

Low-level matrix construction and numerical utilities used internally.

### Utility Functions

| Function | Description |
|----------|-------------|
| `construct_var_matrices(Y, p)` | Build VAR design matrices |
| `companion_matrix(B, n, p)` | VAR companion form |
| `robust_inv(A)` | Robust matrix inverse |
| `safe_cholesky(A; ...)` | Stable Cholesky decomposition |

Specify, solve, simulate, and estimate Dynamic Stochastic General Equilibrium models. See [DSGE Models](@ref dsge_page) for the full guide.

### DSGE Specification and Solution

| Function | Description |
|----------|-------------|
| `@dsge begin ... end` | Parse DSGE model specification |
| `compute_steady_state(spec)` | Compute deterministic steady state |
| `linearize(spec)` | Linearize around steady state (Sims canonical form) |
| `solve(spec; method=:gensys)` | Solve rational expectations model |
| `gensys(Γ₀, Γ₁, C, Ψ, Π)` | Sims (2002) QZ decomposition solver |
| `blanchard_kahn(ld, spec)` | Blanchard-Kahn (1980) eigenvalue counting |
| `klein(ld, spec)` | Klein (2000) generalized Schur solver |
| `solve(spec; sparse=true)` | Matrix-free Newton route for large sparse models (`:auto` by default) |
| `perturbation_solver(spec; order=2)` | Higher-order perturbation solver |
| `collocation_solver(spec; ...)` | Chebyshev collocation projection (isotropic/anisotropic Smolyak, `adaptive=true` refinement) |
| `pfi_solver(spec; ...)` | Policy function iteration |
| `vfi_solver(spec; ...)` | Value function iteration |
| `is_determined(sol)` | Check existence and uniqueness (Sims 2002 rank test) |
| `is_stable(sol)` | Check stability of solution |
| `determinacy_region(spec; params, grids)` | Determinacy verdict over a 1- or 2-parameter grid |
| `determinacy_boundary(m)` | Grid location of the boundary in a 1-parameter sweep |

### DSGE Simulation and Analysis

| Function | Description |
|----------|-------------|
| `simulate(sol, T)` | Stochastic simulation |
| `irf(sol, H)` | Analytical impulse responses |
| `fevd(sol, H)` | Forecast error variance decomposition |
| `historical_decomposition(sol, data, obs)` | DSGE historical decomposition |
| `solve_lyapunov(G1, impact)` | Unconditional covariance (Lyapunov equation) |
| `analytical_moments(sol; lags)` | Analytical variance and autocovariances |
| `pruned_state_space(sol)` | Pruned state-space object (shared recursion + control map) |
| `perfect_foresight(spec; T_periods, shock_path)` | Deterministic transition path |

### DSGE Estimation

| Function | Description |
|----------|-------------|
| `estimate_dsge(spec, data, params; method)` | GMM estimation (IRF matching, Euler, SMM, analytical) |
| `estimate_dsge_bayes(spec, data, θ0; ...)` | Bayesian estimation (SMC/SMC²/MH) |

Heterogeneous-agent (Reiter/SSJ/Krusell-Smith), continuous-time (HJB/KFE), and OLG solvers. See [Heterogeneous Agents](@ref dsge_ha), [Continuous Time](@ref dsge_continuous), and [Overlapping Generations](@ref dsge_olg).

### Heterogeneous-Agent DSGE

| Function | Description |
|----------|-------------|
| `load_ha_example(:krusell_smith)` | Built-in HA-DSGE model specs (see [Heterogeneous Agents](@ref dsge_ha)) |
| `compute_steady_state(spec::ModelSpec)` | HA stationary equilibrium (EGM + distribution + market clearing) |
| `solve(spec::ModelSpec; method=:ssj)` | HA-DSGE solution (SSJ/Reiter/Krusell-Smith via `combine_blocks`) |
| `rouwenhorst(ρ, σ, n)` / `tauchen(ρ, σ, n)` | Income process discretization (`σ` = **innovation** sd; pass `sigma_is=:unconditional` for sd(y)) |
| `distribution_irf(sol, H)` / `inequality_irf(sol, H)` | Distribution dynamics / Gini response |
| `simulate_panel(ss; N_agents, T_periods)` | Simulate individual-level panel from HA steady state |
| `den_haan_test(ks_sol)` / `den_haan_test(ha_sol)` | Den Haan (2010) accuracy for a Krusell-Smith PLM or an `:ssj`/`:reiter` linearization |
| `ha_grid_diagnostics(ss)` | Asset-grid adequacy: ceiling mass, `∫a′dμ − ∫a dμ` clearing residual |
| `adapt_ha_grid(spec, ss)` / `adaptive_asset_grid(nodes, mass)` | Re-place asset nodes by stationary-density curvature (de Boor equidistribution) |
| `LaborSupply(; kind=:ghh, psi, frisch)` | Endogenous labor supply (GHH / separable) for an `IndividualProblem` |
| `labor_supply(ls, w*e[, u′(c)])` / `labor_policy(ip, …)` | Intratemporal hours FOC / hours policy `n(a,e)` |
| `HetBlock(spec, ss)` / `SimpleBlock(f; ...)` | Sequence-space blocks (household / equation) |
| `combine_blocks(blocks...)` | Assemble blocks into a DAG (topological sort) |
| `ssj_jacobian(model; unknowns, targets, shocks)` | General-equilibrium sequence-space Jacobian `H_U`, `H_Z` |
| `ssj_irf(gej, dZ; order=2)` | First- and second-order sequence-space impulse responses |
| `dcegm_solve(prob)` | DCEGM discrete-continuous choice with upper envelope (Iskhakov et al. 2017) |
| `ct_two_asset_solve(m)` | Continuous-time two-asset household block (KMV upwind; `cost=:quadratic`/`:kinked`) |
| `ct_two_asset_ge(m)` | Two-asset stationary general equilibrium (capital + bond market clearing) |
| `ct_two_asset_mit(m, ge, Z_path)` | Two-asset MIT-shock transition (backward HJB / forward KFE shooting) |
| `hand_to_mouth(sol)` / `ceiling_mass(sol)` | Poor vs wealthy hand-to-mouth shares; mass on the grid ceilings |
| `ct_two_asset_stationarity(m)` | Whether the calibration can bound illiquid wealth (see #509) |
| `dcegm_retirement_model(; ...)` | Canonical work/retire problem |
| `dcegm_threshold(sol, t, d_prev, j; ...)` | Cash-on-hand at which the discrete choice switches |
| `dcegm_simulate(sol, grid)` | Young histogram respecting the discrete choice |
| `fit_parametric_density(moments; bounds)` | Winberry (2018) exponential family matching centered moments |
| `parametric_density(pd, a)` / `parametric_moments(pd, nodes, weights)` | Evaluate a fitted density / invert `λ` back to moments |
| `fit_winberry(ss; n_moments=3)` | Fit the parametric family to a Young histogram (one density per income state) |
| `winberry_moments(dist, grid)` / `winberry_histogram(fam, grid)` | Histogram → conditional moments / family → grid rendering |
| `winberry_quadrature(grid; n_quad=4)` | Composite Gauss-Legendre rule on the asset-grid intervals |
| `LifeCycleOLG(; J, J_retire, ...)` | True life-cycle OLG (age-dependent EGM) |
| `lifecycle_steady_state(m)` | Stationary equilibrium: backward age sweep + age-extended histogram |
| `lifecycle_income(ρ, σ, n)` / `lifecycle_survival(J)` | Unit-mean level income process / Gompertz-Makeham mortality |

### Occasionally Binding Constraints (OccBin)

| Function | Description |
|----------|-------------|
| `parse_constraint(expr, spec)` | Parse constraint expression |
| `occbin_solve(spec, constraint; ...)` | Piecewise-linear OccBin solution (1 or 2 constraints) |
| `occbin_irf(spec, constraint, shock_idx, H; ...)` | OccBin impulse responses |

### DSGE Smoothers and Diagnostics

| Function | Description |
|----------|-------------|
| `dsge_smoother(ss, data)` | RTS Kalman smoother for linear DSGE |
| `dsge_particle_smoother(nss, data)` | FFBSi particle smoother for nonlinear DSGE |
| `evaluate_policy(sol, grid)` | Evaluate policy function on grid |
| `max_euler_error(sol, grid)` | Maximum Euler equation error |

OLS, WLS, IV/2SLS, logit, probit, ordered, and multinomial estimation for cross-sectional data. See [Linear Regression](@ref regression_page) and [Binary Choice](@ref binary_choice_page) for theory and examples.

### Cross-Sectional Models

| Function | Description |
|----------|-------------|
| `estimate_reg(y, X; ...)` | OLS/WLS regression (HC0–HC3, cluster-robust, Conley spatial SEs) |
| `conley_se(m; coords, cutoff)` | Conley (1999) spatial HAC SEs (also `cov_type=:conley`) |
| `estimate_qreg(y, X, tau)` | Quantile regression (Koenker-Bassett; `:iid`/`:robust`/`:boot` SEs) |
| `estimate_rdd(y, running; cutoff)` | Sharp/fuzzy RDD with CCT robust bias-corrected inference |
| `estimate_iv(y, X, Z; ...)` | IV/2SLS estimation |
| `estimate_logit(y, X)` | Logit MLE via IRLS |
| `estimate_probit(y, X)` | Probit MLE via IRLS |
| `estimate_ologit(y, X)` | Ordered logit MLE |
| `estimate_oprobit(y, X)` | Ordered probit MLE |
| `estimate_mlogit(y, X)` | Multinomial logit MLE |
| `estimate_poisson(y, X; exposure)` | Poisson regression, QMLE sandwich SEs by default |
| `estimate_nbreg(y, X)` | Negative-Binomial-2 regression (`Var = mu + alpha*mu^2`) |
| `dispersion_test(m::PoissonModel)` | Cameron-Trivedi (1990) overdispersion test (NB1 & NB2 forms) |
| `incidence_rate_ratio(m)` | Incidence-rate ratios `exp(beta)` for count models |
| `marginal_effects(m; ...)` | AME/MEM/MER with delta-method SEs |
| `odds_ratio(m)` | Odds ratios for logit models |
| `classification_table(m)` | Classification accuracy table |
| `vif(m)` | Variance inflation factors |
| `white_test(m)` | White heteroskedasticity test |
| `breusch_pagan_test(m::RegModel)` | Breusch-Pagan/Koenker heteroskedasticity test |
| `glejser_test(m)` | Glejser heteroskedasticity test |
| `harvey_test(m)` | Harvey multiplicative heteroskedasticity test |
| `breusch_godfrey_test(m; lags)` | Breusch-Godfrey serial-correlation LM test |
| `reset_test(m; powers)` | Ramsey RESET functional-form test |
| `chow_test(m, break_index; type)` | Chow structural-break test at a known break |
| `cusum_test(m; level=0.05)` / `cusumsq_test(m; level=0.05)` | CUSUM and CUSUM-of-squares stability tests |
| `anderson_rubin_test(m, beta0)` / `anderson_rubin_ci(m; level)` | Weak-instrument-robust IV inference |
| `residuals(m; kind)` | Ordered/multinomial residual matrix (`:response`/`:pearson`/`:deviance`) |
| `generalized_residuals(m)` | Ordered-model score residual (Chesher-Irish), length `n` |
| `brant_test(m)` | Brant test for parallel regression |
| `hausman_iia(m)` | Hausman test for IIA assumption |

Penalized, robust, and limited-dependent-variable estimators for cross-sectional data. See [Linear Regression](@ref regression_page).

### Penalized, Robust, and Limited-Dependent Regression

| Function | Description |
|----------|-------------|
| `estimate_lasso(y, X; ...)` / `estimate_ridge(y, X; ...)` | L1 and L2 penalized regression |
| `estimate_elastic_net(y, X; alpha=1.0, lambda=:cv)` | Elastic net; `alpha=1` is LASSO, `alpha=0` is ridge |
| `estimate_robust(y, X; psi=:huber, method=:m)` | M / MM robust regression |
| `estimate_tobit(y, X; lower=0.0, upper=Inf)` | Censored (Tobit) regression |
| `estimate_truncreg(y, X; lower, upper)` | Truncated regression |
| `estimate_heckman(y, X, d, Z; method=:twostep)` | Heckman selection model |

FE, RE, FD, Between, CRE, Arellano-Bond, and Blundell-Bond panel estimators. See [Panel Regression](@ref panel_reg_page) for theory and examples.

### Panel Regression

| Function | Description |
|----------|-------------|
| `estimate_xtreg(pd, :y, :x1, :x2; ...)` | Panel FE/RE/FD/Between/CRE/AB/BB |
| `estimate_xtreg(pd, :y, :x; absorb=[...])` | High-dimensional FE (reghdfe-style alternating projections) |
| `absorb_fe(y, X, fe_groups; ...)` | Absorb HDFE dimensions from raw arrays |
| `estimate_xtiv(pd, :y, :x; ...)` | Panel IV (FE-IV/RE-IV/FD-IV/Hausman-Taylor) |
| `estimate_xtlogit(pd, :y, :x; ...)` | Panel logit (pooled/FE/RE/CRE) |
| `estimate_xtprobit(pd, :y, :x; ...)` | Panel probit (pooled/FE/RE/CRE) |
| `hausman_test(m_fe, m_re)` | Hausman FE vs RE specification test |
| `breusch_pagan_test(m)` | Breusch-Pagan LM test |
| `pesaran_cd_test(m)` | Pesaran CD cross-sectional dependence test |
| `wooldridge_ar_test(m)` | Wooldridge AR(1) test |
| `modified_wald_test(m)` | Modified Wald heteroskedasticity test |
| `f_test_fe(m)` | F-test for fixed effects |

TWFE, Callaway-Sant'Anna, Sun-Abraham, BJS, and did_multiplegt estimators plus LP-DiD and diagnostics. See [DiD](@ref did_page) and [Event Study LP](@ref event_study_page) for theory and examples.

### Difference-in-Differences

| Function | Description |
|----------|-------------|
| `estimate_did(pd, :y, :treat; ...)` | DiD estimation (5 methods: twfe/cs/sa/bjs/did_multiplegt) |
| `estimate_event_study_lp(pd, :y, :treat; ...)` | Event study LP for panel data |
| `estimate_lp_did(pd, :y, :treat; ...)` | LP-DiD (Dube et al. 2025) |
| `bacon_decomposition(pd, :y, :treat)` | Goodman-Bacon (2021) decomposition |
| `pretrend_test(result)` | Pre-trend parallel trends test |
| `negative_weight_check(pd, :y, :treat)` | Negative weight diagnostic |
| `honest_did(result; ...)` | HonestDiD sensitivity analysis |

Two-step or Bayesian Gibbs FAVAR with factor-to-observable IRF mapping. See [FAVAR](@ref favar_page) for theory and examples.

### FAVAR

| Function | Description |
|----------|-------------|
| `estimate_favar(Y_slow, Y_fast, r, p; ...)` | FAVAR (two-step or Bayesian Gibbs) |
| `favar_panel_irf(favar, H)` | Map factor IRFs to N observables |
| `favar_panel_forecast(favar, h)` | FAVAR multi-step forecasting |

Structural DFM combining GDFM spectral estimation with structural VAR identification. See [Factor Models](@ref factor_page) for theory and examples.

### Structural DFM

| Function | Description |
|----------|-------------|
| `estimate_structural_dfm(X, q; ...)` | Structural DFM (FGLR 2009; `method=:fglr`) |
| `sdfm_panel_irf(sdfm, H)` | Map structural factor IRFs to observables |
| `structural_shocks(sdfm)` | Estimated structural shock series |
| `forecast(sdfm, h; ...)` | Panel forecast from the factor VAR |
| `varindex(sdfm, name)` | Panel variable index for sign restrictions |
| `fevd(sdfm, H; space=:panel)` | Observable FEVD (optional idiosyncratic column) |
| `historical_decomposition(sdfm)` | Panel HD from the stored rotation |
| `historical_decomposition(gdfm)` | Dynamic-PC decomposition of the common component |

Periodogram, Welch/Daniell/AR spectral density, cross-spectrum, coherence, and autocorrelation functions. See [Spectral Analysis](@ref spectral_page) for theory and examples, and [Model Diagnostics](@ref tests_diagnostics_page) for serial correlation tests.

### Spectral Analysis

| Function | Description |
|----------|-------------|
| `periodogram(y; ...)` | Raw periodogram |
| `spectral_density(y; ...)` | Smoothed spectral density (Welch/Daniell/AR) |
| `cross_spectrum(x, y; ...)` | Cross-spectral analysis |
| `acf(y, maxlag)` | Sample autocorrelation function |
| `pacf(y, maxlag)` | Partial autocorrelation function |
| `ccf(x, y, maxlag)` | Cross-correlation function |
| `coherence(cs)` | Coherence from cross-spectrum |
| `phase(cs)` | Phase spectrum |
| `gain(cs)` | Gain function |
| `ideal_bandpass(y; pl, pu)` | Ideal bandpass filter |
| `transfer_function(b, a; ...)` | Filter transfer function |

Ljung-Box, Box-Pierce, and Durbin-Watson tests for autocorrelation and serial correlation. Their worked examples live on [Spectral Analysis](@ref spectral_page); see [Model Diagnostics](@ref tests_diagnostics_page) for the regression-residual diagnostics.

### Portmanteau and Serial Correlation Tests

| Function | Description |
|----------|-------------|
| `ljung_box_test(y, K)` | Ljung-Box autocorrelation test |
| `box_pierce_test(y, K)` | Box-Pierce autocorrelation test |
| `durbin_watson_test(m)` | Durbin-Watson serial correlation test |
| `bartlett_white_noise_test(y)` | Bartlett white noise test |
| `fisher_test(y)` | Fisher exact periodogram test |

Fourier ADF/KPSS, DF-GLS, LM unit root, two-break ADF, and Gregory-Hansen cointegration tests. See [Advanced Unit Root Tests](@ref tests_unitroot_advanced_page) for details.

### Advanced Unit Root Tests

| Function | Description |
|----------|-------------|
| `fourier_adf_test(y; ...)` | Fourier ADF test (Enders & Lee 2012) |
| `fourier_kpss_test(y; ...)` | Fourier KPSS test |
| `dfgls_test(y; ...)` | DF-GLS/ERS unit root test |
| `lm_unitroot_test(y; ...)` | LM unit root test with breaks |
| `adf_2break_test(y; ...)` | Two-break ADF test (Narayan & Popp 2010) |
| `gregory_hansen_test(Y; ...)` | Gregory-Hansen cointegration test with break |

Seasonal unit roots, explosive-bubble detection, nonlinearity, distributional comparison, and random-walk tests. See [Advanced Unit Root Tests](@ref tests_unitroot_advanced_page) and [Model Diagnostics](@ref tests_diagnostics_page).

### Higher-Moment, Bubble, and Distribution Tests

| Function | Description |
|----------|-------------|
| `hegy_test(y; frequency, deterministic)` | HEGY seasonal unit root test |
| `ers_test(y; trend)` | Elliott-Rothenberg-Stock point-optimal test |
| `sadf_test(y; r0, adflag)` / `gsadf_test(y; r0, adflag)` | PSY supremum ADF bubble detection |
| `bds_test(y; m, eps_frac)` | Brock-Dechert-Scheinkman i.i.d. test |
| `variance_ratio_test(y; q, robust)` | Lo-MacKinlay variance ratio test |
| `edf_test(y; dist, test)` | EDF goodness-of-fit (KS, Anderson-Darling, Cramer-von Mises) |
| `cor_test(x, y; method)` | Pearson/Spearman/Kendall correlation test |
| `equality_test(y, g; test)` / `anova_test(y, g)` / `ttest(x; mu)` | Group mean/variance equality tests |
| `symmetry_test(m)` | Wald test of long-run symmetry in a fitted NARDL |

Andrews SupWald/SupLM/SupLR, Bai-Perron multiple break detection, and factor structural break tests. See [Structural Breaks](@ref tests_breaks_page) for details.

### Structural Break Tests

| Function | Description |
|----------|-------------|
| `andrews_test(y, X; ...)` | Andrews (1993) SupWald/SupLM/SupLR |
| `bai_perron_test(y, X; ...)` | Bai-Perron (1998) multiple break detection |
| `factor_break_test(X; ...)` | Factor structural break test |

PANIC, Pesaran CIPS, and Moon-Perron panel unit root tests. See [Panel Tests](@ref tests_panel_page) for details.

### Panel Unit Root Tests

| Function | Description |
|----------|-------------|
| `panic_test(pd; ...)` | Bai-Ng (2004) PANIC test |
| `pesaran_cips_test(pd; ...)` | Pesaran (2007) CIPS test |
| `moon_perron_test(pd; ...)` | Moon-Perron (2004) test |
| `llc_test(X; deterministic, lags)` | Levin-Lin-Chu common-root test |
| `ips_test(X; deterministic, lags)` | Im-Pesaran-Shin heterogeneous-root test |
| `breitung_panel_test(X; deterministic)` | Breitung unbiased panel test |
| `hadri_test(X; deterministic, hetero)` | Hadri panel stationarity (KPSS-type) test |
| `fisher_panel_test(X; base=:adf, combine)` | Fisher-type combination of per-unit tests |
| `dh_causality_test(pd, x, y; p)` | Dumitrescu-Hurlin panel Granger causality |
| `panel_unit_root_summary(pd; ...)` | Run all panel unit root tests |

### Panel Cointegration Tests

| Function | Description |
|----------|-------------|
| `pedroni_test(pd, y, xs...; ...)` | Pedroni (1999, 2004) residual-based test (7 statistics) |
| `kao_test(pd, y, xs...; ...)` | Kao (1999) residual-based test (5 DF-type statistics) |
| `westerlund_test(pd, y, xs...; ...)` | Westerlund (2007) ECM test (Gt/Ga/Pt/Pa) |
| `fisher_johansen_test(pd, ys...; ...)` | Fisher-type (Maddala-Wu/Choi) combined Johansen test |

Leontief and Ghosh accounting, multipliers and linkages, structural decomposition, environmental extensions, MRIO trade accounting (KWW 2014), and the production-network approach of Baqaee & Farhi (2019). See [Input-Output Analysis](@ref io_page) for the hub and its child pages.

### Input-Output Analysis

| Function | Description |
|----------|-------------|
| `technical_coefficients(io)` | Direct requirements matrix ``A`` |
| `leontief(io)` / `leontief_inverse(io)` | Leontief system and total requirements ``L = (I-A)^{-1}`` |
| `ghosh(io)` / `ghosh_inverse(io)` | Supply-side (allocation) Ghosh model |
| `allocation_coefficients(io)` | Output allocation matrix ``B`` |
| `multipliers(io; kind=:output)` | Output, income, and employment multipliers |
| `linkages(io; forward=:ghosh)` | Backward and forward linkage indices |
| `key_sectors(io)` | Key-sector classification from the linkage quadrants |
| `hypothetical_extraction(io, sectors; mode, share)` | Output loss from extracting a sector (complete/backward/forward/partial) |
| `price_model(io; dva, dtax, mode)` | Leontief cost-push price model ``\Delta p = (I-A')^{-1}\Delta v`` |
| `impact(io, dy; kind, type, fix)` | Final-demand impact scenario (Type I/II or mixed model) |
| `network_stats(io)` | Domar HHI, APL matrix, degree structure, upstreamness |
| `sda(io0, io1; method=:additive, factors, on=:output)` | Structural decomposition (n-factor two-polar; emission SDA via `on`) |
| `ras(A0, u, v)` / `gras(A0, u, v)` | Biproportional matrix balancing (GRAS is sign-preserving) |
| `balance(io; method=:ras)` | Repair `IOData` intermediate flows to accounting margins |
| `domar_weights(io)` | Sales-to-GDP (Domar) weights |
| `baqaee_farhi(io; theta, sigma)` | Production-network shock propagation and influence vector |
| `add_extension!(io, name, F; unit)` | Attach an environmental/satellite account |
| `emission_multipliers(io, name)` / `footprint(io, name)` | Extension multipliers and consumption-based footprints |

Downloaders and parsers for the public multi-region input-output databases. See [Downloading IO Data](@ref io_download_page).

### Input-Output Data Sources

| Function | Description |
|----------|-------------|
| `list_io_sources()` | Available databases and their required credentials |
| `download_io(source; storage_folder, years)` | Download WIOD, OECD ICIO, EXIOBASE3, Eora26, or GLORIA |
| `parse_io(path; source, year)` | Parse a downloaded table into `IOData` |
| `parse_icio(path; year, …)` | OECD ICIO recipe → labeled multi-region `IOData` |
| `parse_wiod(path; year, …)` | WIOD 2013 WIOT recipe → labeled multi-region `IOData` |
| `io_file_digest(path)` | Content hash of a downloaded file |

D3.js visualizations for every model family, plus backend and file-export helpers. See [Plotting](@ref plotting_page) for the gallery.

### Visualization

| Function | Description |
|----------|-------------|
| `plot_result(obj; ...)` | Render a fitted model or result as an interactive plot |
| `save_plot(p, path)` | Write a `PlotOutput` to an HTML file |
| `display_plot(p)` | Show a `PlotOutput` in the active display |
| `with_display_backend(f, backend)` | Run `f()` with a scoped display backend |

Within-group lag, lead, and differencing utilities for panel data construction. See [Data Management](@ref data_page) for details.

### Panel Data Utilities

| Function | Description |
|----------|-------------|
| `panel_lag(pd, :var, k)` | Within-group lagged variable |
| `panel_lead(pd, :var, k)` | Within-group lead variable |
| `panel_diff(pd, :var)` | Within-group first difference |
| `add_panel_lag(pd, :var, k)` | Add lagged column to panel |
| `add_panel_lead(pd, :var, k)` | Add lead column to panel |
| `add_panel_diff(pd, :var)` | Add differenced column to panel |
| `balance_panel(d; ...)` | Fill NaN via DFM imputation |
