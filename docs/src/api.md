# API Reference

This section provides the complete API documentation for **MacroEconometricModels.jl**.

The API documentation is organized into the following pages:

- **[Types](@ref api_types)**: Core type definitions for models, results, and estimators
- **[Functions](@ref api_functions)**: Function documentation organized by module

The quick reference tables below cover all modules: data management, time series, multivariate models, cross-sectional and panel models, DSGE, difference-in-differences, factor models, spectral analysis, volatility, nowcasting, hypothesis tests, and output utilities.

## Quick Reference Tables

Typed data containers, built-in datasets (FRED-MD, FRED-QD, Penn World Table), and data cleaning utilities. See [Data Management](data.md) for theory and examples.

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
| `load_example(:fred_md)` / `load_example(:fred_qd)` / `load_example(:pwt)` | Load built-in datasets (FRED-MD, FRED-QD, PWT) |

AR, MA, ARMA, and ARIMA model estimation with automatic order selection. See [ARIMA Models](arima.md) for estimation methods, forecasting, and model selection.

### ARIMA Estimation Functions

| Function | Description |
|----------|-------------|
| `estimate_ar(y, p; method=:ols)` | AR(p) via OLS or MLE |
| `estimate_ma(y, q; method=:css_mle)` | MA(q) via CSS, MLE, or CSS-MLE |
| `estimate_arma(y, p, q; method=:css_mle)` | ARMA(p,q) via CSS, MLE, or CSS-MLE |
| `estimate_arima(y, p, d, q; method=:css_mle)` | ARIMA(p,d,q) via differencing + ARMA |
| `forecast(model, h; conf_level=0.95)` | Multi-step forecasting with confidence intervals |
| `select_arima_order(y, max_p, max_q)` | Grid search for optimal ARMA order |
| `auto_arima(y)` | Automatic ARIMA order selection |
| `ic_table(y, max_p, max_q)` | Information criteria comparison table |

Trend-cycle decomposition via HP, Hamilton, Beveridge-Nelson, Baxter-King, and boosted HP filters. See [Time Series Filters](filters.md) for theory and comparisons.

### Time Series Filters

| Function | Description |
|----------|-------------|
| `hp_filter(y; lambda=1600.0)` | Hodrick-Prescott trend-cycle decomposition |
| `hamilton_filter(y; h=8, p=4)` | Hamilton (2018) regression filter |
| `beveridge_nelson(y; p=:auto, q=:auto)` | Beveridge-Nelson permanent/transitory decomposition |
| `baxter_king(y; pl=6, pu=32, K=12)` | Baxter-King band-pass filter |
| `boosted_hp(y; stopping=:BIC, lambda=1600.0)` | Boosted HP filter (Phillips & Shi 2021) |
| `trend(result)` | Extract trend component from filter result |
| `cycle(result)` | Extract cyclical component from filter result |

VAR, VECM, BVAR, Local Projections, Factor Models, and Panel VAR estimation. See [VAR](manual.md), [VECM](vecm.md), [BVAR](bayesian.md), [LP](lp.md), [Factor Models](factormodels.md), and [Panel VAR](pvar.md) for theory and examples.

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
| `structural_lp(Y, H; method=:cholesky, ...)` | Structural LP with multi-shock IRFs |
| `estimate_vecm(Y, p; rank=:auto, ...)` | Estimate VECM via Johansen MLE or Engle-Granger |
| `to_var(vecm)` | Convert VECM to VAR in levels |
| `select_vecm_rank(Y, p; ...)` | Select cointegrating rank |
| `granger_causality_vecm(vecm, cause, effect)` | VECM Granger causality test |
| `forecast(vecm, h; ci_method=:none, ...)` | VECM forecast preserving cointegration |

Impulse response functions, forecast error variance decomposition, historical decomposition, and 18+ structural identification methods. See [Innovation Accounting](innovation_accounting.md) and [Non-Gaussian Identification](nongaussian.md).

### Structural Analysis Functions

| Function | Description |
|----------|-------------|
| `irf(model, H; ...)` | Compute impulse response functions |
| `fevd(model, H; ...)` | Forecast error variance decomposition |
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

Direct multi-step forecasting from Local Projection models. See [Local Projections](lp.md) for estimation details.

### LP Forecasting Functions

| Function | Description |
|----------|-------------|
| `forecast(lp, shock_path; ...)` | Direct multi-step LP forecast |
| `forecast(slp, shock_idx, shock_path; ...)` | Structural LP conditional forecast |

Augmented Dickey-Fuller, KPSS, Phillips-Perron, Zivot-Andrews, Ng-Perron, and Johansen cointegration tests. See [Hypothesis Tests](tests.md) for interpretation and examples.

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

Likelihood ratio (LR) and Lagrange multiplier (LM/score) tests for comparing nested models across ARIMA, VAR, and GARCH families. See [Hypothesis Tests](tests.md).

### Model Comparison Tests

| Function | Description |
|----------|-------------|
| `lr_test(m1, m2)` | Likelihood ratio test for nested models |
| `lm_test(m1, m2)` | Lagrange multiplier (score) test for nested models |

Pairwise and block Wald tests for Granger causality in VAR models. See [Hypothesis Tests](tests.md) for details.

### Granger Causality Tests

| Function | Description |
|----------|-------------|
| `granger_test(model, cause, effect)` | Pairwise or block Granger causality test |
| `granger_test_all(model)` | All-pairs pairwise Granger causality matrix |

Convenience functions for extracting impulse responses from fitted LP models. See [Local Projections](lp.md).

### LP IRF Extraction

| Function | Description |
|----------|-------------|
| `lp_irf(model; ...)` | Extract IRF from LPModel |
| `lp_iv_irf(model; ...)` | Extract IRF from LPIVModel |
| `smooth_lp_irf(model; ...)` | Extract smoothed IRF |
| `state_irf(model; ...)` | Extract state-dependent IRFs |
| `propensity_irf(model; ...)` | Extract ATE impulse response |

Static PCA, Dynamic Factor, and Generalized Dynamic Factor model estimation, forecasting, and selection criteria. See [Factor Models](factormodels.md).

### Factor Model Functions

| Function | Description |
|----------|-------------|
| `estimate_factors(X, r; ...)` | Estimate r-factor model |
| `estimate_dynamic_factors(X, r, p; ...)` | Dynamic factor model |
| `estimate_gdfm(X, q; ...)` | Generalized dynamic factor model |
| `forecast(fm, h; p=1, ci_method=:none)` | Static FM forecast (fits VAR(p) on factors) |
| `forecast(dfm, h; ci_method=:none)` | DFM forecast (`:none/:theoretical/:bootstrap/:simulation`) |
| `forecast(gdfm, h; ci_method=:none)` | GDFM forecast (`:none/:theoretical/:bootstrap`) |
| `ic_criteria(X, r_max)` | Bai-Ng information criteria |
| `ic_criteria_dynamic(X, max_r, max_p)` | DFM factor/lag selection |
| `ic_criteria_gdfm(X, max_q)` | GDFM dynamic factor selection |
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

Bayesian prior optimization, instrument strength tests, and Panel VAR specification tests. See [BVAR](bayesian.md) and [Panel VAR](pvar.md).

### Diagnostic Functions

| Function | Description |
|----------|-------------|
| `optimize_hyperparameters(Y, p; ...)` | Optimize Minnesota prior (τ only) |
| `optimize_hyperparameters_full(Y, p; ...)` | Joint optimization over (τ, λ, μ) (BGR 2010) |
| `posterior_mean_model(post; ...)` | VARModel from posterior mean |
| `posterior_median_model(post; ...)` | VARModel from posterior median |
| `weak_instrument_test(model; ...)` | Test for weak instruments |
| `sargan_test(model, h)` | Overidentification test |
| `test_regime_difference(model; ...)` | Test regime differences |
| `propensity_diagnostics(model)` | Propensity score diagnostics |
| `pvar_hansen_j(model)` | Hansen J-test for Panel VAR |
| `pvar_mmsc(model)` | Andrews-Lu MMSC for Panel VAR |
| `pvar_lag_selection(pd, max_p; ...)` | Panel VAR lag order selection |
| `j_test(model)` | Hansen J-test for GMM |
| `gmm_summary(model)` | Summary statistics for GMM |

Multivariate normality tests for VAR residuals. See [Non-Gaussian Identification](nongaussian.md) for using these as pre-tests for ICA/ML identification.

### Normality Test Functions

| Function | Description |
|----------|-------------|
| `jarque_bera_test(model; method=:multivariate)` | Multivariate Jarque-Bera test |
| `mardia_test(model; type=:both)` | Mardia skewness/kurtosis tests |
| `doornik_hansen_test(model)` | Doornik-Hansen omnibus test |
| `henze_zirkler_test(model)` | Henze-Zirkler characteristic function test |
| `normality_test_suite(model)` | Run all normality tests |

Diagnostic tests for non-Gaussian SVAR identification validity. See [Non-Gaussian Identification](nongaussian.md).

### Identifiability Test Functions

| Function | Description |
|----------|-------------|
| `test_shock_gaussianity(result)` | Test non-Gaussianity of recovered shocks |
| `test_gaussian_vs_nongaussian(model; ...)` | LR test: Gaussian vs non-Gaussian |
| `test_shock_independence(result; ...)` | Test independence of recovered shocks |
| `test_identification_strength(model; ...)` | Bootstrap identification strength test |
| `test_overidentification(model, result; ...)` | Overidentification test |

ARCH, GARCH, EGARCH, GJR-GARCH, and Stochastic Volatility estimation, forecasting, and diagnostics. See [Volatility Models](volatility.md).

### Volatility Model Functions

| Function | Description |
|----------|-------------|
| `estimate_arch(y, q)` | ARCH(q) via MLE |
| `estimate_garch(y, p, q)` | GARCH(p,q) via MLE |
| `estimate_egarch(y, p, q)` | EGARCH(p,q) via MLE |
| `estimate_gjr_garch(y, p, q)` | GJR-GARCH(p,q) via MLE |
| `estimate_sv(y; variant, ...)` | Stochastic Volatility via KSC Gibbs |
| `forecast(vol_model, h)` | Volatility forecast with simulation CIs |
| `arch_lm_test(y_or_model, q)` | ARCH-LM test for conditional heteroskedasticity |
| `ljung_box_squared(z_or_model, K)` | Ljung-Box test on squared residuals |
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

Mixed-frequency nowcasting via DFM, BVAR, and bridge equations with news decomposition. See [Nowcasting](nowcast.md) for theory and examples.

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

Specify, solve, simulate, and estimate Dynamic Stochastic General Equilibrium models. See [DSGE Models](dsge.md) for the full guide.

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
| `perturbation_solver(spec; order=2)` | Higher-order perturbation solver |
| `collocation_solver(spec; ...)` | Chebyshev collocation projection |
| `pfi_solver(spec; ...)` | Policy function iteration |
| `vfi_solver(spec; ...)` | Value function iteration |
| `is_determined(sol)` | Check existence and uniqueness |
| `is_stable(sol)` | Check stability of solution |

### DSGE Simulation and Analysis

| Function | Description |
|----------|-------------|
| `simulate(sol, T)` | Stochastic simulation |
| `irf(sol, H)` | Analytical impulse responses |
| `fevd(sol, H)` | Forecast error variance decomposition |
| `historical_decomposition(sol, data, obs)` | DSGE historical decomposition |
| `solve_lyapunov(G1, impact)` | Unconditional covariance (Lyapunov equation) |
| `analytical_moments(sol; lags)` | Analytical variance and autocovariances |
| `perfect_foresight(spec; T_periods, shock_path)` | Deterministic transition path |

### DSGE Estimation

| Function | Description |
|----------|-------------|
| `estimate_dsge(spec, data, params; method)` | GMM estimation (IRF matching, Euler, SMM, analytical) |
| `estimate_dsge_bayes(spec, data, θ0; ...)` | Bayesian estimation (SMC/SMC²/MH) |

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

OLS, WLS, IV/2SLS, logit, probit, ordered, and multinomial estimation for cross-sectional data. See [Regression](regression.md) and [Binary Choice](binary_choice.md) for theory and examples.

### Cross-Sectional Models

| Function | Description |
|----------|-------------|
| `estimate_reg(y, X; ...)` | OLS/WLS regression (HC0–HC3, cluster-robust SEs) |
| `estimate_iv(y, X, Z; ...)` | IV/2SLS estimation |
| `estimate_logit(y, X)` | Logit MLE via IRLS |
| `estimate_probit(y, X)` | Probit MLE via IRLS |
| `estimate_ologit(y, X)` | Ordered logit MLE |
| `estimate_oprobit(y, X)` | Ordered probit MLE |
| `estimate_mlogit(y, X)` | Multinomial logit MLE |
| `marginal_effects(m; ...)` | AME/MEM/MER with delta-method SEs |
| `odds_ratio(m)` | Odds ratios for logit models |
| `classification_table(m)` | Classification accuracy table |
| `vif(m)` | Variance inflation factors |
| `brant_test(m)` | Brant test for parallel regression |
| `hausman_iia(m)` | Hausman test for IIA assumption |

FE, RE, FD, Between, CRE, Arellano-Bond, and Blundell-Bond panel estimators. See [Panel Models](pvar.md) for theory and examples.

### Panel Regression

| Function | Description |
|----------|-------------|
| `estimate_xtreg(pd, :y, :x1, :x2; ...)` | Panel FE/RE/FD/Between/CRE/AB/BB |
| `estimate_xtiv(pd, :y, :x; ...)` | Panel IV (FE-IV/RE-IV/FD-IV/Hausman-Taylor) |
| `estimate_xtlogit(pd, :y, :x; ...)` | Panel logit (pooled/FE/RE/CRE) |
| `estimate_xtprobit(pd, :y, :x; ...)` | Panel probit (pooled/FE/RE/CRE) |
| `hausman_test(m_fe, m_re)` | Hausman FE vs RE specification test |
| `breusch_pagan_test(m)` | Breusch-Pagan LM test |
| `pesaran_cd_test(m)` | Pesaran CD cross-sectional dependence test |
| `wooldridge_ar_test(m)` | Wooldridge AR(1) test |
| `modified_wald_test(m)` | Modified Wald heteroskedasticity test |
| `f_test_fe(m)` | F-test for fixed effects |

TWFE, Callaway-Sant'Anna, Sun-Abraham, BJS, and did_multiplegt estimators plus LP-DiD and diagnostics. See [DiD](did.md) and [Event Study](event_study.md) for theory and examples.

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

Two-step or Bayesian Gibbs FAVAR with factor-to-observable IRF mapping. See [FAVAR](favar.md) for theory and examples.

### FAVAR

| Function | Description |
|----------|-------------|
| `estimate_favar(Y_slow, Y_fast, r, p; ...)` | FAVAR (two-step or Bayesian Gibbs) |
| `favar_panel_irf(favar, H)` | Map factor IRFs to N observables |
| `favar_panel_forecast(favar, h)` | FAVAR multi-step forecasting |

Structural DFM combining GDFM spectral estimation with structural VAR identification. See [Factor Models](factormodels.md) for theory and examples.

### Structural DFM

| Function | Description |
|----------|-------------|
| `estimate_structural_dfm(X, q; ...)` | Structural DFM (GDFM + VAR) |
| `sdfm_panel_irf(sdfm, H)` | Map structural factor IRFs to observables |

Periodogram, Welch/Daniell/AR spectral density, cross-spectrum, coherence, and autocorrelation functions. See [Hypothesis Tests](tests_diagnostics.md) for serial correlation tests.

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

Ljung-Box, Box-Pierce, and Durbin-Watson tests for autocorrelation and serial correlation. See [Hypothesis Tests](tests_diagnostics.md) for details.

### Portmanteau and Serial Correlation Tests

| Function | Description |
|----------|-------------|
| `ljung_box_test(y, K)` | Ljung-Box autocorrelation test |
| `box_pierce_test(y, K)` | Box-Pierce autocorrelation test |
| `durbin_watson_test(m)` | Durbin-Watson serial correlation test |
| `bartlett_white_noise_test(y)` | Bartlett white noise test |
| `fisher_test(y)` | Fisher exact periodogram test |

Fourier ADF/KPSS, DF-GLS, LM unit root, two-break ADF, and Gregory-Hansen cointegration tests. See [Advanced Unit Root Tests](tests_unitroot_advanced.md) for details.

### Advanced Unit Root Tests

| Function | Description |
|----------|-------------|
| `fourier_adf_test(y; ...)` | Fourier ADF test (Enders & Lee 2012) |
| `fourier_kpss_test(y; ...)` | Fourier KPSS test |
| `dfgls_test(y; ...)` | DF-GLS/ERS unit root test |
| `lm_unitroot_test(y; ...)` | LM unit root test with breaks |
| `adf_2break_test(y; ...)` | Two-break ADF test (Narayan & Popp 2010) |
| `gregory_hansen_test(Y; ...)` | Gregory-Hansen cointegration test with break |

Andrews SupWald/SupLM/SupLR, Bai-Perron multiple break detection, and factor structural break tests. See [Structural Break Tests](tests_breaks.md) for details.

### Structural Break Tests

| Function | Description |
|----------|-------------|
| `andrews_test(y, X; ...)` | Andrews (1993) SupWald/SupLM/SupLR |
| `bai_perron_test(y, X; ...)` | Bai-Perron (1998) multiple break detection |
| `factor_break_test(X; ...)` | Factor structural break test |

PANIC, Pesaran CIPS, and Moon-Perron panel unit root tests. See [Panel Unit Root Tests](tests_panel.md) for details.

### Panel Unit Root Tests

| Function | Description |
|----------|-------------|
| `panic_test(pd; ...)` | Bai-Ng (2004) PANIC test |
| `pesaran_cips_test(pd; ...)` | Pesaran (2007) CIPS test |
| `moon_perron_test(pd; ...)` | Moon-Perron (2004) test |
| `panel_unit_root_summary(pd; ...)` | Run all panel unit root tests |

Within-group lag, lead, and differencing utilities for panel data construction. See [Data Management](data.md) for details.

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
