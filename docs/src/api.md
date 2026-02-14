# API Reference

This section provides the complete API documentation for **MacroEconometricModels.jl**.

The API documentation is organized into the following pages:

- **[Types](@ref api_types)**: Core type definitions for models, results, and estimators
- **[Functions](@ref api_functions)**: Function documentation organized by module

## Quick Reference Tables

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

### LP Forecasting Functions

| Function | Description |
|----------|-------------|
| `forecast(lp, shock_path; ...)` | Direct multi-step LP forecast |
| `forecast(slp, shock_idx, shock_path; ...)` | Structural LP conditional forecast |

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

### Model Comparison Tests

| Function | Description |
|----------|-------------|
| `lr_test(m1, m2)` | Likelihood ratio test for nested models |
| `lm_test(m1, m2)` | Lagrange multiplier (score) test for nested models |

### Granger Causality Tests

| Function | Description |
|----------|-------------|
| `granger_test(model, cause, effect)` | Pairwise or block Granger causality test |
| `granger_test_all(model)` | All-pairs pairwise Granger causality matrix |

### LP IRF Extraction

| Function | Description |
|----------|-------------|
| `lp_irf(model; ...)` | Extract IRF from LPModel |
| `lp_iv_irf(model; ...)` | Extract IRF from LPIVModel |
| `smooth_lp_irf(model; ...)` | Extract smoothed IRF |
| `state_irf(model; ...)` | Extract state-dependent IRFs |
| `propensity_irf(model; ...)` | Extract ATE impulse response |

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

### Diagnostic Functions

| Function | Description |
|----------|-------------|
| `optimize_hyperparameters(Y, p; ...)` | Optimize Minnesota prior |
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

### Normality Test Functions

| Function | Description |
|----------|-------------|
| `jarque_bera_test(model; method=:multivariate)` | Multivariate Jarque-Bera test |
| `mardia_test(model; type=:both)` | Mardia skewness/kurtosis tests |
| `doornik_hansen_test(model)` | Doornik-Hansen omnibus test |
| `henze_zirkler_test(model)` | Henze-Zirkler characteristic function test |
| `normality_test_suite(model)` | Run all normality tests |

### Identifiability Test Functions

| Function | Description |
|----------|-------------|
| `test_shock_gaussianity(result)` | Test non-Gaussianity of recovered shocks |
| `test_gaussian_vs_nongaussian(model; ...)` | LR test: Gaussian vs non-Gaussian |
| `test_shock_independence(result; ...)` | Test independence of recovered shocks |
| `test_identification_strength(model; ...)` | Bootstrap identification strength test |
| `test_overidentification(model, result; ...)` | Overidentification test |

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

### Covariance Functions

| Function | Description |
|----------|-------------|
| `newey_west(X, residuals; ...)` | Newey-West HAC estimator |
| `white_vcov(X, residuals; ...)` | White heteroskedasticity-robust |
| `driscoll_kraay(X, residuals; ...)` | Driscoll-Kraay panel-robust |
| `long_run_variance(x; ...)` | Long-run variance estimate |
| `long_run_covariance(X; ...)` | Long-run covariance matrix |
| `optimal_bandwidth_nw(residuals)` | Automatic bandwidth selection |

### Utility Functions

| Function | Description |
|----------|-------------|
| `construct_var_matrices(Y, p)` | Build VAR design matrices |
| `companion_matrix(B, n, p)` | VAR companion form |
| `robust_inv(A)` | Robust matrix inverse |
| `safe_cholesky(A; ...)` | Stable Cholesky decomposition |
