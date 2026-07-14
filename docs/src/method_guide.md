# [Choosing a Method](@id method_guide_page)

This guide routes you from a modelling question to the right estimator. Each row pairs a data situation or research goal with the recommended function and links to the page that documents it in full. If you are new to the package, start with [Installation & First Model](@ref getting_started_page) and return here once you know what you want to estimate.

The tables are grouped by data structure — univariate series, multivariate systems, cross sections, panels — and by task — structural analysis, nowcasting, estimation, and testing.

---

## Univariate Time Series

One series observed over time.

| If you want to | Use | Page |
|----------------|-----|------|
| Separate trend from cycle | `hp_filter`, `hamilton_filter`, `baxter_king`, `beveridge_nelson` | [Time Series Filters](@ref filters_page) |
| Seasonally adjust a monthly/quarterly series | `x13_filter` | [X-13ARIMA-SEATS](@ref x13_page) |
| Estimate the spectrum, ACF, or PACF | `periodogram`, `spectral_density`, `acf`, `pacf` | [Spectral Analysis](@ref spectral_page) |
| Fit an ARIMA / ARMA model and forecast | `estimate_arima`, `estimate_arma`, `forecast` | [ARIMA](@ref arima_page) |
| Model time-varying volatility | `estimate_garch`, `estimate_egarch`, `estimate_sv` | [Volatility Models](@ref volatility_page) |

---

## Multivariate Time Series

Several series observed jointly over time.

| If your data / goal is | Use | Page |
|------------------------|-----|------|
| Stationary system, dynamics of interest | `estimate_var` | [VAR](@ref var_page) |
| Small sample or many variables (shrinkage) | `estimate_bvar` | [Bayesian VAR](@ref bvar_page) |
| Cointegrated (unit-root) variables | `estimate_vecm`, `johansen_test` | [VECM](@ref vecm_page) |
| Robust dynamic responses without a full VAR | `estimate_lp` | [Local Projections](@ref lp_page) |
| A high-dimensional panel driven by few factors | `estimate_factors`, `estimate_dynamic_factors` | [Factor Models](@ref factor_page) |
| Factors plus observed variables in one VAR | `estimate_favar` | [FAVAR](@ref favar_page) |

---

## Structural Analysis & Identification

Given an estimated VAR/VECM/LP, recover economically meaningful shocks and trace their effects.

| If you want to | Use | Page |
|----------------|-----|------|
| Compute IRFs, FEVDs, or historical decompositions | `irf`, `fevd`, `historical_decomposition` | [Innovation Accounting](@ref innovation_accounting_page) |
| Read the impulse-response workflow in depth | `irf` | [Impulse Responses](@ref ia_irf_page) |
| Read the variance-decomposition workflow | `fevd` | [Variance Decomposition](@ref ia_fevd_page) |
| Read the historical-decomposition workflow | `historical_decomposition` | [Historical Decomposition](@ref ia_hd_page) |
| Identify shocks by sign, narrative, or long-run restrictions | `identify_sign`, `identify_narrative`, `identify_long_run` | [Structural Identification](@ref structural_identification_page) |
| Identify shocks statistically (non-Gaussianity / ICA) | `compute_Q` (ICA, ML) | [Non-Gaussian Methods](@ref id_nongaussian_page) |
| Identify shocks via heteroskedasticity | `compute_Q` (Markov-switching, GARCH) | [Heteroskedasticity](@ref id_heteroskedastic_page) |
| Choose among statistical schemes | overview | [Statistical Identification](@ref nongaussian_page) |
| Test identifying assumptions | overidentification / independence tests | [Testing](@ref id_testing_page) |

---

## Cross-Sectional Models

Independent observations, no time dimension.

| If your outcome is | Use | Page |
|--------------------|-----|------|
| Continuous | `estimate_reg` (OLS/WLS) | [Linear Regression](@ref regression_page) |
| Continuous with endogenous regressors | `estimate_iv` (2SLS) | [Linear Regression](@ref regression_page) |
| Binary (0/1) | `estimate_logit`, `estimate_probit` | [Binary Choice Models](@ref binary_choice_page) |
| Ordered categories | `estimate_ologit`, `estimate_oprobit` | [Ordered & Multinomial](@ref ordered_multinomial_page) |
| Unordered categories | `estimate_mlogit` | [Ordered & Multinomial](@ref ordered_multinomial_page) |

---

## Panel Data

Multiple units observed over time.

| If your data / goal is | Use | Page |
|------------------------|-----|------|
| Static panel (FE/RE/FD/between/CRE) | `estimate_xtreg` | [Panel Regression](@ref panel_reg_page) |
| Panel with endogenous regressors | `estimate_xtiv` | [Panel Regression](@ref panel_reg_page) |
| Binary outcome in a panel | `estimate_xtlogit`, `estimate_xtprobit` | [Panel Regression](@ref panel_reg_page) |
| Dynamic panel VAR | `estimate_pvar` | [Panel VAR](@ref pvar_page) |
| Treatment effect with staggered adoption | `estimate_did` | [Difference-in-Differences](@ref did_page) |
| Event-study dynamics around a treatment | `estimate_event_study_lp`, `estimate_lp_did` | [Event Study LP](@ref event_study_page) |

---

## DSGE Models

Structural equilibrium models.

| If you want to | Use | Page |
|----------------|-----|------|
| Understand the DSGE toolchain | overview | [DSGE Models](@ref dsge_page) |
| Solve a linearized model | `solve` (`:gensys`/`:klein`/`:blanchard_kahn`) | [Linear Solvers](@ref dsge_linear) |
| Solve with higher-order or global methods | `perturbation_solver`, `collocation_solver`, `pfi_solver` | [Nonlinear Methods](@ref dsge_nonlinear) |
| Impose occasionally-binding constraints (ZLB) | `occbin_solve`, `perfect_foresight` | [Constraints](@ref dsge_constraints) |
| Estimate structural parameters (GMM/Bayesian) | `estimate_dsge`, `estimate_dsge_bayes` | [Estimation](@ref dsge_estimation) |
| Decompose observed data into shocks | `historical_decomposition` | [Historical Decomposition](@ref dsge_hd_page) |
| Model household heterogeneity (HANK/Krusell-Smith) | `solve` (`HADSGESpec`) | [Heterogeneous Agents](@ref dsge_ha) |
| Model life-cycle / overlapping generations | `blanchard_solve` | [Overlapping Generations](@ref dsge_olg) |
| Work in continuous time (HJB/KFE) | `ct_steady_state`, `ct_two_asset_solve` | [Continuous Time](@ref dsge_continuous) |

---

## Method of Moments

Estimate parameters from moment conditions.

| If your model implies | Use | Page |
|-----------------------|-----|------|
| Analytic moment conditions | `estimate_gmm` | [Generalized & Simulated Method of Moments](@ref gmm_page) |
| Moments computable only by simulation | `estimate_smm` | [Generalized & Simulated Method of Moments](@ref gmm_page) |
| Dynamic-panel GMM moments | `estimate_pvar` | [Panel VAR](@ref pvar_page) |
| SMM/GMM estimation of a DSGE | `estimate_dsge` | [Estimation](@ref dsge_estimation) |

---

## Nowcasting

Real-time estimates of the current period.

| If you want to nowcast with | Use | Page |
|-----------------------------|-----|------|
| A dynamic factor model | `nowcast_dfm` | [DFM Nowcasting](@ref nowcast_dfm_page) |
| A Bayesian VAR | `nowcast_bvar` | [BVAR Nowcasting](@ref nowcast_bvar_page) |
| Bridge equations | `nowcast_bridge` | [Bridge Equations](@ref nowcast_bridge_page) |
| A decomposition of forecast revisions | `nowcast_news` | [News Decomposition](@ref nowcast_news_page) |
| An overview of the approaches | overview | [Nowcasting](@ref nowcast_page) |

---

## Hypothesis Tests

Diagnostics and specification tests.

| If you want to test for | Use | Page |
|-------------------------|-----|------|
| Choosing among the test families | overview | [Hypothesis Tests](@ref tests_page) |
| A unit root or cointegration | `adf_test`, `kpss_test`, `johansen_test` | [Unit Root & Cointegration](@ref tests_unitroot_page) |
| Unit roots with breaks or Fourier terms | `fourier_adf_test`, `za_test`, `dfgls_test` | [Advanced Unit Root](@ref tests_unitroot_advanced_page) |
| Structural breaks | `andrews_test`, `bai_perron_test` | [Structural Breaks](@ref tests_breaks_page) |
| Panel unit roots | `pesaran_cips_test`, `panic_test` | [Panel Tests](@ref tests_panel_page) |
| Serial correlation, normality, ARCH | `ljung_box_test`, `normality_test_suite`, `arch_lm_test` | [Model Diagnostics](@ref tests_diagnostics_page) |

---

## See Also

- [Installation & First Model](@ref getting_started_page) — the ten-minute quick-start tutorial
- [Data Management](@ref data_page) — getting your data into the right container first
- [Visualization](@ref plotting_page) — plotting any estimated result with `plot_result`
