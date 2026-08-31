# [Choosing a Method](@id method_guide_page)

This page routes a research question to the estimator that answers it. Find the row that matches your data and goal, then follow the page link for the model specification, keyword arguments, and worked examples. If you have not estimated anything with the package yet, run through [Installation & First Model](@ref getting_started_page) first — it takes ten minutes and establishes the workflow every section below assumes.

---

## Where to Start

The first question is what your data looks like. Everything else follows from it.

| If your data is | Start with | Why | Section |
|-----------------|------------|-----|---------|
| One series observed over time | `hp_filter`, `estimate_arima`, `estimate_garch` | Trend, dynamics, volatility | [Univariate Time Series](@ref mg_univariate) |
| Several series observed jointly | `estimate_var`, `estimate_bvar`, `estimate_lp` | System dynamics and spillovers | [Multivariate Time Series](@ref mg_multivariate) |
| An estimated system needing economic shocks | `irf`, `identify_sign`, `fevd` | Reduced form is not structural | [Structural Analysis](@ref mg_structural) |
| Independent observations, no time dimension | `estimate_reg`, `estimate_logit` | Cross-sectional inference | [Cross-Section and Panel](@ref mg_micro) |
| Many units observed over time | `estimate_xtreg`, `estimate_did`, `estimate_pvar` | Unobserved heterogeneity handled | [Cross-Section and Panel](@ref mg_micro) |
| A theoretical model rather than data | `@dsge`, `solve` | Structural equilibrium restrictions | [DSGE Models](@ref mg_dsge) |
| An inter-industry transactions table | `IOData`, `leontief` | Production-network accounting | [Input-Output](@ref mg_io) |
| Series released at different frequencies and lags | `nowcast_dfm`, `estimate_midas` | Ragged edge handled explicitly | [Nowcasting and Forecasting](@ref mg_forecasting) |
| Raw input of any of the above | `load_example`, `apply_tcode`, `xtset` | Typed containers dispatch everywhere | [Data Management](@ref data_page) |

Every estimator accepts a `TimeSeriesData`, `PanelData`, or `CrossSectionData` container directly, so the data step is the same regardless of which row you land on.

---

## [Univariate Time Series](@id mg_univariate)

One series observed over time. The choice turns on whether you want to remove a component, model the conditional mean, model the conditional variance, or let the data choose the functional form.

| If you want to | Use | Why | Page |
|----------------|-----|-----|------|
| Separate trend from cycle | `hp_filter`, `hamilton_filter`, `baxter_king`, `beveridge_nelson`, `boosted_hp` | No model specification required | [Time Series Filters](@ref filters_page) |
| Remove seasonality | `x13_filter` | X-11 and SEATS in pure Julia | [X-13ARIMA-SEATS](@ref x13_page) |
| Measure cyclical frequency and persistence | `periodogram`, `spectral_density`, `acf`, `pacf`, `cross_spectrum` | Frequency-domain and lag diagnostics | [Spectral Analysis](@ref spectral_page) |
| Test residuals for serial correlation | `ljung_box_test`, `box_pierce_test`, `durbin_watson_test` | Portmanteau tests on the autocorrelations | [Spectral Analysis](@ref spectral_page) |
| Fit an ARMA/ARIMA model and forecast | `estimate_arima`, `auto_arima`, `estimate_sarima` | Automatic order selection | [ARIMA](@ref arima_page) |
| Model long memory | `estimate_arfima`, `gph_test` | Fractional differencing parameter | [ARIMA](@ref arima_page) |
| Extract an unobserved trend or cycle state | `estimate_statespace`, `local_level`, `local_linear_trend` | Kalman filter with estimated hyper-parameters | [State-Space Models](@ref statespace_page) |
| Let regression coefficients drift | `estimate_tvp_reg` | Random-walk coefficient paths | [State-Space Models](@ref statespace_page) |
| Model time-varying volatility | `estimate_garch`, `estimate_egarch`, `estimate_gjr_garch`, `estimate_sv` | Conditional variance dynamics and leverage | [Volatility Models](@ref volatility_page) |
| Model asymmetric or regime-dependent dynamics | `estimate_threshold`, `estimate_setar`, `estimate_star`, `estimate_ms` | Piecewise or smooth regime transitions | [Nonlinear Time Series](@ref nonlinear_page) |
| Estimate a shape without imposing one | `kernel_density`, `kernel_reg`, `lowess` | Data-driven smoothing, no parametric form | [Nonparametric Regression](@ref nonparametric_page) |

Filters and state-space models both deliver a trend-cycle split. Use a filter when the decomposition is a preprocessing step; use a state-space model when the trend is an object of inference with standard errors.

---

## [Multivariate Time Series](@id mg_multivariate)

Several series observed jointly. The dividing lines are stationarity, the ratio of parameters to observations, and whether you want the full system or a single dynamic response.

| If your data or goal is | Use | Why | Page |
|-------------------------|-----|-----|------|
| Stationary system, joint dynamics | `estimate_var`, `select_lag_order` | Reduced-form workhorse, fully tooled | [VAR](@ref var_page) |
| Many variables or a short sample | `estimate_bvar` | Minnesota shrinkage tames the parameter count | [Bayesian VAR](@ref bvar_page) |
| Coefficients that drift over the sample | `estimate_tvpvar` | Time-varying parameters and stochastic volatility | [Bayesian VAR](@ref bvar_page) |
| A system mixing monthly and quarterly series | `estimate_mfvar` | State-space mixed-frequency VAR | [Bayesian VAR](@ref bvar_page) |
| Cointegrated ``I(1)`` variables | `estimate_vecm`, `johansen_test`, `to_var` | Rank restriction imposes long-run equilibria | [VECM](@ref vecm_page) |
| A mix of ``I(0)`` and ``I(1)`` regressors | `estimate_ardl`, `bounds_test`, `long_run` | No unit-root pre-testing required | [ARDL & Bounds Testing](@ref ardl_page) |
| One efficient cointegrating vector | `estimate_cointreg` | Endogeneity-corrected long-run coefficients | [Cointegrating Regression](@ref cointreg_page) |
| Dynamic responses robust to misspecification | `estimate_lp`, `estimate_lp_iv`, `estimate_smooth_lp`, `estimate_state_lp`, `lp_fevd` | Horizon-by-horizon, no iterated bias | [Local Projections](@ref lp_page) |
| A high-dimensional panel driven by few factors | `estimate_factors`, `estimate_dynamic_factors`, `ic_criteria` | Dimension reduction with Bai-Ng criteria | [Factor Models](@ref factor_page) |
| Factors and observed variables in one VAR | `estimate_favar`, `favar_panel_irf` | IRFs for hundreds of observables | [FAVAR](@ref favar_page) |

VAR and local projections answer the same question with different bias-variance trade-offs: the VAR is efficient when correctly specified, local projections are robust when it is not. Estimate both when the impulse responses drive the conclusion.

---

## [Structural Analysis and Identification](@id mg_structural)

Reduced-form residuals are correlated across equations and carry no economic interpretation. Identification recovers the structural shocks; innovation accounting reports what they do.

| If you want to | Use | Why | Page |
|----------------|-----|-----|------|
| Impose a recursive ordering | `irf(m, H; method=:cholesky)` | Fastest, exactly identified | [Structural Identification](@ref structural_identification_page) |
| Restrict the sign of responses | `identify_sign`, `identify_arias`, `identify_uhlig` | Weak, credible economic restrictions | [Structural Identification](@ref structural_identification_page) |
| Use documented historical episodes | `identify_narrative` | Event evidence sharpens the set | [Structural Identification](@ref structural_identification_page) |
| Restrict long-run effects to zero | `identify_long_run` | Blanchard-Quah supply-demand split | [Structural Identification](@ref structural_identification_page) |
| Identify with an external instrument | `identify_proxy`, `irf(m, H; method=:proxy, instruments=Z)` | High-frequency or narrative proxy for one (or k) shock | [Proxy SVAR](@ref id_proxy_page) |
| Impose non-recursive short-run zeros | `estimate_svar`, `irf(m, H; method=:ab, pattern=…)` | Amisano–Giannini AB-model ML with LR over-ID test | [AB-Model SVAR](@ref id_ab_page) |
| Identify from non-Gaussian residuals | `identify_fastica`, `identify_jade`, `identify_sobi`, `identify_student_t` | Statistical identification, no restrictions | [Non-Gaussian Methods](@ref id_nongaussian_page) |
| Identify from changing volatility | `identify_markov_switching`, `identify_garch`, `identify_smooth_transition`, `identify_external_volatility` | Regime variance shifts identify shocks | [Heteroskedasticity](@ref id_heteroskedastic_page) |
| Compare the statistical schemes | `irf(m, H; method=...)` | Eighteen `method=` symbols, one call | [Statistical Identification](@ref nongaussian_page) |
| Check that identification holds | independence and non-Gaussianity tests | Statistical ID needs testable assumptions | [Identification Testing](@ref id_testing_page) |
| Trace a shock through time | `irf` | Dynamic causal effect paths | [Impulse Responses](@ref ia_irf_page) |
| Attribute forecast error variance | `fevd` | Relative importance by horizon | [Variance Decomposition](@ref ia_fevd_page) |
| Explain observed historical movements | `historical_decomposition` | Shock contributions to actual data | [Historical Decomposition](@ref ia_hd_page) |
| See how the three fit together | `irf`, `fevd`, `historical_decomposition` | Shared identification, three questions | [Innovation Accounting](@ref innovation_accounting_page) |

Sign, narrative, and long-run restrictions require economic assumptions you must defend. The statistical schemes require distributional assumptions instead — test them with the tools on [Identification Testing](@ref id_testing_page) before reporting the shocks.

---

## [Cross-Section and Panel Data](@id mg_micro)

Independent observations, or many units followed over time. The choice turns on the outcome type first and the panel structure second.

| If your outcome or design is | Use | Why | Page |
|------------------------------|-----|-----|------|
| Continuous | `estimate_reg` | OLS/WLS with HC0-HC3 and clustered SEs | [Linear Regression](@ref regression_page) |
| Continuous with endogenous regressors | `estimate_iv`, `estimate_3sls`, `estimate_sur` | 2SLS with weak-instrument diagnostics | [Linear Regression](@ref regression_page) |
| High-dimensional with sparse truth | `estimate_lasso`, `estimate_ridge`, `estimate_elastic_net` | Penalized selection and shrinkage | [Linear Regression](@ref regression_page) |
| Censored, truncated, or self-selected | `estimate_tobit`, `estimate_truncreg`, `estimate_heckman` | Corrects the limited-dependent bias | [Linear Regression](@ref regression_page) |
| A count | `estimate_poisson`, `estimate_nbreg`, `dispersion_test` | Overdispersion tested and modelled | [Linear Regression](@ref regression_page) |
| A conditional quantile or a cutoff design | `estimate_qreg`, `estimate_rdd` | Distributional and local causal effects | [Linear Regression](@ref regression_page) |
| A fitted regression needing stability checks | `chow_test`, `cusum_test`, `reset_test`, `white_test`, `vif` | Breaks, functional form, heteroskedasticity | [Linear Regression](@ref regression_page) |
| Binary | `estimate_logit`, `estimate_probit`, `marginal_effects` | AME/MEM/MER with delta-method SEs | [Binary Choice Models](@ref binary_choice_page) |
| Ordered categories | `estimate_ologit`, `estimate_oprobit`, `brant_test` | Cut points plus parallel-regression test | [Ordered & Multinomial](@ref ordered_multinomial_page) |
| Unordered categories | `estimate_mlogit`, `hausman_iia` | Multinomial logit with IIA test | [Ordered & Multinomial](@ref ordered_multinomial_page) |
| A static panel | `estimate_xtreg`, `hausman_test`, `pesaran_cd_test` | FE/RE/FD/Between/CRE plus specification tests | [Panel Regression](@ref panel_reg_page) |
| A dynamic panel or panel IV | `estimate_xtreg` (`:ab`, `:bb`), `estimate_xtiv` | Arellano-Bond and Blundell-Bond GMM | [Panel Regression](@ref panel_reg_page) |
| A binary outcome in a panel | `estimate_xtlogit`, `estimate_xtprobit` | Fixed and random effects discrete choice | [Panel Regression](@ref panel_reg_page) |
| A cointegrated panel | `estimate_xtcointreg` | Group-mean and pooled panel FMOLS/DOLS | [Cointegrating Regression](@ref cointreg_page) |
| A panel system with feedback | `estimate_pvar` | Dynamic panel VAR by GMM | [Panel VAR](@ref pvar_page) |
| A treatment adopted at different dates | `estimate_did` | TWFE plus four heterogeneity-robust estimators | [Difference-in-Differences](@ref did_page) |
| Dynamics around a treatment date | `estimate_event_study_lp`, `estimate_lp_did` | Pre-trends and post-treatment paths | [Event Study LP](@ref event_study_page) |

Two-way fixed effects is biased under staggered adoption with heterogeneous effects. Start from `estimate_did` with a modern estimator, then use the Bacon decomposition and negative-weight diagnostics on [Difference-in-Differences](@ref did_page) to show what TWFE would have missed.

---

## [DSGE Models](@id mg_dsge)

Structural equilibrium models specified with the `@dsge` macro. The solution method follows from the accuracy you need and the nonlinearity you cannot linearize away.

| If you want to | Use | Why | Page |
|----------------|-----|-----|------|
| Understand the toolchain end to end | `@dsge`, `compute_steady_state`, `solve` | Specification through analysis | [DSGE Models](@ref dsge_page) |
| Solve a linearized model | `solve` (`:gensys`, `:klein`, `:blanchard_kahn`) | First-order accuracy, near-instant | [Linear Solvers](@ref dsge_linear) |
| Capture risk premia or precaution | `perturbation_solver` (order 2 or 3) | Certainty equivalence broken | [Nonlinear Methods](@ref dsge_nonlinear) |
| Solve accurately far from steady state | `collocation_solver`, `pfi_solver`, `vfi_solver` | Global accuracy on the state space | [Nonlinear Methods](@ref dsge_nonlinear) |
| Impose a zero lower bound | `occbin_solve`, `perfect_foresight` | Occasionally binding constraints | [Constraints](@ref dsge_constraints) |
| Estimate structural parameters | `estimate_dsge`, `estimate_dsge_bayes` | GMM/SMM and SMC/MH posteriors | [Estimation](@ref dsge_estimation) |
| Attribute observed data to shocks | `historical_decomposition` | Smoothed structural shock contributions | [Historical Decomposition](@ref dsge_hd_page) |
| Model household heterogeneity | `solve` on a `ModelSpec` with a `HouseholdSystem` | Krusell-Smith, HANK, multi-pop, plants, banks | [Heterogeneous Agents](@ref dsge_ha) |
| Model finite lifetimes or cohorts | `blanchard_solve` | Life-cycle and generational effects | [Overlapping Generations](@ref dsge_olg) |
| Work in continuous time | `ct_steady_state`, `ct_two_asset_solve`, `ct_hjb`, `ct_kfe` | HJB and Kolmogorov forward equations | [Continuous Time](@ref dsge_continuous) |

Solve linearly first. Move to higher-order perturbation or a global method only when the question depends on curvature — risk pricing, large shocks, or a binding constraint.

---

## [Method of Moments](@id mg_gmm)

Estimation from moment conditions rather than a likelihood.

| If your moments are | Use | Why | Page |
|---------------------|-----|-----|------|
| Analytic | `estimate_gmm`, `j_test` | One-step, two-step, iterated weighting | [Generalized & Simulated Method of Moments](@ref gmm_page) |
| Computable only by simulation | `estimate_smm` | Simulated moments match the data | [Generalized & Simulated Method of Moments](@ref gmm_page) |
| Dynamic-panel orthogonality conditions | `estimate_pvar`, `pvar_hansen_j`, `pvar_mmsc` | Arellano-Bond instruments, overidentification tested | [Panel VAR](@ref pvar_page) |
| Implied by a solved DSGE | `estimate_dsge` | IRF matching, Euler equations, SMM | [Estimation](@ref dsge_estimation) |

---

## [Input-Output and Production Networks](@id mg_io)

Inter-industry transactions tables, multipliers, and network propagation.

| If you want to | Use | Why | Page |
|----------------|-----|-----|------|
| Set up a table and check the identities | `IOData`, `load_example(:wiot)` | Container plus accounting checks | [Input-Output Analysis](io.md) |
| Compute multipliers and linkages | `leontief`, `ghosh`, `multipliers`, `linkages`, `key_sectors` | Demand- and supply-driven analysis | [Classical Analysis](io_classical.md) |
| Decompose change or remove a sector | `sda`, `hypothetical_extraction` | Structural decomposition and counterfactuals | [Classical Analysis](io_classical.md) |
| Attach emissions or employment accounts | `add_extension!`, `intensities`, `emission_multipliers`, `footprint` | Consumption-based accounting | [Environmental Extensions](io_environmental.md) |
| Decompose gross exports into value-added | `export_decomposition`, `vertical_specialization`, `aggregate` | KWW (2014) DVA/RDV/FVA/PDC, HIY VS | [MRIO Trade Accounting](io_mrio.md) |
| Go beyond Hulten's theorem | `baqaee_farhi`, `domar_weights` | Nonlinear network propagation | [Baqaee & Farhi (2019)](io_baqaee_farhi.md) |
| Fetch a public MRIO database | `list_io_sources`, `download_io`, `parse_io` | WIOD, OECD, EXIOBASE, EORA, GLORIA | [Downloading Data](io_download.md) |

---

## [Nowcasting and Forecasting](@id mg_forecasting)

Producing and evaluating forecasts, including current-quarter estimates from an incomplete data panel.

| If you want to | Use | Why | Page |
|----------------|-----|-----|------|
| Forecast from a model you already fitted | `forecast` | Same interface for every model family | *the estimator's own page* |
| Nowcast from a large mixed-frequency panel | `nowcast_dfm` | EM plus Kalman handles the ragged edge | [DFM Nowcasting](@ref nowcast_dfm_page) |
| Nowcast with shrinkage instead of factors | `nowcast_bvar` | Large BVAR with optimized hyper-parameters | [BVAR Nowcasting](@ref nowcast_bvar_page) |
| Keep the method transparent | `nowcast_bridge` | OLS bridge regressions, fast baseline | [Bridge Equations](@ref nowcast_bridge_page) |
| Attribute a revision to a release | `nowcast_news` | Kalman-gain weights per data release | [News Decomposition](@ref nowcast_news_page) |
| Compare the nowcasting approaches | `nowcast_dfm`, `nowcast_bvar`, `nowcast_bridge` | Same panel, three model classes | [Nowcasting](@ref nowcast_page) |
| Exploit within-quarter timing directly | `estimate_midas` | Parsimonious high-frequency lag weights | [MIDAS Regression](midas.md) |
| Score forecasts and test differences | `forecast_evaluate`, `diebold_mariano`, `clark_west`, `mincer_zarnowitz` | Accuracy metrics and equal-accuracy tests | [Forecast Evaluation](forecast_evaluation.md) |
| Combine competing forecasts | `combine_forecasts` | Equal, inverse-MSE, and regression weights | [Forecast Evaluation](forecast_evaluation.md) |

Bridge equations aggregate the high-frequency indicator before regressing; MIDAS keeps every high-frequency observation and estimates the weights. Use the bridge as a baseline and MIDAS when release timing within the quarter carries information.

---

## [Hypothesis Testing](@id mg_testing)

Specification and diagnostic tests. Run the stationarity tests before any time-series estimation.

| If you want to test for | Use | Why | Page |
|-------------------------|-----|-----|------|
| Which test family applies | `unit_root_summary`, `test_all_variables` | One call screens every series | [Hypothesis Tests](@ref tests_page) |
| A unit root or system cointegration | `adf_test`, `kpss_test`, `pp_test`, `johansen_test` | Standard stationarity and rank tests | [Unit Root & Cointegration](@ref tests_unitroot_page) |
| A unit root with breaks or nonlinear trend | `za_test`, `lm_unitroot_test`, `fourier_adf_test`, `dfgls_test` | Breaks destroy standard ADF power | [Advanced Unit Root](@ref tests_unitroot_advanced_page) |
| Cointegration in a single equation | `engle_granger_test`, `phillips_ouliaris_test`, `hansen_instability_test`, `park_added_test` | Residual-based, no system rank | [Residual-Based Cointegration](@ref tests_cointegration_page) |
| A structural break at an unknown date | `andrews_test`, `bai_perron_test`, `factor_break_test` | Sup-Wald and multiple-break dating | [Structural Breaks](@ref tests_breaks_page) |
| A unit root or cointegration in a panel | `pesaran_cips_test`, `panic_test`, `ips_test`, `pedroni_test` | Cross-sectional dependence handled | [Panel Tests](@ref tests_panel_page) |
| Panel causality across units | `dh_causality_test` | Dumitrescu-Hurlin heterogeneous causality | [Panel Tests](@ref tests_panel_page) |
| Residual pathologies | `arch_lm_test`, `normality_test_suite`, `jarque_bera_test`, `bds_test` | ARCH, non-normality, hidden nonlinearity | [Model Diagnostics](@ref tests_diagnostics_page) |
| Predictive ordering between series | `granger_test` | Granger causality, pairwise and block | [Model Diagnostics](@ref tests_diagnostics_page) |

ADF and KPSS have opposite nulls. Agreement between them is evidence; disagreement means the deterministic specification or the break structure needs attention before you difference.

---

## [Output, Notation, and Reference](@id mg_reference)

Once a model is estimated, these route the result into a paper.

| If you want to | Use | Why | Page |
|----------------|-----|-----|------|
| Plot any estimated result | `plot_result`, `save_plot` | Self-contained interactive D3.js output | [Visualization](@ref plotting_page) |
| Produce a publication table | `report`, `print_table`, `table`, `set_display_backend` | Text, LaTeX, and HTML backends | [Utilities & Display API](@ref api_utilities) |
| Cite the methods you used | `refs` | AEA, BibTeX, LaTeX, or HTML output | [How to Cite](@ref citation) |
| Check a symbol's meaning | notation dictionary | One convention across every page | [Notation](@ref notation) |
| Find a full bibliographic entry | canonical reference list | Every cited work, one entry each | [Bibliography](@ref bibliography) |
| See what changed between versions | release notes | Breaking changes and new features | [Changelog](@ref changelog) |
| Read a docstring or signature | `?` at the REPL | Complete exported-API catalog | [API Reference](api.md) |

---

## See Also

- [Installation & First Model](@ref getting_started_page) — the ten-minute onboarding tutorial
- [Data Management](@ref data_page) — the containers and transformations every row above assumes
- [Home](index.md) — the full feature list and package structure
