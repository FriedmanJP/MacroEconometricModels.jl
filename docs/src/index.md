# MacroEconometricModels.jl

*A comprehensive Julia package for macroeconometric research and analysis*

## Overview

**MacroEconometricModels.jl** provides a unified, high-performance framework for estimating and analyzing macroeconometric models in Julia. The package implements state-of-the-art methods spanning the full empirical macro workflow: from unit root testing and trend-cycle decomposition, through univariate, multivariate, panel, and structural model estimation, to identification, nowcasting, forecast evaluation, and publication-quality output.

New users start at [Installation & First Model](@ref getting_started_page); readers who know the problem but not the method go to [Choosing a Method](@ref method_guide_page).

### Key Features

**Univariate Models**

- **Time Series Filters**: Hodrick-Prescott (1997), Hamilton (2018) regression, Beveridge-Nelson (1981), Baxter-King (1999) band-pass, and boosted HP (Phillips & Shi 2021) with unified `trend()`/`cycle()` accessors
- **ARIMA**: AR, MA, ARMA, ARIMA estimation via OLS, CSS, MLE (Kalman filter), and CSS-MLE; automatic order selection (`auto_arima`); ARFIMA long-memory estimation; multi-step forecasting with confidence intervals
- **State-Space Models**: User-specified linear-Gaussian systems with Kalman-MLE hyperparameter estimation, RTS smoothing, and forecasting; unobserved-components and structural time series (Harvey 1989); time-varying-parameter regression (Durbin & Koopman 2012)
- **Volatility Models**: ARCH (Engle 1982), GARCH (Bollerslev 1986), EGARCH (Nelson 1991), GJR-GARCH (Glosten et al. 1993), IGARCH, Component GARCH (Engle & Lee 1999), APARCH, FIGARCH/FIEGARCH long memory, and GARCH-MIDAS (Engle, Ghysels & Sohn 2013) via MLE; multivariate CCC/DCC/BEKK; Stochastic Volatility via Kim-Shephard-Chib (1998) Gibbs sampler (basic, leverage, Student-t variants); news impact curves, ARCH-LM diagnostics, multi-step forecasting
- **Nonlinear Time Series**: Threshold regression and SETAR with grid-search threshold estimation, STAR/LSTR smooth transition, and Markov-switching models; Hansen (1996) linearity test and Hansen (2000) threshold confidence intervals
- **Nonparametric Regression and Density**: Kernel density with Silverman and Sheather-Jones (1991) bandwidths, Nadaraya-Watson and Fan-Gijbels local-linear regression, local polynomials, and Cleveland (1979) LOWESS
- **Spectral Analysis**: Periodogram, Welch method, smoothed periodogram, AR spectral estimation; cross-spectrum with coherence, phase, gain functions; ACF/PACF/CCF with Ljung-Box/Box-Pierce/Durbin-Watson portmanteau tests
- **X-13ARIMA-SEATS**: Pure-Julia seasonal adjustment via X-11 (Henderson trend filters, seasonal moving averages) and SEATS (Gomez & Maravall 1996) Wiener-Kolmogorov signal extraction; TRAMO-style automatic ARIMA identification; additive-outlier, level-shift, and temporary-change detection; trading-day and Easter calendar regressors; unified `trend()`/`cycle()`/`seasonal()`/`adjusted()` accessors

**Multivariate Models**

- **VAR**: OLS estimation with lag order selection (AIC, BIC, HQ), stability diagnostics, companion matrix
- **Bayesian VAR**: Conjugate Normal-Inverse-Wishart posterior with Minnesota prior; direct and Gibbs samplers; automatic hyperparameter optimization via marginal likelihood (Giannone, Lenza & Primiceri 2015)
- **VECM**: Johansen MLE and Engle-Granger two-step estimation for cointegrated systems; automatic rank selection; IRF/FEVD/HD via VAR conversion (`to_var`); VECM-specific forecasting; Granger causality (short-run, long-run, strong)
- **ARDL and Bounds Testing**: ARDL and NARDL estimation with long-run multipliers, the conditional error-correction form, and the Pesaran-Shin-Smith (2001) bounds test; panel ARDL via pooled mean group
- **Cointegrating Regression**: FMOLS (Phillips-Hansen 1990), CCR (Park 1992), and DOLS (Saikkonen 1991; Stock & Watson 1993) for single-equation and panel cointegrating vectors
- **Systems of Equations**: Seemingly unrelated regressions and three-stage least squares
- **Panel VAR**: GMM estimation via Arellano-Bond (1991) first-difference and Blundell-Bond (1998) system GMM; fixed-effects OLS; Windmeijer (2005) corrected standard errors; Hansen J-test, Andrews-Lu MMSC; OIRF, GIRF, FEVD; group-level bootstrap CIs; lag selection
- **Local Projections**: Jorda (2005) with extensions for IV (Stock & Watson 2018), smooth LP (Barnichon & Brownlees 2019), state-dependence (Auerbach & Gorodnichenko 2013), propensity score weighting (Angrist et al. 2018), structural LP (Plagborg-Moller & Wolf 2021), LP forecasting, and LP-FEVD (Gorodnichenko & Lee 2019)
- **Factor Models**: Static (PCA), dynamic (two-step/EM), and generalized dynamic (spectral GDFM) with Bai-Ng information criteria; unified forecasting with theoretical and bootstrap CIs
- **FAVAR**: Factor-augmented VAR via two-step (PCA + VAR) or Bayesian Gibbs (Carter-Kohn smoother + NIW); `favar_panel_irf` maps factor IRFs to N observables via loadings
- **Structural DFM**: Structural dynamic factor model wrapping GDFM + VAR for identified factor shocks
- **GMM and SMM**: Flexible moment estimation with one-step, two-step, and iterated weighting; Hansen J-test; simulated method of moments

**Innovation Accounting**

- **IRF**: Impulse responses with bootstrap, theoretical, and Bayesian credible intervals
- **FEVD**: Forecast error variance decomposition (frequentist and Bayesian)
- **Historical Decomposition**: Decompose observed movements into structural shock contributions
- **LP-FEVD**: R-squared, LP-A, and LP-B estimators (Gorodnichenko & Lee 2019)

**DSGE Models**

- **Specification**: `@dsge` macro for domain-specific model specification with time-indexed variables, analytical or numerical steady state, and automatic Jacobian computation
- **Linear Solvers**: Gensys (Sims 2002), Blanchard-Kahn (1980), Klein (2000) with automatic eigenvalue decomposition
- **Nonlinear Perturbation**: Second-order (Schmitt-Grohe & Uribe 2004) and third-order perturbation with Andreasen, Fernandez-Villaverde & Rubio-Ramirez (2018) pruned simulation; Kim et al. (2008) second-order pruning
- **Global Methods**: Chebyshev collocation projection (tensor and Smolyak grids), policy function iteration, value function iteration (with Howard improvement steps and Anderson acceleration)
- **Simulation and IRFs**: Stochastic simulation, pruned higher-order simulation, analytical and generalized IRFs, FEVD, Lyapunov-based unconditional moments
- **GMM Estimation**: IRF matching, Euler equation GMM, Simulated Method of Moments, analytical GMM via Lyapunov equation
- **Bayesian Estimation**: Sequential Monte Carlo (SMC with adaptive tempering), SMC-squared (SMC² with particle filter likelihood), random-walk Metropolis-Hastings; delayed acceptance for accelerated sampling; nonlinear particle filter for higher-order solutions; posterior mode and marginal likelihood, MCMC convergence diagnostics, and Iskrev identification tests
- **Constraints**: Perfect foresight (Newton solver), OccBin occasionally binding constraints (Guerrieri & Iacoviello 2015), built-in constrained solvers (Optim.jl box constraints, NLopt.jl nonlinear inequalities, projected Newton, JuMP+Ipopt NLP) with an optional PATH (MCP) backend
- **Heterogeneity and Continuous Time**: Heterogeneous agents via the sequence-space Jacobian (Auclert et al. 2021), Reiter (2009), and Krusell-Smith (1998); overlapping generations (Blanchard 1985 perpetual youth, Auerbach-Kotlikoff life cycle); continuous-time HJB / Kolmogorov-Forward solvers on sparse grids

**Input-Output Analysis**

- **Container**: `IOData` stores intermediate flows, final demand, value added, gross output, sector and region labels, and satellite accounts, validating both accounting identities at construction
- **Classical Analysis**: Leontief (1936) demand-driven and Ghosh (1958) supply-driven inverses, Type I and Type II multipliers, Rasmussen linkages and key sectors, structural decomposition, hypothetical extraction
- **Environmental Extensions**: Stressor intensities, consumption-based emission multipliers, and footprints for any satellite account
- **Production Networks**: Domar weights, Hulten's theorem, the second-order Baqaee & Farhi (2019) "beyond Hulten" decomposition, and network centralities
- **Data Downloaders**: pymrio-style loaders for the major public multi-regional input-output databases

**Policy Counterfactuals**

- **Rule Counterfactuals**: McKay & Wolf (2023) news-implemented rule changes and optimal-policy projections from estimated causal effects; peg/Taylor/NGDP/AIT templates; second-moment (Wold) counterfactuals; implementation-error honesty diagnostics throughout
- **Optimal Policy Perturbations**: Barnichon & Mesters (2023) OPP statistic and optimality test from gap forecasts and policy-shock IRFs; two-source inference with 60/75/90% bands; ZLB and pre-commitment constraints via SLSQP; decision-date sequences with the time-consistency decomposition
- **Model Bank**: Caravello, McKay & Wolf (2025) limited-information IRF matching with Geweke marginal likelihoods and posterior model averaging; DSGE news menus, one-asset HANK sequence-space menus, and Gabaix/sticky-expectations behavioral variants; historical counterfactuals from forecast revisions; spanning and forecast-sufficiency diagnostics

**Structural Identification**

- Six schemes impose economic restrictions: Cholesky, sign restrictions, long-run (Blanchard-Quah), narrative restrictions, Arias et al. (2018) sign-and-zero, and the Mountford-Uhlig (2009) penalty function
- Fourteen further schemes exploit statistical structure instead, so the `method=` keyword accepts eighteen symbols in total, every one of them returning the same result object
- Non-Gaussian ICA: FastICA, JADE, SOBI, dCov, HSIC
- Non-Gaussian ML: Student-t, mixture-normal, PML, skew-normal, plus the `:nongaussian_ml` dispatcher that selects the distribution at runtime
- Heteroskedasticity-based: Markov-switching, GARCH, smooth-transition, external volatility
- Identifiability diagnostics: gaussianity tests, independence tests, bootstrap strength tests

**Nowcasting**

- **Dynamic Factor Model (DFM)**: EM algorithm with Kalman smoother for mixed-frequency data with arbitrary missing patterns (Banbura & Modugno 2014); Mariano-Murasawa temporal aggregation; block factor structure; AR(1)/IID idiosyncratic components
- **Large Bayesian VAR**: GLP-style Normal-Inverse-Wishart prior with hyperparameter optimization via marginal likelihood (Cimadomo et al. 2022); Minnesota shrinkage with sum-of-coefficients and co-persistence priors
- **Bridge Equations**: OLS bridge regressions combining pairs of monthly indicators via median (Banbura et al. 2023); transparent and fast baseline
- **MIDAS**: Mixed-data-sampling regressions that keep every high-frequency observation under a parsimonious weighting function (Ghysels, Santa-Clara & Valkanov 2006)
- **News Decomposition**: Attribute nowcast revisions to individual data releases via Kalman gain weights
- **Panel Balancing**: `balance_panel()` fills NaN in `TimeSeriesData`/`PanelData` using DFM imputation

**Forecast Evaluation**

- Point-forecast accuracy scores, Diebold-Mariano and Harvey-Leybourne-Newbold equal-accuracy tests, forecast encompassing, and combination weights (Bates-Granger, inverse-MSE, regression-based); model-agnostic, consuming plain vectors of realizations and forecasts from any estimator

**Cross-Sectional Models**

- **Linear Regression**: OLS with HC0--HC3 robust and cluster-robust standard errors; Weighted Least Squares (WLS); IV/2SLS and k-class estimators with first-stage F-statistic and Sargan test; VIF multicollinearity diagnostics; penalized (ridge, lasso, elastic net) and robust (M, S, MM) estimators
- **Binary Choice**: Logit and Probit MLE via IRLS; marginal effects (AME/MEM/MER) with delta-method SEs; `odds_ratio()`, `classification_table()`
- **Ordered and Multinomial**: Ordered Logit/Probit MLE with cut-point estimation; Multinomial Logit MLE; Brant (1990) parallel regression test; Hausman-McFadden IIA test
- **Limited Dependent and Count**: Tobit and Heckman selection models; Poisson and Negative-Binomial-2 regression with pseudo-ML sandwich standard errors, dispersion testing, and incidence-rate ratios

**Panel Models**

- **Panel Regression**: `estimate_xtreg` for FE/RE/FD/Between/CRE/Arellano-Bond/Blundell-Bond; `estimate_xtiv` for panel IV (FE-IV/RE-IV/FD-IV/Hausman-Taylor); `estimate_xtlogit`/`estimate_xtprobit` for panel discrete choice; PCSE and Prais-Winsten corrections; Hausman, Breusch-Pagan, Pesaran CD specification tests
- **Difference-in-Differences**: Five estimators — TWFE, Callaway-Sant'Anna (2021), Sun-Abraham (2021), Borusyak-Jaravel-Spiess (2024), de Chaisemartin-D'Haultfoeuille (2020); Bacon (2021) decomposition; pretrend tests; negative weight diagnostics; HonestDiD (Rambachan & Roth 2023) sensitivity analysis
- **Event Study LP**: Local projection event study with staggered treatment, cluster-robust SEs
- **LP-DiD**: Dube, Girardi, Jorda & Taylor (2025) LP-DiD estimator with clean control samples, PMD, and IPW reweighting

**Hypothesis Tests**

- **Unit root**: ADF, KPSS, Phillips-Perron, Zivot-Andrews, Ng-Perron, Fourier ADF/KPSS (Enders & Lee 2012; Becker, Enders & Lee 2006), DF-GLS/ERS (Elliott, Rothenberg & Stock 1996), LM unit root with 0/1/2 breaks (Schmidt-Phillips 1992; Lee-Strazicich 2003, 2013), two-break ADF (Narayan & Popp 2010), HEGY seasonal unit root
- **Cointegration**: Johansen trace and max-eigenvalue tests, Engle-Granger and Phillips-Ouliaris residual-based tests, Gregory-Hansen (1996) with regime shift, panel cointegration
- **Structural breaks**: Andrews (1993) SupWald/SupLM/SupLR with 9 test variants; Bai-Perron (1998) multiple break detection via dynamic programming with BIC/LWZ/sequential selection; CUSUM and CUSUMSQ; factor break tests — Breitung-Eickmeier (2011), Chen-Dolado-Gonzalo (2014), Han-Inoue (2015)
- **Panel unit root**: Bai-Ng (2004) PANIC with factor-adjusted pooled/individual tests; Pesaran (2007) CIPS with cross-sectional augmentation; Moon-Perron (2004) factor-adjusted t-statistics; first-generation Levin-Lin-Chu, Im-Pesaran-Shin, Breitung, and Fisher-type tests; Dumitrescu-Hurlin panel causality
- **Bubbles and dependence**: SADF/GSADF right-tailed explosive-root tests, BDS independence, variance-ratio tests
- **Granger causality**: pairwise Wald, block (multivariate), all-pairs matrix
- **Model comparison**: likelihood ratio (LR) and Lagrange multiplier (LM/score) tests for nested models
- **Normality and distribution**: Jarque-Bera, Mardia multivariate, Doornik-Hansen, Henze-Zirkler, Royston, and EDF goodness-of-fit tests; unified `normality_test_suite()`
- **ARCH diagnostics**: ARCH-LM test, Ljung-Box on squared residuals
- **Stationarity diagnostics**: `unit_root_summary()`, `test_all_variables()`
- **Panel VAR specification**: Hansen J-test, Andrews-Lu MMSC, lag selection criteria

**Visualization**

- **D3.js Plotting**: Zero-dependency interactive visualization via D3.js v7, with a `plot_result` dispatch for every result type the package produces; IRF, FEVD, historical decomposition, filter output, forecasts, model diagnostics, DiD event studies, nowcast fan charts
- **Output Formats**: Self-contained HTML files with Solarized Light/Dark themes; embeddable in documentation and presentations

**Data Management**

- **Typed Containers**: `TimeSeriesData`, `PanelData`, `CrossSectionData`, and `IOData` with metadata (frequency, variable names, transformation codes)
- **Built-in Datasets**: Twelve example datasets via `load_example()` — FRED-MD, FRED-QD, the Penn World Table, the Acemoglu et al. (2019) democracy panel, the Callaway-Sant'Anna minimum-wage panel, a world input-output table, and the canonical Nile, Mroz, Hamilton GNP, stack-loss, Grunfeld, and Danish money-demand series
- **Validation**: `diagnose()` detects NaN/Inf/constant columns; `fix()` repairs via listwise deletion, interpolation, or mean imputation
- **FRED Transforms**: `apply_tcode()` / `inverse_tcode()` implement all 7 FRED-MD transformation codes (McCracken & Ng 2016)
- **Panel Support**: Stata-style `xtset()` for panel construction, `group_data()` for per-entity extraction
- **Summary Statistics**: `describe_data()` with N, Mean, Std, Min, P25, Median, P75, Max, Skewness, Kurtosis
- **Estimation Dispatch**: All estimation functions accept `TimeSeriesData` directly

**Output, Reproducibility, and References**

- Display backends: switchable text, LaTeX, and HTML table output via `set_display_backend()`
- Publication-quality tables: `report()`, `table()`, `print_table()`, with Tables.jl integration for result types
- Reproducibility: `ReproManifest` / `reproduce` and versioned `save_model` / `load_model`
- Bibliographic references: `refs(model)` in AEA text, BibTeX, LaTeX, or HTML format

## Installation

```julia
using Pkg
Pkg.add("MacroEconometricModels")
```

Or from the Julia REPL package mode:

```
] add MacroEconometricModels
```

## Package Structure

The package is organized into the following modules:

| Module | Description |
|--------|-------------|
| `data/` | Data containers, validation, FRED transforms, panel support, example datasets, summary statistics |
| `core/` | Shared infrastructure: types, utilities, display backends, covariance estimators, identification dispatch |
| `arima/` | ARIMA suite: types, Kalman filter, estimation (CSS/MLE), ARFIMA, forecasting, order selection |
| `statespace/` | Linear-Gaussian state-space models: Kalman-MLE estimation, RTS smoothing, TVP regression |
| `filters/` | Time series filters: HP, Hamilton, Beveridge-Nelson, Baxter-King, boosted HP |
| `x13/` | X-13ARIMA-SEATS seasonal adjustment: X-11 decomposition, SEATS signal extraction, TRAMO auto-ARIMA, outlier detection, calendar effects |
| `arch/` | ARCH(q) estimation via MLE, volatility forecasting |
| `garch/` | GARCH family (GARCH, EGARCH, GJR, IGARCH, Component, APARCH, FIGARCH/FIEGARCH, GARCH-MIDAS) via MLE, news impact curves, forecasting |
| `mgarch/` | Multivariate GARCH: CCC, DCC, and BEKK |
| `sv/` | Stochastic Volatility via KSC (1998) Gibbs sampler, posterior predictive forecasts |
| `nonlinear/` | Nonlinear time series: threshold/SETAR, STAR/LSTR, Markov-switching, linearity tests |
| `nonparametric/` | Kernel density, Nadaraya-Watson and local-polynomial regression, LOWESS |
| `spectral/` | Spectral analysis: periodogram, Welch, AR, cross-spectrum, ACF/PACF/CCF, portmanteau tests |
| `reg/` | Cross-sectional regression: OLS, WLS, IV/2SLS, Logit, Probit, Ordered, Multinomial, count, Tobit/Heckman, penalized, robust, marginal effects |
| `preg/` | Panel regression: FE/RE/FD/Between/CRE/AB/BB, Panel IV, Panel Logit/Probit, PCSE, specification tests |
| `var/` | VAR estimation (OLS), structural identification, IRF, FEVD, historical decomposition |
| `vecm/` | VECM: Johansen MLE, Engle-Granger, cointegrating vectors, forecasting, Granger causality |
| `bvar/` | Bayesian VAR: conjugate NIW posterior sampling, Minnesota prior, hyperparameter optimization |
| `ardl/` | ARDL and NARDL, bounds testing, long-run multipliers, panel ARDL (pooled mean group) |
| `cointreg/` | Cointegrating regression: FMOLS, CCR, DOLS for single equations and panels |
| `system/` | Systems of equations: seemingly unrelated regressions and 3SLS |
| `lp/` | Local Projections: core, IV, smooth, state-dependent, propensity, structural LP, forecast, LP-FEVD |
| `factor/` | Static (PCA), dynamic (two-step/EM), generalized (spectral) factor models with forecasting |
| `favar/` | Factor-Augmented VAR: two-step (PCA + VAR) and Bayesian Gibbs estimation, panel IRFs |
| `nongaussian/` | Statistical identification: non-Gaussian ICA and ML, heteroskedasticity-based schemes, identifiability tests |
| `teststat/` | Statistical tests: unit root, cointegration, structural breaks, panel unit root, normality, Granger causality, bubbles, LR/LM, ARCH diagnostics |
| `pvar/` | Panel VAR: types, transforms, instruments, estimation (GMM/FE-OLS), analysis, bootstrap, tests |
| `did/` | Difference-in-Differences: TWFE, Callaway-Sant'Anna, Sun-Abraham, BJS, de Chaisemartin-D'Haultfoeuille; event study LP; LP-DiD |
| `dsge/` | DSGE: specification, linearization, solution (Gensys/BK/Klein/perturbation/projection/PFI/VFI), constrained solvers (Optim/NLopt/projected Newton), OccBin, heterogeneous agents, Bayesian estimation (SMC/SMC²/MH) |
| `olg/` | Overlapping generations: perpetual-youth and life-cycle solvers |
| `ct/` | Continuous-time heterogeneous agents: HJB and Kolmogorov-Forward solvers |
| `io/` | Input-Output analysis: Leontief/Ghosh models, multipliers, linkages, SDA, extraction, environmental extensions, Baqaee-Farhi (2019), MRIO downloaders |
| `counterfactual/` | Policy counterfactuals: McKay-Wolf rule counterfactuals and optimal policy, Barnichon-Mesters OPP (with ZLB constraints and sequences), Caravello-McKay-Wolf model bank, spanning and forecast-sufficiency diagnostics |
| `gmm/` | Generalized Method of Moments and Simulated Method of Moments |
| `midas/` | MIDAS mixed-frequency regression with restricted and unrestricted lag weighting |
| `nowcast/` | Nowcasting: DFM (EM + Kalman), large BVAR, bridge equations, news decomposition |
| `fceval/` | Forecast evaluation and combination: accuracy scores, equal-accuracy tests, combination weights |
| `plotting/` | D3.js interactive visualization: plot dispatches across model families, Solarized Light/Dark themes |
| `summary*.jl` | Publication-quality summary tables, display backends, and `refs()` bibliographic references |

## Mathematical Notation

The consistent notation dictionary used across every documentation page lives on a dedicated page:
see [Notation](@ref notation).

## References

The complete, canonical bibliography — every work cited anywhere in the documentation, one entry per
work, alphabetical by author — lives on a dedicated page: see [Bibliography](@ref bibliography).

## License

This package is released under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html). The GPL warranty-disclaimer and redistribution-condition notices are available at the REPL via [`warranty`](@ref) and [`conditions`](@ref).

## Contributing

Contributions are welcome! Please see the [GitHub repository](https://github.com/FriedmanJP/MacroEconometricModels.jl) for contribution guidelines.

## Contents

```@contents
Pages = ["getting_started.md", "method_guide.md", "data.md", "filters.md", "x13.md", "spectral.md", "arima.md", "statespace.md", "volatility.md", "nonparametric.md", "manual.md", "bayesian.md", "vecm.md", "ardl.md", "cointreg.md", "lp.md", "factormodels.md", "favar.md", "nonlinear.md", "regression.md", "binary_choice.md", "ordered_multinomial.md", "panel_reg.md", "pvar.md", "did.md", "event_study.md", "gmm.md", "dsge.md", "dsge_linear.md", "dsge_nonlinear.md", "dsge_constraints.md", "dsge_estimation.md", "dsge_hd.md", "dsge_heterogeneity.md", "dsge_ha.md", "dsge_olg.md", "dsge_continuous.md", "io.md", "io_classical.md", "io_environmental.md", "io_mrio.md", "io_baqaee_farhi.md", "io_download.md", "structural_identification.md", "nongaussian.md", "id_nongaussian.md", "id_heteroskedastic.md", "id_testing.md", "innovation_accounting.md", "ia_irf.md", "ia_fevd.md", "ia_hd.md", "nowcast.md", "nowcast_dfm.md", "nowcast_bvar.md", "nowcast_bridge.md", "nowcast_news.md", "midas.md", "forecast_evaluation.md", "tests.md", "tests_unitroot.md", "tests_unitroot_advanced.md", "tests_cointegration.md", "tests_breaks.md", "tests_panel.md", "tests_diagnostics.md", "plotting.md", "notation.md", "bibliography.md", "changelog.md", "citation.md", "api.md", "api/data.md", "api/univariate.md", "api/multivariate.md", "api/cross_section.md", "api/panel.md", "api/dsge.md", "api/structural.md", "api/gmm.md", "api/tests.md", "api/nowcasting.md", "api/visualization.md", "api/utilities.md"]
Depth = 2
```
