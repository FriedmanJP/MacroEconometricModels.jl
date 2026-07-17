<img width="1760" height="600" alt="MacroEconometricModels.jl" src="https://raw.githubusercontent.com/FriedmanJP/MacroEconometricModels.jl/main/banner.png" />

# MacroEconometricModels.jl

[![CI](https://github.com/FriedmanJP/MacroEconometricModels.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/FriedmanJP/MacroEconometricModels.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/FriedmanJP/MacroEconometricModels.jl/graph/badge.svg)](https://codecov.io/gh/FriedmanJP/MacroEconometricModels.jl)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://FriedmanJP.github.io/MacroEconometricModels.jl/stable/)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18439170.svg)](https://doi.org/10.5281/zenodo.18439170)

A comprehensive Julia package for macroeconometric research and analysis.

**Univariate:** ARIMA/ARFIMA, ARCH/GARCH family (EGARCH, GJR, IGARCH, Component-GARCH, APARCH, FIGARCH/FIEGARCH, GARCH-MIDAS), Stochastic Volatility, HP/Hamilton/BN/BK/Boosted HP filters, X-13ARIMA-SEATS seasonal adjustment, Spectral Analysis, ACF/PACF/CCF

**Nonlinear & State-Space:** Threshold/SETAR (Hansen), STAR/LSTR/ESTR (Teräsvirta), Markov-switching regression & MS-AR (Hamilton), general linear-Gaussian state-space (Kalman MLE), time-varying-parameter regression

**Multivariate:** VAR, VECM (+ Johansen β/α restriction testing), Bayesian VAR, Local Projections, Factor Models, FAVAR, Structural DFM, Multivariate GARCH (CCC/DCC/BEKK), cointegrating regression (FMOLS/CCR/DOLS), SUR/3SLS systems, MIDAS regression

**Panel:** Panel VAR (FD-GMM, System GMM, FE-OLS), Panel Regression (FE/RE/FD/Between/CRE/AB/BB + PCSE/Prais-Winsten), Panel IV (FE-IV/RE-IV/FD-IV/Hausman-Taylor), Panel Logit/Probit, Panel ARDL (PMG/MG/DFE), panel cointegrating regression (FMOLS/DOLS), Difference-in-Differences (TWFE, Callaway-Sant'Anna, Sun-Abraham, BJS, dCDH, HonestDiD), Event Study LP, LP-DiD (Dube et al. 2025)

**DSGE:** 7 solvers (Gensys, Blanchard-Kahn, Klein, 2nd/3rd-order perturbation with pruning, Chebyshev projection, PFI, VFI), model(linear) for pre-linearized models, built-in constrained solvers (Optim.jl, NLopt.jl, projected Newton, JuMP+Ipopt) with optional PATH (MCP), OccBin, GMM/SMM estimation, Bayesian estimation (SMC/SMC²/MH) with posterior mode + Laplace/bridge-sampling marginal likelihood, MCMC & identification diagnostics, prior/posterior predictive checks, order≥2 unconditional FEVD (Andreasen et al. 2018), 24-model Dynare replication suite

**Heterogeneous Agent DSGE:** Reiter, Sequence-Space Jacobian, Krusell-Smith; one-asset and two-asset HANK; continuous-time Aiyagari & two-asset HANK (HJB / Kolmogorov-Forward, Achdou et al. 2022); Blanchard (1985) perpetual-youth OLG; EGM/VFI individual solvers; Bayesian estimation

**Input-Output:** IOData container, Leontief/Ghosh models, output/income/employment multipliers, backward/forward linkages (Rasmussen) & key sectors, structural decomposition analysis, hypothetical extraction, environmental satellite accounts, Baqaee-Farhi (2019), pymrio-style MRIO downloaders (OECD/WIOD/Exiobase3/Eora26/GLORIA)

**Cross-Sectional:** OLS, WLS, IV/2SLS (+ LIML/Fuller/k-class), penalized (ridge/LASSO/elastic net), robust (Huber/bisquare/MM), Tobit/truncated, Heckman selection, Logit, Probit, Ordered Logit/Probit, Multinomial Logit (MLE), marginal effects (AME/MEM/MER)

**Nonparametric:** kernel density estimation (Sheather-Jones), Nadaraya-Watson & local-polynomial regression, LOWESS

**Forecasting:** multi-model forecasting, forecast evaluation (Diebold-Mariano, Clark-West, Mincer-Zarnowitz, encompassing), forecast combination, nowcasting (DFM/BVAR/bridge)

**Estimation:** OLS, MLE, GMM, SMM, Bayesian (Gibbs/conjugate/SMC/MH), Kalman filter/smoother

**Features:** IRF, FEVD, historical decomposition, structural identification, spectral analysis, structural-break & explosive-bubble detection, unit-root & panel unit-root tests, cointegration tests, hypothesis testing, reproducibility manifests, versioned model serialization, Tables.jl integration, interactive D3.js visualization

## Installation

```julia
using Pkg
Pkg.add("MacroEconometricModels")
```

## Features

### Univariate Models
- **Time Series Filters** - Trend-cycle decomposition:
  - Hodrick-Prescott filter
  - Hamilton (2018) regression filter
  - Beveridge-Nelson decomposition (ARIMA psi-weights)
  - Baxter-King band-pass filter
  - Boosted HP filter (Phillips & Shi 2021) with ADF/BIC/fixed stopping
- **X-13ARIMA-SEATS** - Pure-Julia port of the US Census Bureau's seasonal adjustment program:
  - X-11 decomposition (iterative moving average with Henderson trend filters)
  - SEATS decomposition (Wiener-Kolmogorov spectral signal extraction)
  - Automatic ARIMA model identification (TRAMO-style search with AICC selection)
  - Outlier detection (additive outliers, level shifts, temporary changes)
  - Calendar effects (trading day and Easter regressors)
  - Log/level transformation selection with Jacobian-corrected AICC comparison
- **ARIMA** - AR, MA, ARMA, ARIMA estimation via CSS, exact MLE (Kalman filter), or CSS-MLE
  - Automatic order selection via `auto_arima` with grid search over (p,d,q), AIC/BIC information criteria
  - Multi-step ahead forecasting with confidence intervals via psi-weight accumulation
- **ARFIMA** - Fractionally-integrated ARFIMA(p,d,q) via CSS or exact MLE (`estimate_arfima`); semiparametric long-memory estimators of the differencing parameter d: GPH log-periodogram regression (`gph_test`; Geweke & Porter-Hudak 1983) and local-Whittle (`local_whittle`; Robinson 1995)
- **Volatility Models**:
  - ARCH - Engle (1982) ARCH(q) with MLE, ARCH-LM test, Ljung-Box squared residuals
  - GARCH - GARCH(p,q), EGARCH (Nelson 1991), GJR-GARCH (Glosten, Jagannathan & Runkle 1993)
  - Extended GARCH - IGARCH (`estimate_igarch`; Engle & Bollerslev 1986), Component-GARCH (`estimate_cgarch`; Engle & Lee 1999), APARCH (`estimate_aparch`; Ding, Granger & Engle 1993)
  - Long-memory volatility - FIGARCH (`estimate_figarch`; Baillie, Bollerslev & Mikkelsen 1996) and FIEGARCH (`estimate_fiegarch`; Bollerslev & Mikkelsen 1996)
  - GARCH-MIDAS - long/short-run volatility components with a mixed-frequency secular component (`estimate_garch_midas`; Engle, Ghysels & Sohn 2013)
  - Stochastic Volatility - Bayesian SV via Kim-Shephard-Chib Gibbs sampler (basic, leverage, Student-t variants)
  - Multi-step volatility forecasts with simulation CIs
  - Diagnostics: news impact curves, persistence, half-life, unconditional variance; sign-bias test (`sign_bias_test`; Engle & Ng 1993) and Nyblom-Hansen parameter-stability test (`nyblom_test`)
- **Spectral Analysis**:
  - Periodogram, Welch method, smoothed periodogram (Daniell kernel), AR spectral estimation
  - Cross-spectrum: coherence, phase, gain functions
  - Fisher exact test and Bartlett white noise test
  - Ideal bandpass filter, transfer function visualization
- **Autocorrelation**:
  - ACF, PACF (Levinson-Durbin and OLS), CCF for cross-correlation
  - Correlogram display with cumulative Ljung-Box Q-statistics
  - Ljung-Box, Box-Pierce, and Durbin-Watson tests

### Nonlinear Time Series
- **Threshold & SETAR** - Two-regime threshold least squares and SETAR(2;p,p) with automatic delay/threshold selection (`estimate_threshold`, `estimate_setar`)
  - Hansen (1996) fixed-regressor bootstrap linearity test (`hansen_linearity_test`) and Hansen (2000) likelihood-ratio-inversion threshold confidence interval
- **Smooth-Transition (STAR)** - LSTR1, LSTR2, and ESTR transition families via NLS (`estimate_star`)
  - Luukkonen-Saikkonen-Teräsvirta (1988) LM linearity test and Teräsvirta (1994) sequential transition-family selection (`star_linearity_test`)
- **Markov-Switching** - K-state Markov-switching regression (`estimate_ms`) and mean-switching MS-AR (`estimate_ms_ar`; Hamilton 1989) via the Hamilton forward filter, Kim (1994) smoother, and EM/Optim MLE

### State-Space Models
- **General linear-Gaussian state-space** - `estimate_statespace` fits an arbitrary parameterized state-space model by prediction-error-decomposition MLE, routed through the consolidated Kalman filter + RTS smoother
- **Structural components** - `local_level` (random walk + noise) and `local_linear_trend` convenience constructors
- **Time-varying-parameter regression** - `estimate_tvp_reg` for regression coefficients following a random walk

### Multivariate Models
- **Vector Autoregression (VAR)** - OLS estimation with lag order selection (AIC, BIC, HQ)
- **Bayesian VAR (BVAR)** - Minnesota priors with hyperparameter optimization (Giannone, Lenza & Primiceri 2015)
- **Vector Error Correction Model (VECM)** - Johansen MLE and Engle-Granger two-step estimation
  - Automatic cointegrating rank selection (trace/max-eigenvalue)
  - VAR conversion (`to_var`) enabling all 13 statistical identification methods
  - VECM-specific forecasting preserving cointegrating relationships
  - Granger causality: short-run, long-run, and strong tests
  - Johansen LR restriction testing on the cointegrating structure: `test_beta_restriction`, `test_alpha_restriction`, `test_weak_exogeneity`, `test_known_beta`, `test_joint_restriction`
- **Local Projections (LP)** - Jorda (2005) with extensions:
  - HAC standard errors (Newey-West, White, Driscoll-Kraay)
  - Instrumental Variables (Stock & Watson 2018)
  - Smooth IRF via B-splines (Barnichon & Brownlees 2019)
  - State-dependent LP (Auerbach & Gorodnichenko 2013)
  - Propensity Score Matching (Angrist et al. 2018)
  - Structural LP with multi-shock IRFs (Plagborg-Møller & Wolf 2021)
  - Direct multi-step forecasting with analytical/bootstrap CIs
- **Factor Models**
  - Static factors via PCA with Bai-Ng information criteria (IC1, IC2, IC3)
  - Dynamic Factor Models (two-step and EM estimation)
  - Generalized Dynamic Factor Models (spectral methods, Forni et al. 2000)
  - Unified forecasting with theoretical (analytical) and bootstrap confidence intervals for all three factor model types
- **FAVAR** - Factor-Augmented VAR (Bernanke, Boivin & Eliasz 2005):
  - Two-step estimation (PCA + VAR) and Bayesian Gibbs (Carter-Kohn smoother + NIW)
  - `favar_panel_irf` maps factor IRFs to N observables via loadings
- **Structural DFM** - Structural dynamic factor model wrapping GDFM + VAR for identified factor shocks
  - `sdfm_panel_irf` maps structural factor IRFs to all N observable panel variables via loadings
- **Multivariate GARCH** - Conditional-covariance models for vector return series (`estimate_ccc`, `estimate_dcc`, `estimate_bekk`)
  - CCC-GARCH (Bollerslev 1990), DCC-GARCH with optional cDCC correction (Engle 2002; Aielli 2013), scalar/diagonal BEKK(1,1) (Engle & Kroner 1995)
  - Two-step QMLE reusing univariate GARCH margins; multi-step covariance forecasting; `covariances`/`correlations`/`variances` accessors
- **Cointegrating Regression** - Single-equation estimators of a cointegrating vector (`estimate_cointreg`)
  - Fully-modified OLS (Phillips & Hansen 1990), canonical cointegrating regression (Park 1992), dynamic OLS (Saikkonen 1991; Stock & Watson 1993)
- **MIDAS Regression** - Mixed-data-sampling regression of a low-frequency target on high-frequency predictors (`estimate_midas`)
  - Exponential-Almon, Beta (2/3-param), polynomial-Almon, and unrestricted U-MIDAS weighting; ADL-MIDAS with autoregressive lags; direct multi-horizon forecasting (Ghysels, Sinko & Valkanov 2007; Foroni, Marcellino & Schumacher 2015)

### Systems of Equations
- **Seemingly Unrelated Regressions (SUR)** - Feasible/iterated GLS (converging to Gaussian MLE) with McElroy (1977) system R² (`estimate_sur`; Zellner 1962)
- **Three-Stage Least Squares (3SLS)** - Zellner-Theil system IV with common or per-equation instruments (`estimate_3sls`; Zellner & Theil 1962)
- Built-in `load_example(:grunfeld)` panel for classic SUR demonstrations

### Nonparametric Methods
- **Kernel density estimation** - `kernel_density` with Gaussian/Epanechnikov/triangular/uniform kernels and Sheather-Jones (1991) plug-in bandwidth
- **Kernel & local-polynomial regression** - `kernel_reg` (degree 0 → Nadaraya-Watson, degree ≥ 1 → Fan-Gijbels local polynomial)
- **LOWESS** - Cleveland (1979) tricube-weighted robust local-linear scatterplot smoother (`lowess`)

### Cross-Sectional Models
- **Linear Regression** - OLS with HC0–HC3 robust and cluster-robust standard errors
  - Weighted Least Squares (WLS) with analytic or user-supplied weights
  - Variance Inflation Factor (VIF) for multicollinearity diagnostics
- **Instrumental Variables** - IV/2SLS estimation with first-stage F-statistic and Sargan overidentification test
  - k-class estimators: LIML, Fuller-modified LIML, and generic k-class via `estimate_iv(...; method=:liml/:fuller/:kclass)` (Fuller 1977)
- **Penalized Regression** - Ridge, LASSO, and elastic net with coordinate descent and a cross-validated regularization path (`estimate_ridge`, `estimate_lasso`, `estimate_elastic_net`; Hoerl & Kennard 1970; Tibshirani 1996; Zou & Hastie 2005)
- **Robust Regression** - Huber and Tukey-bisquare M-estimators and Yohai (1987) MM-estimation (`estimate_robust`; Huber 1964)
- **Censored & Truncated** - Tobit (`estimate_tobit`; Tobin 1958, Olsen 1978 reparameterization) and truncated regression (`estimate_truncreg`) with McDonald-Moffitt marginal effects
- **Sample Selection** - Heckman two-step and full-information MLE selection model (`estimate_heckman`; Heckman 1979)
- **Variable Selection** - Forward/backward/bidirectional stepwise, best-subset, and LSE general-to-specific (GETS) (`select_variables`)
- **Regression Diagnostics** - OLS residual tests: White, Breusch-Pagan, Glejser, Harvey, Breusch-Godfrey, and Ramsey RESET (`white_test`, `breusch_pagan_test`, `glejser_test`, `harvey_test`, `breusch_godfrey_test`, `reset_test`); stability & influence: recursive residuals, CUSUM/CUSUMSQ, Chow, and influence measures (`recursive_residuals`, `cusum_test`, `cusumsq_test`, `chow_test`, `influence_stats`; Brown, Durbin & Evans 1975)
- **Binary Choice** - Logit and Probit MLE via IRLS (Fisher scoring)
  - Marginal effects: average (AME), at-means (MEM), at-representative (MER) with delta-method SEs
  - `odds_ratio()`, `classification_table()`, McFadden/AIC/BIC fit statistics
- **Ordered Choice** - Ordered Logit and Ordered Probit MLE with cut-point estimation
  - Marginal effects (AME/MEM/MER) per outcome category with delta-method SEs
  - Brant (1990) test for parallel regression assumption
- **Multinomial Logit** - MLE with IIA assumption
  - Marginal effects per alternative with delta-method SEs
  - Hausman-McFadden IIA test
- **CrossSectionData** container with `diagnose()` / `fix()` and direct estimation dispatch

### Panel Models
- **Panel VAR (PVAR)** - GMM estimation for dynamic panel data:
  - First-difference GMM (Arellano & Bond 1991) and System GMM (Blundell & Bond 1998)
  - Forward orthogonal deviations (Helmert transform) as alternative to first-differencing
  - Fixed-Effects OLS within estimator
  - Windmeijer (2005) finite-sample corrected standard errors for two-step GMM
  - Specification tests: Hansen J-test, Andrews-Lu (2001) MMSC for lag/moment selection
  - Structural analysis: OIRF, GIRF (Pesaran & Shin 1998), FEVD, stability
  - Group-level block bootstrap confidence intervals for IRFs
  - Instrument management: min/max lag truncation, collapse, PCA reduction
- **Panel Regression** - `estimate_xtreg` unified dispatcher for linear panel models:
  - Fixed Effects (within estimator), Random Effects (GLS), First Differences
  - Between estimator, Correlated Random Effects (Mundlak/Chamberlain)
  - Dynamic panels: Arellano-Bond (1991) and Blundell-Bond (1998) GMM
  - Covariance estimators: conventional, robust (HC1), cluster-robust, Driscoll-Kraay, and Beck-Katz panel-corrected SEs (PCSE; `cov_type=:pcse`)
  - Prais-Winsten AR(1) FGLS quasi-differencing (`ar1=:common`)
  - Specification tests: Hausman FE vs RE, Breusch-Pagan LM, Pesaran CD, Wooldridge AR(1), Modified Wald
- **Panel IV** - `estimate_xtiv` for instrumental variables in panel data:
  - FE-IV, RE-IV, FD-IV, Hausman-Taylor estimator
  - First-stage F-statistics, Sargan-Hansen overidentification test
- **Panel Discrete Choice** - `estimate_xtlogit` / `estimate_xtprobit`:
  - Pooled, Fixed Effects (conditional logit), Random Effects (Gauss-Hermite quadrature)
  - Correlated Random Effects (Mundlak projection)
  - Panel marginal effects with delta-method SEs
- **Difference-in-Differences (DiD)** - Unified `estimate_did()` dispatcher for staggered treatment designs:
  - TWFE event-study regression with double-demeaned panel fixed effects
  - Callaway & Sant'Anna (2021) group-time ATT with doubly robust estimation
  - Sun & Abraham (2021) interaction-weighted estimator with cohort-specific regressions
  - Borusyak, Jaravel & Spiess (2024) imputation estimator via counterfactual prediction
  - de Chaisemartin & D'Haultfoeuille (2020) first-difference DID with bootstrap SEs
  - Diagnostics: Bacon decomposition (Goodman-Bacon 2021), pre-trend testing, negative weight checks
  - Honest DiD sensitivity analysis: relative-magnitudes and second-difference restriction sets with Armstrong-Kolesár FLCIs and breakdown values (Rambachan & Roth 2023)
- **Event Study LP** - Local projection event study for panel data:
  - Switching indicator treatment with time-only FE (Acemoglu et al. 2019)
  - Flexible lead/lag window specification with clustered standard errors
  - Interactive D3.js event-study plots with confidence bands
- **LP-DiD** - Full-featured LP-DiD estimator (Dube, Girardi, Jordà & Taylor 2025):
  - Clean control samples (CCS): absorbing, non-absorbing, one-off treatment
  - Pre-mean differencing (PMD), IPW reweighting, nocomp restriction
  - Pooled post-treatment and pre-treatment estimates
  - `panel_lag`, `panel_lead`, `panel_diff` for within-group transformations
- **Panel ARDL** - Heterogeneous dynamic panels via `estimate_pmg`: Pooled Mean Group, Mean Group, and Dynamic Fixed Effects (`method=:pmg/:mg/:dfe`) with a generalized Hausman selection test (Pesaran, Shin & Smith 1999; Pesaran & Smith 1995)
- **Panel Cointegrating Regression** - Panel FMOLS/DOLS with group-mean (between) or pooled (within) dimensions (`estimate_xtcointreg`; Pedroni 2000, 2001; Kao & Chiang 2000)

### DSGE
- **Model specification** - `@dsge` macro with declarative syntax for parameters, variables, shocks, and equilibrium equations
- **Steady state** - Numerical solver (Newton's method) or analytical closed-form; built-in constrained solvers (Optim.jl box constraints, NLopt.jl nonlinear inequalities, JuMP+Ipopt NLP) with an optional PATH backend
- **Linear solvers** - Gensys (Sims 2002), Blanchard-Kahn (1980), Klein (2000) via unified `solve(spec; method=...)` interface
- **Higher-order perturbation** - 2nd-order (Schmitt-Grohe & Uribe 2004) and 3rd-order with Andreasen, Fernandez-Villaverde & Rubio-Ramirez (2018) pruned simulation; Kim et al. (2008) 2nd-order pruning
- **Global methods** - Chebyshev collocation (tensor/Smolyak grids, Gauss-Hermite quadrature; Judd 1998); Policy Function Iteration (Coleman 1990, Rendahl 2017); Value Function Iteration (Stokey, Lucas & Prescott 1989) with Howard improvement steps and Anderson acceleration (Walker & Ni 2011)
  - Opt-in multi-threading for VFI, PFI, and collocation grid evaluation
- **Constrained solvers** - Built-in projected Newton (box-constrained PF), Optim.jl Fminbox (box-constrained SS), NLopt.jl SLSQP (nonlinear inequalities), and JuMP+Ipopt (NLP); optional PATH (MCP) backend
- **Perfect foresight** - Newton solver on stacked system with block-tridiagonal Jacobian; built-in box and nonlinear constraint support
- **OccBin** - Occasionally binding constraints via piecewise-linear regime switching (Guerrieri & Iacoviello 2015)
- **Pre-linearized models** - `model(linear)` support via `DSGESpec(... ; linear=true)` for Dynare-style pre-linearized models (e.g., Smets-Wouters 2007); automatic zero steady state, gensys constant handling in Kalman filter
- **Simulation & IRF** - `simulate`, `irf`, `fevd` for linear, pruned higher-order, and projection solutions; `fevd(sol, H; unconditional=true)` for order≥2 asymptotic FEVD via Andreasen et al. (2018) augmented Lyapunov with per-shock variance decomposition; Bayesian posterior credible bands (dual 68%/90%) via `irf(::BayesianDSGE)`, `fevd(::BayesianDSGE)`, `simulate(::BayesianDSGE)`
- **Historical decomposition** - `historical_decomposition(sol, data, observables)` for linear (Kalman/RTS smoother), nonlinear (FFBSi particle smoother + counterfactual), and Bayesian (posterior draws) DSGE models; standalone `dsge_smoother` and `dsge_particle_smoother`
- **Analytical moments** - Order 1: Lyapunov equation for unconditional covariance; Order ≥2: Andreasen et al. (2018) augmented state-space Lyapunov for means, variances, and autocovariances; `analytical_moments` for both
- **GMM Estimation** - IRF matching, Euler equation GMM, SMM, analytical GMM via `estimate_dsge`
- **Bayesian Estimation** - Sequential Monte Carlo (SMC with adaptive tempering), SMC² with particle filter likelihood, random-walk Metropolis-Hastings; delayed acceptance for accelerated sampling; nonlinear particle filter for higher-order solutions via `estimate_dsge_bayes`
- **Posterior mode & marginal likelihood** - `posterior_mode` maximizes the log posterior (optionally in unconstrained space) and returns the inverse Hessian (reusable as an RWMH proposal) and the Laplace marginal likelihood; `bridge_sampling_ml` computes a bridge-sampling marginal likelihood from stored draws (Meng & Wong 1996)
- **MCMC convergence diagnostics** - `mcmc_diagnostics` reports rank-normalized split-R-hat, bulk/tail ESS, and Geweke z-statistics; `trace` and `acf` accessors expose per-parameter draw sequences (Vehtari et al. 2021; Geweke 1992)
- **Identification diagnostics** - Iskrev (2010) rank test (`identification_diagnostics`), Koop-Pesaran-Smith learning-rate test (`learning_rate_check`), and prior/posterior overlap (`prior_posterior_overlap`)
- **Predictive checks** - Prior predictive simulation (`prior_predictive`) and posterior predictive checks with per-statistic predictive p-values (`posterior_predictive_check`; Gelman, Meng & Stern 1996)
- **Sampler infrastructure** - Bijective parameter transforms with Jacobian correction for unconstrained sampling (`to_unconstrained`/`to_constrained`) and Dynare prior-convention shims (`dynare_prior`, `InverseGamma1`)
- **Dynare replication** - 24-model replication suite (`test/dynare_replication/`) with automated steady-state, IRF, variance decomposition, and theoretical moment comparison against Dynare 6.5+ reference values; includes Smets-Wouters (2007) full estimation pipeline

### Heterogeneous Agent DSGE
- **Model specification** - `@dsge` macro with `heterogeneous:`, `idiosyncratic:`, `aggregation:` blocks for declaring individual state space, income process, and market clearing conditions
- **Built-in examples** - `load_ha_example(:krusell_smith)`, `load_ha_example(:one_asset_hank)`, `load_ha_example(:two_asset_hank)` with published calibrations
- **Individual solvers**:
  - Endogenous Grid Method (Carroll 2006) for one-asset models with borrowing constraints
  - Nested EGM for two-asset HANK (Kaplan, Moll & Violante 2018) with portfolio adjustment costs
  - Value Function Iteration with Howard (1960) improvement steps as fallback
- **Distribution tracking** - Young (2010) non-stochastic histogram method with sparse transition matrices and power iteration for stationary distribution
- **Income discretization** - Rouwenhorst (1995) and Tauchen (1986) methods for AR(1) processes
- **Steady state** - Bisection on the interest rate iterating EGM + distribution + market clearing until capital supply equals demand
- **Solution methods**:
  - Sequence-Space Jacobian (Auclert, Bardóczy, Rognlie & Straub 2021) with fake news algorithm and Ho-Kalman state-space reduction
  - Reiter (2009) linearization with observability-based SVD dimensionality reduction
  - Krusell-Smith (1998) bounded rationality via perceived law of motion simulation
- **Bayesian estimation** - `estimate_dsge_bayes(spec::HADSGESpec, ...)` with adaptive RWMH; re-solves HA steady state + linearizes at each draw; Kalman filter on reduced system
- **Continuous-time methods** - Continuous-time Aiyagari solved by implicit upwind finite-difference HJB (`ct_hjb`) with a Kolmogorov-Forward stationary distribution (`ct_kfe`), steady state (`ct_steady_state`), and MIT-shock transition dynamics (`ct_mit_shock`); a two-asset HANK household block with convex deposit-adjustment costs (`ct_two_asset_solve`) (Achdou et al. 2022; Kaplan, Moll & Violante 2018)
- **Overlapping generations** - Blanchard (1985) perpetual-youth OLG: steady state (`blanchard_steady_state`), full solution (`blanchard_solve`), and transition-path dynamics (`blanchard_transition`)
- **Analysis** - `irf`, `fevd`, `simulate` dispatch via embedded `DSGESolution`; `distribution_irf` for wealth distribution dynamics; `inequality_irf` for Gini/percentile responses; `simulate_panel` for individual-level data
- **Visualization** - `plot_result(ss; view=:distribution)` (wealth histogram), `:lorenz` (Lorenz curve with Gini), `:policy` (consumption and savings functions by income state)

### Input-Output Analysis
- **`IOData` container** - Symmetric input-output tables (intermediate flows, final demand, value added); `load_example(:wiot)` loads the Miller & Blair 2-sector table
- **Leontief & Ghosh models** - Technical coefficients `A` and Leontief inverse `L` (demand-driven); allocation coefficients `B` and Ghosh inverse `G` (supply-driven)
- **Multipliers** - Output, income, and employment multipliers (Type I and Type II with household endogenization)
- **Linkages** - Backward and forward linkages, Rasmussen (1956) power/sensitivity-of-dispersion indices, key-sector classification
- **Structural decomposition analysis (SDA)** - Additive two-polar and multiplicative decomposition of output change between two periods
- **Hypothetical extraction** - Backward/forward/total linkage extraction quantifying a sector's importance
- **Environmental extensions** - Satellite accounts via `add_extension!`; direct/total intensities, emission multipliers, and consumption-based footprints
- **Baqaee & Farhi (2019)** - Nonlinear production-network model: Domar weights (Hulten first-order), second-order Hessian term, Cobb-Douglas (θ=σ=1) special case
- **MRIO downloaders** - pymrio-style `download_io(:oecd/:wiod/:exiobase3/:eora26/:gloria)` on the `Downloads` stdlib + URL registry; `parse_io` reads CSV/TSV in-core and `.zip`/`.xlsx` via `ZipFile`/`XLSX` package extensions

### GMM
- **Generalized Method of Moments** - One-step, two-step, and iterated; Hansen J-test
- **Simulated Method of Moments (SMM)** - Simulation-based estimation with HAC-optimal weighting, two-step and iterated
- **Linear GMM** - Closed-form solver for panel IV estimation
- **Sandwich covariance** - Robust GMM variance with Windmeijer correction

### Structural Identification
- **Cholesky** (recursive) - Wold causal ordering (Sims 1980)
- **Sign restrictions** - Rotation-based identification (Rubio-Ramirez et al. 2010)
- **Narrative restrictions** - Historical event constraints on shocks (Antolin-Diaz & Rubio-Ramirez 2018)
- **Long-run restrictions** - Permanent/transitory decomposition (Blanchard & Quah 1989)
- **Zero and sign restrictions** - Joint zero+sign with importance sampling (Arias et al. 2018)
- **Penalty function** - Frequentist sign and zero restrictions via constrained optimization (Mountford & Uhlig 2009)

### Innovation Accounting
- **Impulse Response Functions (IRF)** - Bootstrap, theoretical, and Bayesian credible intervals
- **Forecast Error Variance Decomposition (FEVD)** - Frequentist and Bayesian
- **LP-FEVD** - R², LP-A, LP-B estimators with bootstrap CIs (Gorodnichenko & Lee 2019)
- **Historical Decomposition (HD)** - Decompose observed movements into structural shock contributions
- **Summary Tables** - Publication-quality output with `report()`, `table()`, `print_table()`
- **Display Backends** - Switch between text, LaTeX, and HTML table output with `set_display_backend()`
- **Bibliographic References** - `refs(model)` outputs AEA-style citations in text, LaTeX, BibTeX, or HTML

### Nowcasting
- **Dynamic Factor Model (DFM)** - EM algorithm with Kalman smoother for mixed-frequency data (Banbura & Modugno 2014)
  - Arbitrary missing patterns and ragged edges
  - Mariano-Murasawa [1 2 3 2 1] temporal aggregation for quarterly variables
  - Block factor structure, AR(1)/IID idiosyncratic components
- **Large Bayesian VAR** - GLP-style Normal-Inverse-Wishart prior (Cimadomo et al. 2022)
  - Hyperparameter optimization via marginal log-likelihood maximization
  - Minnesota shrinkage with sum-of-coefficients and co-persistence priors
- **Bridge Equations** - OLS regressions combining pairs of monthly indicators via median (Banbura et al. 2023)
- **News Decomposition** - Attribute nowcast revisions to individual data releases with group aggregation and named groups
- **Panel Balancing** - `balance_panel()` fills NaN in TimeSeriesData/PanelData using DFM imputation

### Forecast Evaluation & Combination
- **Accuracy metrics** - `forecast_evaluate` reports ME, MAE, RMSE, MAPE, sMAPE, MASE, and Theil U1/U2
- **Predictive-accuracy tests** - Diebold-Mariano with Harvey-Leybourne-Newbold small-sample correction (`diebold_mariano`; 1995), Clark-West nested-model adjusted-MSPE test (`clark_west`; 2007), Mincer-Zarnowitz efficiency regression (`mincer_zarnowitz`; 1969), and forecast encompassing (`forecast_encompassing`)
- **Combination** - Equal-weight, Bates-Granger inverse-MSE, and Granger-Ramanathan constrained-least-squares combination (`combine_forecasts`; Bates & Granger 1969; Granger & Ramanathan 1984)

### Statistical Identification via Higher Moments
- **Heteroskedasticity-based** - Markov-switching (Lanne & Lütkepohl 2008), GARCH (Normandin & Phaneuf 2004), smooth-transition (Lütkepohl & Netšunajev 2017), external volatility (Rigobon 2003)
- **Non-Gaussian ICA** - FastICA, JADE, SOBI, distance covariance, HSIC (Hyvärinen et al. 2010, Matteson & Tsay 2017)
- **Non-Gaussian ML** - Student-t, mixture-normal, pseudo-ML, skew-normal (Lanne et al. 2017, Gourieroux et al. 2017)
- **Multivariate normality tests** - Jarque-Bera, Mardia, Doornik-Hansen, Henze-Zirkler
- **Identifiability diagnostics** - Shock gaussianity, independence, identification strength, LR tests
- Seamless integration: `irf(model, 20; method=:fastica)` works out of the box
- See Lewis (2025) for a comprehensive review

### Hypothesis Tests
- **Unit Root Tests** - ADF, KPSS, Phillips-Perron, Zivot-Andrews, Ng-Perron (MZa, MZt, MSB, MPT)
- **Advanced Unit Root** - Fourier ADF/KPSS (Enders & Lee 2012), DF-GLS and ERS point-optimal (`dfgls_test`, `ers_test`; Elliott, Rothenberg & Stock 1996), LM unit root with 0/1/2 breaks (Lee & Strazicich 2003, 2013), two-break ADF (Narayan & Popp 2010)
- **Seasonal Unit Root** - HEGY test for unit roots at seasonal frequencies (`hegy_test`; Hylleberg, Engle, Granger & Yoo 1990)
- **Cointegration** - Johansen test (trace and max-eigenvalue), Gregory-Hansen (1996) test with structural break (level shift, trend, regime), and residual-based tests: Engle-Granger, Phillips-Ouliaris, Hansen L_c instability, and Park added-variable (`engle_granger_test`, `phillips_ouliaris_test`, `hansen_instability_test`, `park_added_test`)
- **Explosive Bubbles** - SADF and generalized supADF (GSADF) right-tailed tests with BSADF date-stamping (`sadf_test`, `gsadf_test`; Phillips, Wu & Yu 2011; Phillips, Shi & Yu 2015)
- **Structural Breaks** - Andrews (1993) SupWald/SupLM/SupLR with 9 test variants; Bai-Perron (1998) multiple break detection via dynamic programming with BIC/LWZ/sequential selection; factor break tests — Breitung-Eickmeier (2011), Chen-Dolado-Gonzalo (2014), Han-Inoue (2015)
- **Panel Unit Root** - First-generation tests: Levin-Lin-Chu (`llc_test`), Im-Pesaran-Shin (`ips_test`), Breitung (`breitung_panel_test`), Fisher/Maddala-Wu (`fisher_panel_test`), and Hadri (`hadri_test`); second-generation Bai-Ng (2004) PANIC, Pesaran (2007) CIPS, and Moon-Perron (2004) factor-adjusted tests; `panel_unit_root_summary()` battery
- **Panel Cointegration** - Pedroni (1999, 2004), Kao (1999), Westerlund (2007), and Fisher-Johansen combined tests (`pedroni_test`, `kao_test`, `westerlund_test`, `fisher_johansen_test`)
- **Panel Granger** - Dumitrescu-Hurlin (2012) heterogeneous panel non-causality test (`dh_causality_test`)
- **Granger Causality** - Pairwise and block Wald tests, all-pairs matrix
- **Normality** - Jarque-Bera, Mardia multivariate, Doornik-Hansen, Henze-Zirkler, Royston; unified `normality_test_suite()`
- **Portmanteau Tests** - Ljung-Box, Box-Pierce autocorrelation tests; Durbin-Watson test for first-order serial correlation
- **ARCH Diagnostics** - ARCH-LM test, Ljung-Box on squared residuals
- **Panel VAR** - Hansen J-test for overidentifying restrictions, Andrews-Lu MMSC for lag/moment selection
- **Model Comparison** - Likelihood ratio (LR) and Lagrange multiplier (LM/score) tests for nested models
- **Independence** - BDS nonlinear-dependence test (`bds_test`; Brock, Dechert, Scheinkman & LeBaron 1996)
- **Variance-Ratio / Random Walk** - Lo-MacKinlay, Chow-Denning joint, Wright rank/sign, and Kim wild-bootstrap variants (`variance_ratio_test`; Lo & MacKinlay 1988)
- **Goodness-of-Fit** - EDF battery: Kolmogorov-Smirnov, Lilliefors, Cramér-von Mises, Anderson-Darling, and Watson (`edf_test`)
- **Two-Sample & Rank** - Equality-of-distribution battery (Wilcoxon/Mann-Whitney, Kruskal-Wallis, van der Waerden, Levene/Bartlett; `equality_test`, plus `ttest`/`anova_test`) and Pearson/Spearman/Kendall rank-correlation tests (`cor_test`)
- **Long-Run Variance** - Kernel HAC and VARHAC long-run (co)variance toolkit with Andrews (1991) and Newey-West (1994) automatic bandwidth (`lrvar`, `lrcov`, `lrcov_oneside`, `varhac`)
- **Stationarity diagnostics** - `unit_root_summary()`, `test_all_variables()`

### Visualization
- **Interactive D3.js plots** - `plot_result()` renders self-contained HTML with inline D3.js v7 (no additional dependencies)
  - 78 dispatch methods covering IRF, FEVD, historical decomposition, filters, forecasts, volatility models (incl. multivariate GARCH), factor models, data containers, nowcasting, regression, nonlinear time series, nonparametric fits, MIDAS/ARDL, input-output, and difference-in-differences
  - Four chart types: line (with confidence bands), stacked area, bar, and heatmap
  - Interactive tooltips, responsive layout, multi-panel grid figures
  - Nowcast views: `view=:default` (+ DFM factor panels), `:heatmap` (z-score ragged edge), `:contributions` (group stacked bar); news views: `:releases`, `:groups`, `:individual`
  - `save_plot(p, "file.html")` saves to disk; `display_plot(p)` opens in browser; auto-renders in Jupyter
  - Common kwargs: `var`, `shock`, `title`, `save_path`, `ncols`, `view`

### Reproducibility & Interoperability
- **Reproducibility manifests** - `capture_manifest` records the RNG seed, thread count, Julia/package/dependency versions, OS, UTC timestamp, and package git SHA + dirty flag; bootstrap IRFs and BVAR posteriors carry a `ReproManifest`, and `reproduce()` re-runs from the stored seed and reports a bit-for-bit `ReproReport` (with a thread-count caveat)
- **Versioned serialization** - `save_model` / `load_model` write a self-describing, version-tagged container (via an optional JLD2 backend) that survives package upgrades; `load_model` raises a typed `SerializationError` on a format/type mismatch
- **Tables.jl integration** - Coefficient-bearing result types are Tables.jl column sources, so `DataFrame(result)` works with no hard DataFrames dependency; `long_table` gives tidy views of array-valued results (IRF/FEVD/forecasts) and `write_csv` exports via stdlib
- **Structured logging** - Library diagnostics route through the `Logging` stdlib (quiet by default); `set_log_level` and `with_min_level` control verbosity

### Data Management
- **Typed containers** - `TimeSeriesData`, `PanelData`, `CrossSectionData` with variable names, frequency, transformation codes, and descriptions
- **Built-in datasets** - FRED-MD (126 monthly variables), FRED-QD (245 quarterly variables), Penn World Table (38 OECD countries, 1950–2023), DDCG democracy-GDP (184 countries, 1960–2010; Acemoglu et al. 2019), mpdta minimum wage panel (500 US counties, 2003–2007; Callaway & Sant'Anna 2021), Grunfeld (1958) investment panel, Mroz (1987) labor-supply cross-section, the Nile flow series, and the WIOT input-output table (`load_example(:wiot)`)
- **Data diagnostics** - `diagnose()` scans for NaN/Inf/constant columns; `fix()` cleans via listwise deletion, interpolation, or mean imputation
- **FRED transformations** - `apply_tcode()` / `inverse_tcode()` for all 7 FRED transformation codes
- **Filtering** - `apply_filter()` applies HP, Hamilton, BN, BK, or boosted HP per-variable to `TimeSeriesData` and `PanelData`
- **Panel support** - `xtset()` for Stata-style panel construction; balanced/unbalanced detection
- **Summary statistics** - `describe_data()` with per-variable N, mean, std, quantiles, skewness, kurtosis
- **Seamless estimation** - All estimation functions accept `TimeSeriesData` directly

## Documentation

Full documentation available at [https://FriedmanJP.github.io/MacroEconometricModels.jl/dev/](https://FriedmanJP.github.io/MacroEconometricModels.jl/dev/)

All documentation code examples execute during the build — `report()` output, estimation tables, and test statistics appear directly on each page.

## References

### Time Series Filters

- Baxter, Marianne, and Robert G. King. 1999. "Measuring Business Cycles: Approximate Band-Pass Filters for Economic Time Series." *Review of Economics and Statistics* 81 (4): 575–593. [https://doi.org/10.1162/003465399558454](https://doi.org/10.1162/003465399558454)
- Beveridge, Stephen, and Charles R. Nelson. 1981. "A New Approach to Decomposition of Economic Time Series into Permanent and Transitory Components with Particular Attention to Measurement of the 'Business Cycle'." *Journal of Monetary Economics* 7 (2): 151–174. [https://doi.org/10.1016/0304-3932(81)90040-4](https://doi.org/10.1016/0304-3932(81)90040-4)
- Hamilton, James D. 2018. "Why You Should Never Use the Hodrick-Prescott Filter." *Review of Economics and Statistics* 100 (5): 831–843. [https://doi.org/10.1162/rest_a_00706](https://doi.org/10.1162/rest_a_00706)
- Hodrick, Robert J., and Edward C. Prescott. 1997. "Postwar U.S. Business Cycles: An Empirical Investigation." *Journal of Money, Credit and Banking* 29 (1): 1–16. [https://doi.org/10.2307/2953682](https://doi.org/10.2307/2953682)
- Phillips, Peter C. B., and Zhentao Shi. 2021. "Boosting: Why You Can Use the HP Filter." *International Economic Review* 62 (2): 521–570. [https://doi.org/10.1111/iere.12495](https://doi.org/10.1111/iere.12495)

### X-13ARIMA-SEATS

- Dagum, Estela Bee, and Silvia Bianconcini. 2016. *Seasonal Adjustment Methods and Real Time Trend-Cycle Estimation*. Springer. [https://doi.org/10.1007/978-3-319-31822-6](https://doi.org/10.1007/978-3-319-31822-6)
- Findley, David F., Brian C. Monsell, William R. Bell, Mark C. Otto, and Bor-Chung Chen. 1998. "New Capabilities and Methods of the X-12-ARIMA Seasonal-Adjustment Program." *Journal of Business and Economic Statistics* 16 (2): 127–152. [https://doi.org/10.1080/07350015.1998.10524743](https://doi.org/10.1080/07350015.1998.10524743)
- Gómez, Víctor, and Agustín Maravall. 1996. "Programs TRAMO and SEATS: Instructions for the User." *Banco de España Working Papers* 9628.

### ARIMA

- Box, George E. P., and Gwilym M. Jenkins. 1970. *Time Series Analysis: Forecasting and Control*. San Francisco: Holden-Day. ISBN 978-0-8162-1094-7.
- Hyndman, Rob J., and Yeasmin Khandakar. 2008. "Automatic Time Series Forecasting: The forecast Package for R." *Journal of Statistical Software* 27 (3): 1–22. [https://doi.org/10.18637/jss.v027.i03](https://doi.org/10.18637/jss.v027.i03)

### Volatility Models

- Bollerslev, Tim. 1986. "Generalized Autoregressive Conditional Heteroskedasticity." *Journal of Econometrics* 31 (3): 307–327. [https://doi.org/10.1016/0304-4076(86)90063-1](https://doi.org/10.1016/0304-4076(86)90063-1)
- Engle, Robert F. 1982. "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation." *Econometrica* 50 (4): 987–1007. [https://doi.org/10.2307/1912773](https://doi.org/10.2307/1912773)
- Glosten, Lawrence R., Ravi Jagannathan, and David E. Runkle. 1993. "On the Relation Between the Expected Value and the Volatility of the Nominal Excess Return on Stocks." *Journal of Finance* 48 (5): 1779–1801. [https://doi.org/10.1111/j.1540-6261.1993.tb05128.x](https://doi.org/10.1111/j.1540-6261.1993.tb05128.x)
- Nelson, Daniel B. 1991. "Conditional Heteroskedasticity in Asset Returns: A New Approach." *Econometrica* 59 (2): 347–370. [https://doi.org/10.2307/2938260](https://doi.org/10.2307/2938260)
- Taylor, Stephen J. 1986. *Modelling Financial Time Series*. Chichester: Wiley. ISBN 978-0-471-90993-4.

### VAR and Structural Identification

- Arias, Jonas E., Juan F. Rubio-Ramírez, and Daniel F. Waggoner. 2018. "Inference Based on Structural Vector Autoregressions Identified with Sign and Zero Restrictions: Theory and Applications." *Econometrica* 86 (2): 685–720. [https://doi.org/10.3982/ECTA14468](https://doi.org/10.3982/ECTA14468)
- Antolín-Díaz, Juan, and Juan F. Rubio-Ramírez. 2018. "Narrative Sign Restrictions for SVARs." *American Economic Review* 108 (10): 2802–2829. [https://doi.org/10.1257/aer.20161852](https://doi.org/10.1257/aer.20161852)
- Mountford, Andrew, and Harald Uhlig. 2009. "What Are the Effects of Fiscal Policy Shocks?" *Journal of Applied Econometrics* 24 (6): 960–992. [https://doi.org/10.1002/jae.1079](https://doi.org/10.1002/jae.1079)
- Blanchard, Olivier Jean, and Danny Quah. 1989. "The Dynamic Effects of Aggregate Demand and Supply Disturbances." *American Economic Review* 79 (4): 655–673.
- Kilian, Lutz, and Helmut Lütkepohl. 2017. *Structural Vector Autoregressive Analysis*. Cambridge: Cambridge University Press. [https://doi.org/10.1017/9781108164818](https://doi.org/10.1017/9781108164818)
- Lütkepohl, Helmut. 2005. *New Introduction to Multiple Time Series Analysis*. Berlin: Springer. ISBN 978-3-540-40172-8.
- Rubio-Ramírez, Juan F., Daniel F. Waggoner, and Tao Zha. 2010. "Structural Vector Autoregressions: Theory of Identification and Algorithms for Inference." *Review of Economic Studies* 77 (2): 665–696. [https://doi.org/10.1111/j.1467-937X.2009.00578.x](https://doi.org/10.1111/j.1467-937X.2009.00578.x)
- Sims, Christopher A. 1980. "Macroeconomics and Reality." *Econometrica* 48 (1): 1–48. [https://doi.org/10.2307/1912017](https://doi.org/10.2307/1912017)

### Bayesian Methods

- Bańbura, Marta, Domenico Giannone, and Lucrezia Reichlin. 2010. "Large Bayesian Vector Auto Regressions." *Journal of Applied Econometrics* 25 (1): 71–92. [https://doi.org/10.1002/jae.1137](https://doi.org/10.1002/jae.1137)
- Giannone, Domenico, Michele Lenza, and Giorgio E. Primiceri. 2015. "Prior Selection for Vector Autoregressions." *Review of Economics and Statistics* 97 (2): 436–451. [https://doi.org/10.1162/REST_a_00483](https://doi.org/10.1162/REST_a_00483)
- Litterman, Robert B. 1986. "Forecasting with Bayesian Vector Autoregressions—Five Years of Experience." *Journal of Business & Economic Statistics* 4 (1): 25–38. [https://doi.org/10.1080/07350015.1986.10509491](https://doi.org/10.1080/07350015.1986.10509491)

### VECM and Cointegration

- Engle, Robert F., and Clive W. J. Granger. 1987. "Co-Integration and Error Correction: Representation, Estimation, and Testing." *Econometrica* 55 (2): 251–276. [https://doi.org/10.2307/1913236](https://doi.org/10.2307/1913236)
- Johansen, Søren. 1991. "Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models." *Econometrica* 59 (6): 1551–1580. [https://doi.org/10.2307/2938278](https://doi.org/10.2307/2938278)

### Local Projections

- Angrist, Joshua D., Òscar Jordà, and Guido M. Kuersteiner. 2018. "Semiparametric Estimates of Monetary Policy Effects: String Theory Revisited." *Journal of Business & Economic Statistics* 36 (3): 371–387. [https://doi.org/10.1080/07350015.2016.1204919](https://doi.org/10.1080/07350015.2016.1204919)
- Auerbach, Alan J., and Yuriy Gorodnichenko. 2013. "Fiscal Multipliers in Recession and Expansion." In *Fiscal Policy after the Financial Crisis*, edited by Alberto Alesina and Francesco Giavazzi, 63–98. Chicago: University of Chicago Press. [https://doi.org/10.7208/chicago/9780226018584.003.0003](https://doi.org/10.7208/chicago/9780226018584.003.0003)
- Barnichon, Regis, and Christian Brownlees. 2019. "Impulse Response Estimation by Smooth Local Projections." *Review of Economics and Statistics* 101 (3): 522–530. [https://doi.org/10.1162/rest_a_00778](https://doi.org/10.1162/rest_a_00778)
- Jordà, Òscar. 2005. "Estimation and Inference of Impulse Responses by Local Projections." *American Economic Review* 95 (1): 161–182. [https://doi.org/10.1257/0002828053828518](https://doi.org/10.1257/0002828053828518)
- Stock, James H., and Mark W. Watson. 2018. "Identification and Estimation of Dynamic Causal Effects in Macroeconomics Using External Instruments." *Economic Journal* 128 (610): 917–948. [https://doi.org/10.1111/ecoj.12593](https://doi.org/10.1111/ecoj.12593)

### Factor Models

- Bai, Jushan, and Serena Ng. 2002. "Determining the Number of Factors in Approximate Factor Models." *Econometrica* 70 (1): 191–221. [https://doi.org/10.1111/1468-0262.00273](https://doi.org/10.1111/1468-0262.00273)
- Forni, Mario, Marc Hallin, Marco Lippi, and Lucrezia Reichlin. 2000. "The Generalized Dynamic-Factor Model: Identification and Estimation." *Review of Economics and Statistics* 82 (4): 540–554. [https://doi.org/10.1162/003465300559037](https://doi.org/10.1162/003465300559037)
- Stock, James H., and Mark W. Watson. 2002. "Forecasting Using Principal Components from a Large Number of Predictors." *Journal of the American Statistical Association* 97 (460): 1167–1179. [https://doi.org/10.1198/016214502388618960](https://doi.org/10.1198/016214502388618960)

### Panel VAR

- Andrews, Donald W. K., and Biao Lu. 2001. "Consistent Model and Moment Selection Procedures for GMM Estimation with Application to Dynamic Panel Data Models." *Journal of Econometrics* 101 (1): 123–164. [https://doi.org/10.1016/S0304-4076(00)00077-4](https://doi.org/10.1016/S0304-4076(00)00077-4)
- Arellano, Manuel, and Stephen Bond. 1991. "Some Tests of Specification for Panel Data: Monte Carlo Evidence and an Application to Employment Equations." *Review of Economic Studies* 58 (2): 277–297. [https://doi.org/10.2307/2297968](https://doi.org/10.2307/2297968)
- Blundell, Richard, and Stephen Bond. 1998. "Initial Conditions and Moment Restrictions in Dynamic Panel Data Models." *Journal of Econometrics* 87 (1): 115–143. [https://doi.org/10.1016/S0304-4076(98)00009-8](https://doi.org/10.1016/S0304-4076(98)00009-8)
- Holtz-Eakin, Douglas, Whitney Newey, and Harvey S. Rosen. 1988. "Estimating Vector Autoregressions with Panel Data." *Econometrica* 56 (6): 1371–1395. [https://doi.org/10.2307/1913103](https://doi.org/10.2307/1913103)
- Windmeijer, Frank. 2005. "A Finite Sample Correction for the Variance of Linear Efficient Two-Step GMM Estimators." *Journal of Econometrics* 126 (1): 25–51. [https://doi.org/10.1016/j.jeconom.2004.02.005](https://doi.org/10.1016/j.jeconom.2004.02.005)

### Difference-in-Differences

- Acemoglu, Daron, Suresh Naidu, Pascual Restrepo, and James A. Robinson. 2019. "Democracy Does Cause Growth." *Journal of Political Economy* 127 (1): 47–100. [https://doi.org/10.1086/700936](https://doi.org/10.1086/700936)
- Borusyak, Kirill, Xavier Jaravel, and Jann Spiess. 2024. "Revisiting Event-Study Designs: Robust and Efficient Estimation." *Review of Economic Studies* 91 (6): 3253–3285. [https://doi.org/10.1093/restud/rdae007](https://doi.org/10.1093/restud/rdae007)
- Callaway, Brantly, and Pedro H. C. Sant'Anna. 2021. "Difference-in-Differences with Multiple Time Periods." *Journal of Econometrics* 225 (2): 200–230. [https://doi.org/10.1016/j.jeconom.2020.12.001](https://doi.org/10.1016/j.jeconom.2020.12.001)
- Dube, Arindrajit, Daniele Girardi, Óscar Jordà, and Alan M. Taylor. 2025. "A Local Projections Approach to Difference-in-Differences." *Journal of Applied Econometrics*. [https://doi.org/10.1002/jae.3117](https://doi.org/10.1002/jae.3117)
- de Chaisemartin, Clement, and Xavier D'Haultfoeuille. 2020. "Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects." *American Economic Review* 110 (9): 2964–2996. [https://doi.org/10.1257/aer.20181169](https://doi.org/10.1257/aer.20181169)
- Goodman-Bacon, Andrew. 2021. "Difference-in-Differences with Variation in Treatment Timing." *Journal of Econometrics* 225 (2): 254–277. [https://doi.org/10.1016/j.jeconom.2021.03.014](https://doi.org/10.1016/j.jeconom.2021.03.014)
- Rambachan, Ashesh, and Jonathan Roth. 2023. "A More Credible Approach to Parallel Trends." *Review of Economic Studies* 90 (5): 2555–2591. [https://doi.org/10.1093/restud/rdad018](https://doi.org/10.1093/restud/rdad018)
- Roth, Jonathan. 2022. "Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends." *American Economic Review: Insights* 4 (3): 305–322. [https://doi.org/10.1257/aeri.20210236](https://doi.org/10.1257/aeri.20210236)
- Sun, Liyang, and Sarah Abraham. 2021. "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects." *Journal of Econometrics* 225 (2): 175–199. [https://doi.org/10.1016/j.jeconom.2020.09.006](https://doi.org/10.1016/j.jeconom.2020.09.006)

### FAVAR

- Bernanke, Ben S., Jean Boivin, and Piotr Eliasz. 2005. "Measuring the Effects of Monetary Policy: A Factor-Augmented Vector Autoregressive (FAVAR) Approach." *Quarterly Journal of Economics* 120 (1): 387–422. [https://doi.org/10.1162/0033553053327452](https://doi.org/10.1162/0033553053327452)

### DSGE

- Andreasen, Martin M., Jesús Fernández-Villaverde, and Juan F. Rubio-Ramírez. 2018. "The Pruned State-Space System for Non-Linear DSGE Models: Theory and Empirical Applications." *Review of Economic Studies* 85 (1): 1–49. [https://doi.org/10.1093/restud/rdx037](https://doi.org/10.1093/restud/rdx037)
- Blanchard, Olivier Jean, and Charles M. Kahn. 1980. "The Solution of Linear Difference Models Under Rational Expectations." *Econometrica* 48 (5): 1305–1311. [https://doi.org/10.2307/1912186](https://doi.org/10.2307/1912186)
- Guerrieri, Luca, and Matteo Iacoviello. 2015. "OccBin: A Toolkit for Solving Dynamic Models with Occasionally Binding Constraints Easily." *Journal of Monetary Economics* 70: 22–38. [https://doi.org/10.1016/j.jmoneco.2014.08.005](https://doi.org/10.1016/j.jmoneco.2014.08.005)
- Hamilton, James D. 1994. *Time Series Analysis*. Princeton: Princeton University Press. ISBN 978-0-691-04289-3.
- Herbst, Edward, and Frank Schorfheide. 2015. *Bayesian Estimation of DSGE Models*. Princeton: Princeton University Press. ISBN 978-0-691-16108-2.
- Kim, Jinill, Sunghyun Kim, Ernst Schaumburg, and Christopher A. Sims. 2008. "Calculating and Using Second-Order Accurate Solutions of Discrete Time Dynamic Equilibrium Models." *Journal of Economic Dynamics and Control* 32 (11): 3397–3414. [https://doi.org/10.1016/j.jedc.2008.02.003](https://doi.org/10.1016/j.jedc.2008.02.003)
- Klein, Paul. 2000. "Using the Generalized Schur Form to Solve a Multivariate Linear Rational Expectations Model." *Journal of Economic Dynamics and Control* 24 (10): 1405–1423. [https://doi.org/10.1016/S0165-1889(99)00045-7](https://doi.org/10.1016/S0165-1889(99)00045-7)
- Schmitt-Grohé, Stephanie, and Martín Uribe. 2004. "Solving Dynamic General Equilibrium Models Using a Second-Order Approximation to the Policy Function." *Journal of Economic Dynamics and Control* 28 (4): 755–775. [https://doi.org/10.1016/S0165-1889(03)00043-5](https://doi.org/10.1016/S0165-1889(03)00043-5)
- Sims, Christopher A. 2002. "Solving Linear Rational Expectations Models." *Computational Economics* 20 (1): 1–20. [https://doi.org/10.1023/A:1020517101123](https://doi.org/10.1023/A:1020517101123)
- Smets, Frank, and Rafael Wouters. 2007. "Shocks and Frictions in US Business Cycles: A Bayesian DSGE Approach." *American Economic Review* 97 (3): 586–606. [https://doi.org/10.1257/aer.97.3.586](https://doi.org/10.1257/aer.97.3.586)
- Stokey, Nancy L., Robert E. Lucas, and Edward C. Prescott. 1989. *Recursive Methods in Economic Dynamics*. Cambridge, MA: Harvard University Press. ISBN 978-0-674-75096-8.
- Walker, Homer F., and Peng Ni. 2011. "Anderson Acceleration for Fixed-Point Iterations." *SIAM Journal on Numerical Analysis* 49 (4): 1715–1735. [https://doi.org/10.1137/10078356X](https://doi.org/10.1137/10078356X)

### Heterogeneous Agent DSGE

- Auclert, Adrien, Bence Bardóczy, Matthew Rognlie, and Ludwig Straub. 2021. "Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models." *Econometrica* 89 (5): 2375–2408. [https://doi.org/10.3982/ECTA17434](https://doi.org/10.3982/ECTA17434)
- Carroll, Christopher D. 2006. "The Method of Endogenous Gridpoints for Solving Dynamic Stochastic Optimization Problems." *Economics Letters* 91 (3): 312–320. [https://doi.org/10.1016/j.econlet.2005.09.013](https://doi.org/10.1016/j.econlet.2005.09.013)
- Kaplan, Greg, Benjamin Moll, and Giovanni L. Violante. 2018. "Monetary Policy According to HANK." *American Economic Review* 108 (3): 697–743. [https://doi.org/10.1257/aer.20160042](https://doi.org/10.1257/aer.20160042)
- Krusell, Per, and Anthony A. Smith Jr. 1998. "Income and Wealth Heterogeneity in the Macroeconomy." *Journal of Political Economy* 106 (5): 867–896. [https://doi.org/10.1086/250034](https://doi.org/10.1086/250034)
- Reiter, Michael. 2009. "Solving Heterogeneous-Agent Models by Projection and Perturbation." *Journal of Economic Dynamics and Control* 33 (3): 649–665. [https://doi.org/10.1016/j.jedc.2008.08.010](https://doi.org/10.1016/j.jedc.2008.08.010)
- Young, Eric R. 2010. "Solving the Incomplete Markets Model with Aggregate Uncertainty Using the Krusell–Smith Algorithm and Non-Stochastic Simulations." *Journal of Economic Dynamics and Control* 34 (1): 36–41. [https://doi.org/10.1016/j.jedc.2008.11.010](https://doi.org/10.1016/j.jedc.2008.11.010)

### Input-Output Analysis

- Baqaee, David Rezza, and Emmanuel Farhi. 2019. "The Macroeconomic Impact of Microeconomic Shocks: Beyond Hulten's Theorem." *Econometrica* 87 (4): 1155–1203. [https://doi.org/10.3982/ECTA15202](https://doi.org/10.3982/ECTA15202)
- Ghosh, Ambica. 1958. "Input-Output Approach in an Allocation System." *Economica* 25 (97): 58–64. [https://doi.org/10.2307/2550694](https://doi.org/10.2307/2550694)
- Leontief, Wassily W. 1936. "Quantitative Input and Output Relations in the Economic System of the United States." *Review of Economics and Statistics* 18 (3): 105–125. [https://doi.org/10.2307/1927837](https://doi.org/10.2307/1927837)
- Miller, Ronald E., and Peter D. Blair. 2009. *Input-Output Analysis: Foundations and Extensions.* 2nd ed. Cambridge: Cambridge University Press. [https://doi.org/10.1017/CBO9780511626982](https://doi.org/10.1017/CBO9780511626982)
- Rasmussen, P. Nørregaard. 1956. *Studies in Inter-Sectoral Relations.* Amsterdam: North-Holland.

### GMM and Covariance Estimation

- Hansen, Lars Peter. 1982. "Large Sample Properties of Generalized Method of Moments Estimators." *Econometrica* 50 (4): 1029–1054. [https://doi.org/10.2307/1912775](https://doi.org/10.2307/1912775)
- Newey, Whitney K., and Kenneth D. West. 1987. "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica* 55 (3): 703–708. [https://doi.org/10.2307/1913610](https://doi.org/10.2307/1913610)

### Statistical Identification via Higher Moments

- Lewis, Daniel J. 2025. "Identification Based on Higher Moments in Macroeconometrics." *Annual Review of Economics* 17: 665–693. [https://doi.org/10.1146/annurev-economics-070124-051419](https://doi.org/10.1146/annurev-economics-070124-051419)
- Lewis, Daniel J. 2021. "Identifying Shocks via Time-Varying Volatility." *Review of Economic Studies* 88 (6): 3086–3124. [https://doi.org/10.1093/restud/rdab009](https://doi.org/10.1093/restud/rdab009)
- Gourieroux, Christian, Alain Monfort, and Jean-Paul Renne. 2017. "Statistical Inference for Independent Component Analysis: Application to Structural VAR Models." *Journal of Econometrics* 196 (1): 111–126. [https://doi.org/10.1016/j.jeconom.2016.09.007](https://doi.org/10.1016/j.jeconom.2016.09.007)
- Hyvärinen, Aapo. 1999. "Fast and Robust Fixed-Point Algorithms for Independent Component Analysis." *IEEE Transactions on Neural Networks* 10 (3): 626–634. [https://doi.org/10.1109/72.761722](https://doi.org/10.1109/72.761722)
- Keweloh, Sascha A. 2021. "A Generalized Method of Moments Estimator for Structural Vector Autoregressions Based on Higher Moments." *Journal of Business & Economic Statistics* 39 (3): 772–882. [https://doi.org/10.1080/07350015.2020.1730858](https://doi.org/10.1080/07350015.2020.1730858)
- Lanne, Markku, Mika Meitz, and Pentti Saikkonen. 2017. "Identification and Estimation of Non-Gaussian Structural Vector Autoregressions." *Journal of Econometrics* 196 (2): 288–304. [https://doi.org/10.1016/j.jeconom.2016.06.002](https://doi.org/10.1016/j.jeconom.2016.06.002)
- Lanne, Markku, and Helmut Lütkepohl. 2010. "Structural Vector Autoregressions with Nonnormal Residuals." *Journal of Business & Economic Statistics* 28 (1): 159–168. [https://doi.org/10.1198/jbes.2009.06003](https://doi.org/10.1198/jbes.2009.06003)
- Rigobon, Roberto. 2003. "Identification through Heteroskedasticity." *Review of Economics and Statistics* 85 (4): 777–792. [https://doi.org/10.1162/003465303772815727](https://doi.org/10.1162/003465303772815727)

### Unit Root and Cointegration Tests

- Dickey, David A., and Wayne A. Fuller. 1979. "Distribution of the Estimators for Autoregressive Time Series with a Unit Root." *Journal of the American Statistical Association* 74 (366): 427–431. [https://doi.org/10.1080/01621459.1979.10482531](https://doi.org/10.1080/01621459.1979.10482531)
- Johansen, Søren. 1991. "Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models." *Econometrica* 59 (6): 1551–1580. [https://doi.org/10.2307/2938278](https://doi.org/10.2307/2938278)
- Kwiatkowski, Denis, Peter C. B. Phillips, Peter Schmidt, and Yongcheol Shin. 1992. "Testing the Null Hypothesis of Stationarity Against the Alternative of a Unit Root." *Journal of Econometrics* 54 (1–3): 159–178. [https://doi.org/10.1016/0304-4076(92)90104-Y](https://doi.org/10.1016/0304-4076(92)90104-Y)
- Elliott, Graham, Thomas J. Rothenberg, and James H. Stock. 1996. "Efficient Tests for an Autoregressive Unit Root." *Econometrica* 64 (4): 813–836. [https://doi.org/10.2307/2171846](https://doi.org/10.2307/2171846)
- Enders, Walter, and Junsoo Lee. 2012. "A Unit Root Test Using a Fourier Series to Approximate Smooth Breaks." *Oxford Bulletin of Economics and Statistics* 74 (4): 574–599. [https://doi.org/10.1111/j.1468-0084.2011.00662.x](https://doi.org/10.1111/j.1468-0084.2011.00662.x)
- Gregory, Allan W., and Bruce E. Hansen. 1996. "Residual-Based Tests for Cointegration in Models with Regime Shifts." *Journal of Econometrics* 70 (1): 99–126. [https://doi.org/10.1016/0304-4076(69)41685-7](https://doi.org/10.1016/0304-4076(69)41685-7)
- Lee, Junsoo, and Mark C. Strazicich. 2003. "Minimum Lagrange Multiplier Unit Root Test with Two Structural Breaks." *Review of Economics and Statistics* 85 (4): 1082–1089. [https://doi.org/10.1162/003465303772815961](https://doi.org/10.1162/003465303772815961)
- Narayan, Paresh Kumar, and Stephan Popp. 2010. "A New Unit Root Test with Two Structural Breaks in Level and Slope at Unknown Time." *Journal of Applied Statistics* 47 (12): 1425–1438. [https://doi.org/10.1080/02664760903039883](https://doi.org/10.1080/02664760903039883)
- Ng, Serena, and Pierre Perron. 2001. "Lag Length Selection and the Construction of Unit Root Tests with Good Size and Power." *Econometrica* 69 (6): 1519–1554. [https://doi.org/10.1111/1468-0262.00256](https://doi.org/10.1111/1468-0262.00256)

### Structural Breaks

- Andrews, Donald W. K. 1993. "Tests for Parameter Instability and Structural Change with Unknown Change Point." *Econometrica* 61 (4): 821–856. [https://doi.org/10.2307/2951764](https://doi.org/10.2307/2951764)
- Bai, Jushan, and Pierre Perron. 1998. "Estimating and Testing Linear Models with Multiple Structural Changes." *Econometrica* 66 (1): 47–78. [https://doi.org/10.2307/2998540](https://doi.org/10.2307/2998540)
- Breitung, Jörg, and Sandra Eickmeier. 2011. "Testing for Structural Breaks in Dynamic Factor Models." *Journal of Econometrics* 163 (1): 71–84. [https://doi.org/10.1016/j.jeconom.2010.11.008](https://doi.org/10.1016/j.jeconom.2010.11.008)

### Panel Unit Root Tests

- Bai, Jushan, and Serena Ng. 2004. "A PANIC Attack on Unit Roots and Cointegration." *Econometrica* 72 (4): 1127–1177. [https://doi.org/10.1111/j.1468-0262.2004.00528.x](https://doi.org/10.1111/j.1468-0262.2004.00528.x)
- Moon, Hyungsik Roger, and Benoît Perron. 2004. "Testing for a Unit Root in Panels with Dynamic Factors." *Journal of Econometrics* 122 (1): 81–126. [https://doi.org/10.1016/j.jeconom.2003.10.020](https://doi.org/10.1016/j.jeconom.2003.10.020)
- Pesaran, M. Hashem. 2007. "A Simple Panel Unit Root Test in the Presence of Cross-Section Dependence." *Journal of Applied Econometrics* 22 (2): 265–312. [https://doi.org/10.1002/jae.951](https://doi.org/10.1002/jae.951)

### Panel Regression

- Hausman, Jerry A., and William E. Taylor. 1981. "Panel Data and Unobservable Individual Effects." *Econometrica* 49 (6): 1377–1398. [https://doi.org/10.2307/1911406](https://doi.org/10.2307/1911406)
- Mundlak, Yair. 1978. "On the Pooling of Time Series and Cross Section Data." *Econometrica* 46 (1): 69–85. [https://doi.org/10.2307/1913646](https://doi.org/10.2307/1913646)
- Wooldridge, Jeffrey M. 2010. *Econometric Analysis of Cross Section and Panel Data*. 2nd ed. Cambridge, MA: MIT Press. ISBN 978-0-262-23258-6.

### Ordered and Multinomial Choice

- Brant, Rollin. 1990. "Assessing Proportionality in the Proportional Odds Model for Ordinal Logistic Regression." *Biometrics* 46 (4): 1171–1178. [https://doi.org/10.2307/2532457](https://doi.org/10.2307/2532457)
- McFadden, Daniel. 1974. "Conditional Logit Analysis of Qualitative Choice Behavior." In *Frontiers in Econometrics*, edited by Paul Zarembka, 105–142. New York: Academic Press.

### Spectral Analysis

- Welch, Peter D. 1967. "The Use of Fast Fourier Transform for the Estimation of Power Spectra." *IEEE Transactions on Audio and Electroacoustics* 15 (2): 70–73. [https://doi.org/10.1109/TAU.1967.1161901](https://doi.org/10.1109/TAU.1967.1161901)
- Priestley, Maurice B. 1981. *Spectral Analysis and Time Series*. London: Academic Press. ISBN 978-0-12-564922-3.

### Granger Causality

- Granger, Clive W. J. 1969. "Investigating Causal Relations by Econometric Models and Cross-spectral Methods." *Econometrica* 37 (3): 424–438. [https://doi.org/10.2307/1912791](https://doi.org/10.2307/1912791)

### Data Sources

- Feenstra, Robert C., Robert Inklaar, and Marcel P. Timmer. 2015. "The Next Generation of the Penn World Table." *American Economic Review* 105 (10): 3150–3182. [https://doi.org/10.1257/aer.20130954](https://doi.org/10.1257/aer.20130954)
- McCracken, Michael W., and Serena Ng. 2016. "FRED-MD: A Monthly Database for Macroeconomic Research." *Journal of Business & Economic Statistics* 34 (4): 574–589. [https://doi.org/10.1080/07350015.2015.1086655](https://doi.org/10.1080/07350015.2015.1086655)
- McCracken, Michael W., and Serena Ng. 2020. "FRED-QD: A Quarterly Database for Macroeconomic Research." *Federal Reserve Bank of St. Louis Working Paper* 2020-005. [https://doi.org/10.20955/wp.2020.005](https://doi.org/10.20955/wp.2020.005)

### Cross-Sectional Regression

- White, Halbert. 1980. "A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity." *Econometrica* 48 (4): 817–838. [https://doi.org/10.2307/1912934](https://doi.org/10.2307/1912934)
- MacKinnon, James G., and Halbert White. 1985. "Some Heteroskedasticity-Consistent Covariance Matrix Estimators with Improved Finite Sample Properties." *Journal of Econometrics* 29 (3): 305–325. [https://doi.org/10.1016/0304-4076(85)90158-7](https://doi.org/10.1016/0304-4076(85)90158-7)
- Wooldridge, Jeffrey M. 2010. *Econometric Analysis of Cross Section and Panel Data*. 2nd ed. Cambridge, MA: MIT Press. ISBN 978-0-262-23258-6.

### Nowcasting

- Bańbura, Marta, and Michele Modugno. 2014. "Maximum Likelihood Estimation of Factor Models on Datasets with Arbitrary Pattern of Missing Data." *Journal of Applied Econometrics* 29 (1): 133–160. [https://doi.org/10.1002/jae.2306](https://doi.org/10.1002/jae.2306)
- Cimadomo, Jacopo, Domenico Giannone, Michele Lenza, Francesca Monti, and Andrej Sokol. 2022. "Nowcasting with Large Bayesian Vector Autoregressions." *ECB Working Paper* No. 2696.
- Bańbura, Marta, Irina Belousova, Katalin Bodnár, and Máté Barnabás Tóth. 2023. "Nowcasting Employment in the Euro Area." *ECB Working Paper* No. 2815.
- Linzenich, Jan, and Baptiste Meunier. 2024. "Nowcasting with Mixed Frequency Data Using a Simple Modelling Setup: An Update of the ECB Nowcasting Framework." *ECB Working Paper* No. 3004.

### Model Comparison Tests

- Neyman, Jerzy, and Egon S. Pearson. 1933. "On the Problem of the Most Efficient Tests of Statistical Hypotheses." *Philosophical Transactions of the Royal Society A* 231 (694–706): 289–337. [https://doi.org/10.1098/rsta.1933.0009](https://doi.org/10.1098/rsta.1933.0009)
- Rao, C. Radhakrishna. 1948. "Large Sample Tests of Statistical Hypotheses Concerning Several Parameters with Applications to Problems of Estimation." *Mathematical Proceedings of the Cambridge Philosophical Society* 44 (1): 50–57. [https://doi.org/10.1017/S0305004100023987](https://doi.org/10.1017/S0305004100023987)
- Silvey, S. D. 1959. "The Lagrangian Multiplier Test." *Annals of Mathematical Statistics* 30 (2): 389–407. [https://doi.org/10.1214/aoms/1177706259](https://doi.org/10.1214/aoms/1177706259)
- Wilks, Samuel S. 1938. "The Large-Sample Distribution of the Likelihood Ratio for Testing Composite Hypotheses." *Annals of Mathematical Statistics* 9 (1): 60–62. [https://doi.org/10.1214/aoms/1177732360](https://doi.org/10.1214/aoms/1177732360)

### MIDAS and Mixed-Frequency Regression

- Ghysels, Eric, Arthur Sinko, and Rossen Valkanov. 2007. "MIDAS Regressions: Further Results and New Directions." *Econometric Reviews* 26 (1): 53–90. [https://doi.org/10.1080/07474930600972467](https://doi.org/10.1080/07474930600972467)
- Foroni, Claudia, Massimiliano Marcellino, and Christian Schumacher. 2015. "Unrestricted Mixed Data Sampling (MIDAS): MIDAS Regressions with Unrestricted Lag Polynomials." *Journal of the Royal Statistical Society: Series A* 178 (1): 57–82. [https://doi.org/10.1111/rssa.12043](https://doi.org/10.1111/rssa.12043)
- Engle, Robert F., Eric Ghysels, and Bumjean Sohn. 2013. "Stock Market Volatility and Macroeconomic Fundamentals." *Review of Economics and Statistics* 95 (3): 776–797. [https://doi.org/10.1162/REST_a_00300](https://doi.org/10.1162/REST_a_00300)

### ARDL and Cointegrating Regression

- Pesaran, M. Hashem, Yongcheol Shin, and Richard J. Smith. 2001. "Bounds Testing Approaches to the Analysis of Level Relationships." *Journal of Applied Econometrics* 16 (3): 289–326. [https://doi.org/10.1002/jae.616](https://doi.org/10.1002/jae.616)
- Pesaran, M. Hashem, Yongcheol Shin, and Ron P. Smith. 1999. "Pooled Mean Group Estimation of Dynamic Heterogeneous Panels." *Journal of the American Statistical Association* 94 (446): 621–634. [https://doi.org/10.1080/01621459.1999.10474156](https://doi.org/10.1080/01621459.1999.10474156)
- Shin, Yongcheol, Byungchul Yu, and Matthew Greenwood-Nimmo. 2014. "Modelling Asymmetric Cointegration and Dynamic Multipliers in a Nonlinear ARDL Framework." In *Festschrift in Honor of Peter Schmidt*, edited by Robin C. Sickles and William C. Horrace, 281–314. New York: Springer. [https://doi.org/10.1007/978-1-4899-8008-3_9](https://doi.org/10.1007/978-1-4899-8008-3_9)
- Phillips, Peter C. B., and Bruce E. Hansen. 1990. "Statistical Inference in Instrumental Variables Regression with I(1) Processes." *Review of Economic Studies* 57 (1): 99–125. [https://doi.org/10.2307/2297545](https://doi.org/10.2307/2297545)
- Stock, James H., and Mark W. Watson. 1993. "A Simple Estimator of Cointegrating Vectors in Higher Order Integrated Systems." *Econometrica* 61 (4): 783–820. [https://doi.org/10.2307/2951763](https://doi.org/10.2307/2951763)

### Nonlinear Time Series

- Hamilton, James D. 1989. "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica* 57 (2): 357–384. [https://doi.org/10.2307/1912559](https://doi.org/10.2307/1912559)
- Hansen, Bruce E. 2000. "Sample Splitting and Threshold Estimation." *Econometrica* 68 (3): 575–603. [https://doi.org/10.1111/1468-0262.00124](https://doi.org/10.1111/1468-0262.00124)
- Teräsvirta, Timo. 1994. "Specification, Estimation, and Evaluation of Smooth Transition Autoregressive Models." *Journal of the American Statistical Association* 89 (425): 208–218. [https://doi.org/10.1080/01621459.1994.10476462](https://doi.org/10.1080/01621459.1994.10476462)

### Multivariate and Long-Memory Volatility

- Bollerslev, Tim. 1990. "Modelling the Coherence in Short-Run Nominal Exchange Rates: A Multivariate Generalized ARCH Model." *Review of Economics and Statistics* 72 (3): 498–505. [https://doi.org/10.2307/2109358](https://doi.org/10.2307/2109358)
- Engle, Robert F. 2002. "Dynamic Conditional Correlation: A Simple Class of Multivariate Generalized Autoregressive Conditional Heteroskedasticity Models." *Journal of Business & Economic Statistics* 20 (3): 339–350. [https://doi.org/10.1198/073500102288618487](https://doi.org/10.1198/073500102288618487)
- Engle, Robert F., and Kenneth F. Kroner. 1995. "Multivariate Simultaneous Generalized ARCH." *Econometric Theory* 11 (1): 122–150. [https://doi.org/10.1017/S0266466600009063](https://doi.org/10.1017/S0266466600009063)
- Baillie, Richard T., Tim Bollerslev, and Hans Ole Mikkelsen. 1996. "Fractionally Integrated Generalized Autoregressive Conditional Heteroskedasticity." *Journal of Econometrics* 74 (1): 3–30. [https://doi.org/10.1016/S0304-4076(95)01749-6](https://doi.org/10.1016/S0304-4076(95)01749-6)
- Ding, Zhuanxin, Clive W. J. Granger, and Robert F. Engle. 1993. "A Long Memory Property of Stock Market Returns and a New Model." *Journal of Empirical Finance* 1 (1): 83–106. [https://doi.org/10.1016/0927-5398(93)90006-D](https://doi.org/10.1016/0927-5398(93)90006-D)
- Geweke, John, and Susan Porter-Hudak. 1983. "The Estimation and Application of Long Memory Time Series Models." *Journal of Time Series Analysis* 4 (4): 221–238. [https://doi.org/10.1111/j.1467-9892.1983.tb00371.x](https://doi.org/10.1111/j.1467-9892.1983.tb00371.x)
- Robinson, Peter M. 1995. "Gaussian Semiparametric Estimation of Long Range Dependence." *Annals of Statistics* 23 (5): 1630–1661. [https://doi.org/10.1214/aos/1176324317](https://doi.org/10.1214/aos/1176324317)

### State-Space Models

- Durbin, James, and Siem Jan Koopman. 2012. *Time Series Analysis by State Space Methods.* 2nd ed. Oxford: Oxford University Press. [https://doi.org/10.1093/acprof:oso/9780199641178.001.0001](https://doi.org/10.1093/acprof:oso/9780199641178.001.0001)
- Harvey, Andrew C. 1989. *Forecasting, Structural Time Series Models and the Kalman Filter.* Cambridge: Cambridge University Press. [https://doi.org/10.1017/CBO9781107049994](https://doi.org/10.1017/CBO9781107049994)

### Systems of Equations

- Zellner, Arnold. 1962. "An Efficient Method of Estimating Seemingly Unrelated Regressions and Tests for Aggregation Bias." *Journal of the American Statistical Association* 57 (298): 348–368. [https://doi.org/10.1080/01621459.1962.10480664](https://doi.org/10.1080/01621459.1962.10480664)
- Zellner, Arnold, and Henri Theil. 1962. "Three-Stage Least Squares: Simultaneous Estimation of Simultaneous Equations." *Econometrica* 30 (1): 54–78. [https://doi.org/10.2307/1911287](https://doi.org/10.2307/1911287)

### Nonparametric Methods

- Cleveland, William S. 1979. "Robust Locally Weighted Regression and Smoothing Scatterplots." *Journal of the American Statistical Association* 74 (368): 829–836. [https://doi.org/10.1080/01621459.1979.10481038](https://doi.org/10.1080/01621459.1979.10481038)
- Sheather, Simon J., and Michael C. Jones. 1991. "A Reliable Data-Based Bandwidth Selection Method for Kernel Density Estimation." *Journal of the Royal Statistical Society: Series B* 53 (3): 683–690. [https://doi.org/10.1111/j.2517-6161.1991.tb01857.x](https://doi.org/10.1111/j.2517-6161.1991.tb01857.x)
- Fan, Jianqing, and Irène Gijbels. 1996. *Local Polynomial Modelling and Its Applications.* London: Chapman & Hall. ISBN 978-0-412-98321-4.

### Forecast Evaluation and Combination

- Diebold, Francis X., and Roberto S. Mariano. 1995. "Comparing Predictive Accuracy." *Journal of Business & Economic Statistics* 13 (3): 253–263. [https://doi.org/10.1080/07350015.1995.10524599](https://doi.org/10.1080/07350015.1995.10524599)
- Clark, Todd E., and Kenneth D. West. 2007. "Approximately Normal Tests for Equal Predictive Accuracy in Nested Models." *Journal of Econometrics* 138 (1): 291–311. [https://doi.org/10.1016/j.jeconom.2006.05.023](https://doi.org/10.1016/j.jeconom.2006.05.023)
- Bates, John M., and Clive W. J. Granger. 1969. "The Combination of Forecasts." *Operational Research Quarterly* 20 (4): 451–468. [https://doi.org/10.1057/jors.1969.103](https://doi.org/10.1057/jors.1969.103)

### Continuous-Time and Life-Cycle Heterogeneous Agents

- Achdou, Yves, Jiequn Han, Jean-Michel Lasry, Pierre-Louis Lions, and Benjamin Moll. 2022. "Income and Wealth Distribution in Macroeconomics: A Continuous-Time Approach." *Review of Economic Studies* 89 (1): 45–86. [https://doi.org/10.1093/restud/rdab002](https://doi.org/10.1093/restud/rdab002)
- Aiyagari, S. Rao. 1994. "Uninsured Idiosyncratic Risk and Aggregate Saving." *Quarterly Journal of Economics* 109 (3): 659–684. [https://doi.org/10.2307/2118417](https://doi.org/10.2307/2118417)
- Blanchard, Olivier J. 1985. "Debt, Deficits, and Finite Horizons." *Journal of Political Economy* 93 (2): 223–247. [https://doi.org/10.1086/261297](https://doi.org/10.1086/261297)

### DSGE Bayesian Estimation Diagnostics

- Iskrev, Nikolay. 2010. "Local Identification in DSGE Models." *Journal of Monetary Economics* 57 (2): 189–202. [https://doi.org/10.1016/j.jmoneco.2009.12.007](https://doi.org/10.1016/j.jmoneco.2009.12.007)
- Vehtari, Aki, Andrew Gelman, Daniel Simpson, Bob Carpenter, and Paul-Christian Bürkner. 2021. "Rank-Normalization, Folding, and Localization: An Improved R̂ for Assessing Convergence of MCMC." *Bayesian Analysis* 16 (2): 667–718. [https://doi.org/10.1214/20-BA1221](https://doi.org/10.1214/20-BA1221)
- Meng, Xiao-Li, and Wing Hung Wong. 1996. "Simulating Ratios of Normalizing Constants via a Simple Identity: A Theoretical Exploration." *Statistica Sinica* 6 (4): 831–860.
- Geweke, John. 1992. "Evaluating the Accuracy of Sampling-Based Approaches to the Calculation of Posterior Moments." In *Bayesian Statistics 4*, edited by J. M. Bernardo, J. O. Berger, A. P. Dawid, and A. F. M. Smith, 169–193. Oxford: Oxford University Press.

### Panel Cointegration and First-Generation Panel Unit Root Tests

- Levin, Andrew, Chien-Fu Lin, and Chia-Shang James Chu. 2002. "Unit Root Tests in Panel Data: Asymptotic and Finite-Sample Properties." *Journal of Econometrics* 108 (1): 1–24. [https://doi.org/10.1016/S0304-4076(01)00098-7](https://doi.org/10.1016/S0304-4076(01)00098-7)
- Im, Kyung So, M. Hashem Pesaran, and Yongcheol Shin. 2003. "Testing for Unit Roots in Heterogeneous Panels." *Journal of Econometrics* 115 (1): 53–74. [https://doi.org/10.1016/S0304-4076(03)00092-7](https://doi.org/10.1016/S0304-4076(03)00092-7)
- Pedroni, Peter. 2004. "Panel Cointegration: Asymptotic and Finite Sample Properties of Pooled Time Series Tests with an Application to the PPP Hypothesis." *Econometric Theory* 20 (3): 597–625. [https://doi.org/10.1017/S0266466604203073](https://doi.org/10.1017/S0266466604203073)
- Westerlund, Joakim. 2007. "Testing for Error Correction in Panel Data." *Oxford Bulletin of Economics and Statistics* 69 (6): 709–748. [https://doi.org/10.1111/j.1468-0084.2007.00477.x](https://doi.org/10.1111/j.1468-0084.2007.00477.x)
- Dumitrescu, Elena-Ivona, and Christophe Hurlin. 2012. "Testing for Granger Non-Causality in Heterogeneous Panels." *Economic Modelling* 29 (4): 1450–1460. [https://doi.org/10.1016/j.econmod.2012.02.014](https://doi.org/10.1016/j.econmod.2012.02.014)

### Seasonal Unit Roots, Bubbles, and Distribution Tests

- Hylleberg, Svend, Robert F. Engle, Clive W. J. Granger, and Byung Sam Yoo. 1990. "Seasonal Integration and Cointegration." *Journal of Econometrics* 44 (1–2): 215–238. [https://doi.org/10.1016/0304-4076(90)90080-D](https://doi.org/10.1016/0304-4076(90)90080-D)
- Phillips, Peter C. B., Shuping Shi, and Jun Yu. 2015. "Testing for Multiple Bubbles: Historical Episodes of Exuberance and Collapse in the S&P 500." *International Economic Review* 56 (4): 1043–1078. [https://doi.org/10.1111/iere.12132](https://doi.org/10.1111/iere.12132)
- Lo, Andrew W., and A. Craig MacKinlay. 1988. "Stock Market Prices Do Not Follow Random Walks: Evidence from a Simple Specification Test." *Review of Financial Studies* 1 (1): 41–66. [https://doi.org/10.1093/rfs/1.1.41](https://doi.org/10.1093/rfs/1.1.41)
- Brock, William A., W. Davis Dechert, José A. Scheinkman, and Blake LeBaron. 1996. "A Test for Independence Based on the Correlation Dimension." *Econometric Reviews* 15 (3): 197–235. [https://doi.org/10.1080/07474939608800353](https://doi.org/10.1080/07474939608800353)

### Regularized, Robust, and Limited-Dependent-Variable Regression

- Tibshirani, Robert. 1996. "Regression Shrinkage and Selection via the Lasso." *Journal of the Royal Statistical Society: Series B* 58 (1): 267–288. [https://doi.org/10.1111/j.2517-6161.1996.tb02080.x](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x)
- Zou, Hui, and Trevor Hastie. 2005. "Regularization and Variable Selection via the Elastic Net." *Journal of the Royal Statistical Society: Series B* 67 (2): 301–320. [https://doi.org/10.1111/j.1467-9868.2005.00503.x](https://doi.org/10.1111/j.1467-9868.2005.00503.x)
- Huber, Peter J. 1964. "Robust Estimation of a Location Parameter." *Annals of Mathematical Statistics* 35 (1): 73–101. [https://doi.org/10.1214/aoms/1177703732](https://doi.org/10.1214/aoms/1177703732)
- Tobin, James. 1958. "Estimation of Relationships for Limited Dependent Variables." *Econometrica* 26 (1): 24–36. [https://doi.org/10.2307/1907382](https://doi.org/10.2307/1907382)
- Heckman, James J. 1979. "Sample Selection Bias as a Specification Error." *Econometrica* 47 (1): 153–161. [https://doi.org/10.2307/1912352](https://doi.org/10.2307/1912352)
- Fuller, Wayne A. 1977. "Some Properties of a Modification of the Limited Information Estimator." *Econometrica* 45 (4): 939–953. [https://doi.org/10.2307/1912683](https://doi.org/10.2307/1912683)

## License

GPL-3.0-or-later
