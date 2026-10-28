<img width="1760" height="600" alt="MacroEconometricModels.jl" src="https://raw.githubusercontent.com/FriedmanJP/MacroEconometricModels.jl/main/banner.png" />

# MacroEconometricModels.jl

[![CI](https://github.com/FriedmanJP/MacroEconometricModels.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/FriedmanJP/MacroEconometricModels.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/FriedmanJP/MacroEconometricModels.jl/graph/badge.svg)](https://codecov.io/gh/FriedmanJP/MacroEconometricModels.jl)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://api.friedman.jp/MacroEconometricModels.jl/stable/)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://api.friedman.jp/MacroEconometricModels.jl/dev/)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18439170.svg)](https://doi.org/10.5281/zenodo.18439170)

**MacroEconometricModels.jl** is a comprehensive Julia package for macroeconometrics, time-series econometrics, structural macroeconomic modeling, panel data analysis, and causal inference. It provides a unified, high-performance environment spanning **Vector Autoregressions (VAR)**, **Bayesian VAR (BVAR)**, **Structural VAR (SVAR)**, **Vector Error Correction Models (VECM)**, **Local Projections (LP, LP-IV, Smooth LP, State-Dependent LP, LP-FEVD)**, **Dynamic Factor Models (DFM, GDFM, FAVAR)**, **Dynamic Stochastic General Equilibrium (DSGE)** models, **Heterogeneous Agent New Keynesian (HANK)** and **Overlapping Generations (OLG)** models in discrete and continuous time, **Policy Counterfactuals**, **Input-Output (IO) and Production Network Analysis**, **Mixed-Frequency Nowcasting and MIDAS**, **Univariate Time Series (ARIMA, ARFIMA, X-13ARIMA-SEATS, ARCH/GARCH, Stochastic Volatility, Spectral Analysis, State-Space, and Trend-Cycle Filters)**, **Nonlinear Time Series (Threshold/SETAR, STAR, Markov-Switching)**, **Cross-Sectional and Panel Econometrics (Difference-in-Differences, LP-DiD, Panel VAR, Panel ARDL, IV/2SLS/LIML, Discrete Choice, Count, Quantile, Censored/Selection, and Regression Discontinuity)**, **Generalized and Simulated Method of Moments (GMM/SMM)**, and **Econometric Hypothesis Testing**.

## Domain Overview

| Domain | Core Methods & Models | Documentation |
| :--- | :--- | :--- |
| **Multivariate Time Series** | Vector Autoregression (VAR), Bayesian VAR (BVAR), Vector Error Correction (VECM), Autoregressive Distributed Lag (ARDL & NARDL), Cointegrating Regression (FMOLS, CCR, DOLS), Seemingly Unrelated Regressions (SUR), Three-Stage Least Squares (3SLS) | [VAR](https://api.friedman.jp/MacroEconometricModels.jl/stable/manual/) · [BVAR](https://api.friedman.jp/MacroEconometricModels.jl/stable/bayesian/) · [VECM](https://api.friedman.jp/MacroEconometricModels.jl/stable/vecm/) · [ARDL](https://api.friedman.jp/MacroEconometricModels.jl/stable/ardl/) |
| **Local Projections & Factor Models** | Jordà Local Projections (LP), Instrumental Variables LP (LP-IV), Smooth B-Spline LP, State-Dependent LP, Propensity-Score LP, Structural LP, Static PCA, Dynamic Factor Models (DFM), Generalized DFM (GDFM), Factor-Augmented VAR (FAVAR), Structural DFM | [Local Projections](https://api.friedman.jp/MacroEconometricModels.jl/stable/lp/) · [Factor Models](https://api.friedman.jp/MacroEconometricModels.jl/stable/factormodels/) · [FAVAR](https://api.friedman.jp/MacroEconometricModels.jl/stable/favar/) |
| **Structural Identification & Innovation Accounting** | 25 SVAR identification schemes: Cholesky, Sign, Narrative, Long-Run (Blanchard-Quah), Zero+Sign (Arias et al.), Mountford-Uhlig Penalty, Proxy SVAR, AB-Model, Max-Share News Shocks, Non-Gaussian ICA/MLE, Heteroskedastic/TVV/SV-SVAR; IRF, FEVD, LP-FEVD, Historical Decomposition | [Structural ID](https://api.friedman.jp/MacroEconometricModels.jl/stable/structural_identification/) · [Statistical ID](https://api.friedman.jp/MacroEconometricModels.jl/stable/nongaussian/) · [Innovation Accounting](https://api.friedman.jp/MacroEconometricModels.jl/stable/innovation_accounting/) |
| **DSGE Models** | Symbolic rational expectations, Gensys, Blanchard-Kahn, Klein, 2nd/3rd-order pruned perturbation, Chebyshev projection, Policy & Value Function Iteration (PFI/VFI), Perfect Foresight, OccBin occasionally binding constraints, Classical GMM/SMM, Bayesian SMC, SMC², and Metropolis-Hastings | [DSGE Overview](https://api.friedman.jp/MacroEconometricModels.jl/stable/dsge/) · [Linear](https://api.friedman.jp/MacroEconometricModels.jl/stable/dsge_linear/) · [Nonlinear](https://api.friedman.jp/MacroEconometricModels.jl/stable/dsge_nonlinear/) · [Estimation](https://api.friedman.jp/MacroEconometricModels.jl/stable/dsge_estimation/) |
| **Heterogeneous Agent DSGE (HANK, OLG & Continuous Time)** | Sequence-Space Jacobian (SSJ), Reiter projection-perturbation, Krusell-Smith, One- and Two-Asset HANK, Endogenous Grid Method (EGM, Nested EGM, DCEGM), Khan-Thomas plants, Bewley banks, Continuous-Time Aiyagari & HANK (HJB/KFE), Blanchard & Life-Cycle OLG | [Heterogeneity Hub](https://api.friedman.jp/MacroEconometricModels.jl/stable/dsge_heterogeneity/) · [HANK](https://api.friedman.jp/MacroEconometricModels.jl/stable/dsge_ha/) · [OLG](https://api.friedman.jp/MacroEconometricModels.jl/stable/dsge_olg/) · [Continuous Time](https://api.friedman.jp/MacroEconometricModels.jl/stable/dsge_continuous/) |
| **Policy Counterfactuals** | McKay-Wolf rule counterfactuals and optimal-policy projections, Barnichon-Mesters Optimal Policy Perturbations (OPP) with ZLB constraints, Caravello-McKay-Wolf model bank and posterior model averaging, spanning and forecast-sufficiency diagnostics | [Counterfactuals Hub](https://api.friedman.jp/MacroEconometricModels.jl/stable/counterfactual/) · [McKay-Wolf](https://api.friedman.jp/MacroEconometricModels.jl/stable/counterfactual_rules/) · [OPP](https://api.friedman.jp/MacroEconometricModels.jl/stable/counterfactual_opp/) · [Model Bank](https://api.friedman.jp/MacroEconometricModels.jl/stable/counterfactual_bank/) |
| **Input-Output & Production Networks** | Leontief & Ghosh models, Type I/II multipliers, Rasmussen linkages, Structural Decomposition Analysis (SDA), Hypothetical Extraction, Environmental satellites, Baqaee-Farhi nonlinear production networks and wedges, KWW export value-added decomposition, MRIO loaders | [Input-Output Hub](https://api.friedman.jp/MacroEconometricModels.jl/stable/io/) · [Classical IO](https://api.friedman.jp/MacroEconometricModels.jl/stable/io_classical/) · [MRIO Trade](https://api.friedman.jp/MacroEconometricModels.jl/stable/io_mrio/) · [Baqaee-Farhi](https://api.friedman.jp/MacroEconometricModels.jl/stable/io_baqaee_farhi/) |
| **Nowcasting, MIDAS & Forecast Evaluation** | Mixed-frequency DFM nowcasting (EM + Kalman smoother), Large BVAR nowcasting, Bridge equations, Bańbura-Modugno news decomposition (DFM & BVAR), MIDAS regression, Diebold-Mariano, Clark-West, Mincer-Zarnowitz, forecast encompassing and combination | [Nowcasting Hub](https://api.friedman.jp/MacroEconometricModels.jl/stable/nowcast/) · [MIDAS](https://api.friedman.jp/MacroEconometricModels.jl/stable/midas/) · [Forecast Evaluation](https://api.friedman.jp/MacroEconometricModels.jl/stable/forecast_evaluation/) |
| **Univariate, Volatility & Nonlinear Time Series** | HP, Hamilton, Beveridge-Nelson, Baxter-King, and Boosted HP filters; X-13ARIMA-SEATS seasonal adjustment; ARIMA & ARFIMA; State-Space & TVP regression; ARCH, GARCH, EGARCH, GJR, IGARCH, CGARCH, APARCH, FIGARCH, FIEGARCH, GARCH-MIDAS, CCC/DCC/BEKK, Bayesian Stochastic Volatility; Spectral analysis; Threshold/SETAR, STAR, Markov-Switching | [Filters](https://api.friedman.jp/MacroEconometricModels.jl/stable/filters/) · [X-13ARIMA-SEATS](https://api.friedman.jp/MacroEconometricModels.jl/stable/x13/) · [ARIMA](https://api.friedman.jp/MacroEconometricModels.jl/stable/arima/) · [Volatility](https://api.friedman.jp/MacroEconometricModels.jl/stable/volatility/) · [State-Space](https://api.friedman.jp/MacroEconometricModels.jl/stable/statespace/) · [Nonlinear](https://api.friedman.jp/MacroEconometricModels.jl/stable/nonlinear/) |
| **Panel Data, Difference-in-Differences & Cross-Section** | Linear Panel (FE, RE, FD, Between, CRE, Arellano-Bond, Blundell-Bond, PCSE, Prais-Winsten), Panel IV (Hausman-Taylor), Panel Logit/Probit, Panel VAR, Panel ARDL (PMG/MG/DFE), Staggered DiD (TWFE, Callaway-Sant'Anna, Sun-Abraham, BJS, dCDH, HonestDiD, Event-Study LP, LP-DiD), OLS/WLS/IV/LIML, Penalized, Robust, Discrete/Ordered/Multinomial Choice, Count, Tobit, Heckman, Quantile, RDD, Nonparametric | [Panel Regression](https://api.friedman.jp/MacroEconometricModels.jl/stable/panel_reg/) · [Panel VAR](https://api.friedman.jp/MacroEconometricModels.jl/stable/pvar/) · [DiD](https://api.friedman.jp/MacroEconometricModels.jl/stable/did/) · [Event Studies](https://api.friedman.jp/MacroEconometricModels.jl/stable/event_study/) · [Cross-Section](https://api.friedman.jp/MacroEconometricModels.jl/stable/regression/) · [GMM & SMM](https://api.friedman.jp/MacroEconometricModels.jl/stable/gmm/) |
| **Hypothesis Testing & Diagnostics** | Unit root (ADF, KPSS, PP, Zivot-Andrews, Ng-Perron, Fourier, DF-GLS/ERS, LM breaks, Narayan-Popp, HEGY), Cointegration (Johansen, Engle-Granger, Phillips-Ouliaris, Gregory-Hansen, Pedroni, Kao, Westerlund), Structural breaks (Andrews, Bai-Perron, factor breaks), Explosive bubbles (SADF/GSADF), Panel unit root (PANIC, CIPS, Moon-Perron, LLC, IPS), Normality, Independence & HAC | [Hypothesis Tests Hub](https://api.friedman.jp/MacroEconometricModels.jl/stable/tests/) · [Unit Root](https://api.friedman.jp/MacroEconometricModels.jl/stable/tests_unitroot/) · [Structural Breaks](https://api.friedman.jp/MacroEconometricModels.jl/stable/tests_breaks/) · [Panel Tests](https://api.friedman.jp/MacroEconometricModels.jl/stable/tests_panel/) · [Diagnostics](https://api.friedman.jp/MacroEconometricModels.jl/stable/tests_diagnostics/) |

---

## Features

### Univariate Time Series Models
- **Time Series Filters & Trend-Cycle Decomposition**
  - Hodrick-Prescott (1997) filter
  - Hamilton (2018) regression filter
  - Beveridge-Nelson (1981) permanent-transitory decomposition
  - Baxter-King (1999) approximate band-pass filter
  - Boosted HP filter (Phillips & Shi 2021) with ADF, BIC, and fixed stopping criteria
- **X-13ARIMA-SEATS Seasonal Adjustment** — Pure-Julia implementation of the US Census Bureau and Bank of Spain seasonal adjustment program:
  - X-11 iterative moving-average decomposition with Henderson trend filters and seasonal moving averages (Findley et al. 1998; Dagum & Bianconcini 2016)
  - SEATS Wiener-Kolmogorov spectral signal extraction (Gómez & Maravall 1996)
  - TRAMO-style automatic ARIMA model identification with AICC selection
  - Automatic outlier detection (additive outliers, level shifts, temporary changes)
  - Trading-day and Easter calendar-effect regressors, plus Jacobian-corrected log/level transformation selection
- **ARIMA & Fractionally Integrated ARFIMA**
  - Autoregressive Moving Average (AR, MA, ARMA, ARIMA) estimation via conditional sum of squares (CSS), exact maximum likelihood (Kalman filter), or hybrid CSS-MLE (Box & Jenkins 1970)
  - Automatic order selection over lag and differencing grids using AIC, AICc, and BIC (Hyndman & Khandakar 2008)
  - Fractionally integrated long-memory ARFIMA estimation via CSS or exact MLE
  - Semiparametric long-memory fractional differencing estimators: Geweke & Porter-Hudak (1983) log-periodogram regression and Robinson (1995) local Whittle estimator
  - Multi-step ahead forecasting with analytical psi-weight confidence intervals
- **Univariate & Long-Memory Volatility Models**
  - Autoregressive Conditional Heteroskedasticity (ARCH; Engle 1982) and Generalized ARCH (GARCH; Bollerslev 1986)
  - Asymmetric and exponential volatility: EGARCH (Nelson 1991), GJR-GARCH (Glosten, Jagannathan & Runkle 1993), and Asymmetric Power ARCH (APARCH; Ding, Granger & Engle 1993)
  - Integrated and component volatility: IGARCH (Engle & Bollerslev 1986) and Component-GARCH (Engle & Lee 1999)
  - Long-memory volatility: Fractionally Integrated GARCH (FIGARCH; Baillie, Bollerslev & Mikkelsen 1996) and Fractionally Integrated EGARCH (FIEGARCH; Bollerslev & Mikkelsen 1996)
  - Mixed-frequency volatility: GARCH-MIDAS decomposing short-run fluctuations and secular macro-driven volatility (Engle, Ghysels & Sohn 2013)
  - Bayesian Stochastic Volatility (SV) via the Kim, Shephard & Chib (1998) Gibbs sampler with basic, leverage, and heavy-tailed Student-t innovations
  - Volatility diagnostics: news impact curves, persistence, half-life, unconditional variance, Engle & Ng (1993) sign-bias tests, and Nyblom (1989) parameter-stability tests
- **Spectral Analysis & Autocorrelation**
  - Periodogram, Welch (1967) overlapping-segment spectral density, Daniell smoothed periodogram, and autoregressive spectral estimation (Priestley 1981)
  - Cross-spectral analysis: coherence, phase, gain, ideal band-pass filtering, and transfer-function analysis
  - Fisher exact periodicity test and Bartlett cumulative periodogram white-noise test
  - Autocorrelation (ACF), partial autocorrelation (PACF via Levinson-Durbin and OLS), cross-correlation (CCF), and cumulative Ljung-Box, Box-Pierce, and Durbin-Watson portmanteau tests

### Nonlinear Time Series & State-Space Models
- **Threshold & Self-Exciting Threshold Autoregression (SETAR)**
  - Two-regime threshold least squares and SETAR with automatic delay and threshold grid search
  - Hansen (1996) fixed-regressor bootstrap linearity test and Hansen (2000) likelihood-ratio-inversion threshold confidence intervals
- **Smooth Transition Autoregression (STAR)**
  - Logistic (LSTR1, LSTR2) and exponential (ESTR) smooth-transition regressions via nonlinear least squares
  - Luukkonen, Saikkonen & Teräsvirta (1988) LM linearity test and Teräsvirta (1994) sequential transition-family selection
- **Markov-Switching Models**
  - Multi-regime Markov-switching regression and mean-switching autoregression (MS-AR; Hamilton 1989) estimated via the Hamilton forward filter, Kim (1994) backward smoother, and Expectation-Maximization / direct maximum likelihood
- **Linear-Gaussian State-Space & Unobserved Components**
  - Prediction-error-decomposition maximum likelihood for arbitrary parameterized linear-Gaussian state-space systems via the Kalman filter and Rauch-Tung-Striebel (RTS) smoother (Harvey 1989; Durbin & Koopman 2012)
  - Structural time-series components (local level and local linear trend) and time-varying-parameter (TVP) regression with random-walk state evolution

### Multivariate Time Series & Factor Models
- **Vector Autoregression (VAR)**
  - Ordinary least squares estimation with automatic lag-order selection (AIC, BIC, HQ), companion-form stability analysis, multi-step forecasting, and Granger causality testing (Sims 1980; Lütkepohl 2005)
- **Bayesian Vector Autoregression (BVAR)**
  - Conjugate Normal-Inverse-Wishart posterior sampling with Minnesota, sum-of-coefficients, and single-unit-root priors (Litterman 1986; Bańbura, Giannone & Reichlin 2010)
  - Hierarchical marginal-likelihood hyperparameter optimization following Giannone, Lenza & Primiceri (2015)
- **Vector Error Correction Model (VECM)**
  - Johansen (1991) maximum likelihood and Engle & Granger (1987) two-step estimation with trace and maximum-eigenvalue cointegration rank selection
  - Exact level-VAR conversion unlocking all structural and statistical identification schemes, cointegration-preserving forecasting, and short-run, long-run, and strong Granger causality tests
  - Likelihood-ratio hypothesis testing on cointegrating vectors and adjustment speeds: linear restrictions on cointegrating vectors, adjustment-coefficient restrictions, weak exogeneity tests, known cointegrating vector tests, and joint restrictions
- **Autoregressive Distributed Lag (ARDL & NARDL) & Bounds Testing**
  - Linear ARDL and asymmetric Nonlinear ARDL (Shin, Yu & Greenwood-Nimmo 2014) with short-run, long-run, and dynamic multipliers
  - Conditional error-correction representation and Pesaran, Shin & Smith (2001) F- and t-bounds tests for level relationships across I(0) and I(1) regressors with exact finite-sample critical values
- **Single-Equation Cointegrating Regression**
  - Fully Modified OLS (FMOLS; Phillips & Hansen 1990), Canonical Cointegrating Regression (CCR; Park 1992), and Dynamic OLS (DOLS; Saikkonen 1991; Stock & Watson 1993)
- **Local Projections (LP)**
  - Jordà (2005) horizon-by-horizon impulse response estimation with Newey-West, White, and Driscoll-Kraay heteroskedasticity and autocorrelation robust standard errors
  - External-instrument Local Projections (LP-IV; Stock & Watson 2018) with weak-instrument diagnostics
  - Smooth Local Projections using penalized B-splines (Barnichon & Brownlees 2019)
  - Nonlinear state-dependent Local Projections with logistic regime switching (Auerbach & Gorodnichenko 2013)
  - Semiparametric propensity-score-weighted Local Projections for discrete policy interventions (Angrist, Jordà & Kuersteiner 2018)
  - Structural multi-shock Local Projections (Plagborg-Møller & Wolf 2021) and direct multi-horizon LP forecasting
- **Static, Dynamic & Generalized Dynamic Factor Models**
  - Principal-component static factor models with Bai & Ng (2002) information criteria (IC1, IC2, IC3)
  - Parametric Dynamic Factor Models (DFM) estimated via two-step principal components and Kalman smoothing or Expectation-Maximization (Stock & Watson 2002)
  - Frequency-domain Generalized Dynamic Factor Models (GDFM; Forni, Hallin, Lippi & Reichlin 2000)
  - Unified factor and observable forecasting with analytical and bootstrap confidence intervals
- **Factor-Augmented VAR (FAVAR) & Structural DFM**
  - Two-step principal-component FAVAR and Bayesian Gibbs-sampled FAVAR via Carter-Kohn state smoothing (Bernanke, Boivin & Eliasz 2005)
  - Structural Dynamic Factor Models combining GDFM extraction with structural VAR identification, mapping structural factor shocks back to all panel observables
- **Multivariate GARCH**
  - Constant Conditional Correlation (CCC-GARCH; Bollerslev 1990), Dynamic Conditional Correlation (DCC-GARCH and corrected cDCC; Engle 2002; Aielli 2013), and scalar/diagonal BEKK(1,1) covariance models (Engle & Kroner 1995) with multi-step covariance forecasting
- **Mixed-Data Sampling (MIDAS) Regression**
  - Mixed-frequency regression linking low-frequency targets to high-frequency predictors via Exponential-Almon, Beta, polynomial-Almon, and unrestricted (U-MIDAS) lag polynomials with autoregressive augmentation and direct multi-horizon forecasting (Ghysels, Sinko & Valkanov 2007; Foroni, Marcellino & Schumacher 2015)
- **Simultaneous Systems of Equations**
  - Feasible and iterated Generalized Least Squares for Seemingly Unrelated Regressions (SUR; Zellner 1962) with McElroy (1977) system R-squared, and Three-Stage Least Squares (3SLS; Zellner & Theil 1962) for simultaneous equations with endogenous regressors

### Structural Identification & Innovation Accounting
- **Economic & Exclusion-Based SVAR Identification**
  - Recursive Cholesky / Wold causal ordering (Sims 1980)
  - Sign restrictions via Haar-measure QR orthogonal rotations (Rubio-Ramírez, Waggoner & Zha 2010)
  - Narrative sign and shock-contribution restrictions on historical episodes (Antolín-Díaz & Rubio-Ramírez 2018)
  - Long-run cumulative impulse response restrictions for permanent-transitory decompositions (Blanchard & Quah 1989)
  - Joint zero and sign restrictions with importance sampling (Arias, Rubio-Ramírez & Waggoner 2018)
  - Penalty-function identification with first-difference or residual standard-deviation normalization (Uhlig 2005; Mountford & Uhlig 2009)
  - Proxy SVAR / External Instrument identification for point- and set-identified structural shocks (Stock & Watson 2012; Mertens & Ravn 2013)
  - Classical AB-model maximum likelihood SVAR (Amisano & Giannini 1997; Lütkepohl 2005)
  - Max-Share identification maximizing forecast error variance share at target horizons for news and TFP shocks (Uhlig 2004; Barsky & Sims 2011; Francis et al. 2014)
  - Structural VECM identification combining cointegration rank with permanent and transitory shock restrictions (King, Plosser, Stock & Watson 1991)
- **Statistical Identification via Higher Moments & Volatility**
  - Non-Gaussian Independent Component Analysis (ICA): FastICA (Hyvärinen 1999), JADE, Second-Order Blind Identification (SOBI), Distance Covariance (Matteson & Tsay 2017), and Hilbert-Schmidt Independence Criterion (HSIC)
  - Non-Gaussian Maximum Likelihood & Higher-Moment GMM: Student-t, mixture-normal, pseudo-ML, skew-normal (Lanne, Meitz & Saikkonen 2017; Gourieroux, Monfort & Renne 2017), and third/fourth-moment coskewness and cokurtosis GMM (Keweloh 2021; Lewis 2025)
  - Heteroskedasticity and Volatility Regimes: Markov-switching covariance (Lanne, Lütkepohl & Maciejowska 2010), GARCH-SVAR (Normandin & Phaneuf 2004), smooth-transition covariance (Lütkepohl & Netšunajev 2017), external volatility regimes (Rigobon 2003), Time-Varying Volatility SVAR (Lewis 2021), and Bayesian Stochastic Volatility SVAR
  - Set-identified inference & diagnostics: Fry-Pagan median-target, Inoue-Kilian modal model, joint credible bands, Giacomini & Kitagawa (2021) robust Bayesian prior-bounds inference, and shock Gaussianity, independence, and distinct-eigenvalue Wald tests
- **Innovation Accounting**
  - Frequentist (analytical and wild/block bootstrap) and Bayesian Impulse Response Functions (IRF)
  - Forecast Error Variance Decomposition (FEVD) and bias-corrected Local Projection Variance Decomposition (LP-FEVD; Gorodnichenko & Lee 2019)
  - Historical Decomposition (HD) attributing observed time-series fluctuations to individual structural shocks

### Dynamic Stochastic General Equilibrium (DSGE) Models
- **Model Specification & Steady-State Solvers**
  - Symbolic rational-expectations model specification with automatic lead/lag handling, analytical or numerical Newton steady-state computation, and box- or inequality-constrained steady-state solvers via interior-point and sequential quadratic programming methods (plus optional mixed complementarity problem support)
  - Native support for pre-linearized DSGE specifications such as Smets & Wouters (2007)
- **First-Order & Higher-Order Perturbation Solvers**
  - Linear rational-expectations solvers: Sims (2002) Gensys QZ decomposition, Blanchard & Kahn (1980) eigenvalue counting, and Klein (2000) generalized Schur decomposition
  - Second-order perturbation (Schmitt-Grohé & Uribe 2004) with Kim, Kim, Schaumburg & Sims (2008) pruning
  - Third-order perturbation with Andreasen, Fernández-Villaverde & Rubio-Ramírez (2018) pruned state-space simulation, exact augmented-Lyapunov analytical moments, and higher-order unconditional variance decomposition
- **Global Projection, Dynamic Programming & Nonlinear Constraints**
  - Chebyshev polynomial collocation on tensor-product and Smolyak sparse grids with Gauss-Hermite quadrature (Judd 1998)
  - Policy Function Iteration (Coleman 1990; Rendahl 2017) and Value Function Iteration (Stokey, Lucas & Prescott 1989) accelerated via Howard policy improvement and Walker & Ni (2011) Anderson acceleration
  - Deterministic perfect-foresight stacked-Newton solver with sparse colored Jacobians, block-tridiagonal factorization, and box/nonlinear inequality constraints
  - Occasionally binding constraints via piecewise-linear regime switching (OccBin; Guerrieri & Iacoviello 2015)
- **Classical & Bayesian DSGE Estimation**
  - Classical estimation: Impulse-response matching, Euler-equation GMM, Simulated Method of Moments (SMM), and Lyapunov analytical-moment GMM
  - Bayesian estimation (Herbst & Schorfheide 2015): Adaptive-tempering Sequential Monte Carlo (SMC), Particle-Filter SMC-squared (SMC²) for nonlinear higher-order models, and Random-Walk Metropolis-Hastings (RWMH) with delayed acceptance
  - Posterior mode optimization, Laplace approximation, and Meng & Wong (1996) bridge-sampling marginal likelihoods
  - Convergence, identification & validation: Vehtari et al. (2021) rank-normalized split-R-hat, bulk/tail effective sample size, Geweke (1992) spectral diagnostics, Iskrev (2010) Jacobian rank identification tests, Koop-Pesaran-Smith learning-rate checks, Gelman-Meng-Stern posterior predictive p-values, and Kalman/particle-smoother historical decompositions
  - Verified against a 24-model Dynare replication suite covering steady states, impulse responses, variance decompositions, and theoretical moments

### Heterogeneous Agent DSGE (HANK, OLG & Continuous Time)
- **Household, Firm & Financial Intermediary Blocks**
  - Endogenous Grid Method (EGM; Carroll 2006) for one-asset incomplete-market models, Nested EGM for two-asset HANK models with illiquid-asset adjustment costs (Kaplan, Moll & Violante 2018), and Discrete-Continuous EGM (DCEGM; Iskhakov, Jørgensen, Rust & Schjerning 2017)
  - Non-stochastic histogram distribution tracking via sparse transition matrices (Young 2010) with Rouwenhorst (1995) and Tauchen (1986) Markov discretization
  - Heterogeneous firm investment with nonconvex adjustment costs (Khan & Thomas 2008) and Bewley-bank financial intermediaries
  - Pre-calibrated benchmark economies including Aiyagari (1994), Huggett (1993), Krusell & Smith (1998), endogenous-labor HA, and one- and two-asset HANK
- **General Equilibrium Solution & Estimation Methods**
  - Sequence-Space Jacobian (SSJ; Auclert, Bardóczy, Rognlie & Straub 2021) with the fake-news algorithm, Directed Acyclic Graph block composition, multi-population aggregation, and Ho-Kalman state-space reduction
  - Reiter (2009) projection-perturbation linearization with observability-based singular-value dimensionality reduction
  - Krusell & Smith (1998) aggregate law-of-motion simulation
  - Bayesian estimation of heterogeneous-agent models re-solving the stationary equilibrium and sequence-space linearization across posterior draws
  - Distributional accounting: wealth-distribution impulse responses, Lorenz curves, Gini and top-wealth-share responses, and micro-panel simulation
- **Continuous-Time Heterogeneous Agents & Overlapping Generations**
  - Upwind finite-difference Hamilton-Jacobi-Bellman (HJB) and Kolmogorov-Forward (KFE) stationary equilibrium and MIT-shock transition solvers for continuous-time Aiyagari and two-asset HANK economies (Achdou, Han, Lasry, Lions & Moll 2022)
  - Blanchard (1985) perpetual-youth overlapping generations (including New Keynesian monetary extensions) and Auerbach-Kotlikoff multi-cohort life-cycle transition dynamics

### Policy Counterfactuals
- **Sufficient-Statistics Rule Counterfactuals (McKay & Wolf 2023)**
  - News-shock policy-rule counterfactuals and optimal-policy projections constructed directly from empirical impulse responses without committing to a single structural model
  - Built-in policy templates for interest-rate pegs, Taylor rules, nominal GDP targeting, average inflation targeting, and second-moment Wold variance counterfactuals with implementation-error diagnostics
- **Optimal Policy Perturbations (Barnichon & Mesters 2023)**
  - Sufficient-statistics optimality test and Optimal Policy Perturbation (OPP) direction from macro gap forecasts and policy-shock impulse responses
  - Two-source confidence sets (60%, 75%, 90%), zero-lower-bound and pre-commitment constrained optimization, and multi-period decision-date sequences with time-consistency decompositions
- **Structural Model Bank & Diagnostics (Caravello, McKay & Wolf 2025)**
  - Limited-information impulse-response matching across structural model banks with Geweke marginal likelihoods and posterior model averaging
  - Representative-agent DSGE news menus, one-asset HANK sequence-space menus, Gabaix cognitive-discounting and sticky-expectations behavioral operators, historical forecast-revision counterfactuals, and spanning/sufficiency tests

### Input-Output Analysis & Production Networks
- **Classical Input-Output Accounting & Multipliers**
  - Demand-driven Leontief (1936) and supply-driven Ghosh (1958) inverses, Type I and Type II output, income, value-added, and employment multipliers (Miller & Blair 2009)
  - Rasmussen (1956) backward and forward dispersion linkages, key-sector classification, additive two-polar and multiplicative Structural Decomposition Analysis (SDA), Dietzenbacher-Lahr hypothetical extraction, and RAS/GRAS biproportional matrix balancing
- **Environmental Input-Output & Multi-Regional Trade Accounting (MRIO)**
  - Satellite accounts for carbon and environmental stressors, direct and total intensity matrices, and consumption-based emission footprints
  - Multi-regional aggregation, regional footprints, and Koopman, Wang & Wei (2014) gross export decomposition into domestic value added, returned domestic value, foreign value added, and pure double counting
  - Automated downloaders and parsers for OECD Inter-Country Input-Output (ICIO), World Input-Output Database (WIOD), Exiobase 3, Eora 26, and GLORIA
- **Nonlinear Production Networks (Baqaee & Farhi 2019)**
  - Multi-factor nested-CES production network general equilibrium, Domar weights, Hulten's theorem, second-order "beyond Hulten" nonlinear propagation of microeconomic TFP shocks, and distorting wedge and misallocation decompositions

### Nowcasting & Forecast Evaluation
- **Mixed-Frequency Nowcasting**
  - Dynamic Factor Model (DFM) nowcasting for mixed-frequency panels with arbitrary missing data and ragged edges via Expectation-Maximization and Kalman smoothing with Mariano-Murasawa temporal aggregation (Bańbura & Modugno 2014)
  - Large Bayesian VAR mixed-frequency nowcasting with Normal-Inverse-Wishart shrinkage and marginal-likelihood hyperparameter selection (Cimadomo et al. 2022)
  - Pairwise monthly-to-quarterly Bridge Equations aggregated via median combinations (Bańbura et al. 2023)
  - Automated panel balancing and missing-data imputation
- **Nowcast News Decomposition**
  - Exact Bańbura & Modugno (2014) decomposition of nowcast revisions into data revisions and individual indicator news releases, with category-group aggregation, supported for both Dynamic Factor Models and Large Bayesian VAR nowcasts
- **Forecast Evaluation & Combination**
  - Accuracy metrics: Mean Error, MAE, RMSE, MAPE, sMAPE, MASE, and Theil U1/U2
  - Predictive accuracy and efficiency tests: Diebold & Mariano (1995) with Harvey-Leybourne-Newbold finite-sample correction, Clark & West (2007) nested-model MSPE test, Mincer & Zarnowitz (1969) efficiency regression, and forecast encompassing tests
  - Forecast combinations: Equal-weight, Bates & Granger (1969) inverse-MSE, and Granger & Ramanathan (1984) constrained least-squares weights

### Cross-Sectional, Panel & Nonparametric Econometrics
- **Cross-Sectional Regression & Causal Microeconometrics**
  - Ordinary and Weighted Least Squares with White (1980) and MacKinnon & White (1985) HC0–HC3 heteroskedasticity-robust and multi-way cluster-robust standard errors, plus Variance Inflation Factor (VIF) diagnostics
  - Instrumental Variables: Two-Stage Least Squares (2SLS), Limited Information Maximum Likelihood (LIML), Fuller (1977) modified LIML, and k-class estimators with weak-instrument F-tests and Sargan-Hansen overidentification tests
  - Penalized & Robust Regression: Coordinate-descent Ridge (Hoerl & Kennard 1970), LASSO (Tibshirani 1996), and Elastic Net (Zou & Hastie 2005) with cross-validated regularization paths; Huber (1964) and Tukey-bisquare M-estimation and Yohai (1987) high-breakdown MM-estimation; stepwise, best-subset, and London School of Economics general-to-specific (GETS) variable selection
  - Discrete, Ordered & Multinomial Choice: Binary Logit and Probit, Ordered Logit and Probit with Brant (1990) parallel-regression test, and Multinomial Logit (McFadden 1974) with Hausman-McFadden IIA test; average marginal effects (AME), marginal effects at means (MEM), and marginal effects at representative values (MER) via the delta method
  - Count, Limited Dependent, Quantile & Discontinuity Models: Poisson and Negative-Binomial-2 count regression with overdispersion tests and incidence-rate ratios; Tobit censored regression (Tobin 1958) and truncated regression with McDonald-Moffitt decomposition; Heckman (1979) two-step and full-information MLE sample selection; Koenker & Bassett (1978) Quantile Regression; Sharp and Fuzzy Regression Discontinuity Designs (RDD)
- **Nonparametric Density & Regression**
  - Univariate kernel density estimation with Sheather & Jones (1991) plug-in and Silverman bandwidths, Nadaraya-Watson and Fan & Gijbels (1996) local-polynomial regression, and Cleveland (1979) locally weighted scatterplot smoothing (LOWESS)
- **Panel Data Econometrics & Panel VAR**
  - Linear Panel Regression (Wooldridge 2010): Fixed Effects (within), Random Effects (Swamy-Arora GLS), First Differences, Between, and Mundlak (1978) / Chamberlain Correlated Random Effects; dynamic panel GMM via Arellano & Bond (1991) and Blundell & Bond (1998); Driscoll-Kraay spatial-temporal HAC, Beck-Katz Panel-Corrected Standard Errors (PCSE), and Prais-Winsten AR(1) FGLS; Hausman, Breusch-Pagan LM, Pesaran CD, Wooldridge autocorrelation, and Modified Wald groupwise heteroskedasticity tests
  - Panel Instrumental Variables (FE-IV, RE-IV, FD-IV, and Hausman & Taylor 1981 endogenous time-invariant covariates) and Panel Discrete Choice (Pooled, Conditional FE, Gauss-Hermite Quadrature RE, and Mundlak CRE Logit/Probit)
  - Panel VAR (Holtz-Eakin, Newey & Rosen 1988): First-difference GMM and System GMM with Helmert forward orthogonal deviations, Windmeijer (2005) finite-sample covariance correction, Hansen J-test, Andrews & Lu (2001) MMSC lag/moment selection, orthogonalized and generalized impulse responses (Pesaran & Shin 1998), and block-bootstrap confidence bands
  - Heterogeneous Dynamic Panels: Panel ARDL Pooled Mean Group (PMG), Mean Group (MG), and Dynamic Fixed Effects (DFE; Pesaran, Shin & Smith 1999), plus Panel FMOLS and DOLS cointegrating regression (Pedroni 2000, 2001; Kao & Chiang 2000)
- **Difference-in-Differences (DiD) & Panel Event Studies**
  - Staggered adoption estimators: Two-Way Fixed Effects (TWFE), Callaway & Sant'Anna (2021) doubly robust group-time average treatment effects, Sun & Abraham (2021) interaction-weighted estimator, Borusyak, Jaravel & Spiess (2024) counterfactual imputation, and de Chaisemartin & D'Haultfoeuille (2020) switching-treatment estimator
  - Diagnostics & Sensitivity: Goodman-Bacon (2021) 2×2 weight decomposition, joint pre-trend Wald tests, negative-weight diagnostics, and Rambachan & Roth (2023) HonestDiD partial-identification breakdown bounds under relative-magnitudes and smoothness restrictions
  - Panel Event-Study Local Projections (Acemoglu, Naidu, Restrepo & Robinson 2019) and Local Projection Difference-in-Differences (LP-DiD; Dube, Girardi, Jordà & Taylor 2025) with clean control samples, pre-mean differencing, and inverse-propensity reweighting

### Generalized & Simulated Method of Moments (GMM & SMM)
- One-step, two-step, and iterated Generalized Method of Moments (Hansen 1982) and Simulated Method of Moments (SMM) with Newey & West (1987) HAC-optimal weighting matrices, Windmeijer finite-sample variance correction, and Hansen J-tests of overidentifying restrictions

### Hypothesis Testing & Econometric Diagnostics
- **Univariate & Seasonal Unit Root Tests**
  - Augmented Dickey-Fuller (ADF; Dickey & Fuller 1979), KPSS stationarity (Kwiatkowski et al. 1992), Phillips-Perron, Ng & Perron (2001) MZ-statistics, and Elliott, Rothenberg & Stock (1996) DF-GLS and point-optimal tests
  - Structural-break and nonlinear unit root tests: Zivot-Andrews single-break, Narayan & Popp (2010) two-break ADF, Schmidt-Phillips and Lee & Strazicich (2003, 2013) one- and two-break LM tests, Enders & Lee (2012) and Becker, Enders & Lee (2006) Fourier unit root and stationarity tests, and Hylleberg, Engle, Granger & Yoo (1990) HEGY seasonal unit root tests
- **Cointegration, Structural Breaks & Explosive Bubbles**
  - Johansen trace and max-eigenvalue tests, Engle-Granger and Phillips-Ouliaris residual-based tests, Gregory & Hansen (1996) regime-shift cointegration, Hansen parameter-instability cointegration test, and Park added-variable test
  - Andrews (1993) SupWald, SupLM, and SupLR unknown-breakpoint tests, Bai & Perron (1998) dynamic-programming multiple structural break estimation and testing, Brown-Durbin-Evans CUSUM and CUSUM-of-squares, Chow stability tests, and Breitung & Eickmeier (2011), Chen-Dolado-Gonzalo (2014), and Han-Inoue (2015) factor structural-break tests
  - Right-tailed SupADF (SADF) and Generalized SupADF (GSADF) explosive bubble detection with backward SADF real-time date-stamping (Phillips, Wu & Yu 2011; Phillips, Shi & Yu 2015)
- **Panel Unit Root, Panel Cointegration & Causality**
  - First-generation panel unit root tests: Levin, Lin & Chu (2002), Im, Pesaran & Shin (2003), Breitung, Maddala-Wu Fisher combinations, and Hadri stationarity
  - Second-generation cross-sectionally dependent panel unit root tests: Bai & Ng (2004) PANIC factor-decomposed tests, Pesaran (2007) cross-sectionally augmented CIPS, and Moon & Perron (2004) factor-adjusted t-tests
  - Panel cointegration & causality: Pedroni (1999, 2004) seven-statistic battery, Kao (1999) residual ADF, Westerlund (2007) error-correction tests, Fisher-Johansen combined panel cointegration, and Dumitrescu & Hurlin (2012) heterogeneous panel Granger non-causality
- **Specification, Distribution, Independence & Long-Run Covariance**
  - Regression & model comparison: White, Breusch-Pagan, Glejser, and Harvey heteroskedasticity tests, Breusch-Godfrey serial correlation, Ramsey RESET functional-form test, and nested Likelihood Ratio (LR; Wilks 1938) and Lagrange Multiplier (LM; Rao 1948; Silvey 1959) tests
  - Normality & goodness-of-fit: Jarque-Bera, Mardia multivariate skewness/kurtosis, Doornik-Hansen, Henze-Zirkler, Royston, and empirical distribution function tests (Kolmogorov-Smirnov, Lilliefors, Cramér-von Mises, Anderson-Darling, Watson)
  - Nonlinear dependence, random walk & two-sample tests: Brock, Dechert, Scheinkman & LeBaron (1996) BDS independence test, Lo & MacKinlay (1988), Chow-Denning, Wright rank/sign, and wild-bootstrap variance-ratio tests, parametric/nonparametric equality-of-distribution tests (Wilcoxon, Kruskal-Wallis, van der Waerden, Levene, Bartlett), and Andrews (1991) / Newey & West (1994) kernel HAC and VARHAC long-run covariance estimators

### Visualization, Tabular Interoperability, Reproducibility & Datasets
- **Interactive D3.js Visualization** — Self-contained HTML figures powered by inline D3.js v7 with zero external runtime dependencies, supporting impulse responses, variance decompositions, historical decompositions, trend-cycle filters, volatility paths, spectral densities, factor panels, nowcast fan charts and ragged-edge heatmaps, DSGE/HANK wealth distributions and Lorenz curves, input-output networks, and event-study plots in Solarized Light and Dark themes
- **Publication Tables & Tidy Data Interoperability** — Switchable plain-text, LaTeX, and HTML regression and summary tables alongside complete Tables.jl integration across nearly 100 result types (coefficients, marginal effects, impulse responses, variance and historical decompositions, forecast evaluations, policy counterfactuals, input-output tables, DSGE/HANK solutions, and the 60-type hypothesis test battery)
- **Reproducibility, Serialization & Bibliographic Citations** — Bit-for-bit reproducibility manifests capturing RNG state, thread count, and environment metadata; versioned JLD2 model serialization; and automatic American Economic Association (AEA) bibliographic reference export in plain text, BibTeX, LaTeX, and HTML
- **Built-in Benchmark Datasets** — Thirteen ready-to-use datasets included out of the box: FRED-MD (McCracken & Ng 2016), FRED-QD (McCracken & Ng 2020), Penn World Table (Feenstra, Inklaar & Timmer 2015), Acemoglu et al. (2019) democracy-growth panel, Callaway & Sant'Anna (2021) county minimum-wage panel, Grunfeld (1958) firm investment panel, Mroz (1987) labor-supply cross-section, World Input-Output Table, US monetary-policy shocks, Johansen-Juselius Danish money demand, Hamilton (1989) US GNP, Nile river flow, and Brownlee stack-loss data

## Documentation & Bibliography

- **User Manual & Tutorials (Stable):** [https://api.friedman.jp/MacroEconometricModels.jl/stable/](https://api.friedman.jp/MacroEconometricModels.jl/stable/)
- **Development Documentation:** [https://api.friedman.jp/MacroEconometricModels.jl/dev/](https://api.friedman.jp/MacroEconometricModels.jl/dev/)
- **Method Selection Guide:** [Choosing a Method](https://api.friedman.jp/MacroEconometricModels.jl/stable/method_guide/)
- **API Reference Catalog:** [API Overview](https://api.friedman.jp/MacroEconometricModels.jl/stable/api/)
- **Complete Academic Bibliography:** [References & DOIs](https://api.friedman.jp/MacroEconometricModels.jl/stable/bibliography/)

## License

GPL-3.0-or-later
