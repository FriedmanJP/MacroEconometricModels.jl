# MacroEconometricModels.jl

[![CI](https://github.com/chung9207/MacroEconometricModels.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/chung9207/MacroEconometricModels.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/chung9207/MacroEconometricModels.jl/graph/badge.svg?token=PB8UPGDJIY)](https://codecov.io/gh/chung9207/MacroEconometricModels.jl)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://chung9207.github.io/MacroEconometricModels.jl/stable/)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18439170.svg)](https://doi.org/10.5281/zenodo.18439170)

A comprehensive Julia package for macroeconomic time series analysis. Provides VAR, VECM, Bayesian VAR, Panel VAR, Local Projections, Factor Models, ARIMA, time series filters, GMM, ARCH/GARCH/Stochastic Volatility estimation, structural identification (including non-Gaussian and heteroskedasticity-based methods), hypothesis testing, typed data containers, and publication-quality output with multi-format bibliographic references.

## Features

### Univariate Models
- **Time Series Filters** - Trend-cycle decomposition:
  - Hodrick-Prescott filter
  - Hamilton (2018) regression filter
  - Beveridge-Nelson decomposition (ARIMA psi-weights)
  - Baxter-King band-pass filter
  - Boosted HP filter (Phillips & Shi 2021) with ADF/BIC/fixed stopping
- **ARIMA** - AR, MA, ARMA, ARIMA estimation via CSS, exact MLE (Kalman filter), or CSS-MLE
  - Automatic order selection via `auto_arima` with grid search over (p,d,q), AIC/BIC information criteria
  - Multi-step ahead forecasting with confidence intervals via psi-weight accumulation
- **Volatility Models**:
  - ARCH - Engle (1982) ARCH(q) with MLE, ARCH-LM test, Ljung-Box squared residuals
  - GARCH - GARCH(p,q), EGARCH (Nelson 1991), GJR-GARCH (Glosten, Jagannathan & Runkle 1993)
  - Stochastic Volatility - Bayesian SV via Kim-Shephard-Chib Gibbs sampler (basic, leverage, Student-t variants)
  - Multi-step volatility forecasts with simulation CIs
  - Diagnostics: news impact curves, persistence, half-life, unconditional variance

### Multivariate Models
- **Vector Autoregression (VAR)** - OLS estimation with lag order selection (AIC, BIC, HQ)
- **Bayesian VAR (BVAR)** - Minnesota priors with hyperparameter optimization (Giannone, Lenza & Primiceri 2015)
- **Vector Error Correction Model (VECM)** - Johansen MLE and Engle-Granger two-step estimation
  - Automatic cointegrating rank selection (trace/max-eigenvalue)
  - VAR conversion (`to_var`) enabling all 18+ identification methods
  - VECM-specific forecasting preserving cointegrating relationships
  - Granger causality: short-run, long-run, and strong tests
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

### GMM
- **Generalized Method of Moments** - One-step, two-step, and iterated; Hansen J-test
- **Linear GMM** - Closed-form solver for panel IV estimation
- **Sandwich covariance** - Robust GMM variance with Windmeijer correction

### Structural Identification
- Cholesky decomposition (recursive)
- Sign restrictions (Rubio-Ramirez, Waggoner & Zha 2010)
- Narrative restrictions (Antolin-Diaz & Rubio-Ramirez 2018)
- Long-run restrictions (Blanchard & Quah 1989)
- Arias, Rubio-Ramirez & Waggoner (2018) algorithm for zero and sign restrictions

### Innovation Accounting
- **Impulse Response Functions (IRF)** - Bootstrap, theoretical, and Bayesian credible intervals
- **Forecast Error Variance Decomposition (FEVD)** - Frequentist and Bayesian
- **LP-FEVD** - R², LP-A, LP-B estimators with bootstrap CIs (Gorodnichenko & Lee 2019)
- **Historical Decomposition (HD)** - Decompose observed movements into structural shock contributions
- **Summary Tables** - Publication-quality output with `report()`, `table()`, `print_table()`
- **Display Backends** - Switch between text, LaTeX, and HTML table output with `set_display_backend()`
- **Bibliographic References** - `refs(model)` outputs AEA-style citations in text, LaTeX, BibTeX, or HTML

### Non-Gaussian Identification
- **Heteroskedasticity-based** - Markov-switching, GARCH, smooth-transition, external volatility
- **ICA-based methods** - FastICA, JADE, SOBI, distance covariance, HSIC
- **Non-Gaussian ML** - Student-t, mixture-normal, pseudo-ML, skew-normal
- **Multivariate normality tests** - Jarque-Bera, Mardia, Doornik-Hansen, Henze-Zirkler
- **Identifiability diagnostics** - Shock gaussianity, LR tests, independence tests, bootstrap strength tests
- Seamless integration: `irf(model, 20; method=:fastica)` works out of the box

### Hypothesis Tests
- **Unit Root Tests** - ADF, KPSS, Phillips-Perron, Zivot-Andrews, Ng-Perron
- **Cointegration** - Johansen test (trace and max-eigenvalue)
- **VAR** - Granger causality (pairwise and block Wald tests), stationarity diagnostics
- **Panel VAR** - Hansen J-test for overidentifying restrictions, Andrews-Lu MMSC for lag/moment selection
- **Model Comparison** - Likelihood ratio (LR) and Lagrange multiplier (LM/score) tests for nested models
- **Stationarity diagnostics** - `unit_root_summary()`, `test_all_variables()`

### Data Management
- **Typed containers** - `TimeSeriesData`, `PanelData`, `CrossSectionData` with variable names, frequency, transformation codes, and descriptions
- **Built-in datasets** - FRED-MD (126 monthly variables) and FRED-QD (245 quarterly variables), January 2026 vintages (McCracken & Ng 2016, 2020)
- **Data diagnostics** - `diagnose()` scans for NaN/Inf/constant columns; `fix()` cleans via listwise deletion, interpolation, or mean imputation
- **FRED transformations** - `apply_tcode()` / `inverse_tcode()` for all 7 FRED transformation codes
- **Filtering** - `apply_filter()` applies HP, Hamilton, BN, BK, or boosted HP per-variable to `TimeSeriesData` and `PanelData`
- **Panel support** - `xtset()` for Stata-style panel construction; balanced/unbalanced detection
- **Summary statistics** - `describe_data()` with per-variable N, mean, std, quantiles, skewness, kurtosis
- **Seamless estimation** - All estimation functions accept `TimeSeriesData` directly

## Installation

```julia
using Pkg
Pkg.add("MacroEconometricModels")
```

## Quick Start

### Time Series Filters

```julia
using MacroEconometricModels

y = cumsum(randn(200))  # simulate I(1) series

# Hodrick-Prescott filter
hp = hp_filter(y; lambda=1600.0)
trend(hp)  # trend component
cycle(hp)  # cyclical component

# Hamilton (2018) regression filter
ham = hamilton_filter(y; h=8, p=4)

# Beveridge-Nelson decomposition
bn = beveridge_nelson(y; p=:auto, q=:auto)

# Baxter-King band-pass filter
bk = baxter_king(y; pl=6, pu=32, K=12)

# Boosted HP (Phillips & Shi 2021)
bhp = boosted_hp(y; lambda=1600.0, stopping=:BIC)
```

### ARIMA Modeling

```julia
using MacroEconometricModels

# Simulate AR(1) data
y = cumsum(randn(200))

# Automatic order selection
auto = auto_arima(y; max_p=5, max_d=2, max_q=5)

# Estimate ARIMA(1,1,1)
model = estimate_arima(y, 1, 1, 1; method=:css_mle)

# Forecast 12 steps ahead
fc = forecast(model, 12)
```

### Volatility Models

```julia
using MacroEconometricModels

y = randn(500)

# GARCH(1,1)
gm = estimate_garch(y, 1, 1)
fc = forecast(gm, 10)

# EGARCH (asymmetric leverage effects)
em = estimate_egarch(y, 1, 1)

# GJR-GARCH (threshold effects)
gjr = estimate_gjr_garch(y, 1, 1)

# Stochastic Volatility (Bayesian MCMC)
sv = estimate_sv(y; variant=:basic, n_samples=1000)
```

### VAR Analysis

```julia
using MacroEconometricModels
using Random

# Generate synthetic VAR(1) data
Random.seed!(42)
T, n = 200, 3
A = [0.5 0.1 0.0; 0.0 0.6 0.1; 0.1 0.0 0.4]
Y = zeros(T, n)
for t in 2:T
    Y[t, :] = A * Y[t-1, :] + randn(n)
end

# Estimate VAR model
model = estimate_var(Y, 2)

# Innovation accounting
irf_result = irf(model, 20; ci_type=:bootstrap)
fevd_result = fevd(model, 20)
hd_result = historical_decomposition(model, 198)

# Publication-quality tables
df = table(irf_result, 1, 1; horizons=[1, 4, 8, 12, 20])
print_table(stdout, fevd_result, 1)

# Bayesian estimation
chain = estimate_bvar(Y, 2; prior=:minnesota)
```

### VECM Analysis

```julia
using MacroEconometricModels
using Random

Random.seed!(42)
# Simulate cointegrated data
T = 200
e = randn(T, 2)
x1 = cumsum(e[:, 1])
x2 = x1 + 0.5 .* e[:, 2]  # cointegrated with x1
Y = hcat(x1, x2)

# Johansen cointegration test
jt = johansen_test(Y, 2)

# Estimate VECM with automatic rank selection
vecm = estimate_vecm(Y, 2; rank=:auto)

# Structural analysis via VAR conversion
irf_result = irf(vecm, 20; method=:cholesky)
fevd_result = fevd(vecm, 20)

# VECM-specific forecasting
fc = forecast(vecm, 12; ci_method=:bootstrap)

# Granger causality (short-run, long-run, strong)
gc = granger_causality_vecm(vecm, 1, 2)
```

### Local Projections

```julia
using MacroEconometricModels

Y = randn(200, 3)
lp_result = estimate_lp(Y, 1, 20; cov_type=:newey_west)
```

### Factor Model Forecasting

```julia
using MacroEconometricModels

X = randn(200, 50)  # 200 observations, 50 variables

# Static factor model with PCA
fm = estimate_factors(X, 3)
fc = forecast(fm, 12; ci_method=:theoretical)
fc.observables       # 12x50 forecasted observables
fc.observables_lower # lower CI bounds
fc.observables_upper # upper CI bounds

# Dynamic factor model with bootstrap CIs
dfm = estimate_dynamic_factors(X, 3, 2)
fc = forecast(dfm, 12; ci_method=:bootstrap, n_boot=500)
```

### Panel VAR

```julia
using MacroEconometricModels, DataFrames, Random

Random.seed!(42)
N, T_total, m = 50, 20, 3
data = zeros(N * T_total, m)
for i in 1:N
    mu_i = randn(m)
    for t in 2:T_total
        idx = (i-1)*T_total + t
        data[idx, :] = mu_i + 0.5 * data[(i-1)*T_total + t - 1, :] + 0.1 * randn(m)
    end
end
df = DataFrame(data, ["y1", "y2", "y3"])
df.id = repeat(1:N, inner=T_total)
df.time = repeat(1:T_total, outer=N)
pd = xtset(df, :id, :time)

# Two-step GMM estimation
model = estimate_pvar(pd, 2; steps=:twostep)

# Structural analysis
irfs = pvar_oirf(model, 10)
fv = pvar_fevd(model, 10)

# Specification tests
j = pvar_hansen_j(model)
stab = pvar_stability(model)
```

### Non-Gaussian SVAR

```julia
using MacroEconometricModels

Y = randn(200, 3)
model = estimate_var(Y, 2)

# Check residual normality
suite = normality_test_suite(model)

# ICA-based identification
ica = identify_fastica(model)
irfs = irf(model, 20; method=:fastica)

# Non-Gaussian ML identification
ml = identify_student_t(model)

# Heteroskedasticity-based identification
ms = identify_markov_switching(model; n_regimes=2)
```

### Unit Root Tests

```julia
using MacroEconometricModels

y = cumsum(randn(200))

# Individual tests
adf_test(y; lags=:aic)
kpss_test(y)
pp_test(y)

# Combined summary
unit_root_summary(y)
```

### Granger Causality

```julia
using MacroEconometricModels

Y = randn(200, 3)
m = estimate_var(Y, 2)

# Pairwise: does variable 1 Granger-cause variable 2?
g = granger_test(m, 1, 2)

# Block: do variables 1 and 2 jointly Granger-cause variable 3?
g_block = granger_test(m, [1, 2], 3)

# All pairwise tests at once (n×n p-value matrix)
results = granger_test_all(m)
```

### Data Management

```julia
using MacroEconometricModels

# Load built-in FRED-MD dataset (126 monthly macro variables)
md = load_example(:fred_md)
desc(md)                            # "FRED-MD Monthly Database, January 2026 Vintage..."
vardesc(md, "INDPRO")               # "IP Index"
refs(md)                            # McCracken & Ng (2016)

# Apply recommended FRED transformation codes
md_transformed = apply_tcode(md, md.tcode)

# Per-variable filtering (HP, Hamilton, BN, BK, boosted HP)
cyclical = apply_filter(md_transformed, :hp; component=:cycle)

# Diagnose and clean data
diag = diagnose(md_transformed)
clean = fix(md_transformed; method=:listwise)

# Estimate directly from TimeSeriesData
model = estimate_var(clean[:, ["INDPRO", "UNRATE", "CPIAUCSL"]], 4)
```

### Bibliographic References

```julia
using MacroEconometricModels

model = estimate_var(randn(100, 3), 2)

# Get references for any model or method
refs(model)                     # AEA plain text
refs(model; format=:bibtex)     # BibTeX entries
refs(:fastica; format=:latex)   # LaTeX \bibitem
refs(:johansen; format=:html)   # HTML with DOI links
```

## Documentation

Full documentation available at [https://chung9207.github.io/MacroEconometricModels.jl/dev/](https://chung9207.github.io/MacroEconometricModels.jl/dev/)

## References

### Time Series Filters

- Baxter, Marianne, and Robert G. King. 1999. "Measuring Business Cycles: Approximate Band-Pass Filters for Economic Time Series." *Review of Economics and Statistics* 81 (4): 575–593. [https://doi.org/10.1162/003465399558454](https://doi.org/10.1162/003465399558454)
- Beveridge, Stephen, and Charles R. Nelson. 1981. "A New Approach to Decomposition of Economic Time Series into Permanent and Transitory Components with Particular Attention to Measurement of the 'Business Cycle'." *Journal of Monetary Economics* 7 (2): 151–174. [https://doi.org/10.1016/0304-3932(81)90040-4](https://doi.org/10.1016/0304-3932(81)90040-4)
- Hamilton, James D. 2018. "Why You Should Never Use the Hodrick-Prescott Filter." *Review of Economics and Statistics* 100 (5): 831–843. [https://doi.org/10.1162/rest_a_00706](https://doi.org/10.1162/rest_a_00706)
- Hodrick, Robert J., and Edward C. Prescott. 1997. "Postwar U.S. Business Cycles: An Empirical Investigation." *Journal of Money, Credit and Banking* 29 (1): 1–16. [https://doi.org/10.2307/2953682](https://doi.org/10.2307/2953682)
- Phillips, Peter C. B., and Zhentao Shi. 2021. "Boosting: Why You Can Use the HP Filter." *International Economic Review* 62 (2): 521–570. [https://doi.org/10.1111/iere.12495](https://doi.org/10.1111/iere.12495)

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
- Auerbach, Alan J., and Yuriy Gorodnichenko. 2013. "Fiscal Multipliers in Recession and Expansion." In *Fiscal Policy after the Financial Crisis*, edited by Alberto Alesina and Francesco Giavazzi, 63–98. Chicago: University of Chicago Press. [https://doi.org/10.7208/9780226018584-004](https://doi.org/10.7208/9780226018584-004)
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

### GMM and Covariance Estimation

- Hansen, Lars Peter. 1982. "Large Sample Properties of Generalized Method of Moments Estimators." *Econometrica* 50 (4): 1029–1054. [https://doi.org/10.2307/1912775](https://doi.org/10.2307/1912775)
- Newey, Whitney K., and Kenneth D. West. 1987. "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica* 55 (3): 703–708. [https://doi.org/10.2307/1913610](https://doi.org/10.2307/1913610)

### Non-Gaussian Identification

- Hyvärinen, Aapo. 1999. "Fast and Robust Fixed-Point Algorithms for Independent Component Analysis." *IEEE Transactions on Neural Networks* 10 (3): 626–634. [https://doi.org/10.1109/72.761722](https://doi.org/10.1109/72.761722)
- Lanne, Markku, Mika Meitz, and Pentti Saikkonen. 2017. "Identification and Estimation of Non-Gaussian Structural Vector Autoregressions." *Journal of Econometrics* 196 (2): 288–304. [https://doi.org/10.1016/j.jeconom.2016.06.002](https://doi.org/10.1016/j.jeconom.2016.06.002)
- Lanne, Markku, and Helmut Lütkepohl. 2010. "Structural Vector Autoregressions with Nonnormal Residuals." *Journal of Business & Economic Statistics* 28 (1): 159–168. [https://doi.org/10.1198/jbes.2009.06003](https://doi.org/10.1198/jbes.2009.06003)
- Rigobon, Roberto. 2003. "Identification through Heteroskedasticity." *Review of Economics and Statistics* 85 (4): 777–792. [https://doi.org/10.1162/003465303772815727](https://doi.org/10.1162/003465303772815727)

### Unit Root and Cointegration Tests

- Dickey, David A., and Wayne A. Fuller. 1979. "Distribution of the Estimators for Autoregressive Time Series with a Unit Root." *Journal of the American Statistical Association* 74 (366): 427–431. [https://doi.org/10.1080/01621459.1979.10482531](https://doi.org/10.1080/01621459.1979.10482531)
- Johansen, Søren. 1991. "Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models." *Econometrica* 59 (6): 1551–1580. [https://doi.org/10.2307/2938278](https://doi.org/10.2307/2938278)
- Kwiatkowski, Denis, Peter C. B. Phillips, Peter Schmidt, and Yongcheol Shin. 1992. "Testing the Null Hypothesis of Stationarity Against the Alternative of a Unit Root." *Journal of Econometrics* 54 (1–3): 159–178. [https://doi.org/10.1016/0304-4076(92)90104-Y](https://doi.org/10.1016/0304-4076(92)90104-Y)
- Ng, Serena, and Pierre Perron. 2001. "Lag Length Selection and the Construction of Unit Root Tests with Good Size and Power." *Econometrica* 69 (6): 1519–1554. [https://doi.org/10.1111/1468-0262.00256](https://doi.org/10.1111/1468-0262.00256)

### Granger Causality

- Granger, Clive W. J. 1969. "Investigating Causal Relations by Econometric Models and Cross-spectral Methods." *Econometrica* 37 (3): 424–438. [https://doi.org/10.2307/1912791](https://doi.org/10.2307/1912791)

### Data Sources

- McCracken, Michael W., and Serena Ng. 2016. "FRED-MD: A Monthly Database for Macroeconomic Research." *Journal of Business & Economic Statistics* 34 (4): 574–589. [https://doi.org/10.1080/07350015.2015.1086655](https://doi.org/10.1080/07350015.2015.1086655)
- McCracken, Michael W., and Serena Ng. 2020. "FRED-QD: A Quarterly Database for Macroeconomic Research." *Federal Reserve Bank of St. Louis Working Paper* 2020-005. [https://doi.org/10.20955/wp.2020.005](https://doi.org/10.20955/wp.2020.005)

### Model Comparison Tests

- Neyman, Jerzy, and Egon S. Pearson. 1933. "On the Problem of the Most Efficient Tests of Statistical Hypotheses." *Philosophical Transactions of the Royal Society A* 231 (694–706): 289–337. [https://doi.org/10.1098/rsta.1933.0009](https://doi.org/10.1098/rsta.1933.0009)
- Rao, C. Radhakrishna. 1948. "Large Sample Tests of Statistical Hypotheses Concerning Several Parameters with Applications to Problems of Estimation." *Mathematical Proceedings of the Cambridge Philosophical Society* 44 (1): 50–57. [https://doi.org/10.1017/S0305004100023987](https://doi.org/10.1017/S0305004100023987)
- Silvey, S. D. 1959. "The Lagrangian Multiplier Test." *Annals of Mathematical Statistics* 30 (2): 389–407. [https://doi.org/10.1214/aoms/1177706259](https://doi.org/10.1214/aoms/1177706259)
- Wilks, Samuel S. 1938. "The Large-Sample Distribution of the Likelihood Ratio for Testing Composite Hypotheses." *Annals of Mathematical Statistics* 9 (1): 60–62. [https://doi.org/10.1214/aoms/1177732360](https://doi.org/10.1214/aoms/1177732360)

## License

MIT
