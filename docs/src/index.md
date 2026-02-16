# MacroEconometricModels.jl

*A comprehensive Julia package for macroeconometric research and analysis*

## Overview

**MacroEconometricModels.jl** provides a unified, high-performance framework for estimating and analyzing macroeconometric models in Julia. The package implements state-of-the-art methods spanning the full empirical macro workflow: from unit root testing and trend-cycle decomposition, through univariate and multivariate model estimation, to structural identification and publication-quality output.

### Key Features

**Univariate Models**

- **Time Series Filters**: Hodrick-Prescott (1997), Hamilton (2018) regression, Beveridge-Nelson (1981), Baxter-King (1999) band-pass, and boosted HP (Phillips & Shi 2021) with unified `trend()`/`cycle()` accessors
- **ARIMA**: AR, MA, ARMA, ARIMA estimation via OLS, CSS, MLE (Kalman filter), and CSS-MLE; automatic order selection (`auto_arima`); multi-step forecasting with confidence intervals
- **Volatility Models**: ARCH (Engle 1982), GARCH (Bollerslev 1986), EGARCH (Nelson 1991), GJR-GARCH (Glosten et al. 1993) via MLE; Stochastic Volatility via Kim-Shephard-Chib (1998) Gibbs sampler (basic, leverage, Student-t variants); news impact curves, ARCH-LM diagnostics, multi-step forecasting

**Multivariate Models**

- **VAR**: OLS estimation with lag order selection (AIC, BIC, HQ), stability diagnostics, companion matrix
- **Bayesian VAR**: Conjugate Normal-Inverse-Wishart posterior with Minnesota prior; direct and Gibbs samplers; automatic hyperparameter optimization via marginal likelihood (Giannone, Lenza & Primiceri 2015)
- **VECM**: Johansen MLE and Engle-Granger two-step estimation for cointegrated systems; automatic rank selection; IRF/FEVD/HD via VAR conversion (`to_var`); VECM-specific forecasting; Granger causality (short-run, long-run, strong)
- **Panel VAR**: GMM estimation via Arellano-Bond (1991) first-difference and Blundell-Bond (1998) system GMM; fixed-effects OLS; Windmeijer (2005) corrected standard errors; Hansen J-test, Andrews-Lu MMSC; OIRF, GIRF, FEVD; group-level bootstrap CIs; lag selection
- **Local Projections**: Jorda (2005) with extensions for IV (Stock & Watson 2018), smooth LP (Barnichon & Brownlees 2019), state-dependence (Auerbach & Gorodnichenko 2013), propensity score weighting (Angrist et al. 2018), structural LP (Plagborg-Moller & Wolf 2021), LP forecasting, and LP-FEVD (Gorodnichenko & Lee 2019)
- **Factor Models**: Static (PCA), dynamic (two-step/EM), and generalized dynamic (spectral GDFM) with Bai-Ng information criteria; unified forecasting with theoretical and bootstrap CIs
- **GMM**: Flexible estimation with one-step, two-step, and iterated weighting; Hansen J-test

**Innovation Accounting**

- **IRF**: Impulse responses with bootstrap, theoretical, and Bayesian credible intervals
- **FEVD**: Forecast error variance decomposition (frequentist and Bayesian)
- **Historical Decomposition**: Decompose observed movements into structural shock contributions
- **LP-FEVD**: R-squared, LP-A, and LP-B estimators (Gorodnichenko & Lee 2019)

**Structural Identification** (18+ methods)

- Cholesky, sign restrictions, long-run (Blanchard-Quah), narrative restrictions, Arias et al. (2018), Mountford-Uhlig (2009) penalty function
- Non-Gaussian ICA: FastICA, JADE, SOBI, dCov, HSIC
- Non-Gaussian ML: Student-t, mixture-normal, PML, skew-normal
- Heteroskedasticity-based: Markov-switching, GARCH, smooth-transition, external volatility
- Identifiability diagnostics: gaussianity tests, independence tests, bootstrap strength tests

**Nowcasting**

- **Dynamic Factor Model (DFM)**: EM algorithm with Kalman smoother for mixed-frequency data with arbitrary missing patterns (Banbura & Modugno 2014); Mariano-Murasawa temporal aggregation; block factor structure; AR(1)/IID idiosyncratic components
- **Large Bayesian VAR**: GLP-style Normal-Inverse-Wishart prior with hyperparameter optimization via marginal likelihood (Cimadomo et al. 2022); Minnesota shrinkage with sum-of-coefficients and co-persistence priors
- **Bridge Equations**: OLS bridge regressions combining pairs of monthly indicators via median (Banbura et al. 2023); transparent and fast baseline
- **News Decomposition**: Attribute nowcast revisions to individual data releases via Kalman gain weights
- **Panel Balancing**: `balance_panel()` fills NaN in `TimeSeriesData`/`PanelData` using DFM imputation

**Hypothesis Tests**

- Unit root: ADF, KPSS, Phillips-Perron, Zivot-Andrews, Ng-Perron
- Cointegration: Johansen trace and max-eigenvalue tests
- Granger causality: pairwise Wald, block (multivariate), all-pairs matrix
- Model comparison: likelihood ratio (LR) and Lagrange multiplier (LM/score) tests for nested models
- Normality: Jarque-Bera, Mardia, Doornik-Hansen, Henze-Zirkler
- Stationarity diagnostics: `unit_root_summary()`, `test_all_variables()`

**Data Management**

- **Typed Containers**: `TimeSeriesData`, `PanelData`, `CrossSectionData` with metadata (frequency, variable names, transformation codes)
- **Validation**: `diagnose()` detects NaN/Inf/constant columns; `fix()` repairs via listwise deletion, interpolation, or mean imputation
- **FRED Transforms**: `apply_tcode()` / `inverse_tcode()` implement all 7 FRED-MD transformation codes (McCracken & Ng 2016)
- **Panel Support**: Stata-style `xtset()` for panel construction, `group_data()` for per-entity extraction
- **Summary Statistics**: `describe_data()` with N, Mean, Std, Min, P25, Median, P75, Max, Skewness, Kurtosis
- **Estimation Dispatch**: All estimation functions accept `TimeSeriesData` directly

**Output and References**

- Display backends: switchable text, LaTeX, and HTML table output via `set_display_backend()`
- Publication-quality tables: `report()`, `table()`, `print_table()`
- Bibliographic references: `refs(model)` in AEA text, BibTeX, LaTeX, or HTML format (209 entries)

## Installation

```julia
using Pkg
Pkg.add("MacroEconometricModels")
```

Or from the Julia REPL package mode:

```
] add MacroEconometricModels
```

## Quick Start

### One-Liner Overview

```julia
using MacroEconometricModels

# Univariate
hp = hp_filter(y; lambda=1600.0)                   # Trend-cycle decomposition
ar = estimate_ar(y, 2)                              # AR(2) via OLS
garch = estimate_garch(y, 1, 1)                     # GARCH(1,1)
sv = estimate_sv(y; n_samples=2000)                  # Stochastic Volatility

# Multivariate
model = estimate_var(Y, 2)                           # VAR(2) via OLS
irfs = irf(model, 20; method=:cholesky)              # Impulse responses
post = estimate_bvar(Y, 2; prior=:minnesota)         # Bayesian VAR
vecm = estimate_vecm(Y, 2; rank=:auto)               # VECM
lp = estimate_lp(Y, 1, 20; cov_type=:newey_west)    # Local Projections
fm = estimate_factors(X, 3)                          # Factor model
gmm = estimate_gmm(g, theta0, data; weighting=:two_step)  # GMM

# Tests & diagnostics
adf = adf_test(y)                                    # Unit root test
g = granger_test(model, 1, 2)                        # Granger causality
suite = normality_test_suite(model)                   # Normality tests

# Output
refs(model)                                           # Bibliographic references
set_display_backend(:latex)                            # Switch to LaTeX tables
```

---

### Time Series Filters

```julia
using MacroEconometricModels

y = cumsum(0.5 .+ randn(200))  # Simulated I(1) quarterly GDP

hp  = hp_filter(y; lambda=1600.0)     # Hodrick-Prescott
ham = hamilton_filter(y; h=8, p=4)    # Hamilton (2018)
bn  = beveridge_nelson(y; p=2, q=0)  # Beveridge-Nelson
bk  = baxter_king(y; pl=6, pu=32)    # Baxter-King band-pass
bhp = boosted_hp(y; stopping=:BIC)   # Boosted HP

trend(hp)  # trend component
cycle(hp)  # cyclical component
```

### ARIMA

```julia
using MacroEconometricModels

best = auto_arima(y)                        # Automatic order selection
arma = estimate_arma(y, 1, 1)              # ARMA(1,1) via CSS-MLE
fc = forecast(arma, 12; conf_level=0.95)   # 12-step forecast with CIs
```

### Volatility Models

```julia
using MacroEconometricModels

garch  = estimate_garch(y, 1, 1)                    # GARCH(1,1)
egarch = estimate_egarch(y, 1, 1)                    # EGARCH(1,1)
gjr    = estimate_gjr_garch(y, 1, 1)                # GJR-GARCH(1,1)
sv     = estimate_sv(y; n_samples=2000, burnin=1000) # Stochastic Volatility

nic = news_impact_curve(egarch)    # Asymmetry diagnostic
fc = forecast(garch, 20)           # Volatility forecast
persistence(garch)                  # Volatility persistence
```

### VAR and Structural Identification

```julia
using MacroEconometricModels, Random

Random.seed!(42)
Y = randn(200, 3)
for t in 2:200
    Y[t, :] = 0.5 * Y[t-1, :] + 0.3 * randn(3)
end

model = estimate_var(Y, 2)                          # OLS estimation
irfs = irf(model, 20; method=:cholesky)             # Cholesky IRF
decomp = fevd(model, 20; method=:cholesky)          # FEVD
hd = historical_decomposition(model)                # Historical decomposition
```

### Bayesian VAR

```julia
using MacroEconometricModels

best_hyper = optimize_hyperparameters(Y, 2; grid_size=20)
post = estimate_bvar(Y, 2; n_draws=1000,
                     prior=:minnesota, hyper=best_hyper)
birf = irf(post, 20; method=:cholesky)   # Bayesian IRF with credible intervals
```

### VECM

```julia
using MacroEconometricModels

joh = johansen_test(Y, 2)                          # Cointegration test
vecm = estimate_vecm(Y, 2; rank=:auto)             # VECM estimation
irfs = irf(vecm, 20; method=:cholesky)             # IRF via VAR conversion
fc = forecast(vecm, 12; ci_method=:bootstrap)      # VECM forecast
gc = granger_causality_vecm(vecm, 1, 2)            # VECM Granger causality
```

### Local Projections

```julia
using MacroEconometricModels

lp = estimate_lp(Y, 1, 20; lags=4, cov_type=:newey_west)   # Standard LP
lpiv = estimate_lp_iv(Y, 1, Z, 20; lags=4)                  # LP-IV
slp = structural_lp(Y, 20; method=:cholesky, lags=4)        # Structural LP
lfevd = lp_fevd(slp, 20)                                     # LP-FEVD
```

### Factor Models

```julia
using MacroEconometricModels

ic = ic_criteria(X, 10)                                # Bai-Ng criteria
fm = estimate_factors(X, ic.r_IC2; standardize=true)   # Static PCA
dfm = estimate_dynamic_factors(X, 3, 2)                # Dynamic factors
fc = forecast(fm, 12; ci_method=:theoretical)           # Forecast with CIs
```

### Hypothesis Tests

```julia
using MacroEconometricModels

# Unit root tests
adf = adf_test(y; lags=:aic)
kpss = kpss_test(y)
unit_root_summary(y)                  # Combined ADF + KPSS

# Granger causality
g = granger_test(model, 1, 2)        # Pairwise
g_block = granger_test(model, [1,2], 3)  # Block
results = granger_test_all(model)    # All pairs

# Model comparison
lr = lr_test(restricted, unrestricted)   # Likelihood ratio
lm = lm_test(restricted, unrestricted)   # Lagrange multiplier
```

### Non-Gaussian Identification

```julia
using MacroEconometricModels

model = estimate_var(Y, 2)
suite = normality_test_suite(model)          # Test for non-Gaussianity
ica = identify_fastica(model)                # ICA identification
ml = identify_student_t(model)               # ML identification
irfs = irf(model, 20; method=:fastica)       # IRF with ICA-identified shocks
```

### Nowcasting

```julia
using MacroEconometricModels

# Mixed-frequency data: nM monthly + nQ quarterly variables (NaN for missing)
dfm = nowcast_dfm(Y, nM, nQ; r=2, p=1)          # DFM (EM + Kalman)
bvar = nowcast_bvar(Y, nM, nQ; lags=5)           # Large BVAR
bridge = nowcast_bridge(Y, nM, nQ)                # Bridge equations

result = nowcast(dfm)                              # Current-quarter nowcast
fc = forecast(dfm, 6)                              # 6-step forecast
news = nowcast_news(X_new, X_old, dfm, T)          # News decomposition
```

### Output and References

```julia
using MacroEconometricModels

set_display_backend(:latex)          # LaTeX tables for papers
print_table(irfs, 1, 1)             # Print IRF table

refs(model)                          # AEA-style references
refs(model; format=:bibtex)          # BibTeX for .bib files
refs(:johansen; format=:html)        # HTML with DOI links
```

## Package Structure

The package is organized into the following modules:

| Module | Description |
|--------|-------------|
| `data/` | Data containers, validation, FRED transforms, panel support, summary statistics |
| `core/` | Shared infrastructure: types, utilities, display backends, covariance estimators |
| `arima/` | ARIMA suite: types, Kalman filter, estimation (CSS/MLE), forecasting, order selection |
| `filters/` | Time series filters: HP, Hamilton, Beveridge-Nelson, Baxter-King, boosted HP |
| `arch/` | ARCH(q) estimation via MLE, volatility forecasting |
| `garch/` | GARCH, EGARCH, GJR-GARCH estimation via MLE, news impact curves, forecasting |
| `sv/` | Stochastic Volatility via KSC (1998) Gibbs sampler, posterior predictive forecasts |
| `var/` | VAR estimation (OLS), structural identification, IRF, FEVD, historical decomposition |
| `vecm/` | VECM: Johansen MLE, Engle-Granger, cointegrating vectors, forecasting, Granger causality |
| `bvar/` | Bayesian VAR: conjugate NIW posterior sampling, Minnesota prior, hyperparameter optimization |
| `lp/` | Local Projections: core, IV, smooth, state-dependent, propensity, structural LP, forecast, LP-FEVD |
| `factor/` | Static (PCA), dynamic (two-step/EM), generalized (spectral) factor models with forecasting |
| `nongaussian/` | Non-Gaussian structural identification: ICA, ML, heteroskedastic-ID |
| `teststat/` | Statistical tests: unit root, Johansen, normality, Granger causality, LR/LM, ARCH diagnostics |
| `pvar/` | Panel VAR: types, transforms, instruments, estimation (GMM/FE-OLS), analysis, bootstrap, tests |
| `gmm/` | Generalized Method of Moments |
| `nowcast/` | Nowcasting: DFM (EM + Kalman), large BVAR, bridge equations, news decomposition |
| `summary.jl` | Publication-quality summary tables and `refs()` bibliographic references |

## Mathematical Notation

Throughout this documentation, we use the following notation conventions:

| Symbol | Description |
|--------|-------------|
| ``y_t`` | ``n \times 1`` vector of endogenous variables at time ``t`` |
| ``Y`` | ``T \times n`` data matrix |
| ``p`` | Number of lags in VAR |
| ``A_i`` | ``n \times n`` coefficient matrix for lag ``i`` |
| ``\Sigma`` | ``n \times n`` reduced-form error covariance |
| ``B_0`` | ``n \times n`` contemporaneous impact matrix |
| ``\varepsilon_t`` | ``n \times 1`` structural shocks |
| ``u_t`` | ``n \times n`` reduced-form residuals |
| ``h`` | Forecast/impulse response horizon |
| ``H`` | Maximum horizon |

## References

### Univariate Time Series

- Box, George E. P., and Gwilym M. Jenkins. 1976. *Time Series Analysis: Forecasting and Control*. San Francisco: Holden-Day. ISBN 978-0-816-21104-3.
- Brockwell, Peter J., and Richard A. Davis. 1991. *Time Series: Theory and Methods*. 2nd ed. New York: Springer. ISBN 978-1-4419-0319-8.
- Harvey, Andrew C. 1993. *Time Series Models*. 2nd ed. Cambridge, MA: MIT Press. ISBN 978-0-262-08224-2.

### Time Series Filters

- Hodrick, Robert J., and Edward C. Prescott. 1997. "Postwar U.S. Business Cycles: An Empirical Investigation." *Journal of Money, Credit and Banking* 29 (1): 1--16. [https://doi.org/10.2307/2953682](https://doi.org/10.2307/2953682)
- Hamilton, James D. 2018. "Why You Should Never Use the Hodrick-Prescott Filter." *Review of Economics and Statistics* 100 (5): 831--843. [https://doi.org/10.1162/rest_a_00706](https://doi.org/10.1162/rest_a_00706)
- Beveridge, Stephen, and Charles R. Nelson. 1981. "A New Approach to Decomposition of Economic Time Series into Permanent and Transitory Components." *Journal of Monetary Economics* 7 (2): 151--174. [https://doi.org/10.1016/0304-3932(81)90040-4](https://doi.org/10.1016/0304-3932(81)90040-4)
- Baxter, Marianne, and Robert G. King. 1999. "Measuring Business Cycles: Approximate Band-Pass Filters for Economic Time Series." *Review of Economics and Statistics* 81 (4): 575--593. [https://doi.org/10.1162/003465399558454](https://doi.org/10.1162/003465399558454)
- Phillips, Peter C. B., and Zhentao Shi. 2021. "Boosting: Why You Can Use the HP Filter." *International Economic Review* 62 (2): 521--570. [https://doi.org/10.1111/iere.12495](https://doi.org/10.1111/iere.12495)

### Volatility Models

- Bollerslev, Tim. 1986. "Generalized Autoregressive Conditional Heteroskedasticity." *Journal of Econometrics* 31 (3): 307--327. [https://doi.org/10.1016/0304-4076(86)90063-1](https://doi.org/10.1016/0304-4076(86)90063-1)
- Engle, Robert F. 1982. "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation." *Econometrica* 50 (4): 987--1007. [https://doi.org/10.2307/1912773](https://doi.org/10.2307/1912773)
- Glosten, Lawrence R., Ravi Jagannathan, and David E. Runkle. 1993. "On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks." *Journal of Finance* 48 (5): 1779--1801. [https://doi.org/10.1111/j.1540-6261.1993.tb05128.x](https://doi.org/10.1111/j.1540-6261.1993.tb05128.x)
- Nelson, Daniel B. 1991. "Conditional Heteroskedasticity in Asset Returns: A New Approach." *Econometrica* 59 (2): 347--370. [https://doi.org/10.2307/2938260](https://doi.org/10.2307/2938260)
- Kim, Sangjoon, Neil Shephard, and Siddhartha Chib. 1998. "Stochastic Volatility: Likelihood Inference and Comparison with ARCH Models." *Review of Economic Studies* 65 (3): 361--393. [https://doi.org/10.1111/1467-937X.00050](https://doi.org/10.1111/1467-937X.00050)
- Taylor, Stephen J. 1986. *Modelling Financial Time Series*. Chichester: Wiley. ISBN 978-0-471-90975-7.

### VAR and Structural Identification

- Blanchard, Olivier Jean, and Danny Quah. 1989. "The Dynamic Effects of Aggregate Demand and Supply Disturbances." *American Economic Review* 79 (4): 655--673.
- Hamilton, James D. 1994. *Time Series Analysis*. Princeton, NJ: Princeton University Press. ISBN 978-0-691-04289-3.
- Kilian, Lutz, and Helmut Lutkepohl. 2017. *Structural Vector Autoregressive Analysis*. Cambridge: Cambridge University Press. [https://doi.org/10.1017/9781108164818](https://doi.org/10.1017/9781108164818)
- Lutkepohl, Helmut. 2005. *New Introduction to Multiple Time Series Analysis*. Berlin: Springer. ISBN 978-3-540-40172-8.
- Sims, Christopher A. 1980. "Macroeconomics and Reality." *Econometrica* 48 (1): 1--48. [https://doi.org/10.2307/1912017](https://doi.org/10.2307/1912017)
- Arias, Jonas E., Juan F. Rubio-Ramirez, and Daniel F. Waggoner. 2018. "Inference Based on Structural Vector Autoregressions Identified with Sign and Zero Restrictions: Theory and Applications." *Econometrica* 86 (2): 685--720. [https://doi.org/10.3982/ECTA14468](https://doi.org/10.3982/ECTA14468)
- Mountford, Andrew, and Harald Uhlig. 2009. "What Are the Effects of Fiscal Policy Shocks?" *Journal of Applied Econometrics* 24 (6): 960--992. [https://doi.org/10.1002/jae.1079](https://doi.org/10.1002/jae.1079)

### Bayesian Methods

- Doan, Thomas, Robert Litterman, and Christopher Sims. 1984. "Forecasting and Conditional Projection Using Realistic Prior Distributions." *Econometric Reviews* 3 (1): 1--100. [https://doi.org/10.1080/07474938408800053](https://doi.org/10.1080/07474938408800053)
- Giannone, Domenico, Michele Lenza, and Giorgio E. Primiceri. 2015. "Prior Selection for Vector Autoregressions." *Review of Economics and Statistics* 97 (2): 436--451. [https://doi.org/10.1162/REST_a_00483](https://doi.org/10.1162/REST_a_00483)
- Litterman, Robert B. 1986. "Forecasting with Bayesian Vector Autoregressions---Five Years of Experience." *Journal of Business & Economic Statistics* 4 (1): 25--38. [https://doi.org/10.1080/07350015.1986.10509491](https://doi.org/10.1080/07350015.1986.10509491)

### VECM and Cointegration

- Engle, Robert F., and Clive W. J. Granger. 1987. "Co-Integration and Error Correction: Representation, Estimation, and Testing." *Econometrica* 55 (2): 251--276. [https://doi.org/10.2307/1913236](https://doi.org/10.2307/1913236)
- Johansen, Soren. 1991. "Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models." *Econometrica* 59 (6): 1551--1580. [https://doi.org/10.2307/2938278](https://doi.org/10.2307/2938278)

### Local Projections

- Angrist, Joshua D., Oscar Jorda, and Guido M. Kuersteiner. 2018. "Semiparametric Estimates of Monetary Policy Effects: String Theory Revisited." *Journal of Business & Economic Statistics* 36 (3): 371--387. [https://doi.org/10.1080/07350015.2016.1204919](https://doi.org/10.1080/07350015.2016.1204919)
- Auerbach, Alan J., and Yuriy Gorodnichenko. 2013. "Fiscal Multipliers in Recession and Expansion." In *Fiscal Policy after the Financial Crisis*, edited by Alberto Alesina and Francesco Giavazzi, 63--98. Chicago: University of Chicago Press. [https://doi.org/10.7208/chicago/9780226018584.003.0003](https://doi.org/10.7208/chicago/9780226018584.003.0003)
- Barnichon, Regis, and Christian Brownlees. 2019. "Impulse Response Estimation by Smooth Local Projections." *Review of Economics and Statistics* 101 (3): 522--530. [https://doi.org/10.1162/rest_a_00778](https://doi.org/10.1162/rest_a_00778)
- Jorda, Oscar. 2005. "Estimation and Inference of Impulse Responses by Local Projections." *American Economic Review* 95 (1): 161--182. [https://doi.org/10.1257/0002828053828518](https://doi.org/10.1257/0002828053828518)
- Stock, James H., and Mark W. Watson. 2018. "Identification and Estimation of Dynamic Causal Effects in Macroeconomics Using External Instruments." *Economic Journal* 128 (610): 917--948. [https://doi.org/10.1111/ecoj.12593](https://doi.org/10.1111/ecoj.12593)
- Plagborg-Moller, Mikkel, and Christian K. Wolf. 2021. "Local Projections and VARs Estimate the Same Impulse Responses." *Econometrica* 89 (2): 955--980. [https://doi.org/10.3982/ECTA17813](https://doi.org/10.3982/ECTA17813)
- Gorodnichenko, Yuriy, and Byoungchan Lee. 2019. "Forecast Error Variance Decompositions with Local Projections." *Journal of Business & Economic Statistics* 38 (4): 921--933. [https://doi.org/10.1080/07350015.2019.1610661](https://doi.org/10.1080/07350015.2019.1610661)

### Factor Models

- Bai, Jushan, and Serena Ng. 2002. "Determining the Number of Factors in Approximate Factor Models." *Econometrica* 70 (1): 191--221. [https://doi.org/10.1111/1468-0262.00273](https://doi.org/10.1111/1468-0262.00273)
- Forni, Mario, Marc Hallin, Marco Lippi, and Lucrezia Reichlin. 2000. "The Generalized Dynamic-Factor Model: Identification and Estimation." *Review of Economics and Statistics* 82 (4): 540--554. [https://doi.org/10.1162/003465300559037](https://doi.org/10.1162/003465300559037)
- Stock, James H., and Mark W. Watson. 2002. "Forecasting Using Principal Components from a Large Number of Predictors." *Journal of the American Statistical Association* 97 (460): 1167--1179. [https://doi.org/10.1198/016214502388618960](https://doi.org/10.1198/016214502388618960)

### Panel VAR

- Holtz-Eakin, Douglas, Whitney Newey, and Harvey S. Rosen. 1988. "Estimating Vector Autoregressions with Panel Data." *Econometrica* 56 (6): 1371--1395. [https://doi.org/10.2307/1913103](https://doi.org/10.2307/1913103)
- Arellano, Manuel, and Stephen Bond. 1991. "Some Tests of Specification for Panel Data." *Review of Economic Studies* 58 (2): 277--297. [https://doi.org/10.2307/2297968](https://doi.org/10.2307/2297968)
- Blundell, Richard, and Stephen Bond. 1998. "Initial Conditions and Moment Restrictions in Dynamic Panel Data Models." *Journal of Econometrics* 87 (1): 115--143. [https://doi.org/10.1016/S0304-4076(98)00009-8](https://doi.org/10.1016/S0304-4076(98)00009-8)
- Windmeijer, Frank. 2005. "A Finite Sample Correction for the Variance of Linear Efficient Two-Step GMM Estimators." *Journal of Econometrics* 126 (1): 25--51. [https://doi.org/10.1016/j.jeconom.2004.02.005](https://doi.org/10.1016/j.jeconom.2004.02.005)
- Andrews, Donald W. K., and Biao Lu. 2001. "Consistent Model and Moment Selection Procedures for GMM Estimation." *Journal of Econometrics* 101 (1): 123--164. [https://doi.org/10.1016/S0304-4076(00)00077-4](https://doi.org/10.1016/S0304-4076(00)00077-4)

### Robust Inference

- Andrews, Donald W. K. 1991. "Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation." *Econometrica* 59 (3): 817--858. [https://doi.org/10.2307/2938229](https://doi.org/10.2307/2938229)
- Hansen, Lars Peter. 1982. "Large Sample Properties of Generalized Method of Moments Estimators." *Econometrica* 50 (4): 1029--1054. [https://doi.org/10.2307/1912775](https://doi.org/10.2307/1912775)
- Newey, Whitney K., and Kenneth D. West. 1987. "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica* 55 (3): 703--708. [https://doi.org/10.2307/1913610](https://doi.org/10.2307/1913610)
- Newey, Whitney K., and Kenneth D. West. 1994. "Automatic Lag Selection in Covariance Matrix Estimation." *Review of Economic Studies* 61 (4): 631--653. [https://doi.org/10.2307/2297912](https://doi.org/10.2307/2297912)

### Non-Gaussian Identification

- Hyvarinen, Aapo. 1999. "Fast and Robust Fixed-Point Algorithms for Independent Component Analysis." *IEEE Transactions on Neural Networks* 10 (3): 626--634. [https://doi.org/10.1109/72.761722](https://doi.org/10.1109/72.761722)
- Lanne, Markku, and Helmut Lutkepohl. 2010. "Structural Vector Autoregressions with Nonnormal Residuals." *Journal of Business & Economic Statistics* 28 (1): 159--168. [https://doi.org/10.1198/jbes.2009.06003](https://doi.org/10.1198/jbes.2009.06003)
- Lanne, Markku, Mika Meitz, and Pentti Saikkonen. 2017. "Identification and Estimation of Non-Gaussian Structural Vector Autoregressions." *Journal of Econometrics* 196 (2): 288--304. [https://doi.org/10.1016/j.jeconom.2016.06.002](https://doi.org/10.1016/j.jeconom.2016.06.002)

### Hypothesis Tests

- Dickey, David A., and Wayne A. Fuller. 1979. "Distribution of the Estimators for Autoregressive Time Series with a Unit Root." *Journal of the American Statistical Association* 74 (366): 427--431. [https://doi.org/10.1080/01621459.1979.10482531](https://doi.org/10.1080/01621459.1979.10482531)
- Kwiatkowski, Denis, Peter C. B. Phillips, Peter Schmidt, and Yongcheol Shin. 1992. "Testing the Null Hypothesis of Stationarity Against the Alternative of a Unit Root." *Journal of Econometrics* 54 (1--3): 159--178. [https://doi.org/10.1016/0304-4076(92)90104-Y](https://doi.org/10.1016/0304-4076(92)90104-Y)
- Granger, Clive W. J. 1969. "Investigating Causal Relations by Econometric Models and Cross-spectral Methods." *Econometrica* 37 (3): 424--438. [https://doi.org/10.2307/1912791](https://doi.org/10.2307/1912791)
- Wilks, Samuel S. 1938. "The Large-Sample Distribution of the Likelihood Ratio for Testing Composite Hypotheses." *Annals of Mathematical Statistics* 9 (1): 60--62. [https://doi.org/10.1214/aoms/1177732360](https://doi.org/10.1214/aoms/1177732360)

### Nowcasting

- Banbura, Marta, and Michele Modugno. 2014. "Maximum Likelihood Estimation of Factor Models on Datasets with Arbitrary Pattern of Missing Data." *Journal of Applied Econometrics* 29 (1): 133--160. [https://doi.org/10.1002/jae.2306](https://doi.org/10.1002/jae.2306)
- Cimadomo, Jacopo, Domenico Giannone, Michele Lenza, Francesca Monti, and Andrej Sokol. 2022. "Nowcasting with Large Bayesian Vector Autoregressions." *ECB Working Paper* No. 2696.
- Banbura, Marta, Irina Belousova, Katalin Bodnar, and Mate Barnabas Toth. 2023. "Nowcasting Employment in the Euro Area." *ECB Working Paper* No. 2815.

## License

This package is released under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Contributing

Contributions are welcome! Please see the [GitHub repository](https://github.com/chung9207/MacroEconometricModels.jl) for contribution guidelines.

## Contents

```@contents
Pages = ["data.md", "filters.md", "arima.md", "volatility.md", "manual.md", "bayesian.md", "vecm.md", "pvar.md", "lp.md", "factormodels.md", "innovation_accounting.md", "nowcast.md", "nongaussian.md", "hypothesis_tests.md", "examples.md", "api.md", "api_types.md", "api_functions.md"]
Depth = 2
```
