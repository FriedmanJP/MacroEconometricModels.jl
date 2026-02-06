# MacroEconometricModels.jl

[![CI](https://github.com/chung9207/MacroEconometricModels.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/chung9207/MacroEconometricModels.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/chung9207/MacroEconometricModels.jl/graph/badge.svg)](https://codecov.io/gh/chung9207/MacroEconometricModels.jl)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://chung9207.github.io/MacroEconometricModels.jl/dev/)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18439170.svg)](https://doi.org/10.5281/zenodo.18439170)

A comprehensive Julia package for macroeconomic time series analysis. Provides VAR, Bayesian VAR, Local Projections, Factor Models, ARIMA, GMM estimation, structural identification (including non-Gaussian and heteroskedasticity-based methods), and hypothesis testing.

## Features

### Univariate Models
- **ARIMA** - AR, MA, ARMA, ARIMA estimation via CSS, exact MLE (Kalman filter), or CSS-MLE
- **Automatic order selection** - `auto_arima` with grid search over (p,d,q), AIC/BIC information criteria
- **Forecasting** - Multi-step ahead with confidence intervals via psi-weight accumulation

### Multivariate Estimation
- **Vector Autoregression (VAR)** - OLS estimation with lag order selection (AIC, BIC, HQ)
- **Bayesian VAR (BVAR)** - Minnesota priors with hyperparameter optimization (Giannone, Lenza & Primiceri 2015)
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
- **Generalized Method of Moments (GMM)** - One-step, two-step, and iterated; Hansen J-test

### Structural Identification
- Cholesky decomposition (recursive)
- Sign restrictions (Rubio-Ramirez, Waggoner & Zha 2010)
- Narrative restrictions (Antolin-Diaz & Rubio-Ramirez 2018)
- Long-run restrictions (Blanchard & Quah 1989)
- Arias, Rubio-Ramirez & Waggoner (2018) algorithm for zero and sign restrictions

### Non-Gaussian SVAR Identification
- **ICA-based methods** - FastICA, JADE, SOBI, distance covariance, HSIC
- **Non-Gaussian ML** - Student-t, mixture-normal, pseudo-ML, skew-normal
- **Heteroskedasticity-based** - Markov-switching, GARCH, smooth-transition, external volatility
- **Multivariate normality tests** - Jarque-Bera, Mardia, Doornik-Hansen, Henze-Zirkler
- **Identifiability diagnostics** - Shock gaussianity, LR tests, independence tests, bootstrap strength tests
- Seamless integration: `irf(model, 20; method=:fastica)` works out of the box

### Innovation Accounting
- **Impulse Response Functions (IRF)** - Bootstrap, theoretical, and Bayesian credible intervals
- **Forecast Error Variance Decomposition (FEVD)** - Frequentist and Bayesian
- **LP-FEVD** - R², LP-A, LP-B estimators with bootstrap CIs (Gorodnichenko & Lee 2019)
- **Historical Decomposition (HD)** - Decompose observed movements into structural shock contributions
- **Summary Tables** - Publication-quality output with `summary()`, `table()`, `print_table()`

### Hypothesis Tests
- **Unit Root Tests** - ADF, KPSS, Phillips-Perron, Zivot-Andrews, Ng-Perron
- **Cointegration** - Johansen test (trace and max-eigenvalue)
- **Stationarity diagnostics** - `unit_root_summary()`, `test_all_variables()`

## Installation

```julia
using Pkg
Pkg.add("MacroEconometricModels")
```

## Quick Start

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

## Documentation

Full documentation available at [https://chung9207.github.io/MacroEconometricModels.jl/dev/](https://chung9207.github.io/MacroEconometricModels.jl/dev/)

## References

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

### Non-Gaussian SVAR

- Hyvärinen, Aapo. 1999. "Fast and Robust Fixed-Point Algorithms for Independent Component Analysis." *IEEE Transactions on Neural Networks* 10 (3): 626–634. [https://doi.org/10.1109/72.761722](https://doi.org/10.1109/72.761722)
- Lanne, Markku, Mika Meitz, and Pentti Saikkonen. 2017. "Identification and Estimation of Non-Gaussian Structural Vector Autoregressions." *Journal of Econometrics* 196 (2): 288–304. [https://doi.org/10.1016/j.jeconom.2016.06.002](https://doi.org/10.1016/j.jeconom.2016.06.002)
- Lanne, Markku, and Helmut Lütkepohl. 2010. "Structural Vector Autoregressions with Nonnormal Residuals." *Journal of Business & Economic Statistics* 28 (1): 159–168. [https://doi.org/10.1198/jbes.2009.06003](https://doi.org/10.1198/jbes.2009.06003)
- Rigobon, Roberto. 2003. "Identification through Heteroskedasticity." *Review of Economics and Statistics* 85 (4): 777–792. [https://doi.org/10.1162/003465303772815727](https://doi.org/10.1162/003465303772815727)

### Unit Root and Cointegration Tests

- Dickey, David A., and Wayne A. Fuller. 1979. "Distribution of the Estimators for Autoregressive Time Series with a Unit Root." *Journal of the American Statistical Association* 74 (366): 427–431. [https://doi.org/10.1080/01621459.1979.10482531](https://doi.org/10.1080/01621459.1979.10482531)
- Johansen, Søren. 1991. "Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models." *Econometrica* 59 (6): 1551–1580. [https://doi.org/10.2307/2938278](https://doi.org/10.2307/2938278)
- Kwiatkowski, Denis, Peter C. B. Phillips, Peter Schmidt, and Yongcheol Shin. 1992. "Testing the Null Hypothesis of Stationarity Against the Alternative of a Unit Root." *Journal of Econometrics* 54 (1–3): 159–178. [https://doi.org/10.1016/0304-4076(92)90104-Y](https://doi.org/10.1016/0304-4076(92)90104-Y)
- Ng, Serena, and Pierre Perron. 2001. "Lag Length Selection and the Construction of Unit Root Tests with Good Size and Power." *Econometrica* 69 (6): 1519–1554. [https://doi.org/10.1111/1468-0262.00256](https://doi.org/10.1111/1468-0262.00256)

### GMM and Covariance Estimation

- Hansen, Lars Peter. 1982. "Large Sample Properties of Generalized Method of Moments Estimators." *Econometrica* 50 (4): 1029–1054. [https://doi.org/10.2307/1912775](https://doi.org/10.2307/1912775)
- Newey, Whitney K., and Kenneth D. West. 1987. "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica* 55 (3): 703–708. [https://doi.org/10.2307/1913610](https://doi.org/10.2307/1913610)

## License

MIT
