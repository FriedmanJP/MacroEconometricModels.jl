# MacroEconometricModels.jl

[![CI](https://github.com/chung9207/MacroEconometricModels.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/chung9207/MacroEconometricModels.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/chung9207/MacroEconometricModels.jl/graph/badge.svg)](https://codecov.io/gh/chung9207/MacroEconometricModels.jl)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://chung9207.github.io/MacroEconometricModels.jl/dev/)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18439170.svg)](https://doi.org/10.5281/zenodo.18439170)

A comprehensive Julia package for macroeconomic time series analysis. Provides VAR, Bayesian VAR, Local Projections, Factor Models, ARIMA, GMM estimation, structural identification, and hypothesis testing.

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

### Innovation Accounting
- **Impulse Response Functions (IRF)** - Bootstrap, theoretical, and Bayesian credible intervals
- **Forecast Error Variance Decomposition (FEVD)** - Frequentist and Bayesian
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
fc.observables       # 12×50 forecasted observables
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
- Arias, J. E., Rubio-Ramirez, J. F., & Waggoner, D. F. (2018). Inference Based on Structural Vector Autoregressions Identified with Sign and Zero Restrictions. *Econometrica*, 86(2), 685-720.
- Antolin-Diaz, J., & Rubio-Ramirez, J. F. (2018). Narrative Sign Restrictions for SVARs. *American Economic Review*, 108(10), 2802-2829.
- Blanchard, O. J., & Quah, D. (1989). The Dynamic Effects of Aggregate Demand and Supply Disturbances. *American Economic Review*, 79(4), 655-673.
- Kilian, L., & Lutkepohl, H. (2017). *Structural Vector Autoregressive Analysis*. Cambridge University Press.
- Lutkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
- Rubio-Ramirez, J. F., Waggoner, D. F., & Zha, T. (2010). Structural vector autoregressions. *Review of Economic Studies*, 77(2), 665-696.
- Sims, C. A. (1980). Macroeconomics and Reality. *Econometrica*, 48(1), 1-48.

### Bayesian Methods
- Banbura, M., Giannone, D., & Reichlin, L. (2010). Large Bayesian vector auto regressions. *Journal of Applied Econometrics*, 25(1), 71-92.
- Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior Selection for Vector Autoregressions. *Review of Economics and Statistics*, 97(2), 436-451.
- Litterman, R. B. (1986). Forecasting with Bayesian Vector Autoregressions. *Journal of the American Statistical Association*, 81(394), 49-54.

### Local Projections
- Angrist, J. D., Jorda, O., & Kuersteiner, G. M. (2018). Semiparametric Estimates of Monetary Policy Effects. *Journal of Business & Economic Statistics*, 36(3), 371-387.
- Auerbach, A. J., & Gorodnichenko, Y. (2013). Fiscal Multipliers in Recession and Expansion. *Fiscal Policy after the Financial Crisis*, 63-98.
- Barnichon, R., & Brownlees, C. (2019). Impulse Response Estimation by Smooth Local Projections. *Review of Economics and Statistics*, 101(3), 522-530.
- Jorda, O. (2005). Estimation and Inference of Impulse Responses by Local Projections. *American Economic Review*, 95(1), 161-182.
- Stock, J. H., & Watson, M. W. (2018). Identification and Estimation of Dynamic Causal Effects in Macroeconomics Using External Instruments. *The Economic Journal*, 128(610), 917-948.

### Factor Models
- Bai, J., & Ng, S. (2002). Determining the Number of Factors in Approximate Factor Models. *Econometrica*, 70(1), 191-221.
- Forni, M., Hallin, M., Lippi, M., & Reichlin, L. (2000). The Generalized Dynamic Factor Model. *Review of Economics and Statistics*, 82(4), 540-554.
- Stock, J. H., & Watson, M. W. (2002). Forecasting Using Principal Components from a Large Number of Predictors. *Journal of the American Statistical Association*, 97(460), 1167-1179.

### Unit Root and Cointegration Tests
- Dickey, D. A., & Fuller, W. A. (1979). Distribution of the Estimators for Autoregressive Time Series with a Unit Root. *Journal of the American Statistical Association*, 74(366), 427-431.
- Johansen, S. (1991). Estimation and Hypothesis Testing of Cointegration Vectors. *Econometrica*, 59(6), 1551-1580.
- Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y. (1992). Testing the Null Hypothesis of Stationarity. *Journal of Econometrics*, 54(1-3), 159-178.
- Ng, S., & Perron, P. (2001). Lag Length Selection and the Construction of Unit Root Tests with Good Size and Power. *Econometrica*, 69(6), 1519-1554.

### GMM and Covariance Estimation
- Hansen, L. P. (1982). Large Sample Properties of Generalized Method of Moments Estimators. *Econometrica*, 50(4), 1029-1054.
- Newey, W. K., & West, K. D. (1987). A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3), 703-708.

## License

MIT
