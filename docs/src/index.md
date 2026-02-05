# MacroEconometricModels.jl

*A comprehensive Julia package for macroeconometric research and analysis*

## Overview

**MacroEconometricModels.jl** provides a unified, high-performance framework for estimating and analyzing macroeconometric models in Julia. The package implements state-of-the-art methods for Vector Autoregression (VAR), Bayesian VAR (BVAR), Local Projections (LP), Factor Models, and Generalized Method of Moments (GMM) estimation.

### Key Features

- **Vector Autoregression (VAR)**: OLS estimation with comprehensive diagnostics, impulse response functions (IRFs), and forecast error variance decomposition (FEVD)
- **Structural Identification**: Multiple identification schemes including Cholesky, sign restrictions, long-run (Blanchard-Quah), and narrative restrictions
- **Bayesian VAR**: Minnesota/Litterman prior with automatic hyperparameter optimization via marginal likelihood (Giannone, Lenza & Primiceri, 2015)
- **Local Projections**: Jordà (2005) methodology with extensions for IV (Stock & Watson, 2018), smooth LP (Barnichon & Brownlees, 2019), state-dependence (Auerbach & Gorodnichenko, 2013), and propensity score methods (Angrist, Jordà & Kuersteiner, 2018)
- **Factor Models**: Static, dynamic, and generalized dynamic factor models with Bai & Ng (2002) information criteria
- **Hypothesis Tests**: Comprehensive unit root tests (ADF, KPSS, Phillips-Perron, Zivot-Andrews, Ng-Perron) and Johansen cointegration test
- **GMM Estimation**: Flexible GMM framework with one-step, two-step, and iterated estimation
- **Robust Inference**: Newey-West, White, and Driscoll-Kraay HAC standard errors with automatic bandwidth selection

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

### Basic VAR Estimation

```julia
using MacroEconometricModels
using Random

# Generate synthetic macroeconomic data
Random.seed!(42)
T, n = 200, 3  # 200 observations, 3 variables
Y = randn(T, n)
for t in 2:T
    Y[t, :] = 0.5 * Y[t-1, :] + 0.3 * randn(3)
end

# Estimate VAR(2) model
model = fit(VARModel, Y, 2)

# Compute impulse responses (20 periods ahead)
irfs = irf(model, 20; method=:cholesky)

# Forecast error variance decomposition
decomp = fevd(model, 20; method=:cholesky)
```

### Bayesian VAR with Minnesota Prior

```julia
using MacroEconometricModels

# Set hyperparameters (or use optimize_hyperparameters)
hyper = MinnesotaHyperparameters(
    τ = 0.5,      # Overall tightness
    d = 2.0,      # Lag decay
    ω_own = 1.0,  # Own-lag variance
    ω_cross = 1.0, # Cross-lag variance
    ω_det = 1.0   # Deterministic terms
)

# Estimate BVAR with MCMC
chain = estimate_bvar(Y, 2; n_samples=2000, n_adapts=500,
                      prior=:minnesota, hyper=hyper)

# Bayesian IRF with credible intervals
birf = irf(chain, 2, 3, 20; method=:cholesky)
```

### Local Projections

```julia
using MacroEconometricModels

# Standard Local Projection (Jordà 2005)
lp_model = estimate_lp(Y, 1, 20; lags=4, cov_type=:newey_west)
lp_irfs = lp_irf(lp_model)

# LP with Instrumental Variables (Stock & Watson 2018)
Z = randn(T, 1)  # External instrument
lpiv_model = estimate_lp_iv(Y, 1, Z, 20; lags=4)
lpiv_irfs = lp_iv_irf(lpiv_model)
```

### Factor Models

```julia
using MacroEconometricModels

# Large panel: T observations, N variables
X = randn(200, 100)

# Determine optimal number of factors (Bai & Ng 2002)
ic = ic_criteria(X, 10)
r_optimal = ic.r_IC2

# Estimate static factor model
fm = estimate_factors(X, r_optimal)

# Extract factors for use in FAVAR
factors = fm.factors
```

### Unit Root Tests

```julia
using MacroEconometricModels

# Test for unit root
y = cumsum(randn(200))  # Random walk (has unit root)

# Augmented Dickey-Fuller test
adf_result = adf_test(y; lags=:aic, regression=:constant)

# KPSS stationarity test (opposite null hypothesis)
kpss_result = kpss_test(y; regression=:constant)

# Johansen cointegration test for multivariate data
Y = randn(200, 3)
johansen_result = johansen_test(Y, 2; deterministic=:constant)
```

## Package Structure

The package is organized into the following modules:

| Module | Description |
|--------|-------------|
| `types.jl` | Core type definitions for VAR, BVAR, IRF, FEVD results |
| `estimation.jl` | VAR/BVAR estimation via OLS and MCMC |
| `bayesian.jl` | Bayesian inference with Turing.jl |
| `priors.jl` | Minnesota prior and hyperparameter optimization |
| `identification.jl` | Structural identification schemes |
| `irf.jl` | Impulse response function computation |
| `fevd.jl` | Forecast error variance decomposition |
| `hd.jl` | Historical decomposition |
| `summary.jl` | Publication-quality summary tables |
| `lp_*.jl` | Local Projections suite |
| `factormodels.jl` | Static, dynamic, and generalized dynamic factor models |
| `unitroot.jl` | Unit root and cointegration tests (ADF, KPSS, PP, ZA, Ng-Perron, Johansen) |
| `gmm.jl` | Generalized Method of Moments |
| `utils.jl` | Numerical utilities |

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

### Core Methodology

- Blanchard, O. J., & Quah, D. (1989). "The Dynamic Effects of Aggregate Demand and Supply Disturbances." *American Economic Review*, 79(4), 655-673.
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
- Kilian, L., & Lütkepohl, H. (2017). *Structural Vector Autoregressive Analysis*. Cambridge University Press.
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
- Sims, C. A. (1980). "Macroeconomics and Reality." *Econometrica*, 48(1), 1-48.

### Bayesian Methods

- Doan, T., Litterman, R., & Sims, C. (1984). "Forecasting and Conditional Projection Using Realistic Prior Distributions." *Econometric Reviews*, 3(1), 1-100.
- Giannone, D., Lenza, M., & Primiceri, G. E. (2015). "Prior Selection for Vector Autoregressions." *Review of Economics and Statistics*, 97(2), 436-451.
- Litterman, R. B. (1986). "Forecasting with Bayesian Vector Autoregressions—Five Years of Experience." *Journal of Business & Economic Statistics*, 4(1), 25-38.

### Local Projections

- Angrist, J. D., Jordà, Ò., & Kuersteiner, G. M. (2018). "Semiparametric Estimates of Monetary Policy Effects: String Theory Revisited." *Journal of Business & Economic Statistics*, 36(3), 371-387.
- Auerbach, A. J., & Gorodnichenko, Y. (2013). "Fiscal Multipliers in Recession and Expansion." In *Fiscal Policy after the Financial Crisis*.
- Barnichon, R., & Brownlees, C. (2019). "Impulse Response Estimation by Smooth Local Projections." *Review of Economics and Statistics*, 101(3), 522-530.
- Jordà, Ò. (2005). "Estimation and Inference of Impulse Responses by Local Projections." *American Economic Review*, 95(1), 161-182.
- Stock, J. H., & Watson, M. W. (2018). "Identification and Estimation of Dynamic Causal Effects in Macroeconomics Using External Instruments." *The Economic Journal*, 128(610), 917-948.

### Factor Models

- Bai, J., & Ng, S. (2002). "Determining the Number of Factors in Approximate Factor Models." *Econometrica*, 70(1), 191-221.
- Stock, J. H., & Watson, M. W. (2002). "Forecasting Using Principal Components from a Large Number of Predictors." *Journal of the American Statistical Association*, 97(460), 1167-1179.

### Robust Inference

- Andrews, D. W. K. (1991). "Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation." *Econometrica*, 59(3), 817-858.
- Hansen, L. P. (1982). "Large Sample Properties of Generalized Method of Moments Estimators." *Econometrica*, 50(4), 1029-1054.
- Newey, W. K., & West, K. D. (1987). "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3), 703-708.
- Newey, W. K., & West, K. D. (1994). "Automatic Lag Selection in Covariance Matrix Estimation." *Review of Economic Studies*, 61(4), 631-653.

## License

This package is released under the MIT License.

## Contributing

Contributions are welcome! Please see the [GitHub repository](https://github.com/chung9207/MacroEconometricModels.jl) for contribution guidelines.

## Contents

```@contents
Pages = ["manual.md", "lp.md", "factormodels.md", "bayesian.md", "innovation_accounting.md", "hypothesis_tests.md", "api.md", "examples.md"]
Depth = 2
```
