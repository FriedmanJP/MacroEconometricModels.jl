# MacroEconometricModels.jl

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://chung9207.github.io/MacroEconometricModels.jl/dev/)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18439170.svg)](https://doi.org/10.5281/zenodo.18439170)

A Julia package for macroeconomic time series analysis.

## Features

- **Vector Autoregression (VAR)** - OLS estimation with lag order selection
- **Bayesian VAR (BVAR)** - Minnesota priors with hyperparameter optimization
- **Structural Identification** - Cholesky, sign restrictions, narrative, long-run restrictions
- **Impulse Response Functions (IRF)** - Bootstrap and Bayesian confidence intervals
- **Forecast Error Variance Decomposition (FEVD)**
- **Factor Models** - Static, Dynamic, and Generalized Dynamic Factor Models
- **Local Projections (LP)**
  - HAC standard errors (Jordà 2005)
  - Instrumental Variables (Stock & Watson 2018)
  - Smooth IRF via B-splines (Barnichon & Brownlees 2019)
  - State-dependent LP (Auerbach & Gorodnichenko 2013)
  - Propensity Score Matching (Angrist et al. 2018)
- **Generalized Method of Moments (GMM)**

## Installation

```julia
using Pkg
Pkg.add("MacroEconometricModels")
```

## Quick Start

```julia
using MacroEconometricModels

# Estimate a VAR model
Y = randn(100, 3)
model = estimate_var(Y, 2)

# Compute IRFs with bootstrap confidence intervals
irf_result = irf(model, 20; ci_type=:bootstrap)

# Local Projection IRFs with HAC standard errors
lp_result = estimate_lp(Y, 1, 20; cov_type=:newey_west)
lp_irf_result = lp_irf(lp_result)

# Bayesian estimation
chain = estimate_bvar(Y, 2; prior=:minnesota)
```

## References

- Bańbura, M., Giannone, D., & Reichlin, L. (2010). Large Bayesian vector auto regressions.
- Lütkepohl, H. (2005). New Introduction to Multiple Time Series Analysis.
- Rubio-Ramírez, J. F., Waggoner, D. F., & Zha, T. (2010). Structural vector autoregressions.
- Jordà, Ò. (2005). Estimation and Inference of Impulse Responses by Local Projections.
- Stock, J. H., & Watson, M. W. (2018). Identification and Estimation of Dynamic Causal Effects.
- Barnichon, R., & Brownlees, C. (2019). Impulse Response Estimation by Smooth Local Projections.
- Auerbach, A. J., & Gorodnichenko, Y. (2013). Fiscal Multipliers in Recession and Expansion.
- Angrist, J. D., Jordà, Ò., & Kuersteiner, G. M. (2018). Semiparametric Estimates of Monetary Policy Effects.

## License

MIT
