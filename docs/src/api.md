# API Reference

This page provides the complete API documentation for **MacroEconometricModels.jl**, organized by functionality.

## Module

```@docs
MacroEconometricModels.MacroEconometricModels
```

---

## Core Types

### VAR Models

```@docs
VARModel
AbstractVARModel
```

### Impulse Response and FEVD

```@docs
ImpulseResponse
BayesianImpulseResponse
AbstractImpulseResponse
FEVD
BayesianFEVD
AbstractFEVD
```

### Local Projection Types

```@docs
AbstractLPModel
LPModel
LPIVModel
SmoothLPModel
StateLPModel
PropensityLPModel
AbstractLPImpulseResponse
LPImpulseResponse
```

### Factor Models

```@docs
FactorModel
DynamicFactorModel
GeneralizedDynamicFactorModel
AbstractFactorModel
```

### GMM Types

```@docs
AbstractGMMModel
GMMModel
GMMWeighting
```

### Prior Types

```@docs
MinnesotaHyperparameters
AbstractPrior
```

### Covariance Estimator Types

```@docs
AbstractCovarianceEstimator
NeweyWestEstimator
WhiteEstimator
DriscollKraayEstimator
```

### SVAR Identification Types

```@docs
ZeroRestriction
SignRestriction
SVARRestrictions
AriasSVARResult
```

### Helper Types

```@docs
BSplineBasis
StateTransition
PropensityScoreConfig
```

---

## VAR Estimation

### Frequentist Estimation

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["estimation.jl"]
Order   = [:function]
```

### Bayesian Estimation

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["bayesian.jl"]
Order   = [:function]
```

### Prior Specification

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["priors.jl"]
Order   = [:function]
```

---

## Structural Analysis

### Identification Schemes

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["identification.jl"]
Order   = [:function]
```

### Impulse Response Functions

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["irf.jl"]
Order   = [:function]
```

### Forecast Error Variance Decomposition

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["fevd.jl"]
Order   = [:function]
```

---

## Local Projections

### Core LP Estimation

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["lp_estimation.jl"]
Order   = [:function]
```

### Covariance Estimation

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["lp_covariance.jl"]
Order   = [:function]
```

### LP with Instrumental Variables

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["lp_iv.jl"]
Order   = [:function]
```

### Smooth Local Projections

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["lp_smooth.jl"]
Order   = [:function]
```

### State-Dependent Local Projections

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["lp_state.jl"]
Order   = [:function]
```

### Propensity Score Local Projections

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["lp_propensity.jl"]
Order   = [:function]
```

---

## Factor Models

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["factormodels.jl"]
Order   = [:function]
```

---

## GMM Estimation

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["gmm.jl"]
Order   = [:function]
```

---

## Utility Functions

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["utils.jl"]
Order   = [:function]
```

---

## Function Index

### Estimation Functions

| Function | Description |
|----------|-------------|
| `estimate_var(Y, p)` | Estimate VAR(p) via OLS |
| `estimate_bvar(Y, p; ...)` | Estimate Bayesian VAR with MCMC |
| `estimate_lp(Y, shock_var, H; ...)` | Standard Local Projection |
| `estimate_lp_iv(Y, shock_var, Z, H; ...)` | LP with instrumental variables |
| `estimate_smooth_lp(Y, shock_var, H; ...)` | Smooth LP with B-splines |
| `estimate_state_lp(Y, shock_var, state_var, H; ...)` | State-dependent LP |
| `estimate_propensity_lp(Y, treatment, covariates, H; ...)` | LP with propensity scores |
| `doubly_robust_lp(Y, treatment, covariates, H; ...)` | Doubly robust LP estimator |
| `estimate_factors(X, r; ...)` | Static factor model via PCA |
| `estimate_gmm(moment_fn, theta0, data; ...)` | GMM estimation |

### Structural Analysis Functions

| Function | Description |
|----------|-------------|
| `irf(model, H; ...)` | Compute impulse response functions |
| `fevd(model, H; ...)` | Forecast error variance decomposition |
| `identify_cholesky(model)` | Cholesky identification |
| `identify_sign(model; ...)` | Sign restriction identification |
| `identify_long_run(model)` | Blanchard-Quah identification |
| `identify_narrative(model; ...)` | Narrative sign restrictions |

### LP IRF Extraction

| Function | Description |
|----------|-------------|
| `lp_irf(model; ...)` | Extract IRF from LPModel |
| `lp_iv_irf(model; ...)` | Extract IRF from LPIVModel |
| `smooth_lp_irf(model; ...)` | Extract smoothed IRF |
| `state_irf(model; ...)` | Extract state-dependent IRFs |
| `propensity_irf(model; ...)` | Extract ATE impulse response |

### Factor Model Functions

| Function | Description |
|----------|-------------|
| `estimate_factors(X, r; ...)` | Estimate r-factor model |
| `ic_criteria(X, r_max)` | Bai-Ng information criteria |
| `scree_plot_data(model)` | Data for scree plot |

### Diagnostic Functions

| Function | Description |
|----------|-------------|
| `optimize_hyperparameters(Y, p; ...)` | Optimize Minnesota prior |
| `weak_instrument_test(model; ...)` | Test for weak instruments |
| `sargan_test(model, h)` | Overidentification test |
| `test_regime_difference(model; ...)` | Test regime differences |
| `propensity_diagnostics(model)` | Propensity score diagnostics |
| `j_test(model)` | Hansen J-test for GMM |
| `gmm_summary(model)` | Summary statistics for GMM |

### Covariance Functions

| Function | Description |
|----------|-------------|
| `newey_west(X, residuals; ...)` | Newey-West HAC estimator |
| `white_vcov(X, residuals; ...)` | White heteroskedasticity-robust |
| `long_run_variance(x; ...)` | Long-run variance estimate |
| `long_run_covariance(X; ...)` | Long-run covariance matrix |
| `optimal_bandwidth_nw(residuals)` | Automatic bandwidth selection |

### Utility Functions

| Function | Description |
|----------|-------------|
| `construct_var_matrices(Y, p)` | Build VAR design matrices |
| `companion_matrix(B, n, p)` | VAR companion form |
| `robust_inv(A)` | Robust matrix inverse |
| `safe_cholesky(A; ...)` | Stable Cholesky decomposition |

---

## Type Hierarchy

```
AbstractVARModel
└── VARModel{T}

AbstractImpulseResponse
├── ImpulseResponse{T}
├── BayesianImpulseResponse{T}
└── AbstractLPImpulseResponse
    └── LPImpulseResponse{T}

AbstractFEVD
├── FEVD{T}
└── BayesianFEVD{T}

AbstractFactorModel
├── FactorModel{T}
├── DynamicFactorModel{T}
└── GeneralizedDynamicFactorModel{T}

AbstractLPModel
├── LPModel{T}
├── LPIVModel{T}
├── SmoothLPModel{T}
├── StateLPModel{T}
└── PropensityLPModel{T}

AbstractCovarianceEstimator
├── NeweyWestEstimator{T}
├── WhiteEstimator
└── DriscollKraayEstimator{T}

AbstractGMMModel
└── GMMModel{T}

AbstractPrior
└── MinnesotaHyperparameters{T}
```
