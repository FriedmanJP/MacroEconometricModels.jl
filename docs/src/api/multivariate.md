# [Multivariate Models API](@id api_multivariate)

VAR, VECM, BVAR, Local Projections, Factor Models, FAVAR, and Structural DFM, plus innovation accounting (IRF/FEVD/HD). See [VAR](../manual.md), [VECM](../vecm.md), [BVAR](../bayesian.md), [Local Projections](../lp.md), [Factor Models](../factormodels.md), and [FAVAR](../favar.md) for theory and examples.

---

## VAR Models

```@docs
VARModel
AbstractVARModel
```

---

## VECM Models

```@docs
VECMModel
VECMForecast
VECMGrangerResult
VECMRestrictionTest
```

---

## Analysis Result Types

```@docs
AbstractAnalysisResult
AbstractFrequentistResult
AbstractBayesianResult
```

---

## Impulse Response and FEVD

```@docs
ImpulseResponse
BayesianImpulseResponse
AbstractImpulseResponse
FEVD
BayesianFEVD
AbstractFEVD
```

---

## Historical Decomposition

```@docs
HistoricalDecomposition
BayesianHistoricalDecomposition
AbstractHistoricalDecomposition
```

---

## Factor Model Types

```@docs
FactorModel
DynamicFactorModel
GeneralizedDynamicFactorModel
FactorForecast
AbstractFactorModel
```

---

## Local Projection Types

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["lp/types.jl"]
Order   = [:type]
```

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["lp/types_variants.jl"]
Order   = [:type]
```

---

## Prior Types

```@docs
MinnesotaHyperparameters
AbstractPrior
```

---

## Bayesian Posterior Types

```@docs
BVARPosterior
```

---

## Forecast Types

```@docs
AbstractForecastResult
VARForecast
BVARForecast
```

---

## FAVAR Models

```@docs
FAVARModel
BayesianFAVAR
```

---

## Structural DFM

```@docs
StructuralDFM
```

---

## VAR Estimation

### Frequentist Estimation

```@docs
estimate_var
select_lag_order
MacroEconometricModels.StatsAPI.vcov(::VARModel)
MacroEconometricModels.StatsAPI.predict
MacroEconometricModels.StatsAPI.r2(::VARModel)
MacroEconometricModels.StatsAPI.loglikelihood
MacroEconometricModels.StatsAPI.confint(::VARModel)
```

### Bayesian Estimation

```@docs
estimate_bvar
posterior_mean_model
posterior_median_model
```

### VAR/BVAR Forecasting

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["var/estimation.jl", "bvar/estimation.jl"]
Order   = [:function]
```

### Forecast Accessors

```@docs
point_forecast
forecast_horizon
lower_bound
upper_bound
```

### Prior Specification

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["bvar/priors.jl"]
Order   = [:function]
```

### VECM Estimation

```@docs
estimate_vecm
to_var
select_vecm_rank
cointegrating_rank
granger_causality_vecm
```

### VECM Analysis and Forecasting

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["vecm/analysis.jl", "vecm/forecast.jl"]
Order   = [:function]
```

---

## Innovation Accounting

### Impulse Response Functions

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["core/irf.jl"]
Order   = [:function]
```

### Forecast Error Variance Decomposition

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["core/fevd.jl"]
Order   = [:function]
```

### Historical Decomposition

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["core/hd.jl"]
Order   = [:function]
```

### Summary Tables

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["summary.jl"]
Order   = [:function]
```

---

## Local Projections

### Core LP Estimation and Covariance

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["lp/core.jl"]
Order   = [:function]
```

### LP-IV (Stock & Watson 2018)

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["lp/iv.jl"]
Order   = [:function]
```

### Smooth LP (Barnichon & Brownlees 2019)

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["lp/smooth.jl"]
Order   = [:function]
```

### State-Dependent LP (Auerbach & Gorodnichenko 2013)

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["lp/state.jl"]
Order   = [:function]
```

### Propensity Score LP (Angrist et al. 2018)

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["lp/propensity.jl"]
Order   = [:function]
```

### LP Forecasting

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["lp/forecast.jl"]
Order   = [:function]
```

### LP-FEVD (Gorodnichenko & Lee 2019)

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["lp/fevd.jl"]
Order   = [:function]
```

---

## Factor Models

### Static Factor Model

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["factor/static.jl"]
Order   = [:function]
```

### Dynamic Factor Model

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["factor/dynamic.jl"]
Order   = [:function]
```

### Generalized Dynamic Factor Model

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["factor/generalized.jl"]
Order   = [:function]
```

---

## FAVAR

```@docs
estimate_favar
favar_panel_irf
favar_panel_forecast
```

### FAVAR/SDFM Structural Analysis

```@docs
irf(::FAVARModel{T}, ::Int) where {T}
irf(::BayesianFAVAR{T}, ::Int) where {T}
irf(::StructuralDFM{T}, ::Int) where {T}
fevd(::FAVARModel{T}, ::Int) where {T}
fevd(::BayesianFAVAR{T}, ::Int) where {T}
fevd(::StructuralDFM{T}, ::Int) where {T}
forecast(::FAVARModel{T}, ::Int) where {T}
```

---

## Structural Dynamic Factor Models

```@docs
estimate_structural_dfm
sdfm_panel_irf
```

---

## Multivariate GARCH

```@docs
AbstractMGARCHModel
```

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["mgarch/types.jl", "mgarch/ccc.jl", "mgarch/dcc.jl", "mgarch/bekk.jl"]
```

---

## Cointegrating Regression

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["cointreg/types.jl", "cointreg/fmols.jl", "cointreg/ccr.jl", "cointreg/dols.jl", "cointreg/panel.jl"]
```

---

## Systems of Equations (SUR / 3SLS)

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["system/types.jl", "system/sur.jl", "system/threesls.jl"]
```

---

## Autoregressive Distributed Lag (ARDL)

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["ardl/types.jl", "ardl/estimation.jl", "ardl/bounds.jl", "ardl/nardl.jl", "ardl/pmg.jl"]
```
