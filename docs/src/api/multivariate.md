# [Multivariate Models API](@id api_multivariate)

Multi-equation models: VAR, VECM, BVAR (including time-varying-parameter and mixed-frequency variants), Local Projections, factor models, FAVAR, and Structural DFM, plus innovation accounting (IRF/FEVD/historical decomposition), multivariate GARCH, cointegrating regression, systems of equations, and ARDL. See [VAR](@ref var_page), [VECM](@ref vecm_page), [BVAR](@ref bvar_page), [Local Projections](@ref lp_page), [Factor Models](@ref factor_page), [FAVAR](@ref favar_page), [ARDL](@ref ardl_page), and [Cointegrating Regression](@ref cointreg_page) for theory and examples.

---

## VAR Models

```@docs
VARModel
AbstractVARModel
effective_nobs(::VARModel)
```

---

## VECM Models

`VECMModel` stores estimated Vector Error Correction Model parameters (``\alpha``, ``\beta``, ``\Gamma``, ``\mu``, ``\Sigma``), cointegrating rank, and test diagnostics. It supports the dynamic property `m.eigenvalues`, which delegates to `m.johansen_result.eigenvalues` to expose the ordered eigenvalues from Johansen reduced-rank regression.

`VECMRestrictionTest` holds log-likelihood ratio test statistics, degrees of freedom, and refitted models under restrictions on ``\beta`` (`test_beta_restriction`), ``\alpha`` (`test_alpha_restriction`), weak exogeneity (`test_weak_exogeneity`), known ``\beta`` (`test_known_beta`), and joint restrictions (`test_joint_restriction`). For Johansen Cases 2 and 4, restrictions on ``\beta`` automatically embed into ``(n+1)``-space via ``H^* = [H \; 0; 0 \; 1]``, leaving restricted deterministic terms unconstrained and preserving ``\mathrm{df} = r(n-s)``.

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
TVPVARPosterior
MFVARPosterior
GLPHyperparameters
```

### Time-Varying Parameter VAR

```@docs
estimate_tvpvar
volatility_path
irf(::TVPVARPosterior{T}, ::Int) where {T<:AbstractFloat}
```

### Mixed-Frequency VAR

```@docs
estimate_mfvar
latent_path
irf(::MFVARPosterior, ::Int)
forecast(::MFVARPosterior, ::Int)
```

### Hyperparameter Selection

```@docs
optimize_hyperparameters_glp
```

---

## Forecast Types

```@docs
AbstractForecastResult
VARForecast
BVARForecast
ConditionalForecast
ForecastCondition
```

### Conditional Forecasting

```@docs
conditional_forecast
forecast_condition
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

Reduced-form estimation, lag-order selection, forecasting, and the `StatsAPI` accessors
defined for `VARModel`.

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["src/var/estimation.jl"]
Order   = [:function]
Private = false
```

### Bayesian Estimation

Posterior sampling, posterior summary models, and predictive simulation for the BVAR.

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["src/bvar/estimation.jl"]
Order   = [:function]
Private = false
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

`estimate_vecm` supports automatic rank selection via the Johansen trace test (`rank=:auto`) or explicit rank specification, and all five Johansen (1995) deterministic specifications via integer IDs (`1:5`) or symbols:
- Case 1: `1` or `:none` (no deterministic terms)
- Case 2: `2`, `:constant`, or `:restricted_constant` (restricted intercept in ``\beta``; default)
- Case 3: `3`, `:unrestricted_constant`, or `:drift` (unrestricted constant in VAR)
- Case 4: `4`, `:trend`, or `:restricted_trend` (restricted linear trend in ``\beta``)
- Case 5: `5` or `:unrestricted_trend` (unrestricted trend in VAR)

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
Private = false
```

---

## Innovation Accounting

### Impulse Response Functions

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["core/irf.jl"]
Order   = [:function]
Private = false
```

### Forecast Error Variance Decomposition

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["core/fevd.jl"]
Order   = [:function]
Private = false
```

### Historical Decomposition

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["core/hd.jl"]
Order   = [:function]
Private = false
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

### LP-IV Weak-Instrument-Robust Inference

```@docs
montiel_olea_pflueger_f
lp_iv_ar_band
MontielOleaPfluegerF
LPIVARBand
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
Private = false
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
Private = false
```

### LP-FEVD (Gorodnichenko & Lee 2019)

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["lp/fevd.jl"]
Order   = [:function]
Private = false
```

---

## Factor Models

### Static Factor Model

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["factor/static.jl"]
Order   = [:function]
Private = false
```

### Dynamic Factor Model

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["factor/dynamic.jl"]
Order   = [:function]
Private = false
```

### Generalized Dynamic Factor Model

```@docs
HallinLiskaResult
BaiNgQResult
AmengualWatsonResult
```

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["factor/generalized.jl"]
Order   = [:function]
Private = false
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
historical_decomposition(::StructuralDFM{T}) where {T}
historical_decomposition(::GeneralizedDynamicFactorModel{T}) where {T}
forecast(::FAVARModel{T}, ::Int) where {T}
```

---

## Structural Dynamic Factor Models

```@docs
estimate_structural_dfm
sdfm_panel_irf
varindex
structural_shocks
forecast(::StructuralDFM{T}, ::Int) where {T<:AbstractFloat}
is_stable(::StructuralDFM)
MacroEconometricModels.StatsAPI.predict(::StructuralDFM)
MacroEconometricModels.StatsAPI.residuals(::StructuralDFM)
MacroEconometricModels.StatsAPI.coef(::StructuralDFM)
```

---

## Multivariate GARCH

```@docs
AbstractMGARCHModel
```

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["mgarch/types.jl", "mgarch/ccc.jl", "mgarch/dcc.jl", "mgarch/bekk.jl", "mgarch/forecast.jl"]
Private = false
```

---

## Cointegrating Regression

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["cointreg/types.jl", "cointreg/fmols.jl", "cointreg/ccr.jl", "cointreg/dols.jl", "cointreg/panel.jl"]
Private = false
```

---

## Systems of Equations (SUR / 3SLS)

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["system/types.jl", "system/sur.jl", "system/threesls.jl"]
Private = false
```

---

## Autoregressive Distributed Lag (ARDL)

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["ardl/types.jl", "ardl/estimation.jl", "ardl/bounds.jl", "ardl/nardl.jl", "ardl/pmg.jl"]
```
