# [API Functions](@id api_functions)

This page documents all functions in **MacroEconometricModels.jl**, organized by module.

---

## Data Management

### Validation and Cleaning

```@docs
diagnose
fix
validate_for_model
```

### FRED Transformations

```@docs
apply_tcode
inverse_tcode
```

### Filtering

```@docs
apply_filter
```

### Summary Statistics

```@docs
describe_data
```

### Panel Data

```@docs
xtset
isbalanced
groups
ngroups
group_data
panel_summary
```

### Data Accessors and Conversion

```@docs
MacroEconometricModels.StatsAPI.nobs(::TimeSeriesData)
nvars
varnames
frequency
time_index
obs_id
desc
vardesc
rename_vars!
set_time_index!
set_obs_id!
set_desc!
set_vardesc!
to_matrix
to_vector
load_example
```

---

## Time Series Filters

```@docs
hp_filter
hamilton_filter
beveridge_nelson
baxter_king
boosted_hp
trend
cycle
```

---

## ARIMA Models

### Estimation

```@docs
estimate_ar
estimate_ma
estimate_arma
estimate_arima
```

### Forecasting

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["arima/forecast.jl"]
Order   = [:function]
```

### Order Selection

```@docs
select_arima_order
auto_arima
ic_table
```

### ARIMA Accessors

```@docs
ar_order
ma_order
diff_order
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

## Structural Identification

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["var/identification.jl"]
Order   = [:function]
```

### Mountford-Uhlig (2009) Penalty Function

```@docs
UhligSVARResult
identify_uhlig
```

---

## Innovation Accounting

### Impulse Response Functions

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["var/irf.jl"]
Order   = [:function]
```

### Forecast Error Variance Decomposition

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["var/fevd.jl"]
Order   = [:function]
```

### Historical Decomposition

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["var/hd.jl"]
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

## Panel VAR

### Estimation

```@docs
estimate_pvar
estimate_pvar_feols
```

### Structural Analysis

```@docs
pvar_oirf
pvar_girf
pvar_fevd
pvar_stability
```

### Bootstrap

```@docs
pvar_bootstrap_irf
```

### Specification Tests

```@docs
pvar_hansen_j
pvar_mmsc
pvar_lag_selection
```

### GMM Utilities

```@docs
linear_gmm_solve
gmm_sandwich_vcov
andrews_lu_mmsc
```

---

## GMM Estimation

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["gmm/gmm.jl"]
Order   = [:function]
```

---

## Unit Root and Cointegration Tests

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["teststat/adf.jl", "teststat/kpss.jl", "teststat/pp.jl", "teststat/za.jl", "teststat/ngperron.jl", "teststat/johansen.jl", "teststat/stationarity.jl", "teststat/convenience.jl"]
Order   = [:function]
```

---

## Model Comparison Tests

```@docs
lr_test
lm_test
```

---

## Granger Causality Tests

```@docs
granger_test
granger_test_all
```

---

## Volatility Models

### ARCH Estimation and Diagnostics

```@docs
estimate_arch
arch_lm_test
ljung_box_squared
```

### GARCH Estimation and Diagnostics

```@docs
estimate_garch
estimate_egarch
estimate_gjr_garch
news_impact_curve
```

### Stochastic Volatility

```@docs
estimate_sv
```

### Volatility Forecasting

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["arch/forecast.jl", "garch/forecast.jl", "sv/forecast.jl"]
Order   = [:function]
```

### Volatility Accessors

```@docs
persistence
halflife
unconditional_variance
arch_order
garch_order
```

### Volatility StatsAPI Interface

```@docs
MacroEconometricModels.StatsAPI.nobs(::ARCHModel)
MacroEconometricModels.StatsAPI.coef(::ARCHModel)
MacroEconometricModels.StatsAPI.residuals(::ARCHModel)
MacroEconometricModels.StatsAPI.aic(::ARCHModel)
MacroEconometricModels.StatsAPI.bic(::ARCHModel)
MacroEconometricModels.StatsAPI.dof(::ARCHModel)
MacroEconometricModels.StatsAPI.islinear(::ARCHModel)
MacroEconometricModels.StatsAPI.nobs(::GARCHModel)
MacroEconometricModels.StatsAPI.coef(::GARCHModel)
MacroEconometricModels.StatsAPI.residuals(::GARCHModel)
MacroEconometricModels.StatsAPI.aic(::GARCHModel)
MacroEconometricModels.StatsAPI.bic(::GARCHModel)
MacroEconometricModels.StatsAPI.dof(::GARCHModel)
MacroEconometricModels.StatsAPI.islinear(::GARCHModel)
MacroEconometricModels.StatsAPI.nobs(::EGARCHModel)
MacroEconometricModels.StatsAPI.coef(::EGARCHModel)
MacroEconometricModels.StatsAPI.residuals(::EGARCHModel)
MacroEconometricModels.StatsAPI.aic(::EGARCHModel)
MacroEconometricModels.StatsAPI.bic(::EGARCHModel)
MacroEconometricModels.StatsAPI.dof(::EGARCHModel)
MacroEconometricModels.StatsAPI.islinear(::EGARCHModel)
MacroEconometricModels.StatsAPI.nobs(::GJRGARCHModel)
MacroEconometricModels.StatsAPI.coef(::GJRGARCHModel)
MacroEconometricModels.StatsAPI.residuals(::GJRGARCHModel)
MacroEconometricModels.StatsAPI.aic(::GJRGARCHModel)
MacroEconometricModels.StatsAPI.bic(::GJRGARCHModel)
MacroEconometricModels.StatsAPI.dof(::GJRGARCHModel)
MacroEconometricModels.StatsAPI.islinear(::GJRGARCHModel)
MacroEconometricModels.StatsAPI.nobs(::SVModel)
MacroEconometricModels.StatsAPI.coef(::SVModel)
MacroEconometricModels.StatsAPI.residuals(::SVModel)
MacroEconometricModels.StatsAPI.islinear(::SVModel)
```

---

## Nowcasting

### Estimation

```@docs
nowcast_dfm
nowcast_bvar
nowcast_bridge
```

### Nowcast and Forecast

```@docs
nowcast
```

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["nowcast/forecast.jl"]
Order   = [:function]
```

### News Decomposition

```@docs
nowcast_news
```

### Panel Balancing

```@docs
balance_panel
```

---

## Display and References

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["core/display.jl"]
Order   = [:function]
```

```@docs
refs
```

---

## Non-Gaussian Identification

### Normality Tests

```@docs
jarque_bera_test
mardia_test
doornik_hansen_test
henze_zirkler_test
normality_test_suite
```

### ICA-based Identification

```@docs
identify_fastica
identify_jade
identify_sobi
identify_dcov
identify_hsic
```

### Non-Gaussian ML Identification

```@docs
identify_student_t
identify_mixture_normal
identify_pml
identify_skew_normal
identify_nongaussian_ml
```

### Heteroskedasticity Identification

```@docs
identify_markov_switching
identify_garch
identify_smooth_transition
identify_external_volatility
```

### Identifiability Tests

```@docs
test_identification_strength
test_shock_gaussianity
test_gaussian_vs_nongaussian
test_shock_independence
test_overidentification
```

---

## Covariance Estimators

```@docs
newey_west
white_vcov
driscoll_kraay
robust_vcov
long_run_variance
long_run_covariance
optimal_bandwidth_nw
```

---

## Utility Functions

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["core/utils.jl"]
Order   = [:function]
```
