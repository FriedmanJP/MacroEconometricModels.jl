# [Univariate Models API](@id api_univariate)

Single-series models: time series filters, ARIMA and long-memory models, state-space models, the ARCH/GARCH/SV volatility family, spectral analysis, nonlinear time series, nonparametric regression, and MIDAS. See [Time Series Filters](@ref filters_page), [ARIMA Models](@ref arima_page), [State-Space Models](@ref statespace_page), [Volatility Models](@ref volatility_page), [Spectral Analysis](@ref spectral_page), [Nonlinear Time Series](@ref nonlinear_page), [Nonparametric Methods](@ref nonparametric_page), and [MIDAS Regression](@ref midas_page) for theory and examples.

---

## Time Series Filter Types

```@docs
AbstractFilterResult
HPFilterResult
HamiltonFilterResult
BeveridgeNelsonResult
BaxterKingResult
BoostedHPResult
X13FilterResult
```

---

## Time Series Filter Functions

```@docs
hp_filter
hamilton_filter
beveridge_nelson
baxter_king
boosted_hp
trend
cycle
```

### X-13ARIMA-SEATS

```@docs
x13_filter
seasonal
adjusted
```

---

## ARIMA Types

```@docs
AbstractARIMAModel
ARModel
MAModel
ARMAModel
ARIMAModel
SARIMAModel
ARIMAForecast
ARIMAOrderSelection
```

---

## ARIMA Estimation

```@docs
estimate_ar
estimate_ma
estimate_arma
estimate_arima
estimate_sarima
```

### Forecasting

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["arima/forecast.jl"]
Order   = [:function]
```

```@docs
forecast(::SARIMAModel{T}, ::Int) where {T<:AbstractFloat}
```

### Order Selection

```@docs
select_arima_order
auto_arima
auto_sarima
ic_table
```

### ARIMA Accessors

```@docs
ar_order
ma_order
diff_order
```

### ARIMA StatsAPI Interface

```@docs
MacroEconometricModels.StatsAPI.stderror(::ARModel)
MacroEconometricModels.StatsAPI.stderror(::MAModel)
MacroEconometricModels.StatsAPI.stderror(::ARMAModel)
MacroEconometricModels.StatsAPI.stderror(::ARIMAModel)
```

---

## State-Space Models

See [State-Space Models](@ref statespace_page) for theory and examples.

```@docs
StateSpaceModel
estimate_statespace
local_level
local_linear_trend
estimate_tvp_reg
```

---

## Volatility Types

```@docs
AbstractVolatilityModel
ARCHModel
GARCHModel
EGARCHModel
GJRGARCHModel
GarchMidasModel
IGARCHModel
CGARCHModel
APARCHModel
FIGARCHModel
FIEGARCHModel
SVModel
VolatilityForecast
```

---

## Volatility Functions

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
estimate_garch_midas
estimate_igarch
estimate_cgarch
estimate_aparch
component_variances
news_impact_curve
```

### Long-Memory GARCH

Fractionally integrated GARCH and EGARCH: estimation, filtering, forecasting, and the
fractional-differencing standard errors.

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["src/garch/figarch.jl"]
Order   = [:function]
```

### Extended GARCH Diagnostics

```@docs
sign_bias_test
nyblom_test
```

### Stochastic Volatility

```@docs
estimate_sv
```

### Volatility Forecasting

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["src/arch/forecast.jl", "src/garch/forecast.jl", "src/sv/forecast.jl"]
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
MacroEconometricModels.StatsAPI.coef(::IGARCHModel)
MacroEconometricModels.StatsAPI.dof(::IGARCHModel)
MacroEconometricModels.StatsAPI.coef(::CGARCHModel)
MacroEconometricModels.StatsAPI.dof(::CGARCHModel)
MacroEconometricModels.StatsAPI.coef(::APARCHModel)
MacroEconometricModels.StatsAPI.dof(::APARCHModel)
MacroEconometricModels.StatsAPI.coef(::FIGARCHModel)
MacroEconometricModels.StatsAPI.dof(::FIGARCHModel)
MacroEconometricModels.StatsAPI.coef(::FIEGARCHModel)
MacroEconometricModels.StatsAPI.dof(::FIEGARCHModel)
MacroEconometricModels.StatsAPI.nobs(::GarchMidasModel)
MacroEconometricModels.StatsAPI.coef(::GarchMidasModel)
MacroEconometricModels.StatsAPI.residuals(::GarchMidasModel)
MacroEconometricModels.StatsAPI.aic(::GarchMidasModel)
MacroEconometricModels.StatsAPI.bic(::GarchMidasModel)
MacroEconometricModels.StatsAPI.dof(::GarchMidasModel)
MacroEconometricModels.StatsAPI.dof_residual(::GarchMidasModel)
MacroEconometricModels.StatsAPI.islinear(::GarchMidasModel)
MacroEconometricModels.StatsAPI.confint(m::AbstractVolatilityModel)
MacroEconometricModels.StatsAPI.vcov(::AbstractVolatilityModel)
MacroEconometricModels.StatsAPI.dof_residual(::ARCHModel)
MacroEconometricModels.StatsAPI.dof_residual(::GARCHModel)
MacroEconometricModels.StatsAPI.dof_residual(::EGARCHModel)
MacroEconometricModels.StatsAPI.dof_residual(::GJRGARCHModel)
```

Standard errors come from the numerical Hessian of each estimator, so they are documented
per model alongside the estimation routine.

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["src/arch/estimation.jl", "src/garch/estimation.jl", "src/garch/midas.jl"]
Order   = [:function]
Filter  = f -> f === MacroEconometricModels.StatsAPI.stderror
```

`predict` returns the conditional variance series and `loglikelihood` the maximized
log-likelihood, for every univariate volatility and nonlinear model.

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["src/arch/types.jl", "src/garch/types.jl", "src/sv/types.jl", "src/nonlinear/types.jl"]
Order   = [:function]
Filter  = f -> f === MacroEconometricModels.StatsAPI.predict || f === MacroEconometricModels.StatsAPI.loglikelihood
```

---

## Spectral Analysis Types

```@docs
SpectralDensityResult
CrossSpectrumResult
TransferFunctionResult
ACFResult
```

---

## Spectral Analysis Functions

### Spectral Estimation

```@docs
periodogram
spectral_density
cross_spectrum
```

### Autocorrelation Functions

```@docs
acf
pacf
ccf
acf_pacf
```

### Spectral Diagnostics

```@docs
coherence
phase
gain
band_power
ideal_bandpass
transfer_function
```

---

## Nonlinear Time Series Types

```@docs
AbstractNonlinearTSModel
ThresholdModel
ThresholdForecast
HansenLinearityTest
STARModel
STARForecast
MSRegModel
MSForecast
fitted(::MSRegModel)
```

---

## Nonlinear Time Series Estimation

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["nonlinear/threshold.jl", "nonlinear/star.jl", "nonlinear/markov_switching.jl"]
```

---

## Nonparametric Regression & Density Types

```@docs
KernelDensity
KernelRegression
LowessFit
```

---

## Nonparametric Regression & Density Estimation

```@docs
kernel_density
kernel_reg
lowess
```

---

## MIDAS Regression

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["midas/types.jl", "midas/weights.jl", "midas/estimation.jl", "midas/forecast.jl"]
```

---

## ARFIMA and Long-Memory

```@docs
ARFIMAModel
GPHResult
LocalWhittleResult
```

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["arima/arfima.jl"]
```

---

## Nonlinear & State-Space Forecast Methods

```@docs
forecast(::StateSpaceModel{T}, ::Integer) where {T<:AbstractFloat}
forecast(::GarchMidasModel{T}, ::Int) where {T}
report(::MSRegModel)
report(::ThresholdModel)
report(::STARModel)
```
