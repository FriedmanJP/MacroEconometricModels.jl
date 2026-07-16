# [Univariate Models API](@id api_univariate)

Time series filters, ARIMA models, volatility models, and spectral analysis. See [Time Series Filters](../filters.md), [ARIMA](../arima.md), [Volatility Models](../volatility.md), and [Spectral Analysis](../spectral.md) for theory and examples.

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

### ARIMA StatsAPI Interface

```@docs
MacroEconometricModels.StatsAPI.stderror(::ARModel)
MacroEconometricModels.StatsAPI.stderror(::MAModel)
MacroEconometricModels.StatsAPI.stderror(::ARMAModel)
MacroEconometricModels.StatsAPI.stderror(::ARIMAModel)
```

---

## Volatility Types

```@docs
AbstractVolatilityModel
ARCHModel
GARCHModel
EGARCHModel
GJRGARCHModel
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
MacroEconometricModels.StatsAPI.stderror(::GJRGARCHModel{T}) where {T}
MacroEconometricModels.StatsAPI.confint(m::AbstractVolatilityModel)
MacroEconometricModels.StatsAPI.vcov(::AbstractVolatilityModel)
MacroEconometricModels.StatsAPI.dof_residual(::ARCHModel)
MacroEconometricModels.StatsAPI.dof_residual(::GARCHModel)
MacroEconometricModels.StatsAPI.dof_residual(::EGARCHModel)
MacroEconometricModels.StatsAPI.dof_residual(::GJRGARCHModel)
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
```

---

## Nonlinear Time Series Estimation

```@docs
estimate_threshold
estimate_setar
hansen_linearity_test
estimate_star
star_linearity_test
```
