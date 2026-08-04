# [Cross-Sectional Models API](@id api_cross_section)

Cross-sectional estimation — OLS/WLS, IV/2SLS, quantile regression, regression discontinuity, binary, ordered and multinomial choice, penalized and robust regression, limited-dependent and count models — together with the regression diagnostics and parameter-stability tests that accompany them. See [Linear Regression](@ref regression_page), [Binary Choice Models](@ref binary_choice_page), and [Ordered & Multinomial Models](@ref ordered_multinomial_page) for theory and examples.

---

## Cross-Sectional Types

```@docs
RegModel
QuantileRegModel
RDDResult
LogitModel
ProbitModel
MarginalEffects
MultinomialMarginalEffects
OddsRatio
WildClusterBootstrap
AndersonRubinTest
AndersonRubinCI
report(::IO, ::QuantileRegModel{T}) where {T}
report(::IO, ::RDDResult{T}) where {T}
```

---

## Ordered and Multinomial Types

```@docs
OrderedLogitModel
OrderedProbitModel
MultinomialLogitModel
```

---

## Cross-Sectional Estimation

```@docs
estimate_reg
conley_se
estimate_qreg
estimate_rdd
estimate_iv
estimate_logit
estimate_probit
```

### Few-Cluster Inference

```@docs
wild_cluster_bootstrap
```

### Weak-Instrument-Robust Inference

```@docs
anderson_rubin_test
anderson_rubin_ci
```

### Marginal Effects and Odds Ratios

```@docs
marginal_effects
odds_ratio
```

### Regression Diagnostics

Collinearity, classification accuracy, and the heteroskedasticity, serial-correlation, and functional-form tests for [`RegModel`](@ref). The panel counterpart of `breusch_pagan_test` — the random-effects LM test — is documented in [Panel Models API](@ref api_panel).

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["reg/diagnostics.jl"]
```

### Stability and Influence Diagnostics

```@docs
StabilityResult
InfluenceStats
```

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["reg/stability.jl"]
```

---

## Ordered and Multinomial Models

```@docs
estimate_ologit
estimate_oprobit
estimate_mlogit
brant_test
hausman_iia
generalized_residuals
```

---

## Regularized, Robust & Limited-Dependent Regression

Lasso, ridge and elastic net; M- and MM-estimators; Tobit, truncated and Heckman selection models; Poisson and negative binomial counts; and stepwise variable selection. Their `marginal_effects` methods are documented under Marginal Effects and Odds Ratios above.

```@docs
PenalizedRegModel
RobustRegModel
HeckmanModel
SelectionResult
```

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["reg/penalized.jl", "reg/robust.jl", "reg/tobit.jl", "reg/heckman.jl", "reg/count.jl", "reg/selection.jl"]
Filter  = f -> f !== MacroEconometricModels.marginal_effects
```

### Prediction

`predict` methods for the cross-sectional model families.

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["src/reg/predict.jl", "reg/quantile.jl", "reg/ordered.jl", "reg/multinomial.jl"]
Order   = [:function]
Filter  = f -> f === MacroEconometricModels.StatsAPI.predict
```
