# [Cross-Sectional Models API](@id api_cross_section)

OLS, WLS, IV/2SLS, logit, probit, ordered, and multinomial estimation for cross-sectional data. See [Regression](../regression.md) and [Binary Choice](../binary_choice.md) for theory and examples.

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

### Diagnostics

```@docs
vif
classification_table
```

### Stability and Influence Diagnostics

```@docs
StabilityResult
InfluenceStats
recursive_residuals
cusum_test
cusumsq_test
chow_test
influence_stats
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

```@docs
PenalizedRegModel
RobustRegModel
HeckmanModel
SelectionResult
```

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["reg/penalized.jl", "reg/robust.jl", "reg/tobit.jl", "reg/heckman.jl", "reg/count.jl", "reg/selection.jl", "reg/stability.jl", "reg/diagnostics.jl"]
```
