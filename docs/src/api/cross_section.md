# [Cross-Sectional Models API](@id api_cross_section)

OLS, WLS, IV/2SLS, logit, probit, ordered, and multinomial estimation for cross-sectional data. See [Regression](../regression.md) and [Binary Choice](../binary_choice.md) for theory and examples.

---

## Cross-Sectional Types

```@docs
RegModel
LogitModel
ProbitModel
MarginalEffects
MultinomialMarginalEffects
OddsRatio
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
estimate_iv
estimate_logit
estimate_probit
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
```
