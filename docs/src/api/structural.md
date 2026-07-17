# [Structural & Statistical Identification API](@id api_structural)

Structural identification schemes (Cholesky, sign, long-run, narrative, Arias, Uhlig) and statistical identification via non-Gaussianity and heteroskedasticity. See [Structural Identification](../structural_identification.md) and [Statistical Identification](../nongaussian.md).

---

## SVAR Identification Types

```@docs
ZeroRestriction
SignRestriction
SVARRestrictions
SignIdentifiedSet
AriasSVARResult
UhligSVARResult
```

---

## Non-Gaussian SVAR Types

```@docs
AbstractNormalityTest
AbstractNonGaussianSVAR
NormalityTestResult
NormalityTestSuite
ICASVARResult
NonGaussianMLResult
MarkovSwitchingSVARResult
GARCHSVARResult
SmoothTransitionSVARResult
ExternalVolatilitySVARResult
IdentifiabilityTestResult
```

---

## Structural Identification

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["core/identification.jl"]
Order   = [:function]
```

### Arias et al. (2018) Sign/Zero Restrictions

```@docs
identify_arias
identify_arias_bayesian
zero_restriction
sign_restriction
```

### Sign-Identified Set

```@docs
irf_bounds
irf_median
irf_mean
irf_percentiles
```

### Mountford-Uhlig (2009) Penalty Function

```@docs
identify_uhlig
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
