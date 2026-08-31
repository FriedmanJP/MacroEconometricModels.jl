# [Structural & Statistical Identification API](@id api_structural)

Six restriction-based identification schemes (Cholesky, sign, long-run, narrative, Arias, Uhlig), AB-model ML, proxy / external-instrument identification, and 14 statistical identification methods that exploit non-Gaussianity or heteroskedasticity instead of restrictions. See [Structural Identification](@ref structural_identification_page), [AB-Model SVAR](@ref id_ab_page), [Proxy SVAR](@ref id_proxy_page), and [Statistical Identification](@ref nongaussian_page) for theory and examples.

In applied use these are reached through `irf(model, H; method=...)` rather than called directly; the entries below document the identification routines themselves.

---

## SVAR Identification Types

```@docs
AbstractSVARRestriction
ZeroRestriction
LongRunZeroRestriction
A0ZeroRestriction
AplusZeroRestriction
SignRestriction
A0SignRestriction
AplusSignRestriction
ElasticityBound
MagnitudeBound
FEVDShareRestriction
CumulativeRestriction
NarrativeShockRestriction
NarrativeContributionRestriction
SVARRestrictions
IdentificationStatus
SignIdentifiedSet
AriasSVARResult
UhligSVARResult
ProxySVARResult
SVARPattern
SVARModel
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
Private = false
```

### Arias et al. (2018) Sign/Zero Restrictions

```@docs
identify_arias
identify_arias_bayesian
check_identification
is_linear_zero
check
sign_check
zero_restriction
sign_restriction
a0_zero_restriction
a0_sign_restriction
aplus_zero_restriction
aplus_sign_restriction
elasticity_bound
magnitude_bound
fevd_share_restriction
cumulative_restriction
narrative_shock_restriction
narrative_contribution_restriction
```

### Posterior Summaries of an Identified Set

`irf_bounds` and `irf_median` summarise the `SignIdentifiedSet` returned by `identify_sign`
and are documented with the identification functions above. The two entries below summarise
the posterior draws of an `AriasSVARResult`.

```@docs
irf_mean
irf_percentiles
```

### Mountford-Uhlig (2009) Penalty Function

```@docs
identify_uhlig
```

### AB-Model ML (Amisano–Giannini)

```@docs
estimate_svar
recursive_pattern
blanchard_quah_pattern
a_model_pattern
b_model_pattern
ab_model_pattern
```

---

## Statistical Identification

When the reduced-form innovations are non-Gaussian or heteroskedastic, the rotation is
point-identified without restrictions. Run the normality tests first: they are the
pre-test that decides whether the non-Gaussian route is available at all.

### Normality Tests

```@docs
jarque_bera_test
mardia_test
doornik_hansen_test
henze_zirkler_test
normality_test_suite
```

### ICA-based Identification

Five independent-component methods, ordered from the classical fixed-point algorithm to
the kernel-dependence criteria.

```@docs
identify_fastica
identify_jade
identify_sobi
identify_dcov
identify_hsic
```

### Non-Gaussian ML Identification

Four parametric likelihoods plus `identify_nongaussian_ml`, the dispatcher that selects
among them.

```@docs
identify_student_t
identify_mixture_normal
identify_pml
identify_skew_normal
identify_nongaussian_ml
```

### Heteroskedasticity Identification

Four ways of modelling the variance shift that delivers identification: discrete regimes,
conditional volatility, a smooth transition, or an externally supplied regime indicator.

```@docs
identify_markov_switching
identify_garch
identify_smooth_transition
identify_external_volatility
```

### Identifiability Tests

Diagnostics that check whether the identifying assumption actually holds in the estimated
system. See [Identification Testing](@ref id_testing_page) for how to read them.

```@docs
test_identification_strength
test_shock_gaussianity
test_gaussian_vs_nongaussian
test_shock_independence
test_overidentification
```
