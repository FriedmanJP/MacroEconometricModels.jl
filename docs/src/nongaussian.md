# [Statistical Identification](@id nongaussian_page)

Statistical identification recovers the structural impact matrix ``B_0`` from higher-moment information --- time-varying variances (heteroskedasticity) or non-Gaussian shock distributions --- without imposing recursive orderings, sign restrictions, or zero restrictions. The classification follows Lewis (2025), the definitive survey of higher-moment identification in macroeconometrics.

- **13 identification methods**: 5 ICA + 4 ML (non-Gaussianity) and 4 heteroskedasticity estimators
- **5 diagnostic tests**: normality suite, shock Gaussianity, LR test, independence, identification strength
- **Full pipeline integration**: all methods produce a rotation ``Q`` consumed by `irf()`, `fevd()`, `historical_decomposition()`
- **Sub-pages**: [Non-Gaussian Methods](@ref id_nongaussian_page) | [Heteroskedasticity](@ref id_heteroskedastic_page) | [Testing](@ref id_testing_page)

## Quick Start

**Recipe 1: FastICA identification**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)
ica = identify_fastica(model)
report(ica)
```

**Recipe 2: Student-t ML identification**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)
ml = identify_student_t(model)
report(ml)
```

**Recipe 3: Markov-switching heteroskedasticity**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)
ms = identify_markov_switching(model; n_regimes=2)
report(ms)
```

**Recipe 4: Normality test suite**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)
suite = normality_test_suite(model)
report(suite)
```

**Recipe 5: Shock Gaussianity test**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)
ica = identify_fastica(model)
result = test_shock_gaussianity(ica)
report(result)
```

**Recipe 6: IRF via statistical identification**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)
irfs = irf(model, 20; method=:fastica)
report(irfs)
plot_result(irfs)
```

---

## The SVAR Setting

The structural VAR decomposes reduced-form residuals into orthogonal structural shocks:

```math
u_t = B_0 \varepsilon_t, \quad \Sigma = B_0 B_0'
```

where
- ``u_t`` is the ``n \times 1`` vector of reduced-form residuals
- ``\varepsilon_t`` is the ``n \times 1`` vector of structural shocks (unit variance, mutually independent)
- ``B_0`` is the ``n \times n`` structural impact matrix

The covariance ``\Sigma = B_0 B_0'`` provides ``n(n+1)/2`` equations for ``n^2`` unknowns, leaving ``n(n-1)/2`` free parameters. Statistical identification resolves this gap without economic restrictions:

- **Heteroskedasticity**: regime-dependent covariances ``\Sigma_k = B_0 \Lambda_k B_0'`` supply additional equations. See [Heteroskedasticity](@ref id_heteroskedastic_page).
- **Non-Gaussianity**: independence beyond uncorrelatedness (coskewness, cokurtosis) identifies ``B_0`` from a single sample. See [Non-Gaussian Methods](@ref id_nongaussian_page).

---

## Method Comparison

All 13 methods return a rotation matrix ``Q`` and structural impact matrix ``B_0 = L Q`` where ``L = \text{chol}(\Sigma)``.

| Approach | Method | Function | Key Feature |
|----------|--------|----------|-------------|
| Non-Gaussian (ICA) | FastICA | `identify_fastica` | Negentropy maximization |
| Non-Gaussian (ICA) | JADE | `identify_jade` | Cumulant diagonalization |
| Non-Gaussian (ICA) | SOBI | `identify_sobi` | Autocovariance-based |
| Non-Gaussian (ICA) | Distance Cov. | `identify_dcov` | Nonparametric independence |
| Non-Gaussian (ICA) | HSIC | `identify_hsic` | Kernel independence |
| Non-Gaussian (ML) | Student-t | `identify_student_t` | Heavy tails |
| Non-Gaussian (ML) | Mixture Normal | `identify_mixture_normal` | Bimodality |
| Non-Gaussian (ML) | PML | `identify_pml` | Skewness + kurtosis |
| Non-Gaussian (ML) | Skew-Normal | `identify_skew_normal` | Asymmetry |
| Heteroskedasticity | Markov-switching | `identify_markov_switching` | Regime changes |
| Heteroskedasticity | GARCH | `identify_garch` | Conditional variance |
| Heteroskedasticity | Smooth Transition | `identify_smooth_transition` | Gradual shifts |
| Heteroskedasticity | External | `identify_external_volatility` | Known regimes |

---

## IRF Pipeline Integration

All 13 methods integrate with `irf()`, `fevd()`, and `historical_decomposition()` via `compute_Q()`. Pass the method name as a symbol:

```julia
irfs_ica = irf(model, 20; method=:fastica)
irfs_ml  = irf(model, 20; method=:student_t)
irfs_ms  = irf(model, 20; method=:markov_switching)
decomp   = fevd(model, 20; method=:jade)
```

Supported symbols: `:fastica`, `:jade`, `:sobi`, `:dcov`, `:hsic`, `:student_t`, `:mixture_normal`, `:pml`, `:skew_normal`, `:markov_switching`, `:garch`.

---

## The Labeling Problem

Statistical identification recovers ``B_0`` only up to **column permutation and sign**. The data alone cannot determine which column corresponds to which economic shock --- economic information is still required to label shocks. The package normalizes ``B_0`` to have a positive diagonal (sign convention). The `test_identification_strength` bootstrap measures column-assignment stability across replications via Procrustes distance. See Lewis (2025, Section 6.4) for a thorough discussion.

---

## Sub-Page Guide

- [Non-Gaussian Methods](@ref id_nongaussian_page) --- ICA (FastICA, JADE, SOBI, distance covariance, HSIC), ML (Student-t, mixture normal, PML, skew-normal), Darmois-Skitovich theorem, contrast functions, unified dispatcher
- [Heteroskedasticity](@ref id_heteroskedastic_page) --- Eigendecomposition identification, Markov-switching, GARCH, smooth transition, external volatility instruments, result field tables
- [Testing](@ref id_testing_page) --- Normality suite (7 tests), shock Gaussianity, LR test, independence, identification strength, overidentification, weak identification diagnostics

---

## Common Pitfalls

1. **Non-Gaussianity is a prerequisite, not an assumption.** ICA and ML methods require at most one Gaussian shock. Run `normality_test_suite(model)` first. If residuals are Gaussian, the problem is unidentified.

2. **Column permutation differs across bootstrap replications.** The Procrustes distance in `test_identification_strength` accounts for this, but naive bootstrap CIs on individual ``B_0`` entries do not. Verify identification stability before interpreting shock-by-shock results.

3. **Heteroskedasticity requires distinct eigenvalues.** If two shocks have identical variance ratios across regimes, the eigendecomposition of ``\Sigma_1^{-1}\Sigma_2`` does not uniquely identify the corresponding columns. Check that `Lambda` values are well-separated.

4. **Weak identification is common in practice.** When variance changes are small or deviations from Gaussianity are mild, Wald tests have poor size properties (Lewis 2022). Run `test_identification_strength` as a diagnostic.

5. **Smooth transition needs an external variable.** Unlike Markov-switching and GARCH, `identify_smooth_transition` requires a transition variable `s` of the same length as the residuals (e.g., a lagged endogenous variable).

---

## References

### Survey

- Lewis, D. J. (2025). Identification Based on Higher Moments in Macroeconometrics. *Annual Review of Economics*, 17, 665--693. [DOI: 10.1146/annurev-economics-070124-051419](https://doi.org/10.1146/annurev-economics-070124-051419)

### Heteroskedasticity

- Rigobon, R. (2003). Identification through Heteroskedasticity. *Review of Economics and Statistics*, 85(4), 777--792. [DOI: 10.1162/003465303772815727](https://doi.org/10.1162/003465303772815727)
- Sentana, E. & Fiorentini, G. (2001). Identification, Estimation and Testing of Conditionally Heteroskedastic Factor Models. *Journal of Econometrics*, 102(2), 143--164. [DOI: 10.1016/S0304-4076(01)00051-3](https://doi.org/10.1016/S0304-4076(01)00051-3)
- Lanne, M. & Lutkepohl, H. (2008). Identifying Monetary Policy Shocks via Changes in Volatility. *Journal of Money, Credit and Banking*, 40(6), 1131--1149. [DOI: 10.1111/j.1538-4616.2008.00151.x](https://doi.org/10.1111/j.1538-4616.2008.00151.x)
- Normandin, M. & Phaneuf, L. (2004). Monetary Policy Shocks: Testing Identification Conditions under Time-Varying Conditional Volatility. *Journal of Monetary Economics*, 51(6), 1217--1243. [DOI: 10.1016/j.jmoneco.2003.11.002](https://doi.org/10.1016/j.jmoneco.2003.11.002)
- Lutkepohl, H. & Netsunajev, A. (2017). Structural VARs with Smooth Transition in Variances. *Journal of Economic Dynamics and Control*, 84, 43--57. [DOI: 10.1016/j.jedc.2017.09.001](https://doi.org/10.1016/j.jedc.2017.09.001)
- Lewis, D. J. (2021). Identifying Shocks via Time-Varying Volatility. *Review of Economic Studies*, 88(6), 3086--3124. [DOI: 10.1093/restud/rdab009](https://doi.org/10.1093/restud/rdab009)

### Non-Gaussianity --- ICA

- Hyvarinen, A. (1999). Fast and Robust Fixed-Point Algorithms for Independent Component Analysis. *IEEE Trans. Neural Networks*, 10(3), 626--634. [DOI: 10.1109/72.761722](https://doi.org/10.1109/72.761722)
- Cardoso, J.-F. & Souloumiac, A. (1993). Blind Beamforming for Non-Gaussian Signals. *IEE Proceedings-F*, 140(6), 362--370. [DOI: 10.1049/ip-f-2.1993.0054](https://doi.org/10.1049/ip-f-2.1993.0054)
- Belouchrani, A. et al. (1997). A Blind Source Separation Technique Using Second-Order Statistics. *IEEE Trans. Signal Processing*, 45(2), 434--444. [DOI: 10.1109/78.554307](https://doi.org/10.1109/78.554307)
- Comon, P. (1994). Independent Component Analysis, A New Concept? *Signal Processing*, 36(3), 287--314. [DOI: 10.1016/0165-1684(94)90029-9](https://doi.org/10.1016/0165-1684(94)90029-9)
- Matteson, D. S. & Tsay, R. S. (2017). Independent Component Analysis via Distance Covariance. *JASA*, 112(518), 623--637. [DOI: 10.1080/01621459.2016.1150851](https://doi.org/10.1080/01621459.2016.1150851)
- Gretton, A. et al. (2005). Measuring Statistical Dependence with Hilbert-Schmidt Norms. In *Algorithmic Learning Theory*, 63--77. Springer. [DOI: 10.1007/11564089_7](https://doi.org/10.1007/11564089_7)
- Szekely, G. J. et al. (2007). Measuring and Testing Dependence by Correlation of Distances. *Annals of Statistics*, 35(6), 2769--2794. [DOI: 10.1214/009053607000000505](https://doi.org/10.1214/009053607000000505)

### Non-Gaussianity --- ML

- Lanne, M., Meitz, M. & Saikkonen, P. (2017). Identification and Estimation of Non-Gaussian SVARs. *Journal of Econometrics*, 196(2), 288--304. [DOI: 10.1016/j.jeconom.2016.06.002](https://doi.org/10.1016/j.jeconom.2016.06.002)
- Gourieroux, C., Monfort, A. & Renne, J.-P. (2017). Statistical Inference for ICA: Application to Structural VAR Models. *Journal of Econometrics*, 196(1), 111--126. [DOI: 10.1016/j.jeconom.2016.09.007](https://doi.org/10.1016/j.jeconom.2016.09.007)
- Lanne, M. & Lutkepohl, H. (2010). SVARs with Nonnormal Residuals. *Journal of Business & Economic Statistics*, 28(1), 159--168. [DOI: 10.1198/jbes.2009.06003](https://doi.org/10.1198/jbes.2009.06003)
- Herwartz, H. (2018). Hodges-Lehmann Detection of Structural Shocks. *Oxford Bulletin of Economics and Statistics*, 80(4), 736--754. [DOI: 10.1111/obes.12234](https://doi.org/10.1111/obes.12234)
- Azzalini, A. (1985). A Class of Distributions Which Includes the Normal Ones. *Scandinavian Journal of Statistics*, 12(2), 171--178. [https://www.jstor.org/stable/4615982](https://www.jstor.org/stable/4615982)
- Keweloh, S. A. (2021). A GMM Estimator for SVARs Based on Higher Moments. *Journal of Business & Economic Statistics*, 39(3), 772--882. [DOI: 10.1080/07350015.2020.1730858](https://doi.org/10.1080/07350015.2020.1730858)
- Lanne, M. & Luoto, J. (2021). GMM Estimation of Non-Gaussian SVAR. *Journal of Business & Economic Statistics*, 39(1), 69--81. [DOI: 10.1080/07350015.2019.1629940](https://doi.org/10.1080/07350015.2019.1629940)

### Diagnostics

- Lewis, D. J. (2022). Robust Inference in Models Identified via Heteroskedasticity. *Review of Economics and Statistics*, 104(3), 510--524. [DOI: 10.1162/rest_a_00977](https://doi.org/10.1162/rest_a_00977)
- Jarque, C. M. & Bera, A. K. (1980). Efficient Tests for Normality, Homoscedasticity and Serial Independence. *Economics Letters*, 6(3), 255--259. [DOI: 10.1016/0165-1765(80)90024-5](https://doi.org/10.1016/0165-1765(80)90024-5)
- Mardia, K. V. (1970). Measures of Multivariate Skewness and Kurtosis with Applications. *Biometrika*, 57(3), 519--530. [DOI: 10.1093/biomet/57.3.519](https://doi.org/10.1093/biomet/57.3.519)
- Doornik, J. A. & Hansen, H. (2008). An Omnibus Test for Univariate and Multivariate Normality. *Oxford Bulletin of Economics and Statistics*, 70, 927--939. [DOI: 10.1111/j.1468-0084.2008.00537.x](https://doi.org/10.1111/j.1468-0084.2008.00537.x)
- Henze, N. & Zirkler, B. (1990). A Class of Invariant Consistent Tests for Multivariate Normality. *Communications in Statistics*, 19(10), 3595--3617. [DOI: 10.1080/03610929008830400](https://doi.org/10.1080/03610929008830400)
- Lutkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer. ISBN 978-3-540-40172-8.
