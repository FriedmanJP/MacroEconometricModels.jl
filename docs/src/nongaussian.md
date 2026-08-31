# [Statistical Identification](@id nongaussian_page)

Statistical identification recovers the structural impact matrix ``B_0`` from higher-moment information --- time-varying variances (heteroskedasticity) or non-Gaussian shock distributions --- without imposing recursive orderings, sign restrictions, or zero restrictions. The classification follows Lewis (2025), the definitive survey of higher-moment identification in macroeconometrics.

Fifteen estimators and six diagnostic tests divide across three child pages: eleven methods exploit non-Gaussianity (five ICA, four maximum likelihood, the adaptive `:nongaussian_ml` dispatcher, and moment-based GMM), four exploit heteroskedasticity, and the testing page covers the diagnostics that decide whether either source of identification is present. Every method produces a rotation ``Q`` consumed by `irf()`, `fevd()`, and `historical_decomposition()`. `label_shocks` assigns economic names to those statistically recovered columns.

```@setup id_overview
using MacroEconometricModels, Random
Random.seed!(42)
```

## Quick Start

Estimate a VAR and identify the structural shocks by FastICA, which requires no ordering, sign, or exclusion restriction:

```@example id_overview
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

ica = identify_fastica(model)
report(ica)
```

---

## Choosing a Method

All 15 methods return a rotation matrix ``Q`` and structural impact matrix ``B_0 = P Q`` where ``P = \text{chol}(\Sigma)``. The shape of the data --- fat tails, bimodality, skewness, regime shifts, volatility clustering --- selects the estimator:

| Feature needed | Recommended | Why |
|----------------|-------------|-----|
| Nonparametric, no distribution assumed | `identify_fastica` | Negentropy maximization |
| Fourth-moment structure | `identify_jade` | Cumulant diagonalization |
| Serially correlated shocks | `identify_sobi` | Autocovariance-based separation |
| Independence beyond fourth moments | `identify_dcov`, `identify_hsic` | Nonparametric and kernel criteria |
| Heavy tails | `identify_student_t` | Parametric ML on fat tails |
| Bimodal shocks | `identify_mixture_normal` | Two-component Gaussian mixture |
| Skewness and kurtosis jointly | `identify_pml` | Pearson Type IV ML on both moments |
| Asymmetric shocks | `identify_skew_normal` | Azzalini skew-normal likelihood |
| Independence via higher moments | `identify_gmm_moments` | Coskewness / cokurtosis GMM |
| Discrete volatility regimes | `identify_markov_switching` | EM over latent regimes |
| Volatility clustering | `identify_garch` | Conditional variance dynamics |
| Gradual variance shifts | `identify_smooth_transition` | Logistic transition variable |
| Externally known regimes | `identify_external_volatility` | Regime dates supplied by the user |
| Whether identification holds at all | [Testing](@ref id_testing_page) | Diagnostics before interpretation |

---

## Child Pages

- [Non-Gaussian Methods](@ref id_nongaussian_page) --- ICA (FastICA, JADE, SOBI, distance covariance, HSIC), ML (Student-t, mixture normal, PML, skew-normal), GMM (coskewness / cokurtosis), `label_shocks`, Darmois-Skitovich theorem, contrast functions, unified dispatcher
- [Heteroskedasticity](@ref id_heteroskedastic_page) --- generalized eigenproblem, K-regime joint ML, Markov-switching, GARCH, smooth transition, external volatility instruments, delta-method SEs
- [Testing](@ref id_testing_page) --- normality suite (7 tests), shock Gaussianity, LR test, independence, distinct-``\lambda`` Wald, label-stability, overidentification

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

## IRF Pipeline Integration

All 15 methods plug into `irf()`, `fevd()`, and `historical_decomposition()` through the same internal rotation interface. Pass the method name as a symbol:

```@example id_overview
irfs = irf(model, 20; method=:fastica)
report(irfs)
```

```julia
plot_result(irfs)
```

```@raw html
<iframe src="../assets/plots/nongaussian_irf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The same symbol works across the whole pipeline:

```@example id_overview
irfs_ml = irf(model, 20; method=:student_t)
irfs_ms = irf(model, 20; method=:markov_switching)
decomp  = fevd(model, 20; method=:jade)
nothing # hide
```

Supported symbols: `:fastica`, `:jade`, `:sobi`, `:dcov`, `:hsic`, `:student_t`, `:mixture_normal`, `:pml`, `:skew_normal`, `:nongaussian_ml`, `:gmm_moments`, `:markov_switching`, and `:garch`. The heteroskedasticity schemes `:smooth_transition` and `:external_volatility` are also accepted but require an additional keyword argument (`transition_var` and `regime_indicator`, respectively).

---

## The Labeling Problem

Statistical identification recovers ``B_0`` only up to **column permutation and sign**. The data alone cannot determine which column corresponds to which economic shock --- economic information is still required to label shocks. The package normalizes ``B_0`` to have a positive diagonal (sign convention). `label_shocks` applies a signed permutation: `by=:max_impact` assigns each column to the variable it moves most, `by=:restrictions` maximises satisfied impact-sign restrictions, and `by=:reference` matches a reference impact via `_match_columns`. Column-assignment *stability* is the match-fraction from `test_label_stability` (no p-value); `test_identification_strength` is a deprecated wrapper that is not a Procrustes strength test. See [Non-Gaussian Methods](@ref id_nongaussian_page), [Identification Testing](@ref id_testing_page), and Lewis (2025, Section 6.4).

```@example id_overview
ica_lab = identify_fastica(model; rng=MersenneTwister(11))
labeled = label_shocks(ica_lab; by=:max_impact, variables=1:3,
                       shock_names=["output", "price", "mp"])
labeled.shock_names
```

`label_shocks` permutes the columns so the named shocks line up with the variables they move most. The rotation is the same ``B_0`` up to that signed permutation; only the labels change.

---

## Common Pitfalls

1. **Weak identification is common in practice.** When variance changes are small or deviations from Gaussianity are mild, Wald tests have poor size properties (Lewis 2022). Run `test_lambda_distinct` (heteroskedastic) or `test_gaussian_shock_count` / `test_label_stability` (non-Gaussian) before interpreting any structural result; see [Identification Testing](@ref id_testing_page).

2. **Smooth transition needs an external variable.** Unlike Markov-switching and GARCH, `identify_smooth_transition` requires a transition variable `s` of the same length as the residuals (e.g., a lagged endogenous variable).

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
- Keweloh, S. A. (2021). A GMM Estimator for SVARs Based on Higher Moments. *Journal of Business & Economic Statistics*, 39(3), 772--782. [DOI: 10.1080/07350015.2020.1730858](https://doi.org/10.1080/07350015.2020.1730858)
- Lanne, M. & Luoto, J. (2021). GMM Estimation of Non-Gaussian SVAR. *Journal of Business & Economic Statistics*, 39(1), 69--81. [DOI: 10.1080/07350015.2019.1629940](https://doi.org/10.1080/07350015.2019.1629940)

### Diagnostics

- Lewis, D. J. (2022). Robust Inference in Models Identified via Heteroskedasticity. *Review of Economics and Statistics*, 104(3), 510--524. [DOI: 10.1162/rest_a_00963](https://doi.org/10.1162/rest_a_00963)
- Jarque, C. M. & Bera, A. K. (1980). Efficient Tests for Normality, Homoscedasticity and Serial Independence. *Economics Letters*, 6(3), 255--259. [DOI: 10.1016/0165-1765(80)90024-5](https://doi.org/10.1016/0165-1765(80)90024-5)
- Mardia, K. V. (1970). Measures of Multivariate Skewness and Kurtosis with Applications. *Biometrika*, 57(3), 519--530. [DOI: 10.1093/biomet/57.3.519](https://doi.org/10.1093/biomet/57.3.519)
- Doornik, J. A. & Hansen, H. (2008). An Omnibus Test for Univariate and Multivariate Normality. *Oxford Bulletin of Economics and Statistics*, 70, 927--939. [DOI: 10.1111/j.1468-0084.2008.00537.x](https://doi.org/10.1111/j.1468-0084.2008.00537.x)
- Henze, N. & Zirkler, B. (1990). A Class of Invariant Consistent Tests for Multivariate Normality. *Communications in Statistics*, 19(10), 3595--3617. [DOI: 10.1080/03610929008830400](https://doi.org/10.1080/03610929008830400)
- Lutkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer. ISBN 978-3-540-40172-8.
