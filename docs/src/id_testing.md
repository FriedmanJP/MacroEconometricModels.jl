# [Identification Testing](@id id_testing_page)

Before relying on statistical identification, the practitioner must verify that the identifying conditions hold in the data. This page covers the diagnostic tests for non-Gaussian and heteroskedasticity-based SVAR identification: multivariate normality tests, shock gaussianity and independence tests, likelihood ratio tests, and bootstrap identification strength assessments.

```@setup id_test
using MacroEconometricModels, Random
Random.seed!(42)
```

## Quick Start

**Recipe 1: Normality test suite**

```@example id_test
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

suite = normality_test_suite(model)
report(suite)
```

**Recipe 2: Multivariate Jarque-Bera**

```@example id_test
jb = jarque_bera_test(model)
report(jb)
```

**Recipe 3: Shock gaussianity test**

```@example id_test
ica = identify_fastica(model)
result = test_shock_gaussianity(ica)
report(result)
```

**Recipe 4: Shock independence test**

```@example id_test
ica = identify_fastica(model)
indep = test_shock_independence(ica; max_lag=10)
report(indep)
```

**Recipe 5: Gaussian vs non-Gaussian LR test**

```@example id_test
lr = test_gaussian_vs_nongaussian(model; distribution=:student_t)
report(lr)
```

---

## Multivariate Normality Tests

Testing for multivariate normality of VAR residuals is the essential first step. If residuals are Gaussian, non-Gaussian identification is not possible --- the Darmois-Skitovich theorem requires at most one Gaussian component for uniqueness. Rejecting normality supports the use of ICA or ML methods.

### Multivariate Jarque-Bera

The multivariate Jarque-Bera test (Lutkepohl 2005, Section 4.5) combines skewness and kurtosis measures:

```math
JB = T \cdot \frac{b_{1,k}}{6} + T \cdot \frac{(b_{2,k} - k(k+2))^2}{24k}
```

where:
- ``b_{1,k} = T^{-2} \sum_{i,j} (u_i' \Sigma^{-1} u_j)^3`` is multivariate skewness
- ``b_{2,k} = T^{-1} \sum_i (u_i' \Sigma^{-1} u_i)^2`` is multivariate kurtosis
- Under ``H_0``: ``JB \sim \chi^2(2k)``

The `:component` method applies univariate JB tests to each standardized residual. The component-wise p-values in the result's `component_pvalues` field pinpoint which variables drive non-Gaussianity.

```@example id_test
# Joint and component-wise tests
jb = jarque_bera_test(model)
report(jb)
```

```@example id_test
jb_comp = jarque_bera_test(model; method=:component)
report(jb_comp)
```

### Mardia's Tests

Mardia (1970) proposed separate tests for multivariate skewness and kurtosis:

```math
b_{1,k} = \frac{1}{T^2} \sum_{i,j} (u_i' \Sigma^{-1} u_j)^3, \quad b_{2,k} = \frac{1}{T} \sum_i (u_i' \Sigma^{-1} u_i)^2
```

Under ``H_0``: ``T \cdot b_{1,k}/6 \sim \chi^2(k(k+1)(k+2)/6)`` and ``(b_{2,k} - k(k+2)) / \sqrt{8k(k+2)/T} \sim N(0,1)``. Three modes are available: `mardia_test(model; type=:skewness)`, `:kurtosis`, or `:both`. Rejecting kurtosis but not skewness suggests heavy-tailed shocks (Student-t appropriate). Rejecting skewness suggests asymmetric shocks (skew-normal may be preferred).

### Doornik-Hansen and Henze-Zirkler

The Doornik-Hansen (2008) omnibus test transforms each component's skewness and kurtosis via the Bowman-Shenton transformation: ``DH = \sum_{j=1}^k (z_{1j}^2 + z_{2j}^2) \sim \chi^2(2k)``. The Henze-Zirkler (1990) test is based on the empirical characteristic function and is consistent against all alternatives. Call via `doornik_hansen_test(model)` and `henze_zirkler_test(model)`.

### Normality Test Suite

The `normality_test_suite` runs all 7 tests simultaneously: multivariate JB, component-wise JB, Mardia skewness, Mardia kurtosis, Mardia combined, Doornik-Hansen, and Henze-Zirkler. Consistent rejection across multiple tests provides strong evidence against Gaussianity. Individual results are accessible via `suite.results[i]`.

**Return value** (`NormalityTestResult`):

| Field | Type | Description |
|-------|------|-------------|
| `test_name` | `Symbol` | Test identifier |
| `statistic` | `T` | Test statistic value |
| `pvalue` | `T` | p-value |
| `df` | `Int` | Degrees of freedom |
| `n_vars` | `Int` | Number of variables |
| `n_obs` | `Int` | Number of observations |
| `components` | `Vector{T}` or `nothing` | Per-component statistics |
| `component_pvalues` | `Vector{T}` or `nothing` | Per-component p-values |

---

## Shock Gaussianity Test

After recovering structural shocks via ICA or ML, this test verifies that shocks are non-Gaussian. A univariate Jarque-Bera test is applied to each shock:

```math
JB_j = T \left( \frac{\hat{s}_j^2}{6} + \frac{\hat{\kappa}_j^2}{24} \right) \sim \chi^2(2)
```

where ``\hat{s}_j`` is sample skewness and ``\hat{\kappa}_j`` is excess kurtosis. The Darmois-Skitovich theorem requires at most one Gaussian shock for identification. The test counts how many shocks fail to reject at the 5% level.

```@example id_test
ica = identify_fastica(model)
result = test_shock_gaussianity(ica)
report(result)
println("Gaussian shocks: ", result.details[:n_gaussian], " (need <= 1)")
```

The test also accepts `NonGaussianMLResult`: `test_shock_gaussianity(ml_result)`.

---

## Gaussian vs Non-Gaussian LR Test

The likelihood ratio test compares Gaussian and non-Gaussian shock specifications:

```math
LR = 2(\ell_1 - \ell_0) \sim \chi^2(p)
```

where:
- ``\ell_0`` is the Gaussian log-likelihood (Cholesky)
- ``\ell_1`` is the non-Gaussian log-likelihood (ML with distribution-specific parameters)
- ``p = n \times p_{\text{dist}}`` is the number of extra distribution parameters

```@example id_test
lr = test_gaussian_vs_nongaussian(model; distribution=:student_t)
report(lr)
```

Rejecting ``H_0`` supports non-Gaussian identification. Available distributions: `:student_t`, `:mixture_normal`, `:pml`, `:skew_normal`.

---

## Shock Independence Test

Independence of recovered shocks is necessary for valid identification. This test combines two measures via Fisher's method:

1. **Cross-correlation portmanteau**: ``Q = T \sum_{i < j} \sum_{\ell=0}^{L} r_{ij,\ell}^2 \sim \chi^2\bigl(\binom{n}{2}(L+1)\bigr)``
2. **Distance covariance** (Szekely et al. 2007): permutation-based p-value (199 replicates)

Fisher's method: ``\chi^2_F = -2 \sum_k \ln p_k \sim \chi^2(2K)``. Failing to reject (``p \geq 0.05``) indicates independent shocks.

```@example id_test
ica = identify_fastica(model)
result = test_shock_independence(ica; max_lag=10)
report(result)
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `max_lag` | `Int` | `10` | Maximum lag for cross-correlation test |

---

## Identification Strength

The bootstrap identification strength test assesses robustness of the estimated ``B_0``. The procedure resamples residuals ``B`` times, re-estimates ``B_0`` via the specified ICA method, and computes the Procrustes distance between each bootstrap ``B_0`` and the original. Small distances indicate strong identification.

!!! warning "Weak Identification"
    Lewis (2022) demonstrates that weak identification is common in empirical applications. When variances change little or deviations from Gaussianity are small, standard Wald tests have poor size properties and confidence intervals are unreliable. Always run the identification strength test before reporting structural results.

```@example id_test
result = test_identification_strength(model; method=:fastica, n_bootstrap=499)
report(result)
println("Normalized distance: ", round(result.details[:normalized_distance], digits=4))
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:fastica` | ICA method (`:fastica`, `:jade`, `:sobi`) |
| `n_bootstrap` | `Int` | `999` | Number of bootstrap replications |

---

## Overidentification Test

When additional restrictions beyond non-Gaussianity are imposed on ``B_0``, this bootstrap test checks consistency. It compares the discrepancy between ``B_0 B_0'`` and ``\Sigma`` to a bootstrap distribution under the null.

```@example id_test
ica = identify_fastica(model)
result = test_overidentification(model, ica; n_bootstrap=499)
report(result)
```

**Return value** (`IdentifiabilityTestResult` --- shared by all tests in this section):

| Field | Type | Description |
|-------|------|-------------|
| `test_name` | `Symbol` | Test identifier |
| `statistic` | `T` | Test statistic |
| `pvalue` | `T` | p-value |
| `identified` | `Bool` | Whether identification appears to hold |
| `details` | `Dict{Symbol, Any}` | Method-specific details |

---

## Complete Example

```@example id_test
# Step 1: Normality diagnostics
suite = normality_test_suite(model)
report(suite)
```

```@example id_test
# Step 2: LR test — Gaussian vs Student-t
lr = test_gaussian_vs_nongaussian(model; distribution=:student_t)
report(lr)
```

```@example id_test
# Step 3: ICA identification + shock diagnostics
ica = identify_fastica(model)
gauss = test_shock_gaussianity(ica)
report(gauss)
println("Gaussian shocks: ", gauss.details[:n_gaussian])
```

```@example id_test
indep = test_shock_independence(ica; max_lag=5)
report(indep)
println("Shocks independent: ", indep.identified)
```

```@example id_test
# Step 4: Identification strength
strength = test_identification_strength(model; method=:fastica, n_bootstrap=199)
report(strength)
```

```@example id_test
# Step 5: Compute structural IRFs if identification holds
if gauss.identified
    irfs = irf(model, 20; method=:fastica)
    report(irfs)
end
```

---

## Common Pitfalls

1. **Confusing normality rejection with identification**: Rejecting multivariate normality is necessary but not sufficient. The shock gaussianity test must also confirm at most one Gaussian shock. Normality tests operate on reduced-form residuals; shock tests operate on structural shocks.

2. **Independence test direction**: The shock independence test uses fail-to-reject logic. `identified == true` means independence is *not* rejected. This is the opposite convention from normality tests.

3. **Bootstrap sample size**: Use `n_bootstrap=199` for preliminary diagnostics and `n_bootstrap=999` for publication results. The identification strength test is computationally expensive.

4. **Multiple testing**: Running 7 normality tests inflates the family-wise error rate. Look for consistent rejection across multiple tests before concluding non-Gaussianity.

---

## References

- Jarque, Carlos M., and Anil K. Bera. 1980. "Efficient Tests for Normality, Homoscedasticity and Serial Independence of Regression Residuals." *Economics Letters* 6 (3): 255--259. [DOI](https://doi.org/10.1016/0165-1765(80)90024-5)

- Mardia, Kanti V. 1970. "Measures of Multivariate Skewness and Kurtosis with Applications." *Biometrika* 57 (3): 519--530. [DOI](https://doi.org/10.1093/biomet/57.3.519)

- Doornik, Jurgen A., and Henrik Hansen. 2008. "An Omnibus Test for Univariate and Multivariate Normality." *Oxford Bulletin of Economics and Statistics* 70: 927--939. [DOI](https://doi.org/10.1111/j.1468-0084.2008.00537.x)

- Henze, Norbert, and Bernhard Zirkler. 1990. "A Class of Invariant Consistent Tests for Multivariate Normality." *Communications in Statistics - Theory and Methods* 19 (10): 3595--3617. [DOI](https://doi.org/10.1080/03610929008830400)

- Lewis, Daniel J. 2022. "Robust Inference in Models Identified via Heteroskedasticity." *Review of Economics and Statistics* 104 (3): 510--524. [DOI](https://doi.org/10.1162/rest_a_00977)

- Lutkepohl, Helmut. 2005. *New Introduction to Multiple Time Series Analysis*. Berlin: Springer. ISBN 978-3-540-40172-8.
