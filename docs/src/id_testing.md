# [Identification Testing](@id id_testing_page)

Statistical identification buys freedom from economic restrictions at the price of distributional assumptions, and those assumptions are testable. This page documents the diagnostics that decide whether non-Gaussian or heteroskedasticity-based identification is admissible on a given sample: multivariate normality tests on the reduced-form residuals, Gaussianity and independence tests on the recovered structural shocks, a likelihood-ratio test against the Gaussian benchmark, and bootstrap assessments of identification strength.

For an overview and method comparison, see [Statistical Identification](@ref nongaussian_page). For the non-Gaussian estimators these tests validate, see [Non-Gaussian Methods](@ref id_nongaussian_page); for the volatility-based ones, see [Heteroskedasticity](@ref id_heteroskedastic_page).

```@setup id_test
using MacroEconometricModels, Random
Random.seed!(42)
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-119:end, :]
model = estimate_var(Y, 2; varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"])
```

## Quick Start

**Recipe 1: Normality test suite on the residuals**

```@example id_test
suite = normality_test_suite(model)
report(suite)
```

**Recipe 2: Which variable drives the non-normality?**

```@example id_test
jb_comp = jarque_bera_test(model; method=:component)
round.(jb_comp.component_pvalues, digits=4)
```

**Recipe 3: Are the recovered shocks non-Gaussian?**

```@example id_test
ica = identify_fastica(model; rng=MersenneTwister(11))
gauss = test_shock_gaussianity(ica)
report(gauss)
```

**Recipe 4: Likelihood-ratio test against the Gaussian model**

```@example id_test
lr = test_gaussian_vs_nongaussian(model; distribution=:student_t)
report(lr)
```

**Recipe 5: How strong is the identification?**

```@example id_test
strength = test_identification_strength(model; method=:fastica, n_bootstrap=499,
                                        rng=MersenneTwister(7))
report(strength)
```

---

## Multivariate Normality Tests

Testing multivariate normality of the VAR residuals is the first step. If the residuals are Gaussian, non-Gaussian identification is impossible --- the Darmois-Skitovich theorem requires at most one Gaussian component for uniqueness --- and heteroskedasticity-based methods become the only statistical route. Rejecting normality is necessary but not sufficient: it licenses the ICA and ML estimators without guaranteeing that the shocks they recover are themselves non-Gaussian.

### Multivariate Jarque-Bera

The multivariate Jarque-Bera test (Lütkepohl 2005, Section 4.5) combines skewness and kurtosis measures:

```math
JB = T \cdot \frac{b_{1,k}}{6} + T \cdot \frac{(b_{2,k} - k(k+2))^2}{24k}
```

where:
- ``b_{1,k} = T^{-2} \sum_{i,j} (u_i' \Sigma^{-1} u_j)^3`` is multivariate skewness
- ``b_{2,k} = T^{-1} \sum_i (u_i' \Sigma^{-1} u_i)^2`` is multivariate kurtosis
- ``k`` is the number of variables, and under ``H_0``: ``JB \sim \chi^2(2k)``

The `:component` method instead applies univariate JB tests to each standardized residual and sums them, so `components` and `component_pvalues` pinpoint which variables drive the rejection.

```@example id_test
jb = jarque_bera_test(model)
report(jb)
```

```@example id_test
jb_comp = jarque_bera_test(model; method=:component)
(statistics = round.(jb_comp.components, digits=2),
 pvalues = round.(jb_comp.component_pvalues, digits=4))
```

The joint statistic of ``1897.3`` on 6 degrees of freedom rejects normality decisively. The component test attributes it to the two ends of the system: industrial production (``448.0``) and the federal funds rate (``790.6``) are massively non-normal, while the CPI component (``13.3``, ``p = 0.0013``) rejects far more mildly. Non-Gaussian identification therefore has two strongly non-Gaussian directions to work with, which satisfies the Darmois-Skitovich requirement of at most one Gaussian shock with one direction to spare.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:multivariate` | `:multivariate` (joint, Lütkepohl 2005) or `:component` (summed univariate) |

### Mardia's Tests

Mardia (1970) proposed separate tests for multivariate skewness and kurtosis:

```math
b_{1,k} = \frac{1}{T^2} \sum_{i,j} (u_i' \Sigma^{-1} u_j)^3, \qquad b_{2,k} = \frac{1}{T} \sum_i (u_i' \Sigma^{-1} u_i)^2
```

where:
- ``T \cdot b_{1,k}/6 \sim \chi^2(k(k+1)(k+2)/6)`` under ``H_0``
- ``(b_{2,k} - k(k+2)) / \sqrt{8k(k+2)/T} \sim N(0,1)`` under ``H_0``

`type=:skewness` and `type=:kurtosis` report those two statistics; `type=:both` (the default) sums the skewness statistic and the squared kurtosis ``z``, giving a ``\chi^2`` with one extra degree of freedom, and stores the two pieces in `components`.

```@example id_test
mardia_kurt = mardia_test(model; type=:kurtosis)
report(mardia_kurt)
```

The kurtosis statistic is a standard normal deviate of ``31.4`` --- the residuals are enormously more heavy-tailed than a multivariate normal. This is the diagnostic that points to a distribution: rejecting kurtosis but not skewness favours Student-t or mixture shocks, while rejecting skewness favours the skew-normal. On this sample both reject, but the [Non-Gaussian Methods](@ref id_nongaussian_page) page shows the fitted skewness parameters collapsing to zero, so the kurtosis rejection is the one carrying identifying content.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `type` | `Symbol` | `:both` | `:skewness`, `:kurtosis`, or `:both` |

### Doornik-Hansen and Henze-Zirkler

The Doornik-Hansen (2008) omnibus test transforms each component's skewness and kurtosis with the Bowman-Shenton transformation and sums the squared transforms: ``DH = \sum_{j=1}^k (z_{1j}^2 + z_{2j}^2) \sim \chi^2(2k)``. The Henze-Zirkler (1990) test compares the empirical characteristic function against the Gaussian one and is consistent against all alternatives; its p-value comes from a log-normal approximation, and it reports `df = 0` because it is not a ``\chi^2`` test.

```@example id_test
(doornik_hansen = round(doornik_hansen_test(model).statistic, digits=2),
 henze_zirkler = round(henze_zirkler_test(model).statistic, digits=4))
```

### Normality Test Suite

`normality_test_suite` runs all seven tests at once --- multivariate JB, component-wise JB, Mardia skewness, Mardia kurtosis, Mardia combined, Doornik-Hansen, and Henze-Zirkler --- and returns a `NormalityTestSuite` whose individual results are available as `suite.results[i]`.

```@example id_test
suite = normality_test_suite(model)
report(suite)
```

Every one of the seven tests rejects at the 1% level, which is the pattern to insist on before proceeding: the tests use different functionals of the same residuals, so unanimity rules out a rejection driven by one moment or one outlier. Consistency matters more than any single p-value here, because running seven tests on one sample inflates the family-wise error rate.

**Return value** (`NormalityTestResult`):

| Field | Type | Description |
|-------|------|-------------|
| `test_name` | `Symbol` | `:jarque_bera`, `:mardia_skewness`, `:mardia_kurtosis`, `:mardia_both`, `:doornik_hansen`, `:henze_zirkler` |
| `statistic` | `T` | Test statistic value |
| `pvalue` | `T` | p-value |
| `df` | `Int` | Degrees of freedom (0 for Henze-Zirkler) |
| `n_vars` | `Int` | Number of variables |
| `n_obs` | `Int` | Number of observations |
| `components` | `Vector{T}` or `nothing` | Per-component statistics |
| `component_pvalues` | `Vector{T}` or `nothing` | Per-component p-values |

---

## Shock Gaussianity Test

Once shocks have been recovered by ICA or ML, this test checks that they are individually non-Gaussian. A univariate Jarque-Bera test is applied to each shock:

```math
JB_j = T \left( \frac{\hat{s}_j^2}{6} + \frac{\hat{\kappa}_j^2}{24} \right) \sim \chi^2(2)
```

where:
- ``\hat{s}_j`` is the sample skewness of shock ``j``
- ``\hat{\kappa}_j`` is its excess kurtosis

The reported statistic is the sum across shocks, distributed ``\chi^2(2n)``. The `identified` flag applies the Darmois-Skitovich condition directly: it is `true` when at most one shock fails to reject Gaussianity at 5%.

```@example id_test
ica = identify_fastica(model; rng=MersenneTwister(11))
gauss = test_shock_gaussianity(ica)
report(gauss)
```

```@example id_test
(jb_statistics = round.(gauss.details[:jb_stats], digits=2),
 jb_pvalues = round.(gauss.details[:jb_pvals], digits=5),
 n_gaussian = gauss.details[:n_gaussian])
```

All three recovered shocks reject Gaussianity, so `n_gaussian` is 0 and the identification condition holds with a margin of one shock. The individual statistics are lopsided --- ``505``, ``2430`` and ``17.2`` --- and the third shock, though it rejects at ``p = 0.0002``, is the one closest to normal and therefore the one whose column in ``B_0`` is most weakly pinned down. The same test accepts a `NonGaussianMLResult`: `test_shock_gaussianity(ml_result)`.

---

## Gaussian vs Non-Gaussian LR Test

The likelihood ratio test compares Gaussian and non-Gaussian shock specifications on the same VAR:

```math
LR = 2(\ell_1 - \ell_0) \sim \chi^2(p)
```

where:
- ``\ell_0`` is the Gaussian log-likelihood (equivalently, the Cholesky solution)
- ``\ell_1`` is the non-Gaussian log-likelihood at the ML estimate
- ``p = n \times p_{\text{dist}}`` is the number of extra distribution parameters: one per shock for `:student_t` and `:skew_normal`, two for `:mixture_normal` and `:pml`

```@example id_test
lr = test_gaussian_vs_nongaussian(model; distribution=:student_t)
report(lr)
```

```@example id_test
(statistic = round(lr.statistic, digits=2),
 df = lr.details[:df],
 loglik_nongaussian = round(lr.details[:loglik_nongaussian], digits=2),
 loglik_gaussian = round(lr.details[:loglik_gaussian], digits=2))
```

The Student-t specification lifts the log-likelihood from ``979.75`` to ``1035.94``, an LR statistic of ``112.39`` on 3 degrees of freedom that rejects Gaussian shocks at any conventional level. Note the sign convention: unlike the independence and overidentification tests below, `identified` is `true` when the test **rejects**, because rejecting Gaussianity is what licenses the identification. Available distributions are `:student_t`, `:mixture_normal`, `:pml`, and `:skew_normal`, though the `:pml` likelihood is not a normalized density and should not be used here (see [Non-Gaussian Methods](@ref id_nongaussian_page)).

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `distribution` | `Symbol` | `:student_t` | Alternative-hypothesis distribution |

---

## Shock Independence Test

Independence of the recovered shocks is necessary for valid identification, and it is strictly stronger than the orthogonality every rotation delivers by construction. This test combines two measures with Fisher's method:

1. **Cross-correlation portmanteau**: ``Q = T \sum_{i < j} \sum_{\ell=0}^{L} r_{ij,\ell}^2 \sim \chi^2\bigl(\binom{n}{2}(L+1)\bigr)``, which detects linear dependence at leads and lags
2. **Distance covariance** (Székely et al. 2007): a permutation p-value from 199 replicates, which detects any dependence

Fisher's method combines them as ``\chi^2_F = -2 \sum_k \ln p_k \sim \chi^2(2K)``. Failing to reject (``p \geq 0.05``) supports independence, so `identified` is `true` when the p-value is **large**.

!!! warning "The permutation p-value is random"
    The distance-covariance leg shuffles the shock series 199 times using the global RNG. Seed before calling it, or the p-value --- and occasionally the `identified` flag --- will move between runs.

```@example id_test
Random.seed!(20260802)
indep = test_shock_independence(ica; max_lag=10)
report(indep)
```

```@example id_test
(cross_correlation = (round(indep.details[:cc_statistic], digits=2),
                      round(indep.details[:cc_pvalue], digits=4)),
 distance_covariance = indep.details[:dcov_pvalue],
 fisher_pvalue = round(indep.pvalue, digits=4))
```

The two legs disagree, which is exactly why they are combined. The portmanteau statistic of ``49.85`` on 33 degrees of freedom rejects at 5% (``p = 0.030``): some linear cross-dependence survives at leads and lags, unsurprising with 118 observations and 33 correlations to fit. The distance covariance, which tests the full independence null, does not come close to rejecting (``p = 0.55``). Fisher's combination lands at ``p = 0.085``, above the 5% threshold, so the test reports the shocks as independent --- but the reader should treat that as a marginal pass and prefer a longer sample.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `max_lag` | `Int` | `10` | Maximum lead/lag for the cross-correlation portmanteau |

---

## Identification Strength

The bootstrap identification-strength test measures how stable ``B_0`` is under resampling. The procedure resamples the residual rows with replacement ``B`` times, re-estimates ``B_0`` with the chosen ICA method, and computes the **Procrustes distance** --- the minimum Frobenius distance over all signed column permutations --- between each bootstrap ``B_0`` and the original. The statistic is the median distance; `identified` is `true` when that median is below half the Frobenius norm of ``B_0``.

!!! warning "Weak Identification"
    Lewis (2022) shows that weak identification is common in practice. When variances change little or departures from Gaussianity are small, standard Wald tests have poor size and confidence intervals are unreliable. Run this test before reporting any structural result.

```@example id_test
strength = test_identification_strength(model; method=:fastica, n_bootstrap=499,
                                        rng=MersenneTwister(7))
report(strength)
```

```@example id_test
(median_distance = round(strength.statistic, digits=4),
 normalized = round(strength.details[:normalized_distance], digits=4),
 successful_bootstraps = strength.details[:n_bootstrap])
```

The median bootstrap Procrustes distance is ``0.0325``, or ``24.3\%`` of ``\Vert B_0 \Vert_F`` --- comfortably inside the 50% threshold, so the identification is classified as strong. The reported p-value of ``0.060`` is the fraction of bootstrap replications *exceeding* that threshold, not a test of a null hypothesis in the usual sense: read it as "6% of resamples produce a materially different ``B_0``". Draws that fail to converge are dropped silently, so check `details[:n_bootstrap]` against the requested count.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:fastica` | ICA method: `:fastica`, `:jade`, or `:sobi` |
| `n_bootstrap` | `Int` | `999` | Number of bootstrap replications |
| `rng` | `AbstractRNG` | `Random.default_rng()` | Random number generator |

---

## Overidentification Test

When restrictions beyond non-Gaussianity are imposed on ``B_0``, this bootstrap test asks whether they are consistent with the data. It compares the relative discrepancy between ``B_0 B_0'`` and ``\Sigma``, plus the orthogonality error of ``Q``, against a bootstrap distribution of the same discrepancy.

```@example id_test
overid = test_overidentification(model, ica; n_bootstrap=499, rng=MersenneTwister(5))
(statistic = overid.statistic,
 discrepancy = overid.details[:discrepancy],
 orthogonality_error = overid.details[:orthogonality_error])
```

The statistic is ``7.8 \times 10^{-16}``. That is the point: a plain ICA or ML solution is exactly identified, ``B_0 B_0' = \Sigma`` holds to machine precision by construction, and the test has nothing to detect. The `identified` flag it returns in that situation reflects floating-point noise rather than evidence, so apply this test only when ``B_0`` carries genuine overidentifying restrictions --- zero restrictions on the impact matrix, or a rotation imported from a different identification scheme.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `n_bootstrap` | `Int` | `499` | Bootstrap replications under the null |
| `rng` | `AbstractRNG` | `Random.default_rng()` | Random number generator |

**Return value** (`IdentifiabilityTestResult` --- shared by every test in this section):

| Field | Type | Description |
|-------|------|-------------|
| `test_name` | `Symbol` | `:shock_gaussianity`, `:gaussian_vs_nongaussian`, `:shock_independence`, `:identification_strength`, `:overidentification` |
| `statistic` | `T` | Test statistic |
| `pvalue` | `T` | p-value |
| `identified` | `Bool` | Whether identification appears to hold (see the sign conventions below) |
| `details` | `Dict{Symbol, Any}` | Method-specific details |

| Test | `identified == true` means | Convention |
|------|----------------------------|------------|
| `test_shock_gaussianity` | At most one shock is Gaussian | Counts non-rejections |
| `test_gaussian_vs_nongaussian` | Gaussianity is rejected | Reject to identify |
| `test_shock_independence` | Independence is not rejected | Fail to reject to identify |
| `test_identification_strength` | Median distance below ``0.5\Vert B_0 \Vert_F`` | Threshold, not a p-value |
| `test_overidentification` | Restrictions are not rejected | Fail to reject to identify |

---

## Complete Example

A full pre-flight check: residual normality, the parametric LR test, shock-level diagnostics, and identification strength, ending with the IRFs the diagnostics have licensed.

```@example id_test
# --- Step 1: Are the residuals non-normal at all? ---
suite = normality_test_suite(model)
report(suite)
```

```@example id_test
# --- Step 2: Formal LR test against the Gaussian benchmark ---
lr = test_gaussian_vs_nongaussian(model; distribution=:student_t)
(LR = round(lr.statistic, digits=2), df = lr.details[:df],
 rejects_gaussian = lr.identified)
```

```@example id_test
# --- Step 3: Diagnostics on the recovered shocks ---
ica = identify_fastica(model; rng=MersenneTwister(11))
gauss = test_shock_gaussianity(ica)

Random.seed!(20260802)
indep = test_shock_independence(ica; max_lag=10)

(n_gaussian_shocks = gauss.details[:n_gaussian],
 gaussianity_ok = gauss.identified,
 independence_pvalue = round(indep.pvalue, digits=4),
 independence_ok = indep.identified)
```

```@example id_test
# --- Step 4: Bootstrap identification strength ---
strength = test_identification_strength(model; method=:fastica, n_bootstrap=199,
                                        rng=MersenneTwister(7))
(normalized_distance = round(strength.details[:normalized_distance], digits=4),
 strong = strength.identified)
```

```@example id_test
# --- Step 5: Structural IRFs, now that all four checks pass ---
irfs = irf(model, 20; method=:fastica, rng=MersenneTwister(11))
report(irfs)
```

The four checks pass in sequence: the residuals are non-normal on all seven tests, the Student-t likelihood beats the Gaussian one by ``112.39``, none of the three recovered shocks is Gaussian, their independence is not rejected at ``p = 0.085``, and the bootstrap ``B_0`` sits ``22.9\%`` of a norm away from the point estimate under 199 replications. Only then are the impulse responses in step 5 interpretable as structural. Had any check failed --- Gaussian residuals, two Gaussian shocks, rejected independence, or a bootstrap distance near the threshold --- the correct response is to switch to [Heteroskedasticity](@ref id_heteroskedastic_page) or to an economically restricted scheme from [Structural Identification](@ref structural_identification_page), not to report the IRFs with a caveat.

---

## Common Pitfalls

1. **Normality rejection is not identification.** Rejecting multivariate normality of the residuals is necessary but not sufficient. The shock Gaussianity test must also confirm at most one Gaussian shock. The two operate on different objects: normality tests on reduced-form residuals, shock tests on the recovered structural shocks.

2. **The `identified` flag changes direction between tests.** `test_gaussian_vs_nongaussian` sets it by *rejecting*; `test_shock_independence` and `test_overidentification` set it by *failing to reject*. The convention table above lists all five.

3. **Permutation and bootstrap p-values move between runs.** The distance-covariance leg of the independence test and both bootstrap tests consume randomness. Seed the RNG or pass `rng=` before quoting any of these numbers in a paper.

4. **Bootstrap sample size.** Use `n_bootstrap=199` for exploratory work and `n_bootstrap=999` for published results; the identification-strength test re-runs the full ICA estimator on every replication and is the most expensive diagnostic here.

5. **Multiple testing.** Seven normality tests on one sample inflate the family-wise error rate. Look for consistent rejection across tests rather than the smallest p-value.

6. **Overidentification needs something to overidentify.** Applied to a just-identified ICA or ML solution, the test compares two quantities that are both zero to machine precision and returns noise. Reserve it for a ``B_0`` carrying extra restrictions.

---

## References

- Doornik, Jurgen A., and Henrik Hansen. 2008. "An Omnibus Test for Univariate and Multivariate Normality." *Oxford Bulletin of Economics and Statistics* 70: 927--939. [DOI](https://doi.org/10.1111/j.1468-0084.2008.00537.x)

- Henze, Norbert, and Bernhard Zirkler. 1990. "A Class of Invariant Consistent Tests for Multivariate Normality." *Communications in Statistics - Theory and Methods* 19 (10): 3595--3617. [DOI](https://doi.org/10.1080/03610929008830400)

- Jarque, Carlos M., and Anil K. Bera. 1980. "Efficient Tests for Normality, Homoscedasticity and Serial Independence of Regression Residuals." *Economics Letters* 6 (3): 255--259. [DOI](https://doi.org/10.1016/0165-1765(80)90024-5)

- Lewis, Daniel J. 2022. "Robust Inference in Models Identified via Heteroskedasticity." *Review of Economics and Statistics* 104 (3): 510--524. [DOI](https://doi.org/10.1162/rest_a_00977)

- Lütkepohl, Helmut. 2005. *New Introduction to Multiple Time Series Analysis*. Berlin: Springer. ISBN 978-3-540-40172-8.

- Mardia, Kanti V. 1970. "Measures of Multivariate Skewness and Kurtosis with Applications." *Biometrika* 57 (3): 519--530. [DOI](https://doi.org/10.1093/biomet/57.3.519)

- Székely, Gábor J., Maria L. Rizzo, and Nail K. Bakirov. 2007. "Measuring and Testing Dependence by Correlation of Distances." *Annals of Statistics* 35 (6): 2769--2794. [DOI](https://doi.org/10.1214/009053607000000505)
