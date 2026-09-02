# [Identification Testing](@id id_testing_page)

Statistical identification buys freedom from economic restrictions at the price of distributional assumptions, and those assumptions are testable. This page documents the diagnostics that decide whether non-Gaussian or heteroskedasticity-based identification is admissible on a given sample: multivariate normality tests on the reduced-form residuals, Gaussianity and independence tests on the recovered structural shocks, a likelihood-ratio test against the Gaussian benchmark, Wald tests of distinct relative variances, a residual-bootstrap label-stability diagnostic, and likelihood-ratio / Wald overidentification tests of extra zeros on ``B_0``.

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

**Recipe 5: Are the shock labels stable?**

```@example id_test
stab = test_label_stability(model; method=:fastica, n_bootstrap=50,
                            rng=MersenneTwister(7))
report(stab)
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

Statistical identification of ``B_0`` fails when two relative variances are equal (heteroskedastic schemes) or when two or more shocks are Gaussian (non-Gaussian schemes), and it is only labelled uniquely up to signed permutation. Three diagnostics replace the old Procrustes-bootstrap threshold.

!!! warning "Deprecated wrapper"
    `test_identification_strength` remains as a one-release wrapper: heteroskedastic results dispatch to `test_lambda_distinct`, non-Gaussian results to `test_gaussian_shock_count`, and a `VARModel` with an ICA `method` to `test_label_stability`. Call the principled functions directly.

### Distinct relative variances

For Markov-switching, GARCH, smooth-transition, and external-volatility fits, identification of the columns of ``B_0`` requires ``\lambda_i \neq \lambda_j`` for every pair (Lanne, Lütkepohl and Maciejowska 2010). The Wald statistic for pair ``(i,j)`` is

```math
W_{ij} = \frac{(\hat\lambda_i - \hat\lambda_j)^2}{\widehat{\mathrm{Var}}(\hat\lambda_i - \hat\lambda_j)} \sim \chi^2(1)
```

where:
- ``\hat\lambda`` is the relative-variance vector (regime ``k=2`` against ``\Lambda_1 = I``, or the GARCH / smooth-transition analogue)
- the denominator is the corresponding contrast from the delta-method covariance of ``\lambda``

`pairs=:all` tests every ``i < j`` and reports Bonferroni-adjusted p-values. Identification of every column requires every pair to reject.

```@example id_test
Random.seed!(20260831)
ev = identify_external_volatility(model, vcat(fill(1, 59), fill(2, size(model.U, 1) - 59)))
wλ = test_lambda_distinct(ev; pairs=:all)
(wald = round.(wλ.statistic, digits=2),
 p_bonferroni = round.(wλ.pvalue_bonferroni, digits=4))
```

A pair with a Bonferroni p-value above 5% means those two columns are not separately identified: the data cannot tell those two shocks apart from a rotation in their plane (Lewis 2022). Use that pair as the reason to drop a shock, split the sample differently, or switch scheme --- not as a licence to report both columns. The midpoint split here is only a demonstration that the Wald runs on an `ExternalVolatilitySVARResult`; a serious application uses a regime indicator with economic content (NBER recessions, a policy-rule break).

### Gaussian-shock count

Non-Gaussian identification of ``B_0`` requires **at most one** Gaussian shock (Darmois–Skitovich; Lanne, Meitz and Saikkonen 2017; Keweloh 2021). `test_gaussian_shock_count` applies a Jarque–Bera test and an excess-kurtosis ``z``-test to each recovered shock and Holm-adjusts the p-values (Holm 1979):

```math
JB_j = T\left(\frac{\hat s_j^2}{6} + \frac{\hat\kappa_j^2}{24}\right) \sim \chi^2(2), \qquad
z_{\kappa,j} = \hat\kappa_j \big/ \sqrt{24/T} \sim N(0,1)
```

where:
- ``\hat s_j`` and ``\hat\kappa_j`` are the sample skewness and excess kurtosis of shock ``j``
- Holm's adjusted p-value is ``\tilde p_{(k)} = \max_{j\le k}\min\bigl(1,(m-j+1)p_{(j)}\bigr)``

`identified` is `true` if and only if the Holm-adjusted JB count of Gaussian shocks is at most one.

```@example id_test
gcount = test_gaussian_shock_count(ica)
report(gcount)
```

```@example id_test
(n_gaussian = gcount.details[:n_gaussian],
 jb_p_holm = round.(gcount.details[:jb_pvals_holm], digits=5),
 kurt_p = round.(gcount.details[:kurt_pvals], digits=5))
```

The second-smallest JB statistic is ``505.1``, and every Holm-adjusted p-value is below ``10^{-3}``, so `n_gaussian` is 0. Identification holds with a margin of one shock: the Darmois–Skitovich count is satisfied even though, as the label-stability diagnostic below shows, that is not the same thing as a unique labelling of the three columns.

### Label-stability bootstrap

Column labels are identified only up to signed permutation. The label-stability diagnostic re-estimates the VAR on a residual bootstrap, re-identifies ``B_0``, and matches each bootstrap impact to the original with `_match_columns`. The statistic is the **fraction of replications whose permutation is the identity**; signs are allowed to flip. There is **no p-value** --- this is a match-score, not a hypothesis test.

```@example id_test
stab = test_label_stability(model; method=:fastica, n_bootstrap=50,
                            rng=MersenneTwister(7))
report(stab)
```

```@example id_test
(match_fraction = round(stab.details[:match_fraction], digits=3),
 n_identity = stab.details[:n_identity],
 n_bootstrap = stab.details[:n_bootstrap])
```

The match fraction is ``0.44``, below one half, so the diagnostic classifies the labels as unstable --- FastICA's three columns on this sample are not a pinned-down assignment. There is no p-value to quote. Draws that fail to converge are dropped, so check `details[:n_bootstrap]` against the requested count. Re-run with `n_bootstrap=999` before treating a published labelling as stable.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:fastica` | Identification method used on each bootstrap VAR |
| `n_bootstrap` | `Int` | `999` | Residual-bootstrap replications |
| `rng` | `AbstractRNG` | `Random.default_rng()` | Random number generator |

---

## Overidentification Test

Extra zeros on ``B_0`` overidentify a statistically identified SVAR. The test is a nested likelihood ratio for every parametric likelihood (non-Gaussian ML, the Amisano–Giannini AB model, SVEC via the AB B-model, and heteroskedastic ML). ICA has no likelihood: `test_overidentification` falls back to label-stability and records `details[:fallback] = :label_stability`.

The LR statistic is the constrained MLE comparison

```math
\mathrm{LR} = 2(\ell_u - \ell_r) \sim \chi^2(q)
```

where:
- ``\ell_u`` is the unrestricted log-likelihood (Givens rotation plus distribution parameters, or the concentrated AB likelihood)
- ``\ell_r`` is the restricted log-likelihood with ``q`` zeros **imposed** on ``B_0``: the corresponding Givens angles are dropped (``Q`` is solved so ``(LQ)_{ij}=0``) and the remaining parameters are re-optimized
- ``q`` is the number of zeros in the restriction mask (`0` = restricted, `NaN` = free)

For an AB `SVARModel`, omitting `restrictions` reports the stored pattern's concentrated LR. A supplied mask is re-estimated as an AB B-model; the `SVARModel`-only method throws `ArgumentError` rather than silently recycling the stored pattern.

When a covariance ``V`` of ``\mathrm{vec}(B_0)`` is stored, the companion statistic is the Wald quadratic form after the same signed-permutation alignment as `test_restrictions`:

```math
W = \bigl(R\,\mathrm{vec}(\hat B_0)\bigr)'\bigl(RVR'\bigr)^{-1}\bigl(R\,\mathrm{vec}(\hat B_0)\bigr) \sim \chi^2(q)
```

where:
- ``R`` selects the restricted entries
- ``V`` is the delta-method covariance of ``\mathrm{vec}(B_0)``
- `details[:wald_approximation] = :rvr`

If only diagonal SEs exist, the package reports the sum of squared ``t``-ratios as an **independence approximation** (`details[:wald_approximation] = :independence`). That figure is not a Wald ``\chi^2``.

Failing to reject (``p \ge 0.05``) supports the extra zeros, so `identified` is `true` when the p-value is **large**. A just-identified fit with no extra zeros returns p-value 1 and `details[:just_identified] = true`.

```@example id_test
ml = identify_student_t(model)
mask_upper = [NaN 0.0 0.0; NaN NaN 0.0; NaN NaN NaN]
overid_ml = test_overidentification(model, ml; restrictions=mask_upper)
report(overid_ml)
```

```@example id_test
(LR = round(overid_ml.statistic, digits=2),
 pvalue = round(overid_ml.pvalue, digits=4),
 wald = haskey(overid_ml.details, :wald_statistic) ?
        round(overid_ml.details[:wald_statistic], digits=2) : missing)
```

The recursive upper-triangle zeros are extra relative to Student-t identification. The nested LR statistic is ``34.44`` on 3 degrees of freedom and rejects at any conventional level: the statistically identified rotation does not sit on that triangle, so those zeros are not a restriction this sample will bear. When a covariance of ``\mathrm{vec}(B_0)`` is stored the same mask is also reported as an ``RVR'`` Wald in `details[:wald_statistic]`; check `details[:wald_approximation]`.

ICA cannot run that comparison. The call records the fallback and reports the label-stability match fraction with no p-value:

```@example id_test
overid_ica = test_overidentification(model, ica; n_bootstrap=50, rng=MersenneTwister(5))
(fallback = overid_ica.details[:fallback],
 match_fraction = round(overid_ica.details[:match_fraction], digits=3),
 pvalue = overid_ica.pvalue)
```

`details[:fallback]` is `:label_stability` and `pvalue` is `NaN`: ICA has nothing to overidentify, and the diagnostic says so rather than reporting a fake χ². The match fraction is the same object as `test_label_stability`, not a test of extra zeros.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `restrictions` | `AbstractMatrix` or `nothing` | `nothing` | Zero mask on ``B_0`` (`0` = restricted, `NaN` = free) |
| `n_bootstrap` | `Int` | `999` | Residual-bootstrap replications used only for the ICA label-stability fallback |
| `rng` | `AbstractRNG` | `Random.default_rng()` | Random number generator for the ICA fallback |

**Return value** (`IdentifiabilityTestResult` --- shared by every test in this section):

| Field | Type | Description |
|-------|------|-------------|
| `test_name` | `Symbol` | `:shock_gaussianity`, `:gaussian_vs_nongaussian`, `:shock_independence`, `:gaussian_shock_count`, `:lambda_distinct`, `:label_stability`, `:overidentification` |
| `statistic` | `T` | Test statistic (match fraction for label-stability) |
| `pvalue` | `T` | p-value, or `NaN` for label-stability |
| `identified` | `Bool` | Whether identification appears to hold (see the sign conventions below) |
| `details` | `Dict{Symbol, Any}` | Method-specific details |

| Test | `identified == true` means | Convention |
|------|----------------------------|------------|
| `test_shock_gaussianity` | At most one shock is Gaussian | Holm-adjusted non-rejections |
| `test_gaussian_shock_count` | At most one shock is Gaussian | Holm-adjusted JB count |
| `test_gaussian_vs_nongaussian` | Gaussianity is rejected | Reject to identify |
| `test_shock_independence` | Independence is not rejected | Fail to reject to identify |
| `test_lambda_distinct` | Every ``\lambda_i \neq \lambda_j`` | Reject equality to identify |
| `test_label_stability` | Match fraction ``\ge 1/2`` | Match-score, **no p-value** |
| `test_overidentification` | Extra zeros are not rejected | Fail to reject to identify; ICA falls back to label-stability |

---

## Complete Example

A full pre-flight check: residual normality, the parametric LR test, shock-level diagnostics, and label-stability, then the IRFs those diagnostics licence --- or refuse to name.

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
# --- Step 4: Label-stability of the ICA columns ---
stab = test_label_stability(model; method=:fastica, n_bootstrap=50,
                            rng=MersenneTwister(7))
(match_fraction = round(stab.details[:match_fraction], digits=3),
 labels_stable = stab.identified)
```

```@example id_test
# --- Step 5: Reduced-form-rotated IRFs; columns are not yet labelled ---
irfs = irf(model, 20; method=:fastica, rng=MersenneTwister(11))
report(irfs)
```

The residual and shock checks pass: the residuals are non-normal on all seven tests, the Student-t likelihood beats the Gaussian one by ``112.39``, none of the three recovered shocks is Gaussian, and their independence is not rejected at ``p = 0.085``. Label-stability does not: the match fraction is ``0.44``, so FastICA's column order is not a pinned-down assignment on this sample. The impulse responses in step 5 are still a rotation of the reduced-form MA, but the shock *names* are not identified. Label them with `label_shocks` from [Non-Gaussian Methods](@ref id_nongaussian_page), switch to [Heteroskedasticity](@ref id_heteroskedastic_page), or impose zeros and test them with `test_overidentification` --- do not report unnamed ICA columns as demand, supply, and monetary policy.

---

## Common Pitfalls

1. **Normality rejection is not identification.** Rejecting multivariate normality of the residuals is necessary but not sufficient. The shock Gaussianity test must also confirm at most one Gaussian shock. The two operate on different objects: normality tests on reduced-form residuals, shock tests on the recovered structural shocks.

2. **The `identified` flag changes direction between tests.** `test_gaussian_vs_nongaussian` and `test_lambda_distinct` set it by *rejecting*; `test_shock_independence` and `test_overidentification` set it by *failing to reject*. Label-stability has no p-value. The convention table above lists every test.

3. **Permutation and bootstrap diagnostics move between runs.** The distance-covariance leg of the independence test and the label-stability bootstrap consume randomness. Seed the RNG or pass `rng=` before quoting any of these numbers in a paper.

4. **Bootstrap sample size.** Use `n_bootstrap=50` for exploratory work and `n_bootstrap=999` for published results. Label-stability re-estimates the VAR and re-identifies ``B_0`` on every replication.

5. **Multiple testing.** Seven normality tests on one sample inflate the family-wise error rate. Look for consistent rejection across tests rather than the smallest p-value. Per-shock JB p-values are Holm-adjusted inside `test_gaussian_shock_count`.

6. **Overidentification needs something to overidentify.** A just-identified ML, GARCH, or AB fit with `restrictions=nothing` returns p-value 1. ICA has no likelihood: `test_overidentification` falls back to label-stability and says so. Pass a zero mask on ``B_0`` for the nested LR; an AB mask is re-estimated rather than tested against the stored pattern. The companion Wald is ``RVR'`` when ``V`` is stored and an independence approximation of diagonal SEs otherwise --- do not quote the latter as a Wald ``\chi^2``.

---

## References

- Amisano, Gianni, and Carlo Giannini. 1997. *Topics in Structural VAR Econometrics*. 2nd ed. Berlin: Springer. ISBN 978-3-540-61942-0.

- Doornik, Jurgen A., and Henrik Hansen. 2008. "An Omnibus Test for Univariate and Multivariate Normality." *Oxford Bulletin of Economics and Statistics* 70: 927--939. [DOI](https://doi.org/10.1111/j.1468-0084.2008.00537.x)

- Henze, Norbert, and Bernhard Zirkler. 1990. "A Class of Invariant Consistent Tests for Multivariate Normality." *Communications in Statistics - Theory and Methods* 19 (10): 3595--3617. [DOI](https://doi.org/10.1080/03610929008830400)

- Holm, Sture. 1979. "A Simple Sequentially Rejective Multiple Test Procedure." *Scandinavian Journal of Statistics* 6 (2): 65--70. [DOI](https://doi.org/10.2307/4615733)

- Jarque, Carlos M., and Anil K. Bera. 1980. "Efficient Tests for Normality, Homoscedasticity and Serial Independence of Regression Residuals." *Economics Letters* 6 (3): 255--259. [DOI](https://doi.org/10.1016/0165-1765(80)90024-5)

- Keweloh, Sascha A. 2021. "A Generalized Method of Moments Estimator for Structural Vector Autoregressions Based on Higher Moments." *Journal of Business & Economic Statistics* 39 (3): 772--782. [DOI](https://doi.org/10.1080/07350015.2020.1730858)

- Lanne, Markku, Mika Meitz, and Pentti Saikkonen. 2017. "Identification and Estimation of Non-Gaussian Structural Vector Autoregressions." *Journal of Econometrics* 196 (2): 288--304. [DOI](https://doi.org/10.1016/j.jeconom.2016.06.002)

- Lanne, Markku, Helmut Lütkepohl, and Katarzyna Maciejowska. 2010. "Structural Vector Autoregressions with Markov Switching." *Journal of Economic Dynamics and Control* 34 (2): 121--131. [DOI](https://doi.org/10.1016/j.jedc.2009.08.002)

- Lewis, Daniel J. 2022. "Robust Inference in Models Identified via Heteroskedasticity." *Review of Economics and Statistics* 104 (3): 510--524. [DOI](https://doi.org/10.1162/rest_a_00963)

- Lütkepohl, Helmut. 2005. *New Introduction to Multiple Time Series Analysis*. Berlin: Springer. ISBN 978-3-540-40172-8.

- Mardia, Kanti V. 1970. "Measures of Multivariate Skewness and Kurtosis with Applications." *Biometrika* 57 (3): 519--530. [DOI](https://doi.org/10.1093/biomet/57.3.519)

- Székely, Gábor J., Maria L. Rizzo, and Nail K. Bakirov. 2007. "Measuring and Testing Dependence by Correlation of Distances." *Annals of Statistics* 35 (6): 2769--2794. [DOI](https://doi.org/10.1214/009053607000000505)
