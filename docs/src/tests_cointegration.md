# [Residual-Based Cointegration Tests](@id tests_cointegration_page)

Four single-equation cointegration tests operate on the residuals — or the coefficient path — of a static cointegrating regression ``y_t = D_t'\delta + x_t'\beta + u_t`` with an I(1) regressor vector ``x_t``. Two are residual unit root tests whose null is **no cointegration**; two are parameter-based tests whose null is **genuine, stable cointegration**. Running one of each is the single-equation analogue of pairing ADF with KPSS: the two families fail in opposite directions, so agreement between them is much stronger evidence than either alone.

For the full test battery and the tables that route a question to a test, see [Hypothesis Tests](@ref tests_page). For the system approach to cointegrating rank, see the Johansen test on [Unit Root & Cointegration](@ref tests_unitroot_page). For cointegration that survives a regime shift, see the Gregory-Hansen test on [Structural Breaks](@ref tests_breaks_page). The `CointRegModel` these tests consume comes from [Cointegrating Regression](@ref cointreg_page).

- **Engle-Granger** — `engle_granger_test(y, X)` runs an augmented Dickey-Fuller regression on the levels residuals; the ``t``-statistic is read against the MacKinnon cointegration surface indexed by the number of I(1) series
- **Phillips-Ouliaris** — `phillips_ouliaris_test(y, X)` forms the semiparametric normalized-bias ``\hat Z_\alpha`` and ``t``-ratio ``\hat Z_t`` on the residual AR(1) root
- **Hansen ``L_c``** — `hansen_instability_test(m)` tests a fitted [`CointRegModel`](@ref) for coefficient stability; a large ``L_c`` signals parameter drift, which is observationally equivalent to no cointegration
- **Park ``H(p,q)``** — `park_added_test(m)` adds superfluous deterministic trends; under genuine cointegration their coefficients are zero, under a spurious regression the statistic diverges

!!! note "The cointegration p-value trap"
    Residual-based cointegration statistics do **not** follow the univariate Dickey-Fuller distribution. Because the residuals are estimated rather than observed, their null distribution depends on ``N = k+1``, the number of I(1) series in the cointegrating vector. All four tests here index the MacKinnon cointegration response surface by ``N`` automatically; feeding the residuals to `adf_test` instead would use the ``N = 1`` surface and badly under-reject.

```@setup coint
using MacroEconometricModels, Random
Random.seed!(20260716)
T = 200
v = randn(T); e = randn(T)
x = cumsum(v)                                   # I(1) regressor (random walk)
u = zeros(T)
for t in 2:T
    u[t] = 0.4 * u[t-1] + e[t]                  # I(0) equation error -> cointegration
end
y = 1.0 .+ 2.0 .* x .+ u                        # cointegrated pair, true vector 2.0
# An independent, non-cointegrated pair for contrast.
xn = cumsum(randn(T)); yn = cumsum(randn(T))
```

## Quick Start

**Recipe 1: Engle-Granger two-step test**

```@example coint
# H0: no cointegration. A small p-value rejects in favour of cointegration.
report(engle_granger_test(y, x; trend=:constant))
```

**Recipe 2: Phillips-Ouliaris ``\hat Z_t`` and ``\hat Z_\alpha``**

```@example coint
report(phillips_ouliaris_test(y, x; trend=:constant))
```

**Recipe 3: Contrast with an independent random-walk pair**

```@example coint
report(engle_granger_test(yn, xn))
```

**Recipe 4: Hansen ``L_c`` stability test on a fitted regression**

```@example coint
m = estimate_cointreg(y, x; method=:fmols, trend=:const)
report(hansen_instability_test(m))
```

**Recipe 5: Park ``H(p,q)`` spurious-regression test**

```@example coint
report(park_added_test(m; q_add=2))
```

---

## Engle-Granger Two-Step Test

Engle & Granger (1987) test the null of **no cointegration** in two steps. Stage 1 estimates the static relationship by OLS in levels. Stage 2 runs an augmented Dickey-Fuller regression **with no deterministic term** on the residuals ``\hat u_t``:

```math
\Delta \hat u_t = \rho\, \hat u_{t-1} + \sum_{j=1}^{p} \gamma_j\, \Delta \hat u_{t-j} + \varepsilon_t
```

where:
- ``\hat u_t`` are the stage-1 levels residuals
- ``\rho`` is the coefficient of interest, with ``H_0: \rho = 0`` (residual unit root, hence no cointegration)
- ``p`` augmenting lags absorb serial correlation in the residual dynamics

The statistic is the ``t``-ratio on ``\rho``. Rejecting the residual unit root is evidence of cointegration. The stage-2 regression carries no intercept because the stage-1 residuals are orthogonal to the deterministics by construction.

```@example coint
result = engle_granger_test(y, x; trend=:constant, lags=:aic)
report(result)
```

The ADF statistic is ``-9.770`` with a MacKinnon p-value below 0.001, so the residual unit root is decisively rejected and the pair is cointegrated — which is correct by construction, since the equation error was generated as a stationary AR(1). AIC selects zero augmenting lags, and ``N = k + 1 = 2`` identifies the surface used for the p-value. The independent pair tells the opposite story:

```@example coint
null_result = engle_granger_test(yn, xn)
(statistic = round(null_result.statistic, digits=3), pvalue = round(null_result.pvalue, digits=3))
```

Here ``-2.410`` gives ``p = 0.320``, nowhere near rejection. Note how much closer to zero this is than a Dickey-Fuller critical value would suggest is needed: on the ``N = 2`` cointegration surface a statistic must reach roughly ``-3.4`` to reject at 5%, against ``-2.86`` for a univariate ADF, precisely because the residuals were fitted rather than observed.

### Options

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `trend` | `Symbol` | `:constant` | Deterministics in the cointegrating regression: `:none`, `:constant` (or `:const`), or `:trend` (or `:linear`). Selects the MacKinnon surface |
| `lags` | `Union{Int,Symbol}` | `:aic` | Augmenting lags ``p``, or `:aic`/`:bic` to select over `0:max_lags` |
| `max_lags` | `Union{Int,Nothing}` | `nothing` | Ceiling for automatic selection (defaults to ``\lfloor 12(T/100)^{1/4} \rfloor``) |

Call it as `engle_granger_test(y, X)` with a vector or matrix of regressors, or as `engle_granger_test(Y)` on a single matrix whose first column is the dependent variable. At least ``3k + 13`` observations are required for ``k`` regressors.

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | ADF ``t``-statistic on the residual lagged level |
| `pvalue` | `T` | MacKinnon (1996/2010) cointegration-surface p-value at ``N = k+1`` |
| `lags` | `Int` | Number of augmenting lags used |
| `regression` | `Symbol` | Normalized deterministic case (`:none`, `:constant`, `:trend`) |
| `k` | `Int` | Number of I(1) regressors |
| `N` | `Int` | Number of I(1) series in the cointegrating vector, ``k + 1`` |
| `nobs` | `Int` | Length of the input series |

The p-value matches `egranger` in Stata and `coint` in statsmodels for the same specification.

---

## Phillips-Ouliaris Test

Phillips & Ouliaris (1990) replace the parametric lag augmentation with a semiparametric correction applied directly to the residual AR(1) root ``\hat\rho``, exactly as the Phillips-Perron test does for a univariate series. With short-run variance ``s^2`` and long-run variance ``\omega^2`` of the AR(1) innovations,

```math
\hat Z_\alpha = T(\hat\rho - 1) - \tfrac{1}{2}(\omega^2 - s^2)\, \frac{T^2}{\sum \hat u_{t-1}^2},
\qquad
\hat Z_t = \sqrt{\tfrac{s^2}{\omega^2}}\, t_{\hat\rho} - \frac{\omega^2 - s^2}{2\,\omega\,(T^{-1}\sum \hat u_{t-1}^2)^{1/2}\sqrt{T}}
```

where:
- ``\hat\rho`` is the OLS root from ``\hat u_t = \hat\rho \hat u_{t-1} + \xi_t`` on the levels residuals
- ``s^2 = T^{-1}\sum \xi_t^2`` is the short-run variance and ``\omega^2`` the HAC long-run variance of ``\xi``
- ``t_{\hat\rho}`` is the uncorrected ``t``-ratio testing ``\hat\rho = 1``

``\hat Z_t`` is the studentized statistic and shares the MacKinnon cointegration surface with Engle-Granger. ``\hat Z_\alpha`` is the normalized bias, which has no closed-form surface, so its p-value brackets Monte-Carlo critical values validated against the Phillips-Ouliaris (1990) tables. Both are left-tailed: large negative values reject no cointegration.

```@example coint
result = phillips_ouliaris_test(y, x; trend=:constant, kernel=:bartlett)
report(result)
```

``\hat Z_t = -9.817`` is within 0.05 of the Engle-Granger statistic on the same data, and ``\hat Z_\alpha = -129.28`` rejects at the 1% level from the normalized-bias side. That agreement is the useful diagnostic: the parametric and semiparametric routes to the same null are handling the residual serial correlation the same way, so the verdict does not depend on which was chosen. When they diverge, the residual dynamics are richer than an AR(1) plus a HAC correction can capture, and the Engle-Granger form with a properly selected lag order is the safer read.

```@example coint
# The independent pair, for contrast
spurious = phillips_ouliaris_test(yn, xn; trend=:constant)
(Z_t = round(spurious.statistic, digits=3), p_t = round(spurious.pvalue, digits=3),
 Z_alpha = round(spurious.z_alpha, digits=2), p_alpha = round(spurious.z_alpha_pvalue, digits=3))
```

### Options

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `trend` | `Symbol` | `:constant` | Deterministics: `:none`, `:constant` (or `:const`), `:trend` (or `:linear`) |
| `kernel` | `Symbol` | `:bartlett` | HAC kernel: `:bartlett`, `:parzen`, `:qs`, or `:tukey_hanning` |
| `bandwidth` | `Symbol` or number | `:nw` | `:nw` uses the fixed rule ``\lfloor 4(T/100)^{1/4} \rfloor``; `:andrews` and `:nw94` are the data-dependent plug-ins; a number is used directly |

The `kernel` and `bandwidth` keywords are forwarded to the shared `lrvar` HAC estimator. Note that the `bandwidth` **field** of the result reports the fixed-rule value whenever a symbolic plug-in was requested, so it does not reflect what `:andrews` or `:nw94` actually chose.

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | ``\hat Z_t``, the studentized statistic (the primary one) |
| `pvalue` | `T` | MacKinnon cointegration-surface p-value for ``\hat Z_t`` at ``N = k+1`` |
| `z_alpha` | `T` | ``\hat Z_\alpha`` normalized-bias statistic |
| `z_alpha_pvalue` | `T` | Bracketing p-value from the Monte-Carlo ``\hat Z_\alpha`` table, floored at 0.01 |
| `regression` | `Symbol` | Normalized deterministic case |
| `kernel` | `Symbol` | HAC kernel used |
| `bandwidth` | `T` | Bandwidth reported (see the caveat above) |
| `k`, `N` | `Int` | Number of I(1) regressors, and ``k+1`` |
| `nobs` | `Int` | Length of the input series |

---

## Hansen ``L_c`` Parameter-Instability Test

Hansen (1992) turns the question around. Instead of testing for a residual unit root, it tests a fitted cointegrating regression for coefficient **stability**. The null is cointegration with constant coefficients; the alternative is that the loadings follow a martingale, which is observationally equivalent to no cointegration. With regressor rows ``Z_t = [D_t; x_t]``, residuals ``\hat u_t``, cumulative scores ``\hat S_t = \sum_{i\le t} Z_i \hat u_i``, and the conditional long-run variance ``\hat\omega^2_{u\cdot v}`` stored on the model,

```math
L_c = \hat\omega_{u\cdot v}^{-2}\; T^{-1} \sum_{t=1}^T \hat S_t' \left(\sum_i Z_i Z_i'\right)^{-1} \hat S_t
```

where:
- ``Z_t`` has ``p = d + k`` columns, ``d`` deterministics and ``k`` I(1) regressors
- ``\hat S_t`` accumulates the scores, so it drifts when the coefficients drift
- ``\hat\omega^2_{u\cdot v}`` is inherited from the `CointRegModel`, not re-estimated

``L_c`` is right-tailed: a **large** value rejects stability.

```@example coint
m = estimate_cointreg(y, x; method=:fmols, trend=:const)
report(hansen_instability_test(m))
```

``L_c = 0.219`` with ``p = 0.309`` fails to reject, so the cointegrating relationship is stable — the right answer for a pair generated with a fixed slope of 2.0, which FMOLS recovers as 2.031. Compare the independent pair, where the "relationship" is spurious and its coefficients therefore wander:

```@example coint
m_spurious = estimate_cointreg(yn, xn; method=:fmols, trend=:const)
(Lc = round(hansen_instability_test(m_spurious).statistic, digits=3),
 pvalue = hansen_instability_test(m_spurious).pvalue)
```

``L_c = 1.361`` rejects at 1%. This is what makes the Hansen test a useful complement: Engle-Granger and Phillips-Ouliaris both *failed to reject* on this pair, which is only weak evidence against cointegration, while Hansen *rejects* its own null, which is direct evidence for instability.

### Options

`hansen_instability_test(m)` takes no keywords. Everything it needs — the deterministic case, the regressors, the residuals, and ``\hat\omega^2_{u\cdot v}`` — comes from the fitted `CointRegModel`, so the kernel and bandwidth are whatever `estimate_cointreg` used.

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | ``L_c`` statistic (right-tailed) |
| `pvalue` | `T` | Bracketing p-value from the Monte-Carlo table, floored at 0.01 |
| `regression` | `Symbol` | Normalized deterministic case |
| `trend` | `Symbol` | The `trend` keyword of the underlying `CointRegModel` |
| `nparam` | `Int` | Number of regressor columns ``p = d + k`` |
| `k` | `Int` | Number of I(1) regressors |
| `nobs` | `Int` | Number of observations |

Critical values are Monte-Carlo quantiles indexed by the deterministic case and by ``k`` (clamped to 1 through 5), spot-checked against Hansen (1992, Table 1). The p-value never falls below 0.01, so read the statistic itself when the rejection is decisive.

---

## Park ``H(p,q)`` Added-Variables Test

Park (1990) augments the cointegrating regression with superfluous deterministic trends and tests whether they matter. Under genuine cointegration the errors are I(0) and the added trends have zero coefficients; under a spurious regression the errors are I(1), the added trends soak up part of that wandering, and the statistic diverges.

Let ``p`` be the highest trend order already present — 0 for `:none` and `:const`, 1 for `:linear`. The regression is re-estimated by OLS with ``q_{\mathrm{add}}`` extra normalized-time trends ``(t/T)^{p+1}, \dots, (t/T)^{p+q_{\mathrm{add}}}`` appended, and their joint significance is tested with a long-run-variance-corrected Wald statistic:

```math
H(p,q) = \hat\gamma'\left(\hat\omega^2\,[(Z'Z)^{-1}]_{AA}\right)^{-1}\hat\gamma \;\sim\; \chi^2(q_{\mathrm{add}})
```

where:
- ``\hat\gamma`` collects the coefficients on the added trends
- ``A`` indexes the added-trend columns of the augmented design ``Z``
- ``\hat\omega^2`` is the long-run variance of the augmented-regression residuals

Unlike the other three tests, the null distribution here is standard, so the p-value is an exact ``\chi^2`` upper tail rather than a table lookup.

```@example coint
report(park_added_test(m; q_add=2))
```

``H(0,2) = 2.875`` against a ``\chi^2(2)`` distribution gives ``p = 0.237``: the two superfluous trends are jointly insignificant, and genuine cointegration survives. The spurious pair behaves completely differently:

```@example coint
(H = round(park_added_test(m_spurious; q_add=2).statistic, digits=2),
 pvalue = park_added_test(m_spurious; q_add=2).pvalue)
```

``H = 82.64`` on 2 degrees of freedom is a p-value around ``10^{-18}``. The divergence is the whole point of the test: with I(1) errors the statistic is not bounded in probability, so the magnitude grows with the sample rather than settling into a ``\chi^2``. Read a very large ``H`` as "the errors are not I(0)", not as a calibrated significance level.

### Options

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `q_add` | `Int` | `2` | Number of superfluous trends added; also the ``\chi^2`` degrees of freedom. Must be ``\geq 1`` |
| `kernel` | `Symbol` | `:bartlett` | HAC kernel for ``\hat\omega^2``: `:bartlett`, `:parzen`, `:qs`, or `:tukey_hanning` |
| `bandwidth` | `Symbol` or number | `:nw` | `:nw` uses ``\lfloor 4(T/100)^{1/4} \rfloor``; `:andrews` and `:nw94` are the plug-ins; a number is used directly |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | ``H(p,q)`` Wald statistic, clipped at zero |
| `pvalue` | `T` | Exact ``\chi^2(q_{\mathrm{add}})`` upper-tail p-value |
| `q_add` | `Int` | Number of superfluous trends, the degrees of freedom |
| `base_order` | `Int` | Highest trend order already in the regression (``p``) |
| `regression` | `Symbol` | Normalized deterministic case |
| `trend` | `Symbol` | The `trend` keyword of the underlying `CointRegModel` |
| `k` | `Int` | Number of I(1) regressors |
| `nobs` | `Int` | Number of observations |

---

## Choosing Between the Four

| Question | Test | Null | Reject means |
|----------|------|------|--------------|
| Is there a long-run relationship at all? | `engle_granger_test` | No cointegration | Cointegration |
| Same, without choosing a lag order | `phillips_ouliaris_test` | No cointegration | Cointegration |
| Is the relationship I already fitted stable? | `hansen_instability_test` | Stable cointegration | Instability |
| Are the errors really I(0)? | `park_added_test` | Genuine cointegration | Spurious regression |

The first two need only the raw series; the last two need a fitted `CointRegModel` and therefore also give you the long-run coefficients. Running one from each half is the informative combination, because a failure to reject in the first half is weak evidence while a rejection in the second half is strong.

---

## Complete Example

A fresh cointegrated pair put through all four tests, from raw series to verdict.

```@example coint
# ── Step 1: Simulate a cointegrated pair ────────────────────────
Random.seed!(7)
Tn = 220
xx = cumsum(randn(Tn))
uu = zeros(Tn)
for t in 2:Tn
    uu[t] = 0.5 * uu[t-1] + 0.8 * randn()      # I(0) AR(1) error -> cointegration
end
yy = 3.0 .+ 1.2 .* xx .+ uu

# ── Step 2: Both no-cointegration nulls ─────────────────────────
eg = engle_granger_test(yy, xx; trend=:constant)
po = phillips_ouliaris_test(yy, xx; trend=:constant)
report(eg)
report(po)
```

```@example coint
# ── Step 3: Fit the long-run relationship ───────────────────────
cr = estimate_cointreg(yy, xx; method=:fmols, trend=:const)
report(cr)
```

```@example coint
# ── Step 4: Both genuine-cointegration nulls ────────────────────
lc = hansen_instability_test(cr)
pk = park_added_test(cr; q_add=2)
report(lc)
report(pk)
```

```@example coint
# ── Step 5: Read the four together ──────────────────────────────
[("Engle-Granger",     round(eg.statistic, digits=2), eg.pvalue < 0.05 ? "cointegrated" : "no cointegration"),
 ("Phillips-Ouliaris", round(po.statistic, digits=2), po.pvalue < 0.05 ? "cointegrated" : "no cointegration"),
 ("Hansen Lc",         round(lc.statistic, digits=3), lc.pvalue < 0.05 ? "unstable" : "stable"),
 ("Park H(p,q)",       round(pk.statistic, digits=3), pk.pvalue < 0.05 ? "spurious" : "genuine")]
```

All four agree. Engle-Granger returns ``-6.314`` and Phillips-Ouliaris ``\hat Z_t = -8.727``, both with p-values below ``10^{-6}``, rejecting no cointegration. Hansen gives ``L_c = 0.267`` (``p = 0.267``) and Park gives ``H = 1.288`` (``p = 0.525``), neither rejecting genuine, stable cointegration. FMOLS recovers a slope of 1.190 against the true 1.2 and an intercept of 2.910 against 3.0. Unanimity across both halves is the outcome to insist on before treating an estimated long-run vector as structural.

---

## Common Pitfalls

1. **Do not use the ADF p-value surface.** Residual-based cointegration statistics depend on ``N = k+1``, because the residuals were estimated. All four tests here index the correct MacKinnon cointegration surface automatically; passing residuals to `adf_test` yourself would use the ``N = 1`` surface and badly under-reject.

2. **Order the columns correctly.** The convenience matrix methods `engle_granger_test(Y)` and `phillips_ouliaris_test(Y)` treat `Y[:, 1]` as the dependent variable and `Y[:, 2:end]` as the I(1) regressors. Swapping them changes the normalization of the cointegrating vector and, in finite samples, the test statistic.

3. **The null direction differs between the two halves.** Engle-Granger and Phillips-Ouliaris test ``H_0``: *no* cointegration, so a small p-value means cointegration. Hansen and Park test ``H_0``: *genuine or stable* cointegration, so a small p-value means instability or spuriousness. Reading all four as "small p is good" inverts half the battery.

4. **Ensure the inputs are I(1).** These tests assume each series has exactly one unit root. A stationary regressor makes the whole framework inapplicable, and an I(2) series invalidates the asymptotics. Pre-test with [`adf_test`](@ref) and [`kpss_test`](@ref), and see [ARDL & Bounds Testing](@ref ardl_page) when the integration orders are mixed.

5. **Kernel and bandwidth move the Phillips-Ouliaris and Park statistics.** Both depend on a HAC long-run variance. Their defaults use the fixed rule ``\lfloor 4(T/100)^{1/4} \rfloor`` rather than a data-dependent plug-in, which is conservative under strong residual persistence. Try `bandwidth=:andrews` as a robustness check, and be aware that the `bandwidth` field the Phillips-Ouliaris result reports still shows the fixed-rule value in that case.

6. **A very large Park ``H`` is not a calibrated p-value.** Under a spurious regression the statistic diverges rather than converging to ``\chi^2``, so p-values of ``10^{-18}`` mean "the errors are not I(0)" and nothing more precise. The same caution applies to the floored p-values of Hansen and ``\hat Z_\alpha``, which never report below 0.01.

---

## References

- Engle, Robert F., and Clive W. J. Granger. 1987. "Co-Integration and Error Correction: Representation, Estimation, and Testing." *Econometrica* 55 (2): 251--276. [https://doi.org/10.2307/1913236](https://doi.org/10.2307/1913236)
- Hansen, Bruce E. 1992. "Tests for Parameter Instability in Regressions with I(1) Processes." *Journal of Business & Economic Statistics* 10 (3): 321--335. [https://doi.org/10.1080/07350015.1992.10509908](https://doi.org/10.1080/07350015.1992.10509908)
- MacKinnon, James G. 1996. "Numerical Distribution Functions for Unit Root and Cointegration Tests." *Journal of Applied Econometrics* 11 (6): 601--618. [https://doi.org/10.1002/(SICI)1099-1255(199611)11:6<601::AID-JAE417>3.0.CO;2-T](https://doi.org/10.1002/(SICI)1099-1255(199611)11:6%3C601::AID-JAE417%3E3.0.CO;2-T)
- MacKinnon, James G. 2010. "Critical Values for Cointegration Tests." Queen's Economics Department Working Paper No. 1227. [https://www.econ.queensu.ca/research/working-papers/1227](https://www.econ.queensu.ca/research/working-papers/1227)
- Park, Joon Y. 1990. "Testing for Unit Roots and Cointegration by Variable Addition." In *Advances in Econometrics* 8, edited by Thomas B. Fomby and George F. Rhodes, 107--133. Greenwich, CT: JAI Press.
- Phillips, Peter C. B., and Bruce E. Hansen. 1990. "Statistical Inference in Instrumental Variables Regression with I(1) Processes." *Review of Economic Studies* 57 (1): 99--125. [https://doi.org/10.2307/2297545](https://doi.org/10.2307/2297545)
- Phillips, Peter C. B., and Sam Ouliaris. 1990. "Asymptotic Properties of Residual Based Tests for Cointegration." *Econometrica* 58 (1): 165--193. [https://doi.org/10.2307/2938339](https://doi.org/10.2307/2938339)
