# [Structural Breaks](@id tests_breaks_page)

This page covers tests for parameter instability and structural change. Structural breaks --- abrupt shifts in regression coefficients, factor loadings, or a cointegrating relationship --- invalidate standard inference and produce misleading forecasts when they go undetected. For the full test battery and the tables that route a question to a test, see [Hypothesis Tests](@ref tests_page). Four complementary frameworks are provided:

- **Andrews (1993)**: One unknown break point in a linear regression. Nine variants combine three base statistics (Wald, LR, LM) with three functionals (supremum, exponential average, mean).
- **Bai-Perron (1998, 2003)**: Multiple unknown break points. Dynamic programming finds globally optimal break dates; sequential sup-F tests and the BIC and LWZ criteria select the number of breaks.
- **Factor model break tests**: Instability in factor loadings --- Breitung-Eickmeier (2011) pooled per-series LM, Chen-Dolado-Gonzalo (2014) regression sup-LM, and Han-Inoue (2015) sup-Wald.
- **Gregory-Hansen (1996)**: Cointegration allowing one structural break in the long-run relationship. Three break models (level shift, level + trend, regime shift) and three residual statistics (ADF\*, Zt\*, Za\*).

Every test on this page returns a result object that `report` renders as a specification block, a statistics block, a critical-value block, and a conclusion line. Unit root and cointegration tests without breaks live on [Unit Root & Cointegration](@ref tests_unitroot_page); the regression-stability tests that condition on a *known* break date (`chow_test`, `cusum_test`) live on [Linear Regression](@ref regression_page).

```@setup test_breaks
using MacroEconometricModels, Random
Random.seed!(42)
```

## Quick Start

Each recipe seeds its own generator, so the numbers below reproduce exactly when the block is copied into a fresh session.

**Recipe 1: Andrews sup-Wald test for one unknown break**

```@example test_breaks
Random.seed!(42)
X = hcat(ones(200), randn(200))                  # intercept + one regressor
y = X * [1.0, 2.0] + 0.5 * randn(200)
y[101:end] .+= 3.0 .* X[101:end, 2]              # slope jumps from 2.0 to 5.0 at t = 100

report(andrews_test(y, X; test=:supwald))
```

**Recipe 2: Bai-Perron search for multiple breaks**

```@example test_breaks
Random.seed!(43)
X_bp = ones(300, 1)                              # intercept-only regression
y_bp = vcat(fill(2.0, 100), fill(5.0, 100), fill(1.0, 100)) + 0.5 * randn(300)

report(bai_perron_test(y_bp, X_bp; max_breaks=5))
```

**Recipe 3: Factor loading stability**

```@example test_breaks
Random.seed!(44)
F = randn(200, 3)                                # 3 common factors
Lambda = randn(60, 3)                            # stable loadings for 60 variables
X_fac = F * Lambda' + 0.5 * randn(200, 60)

report(factor_break_test(X_fac, 3; method=:han_inoue))
```

**Recipe 4: Cointegration with a regime shift**

```@example test_breaks
Random.seed!(45)
x_gh = cumsum(randn(200))                        # I(1) regressor
y_gh = vcat(1.0 .+ 0.8 .* x_gh[1:100],           # slope and intercept both shift
            3.0 .+ 1.5 .* x_gh[101:end]) + 0.5 * randn(200)

report(gregory_hansen_test(hcat(y_gh, x_gh); model=:CS))
```

---

## Andrews Structural Break Test

The Andrews (1993) test detects a single structural break at an unknown date in a linear regression. The null hypothesis is parameter constancy: every coefficient is the same over the whole sample. The test evaluates a break statistic at each candidate date in a trimmed range and aggregates the resulting sequence with a functional.

Consider the linear model:

```math
y_t = X_t' \beta + u_t, \quad t = 1, \ldots, T
```

where:
- ``y_t`` is the dependent variable at time ``t``
- ``X_t`` is the ``k \times 1`` vector of regressors
- ``\beta`` is the ``k \times 1`` parameter vector
- ``u_t`` is the error term

Under the alternative the parameter vector shifts at an unknown break date ``t_b``:

```math
\beta = \begin{cases} \beta_1 & t \leq t_b \\ \beta_2 & t > t_b \end{cases}
```

The **Wald statistic** at a candidate break date ``t_b`` compares the two sub-sample estimates:

```math
W(t_b) = (\hat{\beta}_1 - \hat{\beta}_2)' \left[ V_1 + V_2 \right]^{-1} (\hat{\beta}_1 - \hat{\beta}_2)
```

where:
- ``\hat{\beta}_1`` and ``\hat{\beta}_2`` are OLS estimates from observations ``1{:}t_b`` and ``(t_b{+}1){:}T``
- ``V_1 = \hat{\sigma}_1^2 (X_1' X_1)^{-1}`` and ``V_2 = \hat{\sigma}_2^2 (X_2' X_2)^{-1}`` are their covariance matrices, each using the degrees-of-freedom-corrected sub-sample variance

The **LR statistic** compares the full-sample and split-sample residual sums of squares:

```math
\text{LR}(t_b) = T \left[ \ln(\text{SSR}_0 / T) - \ln(\text{SSR}_{\text{split}} / T) \right]
```

The **LM statistic** is a score test that needs only the full-sample estimates:

```math
\text{LM}(t_b) = \frac{S(t_b)' (X'X)^{-1} S(t_b)}{\hat{\sigma}^2 \cdot \tau (1 - \tau)}
```

where ``S(t_b) = \sum_{t=1}^{t_b} X_t \hat{u}_t`` is the partial sum of scores, ``\tau = t_b / T``, and ``\hat{\sigma}^2 = \text{SSR}_0/T``.

Three **functionals** aggregate the base statistic over the trimmed range ``[\pi T, (1 - \pi) T]``:

| Variant | Functional | Base statistic | Reference |
|---------|-----------|----------------|-----------|
| `:supwald` | Supremum | Wald | Andrews (1993) |
| `:suplr` | Supremum | LR | Andrews (1993) |
| `:suplm` | Supremum | LM | Andrews (1993) |
| `:expwald` | Exponential | Wald | Andrews-Ploberger (1994) |
| `:explr` | Exponential | LR | Andrews-Ploberger (1994) |
| `:explm` | Exponential | LM | Andrews-Ploberger (1994) |
| `:meanwald` | Mean | Wald | Andrews-Ploberger (1994) |
| `:meanlr` | Mean | LR | Andrews-Ploberger (1994) |
| `:meanlm` | Mean | LM | Andrews-Ploberger (1994) |

The **supremum** functional takes the maximum statistic over all candidate dates and has optimal power against a single large break. The **exponential** functional computes ``\log(\text{mean}(\exp(W/2)))`` and the **mean** functional computes ``\text{mean}(W)``; both weight every candidate date and gain power against small or gradual parameter changes (Andrews & Ploberger 1994). All three report the same `break_index` --- the date at which the base statistic peaks --- because only the aggregation, not the argmax, differs across functionals.

!!! note "Technical Note"
    P-values interpolate the Hansen (1997) critical-value tables, which are indexed by the number of parameters ``k`` and the functional. Models with ``k > 10`` reuse the ``k = 10`` column. The tables are computed for the standard trimming ``\pi = 0.15``, so a different `trimming` changes the statistic but not the reference distribution. Interpolated p-values are floored at 0.005 beyond the 1% critical value and rise toward 1.0 below the 10% value; the asymptotic null distribution is a functional of a ``k``-dimensional Brownian bridge.

```@example test_breaks
Random.seed!(101)
X_a = hcat(ones(200), randn(200))
y_a = vcat(X_a[1:100, :] * [1.0, 2.0],           # slope 2.0 for the first half
           X_a[101:end, :] * [1.0, 5.0]) + 0.5 * randn(200)

report(andrews_test(y_a, X_a; test=:supwald))
```

The sup-Wald statistic of 1803.77 sits two orders of magnitude above the 1% critical value of 14.72, so the p-value hits its 0.005 floor and parameter constancy is rejected decisively. The estimated break at observation 100 --- a break fraction of exactly 0.5 --- recovers the true break date, and the 200-observation sample with a slope shift from 2.0 to 5.0 leaves no ambiguity about the split. This is the configuration the supremum functional is built for: one sharp, large change at a single date.

The exponential and mean functionals apply different weights to the same statistic sequence:

```@example test_breaks
report(andrews_test(y_a, X_a; test=:expwald))
```

```@example test_breaks
report(andrews_test(y_a, X_a; test=:meanwald))
```

Exp-Wald returns 896.94 against a 1% critical value of 5.01, and mean-Wald returns 411.73 against 7.58 --- both reject at the same 0.005 floor and both place the break at observation 100. The statistics are not comparable in level across functionals because each has its own null distribution; only the position relative to the matching critical values carries information. Agreement across all three functionals, as here, is the signature of a break sharp enough that the weighting scheme does not matter.

Under the null the test is correctly sized but not conservative, because the supremum searches over every admissible split:

```@example test_breaks
Random.seed!(102)
X_0 = hcat(ones(200), randn(200))
y_0 = X_0 * [1.0, 2.0] + 0.5 * randn(200)        # no break
r_0 = andrews_test(y_0, X_0; test=:supwald)
(statistic = round(r_0.statistic, digits=3),
 pvalue = round(r_0.pvalue, digits=3),
 break_index = r_0.break_index)
```

With no break in the data the statistic falls to 9.967, below the 5% critical value of 11.03, and the test correctly fails to reject at conventional levels. The p-value of 0.078 is nevertheless small enough to look suggestive, which is exactly the size behaviour to expect: the maximum of a Brownian-bridge functional over 140 candidate dates is large by construction, and the reported `break_index` of 96 is a sampling artefact with no economic content. Never read a break date from a test that does not reject.

### Options

| Keyword | Type | Default | Description |
|----------|------|---------|-------------|
| `test` | `Symbol` | `:supwald` | Any of the nine functional/base-statistic combinations above |
| `trimming` | `Real` | `0.15` | Fraction of the sample trimmed from each end, in ``(0, 0.5)`` |

The test requires at least 20 observations and ``y`` and ``X`` with matching row counts.

### Return values

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | Functional applied to the base-statistic sequence |
| `pvalue` | `T` | Interpolated p-value from the Hansen (1997) tables |
| `break_index` | `Int` | Observation index at which the base statistic peaks |
| `break_fraction` | `T` | `break_index / nobs` |
| `test_type` | `Symbol` | Test variant, e.g. `:supwald` |
| `critical_values` | `Dict{Int,T}` | Critical values keyed by `1`, `5`, `10` (percent) |
| `stat_sequence` | `Vector{T}` | Base statistic at every candidate break date |
| `trimming` | `T` | Trimming fraction used |
| `nobs` | `Int` | Number of observations |
| `n_params` | `Int` | Number of regressors ``k`` |

`plot_result(result)` draws the full `stat_sequence` over the trimmed candidate range, with the estimated break date as a vertical line and the 5% critical value as a dashed horizontal reference.

---

## Bai-Perron Multiple Break Test

The Bai-Perron (1998, 2003) procedure detects and dates multiple structural breaks. Dynamic programming finds the break dates that globally minimize the total sum of squared residuals, avoiding the suboptimality of sequential single-break searches.

The model with ``m`` breaks has ``m + 1`` regimes:

```math
y_t = X_t' \beta_j + u_t, \quad t = T_{j-1} + 1, \ldots, T_j, \quad j = 1, \ldots, m + 1
```

where:
- ``T_0 = 0`` and ``T_{m+1} = T`` are the sample boundaries
- ``T_1, \ldots, T_m`` are the unknown break dates
- ``\beta_j`` is the ``k \times 1`` parameter vector for regime ``j``

The optimal break dates minimize the total SSR:

```math
(\hat{T}_1, \ldots, \hat{T}_m) = \arg\min_{T_1, \ldots, T_m} \sum_{j=1}^{m+1} \sum_{t=T_{j-1}+1}^{T_j} (y_t - X_t' \hat{\beta}_j)^2
```

The **sup-F test** of ``l`` breaks against zero breaks is:

```math
\text{sup-}F(l) = \frac{(\text{SSR}_0 - \text{SSR}_l) / (l \cdot k)}{\text{SSR}_l / (T - (l+1) \cdot k)}
```

where:
- ``\text{SSR}_0`` is the full-sample residual sum of squares
- ``\text{SSR}_l`` is the minimized SSR with ``l`` optimally placed breaks
- ``k`` is the number of regressors

The **sequential test** ``\text{sup-}F(l+1 \mid l)`` asks whether one more break improves the fit over the ``l``-break model. Two **information criteria** offer an alternative to sequential testing:

```math
\text{BIC}(m) = T \ln(\text{SSR}_m / T) + (m+1) k \ln T
```

```math
\text{LWZ}(m) = T \ln(\text{SSR}_m / T) + (m+1) k \cdot 0.299 (\ln T)^{2.1}
```

where:
- ``m`` is the number of breaks
- ``(m+1)k`` is the total number of estimated coefficients
- the LWZ criterion (Liu, Wu & Zidek 1997) penalizes each additional regime more heavily than BIC

Both criteria are always computed; `criterion` chooses which one sets the reported `n_breaks`.

!!! note "Technical Note"
    The dynamic program has complexity ``O(T^2 \cdot m_{\max})``. A segment SSR matrix is pre-computed once and reused across all candidate configurations. Each segment needs at least ``h = \max(k+1, \lceil \pi T \rceil)`` observations, so `max_breaks` is silently reduced to ``\lfloor T/h \rfloor - 1`` when the requested value is infeasible. Break-date confidence intervals follow Bai (1997), with a half-width of ``1.96^2 \hat{\sigma}^2 / (\delta' \hat{Q} \delta)`` where ``\delta`` is the coefficient change at the break.

```@example test_breaks
Random.seed!(103)
X_2 = hcat(ones(300), randn(300))
y_2 = vcat(X_2[1:100, :]   * [1.0,  2.0],        # regime 1
           X_2[101:200, :] * [3.0, -1.0],        # regime 2
           X_2[201:300, :] * [0.0,  4.0]) + 0.5 * randn(300)

bp = bai_perron_test(y_2, X_2; max_breaks=5, criterion=:bic)
report(bp)
```

The procedure recovers the design exactly: two breaks at observations 100 and 200, each with a 95% confidence interval two observations wide. The sequential column tells the same story --- sup-F(2|1) is 1142.21 with a p-value at the 0.001 floor, while sup-F(3|2) falls to 3.21 with a p-value of 0.765 --- so the sequential rule stops at two breaks. BIC and LWZ both mark ``m = 2`` as the minimum, so all three selection devices agree; disagreement between them is the normal signal that the break structure is weakly identified.

The regime coefficients recover the true parameter vectors to within two standard errors:

```@example test_breaks
[(regime = j,
  intercept = round(c[1], digits=2), se_intercept = round(s[1], digits=3),
  slope = round(c[2], digits=2), se_slope = round(s[2], digits=3))
 for (j, (c, s)) in enumerate(zip(bp.regime_coefs, bp.regime_ses))]
```

Regime 1 returns ``(0.99, 1.92)`` against the true ``(1.0, 2.0)``, regime 2 returns ``(2.98, -0.99)`` against ``(3.0, -1.0)``, and regime 3 returns ``(0.00, 4.00)`` against ``(0.0, 4.0)``. Standard errors near 0.05 make every deviation statistically indistinguishable from zero. Because the break dates are estimated, these standard errors condition on the selected dates and understate the true uncertainty; use `break_cis` to gauge how much the dates themselves could move.

P-values interpolate the Bai & Perron (1998, 2003) critical-value tables, indexed by the number of breaking regressors ``q`` (every regressor breaks in this pure structural-change model, so ``q = k``, and ``q \leq 10`` is required) and by the trimming fraction ``\varepsilon \in \{0.05, 0.10, 0.15, 0.20, 0.25\}``. A requested trimming off that grid snaps to the nearest tabulated value with a warning. The sup-F tables cover 9, 8, 5, 3, and 2 breaks for the five trimmings respectively; a statistic beyond the tabulated break count returns `NaN` rather than an extrapolated number.

### Options

| Keyword | Type | Default | Description |
|----------|------|---------|-------------|
| `max_breaks` | `Int` | `5` | Maximum number of breaks searched, reduced automatically when segments are too short |
| `trimming` | `Real` | `0.15` | Minimum fraction of observations per segment |
| `criterion` | `Symbol` | `:bic` | Criterion that sets `n_breaks`: `:bic` or `:lwz` |

### Return values

| Field | Type | Description |
|-------|------|-------------|
| `n_breaks` | `Int` | Number of breaks selected by `criterion` |
| `break_dates` | `Vector{Int}` | Estimated break date indices |
| `break_cis` | `Vector{Tuple{Int,Int}}` | 95% confidence intervals for the break dates (Bai 1997) |
| `regime_coefs` | `Vector{Vector{T}}` | OLS coefficients for each of the ``m+1`` regimes |
| `regime_ses` | `Vector{Vector{T}}` | Standard errors for each regime |
| `supf_stats` | `Vector{T}` | sup-F(l) statistics for ``l = 1, \ldots, m_{\max}`` |
| `supf_pvalues` | `Vector{T}` | P-values for sup-F, `NaN` beyond the tabulated break count |
| `sequential_stats` | `Vector{T}` | Sequential sup-F(l+1\|l) statistics |
| `sequential_pvalues` | `Vector{T}` | P-values for the sequential statistics |
| `bic_values` | `Vector{T}` | BIC for 0 through ``m_{\max}`` breaks |
| `lwz_values` | `Vector{T}` | LWZ for 0 through ``m_{\max}`` breaks |
| `trimming` | `T` | Trimming fraction used |
| `nobs` | `Int` | Number of observations |

`plot_result(bp)` draws the BIC and LWZ paths against the number of breaks with the selection marked; `plot_result(bp; view=:breaks)` draws the estimated break dates as a timeline with their confidence bands shaded.

---

## Factor Model Break Tests

A factor model assumes that a large cross-section ``X_{it}`` loads on a few common factors ``F_t`` through time-invariant loadings ``\Lambda``. Instability in the loadings breaks principal-components estimation and everything built on it. Three tests target different forms of that instability, and all three accept either a ``T \times N`` matrix or an estimated [`FactorModel`](@ref).

All three treat the break date as unknown, so all three maximize a statistic over the trimmed grid ``\pi \in [0.15, 0.85]``. A supremum is not ``\chi^2``: Han-Inoue and Chen-Dolado-Gonzalo read their p-values from the Andrews (1993)/Hansen (1997) sup-Wald tables, and Breitung-Eickmeier simulates its null reference conditional on the estimated factors.

The examples in this section use one panel with genuinely stable loadings and one in which every loading shifts halfway through the sample:

```@example test_breaks
Random.seed!(104)
F_t = randn(200, 3)                              # 3 factors, T = 200
Lambda_1 = randn(60, 3)                          # loadings for N = 60 variables
Lambda_2 = Lambda_1 .+ 1.5 .* randn(60, 3)       # every loading shifts at t = 100

X_stable = F_t * Lambda_1' + 0.5 * randn(200, 60)
X_break = vcat(F_t[1:100, :] * Lambda_1',
               F_t[101:end, :] * Lambda_2') + 0.5 * randn(200, 60)
size(X_stable, 1), size(X_stable, 2)
```

### Han-Inoue (2015)

The Han-Inoue test aggregates individual Wald statistics for loading instability across all ``N`` cross-section units. For variable ``i`` and candidate break date ``t``, the loading regression ``x_{it} = F_t' \lambda_i + e_{it}`` is estimated on ``[1, t]`` and ``[t+1, T]``, giving

```math
W_i(t) = (\hat{\lambda}_{1,i} - \hat{\lambda}_{2,i})' \left[ \hat{\sigma}_i^2 \left( (F_1'F_1)^{-1} + (F_2'F_2)^{-1} \right) \right]^{-1} (\hat{\lambda}_{1,i} - \hat{\lambda}_{2,i})
```

and the test statistic averages across units before maximizing over dates:

```math
\text{HI} = \sup_t \frac{1}{N} \sum_{i=1}^{N} W_i(t)
```

where:
- ``\hat{\lambda}_{1,i}`` and ``\hat{\lambda}_{2,i}`` are sub-sample loading estimates for variable ``i``
- ``\hat{\sigma}_i^2`` is the full-sample residual variance for variable ``i``
- ``F_1`` and ``F_2`` are the estimated factor matrices for the two sub-samples

Averaging across the cross-section is what supplies power when many loadings move together. P-values use the Andrews (1993) sup-Wald critical values with ``k = r`` degrees of freedom.

```@example test_breaks
report(factor_break_test(X_stable, 3; method=:han_inoue))
```

```@example test_breaks
report(factor_break_test(X_break, 3; method=:han_inoue))
```

On the stable panel the statistic is 3.4637 with a p-value of 0.3255, so loading stability survives; on the broken panel it is 121.2991, the p-value collapses below 0.001, and the estimated break lands on observation 100, the true date. The gap between 3.46 and 121.30 is the cross-sectional averaging at work: each of the 60 individual Wald statistics contributes a small amount of evidence that would be unconvincing on its own. Note the break date of 52 reported on the stable panel --- meaningless, because the null was not rejected.

Averaging is also what makes the test conservative. At a fixed date each ``W_i(t)`` is ``\chi^2(r)``, so the average concentrates near ``r`` --- 3.46 on the stable panel above, against ``r = 3`` --- while the reference is the sup-Wald critical value for ``k = r``, 12.96 at the 5% level. Han-Inoue therefore rejects only for large breaks, and a non-rejection is weak evidence of stability (see Pitfall 5).

### Breitung-Eickmeier (2011)

The Breitung-Eickmeier test runs one loading-break regression per series and pools the results across the cross-section. For series ``i`` and candidate date ``\tau``, the auxiliary regression

```math
x_{it} = \lambda_i' \hat{F}_t + \delta_i' \hat{F}_t \cdot 1(t > \tau) + e_{it}
```

is tested for ``H_0: \delta_i = 0`` by the LM statistic

```math
LM_i(\tau) = \frac{S_i(\tau)' \left[ A_1(\tau)^{-1} + A_2(\tau)^{-1} \right] S_i(\tau)}{\hat{\sigma}_i^2}
```

where:
- ``S_i(\tau) = \sum_{t \leq \tau} \hat{F}_t \hat{e}_{it}`` is the partial sum of the full-sample loading scores
- ``A_1(\tau) = \sum_{t \leq \tau} \hat{F}_t \hat{F}_t'`` and ``A_2(\tau) = \hat{F}'\hat{F} - A_1(\tau)``
- ``\hat{\sigma}_i^2`` is the full-sample idiosyncratic variance of series ``i``

At a fixed ``\tau`` this is algebraically the Chow-Wald statistic for equality of the pre- and post-break loadings of series ``i``, and is asymptotically ``\chi^2(r)``. The break date is unknown, so the series-level statistic is the supremum ``M_i = \sup_\tau LM_i(\tau)``, and the panel statistic standardizes the pooled sum:

```math
Z = \frac{\sum_{i=1}^{N} M_i - N \hat{\mu}}{\hat{\sigma} \sqrt{N}}
```

where ``\hat{\mu}`` and ``\hat{\sigma}`` are the mean and standard deviation of ``M_i`` under the null. A supremum of ``LM_i(\tau)`` is not ``\chi^2(r)`` and a pool of ``N`` of them is not ``\chi^2(Nr)``, so both moments are simulated rather than tabulated.

!!! note "Technical Note"
    The null reference draws `nsim` independent ``N(0,1)`` series of length ``T``, projects them off ``\hat{F}``, and runs them through the same statistic path, so it conditions on the estimated factors and on the trimmed grid actually in use. The p-value is the Monte Carlo upper tail of ``\sum_i M_i`` under resampling from that pool, which accommodates the right skewness a normal approximation to the sum would miss. `nsim`, `nboot`, and `seed` control the simulation; the fixed default seed makes the p-value reproducible across calls on the same data.

```@example test_breaks
report(factor_break_test(X_stable, 3; method=:breitung_eickmeier))
```

```@example test_breaks
report(factor_break_test(X_break, 3; method=:breitung_eickmeier))
```

On the stable panel the pooled statistic is 0.4886 --- half a standard deviation above the simulated null mean --- with a p-value of 0.3038. On the broken panel it is 280.9414, the p-value hits 0.0005, and the break is dated at observation 100, the true date. That 0.0005 is a floor, not a measurement: with `nboot = 2000` resampling draws the smallest attainable p-value is ``1/2001``. Because ``Z`` is standardized by the cross-section, the statistic is comparable across panels of different width, unlike the raw pooled sum.

The `FactorModel` dispatch reuses factors that have already been estimated:

```@example test_breaks
fm = estimate_factors(X_break, 3)
be = factor_break_test(fm; method=:breitung_eickmeier)
(statistic = round(be.statistic, digits=3),
 pvalue = round(be.pvalue, digits=4),
 break_date = be.break_date)
```

The dispatch reads `fm.X` and `fm.r` off the model and returns the same statistic, 280.941 at observation 100, as the matrix call. Use it whenever a factor model has already been fitted for other purposes, since it avoids re-running the principal-components step.

### Chen-Dolado-Gonzalo (2014)

The Chen-Dolado-Gonzalo test targets a **big break** --- one large enough that the full sample needs more principal components than either sub-sample --- and is the only one of the three that does not require ``r`` as an input. A big break in ``\Lambda`` inflates the apparent number of factors, and the extra estimated factors are mixtures of the pre- and post-break factor spaces. That mixing is the signal. Regressing the first estimated factor on the remaining ones,

```math
\hat{F}_{1t} = c + \beta' \hat{F}_{2:r,t} + u_t
```

gives coefficients that are constant under ``H_0`` --- and zero, since principal components are orthogonal over the full sample --- but shift at the break date under ``H_1``. The test is a sup-LM test for instability in that regression:

```math
\text{CDG} = \sup_{\pi \in [0.15,\, 0.85]} \frac{S(\tau)' \hat{\Omega}^{-1} S(\tau)}{\pi (1 - \pi)}, \qquad \pi = \tau / T
```

where:
- ``S(\tau) = T^{-1/2} \sum_{t \leq \tau} Z_t \hat{u}_t`` is the partial sum of the regression scores, with ``Z_t = (1, \hat{F}_{2:r,t}')'``
- ``\hat{u}_t`` are the full-sample regression residuals
- ``\hat{\Omega}`` is a Newey-West HAC estimate of the long-run variance of ``Z_t u_t``, with the Newey-West (1994) automatic bandwidth

The p-value comes from the Andrews (1993)/Hansen (1997) sup-Wald tables with ``p = r`` parameters under test --- the intercept plus ``r - 1`` slopes --- the same machinery `andrews_test` uses. The HAC variance is not optional: estimated factors are serially correlated, and a homoskedastic-i.i.d. variance makes the same statistic reject far too often.

Omitting ``r`` selects it by the Bai-Ng (2002) IC2 criterion over ``r \leq \min(\lfloor\sqrt{\min(T, N)}\rfloor, 10)``:

```@example test_breaks
report(factor_break_test(X_break; method=:chen_dolado_gonzalo))
```

IC2 reads 7 factors off the broken panel --- more than the 3 that generated it, which is exactly the inflation the test exploits --- and the sup-LM statistic of 23.4356 rejects at 5% with the break dated at observation 100. On the stable panel the same call selects 5 factors and returns 10.0023 with a p-value of 0.3620.

Supplying ``r`` is not automatically the stronger choice:

```@example test_breaks
cdg_r3 = factor_break_test(X_break, 3; method=:chen_dolado_gonzalo)
(statistic = round(cdg_r3.statistic, digits=3),
 pvalue = round(cdg_r3.pvalue, digits=4),
 break_date = cdg_r3.break_date)
```

Fixing ``r`` at 3 --- the number of factors that generated the data --- drops the statistic to 7.418 and the p-value to 0.3875, and the test no longer rejects: the extra factor directions the break creates are precisely what was discarded. Neither call dominates, and which one wins depends on how much of the cross-section moves. In simulation on ``T = 200``, ``N = 60``, ``r = 3`` panels, a sign flip in half the loadings is found 69% of the time at 5% with ``r = 3`` supplied against 41% by the IC2 dispatch, while a flip in only a quarter of them reverses the ranking, 15% against 46%.

!!! warning "A break that flips every loading is not identified"
    If every loading changes sign at ``\tau``, then ``X = \tilde{F}\Lambda'`` with ``\tilde{F}_t = F_t`` before the break and ``-F_t`` after fits the same data with *stable* loadings. No test on this page rejects that alternative, and none should: in simulation all three reject it at roughly their nominal size (0.035, 0.040, and 0.000 at the 5% level). Loading breaks are identified only relative to a fixed normalization of the factors.

### Choosing a method

| Feature needed | Recommended | Why |
|----------------|-------------|-----|
| Reject/fail decision, ``r`` known | `:breitung_eickmeier` | Size 0.017-0.043, power 1.00 |
| Sharpest break date | `:breitung_eickmeier` or `:han_inoue` | Dates within 0.1% of ``T`` |
| ``r`` unknown | `:chen_dolado_gonzalo` | Only method not needing ``r`` |
| Guarding against false positives | `:han_inoue` | Conservative, rejects only large breaks |

Simulated size at the 5% level with stable loadings, 300 replications per configuration, runs 0.017 to 0.043 for Breitung-Eickmeier and 0.013 to 0.063 for Chen-Dolado-Gonzalo across ``(T, N, r)`` from ``(35, 8, 2)`` to ``(300, 120, 3)``. Against a sign flip in half the loadings at ``T/2``, Breitung-Eickmeier and Han-Inoue reject on every replication and date the break to within 0.1% of ``T``; Chen-Dolado-Gonzalo rejects 69% of the time. Highly persistent factors are the one regime where the size guarantees weaken --- with AR(0.9) factors Chen-Dolado-Gonzalo rejects 11.3% of stable panels and Breitung-Eickmeier turns very conservative at 0.7%.

### Options and return values

`factor_break_test` takes `method`, defaulting to `:breitung_eickmeier` for the two-argument form `factor_break_test(X, r)` and to `:chen_dolado_gonzalo` for the one-argument form `factor_break_test(X)`. Passing `:breitung_eickmeier` or `:han_inoue` without `r` raises an `ArgumentError`. All three methods require at least 30 time periods and standardize the panel internally, so no pre-scaling is needed. Breitung-Eickmeier and Han-Inoue need ``r`` as an input: determine it first with `ic_criteria(X, r_max)`, which reports the Bai-Ng (2002) criteria, and see [Factor Models](@ref factor_page).

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:breitung_eickmeier` or `:chen_dolado_gonzalo` | Test method, depending on the dispatch |
| `nsim` | `Int` | `0` | Null-reference draws; `0` selects ``\min(\max(100N, 2000), 20000)`` |
| `nboot` | `Int` | `2000` | Resampling draws behind the pooled p-value |
| `seed` | `Integer` | `20110711` | Seed for the null reference |

`nsim`, `nboot`, and `seed` tune the simulated reference of `:breitung_eickmeier` and are ignored by the other two methods.

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | Test statistic |
| `pvalue` | `T` | Monte Carlo p-value for `:breitung_eickmeier`, Hansen (1997) tables otherwise |
| `break_date` | `Union{Int, Nothing}` | Estimated break index, `nothing` when the trimmed range is empty |
| `method` | `Symbol` | Method used |
| `n_factors` | `Int` | ``r``, or the IC2 selection when Chen-Dolado-Gonzalo is called without it |
| `nobs` | `Int` | Number of time periods ``T`` |
| `n_vars` | `Int` | Number of cross-section units ``N`` |

---

## Gregory-Hansen Cointegration Test

The Gregory-Hansen test (Gregory & Hansen 1996) extends residual-based cointegration testing by allowing one structural break in the cointegrating relationship. Standard tests lose power when the long-run relationship shifts at an unknown date, so what looks like no cointegration may be cointegration with a regime change.

The cointegrating regression is estimated at every candidate break date and residual-based statistics are computed from each fit. The null is no cointegration; the alternative is cointegration with a single break.

Three models control the form of the break:

**Level shift** (`:C`):

```math
y_{1t} = \mu_1 + \mu_2 D_t + \alpha' y_{2t} + e_t
```

**Level + trend shift** (`:CT`):

```math
y_{1t} = \mu_1 + \mu_2 D_t + \beta t + \alpha' y_{2t} + e_t
```

**Regime shift** (`:CS`):

```math
y_{1t} = \mu_1 + \mu_2 D_t + \alpha_1' y_{2t} + \alpha_2' y_{2t} D_t + e_t
```

where:
- ``y_{1t}`` is the dependent variable (the first column of `Y`)
- ``y_{2t}`` is the ``m \times 1`` vector of regressors (the remaining columns)
- ``D_t = \mathbf{1}(t > T_B)`` is the break dummy
- ``T_B`` is the break date, searched over the trimmed range

Three statistics are computed from the residuals ``\hat{e}_t`` at each candidate date, and each takes its own minimum:

- **ADF\***: smallest ADF t-statistic on the residuals across break dates
- **Zt\***: smallest Phillips-Perron ``Z_t`` statistic
- **Za\***: smallest Phillips-Perron ``Z_\alpha`` statistic

ADF\* is the statistic normally reported. ADF\* and Zt\* share one critical-value table; Za\* has its own, because it is a coefficient-based rather than a t-based statistic.

```@example test_breaks
Random.seed!(105)
x_c = cumsum(randn(250))                         # I(1) regressor
y_c = vcat(2.0 .+ 1.0 .* x_c[1:120],             # intercept jumps 2.0 -> 5.0 at t = 120
           5.0 .+ 1.0 .* x_c[121:end]) + 0.3 * randn(250)

report(gregory_hansen_test(hcat(y_c, x_c); model=:C))
```

ADF\* of ``-16.70`` clears the 1% critical value of ``-5.13`` by a wide margin, so the null of no cointegration is rejected and the two series share a long-run relationship once the level shift is allowed. ADF\* and Zt\* both date the break at observation 121, one period after the true shift at 120, while Za\* --- built on the coefficient rather than the t-ratio --- settles on 119. Disagreement of one or two observations across the three statistics is normal and is a useful informal measure of how sharply the break is identified.

When the series genuinely do not cointegrate, the test says so:

```@example test_breaks
Random.seed!(106)
x_i = cumsum(randn(250))                         # two independent random walks
y_i = cumsum(randn(250))
g_i = gregory_hansen_test(hcat(y_i, x_i); model=:C)
(adf_statistic = round(g_i.adf_statistic, digits=3),
 adf_pvalue = round(g_i.adf_pvalue, digits=3),
 adf_break = g_i.adf_break)
```

ADF\* of ``-4.342`` falls short of the 10% critical value of ``-4.34`` by a hair, and the reported p-value of 0.1 is the flat value assigned to any statistic above the 10% threshold rather than an interpolated number. The estimated break at observation 46 is meaningless under a non-rejection, exactly as with the Andrews test. Two independent random walks will always produce *some* minimizing break date; only the statistic decides whether it means anything.

### Interpreting the outcome

**Reject** ``H_0`` (p-value < 0.05): the series cointegrate once a break is allowed, and the estimated break date says when the relationship shifted. **Fail to reject**: no cointegration, even permitting a regime change. The natural use of the test is diagnostic --- when Johansen or Engle-Granger finds nothing but theory insists on a long-run relationship, Gregory-Hansen checks whether a single regime shift explains the discrepancy.

### Options

| Keyword | Type | Default | Description |
|----------|------|---------|-------------|
| `model` | `Symbol` | `:C` | Break model: `:C` (level shift), `:CT` (level + trend), `:CS` (regime shift) |
| `lags` | `Union{Int,Symbol}` | `:aic` | Augmenting lags for the residual ADF, or `:aic`/`:bic` |
| `max_lags` | `Union{Int,Nothing}` | `nothing` | Cap for IC selection, default ``\lfloor 12 (T/100)^{1/4} \rfloor`` |
| `trim` | `Real` | `0.15` | Trimming fraction for the break search |

At least 50 observations and two columns are required.

### Return values

| Field | Type | Description |
|-------|------|-------------|
| `adf_statistic` | `T` | ADF\* statistic (minimum ADF over break dates) |
| `adf_pvalue` | `T` | P-value for ADF\* |
| `zt_statistic` | `T` | Zt\* Phillips-Perron statistic |
| `zt_pvalue` | `T` | P-value for Zt\* |
| `za_statistic` | `T` | Za\* Phillips-Perron statistic |
| `za_pvalue` | `T` | P-value for Za\* |
| `adf_break` | `Int` | Break date minimizing ADF\* |
| `zt_break` | `Int` | Break date minimizing Zt\* |
| `za_break` | `Int` | Break date minimizing Za\* |
| `model` | `Symbol` | Break model used |
| `n_regressors` | `Int` | Number of regressors ``m`` |
| `adf_critical_values` | `Dict{Int,T}` | Critical values for ADF\* and Zt\* at 1%, 5%, 10% |
| `za_critical_values` | `Dict{Int,T}` | Critical values for Za\* at 1%, 5%, 10% |
| `nobs` | `Int` | Number of observations |

!!! note "Technical Note"
    Critical values are tabulated by break model and by the number of regressors ``m = 1, \ldots, 4``; a regression with more than four regressors reuses the ``m = 4`` column, which makes the test conservative. P-values interpolate between the 1%, 5%, and 10% values, are floored at 0.001, and are reported as a flat 0.20 for any statistic above the 10% critical value.

---

## Complete Example

Real GDP growth is the canonical testing ground for parameter instability. This workflow fits an AR(1) to quarterly US real GDP growth from FRED-QD (McCracken & Ng 2021), tests it for one unknown break with Andrews, then lets Bai-Perron search for several.

```@example test_breaks
fred_q = load_example(:fred_qd)
g = filter(isfinite, to_vector(apply_tcode(fred_q[:, ["GDPC1"]])))   # log growth of real GDP

X_ar = hcat(ones(length(g) - 1), g[1:end-1])     # AR(1): intercept + one lag
y_ar = g[2:end]

report(andrews_test(y_ar, X_ar; test=:supwald))
```

The sup-Wald statistic of 18.19 exceeds the 1% critical value of 14.72, so the AR(1) coefficients are not constant over the 265 usable quarters, and the p-value hits the 0.005 floor. The peak sits at observation 200, roughly three-quarters of the way through the post-1959 sample. Andrews stops there: it tests for one break and reports where the evidence is strongest, without ruling out more.

```@example test_breaks
bp_gdp = bai_perron_test(y_ar, X_ar; max_breaks=5)
report(bp_gdp)
```

Bai-Perron confirms a single break and dates it at observation 201 with a 95% confidence interval of ``[193, 209]`` --- four years wide on quarterly data, which is about as sharp as break dating gets in macroeconomic samples. All three selection devices agree: sup-F(1) is 12.88 with a p-value of 0.026, the sequential test sup-F(2|1) falls to 2.74 with a p-value of 0.784, and BIC and LWZ both bottom out at one break. Agreement of the sequential rule with both information criteria is the strongest evidence the procedure can deliver.

```@example test_breaks
[(regime = j,
  intercept = round(c[1], digits=4),
  ar1 = round(c[2], digits=3),
  se_ar1 = round(s[2], digits=3))
 for (j, (c, s)) in enumerate(zip(bp_gdp.regime_coefs, bp_gdp.regime_ses))]
```

The economics is in the AR(1) coefficient. Before the break, growth is positively autocorrelated at 0.301 with a standard error of 0.067 --- a strong quarter tends to be followed by another. After it, the coefficient turns to ``-0.283`` with a standard error of 0.122, so growth now reverses itself from quarter to quarter. Both estimates are more than two standard errors from zero and of opposite sign, which is a change in the character of the process, not a change in its scale: forecasting rules and impulse responses estimated on the pooled sample describe neither regime.

---

## Common Pitfalls

1. **Reading a break date from a test that does not reject.** Every test on this page reports the date that maximizes its statistic, and that date always exists --- even in i.i.d. data with no break at all. The Andrews example above returns `break_index = 96` for a series generated with constant coefficients. The date is only interpretable once the null has been rejected.

2. **Trimming too aggressively or too leniently.** `trimming` controls how much of the sample is excluded from the break search. The default ``\pi = 0.15`` leaves candidate dates in ``[0.15T, 0.85T]``. Setting it near 0.05 admits splits in which one sub-sample has too few observations to estimate ``\beta`` reliably; setting it near 0.30 excludes breaks in the first or last third of the sample entirely. The Bai-Perron critical-value tables are only defined on the ``\{0.05, 0.10, 0.15, 0.20, 0.25\}`` grid, so off-grid values snap to the nearest with a warning.

3. **Confusing the power properties of `:supwald` and `:expwald`.** The supremum has optimal power against one sharp break at an unknown date. The exponential and mean functionals (Andrews & Ploberger 1994) average information over all candidate dates and do better against small or gradual shifts. When the shape of the break is unknown, run all three: agreement, as in the example above, means the result is not an artefact of the weighting.

4. **Bai-Perron silently reducing `max_breaks`.** Each regime needs at least ``\max(k+1, \lceil \pi T \rceil)`` observations, so the feasible maximum is ``\lfloor T/h \rfloor - 1``. With many regressors or a short sample the requested `max_breaks` is cut without an error. Check `length(result.supf_stats)` to see how many breaks were actually searched.

5. **Reading a Han-Inoue non-rejection as evidence of stability.** Its statistic averages the individual Wald statistics, which concentrates near ``r`` under the null, while the reference is the sup-Wald critical value for ``k = r``. The test therefore rejects only for large breaks: in simulation its rejection rate on stable panels is 0.000 at the 5% level for every ``(T, N, r)`` tried, and its p-values on those panels sit between 0.20 and 0.35. Take the reject/fail decision from `:breitung_eickmeier` when ``r`` is known.

6. **Expecting power from a narrow cross-section.** Size holds in small panels --- Breitung-Eickmeier rejects 4.3% of stable panels at ``T = 35``, ``N = 8`` --- but power comes from pooling across units, so a break confined to a few series in a narrow panel goes undetected. For small panels, apply `andrews_test` or `bai_perron_test` to the individual equations instead.

7. **Choosing the Gregory-Hansen model by fit rather than by theory.** The `:CS` regime shift lets every slope change at the break, which makes it the most flexible alternative and therefore the least powerful: it estimates ``m`` extra coefficients under ``H_1``. Start with `:C` and move to `:CT` or `:CS` only when theory says the trend or the slopes should shift.

---

## References

- Andrews, D. W. K. (1993). Tests for parameter instability and structural change with unknown change point. *Econometrica*, 61(4), 821-856. [DOI](https://doi.org/10.2307/2951764)

- Andrews, D. W. K., & Ploberger, W. (1994). Optimal tests when a nuisance parameter is present only under the alternative. *Econometrica*, 62(6), 1383-1414. [DOI](https://doi.org/10.2307/2951753)

- Bai, J. (1997). Estimation of a change point in multiple regression models. *Review of Economics and Statistics*, 79(4), 551-563. [DOI](https://doi.org/10.1162/003465397557132)

- Bai, J., & Ng, S. (2002). Determining the number of factors in approximate factor models. *Econometrica*, 70(1), 191-221. [DOI](https://doi.org/10.1111/1468-0262.00273)

- Bai, J., & Perron, P. (1998). Estimating and testing linear models with multiple structural changes. *Econometrica*, 66(1), 47-78. [DOI](https://doi.org/10.2307/2998540)

- Bai, J., & Perron, P. (2003). Computation and analysis of multiple structural change models. *Journal of Applied Econometrics*, 18(1), 1-22. [DOI](https://doi.org/10.1002/jae.659)

- Breitung, J., & Eickmeier, S. (2011). Testing for structural breaks in dynamic factor models. *Journal of Econometrics*, 163(1), 71-84. [DOI](https://doi.org/10.1016/j.jeconom.2010.11.008)

- Chen, L., Dolado, J. J., & Gonzalo, J. (2014). Detecting big structural breaks in large factor models. *Journal of Econometrics*, 180(1), 30-48. [DOI](https://doi.org/10.1016/j.jeconom.2014.01.006)

- Gregory, A. W., & Hansen, B. E. (1996). Residual-based tests for cointegration in models with regime shifts. *Journal of Econometrics*, 70(1), 99-126. [DOI](https://doi.org/10.1016/0304-4076(69)41685-7)

- Han, X., & Inoue, A. (2015). Tests for parameter instability in dynamic factor models. *Econometric Theory*, 31(5), 1117-1152. [DOI](https://doi.org/10.1017/S0266466614000486)

- Hansen, B. E. (1997). Approximate asymptotic p values for structural-change tests. *Journal of Business & Economic Statistics*, 15(1), 60-67. [DOI](https://doi.org/10.1080/07350015.1997.10524687)

- Liu, J., Wu, S., & Zidek, J. V. (1997). On segmented multivariate regression. *Statistica Sinica*, 7(2), 497-525.

- McCracken, M. W., & Ng, S. (2021). FRED-QD: A quarterly database for macroeconomic research. *Federal Reserve Bank of St. Louis Review*, 103(1), 1-44. [DOI](https://doi.org/10.20955/r.103.1-44)

- Newey, W. K., & West, K. D. (1994). Automatic lag selection in covariance matrix estimation. *Review of Economic Studies*, 61(4), 631-653. [DOI](https://doi.org/10.2307/2297912)
