# [Unit Root & Cointegration](@id tests_unitroot_page)

Pre-estimation stationarity analysis determines whether a time series is stationary (I(0)) or contains a unit root (I(1)). This distinction drives the choice between a VAR in levels, a VAR in first differences, and a VECM. MacroEconometricModels.jl provides five unit root tests, the Johansen system cointegration test, and two batch utilities that screen many series at once.

For the full test battery and the tables that route a question to a test, see [Hypothesis Tests](@ref tests_page). For tests with improved power under smooth breaks, GLS detrending, seasonal frequencies, and explosive alternatives, see [Advanced Unit Root Tests](@ref tests_unitroot_advanced_page). For single-equation cointegration tests on a fitted regression, see [Residual-Based Cointegration Tests](@ref tests_cointegration_page).

ADF and KPSS are complementary: ADF tests the null of a unit root, KPSS the null of stationarity. Running both is stronger than either alone, because the two tests fail in opposite directions.

- **ADF, PP, Ng-Perron**: null hypothesis is a unit root
- **KPSS**: null hypothesis is stationarity (reverses the burden of proof)
- **Zivot-Andrews**: unit root test robust to a single endogenous structural break
- **Johansen**: tests for cointegrating relationships among several I(1) series

```@setup test_ur
using MacroEconometricModels
fred   = load_example(:fred_md)
cpi    = filter(isfinite, fred[:, "CPIAUCSL"])
unrate = filter(isfinite, fred[:, "UNRATE"])
qd     = load_example(:fred_qd)
rates  = to_matrix(qd[:, ["GS10", "TB3MS"]])
rates  = rates[all.(isfinite, eachrow(rates)), :]
```

## Quick Start

**Recipe 1: Confirm a unit root with ADF and KPSS**

```@example test_ur
# The consumer price level: ADF should fail to reject, KPSS should reject
report(adf_test(cpi; lags=:aic, regression=:constant))
report(kpss_test(cpi; regression=:constant))
```

**Recipe 2: Confirm stationarity with the same pair**

```@example test_ur
# The unemployment rate: ADF rejects the unit root, KPSS keeps stationarity
report(adf_test(unrate; lags=:aic, regression=:constant))
report(kpss_test(unrate; regression=:constant))
```

**Recipe 3: Batch summary of several tests on one series**

```@example test_ur
summary = unit_root_summary(cpi; tests=[:adf, :kpss, :pp])
summary.conclusion
```

**Recipe 4: Screen every column of a data matrix**

```@example test_ur
Y = to_matrix(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS", "UNRATE", "M2SL"]])
Y = Y[all.(isfinite, eachrow(Y)), :]

results = test_all_variables(Y; test=:adf)
[(stat = round(r.statistic, digits=3), p = round(r.pvalue, digits=3)) for r in results]
```

**Recipe 5: Johansen cointegration rank**

```@example test_ur
# Ten-year and three-month Treasury yields: does the term spread mean-revert?
report(johansen_test(rates, 2; deterministic=:constant))
```

---

## Augmented Dickey-Fuller Test

The Augmented Dickey-Fuller (ADF) test (Dickey & Fuller 1979) is the most widely used unit root test in applied macroeconometrics. It asks whether a series contains a stochastic trend by testing the coefficient on the lagged level in a first-difference regression.

The ADF regression augments the basic Dickey-Fuller test with lagged differences to absorb serial correlation:

```math
\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{j=1}^{p} \delta_j \Delta y_{t-j} + \varepsilon_t
```

where:
- ``\gamma = \rho - 1`` is the coefficient of interest (``\rho`` is the AR(1) root)
- ``\alpha`` is an intercept, included when `regression=:constant` or `:trend`
- ``\beta t`` is a linear trend, included when `regression=:trend`
- ``p`` lagged differences absorb serial correlation in ``\varepsilon_t``

The hypotheses are ``H_0: \gamma = 0`` (unit root) against ``H_1: \gamma < 0`` (stationary). The statistic is the ``t``-ratio ``\tau = \hat\gamma / \text{se}(\hat\gamma)``, which follows a non-standard distribution that depends on the deterministic specification.

!!! note "Technical Note"
    Critical values come from the MacKinnon response surface ``c_1 + c_2/T + c_3/T^2 + c_4 (p/T) + c_5 (p/T)^2``, so they adjust for both sample size and the selected lag order. The reported p-value is the **asymptotic** MacKinnon (1996) surface ``p = \Phi(P(\tau))`` and carries no finite-sample correction. Near a critical value the two can therefore disagree by a little; the critical values are the finer instrument in short samples.

```@example test_ur
# The consumer price level, with the lag order chosen by AIC
result = adf_test(cpi; lags=:aic, regression=:constant)
report(result)
```

The statistic ``\tau = 2.151`` sits far above the 5% critical value of ``-2.847``, and the p-value rounds to 1.000: there is no evidence at all against a unit root in the price level. AIC selects 15 augmenting lags out of a maximum of 20, leaving 787 usable observations. This is the expected result — a price *level* is I(1), and inflation (its first difference) is the object that may be stationary. Compare the unemployment rate, which behaves differently:

```@example test_ur
(τ = round(adf_test(unrate; lags=:aic, regression=:constant).statistic, digits=3),
 p = round(adf_test(unrate; lags=:aic, regression=:constant).pvalue, digits=4))
```

Here ``\tau = -3.563`` with ``p = 0.0065``, rejecting the unit root at the 1% level.

### Options

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `lags` | `Union{Int,Symbol}` | `:aic` | Number of augmenting lags, or `:aic`/`:bic`/`:hqic` for automatic selection |
| `max_lags` | `Union{Int,Nothing}` | `nothing` | Ceiling for automatic selection (defaults to ``\lfloor 12(T/100)^{1/4} \rfloor``) |
| `regression` | `Symbol` | `:constant` | Deterministic terms: `:none`, `:constant`, or `:trend` |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | ADF test statistic (``\tau``-ratio on ``\hat\gamma``) |
| `pvalue` | `T` | Asymptotic MacKinnon (1996) p-value |
| `lags` | `Int` | Number of augmenting lags used |
| `regression` | `Symbol` | Deterministic specification (`:none`, `:constant`, `:trend`) |
| `critical_values` | `Dict{Int,T}` | Critical values keyed by significance level (`1`, `5`, `10`) |
| `nobs` | `Int` | Number of observations in the test regression (``T - 1 - p``) |

### Interpretation

**Reject** ``H_0`` (p-value < 0.05): evidence against a unit root; the series appears stationary. **Fail to reject** (p-value > 0.05): the unit root null survives, and the series needs differencing or a VECM. The ADF test has notoriously low power against near-unit-root alternatives, so a failure to reject is weak evidence on its own — confirm it with KPSS.

---

## KPSS Stationarity Test

The KPSS test (Kwiatkowski, Phillips, Schmidt & Shin 1992) reverses the ADF hypotheses. By testing the null of stationarity it puts the burden of proof on rejecting stationarity rather than on rejecting the unit root, which is what makes the ADF-KPSS pair informative.

KPSS decomposes the series into a deterministic trend, a random walk, and a stationary error:

```math
y_t = \xi t + r_t + \varepsilon_t, \qquad r_t = r_{t-1} + u_t
```

where:
- ``\xi t`` is a deterministic trend (``\xi = 0`` under `regression=:constant`)
- ``r_t`` is a random walk with innovation ``u_t \sim (0, \sigma_u^2)``
- ``\varepsilon_t`` is a stationary error

Under ``H_0: \sigma_u^2 = 0`` the random walk vanishes and the series is stationary. The statistic is

```math
\text{KPSS} = \frac{\sum_{t=1}^{T} S_t^2}{T^2 \hat{\sigma}^2_{LR}}
```

where:
- ``S_t = \sum_{s=1}^{t} \hat{e}_s`` are partial sums of the residuals from regressing ``y_t`` on the deterministic terms
- ``\hat{\sigma}^2_{LR}`` is the Bartlett-kernel long-run variance of those residuals

Large partial sums indicate an accumulating (random-walk) component, so KPSS is **right-tailed**: reject stationarity for large statistics.

!!! note "Bandwidth selection"
    With `bandwidth=:auto` the Bartlett lag truncation is the Andrews (1991) AR(1) plug-in rule ``\lfloor 1.1447 \, (4 \hat\rho^2 T / (1-\hat\rho^2)^2)^{1/3} \rfloor``, where ``\hat\rho`` is the first-order autocorrelation of the residuals. Strongly persistent residuals therefore draw a long bandwidth, which inflates ``\hat{\sigma}^2_{LR}`` and shrinks the statistic — the mechanism by which KPSS loses power on near-unit-root data.

```@example test_ur
# The unemployment rate: a series that should pass a stationarity test
result = kpss_test(unrate; regression=:constant)
report(result)

# The price level, for contrast
report(kpss_test(cpi; regression=:constant))
```

For the unemployment rate the statistic is 0.098, far below the 10% critical value of 0.347, so stationarity survives comfortably — the same verdict the ADF test reached from the opposite direction. For the price level the statistic is 0.482, above the 5% critical value of 0.463, so stationarity is rejected at 5%. Read together with Recipe 1, the price level lands squarely in the "unit root" cell of the decision matrix below and the unemployment rate in the "stationary" cell.

### Options

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `regression` | `Symbol` | `:constant` | Stationarity type: `:constant` (level) or `:trend` (trend) |
| `bandwidth` | `Union{Int,Symbol}` | `:auto` | Bartlett lag truncation, or `:auto` for the Andrews (1991) plug-in rule |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | KPSS test statistic |
| `pvalue` | `T` | Interpolated p-value, clipped to ``[0.01, 0.50]`` |
| `regression` | `Symbol` | Stationarity type (`:constant` or `:trend`) |
| `critical_values` | `Dict{Int,T}` | Critical values keyed by significance level (`1`, `5`, `10`) |
| `bandwidth` | `Int` | Bartlett lag truncation used |
| `nobs` | `Int` | Number of observations (the full series length) |

### Combining ADF and KPSS

Running both tests resolves the ambiguity inherent in either one:

| ADF Result | KPSS Result | Conclusion |
|------------|-------------|------------|
| Reject (stationary) | Fail to reject (stationary) | **Stationary** |
| Fail to reject (unit root) | Reject (unit root) | **Unit root** |
| Reject | Reject | Conflicting — suspect a structural break |
| Fail to reject | Fail to reject | Inconclusive — the sample is uninformative |

When both tests reject, a structural break that distorts both null distributions is the usual explanation, and the Zivot-Andrews test addresses that case directly. When neither rejects, no amount of re-specification helps: the sample simply does not separate the hypotheses.

---

## Phillips-Perron Test

The Phillips-Perron (PP) test (Phillips & Perron 1988) is the non-parametric counterpart to ADF. Instead of adding lagged differences to soak up serial correlation, it estimates the plain Dickey-Fuller regression

```math
y_t = \alpha + \rho y_{t-1} + u_t
```

and corrects the resulting ``t``-ratio for serial correlation and heteroskedasticity after the fact. The corrected statistic is

```math
Z_t = \sqrt{\frac{\hat{\gamma}_0}{\hat{\lambda}^2}} \, t_\rho \; - \; \frac{\hat{\lambda}^2 - \hat{\gamma}_0}{2 \hat{\lambda}} \cdot \frac{T \cdot \text{se}(\hat{\rho})}{s}
```

where:
- ``\hat{\gamma}_0`` is the short-run variance of the residuals ``\hat u_t``
- ``\hat{\lambda}^2`` is their Bartlett long-run variance and ``\hat\lambda = \sqrt{\hat\lambda^2}``
- ``t_\rho`` is the OLS ``t``-ratio testing ``\rho = 1``
- ``s`` is the standard error of the Dickey-Fuller regression

``Z_t`` shares the ADF null distribution, so the same MacKinnon critical values apply. Its advantage is that no lag order has to be chosen, although bandwidth selection for the long-run variance plays the analogous role.

!!! note "Technical Note"
    The correction term uses the identity ``\text{se}(\hat\rho)/s = 1/\sqrt{S_{ll}}``, where ``S_{ll}`` is the sum of squares of ``y_{t-1}`` after the deterministic terms are projected out of it. Formed that way the correction is invariant to the units of ``y`` — multiplying a series by 100 leaves ``Z_t`` unchanged to machine precision — and stays ``O(1)`` in ``T`` instead of vanishing asymptotically.

```@example test_ur
# The unemployment rate, measured in percent
result = pp_test(unrate; regression=:constant)
report(result)
```

The Andrews rule picks a bandwidth of 2, and ``Z_t = -3.596`` clears the 1% critical value of ``-3.436``, giving ``p = 0.0058``. That is within 0.04 of the ADF statistic for the same series, which is the reassuring case: the parametric lag augmentation and the non-parametric correction agree, so the verdict does not hinge on how serial correlation was handled.

### Options

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `regression` | `Symbol` | `:constant` | Deterministic terms: `:none`, `:constant`, or `:trend` |
| `bandwidth` | `Union{Int,Symbol}` | `:auto` | Bartlett lag truncation, or `:auto` for the Andrews (1991) plug-in rule |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | Phillips-Perron ``Z_t`` statistic |
| `pvalue` | `T` | Asymptotic MacKinnon (1996) p-value |
| `regression` | `Symbol` | Deterministic specification |
| `critical_values` | `Dict{Int,T}` | Critical values keyed by significance level (`1`, `5`, `10`) |
| `bandwidth` | `Int` | Bartlett lag truncation used |
| `nobs` | `Int` | Number of observations in the test regression (``T - 1``) |

---

## Zivot-Andrews Test

The Zivot-Andrews test (Zivot & Andrews 1992) extends the ADF framework to allow one endogenous structural break. Standard unit root tests lose power badly under breaks: a stationary series with a level shift looks like a random walk to ADF. Zivot-Andrews searches every candidate break date and keeps the one giving the strongest evidence against the unit root null, so the reported statistic is a minimum over the trimmed grid.

Three specifications control which deterministic terms may break. All three carry an intercept and a linear trend; they differ only in the break dummies.

**Break in intercept** (`:constant`):

```math
\Delta y_t = \alpha + \beta t + \theta DU_t + \gamma y_{t-1} + \sum_{j=1}^{p} \delta_j \Delta y_{t-j} + \varepsilon_t
```

**Break in trend** (`:trend`): replace ``\theta DU_t`` with ``\phi DT_t``. **Break in both** (`:both`): include both dummies. In each case:

- ``DU_t = \mathbf{1}(t \geq T_B)`` is the level shift dummy
- ``DT_t = (t - T_B + 1) \cdot \mathbf{1}(t \geq T_B)`` is the trend shift dummy
- ``T_B`` is the break date, chosen to minimize the ``t``-statistic on ``\gamma``
- the trimming parameter excludes the first and last `trim` fraction of the sample from the search

!!! note "Innovational and additive outliers"
    `outlier=:io` (the default) puts the break dummies directly in the ADF regression, so the level shift propagates through the dynamics. `outlier=:ao` instead detrends ``y`` on the deterministics and break dummies first, then runs an ADF regression on the residuals with a pulse dummy at ``T_B`` and its lags. Use `:io` when the shift is transmitted gradually and `:ao` when it is a one-off jump in the level.

```@example test_ur
# The price level, allowing a break in both intercept and trend
result = za_test(cpi; regression=:both)
report(result)
```

The minimum ``t``-statistic across all candidate breaks is ``-3.983``, still well above the 5% critical value of ``-5.08``, so ``p = 0.152`` and the unit root survives even with a break allowed. The estimated break sits at observation 667, 83% of the way through the sample. The lesson is that CPI is not a stationary series masquerading as I(1) because of one shift: the unit root is genuine, and the level must be differenced regardless of how the deterministics are specified.

### Options

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `regression` | `Symbol` | `:both` | Break type: `:constant`, `:trend`, or `:both` |
| `trim` | `Real` | `0.15` | Trimming fraction for the break search, in ``(0, 0.5)`` |
| `lags` | `Union{Int,Symbol}` | `:aic` | Augmenting lags, or `:aic`/`:bic` for automatic selection |
| `max_lags` | `Union{Int,Nothing}` | `nothing` | Ceiling for automatic selection (defaults to ``\lfloor 12(T/100)^{1/4} \rfloor``) |
| `outlier` | `Symbol` | `:io` | Break model: `:io` (innovational) or `:ao` (additive) |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | Minimum ``t``-statistic over all candidate break dates |
| `pvalue` | `T` | Interpolated p-value against the Zivot-Andrews table |
| `break_index` | `Int` | Estimated break point (observation index into `y`) |
| `break_fraction` | `T` | Break location as a fraction of the sample |
| `regression` | `Symbol` | Break specification (`:constant`, `:trend`, `:both`) |
| `critical_values` | `Dict{Int,T}` | Zivot-Andrews (1992, Table 4) critical values for this specification |
| `lags` | `Int` | Number of augmenting lags at the selected break |
| `nobs` | `Int` | Number of observations in the test regression (``T - 1 - p``) |

Because the statistic is a minimum over the grid, it is mechanically more negative than a single-date ADF ``t``-ratio, and the critical values are correspondingly further left (``-5.08`` at 5% for `:both`, against ``-2.85`` for ADF). Reading a Zivot-Andrews statistic against ADF critical values is the standard way to manufacture a spurious rejection. The test also assumes **at most one** break under the alternative; with several breaks it loses power, and `lm_unitroot_test` or `adf_2break_test` on the [advanced page](@ref tests_unitroot_advanced_page) are the right tools.

---

## Ng-Perron Tests

The Ng-Perron tests (Ng & Perron 2001) target the size distortions of ADF and PP, which are severe when the errors carry a large negative moving-average root. They apply GLS detrending before computing four modified statistics with better size and power.

GLS detrending quasi-differences the data at a local-to-unity parameter:

```math
\tilde{y}_1 = y_1, \qquad \tilde{y}_t = y_t - \bar{\alpha} \, y_{t-1}, \quad t = 2, \ldots, T
```

where:
- ``\bar{\alpha} = 1 + \bar{c}/T`` is the quasi-differencing coefficient
- ``\bar{c} = -7`` for `regression=:constant` and ``\bar{c} = -13.5`` for `:trend`

The deterministic regressors are quasi-differenced the same way, the coefficients ``\hat\delta`` are estimated from that transformed regression, and the detrended series is ``y_t^d = y_t - Z_t \hat\delta``. All four statistics are built from ``y^d``:

- **MZa** (``MZ_\alpha``): modified Phillips ``Z_\alpha``
- **MZt** (``MZ_t``): modified Phillips ``Z_t``, the most commonly reported of the four
- **MSB**: modified Sargan-Bhargava statistic
- **MPT**: modified point-optimal statistic

MZa, MZt, and MSB reject for **small** values; MPT rejects for **small** values as well, since it measures the residual cost of imposing the unit root.

!!! note "Implementation Detail"
    The long-run variance is the autoregressive spectral density at frequency zero, obtained by fitting an AR(``k``) model to ``\Delta y_t^d`` with ``k = \lfloor 4 (T/100)^{2/9} \rfloor`` and forming ``s^2_{AR} = \hat\sigma^2 / (1 - \sum_j \hat\rho_j)^2``. `dfgls_test` reuses exactly this construction, so the `MZa`/`MZt`/`MSB`/`MPT` fields of a `DFGLSResult` are bit-for-bit identical to those of `ngperron_test` on the same data and specification.

```@example test_ur
# The unemployment rate under GLS detrending
result = ngperron_test(unrate; regression=:constant)
report(result)
```

All four statistics reject: ``MZ_\alpha = -18.51`` against a 5% value of ``-8.1``, ``MZ_t = -3.007`` against ``-1.98``, ``MSB = 0.162`` against ``0.233``, and ``MPT = 1.316`` against ``3.17``. Unanimity across the four is the outcome to look for, because they are different functionals of the same detrended series and disagreement usually signals a badly chosen deterministic specification. Running the same test on the price level gives ``MZ_t = 3.288`` — the wrong side of zero entirely, and no rejection.

### Options

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `regression` | `Symbol` | `:constant` | Deterministic terms: `:constant` or `:trend` |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `MZa` | `T` | Modified Phillips ``Z_\alpha`` statistic |
| `MZt` | `T` | Modified Phillips ``Z_t`` statistic |
| `MSB` | `T` | Modified Sargan-Bhargava statistic |
| `MPT` | `T` | Modified point-optimal statistic |
| `regression` | `Symbol` | Deterministic specification |
| `critical_values` | `Dict{Symbol,Dict{Int,T}}` | Ng-Perron (2001, Table 1) values keyed by statistic (`:MZa`, `:MZt`, `:MSB`, `:MPT`) then by level |
| `nobs` | `Int` | Number of observations (the full series length) |

GLS detrending buys substantially better size than ADF in small samples (``T < 100``), so when ADF is borderline, ``MZ_t`` is the more reliable read. The exception is a large negative MA root in the errors, where GLS-based tests become oversized and plain ADF is safer (Perron & Ng 1996).

---

## Batch Utilities

Two functions turn the individual tests into screening tools.

`unit_root_summary` runs several tests on one series and synthesizes an overall verdict. The verdict is driven by the ADF and KPSS p-values alone, following the decision matrix above; the other tests are reported but do not enter the conclusion.

```@example test_ur
summary = unit_root_summary(cpi; tests=[:adf, :kpss, :pp, :dfgls])

# The individual results stay addressable by test name
(pp = round(summary.results[:pp].statistic, digits=3),
 dfgls = round(summary.results[:dfgls].statistic, digits=3),
 verdict = summary.conclusion)
```

`test_all_variables` applies one test to every column of a matrix and returns a vector of results, which is the fastest way to sort a panel into I(0) and I(1) groups before specifying a VAR.

```@example test_ur
Y = to_matrix(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS", "UNRATE", "M2SL"]])
Y = Y[all.(isfinite, eachrow(Y)), :]

labels = ["INDPRO", "CPIAUCSL", "FEDFUNDS", "UNRATE", "M2SL"]
[(v, round(r.pvalue, digits=3), r.pvalue < 0.05 ? "I(0)" : "I(1)")
 for (v, r) in zip(labels, test_all_variables(Y; test=:adf))]
```

Industrial production (``p = 0.692``), the price level (``p = 1.000``), and the money stock (``p = 1.000``) all fail to reject and enter a VAR in differences; the funds rate (``p = 0.026``) and the unemployment rate (``p = 0.007``) reject and can stay in levels. The money stock's p-value of 1.000 comes from a *positive* ADF statistic of 3.169, the signature of an explosively trending series that needs a trend term or a log transform before any of these tests mean much.

### Options

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `tests` | `Vector{Symbol}` | `[:adf, :kpss, :pp]` | Tests `unit_root_summary` runs: `:adf`, `:kpss`, `:pp`, `:za`, `:ngperron`, `:fourier_adf`, `:dfgls` |
| `regression` | `Symbol` | `:constant` | Deterministic specification forwarded to every test (`:none` becomes `:constant` for tests that reject it) |
| `test` | `Symbol` | `:adf` | Test `test_all_variables` applies: `:adf`, `:kpss`, `:pp`, `:za`, `:ngperron`, `:fourier_adf`, `:dfgls`, `:lm_unitroot` |

The two functions accept different test sets: `:lm_unitroot` works in `test_all_variables` but is silently ignored by `unit_root_summary`. Any further keywords passed to `test_all_variables` are forwarded to the underlying test.

### Return Values

`unit_root_summary` returns a `NamedTuple`:

| Field | Type | Description |
|-------|------|-------------|
| `results` | `Dict{Symbol,AbstractUnitRootTest}` | Individual test results keyed by test name |
| `conclusion` | `String` | Verdict synthesized from the ADF and KPSS p-values |

`test_all_variables` returns a `Vector{AbstractUnitRootTest}` with one entry per column of `Y`, in column order.

---

## Johansen Cointegration Test

The Johansen test (Johansen 1991) asks whether several I(1) series share common stochastic trends — equivalently, whether some linear combination of them is stationary. Cointegration implies a long-run equilibrium that constrains the joint dynamics, and the estimated cointegrating vectors become the error correction terms of a VECM.

Start from a VAR(``p``) in levels and rewrite it in vector error correction form:

```math
\Delta y_t = \Pi y_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta y_{t-i} + u_t
```

where:
- ``\Pi = \alpha \beta'`` is the ``n \times n`` long-run impact matrix
- ``\beta`` holds the cointegrating vectors (the equilibrium relationships)
- ``\alpha`` holds the adjustment coefficients (the speed of return to equilibrium)
- ``\text{rank}(\Pi) = r`` is the number of cointegrating relationships

The procedure tests the rank of ``\Pi`` with two likelihood ratio statistics built from the ordered eigenvalues ``\hat\lambda_1 \geq \cdots \geq \hat\lambda_n`` of a reduced-rank regression.

**Trace test** — ``H_0: \text{rank}(\Pi) \leq r`` against ``H_1: \text{rank}(\Pi) > r``:

```math
\lambda_{\text{trace}}(r) = -T \sum_{i=r+1}^{n} \ln(1 - \hat{\lambda}_i)
```

**Maximum eigenvalue test** — ``H_0: \text{rank}(\Pi) = r`` against ``H_1: \text{rank}(\Pi) = r + 1``:

```math
\lambda_{\max}(r) = -T \ln(1 - \hat{\lambda}_{r+1})
```

!!! note "Deterministic cases and p-values"
    The `deterministic` keyword selects among Johansen's (1995) cases. `:constant` is Case 2: the intercept is restricted to the cointegrating space, so it enters ``\Pi y_{t-1}`` but not the short-run dynamics, ruling out linear trends in levels. `:trend` is Case 4: a linear trend is restricted to the cointegrating space and the constant is unrestricted, allowing linear trends in levels. Critical values are the Osterwald-Lenum (1992) tables for the matching case; p-values are the Doornik (1998) gamma approximation to the MacKinnon-Haug-Michelis (1999) asymptotic distributions. The two sources can disagree marginally near a critical value, and the reported `rank` follows the tabulated critical values.

```@example test_ur
# Ten-year and three-month Treasury yields, two lags in the VECM
result = johansen_test(rates, 2; deterministic=:constant)
report(result)
```

The trace statistic at ``r = 0`` is 30.89 against a 5% critical value of 19.96, so no cointegration is rejected. At ``r \leq 1`` it falls to 3.50 against 9.24 and is not rejected, so the sequence stops and the estimated rank is 1. The maximum eigenvalue test agrees (27.39 against 15.67, then 3.50 against 9.24). One cointegrating vector among two I(1) yields is exactly what the expectations hypothesis of the term structure predicts: the levels wander, but the spread between them does not.

```@example test_ur
# Normalize the cointegrating vector on the long rate
beta = result.eigenvectors[:, 1] ./ result.eigenvectors[1, 1]
(cointegrating_vector = round.(beta, digits=3), adjustment = round.(result.adjustment[:, 1], digits=3))
```

The normalized vector is ``(1, -1.040)``, so the stationary combination is the ten-year yield minus roughly one times the bill rate — the term spread, recovered from the data rather than imposed. The adjustment coefficients ``(0.090, -0.148)`` say that when the spread is above equilibrium the long rate drifts up further while the bill rate falls, with the short end doing most of the correcting.

### Options

The lag order ``p`` is a positional argument giving the number of lags in the VECM representation. Two keywords control the rest:

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `deterministic` | `Symbol` | `:constant` | Deterministic terms: `:none` (Case 1), `:constant` (Case 2), or `:trend` (Case 4) |
| `significance` | `Real` | `0.05` | Level at which the trace sequence selects `rank` (`≤ 0.01`, `≤ 0.05`, else 10%) |

### Return Values

Let ``r`` denote the estimated rank and ``r_{\text{eff}} = \max(r, 1)``.

| Field | Type | Description |
|-------|------|-------------|
| `trace_stats` | `Vector{T}` | Trace statistics, one per rank hypothesis ``r = 0, \ldots, n-1`` |
| `trace_pvalues` | `Vector{T}` | Doornik (1998) p-values for the trace statistics |
| `max_eigen_stats` | `Vector{T}` | Maximum eigenvalue statistics |
| `max_eigen_pvalues` | `Vector{T}` | Doornik (1998) p-values for the maximum eigenvalue statistics |
| `rank` | `Int` | Estimated cointegration rank |
| `eigenvectors` | `Matrix{T}` | ``n \times r_{\text{eff}}`` cointegrating vectors ``\beta`` (columns) |
| `adjustment` | `Matrix{T}` | ``n \times r_{\text{eff}}`` adjustment matrix ``\alpha`` |
| `eigenvalues` | `Vector{T}` | The ``n`` ordered eigenvalues of the reduced-rank regression |
| `critical_values_trace` | `Matrix{T}` | ``n \times 3`` trace critical values, columns ordered **10%, 5%, 1%** |
| `critical_values_max` | `Matrix{T}` | ``n \times 3`` maximum eigenvalue critical values, same column order |
| `deterministic` | `Symbol` | Deterministic specification |
| `lags` | `Int` | Number of VECM lags |
| `nobs` | `Int` | Effective number of observations (``T - p``) |

The critical-value columns run **10%, 5%, 1%**, which is the reverse of the `Dict`-keyed critical values returned by the univariate tests. Indexing `critical_values_trace[:, 1]` for the 1% column is a common and silent error.

### Interpretation

The sequential procedure starts at ``r = 0`` and stops at the first non-rejection:

1. Test ``H_0: r = 0`` (no cointegration). If rejected, continue.
2. Test ``H_0: r \leq 1``. If rejected, continue.
3. Continue until ``H_0: r \leq k`` is not rejected; that ``k`` is the rank.

A rank of ``0`` means no cointegration — difference the data and fit a VAR. A rank equal to ``n`` means ``\Pi`` has full rank, which is only possible if the series were stationary to begin with, and signals that the integration-order pretests were wrong or the deterministic case is misspecified. Anything strictly between is genuine cointegration: estimate `estimate_vecm(Y, p; rank=r)` to impose the long-run constraints. See the [VECM page](@ref vecm_page) for estimation and the [Residual-Based Cointegration Tests](@ref tests_cointegration_page) page for the single-equation alternatives.

---

## Complete Example

The full pre-estimation pipeline: screen each series, confirm the borderline ones with complementary tests, check for breaks, and test the I(1) group for cointegration.

```@example test_ur
# ── Step 1: Screen every candidate series with ADF ───────────────
panel = ["INDPRO", "CPIAUCSL", "FEDFUNDS", "UNRATE", "M2SL"]
Y = to_matrix(fred[:, panel])
Y = Y[all.(isfinite, eachrow(Y)), :]

screen = test_all_variables(Y; test=:adf)
[(v, round(r.pvalue, digits=3)) for (v, r) in zip(panel, screen)]
```

```@example test_ur
# ── Step 2: Confirm the two verdicts with KPSS ───────────────────
report(kpss_test(cpi; regression=:constant))
report(kpss_test(unrate; regression=:constant))
```

```@example test_ur
# ── Step 3: Rule out a structural break as the cause ─────────────
za = za_test(cpi; regression=:both)
(statistic = round(za.statistic, digits=3), cv5 = za.critical_values[5],
 break_fraction = round(za.break_fraction, digits=3))
```

```@example test_ur
# ── Step 4: GLS-detrended robustness check ───────────────────────
np = ngperron_test(cpi; regression=:constant)
(MZt = round(np.MZt, digits=3), cv5 = np.critical_values[:MZt][5])
```

```@example test_ur
# ── Step 5: Test the I(1) pair for cointegration ─────────────────
joh = johansen_test(rates, 2; deterministic=:constant)

joh.rank == 0 ? "No cointegration — VAR in first differences" :
joh.rank == size(rates, 2) ? "Full rank — the series are stationary, use a VAR in levels" :
"Rank $(joh.rank) — estimate a VECM with estimate_vecm(rates, 2; rank=$(joh.rank))"
```

Steps 1 and 2 put CPI and M2 in the I(1) group and the unemployment rate in the I(0) group. Step 3 shows the CPI unit root is not an artifact of a single break: even with the break allowed, ``-3.983`` falls short of ``-5.08``. Step 4 corroborates with GLS detrending, ``MZ_t = 3.288`` against ``-1.98``. Step 5 finds one cointegrating vector among the two Treasury yields, so those two belong in a VECM rather than a differenced VAR.

---

## Common Pitfalls

1. **Wrong deterministic specification.** Including a trend when the series has none costs power; omitting one when the series trends guarantees a failure to reject. Use `:constant` for series fluctuating around a fixed mean, `:trend` when a linear trend is visible, and `:none` only for data that has already been demeaned or detrended.

2. **Treating a failure to reject as proof of a unit root.** ADF has low power against roots near one, so a large p-value often just means the sample is short. Run KPSS as well: concordant results (ADF fails to reject, KPSS rejects) are much stronger than either alone, and the "inconclusive" cell of the decision matrix is an honest answer.

3. **Structural breaks masquerading as unit roots.** A stationary series with a level shift mimics a random walk, which pushes ADF, PP, and KPSS to reject their respective nulls simultaneously. That "conflicting" cell is the signature of a break. Use `za_test` for one break and the tests on the [advanced page](@ref tests_unitroot_advanced_page) for two or more.

4. **Reading a break-search statistic against ADF critical values.** Zivot-Andrews minimizes over the break grid, so its statistic is mechanically more negative than a fixed-date ``t``-ratio. Its 5% value for `:both` is ``-5.08``, not ``-2.85``. Always compare against `result.critical_values`.

5. **Johansen lag sensitivity.** The rank decision moves with the VECM lag order ``p``: too few lags leave serial correlation in the residuals and distort test size, too many waste degrees of freedom. Choose ``p`` by fitting VARs at several orders and comparing AIC/BIC before running the cointegration test, and re-run at neighbouring orders to check the rank is stable.

6. **Column order in the Johansen critical-value matrices.** `critical_values_trace` and `critical_values_max` run 10%, 5%, 1% left to right, the reverse of the `Dict` keys used by the univariate tests. Column 2 is the 5% value.

---

## References

- Andrews, Donald W. K. 1991. "Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation." *Econometrica* 59 (3): 817--858. [https://doi.org/10.2307/2938229](https://doi.org/10.2307/2938229)
- Dickey, David A., and Wayne A. Fuller. 1979. "Distribution of the Estimators for Autoregressive Time Series with a Unit Root." *Journal of the American Statistical Association* 74 (366): 427--431. [https://doi.org/10.1080/01621459.1979.10482531](https://doi.org/10.1080/01621459.1979.10482531)
- Doornik, Jurgen A. 1998. "Approximations to the Asymptotic Distributions of Cointegration Tests." *Journal of Economic Surveys* 12 (5): 573--593. [https://doi.org/10.1111/1467-6419.00068](https://doi.org/10.1111/1467-6419.00068)
- Johansen, Soren. 1991. "Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models." *Econometrica* 59 (6): 1551--1580. [https://doi.org/10.2307/2938278](https://doi.org/10.2307/2938278)
- Johansen, Soren. 1995. *Likelihood-Based Inference in Cointegrated Vector Autoregressive Models*. Oxford: Oxford University Press. ISBN 978-0-19-877450-1.
- Kwiatkowski, Denis, Peter C. B. Phillips, Peter Schmidt, and Yongcheol Shin. 1992. "Testing the Null Hypothesis of Stationarity Against the Alternative of a Unit Root." *Journal of Econometrics* 54 (1--3): 159--178. [https://doi.org/10.1016/0304-4076(92)90104-Y](https://doi.org/10.1016/0304-4076(92)90104-Y)
- MacKinnon, James G. 1994. "Approximate Asymptotic Distribution Functions for Unit-Root and Cointegration Tests." *Journal of Business & Economic Statistics* 12 (2): 167--176. [https://doi.org/10.1080/07350015.1994.10510005](https://doi.org/10.1080/07350015.1994.10510005)
- MacKinnon, James G. 1996. "Numerical Distribution Functions for Unit Root and Cointegration Tests." *Journal of Applied Econometrics* 11 (6): 601--618. [https://doi.org/10.1002/(SICI)1099-1255(199611)11:6<601::AID-JAE417>3.0.CO;2-T](https://doi.org/10.1002/(SICI)1099-1255(199611)11:6%3C601::AID-JAE417%3E3.0.CO;2-T)
- MacKinnon, James G., Alfred A. Haug, and Leo Michelis. 1999. "Numerical Distribution Functions of Likelihood Ratio Tests for Cointegration." *Journal of Applied Econometrics* 14 (5): 563--577. [https://doi.org/10.1002/(SICI)1099-1255(199909/10)14:5<563::AID-JAE530>3.0.CO;2-R](https://doi.org/10.1002/(SICI)1099-1255(199909/10)14:5%3C563::AID-JAE530%3E3.0.CO;2-R)
- Ng, Serena, and Pierre Perron. 2001. "Lag Length Selection and the Construction of Unit Root Tests with Good Size and Power." *Econometrica* 69 (6): 1519--1554. [https://doi.org/10.1111/1468-0262.00256](https://doi.org/10.1111/1468-0262.00256)
- Osterwald-Lenum, Michael. 1992. "A Note with Quantiles of the Asymptotic Distribution of the Maximum Likelihood Cointegration Rank Test Statistics." *Oxford Bulletin of Economics and Statistics* 54 (3): 461--472. [https://doi.org/10.1111/j.1468-0084.1992.tb00013.x](https://doi.org/10.1111/j.1468-0084.1992.tb00013.x)
- Perron, Pierre, and Serena Ng. 1996. "Useful Modifications to Some Unit Root Tests with Dependent Errors and Their Local Asymptotic Properties." *Review of Economic Studies* 63 (3): 435--463. [https://doi.org/10.2307/2297890](https://doi.org/10.2307/2297890)
- Phillips, Peter C. B., and Pierre Perron. 1988. "Testing for a Unit Root in Time Series Regression." *Biometrika* 75 (2): 335--346. [https://doi.org/10.1093/biomet/75.2.335](https://doi.org/10.1093/biomet/75.2.335)
- Zivot, Eric, and Donald W. K. Andrews. 1992. "Further Evidence on the Great Crash, the Oil-Price Shock, and the Unit-Root Hypothesis." *Journal of Business & Economic Statistics* 10 (3): 251--270. [https://doi.org/10.1080/07350015.1992.10509904](https://doi.org/10.1080/07350015.1992.10509904)
