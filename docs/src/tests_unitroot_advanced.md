# [Advanced Unit Root Tests](@id tests_unitroot_advanced_page)

Standard unit root tests behave well when the data-generating process is a simple autoregression with fixed deterministic terms. Real macroeconomic series are not so obliging: they drift through smooth regime changes, shift level abruptly, carry seasonal roots, and occasionally turn explosive. This page covers the tests that handle those cases — Fourier approximation of smooth breaks, GLS detrending for near-optimal power, point-optimal and seasonal-frequency testing, LM testing with breaks under the null, two-break ADF, and the right-tailed sup-ADF family for bubbles.

For the full test battery and the tables that route a question to a test, see [Hypothesis Tests](@ref tests_page). For the standard ADF, KPSS, PP, Zivot-Andrews, and Ng-Perron tests, see [Unit Root & Cointegration](@ref tests_unitroot_page). Break tests that ask *when* a relationship changed rather than *whether* a root is present live on [Structural Breaks](@ref tests_breaks_page), and the martingale-difference variance-ratio tests on [Model Diagnostics](@ref tests_diagnostics_page).

- **Fourier ADF** (Enders & Lee 2012): smooth structural change of unknown form, approximated by trigonometric terms
- **Fourier KPSS** (Becker, Enders & Lee 2006): the stationarity-null counterpart
- **DF-GLS** (Elliott, Rothenberg & Stock 1996): GLS-detrended ADF with near-optimal power
- **ERS point-optimal** (Elliott, Rothenberg & Stock 1996): the standalone ``P_T`` test built on the same detrending
- **HEGY seasonal** (Hylleberg et al. 1990; Beaulieu & Miron 1993): unit roots frequency by frequency, quarterly and monthly
- **LM unit root** (Schmidt & Phillips 1992; Lee & Strazicich 2003, 2013): breaks under the null, so rejection is unambiguous
- **Two-break ADF** (Narayan & Popp 2010): two endogenous breaks in level and slope
- **SADF / GSADF** (Phillips, Wu & Yu 2011; Phillips, Shi & Yu 2015): right-tailed sup-ADF tests for explosive behaviour, with date-stamping

```@setup test_ur_adv
using MacroEconometricModels, Random
fred = load_example(:fred_md)
cpi  = filter(isfinite, fred[:, "CPIAUCSL"])[end-299:end]
```

## Quick Start

**Recipe 1: Fourier ADF for smooth breaks**

```@example test_ur_adv
# No break dates required — the sin/cos pair absorbs gradual change
report(fourier_adf_test(cpi; regression=:constant, fmax=3))
```

**Recipe 2: DF-GLS when the call is close**

```@example test_ur_adv
# GLS detrending buys power against local alternatives
report(dfgls_test(cpi; regression=:constant, lags=:aic))
```

**Recipe 3: Seasonal unit roots frequency by frequency**

```@example test_ur_adv
# Is the zero-frequency root the only one? (CPIAUCSL is seasonally adjusted)
report(hegy_test(cpi; frequency=12, deterministic=:const_trend_seas))
```

**Recipe 4: LM unit root with two breaks**

```@example test_ur_adv
# Stationary AR(1) whose mean shifts at t = 101 and t = 201
Random.seed!(5)
u = zeros(300)
for t in 2:300
    u[t] = 0.5 * u[t-1] + randn()
end
ys = vcat(fill(0.0, 100), fill(4.0, 100), fill(1.0, 100)) .+ u

report(lm_unitroot_test(ys; breaks=2, regression=:level))
```

**Recipe 5: GSADF for explosive-bubble detection**

```@example test_ur_adv
# A random walk with an explosive window in [70, 110], then a collapse
Random.seed!(7)
price = zeros(160)
for t in 2:160
    price[t] = (70 <= t <= 110 ? 1.05 : 1.0) * price[t-1] + randn()
end

# Right-tailed: reject the unit root for LARGE statistics
report(gsadf_test(price; adflag=0, mc_reps=299))
```

---

## Fourier ADF Test

The Fourier ADF test (Enders & Lee 2012) augments the ADF regression with low-frequency trigonometric terms that approximate smooth structural change of unknown form. Where Zivot-Andrews models one abrupt break at a date it has to estimate, the Fourier approach captures gradual drift in the intercept or trend without specifying the number, dates, or shape of the changes.

The test regression is

```math
\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + a_k \sin\!\left(\frac{2\pi k t}{T}\right) + b_k \cos\!\left(\frac{2\pi k t}{T}\right) + \sum_{j=1}^{p} \delta_j \Delta y_{t-j} + \varepsilon_t
```

where:
- ``\gamma`` is the coefficient of interest, with ``H_0: \gamma = 0`` (unit root) against ``H_1: \gamma < 0`` (stationary around a smooth deterministic function)
- ``k`` is the Fourier frequency, chosen from ``1, \ldots, k_{\max}`` to minimize the residual sum of squares
- ``a_k, b_k`` load the sine and cosine terms
- ``\alpha`` is always included; ``\beta t`` only when `regression=:trend`
- ``p`` augmenting lags absorb serial correlation

A joint ``F``-test of ``H_0: a_k = b_k = 0`` says whether the Fourier terms earn their degrees of freedom. It is compared against the Enders-Lee tabulated ``F`` values, not the ``F`` distribution, because ``\gamma`` is not asymptotically normal under the unit root null.

!!! note "Technical Note"
    One Fourier frequency approximates a wide range of smooth changes — gradual level shifts, slow trend rotations, and several smooth breaks at once. Enders & Lee (2012) show ``k_{\max} = 3`` covers essentially every empirically relevant pattern. Both the ``\tau`` and ``F`` critical values are indexed by the selected ``k`` and by a sample-size bracket (``T \le 150``, ``151`` to ``349``, ``350`` to ``500``, above ``500``), so they tighten as ``k`` rises and the deterministic function grows more flexible.

```@example test_ur_adv
result = fourier_adf_test(cpi; regression=:constant, fmax=3)
report(result)
```

```@example test_ur_adv
(frequency = result.frequency, F = round(result.f_statistic, digits=3),
 F_pvalue = result.f_pvalue, F_cv5 = result.f_critical_values[5])
```

The search settles on ``k = 1``, and ``\tau = 0.307`` is nowhere near the 5% critical value of ``-3.78``: the unit root in the price level survives smooth-break adjustment, as it did the abrupt-break adjustment of Zivot-Andrews. The ``F``-statistic of 1.644 falls well short of its 5% value of 7.41, so the sine and cosine terms are not doing any work here and the plain `adf_test` is the more powerful choice on this series. That check matters: adding two insignificant regressors shifts the critical values left without buying anything.

### Options

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `regression` | `Symbol` | `:constant` | Deterministic terms: `:constant` or `:trend` |
| `fmax` | `Int` | `3` | Maximum Fourier frequency searched, in ``1`` to ``5`` |
| `lags` | `Union{Int,Symbol}` | `:aic` | Number of augmenting lags, or `:aic`/`:bic` for automatic selection |
| `max_lags` | `Union{Int,Nothing}` | `nothing` | Ceiling for automatic selection (defaults to ``\lfloor 12(T/100)^{1/4} \rfloor``) |
| `trim` | `Real` | `0.15` | Accepted for interface symmetry; the Fourier form has no break grid to trim, so it has no effect |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | ADF ``\tau`` statistic (``t``-ratio on ``\gamma``) at the selected frequency |
| `pvalue` | `T` | Interpolated p-value against the Fourier ADF table, clipped to ``[0.001, 0.20]`` |
| `frequency` | `Int` | Selected Fourier frequency ``k`` |
| `f_statistic` | `T` | ``F``-statistic for joint significance of the Fourier terms |
| `f_pvalue` | `T` | Interpolated p-value for the ``F``-statistic |
| `lags` | `Int` | Number of augmenting lags used |
| `regression` | `Symbol` | Deterministic specification |
| `critical_values` | `Dict{Int,T}` | ``\tau`` critical values for this ``k`` and sample bracket |
| `f_critical_values` | `Dict{Int,T}` | ``F`` critical values for this sample bracket |
| `nobs` | `Int` | Number of observations in the test regression (``T - 1 - p``) |

Both p-values saturate: anything less extreme than the 10% value reports `0.20`, and anything beyond the 1% value reports `0.001`. Read the statistic against `critical_values` when the exact tail probability matters.

---

## Fourier KPSS Test

The Fourier KPSS test (Becker, Enders & Lee 2006) does for the stationarity null what the Fourier ADF does for the unit root null. Under plain KPSS an unmodelled smooth break in the mean inflates the partial sums of residuals and produces a spurious rejection; the trigonometric regressors absorb that drift and restore correct size.

First regress the level on the deterministic terms plus the Fourier pair:

```math
y_t = \alpha + \beta t + a_k \sin\!\left(\frac{2\pi k t}{T}\right) + b_k \cos\!\left(\frac{2\pi k t}{T}\right) + e_t
```

where:
- ``\alpha`` is always included and ``\beta t`` only when `regression=:trend`
- ``a_k, b_k`` are the Fourier coefficients at the ``k`` minimizing the residual sum of squares

Then form the KPSS statistic from the partial sums ``S_t = \sum_{s \le t} \hat e_s`` of those residuals:

```math
\text{KPSS}_F = \frac{\sum_{t=1}^{T} S_t^2}{T^2 \hat{\sigma}^2_{LR}}
```

with ``\hat{\sigma}^2_{LR}`` the Bartlett long-run variance. The hypotheses are ``H_0: \sigma_u^2 = 0`` (stationary around a smooth deterministic function) against ``H_1: \sigma_u^2 > 0`` (unit root), and the test is right-tailed like standard KPSS.

```@example test_ur_adv
result = fourier_kpss_test(cpi; regression=:constant, fmax=3)
report(result)
```

```@example test_ur_adv
(frequency = result.frequency, F = round(result.f_statistic, digits=1),
 F_pvalue = result.f_pvalue, bandwidth = result.bandwidth)
```

The statistic is 1.844 against a 1% critical value of 0.271, so stationarity is decisively rejected — the price level has a unit root that no smooth deterministic function can explain away. Here the ``F``-statistic of 125.2 is enormous, the opposite of what the Fourier ADF reported on the same data. There is no contradiction: the Fourier KPSS fits the sine and cosine to the strongly trending *level*, where they explain a great deal, while the Fourier ADF fits them to the *differences*, where there is almost nothing smooth left to explain.

### Options

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `regression` | `Symbol` | `:constant` | Stationarity type: `:constant` (level) or `:trend` (trend) |
| `fmax` | `Int` | `3` | Maximum Fourier frequency searched; values above 3 are silently clamped to 3, the limit of the published tables |
| `bandwidth` | `Union{Int,Nothing}` | `nothing` | Bartlett lag truncation; `nothing` uses the fixed rule ``\lfloor 4(T/100)^{1/4} \rfloor`` |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | Fourier KPSS statistic |
| `pvalue` | `T` | Interpolated p-value, clipped to ``[0.001, 0.20]`` |
| `frequency` | `Int` | Selected Fourier frequency ``k`` |
| `f_statistic` | `T` | ``F``-statistic for joint significance of the Fourier terms |
| `f_pvalue` | `T` | Interpolated p-value for the ``F``-statistic |
| `regression` | `Symbol` | Stationarity type |
| `critical_values` | `Dict{Int,T}` | Fourier KPSS critical values for this ``k`` and sample bracket |
| `f_critical_values` | `Dict{Int,T}` | ``F`` critical values for this sample bracket |
| `bandwidth` | `Int` | Bartlett lag truncation used |
| `nobs` | `Int` | Number of observations (the full series length) |

Unlike `kpss_test`, this bandwidth is a deterministic function of ``T`` rather than a data-dependent plug-in, so it does not lengthen when the residuals are persistent.

### Combining Fourier ADF and Fourier KPSS

The pair reads exactly like the standard ADF-KPSS combination:

| Fourier ADF | Fourier KPSS | Conclusion |
|-------------|--------------|------------|
| Reject (stationary) | Fail to reject (stationary) | **Stationary** around a smooth deterministic path |
| Fail to reject (unit root) | Reject (unit root) | **Unit root** survives smooth-break adjustment |
| Reject | Reject | Conflicting — suspect a sharp break or a misspecified frequency |
| Fail to reject | Fail to reject | Inconclusive |

The price level lands in the second row, which is the strongest possible evidence that it is genuinely I(1).

---

## DF-GLS Test

The DF-GLS test (Elliott, Rothenberg & Stock 1996) detrends by GLS before running an ADF-type regression, which buys substantially more power against local alternatives than ordinary detrending. The same detrended series also yields the ERS point-optimal ``P_t`` statistic and the four Ng-Perron ``M^{GLS}`` statistics, so one call returns the whole power-optimized family.

GLS detrending quasi-differences the data at a local-to-unity parameter:

```math
\tilde{y}_1 = y_1, \qquad \tilde{y}_t = y_t - \bar{\alpha} \, y_{t-1}, \quad t = 2, \ldots, T
```

where:
- ``\bar{\alpha} = 1 + \bar{c}/T``
- ``\bar{c} = -7`` for `regression=:constant` and ``\bar{c} = -13.5`` for `:trend`

The deterministic regressors ``Z`` are quasi-differenced the same way, ``\hat{\delta}`` is estimated by regressing ``\tilde{y}`` on ``\tilde{Z}``, and the detrended series is ``y_t^d = y_t - Z_t \hat{\delta}``. The DF-GLS statistic is the ``t``-ratio on the lagged level in

```math
\Delta y_t^d = \gamma \, y_{t-1}^d + \sum_{j=1}^{p} \delta_j \, \Delta y_{t-j}^d + \varepsilon_t
```

which carries no intercept or trend, since GLS already removed them.

!!! note "Technical Note"
    The four ``M^{GLS}`` statistics returned here (``MZ_\alpha``, ``MZ_t``, ``MSB``, ``MP_T``) are computed from the same GLS-detrended series and the same AR spectral long-run variance that `ngperron_test` uses, so they agree with that function bit for bit on identical input. The `pt_statistic` field is likewise the ERS point-optimal ``P_t``, identical to what [`ers_test`](@ref) returns — one shared helper computes all three, so they cannot drift apart.

```@example test_ur_adv
result = dfgls_test(cpi; regression=:constant, lags=:aic)
report(result)
```

```@example test_ur_adv
(tau = round(result.statistic, digits=3), Pt = round(result.pt_statistic, digits=2),
 MZt = round(result.MZt, digits=3), MZt_cv5 = result.mgls_critical_values[:MZt][5])
```

Every member of the family points the same way. The DF-GLS ``\tau = 1.938`` is on the wrong side of zero against a 5% value of ``-2.029``; the point-optimal ``P_t = 124.02`` is enormous against a 5% value of 3.26 (small ``P_t`` rejects); and ``MZ_t = 2.406`` against ``-1.98``. When the most powerful test in the class cannot muster a rejection, the unit root is not a matter of low power. Note that AIC selects 15 lags, which is exactly the `max_lags` ceiling ``\lfloor 12(300/100)^{1/4}\rfloor = 15`` — a signal to raise `max_lags` and confirm the selection is interior.

### Options

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `regression` | `Symbol` | `:constant` | Deterministic terms: `:constant` or `:trend` |
| `lags` | `Union{Int,Symbol}` | `:aic` | Number of augmenting lags, or `:aic`/`:bic` for automatic selection |
| `max_lags` | `Union{Int,Nothing}` | `nothing` | Ceiling for automatic selection (defaults to ``\lfloor 12(T/100)^{1/4} \rfloor``) |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | DF-GLS ``\tau`` statistic |
| `pvalue` | `T` | Interpolated p-value for ``\tau``, clipped to ``[0.001, 0.20]`` |
| `pt_statistic` | `T` | ERS point-optimal ``P_t`` statistic |
| `pt_pvalue` | `T` | Interpolated p-value for ``P_t`` |
| `MZa`, `MZt`, `MSB`, `MPT` | `T` | Ng-Perron ``M^{GLS}`` statistics on the detrended series |
| `lags` | `Int` | Number of augmenting lags used |
| `regression` | `Symbol` | Deterministic specification |
| `critical_values` | `Dict{Int,T}` | DF-GLS ``\tau`` values from the response surface in ``1/T`` and ``p/T`` |
| `pt_critical_values` | `Dict{Int,T}` | ERS ``P_t`` values for the nearest tabulated sample size (50, 100, 200, 500) |
| `mgls_critical_values` | `Dict{Symbol,Dict{Int,T}}` | ``M^{GLS}`` values keyed by statistic then by level |
| `nobs` | `Int` | Number of observations in the test regression (``T - 1 - p``) |

### Interpretation

**Reject** ``H_0`` (p-value < 0.05): the series is stationary. DF-GLS has near-optimal power against alternatives of the form ``\rho = 1 + \bar{c}/T``, which makes it the test to reach for when the question is genuinely borderline. **Fail to reject**: when both ``\tau`` and ``P_t`` fail, the unit root evidence is strong. When they diverge, ``MZ_t`` is the usual tiebreaker, since it is the least sensitive of the family to a negative moving-average root in the errors.

---

## ERS Point-Optimal Test

The ERS feasible **point-optimal** test ``P_T`` is the second pillar of the Elliott-Rothenberg-Stock framework. Where DF-GLS builds a ``\tau``-style statistic from detrended data, ``P_T`` compares how well the local alternative fits against how well the unit root fits, using the residual sums of squares from the two quasi-differenced regressions:

```math
P_T = \frac{S(\bar{\alpha}) - \bar{\alpha}\, S(1)}{\hat{\omega}^2}
```

where:
- ``S(\bar{\alpha})`` is the SSR of the quasi-differenced regression at ``\bar{c} = -7`` (constant) or ``-13.5`` (trend)
- ``S(1)`` is the SSR under the unit root null, i.e. of the plain first-differenced regression
- ``\hat{\omega}^2`` is the AR spectral long-run variance, the same estimate DF-GLS uses

Small ``P_T`` rejects: the local-alternative model fits far better than the unit root model. Critical values are the Elliott, Rothenberg & Stock (1996, Table 1) values for the nearest tabulated sample size.

```@example test_ur_adv
result = ers_test(cpi; trend=false)
report(result)
```

``P_T = 124.02`` against a 10% critical value of 4.48 — an enormous distance in the direction of the null. The unit root model fits the price level essentially as well as the local alternative does, which is the point-optimal restatement of everything the ``\tau``-style tests reported.

The identity with `dfgls_test` is exact, not approximate:

```@example test_ur_adv
ers_test(cpi; trend=false).P_T == dfgls_test(cpi; regression=:constant).pt_statistic
```

### Options

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `trend` | `Bool` | `false` | `false` uses GLS demeaning (``\bar{c} = -7``); `true` uses GLS detrending (``\bar{c} = -13.5``) for trending series |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `P_T` | `T` | Feasible point-optimal statistic (note the field is **not** called `statistic`) |
| `pvalue` | `T` | Interpolated p-value against the ERS table, clipped to ``[0.001, 0.20]`` |
| `regression` | `Symbol` | Detrending case (`:constant` or `:trend`) |
| `critical_values` | `Dict{Int,T}` | ERS (1996, Table 1) values at 1%, 5%, 10% |
| `nobs` | `Int` | Number of observations (the full series length) |

---

## HEGY Seasonal Unit Roots

A seasonal series can carry unit roots not only at the zero (long-run) frequency but at seasonal frequencies too, and the two require different remedies. The Hylleberg, Engle, Granger & Yoo (1990) test — extended to monthly data by Beaulieu & Miron (1993) — tests **frequency by frequency**. For periodicity ``s`` it regresses the seasonal difference ``\Delta_s y_t = (1 - L^s) y_t`` on transform regressors that isolate each spectral frequency, plus deterministics and augmenting lags of ``\Delta_s y``:

```math
\Delta_s y_t = \pi_1 y_{1,t-1} + \pi_2 y_{2,t-1} + \sum_{\text{pairs}} \big(\pi_k y_{k,t-1} + \pi_{k+1} y_{k,t-2}\big) + \text{deterministics} + \sum_{i=1}^{p} \phi_i \Delta_s y_{t-i} + \varepsilon_t
```

where:
- ``y_{1,t}`` isolates the zero frequency, ``y_{2,t}`` the Nyquist frequency ``\omega = \pi``, and each pair ``(y_{k,t-1}, y_{k,t-2})`` a complex-conjugate harmonic
- each transform applies the product of *all other* factors of ``\Delta_s``, so only the root at its own frequency survives
- ``p`` augmenting lags of ``\Delta_s y`` absorb residual serial correlation

For quarterly data the transforms are ``y_{1,t} = (1+L)(1+L^2)y_t``, ``y_{2,t} = -(1-L)(1+L^2)y_t``, and ``y_{3,t} = (1-L^2)y_t`` (the annual harmonic pair at ``\omega = \pi/2``). Monthly data follows Beaulieu & Miron with twelve transforms and five harmonic pairs at ``\omega = \pi/6, \pi/3, \pi/2, 2\pi/3, 5\pi/6``.

At every frequency ``H_0`` is that a unit root is present there. The zero and Nyquist frequencies are tested with left-tailed ``t``-statistics ``t(\pi_1)`` and ``t(\pi_2)``; each harmonic pair with a right-tailed joint ``F``. Joint ``F`` statistics over all seasonal frequencies and over all frequencies are reported as well.

```@example test_ur_adv
# CPIAUCSL is a seasonally adjusted monthly series
result = hegy_test(cpi; frequency=12, deterministic=:const_trend_seas)
report(result)
```

Exactly one null survives. ``t(\pi_1) = -0.843`` is far above the 5% value of ``-3.36``, so the zero-frequency unit root stands; the Nyquist statistic ``-5.614`` clears ``-2.84`` and all five harmonic ``F`` statistics (23.0 to 29.5) clear 6.32, so every seasonal root is rejected. That is precisely the right answer for an officially seasonally adjusted series: the seasonal roots have already been filtered out, and the remaining problem is the ordinary long-run root. Differencing once with ``\Delta`` is appropriate; applying ``\Delta_{12}`` would over-difference at eleven frequencies to fix one.

Quarterly data uses `frequency=4`:

```@example test_ur_adv
# A seasonal random walk y_t = y_{t-4} + e_t has a root at EVERY frequency
Random.seed!(2)
yq = zeros(160)
for t in 5:160
    yq[t] = yq[t-4] + randn()
end

result_q = hegy_test(yq; frequency=4)
report(result_q)
```

Now no null is rejected: ``t(\pi_1) = -2.487`` against ``-3.47``, ``t(\pi_2) = -1.275`` against ``-2.92``, and the harmonic ``F = 2.295`` against 6.57. Since ``\Delta_4 = (1-L)(1+L)(1+L^2)`` contains one factor per frequency, and every frequency carries a root, full seasonal differencing is the correct transformation for this series.

### Options

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `frequency` | `Int` | `4` | `4` (quarterly) or `12` (monthly); anything else throws an `ArgumentError` |
| `deterministic` | `Symbol` | `:const_trend_seas` | `:none`, `:const`, `:const_seas`, `:const_trend`, or `:const_trend_seas`; seasonal dummies span the intercept, so they replace it when present |
| `lags` | `Union{Int,Symbol}` | `:auto` | Augmenting lags of ``\Delta_s y``, or `:auto`/`:aic` for AIC selection and `:bic` for BIC, up to ``\lfloor 12(T/100)^{1/4} \rfloor`` |

The series needs at least ``3s + 5`` observations, and an integer `lags` outside the feasible range throws.

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `frequency` | `Int` | Periodicity used (4 or 12) |
| `deterministic` | `Symbol` | Deterministic specification |
| `lags` | `Int` | Number of augmenting lags used |
| `pi_coefs` | `Vector{T}` | Estimated ``\pi`` coefficients, ordered zero, Nyquist, then pairs |
| `t_zero`, `t_nyquist` | `T` | Zero-frequency and Nyquist ``t``-statistics (left-tailed) |
| `t_zero_cv`, `t_nyquist_cv` | `Dict{Int,T}` | Their critical values at 1%, 5%, 10% |
| `pair_freqs` | `Vector{T}` | Harmonic frequencies ``\omega_k`` in radians |
| `pair_F` | `Vector{T}` | Joint ``F`` statistic for each harmonic pair (right-tailed) |
| `pair_F_cv` | `Dict{Int,T}` | Critical values shared by all harmonic pairs |
| `F_seasonal`, `F_all` | `T` | Joint ``F`` over all seasonal frequencies, and over all frequencies |
| `nobs` | `Int` | Number of rows in the auxiliary regression |

Critical values are the published HEGY (1990) quarterly and Beaulieu-Miron (1993) monthly tables. The Díaz-Emparanza (2014) response surfaces, which would give exact p-values rather than table lookups, are not implemented.

### Interpretation

Rejecting ``t(\pi_1)`` removes the zero-frequency root; rejecting ``t(\pi_2)`` removes the Nyquist root; rejecting a pair ``F`` removes the root at that seasonal frequency. If **no** null is rejected, ``\Delta_s`` is the right filter. If some seasonal nulls reject but the zero-frequency one does not, ``\Delta_s`` over-differences and plain ``\Delta`` plus seasonal dummies is the better specification. The joint ``F_{\text{seasonal}}`` is a convenient summary but is not a substitute for the individual tests, because a single strongly rejected frequency can carry it.

---

## LM Unit Root Test

The LM unit root test (Schmidt & Phillips 1992; Lee & Strazicich 2003, 2013) takes a different route to structural breaks from Zivot-Andrews and Narayan-Popp. Its defining feature is that breaks enter under the **null** as well as the alternative. Under Zivot-Andrews, breaks appear only under the alternative, so a rejection is ambiguous: it may mean the series is stationary, or it may mean the series is a unit root process that happens to break. The LM formulation removes that ambiguity, at least in theory — rejection implies stationarity whether or not breaks are present.

Three variants handle different break counts. `breaks=0` is the basic Schmidt-Phillips test; `breaks=1` grid-searches one break date (Lee & Strazicich 2013); `breaks=2` grid-searches a pair (Lee & Strazicich 2003).

The procedure detrends under the null and then runs an ADF-type regression on the detrended series:

```math
\tilde{S}_t = y_t - Z_t \tilde{\delta}, \qquad
\Delta \tilde{S}_t = \phi \, \tilde{S}_{t-1} + \sum_{j=1}^{p} \delta_j \Delta \tilde{S}_{t-j} + \varepsilon_t
```

where:
- ``Z_t`` collects an intercept, a linear trend, and the break dummies ``DU_{it} = \mathbf{1}(t > T_{Bi})``
- for `regression=:both`, ``Z_t`` also carries the trend-shift dummies ``DT_{it} = (t - T_{Bi}) \cdot \mathbf{1}(t > T_{Bi})``
- ``\tilde{\delta}`` is estimated by OLS of ``y`` on ``Z`` in levels
- ``\phi`` is the coefficient of interest, with ``H_0: \phi = 0`` (unit root with possible breaks) against ``H_1: \phi < 0`` (trend-stationary with possible breaks)

The statistic is the ``t``-ratio on ``\phi``, minimized over the trimmed grid of break dates.

!!! note "Critical values do not vary with the break location"
    Lee & Strazicich tabulate critical values that depend on the break fractions ``\lambda_i = T_{Bi}/T``. This implementation uses a single Model A (level-shift) table for `breaks=1` and another for `breaks=2`, with no interpolation over ``\lambda`` and no separate table for `regression=:both`. The `breaks=0` values are a genuine response surface in ``1/T`` and ``p/T``.

!!! warning "The break-search variants over-reject badly"
    On driftless random walks with no breaks (``T = 150``, 60 replications, `lags=0`), `breaks=0` held its nominal 5% size while `breaks=1` rejected 55% of the time and `breaks=2` 85%. Minimizing over the grid drives the statistic far below what the fixed tables anticipate. Treat a rejection from `breaks=1` or `breaks=2` as suggestive only, and corroborate with `dfgls_test` or `fourier_adf_test` before a differencing decision rests on it.

When the series is genuinely stationary around shifting means, the test rejects and pins the dates:

```@example test_ur_adv
# The Recipe 4 series: stationary AR(1) with mean shifts at t = 101 and t = 201
result = lm_unitroot_test(ys; breaks=2, regression=:level)

(statistic = round(result.statistic, digits=3), pvalue = result.pvalue,
 breaks = result.break_dates, cv5 = result.critical_values[5])
```

Here ``\tau = -12.379`` against a 5% value of ``-3.842``, and the estimated dates 103 and 201 sit within two observations of the truth. Recovering the true break locations is the diagnostic that separates a genuine rejection from an artifact — dates that cluster at the trimming boundary, or that correspond to nothing in the data, mean the grid search found noise.

The size distortion is easy to see directly. Apply the one-break test to a random walk that also shifts level, where the correct answer is "unit root":

```@example test_ur_adv
# A random walk with a level shift at t = 101 — an I(1) series, breaks and all
Random.seed!(202608)
y = vcat(cumsum(randn(100)), 5.0 .+ cumsum(randn(100)))

one_break = lm_unitroot_test(y; breaks=1, regression=:level)
no_break  = lm_unitroot_test(y; breaks=0, regression=:level)

(with_break = (round(one_break.statistic, digits=3), one_break.critical_values[5]),
 without_break = (round(no_break.statistic, digits=3), no_break.critical_values[5]))
```

The theory says this series should survive both tests, since breaks sit under ``H_0``. In practice `breaks=1` returns ``\tau = -4.172`` against a 5% value of ``-3.566`` and rejects, while `breaks=0` returns ``-3.340`` against ``-3.041`` and also rejects. Both are false positives on a series that is I(1) by construction, and the one-break statistic is the more distorted of the two — a concrete instance of the size problem above rather than a counterexample to the LM design. Schmidt-Phillips (`breaks=0`) is correctly sized on average; this particular draw is simply unlucky.

### Options

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `breaks` | `Int` | `0` | Number of structural breaks: 0, 1, or 2 |
| `regression` | `Symbol` | `:level` | Break type: `:level` (intercept shifts) or `:both` (intercept and trend shifts) |
| `lags` | `Union{Int,Symbol}` | `:aic` | Number of augmenting lags, or `:aic`/`:bic` for automatic selection |
| `max_lags` | `Union{Int,Nothing}` | `nothing` | Ceiling for automatic selection (defaults to ``\lfloor 12(T/100)^{1/4} \rfloor``) |
| `trim` | `Real` | `0.15` | Trimming fraction for the break search; with `breaks=2` it also sets the minimum gap between the two dates |

At least 50 observations are required. A trend is included in ``Z_t`` for `breaks=1` and `breaks=2` regardless of `regression`; only `breaks=0` with `:level` omits it.

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | LM statistic (``t``-ratio on ``\phi``), minimized over the break grid |
| `pvalue` | `T` | Interpolated p-value, clipped to ``[0.001, 0.20]`` |
| `breaks` | `Int` | Number of breaks (0, 1, or 2) |
| `break_dates` | `Vector{Int}` | Estimated break dates as observation indices; empty when `breaks=0` |
| `break_fractions` | `Vector{T}` | Break locations as fractions of the sample; empty when `breaks=0` |
| `lags` | `Int` | Number of augmenting lags used |
| `regression` | `Symbol` | Break specification |
| `critical_values` | `Dict{Int,T}` | Critical values at 1%, 5%, 10% |
| `nobs` | `Int` | Number of observations in the test regression (``T - 1 - p``) |

---

## Two-Break ADF Test

The two-break ADF test (Narayan & Popp 2010) extends the ADF framework to two endogenous breaks. Zivot-Andrews allows one, but many macro series carry at least two regime changes — the Great Moderation and the 2008 crisis being the standard example. The test searches every admissible pair of break dates and keeps the pair minimizing the ``t``-statistic on ``\gamma``.

**Level shifts only** (`model=:level`):

```math
\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \theta_1 DU_{1t} + \theta_2 DU_{2t} + \sum_{j=1}^{p} \delta_j \Delta y_{t-j} + \varepsilon_t
```

**Level and trend shifts** (`model=:both`) adds ``\phi_1 DT_{1t} + \phi_2 DT_{2t}``. In both cases:

- ``DU_{it} = \mathbf{1}(t \geq T_{Bi})`` is the level shift dummy for break ``i``
- ``DT_{it} = (t - T_{Bi} + 1) \cdot \mathbf{1}(t \geq T_{Bi})`` is the trend shift dummy
- ``T_{B1} < T_{B2}`` are the break dates, chosen to minimize the ``t``-ratio on ``\gamma``
- ``H_0: \gamma = 0`` (unit root with two breaks) against ``H_1: \gamma < 0`` (stationary with two breaks)

!!! note "Technical Note"
    The grid search is quadratic in the sample: with trimming ``\tau`` the number of candidate pairs is ``O\big((T(1-2\tau))^2\big)``, and each pair re-runs the lag selection. The default ``\tau = 0.10`` excludes the first and last 10% of the sample. The minimum gap between dates is 2 observations for `:level` and 3 for `:both`, enough to identify the break parameters separately. Critical values are the Narayan-Popp (2010) tables selected by the nearest sample-size bracket (50, 200, 400, larger), with no interpolation between brackets.

!!! warning "Empirical size far exceeds nominal"
    In the same Monte Carlo check as the LM test — driftless random walks with no breaks, ``T = 150``, 60 replications, `lags=0` — this test rejected 70% of the time at a nominal 5% level. Grid minimization over pairs drives the statistic well past what the tabulated values allow for. A rejection here is a reason to look at the estimated break dates, not a licence to model the series as stationary.

```@example test_ur_adv
# Stationary AR(1) with level shifts at t = 81 and t = 161
Random.seed!(5)
u = zeros(240)
for t in 2:240
    u[t] = 0.5 * u[t-1] + randn()
end
yb = vcat(fill(0.0, 80), fill(3.0, 80), fill(-2.0, 80)) .+ u

result = adf_2break_test(yb; model=:level, lags=:aic)
report(result)
```

The minimum ``t``-statistic is ``-10.186`` against a 5% value of ``-4.136``, and the breaks land at observations 83 and 163, within three of the true shift points. On this genuinely stationary series the rejection is correct and the dates are informative. The contrast with the LM test matters for how you read it: there, breaks sit under the null, so rejection points to stationarity; here they sit under the alternative, so rejection is consistent with either stationarity or a unit root that breaks. When that distinction drives a modelling decision, `lm_unitroot_test` gives the cleaner inference in principle — subject to the size caveats attached to both.

### Options

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `model` | `Symbol` | `:level` | Break type: `:level` (intercept shifts) or `:both` (intercept and trend shifts) |
| `lags` | `Union{Int,Symbol}` | `:aic` | Number of augmenting lags, or `:aic`/`:bic` for automatic selection |
| `max_lags` | `Union{Int,Nothing}` | `nothing` | Ceiling for automatic selection (defaults to ``\lfloor 12(T/100)^{1/4} \rfloor``) |
| `trim` | `Real` | `0.10` | Trimming fraction for the break search |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | Minimum ADF ``t``-statistic over all candidate break pairs |
| `pvalue` | `T` | Interpolated p-value against the Narayan-Popp table, clipped to ``[0.001, 0.20]`` |
| `break1`, `break2` | `Int` | Estimated break dates as observation indices |
| `break1_fraction`, `break2_fraction` | `T` | Break locations as fractions of the sample |
| `lags` | `Int` | Number of augmenting lags at the selected pair |
| `model` | `Symbol` | Break specification (`:level` or `:both`) |
| `critical_values` | `Dict{Int,T}` | Critical values at 1%, 5%, 10% for the matching sample bracket |
| `nobs` | `Int` | Number of observations in the test regression (``T - 1 - p``) |

---

## SADF and GSADF Bubble Detection

Every test above asks whether a series has a unit root against a *stationary* alternative. Asset-price exuberance poses the opposite question: is the root *explosive*, ``\rho > 1``? The sup-ADF family of Phillips, Wu & Yu (2011) and Phillips, Shi & Yu (2015) answers it with **right-tailed** ADF regressions. The unit root null is rejected in favour of a mildly explosive root for **large** statistics, against **upper** simulated critical values — the reverse of every other test on this page.

For a window ``[r_1, r_2]`` expressed as fractions of ``T``, let ``\mathrm{ADF}_{r_1}^{r_2}`` be the right-tailed ADF ``t``-statistic on the lagged level, fitted with a constant and `adflag` augmenting lags. The three statistics are

```math
\mathrm{SADF} = \sup_{r_2 \in [r_0, 1]} \mathrm{ADF}_0^{r_2}, \qquad
\mathrm{GSADF} = \sup_{\substack{r_2 \in [r_0, 1] \\ r_1 \in [0, r_2 - r_0]}} \mathrm{ADF}_{r_1}^{r_2},
```

```math
\mathrm{BSADF}(r_2) = \sup_{r_1 \in [0, r_2 - r_0]} \mathrm{ADF}_{r_1}^{r_2}
```

where:
- ``r_0`` is the minimum window fraction; `r0=:auto` uses the PSY rule ``r_0 = 0.01 + 1.8/\sqrt{T}``
- **SADF** fixes the start at ``r_1 = 0`` and expands the end — the original PWY (2011) test
- **GSADF** floats both endpoints, a double supremum, giving power against *periodically collapsing* bubbles that SADF misses
- **BSADF**``(r_2)`` is the backward sup-ADF sequence used to date-stamp episodes

Date-stamping compares the whole ``\mathrm{BSADF}(r_2)`` sequence against its own 95% critical-value *sequence*, one value per ``r_2``. Origination is the first crossing above, termination the first subsequent crossing below, subject to the PSY minimum-duration rule of ``\lceil \log T \rceil`` observations.

!!! note "Where the critical values come from"
    Under `cv=:asymptotic` the null is the PSY driftless random walk ``y^*_t = y^*_{t-1} + T^{-1/2}\varepsilon_t``; under `cv=:wildboot` it is the Phillips-Shi (2020) wild bootstrap, which resamples the sample's own demeaned first differences with standard normal multipliers so heteroskedasticity is preserved under the unit root null. Analytic null draws are cached by ``(\text{kind}, T, \text{window}, \texttt{adflag}, \texttt{mc\_reps}, \texttt{seed})``, so a repeated call with identical arguments is free; wild-bootstrap draws depend on the sample and are never cached.

```@example test_ur_adv
# The Recipe 5 series: a random walk, an explosive run over [70, 110], then a collapse
res = gsadf_test(price; adflag=0, mc_reps=299)

(statistic = round(res.statistic, digits=3), pvalue = round(res.pvalue, digits=4),
 cv95 = round(res.critical_values[5], digits=3), episodes = res.episodes)
```

GSADF returns 9.126 against a 95% critical value of 2.018 and a 99% value of 2.503, so no null replication came close and ``p < 0.001``. The stamped episode spans observations 65 to 154. Origination is picked up five observations before the explosive window actually opens at ``t = 70``, and termination runs well past its close at ``t = 110``, because the level accumulated during the explosive run keeps the backward sup above its critical-value sequence for some time afterwards. Read stamped dates as bracketing an episode, not as pinpointing its endpoints.

The fixed-start `sadf_test` shares the interface exactly and stores the recursive-ADF sequence in place of the backward sup:

```@example test_ur_adv
sres = sadf_test(price; adflag=0, mc_reps=299)

(sadf = round(sres.statistic, digits=3), cv95 = round(sres.critical_values[5], digits=3),
 episodes = sres.episodes)
```

SADF gives 8.857 against a 95% value of 1.235 — also a decisive rejection, and its stamped episode opens at exactly ``t = 70``, since the fixed start makes the recursive statistic respond as soon as the explosive segment enters the window. This series contains a single bubble, which is the case SADF handles well. GSADF is nevertheless the standard real-time exuberance monitor, because the double supremum is what recovers the second and third episodes of a periodically collapsing series, where a fixed start dilutes each one against the growing earlier sample.

The signature PSY chart plots the ``\mathrm{BSADF}(r_2)`` sequence against its 95% critical-value sequence with the stamped episodes shaded:

```julia
plot_result(res)   # BSADF vs 95% CV, shaded bubble episodes
```

```@raw html
<iframe src="../assets/plots/bubble_monitor.html" width="100%" height="440" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The sequence begins at observation 24, the first ``r_2`` the minimum window admits, and stays below its critical-value line until observation 65. It peaks at 9.126 at observation 110 — the last period of the explosive segment, where the accumulated run is fully inside the window — and the shaded episode covers 65 to 154. Reading the chart rather than the scalar statistic is the point of the exercise: the peak locates the end of the explosive run, the shading brackets it, and the gap between the two lines outside the episode shows how far from exuberance the series otherwise is.

### Options

Both functions take the same keywords.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `r0` | `Union{Symbol,Real}` | `:auto` | Minimum window fraction in ``(0,1)``; `:auto` uses the PSY rule ``0.01 + 1.8/\sqrt{T}``. The window is additionally floored at `adflag + 5` observations |
| `adflag` | `Int` | `0` | Augmenting lags in each window regression |
| `mc_reps` | `Int` | `999` | Null replications for the critical values |
| `cv` | `Symbol` | `:asymptotic` | Null model: `:asymptotic` (PSY random walk) or `:wildboot` (Phillips-Shi 2020) |
| `seed` | `Int` | `20240716` | RNG seed; each draw is seeded separately, so results are reproducible and independent of thread order |

At least 20 observations are required, though the minimum window leaves too few sub-samples for the supremum to be informative much below 50.

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `kind` | `Symbol` | `:sadf` or `:gsadf` |
| `statistic` | `T` | The supremum statistic |
| `pvalue` | `T` | Share of null replications at or above the statistic |
| `critical_values` | `Dict{Int,T}` | **Upper** null quantiles at 1%, 5%, 10% — reject when the statistic exceeds them |
| `bsadf` | `Vector{T}` | The sequence used for date-stamping: BSADF for `:gsadf`, the recursive ADF for `:sadf` |
| `cv_seq` | `Vector{T}` | The 95% critical-value sequence, one entry per ``r_2`` |
| `r2_index` | `Vector{Int}` | Observation index of each ``r_2``, aligning `bsadf` and `cv_seq` to the input series |
| `episodes` | `Vector{Tuple{Int,Int}}` | Stamped (origination, termination) index pairs into `y` |
| `r0` | `T` | Minimum window fraction actually used |
| `adflag`, `cv_method`, `mc_reps` | — | The settings the result was produced with |
| `nobs` | `Int` | Length of the input series |

---

## Complete Example

Every test on this page applied to one series, with the standard ADF as the baseline.

```@example test_ur_adv
# ── Step 1: Standard ADF as baseline ────────────────────────────
adf = adf_test(cpi; lags=:aic, regression=:constant)
report(adf)
```

```@example test_ur_adv
# ── Step 2: Smooth breaks, both directions ──────────────────────
fadf  = fourier_adf_test(cpi; regression=:constant, fmax=3)
fkpss = fourier_kpss_test(cpi; regression=:constant, fmax=3)

(fourier_adf = round(fadf.statistic, digits=3), F = round(fadf.f_statistic, digits=3),
 fourier_kpss = round(fkpss.statistic, digits=3))
```

```@example test_ur_adv
# ── Step 3: Maximum power via GLS detrending ────────────────────
dg = dfgls_test(cpi; regression=:constant, lags=:aic)
(tau = round(dg.statistic, digits=3), Pt = round(dg.pt_statistic, digits=2),
 MZt = round(dg.MZt, digits=3))
```

```@example test_ur_adv
# ── Step 4: Abrupt breaks, under H0 and under H1 ────────────────
lm1  = lm_unitroot_test(cpi; breaks=1, regression=:level)
adf2 = adf_2break_test(cpi; model=:level, lags=:aic)

(lm_stat = round(lm1.statistic, digits=3), lm_break = lm1.break_dates[1],
 np_stat = round(adf2.statistic, digits=3), np_breaks = (adf2.break1, adf2.break2))
```

```@example test_ur_adv
# ── Step 5: Synthesis across the battery ────────────────────────
verdict(p) = p < 0.05 ? "reject H0" : "fail to reject H0"

[("ADF",            verdict(adf.pvalue)),
 ("Fourier ADF",    verdict(fadf.pvalue)),
 ("DF-GLS",         verdict(dg.pvalue)),
 ("LM (1 break)",   verdict(lm1.pvalue)),
 ("ADF (2 breaks)", verdict(adf2.pvalue)),
 ("Fourier KPSS",   verdict(fkpss.pvalue))]
```

The four tests whose null is a unit root and whose statistic is not a grid minimum — ADF (2.387), Fourier ADF (0.307), DF-GLS (1.938), and the point-optimal ``P_t`` (124.02) — all fail to reject, and Fourier KPSS rejects stationarity at 0.1%. That is a unanimous I(1) verdict from five directions. The two break-search tests dissent: LM with one break gives ``-3.946`` (``p = 0.027``) and the Narayan-Popp two-break test gives ``-5.287`` (``p = 0.001``). Given their measured over-rejection under an I(1) null, the dissent is much better explained by the size distortion of grid minimization than by genuine break-stationarity in the US price level. Difference the series.

---

## Common Pitfalls

1. **Setting `fmax` above 3.** A single low-frequency Fourier component (``k = 1`` or ``2``) approximates almost every empirically relevant smooth break. Larger `fmax` widens the search space, and the extra frequencies fit noise rather than structure while the critical values tighten. Always read `f_pvalue`: when the Fourier terms are insignificant, plain `adf_test` is strictly more powerful.

2. **DF-GLS oversizing with a large negative MA root.** The GLS detrending that gives DF-GLS its power backfires when the errors carry a large negative moving-average root (say ``\theta < -0.8``), where it rejects far too often in finite samples (Perron & Ng 1996). When ADF and DF-GLS disagree and MA contamination is plausible, ADF is the safer read, and `MZt` from the same call is less sensitive than the DF-GLS ``\tau``.

3. **Trusting a break-search rejection on its own.** `lm_unitroot_test` with `breaks=1` or `2` and `adf_2break_test` both minimize over a grid of break dates, and both reject far more often than their nominal size under an I(1) null. Corroborate with a test that has no grid — `dfgls_test`, `fourier_adf_test`, or plain `adf_test` — and check that the estimated break dates correspond to something economically real rather than clustering at the trimming boundary.

4. **Confusing where the breaks sit.** In `lm_unitroot_test` breaks are under ``H_0``, so a rejection points to stationarity. In `za_test` and `adf_2break_test` they are under ``H_1``, so a rejection is consistent with stationarity *or* with a unit root that breaks. Use the LM formulation when that distinction drives the specification.

5. **Reading SADF/GSADF as left-tailed.** Every other test on this page rejects for large *negative* statistics. The sup-ADF tests invert this: they reject in favour of an explosive root when the statistic exceeds an *upper* critical value, so the rejection condition is `res.statistic > res.critical_values[5]`, not `<`. Date-stamp with the ``\mathrm{BSADF}(r_2)`` *sequence* against `cv_seq`, never against the scalar `critical_values`, which would mis-stamp short episodes.

6. **Running GSADF at full `mc_reps` while exploring.** The double supremum over the null replications dominates the cost. Keep `mc_reps` at 199 or 299 while iterating and raise it for the final result; identical analytic calls hit the cache and cost nothing, but changing `mc_reps` or `seed` invalidates it.

---

## References

- Beaulieu, J. Joseph, and Jeffrey A. Miron. 1993. "Seasonal Unit Roots in Aggregate U.S. Data." *Journal of Econometrics* 55 (1--2): 305--328. [https://doi.org/10.1016/0304-4076(93)90018-Z](https://doi.org/10.1016/0304-4076(93)90018-Z)
- Becker, Ralf, Walter Enders, and Junsoo Lee. 2006. "A Stationarity Test in the Presence of an Unknown Number of Smooth Breaks." *Journal of Time Series Analysis* 27 (3): 381--409. [https://doi.org/10.1111/j.1467-9892.2006.00478.x](https://doi.org/10.1111/j.1467-9892.2006.00478.x)
- Díaz-Emparanza, Ignacio. 2014. "Numerical Distribution Functions for Seasonal Unit Root Tests." *Computational Statistics & Data Analysis* 76: 237--247. [https://doi.org/10.1016/j.csda.2013.03.006](https://doi.org/10.1016/j.csda.2013.03.006)
- Elliott, Graham, Thomas J. Rothenberg, and James H. Stock. 1996. "Efficient Tests for an Autoregressive Unit Root." *Econometrica* 64 (4): 813--836. [https://doi.org/10.2307/2171846](https://doi.org/10.2307/2171846)
- Enders, Walter, and Junsoo Lee. 2012. "A Unit Root Test Using a Fourier Series to Approximate Smooth Breaks." *Oxford Bulletin of Economics and Statistics* 74 (4): 574--599. [https://doi.org/10.1111/j.1468-0084.2011.00662.x](https://doi.org/10.1111/j.1468-0084.2011.00662.x)
- Hylleberg, Svend, Robert F. Engle, Clive W. J. Granger, and Byung Sam Yoo. 1990. "Seasonal Integration and Cointegration." *Journal of Econometrics* 44 (1--2): 215--238. [https://doi.org/10.1016/0304-4076(90)90080-D](https://doi.org/10.1016/0304-4076(90)90080-D)
- Lee, Junsoo, and Mark C. Strazicich. 2003. "Minimum Lagrange Multiplier Unit Root Test with Two Structural Breaks." *Review of Economics and Statistics* 85 (4): 1082--1089. [https://doi.org/10.1162/003465303772815961](https://doi.org/10.1162/003465303772815961)
- Lee, Junsoo, and Mark C. Strazicich. 2013. "Minimum LM Unit Root Test with One Structural Break." *Economics Bulletin* 33 (4): 2483--2492.
- Narayan, Paresh Kumar, and Stephan Popp. 2010. "A New Unit Root Test with Two Structural Breaks in Level and Slope at Unknown Time." *Journal of Applied Statistics* 37 (9): 1425--1438. [https://doi.org/10.1080/02664760903039883](https://doi.org/10.1080/02664760903039883)
- Ng, Serena, and Pierre Perron. 2001. "Lag Length Selection and the Construction of Unit Root Tests with Good Size and Power." *Econometrica* 69 (6): 1519--1554. [https://doi.org/10.1111/1468-0262.00256](https://doi.org/10.1111/1468-0262.00256)
- Perron, Pierre, and Serena Ng. 1996. "Useful Modifications to Some Unit Root Tests with Dependent Errors and Their Local Asymptotic Properties." *Review of Economic Studies* 63 (3): 435--463. [https://doi.org/10.2307/2297890](https://doi.org/10.2307/2297890)
- Phillips, Peter C. B., and Shuping Shi. 2020. "Real-Time Monitoring of Asset Markets: Bubbles and Crises." In *Handbook of Statistics* 42, 61--80. Elsevier. [https://doi.org/10.1016/bs.host.2018.12.002](https://doi.org/10.1016/bs.host.2018.12.002)
- Phillips, Peter C. B., Shuping Shi, and Jun Yu. 2015. "Testing for Multiple Bubbles: Historical Episodes of Exuberance and Collapse in the S&P 500." *International Economic Review* 56 (4): 1043--1078. [https://doi.org/10.1111/iere.12132](https://doi.org/10.1111/iere.12132)
- Phillips, Peter C. B., Yangru Wu, and Jun Yu. 2011. "Explosive Behavior in the 1990s Nasdaq: When Did Exuberance Escalate Asset Values?" *International Economic Review* 52 (1): 201--226. [https://doi.org/10.1111/j.1468-2354.2010.00625.x](https://doi.org/10.1111/j.1468-2354.2010.00625.x)
- Schmidt, Peter, and Peter C. B. Phillips. 1992. "LM Tests for a Unit Root in the Presence of Deterministic Trends." *Oxford Bulletin of Economics and Statistics* 54 (3): 257--287. [https://doi.org/10.1111/j.1468-0084.1992.tb00002.x](https://doi.org/10.1111/j.1468-0084.1992.tb00002.x)
