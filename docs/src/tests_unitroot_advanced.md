# [Advanced Unit Root Tests](@id tests_unitroot_advanced_page)

Standard unit root tests (ADF, PP, KPSS) perform well when the data-generating process is a simple autoregressive model with fixed deterministic components. Real macroeconomic series, however, exhibit smooth structural changes, multiple regime shifts, and near-unit-root behavior that erode the power of classical tests. This page covers five advanced unit root tests that address these limitations through Fourier approximation of smooth breaks, GLS detrending for optimal power, LM-based testing with endogenous breaks under the null, and two-break ADF extensions.

- **Fourier ADF** (Enders & Lee 2012): Captures smooth, unknown structural breaks with trigonometric terms
- **Fourier KPSS** (Becker, Enders & Lee 2006): Stationarity test robust to smooth breaks
- **DF-GLS / ERS** (Elliott, Rothenberg & Stock 1996): GLS-detrended ADF with near-optimal power
- **LM Unit Root** (Schmidt & Phillips 1992; Lee & Strazicich 2003, 2013): Breaks under the null hypothesis
- **Two-Break ADF** (Narayan & Popp 2010): ADF with two endogenous structural breaks

## Quick Start

**Recipe 1: Fourier ADF for smooth breaks**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
cpi = filter(isfinite, to_vector(fred[:, "CPIAUCSL"]))

# Fourier ADF — captures smooth structural change without specifying break dates
result = fourier_adf_test(cpi; regression=:constant)
report(result)
```

**Recipe 2: DF-GLS for best power**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
cpi = filter(isfinite, to_vector(fred[:, "CPIAUCSL"]))

# DF-GLS — near-optimal power against local alternatives
result = dfgls_test(cpi; regression=:constant, lags=:aic)
report(result)
```

**Recipe 3: LM unit root with 2 breaks**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Simulated series with two level shifts
y = vcat(cumsum(randn(100)), 5.0 .+ cumsum(randn(100)), cumsum(randn(100)))

# LM test — breaks are included under the null, so rejection is unambiguous
result = lm_unitroot_test(y; breaks=2, regression=:level)
report(result)
```

---

## Fourier ADF Test

The Fourier ADF test (Enders & Lee 2012) augments the standard Augmented Dickey-Fuller regression with low-frequency trigonometric terms that approximate smooth structural change of unknown form. Unlike the Zivot-Andrews test, which models a single abrupt break, the Fourier ADF captures gradual shifts in the intercept or trend without requiring the researcher to specify the number, dates, or functional form of structural changes.

The test regression is:

```math
\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + a_k \sin\!\left(\frac{2\pi k t}{T}\right) + b_k \cos\!\left(\frac{2\pi k t}{T}\right) + \sum_{j=1}^{p} \delta_j \Delta y_{t-j} + \varepsilon_t
```

where:
- ``\gamma`` is the coefficient of interest (``H_0: \gamma = 0`` implies a unit root)
- ``k`` is the Fourier frequency, selected from ``k = 1, \ldots, k_{\max}`` to minimize the sum of squared residuals
- ``a_k`` and ``b_k`` are the Fourier coefficients on the sine and cosine terms
- ``\alpha`` is the intercept (always included)
- ``\beta t`` is the linear trend (included when `regression=:trend`)
- ``p`` augmenting lags absorb serial correlation

The optimal frequency ``k`` is determined by fitting the test regression at each candidate frequency and selecting the one that minimizes the residual sum of squares. A joint F-test for ``H_0: a_k = b_k = 0`` assesses whether the Fourier terms contribute significantly. When the F-test fails to reject, the standard ADF test without Fourier terms is more appropriate.

The null and alternative hypotheses are:
- ``H_0: \gamma = 0`` (unit root)
- ``H_1: \gamma < 0`` (stationary around a smooth deterministic function)

!!! note "Technical Note"
    A single Fourier frequency can approximate a wide range of smooth structural changes, including gradual level shifts, slow trend changes, and multiple smooth breaks. Enders & Lee (2012) show that ``k_{\max} = 3`` captures virtually all empirically relevant break patterns. Setting ``k_{\max}`` too high wastes degrees of freedom and reduces power against the unit root null.

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
cpi = filter(isfinite, to_vector(fred[:, "CPIAUCSL"]))

# Fourier ADF with automatic frequency selection
result = fourier_adf_test(cpi; regression=:constant, fmax=3)
report(result)

# Check whether Fourier terms are needed
result.f_statistic   # F-test statistic for joint significance of sin/cos terms
result.f_pvalue      # p-value of the F-test
result.frequency     # Optimal Fourier frequency k selected
```

### Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `regression` | `Symbol` | `:constant` | Deterministic terms: `:constant` or `:trend` |
| `fmax` | `Int` | `3` | Maximum Fourier frequency to search over (1 to 5) |
| `lags` | `Union{Int,Symbol}` | `:aic` | Number of augmenting lags, or `:aic`/`:bic`/`:hqic` for automatic selection |
| `max_lags` | `Union{Int,Nothing}` | `nothing` | Maximum lags for automatic selection (defaults to ``\lfloor 12(T/100)^{0.25} \rfloor``) |
| `trim` | `Real` | `0.15` | Trimming fraction for endpoint exclusion |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | ADF test statistic (t-ratio on ``\gamma``) |
| `pvalue` | `T` | P-value from Fourier ADF critical value tables |
| `frequency` | `Int` | Optimal Fourier frequency ``k`` selected |
| `f_statistic` | `T` | F-test statistic for joint significance of Fourier terms |
| `f_pvalue` | `T` | P-value of the F-test |
| `lags` | `Int` | Number of augmenting lags used |
| `regression` | `Symbol` | Deterministic specification (`:constant` or `:trend`) |
| `critical_values` | `Dict{Int,T}` | Critical values at 1%, 5%, 10% significance levels |
| `f_critical_values` | `Dict{Int,T}` | F-test critical values at 1%, 5%, 10% significance levels |
| `nobs` | `Int` | Number of observations used |

### Interpretation

**Reject** ``H_0`` (p-value < 0.05): the series is stationary around a smooth deterministic function, and the Fourier terms capture the structural change. **Fail to reject** ``H_0`` (p-value > 0.05): cannot rule out a unit root. Always check the F-test for Fourier terms: if `f_pvalue` exceeds 0.05, the Fourier terms are not statistically significant and the standard ADF test provides a more powerful alternative.

---

## Fourier KPSS Test

The Fourier KPSS test (Becker, Enders & Lee 2006) extends the KPSS stationarity test by incorporating Fourier terms to approximate smooth structural shifts in the deterministic components. Under the standard KPSS test, an unmodeled smooth break in the mean or trend inflates the partial sums of residuals, causing spurious rejection of the stationarity null. The Fourier KPSS absorbs these smooth changes through trigonometric regressors, restoring correct test size.

The test proceeds in two steps. First, regress ``y_t`` on deterministic terms augmented with Fourier components:

```math
y_t = \alpha + \beta t + a_k \sin\!\left(\frac{2\pi k t}{T}\right) + b_k \cos\!\left(\frac{2\pi k t}{T}\right) + e_t
```

where:
- ``\alpha`` is the intercept (always included)
- ``\beta t`` is the linear trend (included when `regression=:trend`)
- ``a_k, b_k`` are the Fourier coefficients at optimal frequency ``k``

Second, compute the KPSS statistic from the partial sums of the OLS residuals:

```math
\text{KPSS}_F = \frac{\sum_{t=1}^{T} S_t^2}{T^2 \hat{\sigma}^2_{LR}}
```

where:
- ``S_t = \sum_{s=1}^{t} \hat{e}_s`` are the partial sums of residuals
- ``\hat{\sigma}^2_{LR}`` is the long-run variance estimated with a Bartlett kernel

The null and alternative hypotheses are:
- ``H_0: \sigma_u^2 = 0`` (stationarity around a smooth deterministic function)
- ``H_1: \sigma_u^2 > 0`` (unit root)

An F-test for ``H_0: a_k = b_k = 0`` tests whether the Fourier terms are needed. If the Fourier terms are not significant, the standard KPSS test is preferable.

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
cpi = filter(isfinite, to_vector(fred[:, "CPIAUCSL"]))

# Fourier KPSS — stationarity test robust to smooth breaks
result = fourier_kpss_test(cpi; regression=:constant, fmax=3)
report(result)

# Verify that Fourier terms are needed
result.f_statistic   # F-test for sin/cos joint significance
result.f_pvalue      # If > 0.05, use standard KPSS instead
result.frequency     # Optimal frequency selected
```

### Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `regression` | `Symbol` | `:constant` | Stationarity type: `:constant` (level) or `:trend` (trend) |
| `fmax` | `Int` | `3` | Maximum Fourier frequency to search over (1 to 3) |
| `bandwidth` | `Union{Int,Nothing}` | `nothing` | Bartlett kernel bandwidth, or `nothing` for automatic Newey-West selection |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | Fourier KPSS test statistic |
| `pvalue` | `T` | P-value from Fourier KPSS critical value tables |
| `frequency` | `Int` | Optimal Fourier frequency ``k`` selected |
| `f_statistic` | `T` | F-test statistic for joint significance of Fourier terms |
| `f_pvalue` | `T` | P-value of the F-test |
| `regression` | `Symbol` | Stationarity type (`:constant` or `:trend`) |
| `critical_values` | `Dict{Int,T}` | Critical values at 1%, 5%, 10% significance levels |
| `f_critical_values` | `Dict{Int,T}` | F-test critical values at 1%, 5%, 10% significance levels |
| `bandwidth` | `Int` | Bartlett kernel bandwidth used |
| `nobs` | `Int` | Number of observations |

### Interpretation

**Reject** ``H_0`` (statistic exceeds critical value, p-value < 0.05): the series has a unit root even after accounting for smooth structural change. **Fail to reject** ``H_0`` (p-value > 0.05): the series is stationary around a smooth deterministic path. As with the standard KPSS test, this test reverses the burden of proof relative to the Fourier ADF: failure to reject is the favorable outcome for stationary data.

### Combining Fourier ADF and Fourier KPSS

The Fourier ADF and Fourier KPSS tests form a complementary pair analogous to the standard ADF-KPSS combination:

| Fourier ADF | Fourier KPSS | Conclusion |
|-------------|--------------|------------|
| Reject (stationary) | Fail to reject (stationary) | **Stationary** around smooth deterministic path |
| Fail to reject (unit root) | Reject (unit root) | **Unit root** present despite smooth break adjustment |
| Reject | Reject | Conflicting -- possible sharp break or misspecified frequency |
| Fail to reject | Fail to reject | Inconclusive |

---

## DF-GLS / ERS Test

The DF-GLS test (Elliott, Rothenberg & Stock 1996) applies GLS detrending to the data before running an ADF-type regression, yielding substantially higher power against local alternatives than the standard ADF test. The same detrended series produces the ERS point-optimal ``P_t`` statistic and the Ng & Perron (2001) ``M^{GLS}`` statistics, making `dfgls_test` a comprehensive power-optimized unit root testing function.

The GLS detrending procedure quasi-differences the data using a local-to-unity parameter:

```math
\tilde{y}_1 = y_1, \qquad \tilde{y}_t = y_t - \bar{\alpha} \, y_{t-1}, \quad t = 2, \ldots, T
```

where:
- ``\bar{\alpha} = 1 + \bar{c}/T``
- ``\bar{c} = -7`` for level stationarity (`regression=:constant`)
- ``\bar{c} = -13.5`` for trend stationarity (`regression=:trend`)

The deterministic regressors ``Z`` (intercept, or intercept plus trend) undergo the same quasi-differencing. GLS coefficients ``\hat{\delta}`` are estimated by regressing ``\tilde{y}`` on ``\tilde{Z}``, and the detrended series is:

```math
y_t^d = y_t - Z_t \hat{\delta}
```

The DF-GLS statistic is the t-ratio on the lagged level in the ADF regression of the detrended series (without an intercept or trend, since these have been removed by GLS detrending):

```math
\Delta y_t^d = \gamma \, y_{t-1}^d + \sum_{j=1}^{p} \delta_j \, \Delta y_{t-j}^d + \varepsilon_t
```

The ERS point-optimal ``P_t`` statistic provides an alternative test based on the ratio of restricted and unrestricted sum of squared residuals:

```math
P_t = \frac{S(\bar{\alpha}) - \bar{\alpha} \, S(1)}{s^2_{AR}}
```

where ``S(\bar{\alpha})`` and ``S(1)`` are the sum of squared residuals from quasi-differenced regressions at ``\bar{\alpha}`` and 1, and ``s^2_{AR}`` is the autoregressive spectral density estimator at frequency zero.

!!! note "Technical Note"
    The function also computes the four Ng-Perron (2001) ``M^{GLS}`` statistics -- ``MZ_\alpha``, ``MZ_t``, ``MSB``, and ``MP_T`` -- from the same GLS-detrended series. These are identical to the statistics reported by `ngperron_test` but computed on the DF-GLS detrended data, providing a unified set of power-optimized unit root diagnostics in a single function call.

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
cpi = filter(isfinite, to_vector(fred[:, "CPIAUCSL"]))

# DF-GLS test with AIC lag selection
result = dfgls_test(cpi; regression=:constant, lags=:aic)
report(result)

# Access the DF-GLS tau statistic
result.statistic         # DF-GLS tau
result.pvalue            # P-value for DF-GLS tau

# Access the ERS point-optimal statistic
result.pt_statistic      # ERS Pt
result.pt_pvalue         # P-value for Pt

# Access MGLS statistics
result.MZa               # Modified Zα
result.MZt               # Modified Zt
result.MSB               # Modified Sargan-Bhargava
result.MPT               # Modified point-optimal
```

### Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `regression` | `Symbol` | `:constant` | Deterministic terms: `:constant` or `:trend` |
| `lags` | `Union{Int,Symbol}` | `:aic` | Number of augmenting lags, or `:aic`/`:bic`/`:hqic` for automatic selection |
| `max_lags` | `Union{Int,Nothing}` | `nothing` | Maximum lags for automatic selection (defaults to ``\lfloor 12(T/100)^{0.25} \rfloor``) |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | DF-GLS ``\tau`` statistic (t-ratio on ``\gamma``) |
| `pvalue` | `T` | P-value for the DF-GLS statistic |
| `pt_statistic` | `T` | ERS point-optimal ``P_t`` statistic |
| `pt_pvalue` | `T` | P-value for the ``P_t`` statistic |
| `MZa` | `T` | Ng-Perron modified ``Z_\alpha`` (GLS-detrended) |
| `MZt` | `T` | Ng-Perron modified ``Z_t`` (GLS-detrended) |
| `MSB` | `T` | Ng-Perron modified Sargan-Bhargava (GLS-detrended) |
| `MPT` | `T` | Ng-Perron modified point-optimal (GLS-detrended) |
| `lags` | `Int` | Number of augmenting lags used |
| `regression` | `Symbol` | Deterministic specification (`:constant` or `:trend`) |
| `critical_values` | `Dict{Int,T}` | DF-GLS ``\tau`` critical values at 1%, 5%, 10% |
| `pt_critical_values` | `Dict{Int,T}` | ERS ``P_t`` critical values at 1%, 5%, 10% |
| `mgls_critical_values` | `Dict{Symbol,Dict{Int,T}}` | ``M^{GLS}`` critical values keyed by statistic name (`:MZa`, `:MZt`, `:MSB`, `:MPT`) |
| `nobs` | `Int` | Number of observations used |

### Interpretation

**Reject** ``H_0`` (p-value < 0.05): the series is stationary. The DF-GLS test has near-optimal power against local alternatives of the form ``\rho = 1 + \bar{c}/T``, making it the preferred test when the question is borderline. **Fail to reject** (p-value > 0.05): cannot reject the unit root null. When both the DF-GLS ``\tau`` and the ERS ``P_t`` fail to reject, the evidence for a unit root is particularly strong. When results diverge, the ``MZ_t`` statistic serves as a tiebreaker.

---

## LM Unit Root Test

The LM unit root test (Schmidt & Phillips 1992; Lee & Strazicich 2003, 2013) takes a fundamentally different approach to structural breaks compared to the Zivot-Andrews and Narayan-Popp tests. The key innovation is that breaks are incorporated under the null hypothesis. In the Zivot-Andrews framework, breaks appear only under the alternative, so rejection could reflect either stationarity or a break in a unit root process. The LM test resolves this ambiguity: rejection unambiguously implies stationarity, regardless of whether structural breaks are present.

Three variants handle different numbers of breaks:

**No breaks** (Schmidt & Phillips 1992, `breaks=0`): The basic LM unit root test without structural change.

**One break** (Lee & Strazicich 2013, `breaks=1`): A grid search over a single break date ``T_{B1}``.

**Two breaks** (Lee & Strazicich 2003, `breaks=2`): A grid search over two break dates ``(T_{B1}, T_{B2})``.

The LM test regression for the general two-break case with level shifts (`:level`) is:

```math
\Delta y_t = d' \Delta Z_t + \phi \tilde{S}_{t-1} + \sum_{j=1}^{p} \delta_j \Delta y_{t-j} + \varepsilon_t
```

where:
- ``\tilde{S}_t = y_t - \tilde{\psi}_x - Z_t \tilde{\delta}`` is the detrended series under the null
- ``Z_t`` includes the intercept and break dummies ``DU_{it} = \mathbf{1}(t \geq T_{Bi} + 1)`` for ``i = 1, \ldots, m``
- ``\tilde{\delta}`` are coefficients from the restricted model under ``H_0: \phi = 0``
- ``\tilde{\psi}_x = y_1 - Z_1 \tilde{\delta}`` ensures ``\tilde{S}_1 = 0``

For the combined level-and-trend model (`:both`), the break dummies include both ``DU_{it}`` (level shift) and ``DT_{it} = (t - T_{Bi}) \cdot \mathbf{1}(t \geq T_{Bi} + 1)`` (trend shift).

The null and alternative hypotheses are:
- ``H_0: \phi = 0`` (unit root with possible structural breaks)
- ``H_1: \phi < 0`` (trend-stationary with possible structural breaks)

The test statistic is the t-ratio on ``\phi``. Break dates are selected by minimizing the test statistic over the trimmed search grid.

!!! note "Technical Note"
    The critical values for the LM test with breaks depend on the break location within the sample. Lee & Strazicich (2003, 2013) provide critical values for specific break fractions ``\lambda = T_B / T``. The implementation interpolates over tabulated values to produce accurate critical values for the estimated break locations.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Simulated series with a level shift at t=100
y = vcat(cumsum(randn(100)), 5.0 .+ cumsum(randn(100)))

# LM test with 1 endogenous break
result = lm_unitroot_test(y; breaks=1, regression=:level)
report(result)

# Access break information
result.break_dates       # Estimated break date(s)
result.break_fractions   # Break location(s) as fraction of sample

# No-break variant (Schmidt-Phillips)
result0 = lm_unitroot_test(y; breaks=0, regression=:level)
report(result0)

# Two-break variant
y3 = vcat(cumsum(randn(100)), 5.0 .+ cumsum(randn(100)), cumsum(randn(100)))
result2 = lm_unitroot_test(y3; breaks=2, regression=:level)
report(result2)
```

### Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `breaks` | `Int` | `0` | Number of structural breaks: 0, 1, or 2 |
| `regression` | `Symbol` | `:level` | Break type: `:level` (intercept shifts) or `:both` (intercept + trend shifts) |
| `lags` | `Union{Int,Symbol}` | `:aic` | Number of augmenting lags, or `:aic`/`:bic`/`:hqic` for automatic selection |
| `max_lags` | `Union{Int,Nothing}` | `nothing` | Maximum lags for automatic selection (defaults to ``\lfloor 12(T/100)^{0.25} \rfloor``) |
| `trim` | `Real` | `0.15` | Trimming fraction for break date search (excludes endpoints) |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | LM test statistic (t-ratio on ``\phi``) |
| `pvalue` | `T` | P-value from LM unit root critical value tables |
| `breaks` | `Int` | Number of structural breaks (0, 1, or 2) |
| `break_dates` | `Vector{Int}` | Estimated break date(s) as observation indices (empty if `breaks=0`) |
| `break_fractions` | `Vector{T}` | Break location(s) as fraction of sample (empty if `breaks=0`) |
| `lags` | `Int` | Number of augmenting lags used |
| `regression` | `Symbol` | Break specification (`:level` or `:both`) |
| `critical_values` | `Dict{Int,T}` | Critical values at 1%, 5%, 10% significance levels |
| `nobs` | `Int` | Number of observations used |

### Interpretation

**Reject** ``H_0`` (p-value < 0.05): the series is stationary. Because structural breaks are included under the null hypothesis, this conclusion holds regardless of whether the data contains breaks. This is the central advantage over the Zivot-Andrews test, where rejection could stem from either stationarity or a break in a unit root process. **Fail to reject** (p-value > 0.05): cannot reject a unit root with structural breaks. The estimated break dates are consistent estimates of the true break locations under both the null and alternative hypotheses.

---

## Two-Break ADF Test

The two-break ADF test (Narayan & Popp 2010) extends the standard ADF framework to allow for two endogenous structural breaks. While the Zivot-Andrews test accommodates a single break, many macroeconomic series exhibit multiple regime changes -- for example, the Great Moderation and the 2008 financial crisis. The Narayan-Popp test searches over all admissible pairs of break dates and selects the combination that yields the strongest evidence against the unit root null.

Two model specifications control the form of structural change:

**Level shifts only** (`:level`):

```math
\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \theta_1 DU_{1t} + \theta_2 DU_{2t} + \sum_{j=1}^{p} \delta_j \Delta y_{t-j} + \varepsilon_t
```

**Level and trend shifts** (`:both`):

```math
\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \theta_1 DU_{1t} + \theta_2 DU_{2t} + \phi_1 DT_{1t} + \phi_2 DT_{2t} + \sum_{j=1}^{p} \delta_j \Delta y_{t-j} + \varepsilon_t
```

where:
- ``DU_{it} = \mathbf{1}(t > T_{Bi})`` is the level shift dummy for break ``i``
- ``DT_{it} = (t - T_{Bi}) \cdot \mathbf{1}(t > T_{Bi})`` is the trend shift dummy for break ``i``
- ``T_{B1}`` and ``T_{B2}`` are the two break dates, estimated by minimizing the ADF t-statistic on ``\gamma``
- ``\gamma`` is the coefficient of interest (``H_0: \gamma = 0``)

The test statistic is the minimum t-ratio on ``\gamma`` over all admissible ``(T_{B1}, T_{B2})`` pairs within the trimmed sample. Critical values depend on the sample size and are interpolated from Narayan & Popp (2010, Tables 3--4).

!!! note "Technical Note"
    The grid search over two break dates is computationally intensive: with trimming parameter ``\tau``, the number of candidate pairs is ``O((T(1 - 2\tau))^2)``. The default trimming ``\tau = 0.10`` excludes the first and last 10% of the sample. The minimum gap between break dates is 2 observations for `:level` and 3 for `:both` to ensure identification of the break parameters.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Simulated series with two level shifts
y = vcat(cumsum(randn(80)), 3.0 .+ cumsum(randn(80)), -2.0 .+ cumsum(randn(80)))

# Two-break ADF test with level shifts
result = adf_2break_test(y; model=:level, lags=:aic)
report(result)

# Access break information
result.break1            # First break date (observation index)
result.break2            # Second break date (observation index)
result.break1_fraction   # First break as fraction of sample
result.break2_fraction   # Second break as fraction of sample

# Level + trend shifts
result_both = adf_2break_test(y; model=:both, lags=:aic)
report(result_both)
```

### Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `model` | `Symbol` | `:level` | Break type: `:level` (intercept shifts) or `:both` (intercept + trend shifts) |
| `lags` | `Union{Int,Symbol}` | `:aic` | Number of augmenting lags, or `:aic`/`:bic`/`:hqic` for automatic selection |
| `max_lags` | `Union{Int,Nothing}` | `nothing` | Maximum lags for automatic selection (defaults to ``\lfloor 12(T/100)^{0.25} \rfloor``) |
| `trim` | `Real` | `0.10` | Trimming fraction for break date search (excludes endpoints) |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | Minimum ADF t-statistic over all candidate break date pairs |
| `pvalue` | `T` | P-value from Narayan-Popp critical value tables |
| `break1` | `Int` | First estimated break date (observation index) |
| `break2` | `Int` | Second estimated break date (observation index) |
| `break1_fraction` | `T` | First break location as fraction of sample |
| `break2_fraction` | `T` | Second break location as fraction of sample |
| `lags` | `Int` | Number of augmenting lags used |
| `model` | `Symbol` | Break specification (`:level` or `:both`) |
| `critical_values` | `Dict{Int,T}` | Critical values at 1%, 5%, 10% significance levels |
| `nobs` | `Int` | Number of observations used |

### Interpretation

**Reject** ``H_0`` (p-value < 0.05): the series is stationary around a deterministic path with two structural breaks. **Fail to reject** (p-value > 0.05): cannot rule out a unit root. As with the Zivot-Andrews test, breaks are modeled under the alternative hypothesis, so rejection could in principle reflect either stationarity or the presence of breaks in a unit root process. When this ambiguity is a concern, the LM unit root test (`lm_unitroot_test`) provides a cleaner inference because it includes breaks under the null.

---

## Complete Example

This workflow applies all five advanced unit root tests to a macroeconomic series, compares results with the standard ADF test, and demonstrates how the tests complement each other for robust inference.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# ── Step 1: Load data ──────────────────────────────────────────
fred = load_example(:fred_md)
cpi = filter(isfinite, to_vector(fred[:, "CPIAUCSL"]))

# ── Step 2: Standard ADF as baseline ──────────────────────────
adf = adf_test(cpi; lags=:aic, regression=:constant)
report(adf)

# ── Step 3: Fourier ADF for smooth structural change ──────────
fadf = fourier_adf_test(cpi; regression=:constant, fmax=3)
report(fadf)

# Check whether Fourier terms contribute
println("Fourier F-test: F=$(round(fadf.f_statistic, digits=3)), p=$(round(fadf.f_pvalue, digits=3))")

# ── Step 4: Fourier KPSS as complementary stationarity test ──
fkpss = fourier_kpss_test(cpi; regression=:constant, fmax=3)
report(fkpss)

# ── Step 5: DF-GLS for maximum power ─────────────────────────
dfgls = dfgls_test(cpi; regression=:constant, lags=:aic)
report(dfgls)

# ── Step 6: LM test with 1 break (breaks under H0) ──────────
lm1 = lm_unitroot_test(cpi; breaks=1, regression=:level)
report(lm1)
println("Break at observation: ", lm1.break_dates[1])

# ── Step 7: Two-break ADF ───────────────────────────────────
adf2 = adf_2break_test(cpi; model=:level, lags=:aic)
report(adf2)
println("Breaks at observations: ", adf2.break1, ", ", adf2.break2)

# ── Step 8: Synthesis ───────────────────────────────────────
println("\n── Summary ──────────────────────────────────────────────")
for (name, pval) in [("ADF", adf.pvalue), ("Fourier ADF", fadf.pvalue),
                      ("DF-GLS", dfgls.pvalue), ("LM (1 break)", lm1.pvalue),
                      ("ADF 2-break", adf2.pvalue)]
    status = pval < 0.05 ? "Reject H0 (stationary)" : "Fail to reject H0 (unit root)"
    println("  $name: $status")
end
fkpss_status = fkpss.pvalue < 0.05 ? "Reject H0 (unit root)" : "Fail to reject H0 (stationary)"
println("  Fourier KPSS: $fkpss_status")
```

---

## Common Pitfalls

1. **Fourier frequency selection -- keep fmax low.** Setting `fmax` higher than 3 rarely improves the Fourier ADF or KPSS tests and wastes degrees of freedom. Enders & Lee (2012) demonstrate that a single low-frequency Fourier component (``k = 1`` or ``k = 2``) approximates most empirically relevant smooth break patterns. With `fmax=5`, the search space expands but the additional frequencies capture noise rather than structural change, reducing power against the unit root null.

2. **DF-GLS oversizing with large negative MA root.** The GLS detrending that gives the DF-GLS test its power advantage can backfire when the error process has a large negative moving average root (e.g., ``\theta < -0.8``). In this case, the DF-GLS test rejects the unit root null too often in finite samples (Perron & Ng 1996). When ADF and DF-GLS disagree and you suspect MA contamination, the standard ADF test is more reliable. The ``MZ_t`` statistic from `dfgls_test` is less sensitive to this problem than the DF-GLS ``\tau``.

3. **LM test break model specification.** The `:level` model allows only intercept shifts, while `:both` allows both level and trend shifts. Using `:both` when only level shifts are present reduces power because the test estimates unnecessary trend-break parameters. More importantly, the LM test includes breaks under ``H_0``, so rejection unambiguously implies stationarity. This contrasts with the Zivot-Andrews and Narayan-Popp tests, where breaks appear under ``H_1`` and rejection could reflect either stationarity or a break in a unit root process. When the goal is clean inference, the LM test is the safer choice.

4. **Two-break ADF trimming and endpoint instability.** The default `trim=0.10` excludes the first and last 10% of the sample from the break search. Reducing the trimming parameter below 0.10 allows breaks near the endpoints, but the parameter estimates become unreliable with few observations on either side of the break. With short samples (``T < 100``), increase the trimming to 0.15 to maintain estimation accuracy. Also verify that the estimated break dates are not clustered at the trimming boundary, which suggests the true break may lie outside the search region.

---

## References

- Becker, Ralf, Walter Enders, and Junsoo Lee. 2006. "A Stationarity Test in the Presence of an Unknown Number of Smooth Breaks." *Journal of Time Series Analysis* 27 (3): 381--409. [https://doi.org/10.1111/j.1467-9892.2006.00478.x](https://doi.org/10.1111/j.1467-9892.2006.00478.x)
- Elliott, Graham, Thomas J. Rothenberg, and James H. Stock. 1996. "Efficient Tests for an Autoregressive Unit Root." *Econometrica* 64 (4): 813--836. [https://doi.org/10.2307/2171846](https://doi.org/10.2307/2171846)
- Enders, Walter, and Junsoo Lee. 2012. "A Unit Root Test Using a Fourier Series to Approximate Smooth Breaks." *Oxford Bulletin of Economics and Statistics* 74 (4): 574--599. [https://doi.org/10.1111/j.1468-0084.2011.00662.x](https://doi.org/10.1111/j.1468-0084.2011.00662.x)
- Lee, Junsoo, and Mark C. Strazicich. 2003. "Minimum Lagrange Multiplier Unit Root Test with Two Structural Breaks." *Review of Economics and Statistics* 85 (4): 1082--1089. [https://doi.org/10.1162/003465303772815961](https://doi.org/10.1162/003465303772815961)
- Lee, Junsoo, and Mark C. Strazicich. 2013. "Minimum LM Unit Root Test with One Structural Break." *Economics Bulletin* 33 (4): 2483--2492.
- Narayan, Paresh Kumar, and Stephan Popp. 2010. "A New Unit Root Test with Two Structural Breaks in Level and Slope at Unknown Time." *Journal of Applied Statistics* 37 (9): 1425--1438. [https://doi.org/10.1080/02664760903039883](https://doi.org/10.1080/02664760903039883)
- Ng, Serena, and Pierre Perron. 2001. "Lag Length Selection and the Construction of Unit Root Tests with Good Size and Power." *Econometrica* 69 (6): 1519--1554. [https://doi.org/10.1111/1468-0262.00256](https://doi.org/10.1111/1468-0262.00256)
- Perron, Pierre, and Serena Ng. 1996. "Useful Modifications to Some Unit Root Tests with Dependent Errors and Their Local Asymptotic Properties." *Review of Economic Studies* 63 (3): 435--463. [https://doi.org/10.2307/2297890](https://doi.org/10.2307/2297890)
- Schmidt, Peter, and Peter C. B. Phillips. 1992. "LM Tests for a Unit Root in the Presence of Deterministic Trends." *Oxford Bulletin of Economics and Statistics* 54 (3): 257--287. [https://doi.org/10.1111/j.1468-0084.1992.tb00002.x](https://doi.org/10.1111/j.1468-0084.1992.tb00002.x)
