# Panel Tests

Panel-level hypothesis testing addresses two distinct phases of the empirical workflow. **Panel unit root tests** detect non-stationarity in datasets with cross-sectional dependence, a prerequisite for correct specification of panel VARs and factor models. **Panel VAR specification tests** validate GMM instrument validity and select optimal lag orders after estimation. This page covers three second-generation panel unit root tests (PANIC, Pesaran CIPS, Moon-Perron) and three Panel VAR diagnostics (Hansen J-test, Andrews-Lu MMSC, lag selection).

- PANIC: factor-based decomposition into common and idiosyncratic components (Bai & Ng 2004, 2010)
- Pesaran CIPS: cross-sectionally augmented IPS test robust to common factors (Pesaran 2007)
- Moon-Perron: factor-adjusted pooled AR(1) with bias correction (Moon & Perron 2004)
- Hansen J-test, Andrews-Lu MMSC, and MMSC-based lag selection for Panel VAR

## Quick Start

**Recipe 1: PANIC test on a panel**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Stationary panel with one common factor
X = randn(100, 20)
result = panic_test(X; r=1)
result.pooled_pvalue  # small p-value rejects unit root null
```

**Recipe 2: Pesaran CIPS test**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

X = randn(50, 20)
result = pesaran_cips_test(X; lags=1, deterministic=:constant)
result.cips_statistic
result.pvalue
```

**Recipe 3: Panel VAR Hansen J-test and lag selection**

```julia
using MacroEconometricModels, DataFrames, Random
Random.seed!(42)
N, T_total, m_vars = 50, 20, 3
data = zeros(N * T_total, m_vars)
for i in 1:N
    mu = randn(m_vars) * 0.5
    for t in 2:T_total
        idx = (i-1)*T_total + t
        data[idx, :] = mu + 0.5 * data[(i-1)*T_total + t - 1, :] + 0.2 * randn(m_vars)
    end
end
df = DataFrame(data, ["y1", "y2", "y3"])
df.id = repeat(1:N, inner=T_total)
df.time = repeat(1:T_total, outer=N)
pd = xtset(df, :id, :time)
model = estimate_pvar(pd, 2; steps=:twostep)
j = pvar_hansen_j(model)
j.pvalue  # p > 0.05 indicates valid instruments
```

---

## PANIC

The **PANIC** (Panel Analysis of Nonstationarity in Idiosyncratic and Common components) test of Bai & Ng (2004, 2010) decomposes panel data into common factors and idiosyncratic residuals via principal components, then tests each component for unit roots separately. This separation is critical because standard panel unit root tests (IPS, LLC) assume cross-sectional independence, an assumption violated in macroeconomic panels where global shocks affect all units.

The panel follows a factor structure:

```math
X_{it} = \lambda_i' F_t + e_{it}
```

where:
- ``X_{it}`` is the observation for unit ``i`` at time ``t``
- ``F_t`` is the ``r \times 1`` vector of common factors
- ``\lambda_i`` is the ``r \times 1`` vector of unit-specific loadings
- ``e_{it}`` is the idiosyncratic error

The test proceeds in three steps. First, estimate ``\hat{F}_t`` and ``\hat{\lambda}_i`` via PCA on the ``T \times N`` panel. Second, run ADF tests on each estimated factor ``\hat{F}_{j,t}`` to determine whether the common components are I(1). Third, run ADF tests on the defactored residuals ``\hat{e}_{it}`` and pool the individual p-values into a standardized statistic:

```math
P_a = \frac{\sum_{i=1}^N p_i - N/2}{\sqrt{N/12}} \xrightarrow{d} N(0,1)
```

where:
- ``p_i`` is the ADF p-value for unit ``i``'s idiosyncratic residual
- ``N`` is the number of cross-sectional units
- The standardization uses the mean (``1/2``) and variance (``1/12``) of the uniform distribution

Under ``H_0`` (all idiosyncratic components have unit roots), the individual p-values are approximately uniform, and ``P_a`` converges to a standard normal. Large negative values of ``P_a`` indicate rejection: many units have small ADF p-values, providing evidence of idiosyncratic stationarity.

!!! note "Technical Note"
    When `r=:auto`, the number of factors is selected using the Bai-Ng (2002) IC2 information criterion via `ic_criteria`. The defactored residuals are tested with ADF regressions without deterministic terms, since the common factor extraction already absorbs any trends.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Stationary panel: idiosyncratic components are I(0)
X_stationary = randn(100, 20)
result_s = panic_test(X_stationary; r=1)
result_s.pooled_pvalue  # expect small p-value (reject unit root)

# I(1) panel: cumulate to create unit roots
X_nonstat = cumsum(randn(100, 20), dims=1)
result_ns = panic_test(X_nonstat; r=1)
result_ns.pooled_pvalue  # expect large p-value (fail to reject)
```

The stationary panel produces a small pooled p-value, indicating that the defactored residuals are stationary. The I(1) panel produces a large p-value, consistent with the unit root null. The factor ADF statistics in `result.factor_adf_stats` reveal whether the common component itself is non-stationary.

### Options

| Argument | Description | Default |
|----------|-------------|---------|
| `r` | Number of common factors, or `:auto` for IC-based selection | `:auto` |
| `method` | Pooling method: `:pooled` (standardized p-value sum) or `:individual` (unit-by-unit) | `:pooled` |

### PANICResult Return Values

| Field | Type | Description |
|-------|------|-------------|
| `factor_adf_stats` | `Vector{T}` | ADF test statistics for each estimated common factor |
| `factor_adf_pvalues` | `Vector{T}` | ADF p-values for each common factor |
| `pooled_statistic` | `T` | Standardized pooled statistic ``P_a`` |
| `pooled_pvalue` | `T` | P-value for the pooled statistic (standard normal) |
| `individual_stats` | `Vector{T}` | Individual ADF statistics for each unit's idiosyncratic residual |
| `individual_pvalues` | `Vector{T}` | Individual ADF p-values for each unit |
| `n_factors` | `Int` | Number of common factors used |
| `method` | `Symbol` | Pooling method (`:pooled` or `:individual`) |
| `nobs` | `Int` | Number of time observations ``T`` |
| `n_units` | `Int` | Number of cross-sectional units ``N`` |

The `PanelData` dispatch `panic_test(pd::PanelData; kwargs...)` extracts the first variable from each group, constructs a balanced ``T \times N`` matrix, and passes it to the matrix method.

---

## Pesaran CIPS

The **Cross-sectionally Augmented IPS** (CIPS) test of Pesaran (2007) augments individual ADF regressions with cross-sectional averages to account for dependence driven by a single unobserved common factor. Unlike PANIC, this test does not require explicit factor estimation: the cross-section average serves as a proxy for the common factor.

The **Cross-sectionally Augmented Dickey-Fuller** (CADF) regression for unit ``i`` is:

```math
\Delta y_{it} = a_i + b_i y_{i,t-1} + c_i \bar{y}_{t-1} + d_i \Delta\bar{y}_t + \sum_{j=1}^{p} \phi_j \Delta y_{i,t-j} + \varepsilon_{it}
```

where:
- ``\Delta y_{it} = y_{it} - y_{i,t-1}`` is the first difference
- ``b_i`` is the coefficient of interest; ``H_0: b_i = 0`` for all ``i``
- ``\bar{y}_t = N^{-1}\sum_{i=1}^{N} y_{it}`` is the cross-sectional average
- ``a_i`` is a unit-specific intercept (when `deterministic=:constant`)
- ``p`` is the number of augmenting lags

The CIPS statistic averages the individual CADF t-statistics with truncation:

```math
\text{CIPS} = N^{-1} \sum_{i=1}^{N} \tilde{t}_i
```

where ``\tilde{t}_i = \max(-6.19, \min(6.19, t_i))`` are the truncated CADF statistics. The truncation at ``\pm 6.19`` ensures that the CIPS statistic has finite moments even when some individual regressions produce extreme t-values.

Critical values depend on ``N``, ``T``, and the deterministic specification. The test rejects when the CIPS statistic falls below the critical value (left-tailed test).

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Stationary panel
X = randn(50, 20)
result_const = pesaran_cips_test(X; lags=1, deterministic=:constant)
result_const.cips_statistic
result_const.critical_values  # Dict with 1%, 5%, 10% critical values

# With trend deterministic
result_trend = pesaran_cips_test(X; lags=1, deterministic=:trend)
result_trend.pvalue
```

A CIPS statistic below the 5% critical value rejects the null of a panel unit root. The `critical_values` dictionary provides thresholds at the 1%, 5%, and 10% levels for the nearest tabulated ``(N, T)`` combination.

### Options

| Argument | Description | Default |
|----------|-------------|---------|
| `lags` | Number of augmenting lags, or `:auto` for ``\lfloor T^{1/3} \rfloor`` rule | `:auto` |
| `deterministic` | Deterministic terms: `:none`, `:constant`, or `:trend` | `:constant` |

### PesaranCIPSResult Return Values

| Field | Type | Description |
|-------|------|-------------|
| `cips_statistic` | `T` | CIPS test statistic (average of truncated CADF t-statistics) |
| `pvalue` | `T` | P-value interpolated from Pesaran (2007) critical value tables |
| `individual_cadf_stats` | `Vector{T}` | Individual CADF t-statistics (before truncation) |
| `critical_values` | `Dict{Int,T}` | Critical values at 1%, 5%, 10% significance levels |
| `lags` | `Int` | Number of augmenting lags used |
| `deterministic` | `Symbol` | Deterministic specification (`:none`, `:constant`, `:trend`) |
| `nobs` | `Int` | Number of time observations ``T`` |
| `n_units` | `Int` | Number of cross-sectional units ``N`` |

---

## Moon-Perron

The Moon & Perron (2004) test constructs **factor-adjusted** panel unit root statistics by projecting out estimated common factors from the data and then pooling unit-level AR(1) regressions. The approach is semi-parametric: it estimates the factor space via PCA but does not require parametric assumptions on the factor dynamics.

The test produces two modified t-statistics:

```math
t_a^* = \frac{\sqrt{N}\, T\, (\hat{\rho}_{pool} - 1) - B_a}{S_a}, \quad
t_b^* = \frac{\sqrt{N}\, \bar{t} - B_b}{S_b}
```

where:
- ``\hat{\rho}_{pool}`` is the pooled AR(1) coefficient from de-factored data
- ``\bar{t}`` is the average of individual t-statistics
- ``B_a, B_b`` are bias corrections based on the ratio of long-run to short-run variance
- ``S_a, S_b`` are variance corrections using fourth moments of the long-run variance

Both statistics converge to ``N(0,1)`` under ``H_0``: all units have unit roots. The bias corrections account for the serial correlation induced by the de-factoring step. The test rejects for large negative values (left-tailed).

!!! note "Technical Note"
    The long-run variance ``\hat{\omega}_i^2`` for each unit is estimated using a Bartlett kernel with Newey-West bandwidth. The projection matrix ``Q_\perp = I_N - \hat{\Lambda}(\hat{\Lambda}'\hat{\Lambda})^{-1}\hat{\Lambda}'`` removes the factor space from the cross-section dimension.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Stationary panel
X = randn(80, 15)
result = moon_perron_test(X; r=1)
result.t_a_statistic  # modified t*_a statistic
result.t_b_statistic  # modified t*_b statistic
result.pvalue_a       # p-value for t*_a (standard normal)
result.pvalue_b       # p-value for t*_b (standard normal)
```

Both p-values test the same null hypothesis using different pooling approaches. When both reject at the 5% level, the evidence against the panel unit root is strong. When only one rejects, the evidence is moderate.

### Options

| Argument | Description | Default |
|----------|-------------|---------|
| `r` | Number of common factors, or `:auto` for IC-based selection | `:auto` |

### MoonPerronResult Return Values

| Field | Type | Description |
|-------|------|-------------|
| `t_a_statistic` | `T` | Modified pooled coefficient statistic ``t_a^*`` |
| `t_b_statistic` | `T` | Modified pooled t-statistic ``t_b^*`` |
| `pvalue_a` | `T` | P-value for ``t_a^*`` (standard normal) |
| `pvalue_b` | `T` | P-value for ``t_b^*`` (standard normal) |
| `n_factors` | `Int` | Number of common factors used |
| `nobs` | `Int` | Number of time observations ``T`` |
| `n_units` | `Int` | Number of cross-sectional units ``N`` |

---

## Panel Unit Root Summary

The convenience function `panel_unit_root_summary` runs all three panel unit root tests (PANIC, Pesaran CIPS, Moon-Perron) and prints a consolidated report. This is the recommended entry point for pre-estimation diagnostics on panel data.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

X = randn(100, 20)
panel_unit_root_summary(X; r=1)
```

The output displays each test's specification, statistics, p-values, and conclusion in sequence. If any individual test fails (e.g., due to insufficient observations), the function prints the error and continues with the remaining tests.

---

## Panel VAR Specification Tests

After estimating a Panel VAR by GMM, three diagnostics validate the specification: the Hansen J-test for instrument validity, Andrews-Lu MMSC criteria for model selection, and MMSC-based lag order selection. These tests apply to GMM-estimated models (`estimate_pvar` with `:onestep` or `:twostep`), not to FE-OLS models.

The following data generation pattern is used throughout this section:

```julia
using MacroEconometricModels, DataFrames, Random
Random.seed!(42)
N, T_total, m_vars = 50, 20, 3
data = zeros(N * T_total, m_vars)
for i in 1:N
    mu = randn(m_vars) * 0.5
    for t in 2:T_total
        idx = (i-1)*T_total + t
        data[idx, :] = mu + 0.5 * data[(i-1)*T_total + t - 1, :] + 0.2 * randn(m_vars)
    end
end
df = DataFrame(data, ["y1", "y2", "y3"])
df.id = repeat(1:N, inner=T_total)
df.time = repeat(1:T_total, outer=N)
pd = xtset(df, :id, :time)
```

### Hansen J-Test

The Hansen (1982) J-test evaluates whether the overidentifying restrictions in a GMM model are satisfied. In Panel VAR, the instrument set (lagged levels for first-difference GMM, or lagged levels and differences for system GMM) typically exceeds the number of parameters, creating overidentifying conditions.

The J-statistic is:

```math
J = N \, \bar{g}' \, W \, \bar{g} \sim \chi^2(q - k)
```

where:
- ``\bar{g} = N^{-1} \sum_{i=1}^{N} Z_i' e_i`` is the average moment condition
- ``W`` is the optimal weighting matrix
- ``q`` is the number of instruments
- ``k`` is the number of parameters per equation
- ``q - k`` are the degrees of freedom (overidentifying restrictions)

Under ``H_0`` (all moment conditions are valid), the J-statistic follows a chi-squared distribution. Rejection indicates that some instruments are invalid, pointing to misspecification or endogeneity problems.

```julia
using MacroEconometricModels, DataFrames, Random
Random.seed!(42)
N, T_total, m_vars = 50, 20, 3
data = zeros(N * T_total, m_vars)
for i in 1:N
    mu = randn(m_vars) * 0.5
    for t in 2:T_total
        idx = (i-1)*T_total + t
        data[idx, :] = mu + 0.5 * data[(i-1)*T_total + t - 1, :] + 0.2 * randn(m_vars)
    end
end
df = DataFrame(data, ["y1", "y2", "y3"])
df.id = repeat(1:N, inner=T_total)
df.time = repeat(1:T_total, outer=N)
pd = xtset(df, :id, :time)
model = estimate_pvar(pd, 2; steps=:twostep)
j = pvar_hansen_j(model)
j.statistic  # J-statistic value
j.pvalue     # p > 0.05 indicates valid instruments
j.df         # degrees of freedom (q - k)
```

A p-value above 0.05 means the test fails to reject instrument validity. This is a necessary but not sufficient condition: the J-test has low power when the number of instruments is large relative to the sample size.

### PVARTestResult Return Values

| Field | Type | Description |
|-------|------|-------------|
| `test_name` | `String` | Name of the test (`"Hansen J-test"`) |
| `statistic` | `T` | J-statistic value |
| `pvalue` | `T` | P-value from ``\chi^2(q-k)`` distribution |
| `df` | `Int` | Degrees of freedom (overidentifying restrictions) |
| `n_instruments` | `Int` | Number of instruments ``q`` |
| `n_params` | `Int` | Number of parameters per equation ``k`` |

### Andrews-Lu MMSC

The Andrews & Lu (2001) **Model and Moment Selection Criteria** extend information criteria to the GMM setting. They penalize the J-statistic by the number of overidentifying restrictions, enabling comparison across models with different lag orders or instrument sets.

The three criteria are:

```math
\text{MMSC-BIC}  = J - (q - k) \ln n
```

```math
\text{MMSC-AIC}  = J - 2(q - k)
```

```math
\text{MMSC-HQIC} = J - Q(q - k) \ln \ln n
```

where:
- ``J`` is the Hansen J-statistic
- ``q - k`` is the number of overidentifying restrictions
- ``n`` is the total number of observations
- ``Q = 2.1`` is the default Hannan-Quinn constant

Lower values indicate a better-fitting specification. MMSC-BIC penalizes most heavily and tends to select parsimonious models.

```julia
using MacroEconometricModels, DataFrames, Random
Random.seed!(42)
N, T_total, m_vars = 50, 20, 3
data = zeros(N * T_total, m_vars)
for i in 1:N
    mu = randn(m_vars) * 0.5
    for t in 2:T_total
        idx = (i-1)*T_total + t
        data[idx, :] = mu + 0.5 * data[(i-1)*T_total + t - 1, :] + 0.2 * randn(m_vars)
    end
end
df = DataFrame(data, ["y1", "y2", "y3"])
df.id = repeat(1:N, inner=T_total)
df.time = repeat(1:T_total, outer=N)
pd = xtset(df, :id, :time)
model = estimate_pvar(pd, 2; steps=:twostep)
mmsc = pvar_mmsc(model)
mmsc.bic   # MMSC-BIC value
mmsc.aic   # MMSC-AIC value
mmsc.hqic  # MMSC-HQIC value
```

### Lag Selection

The `pvar_lag_selection` function estimates Panel VAR models for lag orders ``p = 1, \ldots, p_{max}`` and compares them using Andrews-Lu MMSC criteria. This automates the model comparison workflow and returns the optimal lag order under each criterion.

```julia
using MacroEconometricModels, DataFrames, Random
Random.seed!(42)
N, T_total, m_vars = 50, 20, 3
data = zeros(N * T_total, m_vars)
for i in 1:N
    mu = randn(m_vars) * 0.5
    for t in 2:T_total
        idx = (i-1)*T_total + t
        data[idx, :] = mu + 0.5 * data[(i-1)*T_total + t - 1, :] + 0.2 * randn(m_vars)
    end
end
df = DataFrame(data, ["y1", "y2", "y3"])
df.id = repeat(1:N, inner=T_total)
df.time = repeat(1:T_total, outer=N)
pd = xtset(df, :id, :time)
sel = pvar_lag_selection(pd, 4)
sel.best_bic   # optimal lag by MMSC-BIC
sel.best_aic   # optimal lag by MMSC-AIC
sel.best_hqic  # optimal lag by MMSC-HQIC
```

The `table` field contains a ``p_{max} \times 4`` matrix with columns for the lag order and the three MMSC values. The `models` vector stores all estimated `PVARModel` objects for further analysis.

---

## Complete Example

This example demonstrates a full panel diagnostics workflow: test for unit roots with all three second-generation tests, estimate a Panel VAR, and validate the specification with GMM diagnostics.

```julia
using MacroEconometricModels, DataFrames, Random
Random.seed!(42)

# --- Step 1: Generate panel data with a common factor ---
N, T_obs = 30, 60
r_true = 1
F = cumsum(randn(T_obs, r_true), dims=1)          # I(1) common factor
Lambda = randn(N, r_true)                           # unit-specific loadings
e = randn(T_obs, N) * 0.5                           # stationary idiosyncratic errors
X = F * Lambda' + e                                  # T x N panel

# --- Step 2: PANIC test ---
# Decompose into factor + idiosyncratic, test each
panic_result = panic_test(X; r=1)
panic_result.pooled_pvalue
# Expect small p-value: idiosyncratic components are stationary
# Check whether the common factor is I(1):
panic_result.factor_adf_pvalues
# Expect large p-value for factor (fail to reject unit root)

# --- Step 3: Pesaran CIPS test ---
cips_result = pesaran_cips_test(X; lags=1, deterministic=:constant)
cips_result.cips_statistic
cips_result.pvalue
# CIPS tests the composite null; result depends on factor vs. idiosyncratic balance

# --- Step 4: Moon-Perron test ---
mp_result = moon_perron_test(X; r=1)
mp_result.t_a_statistic
mp_result.pvalue_a
mp_result.t_b_statistic
mp_result.pvalue_b

# --- Step 5: Run all three tests together ---
panel_unit_root_summary(X; r=1)

# --- Step 6: Panel VAR estimation and specification tests ---
# Generate a multivariate panel for PVAR
N_pvar, T_pvar, m_vars = 50, 20, 3
pvar_data = zeros(N_pvar * T_pvar, m_vars)
for i in 1:N_pvar
    mu = randn(m_vars) * 0.5
    for t in 2:T_pvar
        idx = (i-1)*T_pvar + t
        pvar_data[idx, :] = mu + 0.5 * pvar_data[(i-1)*T_pvar + t - 1, :] + 0.2 * randn(m_vars)
    end
end
df = DataFrame(pvar_data, ["y1", "y2", "y3"])
df.id = repeat(1:N_pvar, inner=T_pvar)
df.time = repeat(1:T_pvar, outer=N_pvar)
pd = xtset(df, :id, :time)

# Estimate PVAR(2) by two-step GMM
model = estimate_pvar(pd, 2; steps=:twostep)

# Hansen J-test: validate overidentifying restrictions
j = pvar_hansen_j(model)
j.pvalue  # p > 0.05 supports instrument validity

# MMSC criteria for this specification
mmsc = pvar_mmsc(model)
mmsc.bic

# Lag selection: compare p = 1, ..., 4
sel = pvar_lag_selection(pd, 4)
sel.best_bic  # optimal lag order by MMSC-BIC
```

The PANIC test separates the common factor (I(1)) from the stationary idiosyncratic errors, providing a nuanced view that the composite tests (CIPS, Moon-Perron) cannot. The Panel VAR diagnostics confirm that the GMM instruments are valid and identify the preferred lag order.

---

## Common Pitfalls

1. **Too few cross-sectional units.** Panel unit root tests rely on ``N \to \infty`` asymptotics. With ``N < 20``, all three tests (PANIC, CIPS, Moon-Perron) have reduced size and power. Use at least 20 cross-sectional units for reliable inference.

2. **Sensitivity of automatic factor selection.** The `:auto` option for `r` in PANIC and Moon-Perron uses the Bai-Ng IC2 criterion, which can be sensitive to the signal-to-noise ratio. Always compare results with different fixed values of `r` (e.g., `r=1`, `r=2`, `r=3`) to check robustness.

3. **CIPS truncation with unbalanced panels.** The CADF t-statistics are truncated at ``\pm 6.19`` to ensure finite moments. In heavily unbalanced panels where some units have very short time series, truncation binds frequently, reducing the test's discriminatory power. Ensure adequate time dimension for all units.

4. **J-test low power with many instruments.** In Panel VAR, the number of GMM instruments grows quadratically with the time dimension. When ``q \gg k``, the J-test almost never rejects, even when some instruments are invalid. Non-rejection does not validate the instrument set. Use `system_instruments=false` and shorter lag windows to keep the instrument count manageable.

---

## References

- Andrews, D. W. K., & Lu, B. (2001). Consistent Model and Moment Selection Procedures for GMM Estimation with Application to Dynamic Panel Data Models. *Journal of Econometrics*, 101(1), 123-164. [DOI](https://doi.org/10.1016/S0304-4076(00)00077-4)
- Bai, J., & Ng, S. (2002). Determining the Number of Factors in Approximate Factor Models. *Econometrica*, 70(1), 191-221. [DOI](https://doi.org/10.1111/1468-0262.00273)
- Bai, J., & Ng, S. (2004). A PANIC Attack on Unit Roots and Cointegration. *Econometrica*, 72(4), 1127-1177. [DOI](https://doi.org/10.1111/j.1468-0262.2004.00528.x)
- Bai, J., & Ng, S. (2010). Panel Unit Root Tests with Cross-Section Dependence: A Further Investigation. *Econometric Theory*, 26(4), 1088-1114. [DOI](https://doi.org/10.1017/S0266466609990478)
- Hansen, L. P. (1982). Large Sample Properties of Generalized Method of Moments Estimators. *Econometrica*, 50(4), 1029-1054. [DOI](https://doi.org/10.2307/1912775)
- Moon, H. R., & Perron, B. (2004). Testing for a Unit Root in Panels with Dynamic Factors. *Journal of Econometrics*, 122(1), 81-126. [DOI](https://doi.org/10.1016/j.jeconom.2003.10.020)
- Pesaran, M. H. (2007). A Simple Panel Unit Root Test in the Presence of Cross-Section Dependence. *Journal of Applied Econometrics*, 22(2), 265-312. [DOI](https://doi.org/10.1002/jae.951)
