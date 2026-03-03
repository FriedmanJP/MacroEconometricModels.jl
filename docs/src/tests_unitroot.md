# Unit Root & Cointegration Tests

Pre-estimation stationarity analysis determines whether a time series is stationary (I(0)) or contains a unit root (I(1)). This distinction drives the choice between VAR in levels, VAR in first differences, and VECM specifications. MacroEconometricModels.jl provides five unit root tests, a multivariate cointegration test, and convenience functions for batch analysis.

The ADF and KPSS tests are complementary: ADF tests the null of a unit root, while KPSS tests the null of stationarity. Running both provides stronger inference than either alone. When structural breaks are suspected, the Zivot-Andrews test avoids the size distortions that plague standard tests.

- **ADF, PP, Ng-Perron**: Null hypothesis is unit root
- **KPSS**: Null hypothesis is stationarity (reverses the burden of proof)
- **Zivot-Andrews**: Unit root test robust to a single endogenous structural break
- **Johansen**: Tests for cointegrating relationships among multiple I(1) series

## Quick Start

**Recipe 1: ADF + KPSS combined workflow**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
cpi = filter(isfinite, to_vector(fred[:, "CPIAUCSL"]))

# ADF: H0 = unit root
adf_result = adf_test(cpi; lags=:aic, regression=:constant)
report(adf_result)

# KPSS: H0 = stationarity
kpss_result = kpss_test(cpi; regression=:constant)
report(kpss_result)

# If ADF fails to reject and KPSS rejects → unit root confirmed
```

**Recipe 2: Johansen cointegration**

```julia
using MacroEconometricModels

qd = load_example(:fred_qd)
Y = log.(to_matrix(qd[:, ["GDPC1", "PCECC96"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

# Test for cointegrating relationships with 2 lags
result = johansen_test(Y, 2; deterministic=:constant)
report(result)
```

**Recipe 3: Batch unit root summary**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
cpi = filter(isfinite, to_vector(fred[:, "CPIAUCSL"]))

# Run ADF, KPSS, and PP simultaneously with automatic conclusion
summary = unit_root_summary(cpi; tests=[:adf, :kpss, :pp])
summary.conclusion
```

---

## Augmented Dickey-Fuller Test

The Augmented Dickey-Fuller (ADF) test (Dickey & Fuller, 1979) is the most widely used unit root test in applied macroeconometrics. It examines whether a time series contains a stochastic trend by testing the coefficient on the lagged level in a first-difference regression.

The ADF regression augments the basic Dickey-Fuller test with lagged differences to control for serial correlation:

```math
\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{j=1}^{p} \delta_j \Delta y_{t-j} + \varepsilon_t
```

where:
- ``\gamma = \rho - 1`` is the coefficient of interest (``\rho`` is the AR(1) parameter)
- ``\alpha`` is an optional intercept (included when `regression=:constant` or `:trend`)
- ``\beta t`` is an optional linear trend (included when `regression=:trend`)
- ``p`` lagged differences absorb serial correlation in ``\varepsilon_t``

The null and alternative hypotheses are:
- ``H_0: \gamma = 0`` (unit root, series is non-stationary)
- ``H_1: \gamma < 0`` (stationary)

The ADF statistic is the t-ratio on ``\gamma``:

```math
\tau = \frac{\hat{\gamma}}{\text{se}(\hat{\gamma})}
```

Critical values follow non-standard distributions and depend on the deterministic specification. MacroEconometricModels.jl computes p-values from the MacKinnon (1994, 2010) response surface approximation.

```julia
using MacroEconometricModels

# CPI price level — expected to have a unit root
fred = load_example(:fred_md)
cpi = filter(isfinite, to_vector(fred[:, "CPIAUCSL"]))

# ADF test with automatic lag selection via AIC
result = adf_test(cpi; lags=:aic, regression=:constant)
report(result)

# Access specific fields
result.statistic        # ADF τ-statistic
result.pvalue           # MacKinnon p-value
result.lags             # Number of augmenting lags selected
result.critical_values  # Dict: 1 => cv_1%, 5 => cv_5%, 10 => cv_10%
```

### Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `lags` | `Union{Int,Symbol}` | `:aic` | Number of augmenting lags, or `:aic`/`:bic`/`:hqic` for automatic selection |
| `max_lags` | `Union{Int,Nothing}` | `nothing` | Maximum lags for automatic selection (defaults to ``\lfloor 12(T/100)^{0.25} \rfloor``) |
| `regression` | `Symbol` | `:constant` | Deterministic terms: `:none`, `:constant`, or `:trend` |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | ADF test statistic (``\tau``-ratio) |
| `pvalue` | `T` | Asymptotic p-value (MacKinnon response surface) |
| `lags` | `Int` | Number of augmenting lags used |
| `regression` | `Symbol` | Deterministic specification (`:none`, `:constant`, `:trend`) |
| `critical_values` | `Dict{Int,T}` | Critical values at 1%, 5%, 10% significance levels |
| `nobs` | `Int` | Number of observations used |

### Interpretation

**Reject** ``H_0`` (p-value < 0.05): evidence against a unit root; the series appears stationary. **Fail to reject** ``H_0`` (p-value > 0.05): cannot reject the unit root null; the series is likely non-stationary and requires differencing or a VECM specification. The ADF test has notoriously low power against near-unit-root alternatives, so always confirm with the KPSS test.

---

## KPSS Stationarity Test

The KPSS test (Kwiatkowski, Phillips, Schmidt & Shin, 1992) reverses the hypotheses of the ADF test. By testing the null of stationarity, it places the burden of proof on rejecting stationarity rather than rejecting the unit root. This complementary design makes the ADF-KPSS pair a cornerstone of applied unit root analysis.

The KPSS test decomposes the observed series into a deterministic trend, a random walk, and a stationary error:

```math
y_t = \xi t + r_t + \varepsilon_t, \qquad r_t = r_{t-1} + u_t
```

where:
- ``\xi t`` is a deterministic trend (set ``\xi = 0`` for level stationarity)
- ``r_t`` is a random walk component with innovation ``u_t \sim (0, \sigma_u^2)``
- ``\varepsilon_t`` is a stationary error

Under the null ``H_0: \sigma_u^2 = 0``, the random walk component vanishes and the series is stationary. The KPSS statistic is:

```math
\text{KPSS} = \frac{\sum_{t=1}^{T} S_t^2}{T^2 \hat{\sigma}^2_{LR}}
```

where:
- ``S_t = \sum_{s=1}^{t} \hat{e}_s`` are partial sums of OLS residuals from regressing ``y_t`` on deterministic terms
- ``\hat{\sigma}^2_{LR}`` is the long-run variance estimated using a Bartlett kernel

```julia
using MacroEconometricModels

# CPI inflation rate (Δlog CPI) — expected to be stationary
fred = load_example(:fred_md)
cpi_growth = diff(log.(filter(isfinite, to_vector(fred[:, "CPIAUCSL"]))))

result = kpss_test(cpi_growth; regression=:constant)
report(result)

# Test for trend stationarity
result_trend = kpss_test(cpi_growth; regression=:trend)
report(result_trend)
```

### Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `regression` | `Symbol` | `:constant` | Stationarity type: `:constant` (level) or `:trend` (trend) |
| `bandwidth` | `Union{Int,Symbol}` | `:auto` | Bartlett kernel bandwidth, or `:auto` for Newey-West selection |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | KPSS test statistic |
| `pvalue` | `T` | Asymptotic p-value |
| `regression` | `Symbol` | Stationarity type (`:constant` or `:trend`) |
| `critical_values` | `Dict{Int,T}` | Critical values at 1%, 5%, 10% |
| `bandwidth` | `Int` | Bartlett kernel bandwidth used |
| `nobs` | `Int` | Number of observations |

### Interpretation

**Reject** ``H_0`` (p-value < 0.05): evidence against stationarity; the series has a unit root. **Fail to reject** ``H_0`` (p-value > 0.05): cannot reject stationarity; the series appears mean-reverting. Unlike the ADF test, failure to reject here is the desirable outcome for stationary data.

### Combining ADF and KPSS

Running both tests together resolves the ambiguity inherent in any single test:

| ADF Result | KPSS Result | Conclusion |
|------------|-------------|------------|
| Reject (stationary) | Fail to reject (stationary) | **Stationary** |
| Fail to reject (unit root) | Reject (unit root) | **Unit root** |
| Reject | Reject | Conflicting (possible structural break) |
| Fail to reject | Fail to reject | Inconclusive |

When both tests reject, the series likely contains a structural break that distorts both null distributions. The Zivot-Andrews test addresses this case directly.

---

## Phillips-Perron Test

The Phillips-Perron (PP) test (Phillips & Perron, 1988) is a non-parametric alternative to the ADF test. Instead of adding lagged differences to absorb serial correlation, the PP test applies a Newey-West correction directly to the t-statistic from the simple Dickey-Fuller regression.

The PP test estimates the unadjusted regression:

```math
y_t = \alpha + \rho y_{t-1} + u_t
```

The PP ``Z_t`` statistic corrects the OLS t-ratio for serial correlation and heteroskedasticity:

```math
Z_t = \sqrt{\frac{\hat{\gamma}_0}{\hat{\lambda}^2}} \, t_\rho - \frac{\hat{\lambda}^2 - \hat{\gamma}_0}{2 \hat{\lambda} \cdot \text{se}(\hat{\rho}) \cdot \sqrt{T}}
```

where:
- ``\hat{\gamma}_0 = T^{-1} \sum_{t=1}^{T} \hat{u}_t^2`` is the short-run variance
- ``\hat{\lambda}^2`` is the Newey-West long-run variance estimate
- ``t_\rho`` is the OLS t-ratio on ``\hat{\rho}``

The PP test shares the same null distribution as the ADF test, so MacKinnon critical values apply. Its advantage is that it does not require specifying the number of augmenting lags, although bandwidth selection for the Newey-West estimator plays an analogous role.

```julia
using MacroEconometricModels

# CPI price level — non-parametric unit root test
fred = load_example(:fred_md)
cpi = filter(isfinite, to_vector(fred[:, "CPIAUCSL"]))

result = pp_test(cpi; regression=:constant)
report(result)
```

### Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `regression` | `Symbol` | `:constant` | Deterministic terms: `:none`, `:constant`, or `:trend` |
| `bandwidth` | `Union{Int,Symbol}` | `:auto` | Newey-West bandwidth, or `:auto` for automatic selection |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | Phillips-Perron ``Z_t`` test statistic |
| `pvalue` | `T` | Asymptotic p-value |
| `regression` | `Symbol` | Deterministic specification |
| `critical_values` | `Dict{Int,T}` | Critical values at 1%, 5%, 10% |
| `bandwidth` | `Int` | Newey-West bandwidth used |
| `nobs` | `Int` | Number of observations |

---

## Zivot-Andrews Test

The Zivot-Andrews test (Zivot & Andrews, 1992) extends the ADF framework by allowing for a single endogenous structural break. Standard unit root tests lose power dramatically in the presence of structural breaks: a stationary series with a level shift can appear to have a unit root under the ADF test. The Zivot-Andrews test searches over all candidate break dates and selects the one that provides the strongest evidence against the unit root null.

Three specifications control which deterministic terms admit a structural break:

**Break in intercept** (`:constant`):

```math
\Delta y_t = \alpha + \beta t + \theta DU_t + \gamma y_{t-1} + \sum_{j=1}^{p} \delta_j \Delta y_{t-j} + \varepsilon_t
```

**Break in trend** (`:trend`):

```math
\Delta y_t = \alpha + \beta t + \phi DT_t + \gamma y_{t-1} + \sum_{j=1}^{p} \delta_j \Delta y_{t-j} + \varepsilon_t
```

**Break in both** (`:both`):

```math
\Delta y_t = \alpha + \beta t + \theta DU_t + \phi DT_t + \gamma y_{t-1} + \sum_{j=1}^{p} \delta_j \Delta y_{t-j} + \varepsilon_t
```

where:
- ``DU_t = \mathbf{1}(t > T_B)`` is the level shift dummy
- ``DT_t = (t - T_B) \cdot \mathbf{1}(t > T_B)`` is the trend shift dummy
- ``T_B`` is the break date, selected to minimize the ADF t-statistic on ``\gamma``
- The trimming parameter excludes endpoints from the search (default: 15% on each side)

```julia
using MacroEconometricModels

# CPI price level — test for unit root allowing a structural break
fred = load_example(:fred_md)
cpi = filter(isfinite, to_vector(fred[:, "CPIAUCSL"]))

result = za_test(cpi; regression=:both)
report(result)

# Access break point information
result.break_index     # Observation index of detected break
result.break_fraction  # Break location as fraction of sample
```

### Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `regression` | `Symbol` | `:both` | Break type: `:constant`, `:trend`, or `:both` |
| `trim` | `Real` | `0.15` | Trimming fraction for break search (excludes endpoints) |
| `lags` | `Union{Int,Symbol}` | `:aic` | Augmenting lags, or `:aic`/`:bic` for automatic selection |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | Minimum ADF t-statistic over all candidate break dates |
| `pvalue` | `T` | Asymptotic p-value |
| `break_index` | `Int` | Estimated break point (observation index) |
| `break_fraction` | `T` | Break location as fraction of sample (0 to 1) |
| `regression` | `Symbol` | Break specification (`:constant`, `:trend`, `:both`) |
| `critical_values` | `Dict{Int,T}` | Critical values at 1%, 5%, 10% |
| `lags` | `Int` | Number of augmenting lags |
| `nobs` | `Int` | Number of observations |

!!! note "Technical Note"
    The Zivot-Andrews test assumes at most one structural break under the alternative hypothesis. If the data-generating process contains multiple breaks, the test has reduced power. For multiple breaks, consider the Lumsdaine-Papell (1997) extension or panel unit root tests that exploit cross-sectional information.

---

## Ng-Perron Tests

The Ng-Perron tests (Ng & Perron, 2001) address the well-known size distortions of the ADF test, particularly when the initial condition is far from zero or the errors have a large negative MA root. These tests apply GLS detrending to the data before computing four modified test statistics with superior size and power properties.

The GLS detrending procedure quasi-differences the data using a local-to-unity parameter:

```math
\tilde{y}_1 = y_1, \qquad \tilde{y}_t = y_t - \bar{c}/T \cdot y_{t-1}, \quad t = 2, \ldots, T
```

where ``\bar{c} = -7`` for level stationarity (`:constant`) and ``\bar{c} = -13.5`` for trend stationarity (`:trend``). The four test statistics are:

- **MZa** (``MZ_\alpha``): Modified Phillips ``Z_\alpha`` statistic
- **MZt** (``MZ_t``): Modified Phillips ``Z_t`` statistic (most commonly reported)
- **MSB**: Modified Sargan-Bhargava statistic
- **MPT**: Modified point-optimal statistic

All four statistics are computed from the GLS-detrended series and use the autoregressive spectral density estimator for the long-run variance.

```julia
using MacroEconometricModels

# CPI price level — GLS-detrended unit root tests
fred = load_example(:fred_md)
cpi = filter(isfinite, to_vector(fred[:, "CPIAUCSL"]))

result = ngperron_test(cpi; regression=:constant)
report(result)

# Access individual statistics
result.MZa   # Modified Zα
result.MZt   # Modified Zt (most commonly reported)
result.MSB   # Modified Sargan-Bhargava
result.MPT   # Modified point-optimal
```

### Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `regression` | `Symbol` | `:constant` | Deterministic terms: `:constant` or `:trend` |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `MZa` | `T` | Modified Phillips ``Z_\alpha`` statistic |
| `MZt` | `T` | Modified Phillips ``Z_t`` statistic (most commonly reported) |
| `MSB` | `T` | Modified Sargan-Bhargava statistic |
| `MPT` | `T` | Modified point-optimal statistic |
| `regression` | `Symbol` | Deterministic specification |
| `critical_values` | `Dict{Symbol,Dict{Int,T}}` | Critical values keyed by statistic name (`:MZa`, `:MZt`, `:MSB`, `:MPT`) |
| `nobs` | `Int` | Number of observations |

!!! note "Technical Note"
    The Ng-Perron tests use GLS detrending which provides substantially better size properties than the standard ADF test in small samples (``T < 100``). When the ADF test yields borderline results, the MZt statistic is a more reliable indicator. However, ADF remains preferable when the data-generating process has a large negative MA root, as GLS-based tests can be oversized in that case (Perron & Ng, 1996).

!!! note "Implementation Detail"
    The autoregressive spectral density estimator ``s^2_{AR}`` is computed by fitting an AR model to the differenced GLS-detrended series ``\Delta \tilde{y}_t``, following Ng & Perron (2001, equation 11). This ensures the spectral density estimate is consistent under the unit root null.

---

## Convenience Functions

Two convenience functions simplify batch unit root analysis across multiple tests and multiple variables.

### Unit Root Summary

The `unit_root_summary` function runs multiple unit root tests on a single series and synthesizes the results into an overall conclusion based on the ADF-KPSS decision matrix.

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
cpi = filter(isfinite, to_vector(fred[:, "CPIAUCSL"]))

# Run ADF, KPSS, and PP simultaneously
summary = unit_root_summary(cpi; tests=[:adf, :kpss, :pp])

# Access individual results
summary.results[:adf]
summary.results[:kpss]

# Overall conclusion synthesizes ADF + KPSS
summary.conclusion
```

### Test All Variables

The `test_all_variables` function applies a single unit root test to every column of a data matrix, returning a vector of results for quick screening.

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
vars = fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS", "UNRATE", "M2SL"]]
Y = to_matrix(vars)
Y = Y[all.(isfinite, eachrow(Y)), :]

# Apply ADF test to all columns
results = test_all_variables(Y; test=:adf)

# Screen for unit roots
varnames = ["INDPRO", "CPIAUCSL", "FEDFUNDS", "UNRATE", "M2SL"]
for (i, r) in enumerate(results)
    status = r.pvalue > 0.05 ? "I(1)" : "I(0)"
    println("$(varnames[i]): p=$(round(r.pvalue, digits=3)) → $status")
end
```

---

## Johansen Cointegration Test

The Johansen test (Johansen, 1991) examines whether multiple I(1) series share common stochastic trends, i.e., whether linear combinations of the series are stationary. Cointegration implies a long-run equilibrium relationship that constrains the joint dynamics, and the estimated cointegrating vectors become the error correction terms in a VECM.

Consider a VAR(p) in levels:

```math
y_t = A_1 y_{t-1} + \cdots + A_p y_{t-p} + u_t
```

Rewriting in Vector Error Correction Model (VECM) form gives:

```math
\Delta y_t = \Pi y_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta y_{t-i} + u_t
```

where:
- ``\Pi = \alpha \beta'`` is the ``n \times n`` long-run impact matrix
- ``\beta`` contains the cointegrating vectors (equilibrium relationships)
- ``\alpha`` contains the adjustment coefficients (speed of adjustment to equilibrium)
- ``\text{rank}(\Pi) = r`` equals the number of cointegrating relationships

The Johansen procedure tests the rank of ``\Pi`` using two likelihood ratio statistics:

**Trace test** -- tests ``H_0: \text{rank}(\Pi) \leq r`` against ``H_1: \text{rank}(\Pi) > r``:

```math
\lambda_{\text{trace}}(r) = -T \sum_{i=r+1}^{n} \ln(1 - \hat{\lambda}_i)
```

**Maximum eigenvalue test** -- tests ``H_0: \text{rank}(\Pi) = r`` against ``H_1: \text{rank}(\Pi) = r + 1``:

```math
\lambda_{\max}(r) = -T \ln(1 - \hat{\lambda}_{r+1})
```

where ``\hat{\lambda}_1 \geq \hat{\lambda}_2 \geq \cdots \geq \hat{\lambda}_n`` are the ordered eigenvalues from the reduced-rank regression. Critical values are from Osterwald-Lenum (1992).

!!! note "Technical Note"
    The `deterministic` keyword controls the placement of deterministic terms following Johansen's (1995) five cases. With `:constant` (Case 2), the intercept is restricted to the cointegrating space -- it enters the error correction term ``\Pi y_{t-1}`` but not the short-run dynamics, preventing linear trends in levels. With `:trend` (Case 4), a linear trend is restricted to the cointegrating space, allowing quadratic trends in levels. Critical values are tabulated separately for each case.

```julia
using MacroEconometricModels

# Test cointegration between log real GDP and log real consumption
qd = load_example(:fred_qd)
Y = log.(to_matrix(qd[:, ["GDPC1", "PCECC96"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

# Johansen test with 2 lags in VECM
result = johansen_test(Y, 2; deterministic=:constant)
report(result)

# Access results
result.rank                              # Estimated cointegration rank
result.eigenvectors[:, 1:result.rank]    # Cointegrating vectors
result.adjustment                        # Loading matrix α
```

### Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `p` | `Int` | Required | Number of lags in VECM representation |
| `deterministic` | `Symbol` | `:constant` | Deterministic terms: `:none`, `:constant`, or `:trend` |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `trace_stats` | `Vector{T}` | Trace test statistics for each rank hypothesis |
| `trace_pvalues` | `Vector{T}` | P-values for trace statistics |
| `max_eigen_stats` | `Vector{T}` | Maximum eigenvalue test statistics |
| `max_eigen_pvalues` | `Vector{T}` | P-values for max eigenvalue statistics |
| `rank` | `Int` | Estimated cointegration rank |
| `eigenvectors` | `Matrix{T}` | ``n \times n`` matrix of cointegrating vectors (columns) |
| `adjustment` | `Matrix{T}` | ``n \times n`` adjustment (loading) matrix ``\alpha`` |
| `eigenvalues` | `Vector{T}` | Ordered eigenvalues from reduced-rank regression |
| `critical_values_trace` | `Matrix{T}` | ``n \times 3`` critical values for trace test (1%, 5%, 10%) |
| `critical_values_max` | `Matrix{T}` | ``n \times 3`` critical values for max eigenvalue test |
| `deterministic` | `Symbol` | Deterministic specification (`:none`, `:constant`, `:trend`) |
| `lags` | `Int` | Number of VECM lags |
| `nobs` | `Int` | Number of observations |

### Interpretation

The sequential testing procedure starts from ``r = 0`` and increases:

1. Test ``H_0: r = 0`` (no cointegration). If rejected, proceed.
2. Test ``H_0: r \leq 1``. If rejected, proceed.
3. Continue until ``H_0: r \leq k`` is not rejected.

The first non-rejected hypothesis gives the cointegration rank. With rank ``r > 0``, estimate a VECM with `estimate_vecm(Y, p; rank=r)` to incorporate the long-run equilibrium constraints. See the [VECM page](vecm.md) for estimation details.

---

## Complete Example

This workflow demonstrates the full pre-estimation stationarity analysis pipeline: screen individual series for unit roots, confirm with complementary tests, and test for cointegration among I(1) variables.

```julia
using MacroEconometricModels

# ── Step 1: Load data ──────────────────────────────────────────
fred = load_example(:fred_md)

# Extract key macroeconomic variables
indpro = filter(isfinite, to_vector(fred[:, "INDPRO"]))
cpi    = filter(isfinite, to_vector(fred[:, "CPIAUCSL"]))
ffr    = filter(isfinite, to_vector(fred[:, "FEDFUNDS"]))

# ── Step 2: Individual unit root tests (ADF + KPSS) ───────────
for (name, y) in [("INDPRO", indpro), ("CPI", cpi), ("FFR", ffr)]
    adf  = adf_test(y; lags=:aic, regression=:constant)
    kpss = kpss_test(y; regression=:constant)
    adf_status  = adf.pvalue < 0.05 ? "reject" : "fail to reject"
    kpss_status = kpss.pvalue < 0.05 ? "reject" : "fail to reject"
    println("$name: ADF p=$(round(adf.pvalue, digits=3)) ($adf_status) | ",
            "KPSS p=$(round(kpss.pvalue, digits=3)) ($kpss_status)")
end

# ── Step 3: Comprehensive summary for CPI ──────────────────────
summary = unit_root_summary(cpi; tests=[:adf, :kpss, :pp])
println("CPI conclusion: ", summary.conclusion)

# ── Step 4: Check for structural breaks in CPI ─────────────────
za = za_test(cpi; regression=:both)
report(za)
println("Break detected at observation: ", za.break_index)

# ── Step 5: Ng-Perron as robustness check ──────────────────────
np = ngperron_test(cpi; regression=:constant)
report(np)

# ── Step 6: Test cointegration among I(1) variables ────────────
qd = load_example(:fred_qd)
Y = log.(to_matrix(qd[:, ["GDPC1", "PCECC96"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

johansen = johansen_test(Y, 2; deterministic=:constant)
report(johansen)

if johansen.rank > 0
    println("Cointegration rank = $(johansen.rank) → estimate VECM")
else
    println("No cointegration → use VAR in first differences")
end
```

---

## Common Pitfalls

1. **Wrong regression specification.** Including a trend term (`:trend`) when the series has no deterministic trend reduces power. Use `:constant` for series that fluctuate around a fixed mean, `:trend` only when visual inspection shows a clear linear trend, and `:none` only for demeaned or detrended data.

2. **Ignoring KPSS confirmation.** The ADF test has low power against near-unit-root alternatives. A failure to reject in ADF alone does not confirm a unit root. Always run KPSS as a complementary test: concordant results (ADF fails to reject, KPSS rejects) provide much stronger evidence than either test alone.

3. **Structural breaks biasing unit root tests.** A stationary series with a level shift mimics a unit root process, causing ADF, PP, and KPSS to produce misleading results. When both ADF and KPSS reject their respective nulls (the "conflicting" cell in the decision matrix), a structural break is the most common explanation. Use `za_test` to test for a unit root while allowing for an endogenous break.

4. **Johansen lag sensitivity.** The Johansen test is sensitive to the lag order ``p`` in the VECM representation. Too few lags leave serial correlation in the residuals, distorting the test size. Too many lags waste degrees of freedom. Select ``p`` using information criteria (estimate VARs at multiple lag orders and compare AIC/BIC) before running the cointegration test.

---

## References

- Dickey, David A., and Wayne A. Fuller. 1979. "Distribution of the Estimators for Autoregressive Time Series with a Unit Root." *Journal of the American Statistical Association* 74 (366): 427--431. [https://doi.org/10.1080/01621459.1979.10482531](https://doi.org/10.1080/01621459.1979.10482531)
- Johansen, Soren. 1991. "Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models." *Econometrica* 59 (6): 1551--1580. [https://doi.org/10.2307/2938278](https://doi.org/10.2307/2938278)
- Johansen, Soren. 1995. *Likelihood-Based Inference in Cointegrated Vector Autoregressive Models*. Oxford: Oxford University Press. ISBN 978-0-19-877450-5.
- Kwiatkowski, Denis, Peter C. B. Phillips, Peter Schmidt, and Yongcheol Shin. 1992. "Testing the Null Hypothesis of Stationarity Against the Alternative of a Unit Root." *Journal of Econometrics* 54 (1--3): 159--178. [https://doi.org/10.1016/0304-4076(92)90104-Y](https://doi.org/10.1016/0304-4076(92)90104-Y)
- MacKinnon, James G. 1994. "Approximate Asymptotic Distribution Functions for Unit-Root and Cointegration Tests." *Journal of Business & Economic Statistics* 12 (2): 167--176. [https://doi.org/10.1080/07350015.1994.10510005](https://doi.org/10.1080/07350015.1994.10510005)
- MacKinnon, James G. 2010. "Critical Values for Cointegration Tests." Queen's Economics Department Working Paper No. 1227.
- Ng, Serena, and Pierre Perron. 2001. "Lag Length Selection and the Construction of Unit Root Tests with Good Size and Power." *Econometrica* 69 (6): 1519--1554. [https://doi.org/10.1111/1468-0262.00256](https://doi.org/10.1111/1468-0262.00256)
- Osterwald-Lenum, Michael. 1992. "A Note with Quantiles of the Asymptotic Distribution of the Maximum Likelihood Cointegration Rank Test Statistics." *Oxford Bulletin of Economics and Statistics* 54 (3): 461--472. [https://doi.org/10.1111/j.1468-0084.1992.tb00013.x](https://doi.org/10.1111/j.1468-0084.1992.tb00013.x)
- Phillips, Peter C. B., and Pierre Perron. 1988. "Testing for a Unit Root in Time Series Regression." *Biometrika* 75 (2): 335--346. [https://doi.org/10.1093/biomet/75.2.335](https://doi.org/10.1093/biomet/75.2.335)
- Zivot, Eric, and Donald W. K. Andrews. 1992. "Further Evidence on the Great Crash, the Oil-Price Shock, and the Unit-Root Hypothesis." *Journal of Business & Economic Statistics* 10 (3): 251--270. [https://doi.org/10.1080/07350015.1992.10509904](https://doi.org/10.1080/07350015.1992.10509904)
