# Hypothesis Tests

This chapter covers statistical hypothesis tests for time series analysis, including unit root tests for stationarity detection, cointegration tests for multivariate relationships, and VAR stability diagnostics.

## Introduction

Before fitting dynamic models like VARs or Local Projections, it is essential to understand the stationarity properties of the data. Non-stationary series (those with unit roots) require different treatment than stationary series, as standard regression methods can lead to spurious results.

**MacroEconometricModels.jl** provides a comprehensive suite of unit root and stationarity tests.

## Quick Start

```julia
adf_result = adf_test(y; lags=:aic, regression=:constant)          # ADF unit root test
kpss_result = kpss_test(y; regression=:constant)                    # KPSS stationarity test
pp_result = pp_test(y; regression=:constant)                        # Phillips-Perron test
za_result = za_test(y; regression=:both, trim=0.15)                 # Zivot-Andrews (structural break)
johansen_result = johansen_test(Y, 2; deterministic=:constant)      # Johansen cointegration
```

### Univariate Tests
1. **ADF (Augmented Dickey-Fuller)**: Tests the null of a unit root against stationarity
2. **KPSS**: Tests the null of stationarity against a unit root
3. **Phillips-Perron**: Non-parametric unit root test with autocorrelation correction
4. **Zivot-Andrews**: Unit root test allowing for endogenous structural break
5. **Ng-Perron**: Modified tests with improved size properties

### Multivariate Tests
6. **Johansen Cointegration**: Tests for cointegrating relationships among variables

### Model Diagnostics
7. **VAR Stationarity**: Check if an estimated VAR model is stable

---

## Augmented Dickey-Fuller Test

### Theory

The Augmented Dickey-Fuller (ADF) test examines whether a time series has a unit root. Consider the autoregressive model:

```math
y_t = \rho y_{t-1} + u_t
```

The null hypothesis is ``H_0: \rho = 1`` (unit root) against ``H_1: \rho < 1`` (stationary).

The test is performed via the regression:

```math
\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{j=1}^{p} \delta_j \Delta y_{t-j} + \varepsilon_t
```

where:
- ``\gamma = \rho - 1`` is the coefficient of interest
- ``\alpha`` is an optional constant
- ``\beta t`` is an optional linear trend
- Lagged differences are included to control for serial correlation

The ADF statistic is the t-ratio ``\tau = \hat{\gamma} / \text{se}(\hat{\gamma})``.

**Critical values** depend on the specification (none, constant, or trend) and are tabulated using MacKinnon (1994, 2010) response surfaces.

**Reference**: Dickey & Fuller (1979), MacKinnon (2010)

### Julia Implementation

```julia
using MacroEconometricModels

# Generate a random walk (has unit root)
y = cumsum(randn(200))

# ADF test with automatic lag selection via AIC
result = adf_test(y; lags=:aic, regression=:constant)

# The result displays with publication-quality formatting:
# - Test statistic and significance stars
# - Critical values at 1%, 5%, 10% levels
# - Automatic conclusion
```

### Function Signature

```@docs
adf_test
```

### Options

| Argument | Description | Default |
|----------|-------------|---------|
| `lags` | Number of augmenting lags, or `:aic`/`:bic`/`:hqic` for automatic selection | `:aic` |
| `max_lags` | Maximum lags for automatic selection | `floor(12*(T/100)^0.25)` |
| `regression` | Deterministic terms: `:none`, `:constant`, or `:trend` | `:constant` |

### ADFResult Return Values

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | ADF test statistic (``\tau``-ratio) |
| `pvalue` | `T` | Asymptotic p-value (MacKinnon response surface) |
| `lags` | `Int` | Number of augmenting lags used |
| `regression` | `Symbol` | Deterministic specification (`:none`, `:constant`, `:trend`) |
| `critical_values` | `Dict{Int,T}` | Critical values at 1%, 5%, 10% significance levels |
| `nobs` | `Int` | Number of observations used |

### Interpreting Results

- **Reject H₀** (p-value < 0.05): Evidence against unit root; series appears stationary
- **Fail to reject H₀** (p-value > 0.05): Cannot reject unit root; series may be non-stationary

---

## KPSS Stationarity Test

### Theory

The KPSS test (Kwiatkowski, Phillips, Schmidt & Shin, 1992) reverses the hypotheses of the ADF test:

- ``H_0``: Series is stationary (level or trend stationary)
- ``H_1``: Series has a unit root

This complementary approach is valuable because failure to reject in the ADF test does not confirm stationarity—it may simply reflect low power.

The test decomposes the series:

```math
y_t = \xi t + r_t + \varepsilon_t
```

where ``r_t = r_{t-1} + u_t`` is a random walk. Under ``H_0``, the variance of ``u_t`` is zero.

The KPSS statistic is:

```math
\text{KPSS} = \frac{\sum_{t=1}^T S_t^2}{T^2 \hat{\sigma}^2_{LR}}
```

where ``S_t = \sum_{s=1}^t \hat{e}_s`` are partial sums of residuals and ``\hat{\sigma}^2_{LR}`` is the long-run variance estimated using a Bartlett kernel.

**Reference**: Kwiatkowski et al. (1992)

### Julia Implementation

```julia
using MacroEconometricModels

# Stationary series
y = randn(200)
result = kpss_test(y; regression=:constant)

# For trend stationarity
result_trend = kpss_test(y; regression=:trend)
```

### Function Signature

```@docs
kpss_test
```

### Options

| Argument | Description | Default |
|----------|-------------|---------|
| `regression` | Stationarity type: `:constant` (level) or `:trend` | `:constant` |
| `bandwidth` | Bartlett kernel bandwidth, or `:auto` for Newey-West selection | `:auto` |

### KPSSResult Return Values

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | KPSS test statistic |
| `pvalue` | `T` | Asymptotic p-value |
| `regression` | `Symbol` | Stationarity type (`:constant` or `:trend`) |
| `critical_values` | `Dict{Int,T}` | Critical values at 1%, 5%, 10% |
| `bandwidth` | `Int` | Bartlett kernel bandwidth used |
| `nobs` | `Int` | Number of observations |

### Interpreting Results

- **Reject H₀** (p-value < 0.05): Evidence against stationarity; series has a unit root
- **Fail to reject H₀** (p-value > 0.05): Cannot reject stationarity

### Combining ADF and KPSS

Using both tests together provides stronger inference:

| ADF Result | KPSS Result | Conclusion |
|------------|-------------|------------|
| Reject H₀ (stationary) | Fail to reject H₀ (stationary) | **Stationary** |
| Fail to reject H₀ (unit root) | Reject H₀ (unit root) | **Unit root** |
| Reject H₀ | Reject H₀ | Conflicting (possible structural break) |
| Fail to reject H₀ | Fail to reject H₀ | Inconclusive |

---

## Phillips-Perron Test

### Theory

The Phillips-Perron (PP) test is a non-parametric alternative to the ADF test. Instead of augmenting with lagged differences, the PP test corrects the t-statistic for serial correlation using Newey-West standard errors.

The regression is:

```math
y_t = \alpha + \rho y_{t-1} + u_t
```

The PP ``Z_t`` statistic adjusts the OLS t-ratio:

```math
Z_t = \sqrt{\frac{\hat{\gamma}_0}{\hat{\lambda}^2}} t_\rho - \frac{\hat{\lambda}^2 - \hat{\gamma}_0}{2\hat{\lambda} \cdot \text{se}(\hat{\rho}) \cdot \sqrt{T}}
```

where ``\hat{\gamma}_0`` is the short-run variance and ``\hat{\lambda}^2`` is the long-run variance.

**Advantage**: Does not require specifying the number of lags.

**Reference**: Phillips & Perron (1988)

### Julia Implementation

```julia
using MacroEconometricModels

y = cumsum(randn(200))
result = pp_test(y; regression=:constant)
```

### Function Signature

```@docs
pp_test
```

### Options

| Argument | Description | Default |
|----------|-------------|---------|
| `regression` | Deterministic terms: `:none`, `:constant`, or `:trend` | `:constant` |
| `bandwidth` | Newey-West bandwidth, or `:auto` | `:auto` |

### PPResult Return Values

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

### Theory

The Zivot-Andrews test extends the ADF test by allowing for an **endogenous structural break** in the series. This is important because standard unit root tests have low power against stationary alternatives with structural breaks.

Three specifications are available:

1. **Break in intercept** (`:constant`):
```math
\Delta y_t = \alpha + \beta t + \theta DU_t + \gamma y_{t-1} + \sum_j \delta_j \Delta y_{t-j} + \varepsilon_t
```

2. **Break in trend** (`:trend`):
```math
\Delta y_t = \alpha + \beta t + \phi DT_t + \gamma y_{t-1} + \sum_j \delta_j \Delta y_{t-j} + \varepsilon_t
```

3. **Break in both** (`:both`):
```math
\Delta y_t = \alpha + \beta t + \theta DU_t + \phi DT_t + \gamma y_{t-1} + \sum_j \delta_j \Delta y_{t-j} + \varepsilon_t
```

where:
- ``DU_t = 1`` if ``t > T_B`` (level shift dummy)
- ``DT_t = t - T_B`` if ``t > T_B`` (trend shift dummy)
- ``T_B`` is the break point, selected to minimize the t-statistic on ``\gamma``

**Reference**: Zivot & Andrews (1992)

### Julia Implementation

```julia
using MacroEconometricModels

# Series with structural break
y = vcat(randn(100), randn(100) .+ 2)  # Level shift at t=100
result = za_test(y; regression=:constant)

# Access break point
println("Break detected at observation: ", result.break_index)
println("Break location: ", result.break_fraction * 100, "% of sample")
```

### Function Signature

```@docs
za_test
```

### Options

| Argument | Description | Default |
|----------|-------------|---------|
| `regression` | Break type: `:constant`, `:trend`, or `:both` | `:both` |
| `trim` | Trimming fraction for break search | `0.15` |
| `lags` | Augmenting lags, or `:aic`/`:bic` | `:aic` |

### ZAResult Return Values

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | Minimum ADF t-statistic over break candidates |
| `pvalue` | `T` | Asymptotic p-value |
| `break_index` | `Int` | Estimated break point (observation index) |
| `break_fraction` | `T` | Break location as fraction of sample (0 to 1) |
| `regression` | `Symbol` | Break type (`:constant`, `:trend`, `:both`) |
| `critical_values` | `Dict{Int,T}` | Critical values at 1%, 5%, 10% |
| `lags` | `Int` | Number of augmenting lags |
| `nobs` | `Int` | Number of observations |

---

## Ng-Perron Tests

### Theory

The Ng-Perron tests (2001) are modified unit root tests with improved size and power properties, especially in small samples. They use GLS detrending and report four test statistics:

1. **MZα**: Modified Phillips Zα statistic
2. **MZt**: Modified Phillips Zt statistic (most commonly used)
3. **MSB**: Modified Sargan-Bhargava statistic
4. **MPT**: Modified Point-optimal statistic

The GLS detrending uses the quasi-difference:

```math
\tilde{y}_t = y_t - \bar{c}/T \cdot y_{t-1}
```

where ``\bar{c} = -7`` (constant) or ``\bar{c} = -13.5`` (trend).

**Advantage**: Better size properties than ADF when the initial condition is far from zero.

**Reference**: Ng & Perron (2001)

### Julia Implementation

```julia
using MacroEconometricModels

y = cumsum(randn(100))
result = ngperron_test(y; regression=:constant)

# All four statistics are reported
println("MZα: ", result.MZa)
println("MZt: ", result.MZt)
println("MSB: ", result.MSB)
println("MPT: ", result.MPT)
```

### Function Signature

```@docs
ngperron_test
```

### NgPerronResult Return Values

| Field | Type | Description |
|-------|------|-------------|
| `MZa` | `T` | Modified Phillips ``Z_\alpha`` statistic |
| `MZt` | `T` | Modified Phillips ``Z_t`` statistic (most commonly reported) |
| `MSB` | `T` | Modified Sargan-Bhargava statistic |
| `MPT` | `T` | Modified Point-optimal statistic |
| `regression` | `Symbol` | Deterministic specification |
| `critical_values` | `Dict{Symbol,Dict{Int,T}}` | Critical values keyed by statistic name (`:MZa`, `:MZt`, `:MSB`, `:MPT`) |
| `nobs` | `Int` | Number of observations |

!!! note "Technical Note"
    The Ng-Perron tests use GLS detrending which provides substantially better size properties than the standard ADF test in small samples (``T < 100``). When the ADF test has borderline results, the Ng-Perron MZt statistic is a more reliable indicator. However, ADF remains preferable when the data-generating process has a large negative MA root, as GLS-based tests can be oversized in that case (Perron & Ng, 1996).

---

## Johansen Cointegration Test

### Theory

The Johansen test examines whether multiple I(1) series share common stochastic trends, i.e., are **cointegrated**. Consider a VAR(p) in levels:

```math
y_t = A_1 y_{t-1} + \cdots + A_p y_{t-p} + u_t
```

This can be rewritten in Vector Error Correction Model (VECM) form:

```math
\Delta y_t = \Pi y_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta y_{t-i} + u_t
```

where ``\Pi = \alpha \beta'`` is the long-run matrix:
- ``\beta``: Cointegrating vectors (equilibrium relationships)
- ``\alpha``: Adjustment coefficients (speed of adjustment to equilibrium)
- ``\text{rank}(\Pi) = r``: Number of cointegrating relationships

Two test statistics are computed:

**Trace Test**: Tests ``H_0: \text{rank} \leq r`` against ``H_1: \text{rank} > r``
```math
\lambda_{trace}(r) = -T \sum_{i=r+1}^{n} \ln(1 - \hat{\lambda}_i)
```

**Maximum Eigenvalue Test**: Tests ``H_0: \text{rank} = r`` against ``H_1: \text{rank} = r+1``
```math
\lambda_{max}(r) = -T \ln(1 - \hat{\lambda}_{r+1})
```

**Reference**: Johansen (1991), Osterwald-Lenum (1992)

### Julia Implementation

```julia
using MacroEconometricModels

# Generate cointegrated system
T, n = 200, 3
Y = randn(T, n)
Y[:, 2] = Y[:, 1] + 0.1 * randn(T)  # Y2 cointegrated with Y1
Y[:, 3] = cumsum(randn(T))           # Y3 independent I(1)

# Johansen test with 2 lags in VECM
result = johansen_test(Y, 2; deterministic=:constant)

# Access results
println("Estimated cointegration rank: ", result.rank)
println("Cointegrating vectors:\n", result.eigenvectors[:, 1:result.rank])
println("Adjustment coefficients:\n", result.adjustment)
```

### Function Signature

```@docs
johansen_test
```

### Options

| Argument | Description | Default |
|----------|-------------|---------|
| `p` | Lags in VECM representation | Required |
| `deterministic` | `:none`, `:constant`, or `:trend` | `:constant` |

### JohansenResult Return Values

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

### Interpreting Results

The test sequentially tests:
1. ``H_0: r = 0`` (no cointegration)
2. ``H_0: r \leq 1``
3. ``H_0: r \leq 2``, etc.

Stop at the first non-rejected hypothesis; that gives the cointegration rank.

---

## VAR Stationarity Check

### Theory

A VAR(p) model is **stable** (stationary) if and only if all eigenvalues of the companion matrix lie strictly inside the unit circle:

```math
F = \begin{bmatrix}
A_1 & A_2 & \cdots & A_{p-1} & A_p \\
I_n & 0 & \cdots & 0 & 0 \\
0 & I_n & \cdots & 0 & 0 \\
\vdots & & \ddots & & \vdots \\
0 & 0 & \cdots & I_n & 0
\end{bmatrix}
```

**Stability Condition**: ``|\lambda_i| < 1`` for all eigenvalues ``\lambda_i`` of ``F``.

If violated, the VAR is explosive or contains unit roots, and standard asymptotic theory does not apply.

### Julia Implementation

```julia
using MacroEconometricModels

# Estimate VAR
Y = randn(200, 3)
model = fit(VARModel, Y, 2)

# Check stationarity
result = is_stationary(model)

if result.is_stationary
    println("VAR is stationary")
    println("Maximum eigenvalue modulus: ", result.max_modulus)
else
    println("WARNING: VAR is non-stationary!")
    println("Maximum eigenvalue modulus: ", result.max_modulus)
    println("Consider differencing or VECM specification")
end
```

### Function Signature

```@docs
is_stationary
```

### VARStationarityResult Return Values

| Field | Type | Description |
|-------|------|-------------|
| `is_stationary` | `Bool` | `true` if all eigenvalues lie inside unit circle |
| `eigenvalues` | `Vector{E}` | Eigenvalues of the companion matrix (may be complex) |
| `max_modulus` | `T` | Maximum eigenvalue modulus (should be ``< 1`` for stability) |
| `companion_matrix` | `Matrix{T}` | ``np \times np`` companion matrix |

---

## Convenience Functions

### Summary of Multiple Tests

```julia
using MacroEconometricModels

y = cumsum(randn(200))

# Run multiple tests and get summary
summary = unit_root_summary(y; tests=[:adf, :kpss, :pp])

# Access individual results
summary.results[:adf]
summary.results[:kpss]

# Overall conclusion
println(summary.conclusion)
```

### Test All Variables

```julia
using MacroEconometricModels

Y = randn(200, 5)
Y[:, 1] = cumsum(Y[:, 1])  # Make first column non-stationary

# Apply ADF test to all columns
results = test_all_variables(Y; test=:adf)

# Check which variables have unit roots
for (i, r) in enumerate(results)
    status = r.pvalue > 0.05 ? "I(1)" : "I(0)"
    println("Variable $i: p=$(round(r.pvalue, digits=3)) → $status")
end
```

### Function Signatures

```@docs
unit_root_summary
test_all_variables
```

---

## Result Types

All unit root test results inherit from `AbstractUnitRootTest` and implement the StatsAPI interface:

```julia
using StatsAPI

result = adf_test(y)

# StatsAPI interface
nobs(result)    # Number of observations
dof(result)     # Degrees of freedom
pvalue(result)  # P-value
```

### Type Hierarchy

All unit root test results inherit from `AbstractUnitRootTest` and implement the StatsAPI interface. See the [API Reference](@ref) for detailed type documentation.

- `ADFResult` - Augmented Dickey-Fuller test result
- `KPSSResult` - KPSS stationarity test result
- `PPResult` - Phillips-Perron test result
- `ZAResult` - Zivot-Andrews structural break test result
- `NgPerronResult` - Ng-Perron test result (MZα, MZt, MSB, MPT)
- `JohansenResult` - Johansen cointegration test result
- `VARStationarityResult` - VAR model stationarity check result

---

## Practical Workflow

### Step-by-Step Unit Root Analysis

```julia
using MacroEconometricModels

# 1. Load/generate data
y = your_time_series

# 2. Visual inspection (plot the series)
# Look for trends, structural breaks, etc.

# 3. Test for unit root with ADF
adf_result = adf_test(y; regression=:constant)

# 4. Confirm with KPSS (opposite null)
kpss_result = kpss_test(y; regression=:constant)

# 5. If structural break suspected, use Zivot-Andrews
za_result = za_test(y; regression=:both)

# 6. For small samples, use Ng-Perron
np_result = ngperron_test(y; regression=:constant)

# 7. Decision matrix
if pvalue(adf_result) < 0.05 && pvalue(kpss_result) > 0.05
    println("Series is stationary - proceed with VAR in levels")
elseif pvalue(adf_result) > 0.05 && pvalue(kpss_result) < 0.05
    println("Series has unit root - consider differencing or VECM")
else
    println("Inconclusive - examine further or use robust methods")
end
```

### Pre-VAR Analysis

```julia
using MacroEconometricModels

# Multi-variable dataset
Y = your_data_matrix

# 1. Test each variable for unit root
results = test_all_variables(Y; test=:adf)
n_nonstationary = sum(r.pvalue > 0.05 for r in results)
println("Variables with unit roots: $n_nonstationary / $(size(Y, 2))")

# 2. If all I(1), test for cointegration
if n_nonstationary == size(Y, 2)
    johansen_result = johansen_test(Y, 2)

    if johansen_result.rank > 0
        println("Cointegration detected! Use VECM with rank=$(johansen_result.rank)")
    else
        println("No cointegration - use VAR in first differences")
    end
end

# 3. If mixed I(0)/I(1), be cautious
# Consider ARDL bounds test or transform I(1) variables
```

---

## Model Comparison: LR and LM Tests

The **likelihood ratio (LR) test** and **Lagrange multiplier (LM) test** form two legs of the classical "trinity" of specification tests (alongside the Wald test). Both test whether a restricted (simpler) model is adequate relative to an unrestricted (more general) model.

### Theory

Given nested models ``\mathcal{M}_R \subset \mathcal{M}_U`` with log-likelihoods ``\ell_R`` and ``\ell_U``:

**LR test** (Wilks 1938): Evaluate both models at their respective MLEs.
```math
\text{LR} = -2(\ell_R - \ell_U) \xrightarrow{d} \chi^2(\text{df})
```

**LM test** (Rao 1948, Silvey 1959): Evaluate the score of the unrestricted model at the restricted estimates.
```math
\text{LM} = \mathbf{s}'(-\mathbf{H})^{-1}\mathbf{s} \xrightarrow{d} \chi^2(\text{df})
```

where ``\text{df} = k_U - k_R`` is the difference in the number of parameters, ``\mathbf{s}`` is the score vector (gradient of the log-likelihood), and ``\mathbf{H}`` is the Hessian of the log-likelihood, all evaluated at the restricted estimates.

!!! note "Technical Note"
    The LR test requires estimating both models, while the LM test only requires the restricted model (plus the unrestricted likelihood function). Under the null, LR, LM, and Wald statistics are asymptotically equivalent. In finite samples, the ordering ``\text{Wald} \geq \text{LR} \geq \text{LM}`` typically holds.

### Quick Start

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# --- ARIMA: Is AR(2) adequate vs AR(4)? ---
y = randn(300)
ar2 = estimate_ar(y, 2; method=:mle)
ar4 = estimate_ar(y, 4; method=:mle)

lr_result = lr_test(ar2, ar4)   # generic: any model with loglikelihood
lm_result = lm_test(ar2, ar4)   # score-based: model-family specific

# --- VAR: VAR(1) vs VAR(3) ---
Y = randn(200, 3)
var1 = estimate_var(Y, 1)
var3 = estimate_var(Y, 3)
lr_test(var1, var3)

# --- Volatility: ARCH(1) vs GARCH(1,1) ---
y_vol = randn(500)
arch1 = estimate_arch(y_vol, 1)
garch11 = estimate_garch(y_vol, 1, 1)
lr_test(arch1, garch11)     # LR works across ARCH/GARCH
lm_test(arch1, garch11)     # LM supports ARCH→GARCH nesting
```

**Interpretation.** If the p-value is below your significance level (e.g., 0.05), reject H₀ and conclude the unrestricted model provides a significantly better fit. If the p-value is large, the restricted model is adequate.

### Supported Model Families

| Test | Supported pairs | Notes |
|------|----------------|-------|
| `lr_test` | Any pair with `loglikelihood`, `dof`, `nobs` | Generic — works for VAR, VECM, ARIMA, ARCH, GARCH, EGARCH, GJR-GARCH, DFM |
| `lm_test` | `AbstractARIMAModel` × `AbstractARIMAModel` | Same differencing order `d` required |
| `lm_test` | `VARModel` × `VARModel` | Different lag orders, same data |
| `lm_test` | `ARCHModel` × `ARCHModel` | Different ARCH orders |
| `lm_test` | `GARCHModel` × `GARCHModel` | Different `p` or `q` |
| `lm_test` | `ARCHModel` × `GARCHModel` | Cross-type nesting (ARCH ⊂ GARCH) |
| `lm_test` | `EGARCHModel` × `EGARCHModel` | Different `p` or `q` |
| `lm_test` | `GJRGARCHModel` × `GJRGARCHModel` | Different `p` or `q` |

### Function Signatures

- [`lr_test(m1, m2)`](@ref lr_test) — Likelihood ratio test
- [`lm_test(m1, m2)`](@ref lm_test) — Lagrange multiplier (score) test

### Return Values

**`LRTestResult`**

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | LR = −2(ℓ_R − ℓ_U) |
| `pvalue` | `T` | p-value from χ²(df) |
| `df` | `Int` | Degrees of freedom |
| `loglik_restricted` | `T` | Log-likelihood of restricted model |
| `loglik_unrestricted` | `T` | Log-likelihood of unrestricted model |
| `dof_restricted` | `Int` | Parameters in restricted model |
| `dof_unrestricted` | `Int` | Parameters in unrestricted model |
| `nobs_restricted` | `Int` | Observations in restricted model |
| `nobs_unrestricted` | `Int` | Observations in unrestricted model |

**`LMTestResult`**

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | LM = s'(−H)⁻¹s |
| `pvalue` | `T` | p-value from χ²(df) |
| `df` | `Int` | Degrees of freedom |
| `nobs` | `Int` | Number of observations |
| `score_norm` | `T` | ‖s‖₂ diagnostic |

---

## Granger Causality Tests

The Granger causality test (Granger 1969) examines whether lagged values of one variable help predict another variable in a VAR system.

### Theory

Given a VAR(p) model with n variables, the **pairwise** test examines whether variable j Granger-causes variable i:

```math
H_0: A_1[i,j] = A_2[i,j] = \cdots = A_p[i,j] = 0
```

Under ``H_0``, the Wald statistic ``W = \boldsymbol{\theta}'\mathbf{V}^{-1}\boldsymbol{\theta} \sim \chi^2(p)``, where ``\boldsymbol{\theta}`` collects the lag coefficients and ``\mathbf{V} = \sigma_{ii} (\mathbf{X}'\mathbf{X})^{-1}`` is the coefficient covariance.

The **block** (multivariate) test generalizes to groups of cause variables, with ``\text{df} = p \times |\text{cause group}|``.

!!! note "Technical Note"
    Granger causality is a statistical concept based on predictability, not true causation. Variable j "Granger-causes" variable i if past values of j contain information useful for predicting i beyond what is contained in past values of i and other variables. The test is valid under the assumption that the VAR model is correctly specified and the error terms are white noise.

### Quick Start

```julia
using MacroEconometricModels, Random
Random.seed!(42)

Y = randn(200, 3)
m = estimate_var(Y, 2)

# Pairwise: does variable 1 Granger-cause variable 2?
g = granger_test(m, 1, 2)

# Block: do variables 1 and 2 jointly Granger-cause variable 3?
g_block = granger_test(m, [1, 2], 3)

# All pairwise tests at once
results = granger_test_all(m)
```

**Interpretation.** If the p-value is below your significance level (e.g., 0.05), reject ``H_0`` and conclude the cause variable(s) Granger-cause the effect variable. The `granger_test_all` function returns an n×n matrix of p-values where entry [i,j] tests whether variable j Granger-causes variable i.

### Function Signatures

- [`granger_test(model, cause, effect)`](@ref granger_test) — Pairwise or block Granger causality test
- [`granger_test_all(model)`](@ref granger_test_all) — All-pairs pairwise Granger causality matrix

### Return Values

**`GrangerCausalityResult`**

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | Wald χ² statistic |
| `pvalue` | `T` | p-value from χ²(df) |
| `df` | `Int` | Degrees of freedom (p for pairwise, p×\|cause\| for block) |
| `cause` | `Vector{Int}` | Indices of causing variable(s) |
| `effect` | `Int` | Index of effect variable |
| `n` | `Int` | Number of variables in VAR |
| `p` | `Int` | Lag order |
| `nobs` | `Int` | Effective number of observations |
| `test_type` | `Symbol` | `:pairwise` or `:block` |

### Complete Example

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Generate data with known causal structure:
# Variable 2 depends on lagged Variable 1
T_obs = 300
Y = zeros(T_obs, 3)
Y[1, :] = randn(3)
for t in 2:T_obs
    Y[t, 1] = 0.5 * Y[t-1, 1] + randn()
    Y[t, 2] = 0.3 * Y[t-1, 1] + 0.2 * Y[t-1, 2] + randn()  # 1 causes 2
    Y[t, 3] = 0.4 * Y[t-1, 3] + randn()                       # independent
end

m = estimate_var(Y, 2)

# Test all pairs
results = granger_test_all(m)

# Variable 1 → 2 should show significant Granger causality
println("1 → 2: p = ", round(results[2, 1].pvalue, digits=4))

# Variable 3 should be independent
println("3 → 1: p = ", round(results[1, 3].pvalue, digits=4))
println("3 → 2: p = ", round(results[2, 3].pvalue, digits=4))

# Block test: do variables 1 and 3 jointly Granger-cause variable 2?
g_block = granger_test(m, [1, 3], 2)
println("Block [1,3] → 2: p = ", round(g_block.pvalue, digits=4))
```

---

## References

### Unit Root Tests

- Dickey, David A., and Wayne A. Fuller. 1979. "Distribution of the Estimators for Autoregressive Time Series with a Unit Root." *Journal of the American Statistical Association* 74 (366): 427–431. [https://doi.org/10.1080/01621459.1979.10482531](https://doi.org/10.1080/01621459.1979.10482531)
- Kwiatkowski, Denis, Peter C. B. Phillips, Peter Schmidt, and Yongcheol Shin. 1992. "Testing the Null Hypothesis of Stationarity Against the Alternative of a Unit Root." *Journal of Econometrics* 54 (1–3): 159–178. [https://doi.org/10.1016/0304-4076(92)90104-Y](https://doi.org/10.1016/0304-4076(92)90104-Y)
- MacKinnon, James G. 2010. "Critical Values for Cointegration Tests." Queen's Economics Department Working Paper No. 1227.
- Ng, Serena, and Pierre Perron. 2001. "Lag Length Selection and the Construction of Unit Root Tests with Good Size and Power." *Econometrica* 69 (6): 1519–1554. [https://doi.org/10.1111/1468-0262.00256](https://doi.org/10.1111/1468-0262.00256)
- Phillips, Peter C. B., and Pierre Perron. 1988. "Testing for a Unit Root in Time Series Regression." *Biometrika* 75 (2): 335–346. [https://doi.org/10.1093/biomet/75.2.335](https://doi.org/10.1093/biomet/75.2.335)
- Zivot, Eric, and Donald W. K. Andrews. 1992. "Further Evidence on the Great Crash, the Oil-Price Shock, and the Unit-Root Hypothesis." *Journal of Business & Economic Statistics* 10 (3): 251–270. [https://doi.org/10.1080/07350015.1992.10509904](https://doi.org/10.1080/07350015.1992.10509904)

### Cointegration

- Johansen, Søren. 1991. "Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models." *Econometrica* 59 (6): 1551–1580. [https://doi.org/10.2307/2938278](https://doi.org/10.2307/2938278)
- Johansen, Søren. 1995. *Likelihood-Based Inference in Cointegrated Vector Autoregressive Models*. Oxford: Oxford University Press. ISBN 978-0-19-877450-5.
- Osterwald-Lenum, Michael. 1992. "A Note with Quantiles of the Asymptotic Distribution of the Maximum Likelihood Cointegration Rank Test Statistics." *Oxford Bulletin of Economics and Statistics* 54 (3): 461–472. [https://doi.org/10.1111/j.1468-0084.1992.tb00013.x](https://doi.org/10.1111/j.1468-0084.1992.tb00013.x)

### Granger Causality

- Granger, C. W. J. 1969. "Investigating Causal Relations by Econometric Models and Cross-spectral Methods." *Econometrica* 37 (3): 424–438. [https://doi.org/10.2307/1912791](https://doi.org/10.2307/1912791)

### Model Comparison

- Wilks, Samuel S. 1938. "The Large-Sample Distribution of the Likelihood Ratio for Testing Composite Hypotheses." *Annals of Mathematical Statistics* 9 (1): 60–62. [https://doi.org/10.1214/aoms/1177732360](https://doi.org/10.1214/aoms/1177732360)
- Neyman, Jerzy, and Egon S. Pearson. 1933. "On the Problem of the Most Efficient Tests of Statistical Hypotheses." *Philosophical Transactions of the Royal Society A* 231 (694–706): 289–337. [https://doi.org/10.1098/rsta.1933.0009](https://doi.org/10.1098/rsta.1933.0009)
- Rao, C. Radhakrishna. 1948. "Large Sample Tests of Statistical Hypotheses Concerning Several Parameters with Applications to Problems of Estimation." *Mathematical Proceedings of the Cambridge Philosophical Society* 44 (1): 50–57. [https://doi.org/10.1017/S0305004100023987](https://doi.org/10.1017/S0305004100023987)
- Silvey, S. D. 1959. "The Lagrangian Multiplier Test." *Annals of Mathematical Statistics* 30 (2): 389–407. [https://doi.org/10.1214/aoms/1177706259](https://doi.org/10.1214/aoms/1177706259)

### Textbooks

- Hamilton, James D. 1994. *Time Series Analysis*. Princeton, NJ: Princeton University Press. ISBN 978-0-691-04289-3.
- Lütkepohl, Helmut. 2005. *New Introduction to Multiple Time Series Analysis*. Berlin: Springer. ISBN 978-3-540-40172-8.
- Enders, Walter. 2014. *Applied Econometric Time Series*. 4th ed. Hoboken, NJ: Wiley. ISBN 978-1-118-80856-6.
