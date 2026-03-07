# [Vector Error Correction Models](@id vecm_page)

**MacroEconometricModels.jl** provides full-featured Vector Error Correction Model (VECM) estimation for cointegrated ``I(1)`` systems. The VECM decomposes multivariate dynamics into long-run equilibrium relationships and short-run adjustment, making it the canonical framework for modeling nonstationary variables that share common stochastic trends.

- **Estimation**: Johansen (1991) reduced-rank MLE with automatic rank selection, and Engle-Granger (1987) two-step for bivariate systems
- **Rank selection**: Trace and maximum eigenvalue tests at user-specified significance levels
- **Deterministic specification**: None, constant, or linear trend in the cointegrating relation
- **VAR conversion**: Automatic conversion to VAR in levels for full structural analysis (Cholesky, sign restrictions, ICA, and all 18 identification methods)
- **Forecasting**: Direct VECM iteration preserving cointegrating relationships, with bootstrap and simulation confidence intervals
- **Granger causality**: Short-run, long-run, and strong (joint) causality decomposition
- **TimeSeriesData dispatch**: Pass `TimeSeriesData` objects directly --- variable names propagate automatically

```@setup vecm
using MacroEconometricModels, Random
Random.seed!(42)
qd = load_example(:fred_qd)
Y = log.(to_matrix(qd[:, ["GDPC1", "PCECC96", "GPDIC1"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
```

## Quick Start

**Recipe 1: Estimate with automatic rank selection**

```@example vecm
# Automatic rank via Johansen trace test
vecm = estimate_vecm(Y, 2)
report(vecm)
```

**Recipe 2: Explicit rank and deterministic specification**

```@example vecm
vecm = estimate_vecm(Y, 2; rank=1, deterministic=:constant)
report(vecm)
```

**Recipe 3: Impulse responses via VAR conversion**

```@example vecm
vecm = estimate_vecm(Y, 2; rank=1)
irfs = irf(vecm, 20; method=:cholesky)
```

```julia
plot_result(irfs)
```

```@raw html
<iframe src="../assets/plots/vecm_irf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

**Recipe 4: Forecast with bootstrap confidence intervals**

```@example vecm
vecm = estimate_vecm(Y, 2; rank=1)
fc = forecast(vecm, 10; ci_method=:bootstrap, reps=50, conf_level=0.95)
report(fc)
```

**Recipe 5: Granger causality decomposition**

```@example vecm
vecm = estimate_vecm(Y, 2; rank=1)
g = granger_causality_vecm(vecm, 1, 2)  # GDP -> Consumption
report(g)
```

**Recipe 6: TimeSeriesData dispatch**

```@example vecm
ts = qd[:, ["GDPC1", "PCECC96", "GPDIC1"]]

# Pass TimeSeriesData directly --- variable names propagate
vecm = estimate_vecm(ts, 2; rank=1)
report(vecm)
```

---

## Model Specification

The VECM reparameterizes a VAR(p) in levels to separate long-run equilibrium relationships from short-run dynamics. Consider a VAR(p) for an ``n``-dimensional ``I(1)`` vector ``y_t``:

```math
y_t = c + A_1 y_{t-1} + A_2 y_{t-2} + \cdots + A_p y_{t-p} + u_t
```

where:
- ``y_t`` is the ``n \times 1`` vector of endogenous variables at time ``t``
- ``A_i`` are ``n \times n`` coefficient matrices for lag ``i = 1, \ldots, p``
- ``c`` is the ``n \times 1`` intercept vector
- ``u_t \sim N(0, \Sigma)`` are i.i.d. innovations

When the variables are cointegrated, the Granger representation theorem (Engle & Granger 1987) implies the system admits a **Vector Error Correction** representation:

```math
\Delta y_t = \alpha \beta' y_{t-1} + \Gamma_1 \Delta y_{t-1} + \cdots + \Gamma_{p-1} \Delta y_{t-p+1} + \mu + u_t
```

where:
- ``\Pi = \alpha \beta'`` is the ``n \times n`` **long-run matrix** with rank ``r``
- ``\alpha`` is the ``n \times r`` matrix of **adjustment coefficients** (loading matrix)
- ``\beta`` is the ``n \times r`` matrix of **cointegrating vectors**
- ``\Gamma_i = -(A_{i+1} + \cdots + A_p)`` are the ``n \times n`` **short-run dynamics** matrices
- ``\mu`` is the ``n \times 1`` intercept vector
- ``u_t \sim N(0, \Sigma)`` are i.i.d. innovations

### The Cointegrating Relationship

Each column ``\beta_j`` of ``\beta`` defines a stationary linear combination of the ``I(1)`` variables:

```math
z_{j,t} = \beta_j' y_t \sim I(0), \quad j = 1, \ldots, r
```

where:
- ``z_{j,t}`` is the ``j``-th **error correction term** (deviation from long-run equilibrium)
- ``\beta_j`` is the ``j``-th cointegrating vector

The corresponding column ``\alpha_j`` of ``\alpha`` governs the **speed of adjustment**: ``\alpha_{ij}`` measures how quickly variable ``i`` responds to deviations from the ``j``-th equilibrium. The **cointegrating rank** ``r`` determines the number of independent long-run equilibrium relationships. When ``r = 0``, there is no cointegration and the system reduces to a VAR in first differences. When ``r = n``, all variables are stationary in levels.

!!! note "Phillips Normalization"
    The package applies Phillips normalization to ``\beta`` so that the first ``r`` rows form an identity matrix. This ensures unique identification of the cointegrating vectors and makes the ``\alpha`` coefficients directly interpretable as adjustment speeds toward each equilibrium.

---

## Estimation

### Johansen Maximum Likelihood

The Johansen (1991) reduced-rank regression procedure estimates ``\alpha`` and ``\beta`` jointly via maximum likelihood. The algorithm proceeds in four steps:

1. **Concentrate out short-run dynamics** by regressing ``\Delta Y`` and ``Y_{t-1}`` on lagged differences ``Z = [\Delta Y_{t-1}, \ldots, \Delta Y_{t-p+1}, \mu]``
2. **Compute moment matrices** ``S_{00}``, ``S_{11}``, ``S_{01}`` from the concentrated residuals
3. **Solve the generalized eigenvalue problem** ``|\lambda S_{11} - S_{10} S_{00}^{-1} S_{01}| = 0``
4. **Extract** ``\beta`` from the first ``r`` eigenvectors and compute ``\alpha = S_{01} \beta (\beta' S_{11} \beta)^{-1}``

The eigenvalues ``\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n`` correspond to the canonical correlations between ``\Delta Y`` and ``Y_{t-1}`` after removing the short-run dynamics. The cointegrating rank ``r`` equals the number of statistically significant eigenvalues, determined by the trace test:

```math
\text{LR}_{\text{trace}}(r_0) = -T \sum_{i=r_0+1}^{n} \ln(1 - \hat{\lambda}_i)
```

where:
- ``T`` is the effective sample size
- ``\hat{\lambda}_i`` is the ``i``-th largest eigenvalue
- ``r_0`` is the null hypothesis rank

```@example vecm
# Automatic rank selection via Johansen trace test
vecm = estimate_vecm(Y, 2)
report(vecm)

# Explicit rank specification
vecm = estimate_vecm(Y, 2; rank=1)

# Different deterministic specifications
vecm_none = estimate_vecm(Y, 2; rank=1, deterministic=:none)      # No deterministic terms
vecm_const = estimate_vecm(Y, 2; rank=1, deterministic=:constant)  # Constant (default)
vecm_trend = estimate_vecm(Y, 2; rank=1, deterministic=:trend)     # Linear trend
```

The Johansen method estimates ``\alpha`` and ``\beta`` jointly, producing efficient estimates of the cointegrating space regardless of the number of cointegrating vectors. The trace test selects the rank at the 5% significance level by default.

### Rank Selection

The `select_vecm_rank` function provides fine-grained control over rank determination using either the trace test or the maximum eigenvalue test:

```@example vecm
r_trace = select_vecm_rank(Y, 2; criterion=:trace, significance=0.05)
r_max = select_vecm_rank(Y, 2; criterion=:max_eigen)
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `criterion` | `Symbol` | `:trace` | Test statistic: `:trace` or `:max_eigen` |
| `significance` | `Real` | `0.05` | Significance level for critical values |
| `deterministic` | `Symbol` | `:constant` | Deterministic specification for Johansen test |

### Engle-Granger Two-Step

For bivariate systems with a single cointegrating relationship (``r = 1``), the Engle-Granger (1987) two-step estimator provides a simpler alternative:

1. **Step 1**: Estimate the cointegrating vector via static OLS regression of ``y_{1,t}`` on ``y_{2,t}, \ldots, y_{n,t}``
2. **Step 2**: Estimate the VECM equations using the OLS residuals as the error correction term

```@example vecm
vecm_eg = estimate_vecm(Y, 2; method=:engle_granger, rank=1)
report(vecm_eg)
```

The Engle-Granger estimator is consistent but less efficient than Johansen MLE for multivariate systems. The static OLS regression in step 1 produces superconsistent estimates of ``\beta`` (Stock 1987), but the two-step procedure does not jointly optimize the likelihood.

!!! warning "Engle-Granger supports rank=1 only"
    The Engle-Granger method estimates a single cointegrating vector via static OLS. For systems with multiple cointegrating relationships, use the Johansen method.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `rank` | `Union{Symbol,Int}` | `:auto` | Cointegrating rank; `:auto` selects via trace test |
| `deterministic` | `Symbol` | `:constant` | Deterministic terms: `:none`, `:constant`, `:trend` |
| `method` | `Symbol` | `:johansen` | Estimation method: `:johansen` or `:engle_granger` |
| `significance` | `Real` | `0.05` | Significance level for automatic rank selection |
| `varnames` | `Vector{String}` | `["y1", ...]` | Variable display names |

### Return Value

`estimate_vecm` returns a `VECMModel{T}` with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `Y` | `Matrix{T}` | Original data in levels (``T_{obs} \times n``) |
| `p` | `Int` | Underlying VAR order |
| `rank` | `Int` | Cointegrating rank ``r`` |
| `alpha` | `Matrix{T}` | Adjustment coefficients (``n \times r``) |
| `beta` | `Matrix{T}` | Cointegrating vectors (``n \times r``), Phillips-normalized |
| `Pi` | `Matrix{T}` | Long-run matrix ``\alpha\beta'`` (``n \times n``) |
| `Gamma` | `Vector{Matrix{T}}` | Short-run dynamics matrices ``[\Gamma_1, \ldots, \Gamma_{p-1}]`` |
| `mu` | `Vector{T}` | Intercept vector |
| `U` | `Matrix{T}` | Residuals (``T_{eff} \times n``) |
| `Sigma` | `Matrix{T}` | Residual covariance (``n \times n``) |
| `aic`, `bic`, `hqic` | `T` | Information criteria |
| `loglik` | `T` | Log-likelihood |
| `deterministic` | `Symbol` | Deterministic specification |
| `method` | `Symbol` | Estimation method used |
| `johansen_result` | `JohansenResult{T}` | Johansen test result (if applicable) |
| `varnames` | `Vector{String}` | Variable display names |

---

## VAR Conversion

The `to_var` function converts a VECM back to a VAR in levels, enabling all structural analysis methods. The mapping from VECM to VAR coefficients is:

```math
A_1 = \Pi + I_n + \Gamma_1, \quad A_i = \Gamma_i - \Gamma_{i-1} \text{ for } i = 2, \ldots, p-1, \quad A_p = -\Gamma_{p-1}
```

where:
- ``A_i`` is the ``i``-th VAR coefficient matrix in levels
- ``\Pi = \alpha\beta'`` is the long-run matrix
- ``I_n`` is the ``n \times n`` identity matrix
- ``\Gamma_i`` are the short-run dynamics matrices

```@example vecm
vecm = estimate_vecm(Y, 2; rank=1)
var_model = to_var(vecm)
report(var_model)
```

This conversion is critical because it enables all 18 identification methods (Cholesky, sign restrictions, ICA, narrative, etc.) to work automatically with VECM models. The `irf`, `fevd`, and `historical_decomposition` functions dispatch through `to_var()` internally, so `VECMModel` objects can be passed directly.

---

## Innovation Accounting

All structural analysis functions accept `VECMModel` objects directly. The conversion to VAR in levels is handled automatically via `to_var()`.

```@example vecm
vecm = estimate_vecm(Y, 2; rank=1)

# Impulse response functions (Cholesky identification)
irfs = irf(vecm, 20; method=:cholesky)

# Forecast error variance decomposition
decomp = fevd(vecm, 20)

# Historical decomposition
T_eff = effective_nobs(to_var(vecm))
hd = historical_decomposition(vecm, T_eff)
```

```julia
plot_result(irfs)
plot_result(decomp)
plot_result(hd)
```

```@raw html
<iframe src="../assets/plots/vecm_irf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The Cholesky-identified IRFs trace the dynamic effects of orthogonalized shocks on the level variables. Because the VAR is estimated in levels (via conversion), the IRFs capture both the transitory short-run dynamics and the permanent effects that cointegration implies. For a system with ``r < n`` cointegrating vectors, exactly ``n - r`` shocks have permanent effects on the levels, corresponding to the common stochastic trends. For details on identification methods, see [Innovation Accounting](@ref innovation_accounting_page).

---

## Forecasting

VECM forecasting iterates the VECM equations directly in levels, preserving the cointegrating relationships in the forecast path. This approach is preferable to forecasting from the converted VAR because the error correction mechanism operates explicitly during each forecast step, pulling the system toward the long-run equilibrium:

```math
\hat{y}_{T+h} = \hat{y}_{T+h-1} + \alpha\beta'\hat{y}_{T+h-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta\hat{y}_{T+h-i} + \mu
```

where:
- ``\hat{y}_{T+h}`` is the ``h``-step-ahead forecast in levels
- ``\hat{y}_{T+h-1}`` is the previous forecast (or last observed value for ``h = 1``)
- ``\Delta\hat{y}_{T+h-i}`` are lagged forecast differences

```@example vecm
vecm = estimate_vecm(Y, 2; rank=1)

# Point forecast
fc = forecast(vecm, 10)
report(fc)

# Bootstrap confidence intervals
fc = forecast(vecm, 10; ci_method=:bootstrap, reps=50, conf_level=0.95)
report(fc)

# Simulation-based confidence intervals
fc = forecast(vecm, 10; ci_method=:simulation, reps=50)
report(fc)
```

```julia
plot_result(fc)
```

```@raw html
<iframe src="../assets/plots/forecast_vecm.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The forecast preserves cointegrating relationships by iterating the VECM equations in levels rather than converting to VAR form. Bootstrap CIs resample residuals with replacement; simulation CIs draw from ``N(0, \hat{\Sigma})``. Both methods generate ``R`` replicate forecast paths and extract pointwise quantiles.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `ci_method` | `Symbol` | `:none` | Confidence interval method: `:none`, `:bootstrap`, `:simulation` |
| `reps` | `Int` | `500` | Number of bootstrap or simulation replications |
| `conf_level` | `Real` | `0.95` | Confidence level for intervals |

### Return Value

`forecast` returns a `VECMForecast{T}` with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `levels` | `Matrix{T}` | Forecasts in levels (``h \times n``) |
| `differences` | `Matrix{T}` | Forecasts in first differences (``h \times n``) |
| `ci_lower` | `Matrix{T}` | Lower confidence interval bounds (``h \times n``) |
| `ci_upper` | `Matrix{T}` | Upper confidence interval bounds (``h \times n``) |
| `horizon` | `Int` | Forecast horizon |
| `ci_method` | `Symbol` | CI method used |
| `conf_level` | `T` | Confidence level |

---

## Granger Causality

VECM Granger causality tests (Toda & Phillips 1993) decompose causal channels into **short-run** (through lagged differences ``\Gamma``) and **long-run** (through the error correction term ``\alpha\beta'y_{t-1}``) components. The **strong** test combines both channels in a single joint test.

```@example vecm
vecm = estimate_vecm(Y, 2; rank=1)

# Test: does GDP (var 1) Granger-cause Consumption (var 2)?
g = granger_causality_vecm(vecm, 1, 2)
report(g)
```

The three Wald tests are constructed as follows:

| Test | Null hypothesis | Mechanism |
|------|----------------|-----------|
| **Short-run** | ``\Gamma_i[\text{effect}, \text{cause}] = 0`` for all ``i`` | Causality through lagged differences |
| **Long-run** | ``\alpha[\text{effect}, :] = 0`` | Causality through error correction |
| **Strong** | Joint test of both restrictions | Combined short-run and long-run causality |

Each test reports a Wald ``\chi^2`` statistic, degrees of freedom, and p-value. A significant short-run test indicates that past changes in the cause variable predict current changes in the effect variable. A significant long-run test indicates that the effect variable adjusts to deviations from the cointegrating equilibrium --- the error correction channel is active for that variable.

### Return Value

`granger_causality_vecm` returns a `VECMGrangerResult{T}` with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `short_run_stat` | `T` | Wald ``\chi^2`` for short-run test |
| `short_run_pvalue` | `T` | P-value for short-run test |
| `short_run_df` | `Int` | Degrees of freedom for short-run test |
| `long_run_stat` | `T` | Wald ``\chi^2`` for long-run test |
| `long_run_pvalue` | `T` | P-value for long-run test |
| `long_run_df` | `Int` | Degrees of freedom for long-run test |
| `strong_stat` | `T` | Wald ``\chi^2`` for joint test |
| `strong_pvalue` | `T` | P-value for joint test |
| `strong_df` | `Int` | Degrees of freedom for joint test |
| `cause_var` | `Int` | Index of the cause variable |
| `effect_var` | `Int` | Index of the effect variable |

---

## Complete Example

This example demonstrates the full VECM workflow: cointegration testing, estimation, structural analysis, forecasting, and Granger causality.

```@example vecm
# Step 1: Test for cointegration
joh = johansen_test(Y, 2)
report(joh)

# Step 2: Estimate VECM with rank 1
vecm = estimate_vecm(Y, 2; rank=1)
report(vecm)

# Step 3: Impulse responses (Cholesky identification)
irfs = irf(vecm, 20; method=:cholesky)

# Step 4: Forecast with bootstrap CIs
fc = forecast(vecm, 10; ci_method=:bootstrap, reps=50)
report(fc)

# Step 5: Granger causality --- does GDP Granger-cause Consumption?
g = granger_causality_vecm(vecm, 1, 2)
report(g)

# Step 6: Convert to VAR for FEVD
var_model = to_var(vecm)
decomp = fevd(var_model, 20)
```

```julia
plot_result(irfs)
plot_result(fc)
plot_result(decomp)
```

```@raw html
<iframe src="../assets/plots/vecm_irf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The cointegrating vector ``\beta`` identifies the long-run equilibrium between GDP, consumption, and investment. A cointegrating relationship ``\beta'y_t \sim I(0)`` implies these variables share common stochastic trends, consistent with balanced growth path theory. The adjustment coefficients ``\alpha`` reveal how each variable responds to disequilibrium --- a negative ``\alpha`` for consumption indicates it contracts when the system overshoots the equilibrium ratio. The Granger causality test decomposes predictive content into short-run (through lagged differences ``\Gamma``) and long-run (through the error correction term ``\alpha\beta'y_{t-1}``) channels, providing a richer picture than standard VAR-based Granger tests.

---

## Common Pitfalls

1. **Incorrect cointegrating rank**: Specifying ``r`` too high introduces near-unit-root stationary components that contaminate the short-run dynamics. Specifying ``r`` too low discards genuine equilibrium relationships. Always run `johansen_test` first and examine both the trace and maximum eigenvalue statistics before fixing ``r`` manually.

2. **I(2) data passed without differencing**: The VECM framework assumes all variables are ``I(1)``. Passing ``I(2)`` data (e.g., price levels that need double differencing) produces spurious cointegrating vectors. Test each series with `adf_test` or `kpss_test` before estimation and difference any ``I(2)`` variables once to bring them to ``I(1)``.

3. **Too many lags in levels**: The underlying VAR order ``p`` determines the number of lagged differences ``p - 1`` in the VECM. Over-parameterization wastes degrees of freedom and inflates estimation uncertainty, especially in small samples. Use `select_lag_order` on the levels data or compare `aic`/`bic` across candidate orders.

4. **Misinterpreting the Johansen trace test**: The sequential testing procedure starts from ``r_0 = 0`` and increments until the trace statistic falls below the critical value. Rejecting ``r_0 = 0`` but not ``r_0 = 1`` implies exactly one cointegrating vector. The trace test has well-known size distortions in small samples; the Bartlett correction (not yet implemented) mitigates this, or use a more conservative significance level.

5. **Engle-Granger with multiple cointegrating vectors**: The Engle-Granger two-step method estimates only a single cointegrating vector via static OLS. Applying it to a system with ``r > 1`` recovers at most one linear combination and discards the remaining equilibrium relationships. Use the Johansen method for systems with multiple cointegrating vectors.

6. **Forgetting VAR conversion for structural analysis**: `irf`, `fevd`, and `historical_decomposition` dispatch through `to_var()` automatically, so passing a `VECMModel` directly works. However, if you need the VAR coefficient matrices explicitly (e.g., for custom identification schemes), call `to_var(vecm)` and work with the resulting `VARModel`.

---

## References

- Johansen, S. (1991). Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models.
  *Econometrica*, 59(6), 1551-1580. [DOI](https://doi.org/10.2307/2938278)

- Engle, R. F., & Granger, C. W. J. (1987). Co-Integration and Error Correction: Representation, Estimation, and Testing.
  *Econometrica*, 55(2), 251-276. [DOI](https://doi.org/10.2307/1913236)

- Toda, H. Y., & Phillips, P. C. B. (1993). Vector Autoregressions and Causality.
  *Econometrica*, 61(6), 1367-1393. [DOI](https://doi.org/10.2307/2951647)

- Stock, J. H. (1987). Asymptotic Properties of Least Squares Estimators of Cointegrating Vectors.
  *Econometrica*, 55(5), 1035-1056. [DOI](https://doi.org/10.2307/1911260)

- Lutkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
  Berlin: Springer. ISBN 978-3-540-40172-8.
