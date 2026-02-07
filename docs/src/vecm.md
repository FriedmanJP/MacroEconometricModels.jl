# Vector Error Correction Models

This page documents the Vector Error Correction Model (VECM) implementation in **MacroEconometricModels.jl**, providing estimation via Johansen maximum likelihood and Engle-Granger two-step methods.

## Quick Start

```julia
using MacroEconometricModels

# Estimate VECM with automatic rank selection
vecm = estimate_vecm(Y, 2)

# Estimate with explicit rank and deterministic specification
vecm = estimate_vecm(Y, 2; rank=1, deterministic=:constant)

# Convert to VAR for structural analysis
var_model = to_var(vecm)
irfs = irf(vecm, 20)  # IRFs via automatic VAR conversion
```

---

## Model Specification

### From VAR to VECM

Consider a VAR(p) model in levels for an ``n``-dimensional ``I(1)`` vector ``y_t``:

```math
y_t = c + A_1 y_{t-1} + A_2 y_{t-2} + \cdots + A_p y_{t-p} + u_t
```

When the variables are cointegrated, the Granger representation theorem (Engle & Granger, 1987) implies the system can be written in **Vector Error Correction** form:

```math
\Delta y_t = \alpha \beta' y_{t-1} + \Gamma_1 \Delta y_{t-1} + \cdots + \Gamma_{p-1} \Delta y_{t-p+1} + \mu + u_t
```

where:
- ``\Pi = \alpha \beta'`` is the **long-run matrix** (``n \times n``, rank ``r``)
- ``\alpha`` is the ``n \times r`` matrix of **adjustment coefficients** (loading matrix)
- ``\beta`` is the ``n \times r`` matrix of **cointegrating vectors**
- ``\Gamma_i = -(A_{i+1} + \cdots + A_p)`` are the **short-run dynamics** matrices
- ``\mu`` is the ``n \times 1`` intercept vector
- ``u_t \sim N(0, \Sigma)`` are i.i.d. innovations

The cointegrating rank ``r`` determines the number of long-run equilibrium relationships. When ``r = 0``, there is no cointegration and the system reduces to a VAR in first differences. When ``r = n``, the system is stationary in levels.

### The Cointegrating Relationship

Each column ``\beta_j`` of ``\beta`` defines a stationary linear combination:

```math
z_{j,t} = \beta_j' y_t \quad \sim I(0), \quad j = 1, \ldots, r
```

The corresponding column ``\alpha_j`` of ``\alpha`` governs the speed of adjustment: ``\alpha_{ij}`` measures how quickly variable ``i`` responds to deviations from the ``j``-th equilibrium.

!!! note "Phillips Normalization"
    The package applies Phillips normalization to ``\beta`` so that the first ``r`` rows form an identity matrix. This ensures unique identification of the cointegrating vectors.

---

## Estimation

### Johansen Maximum Likelihood (Default)

The Johansen (1991) reduced-rank regression procedure estimates ``\alpha`` and ``\beta`` jointly via MLE:

1. **Concentrate out short-run dynamics** by regressing ``\Delta Y`` and ``Y_{t-1}`` on lagged differences ``Z = [\Delta Y_{t-1}, \ldots, \Delta Y_{t-p+1}, \mu]``
2. **Compute moment matrices** ``S_{00}``, ``S_{11}``, ``S_{01}`` from the concentrated residuals
3. **Solve the generalized eigenvalue problem** ``|{\lambda S_{11} - S_{10} S_{00}^{-1} S_{01}}| = 0``
4. **Extract** ``\beta`` from the first ``r`` eigenvectors and compute ``\alpha = S_{01} \beta (\beta' S_{11} \beta)^{-1}``

```julia
# Automatic rank selection via Johansen trace test
vecm = estimate_vecm(Y, 2)

# Explicit rank specification
vecm = estimate_vecm(Y, 2; rank=1)

# Different deterministic specifications
vecm = estimate_vecm(Y, 2; rank=1, deterministic=:none)     # No deterministic terms
vecm = estimate_vecm(Y, 2; rank=1, deterministic=:constant)  # Constant (default)
vecm = estimate_vecm(Y, 2; rank=1, deterministic=:trend)     # Linear trend
```

The **rank** can be selected automatically using the Johansen trace test, or specified explicitly. Use `select_vecm_rank` for fine-grained control:

```julia
r = select_vecm_rank(Y, 2; criterion=:trace, significance=0.05)
r_max = select_vecm_rank(Y, 2; criterion=:max_eigen)
```

### Engle-Granger Two-Step

For bivariate systems with a single cointegrating relationship (``r = 1``), the Engle-Granger (1987) two-step estimator is available:

1. **Step 1**: Estimate the cointegrating vector via static OLS regression of ``y_{1,t}`` on ``y_{2,t}, \ldots, y_{n,t}``
2. **Step 2**: Estimate the VECM equation using the OLS residuals as the error correction term

```julia
vecm_eg = estimate_vecm(Y, 2; method=:engle_granger, rank=1)
```

!!! warning
    The Engle-Granger method only supports `rank=1`. For systems with multiple cointegrating vectors, use the Johansen method.

### Return Values

`estimate_vecm` returns a `VECMModel{T}` with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `Y` | `Matrix{T}` | Original data in levels (T_obs x n) |
| `p` | `Int` | Underlying VAR order |
| `rank` | `Int` | Cointegrating rank r |
| `alpha` | `Matrix{T}` | Adjustment coefficients (n x r) |
| `beta` | `Matrix{T}` | Cointegrating vectors (n x r) |
| `Pi` | `Matrix{T}` | Long-run matrix (n x n) |
| `Gamma` | `Vector{Matrix{T}}` | Short-run dynamics matrices |
| `mu` | `Vector{T}` | Intercept |
| `U` | `Matrix{T}` | Residuals |
| `Sigma` | `Matrix{T}` | Residual covariance |
| `aic`, `bic`, `hqic` | `T` | Information criteria |
| `loglik` | `T` | Log-likelihood |
| `deterministic` | `Symbol` | Deterministic specification |
| `method` | `Symbol` | Estimation method |
| `johansen_result` | `JohansenResult{T}` | Johansen test result |

---

## VAR Conversion

The `to_var` function converts a VECM back to a VAR in levels, enabling all structural analysis methods:

```math
A_1 = \Pi + I_n + \Gamma_1, \quad A_i = \Gamma_i - \Gamma_{i-1}, \quad A_p = -\Gamma_{p-1}
```

```julia
var_model = to_var(vecm)
```

This is critical because it allows all 18+ identification methods (Cholesky, sign restrictions, ICA, etc.) to work automatically with VECM models.

---

## Innovation Accounting

All structural analysis functions dispatch through `to_var()`, so VECMModel objects can be passed directly:

```julia
# Impulse Response Functions
irfs = irf(vecm, 20; method=:cholesky)
irfs = irf(vecm, 20; method=:sign, check_func=f, ci_type=:bootstrap)

# Forecast Error Variance Decomposition
decomp = fevd(vecm, 20)

# Historical Decomposition
T_eff = effective_nobs(to_var(vecm))
hd = historical_decomposition(vecm, T_eff)
```

---

## Forecasting

VECM forecasting iterates the VECM equations directly in levels, preserving the cointegrating relationships in the forecast path. This is preferable to forecasting from the converted VAR, as it ensures the error correction mechanism operates during the forecast.

```julia
# Point forecast
fc = forecast(vecm, 10)
fc.levels       # h x n forecast in levels
fc.differences  # h x n forecast in first differences

# With bootstrap confidence intervals
fc = forecast(vecm, 10; ci_method=:bootstrap, reps=500, conf_level=0.95)

# With simulation-based CIs
fc = forecast(vecm, 10; ci_method=:simulation, reps=500)
```

The `VECMForecast{T}` struct contains:

| Field | Description |
|-------|-------------|
| `levels` | Forecasts in levels (h x n) |
| `differences` | Forecasts in first differences (h x n) |
| `ci_lower`, `ci_upper` | Confidence interval bounds (h x n) |
| `horizon` | Forecast horizon |
| `ci_method` | Method used for CIs |

---

## Granger Causality

VECM Granger causality tests decompose causal channels into short-run and long-run components:

```julia
g = granger_causality_vecm(vecm, 1, 2)  # Test: Var 1 → Var 2
```

Three tests are computed:

| Test | Hypothesis | Mechanism |
|------|-----------|-----------|
| **Short-run** | ``\Gamma_i[\text{effect}, \text{cause}] = 0`` for all ``i`` | Causality through lagged differences |
| **Long-run** | ``\alpha[\text{effect}, :] = 0`` | Causality through error correction |
| **Strong** | Joint test of both | Combined short-run and long-run causality |

Each test reports a Wald ``\chi^2`` statistic, degrees of freedom, and p-value.

---

## Complete Example

```julia
using MacroEconometricModels, Random

Random.seed!(42)

# Generate cointegrated data
T_obs = 200
Y = cumsum(randn(T_obs, 3), dims=1)
Y[:, 2] = Y[:, 1] + 0.1 * randn(T_obs)  # Y2 cointegrated with Y1

# Step 1: Test for cointegration
joh = johansen_test(Y, 2)
println("Johansen rank: ", joh.rank)

# Step 2: Estimate VECM
vecm = estimate_vecm(Y, 2)
report(vecm)

# Step 3: Examine cointegrating vectors
println("β (cointegrating vectors):")
println(vecm.beta)
println("α (adjustment speeds):")
println(vecm.alpha)

# Step 4: Impulse responses
irfs = irf(vecm, 20; method=:cholesky)

# Step 5: Forecast
fc = forecast(vecm, 10; ci_method=:bootstrap, reps=200)

# Step 6: Granger causality
for i in 1:3, j in 1:3
    i == j && continue
    g = granger_causality_vecm(vecm, i, j)
    println("Var $i → Var $j: p=$(round(g.strong_pvalue, digits=4))")
end

# Step 7: Convert to VAR for further analysis
var_model = to_var(vecm)
decomp = fevd(var_model, 20)
```

**Interpretation.** The cointegrating vector ``\beta`` identifies the long-run equilibrium. If ``\beta \approx [1, -1, 0]'``, this implies ``y_{1,t} - y_{2,t}`` is stationary --- variables 1 and 2 share a common stochastic trend. The adjustment coefficients ``\alpha`` show how each variable responds when the system deviates from equilibrium. A significant ``\alpha_i`` indicates that variable ``i`` adjusts to restore the long-run relationship.

---

## References

- Johansen, S. (1991). Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models. *Econometrica*, 59(6), 1551--1580.
- Engle, R. F. & Granger, C. W. J. (1987). Co-Integration and Error Correction: Representation, Estimation, and Testing. *Econometrica*, 55(2), 251--276.
- Lutkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
