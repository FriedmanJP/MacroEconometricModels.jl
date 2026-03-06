# [Model Diagnostics](@id tests_diagnostics_page)

Post-estimation specification testing validates the assumptions underlying statistical inference in estimated models. This page covers six categories of diagnostic tests: VAR stability checks via companion matrix eigenvalues, Granger causality tests for predictive relationships, multivariate normality tests for residual distributional assumptions, ARCH diagnostics for conditional heteroskedasticity, likelihood-based model comparison tests for nested specifications, and Panel VAR diagnostics for GMM-estimated models.

- **VAR Stationarity**: Companion matrix eigenvalue check for stable dynamics
- **Granger Causality**: Pairwise and block Wald tests for predictive causality (Granger 1969)
- **Normality Tests**: Jarque-Bera, Mardia, Doornik-Hansen, and Henze-Zirkler tests for multivariate normality
- **ARCH Diagnostics**: Engle (1982) ARCH-LM test and Ljung-Box on squared residuals
- **Model Comparison**: Likelihood ratio (Wilks 1938) and Lagrange multiplier (Rao 1948) tests for nested models
- **Panel VAR Diagnostics**: Hansen J-test, Andrews-Lu MMSC, and lag selection for GMM-estimated Panel VARs

## Quick Start

**Recipe 1: VAR stability and Granger causality**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
m = estimate_var(Y, 2)

# Check VAR stability
stat = is_stationary(m)
report(stat)

# All-pairs Granger causality
results = granger_test_all(m)
```

**Recipe 2: Normality test suite**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
m = estimate_var(Y, 2)

# Run all 7 normality tests at once
suite = normality_test_suite(m)
report(suite)
```

**Recipe 3: ARCH-LM test**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Test for ARCH effects in a return series
result = arch_lm_test(randn(500), 5)
report(result)
```

**Recipe 4: LR test for lag selection**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

m1 = estimate_var(Y, 1)
m2 = estimate_var(Y, 2)
result = lr_test(m1, m2)
report(result)
```

---

## VAR Stationarity

A VAR(``p``) process is covariance-stationary if and only if all eigenvalues of its companion matrix lie strictly inside the unit circle. Stationarity ensures that impulse responses decay, forecasts converge to the unconditional mean, and the asymptotic theory underlying coefficient standard errors and Granger tests remains valid.

The companion form stacks the VAR(``p``) system into a first-order representation:

```math
\xi_t = F \, \xi_{t-1} + v_t
```

where the ``np \times np`` companion matrix is:

```math
F = \begin{bmatrix} A_1 & A_2 & \cdots & A_p \\ I_n & 0 & \cdots & 0 \\ \vdots & & \ddots & \vdots \\ 0 & \cdots & I_n & 0 \end{bmatrix}
```

where:
- ``A_1, \ldots, A_p`` are the ``n \times n`` VAR coefficient matrices
- ``I_n`` is the ``n \times n`` identity matrix
- ``\xi_t = (y_t', y_{t-1}', \ldots, y_{t-p+1}')'`` is the stacked state vector

The stability condition requires ``|\lambda_i(F)| < 1`` for all eigenvalues ``\lambda_i``.

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
m = estimate_var(Y, 2)

result = is_stationary(m)
report(result)

# Access fields directly
result.is_stationary    # true if all eigenvalues inside unit circle
result.max_modulus      # largest eigenvalue modulus
result.eigenvalues      # full eigenvalue vector
```

A maximum modulus below 1.0 confirms stationarity. Values close to but below 1.0 indicate persistent dynamics --- common in macroeconomic VARs where variables exhibit strong serial correlation. A modulus at or above 1.0 signals a unit root or explosive root, and the VAR should be re-specified (e.g., by differencing the data or reducing the lag order).

| Field | Type | Description |
|-------|------|-------------|
| `is_stationary` | `Bool` | `true` if all eigenvalues have modulus strictly less than 1 |
| `eigenvalues` | `Vector` | Eigenvalues of the companion matrix (real or complex) |
| `max_modulus` | `T` | Maximum eigenvalue modulus |
| `companion_matrix` | `Matrix{T}` | The ``np \times np`` companion form matrix ``F`` |

---

## Granger Causality

The Granger causality test (Granger 1969) examines whether lagged values of one variable improve the prediction of another variable in a VAR system. The **pairwise test** evaluates whether a single variable ``j`` Granger-causes variable ``i`` by testing whether all lag coefficients from ``j`` in the ``i``-th equation are jointly zero. The **block test** generalizes this to a group of causing variables.

For the pairwise test, the null hypothesis is:

```math
H_0: A_1[i,j] = A_2[i,j] = \cdots = A_p[i,j] = 0
```

where ``A_l[i,j]`` is the coefficient on the ``l``-th lag of variable ``j`` in the equation for variable ``i``. The Wald test statistic is:

```math
W = \hat{\theta}' \hat{V}^{-1} \hat{\theta} \sim \chi^2(p)
```

where:
- ``\hat{\theta}`` is the ``p \times 1`` vector of restricted coefficients
- ``\hat{V} = \hat{\sigma}_{ii} (X'X)^{-1}_{[\text{restricted}]}`` is the covariance of the restricted coefficients
- ``p`` is the number of lags (degrees of freedom)

For the block test with ``m`` causing variables, ``df = p \times m``.

!!! note "Technical Note"
    Granger causality is a **predictive** concept, not a causal one. Variable ``j`` Granger-causes variable ``i`` if lagged values of ``j`` contain information useful for forecasting ``i`` beyond what is already contained in the lags of ``i`` and all other variables in the VAR. This does not imply a structural causal mechanism.

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
m = estimate_var(Y, 2)

# Pairwise test: does FFR (var 3) Granger-cause INDPRO (var 1)?
g = granger_test(m, 3, 1)
report(g)

# Block test: do CPI and FFR jointly Granger-cause INDPRO?
g_block = granger_test(m, [2, 3], 1)
report(g_block)

# All-pairs causality matrix (n x n, diagonal is nothing)
results = granger_test_all(m)
```

The all-pairs matrix `results[i, j]` tests whether variable ``j`` Granger-causes variable ``i``. Diagonal entries are `nothing`. The matrix display shows p-values with significance stars for quick screening.

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | Wald ``\chi^2`` statistic |
| `pvalue` | `T` | P-value from ``\chi^2(df)`` distribution |
| `df` | `Int` | Degrees of freedom (number of restrictions) |
| `cause` | `Vector{Int}` | Indices of causing variable(s) |
| `effect` | `Int` | Index of effect variable |
| `n` | `Int` | Number of variables in VAR |
| `p` | `Int` | Lag order |
| `nobs` | `Int` | Effective number of observations |
| `test_type` | `Symbol` | `:pairwise` or `:block` |

---

## Normality Tests

Multivariate normality of VAR residuals is required for exact finite-sample inference on coefficient t-statistics and confidence intervals. While OLS coefficient estimates remain consistent regardless of the residual distribution, likelihood-based tests (including the LR test) and bootstrap confidence intervals may perform poorly under non-normality. All normality tests accept both a `VARModel` (extracting residuals automatically) and a raw `AbstractMatrix` for direct use with any residual matrix.

### Jarque-Bera Test

The multivariate Jarque-Bera test (Jarque & Bera 1980; Lutkepohl 2005, Section 4.5) combines measures of multivariate skewness and kurtosis into a single test statistic. The `:multivariate` method computes:

```math
\lambda_{JB} = \lambda_s + \lambda_k = \frac{T \, b_{1,k}}{6} + \frac{T \, (b_{2,k} - k(k+2))^2}{24k}
```

where:
- ``b_{1,k} = \frac{1}{T^2} \sum_{i,j} (u_i' \Sigma^{-1} u_j)^3`` is multivariate skewness
- ``b_{2,k} = \frac{1}{T} \sum_i (u_i' \Sigma^{-1} u_i)^2`` is multivariate kurtosis
- ``k`` is the number of variables
- Under ``H_0``: ``\lambda_{JB} \sim \chi^2(2k)``

The `:component` method applies univariate Jarque-Bera tests to each standardized residual component and sums the statistics, providing per-variable diagnostics via the `components` and `component_pvalues` fields.

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
m = estimate_var(Y, 2)

# Multivariate JB test
jb = jarque_bera_test(m; method=:multivariate)
report(jb)

# Component-wise JB test (per-variable diagnostics)
jb_comp = jarque_bera_test(m; method=:component)
report(jb_comp)
jb_comp.components         # per-variable statistics
jb_comp.component_pvalues  # per-variable p-values
```

### Mardia Test

Mardia's tests (Mardia 1970) assess multivariate normality through skewness and kurtosis measures separately or jointly. Three modes are available:

- `:skewness` --- tests ``H_0: b_{1,k} = 0``. Under the null, ``T \cdot b_{1,k} / 6 \sim \chi^2(k(k+1)(k+2)/6)``
- `:kurtosis` --- tests ``H_0: b_{2,k} = k(k+2)``. The standardized statistic ``(b_{2,k} - k(k+2)) / \sqrt{8k(k+2)/T} \sim N(0,1)``
- `:both` --- combines both tests into a single ``\chi^2`` statistic

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
m = estimate_var(Y, 2)

# Skewness, kurtosis, and combined
ms = mardia_test(m; type=:skewness)
mk = mardia_test(m; type=:kurtosis)
mb = mardia_test(m; type=:both)
report(mb)
```

### Doornik-Hansen Test

The Doornik-Hansen omnibus test (Doornik & Hansen 2008) applies the Bowman-Shenton transformation to each component's skewness and kurtosis, producing approximately ``N(0,1)`` transformed values ``z_1`` and ``z_2``. The test statistic sums ``z_1^2 + z_2^2`` across all ``k`` components:

```math
DH = \sum_{j=1}^{k} (z_{1,j}^2 + z_{2,j}^2) \sim \chi^2(2k)
```

The transformation improves finite-sample size properties compared to the raw Jarque-Bera test, particularly for small samples.

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
m = estimate_var(Y, 2)

dh = doornik_hansen_test(m)
report(dh)
dh.components         # per-component DH statistics
dh.component_pvalues  # per-component p-values
```

### Henze-Zirkler Test

The Henze-Zirkler test (Henze & Zirkler 1990) is based on the empirical characteristic function and provides a consistent test against any non-normal alternative. The test statistic is:

```math
T_{\beta} = \frac{1}{n} \sum_{i,j} e^{-\beta^2 D_{ij}/2} - 2(1+\beta^2)^{-k/2} \sum_i e^{-\beta^2 d_i^2/(2(1+\beta^2))} + n(1+2\beta^2)^{-k/2}
```

where:
- ``D_{ij} = (z_i - z_j)'(z_i - z_j)`` is the squared Euclidean distance between standardized residuals
- ``d_i = z_i' z_i`` is the squared norm of the ``i``-th standardized residual
- ``\beta`` is a bandwidth parameter chosen as a function of ``k`` and ``n``
- The p-value uses a log-normal approximation under ``H_0``

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
m = estimate_var(Y, 2)

hz = henze_zirkler_test(m)
report(hz)
```

### Test Suite

The `normality_test_suite` function runs all seven normality tests at once and returns a `NormalityTestSuite` with a consolidated display. The seven tests are: multivariate Jarque-Bera, component-wise Jarque-Bera, Mardia skewness, Mardia kurtosis, Mardia combined, Doornik-Hansen, and Henze-Zirkler.

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
m = estimate_var(Y, 2)

suite = normality_test_suite(m)
report(suite)

# Access individual results
for r in suite.results
    r.test_name, r.pvalue
end

# Direct matrix dispatch (works without a VARModel)
suite2 = normality_test_suite(m.U)
```

**NormalityTestResult fields:**

| Field | Type | Description |
|-------|------|-------------|
| `test_name` | `Symbol` | Test identifier (`:jarque_bera`, `:mardia_skewness`, etc.) |
| `statistic` | `T` | Test statistic |
| `pvalue` | `T` | P-value |
| `df` | `Int` | Degrees of freedom (for ``\chi^2`` tests; 0 for Henze-Zirkler) |
| `n_vars` | `Int` | Number of variables |
| `n_obs` | `Int` | Number of observations |
| `components` | `Union{Nothing, Vector{T}}` | Per-component statistics (if applicable) |
| `component_pvalues` | `Union{Nothing, Vector{T}}` | Per-component p-values (if applicable) |

**NormalityTestSuite fields:**

| Field | Type | Description |
|-------|------|-------------|
| `results` | `Vector{NormalityTestResult{T}}` | Individual test results (length 7) |
| `residuals` | `Matrix{T}` | The residual matrix tested |
| `n_vars` | `Int` | Number of variables |
| `n_obs` | `Int` | Number of observations |

---

## ARCH Diagnostics

Conditional heteroskedasticity in model residuals violates the constant-variance assumption and affects the efficiency of OLS estimates and the coverage of standard confidence intervals. Two complementary tests detect remaining ARCH effects.

### ARCH-LM Test

The ARCH-LM test (Engle 1982) detects autoregressive conditional heteroskedasticity by regressing squared residuals on their own lags. Under the null of no ARCH effects:

```math
\varepsilon_t^2 = \alpha_0 + \alpha_1 \varepsilon_{t-1}^2 + \cdots + \alpha_q \varepsilon_{t-q}^2 + v_t
```

The test statistic is:

```math
LM = T \cdot R^2 \sim \chi^2(q)
```

where:
- ``T`` is the effective sample size
- ``R^2`` is the coefficient of determination from the auxiliary regression
- ``q`` is the number of lags

The function accepts either a raw data vector (centering and squaring internally) or an `AbstractVolatilityModel` (using standardized residuals to test for remaining ARCH effects after model fitting).

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Test raw series for ARCH effects
y = randn(500)
result = arch_lm_test(y, 5)
report(result)

# Test GARCH model residuals for remaining ARCH effects
g = estimate_garch(y)
result2 = arch_lm_test(g, 10)
report(result2)
```

### Ljung-Box on Squared Residuals

The Ljung-Box test on squared residuals detects serial correlation in the variance process. Under the null of no autocorrelation in ``z_t^2``:

```math
Q = n(n+2) \sum_{k=1}^{K} \frac{\hat{\rho}_k^2}{n-k} \sim \chi^2(K)
```

where:
- ``\hat{\rho}_k`` is the sample autocorrelation of squared residuals at lag ``k``
- ``n`` is the sample size
- ``K`` is the maximum lag

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Test for serial correlation in squared residuals
z = randn(500)
result = ljung_box_squared(z, 10)
report(result)

# Test GARCH standardized residuals
g = estimate_garch(z)
result2 = ljung_box_squared(g, 20)
report(result2)
```

!!! note "Technical Note"
    Use `arch_lm_test` on raw residuals to detect ARCH effects before fitting a volatility model. After fitting, apply both `arch_lm_test` and `ljung_box_squared` to the model object (which uses standardized residuals) to verify the GARCH specification adequately captures the conditional variance dynamics.

---

## Model Comparison Tests

The classical trinity of specification tests --- Wald, likelihood ratio (LR), and Lagrange multiplier (LM) --- provides asymptotically equivalent tests for nested model hypotheses. MacroEconometricModels.jl implements the LR and LM tests with automatic detection of the restricted and unrestricted models.

The **likelihood ratio test** (Wilks 1938) compares the maximized log-likelihoods of the restricted and unrestricted models:

```math
LR = -2(\ell_R - \ell_U) \sim \chi^2(df)
```

where:
- ``\ell_R`` is the maximized log-likelihood under the restricted model
- ``\ell_U`` is the maximized log-likelihood under the unrestricted model
- ``df = k_U - k_R`` is the difference in the number of parameters

The **Lagrange multiplier test** (Rao 1948) evaluates the score of the unrestricted log-likelihood at the restricted parameter estimates:

```math
LM = s' (-H)^{-1} s \sim \chi^2(df)
```

where:
- ``s`` is the score vector (gradient of the log-likelihood) evaluated at the restricted estimates
- ``H`` is the Hessian of the negative log-likelihood at the restricted estimates

!!! note "Technical Note"
    In finite samples, the three test statistics satisfy the inequality ``W \geq LR \geq LM`` (Berndt & Savin 1977). The LR test requires estimating both models, the LM test requires only the restricted model (plus the unrestricted parameterization for score evaluation), and the Wald test requires only the unrestricted model. The LR and LM tests use numerical derivatives with central differences for score and Hessian computation.

The `lr_test` function works for any model pair implementing `loglikelihood`, `dof`, and `nobs` from StatsAPI. The `lm_test` function requires same-family nesting due to its dependence on the parameterization of the unrestricted likelihood.

| Test | Supported Pairs | Notes |
|------|-----------------|-------|
| `lr_test` | Any pair with `loglikelihood`, `dof`, `nobs` | Generic; automatically detects restricted/unrestricted |
| `lm_test` | ARIMA x ARIMA | Same differencing order ``d`` required |
| `lm_test` | VAR x VAR | Same data matrix, different lag orders |
| `lm_test` | ARCH x ARCH, GARCH x GARCH | Same family, different orders |
| `lm_test` | ARCH x GARCH | Cross-type nesting (ARCH is GARCH with ``p=0``) |
| `lm_test` | EGARCH x EGARCH, GJR x GJR | Same family, different orders |

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# --- ARIMA model comparison ---
y = cumsum(randn(200))
ar2 = estimate_ar(diff(y), 2; method=:mle)
ar4 = estimate_ar(diff(y), 4; method=:mle)
report(lr_test(ar2, ar4))
report(lm_test(ar2, ar4))

# --- VAR lag order comparison ---
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
m1 = estimate_var(Y, 1)
m2 = estimate_var(Y, 2)
lr_result = lr_test(m1, m2)
report(lr_result)

# --- GARCH order comparison ---
ret = randn(500)
g11 = estimate_garch(ret; p=1, q=1)
g21 = estimate_garch(ret; p=2, q=1)
report(lr_test(g11, g21))
```

Both functions accept the models in any order --- they automatically determine which is restricted (fewer parameters) and which is unrestricted (more parameters).

**LRTestResult fields:**

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | LR statistic ``= -2(\ell_R - \ell_U)`` |
| `pvalue` | `T` | P-value from ``\chi^2(df)`` distribution |
| `df` | `Int` | Degrees of freedom (``k_U - k_R``) |
| `loglik_restricted` | `T` | Log-likelihood of restricted model |
| `loglik_unrestricted` | `T` | Log-likelihood of unrestricted model |
| `dof_restricted` | `Int` | Number of parameters in restricted model |
| `dof_unrestricted` | `Int` | Number of parameters in unrestricted model |
| `nobs_restricted` | `Int` | Observations in restricted model |
| `nobs_unrestricted` | `Int` | Observations in unrestricted model |

**LMTestResult fields:**

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | LM statistic ``= s'(-H)^{-1}s`` |
| `pvalue` | `T` | P-value from ``\chi^2(df)`` distribution |
| `df` | `Int` | Degrees of freedom (``k_U - k_R``) |
| `nobs` | `Int` | Number of observations |
| `score_norm` | `T` | Euclidean norm of the score vector ``\|s\|_2`` |

---

## Panel VAR Diagnostics

Panel VAR models estimated by GMM require instrument validity checks and model selection criteria. Three functions address these needs: the Hansen J-test for overidentifying restrictions, the Andrews-Lu MMSC criteria for model and moment selection, and an automated lag selection procedure.

### Hansen J-Test

The Hansen (1982) J-test evaluates whether the moment conditions used in GMM estimation are jointly valid. The test statistic is:

```math
J = \left(\sum_{i=1}^{N} Z_i' e_i\right)' \hat{W} \left(\sum_{i=1}^{N} Z_i' e_i\right) \sim \chi^2(q - k)
```

where:
- ``Z_i`` is the instrument matrix for unit ``i``
- ``e_i`` is the residual vector for unit ``i``
- ``\hat{W}`` is the optimal weighting matrix
- ``q`` is the number of instruments, ``k`` is the number of parameters per equation

A significant J-statistic indicates misspecification: either some instruments are invalid or the model is incorrectly specified. The test applies only to GMM-estimated Panel VARs (not FE-OLS).

```julia
using MacroEconometricModels, Random, DataFrames
Random.seed!(42)

# Simulate panel data
N, T_obs = 20, 50
df = DataFrame(
    id = repeat(1:N, inner=T_obs),
    t = repeat(1:T_obs, outer=N),
    y1 = randn(N * T_obs),
    y2 = randn(N * T_obs),
)
pd = xtset(df, :id, :t)
m = estimate_pvar(pd, 2)

j = pvar_hansen_j(m)
report(j)
```

### Andrews-Lu MMSC

The Andrews-Lu (2001) model and moment selection criteria extend standard information criteria to the GMM context by penalizing the Hansen J-statistic:

```math
\begin{aligned}
\text{MMSC-BIC} &= J - (q - k) \cdot \ln(n) \\
\text{MMSC-AIC} &= J - (q - k) \cdot 2 \\
\text{MMSC-HQIC} &= J - Q(q - k) \cdot \ln(\ln(n))
\end{aligned}
```

where ``q - k`` is the number of overidentifying restrictions and ``n`` is the total number of observations. Lower values indicate better model-moment fit.

```julia
using MacroEconometricModels, Random, DataFrames
Random.seed!(42)

N, T_obs = 20, 50
df = DataFrame(
    id = repeat(1:N, inner=T_obs),
    t = repeat(1:T_obs, outer=N),
    y1 = randn(N * T_obs),
    y2 = randn(N * T_obs),
)
pd = xtset(df, :id, :t)
m = estimate_pvar(pd, 2)

mmsc = pvar_mmsc(m)
report(mmsc)
```

### Lag Selection

The `pvar_lag_selection` function estimates Panel VARs for lag orders ``p = 1, \ldots, p_{\max}`` and selects the optimal lag based on MMSC criteria.

```julia
using MacroEconometricModels, Random, DataFrames
Random.seed!(42)

N, T_obs = 20, 50
df = DataFrame(
    id = repeat(1:N, inner=T_obs),
    t = repeat(1:T_obs, outer=N),
    y1 = randn(N * T_obs),
    y2 = randn(N * T_obs),
)
pd = xtset(df, :id, :t)

sel = pvar_lag_selection(pd, 4)
report(sel)
```

---

## Complete Example

A full post-estimation diagnostic workflow for a VAR model estimated on FRED-MD data:

```julia
using MacroEconometricModels

# Load and prepare data
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

# Estimate VAR(2)
m = estimate_var(Y, 2)

# --- Step 1: Check stability ---
stat = is_stationary(m)
report(stat)

# --- Step 2: Granger causality ---
# Does the federal funds rate Granger-cause industrial production?
g = granger_test(m, 3, 1)
report(g)

# All-pairs causality matrix
results = granger_test_all(m)

# --- Step 3: Normality diagnostics ---
suite = normality_test_suite(m)
report(suite)

# --- Step 4: ARCH effects in residuals ---
for j in 1:3
    r = arch_lm_test(m.U[:, j], 5)
    println("Variable $j: ARCH-LM p-value = ", round(r.pvalue, digits=4))
end

# --- Step 5: Compare lag orders ---
m1 = estimate_var(Y, 1)
m3 = estimate_var(Y, 3)

# Test VAR(1) vs VAR(2)
lr12 = lr_test(m1, m)
report(lr12)

# Test VAR(2) vs VAR(3)
lr23 = lr_test(m, m3)
report(lr23)
```

---

## Common Pitfalls

1. **Granger causality does not imply true causation.** Granger causality is a predictive concept conditional on the information set in the VAR. It indicates that lagged values of one variable improve forecasts of another, but this predictive relationship may arise from omitted common causes, measurement timing, or aggregation artifacts rather than from a structural causal mechanism.

2. **Normality rejection does not invalidate the model.** OLS coefficient estimates in a VAR remain consistent and asymptotically normal regardless of the residual distribution. Normality test rejections indicate that exact finite-sample inference (t-tests, F-tests, confidence intervals based on normal critical values) may be unreliable. Consider heteroskedasticity-robust standard errors or bootstrap inference as alternatives.

3. **ARCH-LM: test raw versus standardized residuals.** Apply `arch_lm_test` to raw data or model residuals to detect ARCH effects before fitting a volatility model. After fitting a GARCH model, apply `arch_lm_test` and `ljung_box_squared` to the model object (which uses standardized residuals) to verify that the fitted GARCH specification adequately captures conditional variance dynamics.

4. **LM test requires same-family nesting.** The Lagrange multiplier test computes the score of the unrestricted log-likelihood at the restricted parameter estimates, which requires embedding the restricted parameters into the unrestricted parameter space. This embedding is model-family specific: an ARIMA model cannot be compared with a GARCH model via `lm_test`. Use `lr_test` for cross-family comparisons when both models implement `loglikelihood`.

---

## References

- Andrews, D. W. K., & Lu, B. (2001). Consistent Model and Moment Selection Procedures for GMM Estimation with Application to Dynamic Panel Data Models. *Journal of Econometrics*, 101(1), 123-164. [DOI](https://doi.org/10.1016/S0304-4076(00)00077-4)

- Berndt, E. R., & Savin, N. E. (1977). Conflict Among Criteria for Testing Hypotheses in the Multivariate Linear Regression Model. *Econometrica*, 45(5), 1263-1277. [DOI](https://doi.org/10.2307/1914072)

- Doornik, J. A., & Hansen, H. (2008). An Omnibus Test for Univariate and Multivariate Normality. *Oxford Bulletin of Economics and Statistics*, 70(s1), 927-939. [DOI](https://doi.org/10.1111/j.1468-0084.2008.00537.x)

- Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation. *Econometrica*, 50(4), 987-1007. [DOI](https://doi.org/10.2307/1912773)

- Granger, C. W. J. (1969). Investigating Causal Relations by Econometric Models and Cross-spectral Methods. *Econometrica*, 37(3), 424-438. [DOI](https://doi.org/10.2307/1912791)

- Hansen, L. P. (1982). Large Sample Properties of Generalized Method of Moments Estimators. *Econometrica*, 50(4), 1029-1054. [DOI](https://doi.org/10.2307/1912775)

- Henze, N., & Zirkler, B. (1990). A Class of Invariant Consistent Tests for Multivariate Normality. *Communications in Statistics --- Theory and Methods*, 19(10), 3595-3617. [DOI](https://doi.org/10.1080/03610929008830400)

- Jarque, C. M., & Bera, A. K. (1980). Efficient Tests for Normality, Homoscedasticity and Serial Independence of Regression Residuals. *Economics Letters*, 6(3), 255-259. [DOI](https://doi.org/10.1016/0165-1765(80)90024-5)

- Lutkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Berlin: Springer. ISBN 978-3-540-40172-8.

- Mardia, K. V. (1970). Measures of Multivariate Skewness and Kurtosis with Applications. *Biometrika*, 57(3), 519-530. [DOI](https://doi.org/10.2307/2334770)

- Rao, C. R. (1948). Large Sample Tests of Statistical Hypotheses Concerning Several Parameters with Applications to Problems of Estimation. *Mathematical Proceedings of the Cambridge Philosophical Society*, 44(1), 50-57. [DOI](https://doi.org/10.1017/S0305004100023987)

- Wilks, S. S. (1938). The Large-Sample Distribution of the Likelihood Ratio for Testing Composite Hypotheses. *Annals of Mathematical Statistics*, 9(1), 60-62. [DOI](https://doi.org/10.1214/aoms/1177732360)
