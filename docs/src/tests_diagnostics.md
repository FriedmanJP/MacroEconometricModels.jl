# [Model Diagnostics](@id tests_diagnostics_page)

Post-estimation specification testing validates the assumptions underlying statistical inference in estimated models. This page covers six categories of diagnostic tests: VAR stability checks via companion matrix eigenvalues, Granger causality tests for predictive relationships, multivariate normality tests for residual distributional assumptions, ARCH diagnostics for conditional heteroskedasticity, likelihood-based model comparison tests for nested specifications, and Panel VAR diagnostics for GMM-estimated models.

- **VAR Stationarity**: Companion matrix eigenvalue check for stable dynamics
- **Granger Causality**: Pairwise and block Wald tests for predictive causality (Granger 1969)
- **Normality Tests**: Jarque-Bera, Mardia, Doornik-Hansen, and Henze-Zirkler tests for multivariate normality
- **ARCH Diagnostics**: Engle (1982) ARCH-LM test and Ljung-Box on squared residuals
- **Model Comparison**: Likelihood ratio (Wilks 1938) and Lagrange multiplier (Rao 1948) tests for nested models
- **Panel VAR Diagnostics**: Hansen J-test, Andrews-Lu MMSC, and lag selection for GMM-estimated Panel VARs
- **Variance-Ratio Tests**: Lo-MacKinlay (1988), Chow-Denning (1993), Wright (2000) rank/sign, and Kim (2006) wild bootstrap for the random-walk hypothesis

```@setup test_diag
using MacroEconometricModels, Random, DataFrames
Random.seed!(42)
```

## Quick Start

**Recipe 1: VAR stability and Granger causality**

```@example test_diag
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
m = estimate_var(Y, 2)

# Check VAR stability
stat = is_stationary(m)
show(stdout, stat)

# All-pairs Granger causality
results = granger_test_all(m)
```

**Recipe 2: Normality test suite**

```@example test_diag
# Run all 7 normality tests at once
suite = normality_test_suite(m)
report(suite)
```

**Recipe 3: ARCH-LM test**

```@example test_diag
# Test for ARCH effects in a return series
result = arch_lm_test(randn(500), 5)
(statistic = round(result.statistic, digits=4), pvalue = round(result.pvalue, digits=4), q = result.q)
```

A p-value above 0.05 fails to reject the null of no ARCH effects, consistent with the i.i.d. Gaussian input.

**Recipe 4: LR test for lag selection**

```@example test_diag
m1 = estimate_var(Y, 1)
m2 = estimate_var(Y, 2)
result = lr_test(m1, m2)
show(stdout, result)
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

```@example test_diag
result = is_stationary(m)
show(stdout, result)

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

```@example test_diag
# Pairwise test: does FFR (var 3) Granger-cause INDPRO (var 1)?
g = granger_test(m, 3, 1)
show(stdout, g)

# Block test: do CPI and FFR jointly Granger-cause INDPRO?
g_block = granger_test(m, [2, 3], 1)
show(stdout, g_block)

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

```@example test_diag
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

```@example test_diag
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

```@example test_diag
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

```@example test_diag
hz = henze_zirkler_test(m)
report(hz)
```

### Test Suite

The `normality_test_suite` function runs all seven normality tests at once and returns a `NormalityTestSuite` with a consolidated display. The seven tests are: multivariate Jarque-Bera, component-wise Jarque-Bera, Mardia skewness, Mardia kurtosis, Mardia combined, Doornik-Hansen, and Henze-Zirkler.

```@example test_diag
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

```@example test_diag
# Test raw series for ARCH effects
y = randn(500)
result = arch_lm_test(y, 5)

# Test GARCH model residuals for remaining ARCH effects
g = estimate_garch(y)
result2 = arch_lm_test(g, 10)
(raw = (statistic = round(result.statistic, digits=4), pvalue = round(result.pvalue, digits=4)),
 residuals = (statistic = round(result2.statistic, digits=4), pvalue = round(result2.pvalue, digits=4)))
```

Both tests return large p-values, confirming no ARCH effects in the raw series and none remaining in the fitted GARCH standardized residuals.

### Ljung-Box on Squared Residuals

The Ljung-Box test on squared residuals detects serial correlation in the variance process. Under the null of no autocorrelation in ``z_t^2``:

```math
Q = n(n+2) \sum_{k=1}^{K} \frac{\hat{\rho}_k^2}{n-k} \sim \chi^2(K)
```

where:
- ``\hat{\rho}_k`` is the sample autocorrelation of squared residuals at lag ``k``
- ``n`` is the sample size
- ``K`` is the maximum lag

```@example test_diag
# Test for serial correlation in squared residuals
z = randn(500)
result = ljung_box_squared(z, 10)

# Test GARCH standardized residuals
g_lb = estimate_garch(z)
result2 = ljung_box_squared(g_lb, 20)
(raw = (statistic = round(result.statistic, digits=4), pvalue = round(result.pvalue, digits=4)),
 residuals = (statistic = round(result2.statistic, digits=4), pvalue = round(result2.pvalue, digits=4)))
```

Neither Q² statistic is significant, indicating no serial correlation in the squared series before fitting and none in the standardized residuals after.

!!! note "Technical Note"
    Use `arch_lm_test` on raw residuals to detect ARCH effects before fitting a volatility model. After fitting, apply both `arch_lm_test` and `ljung_box_squared` to the model object (which uses standardized residuals) to verify the GARCH specification adequately captures the conditional variance dynamics.

---

## BDS Independence Test

The Ljung-Box and ARCH-LM tests detect only *linear* dependence or conditional heteroskedasticity of a known form. The Brock-Dechert-Scheinkman-LeBaron (BDS) test (Brock, Dechert, Scheinkman & LeBaron 1996) detects *any* departure from independence and identical distribution --- nonlinear serial dependence, neglected conditional heteroskedasticity, or deterministic chaos --- and is the canonical post-ARIMA/post-GARCH adequacy check.

The test compares the correlation integral of the ``m``-dimensional embedding of the series with what independence would imply. For a distance threshold ``\varepsilon`` and the indicator ``\Theta_{ij} = \mathbf{1}(|y_i - y_j| < \varepsilon)``, the correlation integral is

```math
C_m(\varepsilon) = \frac{2}{T_m(T_m-1)} \sum_{s < t} \prod_{k=0}^{m-1} \Theta_{s+k,\, t+k}, \qquad T_m = T - m + 1 ,
```

and the standardized statistic is

```math
w_m = \sqrt{T}\, \frac{C_m(\varepsilon) - C_1(\varepsilon)^m}{\sigma_m(\varepsilon)} \xrightarrow{d} N(0, 1) ,
```

where:
- ``C_1(\varepsilon)`` is the first-order correlation integral (full sample)
- ``\sigma_m(\varepsilon)`` is the asymptotic standard deviation (Brock et al. 1996), a function of ``C_1`` and the triple-overlap probability ``K``
- ``m`` is the embedding dimension and ``\varepsilon = \texttt{eps\_frac} \cdot \operatorname{sd}(y)``

Under ``H_0`` the observations are iid; large ``|w_m|`` (two-sided) rejects independence. The result reports one row per ``(m, \varepsilon)`` pair.

```@example test_diag
# A: iid data — should NOT reject independence
x = randn(400)
report(bds_test(x; m=2:3, eps_frac=1.0))
```

```@example test_diag
# B: deterministic chaos (logistic map) — strongly rejected
z = Vector{Float64}(undef, 400); z[1] = 0.3
for t in 2:400
    z[t] = 4 * z[t-1] * (1 - z[t-1])
end
report(bds_test(z; m=2:3, eps_frac=0.7))
```

For fitted models, pass the model object directly: `bds_test(model)` tests ARIMA residuals, and for volatility models it tests the **standardized** residuals ``\hat{\varepsilon}_t / \hat{\sigma}_t`` (testing raw returns would merely re-detect the volatility clustering the model already removed).

```@example test_diag
# Post-GARCH adequacy check on standardized residuals
g = estimate_garch(randn(600))
report(bds_test(g; m=2, eps_frac=[1.0, 1.5]))
```

!!! note "Small samples and the bootstrap"
    The asymptotic ``N(0,1)`` approximation is unreliable for ``T < 200`` (a warning is emitted). For short series pass `bootstrap=500` (or more): each replication permutes the series to impose the iid null and recomputes ``w_m``, yielding a permutation p-value that does not rely on the asymptotic distribution.

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

```@example test_diag
# --- ARIMA model comparison ---
y_mc = cumsum(randn(200))
ar2 = estimate_ar(diff(y_mc), 2; method=:mle)
ar4 = estimate_ar(diff(y_mc), 4; method=:mle)
show(stdout, lr_test(ar2, ar4))
show(stdout, lm_test(ar2, ar4))

# --- VAR lag order comparison ---
m1_mc = estimate_var(Y, 1)
m2_mc = estimate_var(Y, 2)
lr_result = lr_test(m1_mc, m2_mc)
show(stdout, lr_result)

# --- GARCH order comparison ---
ret = randn(500)
g11 = estimate_garch(ret, 1, 1)
g21 = estimate_garch(ret, 2, 1)
show(stdout, lr_test(g11, g21))
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

```@example test_diag
using DataFrames  # hide

# Simulate panel data
N_pd, T_pd = 20, 50
df_pd = DataFrame(
    id = repeat(1:N_pd, inner=T_pd),
    t = repeat(1:T_pd, outer=N_pd),
    y1 = randn(N_pd * T_pd),
    y2 = randn(N_pd * T_pd),
)
pd = xtset(df_pd, :id, :t)
m_pd = estimate_pvar(pd, 2)

j = pvar_hansen_j(m_pd)
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

```@example test_diag
mmsc = pvar_mmsc(m_pd)
(bic = round(mmsc.bic, digits=2), aic = round(mmsc.aic, digits=2), hqic = round(mmsc.hqic, digits=2))
```

The specification minimizing MMSC-BIC is preferred; the three criteria differ only in how heavily they penalize additional moment conditions.

### Lag Selection

The `pvar_lag_selection` function estimates Panel VARs for lag orders ``p = 1, \ldots, p_{\max}`` and selects the optimal lag based on MMSC criteria.

```@example test_diag
sel = pvar_lag_selection(pd, 4)
(best_bic = sel.best_bic, best_aic = sel.best_aic, best_hqic = sel.best_hqic)
```

Each criterion reports its own optimal lag order; when they agree, the selected order is robust to the penalty choice.

---

## EDF Goodness-of-Fit Tests

Empirical-distribution-function (EDF) tests compare the sample distribution to a hypothesized continuous distribution through the probability-integral transform ``z_{(i)} = F(y_{(i)}; \theta)``. Unlike the moment-based normality suite above, `edf_test` tests fit against any of seven parametric families and reports the Kolmogorov–Smirnov, Lilliefors, Cramér–von Mises, Anderson–Darling, or Watson statistic. This matches EViews' *Empirical distribution tests* and is the standard tool for checking residuals, PIT transforms, or loss series in risk and forecast-evaluation work.

The five statistics summarise the gap between the empirical and hypothesized CDFs differently:

```math
\begin{aligned}
D   &= \max_i\left[\max\left(\tfrac{i}{n} - z_{(i)},\; z_{(i)} - \tfrac{i-1}{n}\right)\right] & &\text{(Kolmogorov–Smirnov)}\\
W^2 &= \frac{1}{12n} + \sum_{i=1}^n\left(z_{(i)} - \frac{2i-1}{2n}\right)^2 & &\text{(Cramér–von Mises)}\\
A^2 &= -n - \frac{1}{n}\sum_{i=1}^n (2i-1)\left[\ln z_{(i)} + \ln\!\left(1 - z_{(n+1-i)}\right)\right] & &\text{(Anderson–Darling)}\\
U^2 &= W^2 - n\left(\bar{z} - \tfrac{1}{2}\right)^2 & &\text{(Watson)}
\end{aligned}
```

where

- ``z_{(i)}`` are the sorted PIT values,
- ``n`` is the sample size,
- ``\bar{z}`` is the mean of the PIT values.

The Anderson–Darling statistic places the most weight on the tails and is the recommended default.

**Specified versus estimated parameters.** When the distribution parameters are known, `params=:specified` uses distribution-free asymptotics (the Marsaglia–Tsang–Wang exact Kolmogorov CDF for ``n \le 100``, the Marsaglia–Marsaglia ADinf distribution for ``A^2``, and asymptotic tables for ``W^2``/``U^2``). When parameters are estimated from the data, the statistics are no longer distribution-free; for the normal family `edf_test` applies the Stephens (1974) modified statistics with the D'Agostino & Stephens (1986) closed-form p-values, and the Dallal–Wilkinson (1986) approximation for the Lilliefors (estimated-normal KS) statistic.

```@example test_diag
using Distributions
z = rand(MersenneTwister(7), Normal(0.5, 2.0), 300)

# Anderson–Darling with estimated normal parameters (Stephens Case 3)
report(edf_test(z; dist=:normal, test=:ad, params=:estimate))
```

The high p-value fails to reject normality, as expected for Gaussian data. Testing the same series against a fully specified null uses the distribution-free route:

```@example test_diag
# Kolmogorov–Smirnov against a fully specified N(0, 1) — a poor fit here
report(edf_test(z; dist=:normal, test=:ks, params=:specified, theta=(0.0, 1.0)))
```

Non-normal families are supported for the specified route and (where a published null table exists) the estimated route:

```@example test_diag
# Exponential duration data, Anderson–Darling, parameters estimated by ML
d = rand(MersenneTwister(11), Exponential(1.5), 200)
r = edf_test(d; dist=:exponential, test=:ad, params=:specified, theta=(1.5,))
(statistic = round(r.statistic, digits=4), pvalue = round(r.pvalue, digits=4))
```

`edf_test` accepts `dist ∈ (:normal, :exponential, :logistic, :gumbel, :gamma, :weibull, :chisq)` and `test ∈ (:ks, :lilliefors, :cvm, :ad, :watson)`. For estimated-parameter families without a published null distribution, the statistic is returned with `pvalue = NaN` and an explanatory case label rather than an incorrect number.

**Return value.** `EDFTestResult{T}` stores `test`, `dist`, `params`, `statistic` (the value compared to the critical values), `raw_statistic` (the unmodified EDF statistic), `pvalue`, `nobs`, the fitted or specified `theta`, the `critical_values` dictionary, and a human-readable `case` label.
## Variance-Ratio Tests

The variance-ratio test evaluates the **random-walk (martingale) hypothesis** for a level series such as a log price or log exchange rate. Under a random walk the variance of the ``q``-period increment grows linearly in ``q``, so the variance ratio equals one for every aggregation ``q``:

```math
VR(q) = \frac{\operatorname{Var}(y_t - y_{t-q})}{q \, \operatorname{Var}(y_t - y_{t-1})}.
```

`variance_ratio_test` treats its argument as the **level** series and works internally with first differences (returns) ``x_t = y_t - y_{t-1}``. It reports the overlapping Lo–MacKinlay (1988) estimator with the unbiased normalizer ``m = q(N-q+1)(1-q/N)``, the homoskedastic statistic ``Z(q)`` and the heteroskedasticity-robust ``Z^*(q)`` (both asymptotically ``N(0,1)``), and the Chow–Denning (1993) joint statistic ``\max_q |Z(q)|`` whose p-value comes from the studentized-maximum-modulus complement ``1 - (2\Phi(\cdot) - 1)^m``.

- ``VR(q) > 1`` — positive autocorrelation in returns (trending / momentum)
- ``VR(q) < 1`` — negative autocorrelation (mean reversion)

```@example test_diag
rw = cumsum(randn(600))            # a simulated random walk (level series)
vr = variance_ratio_test(rw; q=[2, 4, 8, 16])
show(stdout, vr)
```

The Chow–Denning joint test controls the overall size across all ``q`` simultaneously — unlike inspecting each ``Z(q)`` separately, it does not inflate the familywise error rate.

### Wright rank/sign and wild-bootstrap variants

`method=:wright` adds Wright's (2000) rank (``R1``, ``R2``) and sign (``S1``) statistics, whose exact iid-null distributions are simulated on demand (and cached). These are robust to non-normality and often more powerful in small samples. `bootstrap=B` adds Kim's (2006) wild-bootstrap p-values for ``Z^*(q)`` and the Chow–Denning statistic.

```@example test_diag
ar1 = zeros(800)
for t in 2:800
    ar1[t] = 0.5 * ar1[t-1] + randn()   # mean-reverting: not a random walk
end
vr2 = variance_ratio_test(ar1; q=[2, 4, 8], method=:wright, bootstrap=299)
(vr = round.(vr2.vr, digits=3),
 cd_star = round(vr2.cd_star_stat, digits=3),
 cd_boot_p = round(vr2.cd_boot_pvalue, digits=4))
```

The mean-reverting AR(1) level drives ``VR(q)`` below one and the robust Chow–Denning test rejects the random-walk null.
## Basic Statistics: Equality-of-Distribution and Rank Correlation Tests

The "Basic statistics" battery compares the location, scale, or distribution of a response across the groups of a classifier, and measures the association between two series. This delivers EViews "Equality Tests by Classification" parity. The single entry point `equality_test(y, g; test=...)` groups `y` by the distinct values of `g`; `cor_test(x, y; method=...)` returns a rank/product-moment correlation. Both dispatch on `CrossSectionData`/`PanelData` via column symbols.

Location tests: one/two-sample and paired t (pooled or Welch/Satterthwaite), one-way ANOVA (classic F or Welch), Mann–Whitney U, Wilcoxon signed-rank, Kruskal–Wallis H, van der Waerden normal-scores, and the Mood median χ². Scale tests: two-group variance F, Bartlett, Levene (center = group mean), Brown–Forsythe (center = group median), and Siegel–Tukey. The rank tests use the exact null for small tie-free samples and otherwise a continuity- and tie-corrected normal approximation, mirroring R's `wilcox.test`/`kruskal.test` behavior.

!!! note "Grouped data vs. regression residuals"
    The Levene and Brown–Forsythe tests here operate on **raw data split by a classifier**. For the regression-residual heteroskedasticity variants (deviations of fitted residuals), use the diagnostics in the cross-sectional regression module.

```@setup test_basicstat
using MacroEconometricModels, Random
```

Compare a response across three groups and test dispersion and association:

```@example test_basicstat
# three groups of a response
g1 = [5.1, 4.9, 6.2, 5.7, 6.0, 5.5]
g2 = [6.1, 5.9, 7.2, 6.8, 6.5]
g3 = [7.0, 7.5, 6.9, 8.1, 7.3, 7.8]
y = vcat(g1, g2, g3)
grp = vcat(fill(1, 6), fill(2, 5), fill(3, 6))

# one-way ANOVA (equal-variance F)
report(anova_test(y, grp))
```

```@example test_basicstat
# Welch ANOVA (unequal variances) and the tie-corrected Kruskal–Wallis H
report(equality_test(y, grp; test=:anova, equal_var=false))
report(equality_test(y, grp; test=:kruskal_wallis))
```

```@example test_basicstat
# scale: Bartlett and the robust Brown–Forsythe test
report(equality_test(y, grp; test=:bartlett))
report(equality_test(y, grp; test=:brown_forsythe))
```

Two-sample location and the Mann–Whitney U (exact null when tie-free):

```@example test_basicstat
y12 = vcat(g1, g2); grp12 = vcat(fill(1, 6), fill(2, 5))
report(equality_test(y12, grp12; test=:t, equal_var=false))  # Welch t, groups 1 & 2
```

```@example test_basicstat
report(equality_test(y12, grp12; test=:mann_whitney))
```

Association between two series — Pearson, Spearman, and Kendall τ_b (the
concordant−discordant count is computed in `O(n log n)` via a merge-sort inversion
counter):

```@example test_basicstat
x = [10.0, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
z = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
report(cor_test(x, z; method=:pearson))
```

```@example test_basicstat
report(cor_test(x, z; method=:spearman))
report(cor_test(x, z; method=:kendall))
```

---

## Complete Example

A full post-estimation diagnostic workflow for a VAR model estimated on FRED-MD data:

```@example test_diag
# --- Step 1: Check stability ---
stat = is_stationary(m)
show(stdout, stat)

# --- Step 2: Granger causality ---
# Does the federal funds rate Granger-cause industrial production?
g = granger_test(m, 3, 1)
show(stdout, g)

# All-pairs causality matrix
results = granger_test_all(m)

# --- Step 3: Normality diagnostics ---
suite = normality_test_suite(m)
report(suite)

# --- Step 4: ARCH effects in residuals ---
[(variable=j, arch_lm_pvalue=round(arch_lm_test(m.U[:, j], 5).pvalue, digits=4))
 for j in 1:3]

# --- Step 5: Compare lag orders ---
m1_ce = estimate_var(Y, 1)
m3_ce = estimate_var(Y, 3)

# Test VAR(1) vs VAR(2)
lr12 = lr_test(m1_ce, m)
show(stdout, lr12)

# Test VAR(2) vs VAR(3)
lr23 = lr_test(m, m3_ce)
show(stdout, lr23)
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

- Bartlett, M. S. (1937). Properties of Sufficiency and Statistical Tests. *Proceedings of the Royal Society A*, 160(901), 268-282. [DOI](https://doi.org/10.1098/rspa.1937.0109)

- Brown, M. B., & Forsythe, A. B. (1974). Robust Tests for the Equality of Variances. *Journal of the American Statistical Association*, 69(346), 364-367. [DOI](https://doi.org/10.1080/01621459.1974.10482955)

- Kendall, M. G. (1938). A New Measure of Rank Correlation. *Biometrika*, 30(1/2), 81-93. [DOI](https://doi.org/10.1093/biomet/30.1-2.81)

- Knight, W. R. (1966). A Computer Method for Calculating Kendall's Tau with Ungrouped Data. *Journal of the American Statistical Association*, 61(314), 436-439. [DOI](https://doi.org/10.1080/01621459.1966.10480879)

- Kruskal, W. H., & Wallis, W. A. (1952). Use of Ranks in One-Criterion Variance Analysis. *Journal of the American Statistical Association*, 47(260), 583-621. [DOI](https://doi.org/10.1080/01621459.1952.10483441)

- Mann, H. B., & Whitney, D. R. (1947). On a Test of Whether one of Two Random Variables is Stochastically Larger than the Other. *The Annals of Mathematical Statistics*, 18(1), 50-60. [DOI](https://doi.org/10.1214/aoms/1177730491)

- Spearman, C. (1904). The Proof and Measurement of Association between Two Things. *The American Journal of Psychology*, 15(1), 72-101. [DOI](https://doi.org/10.2307/1412159)

- van der Waerden, B. L. (1952). Order Tests for the Two-Sample Problem and Their Power. *Indagationes Mathematicae*, 14, 453-458.

- Wilcoxon, F. (1945). Individual Comparisons by Ranking Methods. *Biometrics Bulletin*, 1(6), 80-83. [DOI](https://doi.org/10.2307/3001968)

- Berndt, E. R., & Savin, N. E. (1977). Conflict Among Criteria for Testing Hypotheses in the Multivariate Linear Regression Model. *Econometrica*, 45(5), 1263-1277. [DOI](https://doi.org/10.2307/1914072)

- Anderson, T. W., & Darling, D. A. (1954). A Test of Goodness of Fit. *Journal of the American Statistical Association*, 49(268), 765-769. [DOI](https://doi.org/10.1080/01621459.1954.10501232)

- D'Agostino, R. B., & Stephens, M. A. (1986). *Goodness-of-Fit Techniques*. New York: Marcel Dekker. ISBN 978-0-8247-7487-5.

- Dallal, G. E., & Wilkinson, L. (1986). An Analytic Approximation to the Distribution of Lilliefors's Test Statistic for Normality. *The American Statistician*, 40(4), 294-296. [DOI](https://doi.org/10.1080/00031305.1986.10475419)
- Chow, K. V., & Denning, K. C. (1993). A Simple Multiple Variance Ratio Test. *Journal of Econometrics*, 58(3), 385-401. [DOI](https://doi.org/10.1016/0304-4076(93)90051-6)
- Brock, W. A., Dechert, W. D., Scheinkman, J. A., & LeBaron, B. (1996). A Test for Independence Based on the Correlation Dimension. *Econometric Reviews*, 15(3), 197-235. [DOI](https://doi.org/10.1080/07474939608800353)

- Brock, W. A., Hsieh, D. A., & LeBaron, B. (1991). *Nonlinear Dynamics, Chaos, and Instability: Statistical Theory and Economic Evidence*. Cambridge, MA: MIT Press. ISBN 978-0-262-02329-0.

- Doornik, J. A., & Hansen, H. (2008). An Omnibus Test for Univariate and Multivariate Normality. *Oxford Bulletin of Economics and Statistics*, 70(s1), 927-939. [DOI](https://doi.org/10.1111/j.1468-0084.2008.00537.x)

- Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation. *Econometrica*, 50(4), 987-1007. [DOI](https://doi.org/10.2307/1912773)

- Granger, C. W. J. (1969). Investigating Causal Relations by Econometric Models and Cross-spectral Methods. *Econometrica*, 37(3), 424-438. [DOI](https://doi.org/10.2307/1912791)

- Hansen, L. P. (1982). Large Sample Properties of Generalized Method of Moments Estimators. *Econometrica*, 50(4), 1029-1054. [DOI](https://doi.org/10.2307/1912775)

- Henze, N., & Zirkler, B. (1990). A Class of Invariant Consistent Tests for Multivariate Normality. *Communications in Statistics --- Theory and Methods*, 19(10), 3595-3617. [DOI](https://doi.org/10.1080/03610929008830400)

- Jarque, C. M., & Bera, A. K. (1980). Efficient Tests for Normality, Homoscedasticity and Serial Independence of Regression Residuals. *Economics Letters*, 6(3), 255-259. [DOI](https://doi.org/10.1016/0165-1765(80)90024-5)

- Lilliefors, H. W. (1967). On the Kolmogorov-Smirnov Test for Normality with Mean and Variance Unknown. *Journal of the American Statistical Association*, 62(318), 399-402. [DOI](https://doi.org/10.1080/01621459.1967.10482916)
- Kim, J. H. (2006). Wild Bootstrapping Variance Ratio Tests. *Economics Letters*, 92(1), 38-43. [DOI](https://doi.org/10.1016/j.econlet.2006.01.007)

- Lo, A. W., & MacKinlay, A. C. (1988). Stock Market Prices Do Not Follow Random Walks: Evidence from a Simple Specification Test. *Review of Financial Studies*, 1(1), 41-66. [DOI](https://doi.org/10.1093/rfs/1.1.41)

- Lutkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Berlin: Springer. ISBN 978-3-540-40172-8.

- Marsaglia, G., Tsang, W. W., & Wang, J. (2003). Evaluating Kolmogorov's Distribution. *Journal of Statistical Software*, 8(18), 1-4. [DOI](https://doi.org/10.18637/jss.v008.i18)

- Mardia, K. V. (1970). Measures of Multivariate Skewness and Kurtosis with Applications. *Biometrika*, 57(3), 519-530. [DOI](https://doi.org/10.2307/2334770)

- Rao, C. R. (1948). Large Sample Tests of Statistical Hypotheses Concerning Several Parameters with Applications to Problems of Estimation. *Mathematical Proceedings of the Cambridge Philosophical Society*, 44(1), 50-57. [DOI](https://doi.org/10.1017/S0305004100023987)

- Stephens, M. A. (1974). EDF Statistics for Goodness of Fit and Some Comparisons. *Journal of the American Statistical Association*, 69(347), 730-737. [DOI](https://doi.org/10.1080/01621459.1974.10480196)
- Wright, J. H. (2000). Alternative Variance-Ratio Tests Using Ranks and Signs. *Journal of Business & Economic Statistics*, 18(1), 1-9. [DOI](https://doi.org/10.1080/07350015.2000.10524842)

- Wilks, S. S. (1938). The Large-Sample Distribution of the Likelihood Ratio for Testing Composite Hypotheses. *Annals of Mathematical Statistics*, 9(1), 60-62. [DOI](https://doi.org/10.1214/aoms/1177732360)
