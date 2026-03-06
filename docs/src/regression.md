# Linear Regression

MacroEconometricModels.jl provides a complete suite of cross-sectional linear regression tools covering OLS, WLS, and IV/2SLS estimation with modern robust inference. All estimators produce Stata/EViews-style output via `report()` and integrate with the package's D3.js visualization system.

- **OLS / WLS** estimation with automatic intercept handling
- **Heteroskedasticity-robust standard errors**: HC0 (White 1980), HC1, HC2, HC3 (MacKinnon & White 1985)
- **Cluster-robust standard errors** with finite-sample correction (Arellano 1987)
- **Instrumental variables / 2SLS** with first-stage F-statistic and Sargan-Hansen overidentification test
- **Variance Inflation Factors** for multicollinearity diagnostics (Belsley, Kuh & Welsch 1980)
- **CrossSectionData dispatch** for symbol-based formula-like syntax
- **StatsAPI interface**: `coef`, `vcov`, `predict`, `confint`, `stderror`, `nobs`, `r2`

## Quick Start

**Recipe 1: OLS with robust standard errors**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Simulate a cross-sectional dataset
n = 200
X = hcat(ones(n), randn(n, 2))
y = X * [1.0, 2.0, -0.5] + 0.5 * randn(n)
m = estimate_reg(y, X; varnames=["(Intercept)", "x1", "x2"])
report(m)
```

**Recipe 2: Weighted Least Squares**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Heteroskedastic DGP: variance proportional to x1^2
n = 300
x1 = 1.0 .+ abs.(randn(n))
X = hcat(ones(n), x1)
y = 2.0 .+ 3.0 * x1 + x1 .* randn(n)
w = 1.0 ./ (x1 .^ 2)   # Inverse variance weights
m = estimate_reg(y, X; weights=w, varnames=["(Intercept)", "x1"])
report(m)
```

**Recipe 3: HC3 robust standard errors**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

n = 200
X = hcat(ones(n), randn(n, 3))
y = X * [1.0, 0.5, -1.0, 0.3] + randn(n) .* (1.0 .+ abs.(X[:, 2]))
m = estimate_reg(y, X; cov_type=:hc3, varnames=["(Intercept)", "x1", "x2", "x3"])
report(m)
```

**Recipe 4: IV/2SLS estimation**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Endogenous regressor correlated with the error
n = 500
z1, z2 = randn(n), randn(n)
u = randn(n)
x_endog = 0.5 * z1 + 0.3 * z2 + 0.5 * u + randn(n)
y = 1.0 .+ 2.0 * x_endog + u
X = hcat(ones(n), x_endog)
Z = hcat(ones(n), z1, z2)
m = estimate_iv(y, X, Z; endogenous=[2], varnames=["(Intercept)", "x_endog"])
report(m)
```

**Recipe 5: Variance Inflation Factors**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

n = 200
x1 = randn(n)
x2 = 0.9 * x1 + 0.1 * randn(n)   # Highly correlated with x1
x3 = randn(n)
X = hcat(ones(n), x1, x2, x3)
y = X * [1.0, 2.0, -1.0, 0.5] + randn(n)
m = estimate_reg(y, X; varnames=["(Intercept)", "x1", "x2", "x3"])
v = vif(m)
println("VIF: x1 = ", round(v[1], digits=1),
        ", x2 = ", round(v[2], digits=1),
        ", x3 = ", round(v[3], digits=1))
```

**Recipe 6: CrossSectionData dispatch**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Symbol-based formula-like API
n = 300
data = hcat(randn(n), randn(n), randn(n))
d = CrossSectionData(data; varnames=["wage", "educ", "exper"])
m = estimate_reg(d, :wage, [:educ, :exper])
report(m)
```

---

## Ordinary Least Squares

### The Linear Regression Model

The linear regression model relates a scalar dependent variable ``y_i`` to a ``k \times 1`` vector of regressors ``x_i``:

```math
y_i = x_i' \beta + u_i, \quad i = 1, \ldots, n
```

where:
- ``y_i`` is the dependent variable for observation ``i``
- ``x_i`` is the ``k \times 1`` vector of regressors (including a constant if desired)
- ``\beta`` is the ``k \times 1`` vector of population coefficients
- ``u_i`` is the error term with ``E[u_i | x_i] = 0``

In matrix form, stacking all ``n`` observations:

```math
y = X \beta + u
```

where ``y`` is ``n \times 1``, ``X`` is ``n \times k``, and ``u`` is ``n \times 1``. The OLS estimator minimizes the sum of squared residuals:

```math
\hat{\beta}_{OLS} = \arg\min_{\beta} (y - X\beta)'(y - X\beta) = (X'X)^{-1} X'y
```

Under the Gauss-Markov conditions (``E[u|X] = 0``, ``\text{Var}(u|X) = \sigma^2 I_n``), the OLS estimator is the best linear unbiased estimator (BLUE). The classical covariance matrix is:

```math
\text{Var}(\hat{\beta}) = \sigma^2 (X'X)^{-1}
```

where ``\hat{\sigma}^2 = \hat{u}'\hat{u} / (n - k)`` is the unbiased estimator of the error variance.

**Reference**: Gauss (1821), Markov (1912), Greene (2018, Chapter 4)

### Goodness of Fit

The **R-squared** measures the fraction of variance explained by the model:

```math
R^2 = 1 - \frac{SSR}{TSS} = 1 - \frac{\sum_{i=1}^n \hat{u}_i^2}{\sum_{i=1}^n (y_i - \bar{y})^2}
```

The **adjusted R-squared** penalizes for additional regressors:

```math
\bar{R}^2 = 1 - (1 - R^2) \cdot \frac{n - 1}{n - k}
```

The **F-statistic** tests joint significance of all slope coefficients (``H_0: \beta_2 = \cdots = \beta_k = 0``):

```math
F = \frac{(R \hat{\beta})' [R \hat{V} R']^{-1} (R \hat{\beta})}{q} \sim F_{q, n-k}
```

where ``R`` selects the ``q = k - 1`` slope coefficients and ``\hat{V}`` is the estimated covariance matrix.

Information criteria for model comparison:

```math
AIC = -2 \log L + 2(k + 1), \quad BIC = -2 \log L + \log(n) \cdot (k + 1)
```

where ``\log L = -\frac{n}{2} \log(2\pi) - \frac{n}{2} \log(\hat{\sigma}^2_{ML}) - \frac{n}{2}`` is the Gaussian log-likelihood and ``\hat{\sigma}^2_{ML} = SSR / n``.

### Code Example

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Generate data with known coefficients
n = 500
X = hcat(ones(n), randn(n, 3))
beta_true = [2.0, 1.5, -0.8, 0.3]
y = X * beta_true + 0.5 * randn(n)

# OLS with HC1 robust standard errors (default)
m = estimate_reg(y, X; varnames=["(Intercept)", "education", "experience", "tenure"])
report(m)
```

The estimated coefficients recover the true data-generating process. With ``n = 500`` and ``\sigma = 0.5``, the standard errors are small enough to reject ``H_0: \beta_j = 0`` for all slope coefficients. The ``R^2`` is high because the signal-to-noise ratio is large. The F-statistic rejects the null of joint insignificance at all conventional levels.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `cov_type` | `Symbol` | `:hc1` | Covariance estimator: `:ols`, `:hc0`, `:hc1`, `:hc2`, `:hc3`, `:cluster` |
| `weights` | `Union{Nothing,Vector}` | `nothing` | WLS weights (positive values); `nothing` for OLS |
| `varnames` | `Union{Nothing,Vector{String}}` | `nothing` | Coefficient names (auto-generated if `nothing`) |
| `clusters` | `Union{Nothing,Vector}` | `nothing` | Cluster assignments (required for `:cluster`) |

### Return Values

`estimate_reg` returns a `RegModel{T}` with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `y` | `Vector{T}` | Dependent variable |
| `X` | `Matrix{T}` | ``n \times k`` regressor matrix |
| `beta` | `Vector{T}` | ``k \times 1`` estimated coefficients |
| `vcov_mat` | `Matrix{T}` | ``k \times k`` variance-covariance matrix |
| `residuals` | `Vector{T}` | OLS/WLS residuals ``\hat{u} = y - X\hat{\beta}`` |
| `fitted` | `Vector{T}` | Fitted values ``\hat{y} = X\hat{\beta}`` |
| `ssr` | `T` | Sum of squared residuals |
| `tss` | `T` | Total sum of squares (demeaned) |
| `r2` | `T` | R-squared |
| `adj_r2` | `T` | Adjusted R-squared |
| `f_stat` | `T` | F-statistic for joint significance |
| `f_pval` | `T` | p-value of the F-test |
| `loglik` | `T` | Gaussian log-likelihood |
| `aic` | `T` | Akaike Information Criterion |
| `bic` | `T` | Bayesian Information Criterion |
| `varnames` | `Vector{String}` | Coefficient names |
| `method` | `Symbol` | Estimation method (`:ols`, `:wls`, or `:iv`) |
| `cov_type` | `Symbol` | Covariance estimator used |

---

## Weighted Least Squares

When the error variance is heteroskedastic with a known form ``\text{Var}(u_i | x_i) = \sigma^2 / w_i``, **Weighted Least Squares** (WLS) restores efficiency by transforming the model. The WLS estimator is:

```math
\hat{\beta}_{WLS} = (X'WX)^{-1} X'Wy
```

where:
- ``W = \text{diag}(w_1, w_2, \ldots, w_n)`` is the ``n \times n`` diagonal weight matrix
- ``w_i > 0`` are user-specified weights, typically inverse variance weights

WLS is equivalent to applying OLS to the transformed model ``\sqrt{w_i} \, y_i = \sqrt{w_i} \, x_i' \beta + \sqrt{w_i} \, u_i``, which has homoskedastic errors when the weight specification is correct.

!!! note "Technical Note"
    The package computes residuals from the **original** (untransformed) data: ``\hat{u}_i = y_i - x_i' \hat{\beta}_{WLS}``. Robust standard errors (HC0--HC3) applied under WLS use the original ``X`` and residuals with the WLS bread matrix ``(X'WX)^{-1}``, providing double robustness against weight misspecification.

**Reference**: Aitken (1936), Wooldridge (2010, Chapter 8)

### Code Example

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# DGP with heteroskedastic errors: Var(u_i) = sigma^2 * x1_i^2
n = 500
x1 = 1.0 .+ abs.(randn(n))
X = hcat(ones(n), x1)
sigma_i = 0.5 * x1
y = 2.0 .+ 3.0 * x1 + sigma_i .* randn(n)

# OLS (inefficient under heteroskedasticity)
m_ols = estimate_reg(y, X; varnames=["(Intercept)", "x1"])

# WLS with correct weights (inverse variance)
w = 1.0 ./ (x1 .^ 2)
m_wls = estimate_reg(y, X; weights=w, varnames=["(Intercept)", "x1"])

# Compare standard errors
println("OLS SE(x1):  ", round(stderror(m_ols)[2], digits=4))
println("WLS SE(x1):  ", round(stderror(m_wls)[2], digits=4))
report(m_wls)
```

The WLS standard errors are smaller than the OLS standard errors when the weight specification correctly captures the heteroskedasticity pattern. Both estimators are consistent, but WLS is more efficient. In practice, the weights often come from a preliminary regression of the squared OLS residuals on the regressors (feasible GLS).

---

## Robust Standard Errors

### HC Estimators

Under heteroskedasticity (``\text{Var}(u_i | x_i) \neq \sigma^2``), the classical OLS covariance matrix is inconsistent. White (1980) proposes a **heteroskedasticity-consistent** (HC) covariance estimator using the sandwich form:

```math
\hat{V}_{HC} = (X'X)^{-1} \left( \sum_{i=1}^{n} \hat{\omega}_i \, x_i x_i' \right) (X'X)^{-1}
```

where the weight ``\hat{\omega}_i`` depends on the HC variant:

| Variant | Weight ``\hat{\omega}_i`` | Description |
|---------|---------------------------|-------------|
| **HC0** | ``\hat{u}_i^2`` | White (1980) — asymptotically valid, downward biased in small samples |
| **HC1** | ``\frac{n}{n-k} \hat{u}_i^2`` | Degree-of-freedom correction (Stata default) |
| **HC2** | ``\frac{\hat{u}_i^2}{1 - h_{ii}}`` | Leverage-adjusted (unbiased under homoskedasticity) |
| **HC3** | ``\frac{\hat{u}_i^2}{(1 - h_{ii})^2}`` | Jackknife-like (conservative, best small-sample properties) |

where ``h_{ii} = x_i' (X'X)^{-1} x_i`` is the ``i``-th diagonal element of the hat matrix ``H = X(X'X)^{-1}X'``.

!!! note "Technical Note"
    The leverage ``h_{ii}`` measures how influential observation ``i`` is on the regression fit. High-leverage observations (``h_{ii}`` close to 1) have small residuals mechanically, making HC0 and HC1 underestimate the true variance. HC2 corrects for this exactly under homoskedasticity; HC3 provides conservative inference even under heteroskedasticity. MacKinnon & White (1985) recommend HC3 for small samples (``n < 250``).

**Reference**: White (1980), MacKinnon & White (1985)

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Heteroskedastic DGP
n = 200
X = hcat(ones(n), randn(n, 2))
u = randn(n) .* (1.0 .+ abs.(X[:, 2]))   # Variance depends on x1
y = X * [1.0, 2.0, -0.5] + u

# Compare HC variants
for cov in [:ols, :hc0, :hc1, :hc2, :hc3]
    m = estimate_reg(y, X; cov_type=cov, varnames=["(Intercept)", "x1", "x2"])
    se = stderror(m)
    println(rpad(string(cov), 8), " SE(x1) = ", round(se[2], digits=4),
            "  SE(x2) = ", round(se[3], digits=4))
end
```

The classical OLS standard errors understate the true sampling variability when heteroskedasticity is present. HC1 (the default) provides a simple finite-sample correction that performs well in most applied settings. HC3 produces the most conservative inference and is preferred when the sample contains high-leverage observations.

### Cluster-Robust Standard Errors

When observations within groups (e.g., firms, regions, industries) share common unobserved shocks, standard errors must account for within-cluster correlation. The cluster-robust covariance estimator (Arellano 1987) is:

```math
\hat{V}_{CL} = (X'X)^{-1} \left( \sum_{g=1}^{G} X_g' \hat{u}_g \hat{u}_g' X_g \right) (X'X)^{-1} \cdot \frac{G}{G-1} \cdot \frac{n-1}{n-k}
```

where:
- ``G`` is the number of clusters
- ``X_g`` and ``\hat{u}_g`` are the regressors and residuals for cluster ``g``
- The factor ``\frac{G}{G-1} \cdot \frac{n-1}{n-k}`` is the standard finite-sample correction

Cluster-robust standard errors are consistent as ``G \to \infty``, regardless of the within-cluster correlation structure. A rule of thumb requires at least ``G \geq 50`` clusters for reliable inference (Cameron & Miller 2015).

**Reference**: Arellano (1987), Cameron & Miller (2015)

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# 50 clusters with 20 observations each
G, n_per_g = 50, 20
n = G * n_per_g
clusters = repeat(1:G, inner=n_per_g)

# Cluster-level shock
cluster_effect = randn(G)
X = hcat(ones(n), randn(n))
u = repeat(cluster_effect, inner=n_per_g) + 0.5 * randn(n)
y = X * [1.0, 2.0] + u

m = estimate_reg(y, X; cov_type=:cluster, clusters=clusters,
                 varnames=["(Intercept)", "x1"])
report(m)
```

---

## Instrumental Variables / 2SLS

### The Endogeneity Problem

When a regressor ``x_j`` is correlated with the error term (``E[x_j u] \neq 0``), OLS is biased and inconsistent. Common sources of endogeneity include omitted variables, simultaneity, and measurement error. Instrumental variables estimation resolves this by using **instruments** ``Z`` that are:

1. **Relevant**: correlated with the endogenous regressor (``E[Z'x_j] \neq 0``)
2. **Exogenous**: uncorrelated with the structural error (``E[Z'u] = 0``)

### Two-Stage Least Squares

The 2SLS estimator proceeds in two stages:

**Stage 1** — Project the regressors onto the instrument space:

```math
\hat{X} = P_Z X, \quad P_Z = Z(Z'Z)^{-1}Z'
```

where ``P_Z`` is the projection matrix onto the column space of ``Z``.

**Stage 2** — Regress ``y`` on the projected regressors:

```math
\hat{\beta}_{2SLS} = (\hat{X}'X)^{-1} \hat{X}'y
```

where:
- ``\hat{X}`` is the ``n \times k`` matrix of fitted values from Stage 1
- ``X`` is the original regressor matrix (not ``\hat{X}``) in the cross-product ``\hat{X}'X``
- Residuals use the original regressors: ``\hat{u} = y - X\hat{\beta}_{2SLS}``

!!! note "Technical Note"
    The 2SLS estimator uses ``\hat{X}'X`` (not ``\hat{X}'\hat{X}``) in the bread matrix. This ensures that the covariance matrix ``(\hat{X}'X)^{-1} S (\hat{X}'X)^{-1}`` is correctly centered. The residuals ``\hat{u} = y - X\hat{\beta}`` use the original ``X`` because ``X\hat{\beta} \neq \hat{X}\hat{\beta}`` in general.

**Reference**: Theil (1953), Wooldridge (2010, Chapter 5)

### Diagnostics

**First-stage F-statistic** — Tests instrument relevance by regressing each endogenous variable on all instruments and computing the F-statistic for joint significance of the excluded instruments. The package reports the minimum first-stage F across all endogenous variables. The Staiger & Stock (1997) rule of thumb requires ``F > 10`` for strong instruments; ``F < 10`` indicates weak instruments and unreliable 2SLS inference.

**Sargan-Hansen J-test** — Tests overidentifying restrictions when the number of instruments exceeds the number of endogenous regressors. Under ``H_0`` (all instruments are valid):

```math
J = n \cdot R^2_{aux} \sim \chi^2(m - k)
```

where ``R^2_{aux}`` is from regressing the 2SLS residuals on the instruments, ``m`` is the number of instruments, and ``k`` is the number of regressors. Rejection suggests at least one instrument is invalid. The test has no power when the model is exactly identified (``m = k``).

**Reference**: Sargan (1958), Hansen (1982), Staiger & Stock (1997)

### Code Example

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Classical IV setup: returns to education with ability bias
n = 1000
ability = randn(n)                                    # Unobserved ability
z1 = randn(n)                                         # Instrument 1: distance to college
z2 = randn(n)                                         # Instrument 2: quarter of birth
education = 12.0 .+ 0.5 * z1 .+ 0.3 * z2 .+ 0.4 * ability .+ randn(n)
wage = 5.0 .+ 0.8 * education .+ 0.6 * ability .+ randn(n)

# OLS is biased upward due to ability bias
X = hcat(ones(n), education)
m_ols = estimate_reg(wage, X; varnames=["(Intercept)", "education"])

# 2SLS removes the bias
Z = hcat(ones(n), z1, z2)
m_iv = estimate_iv(wage, X, Z; endogenous=[2], varnames=["(Intercept)", "education"])

println("OLS  beta(education) = ", round(m_ols.beta[2], digits=3))
println("2SLS beta(education) = ", round(m_iv.beta[2], digits=3))
println("True beta(education) = 0.800")
println("First-stage F = ", round(m_iv.first_stage_f, digits=1))
report(m_iv)
```

The OLS coefficient on education is biased upward because ability enters both the education equation and the wage equation. The 2SLS estimator removes this bias by instrumenting education with distance to college and quarter of birth. The first-stage F well exceeds 10, confirming that the instruments are strong. The Sargan test p-value fails to reject the null that both instruments are valid.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `endogenous` | `Vector{Int}` | required | Column indices of endogenous regressors in `X` |
| `cov_type` | `Symbol` | `:hc1` | Covariance estimator: `:ols`, `:hc0`, `:hc1`, `:hc2`, `:hc3` |
| `varnames` | `Union{Nothing,Vector{String}}` | `nothing` | Coefficient names |

### Return Values

`estimate_iv` returns a `RegModel{T}` with `method = :iv` and the following additional fields:

| Field | Type | Description |
|-------|------|-------------|
| `Z` | `Matrix{T}` | ``n \times m`` instrument matrix |
| `endogenous` | `Vector{Int}` | Indices of endogenous regressors in `X` |
| `first_stage_f` | `T` | Minimum first-stage F-statistic across endogenous variables |
| `sargan_stat` | `Union{Nothing,T}` | Sargan-Hansen J-statistic (`nothing` if exactly identified) |
| `sargan_pval` | `Union{Nothing,T}` | p-value of the Sargan test |

---

## VIF Diagnostics

The **Variance Inflation Factor** (VIF) quantifies the degree of multicollinearity for each regressor. For regressor ``x_j``, the VIF is:

```math
\text{VIF}_j = \frac{1}{1 - R_j^2}
```

where ``R_j^2`` is the R-squared from regressing ``x_j`` on all other regressors (excluding the intercept).

| VIF Range | Interpretation |
|-----------|----------------|
| 1 | No collinearity |
| 1 -- 5 | Moderate collinearity (usually acceptable) |
| 5 -- 10 | High collinearity (investigate) |
| > 10 | Severe multicollinearity (remedial action needed) |

A VIF of 10 means that the variance of ``\hat{\beta}_j`` is 10 times larger than it would be if ``x_j`` were uncorrelated with the other regressors. Remedial actions include dropping or combining correlated variables, using ridge regression, or collecting more data.

**Reference**: Belsley, Kuh & Welsch (1980), Greene (2018, Chapter 4)

```julia
using MacroEconometricModels, Random
Random.seed!(42)

n = 300
x1 = randn(n)
x2 = 0.95 * x1 + 0.05 * randn(n)   # Nearly collinear with x1
x3 = randn(n)                        # Independent
X = hcat(ones(n), x1, x2, x3)
y = X * [1.0, 2.0, -1.0, 0.5] + randn(n)

m = estimate_reg(y, X; varnames=["(Intercept)", "x1", "x2", "x3"])
v = vif(m)

# VIF output excludes the intercept
for (i, name) in enumerate(["x1", "x2", "x3"])
    println(rpad(name, 5), " VIF = ", round(v[i], digits=1))
end
```

The VIF values for `x1` and `x2` are large because these two variables are correlated at ``r = 0.95``. The inflated standard errors make it difficult to distinguish the individual effects of `x1` and `x2`. The VIF for `x3` is close to 1, confirming that it is not collinear with the other regressors.

---

## CrossSectionData Dispatch

The `CrossSectionData` wrapper provides a symbol-based API that automatically constructs the regressor matrix with an intercept column and maps variable names:

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Create a CrossSectionData container
n = 500
data = hcat(randn(n), randn(n), randn(n), randn(n))
d = CrossSectionData(data; varnames=["wage", "educ", "exper", "tenure"])

# OLS — symbols select columns, intercept added automatically
m = estimate_reg(d, :wage, [:educ, :exper, :tenure])
report(m)
```

For IV estimation, the `CrossSectionData` dispatch accepts symbol-based arguments for endogenous variables and instruments:

```julia
using MacroEconometricModels, Random
Random.seed!(42)

n = 500
z1, z2 = randn(n), randn(n)
u = randn(n)
educ = 0.5 * z1 + 0.3 * z2 + 0.5 * u + randn(n)
exper = randn(n)
wage = 5.0 .+ 0.8 * educ + 0.3 * exper + u

data = hcat(wage, educ, exper, z1, z2)
d = CrossSectionData(data; varnames=["wage", "educ", "exper", "z1", "z2"])

# Symbol-based IV: specify endogenous and instruments by name
m = estimate_iv(d, :wage, [:educ, :exper], [:z1, :z2]; endogenous=[:educ])
report(m)
```

The `CrossSectionData` dispatch automatically:
- Extracts the dependent variable column by name
- Builds the regressor matrix with an `(Intercept)` column prepended
- Maps `endogenous` symbols to column indices in the regressor matrix
- Constructs the instrument matrix from exogenous regressors plus excluded instruments

---

## Visualization

The `plot_result` function generates D3.js diagnostic plots for regression models:

```julia
using MacroEconometricModels, Random
Random.seed!(42)

n = 200
X = hcat(ones(n), randn(n, 2))
y = X * [1.0, 2.0, -0.5] + 0.5 * randn(n)
m = estimate_reg(y, X; varnames=["(Intercept)", "x1", "x2"])

# OLS diagnostics: residual plot, QQ plot, fitted vs actual
p = plot_result(m)
save_plot(p, "reg_diagnostics.html")
```

```@raw html
<iframe src="../assets/plots/reg_ols_diagnostics.html" width="100%" height="520" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

For IV models, the diagnostic plot includes the first-stage fit:

```julia
using MacroEconometricModels, Random
Random.seed!(42)

n = 500
z1, z2 = randn(n), randn(n)
u = randn(n)
x_endog = 0.5 * z1 + 0.3 * z2 + 0.5 * u + randn(n)
y = 1.0 .+ 2.0 * x_endog + u
X = hcat(ones(n), x_endog)
Z = hcat(ones(n), z1, z2)
m = estimate_iv(y, X, Z; endogenous=[2], varnames=["(Intercept)", "x_endog"])

p = plot_result(m)
save_plot(p, "reg_iv.html")
```

```@raw html
<iframe src="../assets/plots/reg_iv.html" width="100%" height="520" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## Complete Example

This example demonstrates a full cross-sectional regression workflow: data generation, OLS estimation, robust standard error comparison, WLS correction, IV estimation for an endogenous regressor, VIF diagnostics, and visualization.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# ──────────────────────────────────────────────────────────────────────
# Step 1: Generate synthetic cross-sectional data
# ──────────────────────────────────────────────────────────────────────
n = 500
ability = randn(n)                                           # Unobserved
z1 = randn(n)                                                # Instrument 1
z2 = randn(n)                                                # Instrument 2
education = 12.0 .+ 0.5 * z1 .+ 0.3 * z2 .+ 0.4 * ability .+ randn(n)
experience = abs.(5.0 .+ 3.0 * randn(n))
wage = 5.0 .+ 0.8 * education .+ 0.3 * experience .+ 0.6 * ability .+ experience .* 0.2 .* randn(n)

# ──────────────────────────────────────────────────────────────────────
# Step 2: OLS with default HC1 robust SEs
# ──────────────────────────────────────────────────────────────────────
X = hcat(ones(n), education, experience)
m_ols = estimate_reg(wage, X; varnames=["(Intercept)", "education", "experience"])
println("=== OLS Results ===")
println("beta(education)  = ", round(m_ols.beta[2], digits=3), " (biased by ability)")
println("beta(experience) = ", round(m_ols.beta[3], digits=3))
println("R-squared = ", round(m_ols.r2, digits=3))

# ──────────────────────────────────────────────────────────────────────
# Step 3: Compare robust SE variants
# ──────────────────────────────────────────────────────────────────────
println("\n=== Robust SE Comparison ===")
for cov in [:ols, :hc0, :hc1, :hc2, :hc3]
    m = estimate_reg(wage, X; cov_type=cov,
                     varnames=["(Intercept)", "education", "experience"])
    se = stderror(m)
    println(rpad(string(cov), 6), " SE(educ) = ", round(se[2], digits=4),
            "  SE(exper) = ", round(se[3], digits=4))
end

# ──────────────────────────────────────────────────────────────────────
# Step 4: WLS (inverse variance weights based on experience)
# ──────────────────────────────────────────────────────────────────────
w = 1.0 ./ (experience .^ 2)
m_wls = estimate_reg(wage, X; weights=w,
                      varnames=["(Intercept)", "education", "experience"])
println("\n=== WLS Results ===")
println("beta(education) = ", round(m_wls.beta[2], digits=3))

# ──────────────────────────────────────────────────────────────────────
# Step 5: IV/2SLS to correct for endogeneity
# ──────────────────────────────────────────────────────────────────────
Z = hcat(ones(n), z1, z2, experience)   # Instruments + exogenous regressors
m_iv = estimate_iv(wage, X, Z; endogenous=[2],
                    varnames=["(Intercept)", "education", "experience"])
println("\n=== IV/2SLS Results ===")
println("beta(education) = ", round(m_iv.beta[2], digits=3), " (ability bias corrected)")
println("First-stage F   = ", round(m_iv.first_stage_f, digits=1))
if m_iv.sargan_pval !== nothing
    println("Sargan p-value  = ", round(m_iv.sargan_pval, digits=3))
end

# ──────────────────────────────────────────────────────────────────────
# Step 6: VIF diagnostics
# ──────────────────────────────────────────────────────────────────────
v = vif(m_ols)
println("\n=== VIF Diagnostics ===")
for (i, name) in enumerate(["education", "experience"])
    println(name, ": VIF = ", round(v[i], digits=2))
end

# ──────────────────────────────────────────────────────────────────────
# Step 7: Full regression table
# ──────────────────────────────────────────────────────────────────────
println("\n=== Full OLS Report ===")
report(m_ols)

println("\n=== Full IV Report ===")
report(m_iv)

# ──────────────────────────────────────────────────────────────────────
# Step 8: Visualization
# ──────────────────────────────────────────────────────────────────────
p_ols = plot_result(m_ols)
p_iv = plot_result(m_iv)
```

The OLS estimate of the return to education is biased upward because ability is positively correlated with both education and wages. The 2SLS estimator instruments education with distance to college and quarter of birth, recovering a coefficient closer to the true value of 0.8. The first-stage F confirms strong instruments, and the Sargan test does not reject instrument validity. The VIF values are low, indicating no multicollinearity concerns between education and experience.

---

## Common Pitfalls

1. **Forgetting the intercept column.** `estimate_reg` requires the user to include a column of ones in `X` for the intercept. If omitted, the model is estimated without a constant, which biases ``R^2`` and the F-test. The `CrossSectionData` dispatch adds the intercept automatically.

2. **Using `:ols` covariance in the presence of heteroskedasticity.** Classical standard errors assume ``\text{Var}(u|X) = \sigma^2 I``. When this assumption fails, HC-robust or cluster-robust standard errors are necessary. The default `cov_type=:hc1` provides a safe baseline for most applications.

3. **Weak instruments.** A first-stage F below 10 indicates weak instruments, causing 2SLS to be biased toward OLS and confidence intervals to have incorrect coverage. Report the first-stage F and consider weak-instrument-robust inference (Anderson-Rubin test) when ``F < 10``.

4. **Too few clusters.** Cluster-robust standard errors require ``G \to \infty`` for consistency. With fewer than 50 clusters, the finite-sample correction is insufficient and inference becomes unreliable. Consider wild cluster bootstrap methods in such cases.

5. **Interpreting VIF as a strict cutoff.** VIF > 10 is a rule of thumb, not a theorem. Perfectly collinear variables (VIF = ``\infty``) prevent estimation, but moderate collinearity (VIF = 5--10) may be acceptable if the research question does not require precise estimates of individual coefficients.

6. **Confusing WLS weights with frequency weights.** The `weights` argument represents inverse-variance weights (``w_i = 1/\text{Var}(u_i)``), not frequency weights. For frequency-weighted regression, multiply each weight by the observation count.

---

## References

- Aitken, Alexander C. 1936. "On Least Squares and Linear Combination of Observations." *Proceedings of the Royal Society of Edinburgh* 55: 42--48. [DOI](https://doi.org/10.1017/S0370164600014346)

- Arellano, Manuel. 1987. "Computing Robust Standard Errors for Within-Groups Estimators." *Oxford Bulletin of Economics and Statistics* 49 (4): 431--434. [DOI](https://doi.org/10.1111/j.1468-0084.1987.mp49004006.x)

- Belsley, David A., Edwin Kuh, and Roy E. Welsch. 1980. *Regression Diagnostics: Identifying Influential Data and Sources of Collinearity*. New York: Wiley. ISBN 978-0-471-05856-4.

- Cameron, A. Colin, and Douglas L. Miller. 2015. "A Practitioner's Guide to Cluster-Robust Inference." *Journal of Human Resources* 50 (2): 317--372. [DOI](https://doi.org/10.3368/jhr.50.2.317)

- Greene, William H. 2018. *Econometric Analysis*. 8th ed. New York: Pearson. ISBN 978-0-13-446136-6.

- Hansen, Lars Peter. 1982. "Large Sample Properties of Generalized Method of Moments Estimators." *Econometrica* 50 (4): 1029--1054. [DOI](https://doi.org/10.2307/1912775)

- MacKinnon, James G., and Halbert White. 1985. "Some Heteroskedasticity-Consistent Covariance Matrix Estimators with Improved Finite Sample Properties." *Journal of Econometrics* 29 (3): 305--325. [DOI](https://doi.org/10.1016/0304-4076(85)90158-7)

- Sargan, John D. 1958. "The Estimation of Economic Relationships Using Instrumental Variables." *Econometrica* 26 (3): 393--415. [DOI](https://doi.org/10.2307/1907619)

- Staiger, Douglas, and James H. Stock. 1997. "Instrumental Variables Regression with Weak Instruments." *Econometrica* 65 (3): 557--586. [DOI](https://doi.org/10.2307/2171753)

- Stock, James H., and Mark W. Watson. 2019. *Introduction to Econometrics*. 4th ed. New York: Pearson. ISBN 978-0-13-486107-8.

- Theil, Henri. 1953. "Repeated Least Squares Applied to Complete Equation Systems." *The Hague: Central Planning Bureau*.

- White, Halbert. 1980. "A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity." *Econometrica* 48 (4): 817--838. [DOI](https://doi.org/10.2307/1912934)

- Wooldridge, Jeffrey M. 2010. *Econometric Analysis of Cross Section and Panel Data*. 2nd ed. Cambridge, MA: MIT Press. ISBN 978-0-262-23258-6.
