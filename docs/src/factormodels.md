# Factor Models

This chapter covers static factor models for dimensionality reduction in large macroeconomic panels, including estimation via principal components and information criteria for selecting the number of factors.

## Introduction

Factor models are fundamental tools in macroeconometrics for extracting common sources of variation from large panels of economic indicators. They enable:

1. **Dimensionality Reduction**: Summarize ``N`` variables with ``r \ll N`` factors
2. **Forecasting**: Use factors as predictors in regressions (diffusion indices)
3. **Structural Analysis**: Identify common shocks driving multiple series
4. **FAVAR**: Combine factors with VARs for high-dimensional structural analysis

**Reference**: Stock & Watson (2002a, 2002b), Bai & Ng (2002)

## Quick Start

```julia
fm = estimate_factors(X, r; standardize=true)                       # Static factor model via PCA
ic = ic_criteria(X, 10)                                             # Bai-Ng IC for factor count
dfm = estimate_dynamic_factors(X, r, p; method=:twostep)            # Dynamic factor model
gdfm = estimate_gdfm(X, q; kernel=:bartlett)                        # Generalized DFM (spectral)
fc = forecast(dfm, h)                                               # DFM forecasting
```

---

## The Static Factor Model

### Model Specification

The static factor model decomposes an ``N``-dimensional vector of observables ``x_t`` into common and idiosyncratic components:

```math
x_{it} = \lambda_i' F_t + e_{it}, \quad i = 1, \ldots, N, \quad t = 1, \ldots, T
```

In matrix form:

```math
X = F \Lambda' + E
```

where:
- ``X`` is the ``T \times N`` data matrix
- ``F`` is the ``T \times r`` matrix of latent factors
- ``\Lambda`` is the ``N \times r`` matrix of factor loadings
- ``E`` is the ``T \times N`` matrix of idiosyncratic errors
- ``r`` is the number of factors (with ``r \ll \min(T, N)``)

### Assumptions

**Factors and Loadings**:
- ``E[F_t] = 0``, ``\text{Var}(F_t) = I_r`` (normalization)
- ``\frac{1}{T} \sum_t F_t F_t' \xrightarrow{p} \Sigma_F`` positive definite
- ``\frac{1}{N} \Lambda' \Lambda \xrightarrow{p} \Sigma_\Lambda`` positive definite

**Idiosyncratic Errors**:
- ``E[e_{it}] = 0``
- Weak cross-sectional and temporal dependence allowed
- Weak correlation with factors: ``\frac{1}{NT} \sum_{i,t} |E[F_t e_{it}]| \to 0``

**Reference**: Bai & Ng (2002), Bai (2003)

---

## Estimation via Principal Components

### Principal Components Analysis (PCA)

The factors and loadings are estimated by minimizing the sum of squared idiosyncratic errors:

```math
\min_{F, \Lambda} \sum_{i=1}^N \sum_{t=1}^T (x_{it} - \lambda_i' F_t)^2
```

subject to the normalization ``F'F/T = I_r``.

### Solution

The solution involves the eigenvalue decomposition of ``X'X`` (or ``XX'``):

**Case 1**: ``T < N`` (short panel)
- Compute ``XX'`` (``T \times T`` matrix)
- ``\hat{F} = \sqrt{T} \times`` (first ``r`` eigenvectors of ``XX'``)
- ``\hat{\Lambda} = X' \hat{F} / T``

**Case 2**: ``N \leq T`` (tall panel)
- Compute ``X'X`` (``N \times N`` matrix)
- ``\hat{\Lambda} = \sqrt{N} \times`` (first ``r`` eigenvectors of ``X'X``)
- ``\hat{F} = X \hat{\Lambda} / N``

### Data Preprocessing

Before estimation, data is typically:
1. **Demeaned**: Center each series to have zero mean
2. **Standardized**: Scale each series to have unit variance

This prevents high-variance series from dominating the factor extraction.

### Identification

The factors and loadings are identified only up to an ``r \times r`` invertible rotation. If ``(F, \Lambda)`` is a solution, so is ``(FH, \Lambda H^{-1})`` for any invertible ``H``.

The normalization ``F'F/T = I_r`` and ``\Lambda'\Lambda`` diagonal pins down rotation up to sign.

**Reference**: Stock & Watson (2002a), Bai & Ng (2002)

### Julia Implementation

```julia
using MacroEconometricModels

# X is T×N data matrix
# Estimate r-factor model

model = estimate_factors(X, r;
    standardize = true,    # Standardize data
    method = :pca          # Principal components
)

# Access results
F = model.factors          # T×r estimated factors
Λ = model.loadings         # N×r estimated loadings
```

### FactorModel Return Values

| Field | Type | Description |
|-------|------|-------------|
| `X` | `Matrix{T}` | Original ``T \times N`` data matrix |
| `factors` | `Matrix{T}` | ``T \times r`` estimated factor matrix |
| `loadings` | `Matrix{T}` | ``N \times r`` estimated loading matrix |
| `eigenvalues` | `Vector{T}` | Eigenvalues from PCA (in descending order) |
| `explained_variance` | `Vector{T}` | Fraction of variance explained by each factor |
| `cumulative_variance` | `Vector{T}` | Cumulative fraction of variance explained |
| `r` | `Int` | Number of factors |
| `standardized` | `Bool` | Whether data was standardized before estimation |

!!! note "Technical Note"
    Factor models are identified only up to an ``r \times r`` rotation: if ``(\hat{F}, \hat{\Lambda})`` is a solution, then ``(\hat{F}H, \hat{\Lambda}H^{-1'})`` is equally valid for any invertible ``H``. The normalization ``F'F/T = I_r`` pins down orientation but not sign. Consequently, individual factor loadings should not be interpreted as structural parameters. To compare estimated factors with "true" factors (e.g., in simulations), compute absolute correlations rather than raw correlations.

---

## Determining the Number of Factors

### The Selection Problem

Choosing ``r`` is crucial:
- Too few factors: Omitted common variation, biased estimates
- Too many factors: Overfitting, including noise as signal

### Bai & Ng (2002) Information Criteria

Bai & Ng propose three information criteria:

**IC1**:
```math
IC_1(r) = \log \hat{\sigma}^2(r) + r \cdot \frac{N + T}{NT} \log\left( \frac{NT}{N+T} \right)
```

**IC2**:
```math
IC_2(r) = \log \hat{\sigma}^2(r) + r \cdot \frac{N + T}{NT} \log(C_{NT}^2)
```

**IC3**:
```math
IC_3(r) = \log \hat{\sigma}^2(r) + r \cdot \frac{\log(C_{NT}^2)}{C_{NT}^2}
```

where:
- ``\hat{\sigma}^2(r) = \frac{1}{NT} \sum_{i,t} \hat{e}_{it}^2`` is the average squared residual
- ``C_{NT}^2 = \min(N, T)``

**Selection Rule**: Choose ``\hat{r}`` that minimizes ``IC_k(r)`` over ``r \in \{1, \ldots, r_{max}\}``.

**Properties**:
- IC2 and IC3 perform best in simulations
- All three are consistent: ``\hat{r} \xrightarrow{p} r_0`` as ``N, T \to \infty``

**Reference**: Bai & Ng (2002)

### Julia Implementation

```julia
using MacroEconometricModels

# Compute IC for r = 1, ..., r_max
r_max = 10
ic = ic_criteria(X, r_max)

# Optimal number by each criterion
println("IC1 selects: ", ic.r_IC1, " factors")
println("IC2 selects: ", ic.r_IC2, " factors")
println("IC3 selects: ", ic.r_IC3, " factors")

# IC values for all r
for r in 1:r_max
    println("r=$r: IC1=$(ic.IC1[r]), IC2=$(ic.IC2[r]), IC3=$(ic.IC3[r])")
end
```

---

## Scree Plot Analysis

### Visual Factor Selection

The scree plot displays eigenvalues (or variance explained) against factor number. The "elbow" in the plot suggests the number of significant factors.

### Variance Explained

For each factor ``j``:

**Individual Variance**:
```math
\text{VarExp}_j = \frac{\mu_j}{\sum_{k=1}^N \mu_k}
```

**Cumulative Variance**:
```math
\text{CumVarExp}_r = \sum_{j=1}^r \text{VarExp}_j
```

where ``\mu_j`` is the ``j``-th largest eigenvalue of ``X'X/T`` (or ``XX'/N``).

### Julia Implementation

```julia
using MacroEconometricModels

model = estimate_factors(X, r)

# Get scree plot data
scree = scree_plot_data(model)

# Variance explained
for j in 1:min(10, length(scree.factors))
    println("Factor $j: $(round(scree.explained_variance[j]*100, digits=2))% ",
            "(cumulative: $(round(scree.cumulative_variance[j]*100, digits=2))%)")
end
```

---

## Model Diagnostics

### R-squared for Each Variable

The ``R^2`` measures how much of variable ``i``'s variation is explained by the common factors:

```math
R^2_i = 1 - \frac{\sum_t \hat{e}_{it}^2}{\sum_t (x_{it} - \bar{x}_i)^2}
```

Variables with low ``R^2`` are mainly driven by idiosyncratic shocks.

### Julia Implementation

```julia
using MacroEconometricModels

model = estimate_factors(X, r)

# R² for each variable
r2_values = r2(model)

# Summary statistics
println("Mean R²: ", round(mean(r2_values), digits=3))
println("Median R²: ", round(median(r2_values), digits=3))
println("Min R²: ", round(minimum(r2_values), digits=3))
println("Max R²: ", round(maximum(r2_values), digits=3))

# Variables well-explained by factors
well_explained = findall(r2_values .> 0.7)
```

### Fitted Values and Residuals

```julia
# Fitted values: X̂ = FΛ'
X_fitted = predict(model)

# Residuals: E = X - X̂
resid = residuals(model)

# Model statistics
println("Number of observations: ", nobs(model))
println("Degrees of freedom: ", dof(model))
```

---

## Applications

### Diffusion Index Forecasting

Use factors as predictors for forecasting a target variable ``y_{t+h}``:

```math
y_{t+h} = \alpha + \beta' \hat{F}_t + \gamma' y_{t:t-p} + \varepsilon_{t+h}
```

Factors summarize information from a large panel, improving forecast accuracy.

**Reference**: Stock & Watson (2002b)

### Factor-Augmented VAR (FAVAR)

Combine factors with key observable variables in a VAR:

```math
\begin{bmatrix} y_t \\ F_t \end{bmatrix} = A_1 \begin{bmatrix} y_{t-1} \\ F_{t-1} \end{bmatrix} + \cdots + A_p \begin{bmatrix} y_{t-p} \\ F_{t-p} \end{bmatrix} + u_t
```

This allows structural analysis with high-dimensional information sets.

**Reference**: Bernanke, Boivin & Eliasz (2005)

### Example: FAVAR Setup

```julia
using MacroEconometricModels

# Estimate factors from large panel X
fm = estimate_factors(X, r)
F = fm.factors

# Combine with key observables (e.g., FFR, GDP, inflation)
Y_key = Matrix(data[:, [:FFR, :GDP, :CPI]])
Y_favar = hcat(Y_key, F)

# Estimate FAVAR
favar_model = estimate_var(Y_favar, p)

# Structural analysis
irf_favar = irf(favar_model, H; method=:cholesky)
```

---

## Asymptotic Theory

### Consistency of Factor Estimates

Under the assumptions of Bai & Ng (2002), as ``T, N \to \infty``:

```math
\frac{1}{T} \sum_{t=1}^T \|\hat{F}_t - H F_t\|^2 = O_p\left( \frac{1}{\min(N, T)} \right)
```

where ``H`` is an ``r \times r`` rotation matrix.

The factors are consistently estimated up to rotation at rate ``\min(\sqrt{N}, \sqrt{T})``.

### Distribution Theory

For large ``N, T``, the factor estimates are asymptotically normal:

```math
\sqrt{T} (\hat{F}_t - H F_t) \xrightarrow{d} N(0, V)
```

where ``V`` depends on the cross-sectional and temporal dependence structure.

**Reference**: Bai (2003), Bai & Ng (2006)

---

## Comparison with Other Methods

### Static vs. Dynamic Factor Models

| Aspect | Static FM | Dynamic FM |
|--------|-----------|------------|
| **Model** | ``X_t = \Lambda F_t + e_t`` | ``X_t = \Lambda(L) f_t + e_t`` |
| **Factors** | Contemporaneous | May include lags |
| **Estimation** | PCA | Spectral methods, Kalman filter |
| **Use case** | Large N, moderate T | Time series dynamics important |

**Reference**: Forni, Hallin, Lippi & Reichlin (2000)

### Maximum Likelihood Estimation

ML estimation assumes Gaussian factors and errors:

```math
F_t \sim N(0, I_r), \quad e_t \sim N(0, \Psi)
```

Estimated via EM algorithm. More efficient than PCA if model is correctly specified, but computationally intensive.

---

## Dynamic Factor Models

### Model Specification

The dynamic factor model extends the static model by allowing factors to follow a VAR process:

**Observation Equation**:
```math
X_t = \Lambda F_t + e_t
```

**State Equation (Factor Dynamics)**:
```math
F_t = A_1 F_{t-1} + A_2 F_{t-2} + \cdots + A_p F_{t-p} + \eta_t
```

where:
- ``F_t`` is the ``r \times 1`` vector of latent factors
- ``\Lambda`` is the ``N \times r`` loading matrix
- ``A_1, \ldots, A_p`` are ``r \times r`` autoregressive coefficient matrices
- ``\eta_t \sim N(0, \Sigma_\eta)`` are factor innovations
- ``e_t \sim N(0, \Sigma_e)`` are idiosyncratic errors (typically diagonal)

### Estimation Methods

**Two-Step Estimation**:
1. Extract factors using PCA (as in static model)
2. Estimate VAR(p) on extracted factors

**EM Algorithm**:
- Iterates between E-step (Kalman smoother) and M-step (parameter updates)
- More efficient but computationally intensive

### Julia Implementation

```julia
using MacroEconometricModels

# Estimate dynamic factor model with r factors and p lags
model = estimate_dynamic_factors(X, r, p;
    method = :twostep,      # or :em
    standardize = true,
    diagonal_idio = true    # Diagonal idiosyncratic covariance
)

# Access results
F = model.factors           # T×r estimated factors
Λ = model.loadings          # N×r loadings
A = model.A                 # Vector of r×r AR coefficient matrices
Σ_η = model.Sigma_eta       # r×r factor innovation covariance
Σ_e = model.Sigma_e         # N×N idiosyncratic covariance
```

### DynamicFactorModel Return Values

| Field | Type | Description |
|-------|------|-------------|
| `X` | `Matrix{T}` | Original ``T \times N`` data matrix |
| `factors` | `Matrix{T}` | ``T \times r`` estimated factors |
| `loadings` | `Matrix{T}` | ``N \times r`` loading matrix |
| `A` | `Vector{Matrix{T}}` | ``r \times r`` autoregressive coefficient matrices |
| `factor_residuals` | `Matrix{T}` | Factor VAR residuals |
| `Sigma_eta` | `Matrix{T}` | ``r \times r`` factor innovation covariance |
| `Sigma_e` | `Matrix{T}` | ``N \times N`` idiosyncratic covariance |
| `eigenvalues` | `Vector{T}` | Eigenvalues from initial PCA |
| `explained_variance` | `Vector{T}` | Variance explained by each factor |
| `cumulative_variance` | `Vector{T}` | Cumulative variance explained |
| `r` | `Int` | Number of factors |
| `p` | `Int` | Number of factor VAR lags |
| `method` | `Symbol` | Estimation method (`:twostep` or `:em`) |
| `standardized` | `Bool` | Whether data was standardized |
| `converged` | `Bool` | Convergence status (relevant for `:em`) |
| `iterations` | `Int` | Number of iterations (relevant for `:em`) |
| `loglik` | `T` | Log-likelihood value |

### Model Selection for DFM

Select the number of factors ``r`` and lag order ``p`` using information criteria:

```julia
# Grid search over (r, p) combinations
ic = ic_criteria_dynamic(X, max_r, max_p;
    method = :twostep,
    standardize = true
)

println("AIC selects: r=$(ic.r_AIC), p=$(ic.p_AIC)")
println("BIC selects: r=$(ic.r_BIC), p=$(ic.p_BIC)")

# View full IC matrices
ic.AIC  # r×p matrix of AIC values
ic.BIC  # r×p matrix of BIC values
```

### Forecasting with DFM

```julia
# Point forecasts h steps ahead
fc = forecast(model, h)
fc.factors      # h×r factor forecasts
fc.observables  # h×N observable forecasts

# Forecasts with confidence intervals
fc = forecast(model, h;
    ci_level = 0.90,
    n_sim = 1000
)
fc.factors_lower, fc.factors_upper      # Factor CIs
fc.observables_lower, fc.observables_upper  # Observable CIs
```

### Stationarity Check

```julia
# Check if factor dynamics are stationary
is_stationary(model)  # true if max|eigenvalue| < 1

# Get companion matrix for factor VAR
C = companion_matrix_factors(model)
eigvals(C)  # Eigenvalues determine stability
```

**Reference**: Stock & Watson (2002a), Doz, Giannone & Reichlin (2011)

---

## Generalized Dynamic Factor Model (GDFM)

### Theoretical Foundation

The Generalized Dynamic Factor Model of Forni, Hallin, Lippi & Reichlin (2000, 2005) provides a fully dynamic approach to factor analysis using spectral methods. Unlike the standard DFM which uses static PCA followed by VAR, the GDFM extracts factors directly in the frequency domain.

### Model Specification

The GDFM decomposes each observable as:

```math
x_{it} = \chi_{it} + \xi_{it}
```

where:
- ``\chi_{it}`` is the **common component** driven by ``q`` common shocks
- ``\xi_{it}`` is the **idiosyncratic component**

The common component has the representation:

```math
\chi_{it} = b_{i1}(L) u_{1t} + b_{i2}(L) u_{2t} + \cdots + b_{iq}(L) u_{qt}
```

where ``b_{ij}(L)`` are square-summable filters and ``u_{jt}`` are orthonormal white noise shocks.

### Spectral Representation

In the frequency domain, the spectral density of ``X_t`` decomposes as:

```math
\Sigma_X(\omega) = \Sigma_\chi(\omega) + \Sigma_\xi(\omega)
```

The key insight is that common factors produce **diverging eigenvalues** (growing with ``N``) while idiosyncratic components produce **bounded eigenvalues**.

### Estimation Algorithm

1. **Spectral Density Estimation**: Estimate ``\hat{\Sigma}_X(\omega)`` using kernel smoothing of the periodogram
2. **Dynamic Eigenanalysis**: Compute eigenvalue decomposition at each frequency
3. **Factor Extraction**: Select top ``q`` eigenvectors (dynamic principal components)
4. **Common Component**: Reconstruct ``\chi_t`` via inverse Fourier transform

### Julia Implementation

```julia
using MacroEconometricModels

# Estimate GDFM with q dynamic factors
model = estimate_gdfm(X, q;
    standardize = true,
    bandwidth = 0,           # Auto-select: T^(1/3)
    kernel = :bartlett,      # :bartlett, :parzen, or :tukey
    r = 0                    # Static factors (0 = same as q)
)

# Access results
F = model.factors                 # T×q time-domain factors
χ = model.common_component        # T×N common component
ξ = model.idiosyncratic           # T×N idiosyncratic component
Λ = model.loadings_spectral       # N×q×n_freq frequency-domain loadings

# Variance explained by dynamic factors
model.variance_explained          # q-vector of variance shares
```

### GeneralizedDynamicFactorModel Return Values

| Field | Type | Description |
|-------|------|-------------|
| `X` | `Matrix{T}` | Original ``T \times N`` data matrix |
| `factors` | `Matrix{T}` | ``T \times q`` time-domain factors |
| `common_component` | `Matrix{T}` | ``T \times N`` common component ``\chi_t`` |
| `idiosyncratic` | `Matrix{T}` | ``T \times N`` idiosyncratic component ``\xi_t`` |
| `loadings_spectral` | `Array{Complex{T},3}` | ``N \times q \times n_{freq}`` frequency-domain loadings |
| `spectral_density_X` | `Array{Complex{T},3}` | Spectral density of ``X_t`` |
| `spectral_density_chi` | `Array{Complex{T},3}` | Spectral density of common component |
| `eigenvalues_spectral` | `Matrix{T}` | ``N \times n_{freq}`` eigenvalues across frequencies |
| `frequencies` | `Vector{T}` | Frequency grid (0 to ``\pi``) |
| `q` | `Int` | Number of dynamic factors |
| `r` | `Int` | Number of static factors |
| `bandwidth` | `Int` | Kernel smoothing bandwidth |
| `kernel` | `Symbol` | Kernel type (`:bartlett`, `:parzen`, `:tukey`) |
| `standardized` | `Bool` | Whether data was standardized |
| `variance_explained` | `Vector{T}` | Variance share per dynamic factor |

### Selecting the Number of Dynamic Factors

The GDFM uses eigenvalue-based criteria rather than information criteria:

```julia
# Compute selection criteria
ic = ic_criteria_gdfm(X, max_q;
    standardize = true,
    bandwidth = 0,
    kernel = :bartlett
)

# Eigenvalue ratio criterion (Ahn & Horenstein 2013)
println("Ratio criterion selects: q=$(ic.q_ratio)")

# Variance threshold criterion (90% of spectral variance)
println("Variance criterion selects: q=$(ic.q_variance)")

# Diagnostic data
ic.eigenvalue_ratios      # λ_i / λ_{i+1} ratios
ic.cumulative_variance    # Cumulative variance explained
ic.avg_eigenvalues        # Average eigenvalues across frequencies
```

### Spectral Diagnostics

```julia
# Get data for eigenvalue plots across frequencies
plot_data = spectral_eigenvalue_plot_data(model)
plot_data.frequencies     # Vector of frequencies (0 to π)
plot_data.eigenvalues     # N×n_freq matrix of eigenvalues

# First eigenvalue should dominate if one strong factor
# Gap between q-th and (q+1)-th eigenvalue indicates factor count
```

### Common Variance Share

```julia
# Fraction of variance explained by common component for each variable
shares = common_variance_share(model)

# Variables well-explained by common factors
well_explained = findall(shares .> 0.5)

# Summary statistics
println("Mean common variance share: ", round(mean(shares), digits=3))
println("Variables with >50% common: ", length(well_explained))
```

### Forecasting with GDFM

```julia
# Forecast h steps ahead
fc = forecast(model, h; method=:ar)

fc.common   # h×N common component forecast
fc.factors  # h×q factor forecast
```

### Comparison: DFM vs GDFM

| Aspect | Dynamic Factor Model | Generalized DFM |
|--------|---------------------|-----------------|
| **Approach** | Time domain (PCA + VAR) | Frequency domain (spectral) |
| **Factor dynamics** | Explicit VAR structure | Implicit through spectral density |
| **Estimation** | Two-step or EM | Kernel-smoothed periodogram |
| **Computational cost** | Moderate | Higher (FFT at each frequency) |
| **Asymptotics** | ``T \to \infty`` | ``N, T \to \infty`` jointly |
| **Best for** | Moderate N, focus on forecasting | Large N, structural decomposition |

### Example: Complete GDFM Workflow

```julia
using MacroEconometricModels

# Load large macroeconomic panel (e.g., FRED-MD)
X = load_data()  # T×N matrix

# Step 1: Select number of factors
ic = ic_criteria_gdfm(X, 10)
q = ic.q_ratio
println("Selected q = $q dynamic factors")

# Step 2: Estimate GDFM
model = estimate_gdfm(X, q; kernel=:parzen)

# Step 3: Diagnostics
println("Variance explained: ", round.(model.variance_explained, digits=3))
println("Mean R²: ", round(mean(r2(model)), digits=3))

# Step 4: Extract common component for further analysis
χ = model.common_component  # Use in FAVAR, forecasting, etc.

# Step 5: Identify variables driven by common vs idiosyncratic shocks
shares = common_variance_share(model)
common_driven = findall(shares .> 0.7)
idio_driven = findall(shares .< 0.3)
```

**References**:
- Forni, M., Hallin, M., Lippi, M., & Reichlin, L. (2000). "The Generalized Dynamic-Factor Model: Identification and Estimation."
- Forni, M., Hallin, M., Lippi, M., & Reichlin, L. (2005). "The Generalized Dynamic Factor Model: One-Sided Estimation and Forecasting."
- Hallin, M., & Liška, R. (2007). "Determining the Number of Factors in the General Dynamic Factor Model."

---

## References

### Core References

- Bai, J. (2003). "Inferential Theory for Factor Models of Large Dimensions." *Econometrica*, 71(1), 135-171.
- Bai, J., & Ng, S. (2002). "Determining the Number of Factors in Approximate Factor Models." *Econometrica*, 70(1), 191-221.
- Bai, J., & Ng, S. (2006). "Confidence Intervals for Diffusion Index Forecasts and Inference for Factor-Augmented Regressions." *Econometrica*, 74(4), 1133-1150.
- Stock, J. H., & Watson, M. W. (2002a). "Forecasting Using Principal Components from a Large Number of Predictors." *Journal of the American Statistical Association*, 97(460), 1167-1179.
- Stock, J. H., & Watson, M. W. (2002b). "Macroeconomic Forecasting Using Diffusion Indexes." *Journal of Business & Economic Statistics*, 20(2), 147-162.

### Dynamic Factor Models

- Doz, C., Giannone, D., & Reichlin, L. (2011). "A Two-Step Estimator for Large Approximate Dynamic Factor Models Based on Kalman Filtering." *Journal of Econometrics*, 164(1), 188-205.
- Doz, C., Giannone, D., & Reichlin, L. (2012). "A Quasi-Maximum Likelihood Approach for Large, Approximate Dynamic Factor Models." *Review of Economics and Statistics*, 94(4), 1014-1024.
- Forni, M., Hallin, M., Lippi, M., & Reichlin, L. (2000). "The Generalized Dynamic-Factor Model: Identification and Estimation." *Review of Economics and Statistics*, 82(4), 540-554.
- Forni, M., Hallin, M., Lippi, M., & Reichlin, L. (2005). "The Generalized Dynamic Factor Model: One-Sided Estimation and Forecasting." *Journal of the American Statistical Association*, 100(471), 830-840.
- Hallin, M., & Liška, R. (2007). "Determining the Number of Factors in the General Dynamic Factor Model." *Journal of the American Statistical Association*, 102(478), 603-617.

### Applications

- Bernanke, B. S., Boivin, J., & Eliasz, P. (2005). "Measuring the Effects of Monetary Policy: A Factor-Augmented Vector Autoregressive (FAVAR) Approach." *Quarterly Journal of Economics*, 120(1), 387-422.
- McCracken, M. W., & Ng, S. (2016). "FRED-MD: A Monthly Database for Macroeconomic Research." *Journal of Business & Economic Statistics*, 34(4), 574-589.
