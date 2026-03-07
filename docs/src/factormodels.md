# [Factor Models](@id factor_page)

**MacroEconometricModels.jl** provides a complete toolkit for estimating, diagnosing, and forecasting with factor models in large macroeconomic panels. The package covers static principal components, dynamic factor models with explicit VAR dynamics, generalized dynamic factor models via spectral methods, and structural identification of common shocks.

- **Static Factor Model**: Principal components estimation (Stock & Watson 2002a) with automatic panel orientation (short vs tall), standardization, and block-restricted EM estimation
- **Information Criteria**: Bai & Ng (2002) IC1--IC3 for selecting the number of static factors, plus AIC/BIC for dynamic factor model specification
- **Dynamic Factor Model**: Two-step (PCA + VAR) or EM (Kalman smoother) estimation with four confidence interval methods for forecasting (Doz, Giannone & Reichlin 2011, 2012)
- **Generalized Dynamic Factor Model**: Spectral estimation via kernel-smoothed periodogram with frequency-domain eigenanalysis (Forni, Hallin, Lippi & Reichlin 2000, 2005)
- **Structural DFM**: SVAR identification (Cholesky or sign restrictions) on common factors with panel-wide structural IRFs (Forni, Giannone, Lippi & Reichlin 2009)
- **Block-Restricted Estimation**: EM algorithm with masked loadings for theory-guided factor structures
- **Forecasting**: Factor-augmented forecasting with theoretical, bootstrap, and simulation-based confidence intervals

All results integrate with `report()` for publication-quality output and `plot_result()` for interactive D3.js visualization.

```@setup factor
using MacroEconometricModels, Random
Random.seed!(42)
fred = load_example(:fred_md)
X = to_matrix(apply_tcode(fred))
X = X[all.(isfinite, eachrow(X)), :]
X = X[end-59:end, :]
```

## Quick Start

**Recipe 1: Static factor model from FRED-MD**

```@example factor
# Estimate 3-factor model via PCA
fm = estimate_factors(X, 3; standardize=true)
report(fm)
```

**Recipe 2: Select the number of factors**

```@example factor
# Bai-Ng information criteria for factor count selection
ic = ic_criteria(X, 10)
println("IC1 selects r = ", ic.r_IC1, ", IC2 selects r = ", ic.r_IC2, ", IC3 selects r = ", ic.r_IC3)
```

**Recipe 3: Dynamic factor model with VAR dynamics**

```@example factor
# 3 factors with VAR(1) dynamics, two-step estimation
dfm = estimate_dynamic_factors(X, 3, 1; method=:twostep, standardize=true)
report(dfm)
```

**Recipe 4: Generalized dynamic factor model (spectral)**

```@example factor
# 2 dynamic factors via spectral analysis
gdfm = estimate_gdfm(X, 2; kernel=:bartlett)
report(gdfm)
```

**Recipe 5: Forecast with bootstrap confidence intervals**

```@example factor
# DFM forecast with bootstrap CIs
fc = forecast(dfm, 12; ci_method=:bootstrap, n_boot=50)
report(fc)
```

**Recipe 6: Structural DFM with Cholesky identification**

```@example factor
# Identify structural shocks in common factors
sdfm = estimate_structural_dfm(X, 2; identification=:cholesky, p=1, H=20)
r = irf(sdfm, 20)
report(r)
```

```julia
plot_result(r)
```

---

## The Static Factor Model

The static factor model decomposes an ``N``-dimensional panel of observables into common and idiosyncratic components. It is the workhorse of modern empirical macroeconomics, enabling dimensionality reduction from hundreds of indicators to a handful of latent factors that summarize aggregate economic conditions.

```math
X = F \Lambda' + E
```

where:
- ``X`` is the ``T \times N`` data matrix of observables
- ``F`` is the ``T \times r`` matrix of latent common factors
- ``\Lambda`` is the ``N \times r`` matrix of factor loadings
- ``E`` is the ``T \times N`` matrix of idiosyncratic errors
- ``r`` is the number of factors (with ``r \ll \min(T, N)``)

The factors and loadings are estimated by minimizing the sum of squared idiosyncratic errors:

```math
\min_{F, \Lambda} \sum_{i=1}^N \sum_{t=1}^T (x_{it} - \lambda_i' F_t)^2
```

where:
- ``\lambda_i`` is the ``r \times 1`` loading vector for variable ``i``
- ``F_t`` is the ``r \times 1`` factor vector at time ``t``

subject to the normalization ``F'F/T = I_r``. The solution involves the eigenvalue decomposition of ``X'X`` (when ``N \leq T``) or ``XX'`` (when ``T < N``), with the first ``r`` eigenvectors forming the estimated factors or loadings.

!!! note "Technical Note"
    The factors and loadings are identified only up to an ``r \times r`` invertible rotation: if ``(\hat{F}, \hat{\Lambda})`` is a solution, then ``(\hat{F}H, \hat{\Lambda}H^{-1'})`` is equally valid for any invertible ``H``. The normalization ``F'F/T = I_r`` pins down orientation but not sign. Individual factor loadings should not be interpreted as structural parameters. To compare estimated factors with "true" factors (e.g., in simulations), compute absolute correlations rather than raw correlations.

```@example factor
# Estimate 3-factor model from FRED-MD indicators
fm = estimate_factors(X, 3; standardize=true)
report(fm)
```

```julia
plot_result(fm)
```

```@raw html
<iframe src="../assets/plots/model_factor_static.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The first three factors capture the dominant sources of common variation across the FRED-MD panel. The scree plot shows a clear decline in eigenvalue magnitude after the first few factors, consistent with a low-dimensional factor structure. The cumulative variance explained indicates how much of the total panel variation is attributable to common factors versus idiosyncratic noise.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `standardize` | `Bool` | `true` | Standardize data to zero mean, unit variance before estimation |
| `method` | `Symbol` | `:pca` | Estimation method (`:pca` for principal components) |
| `blocks` | `Dict` | `nothing` | Block structure for restricted estimation (see [Block-Restricted Estimation](@ref block_restricted)) |

| Field | Type | Description |
|-------|------|-------------|
| `X` | `Matrix{T}` | Original ``T \times N`` data matrix |
| `factors` | `Matrix{T}` | ``T \times r`` estimated factor matrix |
| `loadings` | `Matrix{T}` | ``N \times r`` estimated loading matrix |
| `eigenvalues` | `Vector{T}` | Eigenvalues from PCA (descending order) |
| `explained_variance` | `Vector{T}` | Fraction of variance explained by each factor |
| `cumulative_variance` | `Vector{T}` | Cumulative fraction of variance explained |
| `r` | `Int` | Number of factors |
| `standardized` | `Bool` | Whether data was standardized before estimation |

---

## Determining the Number of Factors

Choosing ``r`` is the central model selection problem in factor analysis. Too few factors omit common variation and bias downstream estimates; too many factors overfit, including noise as signal. Bai & Ng (2002) propose three information criteria that trade off goodness-of-fit against model complexity, with penalty terms designed for the double-indexed (``N, T``) asymptotic framework of factor models.

```math
IC_k(r) = \log \hat{\sigma}^2(r) + r \cdot g_k(N, T)
```

where:
- ``\hat{\sigma}^2(r) = \frac{1}{NT} \sum_{i,t} \hat{e}_{it}^2`` is the average squared residual with ``r`` factors
- ``g_1(N, T) = \frac{N + T}{NT} \log\left(\frac{NT}{N+T}\right)`` is the IC1 penalty
- ``g_2(N, T) = \frac{N + T}{NT} \log(C_{NT}^2)`` is the IC2 penalty
- ``g_3(N, T) = \frac{\log(C_{NT}^2)}{C_{NT}^2}`` is the IC3 penalty
- ``C_{NT}^2 = \min(N, T)``

The optimal ``\hat{r}`` minimizes ``IC_k(r)`` over ``r \in \{1, \ldots, r_{\max}\}``. All three criteria are consistent: ``\hat{r} \xrightarrow{p} r_0`` as ``N, T \to \infty``. IC2 and IC3 perform best in Monte Carlo simulations.

```@example factor
# Bai-Ng information criteria
ic = ic_criteria(X, 10)
println("IC1 selects r = ", ic.r_IC1, ", IC2 selects r = ", ic.r_IC2, ", IC3 selects r = ", ic.r_IC3)
```

The three criteria typically agree on the number of factors for well-separated factor structures. When they disagree, IC2 and IC3 are preferred. The scree plot provides a complementary visual diagnostic: a sharp drop in eigenvalue magnitude after factor ``r`` confirms the information criteria selection.

| Field | Type | Description |
|-------|------|-------------|
| `r_IC1` | `Int` | Number of factors selected by IC1 |
| `r_IC2` | `Int` | Number of factors selected by IC2 |
| `r_IC3` | `Int` | Number of factors selected by IC3 |
| `IC1` | `Vector{T}` | IC1 values for ``r = 1, \ldots, r_{\max}`` |
| `IC2` | `Vector{T}` | IC2 values for ``r = 1, \ldots, r_{\max}`` |
| `IC3` | `Vector{T}` | IC3 values for ``r = 1, \ldots, r_{\max}`` |

---

## Model Diagnostics

The ``R^2`` for each variable measures how much of its variation is explained by the common factors, providing a variable-level diagnostic for the factor model fit.

```math
R^2_i = 1 - \frac{\sum_t \hat{e}_{it}^2}{\sum_t (x_{it} - \bar{x}_i)^2}
```

where:
- ``\hat{e}_{it} = x_{it} - \hat{\lambda}_i' \hat{F}_t`` is the idiosyncratic residual for variable ``i``
- ``\bar{x}_i`` is the sample mean of variable ``i``

Variables with high ``R^2`` are strongly driven by common factors; variables with low ``R^2`` are dominated by idiosyncratic shocks and contribute little to the common factor structure.

```@example factor
fm = estimate_factors(X, 3; standardize=true)
report(fm)

# Per-variable R-squared
r2_vals = r2(fm)

# Fitted values and residuals via StatsAPI
X_hat = predict(fm)       # T x N fitted values
resid = residuals(fm)     # T x N residuals
```

The mean ``R^2`` across all variables summarizes the overall explanatory power of the factor model. A mean ``R^2`` above 0.5 indicates that common factors capture more than half of total panel variation, consistent with a strong factor structure.

### StatsAPI Interface

All factor model types implement the standard StatsAPI interface:

| Function | `FactorModel` | `DynamicFactorModel` | `GeneralizedDynamicFactorModel` |
|----------|:---:|:---:|:---:|
| `predict(m)` | Fitted values ``\hat{X} = F\Lambda'`` | Fitted values ``\hat{X} = F\Lambda'`` | Common component ``\hat{\chi}_t`` |
| `residuals(m)` | Idiosyncratic residuals | Idiosyncratic residuals | Idiosyncratic component |
| `r2(m)` | Per-variable ``R^2`` | Per-variable ``R^2`` | Per-variable ``R^2`` |
| `nobs(m)` | Number of observations | Number of observations | Number of observations |
| `dof(m)` | Degrees of freedom | Degrees of freedom | Degrees of freedom |
| `loglikelihood(m)` | --- | Log-likelihood | --- |
| `aic(m)` | --- | AIC | --- |
| `bic(m)` | --- | BIC | --- |

!!! note "Technical Note"
    `loglikelihood`, `aic`, and `bic` are available only for `DynamicFactorModel` since static PCA and spectral GDFM estimation do not produce a well-defined likelihood.

---

## Dynamic Factor Models

The dynamic factor model (DFM) extends the static model by specifying explicit VAR dynamics for the latent factors. This state-space formulation enables likelihood-based estimation, Kalman filtering, and principled multi-step forecasting with proper uncertainty quantification.

**Observation equation**:

```math
X_t = \Lambda F_t + e_t
```

**State equation**:

```math
F_t = A_1 F_{t-1} + A_2 F_{t-2} + \cdots + A_p F_{t-p} + \eta_t
```

where:
- ``X_t`` is the ``N \times 1`` vector of observables at time ``t``
- ``F_t`` is the ``r \times 1`` vector of latent factors
- ``\Lambda`` is the ``N \times r`` loading matrix
- ``A_1, \ldots, A_p`` are ``r \times r`` autoregressive coefficient matrices
- ``\eta_t \sim N(0, \Sigma_\eta)`` are factor innovations
- ``e_t \sim N(0, \Sigma_e)`` are idiosyncratic errors (typically diagonal)

Two estimation methods are available. **Two-step estimation** extracts factors via PCA and fits a VAR on the extracted factors (Stock & Watson 2002a). **EM estimation** iterates between the Kalman smoother (E-step) and parameter updates (M-step), producing more efficient estimates at higher computational cost (Doz, Giannone & Reichlin 2012).

```@example factor
# 3 factors with VAR(1) dynamics
dfm = estimate_dynamic_factors(X, 3, 1;
    method=:twostep,
    standardize=true,
    diagonal_idio=true    # Diagonal idiosyncratic covariance
)
report(dfm)
```

The two-step estimator is fast and consistent under the Bai & Ng (2002) conditions. The EM estimator is preferred when the Gaussian state-space structure is a reasonable approximation, as it exploits the full likelihood and produces efficient estimates even with moderate ``N``.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:twostep` | Estimation method (`:twostep` or `:em`) |
| `standardize` | `Bool` | `true` | Standardize data before estimation |
| `diagonal_idio` | `Bool` | `true` | Restrict ``\Sigma_e`` to be diagonal |

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
| `method` | `Symbol` | Estimation method used |
| `standardized` | `Bool` | Whether data was standardized |
| `converged` | `Bool` | Convergence status (for `:em`) |
| `iterations` | `Int` | Number of iterations (for `:em`) |
| `loglik` | `T` | Log-likelihood value |

### Model Selection for DFM

The joint selection of factor count ``r`` and lag order ``p`` uses standard information criteria computed from the state-space log-likelihood:

```julia
# Grid search over (r, p) combinations
ic_dyn = ic_criteria_dynamic(X, 5, 3; method=:twostep, standardize=true)

# View full IC matrices
ic_dyn.AIC   # r x p matrix of AIC values
ic_dyn.BIC   # r x p matrix of BIC values
```

### Stationarity Check

```@example factor
# Verify factor dynamics are stationary
is_stationary(dfm)   # true if max|eigenvalue| < 1
```

---

## Forecasting

Factor model forecasts extrapolate factor dynamics forward and project to observables via the loading matrix. For static factor models, a VAR is fitted internally on the extracted factors. Four confidence interval methods are available.

```math
\hat{F}_{T+h|T} = \hat{A}_1 \hat{F}_{T+h-1|T} + \cdots + \hat{A}_p \hat{F}_{T+h-p|T}
```

where:
- ``\hat{F}_{T+h|T}`` is the ``h``-step-ahead factor forecast conditional on information at time ``T``
- ``\hat{A}_1, \ldots, \hat{A}_p`` are the estimated factor VAR coefficient matrices

Observable forecasts are obtained via the loading matrix:

```math
\hat{X}_{T+h|T} = \hat{\Lambda} \hat{F}_{T+h|T}
```

where:
- ``\hat{X}_{T+h|T}`` is the ``N \times 1`` vector of observable forecasts
- ``\hat{\Lambda}`` is the ``N \times r`` estimated loading matrix

**Theoretical CIs** compute the ``h``-step forecast error covariance analytically via the VMA(``\infty``) representation:

```math
\text{MSE}_h = \sum_{j=0}^{h-1} \Psi_j \, \Sigma_\eta \, \Psi_j'
```

where:
- ``\Psi_j = J C^j`` are the VMA coefficient matrices from the companion form
- ``C`` is the companion matrix of the factor VAR
- ``J`` is the selector for the first ``r`` rows
- ``\Sigma_\eta`` is the factor innovation covariance

| `ci_method` | Description | Best for |
|-------------|-------------|----------|
| `:none` | Point forecast only | Quick exploration |
| `:theoretical` | Analytical VMA CIs (Gaussian) | Large samples, fast |
| `:bootstrap` | Residual resampling | Non-Gaussian innovations |
| `:simulation` | Monte Carlo draws from estimated model | Full uncertainty propagation |

```@example factor
# DFM forecast with bootstrap CIs
dfm2 = estimate_dynamic_factors(X, 2, 1)
fc = forecast(dfm2, 10; ci_method=:bootstrap, n_boot=50, conf_level=0.95)
report(fc)
```

```julia
plot_result(fc)
```

```@raw html
<iframe src="../assets/plots/forecast_factor.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The theoretical standard errors increase with the forecast horizon, reflecting growing uncertainty about future factor values. For stationary factor dynamics, the standard errors converge to the unconditional forecast error standard deviation. Bootstrap CIs are preferred when the Gaussian assumption is doubtful.

!!! note "Technical Note"
    Observable forecast standard errors combine factor uncertainty with idiosyncratic variance: ``\text{Var}(\hat{X}_{T+h}) = \Lambda \cdot \text{MSE}_h \cdot \Lambda' + \Sigma_e``. The GDFM forecast uses AR(1) extrapolation of each factor series with closed-form variance: ``\text{Var}(\hat{F}_{T+h,i}) = \sigma_i^2 \sum_{j=0}^{h-1} \phi_i^{2j}``.

| Field | Type | Description |
|-------|------|-------------|
| `factors` | `Matrix{T}` | ``h \times r`` factor point forecasts |
| `observables` | `Matrix{T}` | ``h \times N`` observable point forecasts |
| `factors_lower` | `Matrix{T}` | ``h \times r`` lower CI bounds for factors |
| `factors_upper` | `Matrix{T}` | ``h \times r`` upper CI bounds for factors |
| `observables_lower` | `Matrix{T}` | ``h \times N`` lower CI bounds for observables |
| `observables_upper` | `Matrix{T}` | ``h \times N`` upper CI bounds for observables |
| `factors_se` | `Matrix{T}` | ``h \times r`` factor forecast standard errors |
| `observables_se` | `Matrix{T}` | ``h \times N`` observable forecast standard errors |
| `horizon` | `Int` | Forecast horizon ``h`` |
| `conf_level` | `T` | Confidence level (e.g., 0.95) |
| `ci_method` | `Symbol` | CI method used |

---

## Generalized Dynamic Factor Model

The Generalized Dynamic Factor Model (GDFM) of Forni, Hallin, Lippi & Reichlin (2000, 2005) provides a fully dynamic approach to factor analysis using spectral methods. Unlike the standard DFM that uses static PCA followed by a VAR, the GDFM extracts factors directly in the frequency domain, exploiting the spectral density structure of the panel.

The GDFM decomposes each observable into common and idiosyncratic components:

```math
x_{it} = \chi_{it} + \xi_{it}
```

where:
- ``\chi_{it}`` is the **common component** driven by ``q`` common shocks
- ``\xi_{it}`` is the **idiosyncratic component**

The common component has the dynamic representation:

```math
\chi_{it} = b_{i1}(L) u_{1t} + b_{i2}(L) u_{2t} + \cdots + b_{iq}(L) u_{qt}
```

where:
- ``b_{ij}(L)`` are square-summable lag polynomial filters
- ``u_{jt}`` are orthonormal white noise common shocks
- ``q`` is the number of dynamic factors

In the frequency domain, the spectral density of ``X_t`` decomposes as ``\Sigma_X(\omega) = \Sigma_\chi(\omega) + \Sigma_\xi(\omega)``. The key insight is that common factors produce **diverging eigenvalues** (growing with ``N``) while idiosyncratic components produce **bounded eigenvalues**. This spectral separation identifies the factor space without imposing a finite VAR structure on factor dynamics.

!!! note "Technical Note"
    The estimation algorithm proceeds in four steps: (1) estimate ``\hat{\Sigma}_X(\omega)`` using kernel smoothing of the periodogram, (2) compute eigenvalue decomposition at each frequency, (3) select top ``q`` eigenvectors (dynamic principal components), (4) reconstruct the common component ``\chi_t`` via inverse Fourier transform.

```@example factor
# 2 dynamic factors via spectral analysis
gdfm = estimate_gdfm(X, 2;
    standardize=true,
    bandwidth=0,          # Auto-select: T^(1/3)
    kernel=:bartlett      # :bartlett, :parzen, or :tukey
)
report(gdfm)

# Common vs idiosyncratic decomposition
chi = gdfm.common_component      # T x N common component
xi = gdfm.idiosyncratic          # T x N idiosyncratic component
```

The common variance share for each variable measures the fraction of its total variation attributable to the ``q`` common shocks. Variables with high common variance shares are strongly connected to aggregate fluctuations; those with low shares are driven primarily by sector-specific or idiosyncratic disturbances.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `standardize` | `Bool` | `true` | Standardize data before estimation |
| `bandwidth` | `Int` | `0` | Kernel bandwidth (0 for auto: ``T^{1/3}``) |
| `kernel` | `Symbol` | `:bartlett` | Spectral kernel (`:bartlett`, `:parzen`, `:tukey`) |
| `r` | `Int` | `0` | Number of static factors (0 = same as ``q``) |

| Field | Type | Description |
|-------|------|-------------|
| `X` | `Matrix{T}` | Original ``T \times N`` data matrix |
| `factors` | `Matrix{T}` | ``T \times q`` time-domain factors |
| `common_component` | `Matrix{T}` | ``T \times N`` common component ``\chi_t`` |
| `idiosyncratic` | `Matrix{T}` | ``T \times N`` idiosyncratic component ``\xi_t`` |
| `loadings_spectral` | `Array{Complex{T},3}` | ``N \times q \times n_{freq}`` frequency-domain loadings |
| `spectral_density_X` | `Array{Complex{T},3}` | Spectral density of ``X_t`` |
| `eigenvalues_spectral` | `Matrix{T}` | ``N \times n_{freq}`` eigenvalues across frequencies |
| `frequencies` | `Vector{T}` | Frequency grid (0 to ``\pi``) |
| `q` | `Int` | Number of dynamic factors |
| `bandwidth` | `Int` | Kernel smoothing bandwidth |
| `kernel` | `Symbol` | Kernel type |
| `variance_explained` | `Vector{T}` | Variance share per dynamic factor |

### Selecting the Number of Dynamic Factors

The GDFM uses eigenvalue-based criteria rather than information criteria:

```julia
# Eigenvalue ratio and variance criteria
ic_gdfm = ic_criteria_gdfm(X, 10; kernel=:bartlett)

# Diagnostic data
ic_gdfm.eigenvalue_ratios       # lambda_i / lambda_{i+1} ratios
ic_gdfm.cumulative_variance     # Cumulative variance explained
ic_gdfm.avg_eigenvalues         # Average eigenvalues across frequencies
```

### DFM vs GDFM

| Aspect | Dynamic Factor Model | Generalized DFM |
|--------|---------------------|-----------------|
| **Approach** | Time domain (PCA + VAR) | Frequency domain (spectral) |
| **Factor dynamics** | Explicit VAR structure | Implicit through spectral density |
| **Estimation** | Two-step or EM | Kernel-smoothed periodogram |
| **Computational cost** | Moderate | Higher (FFT at each frequency) |
| **Asymptotics** | ``T \to \infty`` | ``N, T \to \infty`` jointly |
| **Best for** | Moderate ``N``, forecasting focus | Large ``N``, structural decomposition |

---

## [Block-Restricted Estimation](@id block_restricted)

When economic theory suggests that different groups of variables load on distinct factors --- for example, "real activity" versus "nominal" versus "financial" factors --- block-restricted estimation enforces zero loadings outside each block. The EM algorithm with masked updates estimates the model while respecting the prescribed factor structure.

```math
x_{it} = \lambda_i' F_t + e_{it}, \quad \lambda_{ij} = 0 \text{ if variable } i \notin \text{block } j
```

where:
- ``\lambda_{ij}`` is the loading of variable ``i`` on factor ``j``
- The zero restriction ``\lambda_{ij} = 0`` is imposed when variable ``i`` does not belong to block ``j``

!!! note "Technical Note"
    The EM algorithm initializes each block factor via block-wise PCA, then iterates: (1) E-step: compute expected factors given current loadings, (2) M-step: update only non-zero loadings (masked update). Convergence is on log-likelihood. Validation requires: ``r`` blocks, no overlapping indices, at least 2 variables per block.

```@example factor
X_block = randn(200, 15)

# Define 3 blocks: variables 1-5, 6-10, 11-15
blocks = Dict(
    :real => [1, 2, 3, 4, 5],
    :nominal => [6, 7, 8, 9, 10],
    :financial => [11, 12, 13, 14, 15]
)

# Block-restricted estimation via EM
fm_block = estimate_factors(X_block, 3; blocks=blocks)
report(fm_block)
```

The block-restricted model produces loadings that are exactly zero outside each block, enabling economically interpretable factor labeling. The unrestricted PCA estimate is more flexible but cannot distinguish between "real" and "nominal" sources of common variation without additional identification.

---

## Structural Dynamic Factor Model

The Structural DFM (Forni, Giannone, Lippi & Reichlin 2009) identifies structural shocks in large panels by applying SVAR identification to the common factors. It combines the GDFM with a VAR on the time-domain factors and maps identified factor-level shocks to all ``N`` panel variables via the loading matrix.

```math
F_t = c + \sum_{l=1}^p A_l F_{t-l} + B_0 \varepsilon_t
```

where:
- ``F_t`` is the ``q \times 1`` vector of common factors from the GDFM
- ``A_l`` are ``q \times q`` autoregressive coefficient matrices
- ``B_0`` is the ``q \times q`` impact matrix
- ``\varepsilon_t`` are structural shocks

Panel-wide structural IRFs map factor responses to all ``N`` observables:

```math
\text{IRF}_i(h, j) = \sum_{k=1}^q \Lambda_{ik} \cdot \Phi_h \, B_0 \, e_j
```

where:
- ``\Lambda_{ik}`` is the time-domain loading of variable ``i`` on factor ``k``
- ``\Phi_h`` is the ``h``-step reduced-form IRF matrix of the factor VAR
- ``e_j`` is the ``j``-th structural shock selector

Two identification schemes are available: **Cholesky decomposition** and **sign restrictions**.

```@example factor
X_sdfm = randn(200, 20)

# Cholesky identification
sdfm = estimate_structural_dfm(X_sdfm, 2; identification=:cholesky, p=1, H=20)
r_sdfm = irf(sdfm, 20)
report(r_sdfm)

# FEVD of the factor VAR
d = fevd(sdfm, 20)
report(d)
```

```julia
plot_result(r_sdfm)
```

The structural IRFs show how a one-standard-deviation structural shock propagates to each of the ``N`` panel variables over the ``H``-period horizon. Cholesky identification imposes a recursive ordering on the factors; sign restrictions allow the researcher to test alternative identification schemes based on economic theory.

### Sign Restrictions

```@example factor
# Define sign restriction function
sign_fn(irf_matrix) = irf_matrix[1, 1] > 0 && irf_matrix[1, 2] < 0

sdfm_sign = estimate_structural_dfm(X_sdfm, 2;
    identification=:sign, sign_check=sign_fn, max_draws=1000, H=20)
r_sign = irf(sdfm_sign, 20)
report(r_sign)
```

### Two-Step Estimation

A pre-estimated GDFM can be passed directly:

```@example factor
gdfm_pre = estimate_gdfm(X_sdfm, 2)
sdfm_two = estimate_structural_dfm(gdfm_pre; identification=:cholesky, p=1, H=20)
report(sdfm_two)
```

!!! note "Technical Note"
    The Structural DFM proceeds in two steps: (1) estimate GDFM to extract common factors and spectral loadings, (2) fit a VAR on the time-domain factors and apply structural identification. Time-domain loadings are computed via OLS regression ``\hat{\Lambda} = (F'F)^{-1}F'X`` rather than the spectral domain.

---

## Asymptotic Theory

Under the conditions of Bai & Ng (2002) and Bai (2003), the principal components estimator of the factors is consistent and asymptotically normal as both ``N`` and ``T`` grow. The convergence rate depends on ``\min(N, T)``, reflecting the dual large-sample requirement of factor models.

```math
\frac{1}{T} \sum_{t=1}^T \|\hat{F}_t - H F_t\|^2 = O_p\left( \frac{1}{\min(N, T)} \right)
```

where:
- ``\hat{F}_t`` is the estimated factor vector at time ``t``
- ``F_t`` is the true factor vector
- ``H`` is an ``r \times r`` rotation matrix (reflecting the identification-up-to-rotation property)

For large ``N`` and ``T``, the factor estimates are asymptotically normal:

```math
\sqrt{T} (\hat{F}_t - H F_t) \xrightarrow{d} N(0, V)
```

where:
- ``V`` depends on the cross-sectional and temporal dependence structure of the idiosyncratic errors

The consistency result holds under weak cross-sectional and temporal dependence in the idiosyncratic errors, making PCA robust to moderate departures from independence. The ``\sqrt{\min(N, T)}`` convergence rate implies that both dimensions must grow for the factor estimates to be precise --- a panel with large ``T`` but small ``N`` does not benefit from factor extraction.

---

## Applications

### Diffusion Index Forecasting

Estimated factors serve as regressors for forecasting a target variable ``y_{t+h}``:

```math
y_{t+h} = \alpha + \beta' \hat{F}_t + \gamma' y_{t:t-p} + \varepsilon_{t+h}
```

where:
- ``\alpha`` is the intercept
- ``\beta`` is the ``r \times 1`` vector of factor coefficients
- ``\hat{F}_t`` is the ``r \times 1`` vector of estimated factors at time ``t``
- ``\gamma`` is the vector of autoregressive coefficients on own lags of ``y``
- ``\varepsilon_{t+h}`` is the forecast error

Factors summarize information from a large panel into a small number of predictors, improving forecast accuracy relative to simple autoregressive models (Stock & Watson 2002b).

### Factor-Augmented VAR

Factors combine with key observable variables in a VAR system for structural analysis with high-dimensional information sets. See the [FAVAR](favar.md) page for full coverage of the `estimate_favar` function, Bayesian Gibbs sampling, and panel-wide impulse response mapping.

---

## Complete Example

This example demonstrates the full factor model workflow: data preparation, factor selection, estimation of static and dynamic models, forecasting, and visualization.

```@example factor
# Step 1: Select number of factors via Bai-Ng criteria
ic_full = ic_criteria(X, 10)
println("IC1 selects r = ", ic_full.r_IC1, ", IC2 selects r = ", ic_full.r_IC2, ", IC3 selects r = ", ic_full.r_IC3)

# Step 2: Estimate static factor model
fm_full = estimate_factors(X, 3; standardize=true)
report(fm_full)

# Step 3: Estimate dynamic factor model with VAR(1) dynamics
dfm_full = estimate_dynamic_factors(X, 3, 1; method=:twostep, standardize=true)
report(dfm_full)

# Step 4: Diagnostics — per-variable R-squared
r2_static = r2(fm_full)
r2_dynamic = r2(dfm_full)

# Step 5: Forecast with theoretical CIs
fc_full = forecast(dfm_full, 12; ci_method=:theoretical, conf_level=0.95)
report(fc_full)
```

```julia
plot_result(fc_full)
```

The Bai-Ng information criteria select the number of factors from the FRED-MD panel. The static factor model extracts the dominant principal components, and the dynamic factor model augments the static estimate with VAR(1) dynamics on the factors. The per-variable ``R^2`` values identify which FRED-MD indicators are well-explained by the common factors and which are driven by idiosyncratic variation. The 12-step-ahead forecast with theoretical confidence intervals shows factor and observable predictions with uncertainty bands that widen at longer horizons, reflecting the accumulation of forecast error variance through the factor VAR dynamics.

---

## Common Pitfalls

1. **Choosing the number of factors without information criteria.** The scree plot provides a useful visual diagnostic, but formal selection requires the Bai & Ng (2002) information criteria (IC1--IC3). Relying solely on the "elbow" in the scree plot is subjective and can lead to over- or under-extraction. Use `ic_criteria(X, r_max)` and check whether IC1, IC2, and IC3 agree.

2. **Interpreting factor loadings as structural parameters.** Factor loadings are identified only up to an ``r \times r`` rotation matrix. The signs and magnitudes of individual loadings are not invariant to the normalization convention. Use block-restricted estimation or rotate factors post-estimation when structural labeling is required.

3. **Forgetting to standardize before estimation.** If the panel variables have heterogeneous scales (e.g., interest rates in percent vs industrial production index levels), high-variance series dominate the principal components. Always set `standardize=true` (the default) unless all variables are already on comparable scales.

4. **Misspecifying block structure.** When using block-restricted estimation, each block must contain at least 2 variables, blocks must not overlap, and the number of blocks must equal the number of factors ``r``. Violating any of these conditions raises a validation error.

5. **Applying GDFM to short panels.** The GDFM spectral estimator requires ``N, T \to \infty`` jointly. For small ``N`` (fewer than 20 variables), the spectral density estimate is unreliable and the standard DFM with two-step or EM estimation is preferred.

6. **Ignoring stationarity of factor dynamics.** The DFM forecast assumes stationary factor VAR dynamics. If the estimated companion matrix has eigenvalues at or above unity, forecasts diverge. Check `is_stationary(dfm)` before forecasting.

---

## References

- Bai, J. (2003). Inferential Theory for Factor Models of Large Dimensions.
  *Econometrica*, 71(1), 135-171. [DOI](https://doi.org/10.1111/1468-0262.00392)

- Bai, J., & Ng, S. (2002). Determining the Number of Factors in Approximate Factor Models.
  *Econometrica*, 70(1), 191-221. [DOI](https://doi.org/10.1111/1468-0262.00273)

- Bai, J., & Ng, S. (2006). Confidence Intervals for Diffusion Index Forecasts and Inference for Factor-Augmented Regressions.
  *Econometrica*, 74(4), 1133-1150. [DOI](https://doi.org/10.1111/j.1468-0262.2006.00696.x)

- Doz, C., Giannone, D., & Reichlin, L. (2011). A Two-Step Estimator for Large Approximate Dynamic Factor Models Based on Kalman Filtering.
  *Journal of Econometrics*, 164(1), 188-205. [DOI](https://doi.org/10.1016/j.jeconom.2011.02.012)

- Doz, C., Giannone, D., & Reichlin, L. (2012). A Quasi-Maximum Likelihood Approach for Large, Approximate Dynamic Factor Models.
  *Review of Economics and Statistics*, 94(4), 1014-1024. [DOI](https://doi.org/10.1162/REST_a_00225)

- Forni, M., Hallin, M., Lippi, M., & Reichlin, L. (2000). The Generalized Dynamic-Factor Model: Identification and Estimation.
  *Review of Economics and Statistics*, 82(4), 540-554. [DOI](https://doi.org/10.1162/003465300559037)

- Forni, M., Hallin, M., Lippi, M., & Reichlin, L. (2005). The Generalized Dynamic Factor Model: One-Sided Estimation and Forecasting.
  *Journal of the American Statistical Association*, 100(471), 830-840. [DOI](https://doi.org/10.1198/016214504000002050)

- Forni, M., Giannone, D., Lippi, M., & Reichlin, L. (2009). Opening the Black Box: Structural Factor Models with Large Cross-Sections.
  *Econometric Theory*, 25(5), 1319-1347. [DOI](https://doi.org/10.1017/S0266466609090422)

- Hallin, M., & Liska, R. (2007). Determining the Number of Factors in the General Dynamic Factor Model.
  *Journal of the American Statistical Association*, 102(478), 603-617. [DOI](https://doi.org/10.1198/016214506000001275)

- McCracken, M. W., & Ng, S. (2016). FRED-MD: A Monthly Database for Macroeconomic Research.
  *Journal of Business & Economic Statistics*, 34(4), 574-589. [DOI](https://doi.org/10.1080/07350015.2015.1086655)

- Stock, J. H., & Watson, M. W. (2002a). Forecasting Using Principal Components from a Large Number of Predictors.
  *Journal of the American Statistical Association*, 97(460), 1167-1179. [DOI](https://doi.org/10.1198/016214502388618960)

- Stock, J. H., & Watson, M. W. (2002b). Macroeconomic Forecasting Using Diffusion Indexes.
  *Journal of Business & Economic Statistics*, 20(2), 147-162. [DOI](https://doi.org/10.1198/073500102317351921)
