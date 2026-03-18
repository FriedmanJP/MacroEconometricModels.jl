# [Factor-Augmented VAR](@id favar_page)

**MacroEconometricModels.jl** provides a complete Factor-Augmented VAR (FAVAR) framework for structural analysis of large macroeconomic panels. The FAVAR combines latent factors extracted from hundreds of variables with a small-scale VAR, enabling impulse response analysis, forecast error variance decomposition, and forecasting at the panel level.

- **Two-Step Estimation**: Extract ``r`` factors via principal components, remove double-counting against key observed variables, and estimate a VAR on the augmented system (Bernanke, Boivin & Eliasz 2005)
- **Bayesian FAVAR**: Joint Gibbs sampler drawing factors, loadings, and VAR parameters with posterior credible intervals via Carter-Kohn smoothing and Normal-Inverse-Wishart conjugate updates
- **Panel-Wide IRFs**: Map structural shocks from the factor VAR to all ``N`` panel variables through the loading matrix ``\Lambda``, producing impulse responses for every series in the dataset
- **Structural Analysis**: Full integration with Cholesky, sign-restriction, and narrative identification through automatic delegation to the VAR infrastructure
- **Forecasting**: Multi-step-ahead forecasts in the augmented space with panel-wide mapping via `favar_panel_forecast`

All results integrate with `report()` for publication-quality output and `plot_result()` for interactive D3.js visualization.

```@setup favar
using MacroEconometricModels, Random
Random.seed!(42)
fred = load_example(:fred_md)
slow_names = ["INDPRO", "CPIAUCSL", "UNRATE"]
fast_names = ["M2SL", "FEDFUNDS", "TB3MS", "GS10"]
md = fred[:, vcat(slow_names, fast_names)]
X = to_matrix(apply_tcode(md))
X = X[all.(isfinite, eachrow(X)), :]
X = X[end-59:end, :]
Y_slow = X[:, 1:3]
Y_fast = X[:, 4:end]
```

## Quick Start

**Recipe 1: Two-step FAVAR estimation**

```@example favar
# FAVAR: 2 factors, 2 lags, columns 4-7 as key variables
favar = estimate_favar(X, [4, 5, 6, 7], 2, 2)
report(irf(favar, 20; method=:cholesky))
```

**Recipe 2: Bayesian FAVAR with Gibbs sampling**

```@example favar
bfavar = estimate_favar(X, [4, 5], 2, 2;
    method=:bayesian, n_draws=100, burnin=50)
report(irf(bfavar, 20))
```

**Recipe 3: Panel-wide impulse responses**

```@example favar
favar3 = estimate_favar(X, [4, 5], 2, 2)
r3 = irf(favar3, 20; method=:cholesky)
r_panel = favar_panel_irf(favar3, r3)
report(r_panel)
```

**Recipe 4: Panel-wide forecasting**

```@example favar
fc = forecast(favar3, 12)
fc_panel = favar_panel_forecast(favar3, fc)
report(fc_panel)
```

**Recipe 5: FEVD and historical decomposition**

```@example favar
d = fevd(favar3, 20)
h = historical_decomposition(favar3)
report(d)
```

---

## Model Specification

The FAVAR augments a standard VAR with latent factors extracted from a large panel of ``N`` macroeconomic variables. Traditional VARs are limited to a handful of variables due to parameter proliferation. The FAVAR overcomes this curse of dimensionality by summarizing the information content of a large panel into ``r`` factors, then estimating a VAR on the compact augmented system ``[F_t, Y_t^{key}]``.

The **transition equation** governs the dynamics of the augmented system:

```math
\begin{bmatrix} F_t \\ Y_t^{key} \end{bmatrix} = c + \sum_{l=1}^{p} A_l \begin{bmatrix} F_{t-l} \\ Y_{t-l}^{key} \end{bmatrix} + u_t
```

where:
- ``F_t`` is the ``r \times 1`` vector of latent factors extracted from the panel
- ``Y_t^{key}`` is the ``n_{key} \times 1`` vector of key observed variables (e.g., the federal funds rate)
- ``A_l`` are ``(r + n_{key}) \times (r + n_{key})`` coefficient matrices at lag ``l``
- ``c`` is the ``(r + n_{key}) \times 1`` intercept vector
- ``u_t \sim N(0, \Sigma)`` is the ``(r + n_{key}) \times 1`` reduced-form error vector

The **observation equation** links the factors to all ``N`` panel variables:

```math
X_t = \Lambda \, F_t + e_t
```

where:
- ``X_t`` is the ``N \times 1`` vector of panel variables at time ``t``
- ``\Lambda`` is the ``N \times r`` loading matrix mapping factors to observables
- ``e_t`` is the ``N \times 1`` vector of idiosyncratic errors

!!! note "Technical Note"
    The FAVAR nests a standard VAR as a special case when ``r = 0``. With ``r > 0``,
    the model exploits information from the full panel while keeping the VAR dimension
    manageable at ``r + n_{key}``. The number of estimated VAR parameters scales as
    ``(r + n_{key})^2 p`` rather than ``N^2 p``.

---

## Two-Step Estimation

The two-step procedure of Bernanke, Boivin & Eliasz (2005) provides a computationally efficient estimator. The key insight is that PCA consistently estimates the factor space even when the factors are correlated with the key observed variables, but double-counting must be removed before forming the VAR.

The algorithm proceeds in three stages:

1. **Extract factors**: Estimate ``r`` factors from the standardized panel ``X`` via principal components (Stock & Watson 2002)
2. **Remove double-counting**: Regress each extracted factor on ``Y^{key}`` via OLS and retain the residuals as **slow-moving factors** ``\tilde{F}``
3. **Estimate VAR**: Fit VAR(``p``) on the augmented system ``[\tilde{F}, Y^{key}]``

!!! note "Technical Note"
    The double-counting correction is essential: without it, ``Y^{key}`` information
    enters twice --- once through the factors (which load on all variables including
    ``Y^{key}``) and once directly. Bernanke, Boivin & Eliasz (2005) resolve this by
    regressing ``F`` on ``Y^{key}`` and using the residual component as the factors
    in the VAR.

```@example favar
# Two-step FAVAR: 2 factors, 2 lags, key variables at columns 4 and 5
favar_ts = estimate_favar(X, [4, 5], 2, 2)
report(irf(favar_ts, 20; method=:cholesky))
```

The estimation extracts factors from the panel, orthogonalizes them against the 2 key variables, and fits a VAR(2) on the resulting 5-variable system. The `report()` call on the IRF displays the structural impulse responses with Cholesky identification, where the ordering places factors before key variables.

### Specifying Key Variables

Key variables enter the FAVAR directly and receive exact (not factor-mapped) impulse responses. They can be specified as column indices or as a matrix:

```@example favar
# By column indices (preferred)
favar_idx = estimate_favar(X, [4, 5, 6], 2, 2)

# With custom variable names for the panel
favar_named = estimate_favar(X, [4, 5], 2, 2;
    panel_varnames=["Slow1", "Slow2", "Slow3", "Fast1", "Fast2", "Fast3", "Fast4"])
```

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:two_step` | Estimation method: `:two_step` or `:bayesian` |
| `panel_varnames` | `Vector{String}` | `nothing` | Display names for the ``N`` panel variables |
| `n_draws` | `Int` | `5000` | Number of posterior draws (Bayesian only) |
| `burnin` | `Int` | `1000` | Burn-in draws discarded before collection (Bayesian only) |

### Return Value (`FAVARModel{T}`)

| Field | Type | Description |
|-------|------|-------------|
| `Y` | `Matrix{T}` | Augmented data ``[\tilde{F}, Y^{key}]`` (``T_{eff} \times (r + n_{key})``) |
| `p` | `Int` | VAR lag order |
| `B` | `Matrix{T}` | VAR coefficient matrix (``(1 + p(r + n_{key})) \times (r + n_{key})``) |
| `U` | `Matrix{T}` | VAR residuals |
| `Sigma` | `Matrix{T}` | Residual covariance (``(r + n_{key}) \times (r + n_{key})``) |
| `X_panel` | `Matrix{T}` | Original panel data (``T \times N``) |
| `panel_varnames` | `Vector{String}` | Panel variable names (length ``N``) |
| `Y_key_indices` | `Vector{Int}` | Column indices of key variables in `X_panel` |
| `n_factors` | `Int` | Number of latent factors ``r`` |
| `n_key` | `Int` | Number of key observed variables ``n_{key}`` |
| `factors` | `Matrix{T}` | Extracted slow-moving factors (``T \times r``) |
| `loadings` | `Matrix{T}` | Factor loading matrix ``\Lambda`` (``N \times r``) |
| `factor_model` | `FactorModel{T}` | Underlying PCA factor model with variance explained |
| `aic` | `T` | Akaike information criterion |
| `bic` | `T` | Bayesian information criterion |
| `loglik` | `T` | Log-likelihood |

---

## Bayesian FAVAR

The Bayesian approach jointly estimates factors, loadings, and VAR parameters via Gibbs sampling (Bernanke, Boivin & Eliasz 2005, Section IV). Joint estimation accounts for the uncertainty in the extracted factors, which the two-step procedure treats as known. This produces wider and more honest credible intervals on structural analysis.

The Gibbs sampler iterates three blocks:

1. **Draw** ``\Lambda \mid F, X``: Equation-by-equation OLS regression with Normal posterior, drawing each row of ``\Lambda`` conditional on the current factors and idiosyncratic variances
2. **Draw** ``F \mid \Lambda, B, \Sigma, X, Y^{key}``: Posterior regression combining the observation equation likelihood with a standard Normal prior, producing time-``t`` factor draws
3. **Draw** ``(B, \Sigma) \mid F, Y^{key}``: Normal-Inverse-Wishart conjugate posterior from the VAR on the augmented system ``[F, Y^{key}]``

```@example favar
# Bayesian FAVAR with 100 posterior draws and 50 burn-in
bfavar = estimate_favar(X, [4, 5], 2, 2;
    method=:bayesian, n_draws=100, burnin=50)

# Bayesian IRF with posterior credible intervals
birf = irf(bfavar, 20)
report(birf)
```

The Bayesian FAVAR produces posterior draws of the VAR coefficients, covariance matrix, factors, and loadings. The IRF computation propagates parameter uncertainty by computing impulse responses at each draw and reporting posterior median and credible intervals. Wider intervals relative to the two-step approach reflect the additional uncertainty from treating factors as latent.

### Bayesian Structural Analysis

All structural analysis methods --- IRFs, FEVD, and historical decomposition --- operate draw-by-draw through automatic delegation to the `BVARPosterior` infrastructure:

```@example favar
# Bayesian FEVD and historical decomposition
bfevd = fevd(bfavar, 20)
bhd = historical_decomposition(bfavar)
report(bfevd)
```

### Return Value (`BayesianFAVAR{T}`)

| Field | Type | Description |
|-------|------|-------------|
| `B_draws` | `Array{T,3}` | Posterior VAR coefficient draws (``n_{draws} \times k \times n_{var}``) |
| `Sigma_draws` | `Array{T,3}` | Posterior covariance draws (``n_{draws} \times n_{var} \times n_{var}``) |
| `factor_draws` | `Array{T,3}` | Posterior factor draws (``n_{draws} \times T \times r``) |
| `loadings_draws` | `Array{T,3}` | Posterior loading draws (``n_{draws} \times N \times r``) |
| `X_panel` | `Matrix{T}` | Original panel data (``T \times N``) |
| `panel_varnames` | `Vector{String}` | Panel variable names |
| `Y_key_indices` | `Vector{Int}` | Key variable column indices |
| `n_factors` | `Int` | Number of factors ``r`` |
| `n_key` | `Int` | Number of key variables ``n_{key}`` |
| `n` | `Int` | Total VAR dimension (``r + n_{key}``) |
| `p` | `Int` | VAR lag order |
| `data` | `Matrix{T}` | Augmented VAR data ``[F, Y^{key}]`` (posterior mean factors) |
| `varnames` | `Vector{String}` | VAR variable names |

---

## Panel-Wide Impulse Responses

The defining feature of FAVAR is the ability to trace structural shocks through to all ``N`` panel variables. The `favar_panel_irf` function maps factor-space IRFs to the full panel via the loading matrix:

```math
\text{response}_i(h, j) = \sum_{k=1}^{r} \Lambda_{ik} \cdot \text{IRF}_{F_k}(h, j)
```

where:
- ``\text{response}_i(h, j)`` is the response of panel variable ``i`` at horizon ``h`` to structural shock ``j``
- ``\Lambda_{ik}`` is the loading of variable ``i`` on factor ``k``
- ``\text{IRF}_{F_k}(h, j)`` is the impulse response of factor ``k`` at horizon ``h`` to shock ``j``

Key observed variables bypass the factor mapping and use their direct VAR impulse responses, providing exact structural responses for the variables that enter the FAVAR directly.

```@example favar
favar_panel = estimate_favar(X, [4, 5], 2, 2)

# Standard IRF in the augmented space
r_aug = irf(favar_panel, 20; method=:cholesky)

# Map to all panel variables via Lambda
r_panel = favar_panel_irf(favar_panel, r_aug)
report(r_panel)
```

The panel IRF result contains responses for all variables. Key variables use their direct VAR responses, while remaining variables are reconstructed through the loading matrix. The mapping preserves confidence intervals when available.

### Bayesian Panel IRFs

The Bayesian variant uses posterior mean loadings for the panel mapping:

```@example favar
birf_panel = favar_panel_irf(bfavar, birf)
report(birf_panel)
```

---

## Forecasting

FAVAR forecasts operate in the augmented VAR space ``[F, Y^{key}]`` and can be mapped to the full panel via `favar_panel_forecast`. The forecast delegates to the standard VAR forecast infrastructure through `to_var()`.

```@example favar
# VAR-space forecast
fc_favar = forecast(favar_panel, 12)

# Map to all panel variables
fc_panel_full = favar_panel_forecast(favar_panel, fc_favar)
report(fc_panel_full)
```

The panel forecast maps factor forecasts through ``\Lambda`` for non-key variables and uses direct VAR forecasts for key variables. Confidence intervals are mapped through the same loading matrix, preserving the relative width across variables proportional to their factor loadings.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `ci_method` | `Symbol` | `:bootstrap` | Confidence interval method: `:none` or `:bootstrap` |
| `reps` | `Int` | `500` | Number of bootstrap replications |
| `conf_level` | `Real` | `0.95` | Confidence level for intervals |

---

## Structural Identification

The FAVAR inherits all structural identification methods from the VAR infrastructure through automatic `to_var()` delegation. The ordering convention places factors first and key variables last, which determines the Cholesky recursive structure.

```@example favar
# Cholesky identification (factors ordered before key variables)
r_chol = irf(favar_panel, 20; method=:cholesky)
report(r_chol)
```

```julia
# Sign restrictions via check function (irf_matrix is H × n × n)
check_fn(irf_matrix) = irf_matrix[1, 1, 1] > 0 && irf_matrix[1, 3, 1] < 0
r_sign_favar = irf(favar_panel, 20; method=:sign, check_func=check_fn)
```

The Cholesky identification places the slow-moving factors before the fast-moving key variables, consistent with the Bernanke, Boivin & Eliasz (2005) identification scheme where monetary policy (the key variable) responds contemporaneously to factor movements but not vice versa. For a detailed treatment of identification methods, see [Innovation Accounting](@ref innovation_accounting_page).

---

## Complete Example

This example estimates a FAVAR on simulated data, performs structural analysis with both frequentist and Bayesian approaches, and maps results to the full panel:

```@example favar
# --- Two-step FAVAR ---
favar_full = estimate_favar(X, [4, 5], 2, 2)

# Structural analysis in the augmented space
r_aug_full = irf(favar_full, 20; method=:cholesky)
d_aug_full = fevd(favar_full, 20)
hd_full = historical_decomposition(favar_full)

# Panel-wide mapping: IRFs and forecasts for all variables
r_panel_full = favar_panel_irf(favar_full, r_aug_full)
fc_full = forecast(favar_full, 12)
fc_panel_map = favar_panel_forecast(favar_full, fc_full)

# --- Bayesian FAVAR ---
bfavar_full = estimate_favar(X, [4, 5], 2, 2;
    method=:bayesian, n_draws=100, burnin=50)

birf_full = irf(bfavar_full, 20)
birf_panel_full = favar_panel_irf(bfavar_full, birf_full)
bfevd_full = fevd(bfavar_full, 20)

# Display results
report(r_panel_full)
report(bfevd_full)
```

The two-step FAVAR extracts 3 factors from the 50-variable panel, removes the component spanned by the 2 key variables, and estimates a VAR(2) on the resulting 5-variable augmented system. The panel-wide IRFs map structural shocks to all 50 variables through the ``N \times r`` loading matrix ``\Lambda``. The Bayesian FAVAR additionally quantifies uncertainty in the factor extraction through 5000 Gibbs draws, producing wider credible intervals that account for estimation error in both the factors and the VAR parameters.

---

## Common Pitfalls

1. **Slow/fast variable classification**: The ordering ``[F, Y^{key}]`` implies that factors are "slow" and key variables are "fast" under Cholesky identification. Placing a slow-moving variable (e.g., GDP) as a key variable violates this assumption. Use the factor ordering to encode your identification scheme, or switch to sign restrictions.

2. **Number of factors**: Too few factors omit relevant information; too many introduce noise and reduce degrees of freedom in the VAR. Use `ic_criteria(X, r_max)` from the [Factor Models](@ref factor_page) page to select ``r`` via the Bai & Ng (2002) information criteria before estimating the FAVAR.

3. **Gibbs burn-in too short**: The Bayesian FAVAR Gibbs sampler requires adequate burn-in for the factor and loading draws to converge from the PCA initialization. With ``N > 100`` panel variables, set `burnin` to at least 2000. Monitor convergence by comparing results across different `burnin` values.

4. **Panel IRF interpretation**: Panel IRFs for non-key variables are linear projections through ``\Lambda`` and reflect the factor-mediated component only. Idiosyncratic responses (variable-specific shocks) are not captured. A near-zero panel IRF does not mean the variable is unaffected --- it means the variable's response is orthogonal to the common factor structure.

5. **Double-counting correction**: The two-step procedure automatically removes double-counting by orthogonalizing factors against ``Y^{key}``. Manually pre-processing the data to exclude key variables from the panel before factor extraction is unnecessary and discards information that improves factor estimation.

---

## References

- Bernanke, B. S., Boivin, J., & Eliasz, P. (2005). Measuring the Effects of Monetary Policy: A Factor-Augmented Vector Autoregressive (FAVAR) Approach.
  *Quarterly Journal of Economics*, 120(1), 387-422. [DOI](https://doi.org/10.1162/0033553053970344)

- Stock, J. H., & Watson, M. W. (2002). Forecasting Using Principal Components from a Large Number of Predictors.
  *Journal of the American Statistical Association*, 97(460), 1167-1179. [DOI](https://doi.org/10.1198/016214502388618960)

- Bai, J., & Ng, S. (2002). Determining the Number of Factors in Approximate Factor Models.
  *Econometrica*, 70(1), 191-221. [DOI](https://doi.org/10.1111/1468-0262.00273)

- Doan, T., Litterman, R., & Sims, C. (1984). Forecasting and Conditional Projection Using Realistic Prior Distributions.
  *Econometric Reviews*, 3(1), 1-100. [DOI](https://doi.org/10.1080/07474938408800053)

- Carter, C. K., & Kohn, R. (1994). On Gibbs Sampling for State Space Models.
  *Biometrika*, 81(3), 541-553. [DOI](https://doi.org/10.1093/biomet/81.3.541)
