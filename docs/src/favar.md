# Factor-Augmented VAR (FAVAR)

This chapter covers FAVAR models (Bernanke, Boivin & Eliasz 2005), which combine latent factors from large macroeconomic panels with VAR structural analysis. FAVAR enables impulse response analysis, forecast error variance decomposition, and forecasting for hundreds of variables simultaneously.

## Introduction

Traditional VAR models are limited to a small number of variables due to the curse of dimensionality. FAVAR solves this by extracting ``r`` latent factors from a large ``N``-variable panel via principal components, then estimating a VAR on the augmented system ``[F_t; Y_t^{key}]`` where ``Y_t^{key}`` are key observed variables (e.g., the federal funds rate).

The key advantage is **panel-wide impulse responses**: by mapping factor IRFs back through the loading matrix ``\Lambda``, we obtain responses of all ``N`` panel variables to identified structural shocks.

**References**: Bernanke, Boivin & Eliasz (2005), Stock & Watson (2002)

## Quick Start

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Generate panel data (T=200, N=50) with 3 factors
X = randn(200, 50)

# Two-step FAVAR: 3 factors, 2 lags, key variables at columns 1 and 2
favar = estimate_favar(X, [1, 2], 3, 2)

# Structural analysis (inherited from VAR)
r = irf(favar, 20; method=:cholesky)         # Impulse responses
d = fevd(favar, 20)                          # FEVD
h = historical_decomposition(favar)           # Historical decomposition

# Panel-wide IRFs — map to all 50 variables via Λ
r_panel = favar_panel_irf(favar, r)

# Bayesian FAVAR with joint Gibbs sampler
bfavar = estimate_favar(X, [1, 2], 3, 2;
    method=:bayesian, n_draws=5000, burnin=1000)
r_bayes = irf(bfavar, 20)

# Forecasting
fc = forecast(favar, 12)
fc_panel = favar_panel_forecast(favar, fc)
```

---

## Model Specification

The FAVAR model augments a standard VAR with latent factors extracted from a large panel:

```math
\begin{bmatrix} F_t \\ Y_t^{key} \end{bmatrix} = c + \sum_{l=1}^p A_l \begin{bmatrix} F_{t-l} \\ Y_{t-l}^{key} \end{bmatrix} + u_t
```

where:
- ``F_t`` is the ``r \times 1`` vector of latent factors
- ``Y_t^{key}`` is the ``n_{key} \times 1`` vector of key observed variables
- ``A_l`` are ``(r + n_{key}) \times (r + n_{key})`` coefficient matrices
- ``u_t \sim N(0, \Sigma)``

The panel observation equation links the factors to all ``N`` variables:

```math
X_t = \Lambda F_t + e_t
```

where ``\Lambda`` is the ``N \times r`` loading matrix and ``e_t`` are idiosyncratic errors.

!!! note "Technical Note"
    The FAVAR nests a standard VAR as a special case when ``r = 0``. With ``r > 0``,
    the model exploits information from the full panel while keeping the VAR dimension
    manageable at ``r + n_{key}``.

---

## Two-Step Estimation

The two-step procedure of Bernanke, Boivin & Eliasz (2005):

1. **Extract factors**: Estimate ``r`` factors from the panel ``X`` via PCA
2. **Remove double-counting**: Regress extracted factors on ``Y^{key}`` and use residuals as "slow-moving" factors ``\tilde{F}``
3. **Estimate VAR**: Fit VAR(p) on the augmented system ``[\tilde{F}; Y^{key}]``

```julia
using MacroEconometricModels, Random
Random.seed!(42)

X = randn(200, 50)
favar = estimate_favar(X, [1, 2], 3, 2)
println(favar)
```

### Specifying Key Variables

Key variables can be specified as column indices or as a matrix:

```julia
# By column indices (preferred)
favar = estimate_favar(X, [1, 5, 10], 3, 2)

# By matrix (columns must match columns of X)
Y_key = X[:, [1, 5, 10]]
favar = estimate_favar(X, Y_key, 3, 2)

# With custom variable names
favar = estimate_favar(X, [1, 2], 3, 2;
    panel_varnames=["GDP", "CPI", "FFR", fill("x", 47)...])
```

!!! note "Technical Note"
    The double-counting correction is important: without it, ``Y^{key}`` information
    is counted twice---once through the factors (which load on all variables including
    ``Y^{key}``) and once directly. BBE (2005) resolve this by regressing ``F`` on
    ``Y^{key}`` and using the residual component as the factors in the VAR.

---

## Bayesian FAVAR

The Bayesian approach jointly estimates factors, loadings, and VAR parameters via Gibbs sampling:

1. **Initialize factors** via PCA
2. **Gibbs sampler** iterates:
   - Draw ``\Lambda | F, X`` (equation-by-equation regression)
   - Draw ``F | \Lambda, B, \Sigma, X, Y^{key}`` (Carter-Kohn smoother)
   - Draw ``(B, \Sigma) | F, Y^{key}`` (Normal-Inverse-Wishart conjugate)

```julia
using MacroEconometricModels, Random
Random.seed!(42)

X = randn(200, 50)
bfavar = estimate_favar(X, [1, 2], 3, 2;
    method=:bayesian, n_draws=5000, burnin=1000)
println(bfavar)
```

### Bayesian Structural Analysis

Bayesian IRFs, FEVD, and historical decomposition are computed draw-by-draw:

```julia
# Bayesian IRF with credible intervals
r = irf(bfavar, 20)

# Bayesian FEVD
d = fevd(bfavar, 20)

# Historical decomposition
h = historical_decomposition(bfavar)

# Panel-wide Bayesian IRFs
r_panel = favar_panel_irf(bfavar, r)
```

---

## Panel-Wide Impulse Responses

The key feature of FAVAR is mapping structural shocks to all ``N`` panel variables:

```math
\text{response}_i(h, j) = \sum_{k=1}^r \Lambda_{ik} \cdot \text{IRF}_{F_k}(h, j)
```

where ``\text{IRF}_{F_k}(h, j)`` is the response of factor ``k`` at horizon ``h`` to shock ``j``.

Key observed variables use their direct VAR impulse responses rather than the factor mapping.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

X = randn(200, 50)
favar = estimate_favar(X, [1, 2], 3, 2)

# Standard IRF (r + n_key variables)
r = irf(favar, 20; method=:cholesky)

# Map to all 50 panel variables
r_panel = favar_panel_irf(favar, r)
println("Panel IRF variables: ", length(r_panel.variables))
println("Panel IRF shape: ", size(r_panel.values))
```

---

## Forecasting

FAVAR forecasts in the factor-augmented VAR space, which can be mapped to the full panel:

```julia
using MacroEconometricModels, Random
Random.seed!(42)

X = randn(200, 50)
favar = estimate_favar(X, [1, 2], 3, 2)

# VAR-space forecast (r + n_key variables)
fc = forecast(favar, 12)

# Map to all panel variables
fc_panel = favar_panel_forecast(favar, fc)
println("Panel forecast shape: ", size(fc_panel.forecast))
```

---

## Return Values

| Field | Type | Description |
|---|---|---|
| `Y` | `Matrix{T}` | Augmented data ``[F; Y^{key}]`` (``T_{eff} \times n``) |
| `p` | `Int` | VAR lag order |
| `B` | `Matrix{T}` | VAR coefficients |
| `U` | `Matrix{T}` | VAR residuals |
| `Sigma` | `Matrix{T}` | Error covariance |
| `X_panel` | `Matrix{T}` | Original panel (``T \times N``) |
| `panel_varnames` | `Vector{String}` | Panel variable names |
| `n_factors` | `Int` | Number of latent factors ``r`` |
| `n_key` | `Int` | Number of key observed variables |
| `factors` | `Matrix{T}` | Extracted factors (``T \times r``) |
| `loadings` | `Matrix{T}` | Loading matrix ``\Lambda`` (``N \times r``) |

---

## References

- Bernanke, Ben S., Jean Boivin, and Piotr Eliasz. 2005. "Measuring the Effects of Monetary Policy: A Factor-Augmented Vector Autoregressive (FAVAR) Approach." *Quarterly Journal of Economics* 120 (1): 387--422. [https://doi.org/10.1162/0033553053970344](https://doi.org/10.1162/0033553053970344)
- Stock, James H., and Mark W. Watson. 2002. "Forecasting Using Principal Components from a Large Number of Predictors." *Journal of the American Statistical Association* 97 (460): 1167--1179. [https://doi.org/10.1198/016214502388618960](https://doi.org/10.1198/016214502388618960)
- Bai, Jushan, and Serena Ng. 2002. "Determining the Number of Factors in Approximate Factor Models." *Econometrica* 70 (1): 191--221. [https://doi.org/10.1111/1468-0262.00273](https://doi.org/10.1111/1468-0262.00273)
