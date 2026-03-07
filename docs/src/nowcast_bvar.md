# [BVAR Nowcasting](@id nowcast_bvar_page)

The large Bayesian VAR (BVAR) approach to nowcasting estimates a high-dimensional VAR directly on mixed-frequency data, using informative Normal-Inverse-Wishart priors to regularize the parameter space. The implementation follows Giannone, Lenza & Primiceri (2015) with the mixed-frequency extensions of Cimadomo et al. (2022). Hyperparameters governing prior tightness are optimized automatically via marginal log-likelihood maximization.

For an overview of all nowcasting methods and method comparison, see [Nowcasting](@ref). For DFM-based nowcasting, see [DFM Nowcasting](@ref nowcast_dfm_page).

```@setup nc_bvar
using MacroEconometricModels, Random
Random.seed!(42)
```

## Quick Start

**Recipe 1: Basic BVAR nowcast**

```@example nc_bvar
fred = load_example(:fred_md)
nc_md = fred[:, ["INDPRO", "UNRATE", "CPIAUCSL", "M2SL", "FEDFUNDS"]]
Y = to_matrix(apply_tcode(nc_md))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-99:end, :]
nM, nQ = 4, 1
for t in 1:size(Y, 1)
    if mod(t, 3) != 0
        Y[t, end] = NaN
    end
end
Y[end, end] = NaN

# BVAR with 5 lags and data-driven hyperparameters
bvar = nowcast_bvar(Y, nM, nQ; lags=5)
report(bvar)
```

**Recipe 2: BVAR with custom hyperparameters**

```@example nc_bvar
# Tighter shrinkage and stronger unit root prior
bvar = nowcast_bvar(Y, nM, nQ; lags=5, lambda0=0.1, theta0=2.0, miu0=0.5, alpha0=3.0)
report(bvar)
```

**Recipe 3: BVAR forecast**

```@example nc_bvar
N = nM + nQ

# 6-step ahead forecast for all variables
bvar = nowcast_bvar(Y, nM, nQ; lags=5)
fc = forecast(bvar, 6; target_var=N)
```

**Recipe 4: BVAR with TimeSeriesData**

```@example nc_bvar
# TimeSeriesData dispatch works identically to raw matrices
ts = TimeSeriesData(Y; varnames=["INDPRO","UNRATE","CPI","M2","FEDFUNDS"], frequency=Monthly)
bvar = nowcast_bvar(ts, nM, nQ; lags=5)
report(bvar)
```

---

## Model Specification

The BVAR models all ``N`` variables jointly as a vector autoregression with ``p`` lags:

```math
y_t = c + B_1 y_{t-1} + \cdots + B_p y_{t-p} + u_t, \quad u_t \sim N(0, \Sigma)
```

where:
- ``y_t`` is the ``N \times 1`` vector of observed variables at time ``t``
- ``c`` is the ``N \times 1`` intercept vector
- ``B_1, \ldots, B_p`` are ``N \times N`` coefficient matrices
- ``\Sigma`` is the ``N \times N`` error covariance matrix

### Prior Structure

The Normal-Inverse-Wishart prior implements four types of shrinkage via dummy observations (Giannone, Lenza & Primiceri 2015):

1. **Overall tightness** (``\lambda``): Controls the decay of prior precision with lag order. The prior variance for the coefficient on lag ``l`` of variable ``j`` in equation ``i`` scales as ``\sigma_i / (\lambda \cdot l^2)``, so higher lags receive stronger shrinkage toward zero.

2. **Cross-variable shrinkage** (``\theta``): Shrinks cross-variable coefficients relative to own-lag coefficients. The off-diagonal dummy observations scale as ``\sigma_i / (\theta \cdot \lambda \cdot l^2)``, so larger ``\theta`` pulls coefficients on other variables' lags more strongly toward zero.

3. **Sum-of-coefficients** (``\mu``): Implements the unit root prior by constraining the sum of lag coefficients on each variable's own lags to equal one. Controlled by the tightness parameter ``\mu``, where smaller values impose the prior more tightly.

4. **Co-persistence** (``\alpha``): Implements the common stochastic trend prior, shrinking toward a model where all variables share a single unit root. Controlled by the tightness parameter ``\alpha``.

!!! note "Technical Note"
    The prior is implemented via dummy observations appended to the data matrix before OLS estimation. This stacking approach yields the posterior mode of the Normal-Inverse-Wishart conjugate prior analytically, avoiding MCMC sampling. The AR(1) residual standard deviations ``\sigma_i`` are pre-estimated from the data to scale the prior appropriately for each variable.

---

## Estimation

The estimation proceeds in two stages. First, Nelder-Mead optimization maximizes the marginal log-likelihood over the hyperparameter vector ``(\lambda, \theta, \mu, \alpha)`` in log-space. Second, the BVAR is estimated at the optimal hyperparameters via OLS on the augmented data-plus-dummy system, and the Kalman smoother fills missing values in the ragged edge.

```@example nc_bvar
bvar = nowcast_bvar(Y, nM, nQ; lags=5, max_iter=200, thresh=1e-6)
println("Optimized hyperparameters:")
println("  lambda = ", round(bvar.lambda, digits=4))
println("  theta  = ", round(bvar.theta, digits=4))
println("  miu    = ", round(bvar.miu, digits=4))
println("  alpha  = ", round(bvar.alpha, digits=4))
println("Marginal log-likelihood: ", round(bvar.loglik, digits=2))
```

The optimized hyperparameters reveal how much the data favors shrinkage. A small ``\lambda`` indicates strong overall regularization is optimal. A large ``\theta`` means cross-variable dynamics are weak relative to own-lag persistence. The marginal log-likelihood provides a model comparison criterion --- higher values indicate better out-of-sample prediction potential.

---

## Forecasting

The `forecast` function iterates the estimated VAR equation forward from the last smoothed observation:

```math
\hat{y}_{T+h} = \hat{c} + \hat{B}_1 \hat{y}_{T+h-1} + \cdots + \hat{B}_p \hat{y}_{T+h-p}
```

For ``h > 1``, the forecast uses previously generated forecasts as inputs (iterated multi-step forecast).

```@example nc_bvar
bvar = nowcast_bvar(Y, nM, nQ; lags=5)

# Forecast all variables 6 steps ahead
fc_all = forecast(bvar, 6)

# Forecast specific target variable only
fc_target = forecast(bvar, 6; target_var=N)
```

The BVAR forecast leverages the full cross-variable dynamics estimated from the data. Each forecast step uses the posterior mode coefficients ``\hat{B}_1, \ldots, \hat{B}_p`` and chains predictions forward. Missing values in the ragged edge are filled by the Kalman smoother before forecasting begins, ensuring a complete initial condition.

---

## Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `lags` | `Int` | `5` | Number of VAR lags |
| `thresh` | `Real` | ``10^{-6}`` | Nelder-Mead convergence threshold |
| `max_iter` | `Int` | `200` | Maximum optimization iterations |
| `lambda0` | `Real` | `0.2` | Initial overall shrinkage |
| `theta0` | `Real` | `1.0` | Initial cross-variable shrinkage |
| `miu0` | `Real` | `1.0` | Initial sum-of-coefficients weight |
| `alpha0` | `Real` | `2.0` | Initial co-persistence weight |

---

## NowcastBVAR Return Values

| Field | Type | Description |
|-------|------|-------------|
| `X_sm` | `Matrix{T}` | Smoothed data with all NaN values filled |
| `beta` | `Matrix{T}` | Posterior mode VAR coefficients (``(1 + Np) \times N``) |
| `sigma` | `Matrix{T}` | Posterior mode error covariance (``N \times N``) |
| `lambda` | `T` | Optimized overall shrinkage |
| `theta` | `T` | Optimized cross-variable shrinkage |
| `miu` | `T` | Optimized sum-of-coefficients weight |
| `alpha` | `T` | Optimized co-persistence weight |
| `lags` | `Int` | Number of VAR lags |
| `loglik` | `T` | Marginal log-likelihood at optimum |

---

## Complete Example

```@example nc_bvar
# === Step 1: Estimate BVAR ===
bvar = nowcast_bvar(Y, nM, nQ; lags=5, max_iter=200)
report(bvar)

# === Step 2: Extract nowcast and forecast ===
result = nowcast(bvar)
println("Nowcast: ", round(result.nowcast, digits=3))
println("Forecast: ", round(result.forecast, digits=3))

# === Step 3: Multi-step forecast ===
fc = forecast(bvar, 6; target_var=N)

# === Step 4: Compare with DFM ===
dfm = nowcast_dfm(Y, nM, nQ; r=2, p=1)
r_dfm = nowcast(dfm)
r_bvar = nowcast(bvar)
println("DFM nowcast:  ", round(r_dfm.nowcast, digits=3))
println("BVAR nowcast: ", round(r_bvar.nowcast, digits=3))
```

**Interpretation.** The BVAR estimates all cross-variable dynamics directly from the 5-variable mixed-frequency panel, with the Normal-Inverse-Wishart prior preventing overfitting in the ``5 \times (1 + 5 \times 5) = 130``-parameter system. The Nelder-Mead optimizer finds the hyperparameter combination that maximizes marginal likelihood --- a criterion that inherently penalizes complexity. The Kalman smoother fills the quarterly NaN pattern and the ragged edge, producing a complete smoothed panel. Comparing the BVAR nowcast with the DFM nowcast provides a robustness check: agreement across methods strengthens confidence in the current-quarter estimate.

---

## Common Pitfalls

1. **Insufficient complete observations.** The BVAR requires at least `lags + 2` rows without any NaN values for estimation. If the complete portion is too short, the algorithm falls back to column-mean imputation, which degrades prior calibration.

2. **Hyperparameter initial values.** The Nelder-Mead optimizer starts from `(lambda0, theta0, miu0, alpha0)`. Poor starting values can lead to local optima. If results appear unreasonable, try different initial values or increase `max_iter`.

3. **Too many lags for small samples.** Each additional lag adds ``N`` parameters per equation. With `lags=5` and ``N = 10``, each equation has 51 coefficients. The prior regularizes estimation, but very large lag orders relative to the sample size can still cause numerical instability.

4. **Ragged edge depth.** The BVAR fills missing values at the end of the sample using iterated VAR forecasts from the last complete row. Deep ragged edges (many consecutive missing periods) accumulate forecast error and reduce nowcast accuracy.

---

## References

- Giannone, Domenico, Michele Lenza, and Giorgio E. Primiceri. 2015. "Prior Selection for Vector Autoregressions." *Review of Economics and Statistics* 97 (2): 436--451. [DOI: 10.1162/REST_a_00483](https://doi.org/10.1162/REST_a_00483)
- Cimadomo, Jacopo, Domenico Giannone, Michele Lenza, Francesca Monti, and Andrej Sokol. 2022. "Nowcasting with Large Bayesian Vector Autoregressions." *ECB Working Paper* No. 2696.
