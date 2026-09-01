# [Factor Models](@id factor_page)

Factor models compress a panel of hundreds of macroeconomic indicators into a handful of latent common factors, turning a cross-section that no VAR can hold into a system small enough to estimate. This page covers the four estimators the package provides — static principal components, dynamic factor models with explicit VAR dynamics, the generalized dynamic factor model estimated in the frequency domain, and structural identification of the common shocks.

- **Static factor model**: principal components (Stock & Watson 2002a) with automatic panel orientation, standardization, and block-restricted EM estimation
- **Information criteria**: Bai & Ng (2002) IC1--IC3 for the number of static factors; Hallin–Liška (2007), Bai–Ng (2007), and Amengual–Watson (2007) for the number of dynamic factors ``q``; AIC/BIC for the dynamic-factor VAR
- **Dynamic factor model**: two-step (PCA + VAR) or EM (Kalman smoother) estimation with four confidence-interval methods for forecasting (Doz, Giannone & Reichlin 2011, 2012)
- **Generalized dynamic factor model**: lag-window spectral estimation with frequency-by-frequency eigenanalysis (Forni, Hallin, Lippi & Reichlin 2000, 2005); the smoothed periodogram is kept as `spectral=:smoothed_periodogram`
- **Structural DFM**: FGLR (2009) with ``r \ge q`` static factors, rank-``q`` shocks, Cholesky or **observable** sign restrictions, and panel-wide structural impulse responses
- **Block-restricted estimation**: EM with masked loadings for theory-guided factor structures

Factors alone summarize a panel; putting them inside a VAR alongside observed policy variables is the [Factor-Augmented VAR](@ref favar_page). For a factor model built to handle ragged real-time data, see [DFM Nowcasting](@ref nowcast_dfm_page).

All results integrate with `report()` for publication-quality output and `plot_result()` for interactive D3.js visualization.

```@setup factor
using MacroEconometricModels, Random, Statistics, LinearAlgebra
Random.seed!(42)
fred = load_example(:fred_md)
X = to_matrix(apply_tcode(fred))
X = X[all.(isfinite, eachrow(X)), :]
X = X[end-59:end, :]
```

## Quick Start

The recipes below run on `X`, the transformed FRED-MD panel (McCracken & Ng 2016) of 126 monthly indicators.

**Recipe 1: Static factor model from FRED-MD**

```@example factor
fm = estimate_factors(X, 3; standardize=true)
report(fm)
```

**Recipe 2: Select the number of factors**

```@example factor
ic = ic_criteria(X, 10)
(IC1=ic.r_IC1, IC2=ic.r_IC2, IC3=ic.r_IC3)
```

**Recipe 3: Dynamic factor model with VAR dynamics**

```@example factor
dfm = estimate_dynamic_factors(X, 3, 1; method=:twostep, standardize=true)
report(dfm)
```

**Recipe 4: Generalized dynamic factor model**

```@example factor
gdfm = estimate_gdfm(X, 2; kernel=:bartlett)
report(gdfm)
```

**Recipe 5: Forecast with bootstrap confidence intervals**

```@example factor
fc = forecast(dfm, 12; ci_method=:bootstrap, n_boot=50)
report(fc)
```

**Recipe 6: Structural DFM with Cholesky identification**

```@example factor
sdfm = estimate_structural_dfm(X, 2; identification=:cholesky, p=1, H=20)
report(sdfm)
```

---

## The Static Factor Model

The static factor model splits an ``N``-dimensional panel into a common component driven by ``r`` factors and an idiosyncratic remainder. It is the workhorse of empirical macroeconomics with large data sets: a few factors reproduce the bulk of the co-movement in hundreds of series, so the modelling problem collapses from ``N`` dimensions to ``r``.

```math
X = F \Lambda' + E
```

where:
- ``X`` is the ``T \times N`` data matrix of observables
- ``F`` is the ``T \times r`` matrix of latent common factors
- ``\Lambda`` is the ``N \times r`` matrix of factor loadings
- ``E`` is the ``T \times N`` matrix of idiosyncratic errors
- ``r`` is the number of factors, with ``r \ll \min(T, N)``

Factors and loadings minimize the sum of squared idiosyncratic errors:

```math
\min_{F, \Lambda} \sum_{i=1}^N \sum_{t=1}^T (x_{it} - \lambda_i' F_t)^2
```

where:
- ``\lambda_i`` is the ``r \times 1`` loading vector for variable ``i``
- ``F_t`` is the ``r \times 1`` factor vector at time ``t``

subject to the normalization ``F'F/T = I_r``. The solution is the eigendecomposition of the sample covariance matrix: the first ``r`` eigenvectors scaled by ``\sqrt{\lambda}`` form the loadings, and the factors are the corresponding scores rescaled to unit variance.

!!! note "Technical Note"
    Factors and loadings are identified only up to an ``r \times r`` invertible rotation: if ``(\hat{F}, \hat{\Lambda})`` solves the problem, so does ``(\hat{F}H, \hat{\Lambda}H^{-1\prime})`` for any invertible ``H``. The normalization ``F'F/T = I_r`` pins down scale but not sign or orientation. Compare estimated factors with true factors through absolute correlations, never raw ones.

```@example factor
fm = estimate_factors(X, 3; standardize=true)
report(fm)
```

```julia
plot_result(fm)
```

```@raw html
<iframe src="../assets/plots/model_factor_static.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

Three factors account for 17.3%, 11.4%, and 9.0% of the total variance of the 126-series panel, 37.8% cumulatively. The spacing between the first three eigenvalues and the fourth (7.0%) is modest, which is typical of monthly growth-rate panels: FRED-MD co-movement is real but spread across several moderately sized factors rather than concentrated in one dominant cycle.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `standardize` | `Bool` | `true` | Standardize to zero mean and unit variance before estimation |
| `blocks` | `Dict{Symbol,Vector{Int}}` | `nothing` | Block structure for restricted EM estimation (see [Block-Restricted Estimation](@ref block_restricted)) |

| Field | Type | Description |
|-------|------|-------------|
| `X` | `Matrix{T}` | Original ``T \times N`` data matrix |
| `factors` | `Matrix{T}` | ``T \times r`` estimated factor matrix |
| `loadings` | `Matrix{T}` | ``N \times r`` estimated loading matrix |
| `eigenvalues` | `Vector{T}` | Eigenvalues of the sample covariance, descending |
| `explained_variance` | `Vector{T}` | Fraction of variance explained by each factor |
| `cumulative_variance` | `Vector{T}` | Cumulative fraction of variance explained |
| `r` | `Int` | Number of factors |
| `standardized` | `Bool` | Whether the data was standardized before estimation |
| `block_names` | `Union{Vector{Symbol},Nothing}` | Block labels when `blocks` was supplied, `nothing` otherwise |

---

## Determining the Number of Factors

Choosing ``r`` is the central model-selection problem in factor analysis. Too few factors leave common variation in the residual and bias every downstream estimate; too many treat noise as signal. Bai & Ng (2002) propose three information criteria whose penalties are built for the double-indexed ``(N, T)`` asymptotics of factor models.

```math
IC_k(r) = \log \hat{\sigma}^2(r) + r \cdot g_k(N, T)
```

where:
- ``\hat{\sigma}^2(r) = \frac{1}{NT} \sum_{i,t} \hat{e}_{it}^2`` is the average squared residual with ``r`` factors
- ``g_1(N, T) = \frac{N + T}{NT} \log\left(\frac{NT}{N+T}\right)`` is the IC1 penalty
- ``g_2(N, T) = \frac{N + T}{NT} \log(C_{NT}^2)`` is the IC2 penalty
- ``g_3(N, T) = \frac{\log(C_{NT}^2)}{C_{NT}^2}`` is the IC3 penalty
- ``C_{NT}^2 = \min(N, T)``

The selected ``\hat{r}`` minimizes ``IC_k(r)`` over ``r \in \{1, \ldots, r_{\max}\}``. All three criteria are consistent as ``N, T \to \infty``; IC2 performs best in the Bai & Ng Monte Carlo designs.

```@example factor
ic = ic_criteria(X, 10)
(IC1=ic.r_IC1, IC2=ic.r_IC2, IC3=ic.r_IC3)
```

IC1 and IC2 agree on four factors. IC3 runs to the boundary of the search grid, ``\hat{r} = 10``, because its penalty ``\log(C_{NT}^2)/C_{NT}^2`` is the weakest of the three and ``C_{NT}^2 = \min(N,T) = 60`` here — with ``T`` this short the penalty barely offsets the fit gain from an extra factor. A criterion that selects ``r_{\max}`` is reporting that the penalty has failed, not that ten factors exist: prefer IC2, and re-run with a larger `max_factors` to confirm the selection is interior.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `standardize` | `Bool` | `true` | Standardize before each trial estimation |

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

The ``R^2`` of each variable measures how much of its variation the common factors explain, giving a series-level diagnostic that the aggregate variance shares cannot.

```math
R^2_i = 1 - \frac{\sum_t \hat{e}_{it}^2}{\sum_t (x_{it} - \bar{x}_i)^2}
```

where:
- ``\hat{e}_{it} = x_{it} - \hat{\lambda}_i' \hat{F}_t`` is the idiosyncratic residual for variable ``i``
- ``\bar{x}_i`` is the sample mean of variable ``i``

Series with high ``R^2`` are driven by aggregate forces and carry information about the factors; series with low ``R^2`` are dominated by idiosyncratic shocks and contribute little to factor extraction.

```@example factor
fm = estimate_factors(X, 3; standardize=true)

r2_vals = r2(fm)                # per-variable R²
X_hat = predict(fm)             # T × N fitted values
resid = residuals(fm)           # T × N idiosyncratic residuals

(mean_r2=round(mean(r2_vals), digits=3),
 share_above_half=round(count(>(0.5), r2_vals) / length(r2_vals), digits=3),
 max_r2=round(maximum(r2_vals), digits=3))
```

Three factors explain 37.8% of the variance of the average FRED-MD series, and 35% of the panel crosses an ``R^2`` of 0.5. The fit is concentrated in the price block: `CUSR0000SA0L2` (0.938), `CPIAUCSL` (0.936), and `CPIULFSL` (0.930) are almost entirely common, while `BUSLOANS` (0.003) and `CPIMEDSL` (0.018) are essentially idiosyncratic at this factor count. A panel-wide mean ``R^2`` near 0.4 is normal for monthly growth rates; levels panels routinely exceed 0.7.

### StatsAPI Interface

All factor model types implement the standard StatsAPI interface:

| Function | `FactorModel` | `DynamicFactorModel` | `GeneralizedDynamicFactorModel` |
|----------|:---:|:---:|:---:|
| `predict(m)` | Fitted values ``\hat{X} = F\Lambda'`` | Fitted values ``\hat{X} = F\Lambda'`` | Common component ``\hat{\chi}_t`` |
| `residuals(m)` | Idiosyncratic residuals | Idiosyncratic residuals | Idiosyncratic component ``\hat{\xi}_t`` |
| `r2(m)` | Per-variable ``R^2`` | Per-variable ``R^2`` | Per-variable ``R^2`` |
| `nobs(m)` | Number of observations | Number of observations | Number of observations |
| `dof(m)` | Degrees of freedom | Degrees of freedom | Degrees of freedom |
| `loglikelihood(m)` | --- | Log-likelihood | --- |
| `aic(m)` | --- | AIC | --- |
| `bic(m)` | --- | BIC | --- |

!!! note "Technical Note"
    `loglikelihood`, `aic`, and `bic` are defined only for `DynamicFactorModel`. Static PCA and spectral GDFM estimation carry no Gaussian likelihood, so no information criterion built on one is available for them. `StructuralDFM` forwards `nobs`, `dof`, and `r2` to its underlying GDFM and factor VAR.

---

## Dynamic Factor Models

The dynamic factor model (DFM) adds explicit VAR dynamics for the latent factors. That turns the model into a linear Gaussian state-space system, which buys likelihood-based estimation, Kalman filtering, and multi-step forecasts with a proper mean-squared-error calculation.

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
- ``e_t \sim N(0, \Sigma_e)`` are idiosyncratic errors, diagonal by default

**Two-step estimation** extracts the factors by PCA and fits a VAR on them (Stock & Watson 2002a). **EM estimation** iterates a Kalman smoother (E-step) against closed-form parameter updates (M-step), starting from the two-step estimate, and delivers the quasi-maximum-likelihood estimator of Doz, Giannone & Reichlin (2012).

```@example factor
dfm = estimate_dynamic_factors(X, 3, 1;
    method=:twostep,
    standardize=true,
    diagonal_idio=true    # diagonal idiosyncratic covariance
)
report(dfm)
```

```@example factor
# EM refinement of the same specification
dfm_em = estimate_dynamic_factors(X, 3, 1; method=:em, max_iter=50)

(converged=dfm_em.converged, iterations=dfm_em.iterations,
 loglik_twostep=round(loglikelihood(dfm), digits=1),
 loglik_em=round(loglikelihood(dfm_em), digits=1))
```

EM converges in 35 sweeps and raises the log-likelihood from ``-8327.0`` to ``-8218.8``, cutting the AIC from 17692 to 17476 at an unchanged parameter count. The gain comes from the smoother, which uses the whole sample to estimate each ``F_t`` instead of the contemporaneous cross-section alone, and which handles the idiosyncratic serial correlation that two-step PCA ignores. Two-step remains the right default for very large ``N``, where the smoother recursions dominate the cost and the two estimators converge anyway.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:twostep` | Estimation method, `:twostep` or `:em` |
| `standardize` | `Bool` | `true` | Standardize data before estimation |
| `max_iter` | `Int` | `100` | Maximum EM iterations (`:em` only) |
| `tol` | `Float64` | ``10^{-6}`` | Relative log-likelihood tolerance (`:em` only) |
| `diagonal_idio` | `Bool` | `true` | Restrict ``\Sigma_e`` to be diagonal |

| Field | Type | Description |
|-------|------|-------------|
| `X` | `Matrix{T}` | Original ``T \times N`` data matrix |
| `factors` | `Matrix{T}` | ``T \times r`` estimated factors |
| `loadings` | `Matrix{T}` | ``N \times r`` loading matrix |
| `A` | `Vector{Matrix{T}}` | ``p`` autoregressive coefficient matrices, each ``r \times r`` |
| `factor_residuals` | `Matrix{T}` | Factor VAR residuals |
| `Sigma_eta` | `Matrix{T}` | ``r \times r`` factor innovation covariance |
| `Sigma_e` | `Matrix{T}` | ``N \times N`` idiosyncratic covariance |
| `eigenvalues` | `Vector{T}` | Eigenvalues from the initial PCA |
| `explained_variance` | `Vector{T}` | Variance explained by each factor |
| `cumulative_variance` | `Vector{T}` | Cumulative variance explained |
| `r` | `Int` | Number of factors |
| `p` | `Int` | Number of factor VAR lags |
| `method` | `Symbol` | Estimation method used |
| `standardized` | `Bool` | Whether the data was standardized |
| `converged` | `Bool` | Convergence flag (`:em`; always `true` for `:twostep`) |
| `iterations` | `Int` | Number of EM iterations (`1` for `:twostep`) |
| `loglik` | `T` | Gaussian log-likelihood |

### Model Selection for DFM

`ic_criteria_dynamic` searches the ``(r, p)`` grid, estimating a DFM at each node and scoring it with AIC and BIC from the state-space log-likelihood:

```@example factor
ic_dyn = ic_criteria_dynamic(X, 5, 3; method=:twostep, standardize=true)

(r_AIC=ic_dyn.r_AIC, p_AIC=ic_dyn.p_AIC, r_BIC=ic_dyn.r_BIC, p_BIC=ic_dyn.p_BIC)
```

Both criteria select ``p = 1`` and both run to the grid edge at ``r = 5``. The lag choice is informative; the factor choice is not, because the DFM likelihood rises with every factor added while `dof` grows only linearly in ``N``, so AIC and BIC keep buying factors. Use `ic_criteria_dynamic` to pick ``p`` given ``r``, and pick ``r`` from the Bai & Ng criteria above.

| Field | Type | Description |
|-------|------|-------------|
| `AIC` | `Matrix{T}` | ``r_{\max} \times p_{\max}`` AIC values, `Inf` where estimation failed |
| `BIC` | `Matrix{T}` | ``r_{\max} \times p_{\max}`` BIC values |
| `r_AIC`, `p_AIC` | `Int` | Grid point minimizing AIC |
| `r_BIC`, `p_BIC` | `Int` | Grid point minimizing BIC |

### Stationarity Check

`companion_matrix_factors` builds the ``rp \times rp`` companion form of the factor VAR, and `is_stationary` checks that its spectral radius is below one:

```@example factor
(stationary=is_stationary(dfm),
 max_modulus=round(maximum(abs.(eigvals(companion_matrix_factors(dfm)))), digits=4))
```

---

## Forecasting

Factor forecasts extrapolate the factor VAR forward and project the result onto the observables through the loading matrix. `forecast` accepts a `FactorModel` (fitting a VAR on the extracted factors internally), a `DynamicFactorModel`, or a `GeneralizedDynamicFactorModel`.

```math
\hat{F}_{T+h|T} = \hat{A}_1 \hat{F}_{T+h-1|T} + \cdots + \hat{A}_p \hat{F}_{T+h-p|T}
```

where:
- ``\hat{F}_{T+h|T}`` is the ``h``-step-ahead factor forecast given information at ``T``
- ``\hat{A}_1, \ldots, \hat{A}_p`` are the estimated factor VAR coefficient matrices

Observable forecasts follow from the loadings:

```math
\hat{X}_{T+h|T} = \hat{\Lambda} \hat{F}_{T+h|T}
```

where:
- ``\hat{X}_{T+h|T}`` is the ``N \times 1`` vector of observable forecasts
- ``\hat{\Lambda}`` is the ``N \times r`` estimated loading matrix

Theoretical intervals compute the ``h``-step forecast error covariance analytically from the VMA(``\infty``) representation:

```math
\text{MSE}_h = \sum_{j=0}^{h-1} \Psi_j \, \Sigma_\eta \, \Psi_j'
```

where:
- ``\Psi_j = J C^j`` are the VMA coefficient matrices from the companion form
- ``C`` is the companion matrix of the factor VAR
- ``J`` selects the first ``r`` rows
- ``\Sigma_\eta`` is the factor innovation covariance

| `ci_method` | Description | Available for | Best for |
|-------------|-------------|---------------|----------|
| `:none` | Point forecast only, zero bounds | all three model types | Quick exploration |
| `:theoretical` | Analytical VMA intervals, Gaussian | all three model types | Large samples, fastest |
| `:bootstrap` | Resampling of factor VAR residuals | all three model types | Non-Gaussian innovations |
| `:simulation` | Monte Carlo draws from the fitted state space | `DynamicFactorModel` only | Full uncertainty propagation |

```@example factor
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

Factor forecast standard errors widen with the horizon and then flatten: for this two-factor model they are 0.52 and 0.61 at ``h = 1`` and 0.83 and 0.86 by ``h = 10``. Because the factors are normalized to unit variance, a standard error of one *is* the unconditional standard deviation, so the bands are already three-quarters of the way to conveying no information at all — which is what weakly persistent monthly factors imply. Bootstrap intervals are preferred to theoretical ones whenever the innovations are visibly non-Gaussian, since they inherit the empirical residual distribution rather than assuming normality.

!!! note "Technical Note"
    Observable forecast standard errors combine factor uncertainty with idiosyncratic variance: ``\text{Var}(\hat{X}_{T+h}) = \Lambda \cdot \text{MSE}_h \cdot \Lambda' + \Sigma_e``. GDFM forecasts instead extrapolate each factor as an AR(1) with the closed-form variance ``\text{Var}(\hat{F}_{T+h,i}) = \sigma_i^2 \sum_{j=0}^{h-1} \phi_i^{2j}``.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `ci_method` | `Symbol` | `:theoretical` | Interval method, see the table above |
| `conf_level` | `Real` | `0.95` | Confidence level |
| `n_boot` | `Int` | `1000` | Replications for `:bootstrap` and `:simulation` |
| `p` | `Int` | `1` | Factor VAR lag order (`FactorModel` only) |

| Field | Type | Description |
|-------|------|-------------|
| `factors` | `Matrix{T}` | ``h \times r`` factor point forecasts |
| `observables` | `Matrix{T}` | ``h \times N`` observable point forecasts |
| `factors_lower` | `Matrix{T}` | ``h \times r`` lower bounds for factors |
| `factors_upper` | `Matrix{T}` | ``h \times r`` upper bounds for factors |
| `observables_lower` | `Matrix{T}` | ``h \times N`` lower bounds for observables |
| `observables_upper` | `Matrix{T}` | ``h \times N`` upper bounds for observables |
| `factors_se` | `Matrix{T}` | ``h \times r`` factor forecast standard errors |
| `observables_se` | `Matrix{T}` | ``h \times N`` observable forecast standard errors |
| `horizon` | `Int` | Forecast horizon ``h`` |
| `conf_level` | `T` | Confidence level |
| `ci_method` | `Symbol` | Interval method used |

---

## Generalized Dynamic Factor Model

The generalized dynamic factor model (GDFM) of Forni, Hallin, Lippi & Reichlin (2000, 2005) works in the frequency domain. Rather than assume a finite VAR for the factors, it exploits the spectral density of the panel directly, which admits common components driven by two-sided, infinitely many lags of the common shocks.

Each observable decomposes into common and idiosyncratic parts:

```math
x_{it} = \chi_{it} + \xi_{it}
```

where:
- ``\chi_{it}`` is the **common component**, driven by ``q`` common shocks
- ``\xi_{it}`` is the **idiosyncratic component**

The common component has the dynamic representation:

```math
\chi_{it} = b_{i1}(L) u_{1t} + b_{i2}(L) u_{2t} + \cdots + b_{iq}(L) u_{qt}
```

where:
- ``b_{ij}(L)`` are square-summable lag polynomial filters
- ``u_{jt}`` are orthonormal white-noise common shocks
- ``q`` is the number of dynamic factors

In the frequency domain the spectral density splits as ``\Sigma_X(\omega) = \Sigma_\chi(\omega) + \Sigma_\xi(\omega)``. Common factors produce eigenvalues that **diverge** with ``N`` while idiosyncratic components produce **bounded** ones, and that separation identifies the factor space without any finite-order restriction on factor dynamics.

!!! note "Technical Note"
    The default estimator is the FHLR lag-window spectrum ``\hat\Sigma_X(\theta)=(1/2\pi)\sum_{k=-M}^{M} w(k/M)\hat\Gamma_k e^{-ik\theta}``, evaluated on ``\theta_h=\pi h/M``. The Hermitian eigendecomposition at each grid point supplies the leading-``q`` dynamic principal components; those loadings are interpolated onto the Fourier ordinates so the time-domain projector ``L L^H`` can reconstruct ``\chi_t``. Pass `spectral=:smoothed_periodogram` for the older kernel-smoothed periodogram. Under `:lag_window`, `bandwidth=0` selects ``M=\max(3,\mathrm{round}(\tfrac12\sqrt{T}))``, which is 4 for the 60-observation panel used here.

```@example factor
gdfm = estimate_gdfm(X, 2;
    standardize=true,
    bandwidth=0,          # 0 selects ½√T under :lag_window
    kernel=:bartlett      # :bartlett, :parzen, or :tukey
)
report(gdfm)
```

```@example factor
chi = gdfm.common_component      # T × N common component
xi = gdfm.idiosyncratic          # T × N idiosyncratic component
shares = common_variance_share(gdfm)

(mean_share=round(mean(shares), digits=3),
 median_share=round(median(shares), digits=3),
 mean_r2=round(mean(r2(gdfm)), digits=3))
```

Two dynamic factors carry 28.7% and 15.0% of the average spectral mass (43.8% cumulative). The reconstructed common component accounts for a median 26.9% of the variance of an individual series (mean 31.3%). Three *static* factors deliver 37.8% of contemporaneous co-movement on the same panel. With ``T=60`` and ``M=4`` the lag-window common component is more conservative than static PCA: a short lag truncation on a wide panel does not automatically recover more variance than three contemporaneous factors.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `standardize` | `Bool` | `true` | Standardize data before estimation |
| `bandwidth` | `Int` | `0` | Lag-window ``M`` (default ``\max(3,\mathrm{round}(\tfrac12\sqrt{T}))``); periodogram bandwidth if `spectral=:smoothed_periodogram` |
| `spectral` | `Symbol` | `:lag_window` | `:lag_window` or `:smoothed_periodogram` |
| `kernel` | `Symbol` | `:bartlett` | Spectral kernel: `:bartlett`, `:parzen`, or `:tukey` |
| `r` | `Int` | `0` | Number of static factors, `0` sets ``r = q``; must satisfy ``r \geq q`` |

| Field | Type | Description |
|-------|------|-------------|
| `X` | `Matrix{T}` | Original ``T \times N`` data matrix |
| `factors` | `Matrix{T}` | ``T \times q`` time-domain factors, normalized to unit variance |
| `common_component` | `Matrix{T}` | ``T \times N`` common component ``\chi_t``, in original units |
| `idiosyncratic` | `Matrix{T}` | ``T \times N`` idiosyncratic component ``\xi_t = X - \chi`` |
| `loadings_spectral` | `Array{Complex{T},3}` | ``N \times q \times n_{freq}`` frequency-domain loadings |
| `spectral_density_X` | `Array{Complex{T},3}` | ``N \times N \times n_{freq}`` spectral density of ``X_t`` |
| `spectral_density_chi` | `Array{Complex{T},3}` | ``N \times N \times n_{freq}`` spectral density of ``\chi_t`` |
| `eigenvalues_spectral` | `Matrix{T}` | ``N \times n_{freq}`` eigenvalues across frequencies |
| `frequencies` | `Vector{T}` | Frequency grid from 0 to ``\pi`` |
| `q` | `Int` | Number of dynamic factors |
| `r` | `Int` | Number of static factors |
| `bandwidth` | `Int` | Lag-window ``M`` or periodogram bandwidth actually used |
| `kernel` | `Symbol` | Kernel type |
| `spectral` | `Symbol` | `:lag_window` or `:smoothed_periodogram` |
| `standardized` | `Bool` | Whether the data was standardized |
| `variance_explained` | `Vector{T}` | Average spectral variance share of each dynamic factor |
| `Z` | `Matrix{T}` | ``N \times r`` FHLR (2005) one-sided weights (generalized eigenvectors of ``(\\Gamma_\\chi(0), \\Gamma_\\xi(0))``) |
| `factors_onesided` | `Matrix{T}` | ``T \times r`` contemporaneous factors ``F_t = Z' X_t`` |
| `common_component_onesided` | `Matrix{T}` | ``T \times N`` one-sided common component |

`factors` and `common_component` are the **two-sided** FHLR (2000) reconstruction (inverse FFT over the sample). They use future observations and wrap at the sample ends, so they are for in-sample decomposition and historical decomposition. `Z` and `factors_onesided` are the **one-sided** FHLR (2005) filter: ``F_t`` depends on ``X_t`` only, given ``Z``.

### Selecting the Number of Dynamic Factors

`ic_criteria_gdfm` is a heuristic: the ``q`` that maximizes ``\bar\lambda_q / \bar\lambda_{q+1}`` in the frequency-averaged eigenvalues, and the smallest ``q`` reaching 90% cumulative variance. Neither is a consistent estimator of ``q``.

!!! warning "max_q is bounded by the cross-section"
    `ic_criteria_gdfm(X, max_q)` requires ``1 \leq \texttt{max\_q} \leq N``. The spectral density matrix is ``N \times N``, so there is no ``(N+1)``-th eigenvalue to rank; a larger `max_q` raises an `ArgumentError`. When the 90% threshold is never reached, `q_variance == max_q` and `boundary=true`, and the function warns.

```@example factor
ic_gdfm = ic_criteria_gdfm(X, 5; kernel=:bartlett)

(q_ratio=ic_gdfm.q_ratio, q_variance=ic_gdfm.q_variance, boundary=ic_gdfm.boundary,
 ratios=round.(ic_gdfm.eigenvalue_ratios, digits=2),
 cumvar=round.(ic_gdfm.cumulative_variance, digits=3))
```

The eigenvalue ratios are 1.91, 1.83, 1.23, 1.15, 1.15, so the ratio criterion still picks a single dynamic factor, but the first two ratios are close: the lag-window spectrum on this short panel does not separate ``q = 1`` from ``q = 2`` as sharply as the old smoothed periodogram did. The variance criterion reports 5, and the cumulative column shows why — 0.287, 0.438, 0.520, 0.587, 0.645, never crossing 0.9 — so `q_variance` is returning `max_q` as a fallback (`boundary=true`). Raise `max_q` when that happens, or use a consistent criterion.

Hallin–Liška (2007) minimises ``IC(q; c) = \log\bigl((1/N)\sum_{j>q}\bar\lambda_j\bigr) + q\cdot c\cdot p(N,T)`` on a grid of ``c``, then reads ``S_c``, the standard deviation of ``\hat q_j(c)`` across nested sub-panels ``(N_j, T_j)``. The first ``S_c = 0`` plateau near ``c = 0`` is ``q_{\max}`` (no penalty) and is discarded; the estimate is the longest remaining constant-``q`` plateau.

```@example factor
hl = hallin_liska(X, 5; c_grid=range(0, 3; length=40), subpanels=3)
(q=hl.q, interval=(round(hl.stability_interval[1]; digits=2),
                   round(hl.stability_interval[2]; digits=2)))
```

Bai–Ng (2007) fits a VAR on ``r`` static PCA factors and reads rank statistics ``D_{1,k}``, ``D_{2,k}`` on the residual-covariance eigenvalues. Amengual–Watson (2007) applies Bai–Ng (2002) IC to the residuals of ``X_t`` after projection on lagged static factors. `estimate_structural_dfm(X, :auto; q_method=:hallin_liska)` (or `:bai_ng`, `:amengual_watson`) selects ``q`` and then estimates.

```@example factor
bn = bai_ng_q(X, 4; p=1)
aw = amengual_watson_q(X, 4, 1)
(q_D1=bn.q_D1, q_D2=bn.q_D2, q_AW=aw.q)
```

| Field | Type | Description |
|-------|------|-------------|
| `eigenvalue_ratios` | `Vector{T}` | ``\bar\lambda_i / \bar\lambda_{i+1}`` for consecutive averaged eigenvalues |
| `cumulative_variance` | `Vector{T}` | Cumulative share of the averaged eigenvalues |
| `avg_eigenvalues` | `Vector{T}` | Frequency-averaged eigenvalues, first `max_q` |
| `q_ratio` | `Int` | ``q`` maximizing the eigenvalue ratio |
| `q_variance` | `Int` | Smallest ``q`` with cumulative variance ``\geq 0.9`` |
| `boundary` | `Bool` | `true` when the 90% threshold was not reached |

### One-sided estimation and forecasting

The two-sided projector cannot be used for forecasting: it looks into the future. FHLR (2005) invert the stored common and idiosyncratic spectra to lag covariances ``\hat\Gamma_\chi(k)``, ``\hat\Gamma_\xi(k)``, take the ``r`` generalized eigenvectors ``Z`` of the pencil ``(\hat\Gamma_\chi(0), \hat\Gamma_\xi(0))``, and form contemporaneous factors ``\hat F_t = Z' X_t``. The common-component forecast is the projection

```math
\hat\chi_{T+h|T} = \hat\Gamma_\chi(h)\, Z\, (Z'\hat\Gamma_X(0) Z)^{-1} Z' X_T.
```

`forecast(gdfm, h; method=:one_sided)` (or `method=:spectral`) implements that path. `method=:ar` is the older AR(1) recursion on the two-sided factors. `:spectral` is no longer a silent alias for `:ar`.

```@example factor
fc_fhlr = forecast(gdfm, 4; method=:one_sided, ci_method=:none)
fc_ar = forecast(gdfm, 4; method=:ar, ci_method=:none)
(size_os=size(fc_fhlr.observables), size_ar=size(fc_ar.observables),
 maxabs_diff=round(maximum(abs, fc_fhlr.observables - fc_ar.observables); digits=3))
```

### DFM vs GDFM

| Aspect | Dynamic Factor Model | Generalized DFM |
|--------|---------------------|-----------------|
| **Domain** | Time domain, PCA plus VAR | Frequency domain, spectral |
| **Factor dynamics** | Explicit finite VAR(``p``) | Implicit, two-sided filters |
| **Estimation** | Two-step or EM | Lag-window spectral density |
| **Cost** | Moderate | Higher, eigendecomposition per frequency |
| **Asymptotics** | ``T \to \infty`` for fixed ``r`` | ``N, T \to \infty`` jointly |
| **Likelihood available** | Yes | No |
| **Forecast** | Kalman / VAR on factors | FHLR (2005) one-sided projection (`method=:one_sided`) |
| **Best for** | Moderate ``N``, forecasting | Large ``N``, structural decomposition |

---

## [Block-Restricted Estimation](@id block_restricted)

When theory says that distinct groups of variables load on distinct factors — real activity, prices, financial conditions — block restrictions impose exactly that. Each factor is allowed to load only on its own block, and the EM algorithm estimates the model subject to the mask.

```math
x_{it} = \lambda_i' F_t + e_{it}, \quad \lambda_{ij} = 0 \text{ if variable } i \notin \text{block } j
```

where:
- ``\lambda_{ij}`` is the loading of variable ``i`` on factor ``j``
- the zero restriction applies whenever variable ``i`` is outside block ``j``

!!! note "Technical Note"
    Each block factor is initialized by the leading eigenvector of its own block covariance. EM then alternates ``F = X \Lambda (\Lambda'\Lambda)^{-1}`` (E-step) with ``\Lambda = \left[(F'F)^{-1} F'X\right]' \odot R`` (M-step), where ``R`` is the ``N \times r`` 0/1 mask. Iteration stops when the largest absolute change in ``\Lambda`` falls below ``10^{-6}``, or after 500 sweeps. Validation requires exactly ``r`` blocks, no shared indices, and at least 2 variables per block.

```@example factor
block_series = ["INDPRO", "PAYEMS", "UNRATE", "CUMFNS", "DPCERA3M086SBEA",
                "CPIAUCSL", "CPIULFSL", "PCEPI", "WPSFD49207", "PPICMM",
                "FEDFUNDS", "GS10", "BAA", "TB3MS", "S&P 500"]
X_block = X[:, [findfirst(==(v), varnames(fred)) for v in block_series]]

blocks = Dict(
    :real => [1, 2, 3, 4, 5],
    :nominal => [6, 7, 8, 9, 10],
    :financial => [11, 12, 13, 14, 15]
)

fm_block = estimate_factors(X_block, 3; blocks=blocks)
report(fm_block)
```

Every loading outside its block is exactly zero — 30 of the 45 entries of ``\Lambda`` — so each factor is a labelled object rather than a rotation-dependent linear combination. The nominal factor loads at ``-0.97`` on `CPIULFSL` and ``-0.97`` on `CPIAUCSL` but only ``-0.11`` on `PPICMM`, showing that the consumer price series move as a bloc while the commodity-price component does not. The real factor is anchored by `CUMFNS` (``-0.92``) and `INDPRO` (``-0.90``), and the financial factor by the credit spread `BAA` (``-0.85``) and `GS10` (``-0.74``). Sign is arbitrary within each block, as always; magnitudes relative to other members of the same block are what carry meaning.

!!! warning "Variance shares are panel-wide, not block-wise"
    In the `report` output for a block-restricted model, the `Variance Explained` column reports the ordinary principal-component shares of the whole panel, listed against block labels. Read the loadings, and the per-variable `r2`, to judge how much each *block* factor explains.

---

## Structural Dynamic Factor Model

The structural DFM (Forni, Giannone, Lippi & Reichlin 2009) identifies ``q`` common shocks in a large panel from ``r \ge q`` static factors. The default `method=:fglr` estimates static principal components, fits a VAR on those ``r`` series, reduces the residual covariance to rank ``q``, and identifies the shocks on selected observables. Pass `method=:gdfm_var` for the legacy two-sided GDFM-factor VAR. Identification methods besides Cholesky and signs route through `compute_Q` on the factor VAR (`:long_run` is formed in **panel** space on `target_vars`).

```math
X_t = \Lambda F_t + \xi_t, \qquad A(L) F_t = K \varepsilon_t, \qquad C(L) = \Lambda A(L)^{-1} K H
```

where:
- ``F_t`` is the ``r \times 1`` vector of static factors (``r \ge q``)
- ``\Lambda`` is the ``N \times r`` loading matrix
- ``K`` is ``r \times q``, from the leading ``q`` eigenvectors of the factor-VAR residual covariance
- ``H`` is the ``q \times q`` identification rotation (Cholesky on `order`, or sign restrictions)
- ``\varepsilon_t`` are the ``q`` orthonormal common shocks

Panel IRFs are ``\Lambda \Psi_h K H``, with ``\Psi_h`` the reduced-form MA of the factor VAR. `r` vs `q`: ``q`` is the number of primitive shocks; ``r`` stacks those shocks and their lags so variables that load on ``F_{t-1}`` are not misspecified. The two-sided `method=:gdfm_var` path uses the ``q`` GDFM factors directly and is kept for reproducibility.

Cholesky (on `order`, default the first ``q`` observables) and sign restrictions are both available.

```@example factor
struct_series = ["INDPRO", "IPFINAL", "IPMANSICS", "CUMFNS", "PAYEMS", "MANEMP", "UNRATE",
                 "DPCERA3M086SBEA", "RETAILx", "HOUST", "PERMIT",
                 "CPIAUCSL", "CPIULFSL", "PCEPI", "WPSFD49207",
                 "FEDFUNDS", "TB3MS", "GS10", "BAA", "S&P 500"]
X20 = X[:, [findfirst(==(v), varnames(fred)) for v in struct_series]]

sdfm20 = estimate_structural_dfm(X20, 2; identification=:cholesky, p=1, H=20,
                                 varnames=struct_series)
report(sdfm20)
```

```@example factor
d = fevd(sdfm20, 20)
report(d)
```

```@example factor
d_panel = fevd(sdfm20, 20; space=:panel, include_idiosyncratic=true)
(variables=d_panel.variables[1:3],
 shocks=d_panel.shocks,
 INDPRO_idio_h1=round(d_panel.proportions[1, end, 1]; digits=3))
```

The impact on the two static factors is nearly diagonal — 0.9847 and 0.9488 on the diagonal, off-diagonal ``-0.0027`` and 0.0352 — so Cholesky on the first two observables is close to a relabelling of the PCA factors rather than a substantive restriction. The FEVD confirms it: the first structural shock explains 100% of static factor 1's forecast error at impact and 99.4% at ``h = 20``, and 0.1% of static factor 2 at impact, rising to 9.9% by ``h = 4`` and staying there through ``h = 20``. These are innovations of the static-factor VAR after rank-``q`` reduction, not the frequency-domain GDFM eigenvectors. The observable FEVD (`space=:panel`) answers the applied question: with idiosyncratic included, `INDPRO`'s one-step error is mostly idiosyncratic (the common shocks' shares plus that remainder sum to 1). Factor-space FEVD remains the default.

!!! note "Technical Note"
    Under `:fglr`, ``\Lambda`` is the static PCA loading matrix. With `standardize=true` and `units=:raw` (the default), panel IRFs are rescaled by each series' standard deviation so responses are in original units. Under `:gdfm_var`, loadings are OLS of the untransformed panel on the two-sided GDFM factors.

```@example factor
r20 = irf(sdfm20, 20)
report(r20)
```

```julia
plot_result(r20)
```

```@raw html
<iframe src="../assets/plots/sdfm_irf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The first structural shock moves `UNRATE` by ``-0.0481`` and `FEDFUNDS` by ``-0.0046`` on impact, against ``0.0069`` for `INDPRO` — the scale differences are units, not economics, since `INDPRO` enters as a monthly log difference and `FEDFUNDS` as a level difference in percentage points. `UNRATE`'s response is already ``-0.0001`` by ``h = 4`` and numerically negligible past that: with 60 observations of monthly growth rates the extracted static factors are close to serially uncorrelated and essentially all of the action is at impact.

| Field | Type | Description |
|-------|------|-------------|
| `gdfm` | `GeneralizedDynamicFactorModel{T}` | Underlying GDFM estimate |
| `factor_var` | `VARModel{T}` | VAR(``p``) on the ``r`` static factors (`:fglr`) or ``q`` GDFM factors (`:gdfm_var`) |
| `B0` | `Matrix{T}` | ``r \times q`` impact ``K H`` (`:fglr`) or ``q \times q`` (`:gdfm_var`) |
| `Q` | `Matrix{T}` | ``q \times q`` identification rotation ``H`` (first accepted draw under `:sign`) |
| `K` | `Matrix{T}` | ``r \times q`` rank-``q`` loading of factor-VAR residuals |
| `r` | `Int` | Number of static factors |
| `method` | `Symbol` | `:fglr` (default) or `:gdfm_var` |
| `identification` | `Symbol` | `:cholesky`, `:sign`, `:long_run`, or a `compute_Q` method |
| `structural_irf` | `Array{T,3}` | ``H \times N \times q`` panel-wide structural IRFs |
| `loadings_td` | `Matrix{T}` | ``N \times r`` (`:fglr`) or ``N \times q`` (`:gdfm_var`) loadings |
| `loadings_static` | `Matrix{T}` | ``N \times r`` static PCA loadings (`:fglr`) |
| `static_factors` | `Matrix{T}` | ``T \times r`` static PCA factors |
| `p_var` | `Int` | VAR lag order on the factors |
| `shock_names` | `Vector{String}` | Labels for the ``q`` structural shocks |
| `varnames` | `Vector{String}` | Panel variable names (length ``N``) |
| `units` | `Symbol` | `:raw` or `:standardized` IRF units |
| `identified_set` | `SignIdentifiedSet` or `nothing` | Accepted panel IRFs when `store_all=true` |
| `acceptance_rate` | `T` | Fraction of Haar draws accepted under `:sign` |

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `identification` | `Symbol` | `:cholesky` | `:cholesky`, `:sign`, `:long_run`, `:proxy`, `:narrative`, ICA/ML/heteroskedastic `compute_Q` methods, `:arias`/`:uhlig` |
| `target_vars` | `Vector` | `1:q` | Observables whose long-run responses are lower-triangular (`:long_run`) |
| `method` | `Symbol` | `:fglr` | `:fglr` (FGLR 2009) or `:gdfm_var` (legacy two-sided) |
| `r` | `Int` | `q` | Static factors; must satisfy ``r \ge q`` |
| `order` | `Vector{Int}` | `1:q` | Observable indices for Cholesky under `:fglr` |
| `p` | `Int` or `Symbol` | `1` | VAR lag order, or `:aic`/`:bic`/`:hq` over `1:p_max` |
| `p_max` | `Int` | `8` | Grid upper bound when `p` is a criterion |
| `check_stability` | `Bool` | `true` | Warn when the factor-VAR companion modulus is ``\ge 1`` |
| `H` | `Int` | `40` | IRF horizon stored in `structural_irf` |
| `sign_check` | `Function` | `nothing` | Predicate on the IRF array (`H×N×q` under `:panel`) |
| `sign_restrictions` | `Vector` / `SVARRestrictions` | `nothing` | Declarative `(variable, shock, horizons, sign)` tuples |
| `restriction_space` | `Symbol` | `:panel` | `:panel` (observables) or `:factor` (static factors) |
| `store_all` | `Bool` | `false` | Keep the accepted set as `identified_set` |
| `max_draws` | `Int` | `1000` | Rotation draws searched under `:sign` |
| `rng` | `AbstractRNG` | `default_rng()` | Pins the Haar search |
| `varnames` | `Vector{String}` | `nothing` | Panel names (forwarded from `TimeSeriesData`) |
| `shock_names` | `Vector{String}` | `Shock 1, …` | Labels for the ``q`` shocks |
| `standardize` | `Bool` | `true` | Standardize for PCA / GDFM |
| `bandwidth` | `Int` | `0` | GDFM lag-window ``M`` or periodogram bandwidth |
| `kernel` | `Symbol` | `:bartlett` | Spectral kernel |
| `spectral` | `Symbol` | `:lag_window` | `:lag_window` or `:smoothed_periodogram` |
| `instrument` | `AbstractVector` | `nothing` | External instrument for `identification=:proxy` (length ``T`` or ``T_{\mathrm{eff}}``; `NaN` dropped pairwise) |
| `normalize` | `Tuple` | `(1, 1.0)` | Unit-effect pair `(variable, value)` on an observable |

### Sign Restrictions

Applied structural DFMs restrict **observables** — output, the unemployment rate, the policy rate — not the arbitrarily rotated PCA factors (Forni & Gambetti 2010). Default `restriction_space=:panel` hands the predicate the ``H \times N \times q`` panel IRF ``\Lambda \Psi_h K H``, indexed `[horizon, variable, shock]`. Declarative tuples resolve names through `varindex`:

```@example factor
sdfm_sign = estimate_structural_dfm(X20, 2;
    identification=:sign,
    sign_restrictions=[("INDPRO", 1, 1:2, :positive),
                       ("UNRATE", 1, 1:2, :negative)],
    varnames=struct_series, max_draws=1000, H=20, p=1,
    rng=MersenneTwister(42))

(Q11=round(sdfm_sign.Q[1, 1]; digits=4),
 acceptance=round(sdfm_sign.acceptance_rate; digits=3),
 INDPRO=round(irf(sdfm_sign, 20).values[1, varindex(sdfm_sign, "INDPRO"), 1]; digits=4),
 UNRATE=round(irf(sdfm_sign, 20).values[1, varindex(sdfm_sign, "UNRATE"), 1]; digits=4))
```

Shock 1 is required to raise `INDPRO` and lower `UNRATE` at horizons 1 and 2. The first accepted rotation has (1,1) entry ``-0.5537`` (acceptance rate 33.3%). On impact `INDPRO` moves ``0.0016`` and `UNRATE` ``-0.0517``, so both restricted cells hold on the returned **panel** IRFs — that is the economically meaningful object, not the factor-space array. A closure `sign_check` on the same ``H \times N \times q`` array is equivalent; pass `restriction_space=:factor` only if you deliberately want the old factor-space contract. An unsatisfiable restriction throws `IdentificationError` naming the variable with the lowest pass rate.

Sign restrictions identify a **set**. `store_all=true` keeps every accepted Haar draw in `identified_set` (`SignIdentifiedSet`); `irf` then returns the pointwise median with 16/84 bands (`ci_type = :sign_set`). Pass `point=:first` to recover the first-accepted path stored in `Q`:

```@example factor
sdfm_set = estimate_structural_dfm(X20, 2;
    identification=:sign,
    sign_restrictions=[("INDPRO", 1, 1:1, :positive)],
    varnames=struct_series, store_all=true, max_draws=400, H=12, p=1,
    rng=MersenneTwister(42))
rset = irf(sdfm_set, 12)
i_ip = varindex(sdfm_set, "INDPRO")
(n_accepted=sdfm_set.identified_set.n_accepted,
 ci_type=rset.ci_type,
 median=round(rset.values[1, i_ip, 1]; digits=4),
 lo=round(rset.ci_lower[1, i_ip, 1]; digits=4),
 hi=round(rset.ci_upper[1, i_ip, 1]; digits=4))
```

Of 400 draws, 200 pass the single `INDPRO` restriction. The impact median for `INDPRO` is ``0.0046``, inside the 16/84 band ``[0.0017, 0.0066]``. The band is a set-identification statement, not sampling uncertainty (Baumeister & Hamilton 2015): tightening the restriction shrinks the set, not a confidence interval.

### Two-Step Estimation

A pre-estimated GDFM can be passed straight in, which avoids re-running the spectral estimation when several identification schemes are compared on the same factors:

```@example factor
gdfm_pre = estimate_gdfm(X20, 2)
sdfm_two = estimate_structural_dfm(gdfm_pre; identification=:cholesky, p=1, H=20)

round.(sdfm_two.B0, digits=4)
```

```@example factor
sdfm_lr = estimate_structural_dfm(X20, 2; identification=:long_run,
    target_vars=["INDPRO", "IPFINAL"], varnames=struct_series, p=1, H=20)
hd20 = historical_decomposition(sdfm20)
(lr=sdfm_lr.identification,
 hd_ok=verify_decomposition(hd20),
 hd_shocks=hd20.shock_names)
```

### Panel IRFs

`sdfm_panel_irf` projects factor-space impulse responses onto all ``N`` observables through ``\Lambda``, the structural-DFM counterpart of [`favar_panel_irf`](@ref favar_page). Two forms are available:

```@example factor
# Form 1: convenience — recomputes factor IRFs from the VAR, so any horizon is available
panel_irf = sdfm_panel_irf(sdfm20, 20)
report(panel_irf)
```

```@example factor
# Form 2: project an existing factor-space ImpulseResponse
factor_irf = irf(sdfm20.factor_var, 20)
panel_irf2 = sdfm_panel_irf(sdfm20, factor_irf)
size(panel_irf2.values)
```

```julia
plot_result(panel_irf)
```

```@raw html
<iframe src="../assets/plots/sdfm_panel_irf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The first form recomputes the factor IRFs from the stored VAR coefficients and rotation, so horizons beyond the ``H`` used at estimation time are available. The second accepts any factor-space `ImpulseResponse` — from a custom identification, for instance — validates that it has exactly ``q`` variables and ``q`` shocks, and applies the same ``\Lambda`` projection. Point IRFs have `ci_type = :none`. Pass `ci_type=:bootstrap` to resample factor-VAR residuals (`:iid`, `:wild`, or `:block`), re-estimate the VAR and the rank-``q`` reduction, re-apply the stored identification, and take pointwise quantiles of the **panel** draws — never mapped interval endpoints. `sdfm_panel_irf(sdfm, irf(sdfm.factor_var, H; ci_type=:bootstrap))` pushes those factor draws through ``\Lambda`` and keeps `ci_type = :bootstrap`. Row labels come from `sdfm.varnames`, which defaults to `Var 1 … Var N` in column order of the matrix passed to `estimate_structural_dfm` when names are not supplied.

```@example factor
ir_boot = irf(sdfm20, 8; ci_type=:bootstrap, reps=40, rng=MersenneTwister(20))
(ci_type=ir_boot.ci_type, n_draws=size(ir_boot._draws, 1),
 lo=round(ir_boot.ci_lower[1, 1, 1]; digits=4),
 hi=round(ir_boot.ci_upper[1, 1, 1]; digits=4))
```

### Historical decomposition and forecasting

`structural_shocks(sdfm)` returns the ``T_{\mathrm{eff}} \times q`` series ``\hat\varepsilon_t = B_0^+ \hat u_t`` from the stored rotation. `forecast(sdfm, h)` maps the factor-VAR forecast through ``\Lambda``; bootstrap bands, when requested, are quantiles of those mapped draws.

```@example factor
εhat = structural_shocks(sdfm20)
fc20 = forecast(sdfm20, 4; ci_method=:none)
(shock_cov=round.(cov(εhat); digits=2),
 fc_size=size(fc20.observables),
 p_bic=estimate_structural_dfm(X20, 2; p=:bic, p_max=4, H=8).p_var)
```

Lag order ``p`` can be an integer or `:aic`/`:bic`/`:hq`. `show` prints the companion-matrix max-eigenvalue modulus; `is_stable(sdfm)` is true when that modulus is strictly less than one.

### External-instrument identification

Stock & Watson (2012, 2016) identify one common shock at a time with an instrument ``z_t`` that is correlated with ``\varepsilon_{1t}`` and orthogonal to the others. The first impact column is proportional to ``\mathrm{Cov}(\hat u_t, z_t)``, then scaled so a chosen observable moves by one unit. The shipped `:mp_shocks` panel already aligns high-frequency monetary surprises (`mp1`) with the quarterly FRED-style macro block; a monthly FRED-MD exercise uses the same pairwise-missing convention after mapping each surprise to its FOMC month.

```@example factor
mp = load_example(:mp_shocks)
Ymp = to_matrix(mp[:, ["ygap", "infl", "ffr"]])
zmp = vec(to_matrix(mp[:, ["mp1"]]))
keep = [all(isfinite, Ymp[t, :]) for t in 1:size(Ymp, 1)]
Ymp, zmp = Ymp[keep, :], zmp[keep]
sdfm_iv = estimate_structural_dfm(Ymp, 1; r=1, p=1, H=8, standardize=true,
    identification=:proxy, instrument=zmp, normalize=("ygap", 1.0),
    varnames=["ygap", "infl", "ffr"])
(F=round(sdfm_iv.first_stage_F; digits=1),
 impact_ygap=round(irf(sdfm_iv, 1).values[1, 1, 1]; digits=3))
```

A first-stage F below 10 emits a weak-instrument warning (Montiel Olea, Stock & Watson 2021). Bootstrap IRFs resample the instrument jointly with the factor-VAR residuals.

---

## Asymptotic Theory

Under the assumptions of Bai & Ng (2002) and Bai (2003), principal components consistently estimate the factor space as ``N`` and ``T`` grow together. The convergence rate is governed by the smaller of the two dimensions.

```math
\frac{1}{T} \sum_{t=1}^T \|\hat{F}_t - H F_t\|^2 = O_p\left( \frac{1}{\min(N, T)} \right)
```

where:
- ``\hat{F}_t`` is the estimated factor vector at time ``t``
- ``F_t`` is the true factor vector
- ``H`` is the ``r \times r`` rotation matrix that the estimator cannot pin down

For large ``N`` and ``T`` the factor estimates are asymptotically normal:

```math
\sqrt{T} (\hat{F}_t - H F_t) \xrightarrow{d} N(0, V)
```

where:
- ``V`` depends on the cross-sectional and temporal dependence of the idiosyncratic errors

Consistency survives weak cross-sectional and temporal dependence in ``e_{it}``, which is what makes PCA usable on real macro panels where idiosyncratic errors are plainly correlated within sectors. The ``\min(N, T)`` rate carries a practical warning: a panel with long ``T`` but small ``N`` gains nothing from factor extraction, because the factors stay imprecisely estimated no matter how many periods are added. When ``\sqrt{T}/N \to 0`` fails, Bai & Ng (2006) show that factor-augmented regressions need a correction for the generated-regressor problem.

---

## Applications

### Diffusion Index Forecasting

Estimated factors serve as regressors for a target variable ``y_{t+h}``:

```math
y_{t+h} = \alpha + \beta' \hat{F}_t + \gamma' y_{t:t-p} + \varepsilon_{t+h}
```

where:
- ``\alpha`` is the intercept
- ``\beta`` is the ``r \times 1`` vector of factor coefficients
- ``\hat{F}_t`` is the ``r \times 1`` vector of estimated factors at time ``t``
- ``\gamma`` collects the autoregressive coefficients on own lags of ``y``
- ``\varepsilon_{t+h}`` is the forecast error

Compressing a large panel into a few predictors is what makes this regression estimable at all, and Stock & Watson (2002b) document the resulting forecast gains over pure autoregressions.

### Factor-Augmented VAR

Factors can also enter a VAR alongside observed policy variables, which is what makes structural analysis with a large information set possible. See the [Factor-Augmented VAR](@ref favar_page) page for `estimate_favar`, the Bayesian Gibbs sampler, and panel-wide impulse response mapping.

### Real-Time Nowcasting

A DFM with mixed frequencies and a ragged edge is the standard nowcasting device. See [DFM Nowcasting](@ref nowcast_dfm_page) for `estimate_nowcast_dfm`, temporal aggregation, and news decompositions.

---

## Complete Example

The full workflow: select the factor count, estimate static and dynamic models, check the fit, and forecast.

```@example factor
# Step 1: number of factors from the Bai-Ng criteria
ic_full = ic_criteria(X, 10)
(IC1=ic_full.r_IC1, IC2=ic_full.r_IC2, IC3=ic_full.r_IC3)
```

```@example factor
# Step 2: static factor model at the IC2 choice
fm_full = estimate_factors(X, ic_full.r_IC2; standardize=true)
report(fm_full)
```

```@example factor
# Step 3: dynamic factor model with VAR(1) factor dynamics
dfm_full = estimate_dynamic_factors(X, ic_full.r_IC2, 1; method=:twostep, standardize=true)
report(dfm_full)
```

```@example factor
# Step 4: fit diagnostics and stationarity of the factor VAR
(mean_r2_static=round(mean(r2(fm_full)), digits=3),
 mean_r2_dynamic=round(mean(r2(dfm_full)), digits=3),
 stationary=is_stationary(dfm_full))
```

```@example factor
# Step 5: 12-step forecast with theoretical intervals
fc_full = forecast(dfm_full, 12; ci_method=:theoretical, conf_level=0.95)
report(fc_full)
```

```julia
plot_result(fc_full)
```

```@raw html
<iframe src="../assets/plots/forecast_factor.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

IC2 selects four factors, one more than the three used in the sections above, and the fourth factor lifts the panel-average ``R^2`` from 0.378 to 0.448. Static and dynamic estimates share the same PCA factors under `:twostep`, so their ``R^2`` values coincide exactly; the dynamic model adds the VAR(1) law of motion that the forecast needs. The factor VAR is stationary, so the 12-step forecast bands widen and then settle at the unconditional standard deviation instead of diverging.

---

## Saving Results

[`save_model`](@ref) persists the fitted result to a versioned JLD2 file; [`load_model`](@ref) reconstructs it. JLD2 is a package dependency --- no extra `using` is required. Every exported result type on this page is saveable; the living catalog is the [API Reference](@ref api_page) Persistence table. See [Data Management](@ref data_page) for bundles, `note=`, `model_info`, compression, and the reproducibility manifest.

```@example factor
path = joinpath(mktempdir(), "factor.jld2")
save_model(fm, path)
fm2 = load_model(path)
typeof(fm2)
```

---

## Common Pitfalls

1. **Choosing ``r`` from the scree plot alone.** The elbow is subjective and moves with the standardization choice. Run `ic_criteria(X, r_max)` and check that IC1, IC2, and IC3 agree. When a criterion returns exactly `r_max`, as IC3 does on the 60-observation panel here, its penalty has failed rather than found ten factors — enlarge `r_max` and prefer the interior selection.

2. **Reading loadings as structural parameters.** Loadings are identified only up to an ``r \times r`` rotation, so individual signs and magnitudes are not invariant. Use `blocks=` to label factors by construction, or interpret only rotation-invariant quantities such as `r2` and the variance shares.

3. **Skipping standardization.** With heterogeneous scales — interest rates in percent next to index levels — the high-variance series dominate the principal components entirely. `standardize=true` is the default; override it only when every series is already comparably scaled.

4. **Misspecifying the block structure.** Block-restricted estimation requires exactly ``r`` blocks, no variable in two blocks, indices within ``[1, N]``, and at least 2 variables per block. Each violation raises an `ArgumentError` naming the offending block.

5. **Applying the GDFM to a narrow panel.** The spectral estimator needs ``N`` and ``T`` to grow jointly; with fewer than about 20 series the ``N \times N`` spectral density is too noisy to separate diverging from bounded eigenvalues. Use a two-step or EM DFM instead.

6. **Forecasting from non-stationary factor dynamics.** The forecast recursion assumes the factor companion matrix is stable. Check `is_stationary(dfm)` first; if it fails, either reduce ``p`` or difference the offending series before extraction, since a unit root in a factor makes every horizon's interval meaningless.

7. **Comparing panel IRF magnitudes across series.** `sdfm_panel_irf` and `irf(::StructuralDFM, H)` return responses in each variable's own units. Rescale by each series' standard deviation before ranking responses.

8. **Restricting factors instead of observables.** Default `restriction_space=:panel`. A predicate on PCA factor 1 does not constrain `INDPRO` or `FEDFUNDS`. Use `sign_restrictions` with variable names, or `varindex(sdfm, "INDPRO")` inside `sign_check`.

9. **Treating a sign-identified IRF as a point.** With `store_all=true`, `irf` returns a median and 16/84 set bands (`ci_type = :sign_set`). The first accepted `Q` is one draw from that set. Pass `rng=` to pin the search; do not interpret a single draw as the unique structural response.

10. **Setting ``r < q``.** FGLR requires at least as many static factors as shocks. `r < q` throws `ArgumentError`. Two-sided GDFM factors (`method=:gdfm_var`) are non-fundamental: they are two-sided projections and are not valid for real-time forecasting (FHLR 2005). Use `factors_onesided` or `forecast(gdfm, h; method=:one_sided)`.

11. **Treating `ic_criteria_gdfm` as a consistent selector.** The eigenvalue-ratio and 90% rules are heuristics. When `boundary=true`, the 90% threshold was never reached. Use `hallin_liska`, `bai_ng_q`, or `amengual_watson_q`.

12. **Expecting `forecast(gdfm; method=:spectral)` to equal `:ar`.** `:spectral` is the FHLR (2005) projection, not an AR(1) alias. Pass `method=:ar` for the two-sided factor recursion.

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

- Baumeister, C., & Hamilton, J. D. (2015). Sign Restrictions, Structural Vector Autoregressions, and Useful Prior Information.
  *Econometrica*, 83(5), 1963-1999. [DOI](https://doi.org/10.3982/ECTA12356)

- Forni, M., & Gambetti, L. (2010). The Dynamic Effects of Monetary Policy: A Structural Factor Model Approach.
  *Journal of Monetary Economics*, 57(2), 203-216. [DOI](https://doi.org/10.1016/j.jmoneco.2009.11.009)

- Forni, M., Giannone, D., Lippi, M., & Reichlin, L. (2009). Opening the Black Box: Structural Factor Models with Large Cross-Sections.
  *Econometric Theory*, 25(5), 1319-1347. [DOI](https://doi.org/10.1017/S026646660809052X)

- Forni, M., Hallin, M., Lippi, M., & Reichlin, L. (2000). The Generalized Dynamic-Factor Model: Identification and Estimation.
  *Review of Economics and Statistics*, 82(4), 540-554. [DOI](https://doi.org/10.1162/003465300559037)

- Forni, M., Hallin, M., Lippi, M., & Reichlin, L. (2005). The Generalized Dynamic Factor Model: One-Sided Estimation and Forecasting.
  *Journal of the American Statistical Association*, 100(471), 830-840. [DOI](https://doi.org/10.1198/016214504000002050)

- Hallin, M., & Liska, R. (2007). Determining the Number of Factors in the General Dynamic Factor Model.
  *Journal of the American Statistical Association*, 102(478), 603-617. [DOI](https://doi.org/10.1198/016214506000001275)

- Bai, J., & Ng, S. (2007). Determining the Number of Primitive Shocks in Factor Models.
  *Journal of Business & Economic Statistics*, 25(1), 52-60. [DOI](https://doi.org/10.1198/073500106000000413)

- Amengual, D., & Watson, M. W. (2007). Consistent Estimation of the Number of Dynamic Factors in a Large N and T Panel.
  *Journal of Business & Economic Statistics*, 25(1), 91-96. [DOI](https://doi.org/10.1198/073500106000000613)

- McCracken, M. W., & Ng, S. (2016). FRED-MD: A Monthly Database for Macroeconomic Research.
  *Journal of Business & Economic Statistics*, 34(4), 574-589. [DOI](https://doi.org/10.1080/07350015.2015.1086655)

- Stock, J. H., & Watson, M. W. (2002a). Forecasting Using Principal Components from a Large Number of Predictors.
  *Journal of the American Statistical Association*, 97(460), 1167-1179. [DOI](https://doi.org/10.1198/016214502388618960)

- Stock, J. H., & Watson, M. W. (2002b). Macroeconomic Forecasting Using Diffusion Indexes.
  *Journal of Business & Economic Statistics*, 20(2), 147-162. [DOI](https://doi.org/10.1198/073500102317351921)
