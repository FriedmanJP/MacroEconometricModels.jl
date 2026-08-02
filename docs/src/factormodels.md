# [Factor Models](@id factor_page)

Factor models compress a panel of hundreds of macroeconomic indicators into a handful of latent common factors, turning a cross-section that no VAR can hold into a system small enough to estimate. This page covers the four estimators the package provides — static principal components, dynamic factor models with explicit VAR dynamics, the generalized dynamic factor model estimated in the frequency domain, and structural identification of the common shocks.

- **Static factor model**: principal components (Stock & Watson 2002a) with automatic panel orientation, standardization, and block-restricted EM estimation
- **Information criteria**: Bai & Ng (2002) IC1--IC3 for the number of static factors, AIC/BIC grid search for the dynamic specification, and eigenvalue criteria for the number of dynamic factors
- **Dynamic factor model**: two-step (PCA + VAR) or EM (Kalman smoother) estimation with four confidence-interval methods for forecasting (Doz, Giannone & Reichlin 2011, 2012)
- **Generalized dynamic factor model**: spectral estimation via the kernel-smoothed periodogram with frequency-by-frequency eigenanalysis (Forni, Hallin, Lippi & Reichlin 2000, 2005)
- **Structural DFM**: Cholesky or sign-restriction identification on the common factors, with panel-wide structural impulse responses (Forni, Giannone, Lippi & Reichlin 2009)
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
    Estimation proceeds in four steps: kernel-smooth the periodogram to get ``\hat{\Sigma}_X(\omega)``; take the Hermitian eigendecomposition at each Fourier frequency; keep the leading ``q`` eigenvectors as dynamic principal components; and reconstruct ``\chi_t`` by applying the projector ``L L^H`` to the Fourier transform of ``X`` and inverting it. The bandwidth defaults to ``\max(3, \lfloor T^{1/3} \rceil)``, which is 4 for the 60-observation panel used here.

```@example factor
gdfm = estimate_gdfm(X, 2;
    standardize=true,
    bandwidth=0,          # 0 selects T^(1/3)
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

Two dynamic factors carry 45.5% and 18.7% of the average spectral mass, and the reconstructed common component accounts for a median 82.7% of the variance of an individual series — far above the 37.8% that three *static* factors deliver on the same panel. The gap is the point of the GDFM: because ``\chi_{it}`` is a two-sided filter of the common shocks, a series that responds to the aggregate cycle with a lead or a lag is still classified as common, whereas static PCA can only match contemporaneous co-movement.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `standardize` | `Bool` | `true` | Standardize data before estimation |
| `bandwidth` | `Int` | `0` | Kernel bandwidth, `0` selects ``\max(3, T^{1/3})`` |
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
| `bandwidth` | `Int` | Kernel smoothing bandwidth actually used |
| `kernel` | `Symbol` | Kernel type |
| `standardized` | `Bool` | Whether the data was standardized |
| `variance_explained` | `Vector{T}` | Average spectral variance share of each dynamic factor |

### Selecting the Number of Dynamic Factors

The GDFM uses eigenvalue criteria rather than information criteria. `ic_criteria_gdfm` returns the ratio criterion — the ``q`` that maximizes ``\bar\lambda_q / \bar\lambda_{q+1}`` in the frequency-averaged eigenvalues — and the smallest ``q`` reaching 90% cumulative variance.

!!! warning "max_q is bounded by the cross-section"
    `ic_criteria_gdfm(X, max_q)` requires ``1 \leq \texttt{max\_q} \leq N``. The spectral density matrix is ``N \times N``, so there is no ``(N+1)``-th eigenvalue to rank; a larger `max_q` raises an `ArgumentError`.

```@example factor
ic_gdfm = ic_criteria_gdfm(X, 5; kernel=:bartlett)

(q_ratio=ic_gdfm.q_ratio, q_variance=ic_gdfm.q_variance,
 ratios=round.(ic_gdfm.eigenvalue_ratios, digits=2),
 cumvar=round.(ic_gdfm.cumulative_variance, digits=3))
```

The eigenvalue ratios are 2.44, 1.62, 1.35, 1.47, 1.39, so the ratio criterion picks a single dynamic factor: the drop from the first to the second frequency-averaged eigenvalue is much the largest, and the sequence is flat afterwards. The variance criterion reports 5, but read the cumulative column before believing it — the series reaches 0.899 at ``q = 5`` and never crosses the 0.9 threshold, so `q_variance` is returning `max_q` as a fallback rather than a genuine crossing. Raise `max_q` when `q_variance == max_q`. The two criteria bracket the honest answer: one shock dominates the spectrum, but reconstructing 90% of the variance takes more than five.

Hallin & Liska (2007) give the formal information criterion for this problem, with a tuning-constant stability check that neither of these two diagnostics provides.

| Field | Type | Description |
|-------|------|-------------|
| `eigenvalue_ratios` | `Vector{T}` | ``\bar\lambda_i / \bar\lambda_{i+1}`` for consecutive averaged eigenvalues |
| `cumulative_variance` | `Vector{T}` | Cumulative share of the averaged eigenvalues |
| `avg_eigenvalues` | `Vector{T}` | Frequency-averaged eigenvalues, first `max_q` |
| `q_ratio` | `Int` | ``q`` maximizing the eigenvalue ratio |
| `q_variance` | `Int` | Smallest ``q`` with cumulative variance ``\geq 0.9`` |

### DFM vs GDFM

| Aspect | Dynamic Factor Model | Generalized DFM |
|--------|---------------------|-----------------|
| **Domain** | Time domain, PCA plus VAR | Frequency domain, spectral |
| **Factor dynamics** | Explicit finite VAR(``p``) | Implicit, two-sided filters |
| **Estimation** | Two-step or EM | Kernel-smoothed periodogram |
| **Cost** | Moderate | Higher, eigendecomposition per frequency |
| **Asymptotics** | ``T \to \infty`` for fixed ``r`` | ``N, T \to \infty`` jointly |
| **Likelihood available** | Yes | No |
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

The structural DFM (Forni, Giannone, Lippi & Reichlin 2009) identifies structural shocks in a large panel by applying SVAR identification to the common factors. It fits a VAR on the time-domain GDFM factors, identifies that small system, and maps the identified responses out to all ``N`` panel variables through the loading matrix.

```math
F_t = c + \sum_{l=1}^p A_l F_{t-l} + B_0 \varepsilon_t
```

where:
- ``F_t`` is the ``q \times 1`` vector of common factors from the GDFM
- ``A_l`` are ``q \times q`` autoregressive coefficient matrices
- ``B_0`` is the ``q \times q`` impact matrix
- ``\varepsilon_t`` are the structural shocks

Panel-wide structural IRFs map factor responses to every observable:

```math
\text{IRF}_i(h, j) = \sum_{k=1}^q \Lambda_{ik} \cdot \left[\Phi_h B_0\right]_{kj}
```

where:
- ``\Lambda_{ik}`` is the time-domain loading of variable ``i`` on factor ``k``
- ``\Phi_h`` is the ``h``-step reduced-form IRF matrix of the factor VAR
- ``j`` indexes the structural shock

Cholesky and sign restrictions are both available.

```@example factor
struct_series = ["INDPRO", "IPFINAL", "IPMANSICS", "CUMFNS", "PAYEMS", "MANEMP", "UNRATE",
                 "DPCERA3M086SBEA", "RETAILx", "HOUST", "PERMIT",
                 "CPIAUCSL", "CPIULFSL", "PCEPI", "WPSFD49207",
                 "FEDFUNDS", "TB3MS", "GS10", "BAA", "S&P 500"]
X20 = X[:, [findfirst(==(v), varnames(fred)) for v in struct_series]]

sdfm20 = estimate_structural_dfm(X20, 2; identification=:cholesky, p=1, H=20)
report(sdfm20)
```

```@example factor
d = fevd(sdfm20, 20)
report(d)
```

The impact matrix is nearly diagonal — 0.92 on the first factor, 0.97 on the second, with a cross-term of ``-0.06`` — so the recursive ordering is close to a relabelling here rather than a substantive restriction. The FEVD confirms it: the first structural shock explains 100% of the first factor's forecast error at impact and 99.8% at ``h = 20``, while never accounting for as much as 0.4% of the second factor's at any horizon. Nearly orthogonal factor innovations are the normal case for GDFM factors, since the frequency-domain eigenvectors are orthonormal by construction.

!!! note "Technical Note"
    Time-domain loadings are obtained by regression, ``\hat\Lambda = \left[(F'F)^{-1}F'X\right]'``, on the **untransformed** panel — not from the spectral loadings. Panel IRFs are therefore in the original units of each series, so responses are not comparable across variables measured on different scales.

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

The first structural shock moves `UNRATE` by ``-0.024`` and `FEDFUNDS` by ``-0.068`` on impact, against ``3.3 \times 10^{-4}`` for `INDPRO` — the scale differences are units, not economics, since `INDPRO` enters as a monthly log difference and `FEDFUNDS` as a level difference in percentage points. Responses decay by roughly a factor of seven per month and are numerically negligible past ``h = 4``: the factor VAR coefficients are small, so with 60 observations of monthly growth rates the extracted factors are close to serially uncorrelated and essentially all of the action is at impact.

| Field | Type | Description |
|-------|------|-------------|
| `gdfm` | `GeneralizedDynamicFactorModel{T}` | Underlying GDFM estimate |
| `factor_var` | `VARModel{T}` | VAR(``p``) fitted on the ``q`` common factors |
| `B0` | `Matrix{T}` | ``q \times q`` impact matrix, ``B_0 = \text{chol}(\Sigma) Q`` |
| `Q` | `Matrix{T}` | ``q \times q`` rotation matrix (identity under `:cholesky`) |
| `identification` | `Symbol` | `:cholesky` or `:sign` |
| `structural_irf` | `Array{T,3}` | ``H \times N \times q`` panel-wide structural IRFs |
| `loadings_td` | `Matrix{T}` | ``N \times q`` time-domain loadings |
| `p_var` | `Int` | VAR lag order on the factors |
| `shock_names` | `Vector{String}` | Labels for the ``q`` structural shocks |

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `identification` | `Symbol` | `:cholesky` | `:cholesky` or `:sign` |
| `p` | `Int` | `1` | VAR lag order on the factors |
| `H` | `Int` | `40` | IRF horizon stored in `structural_irf` |
| `sign_check` | `Function` | `nothing` | Sign restriction predicate, required for `:sign` |
| `max_draws` | `Int` | `1000` | Rotation draws searched under `:sign` |
| `standardize` | `Bool` | `true` | Passed to the internal `estimate_gdfm` call |
| `bandwidth` | `Int` | `0` | Passed to the internal `estimate_gdfm` call |
| `kernel` | `Symbol` | `:bartlett` | Passed to the internal `estimate_gdfm` call |

### Sign Restrictions

The predicate receives the **factor-space** IRF array, dimensioned ``H \times q \times q`` — the responses of the ``q`` factors, not of the ``N`` panel variables — and is indexed `[horizon, factor, shock]`:

```@example factor
# Shock 1 raises factor 1 on impact; shock 2 lowers it
sign_fn(irf_matrix) = irf_matrix[1, 1, 1] > 0 && irf_matrix[1, 1, 2] < 0

Random.seed!(42)   # the rotation search is random and takes no rng keyword
sdfm_sign = estimate_structural_dfm(X20, 2;
    identification=:sign, sign_check=sign_fn, max_draws=1000, H=20)

round.(sdfm_sign.Q, digits=4)
```

The accepted rotation is a plane rotation of roughly ``48^\circ``. The restriction moves factor 1 in opposite directions under the two shocks, yet `INDPRO` falls on impact under both — ``-1.31 \times 10^{-4}`` and ``-5.62 \times 10^{-4}``. Nothing is wrong: a panel response is ``\Lambda`` applied to a *combination* of both factors, not a copy of the restricted one, so a restriction imposed in factor space fixes nothing about any individual observable until the loadings are applied. Sign restrictions identify a set rather than a point, and `estimate_structural_dfm` returns the first rotation that passes, so the seed is part of the result. It takes no `rng` keyword, which is why the example reseeds immediately beforehand.

### Two-Step Estimation

A pre-estimated GDFM can be passed straight in, which avoids re-running the spectral estimation when several identification schemes are compared on the same factors:

```@example factor
gdfm_pre = estimate_gdfm(X20, 2)
sdfm_two = estimate_structural_dfm(gdfm_pre; identification=:cholesky, p=1, H=20)

round.(sdfm_two.B0, digits=4)
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

The first form recomputes the factor IRFs from the stored VAR coefficients and rotation, so horizons beyond the ``H`` used at estimation time are available. The second accepts any factor-space `ImpulseResponse` — from a custom identification, for instance — validates that it has exactly ``q`` variables and ``q`` shocks, and applies the same ``\Lambda`` projection. Both return an `ImpulseResponse` with `ci_type = :none`: the projection carries no interval, because the loadings are treated as known. `StructuralDFM` stores no panel variable names, so rows are labelled `Var 1 … Var N` in column order of the matrix passed to `estimate_structural_dfm` — keep the name vector used to build that matrix alongside the result.

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

## Common Pitfalls

1. **Choosing ``r`` from the scree plot alone.** The elbow is subjective and moves with the standardization choice. Run `ic_criteria(X, r_max)` and check that IC1, IC2, and IC3 agree. When a criterion returns exactly `r_max`, as IC3 does on the 60-observation panel here, its penalty has failed rather than found ten factors — enlarge `r_max` and prefer the interior selection.

2. **Reading loadings as structural parameters.** Loadings are identified only up to an ``r \times r`` rotation, so individual signs and magnitudes are not invariant. Use `blocks=` to label factors by construction, or interpret only rotation-invariant quantities such as `r2` and the variance shares.

3. **Skipping standardization.** With heterogeneous scales — interest rates in percent next to index levels — the high-variance series dominate the principal components entirely. `standardize=true` is the default; override it only when every series is already comparably scaled.

4. **Misspecifying the block structure.** Block-restricted estimation requires exactly ``r`` blocks, no variable in two blocks, indices within ``[1, N]``, and at least 2 variables per block. Each violation raises an `ArgumentError` naming the offending block.

5. **Applying the GDFM to a narrow panel.** The spectral estimator needs ``N`` and ``T`` to grow jointly; with fewer than about 20 series the ``N \times N`` spectral density is too noisy to separate diverging from bounded eigenvalues. Use a two-step or EM DFM instead.

6. **Forecasting from non-stationary factor dynamics.** The forecast recursion assumes the factor companion matrix is stable. Check `is_stationary(dfm)` first; if it fails, either reduce ``p`` or difference the offending series before extraction, since a unit root in a factor makes every horizon's interval meaningless.

7. **Comparing panel IRF magnitudes across series.** `sdfm_panel_irf` and `irf(::StructuralDFM, H)` return responses in each variable's own units, because the time-domain loadings are regressed on the untransformed panel. Rescale by each series' standard deviation before ranking responses.

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

- Forni, M., Giannone, D., Lippi, M., & Reichlin, L. (2009). Opening the Black Box: Structural Factor Models with Large Cross-Sections.
  *Econometric Theory*, 25(5), 1319-1347. [DOI](https://doi.org/10.1017/S026646660809052X)

- Forni, M., Hallin, M., Lippi, M., & Reichlin, L. (2000). The Generalized Dynamic-Factor Model: Identification and Estimation.
  *Review of Economics and Statistics*, 82(4), 540-554. [DOI](https://doi.org/10.1162/003465300559037)

- Forni, M., Hallin, M., Lippi, M., & Reichlin, L. (2005). The Generalized Dynamic Factor Model: One-Sided Estimation and Forecasting.
  *Journal of the American Statistical Association*, 100(471), 830-840. [DOI](https://doi.org/10.1198/016214504000002050)

- Hallin, M., & Liska, R. (2007). Determining the Number of Factors in the General Dynamic Factor Model.
  *Journal of the American Statistical Association*, 102(478), 603-617. [DOI](https://doi.org/10.1198/016214506000001275)

- McCracken, M. W., & Ng, S. (2016). FRED-MD: A Monthly Database for Macroeconomic Research.
  *Journal of Business & Economic Statistics*, 34(4), 574-589. [DOI](https://doi.org/10.1080/07350015.2015.1086655)

- Stock, J. H., & Watson, M. W. (2002a). Forecasting Using Principal Components from a Large Number of Predictors.
  *Journal of the American Statistical Association*, 97(460), 1167-1179. [DOI](https://doi.org/10.1198/016214502388618960)

- Stock, J. H., & Watson, M. W. (2002b). Macroeconomic Forecasting Using Diffusion Indexes.
  *Journal of Business & Economic Statistics*, 20(2), 147-162. [DOI](https://doi.org/10.1198/073500102317351921)
