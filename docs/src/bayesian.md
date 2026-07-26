# [Bayesian VAR](@id bvar_page)

**MacroEconometricModels.jl** provides a complete Bayesian estimation framework for Vector Autoregression models, combining the Minnesota prior (Litterman 1986) with conjugate Normal-Inverse-Wishart posterior inference and data-driven hyperparameter selection via marginal likelihood optimization (Giannone, Lenza & Primiceri 2015).

- **Minnesota Prior**: Shrinkage toward random walk via dummy observations (Doan, Litterman & Sims 1984), with five tunable hyperparameters controlling tightness, lag decay, and cross-variable penalization
- **Hyperparameter Optimization**: Grid search over ``\tau`` or joint ``(\tau, \lambda, \mu)`` optimization using the closed-form marginal likelihood (Giannone, Lenza & Primiceri 2015; Banbura, Giannone & Reichlin 2010)
- **Conjugate Posterior Sampling**: Two samplers --- i.i.d. draws from the analytical Normal-Inverse-Wishart posterior (`:direct`) or a two-block Gibbs sampler (`:gibbs`) with burn-in and thinning
- **Bayesian Structural Analysis**: Posterior distributions over impulse responses, forecast error variance decomposition, and historical decomposition with credible intervals, supporting Cholesky and sign-restriction identification
- **Forecasting**: Multi-step-ahead forecasts with posterior credible intervals, integrating over parameter uncertainty across all posterior draws
- **Large BVAR**: Scalable estimation for high-dimensional systems (20+ variables) where the Minnesota prior prevents overfitting

All results integrate with `report()` for publication-quality output and `plot_result()` for interactive D3.js visualization.

```@setup bvar
using MacroEconometricModels, Random
Random.seed!(42)
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-59:end, :]
post = estimate_bvar(Y, 2; n_draws=100, prior=:minnesota, varnames=["INDPRO", "CPI", "FFR"])
```

## Quick Start

**Recipe 1: Estimate a BVAR with Minnesota prior**

```@example bvar
report(post)
```

**Recipe 2: Optimize hyperparameters via marginal likelihood**

```@example bvar
best = optimize_hyperparameters(Y, 2; grid_size=20)
post_opt = estimate_bvar(Y, 2; n_draws=100, prior=:minnesota, hyper=best,
                         varnames=["INDPRO", "CPI", "FFR"])
report(post_opt)
```

**Recipe 3: Bayesian IRFs with Cholesky identification**

```@example bvar
birf = irf(post, 20; method=:cholesky)
report(birf)
```

```julia
plot_result(birf)
```

```@raw html
<iframe src="../assets/plots/irf_bayesian.html" style="width:100%; height:520px; border:none;"></iframe>
```

**Recipe 4: Bayesian FEVD and historical decomposition**

```@example bvar
bfevd = fevd(post, 20; method=:cholesky)
bhd = historical_decomposition(post; method=:cholesky)
report(bfevd)
report(bhd)
```

**Recipe 5: Multi-step forecasting with credible intervals**

```@example bvar
fc = forecast(post, 12; conf_level=0.95)
report(fc)
```

**Recipe 6: Joint hyperparameter optimization for large systems**

```@example bvar
safe_idx = [i for i in 1:nvars(fred)
            if fred.tcode[i] < 4 || all(x -> isfinite(x) && x > 0, fred.data[:, i])]
fred_safe = fred[:, varnames(fred)[safe_idx]]
X = to_matrix(apply_tcode(fred_safe))
X = X[all.(isfinite, eachrow(X)), 1:min(20, size(X, 2))]

best_full, ml = optimize_hyperparameters_full(X, 4)
post_large = estimate_bvar(X, 4; n_draws=100, prior=:minnesota, hyper=best_full)
report(post_large)
```

---

## Bayesian Framework

The Bayesian approach treats VAR parameters as random variables and updates prior beliefs via Bayes' theorem. For the reduced-form VAR:

```math
Y_t = c + A_1 Y_{t-1} + \cdots + A_p Y_{t-p} + u_t, \quad u_t \sim N(0, \Sigma)
```

where:
- ``Y_t`` is the ``n \times 1`` vector of endogenous variables at time ``t``
- ``c`` is the ``n \times 1`` intercept vector
- ``A_l`` is the ``n \times n`` coefficient matrix at lag ``l``
- ``\Sigma`` is the ``n \times n`` error covariance matrix
- ``p`` is the lag order

the posterior distribution over parameters ``(B, \Sigma)`` satisfies:

```math
p(B, \Sigma \mid Y) \propto p(Y \mid B, \Sigma) \cdot p(B, \Sigma)
```

where:
- ``p(Y \mid B, \Sigma)`` is the Gaussian likelihood
- ``p(B, \Sigma)`` is the prior distribution
- ``B`` is the ``k \times n`` coefficient matrix (``k = 1 + np``, stacking the intercept and all lag coefficients)

The package uses the **Normal-Inverse-Wishart** (NIW) conjugate prior:

```math
\Sigma \sim \text{IW}(\nu_0, S_0), \quad \text{vec}(B) \mid \Sigma \sim N(\text{vec}(B_0), \Sigma \otimes \Omega_0)
```

where:
- ``\nu_0`` is the prior degrees of freedom
- ``S_0`` is the ``n \times n`` prior scale matrix
- ``B_0`` is the ``k \times n`` prior mean for coefficients
- ``\Omega_0`` is the ``k \times k`` prior covariance for coefficient rows

The conjugate structure yields a closed-form posterior of the same NIW family, enabling exact i.i.d. sampling without MCMC convergence concerns.

---

## The Minnesota Prior

The **Minnesota prior** (Litterman 1986; Doan, Litterman & Sims 1984) shrinks VAR coefficients toward a random walk, reflecting the empirical observation that many macroeconomic time series are well-approximated by unit root processes at short horizons. The prior mean sets each variable's own first lag to unity and all other coefficients to zero:

```math
E[A_{1,ii}] = 1, \quad E[A_{1,ij}] = 0 \text{ for } i \neq j, \quad E[A_l] = 0 \text{ for } l > 1
```

The prior variance for coefficient ``(i,j)`` at lag ``l`` controls the degree of shrinkage:

```math
\text{Var}(A_{l,ij}) = \begin{cases}
\dfrac{\tau^2}{l^d} & \text{if } i = j \\[6pt]
\dfrac{\tau^2 \omega^2}{l^d} \cdot \dfrac{\sigma_i^2}{\sigma_j^2} & \text{if } i \neq j
\end{cases}
```

where:
- ``\tau`` is the **overall tightness** parameter controlling shrinkage intensity (lower values produce stronger shrinkage toward the prior)
- ``d`` is the **lag decay** exponent (higher values penalize distant lags more aggressively)
- ``\omega`` controls **cross-variable shrinkage** (values below 1 penalize other variables' lags relative to own lags)
- ``\sigma_i^2`` is the residual variance from a univariate AR(1) for variable ``i``, used to normalize units across variables

### Hyperparameter Interpretation

| Hyperparameter | Field | Default | Effect |
|----------------|-------|---------|--------|
| ``\tau`` | `tau` | `3.0` | Overall shrinkage (lower = tighter prior, closer to random walk) |
| ``d`` | `decay` | `0.5` | Lag decay exponent (higher = faster decay of distant lags) |
| ``\lambda`` | `lambda` | `5.0` | Sum-of-coefficients scaling (controls unit root prior tightness) |
| ``\mu`` | `mu` | `2.0` | Co-persistence scaling (controls common stochastic trend prior) |
| ``\omega`` | `omega` | `2.0` | Covariance scaling (controls prior on error covariance) |

!!! note "Technical Note"
    The Minnesota prior is implemented via **dummy observations** (Theil-Goldberger mixed estimation). Augmenting the data with pseudo-observations and running OLS on the combined system is algebraically equivalent to computing the posterior mean under the NIW conjugate prior. This approach avoids explicit construction of the ``\Sigma \otimes \Omega_0`` Kronecker prior covariance.

```@example bvar
# Define hyperparameters explicitly
hyper = MinnesotaHyperparameters(
    tau = 0.5,      # Moderate overall tightness
    decay = 2.0,    # Quadratic lag decay
    lambda = 1.0,   # Sum-of-coefficients scaling
    mu = 1.0,       # Co-persistence scaling
    omega = 1.0     # Covariance scaling
)

post_hyper = estimate_bvar(Y, 2; n_draws=100, prior=:minnesota, hyper=hyper,
                           varnames=["INDPRO", "CPI", "FFR"])
report(post_hyper)
```

The `tau=0.5` setting provides moderate shrinkage --- coefficient estimates are pulled halfway between the data-driven OLS values and the random walk prior. With `decay=2.0`, the prior variance for lag-``l`` coefficients decays as ``1/l^2``, so distant lags are strongly penalized. Setting `mu=1.0` treats cross-variable and own lags symmetrically; reducing `mu` (e.g., to 0.5) imposes stronger cross-variable shrinkage, reflecting the common finding that own lags carry more predictive power.

### `MinnesotaHyperparameters` Fields

| Field | Type | Description |
|-------|------|-------------|
| `tau` | `T` | Overall tightness (lower = more shrinkage toward random walk prior) |
| `decay` | `T` | Lag decay exponent (higher = faster decay of lag importance) |
| `lambda` | `T` | Sum-of-coefficients scaling (controls unit root belief) |
| `mu` | `T` | Co-persistence scaling (controls common trend belief) |
| `omega` | `T` | Covariance scaling (controls prior on error covariance) |

---

## Hyperparameter Optimization

Rather than setting ``\tau`` subjectively, the marginal likelihood (Giannone, Lenza & Primiceri 2015) provides a data-driven criterion for hyperparameter selection. The marginal likelihood integrates out all model parameters:

```math
p(Y \mid \tau) = \int p(Y \mid B, \Sigma) \, p(B, \Sigma \mid \tau) \, dB \, d\Sigma
```

where:
- ``p(Y \mid B, \Sigma)`` is the Gaussian likelihood
- ``p(B, \Sigma \mid \tau)`` is the NIW prior indexed by hyperparameters

For the NIW prior with dummy observations, the log marginal likelihood has an analytical form:

```math
\log p(Y \mid \tau) = c + \frac{n}{2}\left(\log|X_d'X_d| - \log|X_a'X_a|\right) - \frac{\nu_a}{2}\log|\hat{S}_a| + \frac{\nu_d}{2}\log|\hat{S}_d|
```

where:
- ``c`` is a normalization constant involving multivariate gamma functions
- ``X_d, X_a`` are the dummy-only and augmented (data + dummy) regressor matrices
- ``\hat{S}_a, \hat{S}_d`` are the residual sum-of-squares matrices from OLS on augmented and dummy-only systems
- ``\nu_a = T + \nu_d``, ``\nu_d = T_d - k`` are the posterior and prior degrees of freedom
- ``T_d`` is the number of dummy observations, ``k = 1 + np`` is the number of regressors per equation

### Tau-Only Optimization

The `optimize_hyperparameters` function performs a one-dimensional grid search over ``\tau``, holding all other hyperparameters at their defaults:

```@example bvar
# Optimize tau via marginal likelihood
best_tau = optimize_hyperparameters(Y, 2; grid_size=20, tau_range=(0.01, 10.0))

# Use optimized hyperparameters in estimation
post_tau = estimate_bvar(Y, 2; n_draws=100, prior=:minnesota, hyper=best_tau,
                         varnames=["INDPRO", "CPI", "FFR"])
report(post_tau)
```

The optimal ``\tau`` balances fit and complexity: values near 0.01 produce near-dogmatic shrinkage to the random walk (useful for high-dimensional systems), while values near 1.0 produce minimal shrinkage (approaching OLS). The marginal likelihood automatically penalizes overfitting, so the optimal ``\tau`` increases with sample size as data evidence accumulates.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `grid_size` | `Int` | `20` | Number of grid points for ``\tau`` search |
| `tau_range` | `Tuple{Real,Real}` | `(0.01, 10.0)` | Lower and upper bounds for ``\tau`` grid |

### Joint Optimization (BGR 2010)

Banbura, Giannone & Reichlin (2010) recommend jointly optimizing ``(\tau, \lambda, \mu)`` to maximize the marginal likelihood, especially for large systems where the interaction between overall tightness and cross-variable shrinkage matters:

```math
(\hat{\tau}, \hat{\lambda}, \hat{\mu}) = \arg\max_{\tau, \lambda, \mu} \log p(Y \mid \tau, \lambda, \mu)
```

where:
- ``\hat{\tau}`` is the optimal overall tightness
- ``\hat{\lambda}`` is the optimal sum-of-coefficients scaling
- ``\hat{\mu}`` is the optimal co-persistence scaling

```@example bvar
# Three-dimensional grid search
best_joint, ml_joint = optimize_hyperparameters_full(Y, 2;
    tau_grid    = range(0.1, 5.0, length=10),
    lambda_grid = [1.0, 5.0, 10.0],
    mu_grid     = [1.0, 2.0, 5.0]
)

post_joint = estimate_bvar(Y, 2; n_draws=100, prior=:minnesota, hyper=best_joint,
                           varnames=["INDPRO", "CPI", "FFR"])
report(post_joint)
```

Joint optimization is particularly important for large systems (``n \geq 10``), where the optimal ``\mu`` is often substantially below 1.0 --- imposing strong cross-variable shrinkage while allowing own lags to remain relatively free. For small systems (``n \leq 5``), the simpler tau-only search is usually sufficient.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `tau_grid` | `AbstractRange` | `range(0.1, 5.0, length=10)` | Grid values for ``\tau`` |
| `lambda_grid` | `Vector` | `[1.0, 5.0, 10.0]` | Grid values for ``\lambda`` |
| `mu_grid` | `Vector` | `[1.0, 2.0, 5.0]` | Grid values for ``\mu`` |

---

## Posterior Sampling

The package provides two samplers for drawing from the conjugate NIW posterior. Both produce a `BVARPosterior{T}` object containing coefficient and covariance draws.

### Direct Sampler

The `:direct` sampler (default) draws i.i.d. from the analytical NIW posterior. No burn-in or thinning is needed because each draw is independent:

1. Draw ``\Sigma^{(s)} \sim \text{IW}(\nu_{\text{post}}, S_{\text{post}})`` via Bartlett decomposition
2. Draw ``B^{(s)} \mid \Sigma^{(s)} \sim \text{MN}(B_{\text{post}}, \Omega_{\text{post}}, \Sigma^{(s)})``

### Gibbs Sampler

The `:gibbs` sampler alternates between two conditional draws in a Markov chain:

1. Draw ``B^{(s)} \mid \Sigma^{(s-1)}, Y``
2. Draw ``\Sigma^{(s)} \mid B^{(s)}, Y``

The Gibbs sampler is useful for diagnostics, extensions, or cross-validation against the direct sampler. It supports `burnin` and `thinning` parameters to reduce autocorrelation.

!!! note "Technical Note"
    The Gibbs sampler pre-computes the posterior variance ``\Omega_{\text{post}}`` and its Cholesky factor before the sampling loop, since these depend only on the data and prior (not on the current draw of ``\Sigma``). Workspace buffers are pre-allocated to minimize allocations during the MCMC loop.

```@example bvar
# Direct sampler (i.i.d. draws, fast)
post_direct = estimate_bvar(Y, 2; n_draws=100, sampler=:direct,
                            prior=:minnesota,
                            varnames=["INDPRO", "CPI", "FFR"])

# Gibbs sampler (MCMC, for diagnostics)
post_gibbs = estimate_bvar(Y, 2; n_draws=100, sampler=:gibbs,
                           burnin=500, thin=2, prior=:minnesota,
                           varnames=["INDPRO", "CPI", "FFR"])
report(post_direct)
```

The `:direct` sampler is typically 10--100x faster than Gibbs because it avoids iterative sampling. For a 3-variable VAR(2) with `n_draws=1000`, estimation takes under 1 second. If the posterior summaries from `:direct` and `:gibbs` agree closely, the implementation is validated.

### `estimate_bvar` Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `n_draws` | `Int` | `1000` | Number of posterior draws to retain |
| `sampler` | `Symbol` | `:direct` | Sampling algorithm (`:direct` or `:gibbs`) |
| `burnin` | `Int` | `0` | Burn-in period (`:gibbs` only; defaults to 200 when `sampler=:gibbs`) |
| `thin` | `Int` | `1` | Thinning interval (`:gibbs` only) |
| `prior` | `Symbol` | `:normal` | Prior type (`:normal` for diffuse, `:minnesota` for Minnesota) |
| `hyper` | `MinnesotaHyperparameters` | `nothing` | Minnesota hyperparameters (auto-optimized when `nothing` and `prior=:minnesota`) |
| `varnames` | `Vector{String}` | `nothing` | Variable display names |

### `BVARPosterior{T}` Fields

| Field | Type | Description |
|-------|------|-------------|
| `B_draws` | `Array{T,3}` | Coefficient draws (``\text{n\_draws} \times k \times n``), where ``k = 1 + np`` |
| `Sigma_draws` | `Array{T,3}` | Covariance draws (``\text{n\_draws} \times n \times n``) |
| `n_draws` | `Int` | Number of posterior draws |
| `p` | `Int` | Number of VAR lags |
| `n` | `Int` | Number of variables |
| `data` | `Matrix{T}` | Original ``Y`` matrix (used for residual computation downstream) |
| `prior` | `Symbol` | Prior used (`:normal` or `:minnesota`) |
| `sampler` | `Symbol` | Sampler used (`:direct` or `:gibbs`) |
| `varnames` | `Vector{String}` | Variable names |

---

## Posterior Point Estimates

After estimation, it is often useful to extract a single `VARModel` based on the posterior mean or median. This enables all frequentist tools --- stationarity checks, Granger causality, information criteria --- on the Bayesian point estimate.

```@example bvar
# Extract VARModel with posterior mean parameters
mean_model = posterior_mean_model(post)
report(mean_model)

# Extract VARModel with posterior median parameters
median_model = posterior_median_model(post)

# Standard VAR tools work on the point estimate
stab = is_stationary(mean_model)
irfs_mean = irf(mean_model, 20; method=:cholesky)
nothing # hide
```

The `posterior_mean_model` averages ``B`` and ``\Sigma`` across all posterior draws, providing a point estimate that integrates over parameter uncertainty. The `posterior_median_model` uses the element-wise median, which is more robust to outlier draws. The `BVARPosterior` stores the original data, so residuals are computed automatically for downstream analyses such as `historical_decomposition`.

---

## Bayesian Impulse Response Functions

For each posterior draw ``(B^{(s)}, \Sigma^{(s)})``, the package computes impulse responses from the VMA representation, yielding a full posterior distribution over IRFs. The central tendency (posterior median by default) and credible intervals (16th--84th percentile by default) are reported.

### Cholesky Identification

```@example bvar
birf_chol = irf(post, 20; method=:cholesky)
report(birf_chol)
```

```julia
plot_result(birf_chol)
```

```@raw html
<iframe src="../assets/plots/irf_bayesian.html" style="width:100%; height:520px; border:none;"></iframe>
```

The Cholesky ordering [INDPRO, CPI, FFR] identifies a monetary policy shock as the third orthogonalized innovation. The posterior median IRF at ``h = 0`` for INDPRO is zero by construction (ordered first, so it does not respond on impact). Unlike frequentist bootstrap confidence intervals, Bayesian credible intervals integrate over parameter uncertainty in ``(B, \Sigma)`` across all posterior draws, providing a complete characterization of inference uncertainty.

!!! note "Point Estimate Selection"
    By default, `irf`, `fevd`, and `historical_decomposition` use the **posterior mean** as the central tendency (`point_estimate=:mean`). Pass `point_estimate=:median` to use the posterior median instead. The `.point_estimate` field of the result stores whichever was selected.

### Sign Restrictions

Sign restrictions provide set identification by retaining only rotation matrices ``Q`` that produce economically meaningful impulse responses:

```@example bvar
# Contractionary monetary shock: FFR rises, INDPRO and CPI fall on impact
function check_monetary(irf_array)
    return irf_array[1, 3, 3] > 0 &&   # FFR rises
           irf_array[1, 1, 3] < 0 &&   # INDPRO falls
           irf_array[1, 2, 3] < 0       # CPI falls
end

birf_sign = irf(post, 20; method=:sign, check_func=check_monetary, max_draws=5000)
report(birf_sign)
```

The sign-restricted credible intervals combine both parameter uncertainty (from posterior draws of ``(B, \Sigma)``) and identification uncertainty (from the rotation ``Q``). The sign restrictions ensure a contractionary monetary shock raises the federal funds rate and lowers output and prices on impact, consistent with conventional monetary transmission.

### `irf` Keyword Arguments (BVAR dispatch)

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:cholesky` | Identification method (`:cholesky`, `:sign`, `:narrative`, etc.) |
| `quantiles` | `Vector{Real}` | `[0.16, 0.5, 0.84]` | Quantile levels for credible bands |
| `point_estimate` | `Symbol` | `:mean` | Central tendency (`:mean` or `:median`) |
| `check_func` | `Function` | `nothing` | Sign restriction check function |
| `narrative_check` | `Function` | `nothing` | Narrative restriction check function |
| `threaded` | `Bool` | `false` | Use threaded quantile computation |

### `BayesianImpulseResponse{T}` Fields

| Field | Type | Description |
|-------|------|-------------|
| `quantiles` | `Array{T,4}` | ``(H+1) \times n \times n \times n_q``: posterior quantiles of IRFs |
| `point_estimate` | `Array{T,3}` | ``(H+1) \times n \times n`` posterior point estimate (mean or median) |
| `horizon` | `Int` | Maximum IRF horizon |
| `variables` | `Vector{String}` | Variable names |
| `shocks` | `Vector{String}` | Shock names |
| `quantile_levels` | `Vector{T}` | Quantile levels used |

---

## Bayesian FEVD

The **forecast error variance decomposition** (FEVD) measures the share of each variable's forecast error variance attributable to each structural shock. For each posterior draw, the FEVD is computed from the VMA coefficients, yielding a posterior distribution over variance shares.

```@example bvar
bfevd_sec = fevd(post, 20; method=:cholesky)
report(bfevd_sec)
```

```julia
plot_result(bfevd_sec)
```

```@raw html
<iframe src="../assets/plots/fevd_bayesian.html" style="width:100%; height:520px; border:none;"></iframe>
```

At short horizons, the monetary shock (shock 3) explains a small fraction of INDPRO forecast error variance --- consistent with the Cholesky ordering where INDPRO does not respond on impact. As the horizon increases, the monetary transmission mechanism operates through lagged effects, and the monetary shock's contribution grows. The wide credible intervals at long horizons reflect cumulating parameter uncertainty through the VMA representation.

### `BayesianFEVD{T}` Fields

| Field | Type | Description |
|-------|------|-------------|
| `quantiles` | `Array{T,4}` | ``H \times n \times n \times n_q``: posterior quantiles of FEVD shares |
| `point_estimate` | `Array{T,3}` | ``H \times n \times n`` posterior point estimate FEVD proportions |
| `horizon` | `Int` | Maximum horizon |
| `variables` | `Vector{String}` | Variable names |
| `shocks` | `Vector{String}` | Shock names |
| `quantile_levels` | `Vector{T}` | Quantile levels used |

---

## Bayesian Historical Decomposition

The **historical decomposition** decomposes the actual realization of each variable into contributions from each structural shock and an initial condition component. For each posterior draw, structural shocks are recovered and cumulated through the VMA representation.

```@example bvar
bhd_sec = historical_decomposition(post; method=:cholesky)
report(bhd_sec)
```

```julia
plot_result(bhd_sec)
```

```@raw html
<iframe src="../assets/plots/hd_bayesian.html" style="width:100%; height:520px; border:none;"></iframe>
```

The historical decomposition reveals which structural shocks drove each variable's movements at each point in time. Credible intervals on the shock contributions reflect posterior uncertainty in both the VAR parameters and the structural identification. For the [INDPRO, CPI, FFR] system, the decomposition shows how supply, demand, and monetary policy shocks combine to explain the observed dynamics of output, prices, and the policy rate.

### `BayesianHistoricalDecomposition{T}` Fields

| Field | Type | Description |
|-------|------|-------------|
| `quantiles` | `Array{T,4}` | ``T_{\text{eff}} \times n \times n_{\text{shocks}} \times n_q``: shock contribution quantiles |
| `point_estimate` | `Array{T,3}` | ``T_{\text{eff}} \times n \times n_{\text{shocks}}`` point estimate contributions |
| `initial_quantiles` | `Array{T,3}` | ``T_{\text{eff}} \times n \times n_q``: initial condition quantiles |
| `initial_point_estimate` | `Matrix{T}` | ``T_{\text{eff}} \times n`` initial condition point estimate |
| `actual` | `Matrix{T}` | ``T_{\text{eff}} \times n`` actual observed values |
| `T_eff` | `Int` | Effective sample size |
| `variables` | `Vector{String}` | Variable names |
| `shock_names` | `Vector{String}` | Shock names |
| `method` | `Symbol` | Identification method used |

---

## Forecasting

The BVAR forecast integrates over parameter uncertainty by iterating the VAR recursion forward for each posterior draw. For each draw ``(B^{(s)}, \Sigma^{(s)})``, future shocks are drawn from ``N(0, \Sigma^{(s)})`` and the VAR is simulated forward ``h`` steps. The distribution of forecast paths across draws produces posterior credible intervals.

```@example bvar
fc_sec = forecast(post, 12; conf_level=0.90, point_estimate=:median)
report(fc_sec)
```

```julia
plot_result(fc_sec)
```

```@raw html
<iframe src="../assets/plots/forecast_bvar.html" style="width:100%; height:520px; border:none;"></iframe>
```

The posterior credible intervals widen with the forecast horizon, reflecting both parameter uncertainty (from the posterior distribution of ``(B, \Sigma)``) and shock uncertainty (from the stochastic future innovations). Non-stationary draws are automatically filtered out to prevent explosive forecast paths.

### `forecast` Keyword Arguments (BVAR dispatch)

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `reps` | `Int` | `nothing` | Number of posterior draws to use (default: all) |
| `conf_level` | `Real` | `0.95` | Credible interval level |
| `point_estimate` | `Symbol` | `:mean` | Central tendency (`:mean` or `:median`) |

### `BVARForecast{T}` Fields

| Field | Type | Description |
|-------|------|-------------|
| `forecast` | `Matrix{T}` | ``h \times n`` point forecast (posterior mean or median) |
| `ci_lower` | `Matrix{T}` | ``h \times n`` lower credible interval bound |
| `ci_upper` | `Matrix{T}` | ``h \times n`` upper credible interval bound |
| `horizon` | `Int` | Forecast horizon |
| `conf_level` | `T` | Credible interval level |
| `point_estimate` | `Symbol` | Central tendency used (`:mean` or `:median`) |
| `varnames` | `Vector{String}` | Variable names |

### Conditional Forecasts

`conditional_forecast` runs a Waggoner & Zha (1999) scenario through the posterior. Every draw supplies its own coefficients, structural impact matrix, unconditional path and restriction system, and contributes one conditional shock draw — so the bands carry both parameter and shock uncertainty, unlike the `VARModel` dispatch, which conditions on the point estimate. Non-stationary draws are skipped exactly as in `forecast`.

The mechanics of the restriction system, hard versus soft conditions, and the identification-invariance result are covered on the [VAR](@ref var_page) page.

```@example bvar
# Hold the policy rate at 2% for the first four quarters
scenario = Dict(("FFR", h) => 2.0 for h in 1:4)
cfc = conditional_forecast(post, scenario, 12)
report(cfc)
```

The conditioned rows are pinned to 2.0 with a degenerate band — a hard condition holds in every posterior draw — while the other variables show the scenario's spillovers with credible intervals that reflect the full posterior.

```julia
plot_result(cfc)
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `Q` | `AbstractMatrix` | `nothing` | Rotation matrix; `nothing` is Cholesky. Affects only the reported `shocks` |
| `reps` | `Int` | `nothing` | Posterior draws to use (default: all) |
| `conf_level` | `Real` | `0.95` | Credible-band coverage |
| `point_estimate` | `Symbol` | `:mean` | Central tendency across draws (`:mean` or `:median`) |
| `rng` | `AbstractRNG` | `default_rng()` | Random number generator |

The return value is a `ConditionalForecast{T}`; its fields are documented on the [VAR](@ref var_page) page.

---

## Time-Varying Parameters and Stochastic Volatility

A constant-coefficient, homoskedastic VAR cannot represent the Great Moderation, drifting inflation persistence, or a monetary transmission mechanism that changes across regimes. `estimate_tvpvar` implements Primiceri (2005), in which the coefficients, the contemporaneous structural matrix, and the shock volatilities all drift:

```math
y_t = X_t' B_t + A_t^{-1}\Sigma_t\varepsilon_t,\qquad
B_t = B_{t-1} + \nu_t,\quad
a_t = a_{t-1} + \zeta_t,\quad
\log\sigma^2_{i,t} = \log\sigma^2_{i,t-1} + \eta_{i,t}
```

where:
- ``X_t' = I_n \otimes [1, y_{t-1}', \ldots, y_{t-p}']``, so ``B_t`` stacks every equation's coefficients
- ``A_t`` is lower triangular with a unit diagonal; its free elements are the contemporaneous structural coefficients, which impose a recursive identification that itself drifts
- ``\Sigma_t = \mathrm{diag}(\sigma_{1,t},\ldots,\sigma_{n,t})``
- ``\nu_t\sim N(0,Q)``, ``\zeta_t\sim N(0,S)`` block diagonal by equation, ``\eta_t\sim N(0,W)`` diagonal

!!! note "Technical Note"
    Each Gibbs sweep draws, in the Del Negro & Primiceri (2015) **corrected** order: ``B_{1:T}`` by Carter-Kohn with per-period observation covariance ``A_t^{-1}\Sigma_t^2A_t^{-1\prime}``; ``a_{1:T}`` equation by equation on the VAR residuals; the Kim-Shephard-Chib mixture indicators ``s_t`` given the **current** volatilities, then ``\log\sigma^2_{1:T}`` given ``s``; and finally ``Q``, ``S``, ``W`` from conjugate inverse-Wishart / inverse-gamma updates. The corrigendum's point is that ``s`` must be drawn *before* the volatility update, and that the coefficient blocks are drawn without conditioning on ``s``.

Priors are calibrated on a training sample in Primiceri's fashion: ``B_0\sim N(\hat B_{OLS}, 4V(\hat B_{OLS}))``, ``Q\sim IW(k_Q^2\tau V(\hat B_{OLS}), \tau)``, and analogously for ``A_0``, ``S`` and ``W``. The constants ``k_Q = 0.01``, ``k_S = 0.1``, ``k_W = 0.01`` control how much drift the prior permits.

```@example bvar
# Drifting coefficients and drifting volatilities
tvp = estimate_tvpvar(Y, 1; n_draws=100, n_burn=100,
                      varnames=["INDPRO", "CPI", "FFR"])
report(tvp)
```

The report shows the posterior mean volatility path at several dates — the object that carries most of the economic content in this class of models. `volatility_path` returns the full path with credible bands (note that the stored state is the log **variance**, so the standard deviation is `exp(h/2)`; `volatility_path` applies that transform):

```@example bvar
vol, vol_bands = volatility_path(tvp)
size(vol), size(vol_bands)
```

### Time-Varying Impulse Responses

`irf(tvp, H; t)` computes the impulse response **at a chosen date**, integrating over posterior draws. Both the propagation (from ``B_t``) and the impact matrix (``A_t^{-1}\Sigma_t``) are taken at that date, so comparing two dates is how time variation in transmission is read off:

```@example bvar
early = irf(tvp, 12; t=5, n_draws=50)
late  = irf(tvp, 12; t=tvp.T_eff, n_draws=50)
round.([early.point_estimate[1, :, 3]  late.point_estimate[1, :, 3]], digits=4)
```

```julia
plot_result(late)
```

### The Constant-Coefficient Special Case

`tvp=false` fixes ``B_t`` and ``a_t`` and leaves only the volatilities drifting — the Cogley & Sargent (2005) SV-BVAR. On homoskedastic data its coefficient posterior mean reproduces the conjugate BVAR's on the same estimation window:

```@example bvar
sv_only = estimate_tvpvar(Y, 1; tvp=false, n_draws=100, n_burn=100,
                          varnames=["INDPRO", "CPI", "FFR"])
sv_only.tvp, sv_only.sv
```

Symmetrically, `sv=false` freezes the volatilities at their training-sample values and lets only the coefficients drift.

### `estimate_tvpvar` Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `tvp` | `Bool` | `true` | Let ``B_t`` and ``a_t`` drift; `false` gives the Cogley-Sargent SV-BVAR |
| `sv` | `Bool` | `true` | Let the volatilities drift |
| `n_draws` | `Int` | `2000` | Retained posterior draws |
| `n_burn` | `Int` | `1000` | Burn-in sweeps discarded |
| `thin` | `Int` | `1` | Keep every `thin`-th post-burn-in sweep |
| `n_train` | `Int` | `0` | Training-sample length; `0` uses `max(4p+n+2, T÷4)` |
| `k_Q`, `k_S`, `k_W` | `Real` | `0.01`, `0.1`, `0.01` | Primiceri's prior-drift constants |
| `varnames` | `Vector{String}` | `y1, y2, …` | Variable names |
| `rng` | `AbstractRNG` | `default_rng()` | Random number generator |

### `TVPVARPosterior{T}` Fields

| Field | Type | Description |
|-------|------|-------------|
| `B_draws` | `Array{T,3}` | ``n_{draws}\times T_{eff}\times k`` drifting VAR coefficients, ``k = n(1+np)`` |
| `A_draws` | `Array{T,3}` | ``n_{draws}\times T_{eff}\times n_a`` free elements of ``A_t``, ``n_a = n(n-1)/2`` |
| `H_draws` | `Array{T,3}` | ``n_{draws}\times T_{eff}\times n`` log-**variances** ``\log\sigma^2_{i,t}`` |
| `Q_draws` | `Array{T,3}` | Coefficient random-walk covariance draws |
| `S_draws` | `Array{T,3}` | Block-diagonal ``A_t`` random-walk covariance draws |
| `W_draws` | `Matrix{T}` | Log-volatility random-walk variances |
| `T_eff` | `Int` | Effective sample after the training block |
| `n_train` | `Int` | Training-sample length used for the priors |
| `tvp` / `sv` | `Bool` | Which blocks were allowed to drift |

---

## Mixed-Frequency VAR

Macro data arrive at different frequencies: GDP quarterly, employment and prices monthly. `estimate_mfvar` implements Schorfheide & Song (2015), which puts the VAR entirely at the **high** frequency and treats each low-frequency series as a latent high-frequency process observed only at reference dates through a temporal-aggregation identity:

```math
y^{lo}_{i,t} = \sum_{j} w_j\, z_{i,t-j+1}
```

where:
- ``z_{i,t}`` is the latent high-frequency value of series ``i``
- ``w`` is the aggregation filter: ``[1]`` for a `:stock` (end-of-period level), ``\mathbf{1}_m`` for a `:flow` sum, ``\mathbf{1}_m/m`` for an `:average`, and the Mariano-Murasawa triangular filter ``[1,2,\ldots,m,\ldots,2,1]/m`` for a `:growth` rate
- ``m`` is the frequency ratio (3 for monthly/quarterly)

Input is a high-frequency panel with `NaN` wherever a series is not observed — a quarterly series in a monthly panel carries a value every third row.

!!! note "Technical Note"
    A two-block Gibbs sampler alternates between the conjugate NIW draw of ``(B, \Sigma)`` given the completed path — the same draw the conjugate BVAR uses, Minnesota dummies included — and a draw of the latent path given ``(B, \Sigma)``. The path draw uses the **Durbin-Koopman (2002) simulation smoother** rather than a backward sampler: the companion state noise is singular, so a Kim-Nelson backward step conditions only on ``z_{t+1}``, but an aggregation row links ``z_t, z_{t-1}, \ldots`` across several lag blocks and lags redrawn at successive backward steps are then mutually inconsistent with that constraint. Durbin-Koopman simulates an unconditional path ``s^+``, smooths ``y - y^+``, and sets ``\tilde s = \hat s(y - y^+) + s^+``; because the aggregation is noiseless, applying the observation map returns ``y`` **exactly** at every reference date.

```@example bvar
# Build a monthly panel in which the third series is observed only quarterly, as a
# within-quarter sum of its latent monthly values.
Y_m = copy(Y)
mf_data = copy(Y_m)
mf_data[:, 3] .= NaN
for t in 3:size(Y_m, 1)
    t % 3 == 0 || continue
    mf_data[t, 3] = sum(Y_m[t-j+1, 3] for j in 1:3)
end

mf = estimate_mfvar(mf_data, 1; low_freq=[3], aggregation=:flow, freq_ratio=3,
                    n_draws=100, n_burn=100,
                    varnames=["INDPRO", "CPI", "FFR"])
report(mf)
```

The report shows the interpolated high-frequency path of each low-frequency series with credible bands — the object the model exists to produce. `latent_path` returns it in full:

```@example bvar
mu, bands = latent_path(mf)
# The aggregation identity holds at every reference date, by construction
maximum(abs(sum(mu[t-j+1, 3] for j in 1:3) - mf_data[t, 3])
        for t in 3:size(mf_data, 1) if !isnan(mf_data[t, 3]))
```

Because the parameter draws are ordinary VAR draws, the existing analysis dispatches apply at the high frequency:

```@example bvar
fc_mf = forecast(mf, 6)
report(fc_mf)
```

With `low_freq` empty nothing is latent and the sampler reduces to the conjugate BVAR. See also the [Nowcasting](nowcast.md) pages, which solve the same ragged-edge problem with a dynamic factor model instead.

### `estimate_mfvar` Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `low_freq` | `Vector{Int}` | `Int[]` | Column indices observed at the low frequency |
| `freq_ratio` | `Int` | `3` | High-frequency periods per low-frequency period |
| `aggregation` | `Symbol` or `Vector{Symbol}` | `:growth` | `:growth`, `:flow`, `:average`, `:stock`; one rule for all low-frequency series or one per series |
| `n_draws` | `Int` | `1000` | Retained posterior draws |
| `n_burn` | `Int` | `500` | Burn-in sweeps |
| `prior` | `Symbol` | `:minnesota` | `:minnesota` (dummy observations) or `:diffuse` |
| `hyper` | `MinnesotaHyperparameters` | `nothing` | Fixed hyperparameters; `nothing` optimizes once on the initial path |
| `varnames` | `Vector{String}` | `y1, y2, …` | Variable names |
| `rng` | `AbstractRNG` | `default_rng()` | Random number generator |

### `MFVARPosterior{T}` Fields

| Field | Type | Description |
|-------|------|-------------|
| `B_draws` | `Array{T,3}` | ``n_{draws}\times k\times n`` VAR coefficients, ``k = 1+np`` |
| `Sigma_draws` | `Array{T,3}` | ``n_{draws}\times n\times n`` innovation covariances |
| `Z_draws` | `Array{T,3}` | ``n_{draws}\times T_{hf}\times n`` latent high-frequency paths |
| `data` | `Matrix{T}` | The input panel, `NaN` where unobserved |
| `low_freq` | `Vector{Int}` | Low-frequency column indices |
| `freq_ratio` | `Int` | High-frequency periods per low-frequency period |
| `aggregation` | `Vector{Symbol}` | Aggregation rule per low-frequency series |

---

## Large BVAR

For high-dimensional systems (20+ variables), the number of VAR parameters ``n^2 p + n`` grows quadratically with the number of variables, quickly exceeding the sample size. The Minnesota prior prevents overfitting by shrinking coefficient estimates, making large-scale Bayesian VAR estimation feasible.

Banbura, Giannone & Reichlin (2010) show that BVAR with optimized shrinkage outperforms both unrestricted VAR and small-scale models for macroeconomic forecasting. The key insight is that stronger shrinkage is needed as the system dimension grows:

```@example bvar
# Select variables with safe transformations
safe_idx_lg = [i for i in 1:nvars(fred)
               if fred.tcode[i] < 4 || all(x -> isfinite(x) && x > 0, fred.data[:, i])]
fred_safe_lg = fred[:, varnames(fred)[safe_idx_lg]]
X_lg = to_matrix(apply_tcode(fred_safe_lg))
X_lg = X_lg[all.(isfinite, eachrow(X_lg)), 1:min(20, size(X_lg, 2))]

# Tighter prior for large systems
hyper_large = MinnesotaHyperparameters(
    tau = 0.1,      # Strong overall shrinkage
    decay = 2.0,    # Quadratic lag decay
    lambda = 1.0,   # Sum-of-coefficients prior
    mu = 0.5,       # Penalize cross-variable coefficients
    omega = 1.0     # Covariance scaling
)

post_lg = estimate_bvar(X_lg, 4; n_draws=100, prior=:minnesota, hyper=hyper_large)
report(post_lg)
```

For 20 variables at 4 lags, the VAR has ``20^2 \times 4 + 20 = 1620`` parameters per equation. With a typical monthly sample of 600 observations, OLS is ill-conditioned. The Minnesota prior with `tau=0.1` and `mu=0.5` regularizes the system by imposing strong cross-variable shrinkage while allowing own lags to retain flexibility. The `optimize_hyperparameters_full` function automates the selection of ``(\tau, \lambda, \mu)`` for large systems.

---

## Complete Example

This example demonstrates the full BVAR workflow: hyperparameter optimization, estimation, structural analysis, and forecasting using FRED-MD data.

```@example bvar
# Step 1: Optimize hyperparameters via marginal likelihood
best_ce = optimize_hyperparameters(Y, 2; grid_size=20)
report(best_ce)

# Step 2: Estimate BVAR with optimized Minnesota prior
post_ce = estimate_bvar(Y, 2; n_draws=100, prior=:minnesota, hyper=best_ce,
                        varnames=["INDPRO", "CPI", "FFR"])
report(post_ce)

# Step 3: Bayesian IRFs — response to monetary policy shock
birf_ce = irf(post_ce, 20; method=:cholesky)
report(birf_ce)

# Step 4: FEVD — variance decomposition with credible bands
bfevd_ce = fevd(post_ce, 20; method=:cholesky)
report(bfevd_ce)

# Step 5: Historical decomposition
bhd_ce = historical_decomposition(post_ce; method=:cholesky)
report(bhd_ce)

# Step 6: 12-step-ahead forecast
fc_ce = forecast(post_ce, 12; conf_level=0.95)
report(fc_ce)

# Step 7: Extract posterior mean VARModel for stationarity check
mean_model_ce = posterior_mean_model(post_ce)
stab_ce = is_stationary(mean_model_ce)
```

This workflow demonstrates the complete Bayesian pipeline: hyperparameter optimization selects the optimal shrinkage ``\tau`` via marginal likelihood, the conjugate NIW sampler produces 100 posterior draws, and the structural analysis functions compute IRFs, FEVD, and historical decomposition with credible intervals. The Cholesky ordering [INDPRO, CPI, FFR] identifies a monetary policy shock as the third innovation. The forecast integrates over the full posterior distribution of ``(B, \Sigma)``, providing credible intervals that account for both parameter and shock uncertainty. The posterior mean model confirms stationarity of the system.

---

## Common Pitfalls

1. **Too few posterior draws**: With `n_draws=100`, credible intervals are noisy and quantile estimates are unreliable. Use at least `n_draws=1000` for stable inference. For sign restrictions, which discard non-conforming draws, increase to `n_draws=5000` or more.

2. **Prior sensitivity with diffuse prior**: Setting `prior=:normal` (the default) uses a diffuse NIW prior that provides minimal regularization. For systems with more than 5 variables, switch to `prior=:minnesota` to avoid overfitting. The diffuse prior is appropriate only for small, well-identified systems with ample data.

3. **Minnesota prior assumes random walk**: The Minnesota prior centers on a random walk --- each variable's own first lag is 1, all others are 0. For stationary variables (e.g., interest rates, unemployment), the prior mean is inappropriate. Consider demeaning or detrending before estimation, or use a lower `tau` to let the data dominate.

4. **Hyperparameter optimization convergence**: The `optimize_hyperparameters` function uses a discrete grid search, so the result depends on `grid_size` and `tau_range`. If the optimal ``\tau`` is at a grid boundary, widen `tau_range`. Increase `grid_size` for finer resolution.

5. **Non-stationary posterior draws**: The forecast function automatically filters out non-stationary draws (those with companion matrix eigenvalues at or above unity). If more than half the draws are non-stationary, estimation raises a warning. This typically indicates the prior is too diffuse --- increase shrinkage by lowering `tau`.

6. **Gibbs sampler autocorrelation**: The `:gibbs` sampler produces correlated draws. Without thinning, effective sample size is smaller than `n_draws`. Use `thin=5` or `thin=10` and increase `burnin` to at least 500 for reliable posterior summaries. The `:direct` sampler avoids this issue entirely.

---

## References

- Banbura, M., Giannone, D., & Reichlin, L. (2010). Large Bayesian Vector Auto Regressions.
  *Journal of Applied Econometrics*, 25(1), 71-92. [DOI](https://doi.org/10.1002/jae.1137)

- Carriero, A., Clark, T. E., & Marcellino, M. (2015). Bayesian VARs: Specification Choices and Forecast Accuracy.
  *Journal of Applied Econometrics*, 30(1), 46-73. [DOI](https://doi.org/10.1002/jae.2315)

- Doan, T., Litterman, R., & Sims, C. (1984). Forecasting and Conditional Projection Using Realistic Prior Distributions.
  *Econometric Reviews*, 3(1), 1-100. [DOI](https://doi.org/10.1080/07474938408800053)

- Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior Selection for Vector Autoregressions.
  *Review of Economics and Statistics*, 97(2), 436-451. [DOI](https://doi.org/10.1162/REST_a_00483)

- Kadiyala, K. R., & Karlsson, S. (1997). Numerical Methods for Estimation and Inference in Bayesian VAR-Models.
  *Journal of Applied Econometrics*, 12(2), 99-132. [DOI](https://doi.org/10.1002/(SICI)1099-1255(199703)12:2<99::AID-JAE429>3.0.CO;2-A)

- Litterman, R. B. (1986). Forecasting with Bayesian Vector Autoregressions --- Five Years of Experience.
  *Journal of Business & Economic Statistics*, 4(1), 25-38. [DOI](https://doi.org/10.1080/07350015.1986.10509491)

- Waggoner, D. F., & Zha, T. (1999). Conditional Forecasts in Dynamic Multivariate Models.
  *Review of Economics and Statistics*, 81(4), 639-651. [DOI](https://doi.org/10.1162/003465399558508)

- Primiceri, G. E. (2005). Time Varying Structural Vector Autoregressions and Monetary Policy.
  *Review of Economic Studies*, 72(3), 821-852. [DOI](https://doi.org/10.1111/j.1467-937X.2005.00353.x)

- Del Negro, M., & Primiceri, G. E. (2015). Time Varying Structural Vector Autoregressions and Monetary Policy: A Corrigendum.
  *Review of Economic Studies*, 82(4), 1342-1345. [DOI](https://doi.org/10.1093/restud/rdv024)

- Cogley, T., & Sargent, T. J. (2005). Drifts and Volatilities: Monetary Policies and Outcomes in the Post WWII US.
  *Review of Economic Dynamics*, 8(2), 262-302. [DOI](https://doi.org/10.1016/j.red.2004.10.009)

- Kim, S., Shephard, N., & Chib, S. (1998). Stochastic Volatility: Likelihood Inference and Comparison with ARCH Models.
  *Review of Economic Studies*, 65(3), 361-393. [DOI](https://doi.org/10.1111/1467-937X.00050)

- Schorfheide, F., & Song, D. (2015). Real-Time Forecasting with a Mixed-Frequency VAR.
  *Journal of Business & Economic Statistics*, 33(3), 366-380. [DOI](https://doi.org/10.1080/07350015.2014.954707)

- Mariano, R. S., & Murasawa, Y. (2003). A New Coincident Index of Business Cycles Based on Monthly and Quarterly Series.
  *Journal of Applied Econometrics*, 18(4), 427-443. [DOI](https://doi.org/10.1002/jae.695)

- Durbin, J., & Koopman, S. J. (2002). A Simple and Efficient Simulation Smoother for State Space Time Series Analysis.
  *Biometrika*, 89(3), 603-616. [DOI](https://doi.org/10.1093/biomet/89.3.603)
