# [Bayesian VAR](@id bvar_page)

**MacroEconometricModels.jl** provides a complete Bayesian estimation framework for Vector Autoregression models, combining the Minnesota prior (Litterman 1986) with conjugate Normal-Inverse-Wishart posterior inference and data-driven hyperparameter selection via marginal likelihood optimization (Giannone, Lenza & Primiceri 2015).

- **Minnesota Prior**: Shrinkage toward random walk via dummy observations (Doan, Litterman & Sims 1984), with five tunable hyperparameters controlling tightness, lag decay, and cross-variable penalization
- **Hyperparameter Optimization**: Grid search over ``\tau`` or joint ``(\tau, \lambda, \mu)`` optimization using the closed-form marginal likelihood (Giannone, Lenza & Primiceri 2015; Banbura, Giannone & Reichlin 2010)
- **Conjugate Posterior Sampling**: Two samplers --- i.i.d. draws from the analytical Normal-Inverse-Wishart posterior (`:direct`) or a two-block Gibbs sampler (`:gibbs`) with burn-in and thinning
- **Bayesian Structural Analysis**: Posterior distributions over impulse responses, forecast error variance decomposition, and historical decomposition with credible intervals, supporting Cholesky and sign-restriction identification
- **Forecasting**: Multi-step-ahead forecasts with posterior credible intervals, integrating over parameter uncertainty across all posterior draws
- **Large BVAR**: Scalable estimation for high-dimensional systems (20+ variables) where the Minnesota prior prevents overfitting

All results integrate with `report()` for publication-quality output and `plot_result()` for interactive D3.js visualization.

## Quick Start

**Recipe 1: Estimate a BVAR with Minnesota prior**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Load FRED-MD: standard monetary VAR (slow-to-fast ordering)
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

# Estimate BVAR(2) with Minnesota prior and automatic tau optimization
post = estimate_bvar(Y, 2; n_draws=1000, prior=:minnesota,
                     varnames=["INDPRO", "CPI", "FFR"])
report(post)
```

**Recipe 2: Optimize hyperparameters via marginal likelihood**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

best = optimize_hyperparameters(Y, 2; grid_size=20)
post = estimate_bvar(Y, 2; n_draws=1000, prior=:minnesota, hyper=best,
                     varnames=["INDPRO", "CPI", "FFR"])
report(post)
```

**Recipe 3: Bayesian IRFs with Cholesky identification**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

post = estimate_bvar(Y, 2; n_draws=1000, prior=:minnesota,
                     varnames=["INDPRO", "CPI", "FFR"])
birf = irf(post, 20; method=:cholesky)
plot_result(birf)
```

**Recipe 4: Bayesian FEVD and historical decomposition**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

post = estimate_bvar(Y, 2; n_draws=1000, prior=:minnesota,
                     varnames=["INDPRO", "CPI", "FFR"])
bfevd = fevd(post, 20; method=:cholesky)
bhd = historical_decomposition(post; method=:cholesky)
report(bfevd)
```

**Recipe 5: Multi-step forecasting with credible intervals**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

post = estimate_bvar(Y, 2; n_draws=1000, prior=:minnesota,
                     varnames=["INDPRO", "CPI", "FFR"])
fc = forecast(post, 12; conf_level=0.95)
report(fc)
```

**Recipe 6: Joint hyperparameter optimization for large systems**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
safe_idx = [i for i in 1:nvars(fred)
            if fred.tcode[i] < 4 || all(x -> isfinite(x) && x > 0, fred.data[:, i])]
fred_safe = fred[:, varnames(fred)[safe_idx]]
X = to_matrix(apply_tcode(fred_safe))
X = X[all.(isfinite, eachrow(X)), 1:min(20, size(X, 2))]

best, ml = optimize_hyperparameters_full(X, 4)
post = estimate_bvar(X, 4; n_draws=1000, prior=:minnesota, hyper=best)
report(post)
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

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Load FRED-MD monetary policy variables
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

# Define hyperparameters explicitly
hyper = MinnesotaHyperparameters(
    tau = 0.5,      # Moderate overall tightness
    decay = 2.0,    # Quadratic lag decay
    lambda = 1.0,   # Sum-of-coefficients scaling
    mu = 1.0,       # Co-persistence scaling
    omega = 1.0     # Covariance scaling
)

post = estimate_bvar(Y, 2; n_draws=1000, prior=:minnesota, hyper=hyper,
                     varnames=["INDPRO", "CPI", "FFR"])
report(post)
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

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

# Optimize tau via marginal likelihood
best = optimize_hyperparameters(Y, 2; grid_size=20, tau_range=(0.01, 10.0))

# Use optimized hyperparameters in estimation
post = estimate_bvar(Y, 2; n_draws=1000, prior=:minnesota, hyper=best,
                     varnames=["INDPRO", "CPI", "FFR"])
report(post)
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

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

# Three-dimensional grid search
best, ml = optimize_hyperparameters_full(Y, 2;
    tau_grid    = range(0.1, 5.0, length=10),
    lambda_grid = [1.0, 5.0, 10.0],
    mu_grid     = [1.0, 2.0, 5.0]
)

post = estimate_bvar(Y, 2; n_draws=1000, prior=:minnesota, hyper=best,
                     varnames=["INDPRO", "CPI", "FFR"])
report(post)
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

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

# Direct sampler (i.i.d. draws, fast)
post_direct = estimate_bvar(Y, 2; n_draws=1000, sampler=:direct,
                            prior=:minnesota,
                            varnames=["INDPRO", "CPI", "FFR"])

# Gibbs sampler (MCMC, for diagnostics)
post_gibbs = estimate_bvar(Y, 2; n_draws=1000, sampler=:gibbs,
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

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

post = estimate_bvar(Y, 2; n_draws=1000, prior=:minnesota,
                     varnames=["INDPRO", "CPI", "FFR"])

# Extract VARModel with posterior mean parameters
mean_model = posterior_mean_model(post)
report(mean_model)

# Extract VARModel with posterior median parameters
median_model = posterior_median_model(post)

# Standard VAR tools work on the point estimate
stab = is_stationary(mean_model)
irfs_mean = irf(mean_model, 20; method=:cholesky)
```

The `posterior_mean_model` averages ``B`` and ``\Sigma`` across all posterior draws, providing a point estimate that integrates over parameter uncertainty. The `posterior_median_model` uses the element-wise median, which is more robust to outlier draws. The `BVARPosterior` stores the original data, so residuals are computed automatically for downstream analyses such as `historical_decomposition`.

---

## Bayesian Impulse Response Functions

For each posterior draw ``(B^{(s)}, \Sigma^{(s)})``, the package computes impulse responses from the VMA representation, yielding a full posterior distribution over IRFs. The central tendency (posterior median by default) and credible intervals (16th--84th percentile by default) are reported.

### Cholesky Identification

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

post = estimate_bvar(Y, 2; n_draws=1000, prior=:minnesota,
                     varnames=["INDPRO", "CPI", "FFR"])
birf = irf(post, 20; method=:cholesky)
report(birf)
plot_result(birf)
```

```@raw html
<iframe src="../assets/plots/irf_bayesian.html" style="width:100%; height:520px; border:none;"></iframe>
```

The Cholesky ordering [INDPRO, CPI, FFR] identifies a monetary policy shock as the third orthogonalized innovation. The posterior median IRF at ``h = 0`` for INDPRO is zero by construction (ordered first, so it does not respond on impact). Unlike frequentist bootstrap confidence intervals, Bayesian credible intervals integrate over parameter uncertainty in ``(B, \Sigma)`` across all posterior draws, providing a complete characterization of inference uncertainty.

!!! note "Point Estimate Selection"
    By default, `irf`, `fevd`, and `historical_decomposition` use the **posterior mean** as the central tendency (`point_estimate=:mean`). Pass `point_estimate=:median` to use the posterior median instead. The `.point_estimate` field of the result stores whichever was selected.

### Sign Restrictions

Sign restrictions provide set identification by retaining only rotation matrices ``Q`` that produce economically meaningful impulse responses:

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

post = estimate_bvar(Y, 2; n_draws=1000, prior=:minnesota,
                     varnames=["INDPRO", "CPI", "FFR"])

# Contractionary monetary shock: FFR rises, INDPRO and CPI fall on impact
function check_monetary(irf_array)
    return irf_array[1, 3, 3] > 0 &&   # FFR rises
           irf_array[1, 1, 3] < 0 &&   # INDPRO falls
           irf_array[1, 2, 3] < 0       # CPI falls
end

birf_sign = irf(post, 20; method=:sign, check_func=check_monetary)
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

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

post = estimate_bvar(Y, 2; n_draws=1000, prior=:minnesota,
                     varnames=["INDPRO", "CPI", "FFR"])
bfevd = fevd(post, 20; method=:cholesky)
report(bfevd)
plot_result(bfevd)
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

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

post = estimate_bvar(Y, 2; n_draws=1000, prior=:minnesota,
                     varnames=["INDPRO", "CPI", "FFR"])
bhd = historical_decomposition(post; method=:cholesky)
report(bhd)
plot_result(bhd)
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

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

post = estimate_bvar(Y, 2; n_draws=1000, prior=:minnesota,
                     varnames=["INDPRO", "CPI", "FFR"])
fc = forecast(post, 12; conf_level=0.90, point_estimate=:median)
report(fc)
plot_result(fc)
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

---

## Large BVAR

For high-dimensional systems (20+ variables), the number of VAR parameters ``n^2 p + n`` grows quadratically with the number of variables, quickly exceeding the sample size. The Minnesota prior prevents overfitting by shrinking coefficient estimates, making large-scale Bayesian VAR estimation feasible.

Banbura, Giannone & Reichlin (2010) show that BVAR with optimized shrinkage outperforms both unrestricted VAR and small-scale models for macroeconomic forecasting. The key insight is that stronger shrinkage is needed as the system dimension grows:

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Load full FRED-MD dataset (100+ variables)
fred = load_example(:fred_md)

# Select variables with safe transformations
safe_idx = [i for i in 1:nvars(fred)
            if fred.tcode[i] < 4 || all(x -> isfinite(x) && x > 0, fred.data[:, i])]
fred_safe = fred[:, varnames(fred)[safe_idx]]
X = to_matrix(apply_tcode(fred_safe))
X = X[all.(isfinite, eachrow(X)), 1:min(20, size(X, 2))]

# Tighter prior for large systems
hyper_large = MinnesotaHyperparameters(
    tau = 0.1,      # Strong overall shrinkage
    decay = 2.0,    # Quadratic lag decay
    lambda = 1.0,   # Sum-of-coefficients prior
    mu = 0.5,       # Penalize cross-variable coefficients
    omega = 1.0     # Covariance scaling
)

post = estimate_bvar(X, 4; n_draws=1000, prior=:minnesota, hyper=hyper_large)
report(post)
```

For 20 variables at 4 lags, the VAR has ``20^2 \times 4 + 20 = 1620`` parameters per equation. With a typical monthly sample of 600 observations, OLS is ill-conditioned. The Minnesota prior with `tau=0.1` and `mu=0.5` regularizes the system by imposing strong cross-variable shrinkage while allowing own lags to retain flexibility. The `optimize_hyperparameters_full` function automates the selection of ``(\tau, \lambda, \mu)`` for large systems.

---

## Complete Example

This example demonstrates the full BVAR workflow: hyperparameter optimization, estimation, structural analysis, and forecasting using FRED-MD data.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Load FRED-MD: industrial production, CPI, federal funds rate
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
p = 2

# Step 1: Optimize hyperparameters via marginal likelihood
best = optimize_hyperparameters(Y, p; grid_size=20)
report(best)

# Step 2: Estimate BVAR with optimized Minnesota prior
post = estimate_bvar(Y, p; n_draws=1000, prior=:minnesota, hyper=best,
                     varnames=["INDPRO", "CPI", "FFR"])
report(post)

# Step 3: Bayesian IRFs — response to monetary policy shock
birf = irf(post, 20; method=:cholesky)
report(birf)

# Step 4: FEVD — variance decomposition with credible bands
bfevd = fevd(post, 20; method=:cholesky)
report(bfevd)

# Step 5: Historical decomposition
bhd = historical_decomposition(post; method=:cholesky)
report(bhd)

# Step 6: 12-step-ahead forecast
fc = forecast(post, 12; conf_level=0.95)
report(fc)

# Step 7: Extract posterior mean VARModel for stationarity check
mean_model = posterior_mean_model(post)
stab = is_stationary(mean_model)
```

This workflow demonstrates the complete Bayesian pipeline: hyperparameter optimization selects the optimal shrinkage ``\tau`` via marginal likelihood, the conjugate NIW sampler produces 1000 posterior draws, and the structural analysis functions compute IRFs, FEVD, and historical decomposition with credible intervals. The Cholesky ordering [INDPRO, CPI, FFR] identifies a monetary policy shock as the third innovation. The forecast integrates over the full posterior distribution of ``(B, \Sigma)``, providing credible intervals that account for both parameter and shock uncertainty. The posterior mean model confirms stationarity of the system.

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
