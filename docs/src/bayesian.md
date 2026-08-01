# [Bayesian VAR](@id bvar_page)

**MacroEconometricModels.jl** provides a complete Bayesian estimation framework for Vector Autoregression models, combining the Minnesota prior (Litterman 1986) with conjugate Normal-Inverse-Wishart posterior inference and data-driven hyperparameter selection via marginal likelihood optimization (Giannone, Lenza & Primiceri 2015).

- **Minnesota Prior**: Shrinkage toward random walk via dummy observations (Doan, Litterman & Sims 1984), with five tunable hyperparameters controlling tightness, lag decay, and cross-variable penalization
- **Hyperparameter Optimization**: Grid search over ``\tau`` or joint ``(\tau, \lambda, \mu)`` optimization using the closed-form marginal likelihood (Giannone, Lenza & Primiceri 2015; Banbura, Giannone & Reichlin 2010)
- **Conjugate Posterior Sampling**: Two samplers --- i.i.d. draws from the analytical Normal-Inverse-Wishart posterior (`:direct`) or a two-block Gibbs sampler (`:gibbs`) with burn-in and thinning
- **Bayesian Structural Analysis**: Posterior distributions over impulse responses, forecast error variance decomposition, and historical decomposition with credible intervals, supporting Cholesky and sign-restriction identification
- **Forecasting**: Multi-step-ahead forecasts with posterior credible intervals, integrating over parameter uncertainty across all posterior draws
- **Large BVAR**: Scalable estimation for high-dimensional systems (20+ variables) where the Minnesota prior prevents overfitting

All results integrate with `report()` for publication-quality output and `plot_result()` for interactive D3.js visualization. The reduced-form model, its identification schemes, and the frequentist counterparts of every function here are documented on the [VAR](@ref var_page) page; cointegrated systems belong in a [VECM](@ref vecm_page).

The examples use the same three FRED-MD series as the [VAR](@ref var_page) page, under their official transformation codes: `INDPRO` (`tcode=5`, log difference) and `CPIAUCSL` (`tcode=6`, second log difference) rescaled to percent, and `FEDFUNDS` (`tcode=2`, first difference), already in percentage points.

!!! warning "Scale the data before estimating"
    The inverse-Wishart prior scale is fixed at ``S_0 = I_n``, so it is *not* invariant to the
    units of `Y`. On raw log differences, whose standard deviation is around 0.008, that prior
    swamps the likelihood: the posterior draws of ``\Sigma`` come out two orders of magnitude
    too large and nearly every draw is explosive. Putting all series on a percent scale, as the
    setup below does, is what keeps the posterior usable.

```@setup bvar
using MacroEconometricModels, Random
Random.seed!(42)
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y[:, 1:2] .*= 100          # log differences to percent; FEDFUNDS is already in percentage points
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

The prior is imposed through **dummy observations** in the Bańbura-Giannone-Reichlin stacked form. The autoregressive block contributes one pseudo-observation per (lag, variable) pair, scaled by ``\sigma_i l^{d} / \tau``, which under the conjugate Kronecker structure ``\Sigma \otimes \Omega_0`` implies

```math
\text{Var}(A_{l,ji} \mid \Sigma) = \Sigma_{jj} \cdot \frac{\tau^2}{\sigma_i^2 \, l^{2d}}
```

where:
- ``A_{l,ji}`` is the coefficient on lag ``l`` of variable ``i`` in the equation for variable ``j``
- ``\tau`` is the **overall tightness**, entering as an *inverse* tightness --- the dummy observations are divided by ``\tau``, so a **larger** ``\tau`` gives a **looser** prior and a smaller one shrinks harder toward the random walk
- ``d`` is the **lag decay** exponent; the prior variance falls as ``l^{-2d}``, so distant lags are shrunk more aggressively
- ``\sigma_i^2`` is the residual variance of a univariate AR(1) fitted to variable ``i``, which puts the regressor side on a common scale
- ``\Sigma_{jj}`` is the equation-``j`` innovation variance, supplied by the Kronecker structure rather than by a hyperparameter

Own and cross-variable lags carry the same prior variance in this parameterization: the asymmetry that distinguishes them comes from the prior *mean*, which is 1 for a variable's own first lag and 0 everywhere else. Two further dummy blocks impose the **sum-of-coefficients** prior (weight ``\lambda``) and the **dummy initial observation**, or co-persistence, prior (weight ``\mu``). Both divide by their hyperparameter, so as with ``\tau``, larger means looser.

### Hyperparameter Interpretation

| Hyperparameter | Field | Default | Effect |
|----------------|-------|---------|--------|
| ``\tau`` | `tau` | `3.0` | Overall tightness, inverse scale (lower = tighter, closer to the random walk) |
| ``d`` | `decay` | `0.5` | Lag decay exponent (higher = faster decay of distant lags) |
| ``\lambda`` | `lambda` | `5.0` | Sum-of-coefficients prior, inverse scale (lower = tighter unit-root prior) |
| ``\mu`` | `mu` | `2.0` | Co-persistence / dummy-initial-observation prior, inverse scale (lower = tighter) |
| ``\omega`` | `omega` | `2.0` | Switches the residual-covariance dummy block on when positive |

!!! warning "Naming clashes with other toolboxes"
    Two conventions differ from the reference implementations. First, `omega` acts as a
    **switch**: any positive value appends the residual-covariance dummy block ``\text{diag}(\sigma)``,
    and its magnitude never enters the block. Second, the roles of `lambda` and `mu` are
    **swapped** relative to Ferroni-Canova's `BVAR_`/`rfvar3`, where `lambda` is co-persistence
    and `mu` is sum-of-coefficients. Translating hyperparameters from a paper that uses that
    toolbox requires exchanging the two.

!!! note "Technical Note"
    Dummy observations implement Theil-Goldberger mixed estimation: augmenting the data with pseudo-observations and running OLS on the combined system is algebraically equivalent to computing the posterior mean under the NIW conjugate prior. This avoids ever forming the ``\Sigma \otimes \Omega_0`` Kronecker prior covariance.

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

The shrinkage is visible in the own-first-lag coefficients, which the prior pulls toward 1. OLS puts them at ``-0.447``, ``-0.454`` and ``0.697`` for output, prices and the funds rate; under `tau=0.5` the posterior means move to ``-0.288``, ``-0.284`` and ``0.738``. Every coefficient moves toward 1, but the two badly-determined growth-rate equations move by about 0.16 while the funds-rate equation --- which the data pin down well --- moves by only 0.04. With `decay=2.0` the prior variance falls as ``l^{-4}``, so the second lag is shrunk sixteen times harder than the first.

### `MinnesotaHyperparameters` Fields

| Field | Type | Description |
|-------|------|-------------|
| `tau` | `T` | Overall tightness, inverse scale (lower = more shrinkage toward the random-walk prior) |
| `decay` | `T` | Lag decay exponent (higher = faster decay of lag importance) |
| `lambda` | `T` | Sum-of-coefficients prior, inverse scale (lower = tighter unit-root belief) |
| `mu` | `T` | Co-persistence prior, inverse scale (lower = tighter common-trend belief) |
| `omega` | `T` | Residual-covariance dummy block: included when positive, magnitude unused |

The constructor is keyword-only and every field defaults, so `MinnesotaHyperparameters(tau=0.5)` keeps the package defaults for the rest.

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

The search returns ``\hat{\tau} = 0.536``, an interior point of the ``[0.01, 10]`` range, at a log marginal likelihood of ``-70.07``. The bottom of the range is near-dogmatic shrinkage to the random walk, useful for high-dimensional systems; the top approaches unrestricted OLS. Because the marginal likelihood integrates out the parameters it penalizes overfitting automatically, so the selected ``\tau`` loosens as the sample grows and the data carry more of the inference. A value pinned to either endpoint means the range is too narrow, not that the prior should be that extreme.

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

The three-dimensional grid returns ``(\hat{\tau}, \hat{\lambda}, \hat{\mu}) = (0.644, 10.0, 1.0)`` at a log marginal likelihood of ``-68.35``, better than the ``-70.07`` the tau-only search reaches. Letting ``\lambda`` move to the top of its grid all but switches off the sum-of-coefficients prior, and the freed-up slack lets ``\tau`` sit looser than the one-dimensional search could afford. Joint selection matters most in large systems, where these priors interact strongly; for ``n \leq 5`` the gain over a tau-only search is usually small. Note that ``\hat{\lambda}`` here sits on a grid boundary, which is the signal to widen `lambda_grid` rather than to accept 10.0 as the optimum.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `tau_grid` | `AbstractRange` | `range(0.1, 5.0, length=10)` | Grid values for ``\tau`` |
| `lambda_grid` | `Vector` | `[1.0, 5.0, 10.0]` | Grid values for ``\lambda`` |
| `mu_grid` | `Vector` | `[1.0, 2.0, 5.0]` | Grid values for ``\mu`` |

### Hierarchical Optimization (GLP 2015) --- the default

Giannone, Lenza & Primiceri (2015) treat the hyperparameters as **random**, with hyperpriors, and select them by maximizing the posterior rather than the marginal likelihood alone:

```math
\hat{\gamma} = \arg\max_{\gamma}\ \log p(Y \mid \gamma) + \log p(\gamma),
\qquad \gamma = (\tau, \lambda, \mu)
```

where:
- ``p(Y \mid \gamma)`` is the closed-form conjugate marginal likelihood above
- ``p(\gamma)`` are Gamma hyperpriors in GLP's mode/standard-deviation parameterization: ``\tau`` with mode 0.2 and sd 0.4, ``\lambda`` and ``\mu`` with mode 1 and sd 1

This is what `estimate_bvar` uses by default when `prior=:minnesota` and no `hyper` is supplied. It matters because a one-dimensional grid over ``\tau`` cannot trade overall tightness against the sum-of-coefficients and initial-observation priors: holding the latter at their defaults, the grid can be driven to an endpoint of its own range and report that endpoint as though it had been selected.

```@example bvar
glp = optimize_hyperparameters_glp(Y, 2)
report(glp)
```

The joint optimizer converges in 74 iterations to ``(\hat{\tau}, \hat{\lambda}, \hat{\mu}) = (0.561, 1.264, 0.608)`` with no hyperparameter on a bound, so `converged` is `true`. Its log marginal likelihood of ``-67.82`` beats both grid searches and improves on the ``-88.87`` obtained at the package defaults --- selection is worth roughly 21 log points here, which is exactly the comparison `log_ml_default` exists to support. The maximized objective, `log_posterior = -69.64`, sits below `log_ml` because the Gamma hyperpriors penalize a ``\tau`` above their mode of 0.2. A pinned hyperparameter would instead set `at_bound` and clear `converged`, and such a result must not be used as an estimate.

!!! note "Technical Note"
    Optimization runs in log space, so every hyperparameter stays positive without a constrained solver, and uses a derivative-free Nelder-Mead restarted from several dispersed starting points --- the marginal-likelihood surface is not concave in the hyperparameters, and a single start can settle in a local optimum. The lag decay and covariance scaling are held fixed (GLP fix the lag decay rather than estimate it) and can be shifted with the `decay` and `omega` keywords.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `decay` | `Real` | `0.5` | Lag-decay exponent, held fixed |
| `omega` | `Real` | `2.0` | Covariance scaling, held fixed |
| `starts` | `Int` | `4` | Dispersed restarts of the optimizer |
| `max_iter` | `Int` | `500` | Iterations per restart |
| `f_reltol` | `Real` | `1e-8` | Relative objective tolerance |
| `verbose` | `Bool` | `true` | Warn on non-convergence |

`GLPHyperparameters{T}` return value:

| Field | Type | Description |
|-------|------|-------------|
| `hyper` | `MinnesotaHyperparameters{T}` | The optimized hyperparameters |
| `log_ml` | `T` | Log marginal likelihood at the optimum |
| `log_posterior` | `T` | `log_ml` plus the log hyperprior --- the maximized objective |
| `converged` | `Bool` | Converged **and** no hyperparameter on a bound |
| `at_bound` | `Bool` | Some hyperparameter is pinned to the search box |
| `iterations` | `Int` | Optimizer iterations |
| `log_ml_default` | `T` | Log marginal likelihood at the package defaults |

Selection is controlled from `estimate_bvar` through `hyperopt`:

```@example bvar
post_glp  = estimate_bvar(Y, 2; n_draws=100, prior=:minnesota,
                          varnames=["INDPRO", "CPI", "FFR"])            # :glp (default)
post_grid = estimate_bvar(Y, 2; n_draws=100, prior=:minnesota, hyperopt=:grid,
                          varnames=["INDPRO", "CPI", "FFR"])            # tau-only grid
using Statistics
round.([mean(post_glp.B_draws[:, 2, 1]) mean(post_grid.B_draws[:, 2, 1])], digits=4)
```

On a well-behaved three-variable system the two routes agree closely --- the posterior mean of the own first-lag coefficient in the output equation is ``-0.301`` under `:glp` and ``-0.355`` under `:grid`. The gap widens with the system dimension, where the tau-only grid cannot trade overall tightness against the other two priors. An explicit `hyper=` bypasses selection entirely under either setting.

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

```julia
plot_result(post_direct; view=:trace, params=[1, 2, 3])
```

```@raw html
<iframe src="../assets/plots/mcmc_trace.html" style="width:100%; height:520px; border:none;"></iframe>
```

The `:direct` sampler is typically 10--100x faster than Gibbs because it avoids iterative sampling; a 3-variable VAR(2) with `n_draws=1000` finishes in well under a second. Its trace plot is pure white noise by construction --- there is no chain, so there is nothing to converge. The Gibbs trace is the one worth inspecting, and close agreement between the two posterior summaries is the standard check that the sampler is correct. `plot_result` on a posterior also accepts `view=:density`, `:running`, and `:acf`.

### `estimate_bvar` Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `n_draws` | `Int` | `1000` | Number of posterior draws to retain |
| `sampler` | `Symbol` | `:direct` | Sampling algorithm (`:direct` or `:gibbs`) |
| `burnin` | `Int` | `0` | Burn-in period (`:gibbs` only; `0` becomes 200 when `sampler=:gibbs`) |
| `thin` | `Int` | `1` | Thinning interval (`:gibbs` only) |
| `prior` | `Symbol` | `:normal` | Prior type (`:normal` for diffuse, `:minnesota` for Minnesota) |
| `hyper` | `MinnesotaHyperparameters` | `nothing` | Fixed hyperparameters; `nothing` selects them via `hyperopt` when `prior=:minnesota` |
| `hyperopt` | `Symbol` | `:glp` | Selection route when `hyper=nothing`: `:glp` (joint GLP optimization) or `:grid` (tau-only) |
| `varnames` | `Vector{String}` | `nothing` | Variable display names |
| `seed` | `Integer` | `nothing` | Seed owning the RNG; records a manifest so `reproduce(post)` can re-draw bit-for-bit |
| `rng` | `AbstractRNG` | `default_rng()` | Random number generator, when no `seed` is given |

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
| `manifest` | `ReproManifest` | Seed and environment, or `nothing` unless `seed=` was passed |

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
round.([stab.max_modulus is_stationary(median_model).max_modulus], digits=4)
```

`posterior_mean_model` averages ``B`` and ``\Sigma`` element-wise across all draws; `posterior_median_model` takes the element-wise median, which is more robust to outlier draws. Both are stationary here, with largest companion eigenvalue moduli of 0.859 and 0.862 --- note that this is a property of the *averaged* coefficients and does not imply that individual draws are stationary, which is why the analysis functions still filter draw by draw. Because `BVARPosterior` stores the original data, residuals are reconstructed automatically for downstream analyses such as `historical_decomposition`.

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

The Cholesky ordering [INDPRO, CPI, FFR] identifies a monetary policy shock as the third orthogonalized innovation. Its impact effect on the funds rate is 0.149 percentage points with a 68% credible interval of ``[0.139, 0.162]``; the response of industrial production is exactly zero on impact, by construction, and reaches ``-0.047`` percent at ``h = 6`` with an interval of ``[-0.077, -0.018]`` that excludes zero. Unlike frequentist bootstrap bands, which resample around a fixed point estimate, these intervals integrate over the posterior of ``(B, \Sigma)`` directly.

!!! note "Draws are silently dropped"
    Non-stationary posterior draws and draws for which identification fails are skipped, and the
    result records the accounting: `n_requested`, `n_effective`, and `n_failed`. Here 89 of 100
    draws are usable. A warning is emitted only once **more than half** the draws are lost, so
    always read `n_effective` rather than assuming it equals `n_draws`.

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

Set identification changes the answer materially. Because every draw is rotated until the impact signs hold, the impact response of industrial production is now ``-0.35`` percent with a 68% interval of ``[-0.59, -0.10]``, an order of magnitude larger than the ``-0.047`` the Cholesky scheme produces at ``h = 6`` and no longer forced to zero on impact. The intervals combine parameter uncertainty (posterior draws of ``(B, \Sigma)``) with identification uncertainty (the rotation ``Q``), so they are wider than a Cholesky band computed on the same draws. `max_draws` caps the rotation attempts per posterior draw; a draw for which no admissible rotation is found is dropped and counted in `n_failed`.

### `irf` Keyword Arguments (BVAR dispatch)

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:cholesky` | Identification method (`:cholesky`, `:sign`, `:narrative`, etc.) |
| `quantiles` | `Vector{Real}` | `[0.16, 0.5, 0.84]` | Quantile levels for credible bands |
| `point_estimate` | `Symbol` | `:mean` | Central tendency (`:mean` or `:median`) |
| `check_func` | `Function` | `nothing` | Sign restriction check function |
| `narrative_check` | `Function` | `nothing` | Narrative restriction check function |
| `max_draws` | `Int` | `1000` | Rotation attempts per posterior draw, for set-identified methods |
| `data` | `AbstractMatrix` | `post.data` | Override the data used for residuals and narrative checks |
| `shock_names` | `Vector{String}` | `nothing` | Shock display names |
| `threaded` | `Bool` | `false` | Use threaded quantile computation |

### `BayesianImpulseResponse{T}` Fields

| Field | Type | Description |
|-------|------|-------------|
| `quantiles` | `Array{T,4}` | ``H \times n \times n \times n_q``: posterior quantiles of IRFs |
| `point_estimate` | `Array{T,3}` | ``H \times n \times n`` posterior point estimate (mean or median) |
| `horizon` | `Int` | Maximum IRF horizon |
| `variables` | `Vector{String}` | Variable names |
| `shocks` | `Vector{String}` | Shock names |
| `quantile_levels` | `Vector{T}` | Quantile levels used |
| `n_requested` | `Int` | Posterior draws supplied (`post.n_draws`) |
| `n_effective` | `Int` | Draws that were stationary and identifiable |
| `n_failed` | `Int` | Draws dropped (`n_requested - n_effective`) |

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

At ``h = 1`` the monetary shock explains exactly none of the forecast error variance of industrial production --- the Cholesky ordering forbids a contemporaneous response, so the share is 0 with a degenerate ``[0, 0]`` interval. Transmission then works through lagged effects: by ``h = 20`` the shares are 83.4% own, 8.4% price and 8.2% monetary, and the credible interval on the monetary share, ``[0.026, 0.127]``, is wide enough to span a factor of five. That width is the honest summary of what 60 observations can say about a variance share twenty months out.

### `BayesianFEVD{T}` Fields

| Field | Type | Description |
|-------|------|-------------|
| `quantiles` | `Array{T,4}` | ``H \times n \times n \times n_q``: posterior quantiles of FEVD shares |
| `point_estimate` | `Array{T,3}` | ``H \times n \times n`` posterior point estimate FEVD proportions |
| `horizon` | `Int` | Maximum horizon |
| `variables` | `Vector{String}` | Variable names |
| `shocks` | `Vector{String}` | Shock names |
| `quantile_levels` | `Vector{T}` | Quantile levels used |
| `n_requested` / `n_effective` / `n_failed` | `Int` | Draw accounting, as for `BayesianImpulseResponse` |

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

The decomposition covers the 58 effective observations left after two lags, splitting each into a contribution from every structural shock plus the initial-condition path carried in `initial_point_estimate`. Reading it alongside the variance decomposition is the point: the FEVD says *on average* how much each shock matters, while the historical decomposition says *which* episodes each shock produced. Credible intervals on the contributions reflect posterior uncertainty in the VAR parameters and, under set identification, in the rotation as well.

### `BayesianHistoricalDecomposition{T}` Fields

| Field | Type | Description |
|-------|------|-------------|
| `quantiles` | `Array{T,4}` | ``T_{\text{eff}} \times n \times n_{\text{shocks}} \times n_q``: shock contribution quantiles |
| `point_estimate` | `Array{T,3}` | ``T_{\text{eff}} \times n \times n_{\text{shocks}}`` point estimate contributions |
| `initial_quantiles` | `Array{T,3}` | ``T_{\text{eff}} \times n \times n_q``: initial condition quantiles |
| `initial_point_estimate` | `Matrix{T}` | ``T_{\text{eff}} \times n`` initial condition point estimate |
| `shocks_point_estimate` | `Matrix{T}` | ``T_{\text{eff}} \times n_{\text{shocks}}`` point estimate of the structural shocks |
| `actual` | `Matrix{T}` | ``T_{\text{eff}} \times n`` actual observed values |
| `T_eff` | `Int` | Effective sample size |
| `variables` | `Vector{String}` | Variable names |
| `shock_names` | `Vector{String}` | Shock names |
| `quantile_levels` | `Vector{T}` | Quantile levels used |
| `method` | `Symbol` | Identification method used |
| `n_requested` / `n_effective` / `n_failed` | `Int` | Draw accounting, as for `BayesianImpulseResponse` |

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

The intervals widen with the horizon because they carry two distinct sources of uncertainty: the posterior spread of ``(B, \Sigma)`` and the fresh ``N(0, \Sigma^{(s)})`` innovations drawn along each simulated path. For the funds rate the 90% band runs from ``[-0.31, 0.17]`` at ``h = 1`` to ``[-0.36, 0.44]`` at ``h = 12``, 66% wider. Industrial production widens by only 30% over the same range, because a stationary growth rate has a bounded forecast variance and almost all of its band comes from parameter uncertainty rather than from accumulating shocks. Non-stationary draws are filtered out before simulation, so explosive paths never enter the quantiles.

### `forecast` Keyword Arguments (BVAR dispatch)

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `reps` | `Int` | `nothing` | Number of posterior draws to use (default: all) |
| `conf_level` | `Real` | `0.95` | Credible interval level |
| `point_estimate` | `Symbol` | `:mean` | Central tendency (`:mean` or `:median`) |
| `rng` | `AbstractRNG` | `default_rng()` | Random number generator for the simulated shock paths |

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

The conditioned rows are pinned to 2.0 with a degenerate ``[2.0, 2.0]`` band — a hard condition holds in every posterior draw — against an unconditional funds-rate path of about ``-0.04``. The spillovers carry credible intervals that reflect the full posterior: industrial production is put at ``+1.35`` percent in the first conditioned month, but with a 95% band of ``[-1.48, 4.14]`` that straddles zero. Compare this with the [VAR](@ref var_page) dispatch, which conditions on a single point estimate and therefore reports a far tighter band for the same scenario. `n_draws` on the result records how many posterior draws survived the stationarity filter — 89 of 100 here.

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
round.([vol[1, :] vol[end, :]], digits=4)
```

The training block absorbs the first 15 observations, leaving 45 effective periods, so `vol` is ``45 \times 3`` and `vol_bands` is ``45 \times 3 \times 3``. Comparing the two ends of the sample shows how much drift the posterior actually finds: 0.518 to 0.518 for industrial production, 0.207 to 0.205 for inflation, 0.073 to 0.071 for the funds rate. The paths are nearly flat here, which is the honest result on 45 periods at the reduced draw counts these docs use --- Primiceri-style estimation needs `n_draws` in the thousands and a sample spanning a genuine change in regime before drift is identified.

### Time-Varying Impulse Responses

`irf(tvp, H; t)` computes the impulse response **at a chosen date**, integrating over posterior draws. Both the propagation (from ``B_t``) and the impact matrix (``A_t^{-1}\Sigma_t``) are taken at that date, so comparing two dates is how time variation in transmission is read off:

```@example bvar
early = irf(tvp, 12; t=5, n_draws=50)
late  = irf(tvp, 12; t=tvp.T_eff, n_draws=50)
round.([early.point_estimate[1, :, 3]  late.point_estimate[1, :, 3]], digits=4)
```

The impact column of the third shock is the recursive impact matrix ``A_t^{-1}\Sigma_t`` at the chosen date: zero for the two variables ordered above the funds rate, and 0.072 early against 0.073 late for the rate itself. The two dates are indistinguishable, consistent with the flat volatility path above; on a longer sample spanning a genuine regime change this comparison is where a change in transmission would show up. `t` indexes the effective sample, so `t=1` is the first post-training period and `t=tvp.T_eff` the last.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `t` | `Int` | `post.T_eff` | Date within the effective sample at which to evaluate the response |
| `n_draws` | `Int` | `500` | Posterior draws used, capped at the number available |
| `quantile_levels` | `Vector{<:Real}` | `[0.05, 0.16, 0.84, 0.95]` | Credible-band levels |
| `stationary_only` | `Bool` | `true` | Drop draws whose companion matrix at date `t` is explosive |

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
| `A_draws` | `Array{T,3}` | ``n_{draws}\times T_{eff}\times n_a`` free elements of ``A_t``, ``n_a = n(n-1)/2``, stacked row-wise |
| `H_draws` | `Array{T,3}` | ``n_{draws}\times T_{eff}\times n`` log-**variances** ``\log\sigma^2_{i,t}`` |
| `Q_draws` | `Array{T,3}` | Coefficient random-walk covariance draws |
| `S_draws` | `Array{T,3}` | Block-diagonal ``A_t`` random-walk covariance draws |
| `W_draws` | `Matrix{T}` | Log-volatility random-walk variances |
| `Y` | `Matrix{T}` | The original data |
| `p` / `n` | `Int` | Lag order and number of variables |
| `T_eff` | `Int` | Effective sample after the training block |
| `n_train` | `Int` | Training-sample length used for the priors |
| `tvp` / `sv` | `Bool` | Which blocks were allowed to drift |
| `varnames` | `Vector{String}` | Variable names |

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

The largest violation of the aggregation identity across all reference dates is of order ``10^{-7}``, i.e. numerical noise rather than approximation error. That exactness is the point of the Durbin-Koopman construction: because the aggregation carries no measurement error, applying the observation map to the simulated path reproduces the observed low-frequency value exactly, so the interpolated monthly series is guaranteed to sum back to the published quarterly figure.

Because the parameter draws are ordinary VAR draws, the existing analysis dispatches apply at the high frequency:

```@example bvar
fc_mf = forecast(mf, 6)
report(fc_mf)
```

With `low_freq` empty nothing is latent and the sampler reduces to the conjugate BVAR. See also the [Nowcasting](@ref nowcast_page) pages, which solve the same ragged-edge problem with a dynamic factor model instead.

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
| `p` / `n` | `Int` | Lag order and number of variables |
| `T_hf` | `Int` | Number of high-frequency periods |
| `low_freq` | `Vector{Int}` | Low-frequency column indices |
| `freq_ratio` | `Int` | High-frequency periods per low-frequency period |
| `aggregation` | `Vector{Symbol}` | Aggregation rule per low-frequency series |
| `varnames` | `Vector{String}` | Variable names |

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
    lambda = 1.0,   # Tight sum-of-coefficients prior
    mu = 0.5,       # Tight co-persistence prior
    omega = 1.0     # Include the residual-covariance dummy block
)

post_lg = estimate_bvar(X_lg, 4; n_draws=100, prior=:minnesota, hyper=hyper_large)
report(post_lg)
```

At 20 variables and 4 lags the system carries ``20^2 \times 4 + 20 = 1620`` free parameters in total, or ``1 + 20 \times 4 = 81`` per equation. The sample used here has 570 usable observations, so OLS is estimable but poorly conditioned, and unrestricted estimates would be dominated by sampling noise. The Minnesota prior with `tau=0.1` shrinks hard toward the random walk, `decay=2.0` all but eliminates the distant lags, and the tight sum-of-coefficients and co-persistence priors keep the implied long-run behavior sensible: all 100 posterior draws come back stationary. `optimize_hyperparameters_full` automates the ``(\tau, \lambda, \mu)`` choice when a hand-set prior is not wanted.

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
stab_ce
```

The pipeline runs end to end: the grid search selects ``\hat{\tau} = 0.536``, the conjugate NIW sampler draws 100 posteriors from it, and the structural functions turn those draws into IRFs, variance shares, and a historical decomposition, each with credible intervals rather than point estimates. The Cholesky ordering [INDPRO, CPI, FFR] identifies the monetary policy shock as the third innovation, so output and prices are held to zero on impact and respond only with a lag. The forecast integrates over the full posterior of ``(B, \Sigma)`` and adds fresh innovations along each path, which is why its bands are wider than any bootstrap band computed at a single point estimate. The final stationarity check on the posterior mean model closes the loop --- it is stationary, so the reported IRFs and variance decompositions are well defined.

---

## Common Pitfalls

1. **Too few posterior draws**: With `n_draws=100`, credible intervals are noisy and quantile estimates are unreliable. Use at least `n_draws=1000` for stable inference. For sign restrictions, which discard non-conforming draws, increase to `n_draws=5000` or more.

2. **Prior sensitivity with diffuse prior**: Setting `prior=:normal` (the default) uses a diffuse NIW prior that provides minimal regularization. For systems with more than 5 variables, switch to `prior=:minnesota` to avoid overfitting. The diffuse prior is appropriate only for small, well-identified systems with ample data.

3. **Minnesota prior assumes random walk**: The Minnesota prior centers on a random walk --- each variable's own first lag is 1, all others are 0. For stationary variables (interest rates, unemployment, any differenced series) that prior mean is wrong, and it pulls the posterior toward the unit circle. Demean or detrend before estimation, or raise `tau` to loosen the prior and let the data dominate. Note the direction: `tau` is an *inverse* tightness, so a **larger** value means **less** shrinkage.

4. **Hyperparameter optimization convergence**: `optimize_hyperparameters` is a discrete grid search, so its answer depends on `grid_size` and `tau_range`; `optimize_hyperparameters_full` searches a three-dimensional grid with the same caveat. If the selected value sits on a boundary, widen the range rather than accepting it. `optimize_hyperparameters_glp` reports this directly through `at_bound` and refuses to set `converged` when a hyperparameter is pinned.

5. **Non-stationary posterior draws**: `irf`, `fevd`, `historical_decomposition`, `forecast`, and `conditional_forecast` all silently discard draws whose companion matrix has an eigenvalue at or above unity, and warn only once more than half are gone. Read `n_effective` on the result. Mass rejection usually means the prior is too loose (raise the shrinkage by lowering `tau`) or, more often, that `Y` is on a scale where the fixed ``S_0 = I_n`` inverse-Wishart prior dominates the likelihood --- see the warning at the top of this page.

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
