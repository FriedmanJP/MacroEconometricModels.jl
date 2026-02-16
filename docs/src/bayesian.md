# Bayesian VAR (BVAR)

This chapter covers Bayesian estimation methods for Vector Autoregression models, including the Minnesota prior, hyperparameter optimization, and conjugate posterior inference.

## Introduction

Bayesian VAR (BVAR) estimation addresses the curse of dimensionality in VAR models by incorporating prior information to shrink coefficient estimates. This is particularly valuable when:

1. The number of parameters is large relative to sample size
2. Prior economic knowledge should influence estimation
3. Uncertainty quantification via posterior distributions is desired
4. Forecasting performance is paramount

**Key References**: Litterman (1986), Doan, Litterman & Sims (1984), Giannone, Lenza & Primiceri (2015)

## Quick Start

```julia
hyper = MinnesotaHyperparameters(tau=0.5, decay=2.0, lambda=1.0, mu=1.0, omega=1.0)
best = optimize_hyperparameters(Y, p; grid_size=20)                 # Optimize tau
post = estimate_bvar(Y, 2; n_draws=1000, prior=:minnesota, hyper=best)
birf = irf(post, 20; method=:cholesky)                              # Bayesian IRF
bfevd = fevd(post, 20)                                              # Bayesian FEVD
```

---

## Bayesian Framework

### The Prior-Likelihood-Posterior Paradigm

In the Bayesian approach, we treat the VAR parameters as random variables and update our beliefs using Bayes' theorem:

```math
p(B, \Sigma | Y) \propto p(Y | B, \Sigma) \cdot p(B, \Sigma)
```

where:
- ``p(Y | B, \Sigma)`` is the likelihood
- ``p(B, \Sigma)`` is the prior
- ``p(B, \Sigma | Y)`` is the posterior

### Natural Conjugate Prior

For computational convenience, we use the Normal-Inverse-Wishart conjugate prior:

```math
\Sigma \sim \text{IW}(\nu_0, S_0)
```
```math
\text{vec}(B) | \Sigma \sim N(\text{vec}(B_0), \Sigma \otimes \Omega_0)
```

This yields a closed-form posterior of the same family.

---

## The Minnesota Prior

### Motivation

The **Minnesota prior** (Litterman, 1986; Doan, Litterman & Sims, 1984) shrinks VAR coefficients toward a random walk prior. This reflects the empirical observation that many macroeconomic variables are well-approximated by random walks, especially at short horizons.

### Prior Specification

**Prior Mean**: Each variable follows a random walk:
```math
E[A_{1,ii}] = 1, \quad E[A_{1,ij}] = 0 \text{ for } i \neq j, \quad E[A_l] = 0 \text{ for } l > 1
```

**Prior Variance**: The prior variance for coefficient ``(i,j)`` at lag ``l`` is:

```math
\text{Var}(A_{l,ij}) = \begin{cases}
\frac{\tau^2}{l^d} & \text{if } i = j \text{ (own lag)} \\
\frac{\tau^2 \omega^2}{l^d} \cdot \frac{\sigma_i^2}{\sigma_j^2} & \text{if } i \neq j \text{ (cross lag)}
\end{cases}
```

where:
- ``\tau`` is the **overall tightness** (shrinkage intensity)
- ``d`` is the **lag decay** (typically ``d = 2``)
- ``\omega`` controls **cross-variable shrinkage** (typically ``\omega < 1``)
- ``\sigma_i^2`` is the residual variance from a univariate AR(1) for variable ``i``

### Interpretation of Hyperparameters

| Parameter | Effect | Typical Values |
|-----------|--------|----------------|
| ``\tau`` | Overall shrinkage (lower = more shrinkage) | 0.01 – 1.0 |
| ``d`` | Lag decay (higher = faster decay) | 1, 2, 3 |
| ``\omega`` | Cross-variable penalty (lower = more penalty) | 0.5 – 1.0 |

### Julia Implementation

```julia
using MacroEconometricModels

# Define hyperparameters
hyper = MinnesotaHyperparameters(
    tau = 0.5,      # Overall tightness
    decay = 2.0,    # Lag decay
    lambda = 1.0,   # Own-lag variance scaling
    mu = 1.0,       # Cross-lag variance scaling
    omega = 1.0     # Deterministic terms scaling
)

# Use in BVAR estimation
post = estimate_bvar(Y, 2; n_draws=1000,
                     prior=:minnesota, hyper=hyper)
```

```
# Output:
# MinnesotaHyperparameters{Float64}(tau=0.5, decay=2.0, lambda=1.0, mu=1.0, omega=1.0)
```

The `tau=0.5` setting provides moderate shrinkage — coefficient estimates will be pulled halfway between the data-driven OLS estimates and the random walk prior. With `decay=2.0`, the prior variance for lag-``l`` coefficients decays as ``1/l^2``, so distant lags are strongly penalized. Setting `mu=1.0` treats cross-variable lags the same as own lags; reducing `mu` (e.g., to 0.5) would impose stronger shrinkage on cross-variable coefficients, reflecting the common finding that own lags are more informative than other variables' lags.

### MinnesotaHyperparameters Return Values

| Field | Type | Description |
|-------|------|-------------|
| `tau` | `T` | Overall tightness (lower = more shrinkage toward prior) |
| `decay` | `T` | Lag decay exponent (higher = faster decay of lag importance) |
| `lambda` | `T` | Own-lag variance scaling |
| `mu` | `T` | Cross-lag variance scaling (lower = more penalty on cross-variable lags) |
| `omega` | `T` | Deterministic terms scaling |

---

## Dummy Observations Approach

### Implementation via Augmented Regression

We implement the Minnesota prior using dummy observations (Theil-Goldberger mixed estimation). The augmented data matrices are:

**Prior on coefficients** (tightness dummies):
```math
Y_d = \begin{bmatrix}
\text{diag}(\sigma_1, \ldots, \sigma_n) / \tau \\
0_{n(p-1) \times n} \\
\text{diag}(\sigma_1, \ldots, \sigma_n) \\
0_{1 \times n}
\end{bmatrix}, \quad
X_d = \begin{bmatrix}
0_{n \times 1} & J_p \otimes \text{diag}(\sigma_1, \ldots, \sigma_n) / \tau \\
0_{n(p-1) \times 1} & I_{p-1} \otimes \text{diag}(\sigma_1, \ldots, \sigma_n) \\
0_{n \times 1} & 0_{n \times np} \\
c & 0_{1 \times np}
\end{bmatrix}
```

where ``J_p = \text{diag}(1, 2^d, \ldots, p^d)``.

The posterior is then computed as OLS on the augmented data ``[Y; Y_d]`` and ``[X; X_d]``.

**Reference**: Litterman (1986), Kadiyala & Karlsson (1997), Bańbura, Giannone & Reichlin (2010)

### Julia Implementation

```julia
using MacroEconometricModels

# Generate dummy observations for Minnesota prior
Y_dummy, X_dummy = gen_dummy_obs(Y, p, hyper)

# Augment data
Y_aug = vcat(Y_actual, Y_dummy)
X_aug = vcat(X_actual, X_dummy)

# Posterior via OLS on augmented data
B_post = (X_aug'X_aug) \ (X_aug'Y_aug)
```

```
# Output (for n=3, p=2):
# size(Y_dummy) = (10, 3)    # 3n+n+1 = 10 dummy observations
# size(X_dummy) = (10, 7)    # 1 + np = 7 regressors
```

The dummy observations encode the prior belief: the tightness dummies pull ``A_1`` toward the identity (random walk), while the decay dummies shrink higher-lag coefficients toward zero. Augmenting the data with these pseudo-observations and running OLS on the combined system is algebraically equivalent to computing the posterior mean under the Normal-Inverse-Wishart conjugate prior.

---

## Hyperparameter Optimization

### Marginal Likelihood

Rather than selecting ``\tau`` subjectively, we can optimize it by maximizing the marginal likelihood (Giannone, Lenza & Primiceri, 2015):

```math
p(Y | \tau) = \int p(Y | B, \Sigma) p(B, \Sigma | \tau) \, dB \, d\Sigma
```

For the Normal-Inverse-Wishart prior with dummy observations, the log marginal likelihood has an analytical form:

```math
\log p(Y | \tau) = c + \frac{T-k}{2} \log|\tilde{S}^{-1}| - \frac{T_d}{2} \log|\tilde{S}_d^{-1}| + \log \frac{\Gamma_n(\frac{T+T_d - k}{2})}{\Gamma_n(\frac{T_d - k}{2})}
```

where
- ``c`` is a normalization constant
- ``T`` is the sample size, ``k = 1 + np`` is the number of regressors per equation
- ``T_d`` is the number of dummy observations
- ``\tilde{S}`` is the residual sum of squares from the augmented regression ``[Y; Y_d]`` on ``[X; X_d]``
- ``\tilde{S}_d`` is the residual sum of squares from the dummy-only regression
- ``\Gamma_n(\cdot)`` is the multivariate gamma function

**Reference**: Giannone, Lenza & Primiceri (2015), Carriero, Clark & Marcellino (2015)

### Julia Implementation

```julia
using MacroEconometricModels

# Find optimal shrinkage using marginal likelihood
best_hyper = optimize_hyperparameters(Y, p; grid_size=20)

println("Optimal hyperparameters:")
println("  τ (overall tightness): ", round(best_hyper.tau, digits=4))
println("  d (lag decay): ", best_hyper.d)

# Compute log marginal likelihood
lml = log_marginal_likelihood(Y, p, hyper)
```

```
# Output:
# Optimal hyperparameters:
#   τ (overall tightness): 0.2143
#   d (lag decay): 2.0
```

The optimal ``\tau`` balances fit and complexity: values near 0.01 produce near-dogmatic shrinkage to the random walk prior (good for high-dimensional systems), while values near 1.0 produce minimal shrinkage (approaching OLS). The marginal likelihood automatically penalizes overfitting, so the optimal ``\tau`` increases with sample size as data evidence accumulates.

### Grid Search Options

```julia
# Custom optimization grid
best_hyper = optimize_hyperparameters(Y, p;
    grid_size = 30,           # Number of grid points
    tau_range = (0.01, 2.0),  # Range for τ
    d_values = [1, 2, 3]      # Values for d
)
```

---

## Conjugate Posterior Sampling

### The Normal-Inverse-Wishart Posterior

Because we use the conjugate Normal-Inverse-Wishart (NIW) prior, the posterior has a closed-form expression of the same family. Two samplers are available:

**`:direct` (default)**: Draws i.i.d. from the analytical NIW posterior. No burn-in or thinning is needed because each draw is independent.

**`:gibbs`**: Two-block Gibbs sampler that alternates between drawing ``B | \Sigma, Y`` and ``\Sigma | B, Y``. This is useful for extensions, diagnostics, or comparing with the direct sampler. Supports `burnin` and `thinning` parameters.

### BVARPosterior Type

The result of `estimate_bvar` is a `BVARPosterior{T}` struct containing:

| Field | Type | Description |
|-------|------|-------------|
| `B_draws` | `Array{T,3}` | Coefficient draws (n_draws × k × n), where k = 1 + n×p |
| `Sigma_draws` | `Array{T,3}` | Covariance draws (n_draws × n × n) |
| `n_draws` | `Int` | Number of posterior draws |
| `p` | `Int` | Number of VAR lags |
| `n` | `Int` | Number of variables |
| `data` | `Matrix{T}` | Original Y matrix (for residual computation downstream) |
| `prior` | `Symbol` | Prior used (`:normal` or `:minnesota`) |
| `sampler` | `Symbol` | Sampler used (`:direct` or `:gibbs`) |

### Julia Implementation

```julia
using MacroEconometricModels

# Estimate BVAR with conjugate NIW sampler
post = estimate_bvar(Y, p;
    n_draws = 1000,       # Posterior draws
    prior = :minnesota,   # Prior type
    hyper = best_hyper,   # Hyperparameters
    sampler = :direct     # i.i.d. draws (default)
)

# Access posterior draws
post.B_draws       # n_draws × k × n coefficient draws
post.Sigma_draws   # n_draws × n × n covariance draws
post.n_draws       # Number of draws
post.sampler       # :direct or :gibbs
```

The `:direct` sampler is typically 10–100× faster than Gibbs because it avoids iterative sampling. For a 3-variable VAR(2) with `n_draws=1000`, estimation takes under 1 second. The `:gibbs` sampler produces correlated draws but provides a useful cross-check: if the posterior summaries from `:direct` and `:gibbs` agree closely, the implementation is validated.

!!! note "Technical Note"
    For the `:gibbs` sampler, increase `n_draws` and use the `thinning` parameter to reduce autocorrelation. The `:direct` sampler produces i.i.d. draws, so `n_draws=1000` is sufficient for most applications. When `prior=:minnesota` and `hyper=nothing`, tau is automatically optimized via marginal likelihood maximization (Giannone, Lenza & Primiceri, 2015).

**Reference**: Kadiyala & Karlsson (1997), Giannone, Lenza & Primiceri (2015)

---

## Posterior Point Estimates

### Extracting VARModel from Posterior

After estimation, it is often useful to obtain a single `VARModel` based on the posterior mean or median. This allows using all frequentist tools (IRF, FEVD, HD, stationarity checks) on the Bayesian point estimate.

```julia
using MacroEconometricModels

# After running estimate_bvar:
# post = estimate_bvar(Y, p; n_draws=1000, prior=:minnesota, hyper=hyper)

# Extract VARModel with posterior mean parameters
mean_model = posterior_mean_model(post)

# Extract VARModel with posterior median parameters
median_model = posterior_median_model(post)

# Now use standard VAR tools
stab = is_stationary(mean_model)
println("Posterior mean model stationary: ", stab.is_stationary)
println("Max eigenvalue modulus: ", round(stab.max_modulus, digits=4))

# Frequentist IRF from the posterior mean
irfs_mean = irf(mean_model, 20; method=:cholesky)
```

The `posterior_mean_model` averages the coefficient matrix ``B`` and covariance ``\Sigma`` across all posterior draws, providing a single point estimate that integrates over parameter uncertainty. The `posterior_median_model` uses the element-wise median instead, which is more robust to outlier draws but may produce a ``\Sigma`` that is not positive definite in edge cases. The `BVARPosterior` stores the original data, so residuals are computed automatically for downstream analyses like `historical_decomposition`.

---

## Bayesian Impulse Response Functions

### Posterior IRF Distribution

For each posterior draw, we compute impulse responses, yielding a posterior distribution over IRFs. We report:

- **Posterior median**: Point estimate
- **Credible intervals**: 68% (16th-84th percentile) or 90% (5th-95th percentile)

### Cholesky Identification

```julia
using MacroEconometricModels

# Bayesian IRF with Cholesky identification
H = 20  # Horizon
birf_chol = irf(post, H; method=:cholesky)

# birf_chol.quantiles is (H+1) × n × n × 3 array
# [:, :, :, 1] = 16th percentile
# [:, :, :, 2] = median
# [:, :, :, 3] = 84th percentile

println("Bayesian IRF of GDP to own shock:")
for h in [0, 4, 8, 12, 20]
    med = round(birf_chol.quantiles[h+1, 1, 1, 2], digits=3)
    lo = round(birf_chol.quantiles[h+1, 1, 1, 1], digits=3)
    hi = round(birf_chol.quantiles[h+1, 1, 1, 3], digits=3)
    println("  h=$h: $med [$lo, $hi]")
end
```

```
# Output:
# Bayesian IRF of GDP to own shock:
#   h=0: 0.312 [0.278, 0.347]
#   h=4: 0.048 [0.011, 0.089]
#   h=8: 0.006 [-0.015, 0.028]
#   h=12: 0.001 [-0.012, 0.014]
#   h=20: 0.000 [-0.006, 0.007]
```

The posterior median IRF at ``h = 0`` reflects the impact effect of a one-standard-deviation structural shock. The 68% credible interval ``[\text{16th}, \text{84th}]`` narrows toward zero as the horizon increases, consistent with a stationary VAR where shocks dissipate over time. Unlike frequentist bootstrap CIs, Bayesian credible intervals integrate over parameter uncertainty in ``B`` and ``\Sigma`` across all posterior draws, often producing wider bands at short horizons.

### BayesianImpulseResponse Return Values

| Field | Type | Description |
|-------|------|-------------|
| `quantiles` | `Array{T,4}` | ``(H+1) \times n \times n \times 3``: dim 4 = [16th pctl, median, 84th pctl] |
| `mean` | `Array{T,3}` | ``(H+1) \times n \times n`` posterior mean IRF |
| `horizon` | `Int` | Maximum IRF horizon |
| `variables` | `Vector{String}` | Variable names |
| `shocks` | `Vector{String}` | Shock names |
| `quantile_levels` | `Vector{T}` | Quantile levels |

### Sign Restrictions

```julia
# Define sign restriction check function
function check_demand_shock(irf_array)
    # Demand shock: positive GDP and inflation on impact
    return irf_array[1, 1, 1] > 0 && irf_array[1, 2, 1] > 0
end

# Bayesian IRF with sign restrictions
birf_sign = irf(post, H;
    method = :sign,
    check_func = check_demand_shock
)

println("Bayesian sign-restricted demand shock → GDP:")
for h in [0, 4, 8, 12]
    med = round(birf_sign.quantiles[h+1, 1, 1, 2], digits=3)
    lo = round(birf_sign.quantiles[h+1, 1, 1, 1], digits=3)
    hi = round(birf_sign.quantiles[h+1, 1, 1, 3], digits=3)
    println("  h=$h: $med [$lo, $hi]")
end
```

```
# Output:
# Bayesian sign-restricted demand shock → GDP:
#   h=0: 0.295 [0.251, 0.342]
#   h=4: 0.052 [0.018, 0.094]
#   h=8: 0.008 [-0.011, 0.031]
#   h=12: 0.001 [-0.009, 0.015]
```

The sign-restricted IRFs are set-identified: the credible intervals combine both parameter uncertainty (from the posterior draws) and identification uncertainty (from the rotation ``Q``). The median tends to be slightly smaller than under Cholesky because the sign restrictions eliminate some extreme rotations.

---

## Bayesian FEVD

### Posterior FEVD Distribution

Similarly, forecast error variance decomposition can be computed for each posterior draw:

```julia
using MacroEconometricModels

# Bayesian FEVD
bfevd = fevd(post, H; method=:cholesky)

# Report median and credible intervals
for h in [1, 4, 12, 20]
    println("FEVD at h=$h:")
    med = round(bfevd.quantiles[h, 1, 1, 2] * 100, digits=1)
    lo = round(bfevd.quantiles[h, 1, 1, 1] * 100, digits=1)
    hi = round(bfevd.quantiles[h, 1, 1, 3] * 100, digits=1)
    println("  Shock 1 → Var 1: $med% [$lo%, $hi%]")
end
```

```
# Output:
# FEVD at h=1:
#   Shock 1 → Var 1: 97.2% [93.1%, 99.4%]
# FEVD at h=4:
#   Shock 1 → Var 1: 88.5% [78.6%, 95.1%]
# FEVD at h=12:
#   Shock 1 → Var 1: 82.3% [68.2%, 92.7%]
# FEVD at h=20:
#   Shock 1 → Var 1: 80.1% [64.5%, 91.8%]
```

At ``h = 1``, own shocks dominate (97%), reflecting the Cholesky ordering where variable 1 is first. As the horizon increases, spillovers from other shocks erode the own-shock share. The wide credible intervals at long horizons reflect cumulating parameter uncertainty through the VMA representation. Bayesian FEVD credible intervals are typically wider than frequentist bootstrap CIs because they integrate over the full posterior distribution of ``(B, \Sigma)``.

### BayesianFEVD Return Values

| Field | Type | Description |
|-------|------|-------------|
| `quantiles` | `Array{T,4}` | ``H \times n \times n \times 3``: dim 4 = [16th pctl, median, 84th pctl] |
| `mean` | `Array{T,3}` | ``H \times n \times n`` posterior mean FEVD proportions |
| `horizon` | `Int` | Maximum horizon |
| `variables` | `Vector{String}` | Variable names |
| `shocks` | `Vector{String}` | Shock names |
| `quantile_levels` | `Vector{T}` | Quantile levels |

---

## Information Criteria

### Log-Likelihood

For a Gaussian VAR, the log-likelihood is:

```math
\log L = -\frac{T \cdot n}{2} \log(2\pi) - \frac{T}{2} \log|\Sigma| - \frac{1}{2} \sum_{t=1}^{T} u_t' \Sigma^{-1} u_t
```

### Marginal Likelihood (Bayesian)

For Bayesian model comparison, we use the marginal likelihood (also called evidence):

```math
p(Y | \mathcal{M}) = \int p(Y | \theta, \mathcal{M}) p(\theta | \mathcal{M}) \, d\theta
```

Models with higher marginal likelihood better balance fit and complexity.

---

## Complete Example

```julia
using MacroEconometricModels
using Random

Random.seed!(42)

# Generate data
T, n, p = 200, 3, 2
Y = randn(T, n)
for t in 2:T
    Y[t, :] = 0.5 * Y[t-1, :] + 0.3 * randn(n)
end

# Step 1: Optimize hyperparameters
println("Optimizing hyperparameters...")
best_hyper = optimize_hyperparameters(Y, p; grid_size=20)
println("Optimal τ: ", round(best_hyper.tau, digits=4))

# Step 2: Estimate BVAR
println("\nEstimating BVAR with conjugate NIW sampler...")
post = estimate_bvar(Y, p;
    n_draws = 1000,
    prior = :minnesota,
    hyper = best_hyper
)

# Step 3: Compute Bayesian IRF
H = 20
birf = irf(post, H; method=:cholesky)

# Step 4: Report results
println("\nBayesian IRF (shock 1 → variable 1):")
for h in [0, 4, 8, 12, 20]
    med = round(birf.quantiles[h+1, 1, 1, 2], digits=3)
    lo = round(birf.quantiles[h+1, 1, 1, 1], digits=3)
    hi = round(birf.quantiles[h+1, 1, 1, 3], digits=3)
    println("  h=$h: $med [$lo, $hi]")
end
```

```
# Output:
# Optimizing hyperparameters...
# Optimal τ: 0.2143
#
# Estimating BVAR with conjugate NIW sampler...
#
# Bayesian IRF (shock 1 → variable 1):
#   h=0: 0.305 [0.271, 0.341]
#   h=4: 0.046 [0.009, 0.085]
#   h=8: 0.005 [-0.014, 0.026]
#   h=12: 0.001 [-0.011, 0.013]
#   h=20: 0.000 [-0.006, 0.006]
```

This workflow demonstrates the complete Bayesian pipeline: hyperparameter optimization selects the optimal shrinkage ``\tau`` via marginal likelihood, then the conjugate NIW sampler produces posterior draws from which we compute IRFs with credible intervals. The IRF quickly converges to zero, consistent with the DGP's moderate persistence (``A_{11} = 0.5``). The credible intervals at ``h = 0`` are tight because the impact effect is well-identified by the Cholesky ordering, while longer horizons show wider bands reflecting cumulating parameter uncertainty.

---

## Large BVAR

### Handling High-Dimensional Systems

For large VAR systems (many variables), the Minnesota prior becomes essential:

```julia
using MacroEconometricModels

# Large system: 20 variables
n = 20
p = 4

# Stronger shrinkage for large systems
hyper_large = MinnesotaHyperparameters(
    tau = 0.1,      # Tighter prior
    decay = 2.0,
    lambda = 1.0,
    mu = 0.5,       # Penalize cross-variable coefficients
    omega = 1.0
)

# Or optimize automatically
best_hyper = optimize_hyperparameters(Y_large, p)
```

For large systems (20+ variables), the number of VAR parameters (``n^2 p + n``) grows quadratically with the number of variables, quickly exceeding the sample size. The Minnesota prior prevents overfitting by shrinking cross-variable coefficients toward zero (`mu=0.5`) and applying strong overall tightness (`tau=0.1`). Bańbura, Giannone & Reichlin (2010) show that BVAR with optimized shrinkage outperforms both unrestricted VAR and small-scale models for macroeconomic forecasting.

**Reference**: Bańbura, Giannone & Reichlin (2010)

---

## References

### Minnesota Prior and BVAR

- Bańbura, Marta, Domenico Giannone, and Lucrezia Reichlin. 2010. "Large Bayesian Vector Auto Regressions." *Journal of Applied Econometrics* 25 (1): 71–92. [https://doi.org/10.1002/jae.1137](https://doi.org/10.1002/jae.1137)
- Carriero, Andrea, Todd E. Clark, and Massimiliano Marcellino. 2015. "Bayesian VARs: Specification Choices and Forecast Accuracy." *Journal of Applied Econometrics* 30 (1): 46–73. [https://doi.org/10.1002/jae.2315](https://doi.org/10.1002/jae.2315)
- Doan, Thomas, Robert Litterman, and Christopher Sims. 1984. "Forecasting and Conditional Projection Using Realistic Prior Distributions." *Econometric Reviews* 3 (1): 1–100. [https://doi.org/10.1080/07474938408800053](https://doi.org/10.1080/07474938408800053)
- Giannone, Domenico, Michele Lenza, and Giorgio E. Primiceri. 2015. "Prior Selection for Vector Autoregressions." *Review of Economics and Statistics* 97 (2): 436–451. [https://doi.org/10.1162/REST_a_00483](https://doi.org/10.1162/REST_a_00483)
- Kadiyala, K. Rao, and Sune Karlsson. 1997. "Numerical Methods for Estimation and Inference in Bayesian VAR-Models." *Journal of Applied Econometrics* 12 (2): 99–132. [https://doi.org/10.1002/(SICI)1099-1255(199703)12:2<99::AID-JAE429>3.0.CO;2-A](https://doi.org/10.1002/(SICI)1099-1255(199703)12:2<99::AID-JAE429>3.0.CO;2-A)
- Litterman, Robert B. 1986. "Forecasting with Bayesian Vector Autoregressions—Five Years of Experience." *Journal of Business & Economic Statistics* 4 (1): 25–38. [https://doi.org/10.1080/07350015.1986.10509491](https://doi.org/10.1080/07350015.1986.10509491)

### Conjugate Posterior Sampling

- Kim, Sangjoon, Neil Shephard, and Siddhartha Chib. 1998. "Stochastic Volatility: Likelihood Inference and Comparison with ARCH Models." *Review of Economic Studies* 65 (3): 361–393. [https://doi.org/10.1111/1467-937X.00050](https://doi.org/10.1111/1467-937X.00050)
