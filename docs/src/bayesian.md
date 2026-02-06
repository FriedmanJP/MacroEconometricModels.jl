# Bayesian VAR (BVAR)

This chapter covers Bayesian estimation methods for Vector Autoregression models, including the Minnesota prior, hyperparameter optimization, and MCMC inference.

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
chain = estimate_bvar(Y, 2; n_samples=2000, prior=:minnesota, hyper=best)
birf = irf(chain, 2, 3, 20; method=:cholesky)                      # Bayesian IRF
bfevd = fevd(chain, 2, 3, 20)                                      # Bayesian FEVD
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
chain = estimate_bvar(Y, 2; n_samples=2000, n_adapts=500,
                      prior=:minnesota, hyper=hyper)
```

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

## MCMC Estimation with Turing.jl

### The BVAR Model

For more flexible priors or non-conjugate settings, we use MCMC via Turing.jl with the NUTS sampler:

```julia
@model function bvar_model(Y, X, prior_mean, prior_var, ν₀, S₀)
    n = size(Y, 2)
    k = size(X, 2)

    # Prior on error covariance
    Σ ~ InverseWishart(ν₀, S₀)

    # Prior on coefficients
    B ~ MatrixNormal(prior_mean, prior_var, Σ)

    # Likelihood
    for t in axes(Y, 1)
        Y[t, :] ~ MvNormal(X[t, :]' * B, Σ)
    end
end
```

### Julia Implementation

```julia
using MacroEconometricModels

# Estimate BVAR with MCMC
chain = estimate_bvar(Y, p;
    n_samples = 2000,     # Posterior samples
    n_adapts = 500,       # Adaptation samples
    prior = :minnesota,   # Prior type
    hyper = best_hyper    # Hyperparameters
)

# Access posterior draws
# chain.samples contains the MCMC draws
```

### Convergence Diagnostics

```julia
# Extract chain parameters
params = extract_chain_parameters(chain)

# Check R-hat statistics
# Check effective sample sizes
# Trace plots for visual inspection
```

!!! note "Technical Note"
    MCMC convergence should be assessed before interpreting results. Key diagnostics include: (1) ``\hat{R}`` (R-hat) statistics should be below 1.05 for all parameters; (2) effective sample size (ESS) should be at least 400 for reliable posterior quantile estimation; (3) trace plots should show good mixing without trends or multimodality. If `n_samples=2000` with `n_adapts=500` shows poor convergence, try increasing both values or switching to a more informative prior (lower `tau`).

**Reference**: Gelman et al. (2013), Hoffman & Gelman (2014)

---

## Bayesian Impulse Response Functions

### Posterior IRF Distribution

For each MCMC draw, we compute impulse responses, yielding a posterior distribution over IRFs. We report:

- **Posterior median**: Point estimate
- **Credible intervals**: 68% (16th-84th percentile) or 90% (5th-95th percentile)

### Cholesky Identification

```julia
using MacroEconometricModels

# Bayesian IRF with Cholesky identification
H = 20  # Horizon
birf_chol = irf(chain, p, n, H; method=:cholesky)

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
birf_sign = irf(chain, p, n, H;
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

---

## Bayesian FEVD

### Posterior FEVD Distribution

Similarly, forecast error variance decomposition can be computed for each posterior draw:

```julia
using MacroEconometricModels

# Bayesian FEVD
bfevd = fevd(chain, p, n, H; method=:cholesky)

# Report median and credible intervals
for h in [1, 4, 12, 20]
    println("FEVD at h=$h:")
    med = round(bfevd.quantiles[h, 1, 1, 2] * 100, digits=1)
    lo = round(bfevd.quantiles[h, 1, 1, 1] * 100, digits=1)
    hi = round(bfevd.quantiles[h, 1, 1, 3] * 100, digits=1)
    println("  Shock 1 → Var 1: $med% [$lo%, $hi%]")
end
```

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
println("\nEstimating BVAR with MCMC...")
chain = estimate_bvar(Y, p;
    n_samples = 2000,
    n_adapts = 500,
    prior = :minnesota,
    hyper = best_hyper
)

# Step 3: Compute Bayesian IRF
H = 20
birf = irf(chain, p, n, H; method=:cholesky)

# Step 4: Report results
println("\nBayesian IRF (shock 1 → variable 1):")
for h in [0, 4, 8, 12, 20]
    med = round(birf.quantiles[h+1, 1, 1, 2], digits=3)
    lo = round(birf.quantiles[h+1, 1, 1, 1], digits=3)
    hi = round(birf.quantiles[h+1, 1, 1, 3], digits=3)
    println("  h=$h: $med [$lo, $hi]")
end
```

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

**Reference**: Bańbura, Giannone & Reichlin (2010)

---

## References

### Minnesota Prior and BVAR

- Bańbura, Marta, Domenico Giannone, and Lucrezia Reichlin. 2010. "Large Bayesian Vector Auto Regressions." *Journal of Applied Econometrics* 25 (1): 71–92. [https://doi.org/10.1002/jae.1137](https://doi.org/10.1002/jae.1137)
- Carriero, Andrea, Todd E. Clark, and Massimiliano Marcellino. 2015. "Bayesian VARs: Specification Choices and Forecast Accuracy." *Journal of Applied Econometrics* 30 (1): 46–73. [https://doi.org/10.1002/jae.2272](https://doi.org/10.1002/jae.2272)
- Doan, Thomas, Robert Litterman, and Christopher Sims. 1984. "Forecasting and Conditional Projection Using Realistic Prior Distributions." *Econometric Reviews* 3 (1): 1–100. [https://doi.org/10.1080/07474938408800053](https://doi.org/10.1080/07474938408800053)
- Giannone, Domenico, Michele Lenza, and Giorgio E. Primiceri. 2015. "Prior Selection for Vector Autoregressions." *Review of Economics and Statistics* 97 (2): 436–451. [https://doi.org/10.1162/REST_a_00483](https://doi.org/10.1162/REST_a_00483)
- Kadiyala, K. Rao, and Sune Karlsson. 1997. "Numerical Methods for Estimation and Inference in Bayesian VAR-Models." *Journal of Applied Econometrics* 12 (2): 99–132. [https://doi.org/10.1002/(SICI)1099-1255(199703)12:2<99::AID-JAE429>3.0.CO;2-A](https://doi.org/10.1002/(SICI)1099-1255(199703)12:2<99::AID-JAE429>3.0.CO;2-A)
- Litterman, Robert B. 1986. "Forecasting with Bayesian Vector Autoregressions—Five Years of Experience." *Journal of Business & Economic Statistics* 4 (1): 25–38. [https://doi.org/10.1080/07350015.1986.10509491](https://doi.org/10.1080/07350015.1986.10509491)

### MCMC and Bayesian Inference

- Gelman, Andrew, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, and Donald B. Rubin. 2013. *Bayesian Data Analysis*. 3rd ed. Boca Raton, FL: CRC Press. ISBN 978-1-4398-4095-5.
- Hoffman, Matthew D., and Andrew Gelman. 2014. "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo." *Journal of Machine Learning Research* 15 (1): 1593–1623.
