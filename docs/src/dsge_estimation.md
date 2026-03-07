# [DSGE Estimation](@id dsge_estimation)

**MacroEconometricModels.jl** provides two paradigms for estimating the deep structural parameters of DSGE models. **Frequentist estimation** via `estimate_dsge` matches model-implied moments to data moments using Generalized Method of Moments (GMM) with four moment conditions. **Bayesian estimation** via `estimate_dsge_bayes` combines prior distributions with the likelihood function, targeting the posterior with Sequential Monte Carlo (SMC), SMC``^2``, or Random-Walk Metropolis-Hastings (RWMH). Both approaches build on the solution infrastructure documented in [DSGE Models](@ref dsge_page).

```julia
# Setup: define model and simulate data
using MacroEconometricModels, Random, Distributions
Random.seed!(42)

spec = @dsge begin
    parameters: β = 0.99, α = 0.36, δ = 0.025, ρ = 0.9, σ = 0.01
    endogenous: Y, C, K, A
    exogenous: ε_A

    Y[t] = A[t] * K[t-1]^α
    C[t] + K[t] = Y[t] + (1 - δ) * K[t-1]
    1 = β * (C[t] / C[t+1]) * (α * A[t+1] * K[t]^(α - 1) + 1 - δ)
    A[t] = ρ * A[t-1] + σ * ε_A[t]

    steady_state = begin
        A_ss = 1.0
        K_ss = (α * β / (1 - β * (1 - δ)))^(1 / (1 - α))
        Y_ss = K_ss^α
        C_ss = Y_ss - δ * K_ss
        [Y_ss, C_ss, K_ss, A_ss]
    end
end

sol = solve(spec)
Y_data = simulate(sol, 200)
```

## Quick Start

**Recipe 1: IRF matching GMM**

```julia
est = estimate_dsge(spec, Y_data, [:ρ, :σ];
                    method=:irf_matching, var_lags=4, irf_horizon=20)
report(est)
```

**Recipe 2: Bayesian SMC**

```julia
using Distributions
result = estimate_dsge_bayes(spec, Y_data, [0.9, 0.01];
    priors=Dict(:ρ => Beta(5, 2), :σ => InverseGamma(2, 0.01)),
    method=:smc, observables=[:Y, :C], n_smc=100)
report(result)
```

**Recipe 3: SMC``^2`` with projection solver**

```julia
result = estimate_dsge_bayes(spec, Y_data, [0.9, 0.01];
    priors=Dict(:ρ => Beta(5, 2), :σ => InverseGamma(2, 0.01)),
    method=:smc2, observables=[:Y], n_smc=200, n_particles=100,
    solver=:projection, solver_kwargs=(degree=5,))
report(result)
```

**Recipe 4: Bayesian IRFs with credible bands**

```julia
# Dual 68%/90% credible bands from posterior draws
birf = irf(result, 20; n_draws=100)
report(birf)
```

```julia
plot_result(birf)
```

**Recipe 5: Bayesian FEVD with credible bands**

```julia
bfevd = fevd(result, 20; n_draws=100)
report(bfevd)
```

**Recipe 6: Model comparison via Bayes factor**

```julia
log_bf = bayes_factor(result1, result2)
```

A positive value favors model 1; following Kass & Raftery (1995), ``2 \cdot \log \text{BF} > 6`` constitutes strong evidence.

---

## GMM Estimation

`estimate_dsge` estimates deep structural parameters by matching model-implied moments to data moments via Generalized Method of Moments. Four moment conditions are available: IRF matching, Euler equation GMM, simulated method of moments (SMM), and analytical GMM.

### IRF Matching (Christiano, Eichenbaum & Evans 2005)

The IRF matching estimator minimizes the distance between model-implied and empirical impulse response functions:

```math
\hat{\theta} = \arg\min_\theta \; \big(\Phi^m(\theta) - \Phi^d\big)' \, W \, \big(\Phi^m(\theta) - \Phi^d\big)
```

where:
- ``\Phi^m(\theta)`` is the vector of stacked model-implied IRFs at parameter ``\theta``
- ``\Phi^d`` is the vector of empirical IRFs estimated from a VAR(p) on the data
- ``W`` is the GMM weighting matrix (identity for one-step, inverse of ``\hat{\Omega}`` for two-step)

The procedure first estimates a reduced-form VAR on the observed data, computes Cholesky-identified IRFs, then searches over the structural parameter space to find the ``\theta`` that best replicates those empirical IRFs. This is the workhorse approach for medium-scale DSGE estimation in the frequency domain.

```julia
est_irf = estimate_dsge(spec, Y_data, [:ρ, :σ];
                    method=:irf_matching, var_lags=4, irf_horizon=20,
                    weighting=:two_step)
report(est_irf)
```

The two-step weighting uses the inverse of the estimated variance of the moment conditions from the first-step residuals. For pre-computed target IRFs (e.g., from a sign-identified VAR), pass them via `target_irfs` to bypass the internal VAR estimation.

### Euler Equation GMM (Hansen & Singleton 1982)

The Euler equation approach exploits the model's optimality conditions directly as moment conditions:

```math
E\Big[f\big(y_t, y_{t-1}, y_{t+1}, \varepsilon_t, \theta\big) \otimes z_t\Big] = 0
```

where:
- ``f(\cdot)`` is the vector of Euler equation residuals from the DSGE specification
- ``z_t`` is a vector of instruments (lagged endogenous variables)
- ``\otimes`` denotes the Kronecker product forming the interaction of residuals and instruments

This method does not require solving the model --- it evaluates the equilibrium conditions directly on the data. The instrument set consists of ``n_{\text{lags}}`` lags of the endogenous variables, producing ``n_{\text{eq}} \times n_{\text{vars}} \times n_{\text{lags}}`` moment conditions.

```julia
est_euler = estimate_dsge(spec, Y_data, [:ρ, :σ];
                    method=:euler_gmm, n_lags_instruments=4,
                    weighting=:two_step)
report(est_euler)
```

### Simulated Method of Moments

When analytical moments are unavailable, SMM matches sample moments from simulated data to their empirical counterparts. The simulation ratio ``S/T`` (default: 5) controls the trade-off between computational cost and simulation noise:

```math
\hat{\theta} = \arg\min_\theta \; \big(\hat{m}_S(\theta) - \hat{m}_T\big)' \, W \, \big(\hat{m}_S(\theta) - \hat{m}_T\big)
```

where:
- ``\hat{m}_S(\theta)`` is the vector of moments computed from a simulated path of length ``S``
- ``\hat{m}_T`` is the vector of sample moments from the observed data of length ``T``
- ``W`` accounts for both sampling and simulation uncertainty

```julia
est_smm = estimate_dsge(spec, Y_data, [:ρ, :σ];
                    method=:smm, sim_ratio=5)
report(est_smm)
```

The default moment function computes autocovariances at lag 1. Supply a custom `moments_fn` to target specific features of the data.

### Analytical GMM

Analytical GMM computes model-implied moments from the unconditional distribution without simulation. For linear models, the Lyapunov equation provides exact second moments via `analytical_moments`. For higher-order perturbation solutions, moments are computed from pruned simulations.

```julia
est_agmm = estimate_dsge(spec, Y_data, [:ρ, :σ];
                    method=:analytical_gmm, solve_method=:gensys,
                    solve_order=1, lags=1)
report(est_agmm)
```

Use `solve_method=:perturbation` with `solve_order=2` to match moments from second-order solutions, which capture risk premia and precautionary behavior absent from linear approximations.

### Hansen J-test

When the number of moment conditions exceeds the number of parameters, the Hansen (1982) J-statistic tests whether the over-identifying restrictions hold:

```math
J = T \cdot g(\hat{\theta})' \, \hat{W} \, g(\hat{\theta}) \sim \chi^2(q - p)
```

where:
- ``q`` is the number of moment conditions
- ``p`` is the number of estimated parameters
- ``g(\hat{\theta})`` is the sample moment vector evaluated at the estimated parameters

A large J-statistic (low p-value) indicates model misspecification --- the model cannot simultaneously satisfy all moment conditions.

```julia
est.J_stat     # Hansen J-statistic
est.J_pvalue   # p-value under chi-squared distribution
```

### GMM Keywords

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:irf_matching` | Moment condition: `:irf_matching`, `:euler_gmm`, `:smm`, `:analytical_gmm` |
| `weighting` | `Symbol` | `:two_step` | Weighting scheme: `:one_step`, `:two_step`, `:iterative`, `:cu` |
| `var_lags` | `Int` | `4` | VAR lag order for empirical IRFs (IRF matching only) |
| `irf_horizon` | `Int` | `20` | IRF horizon for matching (IRF matching only) |
| `target_irfs` | `ImpulseResponse` | `nothing` | Pre-computed target IRFs (bypasses internal VAR) |
| `n_lags_instruments` | `Int` | `4` | Instrument lags (Euler GMM only) |
| `sim_ratio` | `Int` | `5` | Simulation-to-data length ratio (SMM only) |
| `moments_fn` | `Function` | autocovariance | Custom moment function (SMM only) |
| `bounds` | `ParameterTransform` | `nothing` | Parameter bounds via `ParameterTransform` |
| `solve_method` | `Symbol` | `:gensys` | DSGE solver for analytical GMM |
| `solve_order` | `Int` | `1` | Perturbation order for analytical GMM |
| `auto_lags` | `Vector{Int}` | `[1]` | Autocovariance lags for analytical GMM |
| `observable_indices` | `Vector{Int}` | `nothing` | Observable variable indices for analytical GMM |

### GMM Return Value (`DSGEEstimation{T}`)

| Field | Type | Description |
|-------|------|-------------|
| `theta` | `Vector{T}` | Estimated parameter vector |
| `vcov` | `Matrix{T}` | Asymptotic variance-covariance matrix |
| `param_names` | `Vector{Symbol}` | Parameter names |
| `method` | `Symbol` | Estimation method used |
| `J_stat` | `T` | Hansen J-statistic for over-identification |
| `J_pvalue` | `T` | p-value of the J-test |
| `solution` | Union type | Model solution at estimated parameters |
| `converged` | `Bool` | Optimizer convergence flag |
| `spec` | `DSGESpec{T}` | Back-reference to model specification |

**StatsAPI interface**: `coef(est)` returns the parameter vector, `vcov(est)` the covariance matrix, `stderror(est)` the standard errors, and `dof(est)` the number of estimated parameters.

---

## Bayesian Estimation

Bayesian estimation combines prior distributions ``\pi(\theta)`` with the data likelihood ``\mathcal{L}(Y|\theta)`` to characterize the posterior distribution:

```math
p(\theta | Y) \propto \mathcal{L}(Y | \theta) \cdot \pi(\theta)
```

where:
- ``p(\theta | Y)`` is the posterior distribution over structural parameters
- ``\mathcal{L}(Y | \theta)`` is the likelihood function (Kalman filter for linear models, particle filter for nonlinear models)
- ``\pi(\theta)`` is the prior distribution encoding economic beliefs

Three sampling algorithms target the posterior: Sequential Monte Carlo (SMC), SMC``^2``, and Random-Walk Metropolis-Hastings (RWMH). SMC is the recommended default --- it produces the marginal likelihood as a by-product, handles multimodal posteriors, and parallelizes naturally.

### Prior Specification

Priors are specified as a `Dict{Symbol, Distribution}` mapping parameter names to distributions from `Distributions.jl`. Parameter bounds are inferred automatically from the distribution support:

```julia
priors = Dict(
    :ρ => Beta(5, 2),           # persistence: mean ≈ 0.71, support [0,1]
    :σ => InverseGamma(2, 0.01) # shock std: positive, heavy-tailed
)
nothing # hide
```

| Distribution | Support | Typical Use |
|---|---|---|
| `Beta(a, b)` | ``[0, 1]`` | Persistence, autocorrelation |
| `Gamma(a, b)` | ``[0, \infty)`` | Adjustment costs, elasticities |
| `InverseGamma(a, b)` | ``[0, \infty)`` | Shock standard deviations |
| `Normal(mu, sigma)` | ``(-\infty, \infty)`` | Unbounded parameters |
| `Uniform(a, b)` | ``[a, b]`` | Weakly informative, bounded |

### Sequential Monte Carlo (Herbst & Schorfheide 2014)

**SMC** draws from a sequence of tempered distributions that bridge the prior to the posterior:

```math
p_\phi(\theta) \propto \mathcal{L}(Y|\theta)^\phi \; \pi(\theta), \qquad \phi \in [0, 1]
```

where:
- At ``\phi = 0``, particles are distributed according to the prior ``\pi(\theta)``
- At ``\phi = 1``, particles approximate the full posterior ``p(\theta|Y)``
- The tempering schedule ``0 = \phi_0 < \phi_1 < \cdots < \phi_S = 1`` bridges between the two

The algorithm proceeds in six steps:

1. **Initialize**: Draw ``N`` particles from the prior: ``\theta^{(i)} \sim \pi(\theta)``
2. **Temper**: Set the adaptive schedule ``0 = \phi_0 < \phi_1 < \cdots < \phi_S = 1`` targeting a fixed ESS fraction
3. **Reweight**: At stage ``s``, compute incremental weights ``w^{(i)} \propto \mathcal{L}(Y|\theta^{(i)})^{\phi_s - \phi_{s-1}}``
4. **Resample**: If ESS falls below the threshold, resample particles via systematic resampling
5. **Mutate**: Apply ``n_{\text{mh}}`` Metropolis-Hastings steps with proposal ``q(\theta^*|\theta) = \mathcal{N}(\theta, \hat{\Sigma})``
6. **Marginal likelihood**: Estimate the normalizing constant: ``\hat{p}(Y) = \prod_{s=1}^{S} \frac{1}{N} \sum_{i=1}^{N} w_s^{(i)}``

The adaptive tempering schedule selects ``\phi_s`` to maintain the effective sample size at the target fraction (default: 50%). This avoids both degenerate weights (too large a step) and unnecessary computation (too small a step).

```julia
result_smc = estimate_dsge_bayes(spec, Y_data, [0.9, 0.01];
    priors=Dict(:ρ => Beta(5, 2), :σ => InverseGamma(2, 0.01)),
    method=:smc, observables=[:Y, :C], n_smc=100)
report(result_smc)
```

The likelihood is evaluated via the Kalman filter, which is exact for linear state-space models produced by [Linear Solvers](@ref dsge_linear) (`:gensys`, `:blanchard_kahn`, `:klein`).

### SMC``^2`` (Chopin, Jacob & Papaspiliopoulos 2013)

**SMC``^2``** nests a particle filter inside the outer SMC loop. At each mutation step, the likelihood ``\mathcal{L}(Y|\theta^*)`` is evaluated by running a bootstrap particle filter rather than the Kalman filter. This enables Bayesian estimation of nonlinear DSGE models solved with [Nonlinear Methods](@ref dsge_nonlinear) --- perturbation (order ``\geq 2``), Chebyshev projection, or policy function iteration --- where the Kalman filter approximation breaks down.

The particle filter estimates the likelihood as:

```math
\hat{\mathcal{L}}(Y|\theta) = \prod_{t=1}^{T} \frac{1}{M} \sum_{j=1}^{M} w_t^{(j)}
```

where ``M`` is the number of inner particles (set via `n_particles`) and ``w_t^{(j)}`` are the importance weights at time ``t``.

```julia
result = estimate_dsge_bayes(spec, Y_data, [0.9, 0.01];
    priors=Dict(:ρ => Beta(5, 2), :σ => InverseGamma(2, 0.01)),
    method=:smc2, observables=[:Y],
    n_smc=200, n_particles=100,
    solver=:projection, solver_kwargs=(degree=5,))
report(result)
```

The mutation step uses Conditional SMC (CSMC) to update both the parameter ``\theta`` and the latent state trajectory jointly, maintaining a valid reference trajectory that prevents particle degeneracy.

!!! note "Technical Note"
    For linear solvers (`:gensys`, `:blanchard_kahn`, `:klein`), use `:smc` with the Kalman filter likelihood --- it is exact and orders of magnitude faster than the particle filter. Reserve `:smc2` for nonlinear solvers (`:projection`, `:pfi`, `:perturbation` with `order >= 2`) where the Kalman approximation breaks down.

### Delayed Acceptance (Christen & Fox 2005)

**Two-stage delayed acceptance** pre-screens Metropolis-Hastings proposals with a cheap particle filter before running the expensive Conditional SMC. This preserves detailed balance while avoiding wasted computation on proposals that would be rejected.

**Stage 1** (screening): Accept the proposal ``\theta^*`` with probability

```math
\alpha_1 = \min\!\Big(1, \; \exp\big[\phi \cdot \hat{\ell}_{\text{screen}}(\theta^*) + \log\pi(\theta^*) - \phi \cdot \hat{\ell}_{\text{screen}}(\theta) - \log\pi(\theta)\big]\Big)
```

**Stage 2** (correction, only if Stage 1 accepts): Accept with probability

```math
\alpha_2 = \min\!\Big(1, \; \exp\big[\phi \cdot \big(\hat{\ell}_{\text{full}}(\theta^*) - \hat{\ell}_{\text{screen}}(\theta^*)\big) - \phi \cdot \big(\hat{\ell}_{\text{full}}(\theta) - \hat{\ell}_{\text{screen}}(\theta)\big)\big]\Big)
```

where:
- ``\hat{\ell}_{\text{screen}}`` is the log-likelihood from a bootstrap PF with ``n_{\text{screen}}`` particles (cheap)
- ``\hat{\ell}_{\text{full}}`` is the log-likelihood from CSMC with ``n_{\text{particles}}`` particles (expensive)
- ``\phi`` is the current tempering parameter

The product ``\alpha_1 \cdot \alpha_2`` equals the standard MH acceptance probability in expectation, so the chain targets the exact posterior. The computational savings come from rejecting bad proposals cheaply at Stage 1 without ever running the full CSMC.

```julia
result = estimate_dsge_bayes(spec, Y_data, [0.9, 0.01];
    priors=Dict(:ρ => Beta(5, 2), :σ => InverseGamma(2, 0.01)),
    method=:smc2, observables=[:Y],
    n_smc=200, n_particles=500,
    solver=:projection, solver_kwargs=(degree=5,),
    delayed_acceptance=true, n_screen=200)
```

!!! warning "Particle Count Tuning"
    Set ``n_{\text{screen}}`` large enough that the screening likelihood is informative (typically 100--300), but small relative to ``n_{\text{particles}}`` (which should be 500+). If ``n_{\text{screen}} \approx n_{\text{particles}}``, there is no computational benefit.

### Random-Walk Metropolis-Hastings

Standard Random-Walk Metropolis-Hastings with adaptive proposal scaling targeting the optimal 23.4% acceptance rate (Roberts & Rosenthal 2001). The proposal is a multivariate normal centered at the current draw:

```math
\theta^* \sim \mathcal{N}\!\big(\theta^{(s)}, \; c \cdot \hat{\Sigma}\big)
```

where:
- ``\hat{\Sigma}`` is the estimated posterior covariance (initialized from the prior, updated during burnin)
- ``c`` is the step-size scalar adapted to target 23.4% acceptance

```julia
result_mh = estimate_dsge_bayes(spec, Y_data, [0.9, 0.01];
    priors=Dict(:ρ => Beta(5, 2), :σ => InverseGamma(2, 0.01)),
    method=:mh, observables=[:Y, :C],
    n_draws=100, burnin=50)
report(result_mh)
```

RWMH is simple to implement and diagnose but converges slowly for high-dimensional parameter spaces. For models with more than 5--10 parameters, SMC is strongly preferred.

### Bayesian Keywords

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `priors` | `Dict{Symbol, Distribution}` | required | Prior distributions keyed by parameter name |
| `method` | `Symbol` | `:smc` | Sampling method: `:smc`, `:smc2`, or `:mh` |
| `observables` | `Vector{Symbol}` | `spec.endog` | Observed endogenous variables |
| `n_smc` | `Int` | `5000` | Number of SMC/SMC``^2`` particles |
| `n_particles` | `Int` | `500` | Number of PF particles (SMC``^2`` only) |
| `n_mh_steps` | `Int` | `1` | MH mutation steps per SMC stage |
| `n_draws` | `Int` | `10000` | Total draws for RWMH (including burnin) |
| `burnin` | `Int` | `5000` | Burnin draws for RWMH |
| `ess_target` | `Float64` | `0.5` | Target ESS fraction for adaptive tempering |
| `measurement_error` | `Vector{<:Real}` | `nothing` | Measurement error standard deviations |
| `solver` | `Symbol` | `:gensys` | DSGE solver method |
| `solver_kwargs` | `NamedTuple` | `()` | Additional solver keyword arguments |
| `delayed_acceptance` | `Bool` | `false` | Two-stage delayed acceptance (SMC``^2`` only) |
| `n_screen` | `Int` | `200` | Screening PF particles (delayed acceptance only) |

### Posterior Analysis

After estimation, five functions extract information from the `BayesianDSGE` result:

```julia
# Posterior summary: mean, median, std, 95% credible interval per parameter
ps = posterior_summary(result_smc)
ps[:ρ][:mean]       # posterior mean of ρ
```

```julia
ps[:σ][:ci_lower]   # lower bound of 95% CI for σ
```

```julia
# Log marginal likelihood (model evidence)
ml = marginal_likelihood(result_smc)
```

```julia
# Bayes factor: log p(Y|M₁) - log p(Y|M₂)
log_bf = bayes_factor(result1, result2)
```

```julia
# Prior vs posterior comparison table
tbl = prior_posterior_table(result_smc)
nothing # hide
```

```julia
# Posterior predictive simulation
Y_pred = posterior_predictive(result_smc, 10; T_periods=50)
size(Y_pred)
```

`posterior_summary` returns a `Dict{Symbol, Dict{Symbol, T}}` with keys `:mean`, `:median`, `:std`, `:ci_lower` (2.5th percentile), and `:ci_upper` (97.5th percentile) for each parameter. `prior_posterior_table` returns a vector of named tuples suitable for tabular display, comparing prior and posterior moments side by side. `posterior_predictive` draws `n_sim` parameter vectors from the posterior, solves the model at each, and simulates forward, returning an `n_sim x T_periods x n_vars` array of simulated paths.

### Posterior IRFs and FEVD (Herbst & Schorfheide 2015)

Bayesian DSGE estimation quantifies parameter uncertainty. `irf` and `fevd` propagate this uncertainty into impulse responses and variance decompositions by re-solving the model at posterior parameter draws and computing pointwise credible bands.

For each of `n_draws` randomly selected posterior draws, the model is re-solved at those parameter values and the analytical IRF (or FEVD) is computed. The results are stacked and summarized with pointwise quantile bands. The default quantiles ``[0.05, 0.16, 0.84, 0.95]`` produce dual 68% and 90% credible bands --- the standard reporting convention in the Bayesian DSGE literature.

```julia
# Bayesian IRFs with dual credible bands
birf_smc = irf(result_smc, 20; n_draws=100)
report(birf_smc)
```

```julia
plot_result(birf_smc)
```

```julia
# Bayesian FEVD
bfevd_smc = fevd(result_smc, 20; n_draws=100)
report(bfevd_smc)
```

```julia
# Custom quantiles (90% band only)
birf_90 = irf(result_smc, 20; n_draws=100, quantiles=[0.05, 0.95])
nothing # hide
```

Both methods return `BayesianImpulseResponse{T}` and `BayesianFEVD{T}` respectively --- the same types used by Bayesian VAR, so all existing `report()`, `plot_result()`, `table()`, and `cumulative_irf()` infrastructure works automatically.

Draws that produce indeterminate or explosive solutions are silently skipped. If all draws fail, an error is raised.

### Posterior Predictive Simulation

`simulate` draws from the posterior predictive distribution with credible bands. For each posterior parameter draw, the model is re-solved and simulated forward `T_periods` periods:

```julia
bsim = simulate(result_smc, 50; n_draws=100)
report(bsim)
```

The result is a `BayesianDSGESimulation{T}` containing the pointwise median, quantile bands, and all raw simulation paths.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `n_draws` | `Int` | `200` | Number of posterior draws to subsample |
| `quantiles` | `Vector{<:Real}` | `[0.05, 0.16, 0.84, 0.95]` | Quantile levels for credible bands |
| `solver` | `Symbol` | `:gensys` | DSGE solver for re-solving at each draw |
| `solver_kwargs` | `NamedTuple` | `()` | Additional solver keyword arguments |

### `BayesianDSGESimulation{T}` Return Value

| Field | Type | Description |
|-------|------|-------------|
| `quantiles` | `Array{T,3}` | ``T \times n_{\text{vars}} \times n_q`` pointwise quantile bands |
| `point_estimate` | `Matrix{T}` | ``T \times n_{\text{vars}}`` posterior median |
| `T_periods` | `Int` | Number of simulation periods |
| `variables` | `Vector{String}` | Variable names |
| `quantile_levels` | `Vector{T}` | Quantile levels used |
| `all_paths` | `Array{T,3}` | ``n_{\text{draws}} \times T \times n_{\text{vars}}`` raw simulation paths |

### Bayesian Return Value (`BayesianDSGE{T}`)

| Field | Type | Description |
|-------|------|-------------|
| `theta_draws` | `Matrix{T}` | ``N \times p`` posterior parameter draws |
| `log_posterior` | `Vector{T}` | Log posterior at each draw |
| `param_names` | `Vector{Symbol}` | Parameter names |
| `priors` | `DSGEPrior{T}` | Prior specification |
| `log_marginal_likelihood` | `T` | Log marginal likelihood estimate |
| `method` | `Symbol` | `:smc`, `:smc2`, or `:rwmh` |
| `acceptance_rate` | `T` | MH/CSMC acceptance rate |
| `ess_history` | `Vector{T}` | ESS at each tempering stage (empty for RWMH) |
| `phi_schedule` | `Vector{T}` | Tempering schedule ``\phi_0, \ldots, \phi_S`` (empty for RWMH) |
| `spec` | `DSGESpec{T}` | Back-reference to model specification |
| `solution` | Union type | Model solution at posterior mean |
| `state_space` | Union type | State-space representation at posterior mean |

**StatsAPI interface**: `coef(result)` returns the posterior mean parameter vector.

---

## Complete Example

This example estimates the RBC model's persistence and shock volatility using both GMM and Bayesian methods, then compares the results:

```julia
using MacroEconometricModels, Random, Distributions
Random.seed!(42)

# 1. Specify the RBC model
spec = @dsge begin
    parameters: β = 0.99, α = 0.36, δ = 0.025, ρ = 0.9, σ = 0.01
    endogenous: Y, C, K, A
    exogenous: ε_A

    Y[t] = A[t] * K[t-1]^α
    C[t] + K[t] = Y[t] + (1 - δ) * K[t-1]
    1 = β * (C[t] / C[t+1]) * (α * A[t+1] * K[t]^(α - 1) + 1 - δ)
    A[t] = ρ * A[t-1] + σ * ε_A[t]

    steady_state = begin
        A_ss = 1.0
        K_ss = (α * β / (1 - β * (1 - δ)))^(1 / (1 - α))
        Y_ss = K_ss^α
        C_ss = Y_ss - δ * K_ss
        [Y_ss, C_ss, K_ss, A_ss]
    end
end

# 2. Simulate data from the true model
sol = solve(spec)
Y_data = simulate(sol, 200)

# 3. Frequentist: IRF matching GMM
est_gmm = estimate_dsge(spec, Y_data, [:ρ, :σ];
                         method=:irf_matching, var_lags=4, irf_horizon=20)
report(est_gmm)

# 4. Bayesian: SMC estimation
priors = Dict(:ρ => Beta(5, 2), :σ => InverseGamma(2, 0.01))
est_bayes = estimate_dsge_bayes(spec, Y_data, [0.9, 0.01];
    priors=priors, method=:smc, observables=[:Y, :C], n_smc=2000)
report(est_bayes)

# 5. Posterior analysis
ps = posterior_summary(est_bayes)
tbl = prior_posterior_table(est_bayes)
ml = marginal_likelihood(est_bayes)

# 6. Bayesian IRFs with credible bands
birf = irf(est_bayes, 20; n_draws=100)
report(birf)

# 7. Bayesian FEVD
bfevd = fevd(est_bayes, 20; n_draws=100)
report(bfevd)

# 8. Posterior predictive simulation
bsim = simulate(est_bayes, 200; n_draws=100)
report(bsim)
```

The GMM point estimates and Bayesian posterior means should cluster near the true values ``\rho = 0.9`` and ``\sigma = 0.01``. The Bayesian IRFs propagate parameter uncertainty into impulse response bands, and the marginal likelihood enables formal model comparison.

---

## Common Pitfalls

1. **Wrong steady state**: If the steady state is incorrect, all estimation methods fail silently --- the model solves to a nonsensical equilibrium, and the optimizer converges to economically meaningless parameter values. Always verify `compute_steady_state` and check that the solution satisfies `is_determined(sol)` before estimation.

2. **Indeterminate model at prior draws**: SMC initializes particles from the prior. If many prior draws produce indeterminate or explosive models, the likelihood evaluates to ``-\infty`` and particles are wasted. Tighten priors to concentrate mass on the determinacy region, or increase `n_smc` to compensate.

3. **Too few SMC particles**: For posteriors with ridges, multimodality, or strong correlations, ``n_{\text{smc}} = 1000`` may not suffice. Start with 5000+ and reduce only after confirming that the ESS history remains above the target and that repeated runs produce consistent marginal likelihood estimates.

4. **Observable mismatch**: The `observables` keyword specifies which endogenous variables in the model correspond to columns in the data matrix, in order. Mismatched dimensions or incorrect ordering produce nonsensical likelihood values. The number of observables must equal the number of data columns.

5. **Solver choice for Bayesian estimation**: Use `:smc` with `:gensys` (or `:blanchard_kahn`, `:klein`) for linear models --- the Kalman filter provides the exact likelihood. Use `:smc2` with `:projection` or `:pfi` for nonlinear models where the particle filter is necessary. Using `:smc` with a nonlinear solver silently falls back to a first-order Kalman approximation that ignores higher-order dynamics.

---

## References

- An, S., & Schorfheide, F. (2007). Bayesian Analysis of DSGE Models. *Econometric Reviews*, 26(2-4), 113-172. [DOI](https://doi.org/10.1080/07474930701220071)

- Chopin, N., Jacob, P. E., & Papaspiliopoulos, O. (2013). SMC``^2``: An Efficient Algorithm for Sequential Analysis of State Space Models. *Journal of the Royal Statistical Society: Series B*, 75(3), 397-426. [DOI](https://doi.org/10.1111/j.1467-9868.2012.01046.x)

- Christen, J. A., & Fox, C. (2005). Markov Chain Monte Carlo Using an Approximation. *Journal of Computational and Graphical Statistics*, 14(4), 795-810. [DOI](https://doi.org/10.1198/106186005X76983)

- Christiano, L. J., Eichenbaum, M., & Evans, C. L. (2005). Nominal Rigidities and the Dynamic Effects of a Shock to Monetary Policy. *Journal of Political Economy*, 113(1), 1-45. [DOI](https://doi.org/10.1086/426038)

- Hansen, L. P. (1982). Large Sample Properties of Generalized Method of Moments Estimators. *Econometrica*, 50(4), 1029-1054. [DOI](https://doi.org/10.2307/1912775)

- Hansen, L. P., & Singleton, K. J. (1982). Generalized Instrumental Variables Estimation of Nonlinear Rational Expectations Models. *Econometrica*, 50(5), 1269-1286. [DOI](https://doi.org/10.2307/1911873)

- Herbst, E., & Schorfheide, F. (2014). Sequential Monte Carlo Sampling for DSGE Models. *Journal of Applied Econometrics*, 29(7), 1073-1098. [DOI](https://doi.org/10.1002/jae.2397)

- Kass, R. E., & Raftery, A. E. (1995). Bayes Factors. *Journal of the American Statistical Association*, 90(430), 773-795. [DOI](https://doi.org/10.1080/01621459.1995.10476572)

- Roberts, G. O., & Rosenthal, J. S. (2001). Optimal Scaling for Various Metropolis-Hastings Algorithms. *Statistical Science*, 16(4), 351-367. [DOI](https://doi.org/10.1214/ss/1015346320)
