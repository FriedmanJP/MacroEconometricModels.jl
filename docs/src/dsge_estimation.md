# [DSGE Estimation](@id dsge_estimation)

**MacroEconometricModels.jl** provides two paradigms for estimating the deep structural parameters of DSGE models. **Frequentist estimation** via `estimate_dsge` matches model-implied moments to data moments using Generalized Method of Moments (GMM) with four moment conditions. **Bayesian estimation** via `estimate_dsge_bayes` combines prior distributions with the likelihood function, targeting the posterior with Sequential Monte Carlo (SMC), SMC``^2``, or Random-Walk Metropolis-Hastings (RWMH). Both approaches build on the solution infrastructure documented in [DSGE Models](@ref dsge_page).

```@setup dsge_estimation
using MacroEconometricModels, Random, Distributions
Random.seed!(42)
# Small two-shock DSGE: one shock per endogenous variable, so a bivariate VAR on
# the simulated data delivers IRF targets of exactly the model's dimensions.
spec = @dsge begin
    parameters: rho = 0.9, phi = 1.5, psi = 0.5
    endogenous: y, i
    exogenous: e_y, e_i
    y[t] = rho * y[t-1] + e_y[t]
    i[t] = phi * y[t] + psi * e_i[t]
end
spec = compute_steady_state(spec)
sol = solve(spec; method=:gensys)
Y_data = simulate(sol, 200)
y_obs = Y_data[:, [1]]  # observed series for :y — data must be T × n_obs
```

## Quick Start

**Recipe 1: IRF matching GMM**

```@example dsge_estimation
est = estimate_dsge(spec, Y_data, [:rho];
                    method=:irf_matching, var_lags=4, irf_horizon=20)
report(est)
```

**Recipe 2: Bayesian SMC**

```@example dsge_estimation
fit_smc = estimate_dsge_bayes(spec, y_obs, [0.9];
    priors=Dict(:rho => Beta(5, 2)),
    method=:smc, observables=[:y], n_smc=50)
report(fit_smc)
```

**Recipe 3: SMC``^2`` with projection solver**

```julia
fit_smc2 = estimate_dsge_bayes(spec, y_obs, [0.9];
    priors=Dict(:rho => Beta(5, 2)),
    method=:smc2, observables=[:y], n_smc=200, n_particles=100,
    solver=:projection, solver_kwargs=(degree=5,))
report(fit_smc2)
```

**Recipe 4: Bayesian IRFs with credible bands**

```@example dsge_estimation
# Dual 68%/90% credible bands from posterior draws
birf = irf(fit_smc, 20; n_draws=10)
report(birf)
```

```julia
plot_result(birf)
```

```@raw html
<iframe src="../assets/plots/dsge_bayes_irf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

**Recipe 5: Bayesian FEVD with credible bands**

```@example dsge_estimation
bfevd = fevd(fit_smc, 20; n_draws=10)
report(bfevd)
```

**Recipe 6: Model comparison via Bayes factor**

```@example dsge_estimation
# Alternative model: a much flatter persistence prior
fit_alt = estimate_dsge_bayes(spec, y_obs, [0.9];
    priors=Dict(:rho => Beta(2, 2)),
    method=:smc, observables=[:y], n_smc=50)
round(bayes_factor(fit_smc, fit_alt); digits=2)
```

A positive value favors the first model; following Kass & Raftery (1995), ``2 \cdot \log \text{BF} > 6`` constitutes strong evidence.

---

## [GMM Estimation](@id dsge_est_gmm)

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

The procedure first estimates a reduced-form VAR on the observed data, computes Cholesky-identified IRFs with bootstrap draws, then searches over the structural parameter space to find the ``\theta`` that best replicates those empirical IRFs. This is the workhorse approach for medium-scale DSGE estimation in the frequency domain.

!!! warning "Model and VAR IRFs must have the same dimensions"
    The stacked model IRF is ``H \times n \times n_{\text{shocks}}`` while the VAR target is ``H \times n \times n``. A model with fewer structural shocks than the VAR has variables produces a dimension mismatch: every candidate ``\theta`` returns the same penalty vector, the objective is flat, and the "estimate" is just the starting value. Match the shock count to the observed variables, or supply an explicitly sliced `target_irfs`.

```@example dsge_estimation
est_irf = estimate_dsge(spec, Y_data, [:rho];
                    method=:irf_matching, var_lags=4, irf_horizon=20,
                    weighting=:two_step)
report(est_irf)
```

Estimating one parameter against 80 stacked IRF ordinates leaves 79 over-identifying restrictions, and the Hansen J-test does not reject them: the model reproduces the empirical impulse responses within their bootstrap sampling variability. The persistence estimate sits slightly below the data-generating ``\rho = 0.9`` because the VAR targets are themselves estimated with small-sample bias toward zero. Its standard error is small precisely because 80 moments — the full IRF surface out to horizon 20 — bear on a single parameter.

Under `weighting=:two_step` (also spelled `:efficient` or `:optimal`) the weighting matrix is ``W = \hat\Omega^{-1}``, the inverse bootstrap covariance of the empirical IRF targets, and only then is the reported J-statistic ``\chi^2``. `:diagonal` (equivalently `:cee`) uses ``W = \text{diag}(\hat\Omega)^{-1}``, the original CEE choice, and `:identity` uses ``W = I``; both suppress the J-test. For pre-computed target IRFs (e.g., from a sign-identified VAR), pass them via `target_irfs` to bypass the internal VAR estimation.

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

```@example dsge_estimation
est_euler = estimate_dsge(spec, Y_data, [:rho];
                    method=:euler_gmm, n_lags_instruments=4,
                    weighting=:two_step)
report(est_euler)
```

Two equations interacted with eight instruments (four lags of two variables) give 16 moment conditions for one parameter, so the J-test carries 15 degrees of freedom. Because the residuals are evaluated on the data rather than on a solved model, this estimator never calls the solver — it is the cheapest of the four, and the only one that remains available when the model does not solve at some candidate parameters.

### Simulated Method of Moments (Lee & Ingram 1991; Ruge-Murcia 2012)

When analytical moments are unavailable, SMM matches sample moments from simulated data to their empirical counterparts. The simulation ratio ``S/T`` (default: 5) controls the trade-off between computational cost and simulation noise:

```math
\hat{\theta} = \arg\min_\theta \; \big(\hat{m}_S(\theta) - \hat{m}_T\big)' \, W \, \big(\hat{m}_S(\theta) - \hat{m}_T\big)
```

where:
- ``\hat{m}_S(\theta)`` is the vector of moments computed from a simulated path of length ``S``
- ``\hat{m}_T`` is the vector of sample moments from the observed data of length ``T``
- ``W`` accounts for both sampling and simulation uncertainty

```@example dsge_estimation
est_smm = estimate_dsge(spec, Y_data, [:rho];
                    method=:smm, sim_ratio=5)
report(est_smm)
```

The default moment vector is `autocovariance_moments(d; lags=1)`: the three distinct second moments of the bivariate system plus two lag-1 autocovariances, five moments against one parameter. Supply a custom `moments_fn` — together with a matching `contributions_fn`, see [Generalized & Simulated Method of Moments](@ref gmm_page) — to target other features of the data. The asymptotic covariance carries the ``(1 + 1/\tau)`` simulation-noise inflation factor, so raising `sim_ratio` buys precision at the cost of simulation time.

### Analytical GMM

Analytical GMM computes model-implied moments from the unconditional distribution without simulation. For linear models, the Lyapunov equation provides exact second moments via `analytical_moments`. For higher-order perturbation solutions, moments are computed from pruned simulations.

```@example dsge_estimation
est_agmm = estimate_dsge(spec, Y_data, [:rho];
                    method=:analytical_gmm, solve_method=:gensys,
                    solve_order=1, lags=1)
report(est_agmm)
```

Analytical GMM matches the same five second moments as SMM but computes the model side exactly from the Lyapunov equation, so it carries no simulation noise and drives the moment discrepancy essentially to zero — the J-statistic here is 0.006 across five moments. Because the model moments are deterministic, all sampling uncertainty enters through the *data* moments, and the reported covariance is the classical minimum-distance sandwich:

```math
V(\hat{\theta}) = (G'G)^{-1} G' \, \hat{\Omega} \, G \, (G'G)^{-1} / T
```

where:
- ``G = \partial g / \partial \theta`` is the Jacobian of the moment discrepancy at ``\hat\theta``
- ``\hat{\Omega}`` is the HAC long-run covariance of the per-observation data-moment contributions
- ``T`` is the number of observations

The standard errors this delivers are genuine: ``\hat\rho = 0.897`` carries a standard error of 0.027 on 200 observations, and refitting the same model on 3200 observations shrinks it to 0.007 — the ``1/\sqrt{T}`` rate. What stays `NaN` is the J p-value, because identity weighting is not the efficient choice and the statistic has no ``\chi^2`` limit. Use `solve_method=:perturbation` with `solve_order=2` to match moments from second-order solutions, which capture risk premia and precautionary behavior absent from linear approximations; that path switches the moment vector to the richer means-plus-product-moments format, reads its lags from `auto_lags` rather than `lags`, and reports a `NaN` covariance, since the per-observation contributions the sandwich needs are not defined for that format.

### Hansen J-test

When the number of moment conditions exceeds the number of parameters, the Hansen (1982) J-statistic tests whether the over-identifying restrictions hold:

```math
J = n \cdot g(\hat{\theta})' \, \hat{W} \, g(\hat{\theta}) \; \xrightarrow{d} \; \chi^2(q - p)
```

where:
- ``q`` is the number of moment conditions
- ``p`` is the number of estimated parameters
- ``g(\hat{\theta})`` is the sample moment vector evaluated at the estimated parameters
- ``n`` is the number of observations contributing to ``g``

A large J-statistic (low p-value) indicates model misspecification --- the model cannot simultaneously satisfy all moment conditions. The ``\chi^2`` limit requires efficient weighting ``\hat{W} = \hat{\Omega}^{-1}``; under any other weighting the statistic is a weighted sum of ``\chi^2(1)`` variables and `J_pvalue` is `NaN`. IRF matching is a minimum-distance estimator whose ``\hat\Omega`` already carries the sampling scale of the targets, so its statistic is ``J = g' \hat\Omega^{-1} g`` with no ``n`` factor.

```@example dsge_estimation
(J = round(est.J_stat; digits=3), p = round(est.J_pvalue; digits=3))
```

### GMM Keywords

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:irf_matching` | Moment condition: `:irf_matching`, `:euler_gmm`, `:smm`, `:analytical_gmm` |
| `weighting` | `Symbol` | `nothing` | Weighting scheme; `nothing` takes the method default, `:two_step` for the sample-moment methods and `:identity` for `:analytical_gmm` (see the note below) |
| `var_lags` | `Int` | `4` | VAR lag order for empirical IRFs (IRF matching only) |
| `irf_horizon` | `Int` | `20` | IRF horizon for matching (IRF matching only) |
| `target_irfs` | `ImpulseResponse` | `nothing` | Pre-computed target IRFs (bypasses internal VAR) |
| `n_boot` | `Int` | `200` | Bootstrap replications for the target-IRF covariance (IRF matching only) |
| `n_lags_instruments` | `Int` | `4` | Instrument lags (Euler GMM only) |
| `sim_ratio` | `Int` | `5` | Simulation-to-data length ratio (SMM only) |
| `burn` | `Int` | `100` | Burn-in periods discarded per simulated path (SMM only) |
| `moments_fn` | `Function` | `d -> autocovariance_moments(d; lags=1)` | Custom moment function (SMM only) |
| `contributions_fn` | `Function` | `d -> autocovariance_moment_contributions(d; lags=1)` | Per-observation moment contributions matching `moments_fn` (SMM only) |
| `rng` | `AbstractRNG` | `Random.default_rng()` | Random number generator (SMM only) |
| `bounds` | `ParameterTransform` | `nothing` | Parameter bounds via `ParameterTransform` (SMM and analytical GMM) |
| `solve_method` | `Symbol` | `:gensys` | DSGE solver for analytical GMM |
| `solve_order` | `Int` | `1` | Perturbation order for analytical GMM |
| `lags` | `Int` | `1` | Autocovariance lags when `solve_order=1` (analytical GMM) |
| `auto_lags` | `Vector{Int}` | `[1]` | Autocovariance lags when `solve_order ≥ 2` (analytical GMM) |
| `observable_indices` | `Vector{Int}` | `nothing` | Observable variable indices for analytical GMM |

!!! note "Weighting vocabularies differ by method"
    `:euler_gmm` and `:smm` route through `estimate_gmm`/`estimate_smm` and accept
    `:identity`, `:optimal`, `:two_step`, and `:iterated` (SMM: `:identity` or `:two_step`).
    `:irf_matching` accepts `:two_step`/`:efficient`/`:optimal`, `:diagonal`/`:cee`, and
    `:identity`. `:analytical_gmm` supports `:identity` only — its moment residual is a
    single ``1 \times q`` vector, so HAC and two-step weighting are degenerate. An explicit
    non-identity request warns and falls back to identity; the default call is silent.

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
| `spec` | `ModelSpec{T}` | Back-reference to model specification |

`J_stat` and `J_pvalue` are both `NaN` for IRF matching whenever the weighting is not efficient (`:diagonal`, `:cee`, `:identity`, or the identity fallback taken when no bootstrap draws are available); `J_pvalue` alone is `NaN` for the other methods under identity weighting.

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

```@example dsge_estimation
priors = Dict(
    :rho => Beta(5, 2)          # persistence: mean ≈ 0.71, support [0,1]
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

### Porting Dynare Priors

Published priors are almost always declared in Dynare's **(mean, std)** convention. `dynare_prior` converts them into correctly parameterized `Distributions` objects — solving the moment-matching equations and, crucially, handling the inverse-gamma convention mismatch:

| Dynare pdf | `dynare_prior` call | Returns |
|---|---|---|
| `normal_pdf(m, s)` | `dynare_prior(:normal, m, s)` | `Normal(m, s)` |
| `gamma_pdf(m, s)` | `dynare_prior(:gamma, m, s)` | `Gamma(m²/s², s²/m)` |
| `beta_pdf(m, s)` | `dynare_prior(:beta, m, s)` | moment-matched `Beta(α, β)` |
| `beta_pdf(m, s, p3, p4)` | `dynare_prior(:beta, m, s; lower=p3, upper=p4)` | shifted/scaled Beta on `(p3, p4)` |
| `inv_gamma_pdf(m, s)` | `dynare_prior(:inv_gamma, m, s)` (alias `:inv_gamma1`) | `InverseGamma1` **on σ** |
| `inv_gamma2_pdf(m, s)` | `dynare_prior(:inv_gamma2, m, s)` | `InverseGamma` **on σ²** |
| `uniform_pdf(...)` | `dynare_prior(:uniform, m, s)` or `; lower=a, upper=b` | `Uniform(a, b)` |

```@example dsge_estimation
priors_dynare = Dict{Symbol,Distribution}(
    :rho   => dynare_prior(:beta, 0.7, 0.1),
    :sigma => dynare_prior(:inv_gamma, 0.02, 0.05))
(rho_moments = round.((mean(priors_dynare[:rho]), std(priors_dynare[:rho])); digits=4),
 sigma_moments = round.((mean(priors_dynare[:sigma]), std(priors_dynare[:sigma])); digits=4))
```

Both distributions reproduce the requested Dynare moments exactly: the beta prior on persistence has mean 0.7 and standard deviation 0.1, and the inverse-gamma prior on the shock standard deviation has mean 0.02 and standard deviation 0.05. Without `dynare_prior` the second one is the classic porting bug — the same two numbers handed to `Distributions.InverseGamma` describe a distribution over the *variance* with entirely different moments.

!!! danger "Dynare's inverse gamma is on σ, not σ²"
    Dynare's `inv_gamma_pdf` is the **type-1** inverse gamma on the *standard deviation*; `Distributions.InverseGamma` is on the *variance*. Feeding Dynare's numbers straight into `Distributions.InverseGamma` — the natural-looking port — silently produces a completely different prior. `dynare_prior(:inv_gamma, m, s)` returns an [`InverseGamma1`](@ref) whose draws are σ values with exactly the requested mean and standard deviation (`σ² ~ InverseGamma(ν/2, s/2)` internally, matching Dynare's `(s, ν)` parameterization).

---

## Posterior Samplers

Three algorithms target the posterior. All three share the same likelihood closure, so a model estimated by one can be re-estimated by another without touching the specification.

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

```@example dsge_estimation
result_smc = estimate_dsge_bayes(spec, y_obs, [0.9];
    priors=Dict(:rho => Beta(5, 2)),
    method=:smc, observables=[:y], n_smc=50,
    n_mh_steps=2, ess_target=0.6)
report(result_smc)
```

The posterior concentrates well above the prior mean of ``5/7 \approx 0.71`` and brackets the data-generating ``\rho = 0.9``: 200 observations of a persistent series are highly informative about persistence. The header reports the number of tempering stages the adaptive schedule needed to walk ``\phi`` from 0 to 1 — raising `ess_target` forces smaller steps and therefore more stages, while `n_mh_steps` sets how many mutation moves each particle takes per stage. Both raise cost and reduce particle degeneracy; the acceptance rate reported alongside them is the mutation-step acceptance rate, not a chain diagnostic.

The likelihood is evaluated via the Kalman filter, which is exact for linear state-space models produced by [Linear Solvers](@ref dsge_linear) (`:gensys`, `:blanchard_kahn`, `:klein`).

### SMC``^2`` (Chopin, Jacob & Papaspiliopoulos 2013)

**SMC``^2``** nests a particle filter inside the outer SMC loop. At each mutation step, the likelihood ``\mathcal{L}(Y|\theta^*)`` is evaluated by running a bootstrap particle filter rather than the Kalman filter. This enables Bayesian estimation of nonlinear DSGE models solved with [Nonlinear Methods](@ref dsge_nonlinear) --- perturbation (order ``\geq 2``), Chebyshev projection, or policy function iteration --- where the Kalman filter approximation breaks down.

The particle filter estimates the likelihood as:

```math
\hat{\mathcal{L}}(Y|\theta) = \prod_{t=1}^{T} \frac{1}{M} \sum_{j=1}^{M} w_t^{(j)}
```

where ``M`` is the number of inner particles (set via `n_particles`) and ``w_t^{(j)}`` are the importance weights at time ``t``.

```julia
fit_smc2 = estimate_dsge_bayes(spec, y_obs, [0.9];
    priors=Dict(:rho => Beta(5, 2)),
    method=:smc2, observables=[:y],
    n_smc=200, n_particles=100,
    solver=:projection, solver_kwargs=(degree=5,))
report(fit_smc2)
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
fit_da = estimate_dsge_bayes(spec, y_obs, [0.9];
    priors=Dict(:rho => Beta(5, 2)),
    method=:smc2, observables=[:y],
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

```@example dsge_estimation
result_mh = estimate_dsge_bayes(spec, y_obs, [0.9];
    priors=Dict(:rho => Beta(5, 2)),
    method=:mh, observables=[:y],
    n_draws=50, burnin=25)
report(result_mh)
```

The chain is short by design here — 50 draws with 25 discarded as burn-in — so the posterior summary is indicative rather than reliable; [Convergence Diagnostics](@ref) shows how to tell the difference. RWMH is simple to implement and diagnose but converges slowly for high-dimensional parameter spaces. For models with more than 5--10 parameters, SMC is strongly preferred.

---

## Posterior Mode and Marginal Likelihood

The posterior mode is the standard first step of a Dynare-style workflow and the natural seed for a random walk. The marginal likelihood it approximates, and the two sampler-based estimates of the same quantity, are what `bayes_factor` compares.

### Posterior Mode and the Laplace Approximation

`posterior_mode` implements the standard Dynare-style first step of Bayesian estimation: numerically maximize the log posterior, report the mode together with a Laplace approximation of the marginal likelihood, and expose the inverse Hessian at the mode as an RWMH proposal covariance.

```math
\theta^* = \arg\max_\theta \; \big[\log \mathcal{L}(Y|\theta) + \log \pi(\theta)\big]
```

The optimizer works in a **prior-transformed unconstrained space** (log for positive supports, logit for bounded intervals, via `ParameterTransform`), so bounded parameters never collide with their boundaries; the reported mode is mapped back to the natural parameter space. The Laplace approximation of the log marginal likelihood is

```math
\log \hat{p}(Y) = \log \mathcal{L}(\theta^*) + \log \pi(\theta^*) + \frac{d}{2}\log(2\pi) - \frac{1}{2}\log\det H
```

where ``H`` is the Hessian of the negative log posterior at the mode and ``d`` the number of estimated parameters (Tierney & Kadane 1986). If ``H`` is not positive definite, `laplace_log_ml` is `NaN` (with a warning) and the inverse Hessian falls back to a diagonal matrix, so downstream proposal seeding never receives a garbage covariance.

```@example dsge_estimation
pm = posterior_mode(spec, y_obs, [0.9];
    priors=Dict(:rho => Beta(5, 2)), observables=[:y])
pm
```

The mode lands essentially on top of the SMC posterior mean, and the Laplace standard error — the square root of the diagonal of ``H^{-1}`` — is close to the SMC posterior standard deviation, which is what one expects when the posterior is nearly Gaussian in a single well-identified parameter. The Laplace log marginal likelihood likewise tracks the SMC tempering-path estimate to about a nat. LBFGS reaches the optimum in a handful of iterations because the transformed objective is smooth and one-dimensional; medium-scale models routinely need the full `max_iter` budget.

**Keywords** (`posterior_mode`):

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `priors` | `Dict{Symbol, Distribution}` | required | Prior distributions keyed by parameter name |
| `observables` | `Vector{Symbol}` | `spec.endog` | Observed endogenous variables |
| `measurement_error` | `Vector{<:Real}` | `nothing` | Measurement error standard deviations |
| `solver` | `Symbol` | `:gensys` | DSGE solver method |
| `solver_kwargs` | `NamedTuple` | `NamedTuple()` | Additional solver keyword arguments |
| `trends` | `ObservationTrends` | `nothing` | Deterministic observation trends, so the mode uses the sampler's measurement equation |
| `transform` | `Bool` | `true` | Optimize in the unconstrained (prior-transformed) space |
| `optimizer` | `Optim` method | `Optim.LBFGS()` | Any first-order `Optim.jl` optimizer |
| `f_reltol` | `Real` | `1e-8` | Relative objective tolerance |
| `max_iter` | `Int` | `500` | Maximum optimizer iterations |

**Return value** (`PosteriorMode` fields):

| Field | Type | Description |
|-------|------|-------------|
| `mode` | `Vector{T}` | Posterior mode in the natural parameter space |
| `inv_hessian` | `Matrix{T}` | Inverse Hessian at the mode (asymptotic posterior covariance) |
| `hessian` | `Matrix{T}` | Hessian of the negative log posterior at the mode |
| `log_posterior` | `T` | Log posterior at the mode |
| `log_likelihood` | `T` | Log-likelihood at the mode |
| `laplace_log_ml` | `T` | Laplace log marginal likelihood (`NaN` if Hessian not PD) |
| `param_names` | `Vector{Symbol}` | Parameter names (sorted prior order) |
| `converged` | `Bool` | Optimizer convergence flag |
| `n_iterations` | `Int` | Optimizer iterations used |

To seed RWMH from the mode, pass `proposal=:mode`: the chain starts at ``\theta^*`` with proposal covariance ``c^2 H^{-1}``, ``c = 2.38/\sqrt{d}`` (Roberts & Rosenthal 2001), which typically lands the acceptance rate in the 0.2--0.4 range without hand-tuning:

```@example dsge_estimation
result_mode_mh = estimate_dsge_bayes(spec, y_obs, [0.9];
    priors=Dict(:rho => Beta(5, 2)),
    method=:mh, proposal=:mode, observables=[:y],
    n_draws=50, burnin=25)
round(result_mode_mh.acceptance_rate; digits=2)
```

The mode-seeded chain lands at the top of the 0.2--0.4 band that Roberts & Rosenthal's optimal-scaling result targets, without any hand-tuning — four times the acceptance rate of the identity-proposal chain above. Starting at the mode also removes the transient that an arbitrary initial value leaves in the first hundreds of draws, which is why `proposal=:mode` is worth its one-off optimization cost on production runs.

### Marginal Likelihood and Bayes Factors

`marginal_likelihood` reads the log marginal likelihood off the result. For SMC it is the tempering-path estimate ``\hat{p}(Y) = \prod_s N^{-1}\sum_i w_s^{(i)}``, produced as a by-product of the sampler at no extra cost:

```@example dsge_estimation
ml = marginal_likelihood(result_smc)
```

For RWMH output, `bridge_sampling_ml` provides a marginal-likelihood estimate that is far more stable than harmonic-mean-style estimators (whose importance weights can have infinite variance). It fits a proposal density to the posterior draws in the prior-transformed unconstrained space and iterates the Meng & Wong (1996) optimal-bridge recursion to convergence (Gronau et al. 2017):

```@example dsge_estimation
bml = bridge_sampling_ml(result_mode_mh)
```

The three estimators — SMC tempering path, bridge sampling on the RWMH chain, and the Laplace approximation at the mode — agree to within about half a nat on this model, as they should when the posterior is close to Gaussian and every estimator is consistent for the same quantity. Differences of that size are noise; the Kass & Raftery thresholds only start to bite at ``2 \cdot \log \text{BF} > 6``, i.e. three nats of log-ML difference.

**Keywords** (`bridge_sampling_ml`):

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `proposal` | `Symbol` | `:normal` | Proposal family fitted to the draws: `:normal` or `:t` |
| `df` | `Real` | `5` | Degrees of freedom for the `:t` proposal |
| `n_proposal` | `Int` | `0` | Number of proposal draws (`0` → same as the bridge half) |
| `max_iter` | `Int` | `1000` | Maximum bridge-recursion iterations |
| `tol` | `Real` | `1e-10` | Relative convergence tolerance on the bridge ratio |
| `rng` | `AbstractRNG` | `Random.default_rng()` | RNG for proposal draws |

**Returns**: the scalar log marginal likelihood estimate, on the same additive-constant convention as the SMC tempering-path estimate and the Laplace approximation from `posterior_mode` — the three are directly comparable via `bayes_factor`. Failure cases (chain too short, proposal too diffuse, recursion non-convergence) return `NaN` with a warning, never a silently wrong number.

!!! tip "Which marginal-likelihood estimator?"
    - **SMC** (`method=:smc`): the tempering-path estimate is a by-product — use it.
    - **RWMH chains**: prefer **bridge sampling** over harmonic-mean estimators; it is consistent with much lighter tail conditions.
    - **Quick model comparison at the mode**: the Laplace approximation from `posterior_mode` is instantaneous and accurate when the posterior is approximately Gaussian.

`bayes_factor(r1, r2)` differences two such estimates; both arguments must come from the same estimator family, since the additive constants only cancel when they are on a common convention.

---

## Estimation on Trending Data

DSGE observables are stationary deviations from steady state, but GDP, consumption and investment trend. `estimate_dsge_bayes` reconciles the two at the estimation entry point in either of the two standard ways, matching Dynare's `prefilter` and `observation_trends`.

**Prefilter transforms** remove the trend from the data before filtering. `prefilter=:demean` subtracts each observable's sample mean, `:first_difference` takes ``\Delta y_t`` (dropping the first observation), `:linear_detrend` removes the OLS fit on ``[1, t]``, and `:hp` keeps the Hodrick-Prescott cycle at smoothing parameter `hp_lambda`.

**Observation trends** instead carry the trend inside the measurement equation:

```math
y_t^{obs} = d + Z s_t + \underbrace{(c_0 + c_1 t + c_2 t^2)}_{\text{trend}_t} + v_t
```

where:
- ``y_t^{obs}`` is the ``n_{obs} \times 1`` vector of observed series
- ``d`` is the steady-state observation offset and ``Z`` the selection matrix
- ``s_t`` is the model state vector and ``v_t \sim N(0, H)`` measurement error
- ``c_0, c_1, c_2`` are per-observable constant, linear and quadratic coefficients

Each coefficient is either a fixed number or a `Symbol` naming a **declared model parameter** — in which case it is estimated like any other parameter simply by giving it a prior. The model variable can therefore stay a stationary gap while the observed series is a trending level.

```@example dsge_estimation
# A trending level series: stationary model variable plus 1.0 + 0.02t
T_obs = size(y_obs, 1)
y_level = y_obs .+ 1.0 .+ 0.02 .* collect(1.0:T_obs)

res_trend = estimate_dsge_bayes(spec, y_level, [0.9];
    priors=Dict(:rho => Beta(5, 2)),
    method=:smc, observables=[:y], n_smc=50,
    observation_trends=Dict(:y => (constant=1.0, linear=0.02)))
report(res_trend)
```

The trend is removed inside the likelihood, so the persistence estimate is unaffected by the level drift: the posterior for `rho` sits on the data-generating 0.9, exactly where the untrended estimate of the same series landed. Feeding `y_level` in without either remedy instead drives persistence toward one, because the only way a stationary model can track a drifting series is to make it nearly a random walk.

The same data through `prefilter=:linear_detrend` reaches the same place without committing to trend values:

```@example dsge_estimation
res_pf = estimate_dsge_bayes(spec, y_level, [0.9];
    priors=Dict(:rho => Beta(5, 2)),
    method=:smc, observables=[:y], n_smc=50,
    prefilter=:linear_detrend)
round(res_pf.prefilter.slopes[1], digits=4)   # recovered drift ≈ 0.02
```

The applied transform is stored on the result as `res_pf.prefilter`, and [`invert_prefilter`](@ref) maps filtered paths back to the observed scale — pass `time_offset` equal to the estimation sample length to invert a forecast.

To **estimate** the drift, declare a model parameter for it and name that parameter in the trend. It need not appear in any equation:

```@example dsge_estimation
spec_g = @dsge begin
    parameters: rho = 0.9, phi = 1.5, psi = 0.5, g = 0.0
    endogenous: y, i
    exogenous: e_y, e_i
    y[t] = rho * y[t-1] + e_y[t]
    i[t] = phi * y[t] + psi * e_i[t]
end
spec_g = compute_steady_state(spec_g)

res_est = estimate_dsge_bayes(spec_g, y_level, Dict(:rho => 0.9, :g => 0.0);
    priors=Dict(:rho => Beta(5, 2), :g => Normal(0.0, 0.1)),
    method=:mh, n_draws=2500, burnin=800, observables=[:y],
    observation_trends=Dict(:y => (constant=1.0, linear=:g)))
report(res_est)
```

The 95% credible interval for `g` covers the data-generating drift of 0.02. The short chain used here is flagged for low effective sample size --- see [Convergence Diagnostics](@ref). Symbolic trend coefficients require a Kalman method (`:smc` or `:mh`); the `:smc2` particle filter cannot re-evaluate the trend per draw and raises an error naming the offending symbols.

When neither remedy is used and an observable shows a strong trend, estimation emits a guidance warning. [`detect_trend`](@ref) implements the check standalone and uses a Newey-West HAC standard error for the slope, so persistent-but-stationary observables are not flagged spuriously:

```@example dsge_estimation
detect_trend(vec(y_level))
```

The OLS slope recovers the drift, but `trending` is `false`: the HAC t-statistic falls short of the `tstat_threshold=4.0` cutoff because a persistent AR(1) inflates the Newey-West standard error of a trend slope. That conservatism is deliberate — a strongly autocorrelated but stationary series would otherwise be flagged as trending in almost every sample. Treat a `false` here as "the warning will not fire", not as "there is no trend"; the decision to prefilter belongs to the modeller.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `prefilter` | `Symbol` | `:none` | `:none`, `:demean`, `:first_difference`, `:linear_detrend`, `:hp` |
| `hp_lambda` | `Real` | `1600` | HP smoothing parameter (`:hp` only) |
| `observation_trends` | `Dict{Symbol}` | `nothing` | Per-observable `Real`/`Symbol` (linear term), `NamedTuple` of `constant`/`linear`/`quadratic`, or a tuple in that order |
| `warn_trends` | `Bool` | `true` | Emit the trending-data guidance warning |
| `tstat_threshold` | `Real` | `4.0` | HAC t-statistic above which [`detect_trend`](@ref) reports `trending` |

---

## Sampler Keywords

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
| `measurement_error` | `Vector{<:Real}`/`Symbol` | `nothing` | Measurement error std devs, `:auto` (10% of each series' variance), or `nothing` for zero ME (requires `n_obs ≤ n_shocks`) |
| `likelihood` | `Symbol` | `:auto` | Likelihood evaluation method (currently always Kalman) |
| `solver` | `Symbol` | `:gensys` | DSGE solver method |
| `solver_kwargs` | `NamedTuple` | `NamedTuple()` | Additional solver keyword arguments |
| `delayed_acceptance` | `Bool` | `false` | Two-stage delayed acceptance (SMC``^2`` only) |
| `n_screen` | `Int` | `200` | Screening PF particles (delayed acceptance only) |
| `max_stages` | `Int` | `500` | Hard cap on adaptive-tempering stages (`:smc`/`:smc2`); exceeding it raises an error |
| `min_dphi` | `Real` | `1e-10` | Minimum tempering step; a smaller adaptive ``\Delta\phi`` while ``\phi < 1`` aborts rather than spins |
| `keep_burnin` | `Bool` | `false` | Retain the full RWMH chain including burnin (e.g. for trace plots) |
| `proposal` | `Symbol` | `:adaptive` | RWMH proposal init: `:adaptive` or `:mode` (seed from `posterior_mode`) |
| `transform` | `Bool` | `true` | RWMH walks in the prior-transformed unconstrained space with Jacobian correction |
| `prefilter` | `Symbol` | `:none` | Observable transform: `:demean`, `:first_difference`, `:linear_detrend`, `:hp` |
| `hp_lambda` | `Real` | `1600` | HP smoothing parameter when `prefilter=:hp` |
| `observation_trends` | `Dict{Symbol}` | `nothing` | Deterministic constant/linear/quadratic trends in the measurement equation |
| `warn_trends` | `Bool` | `true` | Emit the trending-data guidance warning when neither remedy is used |
| `rng` | `AbstractRNG` | `Random.default_rng()` | Random number generator for the sampler |

!!! note "Sampling in the unconstrained space"
    With `transform=true` (the default for `method=:mh`), the random walk runs on ``y = T(\theta)`` — ``\log`` for positive supports, logit for bounded intervals, inferred from each prior's support — and the acceptance ratio uses ``\log p(\theta(y)|Y) + \log|J(y)|``, the correct pushforward density (Stan reference manual). A walk on a persistence near 1 or a shock standard deviation near 0 then never wastes proposals outside the support; draws are back-transformed to ``\theta`` before storage, so results are directly comparable to `transform=false`.

!!! note "Pre-Linearized Models"
    For `ModelSpec` with `linear=true` (e.g., Smets & Wouters 2007), `spec.steady_state` is all zeros, so the observation-equation offset is instead the effective steady state ``d = (I - G_1)^{-1} C_{\text{sol}}``, where ``C_{\text{sol}}`` carries the constant terms from the solver. This handles observation equations with trend growth, steady-state inflation, or other constant offsets absent from the zero steady state. The substitution fires whenever `spec.linear` is set and ``C_{\text{sol}}`` is non-zero, and the Kalman and particle-filter paths share it, so `estimate_dsge_bayes`, `posterior_mode`, and `identification_diagnostics` all see the same measurement equation. No user intervention is required.

---

## Posterior Analysis

After estimation, three functions extract information from the `BayesianDSGE` result:

```@example dsge_estimation
# Posterior summary: mean, median, std, 95% credible interval per parameter
ps = posterior_summary(result_smc)
ps[:rho][:mean]       # posterior mean of rho
```

```@example dsge_estimation
ps[:rho][:ci_lower]   # lower bound of 95% CI for rho
```

```@example dsge_estimation
# Prior vs posterior comparison table
tbl = prior_posterior_table(result_smc)
nothing # hide
```

```@example dsge_estimation
# Posterior predictive simulation
Y_pred = posterior_predictive(result_smc, 10; T_periods=50)
size(Y_pred)
```

`posterior_summary` returns a `Dict{Symbol, Dict{Symbol, T}}` with keys `:mean`, `:median`, `:std`, `:ci_lower` (2.5th percentile), and `:ci_upper` (97.5th percentile) for each parameter. `prior_posterior_table` returns a vector of named tuples (`param`, `prior_dist`, `prior_mean`, `prior_std`, `post_mean`, `post_std`, `ci_lower`, `ci_upper`, `low_ess`) suitable for tabular display, comparing prior and posterior moments side by side. `posterior_predictive` draws `n_sim` parameter vectors from the posterior, solves the model at each, and simulates forward. Draws that fail to solve are dropped with a warning, so the returned array is `n_valid x T_periods x n_vars` with `n_valid ≤ n_sim` — the first dimension counts successful draws, never the requested ones.

For RWMH chains, `posterior_summary` additionally annotates each parameter with its bulk effective sample size (`:ess_bulk`) and a `:low_ess` flag, and **warns** when any parameter's ESS falls below `min_ess` (default 400, per Vehtari et al. 2021) rather than silently presenting unreliable credible intervals. `prior_posterior_table` carries the same flag in a `low_ess` column.

---

## Convergence Diagnostics

Credible intervals from an MCMC chain are only as good as the chain's mixing. `mcmc_diagnostics` computes the modern standard set of per-parameter convergence diagnostics on the retained (post-burn-in) draws:

- **Rank-normalized split-``\hat{R}``** (Vehtari et al. 2021): the chain is split in half, pooled draws are rank-normalized (rank → z-score via the inverse normal CDF), and ``\hat{R} = \sqrt{\widehat{\mathrm{var}}^+ / W}`` is computed from the between/within half-chain variances. The reported value is the maximum of the bulk statistic and the *folded* statistic on ``|\theta - \mathrm{median}|``, which catches scale (not just location) non-convergence. Values ``\lesssim 1.01`` indicate convergence.
- **Bulk / tail ESS**: effective sample size from the integrated autocorrelation time ``\mathrm{ESS} = S / (1 + 2\sum_k \hat\rho_k)`` with Geyer's initial-monotone-sequence truncation. Bulk-ESS is computed on rank-normalized draws; tail-ESS is the minimum ESS of the 5% and 95% quantile indicators. Vehtari et al. recommend ESS ``\geq 400`` before trusting reported intervals.
- **Geweke (1992) z**: spectral test comparing the mean of the first 10% against the last 50% of the chain, with numerical-standard-error variance estimates; under convergence ``z \sim N(0,1)``.

```@example dsge_estimation
diag = mcmc_diagnostics(result_mode_mh)
diag
```

With only 25 retained draws every diagnostic fails at once, which is exactly the intended behaviour: ``\hat{R}`` is far above the 1.01 threshold, bulk ESS is an order of magnitude below the recommended 400, tail ESS is smaller still, and the Geweke z rejects equality of the early and late chain means decisively. Nothing about the posterior summary printed earlier should be trusted at this chain length; the fix is more draws, not a different diagnostic.

`trace` and `acf` expose the raw per-parameter draw sequence and its autocorrelation function for plotting:

```@example dsge_estimation
tr = trace(result_mode_mh, :rho)      # retained draw sequence
length(tr)
```

```@example dsge_estimation
a = acf(result_mode_mh, :rho; lags=5) # ACFResult (spectral acf on the chain)
round.(a.acf; digits=2)
```

Adjacent draws are strongly correlated and the sequence then oscillates around zero — the sampling noise of an autocorrelation estimated from 25 points, not a real cycle. That first-lag correlation is the direct cause of the low ESS above: each draw carries well under one independent observation. Thinning does not fix this; only a longer chain or a better-scaled proposal does.

**Keywords / accessors**:

| Function | Arguments | Returns |
|----------|-----------|---------|
| `mcmc_diagnostics(result)` | `BayesianDSGE` | `MCMCDiagnostics` (see field table below) |
| `trace(result, param)` | result + parameter `Symbol` | `Vector{T}` of retained draws |
| `acf(result, param; lags, conf_level)` | result + parameter `Symbol` | `ACFResult` (see [Spectral Analysis](@ref spectral_page)) |
| `posterior_summary(result; min_ess=400)` | ESS warning threshold | summary dict + `:ess_bulk`/`:low_ess` keys (RWMH) |

**Return value** (`MCMCDiagnostics` fields):

| Field | Type | Description |
|-------|------|-------------|
| `param_names` | `Vector{Symbol}` | Parameter names |
| `rhat` | `Vector{T}` | Rank-normalized split-``\hat{R}`` (max of bulk and folded) |
| `ess_bulk` | `Vector{T}` | Bulk effective sample size |
| `ess_tail` | `Vector{T}` | Tail effective sample size (min of 5%/95% indicators) |
| `geweke_z` | `Vector{T}` | Geweke z-statistic |
| `geweke_p` | `Vector{T}` | Two-sided Geweke p-value |
| `mean`, `sd` | `Vector{T}` | Posterior mean and standard deviation |
| `n_draws` | `Int` | Retained draws used |
| `method` | `Symbol` | Sampler that produced the draws |

!!! note "SMC draws are not a chain"
    `mcmc_diagnostics` assumes Markov chain draws. Calling it on `:smc`/`:smc2` results (weighted particle systems) emits a warning — the SMC-native convergence measures are the ESS history (`result.ess_history`) and the tempering schedule (`result.phi_schedule`).

---

## Identification Diagnostics

Estimating a DSGE whose parameters are not identified by the chosen observables produces confident-looking but meaningless posteriors. Three diagnostics catch this failure mode — one before estimation, two after:

**Iskrev (2010) rank test** (pre-estimation). The parameters are locally identified from the data iff the Jacobian ``J(\theta) = \partial m(\theta) / \partial \theta'`` of the model-implied data moments has full column rank. The moment vector ``m(\theta)`` stacks the observable steady-state means, the lower triangle of the contemporaneous covariance, and the autocovariance matrices at lags ``1..L`` computed from the first-order state-space solution via the Lyapunov equation. Rank deficiency names the unidentified directions through the null space of ``J``:

```@example dsge_estimation
# a and b enter the model only as the product a·b — not separately identified
spec_bad = @dsge begin
    parameters: a = 0.5, b = 0.9, sigma = 0.5
    endogenous: y
    exogenous: e
    y[t] = a * b * y[t-1] + sigma * e[t]
    steady_state = [0.0]
end
spec_bad = compute_steady_state(spec_bad)
idd = identification_diagnostics(spec_bad, [:a, :b]; observables=[:y])
idd
```

The Jacobian of four moments with respect to two parameters has rank 1, so `Locally identified` is `NO` and the smallest singular value is numerically zero. The reported null direction has non-zero loadings on both `a` and `b` with opposite signs: moving along it changes the two parameters while holding the product ``a \cdot b`` — the only thing the data see — fixed. Any sampler run on this pair returns a posterior ridge, not a point.

**Koop-Pesaran-Smith (2013) learning-rate check** (post-estimation). For an identified parameter the posterior variance shrinks at the ``1/T`` rate; `learning_rate_check` re-estimates on nested subsamples and reports the implied rate ``\alpha`` in ``\mathrm{var} \propto T^{-\alpha}`` — ``\alpha \approx 1`` is healthy, ``\alpha \approx 0`` flags a parameter whose posterior barely updates.

**Prior/posterior overlap** (post-estimation). `prior_posterior_overlap` computes ``\int \min(\pi(\theta_i), p(\theta_i|Y))\,d\theta_i`` per parameter; overlap near 1 means the data never moved the prior.

```@example dsge_estimation
ppo = prior_posterior_overlap(result_smc)   # instantaneous
ppo
```

```julia
plot_result(ppo)
```

```@raw html
<iframe src="../assets/plots/prior_posterior.html" width="100%" height="440" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

An overlap near 0.3 means only about a third of the prior density is shared with the posterior — the data moved the persistence prior substantially, which is the healthy case, and the row is marked `ok`. Overlap at or above the 0.8 threshold is flagged and means the posterior is essentially reporting the prior back. `learning_rate_check` is the more expensive companion, since each subsample fraction triggers a fresh SMC run:

```julia
lrc = learning_rate_check(result_smc; fractions=[0.5, 1.0], n_smc=300)
```

**Keywords and return values**:

| Function | Key keywords | Returns |
|----------|--------------|---------|
| `identification_diagnostics(spec, params; ...)` | `theta`, `observables`, `n_lags=2`, `tol_rel=√eps` | `IdentificationDiagnostics`: `rank`, `n_params`, `singular_values`, `null_space`, `identified` |
| `learning_rate_check(result; ...)` | `fractions=[0.5,1.0]`, `n_smc=300`, `threshold=0.2` | `LearningRateCheck`: `sample_sizes`, `post_vars`, `learning_rate` (α), `flagged` |
| `prior_posterior_overlap(result; ...)` | `n_grid=0` (auto ≈√N), `threshold=0.8` | `PriorPosteriorOverlap`: `overlap` ∈ [0,1], `flagged` |

All three emit a warning naming the offending parameters when something looks unidentified; none of them throws.

!!! warning "Identification is observables-dependent"
    A parameter can be perfectly identified with one observable set and unidentified with another. Re-run `identification_diagnostics` with exactly the `observables` you pass to `estimate_dsge_bayes`, and check several `n_lags` horizons (Iskrev's recommendation).

---

## Predictive Checks

**Prior predictive analysis** (Geweke 2005) answers "what kind of data does my prior believe in?" *before* estimation: draw parameters from the prior, solve, simulate, and summarize. A prior that implies absurd volatilities or persistence should be revised before it distorts the posterior:

```@example dsge_estimation
ppr = prior_predictive(spec, Dict(:rho => Beta(5, 2));
    n_draws=50, T_periods=100, observables=[:y])
ppr
```

Every prior draw solved, and the implied data are economically sensible: output is centred on zero with a variance a little above the unit shock variance, and its first-order autocorrelation spreads over roughly 0.3 to 0.9 — the persistence range the `Beta(5, 2)` prior actually believes in. A prior implying, say, a 5% quantile of 0.99 on that AR(1) statistic would be asserting near-unit-root data before seeing any, and should be revised.

**Posterior predictive checks** (Gelman, Meng & Stern 1996) assess model adequacy *after* estimation: draw from the posterior, simulate replicated datasets of the observed length, and compare summary statistics. The posterior predictive p-value ``p_j = \Pr(T_j(y^{\mathrm{rep}}) \geq T_j(y^{\mathrm{obs}}))`` should be interior — extreme values (marked `*` in the table) flag the data feature the model cannot reproduce:

```@example dsge_estimation
ppc = posterior_predictive_check(result_smc; n_draws=25)
ppc
```

Every p-value is interior and none is starred, so the fitted model reproduces the mean, variance, and first-order autocorrelation of the observed series — as it must, since the data were simulated from this very specification. In applied work the AR(1) statistic is the one that usually goes extreme first: a p-value near 0 or 1 there says the model cannot match the persistence of the data at any parameter value the posterior supports.

**Keywords**:

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `n_draws` | `Int` | `500` / `200` | Prior / posterior draws to simulate |
| `T_periods` | `Int` | `200` | Simulated periods per draw (prior predictive only) |
| `observables` | `Vector{Symbol}` | all endogenous | Which variables to summarize (prior predictive) |
| `data` | matrix | stored sample | Observed data override (posterior check) |
| `stats` | function | mean/var/AR(1)/cross-corr | `Y::Matrix → (names, values)` or `NamedTuple` of scalars |
| `solver` | `Symbol` | `:gensys` | DSGE solver (prior predictive; the posterior check reuses the fitted solver) |
| `solver_kwargs` | `NamedTuple` | `NamedTuple()` | Additional solver keyword arguments (prior predictive) |
| `rng` | `AbstractRNG` | `Random.default_rng()` | Random number generator |

**Return values**: `PriorPredictiveResult` carries the `n_effective × n_stats` draw-level statistic matrix (`stats`), the labels (`stat_names`), and the effective draw count. `PosteriorPredictiveCheck` adds the `observed` statistic vector and `p_values`. In both, parameter draws for which the model fails to solve are **dropped and counted** — `n_effective` reports the draws actually used, and a warning fires when more than 10% are lost (a symptom of a prior straddling the determinacy boundary).

---

## Posterior IRFs, FEVD, and Simulation

Bayesian DSGE estimation quantifies parameter uncertainty; `irf`, `fevd`, and `simulate` propagate it into impulse responses, variance decompositions, and predictive paths by re-solving the model at posterior draws.

### Posterior IRFs and FEVD (Herbst & Schorfheide 2015)

For each of `n_draws` randomly selected posterior draws, the model is re-solved at those parameter values and the analytical IRF (or FEVD) is computed. The results are stacked and summarized with pointwise quantile bands. The default quantiles ``[0.05, 0.16, 0.84, 0.95]`` produce dual 68% and 90% credible bands --- the standard reporting convention in the Bayesian DSGE literature.

```@example dsge_estimation
# Bayesian IRFs with dual credible bands
birf_smc = irf(result_smc, 20; n_draws=10)
report(birf_smc)
```

```julia
plot_result(birf_smc)
```

```@raw html
<iframe src="../assets/plots/dsge_bayes_irf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The response of `y` to its own shock decays at the estimated persistence and the credible bands exclude zero out to a horizon where ``\rho^h`` is still economically meaningful — the starred entries in the table. The response of `i` is exactly ``\phi`` times the `y` response at every horizon, because `phi` is held at its calibrated value and only `rho` carries posterior uncertainty. The `e_i` shock moves `i` on impact and nothing thereafter, since it enters no state.

```@example dsge_estimation
# Bayesian FEVD
bfevd_smc = fevd(result_smc, 20; n_draws=10)
report(bfevd_smc)
```

The decomposition confirms the same recursive structure: `y` owes 100% of its forecast error variance to `e_y` at every horizon, while `i` starts with a small `e_i` share on impact that is progressively diluted as the persistent `e_y` component accumulates.

```@example dsge_estimation
# Custom quantiles (90% band only)
birf_90 = irf(result_smc, 20; n_draws=10, quantiles=[0.05, 0.95])
nothing # hide
```

Both methods return `BayesianImpulseResponse{T}` and `BayesianFEVD{T}` respectively --- the same types used by Bayesian VAR, so all existing `report()`, `plot_result()`, `table()`, and `cumulative_irf()` infrastructure works automatically.

Draws that produce indeterminate or explosive solutions are silently skipped. If all draws fail, an error is raised.

### Posterior Predictive Simulation

`simulate` draws from the posterior predictive distribution with credible bands. For each posterior parameter draw, the model is re-solved and simulated forward `T_periods` periods:

```@example dsge_estimation
bsim = simulate(result_smc, 50; n_draws=10)
report(bsim)
```

The result is a `BayesianDSGESimulation{T}` containing the pointwise median, quantile bands, and all raw simulation paths. Unlike `posterior_predictive`, which returns the raw array, this carries the quantile summary needed for a fan chart: `point_estimate` is the pointwise median path and `quantiles` holds the bands at the levels in `quantile_levels`.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `n_draws` | `Int` | `200` | Number of posterior draws to subsample |
| `quantiles` | `Vector{<:Real}` | `[0.05, 0.16, 0.84, 0.95]` | Quantile levels for credible bands |
| `solver` | `Symbol` | `:gensys` | DSGE solver for re-solving at each draw |
| `solver_kwargs` | `NamedTuple` | `NamedTuple()` | Additional solver keyword arguments |
| `rng` | `AbstractRNG` | `Random.default_rng()` | Random number generator |

---

## Return Values

### `BayesianDSGESimulation{T}`

| Field | Type | Description |
|-------|------|-------------|
| `quantiles` | `Array{T,3}` | ``T \times n_{\text{vars}} \times n_q`` pointwise quantile bands |
| `point_estimate` | `Matrix{T}` | ``T \times n_{\text{vars}}`` posterior median |
| `T_periods` | `Int` | Number of simulation periods |
| `variables` | `Vector{String}` | Variable names |
| `quantile_levels` | `Vector{T}` | Quantile levels used |
| `all_paths` | `Array{T,3}` | ``n_{\text{draws}} \times T \times n_{\text{vars}}`` raw simulation paths |

### `BayesianDSGE{T}`

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
| `spec` | `ModelSpec{T}` | Back-reference to model specification |
| `solution` | Union type | Model solution at the point named by `solved_at` |
| `state_space` | Union type | State-space representation at the same point |
| `n_failed_draws` | `Int` | Likelihood evaluations that failed to solve |
| `n_lik_evals` | `Int` | Total likelihood evaluations |
| `solved_at` | `Symbol` | `:posterior_mean`, or `:highest_posterior_draw` when the mean does not solve |
| `data` | `Matrix{T}` | ``n_{obs} \times T_{obs}`` estimation sample (post-prefilter) |
| `observables` | `Vector{Symbol}` | Observed endogenous variables |
| `measurement_error` | `Vector{T}` or `nothing` | Resolved measurement error standard deviations |
| `solver` | `Symbol` | Solver used for the likelihood |
| `solver_kwargs` | `NamedTuple` | Solver keyword arguments used |
| `prefilter` | `PrefilterSpec{T}` or `nothing` | Observable transform applied before estimation |
| `trends` | `ObservationTrends{T}` or `nothing` | Deterministic trends carried in the measurement equation |

When called with `method=:mh`, the stored `method` field reports `:rwmh` --- the random-walk Metropolis-Hastings sampler that `:mh` selects. The stored `data`, `observables`, `solver`, and `solver_kwargs` are what let `bridge_sampling_ml`, `learning_rate_check`, and `posterior_predictive_check` re-evaluate the likelihood without being handed the sample again.

**StatsAPI interface**: `coef(result)` returns the posterior mean parameter vector.

---

## Complete Example

This example estimates the persistence of the RBC technology process by GMM and by Bayesian SMC on the same simulated sample, then propagates posterior uncertainty into impulse responses.

```@example dsge_estimation
# 1. Specify the RBC model
spec_rbc = @dsge begin
    parameters: β = 0.99, α = 0.36, δ = 0.025, ρ = 0.9, σ = 0.01
    endogenous: Y, C, K, A
    exogenous: ε_A

    Y[t] = A[t] * K[t-1]^α
    C[t] + K[t] = Y[t] + (1 - δ) * K[t-1]
    1 = β * (C[t] / C[t+1]) * (α * A[t+1] * K[t]^(α - 1) + 1 - δ)
    A[t] = A[t-1]^ρ * exp(σ * ε_A[t])

    steady_state = begin
        A_ss = 1.0
        K_ss = (α * β / (1 - β * (1 - δ)))^(1 / (1 - α))
        Y_ss = K_ss^α
        C_ss = Y_ss - δ * K_ss
        [Y_ss, C_ss, K_ss, A_ss]
    end
end

# 2. Simulate data from the true model
sol_rbc = solve(spec_rbc)
Y_rbc = simulate(sol_rbc, 200)
nothing # hide
```

The RBC model carries one structural shock against four endogenous variables, so IRF matching against a four-variable VAR is not available here (see the dimension warning under [GMM Estimation](@ref dsge_est_gmm)) and the Kalman likelihood needs either a single observable or measurement error. SMM sidesteps both constraints, provided the moments are chosen to speak to the parameter: the default moment vector mixes the level-scale variances of ``Y``, ``C``, and ``K``, and capital's variance dominates the objective. Targeting the technology process — the only place ``\rho`` appears — is both faster and better identified:

```@example dsge_estimation
# 3. Frequentist: SMM on the technology process alone (column 4 of the data)
tech_moments(d)  = autocovariance_moments(d[:, [4]]; lags=2)
tech_contribs(d) = autocovariance_moment_contributions(d[:, [4]]; lags=2)

est_rbc = estimate_dsge(spec_rbc, Y_rbc, [:ρ];
    method=:smm, sim_ratio=5,
    moments_fn=tech_moments, contributions_fn=tech_contribs,
    bounds=ParameterTransform([0.0], [0.99]))
report(est_rbc)
```

```@example dsge_estimation
# 4. Bayesian: SMC on output alone (1 observable ≤ 1 shock ⇒ no measurement error)
fit_rbc = estimate_dsge_bayes(spec_rbc, Y_rbc[:, [1]], [0.9];
    priors=Dict(:ρ => Beta(5, 2)),
    method=:smc, observables=[:Y], n_smc=100)
report(fit_rbc)
```

```@example dsge_estimation
# 5. Propagate posterior uncertainty into impulse responses
birf_rbc = irf(fit_rbc, 20; n_draws=20)
report(birf_rbc)
```

```julia
plot_result(birf_rbc)
```

```@raw html
<iframe src="../assets/plots/dsge_bayes_irf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The SMM point estimate and the SMC posterior mean both sit near the data-generating ``\rho = 0.9``, reached from opposite directions: SMM matches the variance and first two autocovariances of the technology process, while SMC evaluates the exact Kalman likelihood of output alone. SMM's over-identification test has three moments against one parameter and does not reject, confirming that an AR(1) with the estimated persistence reproduces those moments. The Bayesian impulse responses inherit the posterior spread of ``\rho``, and every band excludes zero over the horizon shown. Output jumps on impact and decays with the technology process, consumption rises for a couple of years as households smooth the windfall, and capital accumulates to a peak around three years out before depreciating back — the standard RBC propagation mechanism, now reported with parameter uncertainty attached.

---

## Common Pitfalls

1. **Wrong steady state**: If the steady state is incorrect, all estimation methods fail silently --- the model solves to a nonsensical equilibrium, and the optimizer converges to economically meaningless parameter values. Always verify `compute_steady_state` and check that the solution satisfies `is_determined(sol)` before estimation.

2. **Indeterminate model at prior draws**: SMC initializes particles from the prior. If many prior draws produce indeterminate or explosive models, the likelihood evaluates to ``-\infty`` and particles are wasted. Tighten priors to concentrate mass on the determinacy region, or increase `n_smc` to compensate.

3. **Too few SMC particles**: For posteriors with ridges, multimodality, or strong correlations, ``n_{\text{smc}} = 1000`` may not suffice. Start with 5000+ and reduce only after confirming that the ESS history remains above the target and that repeated runs produce consistent marginal likelihood estimates.

4. **Observable mismatch**: The `observables` keyword specifies which endogenous variables in the model correspond to columns in the data matrix, in order. Mismatched dimensions or incorrect ordering produce nonsensical likelihood values. The number of observables must equal the number of data columns.

5. **Trending observables against a stationary model**: Feeding levels of GDP or consumption to a model whose variables are steady-state deviations biases persistence toward one and distorts every other parameter. Use `prefilter=` or `observation_trends=` (see [Estimation on Trending Data](@ref)); the entry-point warning names the offending series. Note that `prefilter=:first_difference` changes what the model variable means — the observable is now a growth rate, so the measured DSGE variable must be one too.

6. **Solver choice for Bayesian estimation**: Use `:smc` with `:gensys` (or `:blanchard_kahn`, `:klein`) for linear models --- the Kalman filter provides the exact likelihood. Use `:smc2` with `:projection` or `:pfi` for nonlinear models where the particle filter is necessary. Using `:smc` with a nonlinear solver silently falls back to a first-order Kalman approximation that ignores higher-order dynamics.

7. **IRF matching with too few shocks**: The model IRF is ``H \times n \times n_{\text{shocks}}`` and the VAR target is ``H \times n \times n``. When they differ, `estimate_dsge` returns a constant penalty at every ``\theta``, so the optimizer never moves and the reported "estimate" equals the starting value while the J-statistic is astronomically large. Give the model as many shocks as the VAR has variables, or pass a `target_irfs` object sliced to the model's shock count.

8. **Reading a J p-value that is `NaN`**: The ``\chi^2`` limit requires efficient weighting. `:analytical_gmm` always uses identity weighting, and IRF matching under `:diagonal`/`:cee`/`:identity` reports no statistic at all. A `NaN` there is a signal to re-estimate under `:two_step`, not a numerical failure.

9. **Interpreting a short chain**: `posterior_summary` on an RWMH result warns when bulk ESS falls under `min_ess`, and `mcmc_diagnostics` reports ``\hat{R}`` and ESS explicitly. Credible intervals from a chain with ``\hat{R} > 1.01`` or ESS well below 400 are not usable, however tight they look.

---

## References

- An, S., & Schorfheide, F. (2007). Bayesian Analysis of DSGE Models. *Econometric Reviews*, 26(2-4), 113-172. [DOI](https://doi.org/10.1080/07474930701220071)

- Chopin, N., Jacob, P. E., & Papaspiliopoulos, O. (2013). SMC``^2``: An Efficient Algorithm for Sequential Analysis of State Space Models. *Journal of the Royal Statistical Society: Series B*, 75(3), 397-426. [DOI](https://doi.org/10.1111/j.1467-9868.2012.01046.x)

- Christen, J. A., & Fox, C. (2005). Markov Chain Monte Carlo Using an Approximation. *Journal of Computational and Graphical Statistics*, 14(4), 795-810. [DOI](https://doi.org/10.1198/106186005X76983)

- Christiano, L. J., Eichenbaum, M., & Evans, C. L. (2005). Nominal Rigidities and the Dynamic Effects of a Shock to Monetary Policy. *Journal of Political Economy*, 113(1), 1-45. [DOI](https://doi.org/10.1086/426038)

- Gelman, A., Meng, X.-L., & Stern, H. (1996). Posterior Predictive Assessment of Model Fitness via Realized Discrepancies. *Statistica Sinica*, 6(4), 733-807. [Article](https://www3.stat.sinica.edu.tw/statistica/j6n4/j6n41/j6n41.htm)

- Geweke, J. (1992). Evaluating the Accuracy of Sampling-Based Approaches to the Calculation of Posterior Moments. In *Bayesian Statistics 4*, 169-193. Oxford University Press. [DOI](https://doi.org/10.1093/oso/9780198522669.003.0010)

- Geweke, J. (2005). *Contemporary Bayesian Econometrics and Statistics*. Wiley. ISBN 978-0-471-67932-5. [DOI](https://doi.org/10.1002/0471744735)

- Gronau, Q. F., Sarafoglou, A., Matzke, D., Ly, A., Boehm, U., Marsman, M., Leslie, D. S., Forster, J. J., Wagenmakers, E.-J., & Steingroever, H. (2017). A Tutorial on Bridge Sampling. *Journal of Mathematical Psychology*, 81, 80-97. [DOI](https://doi.org/10.1016/j.jmp.2017.09.005)

- Hansen, L. P. (1982). Large Sample Properties of Generalized Method of Moments Estimators. *Econometrica*, 50(4), 1029-1054. [DOI](https://doi.org/10.2307/1912775)

- Hansen, L. P., & Singleton, K. J. (1982). Generalized Instrumental Variables Estimation of Nonlinear Rational Expectations Models. *Econometrica*, 50(5), 1269-1286. [DOI](https://doi.org/10.2307/1911873)

- Herbst, E., & Schorfheide, F. (2014). Sequential Monte Carlo Sampling for DSGE Models. *Journal of Applied Econometrics*, 29(7), 1073-1098. [DOI](https://doi.org/10.1002/jae.2397)

- Herbst, E. P., & Schorfheide, F. (2015). *Bayesian Estimation of DSGE Models*. Princeton University Press. ISBN 978-0-691-16108-2.

- Iskrev, N. (2010). Local Identification in DSGE Models. *Journal of Monetary Economics*, 57(2), 189-202. [DOI](https://doi.org/10.1016/j.jmoneco.2009.12.007)

- Kass, R. E., & Raftery, A. E. (1995). Bayes Factors. *Journal of the American Statistical Association*, 90(430), 773-795. [DOI](https://doi.org/10.1080/01621459.1995.10476572)

- Koop, G., Pesaran, M. H., & Smith, R. P. (2013). On Identification of Bayesian DSGE Models. *Journal of Business & Economic Statistics*, 31(3), 300-314. [DOI](https://doi.org/10.1080/07350015.2013.773905)

- Lee, B.-S., & Ingram, B. F. (1991). Simulation Estimation of Time-Series Models. *Journal of Econometrics*, 47(2-3), 197-205. [DOI](https://doi.org/10.1016/0304-4076(91)90098-X)

- Meng, X.-L., & Wong, W. H. (1996). Simulating Ratios of Normalizing Constants via a Simple Identity: A Theoretical Exploration. *Statistica Sinica*, 6(4), 831-860. [Article](https://www3.stat.sinica.edu.tw/statistica/j6n4/j6n43/j6n43.htm)

- Roberts, G. O., & Rosenthal, J. S. (2001). Optimal Scaling for Various Metropolis-Hastings Algorithms. *Statistical Science*, 16(4), 351-367. [DOI](https://doi.org/10.1214/ss/1015346320)

- Ruge-Murcia, F. (2012). Estimating Nonlinear DSGE Models by the Simulated Method of Moments. *Journal of Economic Dynamics and Control*, 36(6), 914-938. [DOI](https://doi.org/10.1016/j.jedc.2012.01.008)

- Tierney, L., & Kadane, J. B. (1986). Accurate Approximations for Posterior Moments and Marginal Densities. *Journal of the American Statistical Association*, 81(393), 82-86. [DOI](https://doi.org/10.1080/01621459.1986.10478240)

- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-Normalization, Folding, and Localization: An Improved ``\hat{R}`` for Assessing Convergence of MCMC. *Bayesian Analysis*, 16(2), 667-718. [DOI](https://doi.org/10.1214/20-BA1221)
