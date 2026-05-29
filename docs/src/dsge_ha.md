# [Heterogeneous Agent DSGE](@id dsge_ha)

Standard DSGE models assume a **representative agent** whose decisions aggregate to macroeconomic outcomes. In reality, households differ in wealth, income, and consumption --- heterogeneity that shapes aggregate responses to shocks, especially monetary and fiscal policy. MacroEconometricModels.jl provides a complete toolkit for **heterogeneous agent DSGE (HA-DSGE)** models: the **Endogenous Grid Method** (Carroll 2006) and **VFI** for solving individual problems, **Young (2010) histogram** tracking for the wealth distribution, and three aggregate solution methods --- **Sequence-Space Jacobian** (Auclert, Bardóczy, Rognlie & Straub 2021), **Reiter (2009) linearization**, and **Krusell-Smith (1998) simulation**. The module supports one-asset and two-asset HANK models with Bayesian estimation.

- Individual problem solvers: EGM (one-asset and nested two-asset) and VFI with Howard improvement
- Income discretization: Rouwenhorst (1995) and Tauchen (1986)
- Distribution: Young (2010) non-stochastic histogram with sparse transition matrices
- Steady state: bisection on the interest rate with EGM + distribution + market clearing
- Three aggregate solution methods: SSJ, Reiter, Krusell-Smith
- Built-in models: Krusell-Smith (1998), one-asset HANK, two-asset HANK
- Bayesian estimation via RWMH + Kalman filter on reduced system
- Visualization: wealth distribution, Lorenz curve, policy functions

```@setup dsge_ha
using MacroEconometricModels, Random
Random.seed!(42)
```

## Quick Start

**Recipe 1: Krusell-Smith steady state**

```@example dsge_ha
spec = load_ha_example(:krusell_smith)
ss = compute_steady_state(spec; K_init=10.0, r_bounds=(-0.02, 0.04),
                           max_iter=80, tol=1e-4)
report(ss)
```

**Recipe 2: Solve and compute IRFs via SSJ**

```@example dsge_ha
sol = solve(spec; method=:ssj, ss=ss, T_horizon=50, n_reduced=15)
report(sol)
```

**Recipe 3: Wealth distribution visualization**

```julia
plot_result(ss)                     # wealth histogram with Gini
plot_result(ss; view=:lorenz)       # Lorenz curve
plot_result(ss; view=:policy)       # consumption and savings by income
```

**Recipe 4: Simulate individual panel data**

```@example dsge_ha
panel = simulate_panel(ss; N_agents=500, T_periods=100)
size(panel)
```

The panel matrix contains simulated asset holdings for 500 agents over 100 periods, enabling cross-sectional and longitudinal analyses of wealth dynamics at the micro level.

**Recipe 5: Inequality dynamics at steady state**

```@example dsge_ha
ineq = inequality_irf(ss; T_periods=10)
(gini = round(ineq[:gini][1], digits=4),
 p50  = round(ineq[:p50][1], digits=2),
 p90  = round(ineq[:p90][1], digits=2))
```

**Recipe 6: Krusell-Smith simulation method**

```@example dsge_ha
ks_result = solve(spec; method=:krusell_smith, ss=ss,
                  T_sim=500, T_burn=100, max_outer=3,
                  rho_z=0.95, sigma_z=0.007)
report(ks_result)
```

---

## Individual Problem

Households solve a consumption-savings problem with idiosyncratic income risk and a borrowing constraint:

```math
V(a, e) = \max_{c, a'} \; u(c) + \beta \, \mathbb{E}\bigl[V(a', e') \mid e\bigr]
```

subject to:

```math
c + a' = (1 + r) \, a + w \, e, \qquad a' \geq \underline{a}
```

where:
- ``a`` is individual asset holdings
- ``e`` is idiosyncratic productivity (Markov chain)
- ``r, w`` are aggregate prices (interest rate, wage)
- ``\underline{a}`` is the borrowing constraint
- ``\beta`` is the discount factor

### Endogenous Grid Method

The **EGM** (Carroll 2006) avoids root-finding by inverting the Euler equation on an endogenous grid:

1. Fix end-of-period assets ``a'`` on the exogenous grid
2. Compute expected marginal utility: ``\text{EMU}_i = \beta (1+r) \sum_{j'} \pi(j, j') \, u'(c(a'_i, e_{j'}))``
3. Invert the Euler equation: ``c_i = (u')^{-1}(\text{EMU}_i)``
4. Recover beginning-of-period assets (endogenous): ``a_i = (c_i + a'_i - w e_j) / (1+r)``
5. Interpolate back to the exogenous grid
6. Apply the borrowing constraint: if ``a < a_{\text{endo},1}``, consume all cash-on-hand

The EGM converges in 200--400 iterations for typical calibrations. The `compute_steady_state` function calls EGM internally at each bisection step:

```@example dsge_ha
spec_ks = load_ha_example(:krusell_smith)
ss_egm = compute_steady_state(spec_ks; K_init=10.0, r_bounds=(-0.02, 0.04),
                               max_iter=80, tol=1e-4)
report(ss_egm)
```

The steady state report displays convergence diagnostics, equilibrium prices, aggregate quantities, and wealth distribution statistics. A negative Euler error (in ``\log_{10}`` units) indicates the accuracy of the consumption policy --- values below ``-3`` are standard in the literature.

### VFI with Howard Improvement

When EGM is not applicable (non-separable utility, complex constraints), **Value Function Iteration** with **Howard improvement steps** provides a robust alternative. Each VFI iteration consists of one policy maximization step followed by ``K`` policy-evaluation steps (default ``K = 20``), which are cheap linear operations that dramatically accelerate convergence. The VFI solver is used automatically when `compute_steady_state` detects a two-asset model.

---

## Income Discretization

Idiosyncratic productivity follows an AR(1) process ``\log e' = \rho \log e + \sigma \varepsilon`` discretized onto a finite Markov chain.

### Rouwenhorst

The **Rouwenhorst (1995)** method constructs the transition matrix recursively. It is more accurate than Tauchen for highly persistent processes (``\rho > 0.9``).

```@example dsge_ha
inc7 = rouwenhorst(0.966, 0.5, 7)
(states = round.(inc7.states, digits=3),
 stationary = round.(inc7.stationary_dist, digits=4))
```

The 7-state discretization spans the ergodic support of the log-productivity process. The stationary distribution concentrates mass near the mean, with thin tails reflecting the high persistence.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `rho` | `Real` | — | Persistence parameter |
| `sigma` | `Real` | — | Standard deviation of innovations |
| `n` | `Int` | — | Number of grid points |

| Field | Type | Description |
|-------|------|-------------|
| `transition` | `Matrix{T}` | ``n \times n`` Markov transition matrix |
| `states` | `Vector{T}` | Grid of log-productivity levels |
| `stationary_dist` | `Vector{T}` | Ergodic distribution over states |

### Tauchen

The **Tauchen (1986)** method uses equally spaced grid points covering ``\pm m`` standard deviations and normal CDF transition probabilities.

```@example dsge_ha
inc_t = tauchen(0.9, 0.2, 5; m=3)
(states = round.(inc_t.states, digits=3),
 stationary = round.(inc_t.stationary_dist, digits=4))
```

With lower persistence (``\rho = 0.9``), Tauchen produces a wider grid and a flatter stationary distribution than Rouwenhorst. The ``m = 3`` setting covers three unconditional standard deviations on each side.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `m` | `Int` | `3` | Number of standard deviations to cover |

---

## Distribution Tracking

The cross-sectional wealth distribution ``\Gamma(a, e)`` evolves according to the **Young (2010) non-stochastic simulation** method. Given the savings policy ``a' = g(a, e)``, the distribution updates via a sparse transition matrix ``\Lambda``:

```math
D_{t+1} = \Lambda \, D_t
```

where ``\Lambda`` uses **lottery weights** to map off-grid savings back to grid points. If ``g(a_i, e_j)`` falls between ``a_k`` and ``a_{k+1}``, mass is split proportionally:

```math
\omega = \frac{a_{k+1} - g(a_i, e_j)}{a_{k+1} - a_k}
```

The transition matrix is sparse --- each column has at most ``2 N_e`` nonzero entries. The **stationary distribution** ``D^*`` satisfies ``D^* = \Lambda D^*`` and is found via power iteration. The `compute_steady_state` function handles distribution tracking internally. The resulting `HASteadyState` stores the distribution:

```@example dsge_ha
(shape = size(ss.distribution),
 total_mass = round(sum(ss.distribution), digits=10),
 aggregate_K = round(ss.aggregates[:K], digits=2))
```

The distribution is an ``N_a \times N_e`` matrix whose entries sum to unity. Aggregate capital integrates asset holdings against the distribution, providing the supply side of the capital market clearing condition.

---

## Steady State

The **stationary equilibrium** requires the individual problem, distribution, and prices to be mutually consistent. `compute_steady_state` bisects on the interest rate ``r``:

1. Guess ``r_{\text{mid}} = (r_{\text{lo}} + r_{\text{hi}}) / 2``
2. Compute prices ``(r, w)`` from the firm's first-order conditions given ``r_{\text{mid}}``
3. Solve the individual problem via EGM at those prices
4. Build the transition matrix and compute the stationary distribution
5. Aggregate capital supply: ``K_s = \int a \, d\Gamma(a, e)``
6. Compute capital demand ``K_d`` from the firm's FOC
7. If ``K_s > K_d``, interest rate is too low → ``r_{\text{hi}} = r_{\text{mid}}``; otherwise ``r_{\text{lo}} = r_{\text{mid}}``
8. Converge when ``|K_s - K_d| < \text{tol}``

```@example dsge_ha
spec_hank = load_ha_example(:one_asset_hank)
ss_hank = compute_steady_state(spec_hank; K_init=10.0, r_bounds=(-0.02, 0.04),
                                max_iter=80, tol=1e-4)
report(ss_hank)
```

The one-asset HANK steady state features a lower equilibrium interest rate than the standard Aiyagari economy due to the New Keynesian markup and dividend income. The Gini coefficient and wealth percentiles characterize the cross-sectional distribution, with the 90th percentile typically several times the median --- consistent with empirical wealth data.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `K_init` | `T` | `10.0` | Initial guess for aggregate capital |
| `r_bounds` | `Tuple{T,T}` | `(-0.01, 0.04)` | Bisection bounds for interest rate |
| `max_iter` | `Int` | `200` | Maximum bisection iterations |
| `tol` | `Real` | ``10^{-8}`` | Convergence tolerance on excess demand |
| `verbose` | `Bool` | `false` | Print iteration progress |

| Field | Type | Description |
|-------|------|-------------|
| `converged` | `Bool` | Whether bisection converged |
| `iterations` | `Int` | Number of bisection iterations |
| `prices` | `Dict{Symbol,T}` | Equilibrium prices (``r``, ``w``) |
| `aggregates` | `Dict{Symbol,T}` | Aggregate quantities (``K``, ``Y``, ``C``) |
| `distribution` | `Matrix{T}` | ``N_a \times N_e`` stationary distribution |
| `policies` | `Dict{Symbol,Matrix{T}}` | Policy functions (`:consumption`, `:savings`) |
| `excess_demand` | `T` | Final excess demand for capital |
| `euler_error` | `T` | ``\log_{10}`` Euler equation error |
| `grid` | `HAGrid{T}` | Asset grid used |

---

## Sequence-Space Jacobian

The **Sequence-Space Jacobian** method (Auclert, Bardóczy, Rognlie & Straub 2021) computes the ``T \times T`` Jacobian of aggregate outputs with respect to aggregate input sequences. The key idea: instead of tracking the full distribution as a state variable (as in Reiter), work directly with the impulse response sequences.

The algorithm computes a **fake news matrix** ``\mathcal{F}`` via:
1. **Backward iteration**: perturb a price at time ``s``, iterate the EGM backward to capture the expectation channel
2. **Forward iteration**: propagate the perturbed policies through the distribution forward in time
3. **Accumulation**: the true Jacobian ``\mathcal{J}`` is the cumulative sum of ``\mathcal{F}``

The resulting ``\mathcal{J}`` is converted to a minimal state-space realization via the **Ho-Kalman algorithm** (SVD of the Hankel matrix of IRF coefficients), producing a reduced `DSGESolution` compatible with all existing analysis functions.

```@example dsge_ha
sol_ssj = solve(spec; method=:ssj, ss=ss, T_horizon=50, n_reduced=15)
report(sol_ssj)
```

The SSJ method reduces the full sequence-space representation (dimension ``T``) to a compact state-space form with `n_reduced` states. The explained variance measures the fraction of aggregate dynamics captured by the truncated Ho-Kalman basis --- values above 99.9% confirm the reduction is adequate. The underlying steady state is reported alongside the reduction diagnostics.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `T_horizon` | `Int` | `300` | Truncation horizon for sequences |
| `n_reduced` | `Int` | `30` | Reduced state-space dimension (Ho-Kalman) |
| `dx` | `Real` | ``10^{-4}`` | Finite-difference step size |

| Field | Type | Description |
|-------|------|-------------|
| `method` | `Symbol` | Solution method (`:ssj` or `:reiter`) |
| `n_full_states` | `Int` | Full state-space dimension before reduction |
| `n_reduced` | `Int` | Reduced state-space dimension |
| `explained_variance` | `T` | Fraction of variance captured by truncation |
| `linear_solution` | `DSGESolution{T}` | Reduced-form state-space representation |
| `steady_state` | `HASteadyState{T}` | Underlying stationary equilibrium |

---

## Reiter Method

The **Reiter (2009)** method linearizes the entire system --- Euler equations, distribution evolution, and aggregate equilibrium --- around the stationary equilibrium. The distribution histogram becomes part of the state vector, yielding a large linear system that is reduced via SVD.

The implementation uses **observability-based SVD**: the reduction basis is built from the observability matrix ``[c', c' \Lambda', c' (\Lambda')^2, \ldots]'`` where ``c`` is the capital aggregation vector. This identifies the distribution directions most relevant for aggregate dynamics, achieving ``>99.9\%`` explained variance with 15--30 reduced states.

```@example dsge_ha
sol_reiter = solve(spec; method=:reiter, ss=ss, n_reduced=15)
report(sol_reiter)
```

The Reiter method produces an equivalent reduced state-space form to SSJ. The explained variance confirms that 15 reduced states capture nearly all aggregate dynamics. The two methods yield identical IRFs up to numerical precision for the same `n_reduced`, but differ in computational cost: SSJ is faster for models with few aggregate inputs, while Reiter scales better when many prices affect household decisions.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `n_reduced` | `Int` | `50` | Maximum reduced dimension |
| `dx` | `Real` | ``10^{-6}`` | Finite-difference step size |

---

## Krusell-Smith Method

The **Krusell-Smith (1998)** method approximates agents' forecasting rule with a **perceived law of motion** (PLM):

```math
\log K_{t+1} = b_0 + b_1 \log K_t
```

The algorithm iterates between simulation (using the PLM to forecast prices) and regression (updating PLM coefficients via OLS). Convergence requires ``R^2 > 0.9999``, reflecting the near-sufficiency of the first moment for forecasting.

!!! note "Technical Note"
    The PLM coefficients are updated with damping: ``b^{\text{new}} = 0.5 \, b^{\text{OLS}} + 0.5 \, b^{\text{old}}``. This prevents oscillation and ensures monotone convergence.

```@example dsge_ha
sol_ks = solve(spec; method=:krusell_smith, ss=ss,
               T_sim=500, T_burn=100, max_outer=3,
               rho_z=0.95, sigma_z=0.007)
report(sol_ks)
```

The PLM ``R^2`` near unity confirms that aggregate capital is approximately sufficient for forecasting prices --- the core insight of Krusell & Smith (1998). The PLM intercept and slope jointly characterize the aggregate law of motion. Unlike SSJ and Reiter, the Krusell-Smith method does not produce a reduced linear state-space form, so it cannot be used directly with `irf` or `fevd`.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `T_sim` | `Int` | `11000` | Simulation length |
| `T_burn` | `Int` | `1000` | Burn-in periods to discard |
| `max_outer` | `Int` | `20` | Maximum PLM iterations |
| `rho_z` | `Real` | `0.95` | Aggregate shock persistence |
| `sigma_z` | `Real` | `0.007` | Aggregate shock standard deviation |

| Field | Type | Description |
|-------|------|-------------|
| `converged` | `Bool` | Whether PLM coefficients converged |
| `iterations` | `Int` | Number of outer PLM iterations |
| `plm_coefficients` | `Dict{Symbol,Vector{T}}` | PLM regression coefficients per variable |
| `r_squared` | `Dict{Symbol,T}` | ``R^2`` of PLM regression per variable |
| `steady_state` | `HASteadyState{T}` | Underlying stationary equilibrium |

---

## Two-Asset HANK

The **two-asset HANK** model (Kaplan, Moll & Violante 2018) features households holding both liquid bonds ``b`` and illiquid equity ``a``, with a portfolio **adjustment cost** ``\chi(d, a)`` for accessing the illiquid asset:

```math
V(b, a, e) = \max_{c, b', d} \; u(c) + \beta \, \mathbb{E}\bigl[V(b', a', e') \mid e\bigr]
```

subject to:
```math
c + b' + d + \chi(d, a) = (1 + r_b) \, b + w \, e + T
```
```math
a' = (1 + r_a) \, a + d, \qquad b' \geq \underline{b}
```

where:
- ``b`` is liquid bonds, ``a`` is illiquid equity
- ``d`` is the deposit/withdrawal from the illiquid account
- ``\chi(d, a) = \chi_0 |d / a|^{\chi_1} \cdot a`` is the convex adjustment cost
- ``r_b, r_a`` are liquid and illiquid returns

The individual problem is solved via **nested EGM**: an outer loop over deposit choices with an inner EGM on the liquid dimension.

```@example dsge_ha
spec_2a = load_ha_example(:two_asset_hank)
(n_dims = spec_2a.grid.n_dims,
 grid_points = spec_2a.grid.n_points,
 labels = spec_2a.grid.labels,
 has_adjustment_cost = spec_2a.individual.adjustment_cost !== nothing)
```

The two-asset model uses a two-dimensional grid (liquid ``\times`` illiquid) with 2500 total points. The presence of the adjustment cost ``\chi(d, a)`` induces an inaction region where households neither deposit nor withdraw from the illiquid account, generating realistic portfolio rebalancing behavior.

---

## Built-in Examples

Three canonical models are available via `load_ha_example`:

| Model | Assets | Grid | Income | Key Feature |
|-------|--------|------|--------|-------------|
| `:krusell_smith` | 1 (``a \in [0, 200]``) | 200 pts | 7 states | Standard Aiyagari economy |
| `:one_asset_hank` | 1 (``b \in [-2, 50]``) | 200 pts | 7 states | NK with dividends, borrowing |
| `:two_asset_hank` | 2 (liquid + illiquid) | 50 × 50 | 7 states | Portfolio choice with adjustment cost |

```@example dsge_ha
for name in [:krusell_smith, :one_asset_hank, :two_asset_hank]
    s = load_ha_example(name)
    n = s.grid.n_dims
    g = join(s.grid.n_points, "×")
    println(rpad(name, 20), n, "-asset   β=", s.individual.beta, "   grid=", g)
end
nothing # hide
```

The Krusell-Smith economy is the simplest benchmark with a single asset and Cobb-Douglas production. The one-asset HANK adds New Keynesian features (sticky prices, monetary policy, dividends) and allows borrowing. The two-asset HANK introduces portfolio choice between liquid and illiquid assets, capturing the empirical finding that most household wealth is illiquid.

---

## Bayesian Estimation

Bayesian estimation of HA-DSGE models uses the **linearized reduced system** (from SSJ or Reiter) with a Kalman filter for likelihood evaluation. For each parameter draw ``\theta`` in the RWMH sampler:

1. Update model parameters
2. Re-solve the HA steady state (the expensive step)
3. Linearize via SSJ → produce a reduced `DSGESolution`
4. Build the state space → evaluate Kalman log-likelihood
5. Accept/reject via the Metropolis-Hastings ratio

```julia
using Distributions
result = estimate_dsge_bayes(spec, data, [0.36];
    priors=Dict(:alpha => Beta(5, 2)),
    observables=[:K], n_draws=5000, burnin=1000,
    ha_method=:ssj, ha_kwargs=(T_horizon=50, n_reduced=15))
```

!!! note "Technical Note"
    The inner loop re-solves the HA steady state at each draw, making estimation
    computationally intensive. For a one-asset model with 200 grid points, each
    likelihood evaluation takes approximately 0.5 seconds, yielding a 5000-draw
    chain in about 40 minutes.

The estimation output is a `BayesianDSGE` object with posterior draws, acceptance rate, and log-likelihood trace. Use `report(result)` for a formatted summary including posterior means, credible intervals, and convergence diagnostics.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `priors` | `Dict{Symbol,Distribution}` | — | Prior distributions keyed by parameter name |
| `observables` | `Vector{Symbol}` | — | Observable variable names matching data columns |
| `n_draws` | `Int` | `5000` | Total RWMH draws |
| `burnin` | `Int` | `1000` | Burn-in draws to discard |
| `ha_method` | `Symbol` | `:ssj` | Aggregate solution method (`:ssj` or `:reiter`) |
| `ha_kwargs` | `NamedTuple` | `()` | Keyword arguments passed to `solve` |

---

## Complete Example

A full workflow for a Krusell-Smith (1998) economy: load the model, compute the steady state, examine the wealth distribution, solve for aggregate dynamics, and simulate panel data.

```@example dsge_ha
ks = load_ha_example(:krusell_smith)
ss_ks = compute_steady_state(ks; K_init=10.0, r_bounds=(-0.02, 0.04),
                              max_iter=80, tol=1e-4)
report(ss_ks)
```

```@example dsge_ha
ineq_ks = inequality_irf(ss_ks; T_periods=5)
(gini = round(ineq_ks[:gini][1], digits=4),
 p50  = round(ineq_ks[:p50][1], digits=2),
 p90  = round(ineq_ks[:p90][1], digits=2))
```

The Gini coefficient reflects the degree of wealth concentration in the stationary equilibrium. The gap between the median and 90th percentile illustrates the right-skewed nature of the wealth distribution --- a robust feature of heterogeneous agent models with borrowing constraints and precautionary savings.

```@example dsge_ha
panel_ks = simulate_panel(ss_ks; N_agents=1000, T_periods=200)
size(panel_ks)
```

The simulated panel tracks 1000 agents over 200 periods, providing micro-level data for computing cross-sectional moments, transition matrices, and mobility statistics.

```julia
plot_result(ss_ks)                    # wealth distribution
plot_result(ss_ks; view=:lorenz)      # Lorenz curve with Gini
plot_result(ss_ks; view=:policy)      # policy functions by income
```

---

## Common Pitfalls

1. **Bisection bounds too narrow.** If `compute_steady_state` does not converge, widen `r_bounds`. The equilibrium interest rate can be negative in Aiyagari economies with patient agents.

2. **Grid too coarse near the borrowing constraint.** The EGM interpolation is least accurate near kinks in the policy function. Use `grid_type=:double_exp` (default) for denser spacing near the lower bound.

3. **Rouwenhorst vs Tauchen for persistent income.** For ``\rho > 0.95``, Rouwenhorst is significantly more accurate. Tauchen requires very fine grids to match the stationary distribution of highly persistent processes.

4. **SSJ truncation horizon too short.** If `T_horizon` is smaller than the half-life of the aggregate shock, the Jacobian is truncated prematurely. Use ``T_{\text{horizon}} \geq 3 / (1 - \rho_z)`` as a rule of thumb.

5. **Ho-Kalman `n_reduced` too small.** Check `explained_variance` in the `HADSGESolution` --- it should exceed 0.999. If not, increase `n_reduced`.

6. **Two-asset deposit grid resolution.** The nested EGM searches over a discrete deposit grid. With too few points (`n_deposit < 20`), the optimal deposit choice may be inaccurate near the adjustment cost kink.

---

## References

- Auclert, Adrien, Bence Bardóczy, Matthew Rognlie, and Ludwig Straub. 2021. "Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models." *Econometrica* 89 (5): 2375--2408. [DOI](https://doi.org/10.3982/ECTA17434)

- Carroll, Christopher D. 2006. "The Method of Endogenous Gridpoints for Solving Dynamic Stochastic Optimization Problems." *Economics Letters* 91 (3): 312--320. [DOI](https://doi.org/10.1016/j.econlet.2005.09.013)

- Kaplan, Greg, Benjamin Moll, and Giovanni L. Violante. 2018. "Monetary Policy According to HANK." *American Economic Review* 108 (3): 697--743. [DOI](https://doi.org/10.1257/aer.20160042)

- Krusell, Per, and Anthony A. Smith Jr. 1998. "Income and Wealth Heterogeneity in the Macroeconomy." *Journal of Political Economy* 106 (5): 867--896. [DOI](https://doi.org/10.1086/250034)

- Reiter, Michael. 2009. "Solving Heterogeneous-Agent Models by Projection and Perturbation." *Journal of Economic Dynamics and Control* 33 (3): 649--665. [DOI](https://doi.org/10.1016/j.jedc.2008.08.010)

- Rouwenhorst, K. Geert. 1995. "Asset Pricing Implications of Equilibrium Business Cycle Models." In *Frontiers of Business Cycle Research*, edited by Thomas F. Cooley, 294--330. Princeton: Princeton University Press.

- Tauchen, George. 1986. "Finite State Markov-Chain Approximations to Univariate and Vector Autoregressions." *Economics Letters* 20 (2): 177--181. [DOI](https://doi.org/10.1016/0165-1765(86)90168-0)

- Young, Eric R. 2010. "Solving the Incomplete Markets Model with Aggregate Uncertainty Using the Krusell--Smith Algorithm and Non-Stochastic Simulations." *Journal of Economic Dynamics and Control* 34 (1): 36--41. [DOI](https://doi.org/10.1016/j.jedc.2008.11.010)
