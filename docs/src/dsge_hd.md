# [DSGE Historical Decomposition](@id dsge_hd_page)

Historical decomposition for DSGE models decomposes observed variable movements into contributions from individual structural shocks plus initial conditions. The package provides three methods spanning linear, nonlinear, and Bayesian DSGE models:

- **Linear DSGE**: Exact additive decomposition via the Kalman smoother (Rauch, Tung & Striebel 1965) and structural MA coefficients
- **Nonlinear DSGE**: Counterfactual simulation using the FFBSi particle smoother (Godsill, Doucet & West 2004) for higher-order perturbation solutions
- **Bayesian DSGE**: Posterior credible bands by re-solving at each posterior draw (Herbst & Schorfheide 2015)

```@setup dsge_hd
using MacroEconometricModels, Random, Distributions
Random.seed!(42)

# RBC model for all examples
_spec_hd = @dsge begin
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
_spec_hd = compute_steady_state(_spec_hd)
_sol_hd = solve(_spec_hd)
_data_hd = simulate(_sol_hd, 100)
```

## Quick Start

**Recipe 1: Linear DSGE historical decomposition**

```@example dsge_hd
sol = solve(_spec_hd)
data = simulate(sol, 100)

# Decompose observed data into shock contributions
hd = historical_decomposition(sol, data, [:Y, :C, :K, :A])
report(hd)
```

**Recipe 2: Verify the decomposition identity**

```@example dsge_hd
# The identity y_t = sum_j HD_j(t) + initial(t) holds to machine precision
verified = verify_decomposition(hd)
```

**Recipe 3: HD visualization**

```@example dsge_hd
hd_plot = historical_decomposition(_sol_hd, _data_hd, [:Y, :C, :K, :A])
nothing # hide
```

```julia
plot_result(hd_plot)
```

```@raw html
<iframe src="../assets/plots/dsge_hd.html" width="100%" height="600" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## Linear DSGE HD

Historical decomposition for linear DSGE models derives from the structural moving average (VMA) representation of the state-space solution. The Kalman smoother (Rauch, Tung & Striebel 1965) extracts smoothed structural shocks from the data, and the structural MA coefficients attribute each variable's movement to individual shocks.

```math
y_{i,t} = \sum_{j=1}^{n_{shocks}} \sum_{s=0}^{t-1} (\Theta_s)_{ij} \, \varepsilon_j(t-s) + \text{initial}_i(t)
```

where:
- ``y_{i,t}`` is the deviation of variable ``i`` from steady state at time ``t``
- ``\Theta_s = Z \cdot G_1^s \cdot \text{impact}`` are the structural MA coefficients at lag ``s``
- ``G_1`` is the state transition matrix from the linear solution ``y_t = G_1 y_{t-1} + \text{impact} \cdot \varepsilon_t``
- ``Z`` is the observation selection matrix mapping states to observables
- ``\varepsilon_j(t-s)`` is the smoothed structural shock ``j`` at time ``t-s``
- ``\text{initial}_i(t)`` captures the contribution of the initial state

The decomposition satisfies an exact additive identity --- the sum of all shock contributions plus initial conditions recovers the observed data to machine precision.

```@example dsge_hd
# Linear HD with Cholesky-ordered state space
hd_lin = historical_decomposition(sol, data, [:Y, :C, :K, :A])

# Verify the additive identity
verify_decomposition(hd_lin)
```

```@example dsge_hd
# Extract the technology shock's contribution to output
tech_to_output = contribution(hd_lin, "Y", "ε_A")

# Total shock-driven component of output (excludes initial conditions)
total_Y = total_shock_contribution(hd_lin, "Y")
nothing # hide
```

The single-shock RBC model attributes all output movements to the technology shock ``\varepsilon_A``. In multi-shock models, the decomposition reveals which shocks drove specific historical episodes.

### Decomposing All States

By default, only observed variables are decomposed. To decompose all state variables including latent states:

```@example dsge_hd
hd_all = historical_decomposition(sol, data, [:Y, :C, :K, :A]; states=:all)
report(hd_all)
```

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `states` | `Symbol` | `:observables` | Decompose `:observables` only or `:all` states |
| `measurement_error` | `Vector` | `nothing` | Measurement error standard deviations (small diagonal default) |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `contributions` | `Array{T,3}` | ``T \times n_{vars} \times n_{shocks}`` shock contributions |
| `initial_conditions` | `Matrix{T}` | ``T \times n_{vars}`` initial condition component |
| `actual` | `Matrix{T}` | ``T \times n_{vars}`` data in deviations from steady state |
| `shocks` | `Matrix{T}` | ``T \times n_{shocks}`` smoothed structural shocks |
| `T_eff` | `Int` | Number of time periods |
| `variables` | `Vector{String}` | Variable names |
| `shock_names` | `Vector{String}` | Shock names |
| `method` | `Symbol` | `:dsge_linear` |

---

## Nonlinear DSGE HD

For higher-order perturbation solutions, the structural MA representation is not available because shock effects are not additive. The package uses the FFBSi particle smoother (Godsill, Doucet & West 2004) to extract smoothed state trajectories, then computes each shock's contribution via counterfactual simulation.

The counterfactual approach computes each shock's contribution as:

```math
\text{HD}_j(t) = x_t^{\text{baseline}} - x_t^{\text{cf}_j}
```

where:
- ``x_t^{\text{baseline}}`` is the baseline path simulated with all smoothed shocks
- ``x_t^{\text{cf}_j}`` is the counterfactual path simulated with shock ``j`` zeroed out
- Nonlinear interaction terms are attributed to initial conditions

!!! note "Technical Note"
    The FFBSi smoother runs a bootstrap particle filter forward pass with `N` particles,
    followed by `N_back` backward simulation trajectories. Increasing `N` improves the
    filtering approximation; increasing `N_back` reduces Monte Carlo variance in the
    smoothed shocks. The default `N=1000, N_back=100` balances accuracy and speed.

```@example dsge_hd
# Second-order perturbation solution
psol = perturbation_solver(_spec_hd; order=2)

# Nonlinear HD via counterfactual simulation
hd_nl = historical_decomposition(psol, _data_hd, [:Y, :A];
                                  N=200, N_back=50, rng=Random.MersenneTwister(42))
report(hd_nl)
```

Since the RBC model has a single shock, the counterfactual approach reduces to zeroing out the only shock and attributing the full baseline path to it. In multi-shock nonlinear models, interaction terms between shocks produce a non-zero initial conditions component even at interior periods.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `states` | `Symbol` | `:observables` | Decompose `:observables` only or `:all` states |
| `measurement_error` | `Vector` | `nothing` | Measurement error standard deviations |
| `N` | `Int` | `1000` | Number of forward particles |
| `N_back` | `Int` | `100` | Number of backward simulation trajectories |
| `rng` | `AbstractRNG` | `default_rng()` | Random number generator |

---

## Bayesian DSGE HD

For Bayesian DSGE posteriors (from `estimate_dsge_bayes`), historical decomposition accounts for parameter uncertainty by re-solving the model and re-smoothing at each posterior draw (Herbst & Schorfheide 2015, Ch. 6).

Two modes are available:

- **`mode_only=true`**: Fast path using only the posterior mode solution. Returns a standard `HistoricalDecomposition{T}` with `method=:dsge_bayes_mode`.
- **`mode_only=false`** (default): Full posterior. Subsamples `n_draws` posterior parameter draws, re-solves and re-smooths at each, and computes pointwise quantile bands. Returns `BayesianHistoricalDecomposition{T}` with `method=:dsge_bayes`.

```@example dsge_hd
# Simulate data and estimate Bayesian DSGE
Y_bayes = simulate(_sol_hd, 100)
Y_obs = Y_bayes[:, [1, 4]]  # observe Y and A

bayes = estimate_dsge_bayes(_spec_hd, Y_obs, [0.9];
    priors=Dict(:ρ => Beta(5, 2)),
    method=:mh, n_draws=2000, burnin=1000,
    observables=[:Y, :A])

# Fast path: posterior mode only
hd_mode = historical_decomposition(bayes, Y_bayes, [:Y, :A]; mode_only=true)
report(hd_mode)
```

```@example dsge_hd
# Full posterior: re-solve at each draw with credible bands
hd_bayes = historical_decomposition(bayes, Y_bayes, [:Y, :A];
                                     n_draws=50, quantiles=[0.16, 0.5, 0.84])
report(hd_bayes)
```

The `mode_only` path is orders of magnitude faster --- it calls the linear HD exactly once. The full posterior path iterates over subsampled draws, discarding any that produce indeterminate solutions. Wide credible bands indicate parameter uncertainty substantially affects the attribution of observed movements to specific shocks.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `mode_only` | `Bool` | `false` | Use posterior mode only (fast, no credible bands) |
| `n_draws` | `Int` | `200` | Number of posterior draws to subsample |
| `quantiles` | `Vector{<:Real}` | `[0.16, 0.5, 0.84]` | Quantile levels for credible bands |
| `measurement_error` | `Vector` | `nothing` | Measurement error standard deviations |
| `states` | `Symbol` | `:observables` | Decompose `:observables` only or `:all` states |

---

## Smoother API

The Kalman and particle smoothers can be used independently of the historical decomposition. The smoother extracts smoothed states, covariances, structural shocks, and the log-likelihood from a DSGE state space.

### RTS Smoother (Linear)

!!! note "Advanced Usage"
    The state-space construction functions (`_build_observation_equation`, `_build_state_space`)
    are internal helpers. The `historical_decomposition` function calls them automatically.
    Use the standalone smoother only when you need smoothed states or shocks without the
    full decomposition.

```@example dsge_hd
# Build the state space (internal helpers — subject to change)
observables = [:Y, :A]
Z, d, H = MacroEconometricModels._build_observation_equation(_spec_hd, observables, nothing)
ss = MacroEconometricModels._build_state_space(_sol_hd, Z, d, H)

# Data in deviations from steady state (n_obs × T_obs)
data_dev = Matrix(_data_hd[:, [1, 4]]' .- _spec_hd.steady_state[[1, 4]])

# Run the RTS smoother
smoother = dsge_smoother(ss, data_dev)
smoother
```

The smoother handles missing data (NaN entries) by reducing the observation dimension for periods with missing values. This enables estimation with ragged-edge or mixed-frequency data.

### FFBSi Particle Smoother (Nonlinear)

```@example dsge_hd
# Build nonlinear state space (internal helper — subject to change)
nss = MacroEconometricModels._build_nonlinear_state_space(psol, Z, d, H)

# Run the particle smoother
psmoother = dsge_particle_smoother(nss, data_dev; N=200, N_back=50)
psmoother
```

### KalmanSmootherResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `smoothed_states` | `Matrix{T}` | ``n_{states} \times T`` smoothed state means |
| `smoothed_covariances` | `Array{T,3}` | ``n_{states} \times n_{states} \times T`` smoothed covariances |
| `smoothed_shocks` | `Matrix{T}` | ``n_{shocks} \times T`` smoothed structural shocks |
| `filtered_states` | `Matrix{T}` | ``n_{states} \times T`` filtered state means |
| `filtered_covariances` | `Array{T,3}` | ``n_{states} \times n_{states} \times T`` filtered covariances |
| `predicted_states` | `Matrix{T}` | ``n_{states} \times T`` one-step-ahead predicted means |
| `predicted_covariances` | `Array{T,3}` | ``n_{states} \times n_{states} \times T`` predicted covariances |
| `log_likelihood` | `T` | Log-likelihood from the forward pass |

---

## Complete Example

This example builds a complete DSGE historical decomposition workflow: specify, solve, simulate data, decompose, verify, and visualize.

```@example dsge_hd
# Specify and solve an RBC model
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

sol_ce = solve(spec)
data_ce = simulate(sol_ce, 100)

# Historical decomposition
hd_ce = historical_decomposition(sol_ce, data_ce, [:Y, :C, :K, :A])
report(hd_ce)
```

```@example dsge_hd
# Verify the additive identity
verify_decomposition(hd_ce)
```

```@example dsge_hd
# Technology shock contribution to output
contribution(hd_ce, "Y", "ε_A")[1:5]
```

```julia
plot_result(hd_ce)
```

```@raw html
<iframe src="../assets/plots/dsge_hd.html" width="100%" height="600" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The stacked bar chart shows the technology shock's contribution to each variable over time. In this single-shock model, the shock explains all variation beyond initial conditions. The actual vs reconstructed line chart confirms the decomposition identity holds: the sum of shock contributions and initial conditions exactly recovers the observed data.

---

## Common Pitfalls

1. **Data must be in levels, not deviations.** The `historical_decomposition` function subtracts the steady state internally. Passing data already in deviations double-subtracts and produces incorrect decompositions.

2. **Observable list must match data columns.** The `observables` vector specifies which endogenous variables are observed. The ordering must match the column ordering of the data matrix. Mismatched ordering silently produces wrong decompositions.

3. **Nonlinear HD is not additive.** For higher-order perturbation solutions, shock contributions computed via counterfactual simulation do not sum exactly to the observed data. The interaction terms are attributed to initial conditions. Use `verify_decomposition` with a looser tolerance for nonlinear models.

4. **Particle smoother is stochastic.** The FFBSi smoother produces different results across runs. Set `rng=Random.MersenneTwister(seed)` for reproducibility. Increase `N` and `N_back` to reduce Monte Carlo variance.

5. **Bayesian HD discards indeterminate draws.** When re-solving at posterior parameter draws, any draw that produces an indeterminate solution is silently discarded. If many draws are discarded, the credible bands may be too narrow. Check that the prior supports the determinacy region.

---

## References

- Canova, F. (2007). *Methods for Applied Macroeconomic Research*.
  Princeton University Press. ISBN 978-0-691-11583-1.

- Godsill, S. J., Doucet, A., & West, M. (2004). Monte Carlo Smoothing for Nonlinear Time Series.
  *Journal of the American Statistical Association*, 99(465), 156--168. [DOI](https://doi.org/10.1198/016214504000000151)

- Herbst, E. P., & Schorfheide, F. (2015). *Bayesian Estimation of DSGE Models*.
  Princeton University Press. ISBN 978-0-691-16108-2.

- Rauch, H. E., Tung, F., & Striebel, C. T. (1965). Maximum Likelihood Estimates of Linear Dynamic Systems.
  *AIAA Journal*, 3(8), 1445--1450. [DOI](https://doi.org/10.2514/3.3166)

```@docs
historical_decomposition
dsge_smoother
dsge_particle_smoother
KalmanSmootherResult
```
