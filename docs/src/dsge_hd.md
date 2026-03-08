# DSGE Historical Decomposition

Historical decomposition for DSGE models decomposes observed variable movements into
contributions from individual structural shocks, using the Kalman smoother (linear models)
or FFBSi particle smoother (nonlinear models) to extract smoothed structural shocks.

## Quick Start

```julia
# Define and solve DSGE model
spec = @dsge begin
    parameters: rho_y = 0.8, rho_pi = 0.5, rho_r = 0.6
    endogenous: y, pi_var, r
    exogenous: eps_y, eps_pi, eps_r
    y[t] = rho_y * y[t-1] + eps_y[t]
    pi_var[t] = rho_pi * pi_var[t-1] + eps_pi[t]
    r[t] = rho_r * r[t-1] + eps_r[t]
end
sol = solve(spec)

# Simulate or load observed data (T_obs x n_endog)
data = simulate(sol, 100)

# Historical decomposition
hd = historical_decomposition(sol, data, [:y, :pi_var, :r])
report(hd)

# Access individual contributions
c = contribution(hd, "y", "eps_y")

# Verify decomposition identity
verify_decomposition(hd)

# Plot
plot_result(hd)
```

## Kalman Smoother

The Rauch-Tung-Striebel (RTS) smoother can be used independently to extract
smoothed states, covariances, and structural shocks from a linear DSGE model:

```julia
observables = [:y, :pi_var, :r]
Z, d, H = MacroEconometricModels._build_observation_equation(spec, observables, nothing)
ss = MacroEconometricModels._build_state_space(sol, Z, d, H)

# Data must be n_obs x T_obs in deviations from steady state
data_dev = Matrix(data' .- spec.steady_state)
smoother = dsge_smoother(ss, data_dev)

smoother.smoothed_states     # n_states x T_obs
smoother.smoothed_shocks     # n_shocks x T_obs
smoother.log_likelihood      # scalar
```

The smoother handles missing data (NaN entries) by reducing the observation
dimension for periods with missing values.

## Nonlinear Models

For higher-order perturbation solutions, historical decomposition uses the
FFBSi particle smoother (Godsill, Doucet & West 2004) to extract smoothed
state trajectories, then computes each shock's contribution via counterfactual
simulation.

```julia
psol = perturbation_solver(spec; order=2)
hd = historical_decomposition(psol, data, [:y, :pi_var];
                               N=1000, N_back=100)
```

Since shock contributions are not additive at higher orders due to nonlinear
interactions, the counterfactual approach computes each shock's contribution
as the difference between the baseline path and a path with that shock zeroed
out. Interaction terms are attributed to initial conditions.

## Bayesian HD

For Bayesian DSGE posteriors, historical decomposition can account for parameter
uncertainty by re-solving and re-smoothing at each posterior draw:

```julia
# Full posterior (re-solve at each draw)
hd = historical_decomposition(posterior, data, observables;
                               n_draws=200, quantiles=[0.16, 0.5, 0.84])

# Fast mode (posterior mode only)
hd = historical_decomposition(posterior, data, observables; mode_only=true)
```

The full posterior path returns a `BayesianHistoricalDecomposition` with
quantile bands. The `mode_only` option returns a standard
`HistoricalDecomposition` using only the posterior mode solution.

## Decomposing All States

By default, only observed variables are decomposed. To decompose all state
variables (including latent states):

```julia
hd = historical_decomposition(sol, data, observables; states=:all)
```

## API Reference

```@docs
dsge_smoother
dsge_particle_smoother
KalmanSmootherResult
```
