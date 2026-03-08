# DSGE Historical Decomposition — Design Document

**Date:** 2026-03-08
**Status:** Approved

## Overview

Add historical decomposition (HD) to the DSGE module, covering linear (1st-order),
nonlinear (higher-order perturbation), and Bayesian estimation paths. Decomposes
observed (and optionally all state) variable movements into contributions from
individual structural shocks, using smoothed shock estimates from a dedicated
Kalman/particle smoother.

## Architecture

Three layers:

1. **Smoother layer** (`src/dsge/smoother.jl`) — standalone Kalman and particle smoothers
2. **HD engine** (`src/dsge/hd.jl`) — shock contribution decomposition
3. **Result types** — reuse existing `HistoricalDecomposition` / `BayesianHistoricalDecomposition`
   from the VAR module; new `KalmanSmootherResult{T}` for smoother output

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Smoother approach | Dedicated DSGE RTS smoother | Full control, stores all intermediate states |
| Nonlinear smoother | FFBSi (Godsill-Doucet-West 2004) | O(N^2 T) but robust; optimized with pre-allocation, BLAS, threading |
| Observable scope | Full states internally, project to observables by default (`states=:all` option) | Maximum flexibility |
| Bayesian default | Re-solve + re-smooth at each posterior draw | Correct parameter uncertainty propagation |
| Bayesian fast path | `mode_only=true` keyword | Smooth at posterior mode only |
| Result types | Reuse existing VAR HD types | All accessors, plotting, `report()` work out of the box |
| Smoother result | New `KalmanSmootherResult{T}` type | Independently useful, first-class export |
| Shock extraction | State-space inversion via pseudo-inverse | Shocks enter linearly at all perturbation orders |
| Nonlinear HD method | Counterfactual simulation | Shock contributions not additive at higher orders |

## New Types

### KalmanSmootherResult{T}

```julia
struct KalmanSmootherResult{T<:AbstractFloat}
    smoothed_states::Matrix{T}        # n_states x T_obs
    smoothed_covariances::Array{T,3}  # n_states x n_states x T_obs
    smoothed_shocks::Matrix{T}        # n_shocks x T_obs
    filtered_states::Matrix{T}        # n_states x T_obs
    filtered_covariances::Array{T,3}  # n_states x n_states x T_obs
    predicted_states::Matrix{T}       # n_states x T_obs
    predicted_covariances::Array{T,3} # n_states x n_states x T_obs
    log_likelihood::T
end
```

## API

### Smoother (standalone)

```julia
dsge_smoother(ss::DSGEStateSpace{T}, data::Matrix{T}) -> KalmanSmootherResult{T}
dsge_particle_smoother(nss::NonlinearStateSpace{T}, data::Matrix{T};
                       N=1000, N_back=100, rng=default_rng()) -> KalmanSmootherResult{T}
```

### Historical Decomposition

```julia
# Frequentist linear
historical_decomposition(sol::DSGESolution{T}, data, observables;
                         states=:observables) -> HistoricalDecomposition{T}

# Frequentist nonlinear
historical_decomposition(sol::PerturbationSolution{T}, data, observables;
                         N=1000, N_back=100, states=:observables) -> HistoricalDecomposition{T}

# Bayesian (default: re-solve at each draw)
historical_decomposition(post::BayesianDSGE{T}, data, observables;
                         mode_only=false, n_draws=200,
                         quantiles=[0.16, 0.5, 0.84]) -> BayesianHistoricalDecomposition{T}
```

## Algorithms

### Linear RTS Smoother

1. **Forward pass**: Kalman filter storing x_{t|t}, P_{t|t}, x_{t|t-1}, P_{t|t-1}
   - Same missing-data handling as existing `_kalman_loglikelihood`
   - Pre-allocated workspace, `mul!` throughout, symmetrize at each step
2. **Backward pass**: RTS smoother
   - J_t = P_{t|t} * G1' * P_{t+1|t}^{-1}
   - x_{t|T} = x_{t|t} + J_t * (x_{t+1|T} - x_{t+1|t})
   - P_{t|T} = P_{t|t} + J_t * (P_{t+1|T} - P_{t+1|t}) * J_t'
3. **Shock extraction**: eps_t = impact^+ * (x_{t|T} - G1 * x_{t-1|T} - C_sol)

### FFBSi Particle Smoother

1. **Forward pass**: Bootstrap particle filter
   - Propagation via full nonlinear transition (pruned components)
   - Systematic resampling when ESS < N/2
   - Store all particles, weights, ancestor indices
2. **Backward simulation** (Godsill-Doucet-West 2004):
   - Draw N_back trajectories from final filtered distribution
   - Backward weights: w_tilde_t^(i) = w_t^(i) * p(x_{t+1}^(b) | x_t^(i))
   - Transition density: Gaussian via first-order impact (shocks enter linearly)
3. **Optimizations**:
   - Pre-compute Cholesky of Q once
   - Vectorize backward weight computation (batch matrix ops)
   - `@threads` over backward trajectories
   - Reject-accept acceleration for low-weight particles
4. **Shock extraction**: eps_t^(b) = impact^+ * (x_t^(b) - f(x_{t-1}^(b)))

### Linear HD Engine

- Structural MA coefficients: Theta_s = Z * G1^s * impact
- contributions[t, i, j] = sum_{s=0}^{t-1} Theta_s[i, j] * eps_j(t-s)
- initial_conditions = actual - sum_j contributions

### Nonlinear HD Engine (Counterfactual)

- Baseline: simulate forward with all smoothed shocks
- For each shock j: simulate with eps_j zeroed out
- contribution_j = baseline - path_without_j
- Residual (interaction terms) attributed to initial conditions

### Bayesian HD

- Default (`mode_only=false`): subsample n_draws from posterior, re-solve + re-smooth each,
  stack contributions, compute quantiles
- Fast path (`mode_only=true`): modal solution only, single smoother run

## Files

| File | Purpose |
|------|---------|
| `src/dsge/smoother.jl` | RTS smoother + FFBSi particle smoother |
| `src/dsge/hd.jl` | HD dispatches for DSGESolution, PerturbationSolution, BayesianDSGE |
| `test/dsge/test_dsge_hd.jl` | Test suite |
| `docs/src/dsge_hd.md` | Documentation page |

## Exports

- `KalmanSmootherResult` (new type)
- `dsge_smoother` (new function)
- `dsge_particle_smoother` (new function)
- `historical_decomposition` (existing — new method dispatches only)

## Testing Strategy

1. **Smoother correctness** — known DGP with known shocks, verify recovery
2. **HD identity** — contributions + initial_conditions approx actual
3. **Single-shock DGP** — one active shock dominates, others approx 0
4. **Bayesian HD** — small model, few draws, verify quantile structure + mode_only path
5. **Nonlinear HD** — 2nd-order model, counterfactual decomposition runs, identity holds approximately
6. **Particle vs RTS** — linear DGP, verify particle smoother converges to RTS answer
7. **Missing data** — NaN handling in smoother
8. **states=:all** — full state decomposition dimensions
9. **Edge cases** — single observable, single shock, short sample

## References

- Godsill, S. J., Doucet, A., & West, M. (2004). Monte Carlo smoothing for nonlinear time series. JASA.
- Koopman, S. J. (1993). Disturbance smoother for state space models. Biometrika.
- Hamilton, J. D. (1994). Time Series Analysis. Ch. 13.
- Herbst, E. & Schorfheide, F. (2015). Bayesian Estimation of DSGE Models.
