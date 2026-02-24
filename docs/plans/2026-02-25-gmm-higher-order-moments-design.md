# GMM Estimation with Higher-Order Perturbation Moments — Design

**Issue**: Extension to #48 (higher-order perturbation)
**Date**: 2026-02-25

## Goal

Extend GMM estimation for DSGE models to exploit the richer moment conditions available from 2nd and 3rd order perturbation solutions. First-order perturbation yields zero-mean variables, so only covariances and autocovariances can be matched. Higher-order solutions produce non-zero means (risk correction), giving additional moment conditions and more efficient estimation.

## References

- Andreasen, Martin M., Jesper Rangvid & Maik Schmeling. 2019. "Risk and Return in Bond, Currency, and Equity Markets." *Review of Finance* 23 (5): 891–923.
- Andreasen, Martin M., Jesús Fernández-Villaverde & Juan F. Rubio-Ramírez. 2018. "The Pruned State-Space System for Non-Linear DSGE Models: Theory and Empirical Applications." *Review of Economic Studies* 85 (1): 1–49.
- Kim, Jinill, Sunghyun Kim, Ernst Schaumburg & Christopher A. Sims. 2008. "Calculating and Using Second-Order Accurate Solutions of Discrete Time Dynamic Equilibrium Models." *Journal of Economic Dynamics and Control* 32 (11): 3397–3414.
- MATLAB reference implementation: `GMM_ThirdOrder_v2` (Andreasen's code)

## Problem

**Current state**: `analytical_moments(sol::PerturbationSolution; lags)` uses simulation (T=100k) for order >= 2, returning covariances + diagonal autocovariances. The `estimate_dsge(...; method=:analytical_gmm)` always solves via `:gensys` (first-order), missing the richer moments.

**What's missing**:
1. Closed-form moments for 2nd order via augmented-state Lyapunov
2. Mean moments (E[y]) — non-zero for order >= 2 due to risk correction
3. Full product moments E[y_i * y_j] (not just centered covariances)
4. Estimation pipeline that passes `solve_method`/`solve_order` to the perturbation solver
5. Proper data moment computation matching the model moment format

## Algorithm

### Augmented-State Lyapunov for 2nd Order (Andreasen et al. 2018)

The pruned state-space system for 2nd order is:

```
x_f(t+1) = h_x · x_f(t) + η · ε(t)
x_s(t+1) = h_x · x_s(t) + (1/2) · H̃_xx · (x_f ⊗ x_f) + (1/2) · h_σσ
```

Define the augmented state:

```
z(t) = [x_f(t); x_s(t); vec(x_f(t) ⊗ x_f(t))]
```

with dimension `nz = 2·nx + nx²`.

**Transition matrix A** (block-lower-triangular):

```
A = [ h_x      0        0              ]   nz × nz
    [ 0        h_x      (1/2)·H̃_xx    ]
    [ 0        0        h_x ⊗ h_x      ]
```

**Constant vector c**:

```
c = [ 0                              ]
    [ (1/2)·h_σσ                     ]
    [ vec(η·η')                      ]
```

**Unconditional mean**: `E[z] = (I - A)⁻¹ · c`

**Innovation covariance** `Var_inov` (nz × nz):

The innovation to z(t) is `u(t) = z(t) - A·z(t-1) - c`, and Var_inov = E[u·u']. Key blocks:

- Block (1,1): `η·η'` (nx × nx)
- Block (1,3): Third-moment term `η · E[ε⊗ε²] · (η⊗η)'` (zero for symmetric shocks)
- Block (3,3): Quartic term involving `E[ε⁴] = 3` for Gaussian shocks, plus cross-terms with `E[x_f · x_f']`

The full Var_inov computation uses the 3rd and 4th moments of the shock distribution: `vectorMom3 = 0` (symmetric), `vectorMom4 = 3` (Gaussian).

**Variance**: Solve `Var_z = A · Var_z · A' + Var_inov` via `_dlyap_doubling`.

**Output mapping**:

```
y(t) = C · z(t) + d + [noise from contemporaneous shocks]
```

where:
```
C = [g_x   g_x   (1/2)·G̃_xx]     ny × nz
d = (1/2)·g_σσ                      ny × 1
```

Then:
```
E[y] = C · E[z] + d + g_ss_adjustment
Var_y = C · Var_z · C' + [contemporaneous shock contribution]
Cov_y(lag) = C · A^lag · Var_z · C'
```

### Moment Conditions for GMM

Following the MATLAB reference, the moment vector for estimation is:

| Moment Type | Count | Description |
|---|---|---|
| E[y_i] | ny | Means (zero for order 1, non-zero for order ≥ 2) |
| E[y_i · y_j], i ≤ j | ny·(ny+1)/2 | Upper-triangle product moments |
| E[y_i,t · y_i,t-k] | ny × num_lags | Diagonal autocovariances at selected lags |
| **Total** | ny + ny·(ny+1)/2 + ny·num_lags | |

For ny=3 variables and 3 autocovariance lags: 3 + 6 + 9 = 18 moment conditions.

**Important**: These are *product moments* E[y·y'], not centered covariances. This is because E[y·y'] = Cov(y) + E[y]·E[y]', and matching both means and product moments is equivalent to matching means and covariances. This follows the MATLAB reference directly.

### Moment Selection

Not all moments need to be included. An `inclMoms` binary selector (following the MATLAB code) allows choosing which means, which product moments, and which autocovariance lags to include. This is important because:
- Some model variables may not be observed in data
- Autocovariances at very long lags add noise without much information
- Excluding weakly informative moments improves finite-sample performance

### Two-Step GMM with Optimal Weighting

**Step 1**: Minimize with diagonal weighting (diagonal of HAC long-run variance estimate):
```
θ̂₁ = argmin (m_data - m_model(θ))' · W_diag · (m_data - m_model(θ))
```

**Step 2**: Compute optimal weighting matrix from Step 1 model moments, re-optimize:
```
W_opt = S⁻¹  where S = Bartlett HAC estimator of long-run variance
θ̂₂ = argmin (m_data - m_model(θ))' · W_opt · (m_data - m_model(θ))
```

The existing `estimate_gmm` already supports `:two_step` weighting with Bartlett kernel HAC. The only adaptation needed is that analytical-moment GMM has a single "observation" (the moment conditions are deterministic given θ), so the weighting matrix is constructed from the data's h-function rather than from residuals.

### Standard Errors

For optimal weighting: `Var(θ̂) = (1/T) · (D' · W · D)⁻¹`

where D is the Jacobian ∂m_model/∂θ (computed via numerical finite differences).

For non-optimal weighting (sandwich):
`Var(θ̂) = (1/T) · (D'WD)⁻¹ · D'W · S · W'D · (D'WD)⁻¹`

### J-Test

Hansen's overidentification test: `J = T · (m_data - m_model)' · W · (m_data - m_model) ~ χ²(n_mom - n_params)`

### Third-Order Moments

For 3rd order perturbation, the augmented state expands to:

```
z = [x_f; x_s; vec(x_f⊗x_f); x_rd; vec(x_f⊗x_s); vec(x_f⊗x_f⊗x_f)]
```

with dimension `3·nx + 2·nx² + nx³`. The innovation variance computation requires 5th and 6th order shock moments (vectorMom5=0, vectorMom6=15 for Gaussian). This is extremely complex (1327 lines in the MATLAB reference).

**Pragmatic approach**: Implement closed-form moments for order 2 only. For order 3, use long-simulation moments (T=500k with antithetic variates and fixed seed for reproducibility in the optimizer). The closed-form 3rd-order can be added as a future enhancement.

## Type Changes

### No New Types

The existing `DSGEEstimation{T}` and `GMMModel{T}` are sufficient. The `final_sol` field of `DSGEEstimation` can hold either `DSGESolution` or `PerturbationSolution`.

**BUT**: `DSGEEstimation{T}` currently types `final_sol` as `DSGESolution{T}`. This needs to be relaxed:

```julia
# In types.jl — change field type
struct DSGEEstimation{T<:AbstractFloat} <: AbstractDSGEModel
    ...
    final_sol::Union{DSGESolution{T}, PerturbationSolution{T}}
    ...
end
```

## New Functions

### `_compute_data_moments(data; lags, observable_indices)`

Compute the data moment vector matching the model moment format:

```julia
function _compute_data_moments(data::Matrix{T};
                                lags::Vector{Int}=[1],
                                observable_indices::Union{Nothing,Vector{Int}}=nothing) where {T}
    # If observable_indices given, select columns
    Y = observable_indices === nothing ? data : data[:, observable_indices]
    ny = size(Y, 2)
    T_obs = size(Y, 1)

    # Mean: E[y]
    Ey = vec(mean(Y; dims=1))

    # Product moments: E[y·y'] upper triangle
    Eyy = Y' * Y / T_obs

    # Autocovariances: E[y_t · y_{t-k}'] diagonal
    # ...

    # Collect into moment vector
    moments = T[]
    append!(moments, Ey)                          # means
    for i in 1:ny, j in i:ny
        push!(moments, Eyy[i,j])                  # product moments
    end
    for lag in lags
        autocov = Y[1+lag:end,:]' * Y[1:end-lag,:] / (T_obs - lag)
        for i in 1:ny
            push!(moments, autocov[i,i])           # diagonal autocov
        end
    end
    return moments
end
```

### `_augmented_moments_2nd(sol; lags)` — Closed-form 2nd-order moments

Builds the augmented state system and solves via Lyapunov:

```julia
function _augmented_moments_2nd(sol::PerturbationSolution{T};
                                 lags::Vector{Int}=[1]) where {T}
    # 1. Build A, c, Var_inov
    # 2. E[z] = (I - A) \ c
    # 3. Var_z = _dlyap_doubling(A, Var_inov)
    # 4. Map to observables: E[y], Var_y, Cov_y
    # 5. Collect into moment vector (means + product moments + autocov)
    ...
end
```

### Updated `analytical_moments(sol::PerturbationSolution; lags, format)`

Add a `format` kwarg to switch between the old format (covariances only, backward compatible) and the new format (means + product moments + autocov for GMM):

```julia
function analytical_moments(sol::PerturbationSolution{T};
                              lags::Int=1,
                              format::Symbol=:covariance) where {T}
    if format == :gmm
        lag_vec = collect(1:lags)
        if sol.order >= 2
            return _augmented_moments_2nd(sol; lags=lag_vec)
        else
            return _first_order_gmm_moments(sol; lags=lag_vec)
        end
    else
        # Existing behavior: covariance + diagonal autocov
        ...
    end
end
```

### Updated `_estimate_dsge_analytical_gmm`

Extended to accept `solve_method` and `solve_order`:

```julia
function _estimate_dsge_analytical_gmm(spec::DSGESpec{T}, data::Matrix{T},
                                         param_names::Vector{Symbol};
                                         lags=1, weighting=:two_step,
                                         bounds=nothing,
                                         solve_method=:gensys,
                                         solve_order=1,
                                         observable_indices=nothing,
                                         auto_lags=[1]) where {T}
    # Data moments (new format with means + product moments + autocov)
    m_data = _compute_data_moments(data; lags=auto_lags,
                                    observable_indices=observable_indices)

    function analytical_moment_fn(theta, _data)
        # Build spec with new params
        new_spec = _update_spec_params(spec, param_names, theta)
        try
            new_spec = compute_steady_state(new_spec)
            sol = solve(new_spec; method=solve_method, order=solve_order)
            m_model = analytical_moments(sol; lags=maximum(auto_lags), format=:gmm)
            reshape(m_data .- m_model, 1, :)
        catch
            fill(T(1e6), 1, length(m_data))
        end
    end

    gmm_result = estimate_gmm(analytical_moment_fn, theta0, data;
                                weighting=weighting, bounds=bounds)
    ...
end
```

## File Changes

| File | Change |
|---|---|
| `dsge/types.jl` | Relax `DSGEEstimation.final_sol` to `Union{DSGESolution{T}, PerturbationSolution{T}}` |
| `dsge/pruning.jl` | Add `_augmented_moments_2nd()`, `_innovation_variance_2nd()`, `_first_order_gmm_moments()`; update `analytical_moments` dispatch to support `:gmm` format |
| `dsge/estimation.jl` | Add `solve_method`/`solve_order`/`observable_indices`/`auto_lags` kwargs to `estimate_dsge` and `_estimate_dsge_analytical_gmm`; add `_compute_data_moments()`; add `_update_spec_params()` helper |
| `MacroEconometricModels.jl` | No changes (no new exports needed) |

## New File

None. All changes fit within existing files.

## Detailed Algorithm: Innovation Variance for 2nd Order

Following `UnconditionalMoments_2nd_Lyap.m`:

```
Var_inov = zeros(nz, nz)    # nz = 2*nx + nx²

sigeta = η   # nx × n_eps

# Block (1,1): first-order shock variance
Var_inov[1:nx, 1:nx] = sigeta * sigeta'

# Block (1,3): E[η·ε ⊗ (x_f⊗η·ε)] cross-term
# For symmetric shocks (vectorMom3 = 0), this is zero
# For non-symmetric: involves E[ε_i * ε_j * ε_k] terms

# Block (2,2): zero (no direct shock to xs)
# Block (2,3): zero

# Block (3,3): E[(hx·xf⊗η·ε + η·ε⊗hx·xf + η·ε⊗η·ε)(...)']
# This requires:
#   - E[xf·xf'] = Var_xf (from 1st-order Lyapunov)
#   - E[ε⁴] = vectorMom4 = 3 for Gaussian
#   - Cross terms between xf and ε
```

The (3,3) block is the most complex. It involves:

```
Var_inov_33 = kron(hx, sigeta) * E[(xf⊗ε)(xf⊗ε)'] * kron(hx, sigeta)'
            + kron(hx, sigeta) * E[(xf⊗ε)(ε⊗xf)'] * kron(sigeta, hx)'
            + kron(sigeta, hx) * E[(ε⊗xf)(xf⊗ε)'] * kron(hx, sigeta)'
            + kron(sigeta, hx) * E[(ε⊗xf)(ε⊗xf)'] * kron(sigeta, hx)'
            + kron(sigeta, sigeta) * E[(ε⊗ε)(ε⊗ε)'] * kron(sigeta, sigeta)'
            - kron(hx, hx) * vec(Var_xf) * vec(Var_xf)' * kron(hx, hx)'
            + (remaining cross-terms)
```

Where `E[(ε⊗ε)(ε⊗ε)']` is a `n_eps² × n_eps²` matrix with entries:
- `E[ε_a·ε_b·ε_c·ε_d]` = 3 if a=b=c=d (vectorMom4), 1 if two pairs match, 0 otherwise

And `E[(xf⊗ε)(xf⊗ε)']` = `E[xf·xf'] ⊗ I_neps` (since xf and ε are independent at same time).

## Warm-Starting the Lyapunov Solver

Following the MATLAB code, the optimizer calls the moment function many times. Caching the previous `Var_z` as a warm-start for `_dlyap_doubling` significantly reduces iteration count (convergence in 2-3 iterations instead of 20-30). This is implemented via a closure that captures a mutable cache.

## API

```julia
# Existing API — unchanged
sol2 = solve(spec; method=:perturbation, order=2)

# Closed-form moments (new format for GMM)
mom = analytical_moments(sol2; lags=5, format=:gmm)

# Estimation with perturbation + higher-order moments
est = estimate_dsge(spec, data, [:BETTA, :ALFA, :RHOA];
                     method=:analytical_gmm,
                     solve_method=:perturbation, solve_order=2,
                     auto_lags=[1, 3, 5],
                     weighting=:two_step,
                     bounds=bounds)

# Access results
est.theta          # estimated parameters
est.vcov           # asymptotic covariance
est.J_stat         # overidentification J-test
est.final_sol      # PerturbationSolution{T}
```

## Backward Compatibility

Fully backward compatible:
- `analytical_moments(sol; lags=1)` returns the same format as before (default `format=:covariance`)
- `estimate_dsge(...; method=:analytical_gmm)` without `solve_method` still uses `:gensys` (first-order)
- `DSGEEstimation.final_sol` field type widened but all existing code works via duck typing

## Testing

1. **Closed-form vs simulation moments**: 2nd-order closed-form moments match long-simulation moments (T=500k) within tolerance
2. **Order-1 equivalence**: GMM-format moments for order 1 match existing `analytical_moments` (means are zero, product moments equal covariances)
3. **Data moment computation**: `_compute_data_moments` matches `autocovariance_moments` plus means
4. **Risk correction**: 2nd-order means are non-zero and shift in the correct direction (consumption/output mean below deterministic SS)
5. **Estimation round-trip**: Generate data from known parameters, estimate via 2nd-order analytical GMM, recover parameters within confidence intervals
6. **Overidentification**: J-test p-value > 0.05 when model is correctly specified
7. **Backward compatibility**: Existing `estimate_dsge(...; method=:analytical_gmm)` tests still pass
