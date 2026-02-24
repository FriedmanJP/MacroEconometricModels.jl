# DSGE: Higher-Order Perturbation with Pruning — Design

**Issue**: #48 (scoped to perturbation + pruning only)
**Date**: 2026-02-25

## Goal

Add 2nd and 3rd order perturbation solvers with Kim et al. (2008) pruning to the DSGE module, accessible via `solve(spec; method=:perturbation, order=2)`.

## References

- Schmitt-Grohé, Stephanie & Martín Uribe. 2004. "Solving Dynamic General Equilibrium Models Using a Second-Order Approximation to the Policy Function." *Journal of Economic Dynamics and Control* 28 (4): 755–775.
- Kim, Jinill, Sunghyun Kim, Ernst Schaumburg & Christopher A. Sims. 2008. "Calculating and Using Second-Order Accurate Solutions of Discrete Time Dynamic Equilibrium Models." *Journal of Economic Dynamics and Control* 32 (11): 3397–3414.
- Andreasen, Martin M., Jesús Fernández-Villaverde & Juan F. Rubio-Ramírez. 2018. "The Pruned State-Space System for Non-Linear DSGE Models: Theory and Empirical Applications." *Review of Economic Studies* 85 (1): 1–49.

## Algorithm

### Perturbation Orders

The existing `linearize()` produces Sims form `Γ₀·y_t = Γ₁·y_{t-1} + C + Ψ·ε_t + Π·η_t`. Higher-order perturbation extends this with derivative tensors of the model's residual equations evaluated at steady state.

**Second order** adds quadratic terms:
```
y_t = ȳ + g_x·x_t + (1/2)·g_xx·(x_t ⊗ x_t) + (1/2)·g_σσ·σ²
```

**Third order** adds cubic terms:
```
+ (1/6)·g_xxx·(x_t ⊗ x_t ⊗ x_t) + (3/6)·g_σσx·σ²·x_t + (1/6)·g_σσσ·σ³
```

### Pruning Decomposition (Kim et al. 2008)

Without pruning, Kronecker products `x⊗x` grow explosively. Pruning decomposes the state into components that are stable by construction:

- **x_f** (first-order): `x_f = h_x·x_f + η·ε_t` — inherits stability of h_x
- **x_s** (second-order correction): `x_s = h_x·x_s + (1/2)·H̃_xx·(x_f⊗x_f) + (1/2)·h_σσ·σ²` — driven only by squared first-order terms
- **x_rd** (third-order remainder): `x_rd = h_x·x_rd + H̃_xx·(x_f⊗x_s) + (1/6)·H̃_xxx·(x_f⊗x_f⊗x_f) + (3/6)·h_σσx·σ²·x_f + (1/6)·h_σσσ·σ³`

Each component inherits the stability of h_x, so the full state `x = x_f + x_s + x_rd` remains bounded.

### Linear Innovations Form

Following Andreasen et al. (2018), the augmented state is `v_t = [x_{t-1}; u_t]` where shocks enter linearly. Derivative tensors are expressed in terms of v_t. The shock loading matrix `η = [0; Σ]` is block-structured with zeros for the state part and the shock covariance square root for the innovation part.

### Derivative Computation

Numerical higher-order derivatives via central differences on the existing `spec.residual_fns`. For the Hessian of residual `i` w.r.t. variables `j,k`:
```
∂²f_i/∂y_j∂y_k ≈ (f(y+h_j+h_k) - f(y+h_j-h_k) - f(y-h_j+h_k) + f(y-h_j-h_k)) / (4·h_j·h_k)
```

Third derivatives use analogous 8-point stencils. No new dependencies — extends the existing numerical Jacobian approach.

## Type System

### `PerturbationSolution{T}`

New type, separate from `DSGESolution{T}`:

```julia
struct PerturbationSolution{T<:AbstractFloat}
    order::Int                    # 1, 2, or 3

    # First-order (always present)
    gx::Matrix{T}                # ny × nx  — control response to state
    hx::Matrix{T}                # nx × nx  — state transition

    # Second-order (order ≥ 2)
    gxx::Union{Nothing, Array{T,3}}   # ny × nx × nx
    hxx::Union{Nothing, Array{T,3}}   # nx × nx × nx
    gσσ::Union{Nothing, Vector{T}}    # ny — uncertainty correction
    hσσ::Union{Nothing, Vector{T}}    # nx

    # Third-order (order == 3)
    gxxx::Union{Nothing, Array{T,4}}  # ny × nx × nx × nx
    hxxx::Union{Nothing, Array{T,4}}  # nx × nx × nx × nx
    gσσx::Union{Nothing, Matrix{T}}   # ny × nx
    hσσx::Union{Nothing, Matrix{T}}   # nx × nx
    gσσσ::Union{Nothing, Vector{T}}   # ny
    hσσσ::Union{Nothing, Vector{T}}   # nx

    # Shock loading & metadata
    eta::Matrix{T}               # (nx+nu) × nu — [0; Σ] block
    steady_state::Vector{T}      # full steady state
    state_indices::Vector{Int}   # which vars are predetermined
    control_indices::Vector{Int} # which vars are jump/control

    eu::Vector{Int}              # [existence, uniqueness] from first-order
    method::Symbol               # :perturbation
    spec::DSGESpec{T}
end
```

### Flattened Kronecker Matrices

For efficient simulation, pre-compute flattened "tilde" matrices:
- `H̃_xx = reshape(hxx, nx, nx²)` so `H̃_xx * kron(x,x)` is a matrix-vector multiply
- `H̃_xxx = reshape(hxxx, nx, nx³)` similarly
- `G̃_xx`, `G̃_xxx` for controls

These are computed once at solution time.

### State/Control Separation

Reuses `_count_predetermined()` from the Klein solver, extended to return indices. Variables with non-zero columns in Γ₁ are states; the rest are controls. For augmented models (#54), auxiliary lag variables are also states.

## Simulation

### Pruned Simulation

New `simulate(sol::PerturbationSolution{T}, T_periods)` dispatches on `sol.order`:

**Order 2:** Track `(x_f, x_s)` separately. Output:
```
y_t = ȳ + g_x·(x_f+x_s) + (1/2)·G̃_xx·(x_f⊗x_f) + (1/2)·g_σσ
```

**Order 3:** Track `(x_f, x_s, x_rd)`. Output adds:
```
+ g_x·x_rd + G̃_xx·(x_f⊗x_s) + (1/6)·G̃_xxx·(x_f⊗x_f⊗x_f) + (3/6)·g_σσx·x_f
```

Antithetic shock option: second half of draws are negatives of first half for variance reduction.

## Closed-Form Moments (Andreasen et al. 2018)

### Second Order

Augmented state `z = [x_f; x_s; vec(x_f⊗x_f)]` of dimension `2·nx + nx²`.

- **Mean**: `z̄ = (I - A)⁻¹·c` where A is the augmented transition and c contains σ² constants
- **Variance**: Lyapunov equation `Var_z = A·Var_z·A' + Var_inov` via doubling algorithm
- **Autocovariances**: `Cov(z_t, z_{t-h}) = A^h·Var_z`

### Third Order

Augmented state `z = [x_f; x_s; vec(x_f⊗x_f); x_rd; vec(x_f⊗x_s); vec(x_f⊗x_f⊗x_f)]` of dimension `3·nx + 2·nx² + nx³`.

Same Lyapunov framework, larger system. Innovation covariance requires higher-order shock moments (E[ε⁴]=3 for Gaussian).

### Lyapunov Solver

`_dlyap_doubling(A, B)` — iterative doubling algorithm. Numerically stable, O(log T) iterations × O(n³) per iteration.

## IRFs

### Analytical IRF (Andreasen et al. 2012)

Closed-form recursion evaluated at the ergodic mean:
```
IRF_xf(:,l) = h_x^{l-1}·η·ε_shock
```
Then feed through 2nd/3rd order recursions for `IRF_xs`, `IRF_xrd`. Returns all orders for comparison.

### GIRF (Generalized)

Simulation-based: `E[y | shock] - E[y | no shock]` averaged over many draws. More standard in VAR literature, accounts for state-dependence.

Both return `ImpulseResponse{T}` compatible with existing `plot_result()`.

## FEVD

Simulation-based: shut off one shock at a time, compute variance reduction. Returns standard `FEVD{T}`.

## New Files

| File | Contents |
|---|---|
| `dsge/derivatives.jl` | `_compute_hessians()`, `_compute_third_derivatives()` via numerical central differences |
| `dsge/perturbation.jl` | `perturbation_solver()` — compute g/h coefficients for orders 1-3; `_state_control_partition()`; `_solve_second_order()`, `_solve_third_order()` |
| `dsge/pruning.jl` | `simulate` dispatch for `PerturbationSolution`; `_dlyap_doubling()`; `analytical_moments` dispatch; analytical IRF + GIRF |

## Include Order

After `klein.jl`, before `perfect_foresight.jl`:
```
include("dsge/klein.jl")
include("dsge/derivatives.jl")    # NEW
include("dsge/perturbation.jl")   # NEW
include("dsge/pruning.jl")        # NEW — after simulation.jl in phase 2
include("dsge/perfect_foresight.jl")
```

Note: `pruning.jl` contains `simulate`/`irf`/`fevd` dispatches, so it goes in DSGE phase 2 (after `simulation.jl`).

## Pipeline Impact

| Component | Change |
|---|---|
| `types.jl` | Add `PerturbationSolution{T}` type |
| `display.jl` | Add `show()` for `PerturbationSolution` |
| `derivatives.jl` | New file |
| `perturbation.jl` | New file |
| `pruning.jl` | New file (simulation/moments/IRF) |
| `gensys.jl` (solve dispatcher) | Add `:perturbation` case |
| `MacroEconometricModels.jl` | Add includes + exports |
| `linearize` | No change |
| `gensys` / `blanchard_kahn` / `klein` | No change |
| `simulate` / `irf` / `fevd` | New dispatch methods in pruning.jl |
| `analytical_moments` | New dispatch method in pruning.jl |
| `estimate_dsge` | No change (uses solve dispatcher) |
| `perfect_foresight` / `occbin` | No change |
| `plot_result` | No change (works via ImpulseResponse/FEVD types) |

## API

```julia
# Second-order perturbation with pruning
sol2 = solve(spec; method=:perturbation, order=2)

# Third-order
sol3 = solve(spec; method=:perturbation, order=3)

# Pruned simulation
Y = simulate(sol2, 1000)

# Closed-form moments
mom = analytical_moments(sol2; lags=4)

# Analytical IRF (Andreasen)
ir = irf(sol2, 40)

# GIRF (simulation-based)
ir_g = irf(sol2, 40; irf_type=:girf, n_draws=1000)

# FEVD
fv = fevd(sol2, 40)

# Works with existing estimation
est = estimate_dsge(spec, data, θ0; solve_method=:perturbation, solve_order=2)
```

## Backward Compatibility

Fully backward compatible. `:gensys`, `:blanchard_kahn`, `:klein` unchanged. New `:perturbation` method is opt-in. No new dependencies.

## Testing

1. **Derivative accuracy**: Compare numerical Hessians against known analytical derivatives for simple models (AR(1), RBC)
2. **Solution equivalence**: Order-1 perturbation matches gensys G1/impact
3. **Pruning stability**: Long simulations (T=100000) don't explode for order 2 and 3
4. **Moment accuracy**: Closed-form moments vs. sample moments from long simulation (should converge)
5. **IRF equivalence**: Order-1 analytical IRF matches existing `irf(DSGESolution)`
6. **Known model**: 2nd-order RBC risk correction — mean consumption/output shift downward relative to deterministic SS
7. **Downstream**: `plot_result()`, `estimate_dsge()` work with `PerturbationSolution`
