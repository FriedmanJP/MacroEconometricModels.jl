# Projection Methods (Chebyshev Collocation) — Design Document

**Issue:** #48 — DSGE: add nonlinear solvers (item 2: Projection Methods)

**Goal:** Add Chebyshev collocation solver for DSGE models, approximating policy functions globally on a Chebyshev grid and solving residual equations via Newton iteration.

**References:** Judd (1998), Malin-Krueger-Kubler (2011), Judd-Maliar-Maliar-Valero (2014), Den Haan & Marcet (1994)

---

## Architecture

**Approach:** Monolithic solver following the `perturbation_solver` pattern — a single `collocation_solver()` function with private helpers, all in one file. Quadrature split into a separate reusable file.

**Key idea:** Instead of Taylor-expanding the policy function around steady state (perturbation), approximate it globally using Chebyshev polynomials on a bounded domain. Solve for polynomial coefficients that make the equilibrium conditions hold at collocation nodes.

---

## Solution Type

```julia
struct ProjectionSolution{T<:AbstractFloat}
    # Policy function: y = Σ_k coeff[k] * T_k(x_scaled)
    coefficients::Matrix{T}         # n_vars × n_basis

    # Grid specification
    state_bounds::Matrix{T}         # nx × 2 ([lower upper] per state)
    grid_type::Symbol               # :tensor or :smolyak
    degree::Int                     # polynomial degree (tensor) or Smolyak level μ

    # Collocation grid (stored for diagnostics)
    collocation_nodes::Matrix{T}    # n_nodes × nx
    residual_norm::T                # final ||R||

    # Basis info
    n_basis::Int
    multi_indices::Matrix{Int}      # n_basis × nx

    # Quadrature
    quadrature::Symbol              # :gauss_hermite or :monomial

    # Back-references
    spec::DSGESpec{T}
    linear::LinearDSGE{T}
    steady_state::Vector{T}         # cached for fast evaluate_policy
    state_indices::Vector{Int}
    control_indices::Vector{Int}

    # Convergence
    converged::Bool
    iterations::Int
    method::Symbol                  # :projection
end
```

Accessors follow existing pattern: `nvars(sol)`, `nshocks(sol)`, `nstates(sol)`, `ncontrols(sol)`, `is_determined(sol)`.

---

## Algorithm

### Step 1: Setup

1. Compute steady state if missing
2. Linearize → `LinearDSGE` → `_state_control_indices` for state/control partition
3. Solve first-order perturbation → get unconditional variance for state bounds
4. Set state bounds: `SS_i ± scale × σ_i` (default scale=3.0)

### Step 2: Grid Construction

- **Tensor-product** (default for nx ≤ 4):
  - Chebyshev nodes: `x_j = cos(πj/(n-1))` for j=0,...,n-1 in each dimension
  - Multi-indices: all combinations `(i_1,...,i_nx)` with `0 ≤ i_k ≤ degree`
  - Total nodes: `(degree+1)^nx`

- **Smolyak sparse grid** (for nx > 4 or by request):
  - Exactness level μ (default 3)
  - Uses nested Chebyshev extrema (Clenshaw-Curtis points)
  - Smolyak construction: `A(d,μ) = Σ_{|q|≤μ} (-1)^{μ-|q|} C(d-1, μ-|q|) × (Q_{q_1} ⊗ ... ⊗ Q_{q_d})`
  - Multi-indices selected by: `|α| ≤ μ + nx` (isotropic Smolyak)
  - Total nodes: O(n · (log n)^{d-1}) vs n^d for tensor

### Step 3: Initial Guess

Convert the first-order perturbation solution to Chebyshev coefficients:
- For each variable i, evaluate the linear policy `y_i = gx_i · x` at collocation nodes
- Fit Chebyshev coefficients via least squares: `c = B \ y_nodes` where B is the basis matrix

### Step 4: Residual System

At each collocation node `x_j` (mapped to physical coordinates):

1. **Evaluate current policy**: `y_j = B_j · c` (basis matrix row × coefficient vector)
2. **For each quadrature point `ε_i`** (Gauss-Hermite or monomial):
   a. Compute next-period states using state transition equations from `spec.residual_fns`
   b. Map next-period states to [-1,1] using state_bounds
   c. Evaluate policy at next-period states: `y'_{j,i} = B(x') · c`
3. **Form expectations**: `E[y'] ≈ Σ_i w_i · y'_{j,i}`
4. **Evaluate equilibrium residuals**: `R_j = f(y_j, x_j_lag, E[y'], 0, θ)` via `spec.residual_fns`

Total residual dimension: `n_equations × n_nodes`, unknowns: `n_vars × n_basis`.

### Step 5: Newton Iteration

Solve `R(c) = 0`:
1. Compute Jacobian `∂R/∂c` via finite differences on the coefficient vector
2. Newton step: `Δc = -J \ R`
3. **Line search**: if `||R(c + Δc)|| > ||R(c)||`, try `c + α·Δc` with `α = 0.5, 0.25, ...`
4. Convergence: `||R|| < tol` (default 1e-8)
5. Max iterations: 100 (default)

### Step 6: Package Result

Return `ProjectionSolution{T}` with converged coefficients, grid info, and metadata.

---

## Quadrature Methods

### Gauss-Hermite (default for ≤ 2 shocks)

- Nodes and weights for `∫ f(x) exp(-x²) dx ≈ Σ w_i f(x_i)`
- Scale by `√2·σ` for `N(0,σ²)` integration
- Tensor product across shock dimensions
- Default: 5 nodes per dimension

### Monomial Integration (for > 2 shocks)

Judd-Maliar-Maliar (2011) monomial formulas:
- `2n+1` evaluation points for n shocks (vs `5^n` for tensor Gauss-Hermite)
- Exact for all monomials up to degree 3
- Points: origin + `±√n` along each axis

---

## Public API

```julia
# Via solve() dispatcher
sol = solve(spec; method=:projection,
            degree=5,              # Chebyshev polynomial degree
            grid=:auto,            # :tensor, :smolyak, or :auto
            smolyak_mu=3,          # Smolyak exactness level
            quadrature=:auto,      # :gauss_hermite, :monomial, or :auto
            n_quad=5,              # quadrature nodes per shock
            scale=3.0,             # state bounds = SS ± scale * σ
            tol=1e-8,              # Newton convergence tolerance
            max_iter=100,          # max Newton iterations
            verbose=false)         # print iteration info

# Evaluate policy at arbitrary state
y = evaluate_policy(sol, x_state)         # x_state: nx-vector → n_vars-vector
Y = evaluate_policy(sol, X_states)        # X_states: n_points × nx → n_points × n_vars

# Simulation (nonlinear, using policy function evaluation)
Y_sim = simulate(sol::ProjectionSolution, T; rng=...)

# IRF (Monte Carlo: compare paths with/without initial shock)
irfs = irf(sol::ProjectionSolution, H; n_sim=1000, rng=...)

# Diagnostics
max_euler_error(sol; n_test=1000, rng=...)   # max |R| on random test points
```

**Auto-selection rules:**
- `grid=:auto` → `:tensor` if `nx ≤ 4`, else `:smolyak`
- `quadrature=:auto` → `:gauss_hermite` if `n_eps ≤ 2`, else `:monomial`

---

## File Structure

**New files:**
- `src/dsge/quadrature.jl` (~80-100 lines) — Gauss-Hermite and monomial quadrature
- `src/dsge/projection.jl` (~500-600 lines) — Chebyshev basis, grids, collocation solver

**Modified files:**
- `src/dsge/types.jl` — add `ProjectionSolution{T}` struct + accessors
- `src/dsge/gensys.jl` — add `:projection` branch to `solve()` dispatcher
- `src/dsge/simulation.jl` — add `simulate(::ProjectionSolution, ...)` and `irf(::ProjectionSolution, ...)`
- `src/MacroEconometricModels.jl` — add includes for quadrature.jl, projection.jl; export `evaluate_policy`, `max_euler_error`
- `test/dsge/test_dsge.jl` — comprehensive tests

**Include order:** quadrature.jl after perturbation.jl, projection.jl after quadrature.jl (before simulation.jl so that simulate/irf dispatch is available).

**Private helpers** (in projection.jl, prefixed with `_`):
- `_chebyshev_nodes(n)` — nodes on [-1,1]
- `_chebyshev_eval(x, degree)` — evaluate T_0...T_n at scalar x
- `_chebyshev_basis_multi(X, multi_indices)` — tensor-product basis at points
- `_smolyak_grid(nx, mu)` — Smolyak sparse grid construction
- `_tensor_grid(nx, degree)` — tensor-product grid
- `_scale_to_unit(x, bounds)` / `_scale_from_unit(z, bounds)` — affine [-1,1] ↔ [a,b]
- `_compute_state_bounds(spec, sol_perturbation, scale)` — ergodic bounds
- `_collocation_residual!(R, coeffs, ...)` — evaluate R(c) in-place
- `_newton_step(coeffs, ...)` — one Newton iteration with line search

**Private helpers** (in quadrature.jl):
- `_gauss_hermite_nodes_weights(n)` — nodes and weights via eigenvalue method
- `_monomial_nodes_weights(n_eps)` — Judd-Maliar-Maliar (2011) rule

---

## Testing Strategy

~55 tests in 7 groups:

### 1. Quadrature unit tests (~8 tests)
- Gauss-Hermite nodes/weights for n=3,5,7
- Exact integration of polynomials up to degree 2n-1
- Monomial rule integrates x², x⁴ correctly
- Monomial rule has correct number of points (2n+1)

### 2. Chebyshev basis unit tests (~8 tests)
- Chebyshev nodes match cos formula
- T_0(x)=1, T_1(x)=x, T_2(x)=2x²-1
- Basis orthogonality at Chebyshev nodes
- Scale/unscale round-trip

### 3. Grid construction (~6 tests)
- Tensor grid: correct size for nx=1,2,3
- Smolyak grid: correct number of points for μ=2,3
- Multi-indices satisfy Smolyak selection rule

### 4. Linear AR(1) model (~10 tests)
- Projection recovers linear policy exactly
- `evaluate_policy(sol, ss)` returns steady state
- `evaluate_policy` matches perturbation at nearby points
- Max Euler error < 1e-6
- Convergence in ≤ 5 iterations

### 5. Nonlinear growth model (~10 tests)
- Neoclassical growth model with CRRA utility
- Convergence (residual_norm < tol)
- Max Euler error < 1e-4
- Simulation produces stationary distribution
- Policy function is monotone in capital

### 6. Accuracy comparison: projection vs perturbation (~5 tests)
- For a model with significant nonlinearity (high σ or low γ)
- Projection Euler errors should be smaller than perturbation (order 1)
- Projection policy should differ from linear at state bounds
- Both agree near steady state

### 7. API integration (~8 tests)
- `simulate(sol, T)` returns T × n matrix
- `irf(sol, H)` returns ImpulseResponse
- `show(io, sol)` works
- Grid auto-selection
- Quadrature auto-selection
- Backward compatibility (existing methods unaffected)

---

## Dependencies

No new external dependencies. Uses existing:
- `LinearAlgebra` — for eigenvalue computation in Gauss-Hermite, matrix operations
- `Optim` (already imported) — not needed for projection (we implement Newton directly)

Chebyshev polynomials, Smolyak grids, and quadrature rules are implemented from scratch using standard algorithms.

---

## Error Handling

- **Non-convergence**: Set `converged=false`, store best coefficients found, warn user
- **State out of bounds**: `evaluate_policy` extrapolates linearly beyond `state_bounds` with a warning (first call only, via `@warn maxlog=1`)
- **Singular Jacobian**: Use `robust_inv` (existing utility) in Newton step
- **Invalid grid for dimensions**: Error if `nx > 4` and `grid=:tensor` (suggest `:smolyak`)
