# [Nonlinear Solution Methods](@id dsge_nonlinear)

First-order linear solutions impose **certainty equivalence** --- agents behave as if shocks have zero variance. This rules out risk premia, precautionary savings, welfare costs of uncertainty, and asymmetric dynamics. Nonlinear methods capture all of these by retaining higher-order terms in the Taylor expansion of the policy function or by solving the functional equation globally. MacroEconometricModels.jl provides three families: **higher-order perturbation** (local, Schmitt-Grohe & Uribe 2004; Andreasen, Fernandez-Villaverde & Rubio-Ramirez 2018), **Chebyshev projection** (global polynomial, Judd 1992, 1998), and **policy function iteration** (global iterative, Coleman 1990). For model specification and linearization, see [DSGE Models](@ref dsge_page). For first-order solvers, see [Linear Solvers](@ref dsge_linear).

## Quick Start

**Recipe 1: Second-order perturbation with pruned simulation**

```julia
using MacroEconometricModels, Random
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

psol = perturbation_solver(spec; order=2)
Y_sim = simulate(psol, 1000)  # pruned simulation (Kim et al. 2008)
```

**Recipe 2: Third-order perturbation with GIRFs**

```julia
psol3 = perturbation_solver(spec; order=3)
Y_sim3 = simulate(psol3, 1000)
girf3 = irf(psol3, 40; irf_type=:girf, n_draws=500)
plot_result(girf3)
```

**Recipe 3: Chebyshev projection and Euler errors**

```julia
proj = collocation_solver(spec; degree=5, grid=:tensor)
y = evaluate_policy(proj, proj.steady_state[proj.state_indices])
err = max_euler_error(proj)
```

**Recipe 4: PFI with damped updates**

```julia
pfi = pfi_solver(spec; degree=5, damping=0.5)
report(pfi)
```

---

## Second-Order Perturbation

The first-order linear solution imposes **certainty equivalence**: agents behave identically regardless of shock variance. This produces four specific deficiencies: (1) zero risk premia on asset returns, (2) no precautionary savings motive, (3) zero welfare cost of business cycles, and (4) perfectly symmetric impulse responses to positive and negative shocks. The second-order perturbation (Schmitt-Grohe & Uribe 2004) resolves all four by retaining quadratic terms in the Taylor expansion of the policy function.

The second-order decision rule takes the form:

```math
z_t = \bar{z} + f_v \, v_t + \tfrac{1}{2} f_{vv} (v_t \otimes v_t) + \tfrac{1}{2} f_{\sigma\sigma} \sigma^2
```

where:
- ``z_t`` is the ``n \times 1`` vector of all endogenous variables (deviations from steady state)
- ``v_t = [x_{t-1}; \varepsilon_t]`` is the ``n_v \times 1`` innovations vector (lagged states + current shocks)
- ``f_v`` is the ``n \times n_v`` first-order coefficient matrix
- ``f_{vv}`` is the ``n \times n_v^2`` second-order coefficient tensor (flattened Kronecker)
- ``f_{\sigma\sigma}`` is the ``n \times 1`` variance correction (shifts the **stochastic steady state**)
- ``\sigma`` is the perturbation scaling parameter

```julia
psol = perturbation_solver(spec; order=2)
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `order` | `Int` | `2` | Perturbation order (1, 2, or 3) |
| `method` | `Symbol` | `:gensys` | First-order solver (`:gensys` or `:blanchard_kahn`) |

### Algorithm

The computation follows Schmitt-Grohe & Uribe (2004) in six steps:

1. Solve the first-order system via Gensys or Blanchard-Kahn to obtain ``G_1`` and the impact matrix
2. Partition variables into states (``x``) and controls (``y``) via the ``\Gamma_1`` structure
3. Build the innovations vector ``v = [x_{t-1}; \varepsilon_t]`` and the mapping matrices that relate each argument slot (current, lag, lead, shock) to ``v``-space
4. Compute all 10 unique Hessian tensors via central finite differences across the four argument slots
5. Assemble the ``n \times n_v^2`` right-hand side from contracting each Hessian with its slot-to-``v`` mapping matrices, then solve the Kronecker system: ``(I_{n_v^2} \otimes f_c + \text{kron}(M,M)' \otimes f_f) \, \text{vec}(f_{vv}) = -\text{vec}(\text{RHS})``
6. Solve the ``\sigma^2`` correction: ``(f_c + f_f) \, f_{\sigma\sigma} = -f_f \, f_{vv} \, \text{vec}(\eta \eta')``

!!! note "Technical Note"
    The **stochastic steady state** ``\bar{z} + \tfrac{1}{2} f_{\sigma\sigma} \sigma^2`` differs from the deterministic steady state ``\bar{z}``. This shift captures precautionary behavior: risk-averse agents accumulate more capital when facing uncertainty. The magnitude depends on the curvature of the utility function and the variance of shocks.

---

## Third-Order Perturbation

The decision rule at third order adds cubic and variance-interaction terms (Andreasen, Fernandez-Villaverde & Rubio-Ramirez 2018):

```math
z_t = \bar{z} + f_v \, v_t + \tfrac{1}{2} f_{vv} (v_t \otimes v_t) + \tfrac{1}{2} f_{\sigma\sigma} \sigma^2 + \tfrac{1}{6} f_{vvv} (v_t \otimes v_t \otimes v_t) + \tfrac{1}{2} f_{\sigma\sigma v} \, \sigma^2 \, v_t + \tfrac{1}{6} f_{\sigma\sigma\sigma} \, \sigma^3
```

where:
- ``f_{vvv}`` is the ``n \times n_v^3`` third-order coefficient tensor (flattened triple Kronecker)
- ``f_{\sigma\sigma v}`` is the ``n \times n_v`` interaction between uncertainty and the state
- ``f_{\sigma\sigma\sigma}`` is the ``n \times 1`` cubic variance correction (zero for Gaussian shocks)

```julia
psol3 = perturbation_solver(spec; order=3)
```

### Algorithm

The third-order computation extends the second-order procedure with six additional steps:

1. Solve first-order and second-order as prerequisites (Steps 1--6 above)
2. Compute all 20 unique third-derivative tensors via central finite differences across the four argument slots (current, lag, lead, shock), storing only canonical orderings
3. Accumulate the ``n \times n_v^3`` right-hand side from two sources: (A) pure third derivatives contracted with mapping matrices across all slot triples, and (B) mixed Hessian-times-second-order interaction terms (3 cyclic permutations per Hessian block)
4. Solve the third-order Kronecker system: ``(I_{n_v^3} \otimes f_c + \text{kron}(M,M,M)' \otimes f_f) \, \text{vec}(f_{vvv}) = -\text{vec}(\text{RHS}_3)``
5. Compute ``f_{\sigma\sigma v}`` correction via contraction of ``f_{vvv}`` with shock covariance ``\eta \eta'`` plus Hessian-times-``f_{\sigma\sigma}`` interaction across all slot pairs
6. Set ``f_{\sigma\sigma\sigma} = 0`` (zero for Gaussian shocks)

!!! note "Technical Note"
    Third-order perturbation captures **skewness** in the ergodic distribution. The ``f_{\sigma\sigma\sigma}`` correction is zero for Gaussian shocks but would be non-zero for non-Gaussian innovations. The ``f_{\sigma\sigma v}`` term captures how the variance correction depends on the state --- agents' precautionary behavior varies with economic conditions.

### Return Value (`PerturbationSolution{T}`)

| Field | Type | Description |
|-------|------|-------------|
| `order` | `Int` | Perturbation order (1, 2, or 3) |
| `gx` | `Matrix{T}` | ``n_y \times n_v`` first-order controls |
| `hx` | `Matrix{T}` | ``n_x \times n_v`` first-order states |
| `gxx` | `Union{Nothing, Matrix{T}}` | ``n_y \times n_v^2`` second-order controls (order ``\geq 2``) |
| `hxx` | `Union{Nothing, Matrix{T}}` | ``n_x \times n_v^2`` second-order states (order ``\geq 2``) |
| `g\sigma\sigma` | `Union{Nothing, Vector{T}}` | ``n_y`` control variance correction (order ``\geq 2``) |
| `h\sigma\sigma` | `Union{Nothing, Vector{T}}` | ``n_x`` state variance correction (order ``\geq 2``) |
| `gxxx` | `Union{Nothing, Matrix{T}}` | ``n_y \times n_v^3`` third-order controls (order = 3) |
| `hxxx` | `Union{Nothing, Matrix{T}}` | ``n_x \times n_v^3`` third-order states (order = 3) |
| `g\sigma\sigma x` | `Union{Nothing, Matrix{T}}` | ``n_y \times n_v`` variance-state interaction (order = 3) |
| `h\sigma\sigma x` | `Union{Nothing, Matrix{T}}` | ``n_x \times n_v`` variance-state interaction (order = 3) |
| `g\sigma\sigma\sigma` | `Union{Nothing, Vector{T}}` | ``n_y`` cubic variance correction (order = 3) |
| `h\sigma\sigma\sigma` | `Union{Nothing, Vector{T}}` | ``n_x`` cubic variance correction (order = 3) |
| `eta` | `Matrix{T}` | ``n_v \times n_\varepsilon`` shock loading ``[0; I]`` block |
| `steady_state` | `Vector{T}` | Deterministic steady state |
| `state_indices` | `Vector{Int}` | Indices of state variables |
| `control_indices` | `Vector{Int}` | Indices of control variables |
| `eu` | `Vector{Int}` | Existence/uniqueness from first-order |
| `spec` | `DSGESpec{T}` | Back-reference to specification |
| `linear` | `LinearDSGE{T}` | Linearized system |

---

## Pruning

Naive simulation of higher-order decision rules produces **explosive sample paths** because the Kronecker products ``(v_t \otimes v_t)`` compound deviations multiplicatively --- a moderate deviation at time ``t`` is squared, generating a larger deviation at ``t+1``, which is squared again. **Pruning** (Kim, Kim, Schaumburg & Sims 2008) prevents this by tracking state components separately and using only first-order states in the Kronecker products.

### Second-Order Pruning (Kim et al. 2008)

The pruned simulation decomposes the state into two components:

1. **First-order state**: ``x_t^{(1)} = h_{x,\text{state}} \cdot x_{t-1}^{(1)} + \eta_x \cdot \varepsilon_t``
2. **Second-order correction**: ``x_t^{(2)} = h_{x,\text{state}} \cdot x_{t-1}^{(2)} + \tfrac{1}{2} h_{xx} (v_t^{(1)} \otimes v_t^{(1)}) + \tfrac{1}{2} h_{\sigma\sigma}``

Total state: ``x_t = x_t^{(1)} + x_t^{(2)}``

where ``v_t^{(1)} = [x_{t-1}^{(1)}; \varepsilon_t]`` contains only first-order states. The key insight is that the Kronecker product ``v_t^{(1)} \otimes v_t^{(1)}`` uses only the stable first-order component, preventing compounding.

### Third-Order Pruning (Andreasen et al. 2018)

Third-order pruning adds a third component:

3. **Third-order correction**: ``x_t^{(3)} = h_{x,\text{state}} \cdot x_{t-1}^{(3)} + h_{xx} (v_t^{(1)} \otimes v_t^{(2)}) + \tfrac{1}{6} h_{xxx} (v_t^{(1)} \otimes v_t^{(1)} \otimes v_t^{(1)}) + \tfrac{1}{2} h_{\sigma\sigma x} \, v_t^{(1)} + \tfrac{1}{6} h_{\sigma\sigma\sigma}``

where ``v_t^{(2)} = [x_{t-1}^{(2)}; 0]`` contains the second-order state correction with zero shocks.

Total: ``x_t = x_t^{(1)} + x_t^{(2)} + x_t^{(3)}``

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# (assume spec defined as in Quick Start)
psol3 = perturbation_solver(spec; order=3)
Y_sim = simulate(psol3, 1000)  # 3-component pruned simulation
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `shock_draws` | `Union{Nothing, Matrix}` | `nothing` | Pre-drawn shocks (T x n_shocks); draws from N(0,1) if `nothing` |
| `rng` | `AbstractRNG` | `default_rng()` | Random number generator |
| `antithetic` | `Bool` | `false` | Antithetic variates for variance reduction |

!!! warning "Explosive paths without pruning"
    Never simulate a second- or third-order perturbation solution without pruning. The naive simulation ``z_t = \bar{z} + f_v v_t + \frac{1}{2} f_{vv} (v_t \otimes v_t) + \ldots`` using the total state in the Kronecker product diverges within 50--100 periods for most calibrations. All `simulate(::PerturbationSolution, ...)` calls use pruning automatically.

---

## Generalized Impulse Responses

Analytical IRFs from the first-order solution miss second- and third-order effects. **Generalized impulse response functions** (GIRFs, Koop, Pesaran & Potter 1996) compute the expected difference between a shocked and baseline path via Monte Carlo simulation:

```math
\text{GIRF}(h, \delta_j) = E\big[y_{t+h} \mid \varepsilon_{j,t} = \delta_j, \Omega_{t-1}\big] - E\big[y_{t+h} \mid \Omega_{t-1}\big]
```

where:
- ``\delta_j`` is the shock impulse to shock ``j``
- ``\Omega_{t-1}`` is the information set at ``t-1``
- The expectation is computed by averaging over ``n_{\text{draws}}`` simulated paths using pruned simulation

The GIRF captures nonlinear dynamics that the analytical first-order IRF misses: asymmetric responses to positive vs. negative shocks, state-dependent propagation, and variance-correction effects. For a first-order solution, the GIRF converges to the analytical IRF as ``n_{\text{draws}} \to \infty``.

```julia
# Analytical IRFs (first-order only, fast)
irf_analytical = irf(psol3, 40)

# GIRFs (captures nonlinear dynamics, Monte Carlo)
girf = irf(psol3, 40; irf_type=:girf, n_draws=500)
plot_result(girf)
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `irf_type` | `Symbol` | `:analytical` | `:analytical` for first-order, `:girf` for simulation-based |
| `n_draws` | `Int` | `500` | Number of Monte Carlo draws for GIRF |
| `shock_size` | `Real` | `1.0` | Impulse size in standard deviations |

The FEVD is also available via `fevd(psol, H)`, which uses the first-order decomposition from the underlying linear solution.

---

## Chebyshev Projection

**Chebyshev collocation** (Judd 1992, 1998) approximates the policy function globally using Chebyshev polynomials on a tensor or Smolyak grid. Unlike perturbation, the approximation is accurate far from the steady state --- essential for models with large shocks, regime switches, or occasionally binding constraints. The solver finds coefficients such that the equilibrium conditions hold exactly at each collocation node.

The policy function approximation takes the form:

```math
y_i(x) \approx \sum_{j=1}^{n_b} c_{i,j} \, T_j(x)
```

where:
- ``c_{i,j}`` are the Chebyshev coefficients (``n_{\text{vars}} \times n_b`` matrix)
- ``T_j(x)`` are the multivariate Chebyshev basis functions (products of univariate ``T_k(x_d)``)
- ``n_b`` is the number of basis functions

### Algorithm

The collocation solver proceeds in five steps:

1. **Linearize** to get the state/control partition and compute state bounds as ``\bar{x}_i \pm \text{scale} \cdot \sigma_i`` from the first-order Lyapunov solution
2. **Build grid**: tensor product of Chebyshev extrema (Gauss-Lobatto) nodes on ``[-1, 1]^{n_x}``; for high-dimensional models, use Smolyak sparse grid instead
3. **Construct basis matrix** by evaluating all multivariate Chebyshev polynomials at each collocation node
4. **Initialize coefficients** from the first-order perturbation solution via least-squares projection onto the Chebyshev basis
5. **Newton iteration**: solve ``R(c) = 0`` where ``R`` is the vector of equilibrium residuals evaluated at all nodes, with Gauss-Hermite or monomial quadrature for expectations of next-period variables. Uses Gauss-Newton with backtracking line search.

```julia
proj = collocation_solver(spec; degree=5, grid=:tensor)
report(proj)
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `degree` | `Int` | `5` | Chebyshev polynomial degree |
| `grid` | `Symbol` | `:auto` | `:tensor`, `:smolyak`, or `:auto` (tensor if ``n_x \leq 4``) |
| `smolyak_mu` | `Int` | `3` | Smolyak approximation level |
| `quadrature` | `Symbol` | `:auto` | `:gauss_hermite` or `:monomial` (auto based on ``n_\varepsilon``) |
| `n_quad` | `Int` | `5` | Quadrature nodes per dimension |
| `scale` | `Real` | `3.0` | State bounds as multiples of unconditional std |
| `tol` | `Real` | ``10^{-8}`` | Newton convergence tolerance |
| `max_iter` | `Int` | `100` | Maximum Newton iterations |
| `initial_coeffs` | `Union{Nothing, Matrix}` | `nothing` | Warm-start coefficients from previous solve |

### Smolyak Sparse Grids

For models with ``n_x > 4`` states, tensor grids become computationally infeasible. A tensor grid with degree 5 and 5 states requires ``6^5 = 7{,}776`` nodes; with 8 states, ``6^8 \approx 1.7 \times 10^6``. **Smolyak sparse grids** (Smolyak 1963; Judd, Maliar, Maliar & Valero 2014) select a subset of grid points that preserve polynomial exactness at a fraction of the cost. The `grid=:smolyak` option uses nested Clenshaw-Curtis points with the Smolyak selection rule ``|\alpha|_1 \leq \mu + n_x`` for multi-index ``\alpha``.

!!! note "Technical Note"
    The `grid=:auto` option selects tensor grids for ``n_x \leq 4`` and Smolyak grids for ``n_x > 4``. Similarly, `quadrature=:auto` selects Gauss-Hermite for ``n_\varepsilon \leq 2`` and monomial rules for ``n_\varepsilon > 2``. Monomial rules scale linearly with ``n_\varepsilon`` while Gauss-Hermite scales exponentially.

### Evaluating the Policy Function

The `evaluate_policy` function maps state vectors to the full vector of endogenous variables using the stored Chebyshev coefficients:

```julia
y = evaluate_policy(proj, x_state)          # single point (vector)
Y = evaluate_policy(proj, X_states)         # multiple points (matrix, n_points x nx)
```

### Euler Equation Errors

The **Euler equation error** measures the accuracy of the global approximation by evaluating residuals at random test points drawn uniformly within the state bounds:

```julia
err = max_euler_error(proj; n_test=1000)
```

The error is reported in levels. Convert to ``\log_{10}`` for the standard accuracy metric:

| ``\log_{10}`` error | Quality |
|----------------------|---------|
| ``< -6`` | Excellent |
| ``-6`` to ``-4`` | Good |
| ``-4`` to ``-2`` | Acceptable |
| ``> -2`` | Poor --- increase degree or tighten tolerance |

---

## Policy Function Iteration

**Policy function iteration** (PFI, Coleman 1990) solves for the equilibrium policy function by iterating on the Euler equation. At each step, the current policy guess determines expected future values via quadrature, and the equilibrium conditions at each grid point produce an updated policy via Newton's method. PFI tends to be more robust than collocation for models with kinks or near-kinks in the policy function, while collocation converges faster for smooth problems.

The algorithm iterates three sub-steps at each grid point ``j``:

1. **Expectation**: compute ``E[y_{t+1}]`` using the current policy coefficients and quadrature
2. **Euler solve**: given ``y_{\text{lag}} = x_j`` (the grid point as lagged state) and ``E[y_{t+1}]``, solve ``F(y_t, y_{\text{lag}}, E[y_{t+1}], 0, \theta) = 0`` for ``y_t`` via Newton iteration
3. **Refit**: project the updated policy values onto the Chebyshev basis via least squares

```julia
pfi = pfi_solver(spec; degree=5, damping=0.5)
report(pfi)
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `degree` | `Int` | `5` | Chebyshev polynomial degree |
| `grid` | `Symbol` | `:auto` | `:tensor`, `:smolyak`, or `:auto` |
| `damping` | `Real` | `1.0` | Damping factor (0.5 for slow convergence, 1.0 for no damping) |
| `tol` | `Real` | ``10^{-8}`` | Sup-norm convergence tolerance |
| `max_iter` | `Int` | `500` | Maximum iterations |
| `initial_coeffs` | `Union{Nothing, Matrix}` | `nothing` | Warm-start from previous solve |

!!! note "Technical Note"
    PFI and Chebyshev collocation return the same `ProjectionSolution{T}` type. Both support `evaluate_policy`, `simulate`, `irf`, and `max_euler_error`. The `method` field distinguishes them: `:projection` for collocation, `:pfi` for policy function iteration.

### Return Value (`ProjectionSolution{T}`)

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | `Matrix{T}` | ``n_{\text{vars}} \times n_b`` Chebyshev coefficients |
| `state_bounds` | `Matrix{T}` | ``n_x \times 2`` state domain bounds |
| `grid_type` | `Symbol` | `:tensor` or `:smolyak` |
| `degree` | `Int` | Polynomial degree (tensor) or Smolyak level ``\mu`` |
| `collocation_nodes` | `Matrix{T}` | ``n_{\text{nodes}} \times n_x`` grid points in ``[-1, 1]`` |
| `residual_norm` | `T` | Final ``\|R\|`` residual (collocation) or sup-norm (PFI) |
| `n_basis` | `Int` | Number of basis functions |
| `multi_indices` | `Matrix{Int}` | ``n_b \times n_x`` multi-index matrix |
| `quadrature` | `Symbol` | `:gauss_hermite` or `:monomial` |
| `converged` | `Bool` | Convergence flag |
| `iterations` | `Int` | Iterations until convergence |
| `state_indices` | `Vector{Int}` | State variable indices |
| `control_indices` | `Vector{Int}` | Control variable indices |
| `method` | `Symbol` | `:projection` or `:pfi` |

---

## Analytical Moments

For first-order solutions, `analytical_moments` computes unconditional moments in closed form via the discrete **Lyapunov equation**:

```math
\Sigma = G_1 \, \Sigma \, G_1' + \text{impact} \cdot \text{impact}'
```

where:
- ``\Sigma`` is the ``n \times n`` unconditional covariance matrix
- ``G_1`` is the state transition matrix from the first-order solution
- ``\text{impact}`` is the ``n \times n_\varepsilon`` shock impact matrix

The `solve_lyapunov` function solves this equation via Kronecker vectorization: ``\text{vec}(\Sigma) = (I_{n^2} - G_1 \otimes G_1)^{-1} \, \text{vec}(\text{impact} \cdot \text{impact}')``. Autocovariances at lag ``h`` follow from ``\Gamma_h = G_1^h \, \Sigma``.

```julia
sol = solve(spec)

# Unconditional covariance matrix
Sigma = solve_lyapunov(sol.G1, sol.impact)

# Moment vector matching autocovariance_moments format
m = analytical_moments(sol; lags=2)
```

The moment vector contains two blocks: (1) the upper triangle of the variance-covariance matrix (``k(k+1)/2`` elements) and (2) diagonal autocovariances at each lag (``k`` elements per lag). This format matches `autocovariance_moments(data, lags)` for direct comparison in [Estimation](@ref dsge_estimation).

---

## Complete Example

This example solves the RBC model at all three perturbation orders, compares analytical and generalized IRFs, and validates a global projection solution with Euler equation errors.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Specify RBC model
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

# First-order: certainty equivalence
sol1 = perturbation_solver(spec; order=1)

# Second-order: risk correction
sol2 = perturbation_solver(spec; order=2)

# Third-order: skewness and state-dependent risk
sol3 = perturbation_solver(spec; order=3)

# Compare simulated ergodic means (stochastic SS shift)
Y1 = simulate(sol1, 10000)
Y2 = simulate(sol2, 10000)
Y3 = simulate(sol3, 10000)

# Generalized IRFs at third order
girf = irf(sol3, 40; irf_type=:girf, n_draws=500)
plot_result(girf)

# Global solution via Chebyshev collocation
proj = collocation_solver(spec; degree=5, grid=:tensor)
report(proj)

# Euler equation accuracy
err = max_euler_error(proj; n_test=1000)

# PFI for comparison
pfi = pfi_solver(spec; degree=5, damping=0.5)
err_pfi = max_euler_error(pfi; n_test=1000)

# Analytical moments for estimation targets
m = analytical_moments(sol1; lags=2)
```

The stochastic steady state shifts from second-order perturbation reflect precautionary savings: the ergodic mean of capital exceeds the deterministic steady state because risk-averse agents over-accumulate capital as a buffer against productivity shocks. Third-order perturbation adds state-dependent risk premia --- the precautionary effect is stronger in recessions than expansions.

---

## Common Pitfalls

1. **Explosive simulation without pruning**: Never extract the second-order coefficients and simulate manually. Always use `simulate(psol, T)`, which applies Kim et al. (2008) pruning automatically.

2. **Tensor grids in high dimensions**: A tensor grid with degree 5 and ``n_x = 6`` states requires ``6^6 = 46{,}656`` nodes. Use `grid=:smolyak` for ``n_x > 4``.

3. **Poor Euler errors**: If `max_euler_error` returns values above ``10^{-2}``, increase the polynomial `degree`, widen the state bounds via the `scale` parameter, or switch from collocation to PFI.

4. **Non-convergence of collocation**: The Gauss-Newton solver uses backtracking line search but can stall at local minima. Warm-start with `initial_coeffs` from a lower-degree solution or from PFI.

5. **Lyapunov equation instability**: `solve_lyapunov` throws an error if the first-order solution has eigenvalues on or outside the unit circle. Check determinacy with `is_determined(sol)` before computing moments.

---

## References

- Andreasen, M. M., Fernandez-Villaverde, J., & Rubio-Ramirez, J. F. (2018). The Pruned State-Space System for Non-Linear DSGE Models: Theory and Empirical Applications. *Review of Economic Studies*, 85(1), 1--49. [DOI](https://doi.org/10.1093/restud/rdx037)

- Coleman, W. J. (1990). Solving the Stochastic Growth Model by Policy-Function Iteration. *Journal of Business & Economic Statistics*, 8(1), 27--29. [DOI](https://doi.org/10.1080/07350015.1990.10509769)

- Judd, K. L. (1992). Projection Methods for Solving Aggregate Growth Models. *Journal of Economic Theory*, 58(2), 410--452. [DOI](https://doi.org/10.1016/0022-0531(92)90061-L)

- Judd, K. L. (1998). *Numerical Methods in Economics*. MIT Press. ISBN: 978-0-262-10071-7.

- Judd, K. L., Maliar, L., Maliar, S., & Valero, R. (2014). Smolyak Method for Solving Dynamic Economic Models: Lagrange Interpolation, Anisotropic Grids and Adaptive Domain. *Journal of Economic Dynamics and Control*, 44, 92--123. [DOI](https://doi.org/10.1016/j.jedc.2014.03.003)

- Kim, J., Kim, S., Schaumburg, E., & Sims, C. A. (2008). Calculating and Using Second-Order Accurate Solutions of Discrete Time Dynamic Equilibrium Models. *Journal of Economic Dynamics and Control*, 32(11), 3397--3414. [DOI](https://doi.org/10.1016/j.jedc.2008.02.003)

- Koop, G., Pesaran, M. H., & Potter, S. M. (1996). Impulse Response Analysis in Nonlinear Multivariate Models. *Journal of Econometrics*, 74(1), 119--147. [DOI](https://doi.org/10.1016/0304-4076(95)01753-4)

- Schmitt-Grohe, S., & Uribe, M. (2004). Solving Dynamic General Equilibrium Models Using a Second-Order Approximation to the Policy Function. *Journal of Economic Dynamics and Control*, 28(4), 755--775. [DOI](https://doi.org/10.1016/S0165-1889(03)00043-5)
