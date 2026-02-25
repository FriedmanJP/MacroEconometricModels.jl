# DSGE Models

**MacroEconometricModels.jl** provides a complete toolkit for specifying, solving, simulating, and estimating Dynamic Stochastic General Equilibrium (DSGE) models. The package covers the full workflow from model definition to structural estimation:

- **Specification**: The `@dsge` macro provides a domain-specific language for writing equilibrium conditions with time-indexed variables
- **Steady State**: Analytical or numerical steady-state computation with two-phase optimization
- **Linearization**: Automatic first-order approximation via numerical Jacobians in the Sims (2002) canonical form
- **Solution Methods**: Four first-order solvers (Gensys, Blanchard-Kahn, Klein, perturbation) plus second-order perturbation with pruning, Chebyshev projection, and policy function iteration
- **Simulation and IRFs**: Stochastic and pruned simulation, analytical and generalized impulse responses, FEVD, unconditional moments via Lyapunov equation
- **Estimation**: Four GMM-based methods for estimating deep structural parameters from data
- **Nonlinear Extensions**: Perfect foresight paths, OccBin occasionally binding constraints, constrained optimization via JuMP/Ipopt (NLP) and PATH (MCP)

All results integrate seamlessly with `plot_result()` for interactive D3.js visualization and `report()` for publication-quality output.

## Quick Start

**Recipe 1: Solve and plot IRFs**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Specify a simple RBC model
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
result = irf(sol, 40)
plot_result(result)
```

**Recipe 2: Second-order perturbation with pruning**

```julia
psol = perturbation_solver(spec; order=2)
Y_sim = simulate(psol, 1000)  # pruned simulation (Kim et al. 2008)
girf = irf(psol, 40; irf_type=:girf, n_draws=500)
```

**Recipe 3: Estimate with GMM**

```julia
est = estimate_dsge(spec, Y_data, [:β, :α, :ρ];
                    method=:irf_matching, var_lags=4, irf_horizon=20)
report(est)
```

**Recipe 4: OccBin with ZLB**

```julia
constraint = parse_constraint(:(R[t] >= 0), spec)
occ_sol = occbin_solve(spec, constraint; shock_path=shocks)
occ_irf = occbin_irf(spec, constraint, 1, 40)
plot_result(occ_irf)
```

**Recipe 5: Chebyshev projection**

```julia
proj = collocation_solver(spec; degree=5, grid=:tensor)
y = evaluate_policy(proj, x_state)
err = max_euler_error(proj)
```

**Recipe 6: Constrained steady state (JuMP/Ipopt)**

```julia
import JuMP, Ipopt
bound = variable_bound(:i, lower=0.0)
spec_c = compute_steady_state(spec, [bound])
```

---

## Model Specification

The `@dsge` macro provides a domain-specific language for specifying DSGE models. It parses the model into a `DSGESpec{T}` object containing equations, parameters, and variable declarations.

### Syntax

```julia
spec = @dsge begin
    parameters: β = 0.99, α = 0.36, δ = 0.025, ρ = 0.9, σ = 0.01
    endogenous: Y, C, K, A
    exogenous: ε_A

    # Equations (one per endogenous variable)
    Y[t] = A[t] * K[t-1]^α
    C[t] + K[t] = Y[t] + (1 - δ) * K[t-1]
    1 = β * (C[t] / C[t+1]) * (α * A[t+1] * K[t]^(α - 1) + 1 - δ)
    A[t] = ρ * A[t-1] + σ * ε_A[t]
end
```

### Blocks

| Block | Syntax | Description |
|-------|--------|-------------|
| `parameters:` | `name = value, ...` | Calibrated parameters with default values |
| `endogenous:` | `var1, var2, ...` | Endogenous variable names |
| `exogenous:` | `shock1, shock2, ...` | Exogenous shock names |
| `steady_state` | `= begin ... [y_ss] end` | Optional analytical steady-state function (must return vector) |
| `varnames:` | `["Label 1", "Label 2", ...]` | Optional display labels for variables |

### Time Subscripts

| Notation | Meaning |
|----------|---------|
| `var[t]` | Current period value |
| `var[t-1]` | One-period lag (predetermined variable) |
| `var[t+1]` | One-period lead (forward-looking / jump variable) |

Variables with `[t+1]` subscripts generate expectation errors in the Sims canonical form. The number of forward-looking equations determines the dimension of the ``\Pi`` matrix and, via the Blanchard-Kahn condition, the number of unstable eigenvalues required for determinacy.

!!! note "Technical Note"
    Equations are written as `LHS = RHS` where both sides can contain endogenous variables at different time subscripts. The `@dsge` macro rearranges each equation into residual form ``f(y_t, y_{t-1}, y_{t+1}, \varepsilon_t, \theta) = 0`` via `LHS - RHS`. The number of equations must equal the number of endogenous variables. Timing convention: ``K_{t}`` chosen at time ``t`` appears as `K[t]`; ``K_{t-1}`` (beginning-of-period capital) as `K[t-1]`.

### Return Value

| Field | Type | Description |
|-------|------|-------------|
| `endog` | `Vector{Symbol}` | Endogenous variable names |
| `exog` | `Vector{Symbol}` | Exogenous shock names |
| `params` | `Vector{Symbol}` | Parameter names |
| `param_values` | `Dict{Symbol,T}` | Calibrated values |
| `equations` | `Vector{Expr}` | Raw equation expressions |
| `n_endog` | `Int` | Number of endogenous variables |
| `n_exog` | `Int` | Number of exogenous shocks |
| `n_expect` | `Int` | Number of expectation errors |
| `forward_indices` | `Vector{Int}` | Indices of forward-looking equations |
| `steady_state` | `Vector{T}` | Steady-state values |
| `varnames` | `Vector{String}` | Display names |

---

## Steady State

The steady state ``\bar{y}`` satisfies the equilibrium system in the absence of shocks:

```math
f(\bar{y}, \bar{y}, \bar{y}, 0, \theta) = 0
```

For the RBC model above, the analytical steady state is:

```math
\bar{A} = 1, \quad \bar{K} = \left(\frac{\alpha\beta}{1 - \beta(1-\delta)}\right)^{\frac{1}{1-\alpha}}, \quad \bar{Y} = \bar{K}^\alpha, \quad \bar{C} = \bar{Y} - \delta\bar{K}
```

### Numerical Computation

```julia
spec = compute_steady_state(spec)
println("Steady state: ", spec.steady_state)
```

`compute_steady_state` uses a two-phase optimizer: Nelder-Mead for robustness (derivative-free global exploration), then L-BFGS for refinement. It minimizes the sum of squared residuals ``\sum_i f_i(\bar{y})^2``.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `initial_guess` | `Vector` | `nothing` | Starting point (default: ones) |
| `method` | `Symbol` | `:auto` | `:auto` (NelderMead then LBFGS) or `:analytical` |

### Analytical Steady State

For models where the steady state has a closed-form solution, specify it in a `steady_state = begin ... end` block. The block must return a vector matching the endogenous variable ordering:

```julia
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
```

When the `steady_state` block is provided, `compute_steady_state` (or `solve`) uses it directly and validates the result against the equations.

### Constrained Steady State (JuMP/Ipopt)

For models with variable bounds --- such as a zero lower bound on the nominal interest rate or non-negativity of consumption --- the steady state can be computed as a constrained optimization problem using JuMP and Ipopt:

```julia
import JuMP, Ipopt

# ZLB: interest rate cannot go below zero
bound = variable_bound(:R, lower=0.0)

# Solve constrained steady state
spec = compute_steady_state(spec, [bound])
```

The solver minimizes the sum of squared equilibrium residuals subject to the variable bounds. It uses Ipopt (Interior Point Optimizer) via JuMP's automatic differentiation.

### Constrained Steady State (PATH MCP)

For problems naturally expressed as complementarity conditions --- where a constraint binds if and only if a corresponding equation holds with slack --- the PATH solver formulates the steady state as a Mixed Complementarity Problem (MCP):

```julia
import JuMP, PATHSolver

bound = variable_bound(:R, lower=0.0)
spec = compute_steady_state(spec, [bound]; solver=:path)
```

The MCP formulation is: for each variable ``i`` with lower bound ``l_i`` and upper bound ``u_i``, find ``y_i`` such that:

```math
l_i \leq y_i \leq u_i, \quad f_i(y) \begin{cases} \geq 0 & \text{if } y_i = l_i \\ = 0 & \text{if } l_i < y_i < u_i \\ \leq 0 & \text{if } y_i = u_i \end{cases}
```

!!! note "When to use PATH vs Ipopt"
    Use **PATH** when constraints are box bounds (variable bounds only) and the model has a natural complementarity structure --- for example, a ZLB where ``R_t \geq 0`` and the Taylor rule holds with equality when ``R_t > 0``. Use **Ipopt** when you have general nonlinear inequality constraints or when the complementarity interpretation is not natural.

---

## Linearization

`linearize` computes a first-order Taylor expansion around the steady state using numerical Jacobians (central differences). It produces the Sims (2002) canonical form:

```math
\Gamma_0 \, y_t = \Gamma_1 \, y_{t-1} + C + \Psi \, \varepsilon_t + \Pi \, \eta_t
```

where ``y_t`` is the vector of endogenous variables (in deviations from steady state), ``\varepsilon_t`` is the vector of exogenous shocks, and ``\eta_t`` is the vector of expectation errors (``\eta_t = y_t - E_{t-1}[y_t]`` for forward-looking variables).

```julia
ld = linearize(spec)
```

!!! note "Technical Note"
    The matrices are computed via central differences with step size ``h = \max(10^{-7}, 10^{-7} |y_j|)``. No analytical derivatives are required. ``\Gamma_0`` contains coefficients on ``y_t``, ``\Gamma_1`` on ``y_{t-1}``, ``\Psi`` on shocks, and ``\Pi`` selects the forward-looking equations for expectation errors. The matrix pair ``(\Gamma_0, \Gamma_1)`` defines a generalized eigenvalue problem whose solution governs the dynamics.

### Return Value

| Field | Type | Description |
|-------|------|-------------|
| `Gamma0` | `Matrix{T}` | ``n \times n`` coefficient on ``y_t`` |
| `Gamma1` | `Matrix{T}` | ``n \times n`` coefficient on ``y_{t-1}`` |
| `C` | `Vector{T}` | ``n \times 1`` constants |
| `Psi` | `Matrix{T}` | ``n \times n_{shocks}`` shock loading |
| `Pi` | `Matrix{T}` | ``n \times n_{expect}`` expectation error selection |
| `spec` | `DSGESpec{T}` | Back-reference to specification |

---

## Solution Methods

### Overview

`solve` is the unified entry point for solving linearized DSGE models. It dispatches to the appropriate solver based on the `method` keyword:

```julia
sol = solve(spec; method=:gensys)
```

All first-order solvers produce the same state-space representation:

```math
y_t = G_1 \, y_{t-1} + C_{sol} + \text{impact} \cdot \varepsilon_t
```

where ``G_1`` is the ``n \times n`` state transition matrix, ``\text{impact}`` is the ``n \times n_{shocks}`` impact matrix, and ``C_{sol}`` captures constant terms.

| Method | Algorithm | Reference |
|--------|-----------|-----------|
| `:gensys` (default) | QZ decomposition | Sims (2002) |
| `:blanchard_kahn` | Eigenvalue decomposition | Blanchard & Kahn (1980) |
| `:klein` | Generalized Schur (QZ) with BK counting | Klein (2000) |
| `:perturbation` | Higher-order Taylor expansion | Schmitt-Grohe & Uribe (2004) |
| `:projection` | Chebyshev collocation | Judd (1998) |
| `:pfi` | Policy function iteration | Coleman (1990) |
| `:perfect_foresight` | Deterministic Newton solver | --- |

`solve` automatically calls `compute_steady_state` and `linearize` if they have not been called already.

### Choosing a Solver

| Feature needed | Recommended | Why |
|----------------|-------------|-----|
| Standard IRFs, FEVD | `:gensys` | Robust, handles singularity |
| Simple model, fast | `:blanchard_kahn` or `:klein` | Faster eigenvalue decomposition |
| Risk premia, welfare | `:perturbation` (order=2) | Captures precautionary effects |
| Large shocks, global | `:projection` or `:pfi` | Globally accurate policy functions |
| Occasionally binding | OccBin or `:pfi` | Handles kinks, complementarity |
| Known future shocks | `:perfect_foresight` | Exact deterministic path |

### Determinacy and Stability

A rational expectations model has a unique bounded solution (is **determinate**) when the number of unstable eigenvalues equals the number of forward-looking (jump) variables. This is the Blanchard-Kahn (1980) condition:

| Eigenvalue count | Result | Interpretation |
|-----------------|--------|----------------|
| ``n_{unstable} = n_{forward}`` | **Determinate** | Unique bounded solution |
| ``n_{unstable} < n_{forward}`` | **Indeterminate** | Multiple solutions, sunspot equilibria |
| ``n_{unstable} > n_{forward}`` | **No stable solution** | No bounded equilibrium |

```julia
sol = solve(spec; method=:gensys)
println("Determined: ", is_determined(sol))    # true if eu == [1, 1]
println("Stable: ", is_stable(sol))             # true if max|eigenvalue(G1)| < 1
```

!!! warning "Common cause of indeterminacy"
    The Taylor principle requires that the monetary policy rule respond more than one-for-one to inflation: ``\phi_\pi > 1`` in the Taylor rule ``i_t = \phi_\pi \pi_t + \phi_y \hat{y}_t``. With ``\phi_\pi < 1``, the 3-equation New Keynesian model is typically indeterminate. More precisely, determinacy requires ``\phi_\pi + \frac{1-\beta}{\kappa}\phi_y > 1``.

### Diagnosing Indeterminacy

When a model fails the Blanchard-Kahn condition or Gensys reports `eu[2] = 0`, inspect the eigenvalues:

```julia
sol = solve(spec; method=:gensys)
println("Eigenvalues: ", abs.(sol.eigenvalues))
println("EU: ", sol.eu)

# Count stable vs unstable roots
n_stable = count(abs.(sol.eigenvalues) .< 1.0)
n_unstable = count(abs.(sol.eigenvalues) .>= 1.0)
n_forward = spec.n_expect

println("Stable: $n_stable, Unstable: $n_unstable, Forward-looking: $n_forward")
```

---

## First-Order Solvers

### Gensys (Sims 2002)

The Gensys algorithm solves the linearized system via QZ (generalized Schur) decomposition of the matrix pencil ``(\Gamma_0, \Gamma_1)``. It separates stable and unstable generalized eigenvalues to construct the state-space solution. Gensys can handle singular ``\Gamma_0`` matrices and provides diagnostic output on the existence and uniqueness of the solution.

```julia
sol = solve(spec; method=:gensys)
println("Existence: ", sol.eu[1] == 1)    # 1 = exists
println("Uniqueness: ", sol.eu[2] == 1)   # 1 = unique
```

!!! note "Technical Note"
    The `div` keyword (default ``1.0 + 10^{-8}``) sets the dividing line between stable and unstable eigenvalues. Eigenvalues with modulus below `div` are classified as stable. The `eu` vector reports `[existence, uniqueness]` where 1 = satisfied, 0 = violated. Use `:gensys` as the default unless you have a specific reason to prefer another solver.

### Blanchard-Kahn (1980)

The Blanchard-Kahn method partitions the system into predetermined (state) and forward-looking (jump) variables. It computes the eigenvalue decomposition of ``\Gamma_0^{-1} \Gamma_1``, orders eigenvalues by modulus, and checks that the number of unstable roots equals the number of jump variables.

```julia
sol = solve(spec; method=:blanchard_kahn)
```

!!! warning
    The Blanchard-Kahn method requires ``\Gamma_0`` to be invertible. If ``\Gamma_0`` is singular (as in some large-scale models), use `:gensys` or `:klein` instead.

### Klein (2000)

The Klein solver uses the generalized Schur (QZ) decomposition like Gensys, but applies the Blanchard-Kahn counting condition based on predetermined variables rather than forward-looking equations. It reorders eigenvalues so that stable roots (``|\lambda| < \text{div}``) come first, then checks that ``n_{stable} = n_{predetermined}``.

```julia
sol = solve(spec; method=:klein)
```

The Klein method is particularly useful for models with a natural partition into predetermined (state) and non-predetermined (control) variables. The partition is detected automatically from the ``\Gamma_1`` matrix: a variable is predetermined if it has a non-zero column in ``\Gamma_1`` (i.e., it appears with a lag).

### Return Value (All First-Order Solvers)

| Field | Type | Description |
|-------|------|-------------|
| `G1` | `Matrix{T}` | ``n \times n`` state transition |
| `impact` | `Matrix{T}` | ``n \times n_{shocks}`` impact matrix |
| `C_sol` | `Vector{T}` | ``n \times 1`` constants |
| `eu` | `Vector{Int}` | `[existence, uniqueness]` (1=yes, 0=no) |
| `method` | `Symbol` | Solver used |
| `eigenvalues` | `Vector{ComplexF64}` | Generalized eigenvalues |
| `spec` | `DSGESpec{T}` | Model specification |
| `linear` | `LinearDSGE{T}` | Linearized form |

---

## Second-Order Perturbation

### Motivation

First-order perturbation imposes **certainty equivalence**: the decision rules are linear in the state variables and agents behave as if they live in a world without uncertainty. This means first-order solutions cannot capture:

- **Precautionary saving** (agents save more when facing risk)
- **Risk premia** (expected excess returns are zero at first order)
- **Mean shifts** (the stochastic steady state differs from the deterministic one)
- **Welfare comparisons** (second-order utility requires second-order policy)

The second-order perturbation method (Schmitt-Grohe & Uribe 2004) computes decision rules of the form:

```math
z_t = \bar{z} + f_v \, v_t + \tfrac{1}{2} f_{vv} (v_t \otimes v_t) + \tfrac{1}{2} f_{\sigma\sigma} \sigma^2
```

where ``v_t = [x_{t-1}; \varepsilon_t]`` stacks lagged states and current shocks, ``f_v`` is the first-order coefficient, ``f_{vv}`` is the second-order Kronecker coefficient, and ``f_{\sigma\sigma}`` is the **variance correction** that shifts the mean of the ergodic distribution.

### Computing the Second-Order Solution

```julia
psol = perturbation_solver(spec; order=2)
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `order` | `Int` | `2` | Perturbation order (1 or 2) |
| `method` | `Symbol` | `:gensys` | First-order solver (`:gensys` or `:blanchard_kahn`) |

### Algorithm

1. **First-order solution**: Solve the linearized system via Gensys or Blanchard-Kahn to obtain ``G_1`` and the impact matrix
2. **Variable partition**: Identify state (predetermined) and control (jump) variables from the ``\Gamma_1`` matrix
3. **Hessian computation**: Compute all second-derivative tensors ``\partial^2 f_i / \partial a_j \partial b_k`` via central finite differences across the four argument slots (current, lag, lead, shock)
4. **Kronecker system**: Assemble and solve the matrix equation ``f_c \cdot f_{vv} + f_f \cdot f_{vv} \cdot (M \otimes M) = -\text{RHS}`` where ``f_c, f_f`` are Jacobians with respect to current and lead variables, and ``M`` is the augmented transition matrix
5. **Variance correction**: Solve ``(f_c + f_f) \cdot f_{\sigma\sigma} = -f_f \cdot f_{vv} \cdot \text{vec}(\eta \eta')``

!!! note "Stochastic Steady State"
    The variance correction ``f_{\sigma\sigma}`` captures the difference between the deterministic steady state ``\bar{z}`` and the stochastic steady state ``\bar{z}^{(2)} = \bar{z} + \frac{1}{2} f_{\sigma\sigma} \sigma^2``. This mean shift reflects agents' precautionary behavior in the face of uncertainty.

### Return Value

| Field | Type | Description |
|-------|------|-------------|
| `order` | `Int` | Perturbation order |
| `gx` | `Matrix{T}` | ``n_y \times n_v`` control decision rule |
| `hx` | `Matrix{T}` | ``n_x \times n_v`` state transition rule |
| `gxx` | `Matrix{T}` | ``n_y \times n_v^2`` second-order control |
| `hxx` | `Matrix{T}` | ``n_x \times n_v^2`` second-order state |
| `gσσ` | `Vector{T}` | ``n_y`` control variance correction |
| `hσσ` | `Vector{T}` | ``n_x`` state variance correction |
| `state_indices` | `Vector{Int}` | Indices of state variables |
| `control_indices` | `Vector{Int}` | Indices of control variables |

---

## Pruning

### The Explosive Simulation Problem

Naive simulation of second-order decision rules can generate explosive sample paths. The quadratic terms ``f_{vv}(v_t \otimes v_t)`` feed back into states, amplifying deviations and causing paths to diverge. This is a well-known problem in the perturbation literature.

### Kim-Kim-Schaumburg-Sims (2008) Pruning

The pruning method prevents explosions by tracking first-order and second-order components separately:

1. **First-order state**: ``x_t^{(1)} = h_x \cdot x_{t-1}^{(1)} + \eta_x \cdot \varepsilon_t`` (standard linear dynamics)
2. **Second-order correction**: ``x_t^{(2)} = h_x \cdot x_{t-1}^{(2)} + \frac{1}{2} h_{xx} (v_t^{(1)} \otimes v_t^{(1)}) + \frac{1}{2} h_{\sigma\sigma}``
3. **Total state**: ``x_t = x_t^{(1)} + x_t^{(2)}``

The key insight is that the Kronecker product ``v_t \otimes v_t`` uses only the **first-order** innovations vector ``v_t^{(1)}``, preventing the feedback loop that causes explosions.

```julia
psol = perturbation_solver(spec; order=2)
Y_sim = simulate(psol, 1000)  # automatically uses pruning for order >= 2
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `shock_draws` | `Matrix` | `nothing` | Pre-drawn shocks (``T \times n_{shocks}``) |
| `rng` | `AbstractRNG` | `default_rng()` | Random number generator |
| `antithetic` | `Bool` | `false` | Antithetic variates for variance reduction |

---

## Chebyshev Projection

### Motivation

Perturbation methods are local approximations: they are accurate near the steady state but can become unreliable far from it. For models with significant nonlinearities, large shocks, or occasionally binding constraints, **projection methods** provide globally accurate policy functions by approximating the solution over the entire state space.

### Algorithm

The Chebyshev collocation method approximates the policy function ``g(x)`` as a linear combination of Chebyshev polynomials:

```math
g(x) \approx \sum_{i=0}^{n} a_i \, T_i\!\left(\frac{2(x - x_{\min})}{x_{\max} - x_{\min}} - 1\right)
```

where ``T_i`` is the ``i``-th Chebyshev polynomial and the state ``x`` is mapped to ``[-1, 1]`` via an affine transformation. The coefficients ``\{a_i\}`` are determined by imposing equilibrium at a set of collocation nodes:

1. **Grid construction**: Build a tensor-product or Smolyak sparse grid of Chebyshev nodes in the state space
2. **State bounds**: Computed as ``\bar{x} \pm \text{scale} \cdot \sigma_x`` using the first-order unconditional variance
3. **Expectation integration**: Approximate ``E_t[f(x')]`` via Gauss-Hermite quadrature (low dimension) or monomial rules (Judd, Maliar & Maliar 2011)
4. **Initial guess**: Warm-start from the first-order perturbation solution
5. **Newton iteration**: Solve the nonlinear system of equilibrium residuals at all collocation nodes via Gauss-Newton with line search

```julia
proj = collocation_solver(spec; degree=5, grid=:tensor)
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `degree` | `Int` | `5` | Chebyshev polynomial degree |
| `grid` | `Symbol` | `:auto` | `:tensor`, `:smolyak`, or `:auto` |
| `smolyak_mu` | `Int` | `3` | Smolyak exactness level |
| `quadrature` | `Symbol` | `:auto` | `:gauss_hermite`, `:monomial`, or `:auto` |
| `n_quad` | `Int` | `5` | Quadrature nodes per shock dimension |
| `scale` | `Real` | `3.0` | State bounds = SS ``\pm`` scale ``\times \sigma`` |
| `tol` | `Real` | ``10^{-8}`` | Newton convergence tolerance |
| `max_iter` | `Int` | `100` | Maximum Newton iterations |

### Smolyak Sparse Grids

For models with more than 3--4 state variables, tensor-product grids suffer from the curse of dimensionality: ``n^k`` grid points for ``k`` states with ``n`` nodes per dimension. Smolyak sparse grids (Judd, Maliar, Maliar & Valero 2014) provide the same polynomial exactness with dramatically fewer points by selecting grid nodes via a combinatorial rule:

```julia
proj = collocation_solver(spec; grid=:smolyak, smolyak_mu=3)
```

The `:auto` grid option selects tensor for ``n_x \leq 4`` and Smolyak otherwise.

### Evaluating the Policy Function

```julia
# Evaluate at a single state point (returns n_vars-vector of levels)
y = evaluate_policy(proj, x_state)

# Evaluate at multiple state points (n_points × nx → n_points × n_vars)
Y = evaluate_policy(proj, X_states)
```

### Euler Equation Errors

The standard diagnostic for projection accuracy is the maximum Euler equation error on a fine test grid (Judd 1992):

```julia
err = max_euler_error(proj; n_test=1000)
println("Max Euler error: ", err)
```

Report ``\log_{10}(|\text{error}|)``:

| ``\log_{10}`` error | Quality |
|---------------------|---------|
| ``< -5`` | Excellent |
| ``-4`` to ``-5`` | Good |
| ``-3`` to ``-4`` | Acceptable |
| ``> -3`` | Poor |

---

## Policy Function Iteration

Policy Function Iteration (PFI), also known as time iteration (Coleman 1990), is an alternative global solution method that iterates on the policy function directly rather than solving for collocation coefficients.

### Algorithm

At each iteration:

1. **Evaluate current policy** at all grid points
2. **Compute expected next-period values** via quadrature: ``E[y'] = \sum_q w_q \cdot g^{(k)}(x'(x, \varepsilon_q))``
3. **Solve Euler equation** at each grid point via Newton: given ``y_{\text{lag}}`` and ``E[y']``, find ``y_t`` such that ``F(y_t, y_{\text{lag}}, E[y'], 0, \theta) = 0``
4. **Refit Chebyshev coefficients** via least squares on the updated policy values
5. **Check convergence**: sup-norm of policy change

PFI has the advantage of guaranteed contraction (under standard conditions) and natural handling of inequality constraints at the Newton step.

```julia
proj = pfi_solver(spec; degree=5, max_iter=500, damping=0.8)
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `degree` | `Int` | `5` | Chebyshev polynomial degree |
| `grid` | `Symbol` | `:auto` | Grid type (same as collocation) |
| `max_iter` | `Int` | `500` | Maximum PFI iterations |
| `damping` | `Real` | `1.0` | Policy mixing factor (``<1`` for damped updates) |
| `tol` | `Real` | ``10^{-8}`` | Sup-norm convergence tolerance |

### Return Value (Projection and PFI)

Both methods return a `ProjectionSolution{T}`:

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | `Matrix{T}` | ``n_{vars} \times n_{basis}`` Chebyshev coefficients |
| `state_bounds` | `Matrix{T}` | ``n_x \times 2`` state domain bounds |
| `grid_type` | `Symbol` | `:tensor` or `:smolyak` |
| `degree` | `Int` | Polynomial degree or Smolyak level |
| `residual_norm` | `T` | Final residual norm |
| `converged` | `Bool` | Convergence flag |
| `iterations` | `Int` | Iterations used |
| `method` | `Symbol` | `:projection` or `:pfi` |

---

## Simulation and IRFs

### Stochastic Simulation

`simulate` generates sample paths from the solved model. For `DSGESolution` (first-order):

```math
y_t = G_1 \, y_{t-1} + \text{impact} \cdot \varepsilon_t + C_{sol}
```

For `PerturbationSolution` (second-order), pruned simulation is used automatically (see [Pruning](@ref)).

```julia
Y_sim = simulate(sol, 200; rng=Random.MersenneTwister(42))
# Returns 200 × n_endog matrix in levels (steady state + deviations)
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `shock_draws` | `Matrix` | `nothing` | Pre-drawn shocks (``T \times n_{shocks}``) |
| `rng` | `AbstractRNG` | `default_rng()` | Random number generator |

### Impulse Response Functions

`irf` computes analytical impulse responses. For first-order solutions:

```math
\Phi_h = G_1^{h-1} \cdot \text{impact}
```

where ``\Phi_h`` is the response at horizon ``h`` to a one-standard-deviation shock.

```julia
result = irf(sol, 40)
plot_result(result)
```

Returns an `ImpulseResponse{T}` object compatible with `plot_result()` for multi-panel IRF plots.

### Generalized IRFs (Second-Order)

For second-order perturbation solutions, analytical IRFs capture only the first-order dynamics. Generalized IRFs (Koop, Pesaran & Potter 1996) use Monte Carlo simulation to capture the nonlinear propagation:

```math
\text{GIRF}_h = E[y_{t+h} \mid \varepsilon_t = \delta] - E[y_{t+h} \mid \varepsilon_t = 0]
```

averaged over random draws of future shocks:

```julia
psol = perturbation_solver(spec; order=2)
girf = irf(psol, 40; irf_type=:girf, n_draws=500, shock_size=1.0)
```

### Forecast Error Variance Decomposition

`fevd` computes the share of forecast variance attributable to each shock at each horizon:

```math
\text{FEVD}_{ij}(h) = \frac{\sum_{s=0}^{h-1} (\Phi_s e_j)' (\Phi_s e_j)}{\sum_{s=0}^{h-1} \text{tr}(\Phi_s \Phi_s')}
```

```julia
result = fevd(sol, 40)
plot_result(result)
```

### Unconditional Moments

#### Lyapunov Equation

`solve_lyapunov` solves the discrete Lyapunov equation for the unconditional covariance:

```math
\Sigma = G_1 \, \Sigma \, G_1' + \text{impact} \cdot \text{impact}'
```

```julia
Sigma = solve_lyapunov(sol.G1, sol.impact)
```

!!! note "Technical Note"
    For first-order solutions, uses Kronecker vectorization: ``\text{vec}(\Sigma) = (I - G_1 \otimes G_1)^{-1} \text{vec}(\text{impact} \cdot \text{impact}')``. For second-order perturbation solutions, the pruning module provides a doubling algorithm that is more numerically stable for large systems (``O(n^3)`` per iteration vs ``O(n^6)`` for the Kronecker approach).

#### Analytical Moments (First-Order)

`analytical_moments` computes the theoretical moment vector from the Lyapunov solution:

```julia
m = analytical_moments(sol; lags=2)
```

Returns a concatenated vector: upper-triangle of the variance-covariance matrix (``k(k+1)/2`` elements) followed by diagonal autocovariances at each lag (``k`` elements per lag). These moments are used internally by the `:analytical_gmm` estimation method.

#### Analytical Moments (Second-Order)

For second-order perturbation solutions, closed-form moments are computed via the augmented-state Lyapunov approach (Andreasen, Fernandez-Villaverde & Rubio-Ramirez 2018). The augmented state ``z = [x^{(1)}; x^{(2)}; \text{vec}(x^{(1)} \otimes x^{(1)})]`` of dimension ``2n_x + n_x^2`` satisfies a linear transition, enabling Lyapunov-based moment computation that captures the mean shift from the variance correction:

```julia
psol = perturbation_solver(spec; order=2)
m = analytical_moments(psol; lags=2, format=:gmm)
```

The `:gmm` format returns means, product moments, and diagonal autocovariances --- suitable for GMM estimation with higher-order perturbation where unconditional means are non-zero.

---

## Estimation

`estimate_dsge` estimates DSGE deep parameters via Generalized Method of Moments:

```julia
est = estimate_dsge(spec, Y_data, [:β, :ρ, :σ]; method=:irf_matching)
report(est)
```

### Methods

| Method | Description | Reference |
|--------|-------------|-----------|
| `:irf_matching` | Match model IRFs to empirical VAR IRFs | Christiano, Eichenbaum & Evans (2005) |
| `:euler_gmm` | Euler equation moment conditions with lagged instruments | Hansen & Singleton (1982) |
| `:smm` | Simulated Method of Moments | Lee & Ingram (1991) |
| `:analytical_gmm` | Analytical moments via Lyapunov equation | --- |

### IRF Matching

Estimates parameters by minimizing the distance between model-implied and data-implied impulse responses:

```math
\hat{\theta} = \arg\min_\theta \, [\Phi^{data}(H) - \Phi^{model}(\theta, H)]' \, W \, [\Phi^{data}(H) - \Phi^{model}(\theta, H)]
```

where ``\Phi^{data}`` is estimated from a VAR on the data and ``W`` is the GMM weighting matrix.

**When to use:** When you have a VAR-identified structural shock and want to match the DSGE model's propagation mechanism to the data. This is the most common method for medium-scale DSGE models.

```julia
est = estimate_dsge(spec, Y_data, [:β, :α, :ρ];
                    method=:irf_matching,
                    var_lags=4,
                    irf_horizon=20,
                    weighting=:two_step)
```

### Euler Equation GMM

Uses Euler equation residuals as moment conditions with lagged variables as instruments:

```math
E[f(y_t, y_{t-1}, y_{t+1}, 0, \theta) \otimes z_t] = 0
```

where ``z_t = [1, y_{t-1}, y_{t-2}, \ldots, y_{t-p}]`` are instruments.

**When to use:** When the model has a well-defined Euler equation and you want to avoid the computational cost of repeatedly solving the model during optimization.

```julia
est = estimate_dsge(spec, Y_data, [:β, :ρ];
                    method=:euler_gmm,
                    n_lags_instruments=4)
```

### Simulated Method of Moments (SMM)

Matches simulated moments to data moments:

```math
\hat{\theta} = \arg\min_\theta \, [m^{data} - m^{sim}(\theta)]' \, W \, [m^{data} - m^{sim}(\theta)]
```

```julia
est = estimate_dsge(spec, Y_data, [:β, :ρ, :σ];
                    method=:smm,
                    sim_ratio=5,
                    burn=100)
```

!!! note "Simulation noise"
    SMM estimates are noisier than analytical GMM due to simulation error. Increase `sim_ratio` to reduce this noise. The asymptotic variance correction factor is ``(1 + 1/\text{sim\_ratio})``.

### Analytical GMM

Uses Lyapunov-equation-based moments (no simulation noise):

**When to use:** This is the most efficient method when the model is linear and you want to match second-moment properties. No simulation noise.

```julia
est = estimate_dsge(spec, Y_data, [:β, :ρ];
                    method=:analytical_gmm,
                    lags=2)
```

### Hansen J-test

All methods report the overidentification test:

```julia
println("J-statistic: ", est.J_stat)
println("p-value: ", est.J_pvalue)
```

A large p-value indicates the model's moment conditions are not rejected by the data.

### Keywords

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:irf_matching` | Estimation method |
| `target_irfs` | `ImpulseResponse` | `nothing` | Pre-computed empirical IRFs |
| `var_lags` | `Int` | `4` | VAR lag order for empirical IRFs |
| `irf_horizon` | `Int` | `20` | IRF horizon for matching |
| `weighting` | `Symbol` | `:two_step` | GMM weighting matrix |
| `n_lags_instruments` | `Int` | `4` | Instrument lags for Euler GMM |
| `sim_ratio` | `Int` | `5` | Simulation ratio for SMM |
| `burn` | `Int` | `100` | Burn-in periods for SMM |
| `moments_fn` | `Function` | `autocovariance_moments` | Moment function for SMM |
| `bounds` | `ParameterTransform` | `nothing` | Parameter bounds |
| `lags` | `Int` | `1` | Autocovariance lags for analytical GMM |

### Return Value

| Field | Type | Description |
|-------|------|-------------|
| `theta` | `Vector{T}` | Estimated deep parameters |
| `vcov` | `Matrix{T}` | Asymptotic covariance matrix |
| `param_names` | `Vector{Symbol}` | Parameter names |
| `method` | `Symbol` | Estimation method used |
| `J_stat` | `T` | Hansen J-test statistic |
| `J_pvalue` | `T` | J-test p-value |
| `solution` | `DSGESolution{T}` | Solution at estimated parameters |
| `converged` | `Bool` | Optimization convergence |
| `spec` | `DSGESpec{T}` | Model specification |

Standard StatsAPI interface: `coef(est)`, `vcov(est)`, `stderror(est)`, `dof(est)`.

---

## Perfect Foresight

`perfect_foresight` solves for the deterministic transition path given a known sequence of future shocks. Agents know the exact realization of shocks in advance (no uncertainty):

```julia
path = perfect_foresight(spec; T_periods=100, shock_path=shock_matrix)
```

Or equivalently via `solve`:

```julia
sol = solve(spec; method=:perfect_foresight,
            T_periods=100, shock_path=shock_matrix)
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `T_periods` | `Int` | `100` | Number of periods |
| `shock_path` | `Matrix` | zeros | ``T \times n_{shocks}`` shock sequence |
| `max_iter` | `Int` | `100` | Newton iteration limit |
| `tol` | `Real` | ``10^{-8}`` | Convergence tolerance |

!!! note "Technical Note"
    Uses a Newton solver with a sparse block-tridiagonal Jacobian. The system stacks all ``T`` periods into a single nonlinear system ``F(X) = 0`` where ``X = [y_1; y_2; \ldots; y_T]`` and iterates ``X^{(k+1)} = X^{(k)} - J^{-1} F(X^{(k)})``. The Jacobian ``J`` has a block-tridiagonal sparsity pattern (each period ``t`` depends only on ``t-1, t, t+1``) which is exploited for efficient factorization.

### Constrained Perfect Foresight (JuMP/Ipopt)

For deterministic paths subject to variable bounds:

```julia
import JuMP, Ipopt

bound = variable_bound(:R, lower=0.0)
path = perfect_foresight(spec, [bound]; T_periods=100, shock_path=shocks)
```

### Constrained Perfect Foresight (PATH MCP)

The MCP formulation handles complementarity at each time period:

```julia
import JuMP, PATHSolver

bound = variable_bound(:R, lower=0.0)
path = perfect_foresight(spec, [bound]; T_periods=100, shock_path=shocks, solver=:path)
```

### Constraint Types

Two constraint types are available:

**Variable bounds** restrict individual endogenous variables:

```julia
variable_bound(:R, lower=0.0)                     # ZLB: R_t >= 0
variable_bound(:C, lower=0.0)                      # Non-negativity
variable_bound(:h, lower=0.0, upper=1.0)           # Hours in [0, 1]
```

**Nonlinear constraints** express general inequality conditions:

```julia
# Collateral constraint: debt <= 0.8 * capital
nonlinear_constraint(
    (y, y_lag, y_lead, e, theta) -> y[3] - 0.8 * y[1];
    label="collateral"
)
```

!!! warning
    `NonlinearConstraint` is supported only with the Ipopt solver, not PATH. PATH requires that constraints reduce to box bounds (variable bounds).

### Return Value

| Field | Type | Description |
|-------|------|-------------|
| `path` | `Matrix{T}` | ``T \times n_{endog}`` level values |
| `deviations` | `Matrix{T}` | ``T \times n_{endog}`` deviations from SS |
| `converged` | `Bool` | Newton convergence flag |
| `iterations` | `Int` | Newton iterations used |
| `spec` | `DSGESpec{T}` | Model specification |

---

## Occasionally Binding Constraints (OccBin)

The OccBin method (Guerrieri & Iacoviello 2015) solves DSGE models with occasionally binding constraints using a piecewise-linear approximation. The economy switches between a **reference regime** (constraint slack) and an **alternative regime** (constraint binding), and the solution iterates until the regime sequence is self-consistent.

### Parsing Constraints

```julia
constraint = parse_constraint(:(R[t] >= 0), spec)
```

Supports `>=` and `<=` directions. The variable must exist in `spec.endog`.

### One Constraint

The typical use case is a zero lower bound on the nominal interest rate:

```julia
# Large negative demand shock pushes economy to ZLB
shocks = zeros(60, 1)
shocks[1, 1] = -8.0

occ_sol = occbin_solve(spec, constraint; shock_path=shocks)

# Inspect convergence
println("Converged: ", occ_sol.converged)
println("Iterations: ", occ_sol.iterations)

# Regime history: 0 = slack, 1 = binding
println("Binding periods: ", findall(vec(occ_sol.regime_history[:, 1]) .== 1))
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `shock_path` | `Matrix{T}` | zeros | ``T \times n_{shocks}`` shock sequence |
| `nperiods` | `Int` | `size(shock_path, 1)` | Number of periods |
| `maxiter` | `Int` | `100` | Max guess-and-verify iterations |

!!! note "Guess-and-Verify Algorithm"
    1. Solve the unconstrained (reference) model and the alternative (binding) model separately
    2. Guess an initial regime sequence (all slack)
    3. Compute time-varying decision rules via backward iteration from the terminal period
    4. Simulate forward using the time-varying rules
    5. Check whether the constraint is satisfied at each period --- update the regime guess
    6. Repeat until the regime sequence converges

### Two Constraints

For models with two occasionally binding constraints (4 possible regime combinations):

```julia
c1 = parse_constraint(:(R[t] >= 0), spec)       # ZLB
c2 = parse_constraint(:(debt[t] <= limit), spec) # Debt ceiling

occ_sol = occbin_solve(spec, c1, c2; shock_path=shocks)
```

The four regimes are: (0,0) neither binds, (1,0) only first binds, (0,1) only second binds, (1,1) both bind.

### OccBin IRFs

Compare linear (unconstrained) and piecewise-linear (constrained) impulse responses:

```julia
occ_irf = occbin_irf(spec, constraint, 1, 40; magnitude=3.0)
plot_result(occ_irf)
```

The plot shows how the ZLB amplifies the output decline: the constrained path (red) drops further than the linear path (blue) because the interest rate cannot go below zero to stimulate the economy.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `magnitude` | `Real` | `1.0` | Shock size (standard deviations) |
| `maxiter` | `Int` | `100` | Max guess-and-verify iterations |

### Return Values

**`OccBinSolution{T}`:**

| Field | Type | Description |
|-------|------|-------------|
| `linear_path` | `Matrix{T}` | ``T \times n`` unconstrained path |
| `piecewise_path` | `Matrix{T}` | ``T \times n`` constrained path |
| `steady_state` | `Vector{T}` | Steady-state values |
| `regime_history` | `Matrix{Int}` | ``T \times n_c`` regime indicators (0=slack, 1=binding) |
| `converged` | `Bool` | Convergence flag |
| `iterations` | `Int` | Guess-and-verify iterations |

**`OccBinIRF{T}`:**

| Field | Type | Description |
|-------|------|-------------|
| `linear` | `Matrix{T}` | ``H \times n`` linear IRF |
| `piecewise` | `Matrix{T}` | ``H \times n`` piecewise IRF |
| `regime_history` | `Matrix{Int}` | ``H \times n_c`` regime indicators |
| `shock_name` | `String` | Name of shocked variable |

---

## Complete Example

A 3-equation New Keynesian model with an IS curve, Phillips curve, and Taylor rule --- solved with multiple methods, estimated, and subjected to a ZLB constraint.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# ── 1. Model specification ──
spec = @dsge begin
    parameters: β = 0.99, σ_c = 1.0, κ = 0.3, φ_π = 1.5, φ_y = 0.5,
                ρ_d = 0.8, ρ_s = 0.7, σ_d = 0.01, σ_s = 0.01
    endogenous: y, π, R, d, s
    exogenous: ε_d, ε_s

    # IS curve (forward-looking demand)
    y[t] = y[t+1] - (1 / σ_c) * (R[t] - π[t+1]) + d[t]

    # New Keynesian Phillips curve (Calvo pricing)
    π[t] = β * π[t+1] + κ * y[t] + s[t]

    # Taylor rule (monetary policy)
    R[t] = φ_π * π[t] + φ_y * y[t]

    # Demand shock (AR(1))
    d[t] = ρ_d * d[t-1] + σ_d * ε_d[t]

    # Supply shock (AR(1))
    s[t] = ρ_s * s[t-1] + σ_s * ε_s[t]
end

# ── 2. First-order solution ──
sol = solve(spec)
println("Determined: ", is_determined(sol))
println("Stable: ", is_stable(sol))

# ── 3. Impulse responses and FEVD ──
irf_result = irf(sol, 40)
plot_result(irf_result)

fevd_result = fevd(sol, 40)
plot_result(fevd_result)

# ── 4. Simulation and analytical moments ──
Y_sim = simulate(sol, 200; rng=Random.MersenneTwister(42))
Sigma = solve_lyapunov(sol.G1, sol.impact)
m = analytical_moments(sol; lags=2)
println("Unconditional variance of output: ", Sigma[1, 1])

# ── 5. Second-order perturbation ──
psol = perturbation_solver(spec; order=2)
Y_sim_2nd = simulate(psol, 1000)  # pruned
girf = irf(psol, 40; irf_type=:girf, n_draws=200)

# ── 6. Estimation (IRF matching) ──
# est = estimate_dsge(spec, Y_data, [:κ, :φ_π, :ρ_d];
#                     method=:irf_matching, var_lags=4, irf_horizon=20)
# report(est)

# ── 7. OccBin: Zero lower bound ──
constraint = parse_constraint(:(R[t] >= 0), spec)

shocks = zeros(40, 2)
shocks[1, 1] = -3.0  # Large negative demand shock

occ_sol = occbin_solve(spec, constraint; shock_path=shocks)
println(occ_sol)

occ_irf = occbin_irf(spec, constraint, 1, 40; magnitude=3.0)
plot_result(occ_irf)
```

---

## Common Pitfalls

### Wrong Steady State

If the numerical solver converges to the wrong steady state, the linearization will be incorrect. Prefer analytical steady states via the `steady_state = begin ... end` block when possible.

```julia
spec = compute_steady_state(spec)
println("Steady state: ", spec.steady_state)
```

### Indeterminate Model

If `is_determined(sol)` returns `false`, the model has sunspot equilibria. Common fix: ensure the Taylor principle holds (``\phi_\pi > 1``).

### Explosive Higher-Order Simulations

If second-order simulations diverge, ensure you are using `PerturbationSolution` (which applies pruning automatically) rather than naively simulating the quadratic decision rules.

### Non-Convergence in Estimation

If `estimate_dsge` doesn't converge:

- **Narrow the parameter space** with `bounds` via `ParameterTransform`
- **Try different starting values** --- the optimizer may be stuck in a local minimum
- **Check identification** --- some parameters may not be identified from your chosen moments
- **Increase horizon** --- for IRF matching, a longer `irf_horizon` may help

### Equation Count Mismatch

The `@dsge` macro requires exactly one equation per endogenous variable. If you need an interest rate smoothing rule, combine the Taylor rule and smoothing into one equation rather than writing two equations for `R`.

### Projection Accuracy

If `max_euler_error` returns values above ``10^{-3}``:

- **Increase polynomial degree** (try `degree=7` or `degree=9`)
- **Widen state bounds** (`scale=4.0` or `scale=5.0`)
- **Use more quadrature nodes** (`n_quad=7`)
- **Check that the model is well-conditioned** --- degenerate near-zero steady-state values can cause numerical issues

---

## References

- Andreasen, M. M., Fernandez-Villaverde, J., & Rubio-Ramirez, J. F. (2018). The Pruned State-Space System for Non-Linear DSGE Models: Theory and Empirical Applications. *Review of Economic Studies*, 85(1), 1--49. [DOI](https://doi.org/10.1093/restud/rdx037)
- Blanchard, O. J., & Kahn, C. M. (1980). The Solution of Linear Difference Models under Rational Expectations. *Econometrica*, 48(5), 1305--1311. [DOI](https://doi.org/10.2307/1912186)
- Christiano, L. J., Eichenbaum, M., & Evans, C. L. (2005). Nominal Rigidities and the Dynamic Effects of a Shock to Monetary Policy. *Journal of Political Economy*, 113(1), 1--45. [DOI](https://doi.org/10.1086/426038)
- Coleman, W. J. (1990). Solving the Stochastic Growth Model by Policy-Function Iteration. *Journal of Business & Economic Statistics*, 8(1), 27--29. [DOI](https://doi.org/10.1080/07350015.1990.10509769)
- Guerrieri, L., & Iacoviello, M. (2015). OccBin: A Toolkit for Solving Dynamic Models with Occasionally Binding Constraints Easily. *Journal of Monetary Economics*, 70, 22--38. [DOI](https://doi.org/10.1016/j.jmoneco.2014.08.005)
- Hansen, L. P. (1982). Large Sample Properties of Generalized Method of Moments Estimators. *Econometrica*, 50(4), 1029--1054. [DOI](https://doi.org/10.2307/1912775)
- Hansen, L. P., & Singleton, K. J. (1982). Generalized Instrumental Variables Estimation of Nonlinear Rational Expectations Models. *Econometrica*, 50(5), 1269--1286. [DOI](https://doi.org/10.2307/1911873)
- Judd, K. L. (1992). Projection Methods for Solving Aggregate Growth Models. *Journal of Economic Theory*, 58(2), 410--452. [DOI](https://doi.org/10.1016/0022-0531(92)90061-L)
- Judd, K. L. (1998). *Numerical Methods in Economics*. Cambridge, MA: MIT Press. ISBN: 0-262-10071-1.
- Judd, K. L., Maliar, L., Maliar, S., & Valero, R. (2014). Smolyak Method for Solving Dynamic Economic Models. *Journal of Economic Dynamics and Control*, 44, 92--123. [DOI](https://doi.org/10.1016/j.jedc.2014.03.003)
- Kim, J., Kim, S., Schaumburg, E., & Sims, C. A. (2008). Calculating and Using Second-Order Accurate Solutions of Discrete Time Dynamic Equilibrium Models. *Journal of Economic Dynamics and Control*, 32(11), 3397--3414. [DOI](https://doi.org/10.1016/j.jedc.2008.02.003)
- Klein, P. (2000). Using the Generalized Schur Form to Solve a Multivariate Linear Rational Expectations Model. *Journal of Economic Dynamics and Control*, 24(10), 1405--1423. [DOI](https://doi.org/10.1016/S0165-1889(99)00045-7)
- Koop, G., Pesaran, M. H., & Potter, S. M. (1996). Impulse Response Analysis in Nonlinear Multivariate Models. *Journal of Econometrics*, 74(1), 119--147. [DOI](https://doi.org/10.1016/0304-4076(95)01753-4)
- Lee, B. S., & Ingram, B. F. (1991). Simulation Estimation of Time-Series Models. *Journal of Econometrics*, 47(2--3), 197--205. [DOI](https://doi.org/10.1016/0304-4076(91)90098-X)
- Schmitt-Grohe, S., & Uribe, M. (2004). Solving Dynamic General Equilibrium Models Using a Second-Order Approximation to the Policy Function. *Journal of Economic Dynamics and Control*, 28(4), 755--775. [DOI](https://doi.org/10.1016/S0165-1889(03)00043-5)
- Sims, C. A. (2002). Solving Linear Rational Expectations Models. *Computational Economics*, 20(1--2), 1--20. [DOI](https://doi.org/10.1023/A:1020517101123)
- Smets, F., & Wouters, R. (2007). Shocks and Frictions in US Business Cycles: A Bayesian DSGE Approach. *American Economic Review*, 97(3), 586--606. [DOI](https://doi.org/10.1257/aer.97.3.586)
