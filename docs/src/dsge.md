# [DSGE Models](@id dsge_page)

**MacroEconometricModels.jl** provides a complete toolkit for specifying, solving, simulating, and estimating Dynamic Stochastic General Equilibrium (DSGE) models. The package covers the full workflow from model definition through structural estimation, with six solution methods spanning linear, higher-order, and global approaches.

- **Specification**: The `@dsge` macro provides a domain-specific language for writing equilibrium conditions with time-indexed variables
- **Steady State**: Analytical or numerical steady-state computation with two-phase optimization and optional JuMP constraints
- **Linearization**: Automatic first-order approximation via numerical Jacobians in the Sims (2002) canonical form
- **Linear Solvers**: Three first-order solvers --- Gensys (Sims 2002), Blanchard-Kahn (1980), and Klein (2000) --- producing the state-space solution; see [Linear Solvers](@ref dsge_linear)
- **Nonlinear Methods**: Up to 3rd-order perturbation with Andreasen, Fernandez-Villaverde & Rubio-Ramirez (2018) pruning, Chebyshev collocation, and policy function iteration for globally accurate policy functions; see [Nonlinear Methods](@ref dsge_nonlinear)
- **Constraints**: Perfect foresight paths, OccBin occasionally-binding constraints (Guerrieri & Iacoviello 2015), and constrained optimization via JuMP/Ipopt (NLP) and PATH (MCP); see [Constraints](@ref dsge_constraints)
- **Estimation**: Four GMM-based methods (one-step, two-step, iterative, CU) for IRF matching, plus Bayesian estimation via SMC, SMC`` ^2 `` with two-stage delayed acceptance, and Random-Walk Metropolis-Hastings; see [Estimation](@ref dsge_estimation)
- **Simulation and IRFs**: Stochastic and pruned simulation, analytical and generalized impulse responses, FEVD, and unconditional moments via Lyapunov equation; see [Nonlinear Methods](@ref dsge_nonlinear)

All results integrate with `plot_result()` for interactive D3.js visualization and `report()` for publication-quality output.

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

**Recipe 6: Bayesian estimation via SMC``^2``**

```julia
using Distributions
result = estimate_dsge_bayes(spec, data, [0.99, 0.9, 0.01];
    priors=Dict(:β => Beta(99, 1), :ρ => Beta(5, 2), :σ => InverseGamma(2, 0.01)),
    method=:smc2, observables=[:y], n_smc=200, n_particles=100,
    solver=:projection, solver_kwargs=(degree=5,))
report(result)
```

---

## Model Specification

The `@dsge` macro provides a domain-specific language for specifying DSGE models. It parses the model block into a `DSGESpec{T}` object containing equations, parameters, and variable declarations.

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

Variables with `[t+1]` subscripts generate expectation errors in the Sims (2002) canonical form. The number of forward-looking equations determines the dimension of the ``\Pi`` matrix and, via the Blanchard-Kahn (1980) condition, the number of unstable eigenvalues required for determinacy.

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

The **steady state** ``\bar{y}`` satisfies the equilibrium system in the absence of shocks:

```math
f(\bar{y}, \bar{y}, \bar{y}, 0, \theta) = 0
```

where:
- ``\bar{y}`` is the ``n \times 1`` vector of endogenous variables at the steady state
- ``\theta`` is the vector of deep structural parameters
- ``f`` is the system of ``n`` equilibrium conditions

For the RBC model above, the analytical steady state is:

```math
\bar{A} = 1, \quad \bar{K} = \left(\frac{\alpha\beta}{1 - \beta(1-\delta)}\right)^{\frac{1}{1-\alpha}}, \quad \bar{Y} = \bar{K}^\alpha, \quad \bar{C} = \bar{Y} - \delta\bar{K}
```

### Numerical Computation

`compute_steady_state` uses a two-phase optimizer: Nelder-Mead for robustness (derivative-free global exploration), then L-BFGS for refinement. It minimizes the sum of squared residuals ``\sum_i f_i(\bar{y})^2``.

```julia
spec = compute_steady_state(spec)
report(spec)
```

The optimizer converges to the steady state from a default initial guess of ones. For models with multiple equilibria, providing a good starting point via `initial_guess` avoids convergence to an economically irrelevant solution.

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

When the `steady_state` block is provided, `compute_steady_state` (or `solve`) uses it directly and validates the result against the equations. The analytical path is faster and avoids numerical convergence issues, but the user is responsible for correctness --- the validator checks that ``\|f(\bar{y})\| < 10^{-10}``.

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

where:
- ``y_i`` is the ``i``-th endogenous variable
- ``l_i, u_i`` are the lower and upper bounds
- ``f_i(y)`` is the ``i``-th equilibrium residual

!!! note "When to use PATH vs Ipopt"
    Use **PATH** when constraints are box bounds (variable bounds only) and the model has a natural complementarity structure --- for example, a ZLB where ``R_t \geq 0`` and the Taylor rule holds with equality when ``R_t > 0``. Use **Ipopt** when you have general nonlinear inequality constraints or when the complementarity interpretation is not natural. For full details on constrained solvers, see [Constraints](@ref dsge_constraints).

---

## Linearization

`linearize` computes a first-order Taylor expansion around the steady state using numerical Jacobians (central differences). It produces the Sims (2002) canonical form:

```math
\Gamma_0 \, y_t = \Gamma_1 \, y_{t-1} + C + \Psi \, \varepsilon_t + \Pi \, \eta_t
```

where:
- ``y_t`` is the ``n \times 1`` vector of endogenous variables (deviations from steady state)
- ``\Gamma_0`` is the ``n \times n`` coefficient matrix on current-period variables
- ``\Gamma_1`` is the ``n \times n`` coefficient matrix on lagged variables
- ``C`` is the ``n \times 1`` constant vector
- ``\Psi`` is the ``n \times n_{shocks}`` shock loading matrix
- ``\Pi`` is the ``n \times n_{expect}`` expectation error selection matrix
- ``\varepsilon_t`` is the vector of exogenous shocks
- ``\eta_t = y_t - E_{t-1}[y_t]`` is the vector of expectation errors for forward-looking variables

```julia
ld = linearize(spec)
```

The matrix pair ``(\Gamma_0, \Gamma_1)`` defines a generalized eigenvalue problem whose solution governs the model dynamics. The three [Linear Solvers](@ref dsge_linear) --- Gensys, Blanchard-Kahn, and Klein --- each decompose this pencil to extract the stable state-space representation.

!!! note "Technical Note"
    The matrices are computed via central differences with step size ``h = \max(10^{-7}, 10^{-7} |y_j|)``. No analytical derivatives are required. ``\Gamma_0`` contains coefficients on ``y_t``, ``\Gamma_1`` on ``y_{t-1}``, ``\Psi`` on shocks, and ``\Pi`` selects the forward-looking equations for expectation errors.

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

## Complete Example

This example specifies, solves, and analyzes a full RBC model using the core functions covered on this page:

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Specify the RBC model with analytical steady state
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

# Verify steady state
report(spec)

# Linearize and inspect the canonical form matrices
ld = linearize(spec)

# Solve with default Gensys method
sol = solve(spec)

# IRFs and FEVD
result = irf(sol, 40)
plot_result(result)
```

The `spec` object stores the parsed model. `linearize` produces the Sims (2002) canonical form ``(\Gamma_0, \Gamma_1, C, \Psi, \Pi)``. `solve` dispatches to the Gensys algorithm and returns a `DSGESolution` with the state-space representation ``y_t = G_1 y_{t-1} + C + \text{impact} \cdot \varepsilon_t``. The IRF and FEVD functions operate on this solution. For higher-order or global solutions, see [Nonlinear Methods](@ref dsge_nonlinear).

---

## Common Pitfalls

1. **Steady-state validation failure**: When providing an analytical `steady_state` block, the validator checks ``\|f(\bar{y})\| < 10^{-10}``. A common cause of failure is mismatched variable ordering --- the returned vector must match the `endogenous:` declaration order exactly.

2. **Equation count mismatch**: The number of equations must equal the number of endogenous variables. Missing an equilibrium condition or double-counting a definition produces `DimensionMismatch`. Each equation is written as `LHS = RHS`; the parser rearranges to residual form automatically.

3. **Timing convention confusion**: ``K_t`` chosen at time ``t`` is written `K[t]`. Beginning-of-period capital (predetermined) is `K[t-1]`. A forward-looking Euler equation uses `C[t+1]`. Misplacing a time subscript silently changes the ``\Gamma_0``/``\Gamma_1`` structure and can cause indeterminacy.

4. **Numerical steady state converges to wrong equilibrium**: For models with multiple equilibria, the default initial guess (vector of ones) may converge to an economically irrelevant solution. Provide `initial_guess` close to the desired equilibrium, or use the analytical `steady_state` block.

5. **Constrained steady state requires JuMP**: Calling `compute_steady_state(spec, [bound])` requires `import JuMP, Ipopt` (for NLP) or `import JuMP, PATHSolver` (for MCP) before the call. Without these imports, the package raises an `ArgumentError`.

---

## References

- Sims, C. A. (2002). Solving Linear Rational Expectations Models.
  *Computational Economics*, 20(1-2), 1-20. [DOI](https://doi.org/10.1023/A:1020517101123)

- Blanchard, O. J., & Kahn, C. M. (1980). The Solution of Linear Difference Models under Rational Expectations.
  *Econometrica*, 48(5), 1305-1311. [DOI](https://doi.org/10.2307/1912186)
