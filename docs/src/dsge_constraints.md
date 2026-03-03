# [Constraints and Occasionally Binding Models](@id dsge_constraints)

Standard linearized DSGE models assume all equilibrium conditions hold with equality at all times. Occasionally binding constraints --- such as the zero lower bound on nominal interest rates, borrowing limits, or irreversible investment --- require specialized solution methods. This page covers three approaches: deterministic perfect foresight, constrained optimization via JuMP (Ipopt and PATH), and the piecewise-linear OccBin algorithm (Guerrieri & Iacoviello 2015). For model specification and linearization, see [DSGE Models](@ref dsge_page). For first-order solvers, see [Linear Solvers](@ref dsge_linear).

## Quick Start

**Recipe 1: Perfect foresight path**

```julia
using MacroEconometricModels

spec = @dsge begin
    parameters: beta = 0.99, alpha = 0.36, delta = 0.025, rho = 0.9, sigma = 0.01
    endogenous: Y, C, K, A
    exogenous: eps_A

    Y[t] = A[t] * K[t-1]^alpha
    C[t] + K[t] = Y[t] + (1 - delta) * K[t-1]
    1 = beta * (C[t] / C[t+1]) * (alpha * A[t+1] * K[t]^(alpha - 1) + 1 - delta)
    A[t] = rho * A[t-1] + sigma * eps_A[t]

    steady_state = begin
        A_ss = 1.0
        K_ss = (alpha * beta / (1 - beta * (1 - delta)))^(1 / (1 - alpha))
        Y_ss = K_ss^alpha
        C_ss = Y_ss - delta * K_ss
        [Y_ss, C_ss, K_ss, A_ss]
    end
end

shocks = zeros(100, 1)
shocks[1, 1] = -3.0  # Large negative TFP shock at period 1
pf = perfect_foresight(spec; shock_path=shocks)
report(pf)
```

**Recipe 2: OccBin ZLB**

```julia
# (assume nk_spec is a New Keynesian model with endogenous variable R)
constraint = parse_constraint(:(R[t] >= 0), nk_spec)
shocks = zeros(100, 1)
shocks[1, 1] = -3.0
occ_sol = occbin_solve(nk_spec, constraint; shock_path=shocks)
report(occ_sol)
```

**Recipe 3: OccBin IRFs --- linear vs constrained**

```julia
occ_irf = occbin_irf(nk_spec, constraint, 1, 40; magnitude=-3.0)
plot_result(occ_irf)
```

---

## Perfect Foresight

A **perfect foresight** path solves for the deterministic transition of the economy given a known sequence of shocks. Agents have perfect information about future shocks --- no uncertainty remains. The solver stacks ``T`` periods of equilibrium conditions into a large nonlinear system:

```math
F(y_1, y_2, \ldots, y_T) = 0
```

where:

- ``y_t`` is the ``n \times 1`` vector of endogenous variables at period ``t``
- ``\bar{y}`` is the steady state
- ``F`` is the ``nT \times 1`` stacked residual vector
- Boundary conditions: ``y_0 = \bar{y}`` (initial steady state) and ``y_{T+1} = \bar{y}`` (terminal steady state)

The function `perfect_foresight` solves this system via Newton iteration. The same solver is accessible through the unified `solve` interface:

```julia
# Direct call
pf = perfect_foresight(spec; T_periods=100, shock_path=shocks)

# Via unified solve interface
pf = solve(spec; method=:perfect_foresight, T_periods=100, shock_path=shocks)
```

The `PerfectForesightPath{T}` result contains both the level path and deviations from steady state:

```julia
pf.path         # 100 x 4 matrix of variable levels
pf.deviations   # 100 x 4 matrix of deviations from steady state
pf.converged    # true if Newton iteration converged
pf.iterations   # number of Newton iterations used
```

!!! note "Technical Note"
    The Newton solver exploits the block-tridiagonal structure of the Jacobian via sparse LU factorization. Each Newton step solves ``J \Delta x = -F(x)`` where ``J`` is ``nT \times nT`` but has only ``3n^2 T`` non-zeros (vs ``n^2 T^2`` for dense). Numerical Jacobians use central differences with adaptive step sizes.

### Keywords

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `T_periods` | `Int` | `100` | Number of simulation periods |
| `shock_path` | `Union{Nothing, Matrix}` | `nothing` | ``T \times n_\varepsilon`` shock realizations (zeros if omitted) |
| `max_iter` | `Int` | `100` | Newton iteration limit |
| `tol` | `Real` | ``10^{-8}`` | Convergence tolerance (max absolute residual) |
| `constraints` | `Vector{<:DSGEConstraint}` | `[]` | Variable bounds and nonlinear constraints |
| `solver` | `Union{Nothing, Symbol}` | `nothing` | `:ipopt` or `:path` (auto-detected from constraint types) |

### Return Value

**`PerfectForesightPath{T}`:**

| Field | Type | Description |
|-------|------|-------------|
| `path` | `Matrix{T}` | ``T \times n`` variable levels |
| `deviations` | `Matrix{T}` | ``T \times n`` deviations from steady state |
| `converged` | `Bool` | Newton convergence flag |
| `iterations` | `Int` | Newton iterations used |
| `spec` | `DSGESpec{T}` | Back-reference to model specification |

---

## Constrained Perfect Foresight

When variable bounds or nonlinear inequality constraints are present, the perfect foresight problem becomes a constrained optimization problem. Two solver backends are available via JuMP.

### Ipopt (NLP)

**Ipopt** (Interior Point Optimizer) handles general nonlinear constraints. Variable bounds and nonlinear inequalities are both supported:

```julia
import JuMP, Ipopt

# Variable bound: ZLB on nominal rate
zlb = variable_bound(:R, lower=0.0)

# Nonlinear constraint: debt-to-GDP ceiling
debt_limit = nonlinear_constraint(
    (y, y_lag, y_lead, e, theta) -> y[debt_idx] / y[gdp_idx] - 0.6;
    label="Debt-to-GDP <= 60%"
)

pf = perfect_foresight(spec; shock_path=shocks,
                        constraints=[zlb, debt_limit])
```

The `nonlinear_constraint` function takes a closure with the standard residual signature `(y, y_lag, y_lead, e, theta) -> scalar`. The constraint is satisfied when the return value is ``\leq 0``.

### PATH (MCP)

**PATH** solves the perfect foresight problem as a Mixed Complementarity Problem (MCP). For each variable ``i`` with bounds ``[l_i, u_i]``:

```math
l_i \leq y_i \leq u_i, \quad f_i(y) \begin{cases} \geq 0 & \text{if } y_i = l_i \\ = 0 & \text{if } l_i < y_i < u_i \\ \leq 0 & \text{if } y_i = u_i \end{cases}
```

where ``f_i(y)`` is the ``i``-th equilibrium residual. This complementarity structure is natural for problems where a constraint replaces an equilibrium condition when binding (e.g., the Taylor rule is replaced by ``R_t = 0`` at the ZLB).

```julia
import JuMP, PATHSolver

zlb = variable_bound(:R, lower=0.0)
pf = perfect_foresight(spec; shock_path=shocks,
                        constraints=[zlb], solver=:path)
```

When `solver` is not specified, the solver is auto-detected: pure `VariableBound` constraints use PATH (if PATHSolver.jl is loaded), and any `NonlinearConstraint` requires Ipopt.

!!! warning "PATH vs Ipopt"
    Use **PATH** for pure box constraints (variable bounds) with a natural complementarity structure. Use **Ipopt** for general nonlinear inequality constraints. PATH requires the PATHSolver.jl package; Ipopt requires Ipopt.jl. Both require JuMP.jl. These are weak dependencies --- load them with `import` before calling `perfect_foresight` with constraints.

### Constraint Constructors

| Constructor | Type | Use Case |
|-------------|------|----------|
| `variable_bound(:var, lower=0.0)` | `VariableBound{T}` | Box constraints (ZLB, positivity, bounded hours) |
| `variable_bound(:var, lower=0.0, upper=1.0)` | `VariableBound{T}` | Two-sided bounds (hours in [0, 1]) |
| `nonlinear_constraint(fn; label="...")` | `NonlinearConstraint{T}` | General inequalities (debt limits, leverage ratios) |

---

## Occasionally Binding Constraints (OccBin)

The **OccBin** algorithm (Guerrieri & Iacoviello 2015) solves DSGE models with occasionally binding constraints using a piecewise-linear approach. Unlike the global methods on the [Nonlinear Methods](@ref dsge_nonlinear) page, OccBin uses the linearized model and switches between regimes (constraint binding vs. slack) period by period. This makes it fast and easy to implement, at the cost of local (rather than global) accuracy.

### Constraint Specification

The `parse_constraint` function converts a Julia expression into an `OccBinConstraint`:

```julia
# ZLB: nominal rate cannot go below zero
constraint = parse_constraint(:(R[t] >= 0), spec)

# Upper bound: output gap capped at 5%
cap = parse_constraint(:(gap[t] <= 0.05), spec)
```

The constraint defines two regimes:

- **Slack regime**: The original model equation holds --- ``R_t`` is determined by the Taylor rule
- **Binding regime**: The constraint replaces the policy equation with ``R_t = 0``

The variable name must match one of the endogenous variables declared in the `@dsge` block.

### One-Constraint Example (ZLB)

```julia
# Large negative demand shock pushes economy to ZLB
shocks = zeros(100, 1)
shocks[1, 1] = -3.0

occ_sol = occbin_solve(nk_spec, constraint; shock_path=shocks)
report(occ_sol)
```

The solution contains both the unconstrained linear path and the piecewise-linear constrained path:

```julia
occ_sol.linear_path      # 100 x n unconstrained path (deviations from SS)
occ_sol.piecewise_path   # 100 x n constrained path (deviations from SS)
occ_sol.regime_history   # 100 x 1 matrix: 0 = slack, 1 = binding
occ_sol.converged        # true if regime sequence converged
```

### Guess-and-Verify Algorithm

The OccBin algorithm proceeds as follows:

1. Solve the unconstrained (reference) model via Gensys to obtain the state-space matrices ``P, Q``
2. Derive the alternative (binding) regime by replacing the constraint equation with the bound
3. Extract linearized coefficient matrices ``(A, B, C, D)`` for both regimes
4. **Initial guess**: assume no periods are binding
5. **Backward iteration**: compute time-varying decision rules from the last binding period back to period 1, using the appropriate regime matrices at each period
6. **Forward simulation**: simulate the piecewise-linear path using the time-varying rules
7. **Constraint evaluation**: check whether the constraint is violated (for slack periods) or the shadow value indicates the constraint should release (for binding periods)
8. **Repeat** steps 5--7 until the regime sequence converges or `maxiter` is reached

!!! note "Technical Note"
    OccBin linearizes the model separately in each regime. The binding regime replaces the constraint equation with the bound (e.g., ``R_t = 0``), producing different ``\Gamma_0^b, \Gamma_1^b`` matrices. The backward iteration substitutes the next-period rule ``\hat{y}_{t+1} = P_{t+1} \hat{y}_t + D_{t+1}`` into the current-period equation to solve for time-varying policy matrices ``P_t``. The unconstrained terminal condition provides the starting point for backward recursion.

### Keywords

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `shock_path` | `Matrix{T}` | `zeros(40, n_exog)` | ``T \times n_\varepsilon`` shock sequence |
| `nperiods` | `Int` | `size(shock_path, 1)` | Number of periods to simulate |
| `maxiter` | `Int` | `100` | Maximum regime iterations |

### Two-Constraint Example

OccBin supports two simultaneous constraints. The algorithm generalizes to four regimes: neither binding, only constraint 1 binding, only constraint 2 binding, and both binding.

```julia
zlb = parse_constraint(:(R[t] >= 0), spec)
borrow = parse_constraint(:(D[t] <= D_max), spec)
occ_sol = occbin_solve(spec, zlb, borrow; shock_path=shocks)
```

The `regime_history` matrix has two columns --- one per constraint --- recording which regimes are active in each period. An optional `curb_retrench=true` keyword limits constraint relaxation to one period per iteration, which helps prevent oscillation in difficult two-constraint problems.

### OccBin IRFs

OccBin impulse responses compare the linear and constrained paths for a given shock:

```julia
occ_irf = occbin_irf(nk_spec, constraint, 1, 40; magnitude=-3.0)
plot_result(occ_irf)
```

The `magnitude` keyword controls the shock size. It must be large enough to trigger the constraint --- small shocks produce identical linear and piecewise paths. The result contains both the unconstrained and constrained IRFs for direct comparison.

For two-constraint IRFs:

```julia
occ_irf = occbin_irf(spec, zlb, borrow, 1, 40; magnitude=-3.0)
```

### Return Values

**`OccBinSolution{T}`:**

| Field | Type | Description |
|-------|------|-------------|
| `linear_path` | `Matrix{T}` | ``T \times n`` unconstrained path (deviations from SS) |
| `piecewise_path` | `Matrix{T}` | ``T \times n`` piecewise-linear constrained path |
| `steady_state` | `Vector{T}` | Steady-state values |
| `regime_history` | `Matrix{Int}` | ``T \times n_c`` regime indicators (0 = slack, 1 = binding) |
| `converged` | `Bool` | Regime convergence flag |
| `iterations` | `Int` | Regime iterations used |
| `spec` | `DSGESpec{T}` | Back-reference to model specification |
| `varnames` | `Vector{String}` | Variable display labels |

**`OccBinIRF{T}`:**

| Field | Type | Description |
|-------|------|-------------|
| `linear` | `Matrix{T}` | ``H \times n`` unconstrained IRF |
| `piecewise` | `Matrix{T}` | ``H \times n`` constrained IRF |
| `regime_history` | `Matrix{Int}` | Regime indicators during IRF horizon |
| `varnames` | `Vector{String}` | Variable display labels |
| `shock_name` | `String` | Name of the shocked variable |

---

## Common Pitfalls

1. **Non-convergence in perfect foresight**: Increase `T_periods` or reduce the shock magnitude. The terminal condition assumes return to steady state --- if the shock is too persistent or too large, the horizon must be long enough for the economy to converge back.

2. **OccBin regime cycling**: The guess-and-verify algorithm can cycle between regime sequences without converging. For one-constraint problems, increase `maxiter`. For two-constraint problems, set `curb_retrench=true` to limit relaxation to one period per iteration.

3. **Missing JuMP extensions**: Constrained solvers require `import JuMP, Ipopt` or `import JuMP, PATHSolver` *before* calling `perfect_foresight` with constraints. Without these imports, the package raises an `ArgumentError` with installation instructions.

4. **Wrong constraint direction**: `:(R[t] >= 0)` means "``R`` must be at least 0" (a lower bound). `:(D[t] <= D_max)` means "``D`` must be at most `D_max`" (an upper bound). Verify that the direction matches the economic interpretation.

5. **Constraint binding at terminal period**: If `regime_history` shows the constraint binding in the last period, the horizon is too short. OccBin warns when this occurs --- increase `nperiods` until the constraint releases before the terminal period.

---

## References

- Guerrieri, L., & Iacoviello, M. (2015). OccBin: A Toolkit for Solving Dynamic Models with Occasionally Binding Constraints Easily. *Journal of Monetary Economics*, 70, 22--38. [DOI](https://doi.org/10.1016/j.jmoneco.2014.08.005)

- Ferris, M. C., & Munson, T. S. (1999). Interfaces to PATH 3.0: Design, Implementation and Usage. *Computational Optimization and Applications*, 12(1), 207--227. [DOI](https://doi.org/10.1023/A:1018636318047)

- Rendahl, P. (2017). Linear Time Iteration. Unpublished manuscript.

- Sims, C. A. (2002). Solving Linear Rational Expectations Models. *Computational Economics*, 20(1), 1--20. [DOI](https://doi.org/10.1023/A:1020517101123)
