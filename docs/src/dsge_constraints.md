# [Constraints and Occasionally Binding Models](@id dsge_constraints)

Standard linearized DSGE models assume all equilibrium conditions hold with equality at all times. Occasionally binding constraints --- such as the zero lower bound on nominal interest rates, borrowing limits, or irreversible investment --- require specialized solution methods. This page covers three approaches: deterministic perfect foresight with built-in constrained solvers (Optim.jl, NLopt.jl), optional JuMP-based backends (Ipopt and PATH), and the piecewise-linear OccBin algorithm (Guerrieri & Iacoviello 2015). For model specification and linearization, see [DSGE Models](@ref dsge_page). For first-order solvers, see [Linear Solvers](@ref dsge_linear).


## Quick Start

```@setup dsge_constraints
using MacroEconometricModels, Random
Random.seed!(42)
```

**Recipe 1: Perfect foresight path**

```@example dsge_constraints
spec = @dsge begin
    parameters: β = 0.99, σ_c = 1.0, κ = 0.024, ϕ_π = 1.5, ϕ_y = 0.125,
                ρ_d = 0.9, σ_d = 0.01
    endogenous: y, π, R, d
    exogenous: ε_d

    y[t] = y[t+1] - σ_c * (R[t] - π[t+1]) + d[t]
    π[t] = β * π[t+1] + κ * y[t]
    R[t] = ϕ_π * π[t] + ϕ_y * y[t]
    d[t] = ρ_d * d[t-1] + σ_d * ε_d[t]
end
spec = compute_steady_state(spec)

shocks = zeros(100, 1)
shocks[1, 1] = -3.0  # Large negative demand shock at period 1
pf = perfect_foresight(spec; shock_path=shocks)
report(pf)
```

**Recipe 2: OccBin borrowing constraint**

```@example dsge_constraints
borrow_spec = @dsge begin
    parameters: β = 20/21, R = 21/20, ρ = 0.9, σ = 0.05, M = 1.0
    endogenous: b, c, y
    exogenous: u

    # Savings optimality (substituted Euler, β*R = 1)
    b[t] = (y[t+1] + b[t+1] + R * b[t-1] - y[t]) / (1 + R)
    # Budget constraint
    c[t] = y[t] + b[t] - R * b[t-1]
    # Income process
    y[t] = y[t-1]^ρ * exp(σ * u[t])
end
borrow_spec = compute_steady_state(borrow_spec;
    method=:analytical, ss_fn = θ -> [0.0, 1.0, 1.0])

# Borrowing limit: debt cannot exceed M
constraint = parse_constraint(:(b[t] <= 1.0), borrow_spec)
borrow_shocks = zeros(60, 1)
borrow_shocks[1, 1] = -40.0  # Large negative income shock
occ_sol = occbin_solve(borrow_spec, constraint; shock_path=borrow_shocks)
report(occ_sol)
```

**Recipe 3: OccBin IRFs --- linear vs constrained**

```@example dsge_constraints
occ_irf = occbin_irf(borrow_spec, constraint, 1, 40; magnitude=-40.0)
nothing # hide
```

```julia
plot_result(occ_irf)
```

```@raw html
<iframe src="../assets/plots/occbin_irf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
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

The function `perfect_foresight` solves this system using [NonlinearSolve.jl](https://github.com/SciML/NonlinearSolve.jl) with `NewtonRaphson()` as the default algorithm. The same solver is accessible through the unified `solve` interface:

```@example dsge_constraints
# Direct call
pf = perfect_foresight(spec; T_periods=100, shock_path=shocks)

# Via unified solve interface
pf = solve(spec; method=:perfect_foresight, T_periods=100, shock_path=shocks)
nothing # hide
```

The `PerfectForesightPath{T}` result contains both the level path and deviations from steady state:

```@example dsge_constraints
pf.path         # T x n matrix of variable levels
pf.deviations   # T x n matrix of deviations from steady state
pf.converged    # true if Newton iteration converged
pf.iterations   # number of Newton iterations used
```

!!! note "Technical Note"
    The solver exploits the block-tridiagonal structure of the Jacobian via sparse LU factorization. Each Newton step solves ``J \Delta x = -F(x)`` where ``J`` is ``nT \times nT`` but has only ``3n^2 T`` non-zeros (vs ``n^2 T^2`` for dense). Numerical Jacobians use central differences with adaptive step sizes. The `algorithm` keyword accepts any NonlinearSolve.jl algorithm (e.g., `NonlinearSolve.TrustRegion()`).

### Keywords

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `T_periods` | `Int` | `100` | Number of simulation periods |
| `shock_path` | `Union{Nothing, Matrix}` | `nothing` | ``T \times n_\varepsilon`` shock realizations (zeros if omitted) |
| `max_iter` | `Int` | `100` | Newton iteration limit |
| `tol` | `Real` | ``10^{-8}`` | Convergence tolerance (max absolute residual) |
| `constraints` | `Vector{<:DSGEConstraint}` | `[]` | Variable bounds and nonlinear constraints |
| `solver` | `Union{Nothing, Symbol}` | `nothing` | `:nonlinearsolve`, `:optim`, `:nlopt`, `:ipopt`, or `:path`; auto-detected |
| `algorithm` | `Any` | `NewtonRaphson()` | Algorithm for chosen backend (e.g., `Optim.IPNewton()`, `:LN_COBYLA`) |

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

When variable bounds or nonlinear inequality constraints are present, the solver uses a five-tier hierarchy. Tiers 1--4 require no additional packages. Tiers 4--5 provide optional JuMP-based backends for large-scale problems.

| Tier | Backend | Handles | Activation |
|------|---------|---------|------------|
| 1 | NonlinearSolve.jl | Unconstrained, non-binding box | Default |
| 2 | Projected Newton | Box-constrained PF | Auto-escalation when bounds violated |
| 3 | Optim.jl `Fminbox(LBFGS())` | Box-constrained SS | Auto-escalation when SS bounds violated |
| 4 | NLopt.jl `LD_SLSQP` | Nonlinear inequality constraints | Default for `NonlinearConstraint` |
| 5 | JuMP+Ipopt / JuMP+PATH | Full NLP / MCP | Explicit `solver=:ipopt` or `:path` |

The solver is auto-detected from constraint types. Pure `VariableBound` constraints start with NonlinearSolve and auto-escalate to projected Newton (PF) or Optim.jl (SS) if bounds are violated. `NonlinearConstraint` dispatches to NLopt.jl when JuMP is absent, or Ipopt when JuMP is loaded. Override with the `solver` keyword.

### Box Constraints (Built-in)

Box constraints work out of the box --- no additional packages required. For perfect foresight, the solver first attempts the unconstrained Newton solve. If any variable violates its bounds, it auto-escalates to a projected Newton method that preserves the sparse block-tridiagonal Jacobian structure:

```julia
# ZLB on nominal rate --- no imports needed
zlb = variable_bound(:R, lower=0.0)
pf = perfect_foresight(spec; shock_path=shocks, constraints=[zlb])
```

The projected Newton solver uses NCP (nonlinear complementarity problem) convergence criteria: at interior points the residual must equal zero, at a binding lower bound the residual must be non-negative, and at a binding upper bound the residual must be non-positive.

### Nonlinear Constraints (NLopt)

For general nonlinear inequality constraints, NLopt.jl `LD_SLSQP` is the default solver. No additional packages required:

```julia
# Variable bound + nonlinear constraint
zlb = variable_bound(:R, lower=0.0)
debt_limit = nonlinear_constraint(
    (y, y_lag, y_lead, e, theta) -> y[debt_idx] / y[gdp_idx] - 0.6;
    label="Debt-to-GDP <= 60%"
)

pf = perfect_foresight(spec; shock_path=shocks,
                        constraints=[zlb, debt_limit])
```

The `nonlinear_constraint` function takes a closure with the standard residual signature `(y, y_lag, y_lead, e, theta) -> scalar`. The constraint is satisfied when the return value is ``\leq 0``. For perfect foresight, NLopt formulates the problem as a feasibility problem with equality constraints (model equations) and inequality constraints (user-specified).

!!! warning "NLopt PF Performance"
    NLopt's SLSQP is a dense algorithm. For large PF problems (T × n > 1000), consider `solver=:ipopt` with JuMP + Ipopt for better scalability. The solver warns when the problem size exceeds this threshold.

### Advanced: JuMP Backends (Ipopt and PATH)

For large-scale problems or complementarity formulations, JuMP-based backends are available as optional weak dependencies:

**Ipopt** (Interior Point Optimizer) handles general NLP problems. It is more robust than NLopt for large systems:

```julia
import JuMP, Ipopt

pf = perfect_foresight(spec; shock_path=shocks,
                        constraints=[zlb, debt_limit], solver=:ipopt)
```

**PATH** solves the problem as a Mixed Complementarity Problem (MCP). For each variable ``i`` with bounds ``[l_i, u_i]``:

```math
l_i \leq y_i \leq u_i, \quad f_i(y) \begin{cases} \geq 0 & \text{if } y_i = l_i \\ = 0 & \text{if } l_i < y_i < u_i \\ \leq 0 & \text{if } y_i = u_i \end{cases}
```

This complementarity structure is natural for problems where a constraint replaces an equilibrium condition when binding (e.g., the Taylor rule is replaced by ``R_t = 0`` at the ZLB).

```julia
import JuMP, PATHSolver

pf = perfect_foresight(spec; shock_path=shocks,
                        constraints=[zlb], solver=:path)
```

!!! note "Solver Selection Guide"
    **Built-in** (no extra packages): Box constraints auto-escalate from NonlinearSolve to projected Newton (PF) or Optim.jl (SS). Nonlinear constraints default to NLopt.jl. **JuMP backends**: Use `:ipopt` for large-scale NLP or when NLopt doesn't converge. Use `:path` for pure box constraints with a natural complementarity structure (ZLB). Both require JuMP.jl as a weak dependency.

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

# Borrowing limit: debt cannot exceed M
borrow = parse_constraint(:(b[t] <= 1.0), spec)

# Upper bound: output gap capped at 5%
cap = parse_constraint(:(gap[t] <= 0.05), spec)
```

The constraint defines two regimes:

- **Slack regime**: The original model equation holds --- the variable is determined by its defining equation (e.g., a Taylor rule for ``R_t``, a savings optimality condition for ``b_t``)
- **Binding regime**: The constraint replaces the defining equation with the bound (e.g., ``R_t = 0`` at the ZLB, ``b_t = M`` at the borrowing limit)

The variable name must match one of the endogenous variables declared in the `@dsge` block.

### One-Constraint Example (Borrowing Limit)

```@example dsge_constraints
# Large negative income shock pushes agent to borrowing limit
borrow_shocks = zeros(60, 1)
borrow_shocks[1, 1] = -40.0

occ_sol = occbin_solve(borrow_spec, constraint; shock_path=borrow_shocks)
report(occ_sol)
```

The solution contains both the unconstrained linear path and the piecewise-linear constrained path:

```@example dsge_constraints
occ_sol.linear_path      # 60 x 3 unconstrained path (deviations from SS)
occ_sol.piecewise_path   # 60 x 3 constrained path (deviations from SS)
occ_sol.regime_history   # 60 x 1 matrix: 0 = slack, 1 = binding
occ_sol.converged        # true if regime sequence converged
```

When the constraint binds, ``b_t = M`` replaces the savings optimality condition. The budget constraint continues to hold, so consumption absorbs the full income shortfall that can no longer be smoothed through additional borrowing.

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
occ_irf = occbin_irf(borrow_spec, constraint, 1, 40; magnitude=-40.0)
plot_result(occ_irf)
```

```@raw html
<iframe src="../assets/plots/occbin_irf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
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

## Complete Example

This example builds a consumption-savings model with an occasionally binding borrowing constraint and compares unconstrained and constrained impulse responses:

```@example dsge_constraints
# Consumption-savings model with borrowing limit
# β*R = 1 ensures a clean unconstrained steady state (b=0, c=1, y=1)
bc_spec = @dsge begin
    parameters: β = 20/21, R = 21/20, ρ = 0.9, σ = 0.05, M = 1.0
    endogenous: b, c, y
    exogenous: u

    # Savings optimality: substituted Euler equation
    # Derived from c[t] = c[t+1] (log utility, β*R = 1) + budget constraint
    b[t] = (y[t+1] + b[t+1] + R * b[t-1] - y[t]) / (1 + R)
    # Budget constraint (accounting identity — always holds)
    c[t] = y[t] + b[t] - R * b[t-1]
    # Income process (log AR(1))
    y[t] = y[t-1]^ρ * exp(σ * u[t])
end
bc_spec = compute_steady_state(bc_spec;
    method=:analytical, ss_fn = θ -> [0.0, 1.0, 1.0])

# Unconstrained solution for comparison
sol_unc = solve(bc_spec)
irf_unc = irf(sol_unc, 40)

# OccBin with borrowing limit: b <= M
bc_constraint = parse_constraint(:(b[t] <= 1.0), bc_spec)

# Large negative income shock pushes agent to borrowing limit
bc_irf = occbin_irf(bc_spec, bc_constraint, 1, 40; magnitude=-40.0)
nothing # hide
```

```julia
plot_result(bc_irf)
```

```@raw html
<iframe src="../assets/plots/occbin_irf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The unconstrained IRF shows optimal consumption smoothing: the agent borrows freely in response to a negative income shock, spreading the impact across many periods. The OccBin IRF reveals that when borrowing hits the limit ``b_t = M``, the savings optimality condition is replaced by the bound. Consumption must absorb the full income shortfall that can no longer be smoothed, producing a sharper drop. The `regime_history` field tracks which periods the constraint binds.

---

## Common Pitfalls

1. **Non-convergence in perfect foresight**: Increase `T_periods` or reduce the shock magnitude. The terminal condition assumes return to steady state --- if the shock is too persistent or too large, the horizon must be long enough for the economy to converge back.

2. **OccBin regime cycling**: The guess-and-verify algorithm can cycle between regime sequences without converging. For one-constraint problems, increase `maxiter`. For two-constraint problems, set `curb_retrench=true` to limit relaxation to one period per iteration.

3. **NLopt PF limits**: NLopt's SLSQP is a dense method that struggles with large perfect foresight problems. For models with T × n > 1000 variables, use `solver=:ipopt` with JuMP + Ipopt. For box-constrained PF, the built-in projected Newton solver handles large problems efficiently.

4. **Wrong constraint direction**: `:(R[t] >= 0)` means "``R`` must be at least 0" (a lower bound). `:(b[t] <= 1.0)` means "debt cannot exceed ``M``" (a borrowing limit). `:(D[t] <= D_max)` means "``D`` must be at most `D_max`" (an upper bound). Verify that the direction matches the economic interpretation.

5. **Constraint binding at terminal period**: If `regime_history` shows the constraint binding in the last period, the horizon is too short. OccBin warns when this occurs --- increase `nperiods` until the constraint releases before the terminal period.

---

## References

- Guerrieri, L., & Iacoviello, M. (2015). OccBin: A Toolkit for Solving Dynamic Models with Occasionally Binding Constraints Easily. *Journal of Monetary Economics*, 70, 22--38. [DOI](https://doi.org/10.1016/j.jmoneco.2014.08.005)

- Ferris, M. C., & Munson, T. S. (1999). Interfaces to PATH 3.0: Design, Implementation and Usage. *Computational Optimization and Applications*, 12(1), 207--227. [DOI](https://doi.org/10.1023/A:1018636318047)

- Pal, A., et al. (2024). NonlinearSolve.jl: High-Performance and Robust Solvers for Systems of Nonlinear Equations in Julia. [GitHub](https://github.com/SciML/NonlinearSolve.jl)

- Johnson, S. G. (2007). The NLopt Nonlinear-Optimization Package. [GitHub](https://github.com/stevengj/nlopt)

- Rendahl, P. (2017). Linear Time Iteration. Unpublished manuscript.

- Sims, C. A. (2002). Solving Linear Rational Expectations Models. *Computational Economics*, 20(1), 1--20. [DOI](https://doi.org/10.1023/A:1020517101123)
