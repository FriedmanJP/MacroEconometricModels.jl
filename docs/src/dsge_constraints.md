# [Constraints and Occasionally Binding Models](@id dsge_constraints)

Standard linearized DSGE models assume all equilibrium conditions hold with equality at all times. Occasionally binding constraints --- such as the zero lower bound on nominal interest rates, borrowing limits, or irreversible investment --- require specialized solution methods. This page covers three approaches: deterministic perfect foresight with built-in constrained solvers (Optim.jl, NLopt.jl, JuMP+Ipopt), the optional PATH complementarity backend, and the piecewise-linear OccBin algorithm (Guerrieri & Iacoviello 2015). For model specification and linearization, see [DSGE Models](@ref dsge_page). For first-order solvers, see [Linear Solvers](@ref dsge_linear).


```@setup dsge_constraints
using MacroEconometricModels, Random, LinearAlgebra, Statistics
Random.seed!(42)
```

## Quick Start

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

The three-standard-deviation demand shock knocks output 8.4% below steady state on impact, inflation 1.85% below, and the nominal rate 3.82% below. Newton converges in a single iteration because this specification is already linear in its variables, so the stacked residual is an affine system and one step solves it exactly. The Taylor rule does the work: the policy rate falls by more than inflation, delivering the real-rate cut that pulls output back toward steady state over the following quarters.

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

The regime sequence converges in two guess-and-verify passes, and the constraint binds in all 60 periods. That is not a horizon that is too short — see the discussion under [Occasionally Binding Constraints (OccBin)](@ref) — but a property of the calibration: with ``\beta R = 1`` the asset position follows a random walk, so once a large enough shock pushes the household to the borrowing limit it never accumulates its way back off. OccBin emits two warnings on this model, one about the terminal period and one about the defining-equation heuristic; both are expected here and are explained below.

```julia
plot_result(occ_sol)
```

```@raw html
<iframe src="../assets/plots/occbin_solution.html" width="100%" height="460" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

**Recipe 3: OccBin IRFs --- linear vs constrained**

```@example dsge_constraints
occ_irf = occbin_irf(borrow_spec, constraint, 1, 40; magnitude=-40.0)
report(occ_irf)
```

```julia
plot_result(occ_irf)
```

```@raw html
<iframe src="../assets/plots/occbin_irf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The constrained and unconstrained paths diverge by up to 12.14 over the 40-period horizon — an enormous gap, because the unconstrained household would borrow more than thirteen times the limit to smooth a shock this large. `magnitude` must be big enough to push the model into the binding regime: at ``-3.0`` standard deviations the desired asset position peaks just under the bound and the two paths coincide exactly.

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
(path_size = size(pf.path),              # T x n matrix of variable levels
 deviations_size = size(pf.deviations),  # T x n deviations from steady state
 converged = pf.converged,
 iterations = pf.iterations,
 impact = round.(pf.path[1, :]; digits=6))
```

Both routes return the identical path — `solve(spec; method=:perfect_foresight, …)` forwards straight to `perfect_foresight`. Levels and deviations coincide here because this model's steady state is the origin; in a model with a non-zero steady state `path` is the series a user plots and `deviations` the one that feeds impulse-response arithmetic.

!!! note "Technical Note"
    The solver exploits the block-tridiagonal structure of the Jacobian via sparse LU factorization. Each Newton step solves ``J \Delta x = -F(x)`` where ``J`` is ``nT \times nT`` but has only ``3n^2 T`` non-zeros (vs ``n^2 T^2`` for dense). Numerical Jacobians use central differences with adaptive step sizes. The `algorithm` keyword accepts any NonlinearSolve.jl algorithm (e.g., `NonlinearSolve.TrustRegion()`).

### Keywords

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `T_periods` | `Int` | `100` | Number of simulation periods |
| `shock_path` | `Union{Nothing, Matrix}` | `nothing` | ``T \times n_\varepsilon`` shock realizations (zeros if omitted) |
| `max_iter` | `Int` | `100` | Newton iteration limit |
| `tol` | `Real` | `default_abstol(Float64)` = ``10^{-8}`` | Convergence tolerance (max absolute residual) |
| `abstol` | `Real` | `tol` | Absolute residual gate passed to the backend; overrides `tol` for the convergence test |
| `constraints` | `Vector` | `DSGEConstraint[]` | Variable bounds and nonlinear constraints |
| `solver` | `Union{Nothing, Symbol}` | `nothing` | `:nonlinearsolve`, `:nlopt`, `:ipopt`, or `:path`; auto-detected when `nothing` |
| `algorithm` | `Any` | `nothing` → `NewtonRaphson()` | Algorithm for the chosen backend (e.g. `NonlinearSolve.TrustRegion()`) |

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

When variable bounds or nonlinear inequality constraints are present, the solver uses a five-tier hierarchy. Tiers 1--5 require no additional packages (JuMP + Ipopt are built in); the PATH MCP solver is an optional add-on.

| Tier | Backend | Handles | Activation |
|------|---------|---------|------------|
| 1 | NonlinearSolve.jl | Unconstrained, non-binding box | Default |
| 2 | Projected Newton | Box-constrained PF | Auto-escalation when bounds violated |
| 3 | Optim.jl `Fminbox(LBFGS())` | Box-constrained SS | Auto-escalation when SS bounds violated |
| 4 | JuMP + Ipopt | Nonlinear inequality constraints | Auto-selected for `NonlinearConstraint` |
| 5 | NLopt.jl `LD_SLSQP` / JuMP + PATH | Dense NLP / MCP | Explicit `solver=:nlopt` or `:path` |

The backend is auto-detected from the constraint types. Pure `VariableBound` constraints start with NonlinearSolve and escalate to projected Newton (perfect foresight) or Optim.jl (steady state) if the bounds are violated. Any `NonlinearConstraint` in the list selects Ipopt, because JuMP and Ipopt are hard dependencies and are always available; NLopt is reached only by asking for it explicitly. Override any of this with the `solver` keyword.

### Box Constraints (Built-in)

Box constraints require no additional packages. The solver solves the unconstrained Newton system first; if any variable violates its bounds it escalates to a projected Newton method that preserves the sparse block-tridiagonal Jacobian structure:

```julia
# The model above is written in deviations, so the ZLB sits at minus the
# steady-state rate, not at zero.
zlb = variable_bound(:R, lower=-0.005)
pf = perfect_foresight(spec; shock_path=shocks, constraints=[zlb])
```

The projected Newton solver uses NCP (nonlinear complementarity problem) convergence criteria: at interior points the residual must equal zero, at a binding lower bound the residual must be non-negative, and at a binding upper bound the residual must be non-positive.

!!! warning "Get the bound right, and check that it converged"
    Two failures are easy to hit here. First, a bound stated in levels is wrong for a model written in deviations: this page's New Keynesian block has an all-zero steady state, so `lower=0.0` on `R` says "the policy rate may never fall at all" rather than "the policy rate may not go below zero". Second, projected Newton is not guaranteed to converge — on this model it exhausts its 100 iterations for every bound that actually binds and throws, with a message pointing at `solver=:ipopt`. Treat that escalation as the supported route for a genuinely binding box constraint on a perfect-foresight path, and use OccBin when the constraint is occasionally binding rather than always.

### Nonlinear Constraints

For general nonlinear inequality constraints the auto-selected backend is Ipopt. NLopt's `LD_SLSQP` remains available through `solver=:nlopt`. Neither needs an extra install:

```julia
# Variable bound + nonlinear constraint
zlb = variable_bound(:R, lower=-0.005)
debt_limit = nonlinear_constraint(
    (y, y_lag, y_lead, e, theta) -> y[debt_idx] / y[gdp_idx] - 0.6;
    label="Debt-to-GDP <= 60%"
)

pf = perfect_foresight(spec; shock_path=shocks,
                        constraints=[zlb, debt_limit])
```

The `nonlinear_constraint` function takes a closure with the standard residual signature `(y, y_lag, y_lead, e, theta) -> scalar`, and the constraint is satisfied when the return value is ``\leq 0``. The closure is validated once at setup by calling it on a dummy argument, so a signature mistake surfaces immediately rather than inside the optimizer. For perfect foresight the problem is posed as a feasibility problem with the model equations as equality constraints and the user's closures as inequalities.

If you route this to NLopt with `solver=:nlopt`, note that SLSQP is a dense algorithm: on problems where ``T \times n`` exceeds 1000 it warns and scales badly, and `solver=:ipopt` is the better choice.

### Advanced: Ipopt and PATH Backends

For large-scale problems or complementarity formulations, JuMP-based backends provide additional power. JuMP + Ipopt are built-in dependencies; the PATH solver is an optional weak dependency (a proprietary binary):

**Ipopt** (Interior Point Optimizer) handles general NLP problems. It is more robust than NLopt for large systems and needs no extra install:

```julia
pf = perfect_foresight(spec; shock_path=shocks,
                        constraints=[zlb, debt_limit], solver=:ipopt)
```

**PATH** solves the problem as a Mixed Complementarity Problem (MCP). For each variable ``i`` with bounds ``[l_i, u_i]``:

```math
l_i \leq y_i \leq u_i, \quad f_i(y) \begin{cases} \geq 0 & \text{if } y_i = l_i \\ = 0 & \text{if } l_i < y_i < u_i \\ \leq 0 & \text{if } y_i = u_i \end{cases}
```

This complementarity structure is natural for problems where a constraint replaces an equilibrium condition when binding (e.g., the Taylor rule is replaced by ``R_t = 0`` at the ZLB).

```julia
import PATHSolver

pf = perfect_foresight(spec; shock_path=shocks,
                        constraints=[zlb], solver=:path)
```

Everything except PATH ships with the package. `:path` requires the PATHSolver weak dependency, a proprietary binary, and is worth the install only when the complementarity formulation is the natural one — a ZLB, where the Taylor rule genuinely *is* replaced by ``R_t = 0`` rather than merely bounded.

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

The solution carries both the unconstrained linear path and the piecewise-linear constrained path, as deviations from the steady state:

```@example dsge_constraints
(linear_impact = round.(occ_sol.linear_path[1, :]; digits=6),
 piecewise_impact = round.(occ_sol.piecewise_path[1, :]; digits=6),
 max_b_linear = round(maximum(occ_sol.linear_path[:, 1]); digits=4),
 max_b_piecewise = round(maximum(occ_sol.piecewise_path[:, 1]); digits=4),
 binding_periods = sum(occ_sol.regime_history .> 0))
```

The variables are ordered ``(b, c, y)``. On impact the unconstrained household would borrow 1.333 and let consumption fall only 0.667; the constraint caps borrowing at exactly 1.0, and consumption absorbs the difference, falling 1.0 instead. Over the full path the gap widens dramatically — the unconstrained household would run its debt up to 13.31, thirteen times the limit — and constrained consumption bottoms out at ``-1.85`` against ``-0.667`` unconstrained. When the constraint binds, ``b_t = M`` replaces the savings optimality condition while the budget constraint continues to hold, so consumption is what gives way.

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
| `curb_retrench` | `Bool` | `false` | Two-constraint solves only: relax at most one period per iteration |

### When the Constraint Never Releases

OccBin warns when the constraint is still binding in the terminal period, because the algorithm's terminal condition assumes the economy has returned to the unconstrained regime by then. Usually that warning means the horizon is too short and `nperiods` should be raised. On the borrowing model above it means something different, and no horizon fixes it: the calibration sets ``\beta R = 1`` exactly, which gives the asset position a unit root. The unconstrained path converges to ``b = 13.33`` and never comes back down, so once a shock large enough to hit the limit arrives, the constraint binds for every remaining period at any horizon — 40, 60, 100 or 150 all bind throughout.

The path is still the correct piecewise-linear solution conditional on that regime sequence; what is missing is the return to steady state that the terminal condition presumes. Read it as the "constraint binds throughout" branch rather than as a completed transition. Shrinking the shock does not produce a partially-binding example either: at ``-3.0`` standard deviations desired borrowing peaks just below the limit and the constraint never binds at all. A model whose constrained variable is genuinely mean-reverting — ``\beta R < 1`` — is the one that gives the textbook bind-then-release pattern.

### Two-Constraint Example

OccBin supports two simultaneous constraints. The algorithm generalizes to four regimes: neither binding, only constraint 1 binding, only constraint 2 binding, and both binding.

```julia
zlb = parse_constraint(:(R[t] >= 0), spec)
borrow = parse_constraint(:(D[t] <= D_max), spec)
occ_sol = occbin_solve(spec, zlb, borrow; shock_path=shocks)
```

The `regime_history` matrix has two columns --- one per constraint --- recording which regimes are active in each period. An optional `curb_retrench=true` keyword limits constraint relaxation to one period per iteration, which helps prevent oscillation in difficult two-constraint problems.

!!! note "Defining-equation assignment"
    Each constrained variable's binding regime replaces its *defining equation*, picked heuristically as the equation whose Jacobian column is most sensitive to that variable. When the runner-up is within 90% of the winner the pick is not decisive and OccBin warns — as it does on the borrowing model above, where the savings rule and the budget constraint are equally sensitive to ``b``. To override the heuristic for a single constraint, build the binding-regime specification yourself and pass it positionally: `occbin_solve(spec, constraint, alt_spec; …)`. For two constraints that map to the same defining equation the solver cannot separate them and raises an `ArgumentError`; supply all three binding regimes through the `Dict` overload, keyed by regime tuple, `Dict((1,0) => alt1, (0,1) => alt2, (1,1) => alt12)`.

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

Because `OccBinSolution` stores the constraints it was solved with, an existing solution can be turned into an IRF without re-specifying them:

```@example dsge_constraints
oirf2 = irf(occ_sol, 40; shock_idx=1, magnitude=-40.0)
(horizon = size(oirf2.piecewise, 1), binding = sum(oirf2.regime_history .> 0))
```

Both `occbin_irf` and this `irf` method accept `maxiter` (default `100`) to bound the regime iteration, and the two-constraint forms additionally accept `curb_retrench`.

### Return Values

**`OccBinSolution{T}`:**

| Field | Type | Description |
|-------|------|-------------|
| `linear_path` | `Matrix{T}` | ``T \times n`` unconstrained path (deviations from SS) |
| `piecewise_path` | `Matrix{T}` | ``T \times n`` piecewise-linear constrained path |
| `steady_state` | `Vector{T}` | Steady-state values |
| `regime_history` | `Matrix{Int}` | ``T \times n_c`` regime indicators (0 = slack, 1 and above = binding) |
| `converged` | `Bool` | Regime convergence flag |
| `iterations` | `Int` | Regime iterations used |
| `spec` | `DSGESpec{T}` | Back-reference to model specification |
| `varnames` | `Vector{String}` | Variable display labels |
| `constraints` | `Vector{OccBinConstraint{T}}` | The constraint(s) used in the solve |

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

(determinate = is_determined(sol_unc),
 max_gap = round(maximum(abs.(bc_irf.piecewise .- bc_irf.linear)); digits=4),
 binding = sum(bc_irf.regime_history .> 0))
```

```julia
plot_result(bc_irf)
```

```@raw html
<iframe src="../assets/plots/occbin_irf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The unconstrained model is determinate, and the two IRFs diverge by up to 12.14 across the 40-period horizon. The unconstrained agent smooths optimally, borrowing freely against future income and spreading the shock across many periods. Once borrowing hits ``b_t = M`` the savings optimality condition is replaced by the bound, the budget constraint alone determines consumption, and the full income shortfall lands on current consumption instead. The constraint binds in all 40 periods here, for the unit-root reason described above.

---

## Common Pitfalls

1. **A bound stated in the wrong units**: `variable_bound` and `parse_constraint` both take the bound in the same units as the model's variables. In a model written in deviations with an all-zero steady state, `lower=0.0` on a rate means "may never fall below steady state", not "may not go negative". Check the steady state before writing the bound.

2. **Projected Newton failing to converge**: a genuinely binding box constraint on a perfect-foresight path can exhaust the projected Newton iteration limit and throw rather than return. The error message points at `solver=:ipopt`, which is the supported escalation. If the constraint is *occasionally* binding rather than always, OccBin is the better tool.

3. **Non-convergence in perfect foresight**: increase `T_periods` or reduce the shock magnitude. The terminal condition assumes a return to steady state, so the horizon must be long enough for the economy to get back.

4. **OccBin regime cycling**: the guess-and-verify loop can cycle between regime sequences. OccBin detects a repeated pattern and stops with a warning and `converged=false` rather than spinning to `maxiter`. For one-constraint problems raise `maxiter`; for two-constraint problems set `curb_retrench=true` to limit relaxation to one period per iteration.

5. **Wrong constraint direction**: `:(R[t] >= 0)` is a lower bound, `:(b[t] <= 1.0)` an upper bound. Verify the direction matches the economics before reading the regime history.

6. **Constraint binding at the terminal period**: usually the horizon is too short and `nperiods` should be raised. But if the constrained variable has a unit root — ``\beta R = 1`` in a consumption-savings block, for instance — the constraint binds forever once triggered and no horizon removes the warning. Check whether the unconstrained path actually mean-reverts before chasing the warning with a longer simulation.

7. **Assuming the defining-equation heuristic got it right**: OccBin picks which equation the bound replaces by Jacobian sensitivity and warns when the choice is close. A wrong pick silently solves a different model. When warned, supply the binding-regime specification explicitly.

---

## References

- Guerrieri, L., & Iacoviello, M. (2015). OccBin: A Toolkit for Solving Dynamic Models with Occasionally Binding Constraints Easily. *Journal of Monetary Economics*, 70, 22--38. [DOI](https://doi.org/10.1016/j.jmoneco.2014.08.005)

- Ferris, M. C., & Munson, T. S. (1999). Interfaces to PATH 3.0: Design, Implementation and Usage. *Computational Optimization and Applications*, 12(1--3), 207--227. [DOI](https://doi.org/10.1023/A:1008636318275)

- Pal, A., et al. (2024). NonlinearSolve.jl: High-Performance and Robust Solvers for Systems of Nonlinear Equations in Julia. [GitHub](https://github.com/SciML/NonlinearSolve.jl)

- Johnson, S. G. (2007). The NLopt Nonlinear-Optimization Package. [GitHub](https://github.com/stevengj/nlopt)

- Sims, C. A. (2002). Solving Linear Rational Expectations Models. *Computational Economics*, 20(1--2), 1--20. [DOI](https://doi.org/10.1023/A:1020517101123)
