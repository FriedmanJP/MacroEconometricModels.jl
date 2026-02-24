# PATH MCP Solver for DSGE — Design Document

**Issue:** #48 — DSGE: add nonlinear solvers (item 4: Nonlinear Constraint Solver, PATH backend)

**Goal:** Add a PATH-based MCP (Mixed Complementarity Problem) solver for constrained steady-state and perfect-foresight problems where variable bounds actually bind — the case where Ipopt's NLP formulation returns infeasible.

**References:** Ferris & Munson (1999), Guerrieri & Iacoviello (2015), Judd (1998)

---

## Why PATH? — Ipopt vs PATH

The existing Ipopt solver poses constrained PF as `min 0 s.t. f(x)=0, bounds`. When bounds bind on a fully-determined system, the equality constraints conflict with the bounds → **infeasible**.

PATH solves **complementarity problems**: for each variable/equation pair, either the equation holds OR the variable is at its bound. This is exactly what "occasionally binding constraints" means:

| Scenario | Ipopt (NLP) | PATH (MCP) |
|---|---|---|
| Non-binding bounds | Works | Works |
| **Binding bounds (ZLB, positivity)** | **Infeasible** | **Correct** |
| General nonlinear inequalities `g(x) ≤ 0` | Works | Cannot handle |
| Collateral/debt constraints (complementarity) | Cannot express | Natural formulation |

PATH complements Ipopt — it does not replace it.

---

## Architecture

**Approach:** Add PATHSolver as a third weak dependency. Auto-detect solver based on constraint types:

- **VariableBounds only** → PATH (MCP)
- **NonlinearConstraints present** → Ipopt (NLP)
- User can override: `solver=:path` or `solver=:ipopt`

**Extension trigger:** The existing `MacroEconometricModelsJuMPExt` already requires JuMP + Ipopt. We add PATHSolver as an optional import within the extension — if PATHSolver is loaded, PATH-based methods become available.

**Dependency:** PATHSolver.jl (MIT license, GPL-compatible). **Important:** PATH has a 300-variable / 2000-nonzero limit without a commercial license. For typical DSGE models (5-50 variables), this is not a constraint. Document the limit.

---

## MCP Formulation

### Complementarity Condition

For each variable `i` with bounds `[l_i, u_i]` and residual function `F_i(x)`:

- If `l_i < x_i < u_i`: `F_i(x) = 0` (interior → equation holds)
- If `x_i = l_i`: `F_i(x) ≥ 0` (at lower bound → slack is non-negative)
- If `x_i = u_i`: `F_i(x) ≤ 0` (at upper bound → slack is non-positive)

Unbounded variables (no `VariableBound`) get `[-Inf, +Inf]` → their equations must hold exactly.

### Variable-Equation Pairing

Variable `i` pairs with equation `i` (standard convention). The `@dsge` macro produces equations in natural order matching variables.

### Steady State MCP

Find `y_ss` such that `residual_i(y_ss, y_ss, y_ss, 0, θ) ⟂ y_ss[i]` for all `i`, with bounds from `VariableBound` constraints.

### Perfect Foresight MCP

Stacked system over `t = 1, ..., T`:

```
residual_i(y_t, y_{t-1}, y_{t+1}, ε_t, θ) ⟂ y[i, t]    ∀ i, t
```

Boundary conditions: `y[0] = y[T+1] = y_ss` (fixed).

Total variables: `n × T`. Total complementarity conditions: `n × T`.

---

## API

No new API surface. The existing `constraints` kwarg auto-dispatches:

```julia
# Ipopt (NLP) — when NonlinearConstraints present
ss = compute_steady_state(spec, θ; constraints=[
    variable_bound(:i, lower=0.0),
    nonlinear_constraint((y,...) -> y[3] - 0.8*y[1]; label="collateral")
])

# PATH (MCP) — when only VariableBounds
ss = compute_steady_state(spec, θ; constraints=[
    variable_bound(:i, lower=0.0)
])

# User override
ss = compute_steady_state(spec, θ; constraints=[...], solver=:ipopt)
path = perfect_foresight(spec, θ, shocks, T; constraints=[...], solver=:path)
```

---

## File Changes

**Modified files:**
- `ext/MacroEconometricModelsJuMPExt.jl` — add PATH-based MCP solvers, auto-detection logic
- `src/dsge/constraints.jl` — add `_path_compute_steady_state`, `_path_perfect_foresight` stubs; update `_check_jump_loaded` for PATH
- `src/dsge/steady_state.jl` — add `solver` kwarg, dispatch logic
- `src/dsge/perfect_foresight.jl` — add `solver` kwarg, dispatch logic
- `Project.toml` — add PATHSolver to `[weakdeps]`, `[compat]`, `[extras]`, `[targets]`
- `test/dsge/test_dsge.jl` — MCP tests

**No new files.** PATH solvers go in the existing JuMP extension.

---

## JuMP Extension Internals

### MCP Steady State

```julia
function _path_compute_steady_state(spec, constraints; initial_guess=nothing)
    model = JuMP.Model(PATHSolver.Optimizer)
    n = spec.n_endog

    # Variables with bounds from constraints (default [-Inf, Inf])
    lower, upper = _extract_bounds(spec, constraints)
    @variable(model, lower[i] <= x[i=1:n] <= upper[i])

    # Set initial guess
    for i in 1:n
        JuMP.set_start_value(x[i], ss_guess[i])
    end

    # Complementarity: residual_i(x, x, x, 0, θ) ⟂ x[i]
    for i in 1:n
        op = add_nonlinear_operator(model, n, _build_ss_residual(spec, i); name=Symbol(:ss_eq_, i))
        @constraint(model, op(x...) ⟂ x[i])
    end

    optimize!(model)
    return value.(x)
end
```

### MCP Perfect Foresight

```julia
function _path_perfect_foresight(spec, T_periods, shocks, constraints)
    model = JuMP.Model(PATHSolver.Optimizer)
    n, n_ε = spec.n_endog, spec.n_exog

    lower, upper = _extract_bounds(spec, constraints)

    # Stacked variables: x[i, t] for t = 1, ..., T
    # Boundary periods fixed to SS via residual construction
    @variable(model, lower[i] <= x[i=1:n, t=1:T_periods] <= upper[i])

    # Set initial guess to SS
    for t in 1:T_periods, i in 1:n
        JuMP.set_start_value(x[i, t], y_ss[i])
    end

    # Complementarity per (equation, variable, period)
    for t in 1:T_periods, i in 1:n
        # Build operator for equation i at period t
        # Inputs: current-period variables x[:, t]
        # y_lag = x[:, t-1] (or y_ss if t=1)
        # y_lead = x[:, t+1] (or y_ss if t=T)
        op = add_nonlinear_operator(model, n_args, wrapper; name=Symbol(:pf_eq_, i, :_, t))
        @constraint(model, op(args...) ⟂ x[i, t])
    end

    optimize!(model)
end
```

### Helper: Extract Bounds

```julia
function _extract_bounds(spec, constraints)
    T = eltype(spec.steady_state)
    lower = fill(T(-Inf), spec.n_endog)
    upper = fill(T(Inf), spec.n_endog)
    for c in constraints
        if c isa VariableBound
            idx = findfirst(==(c.var_name), spec.endog)
            c.lower !== nothing && (lower[idx] = T(c.lower))
            c.upper !== nothing && (upper[idx] = T(c.upper))
        end
    end
    return lower, upper
end
```

### Auto-Detection Logic

```julia
function _select_solver(constraints, solver_override)
    solver_override !== nothing && return solver_override
    has_nlcon = any(c -> c isa NonlinearConstraint, constraints)
    has_nlcon && return :ipopt
    _path_available() ? :path : :ipopt
end

function _path_available()
    try
        @isdefined(PATHSolver) && return true
    catch
    end
    return false
end
```

---

## Error Handling

- **PATH not loaded**: If auto-detection selects PATH but PATHSolver isn't loaded, fall back to Ipopt with a warning that binding bounds may cause infeasibility
- **PATH size limit**: If `n × T > 300`, warn about the free-tier limit
- **Infeasible**: Return `converged=false`, store best iterate
- **NonlinearConstraint + solver=:path**: Error — PATH cannot handle general nonlinear inequalities

---

## Testing Plan (~12 tests)

### MCP Steady State (~4 tests)
1. Non-binding bounds → same as unconstrained SS
2. **Binding lower bound** → variable at bound, residual ≥ 0
3. Binding upper bound → variable at bound, residual ≤ 0
4. Multiple variables, mixed binding/non-binding

### MCP Perfect Foresight (~6 tests)
1. Non-binding bounds → matches unconstrained Newton path
2. **ZLB binding** → AR(1) + large negative shock + `lower=0.0` → y clamped at 0 (the key test)
3. Terminal convergence to SS
4. Path dimensions correct
5. Upper bound binding test
6. Solver status on difficult problems

### Auto-Detection (~2 tests)
1. VariableBounds only → uses PATH (when available)
2. NonlinearConstraints present → uses Ipopt

Tests guarded by `_path_available` flag (same pattern as JuMP availability check).

---

## Dependencies

**New weak dependency:**
- PATHSolver.jl (MIT license)

**Extension trigger change:**
- Current: `MacroEconometricModelsJuMPExt = ["JuMP", "Ipopt"]`
- After: Keep same extension, conditionally import PATHSolver inside it

OR add a second extension:
- `MacroEconometricModelsPATHExt = ["JuMP", "PATHSolver"]`

Recommendation: **Single extension** with conditional PATHSolver import. Simpler, avoids code duplication (shared helpers like `_extract_bounds`, `_build_pf_equation`).
