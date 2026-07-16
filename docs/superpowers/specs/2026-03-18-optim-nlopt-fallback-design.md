# Design: Optim.jl + NLopt.jl Fallback for Constrained DSGE

**Issue:** #91
**Date:** 2026-03-18
**Status:** Approved (rev 3 — final)

## Problem

Constrained DSGE solving requires JuMP + Ipopt (EPL-2.0), which is incompatible with GPL-3.0 for combined binary distribution. The Friedman-cli precompiled releases cannot bundle Ipopt, so constrained DSGE features are unavailable out of the box.

## Solution

Add Optim.jl (already a dep) and NLopt.jl (LGPL, GPL-compatible) as regular dependencies to handle constrained DSGE solving without JuMP/Ipopt.

### Solver Tiers

| Tier | Backend | Handles | When Used |
|------|---------|---------|-----------|
| 1 | NonlinearSolve.jl | Unconstrained, simple box | Default (unchanged) |
| 2 | Projected Newton | Box-constrained PF | Escalation when PF bounds violated |
| 3 | Optim.jl `Fminbox(LBFGS())` | Box-constrained SS | Escalation when SS bounds violated |
| 4 | NLopt.jl `LD_SLSQP` | Nonlinear inequality constraints (SS + PF) | Default for NonlinearConstraint |
| 5 | JuMP+Ipopt / JuMP+PATH | Full NLP / MCP | Explicit request, or auto when JuMP loaded + NonlinearConstraint |

### Escalation Chains

**Perfect Foresight (box constraints):**
1. NonlinearSolve solves unconstrained
2. If bounds violated → `_projected_newton_pf` (always available, preserves sparse Jacobian)
3. Done — no JuMP fallback needed

**Steady State (box constraints):**
1. NonlinearSolve with `lb`/`ub`
2. If bounds violated → `_optim_steady_state` with `Fminbox(LBFGS())`

**Nonlinear inequality constraints (SS or PF):**
1. `_nlopt_*` directly — no escalation. On failure, hard error with suggestion to use `solver=:ipopt`.

Previously: bounds violated → try PATH → try Ipopt → warn and return unconstrained.

## Scope

### In Scope

- Box-constrained steady state via Optim.jl
- Box-constrained perfect foresight via projected Newton
- Nonlinear-inequality-constrained steady state via NLopt.jl
- Nonlinear-inequality-constrained perfect foresight via NLopt.jl
- Updated `_select_solver` auto-detection
- Updated PF/SS escalation logic
- Updated error messages and validation checks
- Tests for all new paths

### Out of Scope

- OccBin (separate piecewise-linear solver, no JuMP dependency)
- Changes to JuMP/Ipopt/PATH extensions (they remain as-is for explicit use)
- API signature changes

## File Changes

| File | Change |
|------|--------|
| `Project.toml` | Add `NLopt`, `ForwardDiff` to `[deps]`; add `NLopt = "1"`, `ForwardDiff = "0.10"` to `[compat]` |
| `src/MacroEconometricModels.jl` | Add `import NLopt` and `import ForwardDiff` |
| `src/dsge/constraints.jl` | `_select_solver`: NonlinearConstraint → `:nlopt`; update error messages; add `:optim`/`:nlopt` to validation checks; add vector-adapter wrappers |
| `src/dsge/steady_state.jl` | Add `_optim_steady_state`, `_nlopt_steady_state`; wire dispatch; add SS escalation |
| `src/dsge/perfect_foresight.jl` | Add `_projected_newton_pf`, `_nlopt_perfect_foresight`; update escalation |
| `test/dsge/test_dsge.jl` | ~9 new tests |

No new files. No new extensions.

## New Functions

### `_optim_steady_state(spec, lower, upper; initial_guess, algorithm)`

- Minimizes sum-of-squared-residuals via `Optim.optimize` with `Fminbox(LBFGS())`
- Box constraints from `_extract_bounds`
- Optim.jl handles gradients internally via `autodiff = :forward`
- **Adapter:** `_build_ss_objective` returns a splatted-scalar closure `f(args::Real...)` (designed for JuMP operators). Wrap with `x::AbstractVector -> f(x...)` for Optim.jl's vector interface.

### `_nlopt_steady_state(spec, constraints; initial_guess, algorithm)`

- Minimizes sum-of-squared-residuals via NLopt `LD_SLSQP`
- Box constraints via `NLopt.lower_bounds!` / `NLopt.upper_bounds!`
- Nonlinear inequality constraints via `NLopt.inequality_constraint!`
- **NLopt callback signature:** `f(x::Vector, grad::Vector)` — compute gradient in-place via `ForwardDiff.gradient!(grad, obj, x)` when `length(grad) > 0`
- **Adapter:** Same vector-to-splat wrapper for `_build_ss_objective` and `_build_ss_nlcon`

### `_projected_newton_pf(spec, T_periods, shocks, lower, upper; max_iter, tol)`

- **Projected Newton with backtracking** for box-constrained PF:
  1. Compute Newton step using the existing block-tridiagonal sparse Jacobian (`_pf_jacobian`)
  2. **Backtracking Armijo line search** on merit function `||F(x)||^2`: find step size `α ∈ (0, 1]` such that `||F(x + α·d)||^2 ≤ ||F(x)||^2 + c·α·∇||F||²·d` (c = 1e-4, halving `α`)
  3. Step: `x_new = x + α·d` where `d = -J \ F(x)`
  4. Clamp to bounds: `x_new = clamp.(x_new, lower_stacked, upper_stacked)`
  5. Iterate until `||F(x)|| < tol`
- **Bound stacking:** `lower_stacked = repeat(lower, T_periods)`, `upper_stacked = repeat(upper, T_periods)` where `lower`/`upper` are `n`-length vectors from `_extract_bounds`
- Preserves the O(T·n) sparsity structure — no dense LBFGS approximation
- Same convergence tolerance and iteration limits as existing NonlinearSolve PF
- Falls back gracefully: if projected Newton doesn't converge, throw with suggestion to use `solver=:ipopt`

### `_nlopt_perfect_foresight(spec, T_periods, shocks, constraints; algorithm)`

- **Formulation:** NLopt solves a constrained feasibility problem:
  - **Objective:** constant 0 (or minimal regularization)
  - **Equality constraints** for model equations: `NLopt.equality_constraint!` for each `f_i(y_t, y_{t-1}, y_{t+1}, ε_t) = 0` at each period
  - **Inequality constraints** for `NonlinearConstraint`: `NLopt.inequality_constraint!`
  - **Box bounds** for `VariableBound`: `NLopt.lower_bounds!` / `NLopt.upper_bounds!`
- **Gradient mapping for PF constraints:** Each PF equality constraint `f_i(y_t, y_{t-1}, y_{t+1}, ε_t)` depends on a `3n + n_ε` slice of the stacked `T*n` decision vector. The NLopt callback must: (a) extract the relevant slice indices `[t*n..(t+1)*n, (t-1)*n..t*n, (t+1)*n..(t+2)*n]` for period `t`, (b) compute the local gradient via ForwardDiff on the slice, (c) scatter the local gradient into the full `T*n`-length grad vector (all other entries zero). Build a `_pf_nlopt_wrap(f, t, n, n_ε, T_periods)` adapter that handles this index mapping.
- `@warn` when T×n > 1000 suggesting `solver=:ipopt` for better performance (NLopt SLSQP is dense — no sparsity exploitation)
- Reuses `_build_pf_equation`, `_build_pf_nlcon` from `constraints.jl` (with PF-specific adapters)

## Adapter Pattern

The existing helpers in `constraints.jl` (`_build_ss_objective`, `_build_ss_nlcon`, `_build_pf_equation`, `_build_pf_nlcon`) return closures with signature `f(args::Real...)` — splatted scalars designed for JuMP's `add_nonlinear_operator`. Optim.jl and NLopt expect `f(x::AbstractVector)`.

Add thin adapter wrappers in `constraints.jl`:

```julia
# SS adapters — decision variable x has length n, maps directly to closure args
_vec_wrap(f) = x -> f(x...)                          # for Optim objective
_nlopt_wrap(f) = (x, grad) -> begin                   # for NLopt SS objective/constraint
    if length(grad) > 0
        ForwardDiff.gradient!(grad, z -> f(z...), x)
    end
    return f(x...)
end

# PF adapter — decision variable x has length T*n, each constraint touches a 3n+n_ε slice
_pf_nlopt_wrap(f, t, n, n_ε, T_periods) = (x, grad) -> begin
    # Extract slice indices for period t: y_t, y_{t-1}, y_{t+1}, ε_t
    # Build local_args from x[relevant_indices]
    # Compute local gradient via ForwardDiff on the slice
    # Scatter into full-length grad vector (zero elsewhere)
    ...  # implementation detail
end
```

## Dispatch Logic

### `_select_solver` (updated)

```
NonlinearConstraint present:
  If JuMP+Ipopt loaded → :ipopt  (preserve existing behavior for power users)
  Else                 → :nlopt  (new fallback)
Only VariableBound → :nonlinearsolve  (unchanged)
Valid values: :nonlinearsolve, :optim, :nlopt, :ipopt, :path
```

### `compute_steady_state` (new branches + escalation)

```
solver == :optim         → _optim_steady_state(spec, lower, upper; ...)
solver == :nlopt         → _nlopt_steady_state(spec, constraints; ...)
solver == :nonlinearsolve → _nonlinearsolve_steady_state(spec, lower, upper; ...)
                            if bounds violated in result → escalate to _optim_steady_state
solver == :ipopt         → _check_jump_loaded(); _jump_compute_steady_state(...)  [unchanged]
solver == :path          → _check_jump_loaded(); _path_compute_steady_state(...)  [unchanged]
```

### `perfect_foresight` (updated escalation + new branches)

```
solver == :nonlinearsolve → _nonlinearsolve_perfect_foresight(...)
                            if bounds violated → _projected_newton_pf(...)  [always available]
solver == :nlopt          → _nlopt_perfect_foresight(...)
solver == :ipopt          → _check_jump_loaded(); _jump_perfect_foresight(...)  [unchanged]
solver == :path           → _check_jump_loaded(); _path_perfect_foresight(...)  [unchanged]
```

### `solver` kwarg valid values

`:nonlinearsolve`, `:optim`, `:nlopt`, `:ipopt`, `:path`

`algorithm` kwarg passes through to the chosen backend.

## Error Handling

- NLopt non-convergence: throw with return code, suggest `solver=:ipopt`. Also check final objective value ≈ 0 for SS (sum-of-squared-residuals < 1e-10); if not, throw "NLopt converged to a non-zero residual — steady state not found"
- Optim non-convergence: throw with Optim result, suggest `solver=:ipopt`
- Projected Newton non-convergence: throw, suggest `solver=:ipopt`
- Large PF with NLopt (T×n > 1000): `@warn` suggesting `solver=:ipopt`
- `solver=:ipopt`/`:path` without JuMP: existing `_JUMP_INSTALL_MSG` error (unchanged)
- PF escalation warning ("JuMP/Ipopt not loaded"): removed — projected Newton always catches it
- Update existing error messages that say "Use solver=:ipopt" to say "Use solver=:nlopt (default) or solver=:ipopt"

## Algorithm Defaults

| Backend | Default Algorithm | Override via `algorithm` |
|---------|-------------------|------------------------|
| Optim.jl | `Fminbox(LBFGS())` | Any Optim algorithm |
| NLopt.jl | `LD_SLSQP` | Any NLopt algorithm symbol |
| Projected Newton | N/A (fixed algorithm) | Not overridable |

## Tests

1. Box-constrained SS via `solver=:optim` — verify bound satisfied
2. Box-constrained SS via `solver=:nlopt` — verify works for box-only
3. Nonlinear-constrained SS via `solver=:nlopt` — verify both constraint types satisfied
4. Box-constrained PF via projected Newton — verify bounds enforced, convergence
5. PF escalation — NonlinearSolve → projected Newton (not JuMP)
6. Nonlinear-constrained PF via `solver=:nlopt` — verify constraints in path
7. NLopt PF with mixed constraints (box bounds + nonlinear inequality) — verify both respected
8. Explicit `solver=:ipopt`/`:path` errors correctly when JuMP not loaded
9. SS escalation — NonlinearSolve → Optim.jl when bounds violated

## License

- Optim.jl: MIT (already a dependency)
- NLopt.jl wrapper: MIT
- NLopt C library: LGPL
- ForwardDiff.jl: MIT
- All compatible with GPL-3.0
