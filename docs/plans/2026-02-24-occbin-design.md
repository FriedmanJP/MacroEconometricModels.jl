# OccBin Solver Design

**Goal:** Implement the piecewise-linear OccBin method (Guerrieri & Iacoviello 2015) for DSGE models with occasionally binding constraints (one or two constraints).

**Architecture:** Guess-and-verify iteration over which periods constraints bind, using time-varying decision rules computed via backward iteration on regime-specific linearized coefficients. Integrates with the existing `@dsge` → `linearize` → `solve` pipeline.

**Reference:** Guerrieri, Luca, and Matteo Iacoviello. 2015. "OccBin: A Toolkit for Solving Dynamic Models with Occasionally Binding Constraints Easily." *Journal of Monetary Economics* 70: 22–38.

---

## 1. Types and Data Structures

### New types in `src/dsge/types.jl`:

**`OccBinConstraint{T<:AbstractFloat}`** — parsed constraint specification:
- `expr::Expr` — original Julia expression (e.g., `:(i[t] >= 0)`)
- `variable::Symbol` — constrained variable (`:i`)
- `bound::T` — bound value (`0.0`)
- `direction::Symbol` — `:geq` or `:leq`
- `bind_expr::Expr` — substitution when binding (e.g., `:(i[t] = 0)`)

**`OccBinRegime{T<:AbstractFloat}`** — linearized coefficients for one regime:
- `A::Matrix{T}` — lag coefficients (n x n), i.e. ∂f/∂y_{t-1}
- `B::Matrix{T}` — current coefficients (n x n), i.e. ∂f/∂y_t
- `C::Matrix{T}` — forward coefficients (n x n), i.e. ∂f/∂y_{t+1}
- `D::Matrix{T}` — shock coefficients (n x n_shocks)

**`OccBinSolution{T<:AbstractFloat}`** — result of `occbin_solve`:
- `linear_path::Matrix{T}` — unconstrained solution (T_periods x n)
- `piecewise_path::Matrix{T}` — constrained solution (T_periods x n)
- `steady_state::Vector{T}` — SS values
- `regime_history::Matrix{Int}` — regime per period per constraint (T_periods x n_constraints)
- `converged::Bool`
- `iterations::Int`
- `spec::DSGESpec{T}`
- `varnames::Vector{String}`

**`OccBinIRF{T<:AbstractFloat}`** — IRF comparison (linear vs piecewise):
- `linear::Matrix{T}` — unconstrained IRF (horizon x n)
- `piecewise::Matrix{T}` — constrained IRF (horizon x n)
- `regime_history::Matrix{Int}`
- `varnames::Vector{String}`
- `shock_name::String`

---

## 2. Core Algorithm

### File: `src/dsge/occbin.jl`

### Constraint parsing

`parse_constraint(expr::Expr, spec::DSGESpec{T})` → `OccBinConstraint{T}`
- Accepts `:(i[t] >= 0)` or `:(i[t] <= bound)` syntax
- Validates that the variable exists in `spec.endog`
- Extracts variable, bound, direction

### Regime derivation

`_derive_alternative_regime(spec::DSGESpec{T}, constraint::OccBinConstraint{T})` → `DSGESpec{T}`
- Auto-generates a modified spec where the constrained equation is replaced by `var[t] = bound`
- If user provides an explicit alternative `DSGESpec`, use that instead

### Linearized regime extraction

`_extract_regime(spec::DSGESpec{T})` → `OccBinRegime{T}`
- Calls `linearize(spec)` to get Gamma0, Gamma1, C, Psi, Pi
- Rearranges to (A, B, C, D) form:
  - A = coefficients on y_{t-1}
  - B = coefficients on y_t
  - C = coefficients on y_{t+1}
  - D = coefficients on shocks

### Regime mapping

`_map_regime(violvec::BitVector)` → `(regimes::Vector{Int}, starts::Vector{Int})`
- Identifies contiguous blocks of binding/non-binding periods

### Time-varying decision rules (backward iteration)

`_backward_iteration(regime_ref, regime_alt, P, Q, violvec, shocks)` → time-varying P_t, D_t, E

Algorithm from last binding period T_max backward:
1. At T_max: `P_T = -[B* + C*·P]⁻¹ · A*`
2. For t = T_max-1 down to 1: select regime coefficients based on `violvec[t]`, compute `P_t = -[B_t + C_t·P_{t+1}]⁻¹ · A_t`
3. Shock impact: `E = -[invmat]⁻¹ · D_t`

### Forward simulation

`_simulate_forward(P_tv, D_tv, E, P_lin, init, T_max, nperiods)` → path matrix

### Guess-and-verify loop

`_guess_verify_one(regime_ref, regime_alt, P, Q, constraint, shocks, nperiods; maxiter=100)` → (path, regime_history, converged, iterations)

### Two-constraint extension

For two constraints, 4 regime combinations: (0,0), (1,0), (0,1), (1,1). The backward iteration selects among 4 sets of coefficients per period based on `violvec[:, 1:2]`. Same guess-and-verify structure but with a 2D violation matrix and optional curb-retrench mode for stability.

### Public API

```julia
# One constraint
occbin_solve(spec, constraint; shock_path, nperiods=40, maxiter=100)
occbin_solve(spec, constraint, alt_spec; ...)  # explicit alternative regime

# Two constraints
occbin_solve(spec, constraint1, constraint2; ...)
occbin_solve(spec, c1, c2, alt_specs::Dict; ...)  # explicit alternatives

# IRF
occbin_irf(spec, constraint, shock_idx, horizon; magnitude=1.0)
```

---

## 3. Integration

### File placement

New `src/dsge/occbin.jl`, included after `analytical.jl` in `MacroEconometricModels.jl`.

### Exports

`occbin_solve`, `occbin_irf`, `parse_constraint`, `OccBinSolution`, `OccBinIRF`, `OccBinConstraint`

### Display

- `show(io, ::OccBinSolution)` — convergence status, regime summary (periods constrained), variable names
- `show(io, ::OccBinIRF)` — shock name and max deviation between linear/piecewise paths
- `report(::OccBinSolution)` dispatches to `show`

### Plotting

- `plot_result(::OccBinIRF)` — side-by-side linear vs piecewise IRFs per variable with shaded binding regions
- `plot_result(::OccBinSolution)` — piecewise path with regime shading

### References

Add Guerrieri & Iacoviello (2015) to `_REFERENCES` and `_TYPE_REFS`.

### Numerical robustness

- `robust_inv` for all `[B + C·P]⁻¹` inversions
- Warn if constraint still binding at terminal period (horizon too short)
- Warn if iteration hits maxiter without convergence
- Validate unconstrained model is stable (`is_stable(sol)`) before proceeding

---

## 4. Test Plan

Tests in `test/dsge/test_dsge.jl`:

1. **Constraint parsing** — `:(i[t] >= 0)`, `:(debt[t] <= collateral[t])`, invalid variable error
2. **Regime derivation** — auto-derived alternative spec has correct equation substitution
3. **One-constraint ZLB** — AR(1) + Taylor rule model, negative shock triggers ZLB, verify piecewise clamps `i >= 0`, linear goes negative, converges, regime history correct
4. **Two-constraint** — Two independent constraints, verify 4-regime logic
5. **OccBin IRF** — Large shock triggers constraint, compare linear vs piecewise
6. **No-binding case** — Small shock never triggers, piecewise == linear
7. **Convergence failure** — Pathological case, verify warning + `converged=false`
8. **Explicit alternative spec** — User-provided second `@dsge` block
9. **Display/report** — `show(io, sol)` outputs correctly
