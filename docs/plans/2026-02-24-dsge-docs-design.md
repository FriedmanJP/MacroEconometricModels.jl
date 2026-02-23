# DSGE Documentation Design

**Goal:** Write comprehensive documentation for the DSGE module, covering model specification, solvers, estimation, OccBin, and visualization.

**Scope:** New `dsge.md` page + additions to `plotting.md` and `examples.md` + API reference updates.

---

## Part 1: New page `docs/src/dsge.md`

Placed in navigation between "Panel Models" and "Innovation Accounting".

### Section 1: Quick Start
Three recipes:
1. Simple NK model → `solve()` → `irf()` → `plot_result()`
2. GMM estimation with `estimate_dsge()`
3. OccBin ZLB with `occbin_solve()`

### Section 2: Model Specification
- `@dsge` macro syntax
- `parameters:`, `endogenous:`, `exogenous:` blocks
- Time subscripts `[t]`, `[t-1]`, `[t+1]`
- `steady_state` block for analytical SS
- `varnames` for display labels

### Section 3: Steady State
- `compute_steady_state(spec)` — numerical (NelderMead + LBFGS)
- Analytical via `ss_fn` in `@dsge` block
- Validation and diagnostics

### Section 4: Linearization
- `linearize(spec)` → `LinearDSGE{T}`
- Numerical Jacobians at steady state
- Sims canonical form: Γ₀ y_t = Γ₁ y_{t-1} + C + Ψ ε_t + Π η_t

### Section 5: Solution Methods
- `solve(spec; method=:gensys)` — unified dispatch
- Gensys (QZ decomposition, Sims 2002)
- Blanchard-Kahn (eigenvalue counting, 1980)
- Determinacy and stability checks: `is_determined()`, `is_stable()`

### Section 6: Simulation and IRFs
- `simulate(sol, T)` — stochastic simulation
- `irf(sol, H)` → `ImpulseResponse{T}` — analytical impulse responses
- `fevd(sol, H)` → `FEVD{T}` — forecast error variance decomposition
- `solve_lyapunov(sol)` — unconditional variance
- `analytical_moments(sol)` — mean, variance, autocovariances, autocorrelations

### Section 7: Estimation
- `estimate_dsge(spec, data, params; method)` → `DSGEEstimation{T}`
- Four methods:
  - `:irf_matching` — match empirical IRFs from VAR
  - `:euler_gmm` — Euler equation moment conditions
  - `:smm` — simulated method of moments
  - `:analytical_gmm` — analytical moments matching
- Hansen J-test for overidentification
- Parameter bounds and identification

### Section 8: Perfect Foresight
- `perfect_foresight(spec; shocks, T)` → `PerfectForesightPath{T}`
- Deterministic transition paths
- Newton solver with sparse block-tridiagonal Jacobian

### Section 9: Occasionally Binding Constraints (OccBin)
- `parse_constraint(:(i[t] >= 0), spec)` — constraint syntax
- `occbin_solve(spec, constraint; shock_path)` — one constraint
- `occbin_solve(spec, c1, c2; ...)` — two constraints (4 regimes)
- `occbin_irf(spec, constraint, shock_idx, H)` — constrained IRFs
- Guess-and-verify algorithm explanation
- Explicit alternative regime specifications

### Section 10: Complete Example
3-equation New Keynesian model (IS + Phillips + Taylor rule):
- Specify with `@dsge`
- Compute steady state
- Solve with Gensys
- Simulate and compute IRFs
- Estimate via IRF matching
- Add ZLB constraint with OccBin

### Section 11: References
Blanchard-Kahn (1980), Sims (2002), Hamilton (1994), Guerrieri-Iacoviello (2015)

---

## Part 2: Plotting additions in `docs/src/plotting.md`

Add "DSGE Models" subsection showing:
- `irf(sol, H)` → `plot_result()` — standard IRF plots
- `fevd(sol, H)` → `plot_result()` — standard FEVD plots
- `plot_result(oirf::OccBinIRF)` — linear vs piecewise IRF comparison
- `plot_result(sol::OccBinSolution)` — piecewise path with regime shading

---

## Part 3: Examples additions in `docs/src/examples.md`

- **Example 18: DSGE — RBC Model** — Specify, solve, simulate, IRFs, moments, estimation
- **Example 19: DSGE — ZLB with OccBin** — NK model, negative demand shock, constrained vs unconstrained

---

## Part 4: API Reference Updates

### `docs/src/api.md`
Add DSGE table with 15 key functions.

### `docs/src/api_types.md`
Add DSGE Types section with `@docs` blocks for all 9 types.

### `docs/src/api_functions.md`
Add DSGE Functions section with `@docs` blocks for all exported functions.

---

## Part 5: Navigation Update

### `docs/make.jl`
Add `"DSGE Models" => "dsge.md"` in the Multivariate Models section, after Panel Models.
