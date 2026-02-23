# DSGE Module Design — MacroEconometricModels.jl

**Date**: 2026-02-24
**Branch**: `dsge`
**Scope**: Linear DSGE models — parsing, solving, simulation, and frequentist GMM estimation

---

## Overview

Add a `src/dsge/` module providing:
1. A `@dsge` macro for specifying DSGE models in natural Julia syntax
2. Automatic steady-state computation via `Optim.jl`
3. Auto-linearization via numerical Jacobians
4. Two rational expectations solvers: Sims' gensys (2002) and Blanchard-Kahn (1980)
5. Perfect foresight (deterministic) solver
6. Frequentist GMM estimation (IRF matching + Euler equation moments)
7. Stochastic simulation and IRF/FEVD via existing infrastructure

**No new dependencies.** Uses `LinearAlgebra` (QZ/Schur), `Optim` (steady state), `SparseArrays` (perfect foresight Jacobian), and existing package GMM/IRF code.

---

## File Structure

```
src/dsge/
├── types.jl              # DSGESpec, LinearDSGE, DSGESolution, PerfectForesightPath, DSGEEstimation
├── parser.jl             # @dsge macro, AST walking, E[t] operator, time-index detection
├── steady_state.jl       # Numerical SS via Optim.jl (NelderMead + LBFGS)
├── linearize.jl          # Numerical Jacobians → Γ₀, Γ₁, C, Ψ, Π matrices
├── gensys.jl             # Sims (2002) QZ decomposition solver
├── blanchard_kahn.jl     # BK (1980) eigenvalue counting solver
├── perfect_foresight.jl  # Newton solver for stacked deterministic system
├── estimation.jl         # GMM estimation (IRF matching + Euler equation)
└── simulation.jl         # Stochastic simulation, IRF/FEVD bridge to existing types
```

---

## Type System

```julia
abstract type AbstractDSGEModel <: StatsAPI.StatisticalModel end

# Parsed symbolic representation from @dsge macro
struct DSGESpec{T<:AbstractFloat}
    endog::Vector{Symbol}           # endogenous variable names
    exog::Vector{Symbol}            # exogenous shock names
    params::Vector{Symbol}          # parameter names
    param_values::Dict{Symbol,T}    # calibrated parameter values
    equations::Vector{Expr}         # raw Julia expressions (from macro)
    n_endog::Int
    n_exog::Int
    n_params::Int
    steady_state::Vector{T}         # SS values (filled by compute_steady_state)
    varnames::Vector{String}        # display names
end

# Linearized canonical form: Γ₀·y_t = Γ₁·y_{t-1} + C + Ψ·ε_t + Π·η_t
struct LinearDSGE{T<:AbstractFloat}
    Gamma0::Matrix{T}    # n × n
    Gamma1::Matrix{T}    # n × n
    C::Vector{T}         # n × 1 constants
    Psi::Matrix{T}       # n × n_shocks
    Pi::Matrix{T}        # n × n_expect (expectation error selection)
    spec::DSGESpec{T}    # back-reference
end

# RE solution: y_t = G1·y_{t-1} + impact·ε_t + C_sol
struct DSGESolution{T<:AbstractFloat}
    G1::Matrix{T}        # n × n state transition
    impact::Matrix{T}    # n × n_shocks impact matrix
    C_sol::Vector{T}     # constants
    eu::Vector{Int}      # [existence, uniqueness] flags
    spec::DSGESpec{T}
    linear::LinearDSGE{T}
end

# Perfect foresight path
struct PerfectForesightPath{T<:AbstractFloat}
    path::Matrix{T}      # T_periods × n_endog
    converged::Bool
    spec::DSGESpec{T}
end

# GMM estimation result
struct DSGEEstimation{T<:AbstractFloat} <: AbstractDSGEModel
    theta::Vector{T}             # estimated parameters
    vcov::Matrix{T}              # asymptotic covariance
    param_names::Vector{Symbol}  # which parameters were estimated
    method::Symbol               # :irf_matching or :euler_gmm
    J_stat::T
    J_pvalue::T
    solution::DSGESolution{T}    # solution at estimated params
    converged::Bool
    spec::DSGESpec{T}
end
```

---

## @dsge Macro and Parser

### User-facing syntax

```julia
spec = @dsge begin
    parameters: α = 0.33, β = 0.99, δ = 0.025, ρ = 0.9, σ_A = 0.01

    endogenous: C, K, Y, A
    exogenous: ε_A

    # Euler equation with expectation operator
    1/C[t] = β * E[t](1/C[t+1]) * (α * K[t]^(α-1) + 1-δ)
    # Resource constraint
    C[t] + K[t] = K[t-1]^α + (1-δ)*K[t-1]
    # Production
    Y[t] = exp(A[t]) * K[t-1]^α
    # Technology shock
    A[t] = ρ * A[t-1] + σ_A * ε_A[t]
end
```

### Parser internals

1. **`_parse_dsge_block(expr)`**: Split declarations (`parameters:`, `endogenous:`, `exogenous:`) from equation lines
2. **`_extract_time_refs(eq)`**: Walk AST, find `var[t±k]` patterns → `Dict{Symbol, Set{Int}}`
3. **`_handle_expectation_operator(eq)`**: Detect `E[t](expr)` wrappers; under RE linearization, `E_t[x_{t+1}] = x_{t+1} - η_{t+1}`. If `[t+1]` appears without `E[t]`, treat implicitly as `E[t](·)`
4. **`_build_residual_fn(eq, endog, exog, params)`**: Transform equation into callable `f(y_t, y_{t-1}, y_{t+1}, ε, θ) → scalar residual`
5. Macro emits `DSGESpec(...)` constructor call

---

## Steady State

```julia
compute_steady_state(spec::DSGESpec{T};
    initial_guess=nothing, method=:auto) → DSGESpec{T}  # returns updated spec
```

- Solves `f(y_ss, y_ss, y_ss, 0, θ) = 0`
- Default: `Optim.NelderMead()` for initial, then `Optim.LBFGS()` for refinement
- User can provide `initial_guess::Vector{T}`
- `:analytical` method: user passes `ss_fn(θ) → y_ss` for closed-form SS

---

## Linearization

```julia
linearize(spec::DSGESpec{T}) → LinearDSGE{T}
```

1. Build vectorized residual function around steady state
2. Numerical Jacobians via central differences (step size `h = max(1e-7, 1e-7 * |y_ss|)`)
3. `∂f/∂y_t → Γ₀`, `∂f/∂y_{t-1} → -Γ₁`, `∂f/∂ε → -Ψ`
4. For forward-looking variables: `∂f/∂y_{t+1} → Π` (expectation error mapping)

---

## Solvers

### Gensys (Sims 2002)

```julia
gensys(Γ0, Γ1, C, Ψ, Π) → (G1, impact, C_sol, eu)
```

1. Generalized Schur: `schur(Γ0, Γ1)` → `(S, T, Q, Z)`
2. Reorder stable eigenvalues (`|T_ii/S_ii| < 1`) first
3. Partition Z; check existence (`rank(Q₂₂·Π) == rank(Π)`) and uniqueness (n_unstable == n_forward)
4. Build solution matrices

### Blanchard-Kahn (1980)

```julia
blanchard_kahn(A, B, n_predetermined) → (G1, impact, eu)
```

1. Eigendecomposition of A
2. Count unstable eigenvalues (`|λ| > 1`)
3. BK condition: n_unstable == n_forward_looking
4. Partition and solve for policy function

### Perfect Foresight

```julia
perfect_foresight(spec, shock_path, T_periods;
    initial_path=nothing) → PerfectForesightPath{T}
```

1. Stack T×n unknowns
2. Block-tridiagonal sparse Jacobian (via `SparseArrays`)
3. Newton iteration: `Δx = -J \ F(x)`
4. Terminal condition: steady state at T

### Unified API

```julia
solve(spec; method=:gensys)                                    → DSGESolution
solve(spec; method=:blanchard_kahn)                            → DSGESolution
solve(spec; method=:perfect_foresight, T=200, shock_path=...) → PerfectForesightPath
```

---

## Estimation

### IRF Matching (CEE 2005)

```julia
estimate_dsge(spec, data, param_names;
    method=:irf_matching, target_irfs=nothing,
    var_lags=4, irf_horizon=20, weighting=:two_step) → DSGEEstimation
```

- If `target_irfs` not provided: estimate VAR on data → compute IRFs
- For each candidate θ: solve DSGE → model IRFs → distance = `vec(model_irf - target_irf)`
- Uses existing `estimate_gmm` infrastructure

### Euler Equation GMM (Hansen & Singleton 1982)

```julia
estimate_dsge(spec, data, param_names;
    method=:euler_gmm, moment_eqs=..., instruments=...) → DSGEEstimation
```

- Euler equation residuals evaluated at data
- Instruments: lagged variables
- Uses existing `estimate_gmm`

---

## Simulation and Analysis Bridge

### Stochastic Simulation

```julia
simulate(sol::DSGESolution, T_periods;
    shock_draws=nothing) → Matrix{T}  # T_periods × n_endog
```

Iterates `y_t = G1·y_{t-1} + impact·ε_t` from steady state.

### IRF/FEVD via Existing Types

```julia
irf(sol::DSGESolution, H; shock=1) → ImpulseResponse{T}  # existing type
fevd(sol::DSGESolution, H)         → FEVD{T}             # existing type
```

Analytical: `Φ_h = G1^h * impact`. Wraps in existing types so `plot_result()`, `report()`, etc. work automatically.

---

## Integration Points

| Existing Infrastructure | DSGE Usage |
|---|---|
| `estimate_gmm` | GMM estimation engine |
| `ImpulseResponse{T}` | DSGE IRF output type |
| `FEVD{T}` | DSGE FEVD output type |
| `plot_result()` | Visualization (no new dispatch needed if types match) |
| `report()` | Display (new dispatch for `DSGEEstimation`) |
| `_coef_table()` | Parameter table display |
| `robust_inv()` | Matrix inversion in solvers |
| `Optim.optimize` | Steady state computation |
| `SparseArrays` | Perfect foresight Jacobian |
| `numerical_gradient` | Linearization Jacobians |

---

## Exports

```julia
# Types
export AbstractDSGEModel, DSGESpec, LinearDSGE, DSGESolution
export PerfectForesightPath, DSGEEstimation

# Macro
export @dsge

# Solving
export solve, compute_steady_state, linearize
export gensys, blanchard_kahn

# Estimation
export estimate_dsge

# Simulation
export simulate
# irf, fevd already exported (new dispatches on DSGESolution)
```

---

## Test Plan

```
test/dsge/
├── test_parser.jl          # @dsge macro parsing, time-index detection, E[t] operator
├── test_steady_state.jl    # SS computation for RBC, NK models
├── test_linearize.jl       # Jacobian accuracy, canonical form matrices
├── test_gensys.jl          # Gensys solver, existence/uniqueness, known solutions
├── test_blanchard_kahn.jl  # BK solver, eigenvalue counting, comparison with gensys
├── test_perfect_foresight.jl # Deterministic paths, convergence
├── test_estimation.jl      # IRF matching + Euler GMM on simulated data
├── test_simulation.jl      # Stochastic simulation, IRF/FEVD bridge
└── test_integration.jl     # Full workflow: specify → solve → estimate → analyze
```

---

## References

- Sims, C. A. (2002). "Solving Linear Rational Expectations Models." Computational Economics, 20(1-2), 1-20.
- Blanchard, O. J. & Kahn, C. M. (1980). "The Solution of Linear Difference Models under Rational Expectations." Econometrica, 48(5), 1305-1311.
- Christiano, L. J., Eichenbaum, M. & Evans, C. L. (2005). "Nominal Rigidities and the Dynamic Effects of a Shock to Monetary Policy." JPE, 113(1), 1-45.
- Hansen, L. P. & Singleton, K. J. (1982). "Generalized Instrumental Variables Estimation of Nonlinear Rational Expectations Models." Econometrica, 50(5), 1269-1286.
