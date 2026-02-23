# Analytical Steady State + Moment Conditions — Design Document

**Goal:** Add analytical steady-state computation via `@dsge` macro and analytical moment conditions (Lyapunov equation) for GMM/SMM estimation without simulation.

---

## 1. Analytical Steady State

### Problem

`compute_steady_state` currently uses numerical optimization (NelderMead + LBFGS) to find the steady state. For most textbook DSGE models, the steady state has a closed-form solution that is faster and more reliable.

### Design

**Two entry points** (both kwarg and macro block):

#### A. `@dsge` Macro — `steady_state` Block

```julia
dsge = @dsge Float64 begin
    parameters: β = 0.99, α = 0.36, δ = 0.025
    endogenous: y, c, k
    exogenous: ε_a

    equations:
        1/c = β * (α * y(+1)/k + 1 - δ) * 1/c(+1)
        y = k(-1)^α * exp(ε_a)
        k = y - c + (1 - δ) * k(-1)

    steady_state:
        k_ss = (α * β / (1 - β * (1 - δ)))^(1 / (1 - α))
        y_ss = k_ss^α
        c_ss = y_ss - δ * k_ss
        return [y_ss, c_ss, k_ss]
end
```

The `steady_state` block is parsed at macro-expansion time into an anonymous function `(θ) -> Vector{T}`, where `θ` is the parameter vector. Parameter names (β, α, δ) are available as local variables extracted from `θ`.

#### B. `ss_fn` Kwarg (existing)

Already supported in `compute_steady_state` via `method=:analytical`. No changes needed — this path already works.

#### C. Integration

`compute_steady_state` gains a check: if `spec.ss_fn !== nothing`, call it directly (bypassing numerical optimization). The `ss_fn` field is added to `DSGESpec{T}`.

```julia
# In DSGESpec (types.jl):
struct DSGESpec{T<:AbstractFloat}
    ...existing fields...
    ss_fn::Union{Nothing, Function}   # analytical steady-state function (θ -> Vector{T})
end
```

Parser changes in `_dsge_impl` (parser.jl):
1. Detect `steady_state:` declaration alongside existing `parameters:`, `endogenous:`, etc.
2. Extract the body, wrap it as `(θ) -> begin ... end` with parameter unpacking
3. Pass `ss_fn` to `DSGESpec` constructor

Steady-state changes in `compute_steady_state` (steady_state.jl):
1. If `spec.ss_fn !== nothing`, call `spec.ss_fn(spec.param_values)` and return
2. Otherwise, proceed with existing numerical optimization

---

## 2. Analytical Moment Conditions

### Problem

`estimate_dsge(...; method=:smm)` requires simulation to compute moments, which is slow and introduces simulation noise. For linear DSGE models, the unconditional variance-covariance matrix can be computed exactly via the discrete Lyapunov equation.

### Design

#### A. `solve_lyapunov(G1, impact)` — Core Computation

Solves `Σ_y = G1 · Σ_y · G1' + impact · impact'` via vectorization:

```
vec(Σ_y) = (I - G1 ⊗ G1)⁻¹ · vec(impact · impact')
```

Returns `Σ_y` (n × n unconditional covariance matrix of state variables).

#### B. `analytical_moments(sol; lags=1)` — Moment Vector

Given a `DSGESolution`, computes the same moment vector as `autocovariance_moments`:

1. **Variance-covariance**: `Σ_y = solve_lyapunov(G1, impact)` — extract unique elements (upper triangle)
2. **Autocovariances at lag h**: `Γ_h = G1^h · Σ_y` — extract diagonal elements

Output format matches `autocovariance_moments(data; lags)` exactly, so SMM weighting matrices are compatible.

#### C. `method=:analytical_gmm` for `estimate_dsge`

New estimation method that minimizes the distance between analytical moments and data moments:

```julia
estimate_dsge(spec, data, param_names;
              method=:analytical_gmm,
              moments_fn=autocovariance_moments,
              lags=1, ...)
```

**Algorithm:**
1. Compute data moments: `m_data = moments_fn(data)`
2. For each candidate θ: update spec → solve → `analytical_moments(sol; lags)` → `m_sim`
3. Minimize `Q(θ) = (m_data - m_sim)' W (m_data - m_sim)` via `estimate_gmm`

**Standard errors:** Same GMM sandwich formula but without the `(1 + 1/τ)` simulation correction (no simulation noise).

---

## 3. File Organization

### New File

- `src/dsge/analytical.jl` — `solve_lyapunov`, `analytical_moments`

### Modified Files

- `src/dsge/types.jl` — add `ss_fn` field to `DSGESpec`
- `src/dsge/parser.jl` — parse `steady_state:` block in `@dsge` macro
- `src/dsge/steady_state.jl` — check `spec.ss_fn` before numerical optimization
- `src/dsge/estimation.jl` — add `:analytical_gmm` method to `estimate_dsge`
- `src/MacroEconometricModels.jl` — include `dsge/analytical.jl`, export new functions
- `test/dsge/test_dsge.jl` — tests for all new functionality

### Exports

```julia
export solve_lyapunov, analytical_moments
```

---

## 4. What This Does NOT Include (YAGNI)

- No second-order moment conditions (linear only)
- No analytical Jacobian of moments w.r.t. parameters (use numerical differentiation)
- No spectral density matching (frequency-domain moments)
- No steady-state solver for nonlinear systems (Newton-Raphson etc.)
- No automatic differentiation through the Lyapunov solver

---

## 5. References

- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. (Ch. 10: Lyapunov equation)
- Hansen, L. P. (1982). "Large Sample Properties of Generalized Method of Moments Estimators." *Econometrica*, 50(4), 1029–1054.
- Ruge-Murcia, F. (2012). "Estimating Nonlinear DSGE Models by the Simulated Method of Moments." *Journal of Economic Dynamics and Control*, 36(6), 914–938.
