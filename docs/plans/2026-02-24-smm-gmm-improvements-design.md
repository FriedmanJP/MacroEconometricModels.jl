# SMM Implementation + GMM Improvements — Design Document

**Goal:** Add Simulated Method of Moments (SMM) estimation and improve the existing GMM implementation with Optim.jl-based optimization and parameter transforms.

**Reference:** Ruge-Murcia (2012), "Estimating Nonlinear DSGE Models by the Simulated Method of Moments," *Journal of Economic Dynamics and Control*.

---

## 1. GMM Optimizer Upgrade

Replace the hand-rolled BFGS in `minimize_gmm` with `Optim.optimize` (LBFGS with NelderMead fallback). The current 80-line implementation duplicates Optim.jl functionality and is less robust. The package already imports Optim everywhere.

**Change is internal only.** `estimate_gmm`, `GMMModel`, `j_test`, etc. retain their existing API. The only behavioral difference is better convergence.

```julia
# Before (hand-rolled):
function minimize_gmm(moment_fn, theta0, data, W; ...)
    # 80 lines of BFGS + backtracking line search
end

# After (Optim.jl):
function minimize_gmm(moment_fn, theta0, data, W; ...)
    obj(t) = gmm_objective(t, moment_fn, data, W)
    result = Optim.optimize(obj, theta0, Optim.LBFGS(),
                            Optim.Options(iterations=max_iter, f_reltol=tol))
    # NelderMead fallback if LBFGS fails
    ...
end
```

---

## 2. Parameter Transforms

Constrained optimization via bijective transforms between constrained (model) and unconstrained (optimizer) spaces.

```julia
struct ParameterTransform{T<:AbstractFloat}
    lower::Vector{T}   # -Inf = unbounded below
    upper::Vector{T}   # Inf = unbounded above
end
```

**Transform rules:**
- `(-Inf, Inf)` → identity
- `(0, Inf)` → exp/log
- `(-Inf, 0)` → -exp/-log
- `(a, b)` → logistic: `a + (b-a) / (1 + exp(-x))`

**Functions:**
- `to_unconstrained(pt, theta)` — model → optimizer
- `to_constrained(pt, phi)` — optimizer → model
- `transform_jacobian(pt, phi)` — diagonal Jacobian for SE correction via delta method

Both `estimate_gmm` and `estimate_smm` accept an optional `bounds` keyword:
```julia
estimate_gmm(moment_fn, theta0, data;
             bounds=ParameterTransform([0.0, -1.0, 0.0], [1.0, 1.0, Inf]),
             ...)
```

When bounds are provided, optimization runs in unconstrained space. Standard errors are adjusted via the delta method using the Jacobian of the inverse transform.

---

## 3. SMM Estimator

### Type

```julia
struct SMMModel{T<:AbstractFloat} <: AbstractGMMModel
    theta::Vector{T}
    vcov::Matrix{T}
    n_moments::Int
    n_params::Int
    n_obs::Int
    weighting::GMMWeighting{T}
    W::Matrix{T}
    g_bar::Vector{T}       # moment discrepancy at solution
    J_stat::T
    J_pvalue::T
    converged::Bool
    iterations::Int
    sim_ratio::Int          # τ = simulation periods / data periods
end
```

`SMMModel` shares the `AbstractGMMModel` interface with `GMMModel` (coef, vcov, nobs, stderror, show, refs, report, j_test all work).

### API

```julia
estimate_smm(simulator_fn, moments_fn, theta0, data;
             sim_ratio::Int=5,
             burn::Int=100,
             weighting::Symbol=:two_step,
             bounds=nothing,
             hac::Bool=true,
             bandwidth::Int=0,
             max_iter::Int=1000,
             tol=1e-8,
             rng=Random.default_rng()) -> SMMModel{T}
```

**Arguments:**
- `simulator_fn(theta, T_periods, burn; rng) -> Matrix{T}` — simulates `T_periods` observations after discarding `burn` burn-in periods
- `moments_fn(data) -> Vector{T}` — computes moment vector from any T×k data matrix
- `theta0` — initial parameter guess
- `data` — observed data matrix (T×k)

**Algorithm:**
1. Compute data moments: `m_data = moments_fn(data)`
2. Compute weighting matrix from data (identity for step 1, HAC-optimal for step 2)
3. Minimize `Q(θ) = (m_data - moments_fn(simulator_fn(θ, τ*n, burn)))' W (m_data - ...)`
4. Compute SEs with simulation correction factor

### Standard Error Formula

For efficient GMM (optimal W):
```
V = (1 + 1/τ) × (D' W⁻¹ D)⁻¹ / n
```

For non-optimal W (sandwich):
```
V = (1 + 1/τ) × (D'WD)⁻¹ D'W Ω W D (D'WD)⁻¹ / n
```

Where:
- `τ = sim_ratio` (simulation-to-sample ratio)
- `D` = numerical Jacobian of simulated moments w.r.t. θ
- `Ω` = long-run variance of data moments (HAC)
- `n` = number of data observations

The `(1 + 1/τ)` correction accounts for simulation noise (Lee & Ingram 1991).

### Weighting Matrix

Computed from data using HAC with Bartlett kernel, matching the existing `long_run_covariance` infrastructure. Bandwidth selection: `floor(4*(n/100)^(2/9))` (Andrews 1991 plug-in rule) when `bandwidth=0`.

---

## 4. Built-in Moment Functions

```julia
autocovariance_moments(data::AbstractMatrix{T}; lags::Int=1) -> Vector{T}
```

Computes the standard DSGE moment vector:
1. Unique elements of variance-covariance matrix (upper triangle, k*(k+1)/2 elements)
2. Diagonal autocovariances at each lag (k elements per lag)

For k=2 variables and lags=1: `[var(y1), cov(y1,y2), var(y2), autocov(y1,1), autocov(y2,1)]` — 5 moments. This matches the MATLAB `Comp.m` pattern.

```julia
smm_weighting_matrix(data, moments_fn; hac=true, bandwidth=0) -> Matrix{T}
```

Computes the optimal weighting matrix for SMM from data. Uses the block approach from Ruge-Murcia's `Wtmat.m`: centers the moment contributions, then applies HAC with Bartlett kernel.

---

## 5. DSGE Integration

Convenience wrapper in `src/dsge/estimation.jl`:

```julia
function estimate_dsge(spec, data, param_names; method=:smm, ...)
```

Adds `:smm` as a third method alongside `:irf_matching` and `:euler_gmm`. Internally builds `simulator_fn` from the DSGE spec: for each candidate θ, updates spec parameters, calls `solve()`, and `simulate()`.

Returns `DSGEEstimation{T}` with `method=:smm`.

---

## 6. File Organization

**Modified files:**
- `src/gmm/gmm.jl` — replace `minimize_gmm` internals with Optim.jl; add `bounds` kwarg to `estimate_gmm`

**New files:**
- `src/gmm/transforms.jl` — `ParameterTransform`, `to_unconstrained`, `to_constrained`, `transform_jacobian`
- `src/gmm/smm.jl` — `SMMModel`, `estimate_smm`, `autocovariance_moments`, `smm_weighting_matrix`

**Modified for integration:**
- `src/dsge/estimation.jl` — add `:smm` method to `estimate_dsge`
- `src/MacroEconometricModels.jl` — includes and exports
- `src/summary_refs.jl` — add SMM references and refs() dispatches
- `src/summary.jl` — add report() for SMMModel

**Test files:**
- `test/gmm/test_gmm.jl` — update existing GMM tests for Optim.jl backend + add transform tests
- `test/gmm/test_smm.jl` — new SMM test file

---

## 7. Exports

```julia
export SMMModel
export estimate_smm
export ParameterTransform, to_unconstrained, to_constrained
export autocovariance_moments
```

---

## 8. What This Does NOT Include (YAGNI)

- No indirect inference (different estimator class)
- No second-order perturbation (linear DSGE only)
- No Monte Carlo experiment framework (scripting concern, not library)
- No analytical moment computation from state-space (optimization for later)
- No CUE (continuously updated estimator) — iterated GMM suffices

---

## 9. References

- Ruge-Murcia, F. (2012). "Estimating Nonlinear DSGE Models by the Simulated Method of Moments." *Journal of Economic Dynamics and Control*, 36(6), 914–938.
- Lee, B.-S. & Ingram, B. F. (1991). "Simulation Estimation of Time-Series Models." *Journal of Econometrics*, 47(2–3), 197–205.
- Duffie, D. & Singleton, K. J. (1993). "Simulated Moments Estimation of Markov Models of Asset Prices." *Econometrica*, 61(4), 929–952.
- Hansen, L. P. (1982). "Large Sample Properties of Generalized Method of Moments Estimators." *Econometrica*, 50(4), 1029–1054.
