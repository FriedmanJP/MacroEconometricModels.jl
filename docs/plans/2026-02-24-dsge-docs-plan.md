# DSGE Documentation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Write comprehensive documentation for the DSGE module covering model specification, solvers, estimation, OccBin, and visualization.

**Architecture:** Five tasks in order: (1) main dsge.md page, (2) plotting additions, (3) examples additions, (4) API reference updates, (5) navigation. Each task writes content, verifies the docs build, and commits.

**Tech Stack:** Documenter.jl, MathJax3, Julia markdown

---

### Task 1: Create `docs/src/dsge.md`

**Files:**
- Create: `docs/src/dsge.md`

**Step 1: Write `docs/src/dsge.md`**

Create the file with the following complete content (11 sections). Follow the project's documentation conventions: H1 → Intro → Quick Start → H2 sections (separated by `---`) → H3 subs → Complete Example → References. Use `!!! note "Technical Note"` for theory, `using MacroEconometricModels`, `Random.seed!(42)`.

```markdown
# DSGE Models

**MacroEconometricModels.jl** provides a complete toolkit for Dynamic Stochastic General Equilibrium (DSGE) models: specify models with the `@dsge` macro, compute steady states, linearize around the steady state, solve via Gensys (QZ decomposition) or Blanchard-Kahn, simulate and compute analytical impulse responses, estimate deep parameters via GMM, solve deterministic perfect-foresight paths, and handle occasionally binding constraints via OccBin. All results integrate with `plot_result()` for interactive visualization.

## Quick Start

**Recipe 1: Solve and plot IRFs**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Specify a simple RBC model
spec = @dsge begin
    parameters: β = 0.99, α = 0.36, δ = 0.025, ρ = 0.9, σ = 0.01
    endogenous: Y, C, K, A
    exogenous: ε_A

    # Production
    Y[t] = A[t] * K[t-1]^α

    # Resource constraint
    C[t] + K[t] = Y[t] + (1 - δ) * K[t-1]

    # Euler equation
    1 = β * (C[t] / C[t+1]) * (α * A[t+1] * K[t]^(α - 1) + 1 - δ)

    # Technology shock
    A[t] = ρ * A[t-1] + σ * ε_A[t]

    steady_state: begin
        A_ss = 1.0
        K_ss = (α * β / (1 - β * (1 - δ)))^(1 / (1 - α))
        Y_ss = K_ss^α
        C_ss = Y_ss - δ * K_ss
    end
end

sol = solve(spec)
result = irf(sol, 40)
plot_result(result)
```

**Recipe 2: Estimate with GMM**

```julia
# Estimate deep parameters from data
est = estimate_dsge(spec, Y_data, [:β, :α, :ρ];
                    method=:irf_matching, var_lags=4, irf_horizon=20)
report(est)
```

**Recipe 3: OccBin with ZLB**

```julia
# Add a zero lower bound on the interest rate
constraint = parse_constraint(:(R[t] >= 0), spec)
occ_sol = occbin_solve(spec, constraint; shock_path=shocks)
occ_irf = occbin_irf(spec, constraint, 1, 40)
plot_result(occ_irf)
```

---

## Model Specification

The `@dsge` macro provides a domain-specific language for specifying DSGE models. It parses the model into a `DSGESpec{T}` object containing equations, parameters, and variable declarations.

### Syntax

```julia
spec = @dsge begin
    parameters: β = 0.99, α = 0.36, δ = 0.025, ρ = 0.9, σ = 0.01
    endogenous: Y, C, K, A
    exogenous: ε_A

    # Equations (one per endogenous variable)
    Y[t] = A[t] * K[t-1]^α
    C[t] + K[t] = Y[t] + (1 - δ) * K[t-1]
    1 = β * (C[t] / C[t+1]) * (α * A[t+1] * K[t]^(α - 1) + 1 - δ)
    A[t] = ρ * A[t-1] + σ * ε_A[t]
end
```

### Blocks

| Block | Syntax | Description |
|-------|--------|-------------|
| `parameters:` | `name = value, ...` | Calibrated parameters with default values |
| `endogenous:` | `var1, var2, ...` | Endogenous variable names |
| `exogenous:` | `shock1, shock2, ...` | Exogenous shock names |
| `steady_state:` | `begin ... end` | Optional analytical steady-state expressions |
| `varnames:` | `["Label 1", "Label 2", ...]` | Optional display labels for variables |

### Time Subscripts

| Notation | Meaning |
|----------|---------|
| `var[t]` | Current period value |
| `var[t-1]` | One-period lag |
| `var[t+1]` | One-period lead (forward-looking) |

Variables with `[t+1]` subscripts generate expectation errors in the Sims canonical form. The number of forward-looking equations determines the dimension of the ``\Pi`` matrix.

!!! note "Technical Note"
    Equations are written as `LHS = RHS` where both sides can contain endogenous variables at different time subscripts. The `@dsge` macro rearranges each equation into residual form ``f(y_t, y_{t-1}, y_{t+1}, \varepsilon_t, \theta) = 0`` via `LHS - RHS`. The number of equations must equal the number of endogenous variables.

### Return Value

| Field | Type | Description |
|-------|------|-------------|
| `endog` | `Vector{Symbol}` | Endogenous variable names |
| `exog` | `Vector{Symbol}` | Exogenous shock names |
| `params` | `Vector{Symbol}` | Parameter names |
| `param_values` | `Dict{Symbol,T}` | Calibrated values |
| `equations` | `Vector{Expr}` | Raw equation expressions |
| `n_endog` | `Int` | Number of endogenous variables |
| `n_exog` | `Int` | Number of exogenous shocks |
| `n_expect` | `Int` | Number of expectation errors |
| `forward_indices` | `Vector{Int}` | Indices of forward-looking equations |
| `steady_state` | `Vector{T}` | Steady-state values |
| `varnames` | `Vector{String}` | Display names |

---

## Steady State

The steady state ``\bar{y}`` satisfies the system in the absence of shocks:

```math
f(\bar{y}, \bar{y}, \bar{y}, 0, \theta) = 0
```

### Numerical Computation

```julia
spec = compute_steady_state(spec)
println("Steady state: ", spec.steady_state)
```

`compute_steady_state` uses a two-phase optimizer: Nelder-Mead for robustness, then L-BFGS for refinement. It minimizes the sum of squared residuals ``\sum_i f_i(\bar{y})^2``.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `initial_guess` | `Vector` | `nothing` | Starting point (default: ones) |
| `method` | `Symbol` | `:auto` | `:auto` (NelderMead → LBFGS) or `:analytical` |

### Analytical Steady State

For models where the steady state has a closed-form solution, specify it in the `steady_state:` block:

```julia
spec = @dsge begin
    parameters: β = 0.99, α = 0.36, δ = 0.025, ρ = 0.9, σ = 0.01
    endogenous: Y, C, K, A
    exogenous: ε_A

    Y[t] = A[t] * K[t-1]^α
    C[t] + K[t] = Y[t] + (1 - δ) * K[t-1]
    1 = β * (C[t] / C[t+1]) * (α * A[t+1] * K[t]^(α - 1) + 1 - δ)
    A[t] = ρ * A[t-1] + σ * ε_A[t]

    steady_state: begin
        A_ss = 1.0
        K_ss = (α * β / (1 - β * (1 - δ)))^(1 / (1 - α))
        Y_ss = K_ss^α
        C_ss = Y_ss - δ * K_ss
    end
end
```

When the `steady_state:` block is provided, `compute_steady_state` (or `solve`) uses it directly and validates the result against the equations.

---

## Linearization

`linearize` computes a first-order approximation around the steady state using numerical Jacobians (central differences). It produces the Sims (2002) canonical form:

```math
\Gamma_0 \, y_t = \Gamma_1 \, y_{t-1} + C + \Psi \, \varepsilon_t + \Pi \, \eta_t
```

where ``y_t`` is the vector of endogenous variables (in deviations from steady state), ``\varepsilon_t`` is the vector of exogenous shocks, and ``\eta_t`` is the vector of expectation errors.

```julia
ld = linearize(spec)
```

!!! note "Technical Note"
    The matrices are computed via central differences with step size ``h = \max(10^{-7}, 10^{-7} |y_j|)``. No analytical derivatives are required. ``\Gamma_0`` contains coefficients on ``y_t``, ``\Gamma_1`` on ``y_{t-1}``, ``\Psi`` on shocks, and ``\Pi`` selects the forward-looking equations for expectation errors.

### Return Value

| Field | Type | Description |
|-------|------|-------------|
| `Gamma0` | `Matrix{T}` | ``n \times n`` coefficient on ``y_t`` |
| `Gamma1` | `Matrix{T}` | ``n \times n`` coefficient on ``y_{t-1}`` |
| `C` | `Vector{T}` | ``n \times 1`` constants |
| `Psi` | `Matrix{T}` | ``n \times n_{shocks}`` shock loading |
| `Pi` | `Matrix{T}` | ``n \times n_{expect}`` expectation error selection |
| `spec` | `DSGESpec{T}` | Back-reference to specification |

---

## Solution Methods

`solve` is the unified entry point for solving DSGE models. It dispatches to the appropriate solver based on the `method` keyword:

```julia
sol = solve(spec; method=:gensys)
```

| Method | Algorithm | Reference |
|--------|-----------|-----------|
| `:gensys` (default) | QZ decomposition | Sims (2002) |
| `:blanchard_kahn` | Eigenvalue counting | Blanchard & Kahn (1980) |
| `:perfect_foresight` | Deterministic Newton solver | — |

`solve` automatically calls `compute_steady_state` and `linearize` if they have not been called already.

### Gensys

The Gensys algorithm (Sims 2002) solves the linearized system via QZ (generalized Schur) decomposition. It separates stable and unstable eigenvalues to construct the state-space solution:

```math
y_t = G_1 \, y_{t-1} + C_{sol} + \text{impact} \cdot \varepsilon_t
```

where ``G_1`` is the ``n \times n`` state transition matrix and ``\text{impact}`` is the ``n \times n_{shocks}`` impact matrix.

```julia
sol = solve(spec; method=:gensys)
println("Existence: ", sol.eu[1] == 1)    # 1 = exists
println("Uniqueness: ", sol.eu[2] == 1)   # 1 = unique
```

!!! note "Technical Note"
    The `div` keyword (default `1.0 + 10^{-8}`) sets the dividing line between stable and unstable eigenvalues. Eigenvalues with modulus below `div` are classified as stable. The `eu` vector reports `[existence, uniqueness]` where 1 = satisfied, 0 = violated.

### Blanchard-Kahn

The Blanchard-Kahn (1980) method partitions the system into predetermined and forward-looking variables. The determinacy condition requires the number of unstable eigenvalues to equal the number of forward-looking variables:

```julia
sol = solve(spec; method=:blanchard_kahn)
```

!!! warning
    The Blanchard-Kahn method requires exactly `n_unstable == n_forward_looking`. If this condition fails, the model is either indeterminate (too few unstable roots) or has no stable solution (too many).

### Determinacy and Stability

```julia
is_determined(sol)   # true if eu == [1, 1]
is_stable(sol)       # true if max|eigenvalue(G1)| < 1
```

### Return Value

| Field | Type | Description |
|-------|------|-------------|
| `G1` | `Matrix{T}` | ``n \times n`` state transition |
| `impact` | `Matrix{T}` | ``n \times n_{shocks}`` impact matrix |
| `C_sol` | `Vector{T}` | ``n \times 1`` constants |
| `eu` | `Vector{Int}` | `[existence, uniqueness]` (1=yes, 0=no) |
| `method` | `Symbol` | `:gensys` or `:blanchard_kahn` |
| `eigenvalues` | `Vector{ComplexF64}` | Generalized eigenvalues |
| `spec` | `DSGESpec{T}` | Model specification |
| `linear` | `LinearDSGE{T}` | Linearized form |

---

## Simulation and IRFs

### Stochastic Simulation

`simulate` generates sample paths from the solved model:

```math
y_t = G_1 \, y_{t-1} + \text{impact} \cdot \varepsilon_t + C_{sol}
```

```julia
Y_sim = simulate(sol, 200; rng=Random.MersenneTwister(42))
# Returns 200 × n_endog matrix in levels (steady state + deviations)
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `shock_draws` | `Matrix` | `nothing` | Pre-drawn shocks (``T \times n_{shocks}``) |
| `rng` | `AbstractRNG` | `default_rng()` | Random number generator |

### Impulse Response Functions

`irf` computes analytical impulse responses:

```math
\Phi_h = G_1^{h-1} \cdot \text{impact}
```

where ``\Phi_h`` is the response at horizon ``h`` to a one-standard-deviation shock.

```julia
result = irf(sol, 40)
plot_result(result)
```

Returns an `ImpulseResponse{T}` object compatible with the standard `plot_result()` method for multi-panel IRF plots.

### Forecast Error Variance Decomposition

`fevd` computes the share of forecast variance attributable to each shock at each horizon:

```math
\text{FEVD}_{ij}(h) = \frac{\sum_{s=0}^{h-1} (\Phi_s e_j)' (\Phi_s e_j)}{\sum_{s=0}^{h-1} \text{tr}(\Phi_s \Phi_s')}
```

```julia
result = fevd(sol, 40)
plot_result(result)
```

Returns a `FEVD{T}` object compatible with `plot_result()`.

### Unconditional Variance

`solve_lyapunov` solves the discrete Lyapunov equation for the unconditional covariance:

```math
\Sigma = G_1 \, \Sigma \, G_1' + \text{impact} \cdot \text{impact}'
```

```julia
Sigma = solve_lyapunov(sol.G1, sol.impact)
```

!!! note "Technical Note"
    Uses Kronecker vectorization: ``\text{vec}(\Sigma) = (I - G_1 \otimes G_1)^{-1} \text{vec}(\text{impact} \cdot \text{impact}')``. Throws `ArgumentError` if ``G_1`` is unstable.

### Analytical Moments

`analytical_moments` computes the theoretical moment vector from the Lyapunov solution:

```julia
m = analytical_moments(sol; lags=2)
```

Returns a concatenated vector: upper-triangle of the variance-covariance matrix (``k(k+1)/2`` elements) followed by diagonal autocovariances at each lag (``k`` elements per lag). These moments are used internally by the `:analytical_gmm` estimation method.

---

## Estimation

`estimate_dsge` estimates DSGE deep parameters via Generalized Method of Moments:

```julia
est = estimate_dsge(spec, Y_data, [:β, :ρ, :σ]; method=:irf_matching)
report(est)
```

### Methods

| Method | Description | Reference |
|--------|-------------|-----------|
| `:irf_matching` | Match model IRFs to empirical VAR IRFs | Christiano, Eichenbaum & Evans (2005) |
| `:euler_gmm` | Euler equation moment conditions with lagged instruments | Hansen & Singleton (1982) |
| `:smm` | Simulated Method of Moments | — |
| `:analytical_gmm` | Analytical moments via Lyapunov equation | — |

### IRF Matching

Estimates parameters by minimizing the distance between model-implied and data-implied impulse responses:

```math
\hat{\theta} = \arg\min_\theta \, [\Phi^{data}(H) - \Phi^{model}(\theta, H)]' \, W \, [\Phi^{data}(H) - \Phi^{model}(\theta, H)]
```

where ``\Phi^{data}`` is estimated from a VAR on the data and ``W`` is the GMM weighting matrix.

```julia
est = estimate_dsge(spec, Y_data, [:β, :α, :ρ];
                    method=:irf_matching,
                    var_lags=4,
                    irf_horizon=20,
                    weighting=:two_step)
```

### Euler Equation GMM

Uses Euler equation residuals as moment conditions with lagged variables as instruments:

```julia
est = estimate_dsge(spec, Y_data, [:β, :ρ];
                    method=:euler_gmm,
                    n_lags_instruments=4)
```

### Simulated Method of Moments (SMM)

Matches simulated moments to data moments:

```julia
est = estimate_dsge(spec, Y_data, [:β, :ρ, :σ];
                    method=:smm,
                    sim_ratio=5,
                    burn=100)
```

### Analytical GMM

Uses Lyapunov-equation-based moments (no simulation noise):

```julia
est = estimate_dsge(spec, Y_data, [:β, :ρ];
                    method=:analytical_gmm,
                    lags=2)
```

### Hansen J-test

All methods report the overidentification test:

```julia
println("J-statistic: ", est.J_stat)
println("p-value: ", est.J_pvalue)
```

A large p-value indicates the model's moment conditions are not rejected by the data.

### Keywords

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:irf_matching` | Estimation method |
| `target_irfs` | `ImpulseResponse` | `nothing` | Pre-computed empirical IRFs |
| `var_lags` | `Int` | `4` | VAR lag order for empirical IRFs |
| `irf_horizon` | `Int` | `20` | IRF horizon for matching |
| `weighting` | `Symbol` | `:two_step` | GMM weighting matrix |
| `n_lags_instruments` | `Int` | `4` | Instrument lags for Euler GMM |
| `sim_ratio` | `Int` | `5` | Simulation ratio for SMM |
| `burn` | `Int` | `100` | Burn-in periods for SMM |
| `moments_fn` | `Function` | `autocovariance_moments` | Moment function for SMM |
| `bounds` | `ParameterTransform` | `nothing` | Parameter bounds |
| `lags` | `Int` | `1` | Autocovariance lags for analytical GMM |

### Return Value

| Field | Type | Description |
|-------|------|-------------|
| `theta` | `Vector{T}` | Estimated deep parameters |
| `vcov` | `Matrix{T}` | Asymptotic covariance matrix |
| `param_names` | `Vector{Symbol}` | Parameter names |
| `method` | `Symbol` | Estimation method used |
| `J_stat` | `T` | Hansen J-test statistic |
| `J_pvalue` | `T` | J-test p-value |
| `solution` | `DSGESolution{T}` | Solution at estimated parameters |
| `converged` | `Bool` | Optimization convergence |
| `spec` | `DSGESpec{T}` | Model specification |

Standard StatsAPI interface: `coef(est)`, `vcov(est)`, `stderror(est)`, `dof(est)`.

---

## Perfect Foresight

`perfect_foresight` solves for the deterministic transition path given a known sequence of future shocks. The agents know the exact realization of shocks in advance (no uncertainty):

```julia
# Solve with perfect foresight
sol = solve(spec; method=:perfect_foresight,
            T_periods=100, shock_path=shock_matrix)
```

Or equivalently:

```julia
path = perfect_foresight(spec; T_periods=100, shock_path=shock_matrix)
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `T_periods` | `Int` | `100` | Number of periods |
| `shock_path` | `Matrix` | zeros | ``T \times n_{shocks}`` shock sequence |
| `max_iter` | `Int` | `100` | Newton iteration limit |
| `tol` | `Real` | `10^{-8}` | Convergence tolerance |

!!! note "Technical Note"
    Uses a Newton solver with a sparse block-tridiagonal Jacobian. The system stacks all ``T`` periods into a single nonlinear system and solves by iterating ``x^{(k+1)} = x^{(k)} - J^{-1} F(x^{(k)})`` where ``J`` exploits the block-tridiagonal sparsity pattern for efficiency.

### Return Value

| Field | Type | Description |
|-------|------|-------------|
| `path` | `Matrix{T}` | ``T \times n_{endog}`` level values |
| `deviations` | `Matrix{T}` | ``T \times n_{endog}`` deviations from SS |
| `converged` | `Bool` | Newton convergence flag |
| `iterations` | `Int` | Newton iterations used |
| `spec` | `DSGESpec{T}` | Model specification |

---

## Occasionally Binding Constraints (OccBin)

The OccBin method (Guerrieri & Iacoviello 2015) solves DSGE models with occasionally binding constraints using a piecewise-linear approximation. The idea is that the economy switches between a reference regime (constraint slack) and an alternative regime (constraint binding), and the solution iterates until the regime sequence is self-consistent.

### Parsing Constraints

```julia
constraint = parse_constraint(:(R[t] >= 0), spec)
```

Supports `>=` and `<=` directions. The variable must exist in `spec.endog`.

### One Constraint

The typical use case is a zero lower bound on the nominal interest rate:

```julia
# Construct shock path (large negative demand shock)
shocks = zeros(40, spec.n_exog)
shocks[1, 1] = -3.0  # 3 std dev negative shock

# Solve with ZLB constraint
occ_sol = occbin_solve(spec, constraint; shock_path=shocks)
println(occ_sol)
plot_result(occ_sol)
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `shock_path` | `Matrix{T}` | zeros | ``T \times n_{shocks}`` shock sequence |
| `nperiods` | `Int` | `size(shock_path, 1)` | Number of periods |
| `maxiter` | `Int` | `100` | Max guess-and-verify iterations |

!!! note "Technical Note"
    **Guess-and-verify algorithm:**
    1. Solve the unconstrained (reference) model and the alternative (binding) model
    2. Guess an initial regime sequence (all slack)
    3. Compute time-varying decision rules via backward iteration from the terminal period
    4. Simulate forward using the time-varying rules
    5. Check whether the constraint is satisfied — update the regime guess
    6. Repeat until the regime sequence converges

### Two Constraints

For models with two occasionally binding constraints (4 possible regime combinations):

```julia
c1 = parse_constraint(:(R[t] >= 0), spec)      # ZLB
c2 = parse_constraint(:(debt[t] <= limit), spec) # Debt ceiling

occ_sol = occbin_solve(spec, c1, c2; shock_path=shocks)
```

The four regimes are: (0,0) neither binds, (1,0) only first binds, (0,1) only second binds, (1,1) both bind.

| Additional Keyword | Type | Default | Description |
|--------------------|------|---------|-------------|
| `curb_retrench` | `Bool` | `false` | Limit constraint relaxation to 1 period/iteration |

### OccBin IRFs

Compare linear (unconstrained) and piecewise-linear (constrained) impulse responses:

```julia
occ_irf = occbin_irf(spec, constraint, 1, 40; magnitude=3.0)
plot_result(occ_irf)
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `magnitude` | `Real` | `1.0` | Shock size (standard deviations) |
| `maxiter` | `Int` | `100` | Max guess-and-verify iterations |

### OccBin Return Values

**`OccBinSolution{T}`:**

| Field | Type | Description |
|-------|------|-------------|
| `linear_path` | `Matrix{T}` | ``T \times n`` unconstrained path |
| `piecewise_path` | `Matrix{T}` | ``T \times n`` constrained path |
| `steady_state` | `Vector{T}` | Steady-state values |
| `regime_history` | `Matrix{Int}` | ``T \times n_c`` regime indicators (0=slack, 1=binding) |
| `converged` | `Bool` | Convergence flag |
| `iterations` | `Int` | Guess-and-verify iterations |

**`OccBinIRF{T}`:**

| Field | Type | Description |
|-------|------|-------------|
| `linear` | `Matrix{T}` | ``H \times n`` linear IRF |
| `piecewise` | `Matrix{T}` | ``H \times n`` piecewise IRF |
| `regime_history` | `Matrix{Int}` | ``H \times n_c`` regime indicators |
| `shock_name` | `String` | Name of shocked variable |

---

## Complete Example

A 3-equation New Keynesian model with an IS curve, Phillips curve, and Taylor rule — estimated and subjected to a ZLB constraint.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# ── 1. Model specification ──
spec = @dsge begin
    parameters: β = 0.99, σ_c = 1.0, κ = 0.3, φ_π = 1.5, φ_y = 0.5,
                ρ_d = 0.8, ρ_s = 0.7, σ_d = 0.01, σ_s = 0.01
    endogenous: y, π, R, d, s
    exogenous: ε_d, ε_s

    # IS curve (demand)
    y[t] = y[t+1] - (1 / σ_c) * (R[t] - π[t+1]) + d[t]

    # Phillips curve (supply)
    π[t] = β * π[t+1] + κ * y[t] + s[t]

    # Taylor rule
    R[t] = φ_π * π[t] + φ_y * y[t]

    # Demand shock
    d[t] = ρ_d * d[t-1] + σ_d * ε_d[t]

    # Supply shock
    s[t] = ρ_s * s[t-1] + σ_s * ε_s[t]
end

# ── 2. Solve ──
sol = solve(spec)
println("Determined: ", is_determined(sol))
println("Stable: ", is_stable(sol))

# ── 3. Impulse responses ──
irf_result = irf(sol, 40)
plot_result(irf_result)

# ── 4. FEVD ──
fevd_result = fevd(sol, 40)
plot_result(fevd_result)

# ── 5. Simulation ──
Y_sim = simulate(sol, 200; rng=Random.MersenneTwister(42))

# ── 6. Analytical moments ──
Sigma = solve_lyapunov(sol.G1, sol.impact)
m = analytical_moments(sol; lags=2)
println("Unconditional variance of output: ", Sigma[1, 1])

# ── 7. Estimation (IRF matching) ──
# est = estimate_dsge(spec, Y_data, [:κ, :φ_π, :ρ_d];
#                     method=:irf_matching, var_lags=4, irf_horizon=20)
# report(est)

# ── 8. OccBin: Zero lower bound ──
constraint = parse_constraint(:(R[t] >= 0), spec)

shocks = zeros(40, 2)
shocks[1, 1] = -3.0  # Large negative demand shock

occ_sol = occbin_solve(spec, constraint; shock_path=shocks)
println(occ_sol)

occ_irf = occbin_irf(spec, constraint, 1, 40; magnitude=3.0)
plot_result(occ_irf)
```

The OccBin IRF plot shows how the ZLB amplifies the output decline: the constrained path (red) drops further than the linear path (blue) because the interest rate cannot go below zero to stimulate the economy.

---

## References

- Blanchard, O. J., & Kahn, C. M. (1980). The Solution of Linear Difference Models under Rational Expectations. *Econometrica*, 48(5), 1305–1311.
- Christiano, L. J., Eichenbaum, M., & Evans, C. L. (2005). Nominal Rigidities and the Dynamic Effects of a Shock to Monetary Policy. *Journal of Political Economy*, 113(1), 1–45.
- Guerrieri, L., & Iacoviello, M. (2015). OccBin: A Toolkit for Solving Dynamic Models with Occasionally Binding Constraints Easily. *Journal of Monetary Economics*, 70, 22–38.
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
- Hansen, L. P. (1982). Large Sample Properties of Generalized Method of Moments Estimators. *Econometrica*, 50(4), 1029–1054.
- Hansen, L. P., & Singleton, K. J. (1982). Generalized Instrumental Variables Estimation of Nonlinear Rational Expectations Models. *Econometrica*, 50(5), 1269–1286.
- Sims, C. A. (2002). Solving Linear Rational Expectations Models. *Computational Economics*, 20(1–2), 1–20.
```

**Step 2: Verify docs build**

Run:
```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()' && julia --project=docs docs/make.jl
```

Note: This will warn about `dsge.md` not in navigation — that's expected until Task 5.

Expected: Build succeeds (warnings OK at this stage).

**Step 3: Commit**

```bash
git add docs/src/dsge.md
git commit -m "docs: add DSGE Models documentation page (11 sections)"
```

---

### Task 2: Add DSGE Plotting Subsection to `docs/src/plotting.md`

**Files:**
- Modify: `docs/src/plotting.md:489-490` (insert before `---` / `## Common Options`)

**Step 1: Insert DSGE subsection**

After line 489 (`Bar chart showing per-release impact on the nowcast revision.`) and before line 490 (`---`), insert:

```markdown

### DSGE Models

DSGE impulse responses and FEVD use the same `plot_result()` methods as VAR models:

```julia
sol = solve(spec)
result = irf(sol, 40)
p = plot_result(result)
```

Standard multi-panel IRF and FEVD plots are generated automatically.

#### OccBin IRF Comparison

`plot_result(::OccBinIRF)` shows side-by-side linear (unconstrained) vs piecewise-linear (constrained) impulse responses, with shaded regions indicating periods when the constraint binds:

```julia
constraint = parse_constraint(:(R[t] >= 0), spec)
oirf = occbin_irf(spec, constraint, 1, 40; magnitude=3.0)
p = plot_result(oirf)
```

Each variable gets a panel with two lines: dashed blue (linear) and solid red (piecewise). Shaded orange regions mark periods when the constraint is binding.

#### OccBin Solution Path

`plot_result(::OccBinSolution)` displays the piecewise-linear path for each variable with regime shading:

```julia
shocks = zeros(40, spec.n_exog)
shocks[1, 1] = -3.0
occ_sol = occbin_solve(spec, constraint; shock_path=shocks)
p = plot_result(occ_sol)
```

```

**Step 2: Verify docs build**

Run: `julia --project=docs docs/make.jl`

Expected: Build succeeds.

**Step 3: Commit**

```bash
git add docs/src/plotting.md
git commit -m "docs: add DSGE plotting subsection to plotting.md"
```

---

### Task 3: Add DSGE Examples to `docs/src/examples.md`

**Files:**
- Modify: `docs/src/examples.md` (two locations)

**Step 1: Update Quick Reference table**

After line 25 (the row for Example 17: Nowcasting), insert a new row:

```markdown
| 18 | DSGE Models | `@dsge`, `solve`, `irf`, `estimate_dsge`, `occbin_solve` | RBC model: specify, solve, simulate, IRFs, moments, estimation, OccBin ZLB |
```

**Step 2: Insert Example 18 before "Best Practices"**

Before line 2282 (`## Best Practices`), insert:

```markdown
## Example 18: DSGE Models

This example demonstrates the complete DSGE workflow: model specification, solution, simulation, impulse responses, analytical moments, and occasionally binding constraints.

### RBC Model

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Specify a Real Business Cycle model
spec = @dsge begin
    parameters: β = 0.99, α = 0.36, δ = 0.025, ρ = 0.9, σ = 0.01
    endogenous: Y, C, K, A
    exogenous: ε_A

    Y[t] = A[t] * K[t-1]^α
    C[t] + K[t] = Y[t] + (1 - δ) * K[t-1]
    1 = β * (C[t] / C[t+1]) * (α * A[t+1] * K[t]^(α - 1) + 1 - δ)
    A[t] = ρ * A[t-1] + σ * ε_A[t]

    steady_state: begin
        A_ss = 1.0
        K_ss = (α * β / (1 - β * (1 - δ)))^(1 / (1 - α))
        Y_ss = K_ss^α
        C_ss = Y_ss - δ * K_ss
    end
end
```

**Solve and check determinacy:**

```julia
sol = solve(spec)
println("Determined: ", is_determined(sol))
println("Stable: ", is_stable(sol))
println("Method: ", sol.method)
```

The Gensys solver uses QZ decomposition (Sims 2002) to find the unique stable rational expectations solution. `is_determined` confirms existence and uniqueness; `is_stable` confirms all eigenvalues of the transition matrix lie inside the unit circle.

**Impulse responses:**

```julia
# Analytical IRFs to a technology shock
irf_result = irf(sol, 40)
plot_result(irf_result)
```

A positive technology shock raises output and consumption on impact. Capital accumulates gradually as higher productivity raises the return to investment. The persistence of the response is governed by the AR(1) coefficient ``\rho = 0.9``.

**Simulation and moments:**

```julia
# Stochastic simulation
Y_sim = simulate(sol, 200; rng=Random.MersenneTwister(42))
println("Simulated output mean: ", round(mean(Y_sim[:, 1]); digits=4))

# Analytical unconditional covariance
Sigma = solve_lyapunov(sol.G1, sol.impact)
println("Unconditional std(Y): ", round(sqrt(Sigma[1, 1]); digits=6))

# Analytical moments (variance + autocovariances)
m = analytical_moments(sol; lags=2)
```

The Lyapunov equation ``\Sigma = G_1 \Sigma G_1' + \text{impact} \cdot \text{impact}'`` gives the model-implied unconditional covariance matrix without simulation noise.

**FEVD:**

```julia
fevd_result = fevd(sol, 40)
plot_result(fevd_result)
```

With a single shock, the FEVD assigns 100% of forecast variance to the technology shock at all horizons.

### New Keynesian Model with ZLB (OccBin)

```julia
# 3-equation NK model
nk = @dsge begin
    parameters: β = 0.99, σ_c = 1.0, κ = 0.3, φ_π = 1.5, φ_y = 0.5,
                ρ_d = 0.8, σ_d = 0.01
    endogenous: y, π, R, d
    exogenous: ε_d

    y[t] = y[t+1] - (1 / σ_c) * (R[t] - π[t+1]) + d[t]
    π[t] = β * π[t+1] + κ * y[t]
    R[t] = φ_π * π[t] + φ_y * y[t]
    d[t] = ρ_d * d[t-1] + σ_d * ε_d[t]
end

nk_sol = solve(nk)
```

**Apply the zero lower bound:**

```julia
# Parse the ZLB constraint
constraint = parse_constraint(:(R[t] >= 0), nk)

# Large negative demand shock
shocks = zeros(40, 1)
shocks[1, 1] = -3.0

# Solve with and without the constraint
occ_sol = occbin_solve(nk, constraint; shock_path=shocks)
println(occ_sol)
```

The `OccBinSolution` shows which periods the ZLB binds (regime_history = 1) and how many guess-and-verify iterations were needed for convergence.

**Compare IRFs:**

```julia
occ_irf = occbin_irf(nk, constraint, 1, 40; magnitude=3.0)
plot_result(occ_irf)
```

The constrained IRF (piecewise path) shows deeper output declines than the linear path because the interest rate is clamped at zero — the central bank cannot provide additional stimulus. This is the "ZLB amplification" mechanism studied by Guerrieri & Iacoviello (2015).

```

**Step 3: Verify docs build**

Run: `julia --project=docs docs/make.jl`

Expected: Build succeeds.

**Step 4: Commit**

```bash
git add docs/src/examples.md
git commit -m "docs: add Example 18 (DSGE Models) to examples.md"
```

---

### Task 4: Update API Reference Pages

**Files:**
- Modify: `docs/src/api.md` (after line 332)
- Modify: `docs/src/api_types.md` (lines 389-390 and after 390)
- Modify: `docs/src/api_functions.md` (after line 506, before line 508)

**Step 1: Add DSGE table to `api.md`**

After line 332 (end of Utility Functions table), append:

```markdown

---

## DSGE Models

Specify, solve, simulate, and estimate Dynamic Stochastic General Equilibrium models. See [DSGE Models](dsge.md) for the full guide.

### DSGE Specification and Solution

| Function | Description |
|----------|-------------|
| `@dsge begin ... end` | Parse DSGE model specification |
| `compute_steady_state(spec)` | Compute deterministic steady state |
| `linearize(spec)` | Linearize around steady state (Sims canonical form) |
| `solve(spec; method=:gensys)` | Solve rational expectations model |
| `gensys(Γ₀, Γ₁, C, Ψ, Π)` | Sims (2002) QZ decomposition solver |
| `blanchard_kahn(ld, spec)` | Blanchard-Kahn (1980) eigenvalue counting |
| `is_determined(sol)` | Check existence and uniqueness |
| `is_stable(sol)` | Check stability of solution |

### DSGE Simulation and Analysis

| Function | Description |
|----------|-------------|
| `simulate(sol, T)` | Stochastic simulation |
| `irf(sol, H)` | Analytical impulse responses |
| `fevd(sol, H)` | Forecast error variance decomposition |
| `solve_lyapunov(G1, impact)` | Unconditional covariance (Lyapunov equation) |
| `analytical_moments(sol; lags)` | Analytical variance and autocovariances |
| `perfect_foresight(spec; T_periods, shock_path)` | Deterministic transition path |

### DSGE Estimation

| Function | Description |
|----------|-------------|
| `estimate_dsge(spec, data, params; method)` | GMM estimation (IRF matching, Euler, SMM, analytical) |

### Occasionally Binding Constraints (OccBin)

| Function | Description |
|----------|-------------|
| `parse_constraint(expr, spec)` | Parse constraint expression |
| `occbin_solve(spec, constraint; ...)` | Piecewise-linear OccBin solution (1 or 2 constraints) |
| `occbin_irf(spec, constraint, shock_idx, H; ...)` | OccBin impulse responses |
```

**Step 2: Add DSGE types to `api_types.md`**

First, in the type hierarchy code block (lines 389-390), before the closing ` ``` `, insert:

```
DSGESpec{T}
LinearDSGE{T}
DSGESolution{T}
PerfectForesightPath{T}

AbstractDSGEModel
└── DSGEEstimation{T}

OccBinConstraint{T}
OccBinRegime{T}
OccBinSolution{T}
OccBinIRF{T}
```

Then after line 390 (after the closing ` ``` `), append:

```markdown

---

## DSGE Models

```@docs
DSGESpec
LinearDSGE
DSGESolution
PerfectForesightPath
DSGEEstimation
OccBinConstraint
OccBinRegime
OccBinSolution
OccBinIRF
```
```

**Step 3: Add DSGE functions to `api_functions.md`**

After line 506 (`balance_panel` closing ` ``` `) and before line 508 (`---`), insert:

```markdown

---

## DSGE Models

### Specification and Steady State

```@docs
compute_steady_state
linearize
```

### Solution Methods

```@docs
solve
gensys
blanchard_kahn
perfect_foresight
```

### Simulation and Analysis

```@docs
simulate
solve_lyapunov
analytical_moments
```

### Estimation

```@docs
estimate_dsge
```

### Occasionally Binding Constraints

```@docs
parse_constraint
occbin_solve
occbin_irf
```
```

**Step 4: Verify docs build**

Run: `julia --project=docs docs/make.jl`

Expected: Build succeeds. Warnings about missing docstrings are OK if types/functions lack docstrings — these will need `@doc` annotations in the source if Documenter warns. Check the build output carefully.

**Step 5: Commit**

```bash
git add docs/src/api.md docs/src/api_types.md docs/src/api_functions.md
git commit -m "docs: add DSGE types and functions to API reference"
```

---

### Task 5: Update Navigation in `docs/make.jl`

**Files:**
- Modify: `docs/make.jl:37-38`

**Step 1: Add DSGE to navigation**

After line 37 (`],` closing Panel Models) and before line 38 (`"Innovation Accounting"`), insert:

```julia
        "DSGE Models" => "dsge.md",
```

The result should look like:

```julia
        "Panel Models" => [
            "Panel VAR" => "pvar.md",
        ],
        "DSGE Models" => "dsge.md",
        "Innovation Accounting" => "innovation_accounting.md",
```

**Step 2: Verify full docs build**

Run: `julia --project=docs docs/make.jl`

Expected: Build succeeds with dsge.md now in navigation. Verify the generated HTML includes the DSGE Models page in the sidebar.

**Step 3: Commit**

```bash
git add docs/make.jl
git commit -m "docs: add DSGE Models to navigation in make.jl"
```
