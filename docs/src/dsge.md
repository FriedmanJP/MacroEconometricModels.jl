# [DSGE Models](@id dsge_page)

Dynamic Stochastic General Equilibrium (DSGE) models describe an economy as the equilibrium outcome of optimizing agents facing stochastic shocks. **MacroEconometricModels.jl** covers the full workflow: the `@dsge` macro parses equilibrium conditions written with time-indexed variables, `compute_steady_state` locates the deterministic steady state, `linearize` produces the Sims (2002) canonical form, and `solve` dispatches to one of eight solution algorithms. Impulse responses, variance decompositions, simulation, historical decomposition, and structural estimation all operate on the resulting solution object.

This page owns the three stages every representative-agent model shares --- specification, steady state, and linearization --- and routes to the child page that owns each solution and estimation method. Those children divide by solution class: linear first-order solvers, higher-order and global methods, and occasionally binding constraints. Models that abandon the representative agent altogether --- heterogeneous-agent, overlapping-generations, and continuous-time --- bypass this pipeline entirely and are grouped under the [Heterogeneity & Continuous Time](@ref dsge_heterogeneity) sub-hub. A 24-model replication suite validates every stage against Dynare 6.5+.

All results integrate with `plot_result()` for interactive D3.js visualization and `report()` for publication-quality output.

```@setup dsge_overview
using MacroEconometricModels, Random
Random.seed!(42)
```

## Quick Start

Specify an RBC model, solve it with the default Gensys algorithm, and trace the impulse responses to a technology shock:

```@example dsge_overview
spec = @dsge begin
    parameters: β = 0.99, α = 0.36, δ = 0.025, ρ = 0.9, σ = 0.01
    endogenous: Y, C, K, A
    exogenous: ε_A

    Y[t] = A[t] * K[t-1]^α
    C[t] + K[t] = Y[t] + (1 - δ) * K[t-1]
    1 = β * (C[t] / C[t+1]) * (α * A[t+1] * K[t]^(α - 1) + 1 - δ)
    A[t] = A[t-1]^ρ * exp(σ * ε_A[t])

    steady_state = begin
        A_ss = 1.0
        K_ss = (α * β / (1 - β * (1 - δ)))^(1 / (1 - α))
        Y_ss = K_ss^α
        C_ss = Y_ss - δ * K_ss
        [Y_ss, C_ss, K_ss, A_ss]
    end
end
spec = compute_steady_state(spec)

sol = solve(spec)
result = irf(sol, 40)
```

```julia
plot_result(result)
```

```@raw html
<iframe src="../assets/plots/dsge_irf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## Choosing a Method

The economic question determines the solution class, and the solution class determines the child page:

| Feature needed | Recommended | Why |
|----------------|-------------|-----|
| Business-cycle IRFs and moments | [Linear Solvers](@ref dsge_linear) | Certainty equivalence suffices |
| Determinacy diagnostics | [Linear Solvers](@ref dsge_linear) | Explicit eigenvalue counting |
| Risk premia, precautionary saving | [Nonlinear Methods](@ref dsge_nonlinear) | Curvature survives the approximation |
| Accuracy far from steady state | [Nonlinear Methods](@ref dsge_nonlinear) | Global solution on a grid |
| Zero lower bound, borrowing limits | [Constraints](@ref dsge_constraints) | Piecewise-linear regime switching |
| Deterministic transition paths | [Constraints](@ref dsge_constraints) | Newton solve over the full path |
| Structural parameters from data | [Estimation](@ref dsge_estimation) | GMM and Bayesian samplers |
| Shock attribution over history | [Historical Decomposition](@ref dsge_hd_page) | Smoother-based additive identity |
| Wealth distribution and MPC heterogeneity | [Heterogeneous Agents](@ref dsge_ha) | Distribution is a state variable |
| Life-cycle and demographic structure | [Overlapping Generations](@ref dsge_olg) | Finite horizons break Ricardian equivalence |
| HJB and Kolmogorov-Forward formulation | [Continuous Time](@ref dsge_continuous) | Sparse finite differences, no simulation |
| Lumpy plant investment | [Heterogeneous Agents](@ref dsge_ha) | Khan–Thomas (2008) `FirmSystem` |
| Bank net-worth distribution | [Heterogeneous Agents](@ref dsge_ha) | Bewley Banks `IntermediarySystem` |

### `solve` Methods

`solve(spec; method=...)` is the single entry point for the representative-agent solvers. Seven algorithms are available:

| `method` | Class | Algorithm | Page |
|----------|-------|-----------|------|
| `:gensys` | Linear | Sims (2002) QZ decomposition (default) | [Linear Solvers](@ref dsge_linear) |
| `:blanchard_kahn` | Linear | Blanchard & Kahn (1980) eigenvalue counting | [Linear Solvers](@ref dsge_linear) |
| `:klein` | Linear | Klein (2000) generalized Schur decomposition | [Linear Solvers](@ref dsge_linear) |
| `:perturbation` | Higher-order | Schmitt-Grohe & Uribe (2004), orders 1--3 with pruning | [Nonlinear Methods](@ref dsge_nonlinear) |
| `:projection` | Global | Chebyshev collocation (Judd 1998) | [Nonlinear Methods](@ref dsge_nonlinear) |
| `:pfi` | Global | Policy function iteration / Euler-equation time iteration (Coleman 1990) | [Nonlinear Methods](@ref dsge_nonlinear) |
| `:vfi` | Global | Bellman value-function iteration (Stokey–Lucas–Prescott / Howard) | [Nonlinear Methods](@ref dsge_nonlinear) |
| `:perfect_foresight` | Deterministic | Newton solver for perfect-foresight paths | [Constraints](@ref dsge_constraints) |

`solve(spec; method=:vfi)` requires the same Bellman keywords as `vfi_solver` (`utility`, `beta`, `transition`, `control_bounds`). Euler-only specs must use `:pfi`. Specs that carry an agent kind dispatch on the type, never on the key name: `HouseholdSystem` uses `method=:ssj` by default (one or more named populations); `to_spec` wrappers for DCEGM, life-cycle OLG, continuous-time households, Khan–Thomas plants, and Bewley Banks call the matching family solver. Blanchard perpetual-youth residuals are `NoAgents` and use `:gensys` like any other representative-agent system.

---

## Child Pages

- [Linear Solvers](@ref dsge_linear) --- Gensys, Blanchard-Kahn, and Klein first-order solutions, determinacy conditions, simulation, IRFs, FEVD, and unconditional moments
- [Nonlinear Methods](@ref dsge_nonlinear) --- second- and third-order perturbation with pruning, generalized IRFs, Chebyshev collocation, policy function iteration, value-function iteration, and analytical moments
- [Constraints](@ref dsge_constraints) --- perfect-foresight paths, constrained steady states, and OccBin occasionally binding constraints (Guerrieri & Iacoviello 2015)
- [Estimation](@ref dsge_estimation) --- GMM IRF matching (one-step, two-step, iterative, CU) and Bayesian estimation via SMC, SMC``^2``, and Random-Walk Metropolis-Hastings
- [Historical Decomposition](@ref dsge_hd_page) --- Kalman-smoother shock attribution for linear models, FFBSi particle smoother for nonlinear models, and posterior bands
- [Heterogeneity & Continuous Time](@ref dsge_heterogeneity) --- sub-hub fronting the three families that drop the representative agent: heterogeneous-agent, overlapping-generations, and continuous-time models

---

## Model Specification

The `@dsge` macro provides a domain-specific language for specifying DSGE models. It parses the model block into a `ModelSpec{T,NoAgents}` object containing named equations, parameters, and variable declarations. Unlabeled `var[t] = ...` sets both the equation name and `defines` to `var`. A lead `x[t+1]` *is* the rational expectation of `x`.

### Syntax

```@example dsge_overview
spec_demo = @dsge begin
    parameters: β = 0.99, α = 0.36, δ = 0.025, ρ = 0.9, σ = 0.01
    endogenous: Y, C, K, A
    exogenous: ε_A

    # Equations (one per endogenous variable)
    Y[t] = A[t] * K[t-1]^α
    C[t] + K[t] = Y[t] + (1 - δ) * K[t-1]
    1 = β * (C[t] / C[t+1]) * (α * A[t+1] * K[t]^(α - 1) + 1 - δ)
    A[t] = A[t-1]^ρ * exp(σ * ε_A[t])
end
```

### Blocks

| Block | Syntax | Description |
|-------|--------|-------------|
| `parameters:` | `name = value, ...` | Calibrated parameters with default values |
| `endogenous:` | `var1, var2, ...` | Endogenous variable names |
| `exogenous:` | `shock1, shock2, ...` | Exogenous shock names |
| `steady_state` | `= begin ... [y_ss] end` | Optional analytical steady-state function (must return vector) |
| `varnames:` | `["Label 1", "Label 2", ...]` | Optional display labels for variables |
| `constraint:` | `var[t] >= bound` | Occasionally binding regime on the defining equation |
| `clock:` | `discrete` or `continuous` | Sets `spec.ir.clock`. Continuous-time households use `to_spec` on [`CTAiyagari`](@ref) |
| `horizon:` | `infinite`, `finite`, `ages`, or `perpetual_youth` | Sets `spec.ir.horizon`. Extra `J=`, `retire=`, `survival=`, `earnings=` keys are stored; life-cycle models use `to_spec` on [`LifeCycleOLG`](@ref) |
| `discrete:` / `absorbing:` | option names | Stored as IR declarations. Discrete-continuous choice uses [`dcegm_retirement_model`](@ref) |

### Time Subscripts

| Notation | Meaning |
|----------|---------|
| `var[t]` | Current period value |
| `var[t-1]` | One-period lag (predetermined variable) |
| `var[t+1]` | One-period lead (forward-looking / jump variable) |

Variables with `[t+1]` subscripts generate expectation errors in the Sims (2002) canonical form. The number of forward-looking equations determines the dimension of the ``\Pi`` matrix and, via the Blanchard-Kahn (1980) condition, the number of unstable eigenvalues required for determinacy.

!!! note "Technical Note"
    Equations are written as `LHS = RHS` where both sides can contain endogenous variables at different time subscripts. The `@dsge` macro rearranges each equation into residual form ``f(y_t, y_{t-1}, y_{t+1}, \varepsilon_t, \theta) = 0`` via `LHS - RHS`. The number of equations must equal the number of endogenous variables. Timing convention: ``K_{t}`` chosen at time ``t`` appears as `K[t]`; ``K_{t-1}`` (beginning-of-period capital) as `K[t-1]`.

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

The **steady state** ``\bar{y}`` satisfies the equilibrium system in the absence of shocks:

```math
f(\bar{y}, \bar{y}, \bar{y}, 0, \theta) = 0
```

where:
- ``\bar{y}`` is the ``n \times 1`` vector of endogenous variables at the steady state
- ``\theta`` is the vector of deep structural parameters
- ``f`` is the system of ``n`` equilibrium conditions

For the RBC model above, the analytical steady state is:

```math
\bar{A} = 1, \quad \bar{K} = \left(\frac{\alpha\beta}{1 - \beta(1-\delta)}\right)^{\frac{1}{1-\alpha}}, \quad \bar{Y} = \bar{K}^\alpha, \quad \bar{C} = \bar{Y} - \delta\bar{K}
```

### Numerical Computation

`compute_steady_state` uses NonlinearSolve.jl to solve the system ``f(\bar{y}, \bar{y}, \bar{y}, 0, \theta) = 0``. The default algorithm is `TrustRegion()`, which is robust to poor starting points. Box constraints (e.g., non-negativity) are handled natively via NonlinearSolve's bounded problem formulation.

```@example dsge_overview
spec = compute_steady_state(spec)
report(spec)
```

The solver converges to the steady state from a default initial guess of ones. For models with multiple equilibria, providing a good starting point via `initial_guess` avoids convergence to an economically irrelevant solution.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `initial_guess` | `Vector` | `nothing` | Starting point (default: ones) |
| `method` | `Symbol` | `:auto` | `:auto` (NonlinearSolve) or `:analytical` |
| `algorithm` | `Any` | `TrustRegion()` | NonlinearSolve.jl algorithm (ignored for JuMP solvers) |

### Analytical Steady State

For models where the steady state has a closed-form solution, specify it in a `steady_state = begin ... end` block. The block must return a vector matching the endogenous variable ordering, as in the Quick Start model above. When the block is provided, `compute_steady_state` (or `solve`) uses it directly. The analytical path is faster and avoids numerical convergence issues, at the cost of shifting correctness onto the modeller.

!!! warning "The analytical path is not validated"
    The numerical route throws a `DSGESolveError` when ``\|f(\bar{y})\|_\infty > 10^{-6}``; an analytical `steady_state` block gets **no such check**. Linearizing around a point that is not a steady state produces a system whose constant is silently taken to be zero, so the first-order solution looks healthy while every higher-order and global solver expands around the wrong point. Verify it yourself --- every entry must be zero:

    ```julia
    ss = spec.steady_state
    [f(ss, ss, ss, zeros(spec.n_exog), spec.param_values) for f in spec.residual_fns]
    ```

The usual way to fail that check is declaring ``\bar{A} = 1`` for a technology process written in levels as `A[t] = ρ*A[t-1] + σ*ε_A[t]`, whose actual steady state is 0. Writing it as `A[t] = A[t-1]^ρ * exp(σ*ε_A[t])` --- the log-AR(1)-in-levels form used by the RBC model throughout these pages --- makes ``\bar{A} = 1`` exact while leaving the first-order dynamics identical.

### Constrained Steady State

For models with variable bounds --- such as a zero lower bound on the nominal interest rate or non-negativity of consumption --- box constraints work out of the box. NonlinearSolve.jl solves the unconstrained system first; if bounds are violated, the solver auto-escalates to Optim.jl `Fminbox(LBFGS())`:

```julia
# ZLB: interest rate cannot go below zero
bound = variable_bound(:R, lower=0.0)

# Solve constrained steady state (auto-escalates to Optim.jl if bounds bind)
spec = compute_steady_state(spec; constraints=[bound])
```

For nonlinear inequality constraints, NLopt.jl `LD_SLSQP` is the default solver --- no additional packages required:

```julia
# Debt-to-GDP ceiling: fn(y, ...) <= 0
debt_limit = nonlinear_constraint(
    (y, y_lag, y_lead, e, theta) -> y[debt_idx] / y[gdp_idx] - 0.6;
    label="Debt-to-GDP <= 60%"
)

spec = compute_steady_state(spec; constraints=[bound, debt_limit])
```

For large-scale problems or complementarity formulations, JuMP-based backends provide additional power:

```julia
# Ipopt (NLP): handles general nonlinear constraints (JuMP + Ipopt are built in)
spec = compute_steady_state(spec; constraints=[bound], solver=:ipopt)

# PATH (MCP): natural for complementarity problems (e.g., ZLB); optional dependency
import PATHSolver
spec = compute_steady_state(spec; constraints=[bound], solver=:path)
```

!!! note "Solver Selection Guide"
    **Built-in solvers** (no extra packages): `:nonlinearsolve` for unconstrained, `:optim` for box constraints, `:nlopt` for nonlinear inequality constraints, and `:ipopt` (JuMP + Ipopt) for large-scale NLP. **Optional:** `:path` for complementarity problems requires the PATHSolver package. The solver is auto-detected from constraint types --- override with the `solver` keyword. For full details, see [Constraints](@ref dsge_constraints).

---

## Linearization

`linearize` computes a first-order Taylor expansion around the steady state using numerical Jacobians (central differences). It produces the Sims (2002) canonical form:

```math
\Gamma_0 \, y_t = \Gamma_1 \, y_{t-1} + C + \Psi \, \varepsilon_t + \Pi \, \eta_t
```

where:
- ``y_t`` is the ``n \times 1`` vector of endogenous variables (deviations from steady state)
- ``\Gamma_0`` is the ``n \times n`` coefficient matrix on current-period variables
- ``\Gamma_1`` is the ``n \times n`` coefficient matrix on lagged variables
- ``C`` is the ``n \times 1`` constant vector
- ``\Psi`` is the ``n \times n_{shocks}`` shock loading matrix
- ``\Pi`` is the ``n \times n_{expect}`` expectation error selection matrix
- ``\varepsilon_t`` is the vector of exogenous shocks
- ``\eta_t = y_t - E_{t-1}[y_t]`` is the vector of expectation errors for forward-looking variables

```@example dsge_overview
ld = linearize(spec)
nothing # hide
```

The matrix pair ``(\Gamma_0, \Gamma_1)`` defines a generalized eigenvalue problem whose solution governs the model dynamics. The three [Linear Solvers](@ref dsge_linear) --- Gensys, Blanchard-Kahn, and Klein --- each decompose this pencil to extract the stable state-space representation.

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
| `spec` | `ModelSpec{T}` | Back-reference to specification |

---

## Dynare Replication Suite

The package includes a 24-model replication suite that validates solutions against [Dynare](https://www.dynare.org/) 6.5+ reference values. The reference `.mod` files come from [Johannes Pfeifer's DSGE\_mod collection](https://github.com/JohannesPfeifer/DSGE_mod), a widely-used repository of Dynare model files for textbook and published DSGE models.

Each replication script specifies the model using `@dsge` (or programmatic `ModelSpec` construction for large models), solves it, and compares the results against Dynare's `.mat` output for:

- **Steady state** — variable-by-variable comparison (typical tolerance: ``10^{-6}``)
- **Impulse response functions** — horizon-by-horizon comparison (typical tolerance: ``10^{-4}``)
- **Variance decomposition** — asymptotic FEVD proportions
- **Theoretical moments** — unconditional variance and lag-1 autocorrelation

The suite is organized by complexity tier:

| Tier | Models | Description |
|------|--------|-------------|
| 1 | RBC Baseline, Hansen (1985), Collard (2001), Fernandez-Villaverde et al. (2007), Gali (2008 Ch.2, 2015 Ch.2--3), SGU (2003), Ascari & Sbordone (2014), Aguiar & Gopinath (2007), Kiyotaki & Moore (1997), McCandless (2008 Ch.9, Ch.13), RBC Capital Stock | Standard textbook models (order 1--2) |
| 2 | Jermann (1998), SGU (2004), Born & Pfeifer (2014), Basu & Bundick (2017) | Asset pricing and nonlinear dynamics (order 2--3) |
| 3 | Solow Transition, Ramsey-Cass-Koopmans | Perfect foresight deterministic transitions |
| 4 | Guerrieri & Iacoviello (2015) RBC/NK | OccBin occasionally binding constraints |
| 5 | Smets & Wouters (2007) | Medium-scale NK (40 vars, 7 shocks, `model(linear)`) |
| 6 | RBC News Shock | Anticipated shocks |

### Running the Suite

The replication scripts live in `test/dynare_replication/`. Each script is self-contained and prints PASS/FAIL for each comparison:

```julia
include("test/dynare_replication/tier1_rbc_baseline.jl")
include("test/dynare_replication/tier5_smets_wouters_2007.jl")
```

Reference `.mat` files in `test/dynare_replication/dynare_results/` are generated by `run_dynare_reference.m` using Octave + Dynare 6.5+. Regenerate after updating the DSGE\_mod source:

```bash
cd test/dynare_replication && octave --no-gui run_dynare_reference.m
```

### Estimation Replication

The estimation pipeline is validated separately in `test/dynare_replication/estimation_sw07.jl`, which generates synthetic data from the Smets & Wouters (2007) model at posterior mode and runs Bayesian estimation via SMC. A toy-model suite tests all three samplers (SMC, RWMH, SMC``^2``), Bayes factors, and posterior IRF/FEVD/simulation.

!!! note "Dynare Comparison Caveats"
    Models where Dynare uses `loglinear` compute second moments in log-deviation space, while our perturbation solver works in level-deviation space. At order ≥ 2, the Hessians differ between coordinate systems, so variance and autocorrelation values are not directly comparable. Steady states and IRFs match exactly regardless of the coordinate system. Variance decomposition proportions from order-1 analysis match across both spaces.

---

## Common Pitfalls

1. **Steady-state validation failure**: When providing an analytical `steady_state` block, the validator checks ``\|f(\bar{y})\| < 10^{-10}``. A common cause of failure is mismatched variable ordering --- the returned vector must match the `endogenous:` declaration order exactly.

2. **Equation count mismatch**: The number of equations must equal the number of endogenous variables. Missing an equilibrium condition or double-counting a definition produces `DimensionMismatch`. Each equation is written as `LHS = RHS`; the parser rearranges to residual form automatically.

3. **Timing convention confusion**: ``K_t`` chosen at time ``t`` is written `K[t]`. Beginning-of-period capital (predetermined) is `K[t-1]`. A forward-looking Euler equation uses `C[t+1]`. Misplacing a time subscript silently changes the ``\Gamma_0``/``\Gamma_1`` structure and can cause indeterminacy.

4. **Numerical steady state converges to wrong equilibrium**: For models with multiple equilibria, the default initial guess (vector of ones) may converge to an economically irrelevant solution. Provide `initial_guess` close to the desired equilibrium, or use the analytical `steady_state` block.

5. **Constrained steady state**: Box constraints (variable bounds) are handled by NonlinearSolve.jl. Nonlinear inequality constraints use the built-in JuMP + Ipopt backend (`solver=:ipopt`). PATH MCP requires the optional PATHSolver package (`import PATHSolver`).

---

## References

- Sims, C. A. (2002). Solving Linear Rational Expectations Models.
  *Computational Economics*, 20(1--2), 1--20. [DOI](https://doi.org/10.1023/A:1020517101123)

- Blanchard, O. J., & Kahn, C. M. (1980). The Solution of Linear Difference Models under Rational Expectations.
  *Econometrica*, 48(5), 1305-1311. [DOI](https://doi.org/10.2307/1912186)

- Pfeifer, J. (2023). DSGE\_mod: A Collection of Dynare Models.
  [https://github.com/JohannesPfeifer/DSGE\_mod](https://github.com/JohannesPfeifer/DSGE_mod)

- Smets, F., & Wouters, R. (2007). Shocks and Frictions in US Business Cycles: A Bayesian DSGE Approach.
  *American Economic Review*, 97(3), 586-606. [DOI](https://doi.org/10.1257/aer.97.3.586)
