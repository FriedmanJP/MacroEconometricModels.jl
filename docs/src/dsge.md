# [DSGE Models](@id dsge_page)

**MacroEconometricModels.jl** provides a complete toolkit for specifying, solving, simulating, and estimating Dynamic Stochastic General Equilibrium (DSGE) models. The package covers the full workflow from model definition through structural estimation, with seven solution methods spanning linear, higher-order, and global approaches.

- **Specification**: The `@dsge` macro provides a domain-specific language for writing equilibrium conditions with time-indexed variables
- **Steady State**: Analytical or numerical steady-state computation via NonlinearSolve.jl (`TrustRegion()` default) with built-in constrained solvers (Optim.jl, NLopt.jl) and optional JuMP backends
- **Linearization**: Automatic first-order approximation via numerical Jacobians in the Sims (2002) canonical form
- **Linear Solvers**: Three first-order solvers --- Gensys (Sims 2002), Blanchard-Kahn (1980), and Klein (2000) --- producing the state-space solution; see [Linear Solvers](@ref dsge_linear)
- **Nonlinear Methods**: Up to 3rd-order perturbation with Andreasen, Fernandez-Villaverde & Rubio-Ramirez (2018) pruning, Chebyshev collocation, policy function iteration, and value function iteration (with Howard steps and Anderson acceleration) for globally accurate policy functions; see [Nonlinear Methods](@ref dsge_nonlinear)
- **Constraints**: Perfect foresight paths, OccBin occasionally-binding constraints (Guerrieri & Iacoviello 2015), and constrained optimization via Optim.jl/NLopt.jl (built-in) with optional JuMP/Ipopt (NLP) and PATH (MCP) backends; see [Constraints](@ref dsge_constraints)
- **Estimation**: Four GMM-based methods (one-step, two-step, iterative, CU) for IRF matching, plus Bayesian estimation via SMC, SMC`` ^2 `` with two-stage delayed acceptance, and Random-Walk Metropolis-Hastings; see [Estimation](@ref dsge_estimation)
- **Simulation and IRFs**: Stochastic and pruned simulation, analytical and generalized impulse responses, FEVD, and unconditional moments via Lyapunov equation; see [Nonlinear Methods](@ref dsge_nonlinear)
- **Historical Decomposition**: Kalman smoother-based shock attribution for linear models, FFBSi particle smoother for nonlinear models, and Bayesian posterior bands; see [Historical Decomposition](@ref dsge_hd_page)
- **Dynare Replication**: 27-model replication suite validated against Dynare 6.5+ reference values for steady states, IRFs, variance decomposition, and theoretical moments; includes Smets & Wouters (2007) Bayesian estimation pipeline

All results integrate with `plot_result()` for interactive D3.js visualization and `report()` for publication-quality output.

```@setup dsge_overview
using MacroEconometricModels, Random
Random.seed!(42)
# Pre-define the RBC spec for reuse across blocks
_spec_rbc = @dsge begin
    parameters: β = 0.99, α = 0.36, δ = 0.025, ρ = 0.9, σ = 0.01
    endogenous: Y, C, K, A
    exogenous: ε_A
    Y[t] = A[t] * K[t-1]^α
    C[t] + K[t] = Y[t] + (1 - δ) * K[t-1]
    1 = β * (C[t] / C[t+1]) * (α * A[t+1] * K[t]^(α - 1) + 1 - δ)
    A[t] = ρ * A[t-1] + σ * ε_A[t]
    steady_state = begin
        A_ss = 1.0
        K_ss = (α * β / (1 - β * (1 - δ)))^(1 / (1 - α))
        Y_ss = K_ss^α
        C_ss = Y_ss - δ * K_ss
        [Y_ss, C_ss, K_ss, A_ss]
    end
end
_spec_rbc = compute_steady_state(_spec_rbc)
```

## Quick Start

**Recipe 1: Solve and plot IRFs**

```@example dsge_overview
# Specify a simple RBC model
spec = @dsge begin
    parameters: β = 0.99, α = 0.36, δ = 0.025, ρ = 0.9, σ = 0.01
    endogenous: Y, C, K, A
    exogenous: ε_A

    Y[t] = A[t] * K[t-1]^α
    C[t] + K[t] = Y[t] + (1 - δ) * K[t-1]
    1 = β * (C[t] / C[t+1]) * (α * A[t+1] * K[t]^(α - 1) + 1 - δ)
    A[t] = ρ * A[t-1] + σ * ε_A[t]

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

**Recipe 2: Second-order perturbation with pruning**

```@example dsge_overview
psol = perturbation_solver(spec; order=2)
Y_sim = simulate(psol, 1000)  # pruned simulation (Kim et al. 2008)
girf = irf(psol, 40; irf_type=:girf, n_draws=100)
nothing # hide
```

**Recipe 3: Estimate with GMM**

```julia
est = estimate_dsge(spec, Y_data, [:β, :α, :ρ];
                    method=:irf_matching, var_lags=4, irf_horizon=20)
report(est)
```

**Recipe 4: OccBin with ZLB**

```julia
constraint = parse_constraint(:(R[t] >= 0), spec)
occ_sol = occbin_solve(spec, constraint; shock_path=shocks)
occ_irf = occbin_irf(spec, constraint, 1, 40)
plot_result(occ_irf)
```

```@raw html
<iframe src="../assets/plots/occbin_irf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

**Recipe 5: Chebyshev projection**

```@example dsge_overview
proj = collocation_solver(spec; degree=5, grid=:tensor, max_iter=200)
y = evaluate_policy(proj, proj.steady_state[proj.state_indices])
err = max_euler_error(proj)
nothing # hide
```

**Recipe 6: Bayesian estimation via SMC``^2``**

```julia
using Distributions
result = estimate_dsge_bayes(spec, data, [0.99, 0.9, 0.01];
    priors=Dict(:β => Beta(99, 1), :ρ => Beta(5, 2), :σ => InverseGamma(2, 0.01)),
    method=:smc2, observables=[:y], n_smc=200, n_particles=100,
    solver=:projection, solver_kwargs=(degree=5,))
report(result)
```

**Recipe 7: Historical decomposition**

```@example dsge_overview
data_hd = simulate(sol, 100)
hd = historical_decomposition(sol, data_hd, [:Y, :C, :K, :A])
report(hd)
```

---

## Model Specification

The `@dsge` macro provides a domain-specific language for specifying DSGE models. It parses the model block into a `DSGESpec{T}` object containing equations, parameters, and variable declarations.

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
    A[t] = ρ * A[t-1] + σ * ε_A[t]
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

For models where the steady state has a closed-form solution, specify it in a `steady_state = begin ... end` block. The block must return a vector matching the endogenous variable ordering:

```@example dsge_overview
spec = @dsge begin
    parameters: β = 0.99, α = 0.36, δ = 0.025, ρ = 0.9, σ = 0.01
    endogenous: Y, C, K, A
    exogenous: ε_A

    Y[t] = A[t] * K[t-1]^α
    C[t] + K[t] = Y[t] + (1 - δ) * K[t-1]
    1 = β * (C[t] / C[t+1]) * (α * A[t+1] * K[t]^(α - 1) + 1 - δ)
    A[t] = ρ * A[t-1] + σ * ε_A[t]

    steady_state = begin
        A_ss = 1.0
        K_ss = (α * β / (1 - β * (1 - δ)))^(1 / (1 - α))
        Y_ss = K_ss^α
        C_ss = Y_ss - δ * K_ss
        [Y_ss, C_ss, K_ss, A_ss]
    end
end
spec = compute_steady_state(spec)
nothing # hide
```

When the `steady_state` block is provided, `compute_steady_state` (or `solve`) uses it directly and validates the result against the equations. The analytical path is faster and avoids numerical convergence issues, but the user is responsible for correctness --- the validator checks that ``\|f(\bar{y})\| < 10^{-10}``.

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

### Advanced: Explicit Solver Backends

For large-scale problems or complementarity formulations, JuMP-based backends provide additional power:

```julia
# Ipopt (NLP): handles general nonlinear constraints
import JuMP, Ipopt
spec = compute_steady_state(spec; constraints=[bound], solver=:ipopt)

# PATH (MCP): natural for complementarity problems (e.g., ZLB)
import JuMP, PATHSolver
spec = compute_steady_state(spec; constraints=[bound], solver=:path)
```

!!! note "Solver Selection Guide"
    **Built-in solvers** (no extra packages): `:nonlinearsolve` for unconstrained, `:optim` for box constraints, `:nlopt` for nonlinear inequality constraints. **JuMP solvers** (require `import`): `:ipopt` for large-scale NLP, `:path` for complementarity problems. The solver is auto-detected from constraint types --- override with the `solver` keyword. For full details, see [Constraints](@ref dsge_constraints).

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
| `spec` | `DSGESpec{T}` | Back-reference to specification |

---

## Complete Example

This example specifies, solves, and analyzes a full RBC model using the core functions covered on this page:

```@example dsge_overview
# Verify steady state (spec defined in Quick Start)
report(spec)
```

```@example dsge_overview
# Linearize and inspect the canonical form matrices
ld = linearize(spec)

# Solve with default Gensys method
sol = solve(spec)

# IRFs and FEVD
result = irf(sol, 40)
report(result)
```

```julia
plot_result(result)
```

```@raw html
<iframe src="../assets/plots/dsge_irf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The `spec` object stores the parsed model. `linearize` produces the Sims (2002) canonical form ``(\Gamma_0, \Gamma_1, C, \Psi, \Pi)``. `solve` dispatches to the Gensys algorithm and returns a `DSGESolution` with the state-space representation ``y_t = G_1 y_{t-1} + C + \text{impact} \cdot \varepsilon_t``. The IRF and FEVD functions operate on this solution. For higher-order or global solutions, see [Nonlinear Methods](@ref dsge_nonlinear).

---

## Dynare Replication Suite

The package includes a 27-model replication suite that validates solutions against [Dynare](https://www.dynare.org/) 6.5+ reference values. The reference `.mod` files come from [Johannes Pfeifer's DSGE\_mod collection](https://github.com/JohannesPfeifer/DSGE_mod), a widely-used repository of Dynare model files for textbook and published DSGE models.

Each replication script specifies the model using `@dsge` (or programmatic `DSGESpec` construction for large models), solves it, and compares the results against Dynare's `.mat` output for:

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

5. **Constrained steady state**: Box constraints (variable bounds) are handled by NonlinearSolve.jl without additional dependencies. Nonlinear inequality constraints require `import JuMP, Ipopt`. PATH MCP requires `import JuMP, PATHSolver`.

---

## References

- Sims, C. A. (2002). Solving Linear Rational Expectations Models.
  *Computational Economics*, 20(1-2), 1-20. [DOI](https://doi.org/10.1023/A:1020517101123)

- Blanchard, O. J., & Kahn, C. M. (1980). The Solution of Linear Difference Models under Rational Expectations.
  *Econometrica*, 48(5), 1305-1311. [DOI](https://doi.org/10.2307/1912186)

- Pfeifer, J. (2023). DSGE\_mod: A Collection of Dynare Models.
  [https://github.com/JohannesPfeifer/DSGE\_mod](https://github.com/JohannesPfeifer/DSGE_mod)

- Smets, F., & Wouters, R. (2007). Shocks and Frictions in US Business Cycles: A Bayesian DSGE Approach.
  *American Economic Review*, 97(3), 586-606. [DOI](https://doi.org/10.1257/aer.97.3.586)
