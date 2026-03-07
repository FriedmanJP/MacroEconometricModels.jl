# [Linear Solution Methods](@id dsge_linear)

First-order linear rational expectations solutions produce the **state-space representation** that is the workhorse of DSGE analysis --- impulse responses, variance decompositions, simulation, and moment matching all flow from this representation. MacroEconometricModels.jl provides three solver implementations (Gensys, Blanchard-Kahn, and Klein) that share the unified `solve()` interface and return the same `DSGESolution{T}` output type. For model specification and linearization, see [DSGE Models](@ref dsge_page).

```@setup dsge_linear
using MacroEconometricModels, Random
Random.seed!(42)
```

## Quick Start

**Recipe 1: Solve with Gensys and check determinacy**

```@example dsge_linear
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

sol = solve(spec; method=:gensys)
report(sol)
```

**Recipe 2: Impulse responses and FEVD**

```@example dsge_linear
result = irf(sol, 40)
decomp = fevd(sol, 40)
nothing # hide
```

```julia
plot_result(result)
plot_result(decomp)
```

**Recipe 3: Unconditional moments via Lyapunov equation**

```julia
Σ = solve_lyapunov(sol.G1, sol.impact)
m = analytical_moments(sol; lags=2)
```

---

## Unified Solver Interface

The `solve(spec; method=:gensys)` function is the single entry point for all DSGE solution methods. The `method` keyword selects the algorithm:

| `method` | Algorithm |
|----------|-----------|
| `:gensys` | Sims (2002) QZ decomposition (default) |
| `:blanchard_kahn` | Blanchard & Kahn (1980) eigenvalue counting |
| `:klein` | Klein (2000) generalized Schur decomposition |
| `:perturbation` | Higher-order perturbation (Schmitt-Grohe & Uribe 2004) |
| `:projection` | Chebyshev collocation (Judd 1998) |
| `:pfi` | Policy function iteration (Coleman 1990) |
| `:perfect_foresight` | Deterministic Newton solver |

This page covers the three **linear** (first-order) methods: `:gensys`, `:blanchard_kahn`, and `:klein`. All three solve the same linearized system and return a `DSGESolution{T}`. For higher-order perturbation, global projection, and policy function iteration, see [Nonlinear Methods](@ref dsge_nonlinear).

### Solver Comparison

| Method | Algorithm | Singularity | Speed | Reference |
|--------|-----------|-------------|-------|-----------|
| `:gensys` | QZ decomposition | Handles singular ``\Gamma_0`` | Fast | Sims (2002) |
| `:blanchard_kahn` | QZ + eigenvalue counting | Requires invertible ``\Gamma_0`` | Fast | Blanchard & Kahn (1980) |
| `:klein` | Generalized Schur | Handles singular ``\Gamma_0`` | Fast | Klein (2000) |

---

## Choosing a Solver

The choice of solver depends on the model structure and the analysis objective. For standard first-order work, all three linear solvers produce identical solutions (up to numerical precision). The following table provides guidance for broader method selection:

| Feature needed | Recommended | Why |
|----------------|-------------|-----|
| Standard IRFs from linearized model | `:gensys` | Robust default, handles singularity |
| Eigenvalue diagnostics | `:blanchard_kahn` | Direct eigenvalue counting |
| Predetermined variable counting | `:klein` | Automatic state/control partition |
| Risk premia, welfare costs | `:perturbation` (order=2) | Captures precautionary effects; see [Nonlinear Methods](@ref dsge_nonlinear) |
| Large deviations, global accuracy | `:projection` or `:pfi` | Globally accurate policy; see [Nonlinear Methods](@ref dsge_nonlinear) |
| ZLB, occasionally binding | `:perfect_foresight` + OccBin | Respects inequality constraints; see [Constraints](@ref dsge_constraints) |

For most applications, `:gensys` is the recommended default. Switch to `:blanchard_kahn` when you need direct eigenvalue diagnostics, or to `:klein` when the state/control partition is central to the analysis.

---

## Determinacy and the Blanchard-Kahn Condition

All three linear solvers share a common foundation: the **linearized canonical form** and the eigenvalue-based **determinacy** check. The `linearize` function produces the Sims (2002) canonical representation:

```math
\Gamma_0 \, y_t = \Gamma_1 \, y_{t-1} + C + \Psi \, \varepsilon_t + \Pi \, \eta_t
```

where:
- ``y_t`` is the ``n \times 1`` vector of endogenous variables (log-deviations from steady state)
- ``\Gamma_0, \Gamma_1`` are ``n \times n`` coefficient matrices on current and lagged variables
- ``C`` is the ``n \times 1`` constant vector (zero when linearized around steady state)
- ``\Psi`` is the ``n \times n_\varepsilon`` shock loading matrix
- ``\Pi`` is the ``n \times n_\eta`` expectation error selection matrix
- ``\varepsilon_t`` is the ``n_\varepsilon \times 1`` vector of exogenous shocks
- ``\eta_t`` is the ``n_\eta \times 1`` vector of expectation errors (``\eta_t = y_t - E_{t-1}[y_t]`` for forward-looking variables)

The solvers reduce this system to the **state-space solution**:

```math
y_t = G_1 \, y_{t-1} + C_{\text{sol}} + \text{impact} \cdot \varepsilon_t
```

where:
- ``G_1`` is the ``n \times n`` state transition matrix
- ``C_{\text{sol}}`` is the ``n \times 1`` solution constant vector
- ``\text{impact}`` is the ``n \times n_\varepsilon`` shock impact matrix

The **generalized eigenvalues** of the matrix pencil ``(\Gamma_0, \Gamma_1)`` determine the system's stability properties. The Blanchard-Kahn (1980) condition requires that the number of eigenvalues with modulus greater than one (unstable roots) equals the number of forward-looking (jump) variables ``n_\eta``. The `eu` vector in the solution reports the outcome:

| `eu` value | Interpretation |
|------------|----------------|
| `[1, 1]` | Existence and uniqueness (determinate) |
| `[1, 0]` | Existence but multiple solutions (indeterminate) |
| `[0, 0]` | No stable solution (explosive) |

```@example dsge_linear
sol = solve(spec; method=:gensys)
sol.eu                          # [1, 1] = determinate
is_determined(sol)              # true
maximum(abs.(sol.eigenvalues))  # largest eigenvalue modulus
```

!!! warning "Common cause of indeterminacy"
    In New Keynesian models, the Taylor principle requires the inflation response coefficient ``\phi_\pi > 1``. With ``\phi_\pi < 1``, the model is indeterminate. If `eu = [1, 0]`, check the policy rule coefficients first.

### Diagnosing Indeterminacy

When `is_determined(sol)` returns `false`, inspect the eigenvalue decomposition to understand why. Count the number of stable and unstable eigenvalues and compare with the number of forward-looking variables:

```@example dsge_linear
sol = solve(spec; method=:blanchard_kahn)
eigenvalues = sol.eigenvalues
n_stable = count(e -> abs(e) < 1.0, eigenvalues)
n_unstable = length(eigenvalues) - n_stable
n_forward = sol.spec.n_expect

# BK condition: n_unstable must equal n_forward
n_unstable == n_forward   # true for determinacy
```

The three outcomes are:

- **``n_{\text{unstable}} = n_{\text{forward}}``**: Exactly one stable solution exists (determinate). This is the desired outcome.
- **``n_{\text{unstable}} < n_{\text{forward}}``**: More forward-looking variables than unstable roots. The system is **indeterminate** --- multiple stable paths exist. Typical fixes: strengthen the Taylor rule (increase ``\phi_\pi``), add fiscal rules, or tighten expectations anchoring.
- **``n_{\text{unstable}} > n_{\text{forward}}``**: More unstable roots than forward-looking variables. No stable solution exists --- all paths are **explosive**. Typical fixes: check parameter calibration (discount factor near unity, reasonable adjustment costs), verify equation timing, or reduce the number of predetermined variables.

---

## Gensys (Sims 2002)

The **Gensys** solver is the default method. It uses the QZ (generalized Schur) decomposition of the matrix pencil ``(\Gamma_0, \Gamma_1)`` and handles singular ``\Gamma_0`` matrices, making it suitable for models with static identities (e.g., ``Y_t = C_t + I_t``).

The `div` keyword sets the dividing line between stable and unstable eigenvalues. The default value of ``1.0 + 10^{-8}`` places the cutoff slightly above the unit circle, ensuring that borderline eigenvalues (exactly at unity) are treated as stable.

```@example dsge_linear
sol = solve(spec; method=:gensys)
```

Or calling the low-level function directly after linearization:

```@example dsge_linear
ld = linearize(spec)
result = gensys(ld.Gamma0, ld.Gamma1, ld.C, ld.Psi, ld.Pi; div=1.0+1e-8)
result.G1       # state transition matrix
result.impact   # shock impact matrix
result.eu       # [existence, uniqueness]
```

!!! note "Technical Note"
    Gensys reorders the Schur decomposition so stable eigenvalues (``|\lambda| < \text{div}``) come first. The solution matrices are then constructed from the stable block: ``G_1 = Z_1 \, S_{11}^{-1} \, T_{11} \, Z_1'`` and ``\text{impact} = Z_1 \, S_{11}^{-1} \, Q_1 \, \Psi``, where subscript 1 denotes the stable partition. The constant vector is ``C_{\text{sol}} = (I - G_1)^{-1} C``.

---

## Blanchard-Kahn (1980)

The **Blanchard-Kahn** method uses the same QZ decomposition but checks the BK condition directly: the number of unstable eigenvalues (``|\lambda| > 1``) must equal the number of forward-looking variables ``n_\eta``. It uses a strict unit circle cutoff (``|\lambda| > 1.0``) rather than Gensys's ``1 + 10^{-8}`` convention.

```@example dsge_linear
sol = solve(spec; method=:blanchard_kahn)
```

!!! warning "Singular ``\\Gamma_0``"
    The Blanchard-Kahn method requires an invertible ``\Gamma_0`` matrix. Models with static identities (e.g., ``Y_t = C_t + I_t``) that make ``\Gamma_0`` singular should use `:gensys` or `:klein` instead.

---

## Klein (2000)

The **Klein** method uses the generalized Schur decomposition with automatic **predetermined variable detection**: variables appearing with ``[t-1]`` subscripts are classified as states by scanning for non-zero columns in ``\Gamma_1``. The BK condition is checked as ``n_{\text{stable}} = n_{\text{predetermined}}``.

```@example dsge_linear
sol = solve(spec; method=:klein)
```

Klein's method handles singular ``\Gamma_0`` and uses a strict dividing line (``|\lambda| < 1.0``) for classifying stable eigenvalues.

!!! note "Technical Note"
    Klein's method detects predetermined variables automatically by scanning non-zero columns of ``\Gamma_1``. A variable ``j`` is classified as predetermined (state) if ``\|\Gamma_1[:, j]\| > 0``. The dividing line for stable eigenvalues is ``|\lambda| < 1.0`` (strict), compared to Gensys's default of ``1 + 10^{-8}``.

### Return Value (All First-Order Solvers)

All three linear solvers return a `DSGESolution{T}` with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `G1` | `Matrix{T}` | ``n \times n`` state transition matrix |
| `impact` | `Matrix{T}` | ``n \times n_\varepsilon`` shock impact matrix |
| `C_sol` | `Vector{T}` | ``n \times 1`` constant vector |
| `eu` | `Vector{Int}` | ``[\text{existence}, \text{uniqueness}]``: 1 = yes, 0 = no |
| `method` | `Symbol` | Solver used (`:gensys`, `:blanchard_kahn`, `:klein`) |
| `eigenvalues` | `Vector{ComplexF64}` | Generalized eigenvalues from QZ decomposition |
| `spec` | `DSGESpec{T}` | Back-reference to model specification |
| `linear` | `LinearDSGE{T}` | Linearized system (``\Gamma_0, \Gamma_1, C, \Psi, \Pi``) |

Accessor functions:

- `nvars(sol)` --- number of endogenous variables
- `nshocks(sol)` --- number of exogenous shocks
- `is_determined(sol)` --- `true` if `eu == [1, 1]`
- `is_stable(sol)` --- `true` if max eigenvalue modulus of ``G_1`` is less than 1

---

## Simulation

Stochastic forward simulation generates sample paths from the state-space solution:

```math
y_t = G_1 \, y_{t-1} + \text{impact} \cdot \varepsilon_t + C_{\text{sol}}
```

where ``\varepsilon_t \sim N(0, I_{n_\varepsilon})`` are i.i.d. standard normal shocks. The simulation returns **levels** (steady state plus deviations), not deviations alone.

```@example dsge_linear
sol = solve(spec)
Y = simulate(sol, 200)  # 200 x n_endog matrix of levels
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `shock_draws` | `Union{Nothing, Matrix}` | `nothing` | Pre-drawn shocks (``T \times n_\varepsilon``); draws from ``N(0,1)`` if `nothing` |
| `rng` | `AbstractRNG` | `default_rng()` | Random number generator |

The return value is a ``T_{\text{periods}} \times n_{\text{endog}}`` matrix of variable levels. For augmented models (those with auxiliary variables introduced by the `@dsge` macro to handle higher-order lags/leads), the output is automatically filtered to the original endogenous variables.

---

## Impulse Response Functions

**Analytical IRFs** for linear solutions exploit the state-space structure directly. The response of all variables to each shock at horizon ``h`` is:

```math
\Phi_h = G_1^{h-1} \cdot \text{impact}
```

where:
- ``\Phi_h`` is the ``n \times n_\varepsilon`` response matrix at horizon ``h``
- ``G_1^{h-1}`` is the ``(h-1)``-th matrix power of the state transition matrix
- ``\text{impact}`` is the shock impact matrix

Each column ``j`` of ``\Phi_h`` gives the response of all ``n`` endogenous variables to a one-standard-deviation shock to exogenous variable ``j``, measured ``h`` periods after impact.

```@example dsge_linear
result = irf(sol, 40)
```

```julia
plot_result(result)
```

The return type is `ImpulseResponse{T}` with field `.values` (``H \times n \times n_\varepsilon``), `.variables` (variable names), and `.shocks` (shock names). The result is directly compatible with `plot_result()` for interactive D3.js visualization.

---

## Forecast Error Variance Decomposition

**FEVD** decomposes the ``h``-step-ahead forecast error variance of each variable into the contribution of each structural shock:

```math
\text{FEVD}_{i,j}(h) = \frac{\sum_{s=0}^{h-1} [\Phi_s]_{i,j}^2}{\sum_{s=0}^{h-1} \sum_{k=1}^{n_\varepsilon} [\Phi_s]_{i,k}^2}
```

where:
- ``\text{FEVD}_{i,j}(h)`` is the fraction of variable ``i``'s ``h``-step forecast error variance due to shock ``j``
- ``[\Phi_s]_{i,j}`` is the response of variable ``i`` to shock ``j`` at horizon ``s``
- The denominator sums over all ``n_\varepsilon`` shocks to normalize to proportions

By construction, ``\sum_j \text{FEVD}_{i,j}(h) = 1`` for every variable ``i`` and horizon ``h``.

```@example dsge_linear
decomp = fevd(sol, 40)
```

```julia
plot_result(decomp)
```

The return type is `FEVD{T}` with fields `.decomposition` (raw cumulative squared IRFs) and `.proportions` (normalized shares). Compatible with `plot_result()`.

In the RBC model, the technology shock ``\varepsilon_A`` is the sole source of fluctuations, so ``\text{FEVD}_{i,A}(h) = 1`` for all variables and horizons. In multi-shock models (e.g., Smets & Wouters 2007), the FEVD reveals which shocks dominate business cycle fluctuations at different frequencies --- demand shocks typically dominate output at short horizons, while supply shocks dominate at longer horizons.

---

## Unconditional Moments

The **discrete Lyapunov equation** computes the unconditional covariance matrix ``\Sigma`` of the endogenous variables under the state-space representation:

```math
\Sigma = G_1 \, \Sigma \, G_1' + \text{impact} \cdot \text{impact}'
```

where:
- ``\Sigma`` is the ``n \times n`` unconditional variance-covariance matrix
- ``G_1`` is the state transition matrix (must be stable: all eigenvalues inside the unit circle)
- ``\text{impact} \cdot \text{impact}'`` is the innovation covariance

The equation is solved via Kronecker vectorization:

```math
\text{vec}(\Sigma) = (I_{n^2} - G_1 \otimes G_1)^{-1} \, \text{vec}(\text{impact} \cdot \text{impact}')
```

```@example dsge_linear
Σ = solve_lyapunov(sol.G1, sol.impact)  # n x n covariance matrix
```

The `analytical_moments` function extracts a moment vector in a format compatible with `autocovariance_moments(data, lags)`, enabling direct comparison between model-implied and data moments for GMM estimation:

```@example dsge_linear
m = analytical_moments(sol; lags=2)
```

The moment vector contains:
1. Upper triangle of the variance-covariance matrix: ``k(k+1)/2`` elements
2. Diagonal autocovariances ``\text{diag}(G_1^h \, \Sigma)`` at each lag ``h = 1, \ldots, \text{lags}``: ``k`` elements per lag

!!! note "Technical Note"
    `analytical_moments` extracts the upper triangle of ``\Sigma`` (``k(k+1)/2`` elements) followed by diagonal autocovariances at each lag. This format matches `autocovariance_moments(data, lags)`, enabling direct comparison between model-implied and data moments for GMM estimation (see [Estimation](@ref dsge_estimation)).

---

## Complete Example

This example combines all the linear solution tools: specification, solving with three methods, simulation, IRFs, FEVD, and unconditional moments.

```@example dsge_linear
# Solve with all three linear methods
sol_g = solve(spec; method=:gensys)
sol_bk = solve(spec; method=:blanchard_kahn)
sol_k = solve(spec; method=:klein)

# Simulate 200 periods
Y_sim = simulate(sol_g, 200)

# IRFs and FEVD
result = irf(sol_g, 40)
decomp = fevd(sol_g, 40)

# Unconditional moments
Σ = solve_lyapunov(sol_g.G1, sol_g.impact)
m = analytical_moments(sol_g; lags=2)
```

```julia
plot_result(result)
plot_result(decomp)
```

All three solvers produce identical state-space representations for a well-specified, determinate model. The Gensys solver handles singularity in ``\Gamma_0`` most robustly; Blanchard-Kahn and Klein are faster for smaller models. The simulation, IRF, FEVD, and moment functions all operate on the common `DSGESolution` type returned by any solver.

---

## Common Pitfalls

1. **Indeterminacy (too many stable eigenvalues)**: The Blanchard-Kahn condition requires the number of unstable eigenvalues to equal the number of forward-looking variables. In New Keynesian models, the Taylor principle ``\phi_\pi > 1`` is necessary for determinacy. With ``\phi_\pi < 1``, the model is typically indeterminate and all three solvers report an error.

2. **Non-existence (too few stable eigenvalues)**: If the model has more unstable eigenvalues than forward-looking variables, no bounded solution exists. Check parameter calibration --- this often indicates economically implausible values (e.g., ``\beta > 1``).

3. **Singular ``\Gamma_0``**: The Blanchard-Kahn and Klein solvers require ``\Gamma_0`` to be nonsingular. If ``\Gamma_0`` is singular (e.g., static equations with no current-period variables on the left), use Gensys which handles this case via the QZ decomposition.

4. **Explosive simulation paths**: `simulate` does not check stability during simulation. If the model is on the boundary of determinacy, numerical errors can produce slowly diverging paths. Verify `is_stable(sol)` returns `true` before long simulations.

5. **Steady-state not computed**: Calling `solve(spec)` without a `steady_state` block or prior `compute_steady_state(spec)` call uses a default guess. For nonlinear models, this can produce incorrect linearization. Always verify the steady state with `report(spec)` before solving.

---

## References

- Blanchard, O. J., & Kahn, C. M. (1980). The Solution of Linear Difference Models under Rational Expectations.
  *Econometrica*, 48(5), 1305--1311. [DOI](https://doi.org/10.2307/1912186)

- Klein, P. (2000). Using the Generalized Schur Form to Solve a Multivariate Linear Rational Expectations Model.
  *Journal of Economic Dynamics and Control*, 24(10), 1405--1423. [DOI](https://doi.org/10.1016/S0165-1889(99)00045-7)

- Koop, G., Pesaran, M. H., & Potter, S. M. (1996). Impulse Response Analysis in Nonlinear Multivariate Models.
  *Journal of Econometrics*, 74(1), 119--147. [DOI](https://doi.org/10.1016/0304-4076(95)01753-4)

- Sims, C. A. (2002). Solving Linear Rational Expectations Models.
  *Computational Economics*, 20(1--2), 1--20. [DOI](https://doi.org/10.1023/A:1020517101123)
