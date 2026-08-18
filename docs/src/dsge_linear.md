# [Linear Solution Methods](@id dsge_linear)

First-order linear rational expectations solutions produce the **state-space representation** that is the workhorse of DSGE analysis --- impulse responses, variance decompositions, simulation, and moment matching all flow from this representation. MacroEconometricModels.jl provides three solver implementations (Gensys, Blanchard-Kahn, and Klein) that share the unified `solve()` interface and return the same `DSGESolution{T}` output type. For model specification and linearization, see [DSGE Models](@ref dsge_page).

```@setup dsge_linear
using MacroEconometricModels, Random, LinearAlgebra, Statistics
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
    A[t] = A[t-1]^ρ * exp(σ * ε_A[t])

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

```@raw html
<iframe src="../assets/plots/dsge_irf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/dsge_fevd.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

**Recipe 3: Unconditional moments via the Lyapunov equation**

```@example dsge_linear
Σ = solve_lyapunov(sol.G1, sol.impact)
round.(sqrt.(diag(Σ)); digits=4)   # unconditional std of Y, C, K, A
```

The numbers are standard deviations in **levels**, not percentages: 0.1071 for output, 0.0540 for consumption, 1.0331 for capital, and 0.0229 for technology. Scaled by their steady states (3.70, 2.75, 37.99, 1.00) they become 2.9%, 2.0%, 2.7% and 2.3%. Both readings matter and they say different things. In levels, capital swings by an order of magnitude more than output, because it integrates every past innovation. Relative to its own mean, capital is the *smoother* series — the household spreads a technology windfall over the whole future rather than spending it, which is why consumption is half as volatile as output in levels and the least volatile of the four in relative terms. That is the permanent-income smoothing that is the RBC model's signature.

**Recipe 4: Map the determinacy region of a policy parameter**

```@example dsge_linear
nk = @dsge begin
    parameters: β = 0.99, σ_c = 1.0, κ = 0.3, φ_π = 1.5, φ_y = 0.5,
                ρ_d = 0.8, σ_d = 0.01
    endogenous: y, π, R, d
    exogenous: ε_d

    y[t] = y[t+1] - (1 / σ_c) * (R[t] - π[t+1]) + d[t]
    R[t] = φ_π * π[t] + φ_y * y[t]
    π[t] = β * π[t+1] + κ * y[t]
    d[t] = ρ_d * d[t-1] + σ_d * ε_d[t]
end
nk = compute_steady_state(nk)

dm = determinacy_region(nk; params=:φ_π, grids=range(0.0, 3.0; length=31))
report(dm)
```

---

## Unified Solver Interface

The `solve(spec; method=:gensys)` function is the single entry point for all DSGE solution methods. This page covers the three **linear** (first-order) methods, selected by the `method` keyword:

| `method` | Algorithm |
|----------|-----------|
| `:gensys` | Sims (2002) QZ decomposition (default) |
| `:blanchard_kahn` | Blanchard & Kahn (1980) eigenvalue counting |
| `:klein` | Klein (2000) generalized Schur decomposition |

All three solve the same linearized system and return a `DSGESolution{T}`. For the full set of `solve()` methods --- including higher-order perturbation, global projection, policy function iteration, and perfect foresight --- see [DSGE Models](@ref dsge_page).

### Solver Comparison

All three share one numerical core, `_solve_qz_quadratic`: the model is recast as the quadratic matrix equation ``f_{\text{lead}} G^2 + f_0 G + f_1 = 0`` and solved through the QZ decomposition of its ``2n \times 2n`` companion pencil. Consequently all three handle a singular ``\Gamma_0``, all three default to the same dividing line ``\text{div} = 1 + 10^{-8}``, and all three take the determinacy verdict from the same Sims (2002) rank test.

| Method | Solution path | Determinacy verdict | Reference |
|--------|---------------|---------------------|-----------|
| `:gensys` | Undetermined coefficients, companion-QZ fallback | Sims rank conditions | Sims (2002) |
| `:blanchard_kahn` | Companion-QZ solvent | Sims rank conditions | Blanchard & Kahn (1980) |
| `:klein` | Companion-QZ solvent | Sims rank conditions | Klein (2000) |

!!! note "What distinguishes the three solvers"
    Only `:gensys` differs numerically: it computes ``G_1`` and `impact` by iterating the undetermined-coefficients recursion ``G_{k+1} = -(f_0 + f_{\text{lead}} G_k)^{-1} f_1``, which is robust in models with many static variables, and falls back to the companion-QZ solvent when that iteration fails to converge or returns an explosive solvent. `:blanchard_kahn` and `:klein` are thin wrappers that take the companion-QZ solvent directly. On a well-specified determinate model the three agree to roughly machine precision.

---

## Choosing a Solver

For standard first-order work all three linear solvers produce the same solution, so the choice is a matter of provenance rather than capability. The table routes the broader method decision:

| Feature needed | Recommended | Why |
|----------------|-------------|-----|
| Standard IRFs from a linearized model | `:gensys` | Robust default, many static variables |
| Reproducing a published Klein/BK result | `:klein`, `:blanchard_kahn` | Pure companion-QZ solvent |
| Determinacy frontier over a parameter | `determinacy_region` | Sweeps and packages the verdict |
| Risk premia, welfare costs | `:perturbation` (order=2) | Captures precautionary effects; see [Nonlinear Methods](@ref dsge_nonlinear) |
| Large deviations, global accuracy | `:projection` or `:pfi` | Globally accurate policy; see [Nonlinear Methods](@ref dsge_nonlinear) |
| ZLB, occasionally binding | `:perfect_foresight` + OccBin | Respects inequality constraints; see [Constraints](@ref dsge_constraints) |

For most applications `:gensys` is the recommended default.

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
- ``\Pi`` is the ``n \times n_\eta`` expectation error selection matrix, built as ``-f_{\text{lead}}`` restricted to the columns of variables that actually appear with a ``[t+1]`` subscript
- ``\varepsilon_t`` is the ``n_\varepsilon \times 1`` vector of exogenous shocks
- ``\eta_t`` is the ``n_\eta \times 1`` vector of expectation errors (``\eta_t = y_t - E_{t-1}[y_t]`` for forward-looking variables)

``n_\eta`` counts distinct **lead variables**, not lead equations, and the two are different numbers. `spec.n_expect` records how many equations contain a lead term; ``n_\eta`` is `size(ld.Pi, 2)`. In the RBC model above the single Euler equation carries both ``C_{t+1}`` and ``A_{t+1}``, so `spec.n_expect` is 1 while ``\Pi`` is ``4 \times 2``.

The solvers reduce this system to the **state-space solution**:

```math
y_t = G_1 \, y_{t-1} + C_{\text{sol}} + \text{impact} \cdot \varepsilon_t
```

where:
- ``G_1`` is the ``n \times n`` state transition matrix
- ``C_{\text{sol}}`` is the ``n \times 1`` solution constant vector
- ``\text{impact}`` is the ``n \times n_\varepsilon`` shock impact matrix

The Blanchard-Kahn (1980) condition — the number of unstable roots of the pencil equals the number of forward-looking variables — is the familiar summary of stability, but it is not what the package computes. The `eu` vector carries the verdict:

| `eu` value | Interpretation |
|------------|----------------|
| `[1, 1]` | Existence and uniqueness (determinate) |
| `[1, 0]` | Existence but multiple solutions (indeterminate) |
| `[0, 0]` | No stable solution |

```@example dsge_linear
sol = solve(spec; method=:gensys)
(eu = sol.eu, determined = is_determined(sol),
 stable = is_stable(sol), max_modulus = maximum(abs.(sol.eigenvalues)))
```

The RBC model is determinate: a unique bounded rational-expectations solution exists. The largest eigenvalue modulus of ``G_1`` is 0.9653, comfortably inside the unit circle, so simulated paths revert to the steady state rather than drifting; the second-largest, 0.9, is the technology persistence ``\rho`` reappearing as a root of the solved system. The remaining two roots are zero because ``Y`` and ``C`` are static functions of the two state variables and carry no independent dynamics of their own.

!!! note "`eu` is a rank test, not a root count"
    The Blanchard-Kahn count is a *theorem* that holds under a rank condition which counting eigenvalues never checks. `eu` is therefore computed from the **Sims (2002) existence and uniqueness conditions** directly, as rank conditions on an augmented canonical system built from the raw Jacobians ``(f_0, f_1, f_{\text{lead}}, f_\varepsilon)`` — not on the ``(\Gamma_0, \Gamma_1)`` pencil that `linearize` returns, which folds ``f_{\text{lead}}`` into ``\Pi`` and drops it. Writing ``Q_1`` and ``Q_2`` for the stable and unstable row blocks of ``Q^{H}`` from the ordered QZ of that augmented system:

    - **Existence**: ``\operatorname{colspan}(Q_2 \Psi) \subseteq \operatorname{colspan}(Q_2 \Pi)`` — every shock that reaches the unstable block can be offset by some expectational error.
    - **Uniqueness**: ``\operatorname{rowspan}(Q_1 \Pi) \subseteq \operatorname{rowspan}(Q_2 \Pi)`` — no expectational error moves the stable block while being left free by the unstable block.

    The two agree on ordinary models. They part company exactly where the rank condition fails: an expectational error that is redundant, or unconstrained, or a shock loading on the unstable block in a direction no error can absorb. In those cases the count can report a unique solution where a continuum exists. All rank tests use scale-relative tolerances, so rescaling the equations cannot change the verdict.

    One case is decided before the rank tests: when there are fewer stable roots than state variables, no stable solution can be constructed at all, and `eu` is `[0, 0]` regardless.

### Diagnosing Indeterminacy

When `is_determined(sol)` returns `false`, `eu` localises the failure: `[1, 0]` means a continuum of bounded solutions (sunspots), `[0, 0]` means none exists. Violating the Taylor principle in the New Keynesian model of Recipe 4 produces the first case:

```@example dsge_linear
nk_weak = @dsge begin
    parameters: β = 0.99, σ_c = 1.0, κ = 0.3, φ_π = 0.5, φ_y = 0.5,
                ρ_d = 0.8, σ_d = 0.01
    endogenous: y, π, R, d
    exogenous: ε_d

    y[t] = y[t+1] - (1 / σ_c) * (R[t] - π[t+1]) + d[t]
    R[t] = φ_π * π[t] + φ_y * y[t]
    π[t] = β * π[t+1] + κ * y[t]
    d[t] = ρ_d * d[t-1] + σ_d * ε_d[t]
end

sol_weak = solve(nk_weak; method=:gensys)
(eu = sol_weak.eu, determined = is_determined(sol_weak))
```

With ``\phi_\pi = 0.5`` the central bank raises the nominal rate by less than one-for-one with inflation, so the real rate *falls* when inflation rises and nothing pins down expectations. `eu = [1, 0]` reports exactly that: bounded solutions exist, but there is a continuum of them, and self-fulfilling sunspot fluctuations are possible. The solver still returns a ``G_1`` — one representative member of the continuum — so a solution object is not by itself evidence of determinacy. Always read `eu`. Estimating which side of the frontier the post-war United States actually sat on is the subject of Lubik & Schorfheide (2004).

!!! warning "Do not diagnose determinacy from `sol.eigenvalues`"
    `sol.eigenvalues` holds the eigenvalues of the **solved transition matrix** ``G_1``, not the generalized eigenvalues of the model pencil. When a stable solution exists every one of them lies inside the unit circle by construction, so counting "unstable" entries always returns zero and comparing that count against the number of forward-looking variables always fails. The pencil roots that the Blanchard-Kahn condition refers to are computed internally and are not exposed on `DSGESolution`; use `eu`, or map the frontier with `determinacy_region`.

---

## Determinacy Regions

A single verdict answers "is *this* calibration determinate?". The more useful applied question is *where* the boundary lies. `determinacy_region` sweeps one or two parameters, re-solving the model at each grid point, and returns a `DeterminacyMap`. Recipe 4 already ran the one-parameter sweep over ``\phi_\pi``; its ASCII map reads `IIIIIIIIIIDDDD…`, ten indeterminate cells followed by twenty-one determinate ones, and the located crossing is:

```@example dsge_linear
determinacy_boundary(dm)     # located to within half a grid step
```

The boundary this locates is the **generalized Taylor principle**, ``\kappa(\phi_\pi - 1) + (1 - \beta)\phi_y > 0``: a strong enough output response substitutes for the inflation response, so the frontier sits slightly *below* ``\phi_\pi = 1`` rather than exactly at it. At ``\kappa = 0.3``, ``\beta = 0.99`` and ``\phi_y = 0.5`` the algebraic frontier is ``\phi_\pi = 1 - (1-\beta)\phi_y/\kappa \approx 0.983``, and the sweep brackets it at 0.95 — the grid spacing of 0.1 is the resolution limit, not an error. Refine the grid to sharpen the estimate.

Sweeping both policy coefficients traces the frontier as a curve rather than a point:

```@example dsge_linear
dm2 = determinacy_region(nk; params=(:φ_π, :φ_y),
                         grids=(range(0.0, 2.0; length=11), range(0.0, 4.0; length=9)))
(determinate = count(==(1), dm2.verdict), cells = length(dm2.verdict))
```

Fifty-three of the ninety-nine cells are determinate. The indeterminate region is not the half-plane ``\phi_\pi < 1``: because the output response substitutes for the inflation response, raising ``\phi_y`` buys determinacy at values of ``\phi_\pi`` below unity, and the frontier bends leftward as ``\phi_y`` grows.

```julia
plot_result(dm2)             # determinacy map (heatmap)
```

| Keyword | Default | Description |
|---------|---------|-------------|
| `params` | --- | a `Symbol`, or 1--2 `Symbol`s to sweep |
| `grids` | --- | matching grid values (a vector, or one vector per parameter) |
| `div` | `1.0 + 1e-8` | stable/unstable eigenvalue boundary |
| `rank_rtol` | `1e-8` | relative tolerance of the Sims rank tests |
| `method` | `:gensys` | linear solver used at each grid point |
| `threaded` | `false` | evaluate grid points in parallel (results are identical) |
| `quiet` | `true` | suppress per-point solver warnings |

Verdicts are stored as ordered integer codes in `dm.verdict`:

| Code | Meaning |
|------|---------|
| `1` | determinate |
| `0` | indeterminate |
| `-1` | no stable solution |
| `-2` | the model could not be solved at that grid point |

!!! note "Failed grid points are not a region"
    A parameter value where the steady state fails to converge is recorded as `-2`, with the error message in `dm.failures`, and the sweep continues rather than aborting. Such cells are drawn in neutral grey on the map and skipped by `determinacy_boundary` --- a solve failure is missing information, not a fourth determinacy region, so the step from "failed" to "determinate" is not a boundary crossing.

---

## Gensys (Sims 2002)

The **Gensys** solver is the default method. It handles singular ``\Gamma_0`` matrices, making it suitable for models with static identities such as ``Y_t = C_t + I_t``, and it is the most robust of the three on models carrying many static variables.

The `div` keyword sets the dividing line between stable and unstable eigenvalues. The default value of ``1.0 + 10^{-8}`` places the cutoff slightly above the unit circle, so a lone unit root is treated as non-explosive — the Sims (2002) convention, which keeps legitimate random-walk models solvable. Since the stable/unstable split defines the ``Q_1``/``Q_2`` partition, `div` also controls the rank conditions that produce `eu`. The `rank_rtol` keyword (default ``10^{-8}``) sets the relative tolerance of those rank and span tests. When two or more roots cluster within `cluster_tol` (default ``10^{-6}``) of the unit circle the split is numerically delicate and the solver warns; pass an explicit `div` to force a deterministic classification.

```@example dsge_linear
sol = solve(spec; method=:gensys, div=1.0+1e-8, rank_rtol=1e-8)
report(sol)
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `div` | `Real` | ``1.0 + 10^{-8}`` | Stable/unstable eigenvalue boundary |
| `rank_rtol` | `Real` | ``10^{-8}`` | Relative tolerance of the Sims rank and span tests |
| `cluster_tol` | `Real` | ``10^{-6}`` | Width of the near-unit-root band that triggers the warning |
| `sparse` | `Union{Bool,Symbol}` | `:auto` | Route the solvent through Newton + GMRES instead of a dense QZ |

!!! warning "The low-level `gensys` is not the entry point for models with leads"
    `gensys(Gamma0, Gamma1, C, Psi, Pi)` solves a system already written in canonical form, where every lead has been folded into ``\Pi``. That folding is lossy: `linearize` drops ``f_{\text{lead}}`` from the ``(\Gamma_0, \Gamma_1)`` pencil, so calling `gensys` on `linearize` output returns a ``G_1`` that is simply *wrong* for any forward-looking model — on the RBC model above it differs from the correct transition matrix by 3.33 in the largest entry, while still reporting `eu = [1, 1]`. Use `solve(spec; method=:gensys)`, which recovers the raw Jacobians and solves the companion pencil. The low-level function is for hand-built canonical systems only.

!!! note "Technical Note"
    The companion-QZ core reorders the Schur decomposition so stable eigenvalues (``|\lambda| < \text{div}``) come first, then recovers the stable solvent as ``G = Z_b Z_t^{-1}`` from the top and bottom ``n``-row blocks of the first ``n`` ordered right Schur vectors. The shock impact follows from ``\text{impact} = -(f_0 + f_{\text{lead}} G)^{-1} f_\varepsilon``. The solution constant is ``C_{\text{sol}} = (I - G_1)\bar{y}`` with ``\bar{y} = (f_0 + f_1 + f_{\text{lead}})^{-1} C`` — the lead block belongs in that sum, and omitting it (as ``\Gamma_0 - \Gamma_1`` does) gives the wrong steady state for any forward-looking model with a constant.

---

## Large Sparse Models

The default route decomposes the `2n × 2n` companion pencil with a complex QZ. For a medium-large model — multi-country, multi-sector, or a reduced HA system — that is the dominant cost, even though each equation touches only a handful of variables. `sparse=` selects an alternative that never forms a QZ: Newton's method on the quadratic `f_{lead} G^2 + f_0 G + f_1 = 0`, whose step

```math
(f_{lead} G_k + f_0)\,\Delta G + f_{lead}\,\Delta G\,G_k = -R_k
```

is a generalized Sylvester equation solved matrix-free by GMRES. Newton converges quadratically — 7 iterations on the benchmark below, independent of `n`.

```julia
sol = solve(spec; method=:gensys, sparse=true)    # force the sparse route
sol = solve(spec; method=:gensys, sparse=false)   # force the dense core
sol = solve(spec; method=:gensys)                 # :auto (default)
```

`:auto` takes the sparse route only when the model is both large (`n ≥ 400`) and sparse (density `≤ 5%`). Measured end-to-end on a multi-sector benchmark:

| `n` | density | speedup vs dense |
|---|---|---|
| 100 | 0.017 | 0.21× (slower — routed to dense) |
| 400 | 0.004 | 1.15× |
| 800 | 0.002 | 1.7--1.8× |
| 1600 | 0.001 | 2.7× |

!!! note "What the sparse route does and does not buy"
    The solvent `G` is dense even when the model is sparse, so the GMRES operator is dominated by dense `O(n^3)` products. The gain is that these are a few BLAS-3 **real** matrix multiplies rather than a **complex** QZ on a `2n × 2n` pencil; sparsity helps second-order by cheapening the `f_0 X` and `f_{lead}(\cdot)` products. On a fully dense model of the same size the advantage is ~1.8× at `n = 400` and gone by `n = 800`, which is why the heuristic requires sparsity as well as size. Because the cost is set by GMRES convergence, the speedup is model-dependent.

    It does **not** speed up the determinacy verdict: `eu` always comes from the Sims rank test, which decomposes the `(n+k)` canonical pencil (~22% of the dense cost). The table above is end-to-end and already includes it.

!!! warning "The sparse route is never allowed to be wrong"
    Newton converges to *a* solvent, not necessarily the stable one. Its result is accepted only if the residual is small **and** `G` is stable (`max|eig(G)| < div`); otherwise the dense core runs and its answer is used. The determinacy verdict is never taken from the sparse path. A model the sparse route cannot handle is therefore slower, never wrong.

---

## Blanchard-Kahn (1980) and Klein (2000)

Both methods recast the model as the quadratic matrix equation ``f_{\text{lead}} G^2 + f_0 G + f_1 = 0`` and take the stable solvent from the QZ decomposition of its companion pencil. They are thin wrappers over the same core, differing from `:gensys` only in that they use the companion-QZ solvent directly instead of the undetermined-coefficients iteration. Both accept the same `div`, `rank_rtol`, `cluster_tol` and `sparse` keywords with the same defaults, and both handle a singular ``\Gamma_0``.

```@example dsge_linear
sol_bk = solve(spec; method=:blanchard_kahn)
sol_k  = solve(spec; method=:klein)

(agree_G1 = maximum(abs.(sol_k.G1 .- sol_bk.G1)),
 agree_gensys = maximum(abs.(sol_k.G1 .- sol.G1)),
 eu_bk = sol_bk.eu, eu_klein = sol_k.eu)
```

The two wrappers return bit-identical matrices — they call the same core — and agree with the Gensys route to ``1.2 \times 10^{-12}``, the accumulated difference between the QZ solvent and the fixed point of the undetermined-coefficients iteration. All three report `eu = [1, 1]`. On a well-specified determinate model the choice of linear solver is therefore immaterial to the answer; it matters only when the undetermined-coefficients iteration struggles, which is the case `:gensys` is built for.

!!! note "The historical distinctions no longer apply"
    Textbook treatments distinguish these methods by their determinacy test — Blanchard-Kahn counts unstable roots against forward-looking variables, Klein counts stable roots against predetermined variables detected from the non-zero columns of ``\Gamma_1``. Neither count is what the package computes: since the rewiring onto the shared companion-QZ core, all three solvers take `eu` from the Sims (2002) rank conditions, and all three use ``\text{div} = 1 + 10^{-8}`` rather than a strict unit circle. The `_count_predetermined` helper still exists but no longer participates in the Klein solve.

### Return Value (All First-Order Solvers)

All three linear solvers return a `DSGESolution{T}` with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `G1` | `Matrix{T}` | ``n \times n`` state transition matrix |
| `impact` | `Matrix{T}` | ``n \times n_\varepsilon`` shock impact matrix |
| `C_sol` | `Vector{T}` | ``n \times 1`` constant vector |
| `eu` | `Vector{Int}` | ``[\text{existence}, \text{uniqueness}]``: 1 = yes, 0 = no. Sims (2002) rank conditions, not a root count |
| `method` | `Symbol` | Solver used (`:gensys`, `:blanchard_kahn`, `:klein`) |
| `eigenvalues` | `Vector{ComplexF64}` | Eigenvalues of ``G_1`` — the *solved* transition matrix, not the model pencil |
| `spec` | `ModelSpec{T}` | Back-reference to model specification |
| `linear` | `LinearDSGE{T}` | Linearized system (``\Gamma_0, \Gamma_1, C, \Psi, \Pi``) |

Accessor functions:

- `nvars(sol)` --- number of endogenous variables
- `nshocks(sol)` --- number of exogenous shocks
- `is_determined(sol)` --- `true` if `eu == [1, 1]`
- `is_stable(sol)` --- `true` if max eigenvalue modulus of ``G_1`` is less than 1

---

## Pre-Linearized Models

Some medium- and large-scale DSGE models (e.g., Smets & Wouters 2007) ship with **pre-linearized equations** where all variables represent log-deviations from a balanced growth path. The `linear` keyword skips the automatic linearization step and passes the equations directly to the solver:

```@example dsge_linear
spec_lin = MacroEconometricModels.ModelSpec{Float64}(
    [:y, :r], [:eps],
    [:rho_p, :phi_p],
    Dict(:rho_p => 0.8, :phi_p => 1.5),
    [:(0 + 0), :(0 + 0)],
    [
        (yt, yl, yle, eps, th) -> yt[1] - th[:rho_p] * yl[1] - eps[1],
        (yt, yl, yle, eps, th) -> yt[2] - th[:phi_p] * yt[1]
    ],
    0, Int[], Float64[], nothing;
    linear=true
)
sol_lin = solve(spec_lin; method=:gensys)
(determined = is_determined(sol_lin), G1 = sol_lin.G1,
 impact = sol_lin.impact, C_sol = sol_lin.C_sol)
```

The solver recovers the model by inspection: ``y`` follows its own AR(1) with ``\rho_p = 0.8``, and ``r`` responds contemporaneously with ``\phi_p = 1.5``, so the ``G_1`` row for ``r`` is ``1.2 = 1.5 \times 0.8`` and its impact loading is ``1.5``. Because this specification carries no constant term, ``C_{\text{sol}}`` is zero and the effective steady state coincides with the origin. In a model like Smets & Wouters (2007), where the observation equations carry trend growth and steady-state inflation, ``C_{\text{sol}}`` is non-zero and the effective steady state must be recovered as ``(I - G_1)^{-1} C_{\text{sol}}``.

For `linear=true` models:
- `compute_steady_state` returns zeros (all variables are deviations from steady state)
- The constant ``C_{\text{sol}}`` from gensys carries observation equation offsets (e.g., trend growth, steady-state inflation)
- The **effective steady state** is ``\bar{y} = (I - G_1)^{-1} C_{\text{sol}}``
- Bayesian estimation via `estimate_dsge_bayes` automatically computes ``\bar{y}`` for the Kalman filter observation offset

!!! note "Technical Note"
    When building observation equations for the Kalman filter, the internal `_build_likelihood_fn` detects `linear=true` and sets ``d = (I - G_1)^{-1} C_{\text{sol}}`` at the observable indices, overriding the zero steady state. This ensures correct likelihood evaluation for models like Smets & Wouters (2007) where observation equations include constant offsets (trend growth ``\bar{\gamma}``, steady-state inflation ``\bar{\pi}``, etc.).

---

## Simulation

Stochastic forward simulation generates sample paths from the state-space solution:

```math
y_t = G_1 \, y_{t-1} + \text{impact} \cdot \varepsilon_t + C_{\text{sol}}
```

where ``\varepsilon_t \sim N(0, I_{n_\varepsilon})`` are i.i.d. standard normal shocks. The simulation returns **levels** (steady state plus deviations), not deviations alone.

```@example dsge_linear
sol = solve(spec)
Y = simulate(sol, 200)                       # 200 x n_endog matrix of levels
round.(vec(std(Y; dims=1)); digits=4)        # realised volatility of Y, C, K, A
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `shock_draws` | `Union{Nothing, Matrix}` | `nothing` | Pre-drawn shocks (``T \times n_\varepsilon``); draws from ``N(0,1)`` if `nothing` |
| `rng` | `AbstractRNG` | `default_rng()` | Random number generator |

The return value is a ``T_{\text{periods}} \times n_{\text{endog}}`` matrix of variable levels. For augmented models (those with auxiliary variables introduced by the `@dsge` macro to handle higher-order lags/leads), the output is automatically filtered to the original endogenous variables.

A 200-period draw reproduces the ordering of the closed-form moments in Recipe 3 — capital the most volatile series, consumption the least among the endogenous trio — but every number is a sample statistic and will differ from the Lyapunov values above, and from run to run, unless `shock_draws` or a seeded `rng` is supplied. For anything that must be reproducible, pass shocks explicitly rather than relying on the global RNG.

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
(impact = round.(result.values[1, :, 1]; digits=5),
 horizon_10 = round.(result.values[10, :, 1]; digits=5),
 variables = result.variables)
```

```julia
plot_result(result)
```

```@raw html
<iframe src="../assets/plots/dsge_irf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The return type is `ImpulseResponse{T}` with field `.values` (``H \times n \times n_\varepsilon``), `.variables` (variable names), and `.shocks` (shock names). The result is directly compatible with `plot_result()` for interactive D3.js visualization.

The one-standard-deviation technology innovation raises output by 0.037 on impact against a consumption response of only 0.006: the household saves almost the entire windfall, and capital rises by 0.031. Ten periods later the ranking has inverted — capital has accumulated to 0.169, five times its impact response, while the direct technology impulse has decayed to 0.004 under ``\rho = 0.9``. Output remains elevated at 0.020 not because technology is still high but because the capital stock built during the boom is still being worked off, which is the RBC propagation mechanism in its purest form.

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
round.(decomp.proportions[:, 1, 40]; digits=4)   # share of each variable's variance from ε_A
```

```julia
plot_result(decomp)
```

```@raw html
<iframe src="../assets/plots/dsge_fevd.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
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

The Kronecker reading of this equation,

```math
\text{vec}(\Sigma) = (I_{n^2} - G_1 \otimes G_1)^{-1} \, \text{vec}(\text{impact} \cdot \text{impact}')
```

forms an ``n^2 \times n^2`` operator and costs ``O(n^6)``. `solve_lyapunov` never builds it: it runs the doubling (squaring) iteration, which converges quadratically in ``O(n^3)`` products, and checks its relative residual; if doubling misses tolerance a direct Bartels-Stewart solve takes over.

```@example dsge_linear
Σ = solve_lyapunov(sol.G1, sol.impact)   # n x n covariance matrix
round.(Σ[1:2, 1:2]; digits=6)            # Var(Y), Cov(Y,C), Var(C)
```

Output has an unconditional variance of ``1.15 \times 10^{-2}`` and consumption ``2.92 \times 10^{-3}``, with a positive covariance of ``4.92 \times 10^{-3}`` between them — consumption and output comove, as they must when a single technology shock drives both.

The `analytical_moments` function extracts a moment vector in a format compatible with `autocovariance_moments(data, lags)`, enabling direct comparison between model-implied and data moments for GMM estimation:

```@example dsge_linear
m = analytical_moments(sol; lags=2)
length(m)
```

The moment vector contains:
1. Upper triangle of the variance-covariance matrix: ``k(k+1)/2`` elements
2. Diagonal autocovariances ``\text{diag}(G_1^h \, \Sigma)`` at each lag ``h = 1, \ldots, \text{lags}``: ``k`` elements per lag

With ``k = 4`` variables and two lags that is ``10 + 4 + 4 = 18`` moments, which is what `length(m)` reports.

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

(solver_agreement = maximum(abs.(sol_k.G1 .- sol_g.G1)),
 determinate = is_determined(sol_g),
 tech_share_of_output_variance = decomp.proportions[1, 1, 40],
 n_moments = length(m))
```

```julia
plot_result(result)
plot_result(decomp)
```

```@raw html
<iframe src="../assets/plots/dsge_irf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/dsge_fevd.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The three solvers agree to ``1.2 \times 10^{-12}``, the model is determinate, and the single technology shock accounts for the entire forecast error variance of output at every horizon. The simulation, IRF, FEVD, and moment functions all operate on the common `DSGESolution` type returned by any solver, so switching solver never requires touching downstream code.

---

## Common Pitfalls

1. **Reading determinacy off `sol.eigenvalues`**: those are the eigenvalues of the solved ``G_1``, which are all stable whenever a solution exists. Counting "unstable" entries there always returns zero and tells you nothing. Read `eu`, or call `is_determined(sol)`.

2. **Treating a returned solution as proof of determinacy**: an indeterminate model still yields a `DSGESolution` — one representative member of the continuum — with no error and usually no warning. Only `eu = [1, 1]` certifies uniqueness. In New Keynesian models a violated Taylor principle (``\phi_\pi`` below the generalized frontier) is the usual cause of `eu = [1, 0]`.

3. **Calling low-level `gensys` on `linearize` output**: `linearize` folds the lead Jacobian into ``\Pi`` and drops it from the ``(\Gamma_0, \Gamma_1)`` pencil, so `gensys(ld.Gamma0, …)` returns a wrong ``G_1`` for any forward-looking model while still reporting `eu = [1, 1]`. Use `solve(spec; method=:gensys)`.

4. **Expecting `solve` to mutate `spec`**: it does not. `solve` computes the steady state into its own copy, so `spec.steady_state` is still empty afterwards. `linearize` and `perturbation_solver` require `spec = compute_steady_state(spec)` first; only `solve` handles it internally.

5. **A steady-state block that is not a steady state**: `compute_steady_state` with an explicit `steady_state` block accepts the values you supply without checking that they zero the residuals — only the numerical path verifies them. Linearizing around a non-stationary point yields a system whose constant ``C`` is silently assumed to be zero, so the first-order solution looks entirely healthy while every higher-order and global solver expands around the wrong point. Verify the residuals at your declared steady state before trusting the solution; see [Analytical Steady State](@ref) for the check.

6. **Explosive simulation paths**: `simulate` does not check stability during simulation. If the model sits on the boundary of determinacy, numerical error can produce slowly diverging paths. Verify `is_stable(sol)` before long simulations.

---

## References

- Blanchard, O. J., & Kahn, C. M. (1980). The Solution of Linear Difference Models under Rational Expectations.
  *Econometrica*, 48(5), 1305--1311. [DOI](https://doi.org/10.2307/1912186)

- Klein, P. (2000). Using the Generalized Schur Form to Solve a Multivariate Linear Rational Expectations Model.
  *Journal of Economic Dynamics and Control*, 24(10), 1405--1423. [DOI](https://doi.org/10.1016/S0165-1889(99)00045-7)

- Lubik, T. A., & Schorfheide, F. (2004). Testing for Indeterminacy: An Application to U.S. Monetary Policy.
  *American Economic Review*, 94(1), 190--217. [DOI](https://doi.org/10.1257/000282804322970760)

- Sims, C. A. (2002). Solving Linear Rational Expectations Models.
  *Computational Economics*, 20(1--2), 1--20. [DOI](https://doi.org/10.1023/A:1020517101123)

- Smets, F., & Wouters, R. (2007). Shocks and Frictions in US Business Cycles: A Bayesian DSGE Approach.
  *American Economic Review*, 97(3), 586--606. [DOI](https://doi.org/10.1257/aer.97.3.586)
