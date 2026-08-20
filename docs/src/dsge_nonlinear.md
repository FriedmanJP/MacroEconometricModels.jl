# [Nonlinear Solution Methods](@id dsge_nonlinear)

First-order linear solutions impose **certainty equivalence** --- agents behave as if shocks have zero variance. This rules out risk premia, precautionary savings, welfare costs of uncertainty, and asymmetric dynamics. Nonlinear methods capture all of these by retaining higher-order terms in the Taylor expansion of the policy function or by solving the functional equation globally. MacroEconometricModels.jl provides four families: **higher-order perturbation** (local, Schmitt-Grohe & Uribe 2004; Andreasen, Fernandez-Villaverde & Rubio-Ramirez 2018), **Chebyshev projection** (global polynomial, Judd 1992, 1998), **policy function iteration** (global iterative, Coleman 1990), and **value-function iteration** (Bellman / Howard, Stokey, Lucas & Prescott 1989). Collocation and PFI support multi-threading; PFI also supports Anderson acceleration (Walker & Ni 2011). `vfi_solver` maximizes ``u + \beta E[V]`` and requires a reward and a state transition --- Euler time iteration is [`pfi_solver`](@ref). For model specification and linearization, see [DSGE Models](@ref dsge_page). For first-order solvers, see [Linear Solvers](@ref dsge_linear).

```@setup dsge_nonlinear
using MacroEconometricModels, Random, LinearAlgebra, Statistics
Random.seed!(42)
```

## Quick Start

**Recipe 1: Second-order perturbation with pruned simulation**

```@example dsge_nonlinear
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

psol = perturbation_solver(spec; order=2)
Y_sim = simulate(psol, 1000)  # pruned simulation (Kim et al. 2008)
nothing # hide
```

**Recipe 2: Third-order perturbation with GIRFs**

```@example dsge_nonlinear
psol3 = perturbation_solver(spec; order=3)
Y_sim3 = simulate(psol3, 1000)
girf3 = irf(psol3, 40; irf_type=:girf, n_draws=100)
nothing # hide
```

```julia
plot_result(girf3)
```

```@raw html
<iframe src="../assets/plots/dsge_girf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

**Recipe 3: Chebyshev projection and Euler errors**

```@example dsge_nonlinear
proj = collocation_solver(spec; degree=5, grid=:tensor, max_iter=200)
err = max_euler_error(proj; n_test=1000, rng=MersenneTwister(42))
(converged = proj.converged, iterations = proj.iterations,
 nodes = size(proj.collocation_nodes, 1), max_euler_error = err)
```

Gauss-Newton drives the collocation residual to zero in 6 iterations on a 36-node tensor grid, and the resulting policy has a maximum Euler error of ``4.9 \times 10^{-4}`` — ``\log_{10}`` of ``-3.3``, "good" on the usual accuracy scale. Note that the residual at the nodes and the Euler error away from them are different quantities; the second is the one that measures accuracy.

**Recipe 4: Bellman `vfi_solver` with Howard policy evaluation**

The Bellman operator needs a state transition, a control box, and a map from ``(x, a)`` back to the full endogenous vector. For this spec the state is ``(K, A)`` and the control is consumption:

```@example dsge_nonlinear
function growth_transition(x, a, ε, θ)
    K, A = x[1], x[2]
    C = a[1]
    Kp = A * K^θ[:α] + (1 - θ[:δ]) * K - C
    Ap = A^θ[:ρ] * exp(θ[:σ] * ε[1])
    [Kp, Ap]
end
function growth_bounds(x, θ)
    K, A = x[1], x[2]
    y = A * K^θ[:α] + (1 - θ[:δ]) * K
    lo = max(y * 1e-4, 1e-8)
    ([lo], [max(y - 1e-8, lo + 1e-8)])
end
function growth_outcome(x, a, θ)
    K, A = x[1], x[2]
    C = a[1]
    Y = A * K^θ[:α]
    [Y, C, Y + (1 - θ[:δ]) * K - C, A]
end

vfi = vfi_solver(spec;
    utility = log, beta = :β, consumption = :C, controls = [:C],
    transition = growth_transition,
    control_bounds = growth_bounds,
    outcome = growth_outcome,
    degree = 3, n_grid = 8, howard_steps = 10, max_iter = 200)
(converged = vfi.converged, iterations = vfi.iterations,
 V_ss = evaluate_value(vfi, spec.steady_state[vfi.state_indices]))
```

`vfi_solver` requires a period reward, a discount factor, and a state transition. It stores a value function: `evaluate_value(vfi, x)` is ``V(K, A)`` at that state. Euler-equation time iteration without a reward is [`pfi_solver`](@ref).

---

## Second-Order Perturbation

The first-order linear solution imposes **certainty equivalence**: agents behave identically regardless of shock variance. This produces four specific deficiencies: (1) zero risk premia on asset returns, (2) no precautionary savings motive, (3) zero welfare cost of business cycles, and (4) perfectly symmetric impulse responses to positive and negative shocks. The second-order perturbation (Schmitt-Grohe & Uribe 2004) resolves all four by retaining quadratic terms in the Taylor expansion of the policy function.

The second-order decision rule takes the form:

```math
z_t = \bar{z} + f_v \, v_t + \tfrac{1}{2} f_{vv} (v_t \otimes v_t) + \tfrac{1}{2} f_{\sigma\sigma} \sigma^2
```

where:
- ``z_t`` is the ``n \times 1`` vector of all endogenous variables (deviations from steady state)
- ``v_t = [x_{t-1}; \varepsilon_t]`` is the ``n_v \times 1`` innovations vector (lagged states + current shocks)
- ``f_v`` is the ``n \times n_v`` first-order coefficient matrix
- ``f_{vv}`` is the ``n \times n_v^2`` second-order coefficient tensor (flattened Kronecker)
- ``f_{\sigma\sigma}`` is the ``n \times 1`` variance correction (shifts the **stochastic steady state**)
- ``\sigma`` is the perturbation scaling parameter

```@example dsge_nonlinear
psol = perturbation_solver(spec; order=2)
(states = spec.varnames[psol.state_indices], h_sigma_sigma = psol.hσσ,
 controls = spec.varnames[psol.control_indices], g_sigma_sigma = psol.gσσ)
```

The variance corrections are the whole point of going to second order. Capital carries ``h_{\sigma\sigma} = +0.0175`` and consumption ``g_{\sigma\sigma} = -0.0175``: facing technology risk, the household holds *more* capital and consumes *less* than the deterministic steady state prescribes. That is the precautionary savings motive, and it is identically zero at first order. Technology and output take no correction, because neither is a choice made under uncertainty — one is exogenous, the other a static function of predetermined inputs.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `order` | `Int` | `2` | Perturbation order (1, 2, or 3) |
| `sylvester_method` | `Symbol` | `:auto` | Solver for the order-2/3 Kronecker-Sylvester systems |
| `sylvester_tol` | `Real` | ``10^{-8}`` | Residual gate on the Sylvester solve |
| `gmres_tol` | `Real` | ``10^{-8}`` | Convergence tolerance of the GMRES fallback |
| `gmres_max_outer` | `Int` | `20` | Outer restarts allowed in the GMRES fallback |

!!! warning "`method=` is accepted but has no effect"
    `perturbation_solver` takes a `method` keyword documented as selecting the first-order solver, but the body never reads it: the first-order step always routes through the companion-QZ core. Passing `method=:blanchard_kahn` silently changes nothing. Do not rely on it.

### Algorithm

The computation follows Schmitt-Grohe & Uribe (2004) in six steps:

1. Solve the first-order system via Gensys or Blanchard-Kahn to obtain ``G_1`` and the impact matrix
2. Partition variables into states (``x``) and controls (``y``) via the ``\Gamma_1`` structure
3. Build the innovations vector ``v = [x_{t-1}; \varepsilon_t]`` and the mapping matrices that relate each argument slot (current, lag, lead, shock) to ``v``-space
4. Compute all 10 unique Hessian tensors via central finite differences across the four argument slots
5. Assemble the ``n \times n_v^2`` right-hand side from contracting each Hessian with its slot-to-``v`` mapping matrices, then solve the Kronecker system: ``(I_{n_v^2} \otimes f_c + \text{kron}(M,M)' \otimes f_f) \, \text{vec}(f_{vv}) = -\text{vec}(\text{RHS})``
6. Solve the ``\sigma^2`` correction: ``(f_c + f_f) \, f_{\sigma\sigma} = -f_f \, f_{vv} \, \text{vec}(\eta \eta')``

!!! note "Technical Note"
    The **stochastic steady state** ``\bar{z} + \tfrac{1}{2} f_{\sigma\sigma} \sigma^2`` differs from the deterministic steady state ``\bar{z}``. This shift captures precautionary behavior: risk-averse agents accumulate more capital when facing uncertainty. The magnitude depends on the curvature of the utility function and the variance of shocks.

---

## Third-Order Perturbation

The decision rule at third order adds cubic and variance-interaction terms (Andreasen, Fernandez-Villaverde & Rubio-Ramirez 2018):

```math
z_t = \bar{z} + f_v \, v_t + \tfrac{1}{2} f_{vv} (v_t \otimes v_t) + \tfrac{1}{2} f_{\sigma\sigma} \sigma^2 + \tfrac{1}{6} f_{vvv} (v_t \otimes v_t \otimes v_t) + \tfrac{1}{2} f_{\sigma\sigma v} \, \sigma^2 \, v_t + \tfrac{1}{6} f_{\sigma\sigma\sigma} \, \sigma^3
```

where:
- ``f_{vvv}`` is the ``n \times n_v^3`` third-order coefficient tensor (flattened triple Kronecker)
- ``f_{\sigma\sigma v}`` is the ``n \times n_v`` interaction between uncertainty and the state
- ``f_{\sigma\sigma\sigma}`` is the ``n \times 1`` cubic variance correction (zero for Gaussian shocks)

```@example dsge_nonlinear
psol3 = perturbation_solver(spec; order=3)
(gxxx = size(psol3.gxxx), hxxx = size(psol3.hxxx),
 g_sigma3 = psol3.gσσσ, h_sigma3 = psol3.hσσσ)
```

With ``n_v = 3`` the third-order tensors are ``2 \times 27`` for both controls and states — the flattened triple Kronecker product. Both cubic variance corrections come back exactly zero, as they must: ``f_{\sigma\sigma\sigma}`` is proportional to the third moment of the shocks, and Gaussian innovations have none. The term becomes active only under skewed innovations, which is the case third-order perturbation exists to handle.

### Algorithm

The third-order computation extends the second-order procedure with six additional steps:

1. Solve first-order and second-order as prerequisites (Steps 1--6 above)
2. Compute all 20 unique third-derivative tensors via central finite differences across the four argument slots (current, lag, lead, shock), storing only canonical orderings
3. Accumulate the ``n \times n_v^3`` right-hand side from two sources: (A) pure third derivatives contracted with mapping matrices across all slot triples, and (B) mixed Hessian-times-second-order interaction terms (3 cyclic permutations per Hessian block)
4. Solve the third-order Kronecker system: ``(I_{n_v^3} \otimes f_c + \text{kron}(M,M,M)' \otimes f_f) \, \text{vec}(f_{vvv}) = -\text{vec}(\text{RHS}_3)``
5. Compute ``f_{\sigma\sigma v}`` correction via contraction of ``f_{vvv}`` with shock covariance ``\eta \eta'`` plus Hessian-times-``f_{\sigma\sigma}`` interaction across all slot pairs
6. Set ``f_{\sigma\sigma\sigma} = 0`` (zero for Gaussian shocks)

!!! note "Technical Note"
    Third-order perturbation captures **skewness** in the ergodic distribution. The ``f_{\sigma\sigma\sigma}`` correction is zero for Gaussian shocks but would be non-zero for non-Gaussian innovations. The ``f_{\sigma\sigma v}`` term captures how the variance correction depends on the state --- agents' precautionary behavior varies with economic conditions.

### Return Value (`PerturbationSolution{T}`)

| Field | Type | Description |
|-------|------|-------------|
| `order` | `Int` | Perturbation order (1, 2, or 3) |
| `gx` | `Matrix{T}` | ``n_y \times n_v`` first-order controls |
| `hx` | `Matrix{T}` | ``n_x \times n_v`` first-order states |
| `gxx` | `Union{Nothing, Matrix{T}}` | ``n_y \times n_v^2`` second-order controls (order ``\geq 2``) |
| `hxx` | `Union{Nothing, Matrix{T}}` | ``n_x \times n_v^2`` second-order states (order ``\geq 2``) |
| `g\sigma\sigma` | `Union{Nothing, Vector{T}}` | ``n_y`` control variance correction (order ``\geq 2``) |
| `h\sigma\sigma` | `Union{Nothing, Vector{T}}` | ``n_x`` state variance correction (order ``\geq 2``) |
| `gxxx` | `Union{Nothing, Matrix{T}}` | ``n_y \times n_v^3`` third-order controls (order = 3) |
| `hxxx` | `Union{Nothing, Matrix{T}}` | ``n_x \times n_v^3`` third-order states (order = 3) |
| `g\sigma\sigma x` | `Union{Nothing, Matrix{T}}` | ``n_y \times n_v`` variance-state interaction (order = 3) |
| `h\sigma\sigma x` | `Union{Nothing, Matrix{T}}` | ``n_x \times n_v`` variance-state interaction (order = 3) |
| `g\sigma\sigma\sigma` | `Union{Nothing, Vector{T}}` | ``n_y`` cubic variance correction (order = 3) |
| `h\sigma\sigma\sigma` | `Union{Nothing, Vector{T}}` | ``n_x`` cubic variance correction (order = 3) |
| `eta` | `Matrix{T}` | ``n_v \times n_\varepsilon`` shock loading ``[0; I]`` block |
| `steady_state` | `Vector{T}` | Deterministic steady state |
| `state_indices` | `Vector{Int}` | Indices of state variables |
| `control_indices` | `Vector{Int}` | Indices of control variables |
| `eu` | `Vector{Int}` | Existence/uniqueness from first-order |
| `method` | `Symbol` | Always `:perturbation` |
| `spec` | `ModelSpec{T}` | Back-reference to specification |
| `linear` | `LinearDSGE{T}` | Linearized system |

---

## Solving the Kronecker-Sylvester System

Both higher orders reduce to the same **generalized Sylvester equation** for the coefficient tensor ``X`` (``f_{vv}`` at order 2, ``f_{vvv}`` at order 3):

```math
f_c \, X + f_f \, X \, \underbrace{(M \otimes M \otimes \cdots \otimes M)}_{d \text{ factors}} = -\text{RHS}
```

where ``d`` is the perturbation order, ``M`` is the ``n_v \times n_v`` transition of the augmented innovation vector, and ``X`` is ``n \times n_v^d``.

Vectorizing this gives an ``(n \cdot n_v^d)``-dimensional linear system. That is the direct reading of the equation, and it is unusable at scale: for a 35-variable model at order 3 the operator alone is 68 GB. The default solver instead uses the **generalized-Schur** method (Kamenik 2005), which decomposes once and back-substitutes through the resulting triangular structure:

1. QZ-decompose the pencil ``(f_c, f_f)`` and Schur-decompose ``M``, making all three operators triangular
2. Transform the equation into that basis, where it separates by column
3. Back-substitute recursively over the ``d`` Kronecker factors
4. Transform back

The ``n_v^d \times n_v^d`` Kronecker operator is **never formed** at any point. Cost falls from ``O((n \cdot n_v^d)^3)`` time and ``O((n \cdot n_v^d)^2)`` memory to ``O(n^3 + n^2 n_v^d + d^2 n \, n_v^{d+1})``.

```julia
# Default: generalized Schur, with automatic fallback
psol = perturbation_solver(spec; order=3)

# Force a specific solver
psol_k = perturbation_solver(spec; order=3, sylvester_method=:kamenik)
psol_d = perturbation_solver(spec; order=3, sylvester_method=:dense)
```

| `sylvester_method` | Description |
|---|---|
| `:auto` (default) | Generalized Schur; falls back to `:dense` (small) or `:gmres` (large) if the residual check fails |
| `:kamenik` | Generalized Schur only; warns if the residual exceeds `sylvester_tol` |
| `:dense` | Vectorize and factor the full ``(n \cdot n_v^d)^2`` system. Exact, but only viable for small models |
| `:gmres` | Matrix-free restarted GMRES. Iterative fallback |

All paths solve the same equation and agree to floating-point tolerance. Representative timings (`n` = model size, `n_v` = states + shocks):

| Case | System size | `:kamenik` | `:dense` |
|---|---|---|---|
| ``n=35, n_v=14``, order 2 | 6,860 | 4.7 ms | 4,651 ms |
| ``n=20, n_v=8``, order 3 | 10,240 | 4.7 ms | 20,370 ms |
| ``n=35, n_v=14``, order 3 | 96,040 | 67 ms | infeasible (68 GB) |

!!! note "Solvability"
    The Sylvester equation is singular exactly when a generalized eigenvalue of the pencil ``(f_c, f_f)`` coincides with a ``d``-fold product of eigenvalues of ``M``. Every solve carries a residual check, so a near-singular system produces a warning and a fallback rather than a silently wrong coefficient tensor.

    Working on the **pencil** rather than on ``f_c^{-1} f_f`` matters in practice: ``f_c`` is singular for any model with a purely static or purely forward-looking equation, and the pencil formulation solves those cases exactly.

---

## Pruning

Naive simulation of higher-order decision rules produces **explosive sample paths** because the Kronecker products ``(v_t \otimes v_t)`` compound deviations multiplicatively --- a moderate deviation at time ``t`` is squared, generating a larger deviation at ``t+1``, which is squared again. **Pruning** (Kim, Kim, Schaumburg & Sims 2008) prevents this by tracking state components separately and using only first-order states in the Kronecker products.

### The Pruned State-Space Object

`pruned_state_space` returns the pruned system as a first-class object. `simulate`, the unconditional FEVD, and the closed-form moments all read the recursion and the observation map from it, so they cannot disagree:

```@example dsge_nonlinear
pss = pruned_state_space(psol)      # psol is the order-2 solution from Quick Start
report(pss)
```

| Field | Description |
|-------|-------------|
| `order` | Perturbation order (1, 2, or 3) |
| `nx`, `ny`, `n_eps`, `nv`, `n` | States, controls, shocks, ``n_v = n_x + n_\varepsilon``, ``n = n_x + n_y`` |
| `state_indices`, `control_indices` | Positions of states/controls in the model's variable order |
| `hx_state`, `eta_x` | First-order state transition and shock loading |
| `gx_state`, `eta_y` | Control loadings on the **lagged** state and the current shock |
| `hxx`, `gxx`, `hss`, `gss` | Second-order blocks over ``v \otimes v`` and the ``\sigma^2`` corrections |
| `hxxx`, `gxxx`, `hssx`, `gssx`, `hsss`, `gsss` | Third-order blocks (zeros below order 3) |
| `steady_state`, `spec` | Deterministic steady state and back-reference, for mapping deviations back to levels |

!!! warning "The control is evaluated on the lagged state"
    Policy functions here are written over ``v_t = [x_{t-1}; \varepsilon_t]``, so ``g_x`` loads the **lagged** state and ``\eta_y`` the current shock. Evaluating the control on the freshly-updated state instead applies a lagged-state loading to a current-dated state --- the state channel propagates twice, and every control series comes out shifted forward by one period. On an exactly linear model, where orders 2 and 3 must reproduce the first-order solution to machine precision, that shift is unmistakable. Both the simulation and the moment routine now evaluate the control on the same ``v`` blocks that drive the state.

### Second-Order Pruning (Kim et al. 2008)

The pruned simulation decomposes the state into two components:

1. **First-order state**: ``x_t^{(1)} = h_{x,\text{state}} \cdot x_{t-1}^{(1)} + \eta_x \cdot \varepsilon_t``
2. **Second-order correction**: ``x_t^{(2)} = h_{x,\text{state}} \cdot x_{t-1}^{(2)} + \tfrac{1}{2} h_{xx} (v_t^{(1)} \otimes v_t^{(1)}) + \tfrac{1}{2} h_{\sigma\sigma}``

Total state: ``x_t = x_t^{(1)} + x_t^{(2)}``

where ``v_t^{(1)} = [x_{t-1}^{(1)}; \varepsilon_t]`` contains only first-order states. The key insight is that the Kronecker product ``v_t^{(1)} \otimes v_t^{(1)}`` uses only the stable first-order component, preventing compounding.

### Third-Order Pruning (Andreasen et al. 2018)

Third-order pruning adds a third component:

3. **Third-order correction**: ``x_t^{(3)} = h_{x,\text{state}} \cdot x_{t-1}^{(3)} + h_{xx} (v_t^{(1)} \otimes v_t^{(2)}) + \tfrac{1}{6} h_{xxx} (v_t^{(1)} \otimes v_t^{(1)} \otimes v_t^{(1)}) + \tfrac{1}{2} h_{\sigma\sigma x} \, v_t^{(1)} + \tfrac{1}{6} h_{\sigma\sigma\sigma}``

where ``v_t^{(2)} = [x_{t-1}^{(2)}; 0]`` contains the second-order state correction with zero shocks.

Total: ``x_t = x_t^{(1)} + x_t^{(2)} + x_t^{(3)}``

```@example dsge_nonlinear
psol3 = perturbation_solver(spec; order=3)
Y_sim = simulate(psol3, 1000)  # 3-component pruned simulation
nothing # hide
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `shock_draws` | `Union{Nothing, Matrix}` | `nothing` | Pre-drawn shocks (T x n_shocks); draws from N(0,1) if `nothing` |
| `rng` | `AbstractRNG` | `default_rng()` | Random number generator |
| `antithetic` | `Bool` | `false` | Antithetic variates for variance reduction |

!!! warning "Explosive paths without pruning"
    Never simulate a second- or third-order perturbation solution without pruning. The naive simulation ``z_t = \bar{z} + f_v v_t + \frac{1}{2} f_{vv} (v_t \otimes v_t) + \ldots`` using the total state in the Kronecker product diverges within 50--100 periods for most calibrations. All `simulate(::PerturbationSolution, ...)` calls use pruning automatically.

---

## Generalized Impulse Responses

Analytical IRFs from the first-order solution miss second- and third-order effects. **Generalized impulse response functions** (GIRFs, Koop, Pesaran & Potter 1996) compute the expected difference between a shocked and baseline path via Monte Carlo simulation:

```math
\text{GIRF}(h, \delta_j) = E\big[y_{t+h} \mid \varepsilon_{j,t} = \delta_j, \Omega_{t-1}\big] - E\big[y_{t+h} \mid \Omega_{t-1}\big]
```

where:
- ``\delta_j`` is the shock impulse to shock ``j``
- ``\Omega_{t-1}`` is the information set at ``t-1``
- The expectation is computed by averaging over ``n_{\text{draws}}`` simulated paths using pruned simulation

The GIRF captures nonlinear dynamics that the analytical first-order IRF misses: asymmetric responses to positive vs. negative shocks, state-dependent propagation, and variance-correction effects. For a first-order solution, the GIRF converges to the analytical IRF as ``n_{\text{draws}} \to \infty``.

```@example dsge_nonlinear
# Analytical IRFs (first-order only, fast)
irf_analytical = irf(psol3, 40)

# GIRFs (captures nonlinear dynamics, Monte Carlo)
girf = irf(psol3, 40; irf_type=:girf, n_draws=100)

(analytical_impact = round.(irf_analytical.values[1, :, 1]; digits=5),
 girf_impact = round.(girf.values[1, :, 1]; digits=5))
```

The two impact responses agree to three decimals — the nonlinear terms contribute little at a one-standard-deviation impulse, which is the regime where certainty equivalence is a good approximation. The residual gap is part genuine second- and third-order curvature and part Monte-Carlo noise from the 100 draws; raising `n_draws` shrinks the second but not the first. GIRFs earn their cost at large impulses and in models with strong asymmetries, not here.

```julia
plot_result(girf)
```

```@raw html
<iframe src="../assets/plots/dsge_girf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `irf_type` | `Symbol` | `:analytical` | `:analytical` for first-order, `:girf` for simulation-based |
| `n_draws` | `Int` | `500` | Number of Monte Carlo draws for GIRF |
| `shock_size` | `Real` | `1.0` | Impulse size in standard deviations |

### Forecast Error Variance Decomposition

The standard `fevd(psol, H)` uses the first-order decomposition from the underlying linear solution. For order ≥ 2, the **unconditional FEVD** decomposes the asymptotic variance by shock source using the Andreasen et al. (2018) augmented Lyapunov approach. For each shock ``j``, the solver zeros out all other shocks in the augmented innovation variance and re-solves the Lyapunov equation to isolate that shock's contribution:

```math
\text{FEVD}_{i,j}^{\infty} = \frac{\text{Var}_j(y_i)}{\sum_{k=1}^{n_\varepsilon} \text{Var}_k(y_i)}
```

where ``\text{Var}_j(y_i)`` is the unconditional variance of variable ``i`` attributable to shock ``j`` alone, computed from the restricted augmented system.

```@example dsge_nonlinear
# Standard FEVD (first-order IRFs)
psol2 = perturbation_solver(spec; order=2)
fv1 = fevd(psol2, 40)

# Unconditional FEVD (order≥2, augmented Lyapunov)
fv2 = fevd(psol2, 1; unconditional=true)

(h_step = round.(fv1.proportions[:, 1, 40]; digits=4),
 unconditional = round.(fv2.proportions[:, 1, 1]; digits=4))
```

Both decompositions assign the entire forecast error variance of every variable to ``\varepsilon_A``, because the RBC model has exactly one shock and the shares must sum to one by construction. The comparison becomes informative only with two or more shocks, where the ``h``-step and unconditional decompositions genuinely differ: the first weights the transition path, the second the ergodic distribution, and nonlinear propagation redistributes variance between them.

The unconditional FEVD captures second-order cross-terms through the Kronecker state block ``\text{vec}(x^f \otimes x^f)`` in the augmented system. At order 1, it reduces to the standard asymptotic first-order decomposition. At order 2, it reflects how nonlinear propagation redistributes variance across shocks.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `unconditional` | `Bool` | `false` | Use augmented Lyapunov instead of IRF-based decomposition |

!!! note "Technical Note"
    At order ≥ 2, per-shock contributions do not sum exactly to the total variance due to cross-shock quartic moment terms. The solver normalizes proportions to sum to 1 for each variable, following Andreasen et al. (2018) §4.2.

---

## Chebyshev Projection

**Chebyshev collocation** (Judd 1992, 1998) approximates the policy function globally using Chebyshev polynomials on a tensor or Smolyak grid. Unlike perturbation, the approximation is accurate far from the steady state --- essential for models with large shocks, regime switches, or occasionally binding constraints. The solver finds coefficients such that the equilibrium conditions hold exactly at each collocation node.

The policy function approximation takes the form:

```math
y_i(x) \approx \sum_{j=1}^{n_b} c_{i,j} \, T_j(x)
```

where:
- ``c_{i,j}`` are the Chebyshev coefficients (``n_{\text{vars}} \times n_b`` matrix)
- ``T_j(x)`` are the multivariate Chebyshev basis functions (products of univariate ``T_k(x_d)``)
- ``n_b`` is the number of basis functions

### Algorithm

The collocation solver proceeds in five steps:

1. **Linearize** to get the state/control partition and compute state bounds as ``\bar{x}_i \pm \text{scale} \cdot \sigma_i`` from the first-order Lyapunov solution
2. **Build grid**: tensor product of Chebyshev extrema (Gauss-Lobatto) nodes on ``[-1, 1]^{n_x}``; for high-dimensional models, use Smolyak sparse grid instead
3. **Construct basis matrix** by evaluating all multivariate Chebyshev polynomials at each collocation node
4. **Initialize coefficients** from the first-order perturbation solution via least-squares projection onto the Chebyshev basis
5. **Newton iteration**: solve ``R(c) = 0`` where ``R`` is the vector of equilibrium residuals evaluated at all nodes, with Gauss-Hermite or monomial quadrature for expectations of next-period variables. Uses Gauss-Newton with backtracking line search.

```@example dsge_nonlinear
proj = collocation_solver(spec; degree=5, grid=:tensor, max_iter=200)
report(proj)
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `degree` | `Int` | `5` | Chebyshev polynomial degree (tensor grid) |
| `grid` | `Symbol` | `:auto` | `:tensor`, `:smolyak`, or `:auto` (tensor if ``n_x \leq 4``) |
| `smolyak_mu` | `Int` or `Vector{Int}` | `3` | Smolyak approximation level: a scalar is isotropic, an ``n_x``-vector is anisotropic |
| `quadrature` | `Symbol` | `:auto` | `:gauss_hermite` or `:monomial` (auto based on ``n_\varepsilon``) |
| `n_quad` | `Int` | `5` | Quadrature nodes per dimension |
| `scale` | `Real` | `3.0` | State bounds as multiples of unconditional std |
| `tol` | `Real` | ``10^{-8}`` | Newton convergence tolerance |
| `max_iter` | `Int` | `100` | Maximum Newton iterations |
| `threaded` | `Bool` | `false` | Compute Jacobian columns in parallel |
| `verbose` | `Bool` | `false` | Print per-iteration residuals |
| `initial_coeffs` | `Union{Nothing, Matrix}` | `nothing` | Warm-start coefficients from previous solve |
| `adaptive` | `Bool` | `false` | Enable dimension-adaptive Smolyak refinement |
| `euler_tol` | `Real` | ``10^{-6}`` | Target max Euler error for adaptive refinement |
| `max_nodes` | `Int` | `1000` | Node budget; refinement stops before exceeding it |
| `max_refinements` | `Int` | `10` | Maximum refinement rounds |
| `n_euler_test` | `Int` | `200` | Random test points used for the Euler-error target |
| `rng` | `AbstractRNG` | `default_rng()` | RNG for the Euler-error test points |

### Smolyak Sparse Grids

For models with ``n_x > 4`` states, tensor grids become computationally infeasible. A tensor grid with degree 5 and 5 states requires ``6^5 = 7{,}776`` nodes; with 8 states, ``6^8 \approx 1.7 \times 10^6``. **Smolyak sparse grids** (Smolyak 1963; Judd, Maliar, Maliar & Valero 2014) select a subset of grid points that preserve polynomial exactness at a fraction of the cost. The `grid=:smolyak` option uses nested Clenshaw-Curtis points, where a level-``\ell`` dimension contributes ``2^\ell + 1`` points and polynomial degrees ``0, \dots, 2^\ell``. The isotropic selection rule admits the level multi-indices

```math
\mathcal{A}_\mu = \{ \ell \in \mathbb{N}_0^{n_x} : |\ell|_1 \leq \mu \},
```

equivalently ``|\alpha|_1 \leq \mu + n_x`` for ``\alpha = \ell + 1``. The nodes and the polynomial basis come out of this one rule, so the collocation matrix is square and unisolvent by construction.

!!! note "Technical Note"
    The `grid=:auto` option selects tensor grids for ``n_x \leq 4`` and Smolyak grids for ``n_x > 4``, except that requesting `adaptive=true` always selects `:smolyak`. Similarly, `quadrature=:auto` selects Gauss-Hermite for ``n_\varepsilon \leq 2`` and monomial rules for ``n_\varepsilon > 2``. Monomial rules scale linearly with ``n_\varepsilon`` while Gauss-Hermite scales exponentially.

### Anisotropic Smolyak Grids

States rarely need equal resolution. Passing an ``n_x``-vector to `smolyak_mu` applies the dimension-adaptive weighting of Gerstner & Griebel (2003), which admits

```math
\mathcal{A}_{\boldsymbol{\mu}} = \left\{ \ell \in \mathbb{N}_0^{n_x} : \sum_{k=1}^{n_x} \frac{\ell_k}{\mu_k} \leq 1 \right\},
```

where:
- ``\ell_k`` is the Clenshaw-Curtis level in state ``k``
- ``\mu_k`` is the requested approximation level for state ``k``; ``\mu_k = 0`` pins that state at level 0

Setting ``\mu_k = \mu`` for every ``k`` recovers the isotropic rule exactly, so anisotropy is a strict generalization. The package derives the grid, the polynomial index set, and the Smolyak combination coefficients from this same rule.

Whether it pays is an empirical question, and the way to answer it is to hold the node count fixed and compare Euler errors. In the stochastic growth model the capital domain is ``[34.19, 41.79]`` while productivity spans only ``[0.90, 1.10]``, which invites the guess that capital deserves the larger share of the level budget:

```@example dsge_nonlinear
iso = collocation_solver(spec; grid=:smolyak, smolyak_mu=3, max_iter=200)
ani = collocation_solver(spec; grid=:smolyak, smolyak_mu=[4, 2], max_iter=200)

(nodes_iso  = size(iso.collocation_nodes, 1),
 nodes_ani  = size(ani.collocation_nodes, 1),
 euler_iso  = max_euler_error(iso; n_test=500, rng=MersenneTwister(42)),
 euler_ani  = max_euler_error(ani; n_test=500, rng=MersenneTwister(42)))
```

The guess is wrong. Both grids carry 29 nodes, and the anisotropic one is the *less* accurate of the two: ``7.4 \times 10^{-4}`` against ``6.3 \times 10^{-4}`` for the isotropic grid. A wide domain is not the same thing as a curved policy, and on this calibration the two states need comparable resolution — shifting levels from productivity to capital sacrifices cross-terms without buying anything back. Raising the isotropic level to ``\mu = 4`` (65 nodes) reaches ``3.4 \times 10^{-4}``; more nodes help here, redistributing them does not. Anisotropy earns its keep when a state is genuinely near-linear — a slow-moving demographic or policy state alongside a sharply curved asset holding — and the way to find out is the comparison above, not the domain widths. The state ordering is `sol.state_indices`, so `spec.varnames[iso.state_indices]` names the entries of the level vector.

### Adaptive Refinement

Choosing ``\boldsymbol{\mu}`` by hand requires knowing where the curvature is. Setting `adaptive=true` discovers it instead, following the dimension-adaptive refinement strategy of Brumm & Scheidegger (2017). Starting from the requested level set, each round:

1. Solves the collocation system on the current grid
2. Measures the max Euler error on a **fixed** set of `n_euler_test` random points (fixed across rounds, so round-to-round comparisons reflect the grid and not resampling noise)
3. Scores every admissible forward neighbour ``\ell + e_k`` by the largest absolute Euler residual at the **new** nodes that block would introduce
4. Adds the highest-scoring block and warm-starts from the previous coefficients

Because the refined index set is a superset of the old one, the warm start is exact: the padded coefficients represent the identical policy function. Refinement stops at `euler_tol`, at the `max_nodes` budget, or after `max_refinements` rounds — and the solver warns when it stops without reaching the target.

```@example dsge_nonlinear
ada = collocation_solver(spec; grid=:smolyak, smolyak_mu=1, adaptive=true,
                         euler_tol=1e-3, max_nodes=300, max_refinements=12,
                         rng=MersenneTwister(42))

(nodes       = size(ada.collocation_nodes, 1),
 refinements = ada.refinements,
 euler_error = ada.euler_error,
 levels      = vec(maximum(ada.smolyak_levels; dims=1)))
```

Five refinement rounds grow the 5-node ``\mu = 1`` grid into a 21-node grid with level 2 in both states, reaching a max Euler error of ``6.3 \times 10^{-4}`` and stopping there because the ``10^{-3}`` target is met. The refiner arrives unprompted at the same verdict the hand comparison reached — the two states want equal resolution — and it does so at 21 nodes rather than the 29 the isotropic ``\mu = 3`` grid spends for comparable accuracy. That is what adaptivity buys: not knowing where the curvature is in advance.

!!! warning "`residual_norm` is not accuracy"
    `residual_norm` measures the fit **at the collocation nodes only**, where the solver forces it to zero. The isotropic ``\mu = 3`` solve above converges to ``\|R\| \approx 6 \times 10^{-14}`` and still has a max Euler error of ``6.3 \times 10^{-4}`` --- ten orders of magnitude apart, because the polynomial is pinned only at the nodes and free between them. Always report `max_euler_error`, which samples the domain uniformly. This is why `adaptive=true` targets `euler_tol` rather than `tol`.

### Evaluating the Policy Function

`evaluate_policy` maps a state vector to the full vector of endogenous variables using the stored Chebyshev coefficients. The argument is the **lagged** state ``x_{t-1}`` in physical units, and the return value is the current-period vector of all ``n`` endogenous variables in `spec.varnames` order — the same ``v_t = [x_{t-1}; \varepsilon_t]`` convention the perturbation solutions use.

```@example dsge_nonlinear
x_lag = [37.0, 1.0]                     # K_{t-1} = 37, A_{t-1} = 1
y = evaluate_policy(proj, x_lag)
(variables = spec.varnames, length_of_y = length(y),
 state_bounds = proj.state_bounds)
```

Passing a matrix of `n_points × nx` states instead returns an `n_points × n_vars` matrix, which is how you tabulate a policy function over a grid.

The Chebyshev basis is fitted only on `state_bounds` — here ``K \in [34.19, 41.79]`` and ``A \in [0.90, 1.10]``, computed as ``\bar{x}_i \pm \text{scale} \cdot \sigma_i`` from the first-order Lyapunov solution. `evaluate_policy` warns once when asked for a point outside that box and clamps to the boundary before evaluating, so out-of-domain values are flat rather than divergent. Widen `scale` rather than reading clamped output.

### Euler Equation Errors

The **Euler equation error** measures the accuracy of the global approximation by evaluating residuals at random test points drawn uniformly within the state bounds. Because the points are random, pass an explicit `rng` whenever the number has to be reproducible:

```@example dsge_nonlinear
err = max_euler_error(proj; n_test=1000, rng=MersenneTwister(42))
(err = err, log10_err = log10(err))
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `n_test` | `Int` | `1000` | Random test points drawn inside the state bounds |
| `rng` | `AbstractRNG` | `default_rng()` | Generator for the test points |

The error is reported in levels. Convert to ``\log_{10}`` for the standard accuracy metric:

| ``\log_{10}`` error | Quality |
|----------------------|---------|
| ``< -6`` | Excellent |
| ``-6`` to ``-4`` | Good |
| ``-4`` to ``-2`` | Acceptable |
| ``> -2`` | Poor --- increase degree or tighten tolerance |

---

## Policy Function Iteration

**Policy function iteration** (PFI, Coleman 1990) solves for the equilibrium policy function by iterating on the Euler equation. At each step, the current policy guess determines expected future values via quadrature, and the equilibrium conditions at each grid point produce an updated policy via Newton's method. PFI tends to be more robust than collocation for models with kinks or near-kinks in the policy function, while collocation converges faster for smooth problems.

The algorithm iterates three sub-steps at each grid point ``j``:

1. **Expectation**: compute ``E[y_{t+1}]`` using the current policy coefficients and quadrature
2. **Euler solve**: given ``y_{\text{lag}} = x_j`` (the grid point as lagged state) and ``E[y_{t+1}]``, solve ``F(y_t, y_{\text{lag}}, E[y_{t+1}], 0, \theta) = 0`` for ``y_t`` via Newton iteration
3. **Refit**: project the updated policy values onto the Chebyshev basis via least squares

```@example dsge_nonlinear
pfi = pfi_solver(spec; degree=5, damping=0.5, anderson_m=3, max_iter=500)
(converged = pfi.converged, iterations = pfi.iterations,
 residual_norm = pfi.residual_norm, method = pfi.method)
```

PFI reaches the ``10^{-8}`` sup-norm target in 223 iterations here — roughly forty times the 6 Gauss-Newton steps collocation needs on the same grid, which is the price of a first-order fixed point against a Newton method. Damping is what costs the iterations: `damping=0.5` halves every update, and dropping it (`damping=1.0`) finishes in 126. The `converged` flag is the field to check regardless — a `ProjectionSolution` is returned either way, and a policy that stopped short of tolerance will quietly produce wrong IRFs and moments downstream.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `degree` | `Int` | `5` | Chebyshev polynomial degree |
| `grid` | `Symbol` | `:auto` | `:tensor`, `:smolyak`, or `:auto` |
| `smolyak_mu` | `Int` or `Vector{Int}` | `3` | Smolyak approximation level (scalar isotropic, vector anisotropic) |
| `quadrature` | `Symbol` | `:auto` | `:gauss_hermite` or `:monomial` |
| `n_quad` | `Int` | `5` | Quadrature nodes per dimension |
| `scale` | `Real` | `3.0` | State bounds as multiples of unconditional std |
| `damping` | `Real` | `1.0` | Damping factor (0.5 for slow convergence, 1.0 for no damping) |
| `anderson_m` | `Int` | `0` | Anderson acceleration depth (0 = disabled; see [Anderson Acceleration](@ref anderson_accel)) |
| `howard_steps` | `Int` | `0` | Extra Euler re-solves per outer iteration (0 = one pass) |
| `next_state` | `Symbol` | `:linear` | `:linear` (policy + gensys impact), `:policy` (policy states only), or `:nonlinear` (Newton residual at the quadrature shock) |
| `threaded` | `Bool` | `false` | Multi-threaded Euler solve and expectation quadrature |
| `verbose` | `Bool` | `false` | Print per-iteration residuals |
| `tol` | `Real` | ``10^{-8}`` | Sup-norm convergence tolerance |
| `max_iter` | `Int` | `500` | Maximum iterations |
| `initial_coeffs` | `Union{Nothing, Matrix}` | `nothing` | Warm-start Chebyshev coefficients (`n_vars × n_basis`) |

!!! note "Technical Note"
    PFI, Chebyshev collocation, and `vfi_solver` all return the same `ProjectionSolution{T}` type. All three support `evaluate_policy`, `simulate`, `irf` (a generalized IRF), and `max_euler_error`. The `method` field distinguishes them: `:projection` for collocation, `:pfi` for policy function iteration, `:vfi` for Bellman value-function iteration. `vfi_solver` additionally stores `value_fn` and supports [`evaluate_value`](@ref).

### Return Value (`ProjectionSolution{T}`)

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | `Matrix{T}` | ``n_{\text{vars}} \times n_b`` Chebyshev coefficients |
| `state_bounds` | `Matrix{T}` | ``n_x \times 2`` state domain bounds |
| `grid_type` | `Symbol` | `:tensor` or `:smolyak` |
| `degree` | `Int` | Polynomial degree (tensor) or highest Smolyak level reached |
| `collocation_nodes` | `Matrix{T}` | ``n_{\text{nodes}} \times n_x`` grid points in ``[-1, 1]`` |
| `residual_norm` | `T` | Final ``\|R\|`` residual (collocation) or sup-norm (PFI/VFI); fit **at the nodes only** |
| `n_basis` | `Int` | Number of basis functions |
| `multi_indices` | `Matrix{Int}` | ``n_b \times n_x`` multi-index matrix |
| `quadrature` | `Symbol` | `:gauss_hermite` or `:monomial` |
| `converged` | `Bool` | Newton convergence flag |
| `iterations` | `Int` | Iterations until convergence |
| `spec` | `ModelSpec{T}` | Back-reference to model specification |
| `linear` | `LinearDSGE{T}` | Linearized system |
| `impact` | `Matrix{T}` | Cached first-order shock-impact matrix used by `irf`/`simulate` |
| `steady_state` | `Vector{T}` | Cached steady-state vector |
| `state_indices` | `Vector{Int}` | State variable indices |
| `control_indices` | `Vector{Int}` | Control variable indices |
| `method` | `Symbol` | `:projection`, `:pfi`, or `:vfi` |
| `euler_error` | `T` | Max Euler error achieved by adaptive refinement; `NaN` when not measured |
| `smolyak_levels` | `Matrix{Int}` | ``n_{\text{blocks}} \times n_x`` admissible level set; ``0 \times 0`` for tensor grids |
| `refinements` | `Int` | Adaptive refinement rounds performed (`0` when `adaptive=false`) |

---

## Value Function Iteration

`vfi_solver` iterates the Bellman operator on a tensor state grid. At each node it maximizes ``u(c) + \beta E[V(x')]`` over a one-dimensional control box, then applies Howard (1960) policy evaluation: hold the policy fixed and iterate ``V = u + \beta P_\pi V``. Convergence is on ``\|V_{\text{new}} - V\|_\infty``. The exported policy is a Chebyshev fit of that grid policy, so `evaluate_policy`, `simulate`, and `irf` keep working.

`ModelSpec` still stores Euler residuals. The reward can be declared on the spec (`utility: log(C)`, `utility: C^(1-σ)/(1-σ)`, `beta: β`, `controls: C` in `@dsge`) and is then picked up automatically. When `transition` is omitted, `vfi_solver` defaults to `next_state=:residual` (drop control FOCs and Newton-solve the leftover residuals) whenever that system is square. `next_state=:linear` remains an explicit opt-in and throws if the linearized `G1` columns for the controls are numerically zero — those columns do not carry the resource constraint. When `control_bounds` is omitted, it is inferred from `constraint:` / `variable_bound` or an SS collar. Calling `vfi_solver(spec)` with no `utility`/`beta` throws `ArgumentError` and points at [`pfi_solver`](@ref).

```math
V(x) = \max_{a \in \mathcal{A}(x)} \Bigl\{ u(x, a) + \beta \sum_q w_q \, V\bigl(g(x, a, \varepsilon_q)\bigr) \Bigr\}
```

where

- ``x`` is the state (here capital and productivity)
- ``a`` is the control (consumption)
- ``u`` is the period reward
- ``g`` is the state transition
- ``(w_q, \varepsilon_q)`` are the shock quadrature nodes

```@example dsge_nonlinear
vfi_plain = vfi_solver(spec;
    utility = log, beta = :β, consumption = :C, controls = [:C],
    transition = growth_transition,
    control_bounds = growth_bounds,
    outcome = growth_outcome,
    degree = 3, n_grid = 8, howard_steps = 0, max_iter = 400)
(converged = vfi_plain.converged, iterations = vfi_plain.iterations,
 residual_norm = vfi_plain.residual_norm)
```

Without Howard steps the contraction is essentially ``\beta = 0.99`` per sweep. After 400 iterations ``\|V_{\mathrm{new}} - V\|_\infty = 1.77 \times 10^{-3}`` and `converged` is `false`. The stored ``V`` is the last iterate, not a fixed point: `evaluate_value(vfi_plain, x)` returns that incomplete object off the grid by multilinear interpolation.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `utility` | `Function` | required | `u(c)` if `consumption` is set, else `u(y, y_lag, ε, θ)` |
| `beta` | `Real` or `Symbol` | required | Discount factor, or a name in `param_values` |
| `transition` | `Function` | inferred | `(x, a, ε, θ) → x′`; `growth_transition` above is the RBC case; omitted → `next_state` |
| `control_bounds` | `Function` | inferred | `(x, θ) → (a_lo, a_hi)`; `growth_bounds` above is the resource-feasible box |
| `next_state` | `Symbol` | `:auto` | `:auto` → `:residual` when FOCs drop cleanly; `:linear` throws if `G1` ignores controls |
| `consumption` | `Symbol` | `nothing` | If set, `utility` is `u(c)` |
| `controls` | `Vector{Symbol}` | non-state endog | Choice variables (one continuous control in v1) |
| `outcome` | `Function` | fill states + controls | `(x, a, θ) → y` full endogenous vector; `growth_outcome` above returns ``(Y, C, K', A)`` |
| `degree` | `Int` | `5` | Chebyshev degree of the exported policy |
| `n_grid` | `Int` | `12` | Uniform tensor nodes per state for the Bellman grid |
| `quadrature` | `Symbol` | `:auto` | `:gauss_hermite` or `:monomial` |
| `n_quad` | `Int` | `5` | Quadrature nodes per shock dimension |
| `n_choice` | `Int` | `41` | Line-search points on the control box |
| `howard_steps` | `Int` | `20` | Howard policy-evaluation steps per iteration |
| `tol` | `Real` | ``10^{-8}`` | Sup-norm tolerance on ``V`` |
| `max_iter` | `Int` | `500` | Maximum VFI iterations |
| `threaded` | `Bool` | `false` | Multi-threaded per-node maximization |

### Howard Policy Evaluation

After each maximization sweep, **Howard policy evaluation** (Howard 1960; Santos & Rust 2003) freezes the policy and iterates the linear map ``V \mapsto u_\pi + \beta P_\pi V``. Those steps do not re-optimize; they only refresh ``V`` under the current policy. The next maximization then sees a much more accurate continuation value.

```@example dsge_nonlinear
vfi_howard = vfi_solver(spec;
    utility = log, beta = :β, consumption = :C, controls = [:C],
    transition = growth_transition,
    control_bounds = growth_bounds,
    outcome = growth_outcome,
    degree = 3, n_grid = 8, howard_steps = 15, max_iter = 200)
(converged = vfi_howard.converged, iterations = vfi_howard.iterations,
 V_ss = evaluate_value(vfi_howard, spec.steady_state[vfi_howard.state_indices]))
```

Fifteen Howard steps cut the iteration count from the unconverged 400-sweep run above to 109, and ``\|V_{\mathrm{new}} - V\|_\infty`` meets ``10^{-8}``. The value at the deterministic steady state is 99.62, against the infinite-horizon payoff of staying there forever, ``u(C_{ss})/(1-\beta) = \log(C_{ss})/0.01 = 101.32``. The 1.7-unit gap is the cost of the coarse ``8 \times 8`` Bellman grid, not a discounting error.

### VFI vs PFI vs Collocation

All three global solvers return `ProjectionSolution{T}` and share `evaluate_policy`, `simulate`, `irf`, and `max_euler_error`. They are not the same algorithm:

- **Collocation** solves the Euler residual by Gauss-Newton on Chebyshev coefficients
- **PFI** is Coleman time iteration on the Euler equation (no value function)
- **VFI** maximizes the Bellman operator and stores ``V``

On a smooth concave problem the three policies agree near the steady state. They need not share an Euler error to six digits: VFI's exported policy is a Chebyshev fit of a grid maximizer, so `max_euler_error` scores that fit, not the native Bellman residual.

```@example dsge_nonlinear
sol_vfi  = vfi
sol_pfi  = pfi_solver(spec; degree=3, damping=0.5, max_iter=400)
sol_proj = collocation_solver(spec; degree=3, max_iter=200)
x_ss = spec.steady_state[sol_vfi.state_indices]
(c_vfi = evaluate_policy(sol_vfi, x_ss)[2],
 c_pfi = evaluate_policy(sol_pfi, x_ss)[2],
 c_proj = evaluate_policy(sol_proj, x_ss)[2],
 V_ss = evaluate_value(sol_vfi, x_ss))
```

Consumption at the deterministic steady state is 2.754 from both PFI and collocation, matching ``C_{ss}`` to four decimals. VFI's exported Chebyshev fit of the grid maximizer returns 2.603 — 5.5% below — because `n_grid = 8` and `degree = 3` undersample the policy. Raise `n_grid` and `degree` together when the object of interest is the Euler error rather than the value function. `evaluate_value` is defined only on a VFI solution.

!!! warning "`max_iter` is part of the comparison"
    PFI at `damping=0.5` reaches the ``10^{-8}`` target in 295 iterations. Capped at `max_iter=200` it stops with `converged=false` and a sup-norm of ``1.2 \times 10^{-7}``. Give every solver the budget it needs, and check `converged` before comparing policies.

---

## [Anderson Acceleration](@id anderson_accel)

**Anderson acceleration** (Walker & Ni 2011) speeds convergence of fixed-point iterations by mixing the last ``m`` iterates. Given iterates ``x_k`` and residuals ``r_k = g(x_k) - x_k``, the method solves:

```math
\min_{\alpha} \left\| \sum_{i=1}^{m} \alpha_i \, r_i \right\|^2 \quad \text{s.t.} \quad \sum_{i=1}^{m} \alpha_i = 1
```

and returns the mixed iterate ``x_{\text{new}} = \sum_{i} \alpha_i (x_i + r_i)``. The depth parameter ``m`` controls how many previous iterates are used. Larger ``m`` captures more history but increases the linear algebra cost. In practice, ``m = 3``--``5`` works well.

Anderson acceleration is available on PFI (`anderson_m` kwarg). It operates on the vectorized Chebyshev coefficient matrix, treating the coefficient update as a fixed-point iteration. Bellman `vfi_solver` does not take `anderson_m`: it accelerates with Howard policy evaluation of ``V``.

```@example dsge_nonlinear
pfi_anderson = pfi_solver(spec; degree=5, damping=0.5, anderson_m=3, max_iter=200)
(converged = pfi_anderson.converged, iterations = pfi_anderson.iterations)
```

---

## Multi-Threading

The three global solvers (collocation, PFI, VFI) support opt-in multi-threading via the `threaded=true` keyword. When enabled:

- **PFI**: Grid-point Euler Newton solves run in parallel via `Threads.@threads`
- **VFI**: Per-node Bellman maximizations run in parallel
- **Collocation**: Jacobian column computation runs in parallel

Threading requires Julia to be started with multiple threads (e.g., `julia -t 4`). On single-threaded Julia, `threaded=true` has no effect. The solutions are numerically identical regardless of the `threaded` setting.

```@example dsge_nonlinear
sol_seq = vfi_solver(spec;
    utility = log, beta = :β, consumption = :C, controls = [:C],
    transition = growth_transition,
    control_bounds = growth_bounds,
    outcome = growth_outcome,
    degree = 3, n_grid = 8, howard_steps = 10, max_iter = 200,
    threaded = false)
sol_par = vfi_solver(spec;
    utility = log, beta = :β, consumption = :C, controls = [:C],
    transition = growth_transition,
    control_bounds = growth_bounds,
    outcome = growth_outcome,
    degree = 3, n_grid = 8, howard_steps = 10, max_iter = 200,
    threaded = true)
abs(evaluate_value(sol_seq, spec.steady_state[sol_seq.state_indices]) -
    evaluate_value(sol_par, spec.steady_state[sol_par.state_indices]))
```

---

## Analytical Moments

### First-Order Moments

For first-order solutions, `analytical_moments` computes unconditional moments in closed form via the discrete **Lyapunov equation**:

```math
\Sigma = G_1 \, \Sigma \, G_1' + \text{impact} \cdot \text{impact}'
```

where:
- ``\Sigma`` is the ``n \times n`` unconditional covariance matrix
- ``G_1`` is the state transition matrix from the first-order solution
- ``\text{impact}`` is the ``n \times n_\varepsilon`` shock impact matrix

The Kronecker reading of this equation, ``\text{vec}(\Sigma) = (I_{n^2} - G_1 \otimes G_1)^{-1} \, \text{vec}(\text{impact} \cdot \text{impact}')``, forms an ``n^2 \times n^2`` matrix and costs ``O(n^6)`` — 11.9 GB at ``n = 200``. `solve_lyapunov` instead uses the doubling (squaring) iteration (Barraud 1977), which converges quadratically in ``O(n^3)`` products, and verifies its relative residual; if doubling misses tolerance, a direct Bartels-Stewart (1972) solve — complex Schur plus a triangular column sweep — takes over. Autocovariances at lag ``h`` follow from ``\Gamma_h = G_1^h \, \Sigma``.

```@example dsge_nonlinear
sol = solve(spec)

# Unconditional covariance matrix
Sigma = solve_lyapunov(sol.G1, sol.impact)

# Moment vector matching autocovariance_moments format
m = analytical_moments(sol; lags=2)
(std_devs = round.(sqrt.(diag(Sigma)); digits=5), n_moments = length(m))
```

The moment vector contains two blocks: (1) the upper triangle of the variance-covariance matrix (``k(k+1)/2`` elements) and (2) diagonal autocovariances at each lag (``k`` elements per lag). With ``k = 4`` and two lags that is ``10 + 4 + 4 = 18`` entries. This format matches `autocovariance_moments(data, lags)` for direct comparison in [Estimation](@ref dsge_estimation).

### Second-Order Moments (Andreasen et al. 2018)

At order ≥ 2, the unconditional distribution is no longer Gaussian and certainty equivalence breaks down. The **augmented state-space Lyapunov** approach (Andreasen, Fernandez-Villaverde & Rubio-Ramirez 2018) computes exact closed-form moments by stacking the first-order state ``x^f``, the second-order correction ``x^s``, and the Kronecker product ``\text{vec}(x^f \otimes x^f)`` into an augmented state:

```math
z_t = \begin{bmatrix} x^f_t \\ x^s_t \\ \text{vec}(x^f_t \otimes x^f_t) \end{bmatrix} \in \mathbb{R}^{2n_x + n_x^2}
```

The augmented system is linear: ``z_{t+1} = A \, z_t + c + u_t``, where ``A`` contains the pruned dynamics and ``c`` captures the stochastic steady-state correction ``\frac{1}{2} h_{\sigma\sigma}``. Unconditional moments follow from the augmented Lyapunov equation ``\text{Var}(z) = A \, \text{Var}(z) \, A' + \text{Var}(u)``, solved via iterative doubling.

The `:gmm` format returns mean shifts and product moments suitable for higher-order GMM estimation:

```@example dsge_nonlinear
# Second-order moments (closed-form augmented Lyapunov)
m2 = analytical_moments(psol2; lags=1, format=:gmm)
mc = analytical_moments(psol2; lags=1)          # default :covariance format
(gmm = length(m2), covariance = length(mc))
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `lags` | `Int` | `1` | Number of autocovariance lags |
| `format` | `Symbol` | `:covariance` | `:covariance` for the backward-compatible vector, `:gmm` for mean shifts plus product moments |

The `:gmm` format returns 18 entries against 14 for `:covariance` at one lag: it carries the mean shifts that second-order perturbation induces, which the covariance format discards. Use `:gmm` when the estimation targets include the stochastic steady-state shift itself, `:covariance` when matching second moments against `autocovariance_moments`.

Both **order 2 and order 3** compute the default `:covariance` format from this closed form; neither draws a shock. Comparing the order-2 covariance moments against the first-order values above shows the two agree to ``5 \times 10^{-5}`` on this calibration — the risk correction shifts the mean far more than it shifts the second moments.

### Order-3 Moments Without Simulation

At third order the augmented state is ``z_t = [x^f; x^s; x^f \otimes x^f; x^{rd}; x^f \otimes x^s; x^f \otimes x^f \otimes x^f]`` and the system is again linear, ``z_t = c + A z_{t-1} + \xi_t``. Andreasen et al.'s companion code obtains ``\text{Var}(\xi)`` and ``\text{Cov}(\xi_t, z_{t-1})`` from an extended hand derivation carrying shock moments to sixth order.

This package takes a shorter route to the same quantities. Expanding each block of ``\xi`` shows that every term is a monomial in ``\varepsilon_t`` times a **single component** of ``z_{t-1}`` — absorbing the nonlinearity into the state coordinates is precisely what the augmentation does. So for a fixed shock, ``\xi`` is *linear* in ``\tilde{z} = [1; z]``:

```math
\xi_t = \Xi(\varepsilon_t) \, \tilde{z}_{t-1}
```

and because ``\varepsilon \perp z``,

```math
\text{Var}(\xi) = E_\varepsilon\!\left[\Xi(\varepsilon) \, E[\tilde{z}\tilde{z}'] \, \Xi(\varepsilon)'\right],
\qquad
\text{Cov}(\xi_t, z_{t-1}) = E_\varepsilon[\Xi(\varepsilon)] \, \text{Cov}(\tilde{z}, z)
```

Both expectations integrate a polynomial of degree ``\le 6`` in ``\varepsilon``, which a **4-node Gauss-Hermite tensor rule evaluates exactly** (an ``m``-node rule is exact through degree ``2m-1 = 7``). Since ``\text{Var}(\xi)`` depends on ``\text{Var}(z)``, the two are solved as a fixed point alongside the Lyapunov equation.

!!! note "The ``\text{Cov}(\xi_t, z_{t-1})`` term is not optional"
    ``E_\varepsilon[\Xi(\varepsilon)]`` is **not** zero: the ``\varepsilon^2 x^f`` terms inside the ``x^f \otimes x^f \otimes x^f`` block correlate the innovation with the state it is added to. The Lyapunov constant is therefore ``\text{Var}(\xi) + \text{Cov}(\xi,z)A' + A\,\text{Cov}(\xi,z)'``, and the same term propagates through the autocovariance recursion. It vanishes identically at order 2, which is why it can be ignored there --- and it was omitted here before, biasing the third-order variance.

At order 2 the innovation variance ``\text{Var}(u)`` already needs quartic shock moments (``E[\varepsilon^4] = 3`` for Gaussian innovations), assembled into a ``(2n_x + n_x^2) \times (2n_x + n_x^2)`` block matrix from ``E[(\varepsilon \otimes \varepsilon)(\varepsilon \otimes \varepsilon)']``; third moments drop out for any symmetric shock distribution.

!!! warning "What the augmented state can and cannot carry"
    This package writes the second-order term over ``v \otimes v`` with ``v = [x; \varepsilon]``, so ``h_{xx}`` carries four blocks. The augmented recursion above is stated for the ``x \otimes x`` block. The ``\varepsilon \otimes \varepsilon`` block has mean ``\text{vec}(I)`` and is folded into the constant exactly. The ``x \otimes \varepsilon`` blocks are bilinear in the lagged state and the current shock: they are mean-zero and uncorrelated with the rest, so means and autocovariance cross-terms are unaffected, but they contribute to the **variance** and are not included. The understatement is ``O(\sigma^2)`` relative to the state variance --- on the RBC benchmark the closed form sits within Monte-Carlo error of a ``2 \times 10^6``-draw pruned simulation.

---

## Complete Example

This example solves the RBC model at all three perturbation orders, compares analytical and generalized IRFs, and validates a global projection solution with Euler equation errors.

```@example dsge_nonlinear
# First-order: certainty equivalence
sol1 = perturbation_solver(spec; order=1)

# Second-order: risk correction
sol2 = perturbation_solver(spec; order=2)

# Third-order: skewness and state-dependent risk
sol3 = perturbation_solver(spec; order=3)

# Compare simulated ergodic means (stochastic SS shift)
Y1 = simulate(sol1, 10000)
Y2 = simulate(sol2, 10000)
Y3 = simulate(sol3, 10000)

# Generalized IRFs at third order
girf = irf(sol3, 40; irf_type=:girf, n_draws=100)

# Global solution via Chebyshev collocation
proj = collocation_solver(spec; degree=5, grid=:tensor, max_iter=200)
report(proj)
```

The three ergodic means are sample averages over 10,000 draws from the global RNG, so they move from run to run; the *systematic* risk correction is the deterministic ``\frac{1}{2}\sigma^2 h_{\sigma\sigma}`` reported earlier, not the difference between these simulated averages, which at this shock size is swamped by Monte-Carlo noise. Read the stochastic steady-state shift off `psol.hσσ`, or pass `shock_draws` so the three orders see identical innovations.

```@example dsge_nonlinear
# Euler equation accuracy, with a fixed RNG so the numbers are reproducible
err = max_euler_error(proj; n_test=1000, rng=MersenneTwister(42))

# Bellman VFI for comparison (reuses the Recipe 4 solution)
err_vfi = max_euler_error(vfi; n_test=200, rng=MersenneTwister(42))

# Analytical moments for estimation targets
m = analytical_moments(sol1; lags=2)

(collocation = (proj.converged, err), vfi = (vfi.converged, err_vfi),
 n_moments = length(m))
```

```julia
plot_result(girf)
```

```@raw html
<iframe src="../assets/plots/dsge_girf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The second-order variance correction shifts the stochastic steady state of capital up by ``+0.0175`` and consumption down by the same amount: risk-averse agents over-accumulate capital as a buffer against productivity shocks, and pay for it with lower average consumption. Third-order perturbation adds the ``f_{\sigma\sigma v}`` term, which makes that precautionary wedge state-dependent — stronger when capital is scarce than when it is abundant. Collocation scores the Euler residual of a degree-5 Chebyshev policy; VFI scores the Chebyshev export of a grid maximizer. They are not required to match to six digits. Use VFI when the object of interest is ``V`` itself (welfare, Howard PE); use collocation or PFI when the object is a low Euler error on a smooth residual system.

---

## Common Pitfalls

1. **Explosive simulation without pruning**: Never extract the second-order coefficients and simulate manually. Always use `simulate(psol, T)`, which applies Kim et al. (2008) pruning automatically.

2. **Tensor grids in high dimensions**: A tensor grid with degree 5 and ``n_x = 6`` states requires ``6^6 = 46{,}656`` nodes. Use `grid=:smolyak` for ``n_x > 4``.

3. **Poor Euler errors**: If `max_euler_error` returns values above ``10^{-2}``, increase the polynomial `degree` or widen the state bounds via the `scale` parameter. Switching solver will not help — on a smooth model the three global solvers reach the same accuracy at the same degree. On a Smolyak grid, raise the overall level or let `adaptive=true` decide which states get it; a hand-chosen `smolyak_mu` vector is as likely to hurt as to help unless you have measured the curvature.

4. **Ignoring the `converged` flag**: every global solver returns a `ProjectionSolution` whether or not it converged. PFI needs 295 iterations on the RBC model above at `damping=0.5`; capped at `max_iter=200` it stops at a sup-norm of ``1.2 \times 10^{-7}`` and still hands back a usable-looking policy. Check `sol.converged` before trusting a global solution — and note the converse, that `converged=true` is a statement about the iteration stopping, not a certificate of accuracy.

5. **Comparing solvers by whether their policies agree**: two unconverged solvers disagreeing tells you nothing about which is right, and two converged ones can differ by more than either's error. Agreement is a cross-check; `max_euler_error` is the comparison that means something, because it scores each policy against the equilibrium conditions rather than against a rival approximation.

6. **Non-convergence of collocation**: The Gauss-Newton solver uses backtracking line search but can stall at local minima. Warm-start with `initial_coeffs` from a lower-degree solution.

7. **Lyapunov equation instability**: `solve_lyapunov` throws an error if the first-order solution has eigenvalues on or outside the unit circle. Check determinacy with `is_determined(sol)` before computing moments.

8. **Calling `vfi_solver` without a Bellman specification**: `vfi_solver` / `method=:vfi` require `utility` and `beta` (from `@dsge` or keywords). `transition` and `control_bounds` are inferred when omitted (`next_state=:residual` when control FOCs can be dropped). Explicit `next_state=:linear` throws if the linearized transition ignores the control. An Euler-only `ModelSpec` with no reward throws `ArgumentError` and names `pfi_solver`. Howard steps on VFI are policy evaluation of ``V``, not extra Euler Newton solves; Anderson acceleration remains a PFI keyword.

9. **Unseeded Monte Carlo**: `max_euler_error`, `irf(...; irf_type=:girf)` and `simulate` all draw from the global RNG by default. Pass `rng=MersenneTwister(...)` or `shock_draws` whenever a reported number must be reproducible.

10. **Trusting `residual_norm`**: A converged collocation solve drives ``\|R\|`` to zero *at the nodes* and says nothing about the points in between. The isotropic ``\mu = 3`` grid in [Anisotropic Smolyak Grids](@ref) reaches ``\|R\| \approx 6 \times 10^{-14}`` with a max Euler error of ``6.3 \times 10^{-4}`` — ten orders of magnitude apart. Report `max_euler_error`, or set `adaptive=true` and read `sol.euler_error`.

11. **`adaptive=true` on a tensor grid**: Refinement grows a Smolyak level set, so it throws an `ArgumentError` when `grid=:tensor` is requested explicitly. Leave `grid=:auto`, which upgrades to `:smolyak` automatically.

---

## References

- Andreasen, M. M., Fernandez-Villaverde, J., & Rubio-Ramirez, J. F. (2018). The Pruned State-Space System for Non-Linear DSGE Models: Theory and Empirical Applications. *Review of Economic Studies*, 85(1), 1--49. [DOI](https://doi.org/10.1093/restud/rdx037)

- Barraud, A. Y. (1977). A Numerical Algorithm to Solve ``A^T X A - X = Q``. *IEEE Transactions on Automatic Control*, 22(5), 883--885. [DOI](https://doi.org/10.1109/TAC.1977.1101604)

- Bartels, R. H., & Stewart, G. W. (1972). Solution of the Matrix Equation ``AX + XB = C``. *Communications of the ACM*, 15(9), 820--826. [DOI](https://doi.org/10.1145/361573.361582)

- Coleman, W. J. (1990). Solving the Stochastic Growth Model by Policy-Function Iteration. *Journal of Business & Economic Statistics*, 8(1), 27--29. [DOI](https://doi.org/10.1080/07350015.1990.10509769)

- Brumm, J., & Scheidegger, S. (2017). Using Adaptive Sparse Grids to Solve High-Dimensional Dynamic Models. *Econometrica*, 85(5), 1575--1612. [DOI](https://doi.org/10.3982/ECTA12216)

- Gerstner, T., & Griebel, M. (2003). Dimension-Adaptive Tensor-Product Quadrature. *Computing*, 71(1), 65--87. [DOI](https://doi.org/10.1007/s00607-003-0015-5)

- Judd, K. L. (1992). Projection Methods for Solving Aggregate Growth Models. *Journal of Economic Theory*, 58(2), 410--452. [DOI](https://doi.org/10.1016/0022-0531(92)90061-L)

- Kamenik, O. (2005). Solving SDGE Models: A New Algorithm for the Sylvester Equation. *Computational Economics*, 25(1--2), 167--187. [DOI](https://doi.org/10.1007/s10614-005-6280-y)

- Judd, K. L. (1998). *Numerical Methods in Economics*. MIT Press. ISBN: 978-0-262-10071-7.

- Judd, K. L., Maliar, L., Maliar, S., & Valero, R. (2014). Smolyak Method for Solving Dynamic Economic Models: Lagrange Interpolation, Anisotropic Grids and Adaptive Domain. *Journal of Economic Dynamics and Control*, 44, 92--123. [DOI](https://doi.org/10.1016/j.jedc.2014.03.003)

- Kim, J., Kim, S., Schaumburg, E., & Sims, C. A. (2008). Calculating and Using Second-Order Accurate Solutions of Discrete Time Dynamic Equilibrium Models. *Journal of Economic Dynamics and Control*, 32(11), 3397--3414. [DOI](https://doi.org/10.1016/j.jedc.2008.02.003)

- Koop, G., Pesaran, M. H., & Potter, S. M. (1996). Impulse Response Analysis in Nonlinear Multivariate Models. *Journal of Econometrics*, 74(1), 119--147. [DOI](https://doi.org/10.1016/0304-4076(95)01753-4)

- Santos, M. S., & Rust, J. (2003). Convergence Properties of Policy Iteration. *SIAM Journal on Control and Optimization*, 42(6), 2094--2115. [DOI](https://doi.org/10.1137/S0363012902399824)

- Schmitt-Grohe, S., & Uribe, M. (2004). Solving Dynamic General Equilibrium Models Using a Second-Order Approximation to the Policy Function. *Journal of Economic Dynamics and Control*, 28(4), 755--775. [DOI](https://doi.org/10.1016/S0165-1889(03)00043-5)

- Walker, H. F., & Ni, P. (2011). Anderson Acceleration for Fixed-Point Iterations. *SIAM Journal on Numerical Analysis*, 49(4), 1715--1735. [DOI](https://doi.org/10.1137/10078356X)
