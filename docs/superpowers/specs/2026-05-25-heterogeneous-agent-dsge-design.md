# Heterogeneous Agent DSGE — Design Spec

**Date:** 2026-05-25
**Version:** 0.5.0
**Issue:** #76
**References:** Krusell & Smith (1998), Reiter (2009), Auclert, Bardóczy, Rognlie & Straub (2021), Kaplan, Moll & Violante (2018), Young (2010), Carroll (2006), Ahn, Kaplan, Moll, Winberry & Wolf (2018), Bhandari, Bourany, Evans & Golosov (2023), Iskhakov, Jørgensen, Rust & Schjerning (2017)

## Overview

Add heterogeneous agent DSGE (HA-DSGE) modeling to MacroEconometricModels.jl. Households face idiosyncratic income risk and borrowing constraints, generating a non-degenerate wealth distribution that interacts with aggregate dynamics. Three solution methods: Reiter (2009) linearization, Sequence-Space Jacobian (Auclert et al. 2021), and Krusell-Smith (1998) simulation. Supports one-asset and two-asset (HANK) models. Bayesian estimation via the existing `estimate_dsge_bayes` infrastructure after linearization.

## Scope

**In scope:**
- `HADSGESpec{T}` type wrapping `DSGESpec{T}` for aggregate block
- `@dsge` macro extensions: `heterogeneous:`, `idiosyncratic_shocks:`, `aggregation:` blocks
- Endogenous Grid Method (EGM) for one-asset and two-asset individual problems
- VFI with Howard improvement as fallback solver
- Young (2010) non-stochastic histogram method for distribution tracking
- HA steady state solver (individual problem + distribution + market clearing)
- Reiter (2009) linearization with SVD dimensionality reduction
- Sequence-Space Jacobian with fake news algorithm and Ho-Kalman state-space conversion
- Krusell-Smith (1998) simulation-based bounded rationality
- `solve(spec::HADSGESpec; method=:ssj)` dispatching to HA solvers → `HADSGESolution{T}`
- `HADSGESolution{T}` embeds `DSGESolution{T}` for reuse of `irf`, `fevd`, `historical_decomposition`, `estimate_dsge_bayes`
- HA-specific analysis: `distribution_irf`, `inequality_irf`, `simulate_panel`
- Built-in examples: Krusell-Smith (1998), one-asset HANK, two-asset HANK
- `report()`, `show()`, `plot_result()` for all new types
- Bayesian estimation of HA-DSGE via existing SMC/MH + Kalman filter on reduced system
- Tests for all methods and examples
- Documentation page `docs/src/dsge_ha.md`

**Out of scope:**
- Continuous-time methods (Achdou-Moll HJB/KFE) — future extension
- Winberry (2018) parametric distribution approximation — future extension
- Lifecycle models (age dimension) — future extension
- Heterogeneous firms — future extension
- GPU acceleration — future extension

## File Structure

```
src/dsge/heterogeneous/
├── types.jl          — HADSGESpec, HAGrid, IncomeProcess, IndividualProblem,
│                       HASteadyState, HADSGESolution, KrusellSmithSolution
├── egm.jl            — EGM for one-asset (Carroll 2006) and two-asset (nested EGM)
├── individual_vfi.jl — VFI with Howard improvement (generic fallback)
├── distribution.jl   — Young (2010) histogram: transition matrix, forward iteration,
│                       stationary distribution
├── steady_state.jl   — HA steady state: iterate (individual + dist + prices)
├── reiter.jl         — Reiter (2009): sparse Jacobians, SVD reduction, → DSGESolution
├── ssj.jl            — Sequence-Space Jacobian: fake news algorithm, Ho-Kalman
├── krusell_smith.jl  — Krusell-Smith (1998): PLM simulation
├── parser.jl         — @dsge macro extensions for heterogeneity blocks
├── analysis.jl       — distribution_irf, inequality_irf, simulate_panel
├── display.jl        — report() / show() for HA types
└── examples.jl       — load_ha_example(:krusell_smith/:one_asset_hank/:two_asset_hank)
```

Test files:
```
test/dsge/
├── test_ha_dsge.jl          — types, steady state, EGM, distribution
├── test_ha_reiter.jl        — Reiter method, SVD reduction
├── test_ha_ssj.jl           — SSJ, fake news, Ho-Kalman
├── test_ha_krusell_smith.jl — KS simulation
└── test_ha_estimation.jl    — Bayesian estimation of HA models
```

## Type Definitions

### HAGrid{T} — Multi-Dimensional Individual State Space

```julia
struct HAGrid{T<:AbstractFloat}
    grids::Vector{Vector{T}}      # [asset1_grid, asset2_grid, ...]
    n_points::Vector{Int}         # [N_a1, N_a2, ...] per asset dimension
    n_dims::Int                   # 1 (one-asset) or 2 (two-asset)
    n_income::Int                 # N_e income states
    bounds::Matrix{T}             # n_dims × 2 [lower upper]
    labels::Vector{Symbol}        # [:liquid, :illiquid] or [:assets]
    total_individual_states::Int  # prod(n_points) * n_income
end
```

Asset grids use denser spacing near the borrowing constraint (double-exponential or Chebyshev nodes). Constructor: `HAGrid(; assets=(-2.0, 50.0, 200), income_states=7)` for one-asset, `HAGrid(; liquid=(-2.0, 50.0, 50), illiquid=(0.0, 100.0, 50), income_states=7)` for two-asset.

### IncomeProcess{T} — Idiosyncratic Markov Chain

```julia
struct IncomeProcess{T<:AbstractFloat}
    transition::Matrix{T}         # N_e × N_e Markov transition matrix
    states::Vector{T}             # N_e income/productivity levels
    stationary_dist::Vector{T}    # N_e stationary distribution of chain
    labels::Vector{Symbol}        # state labels
end
```

Constructors:
- `rouwenhorst(ρ, σ, n_states)` — Rouwenhorst (1995) discretization of AR(1)
- `tauchen(ρ, σ, n_states; m=3)` — Tauchen (1986) method
- `IncomeProcess(transition, states)` — explicit specification

### IndividualProblem{T} — Household Optimization

```julia
struct IndividualProblem{T<:AbstractFloat}
    utility::Function             # u(c) → scalar
    utility_prime::Function       # u'(c) → scalar
    utility_prime_inv::Function   # (u')^{-1}(x) → scalar (for EGM)
    beta::T                       # discount factor
    budget_fn::Function           # (assets..., e, prices...) → cash-on-hand
    borrowing_constraint::Vector{T}  # lower bounds per asset dimension
    adjustment_cost::Union{Nothing,Function}  # χ(d, a) for illiquid (two-asset only)
    n_asset_dims::Int             # 1 or 2
end
```

### HADSGESpec{T} — Heterogeneous Agent DSGE Specification

```julia
struct HADSGESpec{T<:AbstractFloat}
    aggregate_spec::DSGESpec{T}   # aggregate block (NK equations etc.)
    individual::IndividualProblem{T}
    income::IncomeProcess{T}
    grid::HAGrid{T}
    aggregation::Vector{Pair{Symbol,Function}}  # :K => (dist, grid) → aggregate
    het_params::Dict{Symbol,T}    # heterogeneity-specific parameters
    n_assets::Int                 # grid.n_dims
    n_income::Int                 # grid.n_income
end
```

The `@dsge` parser detects `heterogeneous:` blocks and returns an `HADSGESpec{T}` (which is NOT a subtype of `DSGESpec` — it wraps one). This keeps the existing `DSGESpec` type unchanged. The `solve` function gains new methods via Julia's multiple dispatch: `solve(spec::HADSGESpec; method=:ssj)` is a separate method from `solve(spec::DSGESpec; method=:gensys)`, requiring no changes to existing representative-agent code paths.

### HASteadyState{T} — Stationary Equilibrium

```julia
struct HASteadyState{T<:AbstractFloat}
    policies::Dict{Symbol,Array{T}}      # :savings, :consumption, :deposit, etc.
    distribution::Array{T}               # N_a1 [× N_a2] × N_e, sums to 1
    value_fn::Array{T}                   # N_a1 [× N_a2] × N_e
    prices::Dict{Symbol,T}              # :r, :w, :T, :r_b, :r_a, etc.
    aggregates::Dict{Symbol,T}          # :K, :C, :Y, :N, etc.
    grid::HAGrid{T}
    income::IncomeProcess{T}
    converged::Bool
    iterations::Int
    euler_error::T                       # max |1 - LHS/RHS| of Euler equation
    excess_demand::T                     # |K_supply - K_demand| at convergence
end
```

### HADSGESolution{T} — Linearized HA Solution

```julia
struct HADSGESolution{T<:AbstractFloat}
    steady_state::HASteadyState{T}
    linear_solution::DSGESolution{T}     # reduced-dimension linear RE solution
    method::Symbol                       # :reiter or :ssj
    spec::HADSGESpec{T}
    reduction_basis::Matrix{T}           # U_k: maps reduced → full distribution
    n_full_states::Int                   # original dimension before reduction
    n_reduced::Int                       # effective reduced dimension
    explained_variance::T                # fraction captured by reduction
    jacobians::Union{Nothing,Dict{Symbol,Matrix{T}}}  # SSJ block Jacobians
end
```

The `linear_solution::DSGESolution{T}` field is the key integration point. All existing functions that accept `DSGESolution` — `irf()`, `fevd()`, `historical_decomposition()`, `simulate()`, `estimate_dsge_bayes()` — work on `sol.linear_solution` transparently. We add method dispatch:

```julia
irf(sol::HADSGESolution, H; kwargs...) = irf(sol.linear_solution, H; kwargs...)
fevd(sol::HADSGESolution, H; kwargs...) = fevd(sol.linear_solution, H; kwargs...)
simulate(sol::HADSGESolution, T; kwargs...) = simulate(sol.linear_solution, T; kwargs...)
```

### KrusellSmithSolution{T} — Simulation-Based Solution

```julia
struct KrusellSmithSolution{T<:AbstractFloat}
    steady_state::HASteadyState{T}
    plm_coefficients::Dict{Symbol,Vector{T}}  # perceived law of motion
    r_squared::Dict{Symbol,T}                 # forecast accuracy per variable
    spec::HADSGESpec{T}
    converged::Bool
    iterations::Int
end
```

## Algorithm Details

### 1. Endogenous Grid Method (EGM)

#### One-Asset (Carroll 2006)

Input: current value function V_old (or consumption policy), grid, income process, prices.
Output: updated policy c(a, e), savings a'(a, e).

```
For each income state e_j (j = 1, ..., N_e):
  1. Fix end-of-period asset grid: a'_i for i = 1, ..., N_a
  2. Expected marginal utility:
     EMU_i = β(1+r) Σ_{j'} π(j,j') u'(c_old(a'_i, e_j'))
     where c_old is interpolated from previous iteration
  3. Current consumption via Euler inversion:
     c_i = (u')^{-1}(EMU_i)
  4. Beginning-of-period assets (endogenous):
     a_i = (c_i + a'_i - w·e_j) / (1 + r)
  5. Interpolate (a_endo, c_endo) back to exogenous grid a_exo
  6. Borrowing constraint: for a < a_min, set c = (1+r)a + w·e (consume all)
```

Convergence: iterate until max|c_new - c_old| < 1e-10.

#### Two-Asset Nested EGM (Kaplan-Moll-Violante 2018)

The two-asset problem has liquid b and illiquid a with adjustment cost χ(d, a):

```
V(b, a, e) = max_{c, b', d} u(c) + β E[V(b', a', e')]
  s.t. c + b' + d + χ(d, a) = (1+r_b)b + w·e + T
       a' = (1+r_a)a + d
       b' ≥ b_min
```

Solution via nested EGM:

```
OUTER: loop over deposit grid {d_k}
  For each (a_i, e_j, d_k):
    Compute a'_ik = (1+r_a)·a_i + d_k
    
    INNER: standard EGM on liquid dimension
      For each b'_l:
        EMU = β Σ_{j'} π(j,j') (1+r_b) u'(c(b'_l, a'_ik, e_j'))
        c_inner = (u')^{-1}(EMU)
        b_begin = (c_inner + b'_l + d_k + χ(d_k, a_i) - w·e_j - T) / (1+r_b)
      Apply upper envelope (Iskhakov et al. 2017) for non-monotonicity
      Interpolate to exogenous b grid → c(b, a_i, e_j | d_k)
    
    Compute V(b, a_i, e_j | d_k) for this deposit choice

DEPOSIT OPTIMIZATION: for each (b, a, e)
  Primary: FOC-based EGM on illiquid margin
    u'(c) · (1 + χ'(d, a)) = β(1+r_a) E[V_a(b', a', e')]
    Invert for c, recover d from budget constraint
  Fallback: golden section search over {d_k} if FOC-EGM fails
    (handles non-convex regions near constraints)
```

#### Upper Envelope (Iskhakov et al. 2017)

When the EGM endogenous grid is non-monotonic (kinks at borrowing constraint):

```
1. Sort (a_endo, c_endo) pairs by a_endo
2. Detect crossings: regions where increasing a_endo maps to lower c
3. At each crossing, retain the segment yielding higher utility (upper envelope)
4. Interpolate the cleaned policy onto the exogenous grid
```

### 2. Distribution Tracking (Young 2010)

#### Build Sparse Transition Matrix

Given policy a' = g(a_i, e_j) on grid {a_k}:

```
For each (a_i, e_j):
  a' = g(a_i, e_j)
  Find k such that a_k ≤ a' ≤ a_{k+1}
  Lottery weight: ω = (a_{k+1} - a') / (a_{k+1} - a_k)
  For each e_j' reachable from e_j:
    Λ[(k, j'), (i, j)] += ω · π(j, j')
    Λ[(k+1, j'), (i, j)] += (1-ω) · π(j, j')
```

The transition matrix Λ is sparse: at most 2·N_e nonzeros per column. Stored as `SparseMatrixCSC{T, Int}`. Dimension: (N_a·N_e) × (N_a·N_e).

For two-asset: bilinear lottery across (b, a) dimensions, 4·N_e nonzeros per column.

#### Stationary Distribution

```
D_ss = eigenvector of Λ with eigenvalue 1
Computed via power iteration: D_{n+1} = Λ · D_n, normalize to sum = 1
Convergence: max|D_{n+1} - D_n| < 1e-12
Typically 200-500 iterations.
```

#### Forward Iteration (for dynamics)

```
D_{t+1} = Λ(g_t) · D_t
where g_t is the policy at time t (perturbed by aggregate shocks)
```

### 3. Steady State Solver

```
ha_steady_state(spec; tol=1e-10, max_iter=500) → HASteadyState

1. Initialize:
   - Guess prices: r = α·(K_guess)^(α-1)·N^(1-α) - δ, w = (1-α)·(K_guess)^α·N^(-α)
   - Initialize value function: V_0(a, e) = u((1+r)a + w·e) / (1-β)
   
2. OUTER LOOP (bisection/Brent on interest rate r):
   a. Solve individual problem (EGM/VFI until convergence)
   b. Build transition matrix Λ from converged policy
   c. Compute stationary distribution D_ss via power iteration
   d. Compute aggregate capital supply: K_s = Σ_{i,j} a_i · D_ss(a_i, e_j)
   e. Compute aggregate capital demand: K_d from firm FOC given r
   f. Excess demand: K_s - K_d
   g. Update r via bisection (or Brent's method for faster convergence)
   
3. Convergence: |K_s - K_d| < tol

4. Compute remaining aggregates: Y, C, N from cleared prices
5. Compute Euler equation error: max over grid of |1 - β(1+r)E[u'(c')]/u'(c)|
6. Return HASteadyState
```

### 4. Reiter Method (2009)

```
reiter_solve(spec, ss; n_reduced=50) → HADSGESolution

1. Define expanded state vector X_t:
   - Policy deviations: dc(a_i, e_j) for all (i,j) — N = N_a · N_e variables
   - Distribution deviations: dD(a_i, e_j) for all (i,j) — N-1 variables (sums to 0)
   - Aggregate variables: from DSGESpec — n_agg variables
   Total dimension: 2N - 1 + n_agg

2. Write system F(X_t, X_{t+1}, ε_t) = 0:
   Block 1 (Euler equations, N equations):
     β(1+r_t) Σ_{j'} π(j,j') u'(c_{t+1}(a'_t(a_i,e_j), e_j')) - u'(c_t(a_i,e_j)) = 0
   Block 2 (Distribution evolution, N-1 equations):
     D_{t+1} - Λ(g_t) · D_t = 0
   Block 3 (Aggregation, n_agg_link equations):
     K_t - Σ a_i D_t(a_i, e_j) = 0, etc.
   Block 4 (Aggregate model, from DSGESpec):
     Standard DSGE equations evaluated at aggregate variables

3. Compute Jacobians (SPARSE):
   ∂F/∂X_{t+1} = A (sparse, block structure)
   ∂F/∂X_t = B (sparse, block structure)
   ∂F/∂ε_t = C (sparse)
   
   Sparsity structure:
   - Euler block: banded in assets (width 2-4), block in income (N_e × N_e)
   - Distribution block: Λ is already sparse (2·N_e nonzeros per column)
   - Aggregate block: one dense row per aggregation equation, rest sparse
   
   Exploit sparsity via SparseDiffTools.jl: automatic sparsity detection +
   matrix coloring reduces ForwardDiff cost from O(2N + n_agg) to O(colors)
   where colors ≈ 5-10 for the Euler block.

4. SVD dimensionality reduction (Ahn et al. 2018, Bhandari et al. 2023):
   a. Extract distribution block: dD_{t+1} = Λ_D · dD_t + Λ_X · dX_t
   b. Simulate distribution responses for T=500 periods to random shocks
   c. Stack response vectors: M = [dD_1, ..., dD_T] ∈ R^{(N-1) × T}
   d. SVD: M = U Σ V'. Retain top k columns of U where Σ_k/Σ_1 > 1e-6
      Typically k = 20-50 for one-asset, 50-80 for two-asset
   e. Project: dD_t ≈ U_k · d̃_t where d̃_t ∈ R^k
   f. Reduced system dimension: k + n_agg (typically 25-60)
   g. Explained variance: Σ(Σ²_retained) / Σ(Σ²_total)

5. Cast reduced system in Sims form:
   Γ₀ · X̃_{t+1} = Γ₁ · X̃_t + Ψ · ε_t + Π · η_t
   where X̃ = [d̃_t; aggregate_vars_t]

6. Solve with existing gensys() (or blanchard_kahn/klein):
   → G1, impact, C_sol, eu, eigenvalues
   → DSGESolution{T}

7. Return HADSGESolution with:
   - linear_solution = DSGESolution (reduced)
   - reduction_basis = U_k
   - explained_variance = fraction
```

### 5. Sequence-Space Jacobian (Auclert et al. 2021)

```
ssj_solve(spec, ss; T=300, n_reduced=30) → HADSGESolution

1. Compute HA block Jacobian via "fake news" algorithm:
   
   For each input price sequence p ∈ {r, w, T, ...}:
     For each time s = 0, 1, ..., T-1:
       
       a. BACKWARD STEP: starting from steady-state policy at t > s,
          perturb price at time s, iterate EGM backward from t=s to t=0
          → captures how expectations of future price changes affect current decisions
          Result: expectation vector E_s ∈ R^{N_a·N_e}
       
       b. FORWARD STEP: starting from steady-state distribution at t < s,
          apply perturbed policy at time s, iterate distribution forward from t=s to T
          → captures how changed decisions propagate through the distribution
          Result: distribution vectors {D_t}_t=s^T
       
       c. AGGREGATE: for each output y ∈ {K_supply, C, ...}:
          F[t, s] = Σ_{i,j} a_i · D_t(a_i, e_j)  (fake news matrix entry)
     
     Assemble true Jacobian from fake news matrix:
       J[t, s] = Σ_{τ=0}^{t} F[τ, s]  (cumulative sum along columns)
     
     J is T × T, one per (output, input) pair
   
   Computational cost: O(T) backward EGM steps + O(T) forward dist iterations
   Each step is O(N_a · N_e) → total O(T · N_a · N_e) per input variable

2. Compute simple block Jacobians (aggregate equations):
   Firm FOC, Taylor rule, Phillips curve, resource constraint
   → analytical or ForwardDiff, small (n_agg × n_agg) per time step
   → T × T Jacobians with known structure (lower-triangular for causal blocks)

3. Assemble GE system:
   H(U) = 0 where U = {r_0,...,r_{T-1}, w_0,...,w_{T-1}, ...} is (T·n_unknowns)
   H_U = block-sparse Jacobian (DAG structure from model equations)
   
4. GE impulse responses:
   dU = -H_U⁻¹ · H_Z · dZ
   where dZ is the shock sequence (e.g., ε_Z at t=0)
   Solved via sparse LU or iterative methods (block structure)

5. Convert to state-space via Ho-Kalman algorithm:
   a. From IRFs, build Markov parameters: {C·A^k·B} for k=0,...,T-1
   b. Construct Hankel matrix:
      H = [h_1  h_2  ... h_p;
           h_2  h_3  ... h_{p+1};
           ...              ...]
   c. SVD(H) → retain top k singular values
   d. Extract minimal realization: (A_k, B_k, C_k, D_k) of dimension k
   e. Map to DSGESolution: G1 = A_k, impact = B_k
   
   Dimension k is typically 15-30 for one-asset, 30-50 for two-asset

6. Return HADSGESolution with:
   - linear_solution = DSGESolution (from Ho-Kalman)
   - jacobians = Dict of T×T block Jacobians (stored for re-computation)
   - reduction_basis, n_reduced, explained_variance
```

### 6. Krusell-Smith (1998)

```
krusell_smith_solve(spec, ss; T_sim=11000, T_burn=1000, n_moments=1) → KrusellSmithSolution

1. Guess perceived law of motion (PLM):
   log K_{t+1} = b₀(z_t) + b₁(z_t) · log K_t
   (separate intercept/slope per aggregate state z)

2. Simulate:
   For t = 1, ..., T_sim:
     a. Draw aggregate shock z_t from its Markov chain
     b. Compute prices: r_t = α·z_t·K_t^{α-1}·N^{1-α} - δ, w_t = (1-α)·z_t·K_t^α·N^{-α}
     c. Solve individual EGM given prices (one step from previous iteration's policy)
     d. Forward-iterate distribution: D_{t+1} = Λ(g_t) · D_t
     e. Realized aggregate capital: K_{t+1}^{actual} = Σ a_i · D_{t+1}(a_i, e_j)
     f. PLM-predicted capital: K_{t+1}^{PLM} from perceived law of motion

3. Update PLM coefficients:
   Regress log K_{t+1}^{actual} on (1, log K_t) for each z state
   (use t > T_burn observations only)

4. Check convergence:
   - |b_new - b_old| < tol for all coefficients
   - R² > 0.9999 for all regressions

5. Repeat 2-4 until convergence (typically 5-15 outer iterations)

6. Return KrusellSmithSolution with PLM coefficients and R² values
```

### 7. Dimensionality Reduction Details

#### SVD Truncation for Reiter

The distribution is (N_a·N_e - 1)-dimensional. After linearization, the distribution deviation dD_t evolves as:

```
dD_{t+1} = Λ_ss · dD_t + G_policy · dpolicy_t + G_agg · dagg_t
```

The "reachable subspace" is found by:
1. Starting from dD_0 = 0, simulate T responses to unit shocks in each aggregate variable
2. Collect all dD_t vectors as columns of matrix M
3. SVD of M: retain top k singular vectors (U_k) capturing 99.9% of variance
4. Project system: d̃_t = U_k' · dD_t, reconstruct as dD_t ≈ U_k · d̃_t

This reduces the distribution from ~1400 states (200 × 7) to ~30-50 states.

#### Ho-Kalman for SSJ

The SSJ produces impulse responses {h_0, h_1, ..., h_{T-1}} (matrices for multi-variable case). The Ho-Kalman algorithm finds a minimal state-space realization:

```
1. Build Hankel matrix H ∈ R^{(n_out·p) × (n_in·q)} from Markov parameters
   H[i·n_out:(i+1)·n_out, j·n_in:(j+1)·n_in] = h_{i+j+1}
   with p = q = T/2

2. SVD: H = U Σ V'
   Retain top k singular values where Σ_k/Σ_1 > tolerance

3. Observability matrix: O = U_k · Σ_k^{1/2}
   Controllability matrix: C = Σ_k^{1/2} · V_k'

4. State-space matrices:
   C_ss = O[1:n_out, :]                    (observation)
   A = Σ_k^{-1/2} · U_k' · H_shifted · V_k · Σ_k^{-1/2}  (transition)
   B = C[:, 1:n_in]                        (impact)
   D = h_0                                 (direct feedthrough)
   
   where H_shifted is the Hankel matrix shifted by one time step
```

### 8. Sparse Matrix Strategy

The Reiter Jacobian has block structure:

```
Full system dimension: (2N-1+n_agg) × (2N-1+n_agg) where N = N_a · N_e

              [ policy (N)  | dist (N-1) | agg (n_agg) ]
Euler (N)     [ SPARSE-BAND |     0      | DENSE-THIN  ]
Dist (N-1)    [ SPARSE      |  SPARSE(Λ) |     0       ]
Agg (n_agg)   [     0       | DENSE-WIDE | DENSE-SMALL ]
```

Sparsity exploitation:
- Euler Jacobian: banded (width 2-4) in asset dimension, N_e × N_e blocks in income
  → ~O(N_a · N_e²) nonzeros vs O(N²) total → ~0.5% fill for N_a=200, N_e=7
- Distribution Jacobian: Λ has 2·N_e nonzeros per column → ~1% fill
- Aggregate rows: N-1 dense entries but only n_agg rows → negligible

Use `SparseDiffTools.jl` for automatic sparsity detection and matrix coloring:
- Detect sparsity pattern from a single evaluation pass
- Compute optimal column coloring (typically 5-10 colors for banded+block structure)
- Compressed ForwardDiff: evaluate Jacobian in O(colors) function evaluations instead of O(N)

For the reduced system (post-SVD), convert to dense and use existing `gensys()`.

### 9. Performance Considerations

Threading:
- EGM across income states: `Threads.@threads for j in 1:N_e`
- SSJ backward/forward sweeps: parallelizable across time
- Distribution forward iteration: sparse matvec (SparseArrays optimized)
- Two-asset outer loop: `Threads.@threads for (a_i, e_j)` pairs

Expected performance:

| Operation | One-asset (200×7) | Two-asset (50²×7) |
|---|---|---|
| Single EGM step | ~0.1ms | ~5ms |
| Steady state convergence | ~0.5s | ~10s |
| Reiter: sparse Jacobian | ~0.5s | ~5s |
| Reiter: SVD reduction | ~0.2s | ~2s |
| Reiter: reduced gensys | ~0.05s | ~0.2s |
| SSJ: full Jacobian (T=300) | ~1s | ~30s |
| SSJ: Ho-Kalman conversion | ~0.1s | ~0.5s |
| KS: one outer iteration | ~5s | ~60s |
| Single likelihood evaluation | ~0.1s (reduced) | ~0.5s (reduced) |
| Bayesian estimation (50k draws) | ~1.5h | ~7h |

### 10. Estimation Integration

The linearized Reiter/SSJ solution produces a `DSGESolution{T}`. The existing `estimate_dsge_bayes` works with one modification: the inner loop must re-solve the HA steady state for each parameter draw.

```
estimate_dsge_bayes(spec::HADSGESpec, data, observables, prior;
                    ha_method=:ssj, n_reduced=30, kwargs...)

For each parameter draw θ in the SMC/MH sampler:
  1. Update spec parameters: spec_θ = update_params(spec, θ)
  2. Solve HA steady state: ss = ha_steady_state(spec_θ)
  3. Linearize: sol = solve(spec_θ; method=ha_method, ss=ss, n_reduced=n_reduced)
  4. Build state space: ss_model = _build_state_space(sol.linear_solution, observables)
  5. Kalman filter: log_lik = _kalman_log_likelihood(ss_model, data)
  6. Return log_lik + log_prior(θ)
```

Steps 2-3 are the expensive part. For efficiency:
- Cache the steady-state grid and income process (only prices change)
- Warm-start EGM from the previous draw's policy
- Use the reduced (not full) system for Kalman filtering

## @dsge Macro Syntax Extensions

### One-Asset Krusell-Smith Example

```julia
spec = @dsge begin
    parameters: α = 0.36, β = 0.99, δ = 0.025, ρ_z = 0.95, σ_z = 0.007
    endogenous: Y, K, r, w, Z
    exogenous: ε_Z
    
    heterogeneous: households
        assets: a in [0.0, 200.0], n_grid = 200
        utility: log(c)
        discount: β
        budget: c + a_next = (1 + r) * a + w * e
        borrowing: a_next >= 0.0
    end
    
    idiosyncratic_shocks: households
        e ~ Rouwenhorst(ρ = 0.966, σ = 0.5, n_states = 7)
    end
    
    aggregation: households
        K = sum(a, distribution)
    end
    
    Y[t] = Z[t] * K[t-1]^α
    r[t] = α * Z[t] * K[t-1]^(α-1) - δ
    w[t] = (1-α) * Z[t] * K[t-1]^α
    Z[t] = ρ_z * Z[t-1] + σ_z * ε_Z[t]
end
```

### One-Asset HANK Example

```julia
spec = @dsge begin
    parameters: α = 0.36, β = 0.986, δ = 0.025, σ_c = 1.0,
                φ_π = 1.5, φ_y = 0.125, ρ_i = 0.8,
                θ_p = 0.75, ε_p = 6.0,
                ρ_z = 0.95, σ_z = 0.007, σ_m = 0.0025
    endogenous: Y, C, K, N, w, r, mc, π_rate, i, Z, div
    exogenous: ε_Z, ε_m
    
    heterogeneous: households
        assets: b in [-2.0, 50.0], n_grid = 200
        utility: c^(1 - σ_c) / (1 - σ_c)
        discount: β
        budget: c + b_next = (1 + r) * b + w * e + div
        borrowing: b_next >= -2.0
    end
    
    idiosyncratic_shocks: households
        e ~ Rouwenhorst(ρ = 0.966, σ = 0.5, n_states = 7)
    end
    
    aggregation: households
        K = sum(b, distribution)
        C = sum(c, distribution)
    end
    
    # Aggregate New Keynesian block (11 equations for 11 endogenous)
    Y[t] = Z[t] * K[t-1]^α * N[t]^(1-α)
    N[t] = 1.0
    mc[t] = w[t] / ((1-α) * Z[t] * K[t-1]^α * N[t]^(-α))
    w[t] = (1-α) * mc[t] * Y[t] / N[t]
    r[t] = α * mc[t] * Y[t] / K[t-1] - δ
    div[t] = Y[t] - w[t] * N[t] - r[t] * K[t-1] - δ * K[t-1]
    π_rate[t] = β * E[t](π_rate[t+1]) + (ε_p-1)/θ_p * mc[t]
    i[t] = ρ_i * i[t-1] + (1-ρ_i) * (φ_π * π_rate[t] + φ_y * Y[t]) + σ_m * ε_m[t]
    C[t] = Y[t] - δ * K[t-1]
    1 + r[t] = (1 + i[t-1]) / (1 + π_rate[t])
    Z[t] = ρ_z * Z[t-1] + σ_z * ε_Z[t]
end
```

### Two-Asset HANK Example

```julia
spec = @dsge begin
    parameters: α = 0.36, β = 0.986, δ = 0.025, σ_c = 1.0,
                r_b_ss = 0.005, r_a_ss = 0.01,
                χ_0 = 0.5, χ_1 = 2.0,
                φ_π = 1.5, ρ_i = 0.8, ρ_z = 0.95, σ_z = 0.007
    endogenous: Y, C, K_liq, K_illiq, N, w, r_b, r_a, π_rate, i, Z, div
    exogenous: ε_Z, ε_m
    
    heterogeneous: households
        assets: b in [-2.0, 50.0] n_grid = 50,
                a in [0.0, 100.0] n_grid = 50
        utility: c^(1 - σ_c) / (1 - σ_c)
        discount: β
        budget: c + b_next + d + χ_0 * abs(d / a)^χ_1 * a = (1 + r_b) * b + w * e + div
        illiquid: a_next = (1 + r_a) * a + d
        borrowing: b_next >= -2.0
    end
    
    idiosyncratic_shocks: households
        e ~ Rouwenhorst(ρ = 0.966, σ = 0.5, n_states = 7)
    end
    
    aggregation: households
        K_liq = sum(b, distribution)
        K_illiq = sum(a, distribution)
        C = sum(c, distribution)
    end
    
    # Aggregate equations (12 equations for 12 endogenous)
    Y[t] = Z[t] * (K_liq[t-1] + K_illiq[t-1])^α * N[t]^(1-α)
    N[t] = 1.0
    w[t] = (1-α) * Y[t] / N[t]
    r_b[t] = i[t-1] - π_rate[t]
    r_a[t] = α * Y[t] / (K_liq[t-1] + K_illiq[t-1]) - δ
    div[t] = Y[t] - w[t] * N[t] - r_a[t] * (K_liq[t-1] + K_illiq[t-1]) - δ * (K_liq[t-1] + K_illiq[t-1])
    C[t] = Y[t] - δ * (K_liq[t-1] + K_illiq[t-1])
    π_rate[t] = 0.99 * E[t](π_rate[t+1]) + 0.1 * (Y[t] - 1.0)
    i[t] = ρ_i * i[t-1] + (1-ρ_i) * (φ_π * π_rate[t] + 0.125 * Y[t])
    Z[t] = ρ_z * Z[t-1] + σ_z * ε_Z[t]
end
```

### Parser Changes

The `@dsge` macro parser (`src/dsge/parser.jl`) detects `heterogeneous:`, `idiosyncratic_shocks:`, and `aggregation:` blocks. When present, it:

1. Parses the aggregate equations into a standard `DSGESpec{T}` (unchanged logic)
2. Parses heterogeneous blocks into `IndividualProblem{T}`, `HAGrid{T}`, `IncomeProcess{T}`
3. Constructs aggregation functions from the `aggregation:` block
4. Returns `HADSGESpec{T}` wrapping the `DSGESpec{T}` plus heterogeneous components

New block parsing is isolated in `src/dsge/heterogeneous/parser.jl`, called from the main parser when heterogeneous blocks are detected.

## Public API Summary

### New Exported Types
- `HADSGESpec{T}`
- `HAGrid{T}`
- `IncomeProcess{T}`
- `IndividualProblem{T}`
- `HASteadyState{T}`
- `HADSGESolution{T}`
- `KrusellSmithSolution{T}`

### New Exported Functions

| Function | Description |
|---|---|
| `ha_steady_state(spec; method, tol, max_iter)` | Solve HA stationary equilibrium |
| `solve(spec::HADSGESpec; method)` | Dispatch to :reiter, :ssj, :krusell_smith |
| `rouwenhorst(ρ, σ, n)` | Rouwenhorst income discretization |
| `tauchen(ρ, σ, n; m)` | Tauchen income discretization |
| `distribution_irf(sol, H)` | Distribution impulse responses |
| `inequality_irf(sol, H)` | Gini/percentile responses to shocks |
| `simulate_panel(sol, N, T)` | Simulate individual panel data |
| `load_ha_example(name)` | Built-in HA model specifications |

### Extended Existing Functions (via dispatch)

| Function | New dispatch |
|---|---|
| `irf(sol::HADSGESolution, H)` | Delegates to `irf(sol.linear_solution, H)` |
| `fevd(sol::HADSGESolution, H)` | Delegates to `fevd(sol.linear_solution, H)` |
| `historical_decomposition(sol::HADSGESolution, ...)` | Delegates to linear_solution |
| `simulate(sol::HADSGESolution, T)` | Delegates to linear_solution |
| `estimate_dsge_bayes(spec::HADSGESpec, ...)` | HA steady state + linearize in inner loop |
| `report(ss::HASteadyState)` | Prices, Gini, distribution stats |
| `report(sol::HADSGESolution)` | Solution summary + reduction info |
| `report(sol::KrusellSmithSolution)` | PLM coefficients + R² |
| `plot_result(ss::HASteadyState)` | Wealth distribution, Lorenz curve |
| `plot_result(sol::HADSGESolution)` | Aggregate IRFs + distribution dynamics |

### Dependencies

New (lightweight):
- `SparseDiffTools.jl` — sparse Jacobian computation via matrix coloring
- `ArnoldiMethod.jl` — Krylov eigensolver for SVD of large sparse distribution blocks

Existing (already in Project.toml):
- `SparseArrays` (stdlib)
- `LinearAlgebra` (stdlib)
- `ForwardDiff` (already imported)
- `Optim` (already imported, for golden section in two-asset)

## Testing Strategy

1. **Unit tests per file** — EGM convergence, distribution stationarity, Rouwenhorst accuracy
2. **Krusell-Smith reference** — compare steady state K, r, Gini to published values
3. **Method consistency** — Reiter and SSJ IRFs must agree within tolerance for same model
4. **Estimation smoke test** — Bayesian estimation on simulated data, posterior covers true values
5. **Two-asset** — verify nested EGM matches VFI solution, adjustment cost works correctly
6. **Dimension reduction** — verify explained variance > 99.9% for chosen n_reduced
7. **Existing DSGE tests unchanged** — no regressions in representative-agent functionality

## Documentation

New page: `docs/src/dsge_ha.md`

Structure follows `docrule.md`:
- H1: Heterogeneous Agent DSGE Models
- Introduction: brief on HA-DSGE motivation
- Quick Start: 4 recipes (KS steady state, one-asset HANK IRFs, two-asset HANK, estimation)
- H2 sections: Individual Problem, Distribution, Steady State, Reiter Method, SSJ Method, Krusell-Smith, Estimation, Two-Asset HANK
- Complete Example: one-asset HANK with estimation
- Common Pitfalls
- References
