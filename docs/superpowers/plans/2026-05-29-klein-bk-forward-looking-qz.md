# Klein/BK Forward-Looking QZ Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `solve(:klein)` and `solve(:blanchard_kahn)` correct for forward-looking DSGE models (agreeing with gensys), and make gensys's determinacy flags correct, via a shared companion-QZ solver of the quadratic matrix equation.

**Architecture:** A new internal `_solve_qz_quadratic(f_0,f_1,f_lead,f_ε)` solves `f_lead·G²+f_0·G+f_1=0` through the QZ decomposition of the companion pencil and recovers `G`, `impact`, determinacy `eu`, and a residual self-check. `klein` and `blanchard_kahn` become thin `(ld,spec)` wrappers over it; `solve(:gensys)` keeps its UC-based `G1`/`impact` but takes `eu` from the core.

**Tech Stack:** Julia, `LinearAlgebra` (`schur`, `ordschur` generalized Schur / QZ), existing `MacroEconometricModels` DSGE module.

**Reference spec:** `docs/superpowers/specs/2026-05-29-klein-bk-forward-looking-qz-design.md`

**Test rules (repo):** Never run the full suite locally; never serial. Run only the relevant files with `MACRO_MULTIPROCESS_TESTS=1`. To run a single test file, use the harness preamble:

```bash
MACRO_MULTIPROCESS_TESTS=1 julia --project=. -e '
using Test, MacroEconometricModels
const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
include("test/fixtures.jl")
@testset "run" begin include("test/<FILE>") end
'
```

**Key facts (verified in source):**
- `LinearDSGE` fields: `Gamma0, Gamma1, C, Psi, Pi, spec`. Recover Jacobians: `f_0=Gamma0`, `f_1=-Gamma1`, `f_ε=-Psi`, `f_lead=_dsge_jacobian(spec, spec.steady_state, :lead)`.
- `DSGESolution{T}` fields, in order: `G1, impact, C_sol, eu, method, eigenvalues, spec, linear`.
- `is_stable(sol::DSGESolution) = maximum(abs.(eigvals(sol.G1))) < 1.0` — uses `G1`, so the `eigenvalues` field is display-only.
- `is_determined(sol) = sol.eu[1]==1 && sol.eu[2]==1`.
- `blanchard_kahn(ld, spec)` already returns a `DSGESolution`; `klein` currently returns a named tuple wrapped by `solve`. After this plan, both take `(ld, spec)` and return `DSGESolution`.
- `_dsge_jacobian(spec, y_ss, which)` and `_dsge_jacobian_shocks(spec, y_ss)` live in `src/dsge/linearize.jl` (same module).
- The standalone `gensys(Γ0,Γ1,C,Ψ,Π)` function stays exported and unchanged; coverage tests call it directly.

---

## Task 1: Shared companion-QZ core `_solve_qz_quadratic`

**Files:**
- Create: `src/dsge/qz_solve.jl`
- Modify: `src/MacroEconometricModels.jl` (add include before line 203 `include("dsge/gensys.jl")`)
- Test: `test/coverage/test_dsge_coverage.jl` (new testset section, appended before the final `end  # @testset "DSGE Coverage"`)

- [ ] **Step 1: Write failing unit tests for the core**

Append this testset section to `test/coverage/test_dsge_coverage.jl`, immediately **before** the final line `end  # @testset "DSGE Coverage"`:

```julia
# =========================================================================
# 18. _solve_qz_quadratic -- companion-QZ solver of f_lead·G² + f_0·G + f_1 = 0
# =========================================================================
@testset "_solve_qz_quadratic: AR(1) backward model" begin
    # y_t = 0.9 y_{t-1} + ε ; residual f = y_t - 0.9 y_{t-1} - ε
    f_0 = reshape([1.0], 1, 1)
    f_1 = reshape([-0.9], 1, 1)
    f_lead = reshape([0.0], 1, 1)
    f_ε = reshape([-1.0], 1, 1)
    r = MacroEconometricModels._solve_qz_quadratic(f_0, f_1, f_lead, f_ε)
    @test r.eu == [1, 1]
    @test r.n_stable == 1
    @test r.G[1, 1] ≈ 0.9 atol = 1e-10
    @test r.impact[1, 1] ≈ 1.0 atol = 1e-10
    @test r.residual < 1e-8
end

@testset "_solve_qz_quadratic: purely forward model" begin
    # x_t = 0.5 E_t[x_{t+1}] + ε ; f = x_t - 0.5 x_{t+1} - ε
    f_0 = reshape([1.0], 1, 1)
    f_1 = reshape([0.0], 1, 1)
    f_lead = reshape([-0.5], 1, 1)
    f_ε = reshape([-1.0], 1, 1)
    r = MacroEconometricModels._solve_qz_quadratic(f_0, f_1, f_lead, f_ε)
    @test r.eu == [1, 1]            # determinate: stable solvent G = 0
    @test r.n_stable == 1
    @test r.G[1, 1] ≈ 0.0 atol = 1e-10
    @test r.impact[1, 1] ≈ 1.0 atol = 1e-10
    @test r.residual < 1e-8
end

@testset "_solve_qz_quadratic: explosive model" begin
    # y_t = 1.5 y_{t-1} + ε ; no forward var, root 1.5 outside unit circle
    f_0 = reshape([1.0], 1, 1)
    f_1 = reshape([-1.5], 1, 1)
    f_lead = reshape([0.0], 1, 1)
    f_ε = reshape([-1.0], 1, 1)
    r = MacroEconometricModels._solve_qz_quadratic(f_0, f_1, f_lead, f_ε)
    @test r.n_stable == 0
    @test r.eu == [0, 0]           # no stable solution
end

@testset "_solve_qz_quadratic: forward jump + backward state (asset model)" begin
    # p_t = (1/(1+r)) E_t[p_{t+1}] + (r/(1+r)) d_t ;  d_t = ρ d_{t-1} + e
    # Order [p, d]. Closed form: p = 0.2 d, d AR(0.8). r=0.05, ρ=0.8.
    rr = 0.05; ρ = 0.8; β = 1 / (1 + rr); κ = rr / (1 + rr)
    # residuals: f1 = p_t - β p_{t+1} - κ d_t ; f2 = d_t - ρ d_{t-1} - e
    f_0 = [1.0  -κ; 0.0  1.0]          # ∂f/∂y_t
    f_1 = [0.0   0.0; 0.0  -ρ]         # ∂f/∂y_{t-1}
    f_lead = [-β  0.0; 0.0  0.0]       # ∂f/∂y_{t+1}
    f_ε = reshape([0.0, -1.0], 2, 1)   # ∂f/∂ε
    r = MacroEconometricModels._solve_qz_quadratic(f_0, f_1, f_lead, f_ε)
    @test r.eu == [1, 1]
    @test r.n_stable == 2
    @test r.residual < 1e-8
    # impact: d responds 1, p responds 0.2
    @test r.impact[2, 1] ≈ 1.0 atol = 1e-6
    @test r.impact[1, 1] ≈ 0.2 atol = 1e-6
    # G eigenvalues are {0, 0.8}
    @test maximum(abs.(eigvals(r.G))) ≈ 0.8 atol = 1e-6
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
MACRO_MULTIPROCESS_TESTS=1 julia --project=. -e '
using Test, MacroEconometricModels
const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
include("test/fixtures.jl")
@testset "run" begin include("test/coverage/test_dsge_coverage.jl") end
' 2>&1 | tail -20
```
Expected: FAIL — `UndefVarError: _solve_qz_quadratic` (function not yet defined).

- [ ] **Step 3: Create the core file**

Create `src/dsge/qz_solve.jl` with exactly:

```julia
# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
    _solve_qz_quadratic(f_0, f_1, f_lead, f_ε; div=1.0)
        → (G, impact, eigenvalues, n_stable, eu, residual)

Solve the quadratic matrix equation `f_lead·G² + f_0·G + f_1 = 0` for the unique stable
solvent `G` (n×n) via the QZ (generalized Schur) decomposition of its companion pencil, and
recover the shock impact `M = -(f_0 + f_lead·G)⁻¹·f_ε`.

This is the correct Klein (2000) / Blanchard-Kahn (1980) treatment of a linear rational-
expectations model `f_lead·E_t[y_{t+1}] + f_0·y_t + f_1·y_{t-1} + f_ε·ε_t = 0`, valid for
models with forward-looking variables, lags, and static equations alike.

Companion pencil `L·x = λ·M·x` with `x = [a; λ·a]` and `(f_lead·λ² + f_0·λ + f_1)·a = 0`:

    L = [ 0     I    ]      M = [ I   0      ]
        [ -f_1  -f_0 ]          [ 0   f_lead ]

Determinacy is read off the companion's stable-root count `n_stable` (roots with `|λ| < div`):
`n_stable == n` → determinate `[1,1]`; `> n` → indeterminate `[1,0]`; `< n` → no stable
solution `[0,0]`. `residual = ‖f_lead·G² + f_0·G + f_1‖∞` is a convention-independent
self-check on the recovered `G`.

The stable solvent is recovered as `G = Z_b · Z_t⁻¹`, where `[Z_t; Z_b]` are the top/bottom
n-row blocks of the first `n` (stable) columns of the ordered right Schur vectors `Z`.
"""
function _solve_qz_quadratic(f_0::AbstractMatrix{T}, f_1::AbstractMatrix{T},
        f_lead::AbstractMatrix{T}, f_ε::AbstractMatrix{T};
        div::Real=1.0) where {T<:AbstractFloat}
    n = size(f_0, 1)
    N = 2n
    Z0 = zeros(T, n, n)
    In = Matrix{T}(I, n, n)

    # Companion pencil
    L = [Z0    In;
         -Matrix{T}(f_1)  -Matrix{T}(f_0)]
    M = [In    Z0;
         Z0    Matrix{T}(f_lead)]

    F = schur(complex(L), complex(M))
    λ = F.values                                   # 2n generalized eigenvalues (Inf where β≈0)

    stable_select = BitVector(abs.(λ) .< T(div))
    n_stable = count(stable_select)

    eu = n_stable == n ? [1, 1] : (n_stable > n ? [1, 0] : [0, 0])

    G = zeros(T, n, n)
    if n_stable >= n
        Fo = ordschur(F, stable_select)
        Zt = Fo.Z[1:n, 1:n]
        Zb = Fo.Z[n+1:N, 1:n]
        if rank(Zt) == n
            G = real(Zb * inv(Zt))
        else
            eu = [eu[1], 0]
        end
    end

    A = Matrix{T}(f_0) + Matrix{T}(f_lead) * G
    impact = Matrix{T}(-(A \ Matrix{T}(f_ε)))

    residual = maximum(abs.(Matrix{T}(f_lead) * G * G + Matrix{T}(f_0) * G + Matrix{T}(f_1)))

    (G=G, impact=impact, eigenvalues=Vector{ComplexF64}(λ),
     n_stable=n_stable, eu=eu, residual=residual)
end
```

- [ ] **Step 4: Register the include**

In `src/MacroEconometricModels.jl`, add the include line immediately before `include("dsge/gensys.jl")` (currently line 203). After editing, the block reads:

```julia
include("dsge/linearize.jl")
include("dsge/qz_solve.jl")
include("dsge/gensys.jl")
include("dsge/blanchard_kahn.jl")
include("dsge/klein.jl")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run the same command as Step 2.
Expected: PASS — all four `_solve_qz_quadratic` testsets green.

- [ ] **Step 6: Commit**

```bash
git add src/dsge/qz_solve.jl src/MacroEconometricModels.jl test/coverage/test_dsge_coverage.jl
git commit -m "feat(dsge): add companion-QZ quadratic solver core (_solve_qz_quadratic)"
```

---

## Task 2: Rewrite `klein` to `(ld, spec)` over the core; fix klein tests

**Files:**
- Modify: `src/dsge/klein.jl` (replace the `klein(Γ0,…)` function body)
- Modify: `src/dsge/gensys.jl:241-248` (the `:klein` branch of `solve`)
- Modify: `test/dsge/test_dsge.jl:2958-2961` and `:2982-2985` (determinacy assertions)
- Modify: `test/coverage/test_dsge_coverage.jl` (remove the obsolete raw-`klein(...)` testset)

- [ ] **Step 1: Update the klein equivalence tests to expect correct determinacy**

In `test/dsge/test_dsge.jl`, testset "Equivalence with gensys — forward-looking" (~line 2944): replace

```julia
        # Klein reports eu=[1,0] for purely forward-looking models (n_stable > n_predetermined=0)
        # because BK counting flags indeterminacy, but gensys resolves it via Pi.
        # The solution matrices still match.
        @test sol_k.eu[1] == 1  # existence
        @test !is_determined(sol_k)  # Klein BK counting flags indeterminacy
        @test is_determined(sol_g)   # gensys resolves it via Pi rank check
        @test sol_k.G1 ≈ sol_g.G1 atol=1e-8
        @test sol_k.impact ≈ sol_g.impact atol=1e-8
```

with

```julia
        # Companion-QZ Klein is determinate here (unique bounded solution), agreeing
        # with gensys; the stable solvent is G = 0 (no state persistence).
        @test is_determined(sol_k)
        @test is_determined(sol_g)
        @test sol_k.eu == sol_g.eu
        @test sol_k.G1 ≈ sol_g.G1 atol=1e-8
        @test sol_k.impact ≈ sol_g.impact atol=1e-8
```

In testset "Equivalence with gensys — NK 3-equation" (~line 2978): replace

```julia
        # Purely forward-looking NK model: Klein BK counting differs from gensys
        # (n_stable > n_predetermined=0), but solution matrices match.
        @test sol_k.eu[1] == 1  # existence
        @test sol_k.G1 ≈ sol_g.G1 atol=1e-6
        @test sol_k.impact ≈ sol_g.impact atol=1e-6
```

with

```julia
        # Determinate NK model: Klein and gensys agree on solution and determinacy.
        @test is_determined(sol_k)
        @test sol_k.eu == sol_g.eu
        @test sol_k.G1 ≈ sol_g.G1 atol=1e-6
        @test sol_k.impact ≈ sol_g.impact atol=1e-6
```

- [ ] **Step 2: Remove the obsolete raw-`klein(...)` testset**

In `test/coverage/test_dsge_coverage.jl`, delete the entire testset `@testset "klein direct: Q1_adj branch on synthetic unstable pencil" begin … end` (it calls the old raw signature `klein(G0, G1m, Cv, Psi, Pi, 1)`, which no longer exists). Leave the other section-16 testsets in place for now (they are addressed in Task 5).

- [ ] **Step 3: Run klein tests to verify they fail**

Run:
```bash
MACRO_MULTIPROCESS_TESTS=1 julia --project=. -e '
using Test, MacroEconometricModels
const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
include("test/fixtures.jl")
@testset "run" begin include("test/dsge/test_dsge.jl") end
' 2>&1 | tail -25
```
Expected: FAIL — the updated "forward-looking" / "NK 3-equation" assertions fail because the *current* klein still returns `eu=[1,0]`.

- [ ] **Step 4: Rewrite `klein`**

In `src/dsge/klein.jl`, replace the entire `function klein(...) … end` (the `klein(Gamma0, Gamma1, C, Psi, Pi, n_predetermined; div)` definition) with:

```julia
function klein(ld::LinearDSGE{T}, spec::DSGESpec{T}; div::Real=1.0) where {T<:AbstractFloat}
    n = spec.n_endog

    f_0 = ld.Gamma0
    f_1 = -ld.Gamma1
    f_ε = -ld.Psi
    f_lead = _dsge_jacobian(spec, spec.steady_state, :lead)

    res = _solve_qz_quadratic(f_0, f_1, f_lead, f_ε; div=div)
    G1 = res.G
    impact = res.impact

    # Constants: C_sol = (I - G1)·y_ss, y_ss = (Γ0 - Γ1)⁻¹·C
    C_sol = if norm(ld.C) > eps(T)
        y_bar = real(Vector{T}((complex(ld.Gamma0) - complex(ld.Gamma1)) \ complex(ld.C)))
        Vector{T}((I - G1) * y_bar)
    else
        zeros(T, n)
    end

    eigenvalues = Vector{ComplexF64}(eigvals(G1))
    DSGESolution{T}(G1, impact, C_sol, res.eu, :klein, eigenvalues, spec, ld)
end
```

Keep the existing `_count_predetermined` and `_state_control_indices` helper functions in `klein.jl` (still used elsewhere) — only the `klein` function itself changes.

- [ ] **Step 5: Update the `solve(:klein)` call site**

In `src/dsge/gensys.jl`, replace the `:klein` branch (currently lines 241-248):

```julia
    elseif method == :klein
        ld = linearize(spec)
        n_pre = _count_predetermined(ld)
        result = klein(ld.Gamma0, ld.Gamma1, ld.C, ld.Psi, ld.Pi, n_pre)
        return DSGESolution{T}(
            result.G1, result.impact, result.C_sol, result.eu,
            :klein, result.eigenvalues, spec, ld
        )
```

with

```julia
    elseif method == :klein
        ld = linearize(spec)
        return klein(ld, spec)
```

- [ ] **Step 6: Run klein tests to verify they pass**

Run the Step 3 command. Expected: PASS for the whole "Klein" testset group (equivalence backward/forward/NK, BK eu flags, downstream, augmented, display). If any klein assertion outside Steps 1-2 fails because it encoded old behavior, fix it to the correct value and note it in the commit body.

- [ ] **Step 7: Commit**

```bash
git add src/dsge/klein.jl src/dsge/gensys.jl test/dsge/test_dsge.jl test/coverage/test_dsge_coverage.jl
git commit -m "fix(dsge): klein solves forward-looking models via companion-QZ core

klein now takes (ld, spec) and delegates to _solve_qz_quadratic, so it is
correct for models with forward-looking variables and agrees with gensys.
Updates klein equivalence tests that asserted the old (incorrect) spurious
indeterminacy."
```

---

## Task 3: Rewrite `blanchard_kahn` over the core; fix bk tests

**Files:**
- Modify: `src/dsge/blanchard_kahn.jl` (replace the function body)
- Modify: `test/coverage/test_dsge_coverage.jl` ("blanchard_kahn.jl: indeterminate (fewer unstable than forward)" testset, ~line 340-356)
- Modify: `test/dsge/test_dsge.jl:1213-1219` (stale comment)

- [ ] **Step 1: Update the bk "fewer unstable than forward" coverage test**

In `test/coverage/test_dsge_coverage.jl`, the testset `@testset "blanchard_kahn.jl: indeterminate (fewer unstable than forward)" begin … end` currently asserts `@test sol.eu == [1, 0]`. With the corrected solver this model is determinate. Replace the testset name and assertion:

- Rename testset to `"blanchard_kahn.jl: forward + backward determinate"`.
- Replace `@test sol.eu == [1, 0]` with `@test sol.eu == [1, 1]`.
- Replace the stale comment lines describing eu=[1,0] reasoning with: `# Corrected companion-QZ BK: forward roots are now counted, model is determinate.`

(Leave the testset "blanchard_kahn.jl: indeterminate (more unstable than forward)" unchanged — it is a genuinely explosive backward model and still returns `eu[1]==0`.)

- [ ] **Step 2: Update the stale bk comment in test_dsge.jl**

In `test/dsge/test_dsge.jl` (~line 1213-1219), replace the comment

```julia
    # BK eigenvalue counting may flag purely forward-looking models as
    # indeterminate (n_unstable=0 < n_fwd=1), while gensys finds the
    # unique bounded solution. The solution matrices still agree.
```

with

```julia
    # Companion-QZ BK agrees with gensys on the determinate solution.
```

and add, immediately after the existing `@test is_determined(sol_g)` line:

```julia
    @test is_determined(sol_bk)
```

- [ ] **Step 3: Run bk tests to verify they fail**

Run both files:
```bash
MACRO_MULTIPROCESS_TESTS=1 julia --project=. -e '
using Test, MacroEconometricModels
const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
include("test/fixtures.jl")
@testset "run" begin
  include("test/coverage/test_dsge_coverage.jl")
  include("test/dsge/test_dsge.jl")
end
' 2>&1 | tail -25
```
Expected: FAIL — the updated bk assertions fail because the current bk still returns `eu=[1,0]` / is not determined on the forward model.

- [ ] **Step 4: Rewrite `blanchard_kahn`**

In `src/dsge/blanchard_kahn.jl`, replace the entire `function blanchard_kahn(...) … end` with:

```julia
function blanchard_kahn(ld::LinearDSGE{T}, spec::DSGESpec{T}; div::Real=1.0) where {T<:AbstractFloat}
    n = spec.n_endog

    f_0 = ld.Gamma0
    f_1 = -ld.Gamma1
    f_ε = -ld.Psi
    f_lead = _dsge_jacobian(spec, spec.steady_state, :lead)

    res = _solve_qz_quadratic(f_0, f_1, f_lead, f_ε; div=div)
    G1 = res.G
    impact = res.impact

    C_sol = if norm(ld.C) > eps(T)
        y_bar = real(Vector{T}((complex(ld.Gamma0) - complex(ld.Gamma1)) \ complex(ld.C)))
        Vector{T}((I - G1) * y_bar)
    else
        zeros(T, n)
    end

    eigenvalues = Vector{ComplexF64}(eigvals(G1))
    DSGESolution{T}(G1, impact, C_sol, res.eu, :blanchard_kahn, eigenvalues, spec, ld)
end
```

The `solve(:blanchard_kahn)` branch in `gensys.jl` (`return blanchard_kahn(ld, spec)`) is unchanged.

- [ ] **Step 5: Run bk tests to verify they pass**

Run the Step 3 command. Expected: PASS for bk testsets ("BK: AR(1) model", "BK: agrees with gensys", the forward equivalence at ~1213, the coverage "forward + backward determinate" and "more unstable than forward"). Fix any other bk assertion that encoded old behavior to the correct value and note it.

- [ ] **Step 6: Commit**

```bash
git add src/dsge/blanchard_kahn.jl test/coverage/test_dsge_coverage.jl test/dsge/test_dsge.jl
git commit -m "fix(dsge): blanchard_kahn solves forward-looking models via companion-QZ core"
```

---

## Task 4: gensys determinacy from the core (keep UC solution)

**Files:**
- Modify: `src/dsge/gensys.jl` (the `:gensys` branch of `solve`, ~lines 183-234)
- Test: `test/dsge/test_dsge.jl` (existing gensys `eu` tests at ~689-700, ~671-687) and `test/coverage/test_dsge_coverage.jl` section-16 "forward-looking: gensys UC two-phase solves correctly"

- [ ] **Step 1: Confirm the target assertions exist (no new test needed yet)**

The existing gensys tests already assert correct determinacy on backward and forward models:
- `test/dsge/test_dsge.jl:689-700` ("Gensys: existence/uniqueness flags") expects `eu == [1,1]` for AR(1).
- `test/dsge/test_dsge.jl:671-687` ("Gensys: forward-looking model") expects `is_determined`.
- `test/coverage/test_dsge_coverage.jl` section 16 "forward-looking: gensys UC two-phase solves correctly" expects `eu == [1,1]`, `impact ≈ [0.2, 1.0]`, `maximum(abs.(eigenvalues)) ≈ 0.8`.

These must keep passing. No new test is added in this task; the change must not regress them.

- [ ] **Step 2: Rewrite the `:gensys` branch of `solve`**

In `src/dsge/gensys.jl`, replace the entire `if method == :gensys … return DSGESolution{T}(…)` block (from `if method == :gensys` through its `return`, currently ~lines 183-234) with:

```julia
    if method == :gensys
        ld = linearize(spec)

        f_0 = _dsge_jacobian(spec, spec.steady_state, :current)
        f_1 = _dsge_jacobian(spec, spec.steady_state, :lag)
        f_lead = _dsge_jacobian(spec, spec.steady_state, :lead)
        f_ε = _dsge_jacobian_shocks(spec, spec.steady_state)

        # Companion-QZ for correct determinacy + a robust solution fallback
        qz_core = _solve_qz_quadratic(f_0, f_1, f_lead, f_ε)

        # Primary solution via undetermined coefficients (robust to many static vars)
        uc_ok = false
        local uc_result
        try
            uc_result = _solve_undetermined_coefficients(spec)
            resid = (f_0 + f_lead * uc_result.G1) * uc_result.G1 + f_1
            uc_ok = maximum(abs.(resid)) < T(1e-8) && uc_result.converged
        catch
        end

        G1 = uc_ok ? uc_result.G1 : qz_core.G
        impact = uc_ok ? uc_result.impact : qz_core.impact

        # Constants: C_sol = (I - G1)·y_ss, y_ss = (Γ0 - Γ1)⁻¹·C
        if norm(ld.C) > eps(T)
            y_bar = real(Vector{T}((complex(ld.Gamma0) - complex(ld.Gamma1)) \ complex(ld.C)))
            C_sol = (I - G1) * y_bar
        else
            C_sol = zeros(T, spec.n_endog)
        end

        eigenvalues = Vector{ComplexF64}(eigvals(G1))
        return DSGESolution{T}(
            G1, impact, Vector{T}(C_sol), qz_core.eu,
            :gensys, eigenvalues, spec, ld
        )
```

Notes for the implementer:
- `eu` now comes from `qz_core` (correct determinacy). `eigenvalues` is `eigvals(G1)` (the solution's roots, keeping `is_stable` and the `≈ 0.8` assertion valid).
- The standalone `gensys(Γ0,Γ1,C,Ψ,Π)` function above this block is **left unchanged** (still exported, still used by coverage tests that call it directly).

- [ ] **Step 3: Run gensys tests to verify they pass**

```bash
MACRO_MULTIPROCESS_TESTS=1 julia --project=. -e '
using Test, MacroEconometricModels
const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
include("test/fixtures.jl")
@testset "run" begin
  include("test/dsge/test_dsge.jl")
  include("test/coverage/test_dsge_coverage.jl")
end
' 2>&1 | tail -25
```
Expected: PASS for gensys testsets and the section-16 "forward-looking: gensys UC two-phase solves correctly" test. If a test asserted a gensys eigenvalue *count* (not value), reconcile it to the `n`-length `eigvals(G1)`.

- [ ] **Step 4: Commit**

```bash
git add src/dsge/gensys.jl
git commit -m "fix(dsge): gensys determinacy from companion-QZ core (keeps UC solution)"
```

---

## Task 5: Restore strong cross-solver agreement; finalize coverage section 16

**Files:**
- Modify: `test/coverage/test_dsge_coverage.jl` (section 16: remove obsolete Q1_adj testset, add agreement tests)

- [ ] **Step 1: Remove the obsolete forward-jump Q1_adj testset**

In `test/coverage/test_dsge_coverage.jl`, delete the testset `@testset "klein/bk via solve: Q1_adj branch with forward jump + unstable state" begin … end` (it asserted `maximum(abs.(sol.eigenvalues)) ≈ 1.5` on an explosive model with the old reporting; superseded). Keep "forward-looking: gensys UC two-phase solves correctly", "forward-looking: _solve_undetermined_coefficients converges", and "forward-looking: IRFs decay and match closed form".

- [ ] **Step 2: Add restored cross-solver agreement tests**

Append after the "forward-looking: IRFs decay and match closed form" testset (still inside the outer `@testset "DSGE Coverage"`):

```julia
@testset "cross-solver: gensys == klein == bk on asset model" begin
    spec = _asset_pricing_spec()
    sol_g  = solve(spec; method=:gensys)
    sol_k  = solve(spec; method=:klein)
    sol_bk = solve(spec; method=:blanchard_kahn)

    @test sol_g.eu  == [1, 1]
    @test sol_k.eu  == [1, 1]
    @test sol_bk.eu == [1, 1]

    @test sol_k.G1      ≈ sol_g.G1     atol = 1e-6
    @test sol_k.impact  ≈ sol_g.impact atol = 1e-6
    @test sol_bk.G1     ≈ sol_g.G1     atol = 1e-6
    @test sol_bk.impact ≈ sol_g.impact atol = 1e-6

    # Closed form: p = 0.2·d ; endogenous order [p, d]
    @test sol_k.impact[2, 1] ≈ 1.0 atol = 1e-6
    @test sol_k.impact[1, 1] ≈ 0.2 atol = 1e-6
end

@testset "cross-solver: gensys == klein == bk on multi-var forward model" begin
    spec = @dsge begin
        parameters: β = 0.99, κ = 0.3, φ_π = 1.5, ρ = 0.8
        endogenous: pi_v, y, a
        exogenous: e
        pi_v[t] = β * E[t](pi_v[t+1]) + κ * y[t]
        y[t] = E[t](y[t+1]) - (φ_π * pi_v[t]) + a[t]
        a[t] = ρ * a[t-1] + e[t]
    end
    spec = compute_steady_state(spec)

    sol_g  = solve(spec; method=:gensys)
    sol_k  = solve(spec; method=:klein)
    sol_bk = solve(spec; method=:blanchard_kahn)

    @test is_determined(sol_g)
    @test sol_k.eu  == sol_g.eu
    @test sol_bk.eu == sol_g.eu
    @test sol_k.G1      ≈ sol_g.G1     atol = 1e-5
    @test sol_k.impact  ≈ sol_g.impact atol = 1e-5
    @test sol_bk.G1     ≈ sol_g.G1     atol = 1e-5
    @test sol_bk.impact ≈ sol_g.impact atol = 1e-5
end
```

Note for implementer: the second model's determinacy depends on the parameters; if `solve(:gensys)` reports it as not determinate, adjust `φ_π`/`ρ` so the NK block is determinate (e.g. raise `φ_π`), keeping the test meaningful. Verify `is_determined(sol_g)` holds before asserting agreement.

- [ ] **Step 3: Run the coverage file to verify it passes**

```bash
MACRO_MULTIPROCESS_TESTS=1 julia --project=. -e '
using Test, MacroEconometricModels
const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
include("test/fixtures.jl")
@testset "run" begin include("test/coverage/test_dsge_coverage.jl") end
' 2>&1 | tail -20
```
Expected: PASS — all DSGE coverage testsets including the new cross-solver agreement.

- [ ] **Step 4: Commit**

```bash
git add test/coverage/test_dsge_coverage.jl
git commit -m "test(dsge): restore gensys==klein==bk agreement on forward-looking models"
```

---

## Task 6: Full regression over affected files + Dynare suite

**Files:** none (test/source reconciliation only)

- [ ] **Step 1: Run the two main DSGE test files**

```bash
MACRO_MULTIPROCESS_TESTS=1 julia --project=. -e '
using Test, MacroEconometricModels
const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
include("test/fixtures.jl")
@testset "run" begin
  include("test/dsge/test_dsge.jl")
  include("test/coverage/test_dsge_coverage.jl")
end
' 2>&1 | tail -30
```
Expected: PASS. For any failure that is a *correction* of previously-wrong klein/bk/gensys output (determinacy flag or impact value on a forward-looking model), update the assertion to the correct value and note it. Do **not** weaken a test to hide a real regression — if a backward-looking or AR model changes, investigate.

- [ ] **Step 2: Identify and run the Dynare replication test file**

Find the file:
```bash
grep -rln "dynare\|Dynare\|replication" test/ | grep -i dsge
```
Run it with the harness preamble (substitute the path found):
```bash
MACRO_MULTIPROCESS_TESTS=1 julia --project=. -e '
using Test, MacroEconometricModels
const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
include("test/fixtures.jl")
@testset "run" begin include("test/<DYNARE_FILE>") end
' 2>&1 | tail -30
```
Expected: PASS. The suite mostly uses `:gensys`/perturbation; reconcile any `:klein`/`:blanchard_kahn` or determinacy assertion that changes, documenting each as a correction.

- [ ] **Step 3: Run the bayesian DSGE coverage (downstream of solve)**

```bash
MACRO_MULTIPROCESS_TESTS=1 julia --project=. -e '
using Test, MacroEconometricModels
const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
include("test/fixtures.jl")
@testset "run" begin include("test/coverage/test_dsge_bayes_coverage.jl") end
' 2>&1 | tail -20
```
Expected: PASS (these use gensys/UC; determinacy-flag change should not affect estimation, but confirm).

- [ ] **Step 4: Final commit (if any reconciliations were made)**

```bash
git add -A
git commit -m "test(dsge): reconcile klein/bk/gensys assertions after forward-looking QZ fix"
```

---

## Self-Review (run after writing; fix inline)

**Spec coverage check:**
- Spec §"Component 1 — shared companion-QZ core" → Task 1. ✓
- Spec §"Component 2 — klein/bk wrappers", klein API change → Tasks 2 (klein) & 3 (bk). ✓
- Spec §"Component 3 — gensys determinacy" → Task 4. ✓
- Spec §"Determinacy semantics" (`n_stable==n`) → implemented in Task 1 core; asserted in Task 1 tests (backward/forward/explosive). ✓
- Spec §"Edge cases" (pure backward, static, multi-shock, `Z_t` singular) → Task 1 core handles; backward/forward/explosive covered by Task 1 tests; `Z_t` singular sets `eu[2]=0`. ✓
- Spec §"Testing & validation" → Tasks 1,5 (unit + agreement), Task 6 (regression + Dynare). ✓
- Spec §"Risks" (klein/bk outputs change; gensys eu flips) → Tasks 2,3 update flipping assertions; Task 6 reconciles stragglers. ✓

**Placeholder scan:** No "TBD/TODO". The one parameter-dependent spot (Task 5 second model's determinacy) gives an explicit adjustment rule. ✓

**Type/signature consistency:**
- Core returns named tuple `(G, impact, eigenvalues, n_stable, eu, residual)`; callers use `res.G`, `res.impact`, `res.eu` consistently across Tasks 1-4. ✓
- `klein(ld, spec)` and `blanchard_kahn(ld, spec)` both return `DSGESolution{T}` with field order `G1, impact, C_sol, eu, method, eigenvalues, spec, linear` matching `src/dsge/types.jl`. ✓
- `_dsge_jacobian(spec, spec.steady_state, :lead)` / `_dsge_jacobian_shocks(spec, spec.steady_state)` signatures match `src/dsge/linearize.jl`. ✓
