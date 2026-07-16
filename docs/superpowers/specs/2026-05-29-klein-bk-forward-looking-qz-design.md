# Design: Correct Klein/BK (and gensys determinacy) for forward-looking models

- **Date:** 2026-05-29
- **Status:** Approved (pending spec review)
- **Scope:** `src/dsge/klein.jl`, `src/dsge/blanchard_kahn.jl`, `src/dsge/gensys.jl`, new `src/dsge/qz_solve.jl`, tests
- **Out of scope:** `linearize`, the perturbation/estimation/Kalman paths, the `linearize` canonical form

## Problem

`solve(spec; method=:klein)` and `solve(spec; method=:blanchard_kahn)` return incorrect
solutions for forward-looking models. On the asset-pricing test model
`p_t = (1/(1+r))·E_t[p_{t+1}] + (r/(1+r))·d_t`, `d_t = ρ·d_{t-1} + e_t`, gensys returns the
correct `impact = [0.2, 1.0]` while klein/bk return `impact = [0.0476, 1.0]` and the wrong
determinacy flag (`eu = [1,0]` instead of `[1,1]`).

## Root cause

`src/dsge/linearize.jl` (lines 47–71) builds the Sims canonical form as
`Γ0 = f_0`, `Γ1 = -f_1`, `Ψ = -f_ε`, `Π = -f_lead[:, forward_cols]`. The lead Jacobian
`f_lead` is placed **only** into `Π` and is **absent from the pencil `(Γ0, Γ1)`**. Therefore
the generalized eigenvalue problem `det(Γ1 - λ·Γ0) = 0` does not contain the forward
(explosive) roots that Blanchard-Kahn (1980) and Klein (2000) must count. For the asset
model the pencil eigenvalues are `{0, 0.8}` instead of the true `{0.8, 1.05}`.

gensys returns the correct *solution* only because the recent two-phase rewrite makes it
ignore its own QZ result and use the undetermined-coefficients (UC) solver
(`_solve_undetermined_coefficients`, which works directly from `f_0, f_1, f_lead`). Its QZ-based
`eu`/determinacy logic is still computed on the incomplete pencil and is therefore unreliable
for forward-looking models — currently only masked by the UC convergence/residual check.

Klein (2000) and Blanchard-Kahn (1980) are the canonical, textbook-correct methods for
forward-looking linear RE models; the methods are not at fault, only the pencil they are fed.

## Goal

1. `solve(:klein)` and `solve(:blanchard_kahn)` return the correct, unique stable solution for
   forward-looking models, agreeing with `solve(:gensys)` on determinate models.
2. gensys's determinacy flags (`eu`) and reported eigenvalues are correct for forward-looking
   models (its UC-based `G1`/`impact` solution is unchanged).
3. Backward-looking and static models continue to solve correctly (no regression).

## Design

### Component 1 — shared companion-QZ core (new `src/dsge/qz_solve.jl`)

```
_solve_qz_quadratic(f_0, f_1, f_lead, f_ε; div=1.0)
    → (G, impact, eigenvalues, n_stable, eu)
```

Solves the quadratic matrix equation `f_lead·G² + f_0·G + f_1 = 0` for the unique stable
solvent `G` (n×n) via the QZ decomposition of its companion pencil:

```
L = [ 0     I_n  ]      M = [ I_n   0      ]
    [ -f_1  -f_0 ]          [ 0     f_lead ]
```

Eigenvector structure: `L·x = λ·M·x` with `x = [a; λ·a]` and `(f_lead·λ² + f_0·λ + f_1)·a = 0`,
so the companion's generalized eigenvalues are exactly the roots of the quadratic (with `∞`
roots arising from a singular `f_lead`).

Steps:
1. `F = schur(complex(L), complex(M))`; generalized eigenvalues `λ_i = F.S_ii / F.T_ii` guard
   against `|F.T_ii| ≈ 0` (→ `∞`, treated as unstable).
2. Reorder via `ordschur` so stable roots (`|λ| < div`) come first; `n_stable = count`.
3. Determinacy:
   - `n_stable == n` → `eu = [1, 1]` (determinate),
   - `n_stable > n`  → `eu = [1, 0]` (indeterminate / sunspots),
   - `n_stable < n`  → `eu = [0, 0]` (no stable solution / explosive).
4. Recover `G` from the stable deflating subspace (Klein 2000 / `solab`): partition the ordered
   right Schur vectors `Z` into n×n blocks `[Z_t; Z_b]` over the first `n` (stable) columns and
   set `G = real(Z_b · Z_t⁻¹)`. The exact block indexing is fixed in implementation and
   **verified by a self-check** (see below); if `Z_t` is singular, `eu[2] = 0`.
5. `impact = -(f_0 + f_lead·G)⁻¹·f_ε` (n×n_ε). `f_0 + f_lead·G` is invertible for a determinate
   model; identical to the UC solver's impact formula.
6. **Self-check:** assert `maximum(abs.(f_lead·G² + f_0·G + f_1)) < tol` (e.g. `1e-8`). This is a
   convention-independent correctness contract on the recovered `G`. On failure the caller falls
   back to its prior behavior (klein/bk: report the QZ result with a `@warn`; this should not
   occur for determinate models).
7. Use `robust_inv`/`safe_cholesky`-style guarded inverses per repo convention; return real
   matrices (`G`, `impact`) and the complex `eigenvalues` vector for reporting.

The core depends only on the four Jacobians and standard `LinearAlgebra` (`schur`, `ordschur`).
It does not need `n_predetermined` — determinacy is purely `n_stable == n`.

### Component 2 — `klein` and `blanchard_kahn` as thin wrappers

Both compute the four Jacobians from the spec and call the core, then attach the constant:

- `f_0 = Γ0`, `f_1 = -Γ1`, `f_ε = -Ψ` from the `LinearDSGE`.
- `f_lead = _dsge_jacobian(spec, spec.steady_state, :lead)` (full n×n; the lossy `Π` is not used).
- `C_sol`: keep the existing formula — `y_bar = (Γ0 - Γ1)\C`, `C_sol = (I - G)·y_bar` when
  `‖C‖ > eps`, else zeros.

**API change (`klein`):** signature changes from
`klein(Γ0, Γ1, C, Ψ, Π, n_predetermined; div)` to **`klein(ld::LinearDSGE{T}, spec::DSGESpec{T}; div=1.0)`**,
symmetric with the existing `blanchard_kahn(ld, spec)`. Rationale: the raw-matrix signature
cannot recover `f_lead` (`Π` discards the column→variable map). Both remain exported and return
the same named-tuple shape `(G1, impact, C_sol, eu, eigenvalues)`.

`blanchard_kahn(ld, spec)`: signature unchanged; internals rewired to the core. (BK and Klein are
mathematically equivalent for these models; both use the shared QZ core. The 1980
eigendecomposition variant is not implemented separately.)

`solve(:klein)` updates its call to `klein(ld, spec)`; `solve(:blanchard_kahn)` is unchanged at
the call site.

### Component 3 — gensys determinacy

`solve(spec; method=:gensys)` keeps the UC solver for `G1`/`impact` (unchanged working path).
Its `eu` and reported `eigenvalues` are taken from `_solve_qz_quadratic` (the companion count),
replacing the current rank/consistency heuristic on the incomplete pencil. The standalone
`gensys(Γ0, Γ1, C, Ψ, Π)` function (raw QZ) is retained for backward compatibility but is no
longer the source of determinacy in `solve`.

## Determinacy semantics

| Condition          | `eu`     | Meaning                                  |
|--------------------|----------|------------------------------------------|
| `n_stable == n`    | `[1, 1]` | Determinate (unique stable solution)     |
| `n_stable > n`     | `[1, 0]` | Indeterminate (sunspot equilibria)       |
| `n_stable < n`     | `[0, 0]` | No stable solution (explosive)           |

`n = n_endog`. Infinite roots (from singular `f_lead`) are non-stable and excluded from
`n_stable`. Validated on the asset model: companion eigenvalues `{0, 0.8, 1.05, ∞}` → `n_stable = 2 = n` → `[1,1]`.

## Edge cases

- **Pure backward model** (`f_lead = 0`, e.g. AR(1)): `M = [I 0; 0 0]`, companion has finite
  roots `-f_1/f_0` plus `∞` roots; `n_stable == n` for a stable model. Recovers `G = ρ`. ✓
- **Static/no-lag variables**: zero columns in `f_1`; handled by the QZ (no special-casing).
- **Multiple shocks**: `f_ε` is n×n_ε; `impact` formula unchanged.
- **Linear models with measurement constants** (`C ≠ 0`): `C_sol` formula unchanged.
- **`Z_t` singular** (BK rank condition fails): set `eu[2] = 0`, return best-effort `G`.

## Testing & validation

New / updated tests in `test/coverage/test_dsge_coverage.jl` and `test/dsge/test_dsge.jl`:

1. **Core unit tests** for `_solve_qz_quadratic` on synthetic `(f_0,f_1,f_lead,f_ε)`:
   determinate forward, pure backward, indeterminate (`n_stable>n`), explosive (`n_stable<n`);
   assert the residual self-check, `eu`, `n_stable`, and `impact` formula.
2. **Restore strong cross-solver agreement** (weakened in commit 36e4ddd): on the asset model
   and a multi-variable forward model, assert
   `gensys.G1 ≈ klein.G1 ≈ bk.G1` and matching `impact` to `atol≈1e-6`, plus `eu == [1,1]` for all
   three and the closed-form `p = 0.2·d`.
3. **Replace obsolete tests**: the "klein direct: Q1_adj branch on synthetic unstable pencil" and
   "klein/bk via solve: Q1_adj branch …" testsets (which exercised the now-removed `Q1_adj`
   path and the old raw `klein(...)` signature) are replaced by core unit tests + the agreement
   tests above. Update the raw `klein(G0,…)` calls to the new `(ld, spec)` signature.
4. **Determinacy**: forward-looking indeterminate and explosive models return correct `eu` from
   gensys, klein, and bk.
5. **Regression**: re-run `test/dsge/test_dsge.jl` (klein section ~2935–3071; bk cases 718–745,
   1213, 4436; gensys `eu` tests 689–700) and `test/coverage/test_dsge_coverage.jl`. Reconcile
   any klein/bk reference values that change — values that change are corrections of previously
   wrong output; document each.
6. **Dynare suite**: run the Dynare replication tests; the suite mostly uses `:gensys`/perturbation,
   but any `:klein`/`:blanchard_kahn` or `eu` assertions are re-checked.

Run command (per repo rules, multi-process, relevant files only):
`MACRO_MULTIPROCESS_TESTS=1` over `test/dsge/test_dsge.jl` and `test/coverage/test_dsge_coverage.jl`
(and the Dynare replication test file). Never the full suite locally; never serial.

## Risks & mitigations

- **klein/bk numerical outputs change for forward-looking models** (now correct). Mitigation:
  tests asserting old values are updated; each changed reference value is documented as a
  correction.
- **gensys `eu` may flip on determinacy edge cases.** Mitigation: validate against existing `eu`
  tests and the indeterminate/explosive cases; the companion count is the standard criterion.
- **QZ subspace block-indexing convention.** Mitigation: the residual self-check (`‖f_lead·G² +
  f_0·G + f_1‖ < tol`) makes correctness convention-independent and catches a wrong partition
  immediately in tests.
- **Out-of-scope interaction**: `perturbation_solver` no longer calls `blanchard_kahn` (uses
  gensys+UC), so it is unaffected; confirm by re-running its tests.

## Rollback

The change is isolated to three solver files plus one new file and the tests. Reverting the
commit restores prior behavior; `linearize`, perturbation, estimation, and the gensys raw
function are untouched.
