# Huggett (1993) Built-in Example with Aggregate Dynamics — Design

**Date:** 2026-06-23
**Target release:** v0.5.1 (first of four HA-DSGE roadmap items; see CLAUDE.md "Targeted for v0.5.1")
**Status:** Approved (design); ready for implementation plan.

## 1. Goal

Add Huggett (1993) — the canonical pure-exchange, incomplete-markets, risk-free-bond
economy — as a first-class built-in HA-DSGE model:

- `load_ha_example(:huggett)` returns a fully calibrated `HADSGESpec`.
- `compute_steady_state` solves the **stationary equilibrium** via zero-net-supply
  bond-market clearing, reproducing Huggett's central result `r* < 1/β − 1`.
- The model is **dynamically solvable by all three existing methods** — SSJ, Reiter,
  and Krusell-Smith — driven by an **aggregate endowment shock**, with the bond in
  zero net supply (`∫a' dμ = 0`) every period.

## 2. Economic model

Continuum of agents. Each receives a stochastic endowment `e_t ∈ {e_l, e_h}` of a
non-storable good and trades a one-period risk-free bond `a_t ≥ ā` (`ā < 0`, a credit
limit). Recursion (implementation form, compatible with the existing EGM):

```
c + a' = (1+r)·a + w·e ,      a' ≥ ā
```

where **`w` is the aggregate endowment level (`w = 1` in steady state)**. Reading `w·e`
as income matches the existing EGM Euler inversion (`egm.jl:169`) and the existing SSJ
`w→K` Jacobian exactly, so Huggett's budget closure is the *same* function as Aiyagari's
`_ks_budget`. The aggregate endowment shock is a shock to `w`.

**Stationary equilibrium:** find `r*` such that the bond market clears in **zero net
supply**:

```
A(r) := ∫ a'(a,e;r) dμ*(a,e) = 0
```

with `μ*` the stationary distribution induced by the policy at `r`. Precautionary saving
pins `r* < 1/β − 1`.

**Calibration (Huggett 1993 baseline):**
- CRRA utility, `σ = 1.5` (`_crra_utility(1.5)`).
- Two-state endowment `e ∈ {e_l, e_h} = {0.1, 1.0}` with Huggett's Markov transition.
- `β`, the transition matrix, and the credit limit `ā` are taken from Huggett (1993)
  Table II baseline and cited in code.
- **Acceptance criterion (correctness):** the computed equilibrium (annualized) risk-free
  rate reproduces Huggett's reported value within tolerance. The exact constants are
  pinned at implementation from the paper cross-checked against a public replication; a
  regression test asserts the match. This is a hard acceptance test, not a placeholder.

## 3. Architecture & file-level changes

### 3.1 Steady state — pluggable clearing closure (`src/dsge/heterogeneous/steady_state.jl`)

Refactor the Aiyagari-specific block in `_ha_steady_state` (current lines 170–203 +
the Cobb-Douglas `Y` at 247–249) into a pluggable closure:

```julia
clearing_fn(r_mid, params) -> (asset_demand::T, prices::Dict{Symbol,T})
```

- **`_aiyagari_clearing` (default):** current logic verbatim — invert the Cobb-Douglas
  firm FOC to `K_d`, `prices = price_fn(K_d, params)`, then `prices[:r] = r_mid`. The
  default code path is **numerically identical** to today's behavior (regression-tested).
- **`_huggett_clearing`:** returns `(zero(T), Dict(:r => r_mid, :w => one(T)))` — zero net
  supply; `w = 1` is the steady-state endowment level.

Bisection loop is otherwise unchanged: `excess = K_s − asset_demand`; `excess > 0 →
r_hi = r_mid` (same monotonicity as Aiyagari, since `A(r)` is increasing near `r*`).
The final `Y` aggregate is computed by the closure / guarded: for Huggett, `Y` =
aggregate endowment `= Σ_j π_∞(e_j)·e_j` (mean endowment), not Cobb-Douglas output.

`compute_steady_state(spec; clearing=nothing, kwargs...)`: if `clearing === nothing`,
select the closure from `spec.model` (`:aiyagari` → `_aiyagari_clearing`, `:huggett` →
`_huggett_clearing`); an explicit `clearing` kwarg overrides. Huggett uses
`r_bounds` defaulting to roughly `(-0.05, 1/β − 1 − ε)`.

### 3.2 `HADSGESpec` gains a `model::Symbol` field (`src/dsge/heterogeneous/types.jl`)

Add `model::Symbol` to the struct (single source of truth for clearing + dynamics
dispatch; forward-compatible with `:olg`, `:ct_*` model types in later v0.5.1 items).
The one convenience constructor (types.jl:407) gains a keyword default:

```julia
function HADSGESpec{T}(aggregate_spec, individual, income, grid, aggregation, het_params;
                       model::Symbol=:aiyagari) where {T<:AbstractFloat}
    ...
    new{T}(aggregate_spec, individual, income, grid, aggregation, het_params,
           n_assets, n_income, model)
end
```

All existing 6-arg calls are unaffected (they omit the keyword → `:aiyagari`). **Audit
& update** any reconstruction sites: `parser.jl` (HA spec creation) and
`estimation.jl::_update_ha_params` (must thread `model=spec.model`).

### 3.3 The `:huggett` example (`src/dsge/heterogeneous/examples.jl`)

- `_huggett_income()` builds the two-state `IncomeProcess{Float64}` directly from
  Huggett's transition matrix + endowment states + stationary distribution (not
  `rouwenhorst`).
- Budget reuses `_ks_budget` (income `= w·e`).
- `_minimal_huggett_agg_spec()`: minimal `DSGESpec` recording the pure-exchange aggregate
  block — endogenous `[r, A, Y, w_agg]`, shock `eps_e`, `w_agg` (endowment level) AR(1)
  with persistence/volatility `ρ_e, σ_e`; metadata for the dynamic solvers (the
  steady-state solver does not evaluate it, mirroring `_minimal_agg_spec`).
- `_huggett_example()`: CRRA(1.5), 2-state endowment, grid `[ā, a_max]` with `a_max`
  high enough that the stationary distribution does not bind the top, borrowing
  constraint `ā`, `het_params` carrying `β, σ, ā, ρ_e, σ_e` (and harmless Aiyagari
  defaults), constructed with `model = :huggett`.
- `load_ha_example(:huggett)` branch + docstring table row + the error-message list.

### 3.4 Dynamics

Aggregate shock = endowment shock = shock to `w` (the endowment level). Bond stays in
zero net supply each period; `r_t` clears `A = 0`.

**SSJ (`ssj.jl`):** add a market-clearing GE close.
- `H_U = J(r→A)` and `H_Z = J(w→A)` via the existing `_ssj_jacobian` (both `:r` and
  `:w` are prices already supported).
- Solve `dr = − H_U⁻¹ H_Z · dw` for the clearing rate path; build aggregate IRFs;
  Ho-Kalman → `DSGESolution` → `HADSGESolution`.
- Triggered when `spec.model == :huggett`; the existing Aiyagari SSJ path is untouched.

**Reiter (`reiter.jl`):** make the aggregate block model-aware.
- For `:huggett`, the shock channel is a direct perturbation of `w` (`dw`), **not** the
  hard-coded Cobb-Douglas `dr/dZ, dw/dZ` (current lines 217–221). The aggregate state is
  `[d̃; w_shock]`; `r_t` is pinned by the static clearing condition `A(r_t, w_t, d_t)=0`
  (local `J(A→r)`). Distribution evolves via the perturbed transition.
- The Aiyagari path retains current behavior (the magic `0.36/0.025` constants may be
  routed through the same closure as an optional cleanup, but that is not required for
  correctness and is out of scope unless trivially free).

**Krusell-Smith (`krusell_smith.jl`):** add a Huggett-appropriate variant.
- The PLM forecasts the **clearing rate `r`** (not capital) from the aggregate
  endowment state `w` (and optionally one distribution moment). Each simulated period
  runs an inner solve for `r_t` that clears `∫a' = 0`, then regress `r_t` on the
  aggregate state to update the PLM; report R² (also consumed by the Den Haan accuracy
  tests in the next sub-project).
- Dispatched on `spec.model`; the Aiyagari capital-PLM path is untouched.

### 3.5 `solve(spec::HADSGESpec; method, ...)` dispatcher (`parser.jl`)

Thread `spec.model` (and an optional `clearing` kwarg) through `compute_steady_state`
and to the method-specific dynamics so `solve(load_ha_example(:huggett); method=:ssj)`
(and `:reiter`, `:krusell_smith`) works end-to-end.

## 4. Tests (`test/dsge/test_ha_dsge.jl`, new `@testset`s)

1. **Aiyagari regression:** `:krusell_smith` / `:one_asset_hank` steady state with the
   default closure is numerically identical (to tolerance) to pre-refactor values.
2. **Huggett steady state:** `∫a ≈ 0` (clearing), `r* < 1/β − 1`, finite Euler error,
   and **`r*` reproduces Huggett (1993) within tolerance** (the correctness anchor).
3. **Distribution sanity:** Gini ∈ (0,1), Lorenz monotone, percentiles ordered.
4. **SSJ / Reiter / KS:** each returns a stable solution (spectral radius ≤ 1) with
   economically sensible IRFs (a positive endowment shock lowers `r` on impact; KS R²
   high).
5. **Spec/constructor:** `model` field defaults to `:aiyagari`; `:huggett` set correctly;
   `_update_ha_params` preserves `model`.

Run with `MACRO_MULTIPROCESS_TESTS=1` on the HA test file only (plus any directly
affected DSGE downstream), never the full suite, never serial.

## 5. Docs & references

- `docs/src/dsge_ha.md`: new "Huggett (1993) — the risk-free rate" section with a
  quick-start recipe (steady state → report → wealth distribution → one dynamic solve).
  All `@example` blocks must run; verify via
  `julia --project=docs docs/verify_examples.jl docs/src/dsge_ha.md` (follow
  `docs/docrule.md` first).
- `examples.jl` docstring table + `load_ha_example` table updated.
- `memory/refs.md`: add Huggett (1993, JEDC).
- CLAUDE.md / MEMORY.md `load_ha_example` lists updated to include `:huggett`.
- Regenerate plots before any push (`docs/generate_plots.jl`) per CLAUDE.md.

## 6. Implementation sequencing (for the plan)

1. `model` field + clearing-closure refactor + Aiyagari regression test (no behavior
   change to default path).
2. `:huggett` income/example/agg-spec + steady-state correctness test (reproduce paper).
3. SSJ GE-close + test.
4. Reiter model-aware aggregate block + test.
5. Krusell-Smith r-forecasting variant + test.
6. Docs, refs, memory, exports.

This ordering yields a faithful, tested steady-state deliverable (steps 1–2) before the
larger dynamics work (steps 3–5), so the sub-project has value even if a later method
needs iteration.

## 7. Risk & scope notes

- **Largest/riskiest:** the Huggett-aware **Reiter** aggregate block and the
  **KS r-forecasting** variant (genuinely new logic, not config). SSJ GE-close is
  moderate. Steps 1–2 are small.
- **Lowest-risk invariant:** the Aiyagari default path is byte-for-byte preserved
  (closure default + regression test).
- **Out of scope:** continuous-time Huggett, OLG, and the Den Haan accuracy-test
  sub-project (next in the v0.5.1 sequence).

## 8. References

- Huggett, M. (1993). The risk-free rate in heterogeneous-agent incomplete-insurance
  economies. *Journal of Economic Dynamics and Control*, 17(5–6), 953–969.
- Aiyagari, S. R. (1994). Uninsured idiosyncratic risk and aggregate saving. *QJE*.
- Auclert, Bardóczy, Rognlie & Straub (2021). Using the sequence-space Jacobian to
  solve and estimate heterogeneous-agent models. *Econometrica*.
- Reiter, M. (2009). Solving heterogeneous-agent models by projection and perturbation.
  *JEDC*.
- Krusell, P., & Smith, A. A. (1998). Income and wealth heterogeneity in the
  macroeconomy. *JPE*.
- Young, E. R. (2010). Solving the incomplete markets model with aggregate uncertainty
  using the Krusell–Smith algorithm and non-stochastic simulations. *JEDC*.
