# Den Haan (2010) Accuracy Test — Design

**Date:** 2026-06-23 · **Release:** v0.5.1 (item 2 of 4) · **Status:** Approved.

## 1. Goal

Add the Den Haan (2010) dynamic accuracy test for Krusell-Smith solutions: simulate the
aggregate two ways under the same shock path — (i) **reference**, the explicit
cross-sectional (Young) simulation, and (ii) **PLM-only**, iterating the aggregate law of
motion on its *own* forecasts without updating from the simulated cross-section — and
report the maximum and mean errors between them, plus the standard-deviation comparison.
Den Haan's point: R²/standard-error are inadequate (R² can be 0.9999 while σ(K) is 14%
off); the multi-step PLM-only-vs-reference error is the real test.

## 2. KS PLM upgrade (Aiyagari)

In `_krusell_smith_solve` (`:aiyagari` path), change the law of motion from
`log K' = b₁ + b₂ log K` to **`log K' = b₁ + b₂ log K + b₃ z`**: regress `log K_{t+1}` on
`[1, log K_t, z_t]` (the simulation already feeds `z` via `K_eff = K·exp(z)`, so only the
regression and `plm_coefficients[:K]` (now length 3) change). The explicit simulation
loop is unchanged (it uses realized `K`, not the PLM). Update the existing KS testset for
the 3-coefficient PLM.

## 3. `den_haan_test`

```julia
den_haan_test(ks::KrusellSmithSolution{T};
              T_sim=10000, T_burn=1000, rho_z=0.95, sigma_z=0.007, seed=98765)
    → DenHaanAccuracy{T}
```

Dispatch on `ks.spec.model`:

- **`:aiyagari`** (aggregate `:K`): same `z` path and `K₀ = K_ss` for both.
  - Reference `Kᵗ`: explicit Young simulation (reuse the existing KS inner loop —
    factor it into `_simulate_explicit_K(ss, ip, grid, income, price_fn, params, z)`).
  - PLM-only `K̂ᵗ`: `log K̂_{t+1} = b₁ + b₂ log K̂_t + b₃ z_t` (scalar iteration).
  - `dh_max = max_t 100·|log Kᵗ − log K̂ᵗ|`, `dh_mean = mean(...)`, over `t > T_burn`.
  - `sigma_ref = std(log Kᵗ)`, `sigma_plm = std(log K̂ᵗ)`.

- **`:huggett`** (aggregate `:r`): PLM `r = b₁ + b₂ z` (from the Huggett KS).
  - Reference `rᵗ`: explicit per-period clearing (reuse the KS-Huggett Newton-clearing
    simulation — factor into `_simulate_explicit_r(...)`).
  - PLM-only `r̂ᵗ = b₁ + b₂ z_t`.
  - Errors in **percentage points of the rate**: `dh_max = max_t 100·|rᵗ − r̂ᵗ|`,
    `dh_mean = mean(...)`; `sigma_ref/plm = std(rᵗ), std(r̂ᵗ)`.

Shock parameters default to `(0.95, 0.007)` for Aiyagari and to
`spec.het_params[:rho_e]/[:sigma_e]` for Huggett (kwargs override).

## 4. Result type & display

```julia
struct DenHaanAccuracy{T<:AbstractFloat}
    aggregate::Symbol           # :K or :r
    dh_max::T                   # max error (% for K, pp for r)
    dh_mean::T                  # mean error
    sigma_ref::T                # std dev of reference aggregate
    sigma_plm::T                # std dev of PLM-only aggregate
    ref_path::Vector{T}
    plm_path::Vector{T}
    T_sim::Int
    T_burn::Int
end
```

`report(::DenHaanAccuracy)` — labelled table (max/mean error with units, σ comparison,
σ ratio). `show` — one-liner. Both follow the existing display conventions.

## 5. Files

- Modify `src/dsge/heterogeneous/krusell_smith.jl`: z-augmented PLM; factor the two
  explicit simulators; add `den_haan_test` + `DenHaanAccuracy`.
- Modify `src/dsge/heterogeneous/display.jl`: `report`/`show` for `DenHaanAccuracy`.
- Modify `src/MacroEconometricModels.jl`: export `den_haan_test`, `DenHaanAccuracy`.
- Modify `test/dsge/test_ha_dsge.jl`: 3-coef PLM check + Den Haan tests (Aiyagari + Huggett).
- Modify `docs/src/dsge_ha.md`: "Accuracy: the Den Haan (2010) Test" section + reference.

## 6. Tests (acceptance)

1. Aiyagari KS PLM now has 3 coefficients; existing KS testset still passes.
2. `den_haan_test(aiyagari_ks)` → `aggregate == :K`; finite `dh_max ≥ dh_mean ≥ 0`;
   `sigma_ref > 0`; `sigma_plm > 0`; `length(ref_path) == length(plm_path) == T_sim`;
   the z-augmented PLM yields *reasonable* fluctuations (`sigma_plm` within a factor ~3 of
   `sigma_ref`, not ≈0 as the z-free PLM would give).
3. `den_haan_test(huggett_ks)` → `aggregate == :r`; finite, small errors; positive σ.
4. `report(result)` runs without error.

## 7. Scope guard

Only `krusell_smith.jl`, `display.jl`, exports, tests, docs. No change to SSJ/Reiter or
the steady-state solver. Den Haan (2010) is the capital model; the Huggett `:r` analog is
a natural extension reusing existing infra.

## 8. References

- den Haan, W. J. (2010). Assessing the accuracy of the aggregate law of motion in models
  with heterogeneous agents. *Journal of Economic Dynamics and Control*, 34(1), 79–99.
- Krusell, P., & Smith, A. A. (1998). *JPE* 106(5). · Young, E. R. (2010). *JEDC* 34(1).
