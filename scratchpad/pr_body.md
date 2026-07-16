## v0.6.6 — Stage 11: report() display quality (#260–#275)

The display-quality cluster from the road-to-v1 plan. Sixteen issues (T161–T176)
hardening the publication-table layer across **every** result type so `report()`,
`show()`, and `print_table()` output is correct, consistent, and non-truncating in
files, pipes, Documenter, and CI logs. Derived from the report-quality audit
(`docs/plans/2026-07-03-report-quality-audit.md`).

### What landed

**Foundational (core/display.jl):**
- **#260** disable non-TTY table cropping — stars/CI columns are never silently dropped
- **#261** suppress all-empty column-label rows; render legends/notes as plain text
- **#262** rewrite `_fmt` as an `@sprintf` fixed-decimal `String` formatter — no ragged decimals, no `-0.0`, no collapsed exponentials (+ `Printf` dep)
- **#263** dust/reference-row guard + degenerate-fit banner; **#264** dedup `_select_horizons`
- **#265** shared conventions: `_label` dict, `_yesno`, one p-value formatter, `(Intercept)`, CV order, AIC "(per obs.)"

**Dispatches & content standard:**
- **#266** `report()` dispatches for 13 show-existing types + wrappers for bare NamedTuple/Vector/`nothing` returns
- **#267** content standard — every report ends with an estimates/IRF/SS table (LP family, DSGE steady-state, factor/SVAR IRF, nowcast headline)

**Dialect, numerical & one-offs:**
- **#268** unify display dialects (drop ASCII banners, port CT/OLG to tables) + package-wide `report(io, x)` io-plumbing for the bespoke VAR/VECM/DSGEEstimation bodies
- **#269** CI/quantile/SE label fixes (BVAR 2.5%/97.5%, VECM "no CI", LP-FEVD %, volatility note)
- **#270** correct `johansen_test` cointegration-rank off-by-one + dedupe rank selectors *(estimation fix)*
- **#271** panel between/overall-R² excludes the entity fixed effect (was trivially 1.0) *(estimation fix)*
- **#272** flag boundary-hit GLP hyperparameters in the BVAR nowcast report
- **#273** display one-offs batch (HD labels, |Loading|, ARIMA grid, JB names, BN note, HA log10, KS progress)

**Backends & regression harness:**
- **#274** compilable-quality LaTeX booktabs backend + math-moded headers/p-values + merged CI header across all backends
- **#275** goldens + invariants regression harness (`test/display/`, 19 fixtures) locking the display layer against silent regressions

### Verification
Each issue was verified green against its relevant per-module test file(s) locally
(display checks consolidated in `test/core/test_display_backends.jl`; the new
`Display` group runs 124 invariants + 20 goldens). Full suite + docs left to CI.

### Merge note
This branch is stacked on `release/v0.6.5` (PR #380), which is not yet merged to
`dev`. Until #380 merges, the diff here temporarily includes v0.6.5's commits; it
self-corrects to the 19-commit v0.6.6 delta once #380 lands. **Merge #380 first.**

Closes #260
Closes #261
Closes #262
Closes #263
Closes #264
Closes #265
Closes #266
Closes #267
Closes #268
Closes #269
Closes #270
Closes #271
Closes #272
Closes #273
Closes #274
Closes #275

https://claude.ai/code/session_01QZizX88NDsBZrBgdqPcLQp
