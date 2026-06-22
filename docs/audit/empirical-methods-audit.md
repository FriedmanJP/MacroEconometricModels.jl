# Empirical-Methods Validity Audit — Findings Report

**Date started:** 2026-06-23
**Branch:** `fix`
**Reference:** Ferroni & Canova `BVAR_` toolbox (`/Users/chung/Downloads/BVAR_-master-2`)
**Method:** Hybrid — numerical cross-validation (Octave oracle) for reference-overlapping
estimators; algorithmic code review vs canonical formulas elsewhere. See
`docs/superpowers/specs/2026-06-23-empirical-methods-audit-design.md`.

## Severity legend

| Severity | Meaning |
|----------|---------|
| **Critical** | Silently wrong results on valid input, or crash on valid input. |
| **High** | Wrong results in a common/default configuration. |
| **Medium** | Wrong in an edge case, or misleading standard errors / diagnostics. |
| **Low** | Cosmetic, documentation, or correctness-irrelevant performance issue. |

**Status values:** `Open` · `Confirmed` · `Fixed (commit <sha>)` · `Won't-fix (convention)` · `Won't-fix (by design)`.

## Findings

| ID | Module | file:line | Severity | Status | Summary |
|----|--------|-----------|----------|--------|---------|
| F-01 | var | `src/var/estimation.jl:43` | Medium | Confirmed | `dof_adj = max(T_eff-k, T_eff)` always equals `T_eff` (since k≥1), so `Sigma` is the ML covariance `U'U/T_eff`, never the dof-adjusted `U'U/(T_eff-k)` the comment/warning intend. Propagates to `vcov`→`stderror`→`confint`: reported VAR coefficient SEs/CIs are too small (anti-conservative). Reference `ols_reg.m:86` uses `1/(N-K)`. Confirmed numerically (gap 2.35% on a T=300,k=7 fixture; grows as k/T rises). Fix must preserve `loglikelihood` (wants ML cov) and decide IRF/identification convention. |
| F-02 | bvar | `src/bvar/priors.jl:129-131` | High | Open (verify in 1.3) | `log_marginal_likelihood` omits the multivariate-gamma terms `logΓₙ(ν_post/2) − logΓₙ(ν_prior/2)` and the `−(n·T_eff/2)·log π` constant present in the reference `matrictint.m`. Invariant terms cancel for tau-tuning (active dummy blocks fixed), but the returned value is **not** the marginal likelihood and cross-lag/cross-model comparison is invalid. To confirm numerically vs `matrictint` in Task 1.3. |

## Verified-correct ledger

Routines checked and found correct (so we know what was actually verified vs. assumed).

| Module | Routine | Method | Evidence |
|--------|---------|--------|----------|
| var | OLS coefficients `B` (`estimate_var`) | Numerical vs `rfvar3` | `checks_var.jl`: maxrel 3e-14 after lag/const reorder |
| var | Residuals `U` | Numerical vs `rfvar3` | `checks_var.jl`: maxabs 3.6e-15 |
| var | `Sigma` numerator `U'U` (ML form) | Numerical vs `rfvar3` ML | `checks_var.jl`: maxrel 8.6e-16 (denominator is the F-01 bug) |
| var | `companion_matrix` eigenvalues / stationarity | Numerical vs ref companion | `checks_var.jl`: maxrel 1.8e-15 |
| bvar | Minnesota dummy blocks (AR/SOC/co-persistence/cov) form | Code review vs `varprior.m`+`rfvar3.m` | own-first-lag mean 1, higher lags→0 w/ `l^decay`, SOC→ΣAₗ=I, co-persist incl. constant — all correct (BGR parameterization) |

## Notes / convention map

Cross-stack convention differences that are NOT bugs (recorded so they are not re-flagged):

- **VAR regressor layout.** Ours: `X = [1, Y_{t-1}, …, Y_{t-p}]`, constant **first**; `B` rows
  `[const; lag1; …; lagp]`. Reference `rfvar3`: `X = [Y_{t-1}, …, Y_{t-p}, const]`, constant
  **last**. Lag direction (t-1 first) and within-lag variable order (natural) match. Equivalent
  up to a row permutation of `B`.
- **Minnesota parameterization.** Ours uses the BGR stacked `(Y_dummy, X_dummy)`; reference uses
  Sims block-dummies (`varprior` + `rfvar3` generates lagged regressors). Both valid.
- **`tau` is inverse-tightness:** we divide dummies by `tau` (larger `tau` ⇒ looser), whereas the
  reference multiplies by `mnprior.tight` (larger ⇒ tighter). Naming caveat, not a bug.
- **`lambda`/`mu` are swapped vs the reference naming:** our `lambda` drives the
  sum-of-coefficients prior (reference calls this `mu` in `rfvar3`), our `mu` drives the
  co-persistence/dummy-initial-observation prior (reference's `lambda`). Both are present and
  correctly formed. Documentation hazard only — flagged as a Low doc item below if confirmed.
