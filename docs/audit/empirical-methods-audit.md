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
| F-02 | bvar | `src/bvar/priors.jl:129-131` | High | Confirmed | `log_marginal_likelihood` omits the multivariate-gamma terms `logΓₙ(ν_post/2) − logΓₙ(ν_prior/2)` and the `−½·n·T_eff·log π` constant present in the reference `matrictint.m`. Confirmed numerically: our value −2653.18 vs true ML −1343.27 (assembled from Octave `matrictint`); gap 1309.91 = the analytic omitted terms to 1e-13. Invariant for tau-tuning (active dummy blocks fixed), but the returned value is **not** the marginal likelihood — cross-lag/cross-model/cross-n comparison is invalid. `checks_bvar_ml.jl`. |
| F-03 | bvar | `src/var/types.jl:223`, `src/bvar/priors.jl:21` | Low | Confirmed | Hyperparameter naming/convention hazard (NOT a numerical bug). `tau` is inverse-tightness (divides dummies; larger ⇒ looser) vs reference `mnprior.tight` (multiplies; larger ⇒ tighter). `lambda` (our sum-of-coefficients) and `mu` (our co-persistence) are **swapped** vs `rfvar3` semantics, yet the defaults `lambda=5, mu=2` are copied from the reference's `lambda`(co-persistence)`=5`, `mu`(own)`=2` guidance — so the default prior differs from what a user porting reference settings expects. Doc fix recommended. |
| F-04 | bvar | `src/bvar/estimation.jl:76` | Low | Confirmed | `optimize_hyperparameters(Y_eff, p)` passes the already-lagged `Y_eff` as if it were raw data, so the marginal likelihood used for tau selection drops another `p` rows (double-lag) and scales the prior from `Y_eff` while estimation (`gen_dummy_obs(Y, …)` at line 77) scales from the full `Y`. Affects only the auto-selected `tau`, not estimation given a fixed `tau`. Minor sample/scaling inconsistency. |

## Verified-correct ledger

Routines checked and found correct (so we know what was actually verified vs. assumed).

| Module | Routine | Method | Evidence |
|--------|---------|--------|----------|
| var | OLS coefficients `B` (`estimate_var`) | Numerical vs `rfvar3` | `checks_var.jl`: maxrel 3e-14 after lag/const reorder |
| var | Residuals `U` | Numerical vs `rfvar3` | `checks_var.jl`: maxabs 3.6e-15 |
| var | `Sigma` numerator `U'U` (ML form) | Numerical vs `rfvar3` ML | `checks_var.jl`: maxrel 8.6e-16 (denominator is the F-01 bug) |
| var | `companion_matrix` eigenvalues / stationarity | Numerical vs ref companion | `checks_var.jl`: maxrel 1.8e-15 |
| bvar | Minnesota dummy blocks (AR/SOC/co-persistence/cov) form | Code review vs `varprior.m`+`rfvar3.m` | own-first-lag mean 1, higher lags→0 w/ `l^decay`, SOC→ΣAₗ=I, co-persist incl. constant — all correct (BGR parameterization) |
| bvar | `matrictint` reimplementation (used to assemble true ML) | Numerical vs Octave `matrictint` | `checks_bvar_ml.jl`: maxabs 0 on fixed case |
| bvar | `_draw_inverse_wishart(ν,S)` parameterization (scale S, mean S/(ν-n-1)) | Bartlett algebra vs `rand_inverse_wishart.m` + property test | matches standard IW; `checks_bvar_posterior.jl` E[Σ] within 0.24% |
| bvar | Conjugate NIW posterior moments `B_post, V_post, ν_post, S_post` | Code review + sampler reproduces them | textbook conjugate update; `checks_bvar_posterior.jl` |
| bvar | `:direct` and `:gibbs` samplers | Property test vs analytic posterior | `checks_bvar_posterior.jl`: E[B], E[Σ] match analytic moments within MC error |

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
