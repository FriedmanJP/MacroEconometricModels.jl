# Empirical-Methods Validity Audit — Design Spec

**Date:** 2026-06-23
**Branch:** `fix`
**Reference:** Ferroni & Canova `BVAR_` MATLAB toolbox (`/Users/chung/Downloads/BVAR_-master-2`)
**Tooling available:** Octave 11.1.0, MATLAB R2023b

## Objective

Verify that every empirical estimator in MacroEconometricModels.jl (MEMs.jl) **works as
intended** — produces numerically correct results, not merely runs without error. Cover both
frequentist and Bayesian estimators. Where a counterpart exists in the `BVAR_` reference,
validate against it; otherwise self-check against canonical formulas.

## Guardrails — bug vs. convention

Two failure modes are actively distinguished and must never be conflated:

- **Real bug** — wrong formula, wrong degrees of freedom, wrong sign/orientation, indexing
  error, wrong scaling, broken edge case.
- **Convention difference** — lag ordering (most-recent-first vs. last), constant placement
  (first vs. last column), Cholesky orientation (`LL'` vs `L'L`), variable ordering. These are
  **not** bugs when internally consistent and yielding equivalent results. They will not be
  "fixed."

Every candidate finding is confirmed by **reading the actual source file** before it enters the
report. Subagent/exploration summaries are treated as *leads to verify*, never as confirmed
findings.

## Methodology — hybrid verification

### A. Numerical cross-validation (reference-overlapping core)

A reproducible oracle harness under `test/oracle/`:

1. Generate one shared synthetic dataset and load one real example dataset; write to disk
   (CSV/`.mat`) so both stacks read identical inputs.
2. Run the `BVAR_` routine in **Octave headless** (`octave --eval`/script), dumping outputs.
3. Run our Julia routine on identical inputs.
4. Compare within tolerance and record pass/fail + max abs/rel error.

**Deterministic routines** (exact compare, tol ~1e-8…1e-6):
VAR OLS coefficients & residual covariance, companion eigenvalues, Cholesky IRF, FEVD,
historical decomposition, unconditional forecast, Minnesota/dummy-observation prior
construction, NIW posterior moments (`B_post, V_post, S_post, ν_post`), marginal likelihood
(`matrictint` / `mniw_log_dnsty`), long-run / Blanchard–Quah IRF, factor PCA, `baing`
factor-number selection, filters (HP, Baxter–King, Christiano–Fitzgerald, Hamilton).

**Stochastic routines** (compare deterministic conditionals/moments, not RNG draws, because
Octave and Julia RNG streams differ):
Gibbs / IW–MN sampler → compare posterior moments and convergence behavior, plus property
checks (posterior mean → OLS as prior loosens; IW draw mean correct). Sign restrictions /
proxy-SVAR → compare the deterministic impact algebra and restriction-checking logic, plus
distributional checks over many draws.

### B. Algorithmic code review (everything else)

Read our source against the reference math (or canonical formulas where no reference exists):
write out the implemented formula, verify d.o.f., scaling, indexing, sign normalization,
companion-form layout, and edge cases. Cross-reference existing unit tests.

## Coverage & priority (reference-first)

### Tier 1 — deep numerical + code review (reference overlap)
- `var` — reduced-form OLS, companion, covariance d.o.f.
- `bvar` — Minnesota & NIW/dummy priors, Gibbs & direct samplers, marginal likelihood,
  hyperparameter handling.
- `core/identification` + `core/uhlig` — Cholesky / sign / proxy-IV / long-run IRF, FEVD,
  historical decomposition, unconditional & conditional forecasts, Mountford–Uhlig penalty.
- `lp` — local projections (Jordà), horizon-aware Newey–West.
- `factor` — static PCA, dynamic, structural, generalized; factor-number selection.
- `favar` — Bernanke–Boivin–Eliasz.
- `nowcast` — mixed-frequency DFM, Kalman with missing data.
- `pvar` — panel VAR GMM (Arellano–Bond, Blundell–Bond).
- `nongaussian` — ICA / ML / heteroskedasticity identification.
- `filters` — HP, Baxter–King, Christiano–Fitzgerald, Hamilton.

### Tier 2 — code-review self-check (no reference counterpart)
`dsge` (broad sanity pass focused on estimation / likelihood / solver seams; existing
gensys/klein/bk cross-checks are leveraged, not re-derived), `teststat`, `reg`/`preg`, `did`,
`vecm`, `arima`, `garch`/`arch`, `gmm`, `x13`, `spectral`.

## Deliverables

1. **Branch `fix`** off current HEAD. *(done)*
2. **Findings report** — `docs/audit/empirical-methods-audit.md` (tracked; `docs/superpowers/`
   is gitignored). Each issue:
   module, `file:line`, severity (Critical / High / Medium / Low), description, reference
   evidence, suggested fix. Plus a **verified-correct ledger** distinguishing what was actually
   checked from what was assumed.
3. **Oracle harness** — `test/oracle/` (Octave scripts + Julia comparison driver), reproducible.
4. **Fixes** — applied in severity order, each with a regression test, in reviewable batches.
   Test suite runs with `MACRO_MULTIPROCESS_TESTS=1` (never serial).

## Phasing

- **Phase 0** — branch + oracle harness skeleton; verify Octave loads `bvartools` and
  round-trips one VAR end to end.
- **Phase 1** — Tier 1 audit (numerical + code), module by module; accumulate findings.
- **Phase 2** — Tier 2 audit (code review); accumulate findings.
- **Phase 3** — severity-ranked fixes with regression tests, in batches the user reviews.

The report is updated continuously. Nothing is fixed before its finding is verified against
source.

## Already-identified leads (to verify, not yet confirmed)

- `src/var/estimation.jl`: `dof_adj = max(T_eff - k, T_eff)` appears to always select `T_eff`
  (since `T_eff > T_eff - k`), i.e. an ML/biased covariance denominator, vs. the reference's
  `1/(nobs - nk)`. Verify intended behavior and whether it affects reported standard errors.
- Lag ordering & constant placement differ from the reference (ours: const-first then lags by
  variable; reference: lags most-recent-first, const last). Expected convention difference —
  confirm internal consistency only.
- Minnesota dummy-observation construction: confirm our `gen_dummy_obs` block formulas match
  `varprior.m` + `rfvar3.m` (AR block, sum-of-coefficients, co-persistence, covariance block,
  and where `lambda`/`mu` are applied).

## Out of scope

- Performance optimization, refactoring unrelated to correctness, new features.
- Re-deriving the full DSGE solver suite (already cross-checked); only the estimation/likelihood
  seams get a correctness pass.
