# Empirical-Methods Validity Audit — Findings Report

**Date started:** 2026-06-23
**Branch:** `fix`
**Reference:** Ferroni & Canova `BVAR_` toolbox (`/Users/chung/Downloads/BVAR_-master-2`)
**Method:** Hybrid — numerical cross-validation (Octave oracle) for reference-overlapping
estimators; algorithmic code review vs canonical formulas elsewhere. See
`docs/superpowers/specs/2026-06-23-empirical-methods-audit-design.md`.

**The reference is NOT ground truth.** A difference between ours and `BVAR_` does not mean ours is
wrong — sometimes ours is the better implementation. Every discrepancy is judged against correct
econometric theory and classified as: real bug in ours / convention difference / reference is
weaker or wrong. Only the first becomes a fix.

## Severity legend

| Severity | Meaning |
|----------|---------|
| **Critical** | Silently wrong results on valid input, or crash on valid input. |
| **High** | Wrong results in a common/default configuration. |
| **Medium** | Wrong in an edge case, or misleading standard errors / diagnostics. |
| **Low** | Cosmetic, documentation, or correctness-irrelevant performance issue. |

**Status values:** `Open` · `Confirmed` · `Fixed (commit <sha>)` · `Won't-fix (convention)` · `Won't-fix (by design)`.

## Fixes applied on branch `fix`

Only the three findings I verified **myself** (numerically/from source) were fixed, each with a
regression test; all targeted test groups pass (factor 66, FAVAR 2225, VAR statsapi/core/irf/fevd/hd,
BVAR bayesian/minnesota/bgr) plus the full suite.

- **F-06 (Critical)** — `src/factor/static.jl`: normalize PCA factors to unit variance so
  `predict = F·Λ'` is the true projection; fixes `residuals`/`r2`, Bai–Ng factor selection, and the
  FAVAR panel-IRF magnitudes. Regression test in `test/factor/test_factormodel.jl`.
- **F-02 (High)** — `src/bvar/priors.jl`: add the multivariate-gamma + `−½·T_eff·n·log π` terms so
  `log_marginal_likelihood` returns the true ML (cross-model comparison now valid; τ-selection
  unchanged). Regression test in `test/bvar/test_bayesian.jl` (vs `matrictint` assembly).
- **F-01 (Medium)** — `src/var/estimation.jl`: dof-adjust the coefficient covariance inside `vcov`
  (`U'U/(T_eff−k)`), leaving `model.Sigma` as the ML estimate (so IRF/FEVD/HD/loglik/IC are
  unchanged). Regression test in `test/var/test_statsapi.jl`.

**Deliberately NOT fixed (left for review):** F-07 (pvar Windmeijer — needs the full correction
implemented), F-03/F-04/F-05 (Low; conventions/doc/sign-normalization that could disturb existing
behavior), and all "unverified candidate findings" below (the automated review had a high
false-positive rate — see the refuted list).

## Findings

| ID | Module | file:line | Severity | Status | Summary |
|----|--------|-----------|----------|--------|---------|
| F-01 | var | `src/var/estimation.jl:43` | Medium | Fixed | `dof_adj = max(T_eff-k, T_eff)` always equals `T_eff` (since k≥1), so `Sigma` is the ML covariance `U'U/T_eff`, never the dof-adjusted `U'U/(T_eff-k)` the comment/warning intend. Propagates to `vcov`→`stderror`→`confint`: reported VAR coefficient SEs/CIs are too small (anti-conservative). Reference `ols_reg.m:86` uses `1/(N-K)`. Confirmed numerically (gap 2.35% on a T=300,k=7 fixture; grows as k/T rises). Fix must preserve `loglikelihood` (wants ML cov) and decide IRF/identification convention. |
| F-02 | bvar | `src/bvar/priors.jl:129-131` | High | Fixed | `log_marginal_likelihood` omits the multivariate-gamma terms `logΓₙ(ν_post/2) − logΓₙ(ν_prior/2)` and the `−½·n·T_eff·log π` constant present in the reference `matrictint.m`. Confirmed numerically: our value −2653.18 vs true ML −1343.27 (assembled from Octave `matrictint`); gap 1309.91 = the analytic omitted terms to 1e-13. Invariant for tau-tuning (active dummy blocks fixed), but the returned value is **not** the marginal likelihood — cross-lag/cross-model/cross-n comparison is invalid. `checks_bvar_ml.jl`. |
| F-03 | bvar | `src/var/types.jl:223`, `src/bvar/priors.jl:21` | Low | Confirmed | Hyperparameter naming/convention hazard (NOT a numerical bug). `tau` is inverse-tightness (divides dummies; larger ⇒ looser) vs reference `mnprior.tight` (multiplies; larger ⇒ tighter). `lambda` (our sum-of-coefficients) and `mu` (our co-persistence) are **swapped** vs `rfvar3` semantics, yet the defaults `lambda=5, mu=2` are copied from the reference's `lambda`(co-persistence)`=5`, `mu`(own)`=2` guidance — so the default prior differs from what a user porting reference settings expects. Doc fix recommended. |
| F-04 | bvar | `src/bvar/estimation.jl:76` | Low | Confirmed | `optimize_hyperparameters(Y_eff, p)` passes the already-lagged `Y_eff` as if it were raw data, so the marginal likelihood used for tau selection drops another `p` rows (double-lag) and scales the prior from `Y_eff` while estimation (`gen_dummy_obs(Y, …)` at line 77) scales from the full `Y`. Affects only the auto-selected `tau`, not estimation given a fixed `tau`. Minor sample/scaling inconsistency. |
| F-06 | factor | `src/factor/static.jl:100-101` | **Critical** | Fixed | `estimate_factors` mixes PCA normalizations: `factors = X·Vᵣ` (variance λ) paired with `loadings = Vᵣ·√Λ` (which assume unit-variance factors). So `predict = F·Λ' = X·Vᵣ·√Λ·Vᵣ'` overshoots each principal component by `√λ` instead of equalling the projection `X·Vᵣ·Vᵣ'`. Pure-Julia check: `predict` vs `X·Vᵣ·Vᵣ'` fails (maxrel 8660). Propagates to `residuals`, `r2`, and **`ic_criteria` (Bai–Ng) selects the wrong number of factors — 1 vs the true 3** (reference `baing.m` correctly gets 3). Fix: normalize factors to unit variance (`F = X·Vᵣ·Λ^{-1/2}`) keeping `loadings = Vᵣ·Λ^{1/2}`, so `F·Λ' = X·Vᵣ·Vᵣ'`. Fix verified: corrected `predict==projection` and Bai–Ng then selects 3. **Blast radius:** FAVAR (`favar_panel_irf` = `Λ·factor_irf` uses these inconsistent loadings → panel-wide IRFs mis-scaled by `√λ` per factor) and `estimate_favar` `:bayesian` init. Fixing F-06 fixes the FAVAR mapping too. `checks_factor.jl`. |
| F-05 | core | `src/core/identification.jl:184-192` | Low | Confirmed | `identify_long_run` (Blanchard–Quah) is algebraically correct (`L·Q = (I−A_sum)·chol(C1ΣC1')`, verified `Q_BQ Q_BQ'=Σ` and long-run cumulative impact lower-triangular vs reference, IRF² match 1.7e-12), but applies **no sign normalization** of the permanent shock, whereas reference `iresponse_longrun.m` pins `Q(1,1)>0`. The permanent shock's sign is therefore arbitrary/run-dependent — usability/reproducibility nit, not a numerical error. |

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
| core | Cholesky IRF (`compute_irf`) | Numerical vs `iresponse.m` | `checks_irf.jl`: maxrel 1.5e-12 (same Φ,Σ) |
| core | FEVD (`_compute_fevd`) | Numerical vs `fevd.m` | `checks_irf.jl`: maxrel 4.7e-15 |
| core | structural shocks `ε=Q'L⁻¹u` | Numerical vs `ε=L⁻¹u` (Cholesky) | `checks_irf.jl`: maxabs 0 |
| core | historical decomposition identity | Self-consistency | `checks_irf.jl`: contrib+initial==actual exact |
| core | long-run / Blanchard–Quah IRF | Numerical vs `iresponse_longrun.m` (sign-free) | `checks_irf.jl`: IRF² maxrel 1.7e-12; long-run impact lower-triangular |
| var | unconditional forecast recursion (`predict`) | Numerical vs `forecasts.m` no-shock | `checks_forecast.jl`: maxrel 1.1e-13 |
| filters | HP filter (`hp_filter`) | Numerical vs `Hpfilter.m` | `checks_filters.jl`: maxrel 1.8e-11 (identical pentadiagonal system) |
| filters | Hamilton filter (`hamilton_filter`) | Numerical vs `hamfilter.m` | `checks_filters.jl`: cycle/trend maxrel 7e-15 |
| filters | Baxter–King (`baxter_king`), symmetric | Canonical BK property test | zero-sum `a₀+2Σaⱼ`=5.5e-17; loses K each end (reference `bkfilter.m` is actually Christiano–Fitzgerald, not symmetric BK) |
| factor | PCA eigen-decomposition (eigenvalues/explained variance) | Code review + reference projection | eigenvalues/`explained_variance` correct; the **reconstruction** (`predict`) is buggy → F-06 |
| factor | Bai–Ng penalty formulas IC1/IC3 | Code review vs `baing.m` + Bai-Ng 2002 | IC1/IC3 penalties correct (match `baing` jj=1/jj=3); IC2 matches *published* formula (reference adds a non-standard ×2) — but residuals feeding all three are wrong until F-06 fixed |
| favar | BBE two-step structure (PCA → orthogonalize vs Y_key → augmented VAR) + `favar_panel_irf` mapping form | Code review vs BBE 2005 | structure & mapping `Λ·factor_irf` correct in form; magnitudes inherit F-06 (shared `estimate_factors` loadings) |
| lp | `estimate_lp` (Jordà 2005) OLS + IRF extraction | Numerical (manual OLS) + code review | regression coef == manual OLS; standard Jordà form (regress yₜ₊ₕ on shock+lags, IRF=shock coef) |
| lp | horizon-aware NW bandwidth floor `max(auto, h+1)` | Code review vs Jordà MA(h) serial-corr | correct floor; reference `directmethods` uses `lags+h+1` (also valid) |
| lp | `structural_lp` (Plagborg-Møller & Wolf 2021) ε=Q'L⁻¹u → LP per shock | Code review | correct PMW construction; LP-Cholesky h=0 impact ≈ Cholesky L (0.014 finite-sample control-set gap) |

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
- **`bkfilter.m` ≠ Baxter–King.** The reference function *named* `bkfilter.m` actually implements
  the Christiano–Fitzgerald (1999) default filter (drift removal + full-sample asymmetric weights),
  per its own header. Our `baxter_king` is the classic symmetric BK (loses 2K obs) — a different,
  correctly-implemented filter, so a direct numeric comparison would be meaningless.

| F-07 | pvar | `src/pvar/estimation.jl:386-469` | High | Confirmed | `_windmeijer_correct` computes the derivative matrices `D_j` (l.401-415) and `D_jk` (l.420-436) but **never uses them** — it returns the plain two-step sandwich `gmm_sandwich_vcov(S_ZX, W, D_e)` (l.459). The Windmeijer (2005) finite-sample correction is therefore a no-op placeholder, so reported two-step PVAR standard errors are downward-biased (a well-known, sometimes large, bias). Fix = implement the full correction term (non-trivial). |

## Tier-2 audit — method and candidate findings

Tier-2 (no MATLAB reference) was swept by parallel code-review agents, then **every candidate was
re-checked against source and theory by hand**. The agents had a high false-positive rate on
"formula" claims, so only hand-verified items are recorded as findings.

**Refuted agent claims (verified CORRECT in source — do NOT change):**
- `nongaussian/ml.jl:113` Student-t standardized log-pdf — correct: `f_X(x)=f_Z(x·s)·s` with
  `s=√(ν/(ν-2))` is exactly the unit-variance density. (Agent wrongly said "divide not multiply".)
- `nongaussian/ica.jl:251-254` JADE Gaussian-cumulant subtraction — correct: subtracts the three
  pairings `δ_{ij}δ_{kl}+δ_{ik}δ_{jl}+δ_{il}δ_{jk}`, yielding −3 on the `(i,i)` diagonal when `i=j`.
- `core/arias.jl:~220` A0 form — only `|det(A0)|` enters the importance weight, and
  `|det((L')⁻¹Q)|=|det(Q'L⁻¹)|`, so the weight is unaffected (cosmetic at most).
- `spectral/cross.jl:108` phase `atan2(quad,co)` with `quad=−Im` — a self-consistent engineering
  convention for the phase sign, not an error.
- `teststat/helpers.jl:43` ADF p-value normal tail — only used in the fail-to-reject region
  (p>0.10), monotonic and bounded; a Low approximation, not the HIGH the agent claimed.
- `nongaussian/shared.jl` heteroskedasticity uses `eigen(Σ₁⁻¹Σ₂)` — that IS the correct generalized
  eigendecomposition (real eigenvalues; SVD would be wrong here).

**Unverified candidate findings (leads only — need confirmation before any fix):**
- `did/sun_abraham.jl:154-160` — missing-cohort-cell event-times left at 0 (not skipped) could bias
  aggregation when the reporting window exceeds a cohort's estimable range (edge case; Medium?).
- `did/callaway_santanna.jl` — `cluster` kwarg may not feed a cluster-robust SE in all aggregation
  schemes (verify; Medium?).
- `nowcast/dfm.jl:~380` — EM M-step may update only the first `p` factor-VAR lags while the
  state spans `p_eff=max(p,5)` lags for quarterly aggregation (verify; would affect MM cases).
- `reg/iv.jl:54` — first-stage F may use #total instruments instead of #excluded for the
  Stock-Yogo weak-IV diagnostic (Low-Medium; point estimates unaffected).
- `pvar` Hansen-J aggregation across equations / d.o.f. (verify; affects J p-value only).

**Verified correct (hand-checked):** VECM (Johansen trace/max stat, α/β Phillips normalization,
VECM↔VAR mapping), ARIMA (Harvey state-space, exact-ML + CSS likelihood, ψ-weight forecast
variance), ARCH/GARCH/EGARCH/GJR (log-likelihood, persistence constraints incl. GJR α+γ/2+β and
EGARCH |β|<1), GMM (J-test d.o.f. = #moments−#params, HAC sandwich), reg/preg core (OLS, HC0–HC3,
cluster-robust finite-sample factor, logit/probit IRLS + delta-method margins, FE within + RE
Swamy-Arora), most teststat statistics (KPSS, PP, DF-GLS, Ng-Perron, Johansen, Zivot-Andrews,
Bai-Perron, CIPS) and their critical-value tables, DSGE estimation seams (Kalman likelihood,
Lyapunov init, QZ→state-space wiring, prior bounds), spectral periodogram/Welch/Burg + windows,
X-13 seasonal-MA filter coefficients.

## Cases where OURS is correct and the reference deviates

- **Bai–Ng ICp2 penalty.** Reference `baing.m` `jj==2` uses `(N+T)/(NT)·2·log(min(N,T))·k` — an
  extra factor of **2** vs the canonical Bai & Ng (2002) ICp2 penalty `(N+T)/(NT)·log(min(N,T))·k`.
  Our `ic_criteria` IC2 matches the **published** formula (no factor of 2). So here ours is the
  correct one; the reference's IC2 is non-standard. (Currently our IC is unusable anyway until F-06
  is fixed; once fixed, our IC1/IC3 should match `baing` and IC2 will legitimately differ by that
  reference-side factor of 2.)
