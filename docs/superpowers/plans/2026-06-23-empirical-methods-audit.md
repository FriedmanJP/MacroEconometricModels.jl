# Empirical-Methods Validity Audit — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify every empirical estimator in MEMs.jl is numerically correct (works as intended), using the Ferroni–Canova `BVAR_` MATLAB toolbox as the reference where one exists and canonical formulas otherwise; produce a verified findings report and fix confirmed bugs with regression tests.

**Architecture:** Three movements. (1) Stand up a reproducible *oracle harness* (`test/oracle/`) that runs reference routines in headless Octave on shared data and compares against our Julia outputs within tolerance. (2) Audit module-by-module — numerical cross-validation for reference-overlapping code (Tier 1), code review against canonical formulas for the rest (Tier 2) — appending only *source-verified* findings to a report. (3) Fix confirmed findings in severity order, each via TDD (failing regression test → fix → pass).

**Tech Stack:** Julia (package under audit), Octave 11.1.0 (runs `BVAR_` reference), MATLAB R2023b (fallback), bash glue. Reference at `/Users/chung/Downloads/BVAR_-master-2`.

## Global Constraints

- Reference toolbox path: `/Users/chung/Downloads/BVAR_-master-2` (subdirs `bvartools/`, `cmintools/`). Add both to the Octave path before calling any routine.
- The post-2020 main estimator is `bvar_.m` (NOT `bvar.m`). Use `bvar_`.
- Run the Julia test suite ONLY with `MACRO_MULTIPROCESS_TESTS=1` — never serially.
- Never name a Julia variable `eps` (shadows `Base.eps`); use `resid`.
- A **bug** is a wrong formula / d.o.f. / sign / orientation / index / scaling / edge case. A **convention difference** (lag order, constant placement, Cholesky `LL'` vs `L'L`, variable order) is NOT a bug if internally consistent. Never "fix" a convention.
- No finding enters the report until confirmed by reading the actual source file. Subagent/exploration summaries are leads only.
- Findings report: `docs/audit/empirical-methods-audit.md` (tracked). Oracle harness: `test/oracle/` (tracked). The plan/spec live under `docs/superpowers/` (gitignored; force-add if committing).
- Tolerances: deterministic numeric compare default `rtol=1e-6`, `atol=1e-8`; loosen explicitly (and record why) only for inherently ill-conditioned quantities.

---

## Phase 0 — Oracle harness

### Task 0.1: Octave can run the reference toolbox

**Files:**
- Create: `test/oracle/README.md`
- Create: `test/oracle/octave/_setup.m`
- Create: `test/oracle/octave/smoke_var.m`

**Interfaces:**
- Produces: `_setup.m` adds `bvartools/` and `cmintools/` to the Octave path (used by every later Octave script via `source` / `run`). `smoke_var.m` writes `test/oracle/_out/smoke_var.csv`.

- [ ] **Step 1: Write `_setup.m`** — path bootstrap (edit `REF` if the reference moves):

```matlab
% test/oracle/octave/_setup.m — add reference toolbox to path
REF = '/Users/chung/Downloads/BVAR_-master-2';
addpath(fullfile(REF, 'bvartools'));
addpath(fullfile(REF, 'cmintools'));
pkg load statistics   % Octave: Wishart/quantile helpers used by the toolbox
```

- [ ] **Step 2: Write `smoke_var.m`** — minimal end-to-end run proving the toolbox loads and estimates:

```matlab
% test/oracle/octave/smoke_var.m
source(fullfile(fileparts(mfilename('fullpath')), '_setup.m'));
rand('seed', 0); randn('seed', 0);
T = 200; n = 2; p = 2;
y = cumsum(randn(T, n));            % nonstationary-ish smoke data
opt.K = 1;                          % single draw: exercises the path, not inference
[BVAR] = bvar_(y, p, opt);
outdir = fullfile(fileparts(mfilename('fullpath')), '..', '_out');
if ~exist(outdir, 'dir'); mkdir(outdir); end
csvwrite(fullfile(outdir, 'smoke_var.csv'), BVAR.Phi_draws(:,:,1));
disp('smoke_var OK');
```

- [ ] **Step 3: Run it; confirm it loads and writes output**

Run: `octave --no-gui --eval "run('test/oracle/octave/smoke_var.m')"`
Expected: prints `smoke_var OK`, file `test/oracle/_out/smoke_var.csv` exists and is non-empty. If `bvar_` errors on signature, read `bvartools/bvar_.m` header and adjust the options struct — record the real signature in `README.md`.

- [ ] **Step 4: Write `README.md`** documenting: reference path, how to run an Octave script, the verified `bvar_` option-struct fields, and where outputs land (`test/oracle/_out/`, gitignored).

- [ ] **Step 5: Commit**

```bash
echo "test/oracle/_out/" >> .gitignore
git add test/oracle/octave/_setup.m test/oracle/octave/smoke_var.m test/oracle/README.md .gitignore
git commit -m "test(oracle): octave harness loads BVAR_ reference and round-trips a VAR"
```

### Task 0.2: Shared data fixtures

**Files:**
- Create: `test/oracle/octave/make_fixtures.m`
- Create: `test/oracle/fixtures.jl`

**Interfaces:**
- Produces: deterministic CSVs in `test/oracle/_data/` (`synthetic_var.csv`, plus a copy of a real example series). `fixtures.jl` exposes `load_fixture(name) -> Matrix{Float64}` reading the same CSVs into Julia. Both stacks consume identical bytes.

- [ ] **Step 1: Write `make_fixtures.m`** — generate one synthetic stationary VAR(2) with known coefficients and save the data (NOT just a draw — fixed seed, written to CSV):

```matlab
% test/oracle/octave/make_fixtures.m
source(fullfile(fileparts(mfilename('fullpath')), '_setup.m'));
randn('seed', 42);
T = 300; n = 3; p = 2;
A1 = [0.5 0.1 0.0; 0.0 0.4 0.1; 0.1 0.0 0.3];
A2 = [-0.1 0.0 0.0; 0.0 -0.1 0.0; 0.0 0.0 -0.1];
c  = [0.2; 0.0; -0.1];
Sig = [1.0 0.3 0.1; 0.3 1.0 0.2; 0.1 0.2 1.0];
L = chol(Sig, 'lower');
y = zeros(T, n); y(1:2, :) = randn(2, n);
for t = 3:T
  y(t, :) = (c + A1*y(t-1,:)' + A2*y(t-2,:)' + L*randn(n,1))';
end
datadir = fullfile(fileparts(mfilename('fullpath')), '..', '_data');
if ~exist(datadir, 'dir'); mkdir(datadir); end
csvwrite(fullfile(datadir, 'synthetic_var.csv'), y);
disp('fixtures OK');
```

- [ ] **Step 2: Run it**

Run: `octave --no-gui --eval "run('test/oracle/octave/make_fixtures.m')"`
Expected: prints `fixtures OK`; `test/oracle/_data/synthetic_var.csv` is 300×3.

- [ ] **Step 3: Write `fixtures.jl`** — Julia-side loader (no DelimitedFiles surprises; the package already depends on it):

```julia
# test/oracle/fixtures.jl
using DelimitedFiles
const ORACLE_DATA = joinpath(@__DIR__, "_data")
load_fixture(name::AbstractString) = readdlm(joinpath(ORACLE_DATA, name * ".csv"), ',', Float64)
```

- [ ] **Step 4: Verify Julia reads the same matrix**

Run: `julia --project=. -e 'include("test/oracle/fixtures.jl"); y = load_fixture("synthetic_var"); println(size(y)); println(round.(y[1:2,:], digits=4))'`
Expected: prints `(300, 3)` and two rows of numbers matching the CSV head.

- [ ] **Step 5: Commit**

```bash
echo "test/oracle/_data/" >> .gitignore
git add test/oracle/octave/make_fixtures.m test/oracle/fixtures.jl .gitignore
git commit -m "test(oracle): shared synthetic VAR fixtures for Octave/Julia cross-checks"
```

### Task 0.3: Julia comparison driver

**Files:**
- Create: `test/oracle/compare.jl`

**Interfaces:**
- Consumes: `fixtures.jl::load_fixture`.
- Produces: `read_ref(name) -> Array` (reads a CSV dumped by an Octave script from `_out/`); `compare(label, ours, theirs; rtol, atol) -> NamedTuple(pass, maxabs, maxrel)` that prints a one-line verdict and returns the metrics. This is the assertion primitive every Tier-1 task reuses.

- [ ] **Step 1: Write `compare.jl`**

```julia
# test/oracle/compare.jl
using DelimitedFiles
include(joinpath(@__DIR__, "fixtures.jl"))
const ORACLE_OUT = joinpath(@__DIR__, "_out")
read_ref(name::AbstractString) = readdlm(joinpath(ORACLE_OUT, name * ".csv"), ',', Float64)

function compare(label, ours::AbstractArray, theirs::AbstractArray; rtol=1e-6, atol=1e-8)
    size(ours) == size(theirs) || error("$label: size mismatch $(size(ours)) vs $(size(theirs))")
    d = abs.(ours .- theirs)
    maxabs = maximum(d)
    denom = max.(abs.(theirs), 1e-12)
    maxrel = maximum(d ./ denom)
    pass = all(isapprox.(ours, theirs; rtol=rtol, atol=atol))
    println(rpad(label, 36), pass ? "PASS" : "FAIL",
            "  maxabs=", round(maxabs, sigdigits=3), "  maxrel=", round(maxrel, sigdigits=3))
    return (pass=pass, maxabs=maxabs, maxrel=maxrel)
end
```

- [ ] **Step 2: Self-test the driver** (compare a matrix to itself + to a perturbed copy)

Run: `julia --project=. -e 'include("test/oracle/compare.jl"); a=[1.0 2;3 4]; @assert compare("self",a,a).pass; @assert !compare("perturb",a,a.+1e-3).pass; println("compare.jl OK")'`
Expected: two printed verdict lines (PASS then FAIL) and `compare.jl OK`.

- [ ] **Step 3: Commit**

```bash
git add test/oracle/compare.jl
git commit -m "test(oracle): Julia comparison driver (read_ref + tolerance compare)"
```

### Task 0.4: Initialize the findings report

**Files:**
- Create: `docs/audit/empirical-methods-audit.md`

- [ ] **Step 1: Write the report skeleton** with: title, date, reference, methodology summary, a **Findings** table header (columns: ID, Module, file:line, Severity, Status, Summary), a **Verified-correct ledger** table header (columns: Module, Routine, Method, Evidence), and a **Severity legend** (Critical = wrong results silently / crash on valid input; High = wrong results in common config; Medium = wrong in edge case / misleading SE; Low = cosmetic/doc/perf-correctness).

- [ ] **Step 2: Commit**

```bash
git add docs/audit/empirical-methods-audit.md
git commit -m "docs(audit): initialize empirical-methods findings report"
```

---

## Phase 1 — Tier-1 audit (numerical + code review)

**Per-task protocol (applies to every Task 1.x):**
1. Read our source for the routine; write the implemented formula in a scratch note.
2. Read the reference `.m` file(s); write its formula.
3. If deterministic: add an Octave dump script (`test/oracle/octave/ref_<name>.m`) + a Julia check block appended to `test/oracle/checks_<module>.jl` using `compare(...)`. Run both; record metrics.
4. If stochastic: verify the deterministic algebra/moments by code + property tests (no RNG-draw equality).
5. Classify each discrepancy as **bug** or **convention**. Confirm bugs by re-reading source.
6. Append verified findings to the report's Findings table; append checked-and-correct items to the ledger. Commit the harness scripts + report update.

Each Task 1.x deliverable = (oracle scripts if applicable) + report rows + ledger rows + a commit. No code under `src/` is modified in Phase 1.

### Task 1.1: VAR reduced form
**Files:** `src/var/estimation.jl`, `src/var/types.jl`; reference `bvartools/{rfvar3,ols_reg,YXB_,lagX}.m`.
**Checks:** OLS coefficient matrix equals reference `B` after aligning the const/lag ordering convention; residual covariance denominator — confirm whether `dof_adj = max(T_eff - k, T_eff)` divides by `T_eff` (ML) or `T_eff - k`, and whether reported standard errors use the intended one (lead from spec); companion-matrix layout & eigenvalue stationarity check; AIC/BIC/HQIC formulas and the covariance they use. Numerical compare on `synthetic_var` fixture.
- [ ] Run protocol steps 1–6. Commit: `test(oracle): VAR reduced-form cross-check + audit findings`.

### Task 1.2: BVAR priors (Minnesota / dummy observations)
**Files:** `src/bvar/priors.jl`, `src/bvar/types.jl`; reference `bvartools/{varprior,rfvar3}.m`.
**Checks:** dummy-observation block construction — AR block (`σ_i·lag^{-decay}/τ` placement), sum-of-coefficients block (`μ`), co-persistence/dummy-initial block (`λ`), covariance block (`ω`); confirm where `λ`/`μ` enter (reference applies them in `rfvar3`, not `varprior`); tightness/decay parameterization (is smaller τ tighter, matching reference?). Numerical compare of the stacked `(Y_dummy, X_dummy)` against reference dummies (after ordering alignment).
- [ ] Run protocol. Commit: `test(oracle): Minnesota dummy-obs prior cross-check + findings`.

### Task 1.3: BVAR estimation (posterior moments, samplers, marginal likelihood)
**Files:** `src/bvar/estimation.jl`, `src/bvar/utils.jl`; reference `bvartools/{bvar_,rand_inverse_wishart,mniw_log_dnsty,matrictint}.m`.
**Checks:** NIW posterior moments `B_post, V_post, S_post, ν_post` vs reference `posterior.{PhiHat,XXi,S,df}` on identical data + prior (deterministic — exact compare); IW draw scaling (does `rand_inverse_wishart` parameterize `IW(df, S)` the same as ours, incl. Cholesky orientation?); matrix-normal draw `B = B_post + L_V Z L_Σ'` orientation; marginal likelihood vs `matrictint`/`mniw_log_dnsty` (deterministic — exact compare); stability rejection (`non_explosive_`) behavior. Sampler: property tests (posterior mean → OLS as prior loosens; IW draw empirical mean ≈ `S/(ν-n-1)`).
- [ ] Run protocol. Commit: `test(oracle): BVAR posterior-moment + marginal-likelihood cross-check + findings`.

### Task 1.4: Identification — Cholesky / long-run IRF, FEVD, historical decomposition
**Files:** `src/core/identification.jl`, `src/core/irf.jl`, `src/core/fevd.jl`, `src/core/hd.jl`; reference `bvartools/{iresponse,iresponse_longrun,fevd,fevd_,histdecomp,histdecomposition}.m`.
**Checks:** given identical `(Φ, Σ)`, Cholesky impact matrix orientation and IRF recursion match (exact compare of IRF array after ordering alignment); long-run/BQ `(I-A_∞)^{-1}` construction and first-element sign normalization; FEVD accumulation and normalization to shares summing to 1; historical decomposition components (deterministic + stochastic parts, structural-shock extraction `η = A^{-1}Ω^{-1}u`) sum back to the data. Numerical compare driven by passing the *same* `(Φ, Σ)` to both stacks (dump from Julia → feed Octave, or vice-versa) to isolate identification from estimation.
- [ ] Run protocol. Commit: `test(oracle): IRF/FEVD/histdecomp cross-check + findings`.

### Task 1.5: Forecasts (unconditional + conditional)
**Files:** `src/core/*forecast*` (locate), `src/var`/`src/bvar` forecast paths; reference `bvartools/{forecasts,cforecasts,cforecasts2}.m`.
**Checks:** dynamic recursion uses predicted values as new lags; no-shock vs with-shock paths; conditional forecast solves the constrained-shock system (SVD particular + null-space) and the conditioned variables actually hit the path. Numerical compare of the no-shock forecast (deterministic) on shared `(Φ,Σ)`; conditional-path satisfaction as a property test.
- [ ] Run protocol. Commit: `test(oracle): forecast + conditional-forecast cross-check + findings`.

### Task 1.6: Identification — sign / proxy-IV / zeros-signs / Uhlig penalty
**Files:** `src/core/identification.jl`, `src/core/uhlig.jl`, `src/nongaussian/*` (proxy if there); reference `bvartools/{iresponse_sign,iresponse_proxy,iresponse_zeros_signs,iresponse_sign_narrative}.m`.
**Checks (mostly code + algebra, RNG differs):** random orthogonal `Q` is Haar (QR with sign-corrected `R` diagonal); proxy-IV impact algebra (`b21/b11`, `b11²` formula) matches `iresponse_proxy`; zero+sign restriction parsing & null-space projection; Uhlig penalty function form and per-variable standardization; restriction-string semantics `y(var,horizon,shock)` match. Property tests: accepted draws satisfy the stated signs; proxy F-stat sane on a constructed instrument.
- [ ] Run protocol (stochastic variant). Commit: `test(oracle): sign/proxy/zeros-signs identification audit + findings`.

### Task 1.7: Local projections
**Files:** `src/lp/*`; reference `bvartools/{blp_ml,directmethods}.m` + `examples/.../example_5_LP.m`, `example_7_LP.m`.
**Checks:** per-horizon regression design (lags, controls), horizon-aware Newey–West bandwidth floor (`bw ≥ h+1`) and weights `(L+1-j)/(L+1)`; Bayesian LP (`blp_ml`) prior/ML if we implement it. Numerical compare of horizon-0/1 LP coefficients vs `directmethods` on shared data.
- [ ] Run protocol. Commit: `test(oracle): local-projection cross-check + findings`.

### Task 1.8: Factor models + factor-number selection
**Files:** `src/factor/{static,dynamic,structural,generalized}.jl`, `src/factor/kalman.jl`; reference `bvartools/{pc_T,baing,numbfactor,bdfm_,standard,demean}.m`.
**Checks:** PCA factor/loading normalization (`F = U√T`, `Λ = X'F/T` vs ours `Λ=V√λ, F=XV`) — equivalent up to normalization? `baing` IC penalty `CT(k)` per criterion and `DEMEAN` handling; sign/rotation indeterminacy handled in comparison (compare subspace / `|correlation|`, not raw signs). Numerical compare of `baing`-selected k and the recovered factor subspace.
- [ ] Run protocol. Commit: `test(oracle): factor model + Bai-Ng cross-check + findings`.

### Task 1.9: FAVAR
**Files:** `src/favar/estimation.jl`, `src/favar/types.jl`; reference `bvartools/rescaleFAVAR.m` + `examples/.../example_5_favar.m`, `example_3_favar.m`.
**Checks:** BBE three-step (extract factors → purge slow-moving from observed → VAR on `[F̃, Y]`); loading rescale/normalization; panel-wide IRF projection via loadings. Mostly code review + a constructed-data property test (known loadings recovered up to rotation).
- [ ] Run protocol. Commit: `test(oracle): FAVAR audit + findings`.

### Task 1.10: Nowcast / mixed-frequency DFM
**Files:** `src/nowcast/{dfm,kalman_missing,types}.jl`, `src/core/kalman.jl`; reference `bvartools/{nowcast_bvar,kf_dk,kfilternan,m2q,p2p}.m`.
**Checks:** Kalman filter/smoother with missing observations (row-dropping of `C`,`R`; skip-update when all missing) matches `kfilternan`/`kf_dk` (Durbin–Koopman) recursions; Mariano–Murasawa `[1 2 3 2 1]` quarterly aggregation weights; EM M-step updates. Numerical compare of filtered state on a small missing-data example vs `kfilternan`.
- [ ] Run protocol. Commit: `test(oracle): nowcast DFM / Kalman-missing cross-check + findings`.

### Task 1.11: Panel VAR (GMM)
**Files:** `src/pvar/*`; reference `examples/.../example_6_panels.m`, `example_8_panels.m` + any pooled-VAR routine.
**Checks (code review, reference is example-level):** Arellano–Bond FD-GMM moment conditions & instrument lag structure (min lag 2); Blundell–Bond system-GMM level instruments; forward orthogonal deviations; one/two-step weighting + Windmeijer correction; instrument collapsing. Property tests on a simulated dynamic panel (known persistence recovered, AB1 < AR2 sanity).
- [ ] Run protocol (code review). Commit: `audit(pvar): panel-VAR GMM review + findings`.

### Task 1.12: Non-Gaussian / heteroskedastic identification
**Files:** `src/nongaussian/{ica,ml,heteroskedastic,tests,shared}.jl`; reference `bvartools/{iresponse_heterosked,iresponse_sign_hmoments,hmoments2matrix,fourthmom,thirdmom,skewness_,kurtosis_}.m`, `bvar_opt_heterosked.m`.
**Checks:** higher-moment matrices (3rd/4th) construction; heteroskedasticity identification (regime variance ratios) vs `iresponse_heterosked`; ICA objective correctness (FastICA/JADE/SOBI) — code review against canonical algorithms; ML identification likelihoods. Property tests: recover a known mixing matrix from non-Gaussian simulated shocks (up to permutation/sign/scale).
- [ ] Run protocol (code review + property). Commit: `audit(nongaussian): identification review + findings`.

### Task 1.13: Filters
**Files:** `src/filters/{hp,baxter_king,christiano_fitzgerald,hamilton}.jl`; reference `bvartools/{Hpfilter,bkfilter,cffilter,hamfilter,one_sided_hpfilter_serial}.m`.
**Checks:** HP `(I+λD'D)τ=y` second-difference matrix & boundary rows, default λ by frequency; BK weights `B_j=[sin(ω_H j)-sin(ω_L j)]/(πj)` + zero-sum adjustment + lost endpoints; CF filter (random-walk vs stationary variant); Hamilton regression `y_{t+h}` on `[1,y_t,…,y_{t-p+1}]`, lost `h+p-1` obs. Numerical compare each filter's trend/cycle on `synthetic_var` column 1.
- [ ] Run protocol. Commit: `test(oracle): filters (HP/BK/CF/Hamilton) cross-check + findings`.

---

## Phase 2 — Tier-2 self-check (code review vs canonical formulas)

**Per-task protocol:** read source, write implemented formula, compare to canonical textbook/source formula, check d.o.f./scaling/sign/edge cases, cross-reference existing unit tests, classify & verify findings, append to report + ledger, commit. No `src/` edits in Phase 2.

- [ ] **Task 2.1 — teststat** (`src/teststat/*`): unit-root (ADF, PP, KPSS, DF-GLS, NG-Perron, ZA, LM, 2-break), cointegration (Johansen, Gregory–Hansen), panel (PANIC, CIPS, Moon–Perron), Bai–Perron breaks, normality/portmanteau. Check test statistics, critical-value tables/interpolation, and lag-selection. Commit: `audit(teststat): findings`.
- [ ] **Task 2.2 — reg / preg** (`src/reg/*`, `src/preg/*`): OLS, IV/2SLS, logit/probit/ordered/multinomial, margins, robust/cluster covariance; panel (FE/RE/within/between, clustered SE). Check covariance estimators, d.o.f., marginal-effects deltas. Commit: `audit(reg): findings`.
- [ ] **Task 2.3 — did** (`src/did/*`): the 5 estimators, event-study LP, LP-DiD, CS base-period logic, IPW/doubly-robust. Check against Callaway–Sant'Anna / `lpdid` semantics (project memory references mpdta/ddcg verification). Commit: `audit(did): findings`.
- [ ] **Task 2.4 — vecm** (`src/vecm/*`): Johansen ML, rank restriction, α/β normalization, IRF/short-run. Commit: `audit(vecm): findings`.
- [ ] **Task 2.5 — arima** (`src/arima/*`): CSS/ML estimation, differencing, AR/MA stationarity-invertibility enforcement, forecast SE. Commit: `audit(arima): findings`.
- [ ] **Task 2.6 — garch / arch** (`src/garch/*`, `src/arch/*`): likelihood, parameter transforms (log/logistic) & delta-method SE, EGARCH/GJR variants, persistence constraints. Commit: `audit(garch): findings`.
- [ ] **Task 2.7 — gmm** (`src/gmm/*`): moment conditions, weighting matrix iteration, J-test, robust covariance. Commit: `audit(gmm): findings`.
- [ ] **Task 2.8 — dsge seams** (`src/dsge/*`, estimation/likelihood/solver interfaces only): Kalman likelihood wiring, prior transforms, solver→state-space mapping; leverage existing gensys/klein/bk equivalence tests rather than re-deriving. Commit: `audit(dsge): findings`.
- [ ] **Task 2.9 — x13 / spectral** (`src/x13/*`, `src/spectral/*`): spectral density windows/normalization, coherence/phase, X-13 wrapper correctness. Commit: `audit(spectral): findings`.

---

## Phase 3 — Fixes (TDD, severity order)

### Task 3.0: Triage
- [ ] In `docs/audit/empirical-methods-audit.md`, sort the Findings table by severity; for each confirmed finding give a one-line fix approach and the regression-test idea. Commit: `docs(audit): triage and prioritize confirmed findings`.

### Task 3.N (TEMPLATE — instantiate one per confirmed finding, Critical→Low)

> Instantiated only after the audit confirms the finding. One finding = one Task = one commit. If a finding turns out to be a convention difference on closer reading, mark it **Won't-fix (convention)** in the report instead of creating a fix task.

**Files:** the `src/<module>/<file>.jl` named in the finding; test in the module's existing test file under `test/`.

- [ ] **Step 1: Write the failing regression test** that encodes the *correct* expected value — derived from the reference (oracle dump) or the canonical formula, not from our current output. Place it in the relevant `test/.../*.jl`.
- [ ] **Step 2: Run it; confirm it FAILS** against current code.

```bash
MACRO_MULTIPROCESS_TESTS=1 julia --project=. -e 'using Pkg; Pkg.test(test_args=["<group>"])'
```
Expected: the new assertion fails with our wrong value vs the reference value.

- [ ] **Step 3: Apply the minimal fix** in `src/`, matching surrounding style; do not touch unrelated conventions.
- [ ] **Step 4: Run the test; confirm it PASSES.** Then run the affected group to confirm no regressions.
- [ ] **Step 5: Update the report** — set the finding Status to `Fixed (commit <sha>)`.
- [ ] **Step 6: Commit** `fix(<module>): <one-line> [audit <ID>]` (test + src + report together).

### Task 3.Z: Full-suite verification
- [ ] Run the entire suite multi-process; confirm the pre-existing 2 broken tests are the only failures and the new regression tests pass.

```bash
MACRO_MULTIPROCESS_TESTS=1 julia --project=. test/runtests.jl
```
Expected: ~12400+ pass, 2 pre-existing broken, 0 new failures.
- [ ] Final report pass: every finding has Status `Fixed` / `Won't-fix (convention)` / `Open (deferred)` with rationale. Commit: `docs(audit): finalize empirical-methods audit report`.

---

## Self-Review

**Spec coverage:** Every Tier-1 module in the spec has a Task 1.x; every Tier-2 module has a Task 2.x; hybrid methodology (numerical for overlap, code review elsewhere) is encoded in the per-task protocols; oracle harness, report, and severity-ranked TDD fixes are all present. The three spec "leads to verify" map to Tasks 1.1 (d.o.f.), 1.1/1.2 (ordering convention), 1.2 (dummy-obs). ✓

**Placeholder scan:** Phase 0 steps contain runnable code and exact commands. Audit tasks specify concrete check-lists, exact files, and reference `.m` counterparts rather than "audit the module." Phase 3 is an explicit TDD template because findings are not knowable pre-audit — this is intentional, not a placeholder. ✓

**Type/name consistency:** `load_fixture`, `read_ref`, `compare` defined in Tasks 0.2–0.3 and reused by name throughout Phase 1. Output dirs `_data`/`_out` gitignored consistently. ✓
