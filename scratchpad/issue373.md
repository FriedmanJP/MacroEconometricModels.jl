Sequenced remediation backlog for the four internal audits (reliability & overhaul, test-suite speedup, documentation, report()-quality), decomposed into 273 atomic tasks.

**How to use:** solve issues in ascending task order — `[T001]` → `[T273]` (equivalently ascending issue numbers `#100` → `#372`; task `TNNN` = issue `#(99+NNN)`). Each issue is self-contained (context, file:line anchors, formulas, step-by-step instructions, acceptance criteria) and declares its `Depends on:` tasks. Consolidated duplicates (performance program P-1…P-10, QA item Q-3, cross-cutting findings) are folded into owning tasks with explicit notes, so completing the sequence covers every audit finding.

**Labels:** `severity:*` (critical/high/medium/low), `audit:*` (source report), `area:*` (module), plus type labels (`bug`, `enhancement`, `documentation`, `performance`, `test-suite`, `ci`, `qa`, `merge`, `chore`, `econometrics`).


### Stage 1 — Branch merges & CI/docs infrastructure gates (T001–T012)

- [x] #100 — Merge branch `fix` (empirical-audit fixes) into `dev`
- [x] #101 — Merge branch `io` (v0.6.0 Input-Output module) into `dev`
- [x] #102 — CI: run coverage instrumentation only on the ubuntu job that uploads it
- [x] #103 — CI: kill thread oversubscription (JULIA_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1)
- [x] #104 — CI: skip the test workflow for docs-only changes (paths-ignore)
- [x] #105 — CI: set MACRO_FAST_TESTS=1 on windows/macos jobs only (ubuntu stays full-fidelity)
- [x] #106 — CI (decision): move the coverage job to a newer Julia via a 4th matrix cell
- [x] #107 — Docs: replace committed draft=true with a DOCS_DRAFT env switch
- [x] #108 — Docs/CI: shrink make.jl warnonly so example/docs-block/xref failures fail the build
- [x] #109 — Docs: rewrite verify_examples.jl (per-group modules, keep #hide code, per-block eval, show() capture, timeout)
- [x] #110 — Docs/CI: wire verify_examples.jl --all into the Documentation workflow
- [x] #111 — Docs/CI: build (not deploy) docs on push to dev

### Stage 2 — Reliability Phase 0: confirmed criticals, small fixes (T013–T023)

- [x] #112 — Fix LP-IV Sargan J statistic inflated by a factor of T
- [x] #113 — Fix probit robust/cluster SEs: use the score residual, not the raw residual
- [x] #114 — Include f_lead in the steady-state constant C_sol for linear DSGE solvers
- [x] #115 — Introduce a _respec helper; pass linear=spec.linear at all 6 spec-rebuild sites
- [x] #116 — Add missing [compat] entries (LinearAlgebra, Random, Downloads, DelimitedFiles, ZipFile, XLSX)
- [x] #117 — Parenthesize the stepwise-selection guard (&&-precedence bug) in arima/selection.jl
- [x] #118 — Remove committed .cov debris from test/dsge/ and gitignore *.cov
- [x] #119 — Fix perturbation control-variable IRFs double-counting the state channel
- [x] #120 — Fix collocation solver ignoring its quadrature nodes (it solves the certainty-equivalent model)
- [x] #121 — Fix ARIMA/AR exact-MLE forecast bias: c estimated as process mean but forecast as intercept
- [x] #122 — Discard RWMH burn-in for aggregate DSGE (the burnin kwarg is currently a no-op)

### Stage 3 — Test-runner restructure (T024–T028)

- [x] #123 — runtests.jl: split the DSGE group into DSGE Core / DSGE Bayesian & HD / HA-DSGE
- [x] #124 — runtests.jl: replace spawn-all with a concurrency-capped, longest-first work queue
- [x] #125 — runtests.jl: add per-file FILETIME timing instrumentation in child processes
- [x] #126 — runtests.jl: child-process hygiene (Sys.BINDIR julia, --startup-file=no, check-bounds decision)
- [x] #127 — runtests.jl: rebalance groups and merge IO Analysis into a light group

### Stage 4 — Bayesian DSGE estimation validity (T029–T051)

- [x] #128 — Fix particle-filter likelihood estimator under adaptive resampling (all 5 filter variants)
- [x] #129 — Auxiliary particle filter: implement Pitt–Shephard second-stage weights or delete
- [x] #130 — Replace RWMH "marginal likelihood" (currently max log posterior) with Geweke modified harmonic mean
- [x] #131 — Fix SMC marginal-likelihood increment for non-uniform incoming weights
- [x] #132 — Resample SMC particles at termination so stored draws are unweighted
- [x] #133 — Compute adaptive-tempering ESS on actual cumulative weights, not incremental-only
- [x] #134 — Redesign SMC² mutation kernel as standard PMMH (three compounding invalidities)
- [x] #135 — Fix SMC² inner-particle (N_x) adaptation trigger and add the Chopin exchange step
- [x] #136 — Accept theta0 as Dict/NamedTuple and validate length (stop silent alphabetical reordering)
- [x] #137 — Freeze RWMH proposal adaptation after burn-in and window the acceptance-rate signal
- [x] #138 — Make IRF-matching GMM inference valid (target covariance Omega, sandwich SEs, correct J)
- [x] #139 — Kalman initialization: exact diffuse for unit roots; stop P0=10I fallback; rethrow non-stability errors
- [x] #140 — Narrow likelihood-closure exception handling; count and report failed/skipped draws
- [x] #141 — Default measurement error H=1e-4·I → zero plus a stochastic-singularity check (aggregate + HA)
- [x] #142 — Standardize Bayesian DSGE data input on T×n and error instead of orientation-guessing
- [x] #143 — Posterior-mean solve failure: fall back to highest-posterior draw instead of erroring after sampling
- [x] #144 — posterior_summary: use Statistics.quantile (weighted once T033 lands)
- [x] #145 — Add a max_stages guard plus a minimum Δφ to the SMC tempering loops
- [x] #146 — Replace threadid()-indexed workspace pools with chunked/checkout workspaces
- [x] #147 — Thread the SMC² initial-likelihood loop
- [x] #148 — Raise SSJ estimation default T_horizon from 50 to ≥300 and document the truncation tradeoff
- [x] #149 — Mark HA posterior-mean fallback solutions (currently silently solves the original spec)
- [x] #150 — Replace the prior-draw midpoint fallback with an informative error

### Stage 5 — Core numerics / HAC / identification (T052–T063)

- [x] #151 — Fix automatic HAC bandwidth: kernel-consistent Andrews plug-in constants, correct mislabel, revisit the truncation cap
- [x] #152 — Quadratic-spectral HAC kernel: stop truncating like Bartlett; sum lags to n−1
- [x] #153 — Implement Andrews–Monahan VAR(1) prewhitening of the moment vector (or document + warn)
- [x] #154 — System HAC/White/Driscoll–Kraay: fill cross-equation covariance blocks or document diagonal-only validity
- [x] #155 — robust_inv: add a condition-number check (warn/pinv on near-singular) and narrow the caught exceptions
- [x] #156 — safe_cholesky: make jitter scale-relative (∝ tr(A)/n) and return the applied jitter
- [x] #157 — Core Kalman measurement update: use Joseph form + symmetrization + triangular solves
- [x] #158 — Narrow bare `catch; continue` in identification loops (arias/uhlig); retain last error for diagnostics
- [x] #159 — Bootstrap labeled "Kilian (1998)" lacks the bias-correction step — implement it or relabel
- [x] #160 — FEVD: check impact-matrix orthogonality; warn for ICA/heteroskedasticity identifications
- [x] #161 — Core numerics low-severity batch (C-12…C-18): Lyapunov, logdet, generate_Q sign, skew/kurtosis, triangular solves, weighted-quantile eps, data validation
- [x] #162 — Core structural-analysis performance batch: IRF buffers, PSD projection shortcut, precomputed kernel weights, triangular solves

### Stage 6 — Micro / panel / DiD / GMM / volatility / teststat inference (T064–T090)

- [x] #163 — Implement Rambachan–Roth honest DiD properly (current "robust CI" is a naive linear bound)
- [x] #164 — Callaway–Sant'Anna: influence-function or multiplier-bootstrap SEs; honor the cluster kwarg
- [x] #165 — BJS imputation estimator: Prop.-6 influence-function variance with unit clustering
- [x] #166 — Sun–Abraham: period-specific cohort-share weights + joint covariance
- [x] #167 — DiD overall-ATT aggregation: covariance-based SEs across horizons (5 estimators)
- [x] #168 — Pre-trend test: full-covariance joint Wald; store event-study covariance in DIDResult
- [x] #169 — Bacon decomposition: include Goodman-Bacon timing-window terms (match bacondecomp)
- [x] #170 — Ordered logit/probit: analytic observed-information Hessian ("robust" currently cancels to OPG)
- [x] #171 — IV first-stage F: partial out exogenous regressors; q = # excluded instruments; add KP/CD + Stock–Yogo
- [x] #172 — SMM optimal weighting from per-observation influence contributions (propagates to DSGE-SMM)
- [x] #173 — GARCH-family QMLE: Bollerslev–Wooldridge sandwich SEs by default
- [x] #174 — WLS covariance: weighted residuals/design in HC meat; fix the :ols path
- [x] #175 — IV overidentification: Hansen J under robust/cluster weighting (Sargan currently always homoskedastic)
- [x] #176 — Multinomial pseudo-R²: constants-only null log-likelihood (Stata-comparable)
- [x] #177 — Unit-root p-values: MacKinnon response-surface; drop the normal-tail extrapolation
- [x] #178 — ADF lag selection: fix the estimation sample at max-lag (Ng–Perron 1995)
- [x] #179 — PVAR GMM: zero-pad unbalanced instrument blocks instead of truncating to min columns
- [x] #180 — PVAR one-step GMM: include the Arellano–Bond H band matrix (xtabond2 parity)
- [x] #181 — PVAR Σ for IRFs from within/FD residuals (currently inflated by fixed effects)
- [x] #182 — xtreg :ab/:bb: full covariance matrix + AR(1)/AR(2)/Hansen diagnostics
- [x] #183 — PVAR GMM: too-many-instruments warning (Roodman 2009) when n_inst > N
- [x] #184 — Hausman test: report negative statistic + warning (or generalized inverse with reduced df)
- [x] #185 — Bai–Perron sup-F: q-indexed critical values (currently q=1 table used for any q)
- [x] #186 — Panel RE/CRE logit: Newton/Optim optimizer + adaptive Gauss–Hermite quadrature
- [x] #187 — FE conditional-logit: DP recursion in log space (overflow risk)
- [x] #188 — Micro/panel/teststat low-severity batch (M-27, M-28, M-29, M-31, M-32, M-33, M-34, M-35, M-36)
- [x] #189 — Micro/panel performance batch (group-index maps, ICA kernel caching, vectorized panels, SV layout, GARCH Hessian cache)

### Stage 7 — Reduced-form time series (T091–T111)

- [ ] #190 — Estimate real state-dependent LP transition parameters (current NLLS objective is constant in (gamma, c))
- [ ] #191 — Fix smooth-LP cross-validation to score held-out unrestricted LP estimates (currently compares fit to itself, always picks lambda_min)
- [ ] #192 — Bayesian FAVAR: draw real NIW (B, Sigma) and a genuine Carter-Kohn FFBS for the factors
- [ ] #193 — Fix ARIMA(d>=1) prediction intervals: psi-weights of the nondifferenced operator, ci = forecast +/- z*se
- [ ] #194 — Nowcast news: joint release weights via lagged smoother covariances (B = Lambda*Cov(F,I)*Var(I)^-1)
- [ ] #195 — GDFM common component: project on leading dynamic eigenvectors of the smoothed spectrum (not the raw rank-1 periodogram)
- [ ] #196 — DFM EM M-step: fix lag-one smoothed covariance (transpose, per-lag, include t=p+1)
- [ ] #197 — Factor forecast ci_method=:none: stop aliasing one zero array as lower/upper/se into the in-place unstandardizer
- [ ] #198 — FAVAR two-step panel IRFs: include the direct Lambda^y loading and match loadings to the rotated factors
- [ ] #199 — State-LP: use predetermined transition F(z_{t-1}) instead of contemporaneous F(z_t) (Auerbach-Gorodnichenko design)
- [ ] #200 — State-LP: fix inverted expansion/recession regime labels
- [ ] #201 — Doubly-robust LP: apply a HAC/Newey-West floor to the MA(h)-correlated influence functions
- [ ] #202 — Smooth-LP: implement the one-step Barnichon-Brownlees estimator, or document the two-step and report the full covariance
- [ ] #203 — Nowcast bridge: include the incomplete current quarter (cld instead of integer-divide by 3)
- [ ] #204 — Nowcast BVAR: implement a real Kalman smoother for the ragged edge, or re-document (currently interpolation + deterministic projection)
- [ ] #205 — Verify (against X-13ARIMA-SEATS source) and fix if confirmed: X-13 exact-ML likelihood double-counts the determinant term
- [ ] #206 — HP filter transfer-function diagnostic: use sin^4, not sin^2 (half-power period is ~40q at lambda=1600, not ~251q)
- [ ] #207 — ARIMA CSS: use a consistent effective sample across candidate orders for AIC/BIC comparability
- [ ] #208 — VAR forecast bands: add parameter uncertainty (bootstrap-B or analytic MSE), not just residual resampling
- [ ] #209 — Reduced-form low-severity batch (R-22, R-24, R-25, R-26, R-27, R-29, R-30, R-31, R-32, R-33)
- [ ] #210 — Reduced-form performance batch (BVAR containers, hoisted companion/eigvals, forecast ring buffers, EM/Gibbs prealloc, LP factorization reuse, ARIMA rank-1 Kalman)

### Stage 8 — DSGE solver stack (T112–T126)

- [ ] #211 — Route all five raw `gensys(ld…)` callers through the companion-QZ core
- [ ] #212 — Exact AD derivative tensors for order-2/3 perturbation (replace finite differences)
- [ ] #213 — Accept the UC solvent only if it is stable (add eigenvalue check, else companion-QZ fallback)
- [ ] #214 — Steady-state NonlinearSolve path: verify ‖F(u)‖∞ and error/flag instead of warn-and-continue
- [ ] #215 — Matrix-free GMRES Sylvester: add residual check + warn/error and expose tolerance
- [ ] #216 — perfect_foresight projected-Newton failure path: compute merit before the throw (fix UndefVarError)
- [ ] #217 — irf(::ProjectionSolution): one deterministic path or a true KPP GIRF (currently 500 identical paths)
- [ ] #218 — Smolyak: derive the index set and sparse grid from the same |α|₁ ≤ μ+d rule
- [ ] #219 — OccBin: prefer explicit user-specified alternative regimes; warn when the argmax-|Jacobian| heuristic is not decisive
- [ ] #220 — Replace O(n⁶) Kronecker Lyapunov with doubling (_dlyap_doubling) and guard near-unit-root state bounds
- [ ] #221 — vfi_solver: rename/redocument as PFI time iteration, or implement genuine value iteration
- [ ] #222 — Expose `div`; place `divhat` adaptively when roots cluster near 1
- [ ] #223 — Reconcile n_expect (forward equations) with Π columns (forward variables)
- [ ] #224 — DSGE solver low-severity batch (S-17, S-19, S-20, S-21)
- [ ] #225 — Solver performance batch: order-3 matrix-free Kronecker, collocation QR + Jacobian reuse, stored impact, single Jacobian pass

### Stage 9 — HA-DSGE aggregate-dynamics rebuild (T127–T143)

- [ ] #226 — Implement the real fake-news SSJ Jacobian (current default is neither fake-news nor a valid brute-force Jacobian)
- [ ] #227 — Carry the Ho–Kalman observation map (C, D=h0) in HADSGESolution and apply it in irf/fevd/simulate
- [ ] #228 — HA Bayesian estimation: build the observation matrix Z from the reduction's C rows; error on unmatched observables
- [ ] #229 — Krusell–Smith: households must forecast prices through the PLM (outer fixed point currently vacuous)
- [ ] #230 — Reiter Aiyagari path: restore GE price feedback (dLambda/dr * dr/dK + firm-FOC closure)
- [ ] #231 — Normalize example income processes: e = exp(z)/E[exp(z)] (currently mean-zero, often negative income)
- [ ] #232 — Two-asset nested EGM: proper deposit choice (value-based, state-dependent, recomputed post-convergence)
- [ ] #233 — distribution_irf/inequality_irf: store and use the real reduction basis U_k (currently always zeros)
- [ ] #234 — Remove the silent eigenvalue rescaling of G1 (diagnose and warn instead)
- [ ] #235 — EGM Euler inversion through budget_fn/net-income hook (currently hardcodes the KS budget)
- [ ] #236 — Reiter: read rho_z, alpha, delta from spec params (currently hardcoded 0.95/0.36/0.025)
- [ ] #237 — Blanchard OLG: fix debt-service double-count in C and the k-row of M (wrong steady state when b != 0)
- [ ] #238 — KS inner EGM: add warm-start argument and convergence check (cold start gives ~13% error at max_iter=50)
- [ ] #239 — @dsge HA parser: parse the CRRA coefficient, expose budget/model selection (huggett currently unreachable)
- [ ] #240 — HA/CT/OLG low-severity batch (H-16, H-17, H-18, H-20)
- [ ] #241 — Validate the rebuilt HA block against the Python sequence-jacobian toolkit (KS + one-asset HANK)
- [ ] #242 — HA/CT performance batch: sparse KFE & Reiter, Arnoldi stationary distribution, EGM inner-loop cleanups, threading

### Stage 10 — Reliability engineering & QA program (T144–T160)

- [ ] #243 — RNG-everywhere policy: rng kwarg on every drawing function, pre-seeded threaded loops, CI lint
- [ ] #244 — Eliminate silent numerical failures: no fabricated results; n_failed/n_effective on every MC result; narrow _suppress_warnings
- [ ] #245 — Introduce an exception hierarchy (MacroModelError: ConvergenceError, IdentificationError, SingularSystemError)
- [ ] #246 — Consolidate the five Kalman filters into one parametric core kernel (+ steady-state gain, triangular solves, univariate updates)
- [ ] #247 — Central tolerance constants (DEFAULT_ABSTOL/RELTOL from eps(T)), exposed as kwargs
- [ ] #248 — Remove runtime eval of constraint bounds (small recursive numeric evaluator)
- [ ] #249 — Replace global mutable display-backend Ref with ScopedValue or explicit argument
- [ ] #250 — IO downloaders: HTTPS everywhere + sha256 checksum registry verified after download
- [ ] #251 — Re-enable Aqua ambiguities + deps_compat (+ persistent_tasks), exclude specific false positives
- [ ] #252 — Widen CI matrix: current-stable '1', allow-failure nightly, Optim@1-pinned job
- [ ] #253 — Add PrecompileTools workload over top-10 entry points (after SnoopCompile measurement)
- [ ] #254 — Engineering low-severity batch (G-14…G-20)
- [ ] #255 — Evaluate sub-package split along stable seams (Core / IO / DSGE / HA)
- [ ] #256 — Add a seeded weekly statistical regression suite (estimator recovery, CI coverage, SE-vs-MC dispersion, PF/Kalman agreement, order-1 perturbation ≡ gensys)
- [ ] #257 — Extend the oracle harness: Dynare, R (did/HonestDiD/forecast/vars/boottest), Stata worked examples, Python sequence-jacobian, statsmodels
- [ ] #258 — Tighten every isfinite-only assertion guarding a statistical quantity to a value assertion
- [ ] #259 — Create the fixed performance benchmark suite (medium DSGE estimation, 200-var FAVAR, 50×50×7 two-asset HA, 10k-draw BVAR)

### Stage 11 — report() display quality (T161–T176)

- [x] #260 — Disable table cropping in non-TTY output so significance/CI columns are never silently dropped
- [x] #261 — Suppress empty column-label rows; print legends/notes as wrapped text instead of tables
- [x] #262 — Rewrite _fmt as an @sprintf fixed-decimal formatter (aligned columns, -0.0 normalization, scientific fallback)
- [x] #263 — _coef_table dust/reference-row guard (print — and no stars) plus degenerate-fit warning banner
- [x] #264 — Deduplicate _select_horizons endpoint so display no longer emits doubled horizon rows
- [x] #265 — Establish shared display conventions (label dictionary, one _fmt_pvalue, CV order, Yes/No, AIC-scale labels, IRF-horizon note)
- [x] #266 — Add missing report() dispatches and wrap bare NamedTuple/Vector returns in display types
- [x] #267 — Content standard: every report() must end with an estimates table (LP family, DSGE solutions, nowcast headline, factor forecast, Uhlig/Arias)
- [x] #268 — Unify on one display dialect: port DSGESpec/CT/OLG/panel-summary to the shared table infra and add io::IO plumbing package-wide
- [x] #269 — Fix CI/quantile/SE label mismatches (BVAR 3%/97%, ARIMA SE column, volatility CI note, LP-FEVD units, HonestDiD dead column, VECM title)
- [x] #270 — Fix johansen_test cointegration-rank off-by-one (rank = r+1 on rejection) and dedupe the two rank selectors
- [x] #271 — Investigate panel between-R² printing exactly 1.0 on unrelated datasets
- [x] #272 — Flag suspicious GLP hyperparameters in nowcast BVAR report (lambda ≈ 148 suggests unconverged optimizer)
- [x] #273 — Display one-off fixes batch (HD shock labels, |Loading| column, ARIMA order grid, normality names, BN note, HA log10 label, news scaling, KS progress)
- [x] #274 — Make LaTeX backend output compilable-quality (booktabs rules, math-mode/escaped headers, skip legend tables); fix HTML header-split artifacts
- [x] #275 — Add display regression tests and a goldens suite (no `omitted`, no -0.0, no raw e-notation, stars present)

### Stage 12 — Documentation content & architecture (T177–T205)

- [ ] #276 — Register the entire HA/CT/OLG/X-13/smoother export surface in the API reference
- [ ] #277 — Document the 17 fully-undocumented exports (HA/CT/OLG types, distribution_irf, PanelTestResult, utilities)
- [ ] #278 — Fix configuration drift between docrule.md / CLAUDE.md and the actual docs build
- [ ] #279 — arima.md: stop narrating CPI inflation data as "industrial production growth"
- [ ] #280 — volatility.md: stop narrating INDPRO as "S&P 500 returns"; remove invented half-life figure
- [ ] #281 — index.md: fix u_t dimension, surface X-13 from Home, refresh @contents, fix stale counts
- [ ] #282 — favar.md: rewrite Complete Example interpretation to match the actual code
- [ ] #283 — Fix wrong keyword defaults in did.md (leads=0) and event_study.md (lags=4)
- [ ] #284 — Standardize "13 statistical identification methods" across three pages (+ project docs)
- [ ] #285 — Expand plotting.md from a 97-line stub into a real reference of all 53 dispatches
- [ ] #286 — Univariate + data pages [MED] sweep (spectral report() claim, band_power comment, dangling citation, produced-value interpretations)
- [ ] #287 — Multivariate pages [MED] sweep (lp/lp-fevd/bayesian defaults, manual setup exposure, favar static block, ICA links, BQ DOI, factormodels T<N)
- [ ] #288 — Cross-section + panel pages [MED] sweep (last-expression renders, vif display, panel_reg DGP, ordered ending, missing cross-links)
- [ ] #289 — DSGE pages [MED] sweep (dynare-count, solve-method table, hub child links, static recipes, OLG/CT maturation, ha println, hd @docs placement)
- [ ] #290 — Identification + innovation-accounting + nowcast pages [MED] sweep (ia_fevd mislabel, back-link architecture, duplicated table, setup reductions, missing bib entries)
- [ ] #291 — Tests + API pages [MED] sweep (api_types tree, autodocs omits spectral.jl, load_example table, tests.md citations, malformed span, println pages, ref-style split)
- [ ] #292 — Docs [LOW] polish sweep — univariate / multivariate / panel pages
- [ ] #293 — Docs [LOW] polish sweep — DSGE / identification+IA+nowcast / tests+API pages
- [ ] #294 — Retrofit hub/overview pages to the docrule Page-Types skeleton
- [ ] #295 — Restructure docs navigation to the target information architecture
- [ ] #296 — Write gmm.md — the only whole-module narrative gap
- [ ] #297 — Write getting_started.md (FRED data → VAR → IRF → plot in 10 minutes)
- [ ] #298 — Write method_guide.md (decision tables: question → estimator → page)
- [ ] #299 — Extract notation.md + bibliography.md; unify citation style site-wide
- [ ] #300 — Add Changelog / What's New page + How-to-Cite page
- [ ] #301 — Split the API reference into per-domain pages; @eval the type hierarchy; then drop :missing_docs from warnonly
- [ ] #302 — Backfill narrative mentions for the 105 API-only exports
- [ ] #303 — Decide on filename renames (manual→var, bayesian→bvar, nongaussian→id_overview) — bundle with a release or skip
- [ ] #304 — Add docs/lint_docs.jl mechanical docrule checks to CI

### Stage 13 — Test-suite compute cuts (T206–T232)

- [ ] #305 — Cut the HA Bayesian estimation testset (~172s → ~30-45s) via shared steady state + single :ssj solve + n_draws 20→6
- [ ] #306 — Cut the Den Haan (2010) accuracy testset (~71s → ~20-25s): shorten simulation horizons and grid
- [ ] #307 — Cut the Huggett Krusell-Smith testset (~43s → ~12-15s): T_sim 800→300, T_burn 200→75, n_a 150→100
- [ ] #308 — Share one Huggett steady state across SS/SSJ/Reiter/KS testsets; n_a 400→200; SSJ T_horizon 100→50
- [ ] #309 — Extract JuMP/Ipopt/PATH extension tests into one dedicated Extensions test group
- [ ] #310 — test_dsge.jl: add shared AR(1) spec/solution fixtures for the ~150 re-parse sites (optional two-file split)
- [ ] #311 — test_dsge.jl GMM Higher-Order Moments: drop :analytical_gmm solve_order 3→2 (order-3 re-solve per optimizer iteration)
- [ ] #312 — Cap optimizer iterations in DSGE estimation-loop testsets (IRF-matching, analytical GMM, SMM)
- [ ] #313 — test_bayesian_dsge.jl: halve n_smc/n_particles on SMC² dispatch/DA/mechanics tests; keep E2E anchor full-size
- [ ] #314 — Opportunistic simulation-length cuts in test_dsge.jl (500k→50k, 100k→20-30k, GIRF 500→100)
- [ ] #315 — X-13 test dedupe: pass explicit model=spec where automdl isn't under test; fit the seed-42 series once
- [ ] #316 — teststat grid cuts: shared adf_2break grid + T 200→80/lags=1; lm_unitroot, gregory_hansen, Bai-Perron, KPSS
- [ ] #317 — Volatility test dedupe: one fit per GARCH-family block, n 1000→400 where qualitative, explicit n_sim=500 forecasts
- [ ] #318 — SMM test caps: max_iter 1000→200; drop one near-duplicate two-step recovery; reuse one result for show/refs
- [ ] #319 — BVAR hidden hyperparameter grids: pass explicit hyper=/grid_size=3 at ~9 non-hyperopt call sites
- [ ] #320 — FAVAR test sharing: build the 80-iter Gibbs chain once for 5 accessor testsets; explicit small chain at hidden site; trim bootstrap reps
- [ ] #321 — DiD bootstrap cuts: did_multiplegt n_boot 50/30→10 except one SE-validation block; load mpdta once
- [ ] #322 — Nowcast test sharing: one T=60 DFM fit across News/Dispatch/Display (~11 refits→1); keep the two real EM fits
- [ ] #323 — GDFM test caps: N 100→40, share spectra in the kernel loop, halve asymptotic loops
- [ ] #324 — uhlig/arias test dedup: merge 3 near-identical multi-start runs; gate the ungated double-run; reduce arias n_rotations
- [ ] #325 — test_pvar_nongaussian_coverage.jl: FAST-gate the ungated bootstrap-identification blocks
- [ ] #326 — test_examples.jl: reduce bootstrap IRF reps 200→50 (and note sign-restriction / smooth-LP CV sites)
- [ ] #327 — Windmeijer F-07 Monte Carlo: reduce 120→40 reps unconditionally
- [ ] #328 — Add MACRO_FAST_TESTS gating to every file touched in T206-T228 (compounding P0.4 dividends)
- [ ] #329 — Decision: Windows CI sharding via a MACRO_TEST_GROUPS_ONLY env filter (only if still needed after T003-T028)
- [ ] #330 — Decision: threaded-runner hybrid for the Ubuntu coverage job (loads/JITs the package once)
- [ ] #331 — Decision: nightly/weekly full-fidelity all-OS workflow (if FAST-on-win/mac feels too lean)

### Stage 14 — Feature catalogue (T233–T273)

- [ ] #332 — Feature: posterior mode finding + Laplace marginal likelihood + inverse-Hessian RWMH proposal
- [ ] #333 — Feature: MCMC convergence diagnostics (rank-normalized R-hat, ESS, Geweke, trace/ACF accessors)
- [ ] #334 — Feature: bridge sampling for marginal likelihood from MCMC output
- [ ] #335 — Feature: identification diagnostics — Iskrev rank test, KPS learning rate, prior/posterior overlap warnings
- [ ] #336 — Feature: prior predictive & posterior predictive checks as first-class API
- [ ] #337 — Feature: parameter transformations for samplers (log/logit reparameterization with Jacobians)
- [ ] #338 — Feature: Dynare prior-convention shims (dynare_prior(:inv_gamma, mean, std)-style constructors)
- [ ] #339 — Feature: estimation on observables with trends (difference/demean transforms + deterministic trends in the observation equation)
- [ ] #340 — Feature: conditional forecasts / scenario analysis for VAR/BVAR (Waggoner–Zha)
- [ ] #341 — Feature: SARIMA — seasonal (P,D,Q,s) support
- [ ] #342 — Feature: wild cluster bootstrap (boottest-style) for few-cluster inference
- [ ] #343 — Feature: Anderson–Rubin weak-IV-robust confidence intervals (complete the KP/CD/Stock–Yogo suite)
- [ ] #344 — Feature: weak-IV-robust LP-IV inference (Montiel Olea–Pflueger, Anderson–Rubin)
- [ ] #345 — Feature: reproducibility manifest (seed, threads, versions, git SHA) + reproduce(result)
- [ ] #346 — Feature: Tables.jl integration for all result types (DataFrame(result), CSV export, R/Python handoff)
- [ ] #347 — Feature: versioned result serialization (save_model/load_model)
- [ ] #348 — Feature: structured logging (replace println + blanket NullLogger)
- [ ] #349 — Feature: SV-BVAR (Primiceri / Cogley–Sargent) and TVP-VAR reusing src/sv machinery
- [ ] #350 — Feature: mixed-frequency VAR (Schorfheide–Song)
- [ ] #351 — Feature: full GLP (2015) hierarchical hyperparameter optimization as the BVAR default
- [ ] #352 — Feature: SSJ DAG/block model composition + second-order SSJ (Auclert et al. 2023)
- [ ] #353 — Feature: DCEGM for discrete-continuous choice (Iskhakov et al. 2017)
- [ ] #354 — Feature: true life-cycle OLG with age-dependent EGM
- [ ] #355 — Feature: endogenous labor supply in HA blocks (GHH / separable preferences)
- [ ] #356 — Feature: Winberry (2018) parametric distribution dynamics
- [ ] #357 — Feature: adaptive/anisotropic Smolyak + adaptive distribution grids
- [ ] #358 — Feature: continuous-time two-asset general equilibrium + MIT transitions + kinked adjustment costs
- [ ] #359 — Feature: extend Den Haan (2010) accuracy testing beyond Huggett (Krusell–Smith + HANK)
- [ ] #360 — Feature: Conley spatial standard errors
- [ ] #361 — Feature: quantile regression
- [ ] #362 — Feature: regression discontinuity design (rdrobust-style)
- [ ] #363 — Feature: Student-t / GED GARCH likelihoods
- [ ] #364 — Feature: generalized FEVD (Pesaran–Shin)
- [ ] #365 — Feature: Bartels–Stewart / generalized-Schur Sylvester & Lyapunov solvers (Kamenik 2005) for n>50
- [ ] #366 — Feature: true Sims (2002) existence/uniqueness rank test wired into solve(:gensys)
- [ ] #367 — Feature: determinacy-region mapping over parameter grids
- [ ] #368 — Feature: pruned state-space object with correct control mapping + Andreasen et al. (2018) simulation-free order-3 moments
- [ ] #369 — Feature: sparse/structured linear algebra for medium-large DSGE models (sparse Klein)
- [ ] #370 — Feature: wild/block bootstrap options + Kilian bias-corrected bands for LP/VAR IRFs
- [ ] #371 — Feature: reghdfe-style high-dimensional fixed-effect absorption (alternating projections)
- [ ] #372 — Feature: ESS reporting for Arias et al. importance-sampling weights




