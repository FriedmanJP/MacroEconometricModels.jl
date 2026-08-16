# [Changelog / What's New](@id changelog)

Release highlights for MacroEconometricModels.jl, newest first. This is a highlights-level summary of
each release, not a commit-by-commit log. Versions marked with correctness fixes changed numerical
output, not just documentation.

---

## v0.8.3

Patch release: optional **Google Colab precompiled assets** (`#610`) so a new
Colab runtime does not pay a multi-minute cold precompile of
MacroEconometricModels and its heavy dependency graph. A tag-only CI workflow
builds a linux-x86_64 depot tarball for Colab's native Julia pin (1.12.6 as of
the 2026.07 runtime) and attempts a PackageCompiler sysimage; the depot is the
success criterion. In-package `PrecompileTools` and General-registry `Pkg.add`
are unchanged. This tag also carries the genuine Bellman VFI work
(`#615`/`#616`/`#617`) stacked under `#621`.

This is a patch on the `0.8` series. Downstream `[compat]` of
`MacroEconometricModels = "0.8"` resolves here; no bound change is required.

**New**

- **Colab depot** (`environments/colab/`): pinned Project/Manifest,
  `colab_precompile.jl` workload, setup notebook, and
  `.github/workflows/colab-precompiled-env.yml` publishing
  `mem-colab-depot-vX.Y.Z-julia1.12.6-linux-x86_64.tar.zst` on tag or
  `workflow_dispatch`.
- **Bellman VFI**: `vfi_solver` is real value-function iteration (Howard
  policy evaluation of ``V``); HA production accepts `hh_solver=:vfi`.

---

## v0.8.2

Patch release: restore CI on Julia 1.10. `XLSX` 0.12.2 fails to precompile against
`XML` 0.4.5 on LTS (`XLSXError: No sheetData node found in worksheet` inside the
XLSX precompile workload). Compat is tightened to `XLSX = "0.10, 0.11"` until an
XLSX release loads on Julia 1.10 with the current XML 0.4 series. The Excel
parser extension is unchanged; `using XLSX` should resolve 0.11.x.

This is a patch on the `0.8` series. Downstream `[compat]` of
`MacroEconometricModels = "0.8"` resolves here; no bound change is required.

---

## v0.8.1

Patch release: the **Input-Output overhaul** (`#611`) — a Baqaee--Farhi production-network
layer on top of the classical column Leontief toolkit, plus MRIO trade accounting,
general SDA, RAS/GRAS, and labeled ICIO/WIOD parsers. Legacy `baqaee_farhi(io)` remains
the frozen intermediate-only Hessian; the new standard-form path is
`production_network` → `bf_equilibrium` / `baqaee_farhi(net)`.

This is a patch on the `0.8` series. Downstream `[compat]` of
`MacroEconometricModels = "0.8"` resolves here; no bound change is required.

**New**

- **Baqaee--Farhi network** (`ProductionNetwork`, `production_network`): row-oriented
  nested CES (single/two-nest, multi-factor) calibrated from column `IOData` once.
- **Exact nonlinear equilibrium** (`bf_equilibrium` → `BFEquilibrium`): Newton
  unit-cost fixed point, factor-market clearing, GDP numéraire.
- **Local second-order** (`baqaee_farhi(net)`, `BFLocal`, `BFElasticities`,
  `bf_quadratic`, `bf_shock_curve`): 2019 multi-factor ``H_A`` with incidence; shock
  curve for large-``d\log A``.
- **Wedges / markups** (`bf_wedge_decomp`, cost- vs revenue-based Domar): B\&F 2020
  Theorem 1 technology / allocative-efficiency decomposition.
- **Misallocation** (`bf_misallocation`, `bf_wedge_quadratic` → `BFMisallocation`):
  Proposition 5 ``H_\mu`` and the Harberger triangle at the efficient point;
  `:observed` is the Prop 5 Var/Cov object, not the solver Hessian.
- **Classical IO**: Leontief `price_model`, Dietzenbacher--Lahr extraction variants,
  `impact` scenario API (Type I/II + mixed), `network_stats`.
- **MRIO**: `aggregate`, `region_block`, HIY vertical specialization,
  KWW 2014 `export_decomposition` (DVA/RDV/FVA/PDC), regional footprints.
- **SDA / balancing**: general n-determinant Dietzenbacher--Los SDA (incl. emission
  SDA) and `ras` / `gras` / `balance`.
- **Parsers**: `parse_icio` / `parse_wiod` labeled MRIO recipes.

**Also**

- Windows CI: threaded single-process runner, FAST HA-DSGE/DSGE, HA-DSGE OpenBLAS
  threads, skip artifact cache upload.

---

## v0.8.0

Feature release: the **policy-counterfactual module** (`src/counterfactual/`, series
`CF-01`--`CF-24` = `#381`--`#404`) — sufficient-statistics policy analysis after
McKay & Wolf (2023), Barnichon & Mesters (2023), and Caravello, McKay & Wolf (2025).
The module is a general implementation of the *methods*, validated by a ten-identity
theorem-level oracle suite in linear laboratories (Proposition-1 exact recovery on RANK
and HA menus, rule immateriality, the optimal-policy circle, OPP ``\equiv`` optimal
projection, the Sims--Zha vs. news-based Lucas contrast, an end-to-end historical
recursion oracle, Wold-rotation invariance, model-averaging degeneracy, and
second-moment consistency) — not by replicating published tables.

**New**

- **Containers and templates**: `PolicyCausalEffects` (square news menus / thin
  empirical subsets, with draws), `PolicyRule`/`PolicyLoss` with
  peg/target/Taylor/NGDP/AIT/smoothing builders; one shared weighted-projection kernel
  with exact, least-squares, and min-norm regimes and always-surfaced
  implementation-error diagnostics.
- **Inputs from everything the package estimates**: `policy_causal_effects` adapters
  for VAR/BVAR/sign-set/LP IRFs, `baseline_path`, `wold_representation`,
  `policy_forecast` gap containers (BVAR `forecast(...; store_draws=true)` retention;
  SEP-style external route), `stacked_irf_target` + `ctw_covariance`,
  `policy_news_matrix` DSGE news menus (shared news pipeline, linear state growth),
  public `sequence_jacobian` + HA `policy_causal_effects`, and
  `cognitive_discounting`/`sticky_expectations`/`behavioral` operators.
- **Engines**: `policy_counterfactual` (McKay--Wolf), `optimal_policy`/`optimal_rule`,
  `counterfactual_moments` (with business-cycle frequency bands), `opp`/`estimate_opp`
  (reversed 60/75/90% band polarity), `constrained_opp` (ZLB/pledges via SLSQP with
  KKT diagnostics), `opp_sequence`/`opp_sensitivity`/`robust_weights` (exact three-part
  time-consistency decomposition), `irf_match`/`posterior_model_probs`/`model_average`
  (CMW model bank), `counterfactual_forecast`/`counterfactual_history` (forecast-revision
  recursion), `spanning_diagnostic`/`forecast_sufficiency`.
- **Data**: `load_example(:mp_shocks)` — quarterly US monetary panel (1960Q1--2019Q4)
  with the Romer--Romer/Wieland--Yang, Gertler--Karadi, Aruoba--Drechsel, and Ben
  Zeev--Khan shock series (NaN outside published samples).
- **UX**: `report()`/`refs()` for all nine result types (honesty verdicts, band-polarity
  notes, model-probability tables), seven `plot_result` dispatches (path overlays with
  automatic implementation-error panels, OPP fan charts, moment and spanning views), and
  a four-page "Policy Counterfactuals" documentation section.

---

## v0.7.3

Patch release resolving the issues filed by the 2026-08 documentation audit (`#516`--`#595`)
plus its follow-ups (`#598`--`#602`, `#605`--`#607`): a full documentation-site rewrite, two
new capabilities, and correctness fixes across nearly every module. **Many reported numbers
change** --- lag selection, Bayesian FAVAR, panel confidence bands, LP-DiD, RE/CRE panel
logit, the PP/PANIC/Pedroni/break-test battery, and order-1 DSGE analytical moments all
produce different (now correct) output.

**New**

- **Litterman prior for nowcasting BVARs** (`#602`): `nowcast_bvar(...; prior=:litterman)`
  fixes ``\Sigma`` diagonal so the equations separate, enabling a genuine cross-lag
  tightness knob `theta_cross` that the conjugate dummy-observation prior cannot express.
  Posterior mean verified against direct GLS and the marginal likelihood against the
  ``T \times T`` Gaussian to machine precision. Log-likelihoods are not comparable across
  priors --- the conjugate prior integrates ``\Sigma`` out.
- **Per-series break diagnostics** (`#606`): `FactorBreakResult` stores
  `series_statistics` and `series_break_dates` for the pooled factor-break tests, and
  `report` lists the top-5 breaking series. The ranking identifies breakers exactly when a
  modest subset breaks; a half-panel break rotates the factor space and contaminates
  stable series' statistics (documented).

**Correctness** (selected highlights of the ~70 behaviour fixes; the issue tracker has the full list)

- **Lag selection** (`#522`): VAR/VECM information criteria use the system parameter count
  (Lütkepohl 2005) and a common estimation sample --- `select_lag_order` no longer returns
  `max_p` mechanically on short samples.
- **Bayesian FAVAR** (`#523`, `#528`): scale-aware priors, and the BBE measurement
  equation now carries the direct ``\Lambda_y`` loading block, removing the
  factor/observable collinearity that produced explosive coefficient draws.
- **Panel bands** (`#524`): FAVAR panel IRF/forecast intervals take quantiles in panel
  space instead of mapping factor-space endpoints through the loadings.
- **Kilian bootstrap** (`#564`): `bias_correct=true` corrects the point IRF, not only the
  bootstrap draws.
- **LP-DiD** (`#540`): IPW reweighting enters the time-demeaning step, so the reweighted
  estimand matches the weighted least-squares target.
- **DiD cohorts** (`#587`, `#598`): `xtset` stores cohort *values* rather than ranks, and
  non-positive cohorts are treated as adoption periods instead of being silently dropped
  --- four of five estimators returned ATT ``= 0`` on shifted event-time panels.
- **RE/CRE panel logit** (`#600`, `#542`): the adaptive Gauss--Hermite likelihood is
  finite everywhere, the inner mode search converges (safeguarded Newton), and the
  covariance uses the Louis identity --- standard errors recover their information bound.
- **Unit-root and panel tests** (`#576`, `#581`, `#582`, `#584`): the Phillips--Perron
  correction restores a missing factor of ``T``; PANIC pools via the Bai--Ng Fisher
  combination; Moon--Perron projects the standardized panel; ADF regressions solve via QR
  and the Pedroni statistics weight by ``1/\widehat{\text{LRV}}``.
- **Structural break tests** (`#577`, `#583`, `#605`): the LM unit-root, two-break ADF,
  and factor-break family (Breitung--Eickmeier, Chen--Dolado--Gonzalo, Han--Inoue) are
  calibrated against simulated conditional null distributions --- measured sizes now fall
  in ``[0.035, 0.055]`` where several tests previously had size 0.00 or 1.00.
- **Non-Gaussian identification** (`#565`--`#570`): JADE/SOBI rotation-angle fix, a
  genuine normalized Pearson Type IV density for `:pml`, corrected GARCH initialization,
  and a centered bootstrap for restriction tests.
- **DSGE** (`#556`, `#607`): box-constrained perfect foresight solves as a semismooth
  complementarity Newton (the NK ZLB model converges in 2--3 iterations), and order-1
  `analytical_moments` reports contemporaneous control moments --- corr(state, control)
  was ``\rho`` where it should be 1.
- **Nowcasting** (`#571`--`#573`, `#575`, `#588`): full-rank dummy-observation prior
  design, data revisions are no longer mislabeled as news in `nowcast_news`, the MIDAS
  weight Jacobian is clamped consistently with the weights, and `apply_tcode` no longer
  crashes on non-positive columns.

**API and behaviour notes**

- `BayesianFEVD` axes are unified to `(variable, shock, horizon)` (`#527`).
- One-argument `refs(x)` prints to stdout and returns `nothing`; capture text with
  `sprint(io -> refs(io, x))` (`#530`).
- Nowcast-BVAR `theta` is re-purposed as the lag-decay exponent (GLP ``\alpha``) ---
  cross-lag tightness is structurally impossible in the conjugate prior; use
  `prior=:litterman` for a cross-lag knob (`#571`, `#602`).
- The EORA26 downloader throws a clear not-implemented error instead of failing obscurely
  (`#518`).

**Documentation**

- **Full documentation-site rewrite** (PR `#596`): every page rebuilt to a uniform
  skeleton (Quick Start recipes, verified `@example` blocks, keyword and return-value
  tables, topic-scoped bibliographies), a restructured API reference covering every
  export, cross-reference integrity enforced as a hard build error, and all 93 plot
  assets regenerated.
- The reproducibility footer on `report()` output is opt-in (`#521`).

---

## v0.7.2

Patch release closing the last open issues in the `#407`--`#512` range: one new
estimator family, four correctness fixes, and two API gaps. **Two reported accuracy
numbers change** --- see the correctness notes below.

**New**

- **Count-data regression** (`#427`, EV-19): `estimate_poisson` (IRLS on the log link,
  Gourieroux--Monfort--Trognon pseudo-ML sandwich standard errors by default) and
  `estimate_nbreg` (Negative-Binomial-2, fit jointly in ``(\beta, \log\alpha)``), with
  `dispersion_test` (Cameron--Trivedi 1990), `incidence_rate_ratio`, marginal effects,
  and `offset`/`exposure` support. Validated against R's `glm(family=poisson)` and
  `MASS::glm.nb` to 8--10 significant digits. Zero-inflated and hurdle variants are out
  of scope.
- **Markov-switching fitted values and forecasts** (`#510`): `fitted`/`predict` expose
  the regime-probability-weighted conditional mean the estimator already computed
  (`y - fitted(m) == residuals(m)` exactly), with `probs=:filtered` for the real-time
  weighting. `forecast(m, h)` for MS-AR and `forecast(m, X_new)` for switching
  regressions return an exact analytic mean path with simulated mixture bands.
- **Residuals for ordered and multinomial models** (`#507`): `residuals(m; kind=)`
  returns the ``n \times K`` response / Pearson / deviance matrix, and
  `generalized_residuals` the length-``n`` Chesher--Irish score residual for ordered
  models. Note the shape differs from the binary models, which return a vector.

**Correctness**

- **The HA Euler-error statistic was near-zero by construction where the solution was
  worst** (`#508`). It evaluated only at grid nodes, where EGM solves the Euler equation
  exactly, and flat-extrapolated above ``a_{\max}`` so that truncated cells reported
  machine-precision residuals. It is now measured off-node at cell midpoints, with
  out-of-grid cells excluded and counted. **Every HA accuracy number moves by 2.5--3.8
  ``\log_{10}`` units** --- Krusell--Smith from ``-6.04`` to ``-2.25``, one-asset HANK from
  ``-6.06`` to ``-2.28``, Huggett from ``-4.47`` to ``-1.94``. This is a metric-honesty
  change, not a solver regression; `compute_steady_state(spec; euler_points=:nodes)`
  restores the old convention and `ss.euler` carries both.
- **The continuous-time two-asset stationarity check was inverted for the kinked cost**
  (`#509`). In the KMV parameterization ``\chi_1`` multiplies the withdrawal, so a
  *larger* ``\chi_1`` is more stationary; the old test certified divergent calibrations
  (96% of mass on the ceiling) as sound. `ct_two_asset_solve` now also warns when the
  calibration cannot bound illiquid wealth instead of returning a grid artifact with
  `converged = true`.
- **A lone `a1` or `P1` was silently discarded** by `StateSpaceModel` (`#512`), returning
  a different model from the one written --- and the quieter one, differing by three
  orders of magnitude in log-likelihood. It now raises, and the explicit path validates
  dimensions.
- **Panel `report()` printed the raw covariance symbol** (`#407`) --- `cluster` rather than
  `Cluster-robust`. Cosmetic only; the covariance matrix and standard errors were correct.

**Documentation**

- Six state-space entry points documented `init_mode=:diffuse` while defaulting to
  `:kappa` (`#512`); the docstrings now match the code, with a table of what each mode does.
- `docs/src/dsge_ha.md` no longer claims "values below ``-3`` are standard in the
  literature" --- that survived on the node metric and does not survive the corrected one.

---

## v0.7.1

Documentation-only patch. No public API or numerical changes.

- **API reference completed**: added `@docs`/`@autodocs` coverage for the v0.7.0
  EViews-parity and new-module exports --- extended GARCH (IGARCH/CGARCH/APARCH/FIGARCH/
  FIEGARCH, GARCH-MIDAS diagnostics), multivariate GARCH, single-equation & panel
  cointegrating regression, SUR/3SLS, ARDL/NARDL/PMG, penalized/robust/Tobit/Heckman
  regression, MIDAS, ARFIMA, nonlinear & state-space forecast methods, and the
  higher-moment/bubble/distribution test battery --- so every export is registered on a
  reference page (`checkdocs=:exports`).
- Fixed the v0.7.0 documentation examples that were blocking the docs build.

---

## v0.7.0

Major feature release. Breaking (0.6 to 0.7): a large new exported API surface, and `JuMP` + `Ipopt`
are now full dependencies.

- **EViews-parity series** (EV-01--EV-40): nine new modules --- MIDAS (`src/midas`), ARDL/NARDL + panel
  ARDL (`src/ardl`), single-equation & panel cointegrating regression (`src/cointreg`), SUR/3SLS
  (`src/system`), multivariate GARCH CCC/DCC/BEKK (`src/mgarch`), nonlinear time series
  (threshold/SETAR, STAR/LSTR, Markov-switching; `src/nonlinear`), nonparametric density/regression
  (`src/nonparametric`), Kalman-MLE state-space + TVP regression (`src/statespace`), and forecast
  evaluation & combination (`src/fceval`) --- plus GARCH extensions (IGARCH/CGARCH/APARCH/FIGARCH/
  FIEGARCH, GARCH-MIDAS), ARFIMA, penalized/robust/Tobit/Heckman regression, IV k-class, panel
  PCSE/Prais-Winsten, and a large hypothesis-test battery (HEGY, ERS, SADF/GSADF bubbles, BDS,
  variance-ratio, EDF goodness-of-fit, residual/panel cointegration, first-generation panel unit
  root, Dumitrescu-Hurlin, long-run variance).
- **DSGE Bayesian diagnostics**: posterior mode + Laplace/bridge-sampling marginal likelihood, MCMC
  convergence diagnostics (rank-normalized R-hat, bulk/tail ESS, Geweke), Iskrev identification
  tests, prior/posterior predictive checks, sampler parameter transforms, and Dynare prior shims.
- **Reproducibility & serialization**: `ReproManifest`/`reproduce` and versioned
  `save_model`/`load_model` (JLD2 weak-dependency backend).
- **Tables.jl integration** for result types and **structured logging** replacing bare `println`.
- **`JuMP` + `Ipopt` promoted to full dependencies** (GPL-compatible): `solver=:ipopt` works with no
  manual `]add`; `PATHSolver` remains an optional weak dependency.

---

## v0.6.7

Documentation content and architecture, plus test-suite quality. No public API changes.

- New [Getting Started](@ref getting_started_page) tutorial and [Choosing a Method](@ref
  method_guide_page) decision-table router.
- New narrative pages: [GMM & SMM](@ref gmm_page), [Notation](@ref notation),
  [Bibliography](@ref bibliography),
  [How to Cite](@ref citation), and this changelog.
- API reference split into per-domain pages with an auto-generated type hierarchy; every exported
  symbol is now documented on exactly one reference page (`:missing_docs` is a hard build error).
- Corrected stale counts, mislabeled example series, and keyword-table defaults across the corpus.
- Test-suite compute cuts: shared fixtures and shorter simulation/grid/draw settings across the
  DSGE, heterogeneous-agent, volatility, factor, panel, and nowcast suites (with assertions kept
  discriminating), and a dedicated Extensions test group for the optional JuMP/Ipopt/PATH solvers.

---

## v0.6.6

Display quality: publication-grade `report()`/`table()` output.

- Golden-file regression harness and display invariants for the bespoke VAR/VECM/DSGE reports.
- Goldens made robust to cross-version numeric drift.

---

## v0.6.5

Heterogeneous-agent DSGE rebuild plus reliability and QA hardening (issue #380).

---

## v0.6.4

Solver and filter correctness fixes (issue #378). Numerical output changed for affected estimators.

---

## v0.6.3

README and plotting-asset refresh (issue #377).

---

## v0.6.2

Stage-4 Bayesian DSGE estimation validity (issues #128--#150, #376). Correctness fixes to the
posterior samplers, not documentation only.

---

## v0.6.1

Phase-0 correctness criticals and test-runner restructure (issue #375). Numerical output changed for
the affected methods.

---

## v0.6.0

[Input-Output Analysis](@ref io_page) module (issue #374): the [`IOData`](@ref) container,
Leontief/Ghosh models, multipliers/linkages/SDA/extraction, environmental extensions,
Baqaee-Farhi (2019), and the pymrio-style MRIO downloaders.

---

## v0.5.1

Continuous-time and life-cycle heterogeneous-agent methods: [Continuous Time](@ref dsge_continuous)
(HJB / Kolmogorov-Forward), [Overlapping Generations](@ref dsge_olg) (Blanchard 1985 perpetual youth),
and the Huggett (1993) pure-exchange example.

---

## v0.5.0

Heterogeneous-agent DSGE and higher-order analysis.

- [Heterogeneous Agent DSGE](@ref dsge_ha): SSJ (Auclert et al. 2021), Reiter (2009), and
  Krusell-Smith (1998) solvers; EGM/VFI individual solvers; Young (2010) histogram.
- Dynare replication suite (22 models); order ``\geq 2`` unconditional FEVD.
- Pre-linearized models via `model(linear=true)`; [Linear Solution Methods](@ref dsge_linear) rewrite
  on a companion-QZ core.
- [X-13ARIMA-SEATS](@ref x13_page) coverage.

---

See [How to Cite](@ref citation) for how to reference a specific version in your work.
