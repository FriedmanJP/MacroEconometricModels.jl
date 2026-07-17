# [Changelog / What's New](@id changelog)

Release highlights for MacroEconometricModels.jl, newest first. This is a highlights-level summary of
each release, not a commit-by-commit log. Versions marked with correctness fixes changed numerical
output, not just documentation.

---

## v0.7.0

Major feature release. Breaking (0.6 → 0.7): a large new exported API surface, and `JuMP` + `Ipopt`
are now full dependencies.

- **EViews-parity series** (EV-01–EV-40): nine new modules — MIDAS (`src/midas`), ARDL/NARDL + panel
  ARDL (`src/ardl`), single-equation & panel cointegrating regression (`src/cointreg`), SUR/3SLS
  (`src/system`), multivariate GARCH CCC/DCC/BEKK (`src/mgarch`), nonlinear time series
  (threshold/SETAR, STAR/LSTR, Markov-switching; `src/nonlinear`), nonparametric density/regression
  (`src/nonparametric`), Kalman-MLE state-space + TVP regression (`src/statespace`), and forecast
  evaluation & combination (`src/fceval`) — plus GARCH extensions (IGARCH/CGARCH/APARCH/FIGARCH/
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

## v0.6.6

Display quality: publication-grade `report()`/`table()` output.

- Golden-file regression harness and display invariants for the bespoke VAR/VECM/DSGE reports.
- Goldens made robust to cross-version numeric drift.

## v0.6.5

Heterogeneous-agent DSGE rebuild plus reliability and QA hardening (issue #380).

## v0.6.4

Solver and filter correctness fixes (issue #378). Numerical output changed for affected estimators.

## v0.6.3

README and plotting-asset refresh (issue #377).

## v0.6.2

Stage-4 Bayesian DSGE estimation validity (issues #128–#150, #376). Correctness fixes to the
posterior samplers, not documentation only.

## v0.6.1

Phase-0 correctness criticals and test-runner restructure (issue #375). Numerical output changed for
the affected methods.

## v0.6.0

Input-Output analysis module (issue #374): the [Input-Output Analysis](@ref) container, Leontief/Ghosh
models, multipliers/linkages/SDA/extraction, environmental extensions, Baqaee-Farhi (2019), and the
pymrio-style MRIO downloaders.

## v0.5.1

Continuous-time and life-cycle heterogeneous-agent methods: [Continuous Time](@ref dsge_continuous)
(HJB / Kolmogorov-Forward), [Overlapping Generations](@ref dsge_olg) (Blanchard 1985 perpetual youth),
and the Huggett (1993) pure-exchange example.

## v0.5.0

Heterogeneous-agent DSGE and higher-order analysis.

- [Heterogeneous Agent DSGE](@ref dsge_ha): SSJ (Auclert et al. 2021), Reiter (2009), and
  Krusell-Smith (1998) solvers; EGM/VFI individual solvers; Young (2010) histogram.
- Dynare replication suite (22 models); order ≥ 2 unconditional FEVD.
- Pre-linearized models via `model(linear=true)`; [Linear Solution Methods](@ref dsge_linear) rewrite
  on a companion-QZ core.
- [X-13ARIMA-SEATS](@ref x13_page) coverage.

---

See [How to Cite](@ref citation) for how to reference a specific version in your work.
