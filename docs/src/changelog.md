# [Changelog / What's New](@id changelog)

Release highlights for MacroEconometricModels.jl, newest first. This is a highlights-level summary of
each release, not a commit-by-commit log. Versions marked with correctness fixes changed numerical
output, not just documentation.

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
