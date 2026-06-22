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
| _none yet_ | | | | | |

## Verified-correct ledger

Routines checked and found correct (so we know what was actually verified vs. assumed).

| Module | Routine | Method | Evidence |
|--------|---------|--------|----------|
| _none yet_ | | | |

## Notes / convention map

Cross-stack convention differences that are NOT bugs (recorded so they are not re-flagged):

- _to be filled in during Phase 1_
