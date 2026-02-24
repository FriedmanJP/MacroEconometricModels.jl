# DSGE Display & Fixes Design

**Date:** 2026-02-25
**Issues:** #57, #59, #60

## #57 — DSGE LaTeX/Text/HTML Equation Display

### Architecture

New file `src/dsge/display.jl` with two recursive expression converters and a rewritten `show(io, spec::DSGESpec)`.

### Expression Converters

`_expr_to_latex(ex, endog, exog, params)` and `_expr_to_text(ex, endog, exog, params)` — recursive Julia Expr → string converters.

Mappings:
- `var[t]` → `var_t` / `var_t`
- `var[t-1]` → `var_{t-1}` / `var_{t-1}`
- `var[t+1]` → `\mathbb{E}_t[var_{t+1}]` / `E_t[var_{t+1}]`
- Greek params → LaTeX symbols via dict (`β→\beta`, `σ→\sigma`, `κ→\kappa`, `φ→\phi`, `ρ→\rho`, `π→\pi`, etc.)
- Subscripted params `φ_π` → `\phi_\pi` or `\phi_{\pi y}`
- `*` → implicit multiplication or `\,`
- `/` → `\frac{}{}`
- `^` → superscript
- `+`, `-` → infix

### Output Sections (all three backends)

1. **Header**: model dimensions
2. **Parameters**: name = value (text) or `tabular` (LaTeX)
3. **Equations**: numbered, rendered via converters; LaTeX `align` environment
4. **Steady State**: `x̄ = value` (text) or `\bar{x} = value` (LaTeX); computed via `ss_fn` if available
5. **Calibration Table**: Symbol | Value; text uses `_pretty_table`, LaTeX uses `tabular`

### Output Formats

- `:text` — Unicode terminal output (default)
- `:latex` — `\begin{align}...\end{align}` + `\begin{tabular}...\end{tabular}`
- `:html` — MathJax `$$...$$` blocks for Jupyter

### Files

- Create: `src/dsge/display.jl`
- Modify: `src/MacroEconometricModels.jl` — include after `dsge/types.jl`
- Modify: `src/dsge/types.jl` — remove old `show(io, ::DSGESpec)` (moved to display.jl)

## #59 — Test Noise Suppression

One-line fix: wrap `test/dsge/test_dsge.jl` line 2066 `report(sol)` with `@test redirect_stdout(devnull) do; report(sol) end === nothing`.

### Files

- Modify: `test/dsge/test_dsge.jl:2066`

## #60 — Analytical.jl Reference Fix

Update `src/dsge/analytical.jl` line 23:
- Keep Hamilton (1994) Ch. 10 §10.2 as main reference (add section precision)
- Add Fernández-Villaverde, Rubio-Ramírez, and Schorfheide (2016) as DSGE-specific companion

### Files

- Modify: `src/dsge/analytical.jl:22-23`
- Possibly modify: `src/summary_refs.jl` (add FVS 2016 to `_REFERENCES` if not present)
