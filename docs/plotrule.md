# Plot Style & Architecture Guide

This guide defines the design and code-architecture standards for all plotting in MacroEconometricModels.jl — every `plot_result` method, every chart renderer, and every future plot feature. It plays the same role for plots that `docrule.md` plays for documentation pages: **new plot code must follow these rules, and existing plot code is reviewed against them.** The target quality is publication-grade econometrics graphics (FRED / BIS / central-bank working-paper level) delivered as self-contained interactive HTML.

Terminology used throughout:

- **Renderer** — a `_render_*_js` function that emits D3.js code for one chart form (line, area, bar, heatmap, scatter, coefficient plot).
- **Data converter** — a `_*_data_json` helper that turns result-struct arrays into a JSON data string.
- **Plot method** — a `plot_result(::SomeResult; kwargs...)` method that selects data, calls converters and renderers, assembles panels, and returns a `PlotOutput`.
- **Panel** — one chart (`_PanelSpec`); a **figure** is one or more panels composed by `_make_plot`.

---

## Architecture

The plotting module is a strict four-layer stack. Data flows down, never up or sideways:

```
types.jl     PlotOutput, _PanelSpec, theme constants (_PLOT_COLORS, _PLOT_FONT,
             _PLOT_CI_ALPHA), _next_plot_id, the _json serializer
render.jl    CSS, HTML skeleton, shared JS core, ALL chart renderers,
             panel composition (_make_plot), save/display/show
helpers.jl   ALL data converters (_irf_data_json, _fevd_data_json, ...),
             _resolve_var, _series_json
<domain>.jl  plot_result methods only (irf.jl, fevd.jl, hd.jl, filters.jl,
             forecast.jl, models.jl, nowcast.jl, did.jl, reg.jl, spectral.jl, io.jl)
```

**A1 — Renderers live in `render.jl` and nowhere else.** A domain file must never define a `_render_*_js` function. If a domain file needs a chart form that doesn't exist, add the renderer to `render.jl` with full generic parameterization, then call it. (Historical violations to migrate: `_render_vbar_js` in spectral.jl, `_render_scatter_js` in did.jl, `_render_coef_plot_js` in reg.jl, `_render_occbin_panel_js` in models.jl.)

**A2 — Renderers are domain-agnostic.** No domain vocabulary inside a renderer: tooltips, labels, and legends must be built from parameters (`xlabel`, series `name`s), never hardcoded nouns like "Lag", "Period", or "Shock". A renderer that says `'Period '+d.x` in its tooltip will eventually render "Period Agriculture" when reused for a sector heatmap. Tooltip prefix = the `xlabel` parameter (or a dedicated `tip_label` parameter), always.

**A3 — One IIFE per panel; rely on IIFE scoping.** Every renderer wraps its JS in `(function() { ... })();`. Inside that wrapper, use plain identifiers (`data`, `x`, `y`, `svg`). Never suffix identifiers with the panel id (`data_$(id)`) — the IIFE already isolates scope, and suffixing bloats output and obscures diffs.

**A4 — Never re-derive scales outside the renderer.** Post-hoc JS snippets that rebuild the margin object and x/y scales to overlay one extra element (a vertical line, a Q-Q reference line) duplicate renderer internals and silently break when the renderer changes. If a chart needs an overlay, the renderer grows an option:

- Vertical reference lines: `ref_lines_json` entries carry `"axis":"x"` (the scatter renderer already supports this; the line renderer must too).
- Sloped reference lines (Q-Q): a `line_overlays_json` option with `{x1, y1, x2, y2, color, dash}` in *data* coordinates, mapped through the renderer's own scales.

**A5 — Plot methods do selection and assembly, not computation.** A `plot_result` method extracts arrays from the result struct, converts them with helpers, calls renderers, and composes panels. Statistical computation (KDE, histogram binning, z-scores, Lorenz accumulation, Gini) belongs in the owning computational module or a documented `_plot_*` helper directly above the method — never inline in the middle of panel assembly, and never duplicated between plotting and the analysis code that computes the same statistic for `report()`.

**A6 — All data converters live in `helpers.jl`.** If a plot method builds `rows = Vector{Pair{String,String}}[]` inline, that loop should almost always be a named, documented converter in helpers.jl so the next result type with the same shape reuses it.

**A7 — `_json` is the only serializer, and it must be total over what we feed it.**

- Every value type passed to `_json` needs an explicit method (`AbstractString`, `Number`, `Bool`, `Symbol`, `Nothing`, `Missing`, `AbstractVector`, and `Date`/`DateTime` when date axes land). No `_json(::Any)` fallback — a `MethodError` at plot time is better than silently emitting invalid JSON.
- `NaN`/`Inf` serialize to `null` (renderers must treat `null` as a gap via `.defined()`).
- **No duplicate keys in one JSON object.** Building a row and then `push!`-ing a second pair with the same key (the forecast "bridge point" pattern) produces invalid JSON that only works by accident of browser parsing. Overwrite the existing pair instead.
- String escaping must cover `\`, `"`, `\n`, `\r`, `\t`, all control characters < U+0020, and — because output is embedded in a `<script>` block — the sequences `</` (escape as `<\/`) and U+2028/U+2029.

**A8 — Escape everything interpolated into HTML or JS.** User-controlled strings (variable names, shock names, titles, labels) reach three sinks, each with its own rule:

| Sink | Rule |
|---|---|
| HTML text (`<div class="panel-title">$(title)</div>`) | pass through an `_esc_html` helper (`&`, `<`, `>`, `"`) |
| JS string literal (`.text('$(xlabel)')`) | never interpolate raw into quoted JS; emit via `_json(xlabel)` so it arrives as a proper JSON string literal |
| JSON data | already covered by `_json` |

A variable named `GDP's "core"` or `x<y` must render, not break the document.

**A9 — Every renderer ships complete.** A new renderer must provide: margins/responsive width, y-grid, axes with labels, tooltip, legend policy (below), `ref_lines_json` support where meaningful, band support where meaningful, and its own unit test. No "minimal for now" renderers.

**A10 — No positional struct-literal adapters.** Converting one result type to another for plot reuse (e.g. LPDiD → EventStudyLP) must go through a keyword constructor or a dedicated `Base.convert`/`_as_eventstudy(...)` function owned by the struct's module. A positional literal with dummy fields breaks silently when the struct gains a field.

**A11 — IDs and global JS state.** Panel DOM ids come from `_next_plot_id` only. All JS state lives inside the per-panel IIFE except the shared core (`fmt`, `showTip`, `hideTip`, `tooltip`), which must be embed-safe: guard against redefinition (`if (typeof window.__mem_core === 'undefined') { ... }`) and select the tooltip node relative to the plot container, not by a fixed global `#tooltip` id, so that two `PlotOutput`s displayed in the same notebook DOM don't collide. Note the `_plot_counter` `Ref` is *not* atomic; if plots are ever generated from threads, use `Threads.Atomic{Int}`.

**A12 — Self-contained output.** A saved plot must render offline. D3 must be vendored (inlined into the HTML from a copy shipped with the package or an artifact), not loaded from a CDN `<script src>`. If a CDN fallback is kept for size reasons, it requires an SRI `integrity` attribute and a documented offline limitation — but vendoring is the rule. This also protects the Documenter site (iframes) from third-party outages.

---

## API Contract

**C1 — One entry point.** Users plot any result with `plot_result(result; kwargs...)` returning a `PlotOutput`. Exported names are exactly: `plot_result`, `PlotOutput`, `save_plot`, `display_plot`. Everything else is internal (`_`-prefixed).

**C2 — Standard keyword vocabulary.** Use these names with these exact semantics; never invent synonyms:

| Kwarg | Type | Semantics |
|---|---|---|
| `var` | `Union{Int,String,Nothing}=nothing` | select one response variable; `nothing` = all |
| `shock` | `Union{Int,String,Nothing}=nothing` | select one shock; `nothing` = all |
| `vars` | `Union{Vector,Nothing}=nothing` | select multiple variables (data containers) |
| `view` | `Symbol` | switch between named views of one result |
| `ncols` | `Int=0` | grid columns; `0` = auto |
| `title` | `String=""` | figure title; `""` = auto-generated |
| `save_path` | `Union{String,Nothing}=nothing` | also write HTML to this path |
| `history` / `n_history` | vector / `Int` | historical context for forecast fans |

**C3 — Name selection works everywhere names exist.** Any `var`/`shock`/`vars` kwarg accepts **both** `Int` and `String` whenever the result carries names. `Union{Int,Nothing}`-only signatures (current FEVD, LPFEVD, VARForecast, BVARForecast, VECMForecast, LPForecast, FactorForecast) are contract violations to fix. Resolution goes through `_resolve_var`, which must also **bounds-check integer input** and throw a friendly `ArgumentError` (not a downstream `BoundsError`).

**C4 — Every kwarg is live.** A keyword that the body never reads (`stat` in Bayesian FEVD/HD plots, `group_names` in the nowcast heatmap view) is forbidden: either implement it or remove it. Dead kwargs are worse than missing ones — they promise behavior that doesn't happen.

**C5 — `view` dispatch pattern.** Multi-view results (`NowcastResult`, `NowcastNews`, `HASteadyState`) dispatch on `view::Symbol` and must throw `ArgumentError` listing all valid views for anything else. Each view gets a `_plot_<type>_<view>` helper. Kwargs that apply only to some views must be documented per-view, and a kwarg passed to a view that ignores it should be an error, not silence.

**C6 — Docstrings tell the truth.** The docstring describes exactly what is drawn. If it says "vertical line at the posterior mean", "reference period marker", or "horizontal bar chart", the plot must contain a vertical line, a marker, and horizontal bars. Docstring/render mismatches are release blockers because users cite these plots.

**C7 — No silent truncation.** Any cap (first 5 factors, 10 groups, 12 variables, 60 histogram bars, 10 eigenvalues) must be visible in the output — in the panel title (`"Extracted Factors (5 of 12)"`) or a figure note — and documented in the docstring, ideally with a kwarg to raise it. A reader must never believe they are seeing everything when they are not.

**C8 — `save_path` semantics.** When given, write the file *and* return the `PlotOutput`. Never print, never open a browser implicitly (that is `display_plot`'s job).

**C9 — Numbers in titles are formatted, not serialized.** Use `round`/`_fmt` (3–4 significant digits) for values embedded in titles or panel names. Never `_json(x)` into prose — it prints full float precision and renders `NaN` as `null`.

---

## Visual Design Rules

### Form selection

The data's job picks the chart form. The canonical mapping for this package:

| Result | Form | Notes |
|---|---|---|
| IRF (all flavors) | line + CI band + zero line | horizon starts at **0** |
| FEVD | stacked area, 0–100% | horizon starts at **1**; y ticks in % |
| Historical decomposition | diverging stacked bar + actual/reconstructed line panel | zero line mandatory |
| Event study (DiD, LP-DiD) | **point-and-whisker per event time** (coefplot), vertical line at treatment, zero line | a continuous ribbon over-smooths discrete coefficients; keep line+band only as an explicit `style=:ribbon` option |
| Forecast | history line + fan (band) + point path | history context should be available for *every* forecast type, not only ARIMA |
| Filter / decomposition | 2-panel: original+trend, cycle with zero line | |
| ACF/PACF/CCF | thin vertical bars from zero + ±CI lines | lags start at 1 |
| Spectral density | line + band, log₁₀ scale | frequency axis may add period annotations |
| Volatility models | 3-panel: returns, σₜ, standardized residuals | |
| Densities (prior/posterior) | line (KDE) + vertical mean/mode line | |
| Coefficients / marginal effects / multipliers / news impacts | **horizontal** dot-and-whisker or bar when categories are named entities | names read horizontally; vertical bars with rotated or thinned name labels are forbidden |
| Matrices (Leontief, correlation, z-scores) | heatmap **with a color legend** | |
| Two-metric sector/group comparison (Rasmussen) | scatter + quadrant reference lines, direct labels | |
| Distributions over a grid (HA wealth) | area/step density; bin-aggregate when subsampling, never point-sample | mass must be conserved in what is displayed |

If the object is a single headline number (a nowcast, a breakdown value), it belongs in the title or a stat annotation — not a chart of one point.

### Color

- `_PLOT_COLORS` is the single categorical palette. **Assign hues in fixed order and keep color↔entity stable across all panels of a figure** (shock *j* keeps its color in every panel; a filtered replot must not repaint survivors).
- More series than the palette → fold the tail into "Other" or split into small multiples. `mod1` recycling inside `_series_json` is a last-resort guard, not a design; and slicing `_PLOT_COLORS[1:n]` with `n > 20` is an error — always go through `_series_json`'s cycling or pre-check length.
- **Sequential data gets a single-hue ramp** (light→dark). A nonnegative matrix (Leontief inverse) on a red-blue diverging scale is wrong: diverging palettes are reserved for signed data with a meaningful midpoint (z-scores, contributions), and the midpoint must map to the neutral center.
- `#d62728`-class red is the *alert/reference* color (CI bounds, binding regions, treatment lines). Avoid it as an ordinary series color in any figure that also draws red reference elements.
- CI bands use the parent line's color at `_PLOT_CI_ALPHA`. When two bands overlap (HonestDiD original vs robust), they must differ in **both** color and alpha, and both must appear in the legend.
- Never encode meaning by color alone: dashed vs solid (already used for linear/piecewise, trend/original) is the required secondary channel; keep it.
- The palette must be validated for color-vision deficiency once per change (dataviz validator or equivalent); record the result in this file. Pairs like green/red (#2ca02c/#d62728) must never be the only distinction between two adjacent series.

**CVD validation record.** `_PLOT_SERIES` (PLT-13, 2026-07-18): the Tableau-10 categorical set with the two red-class hues (`#d62728`, `#ff9896`) removed so no series can collide with the reserved alert red `_PLOT_ALERT = #d62728`. Reviewed under deuteranopia / protanopia / tritanopia simulation: the first six entries (blue `#1f77b4`, orange `#ff7f0e`, green `#2ca02c`, purple `#9467bd`, brown `#8c564b`, pink `#e377c2`) stay mutually distinguishable under all three; the green/red anti-pair no longer occurs because red is excluded from series. Beyond six entries, dash/solid is the required secondary channel (line/area already carry `dash`), and the legend cap folds the tail (PLT-16). Sequential (`Blues`) and diverging (`RdBu`) matrix ramps are D3 built-ins, both CVD-safe by construction. Re-run this check whenever `_PLOT_SERIES` or `_PLOT_SERIES_DARK` changes.

### Reference lines and annotations

- Zero line on every axis where sign matters: IRF, event study, cycle, HD contributions, news impacts, coefficient plots. (Currently missing on the HD actual-vs-reconstructed panel — add it.)
- Event-time zero gets a vertical treatment line; ACF gets ±CI lines; OccBin gets binding-region shading. These are renderer options (A4), not post-hoc scripts.
- The reference/normalized period in an event study is visually marked (hollow marker at the omitted category).

### Axes, titles, legends

- **Horizon conventions:** IRF x starts at 0; FEVD at 1; document any deviation in the docstring. Horizon, lag, and event-time axes force integer ticks — a tick at h = 2.5 is a bug.
- **Date axes:** whenever the source object carries time metadata (`TimeSeriesData.time_index`, panel `time_id`, nowcast vintages), the x axis shows dates, not 1…T. Plots of HD/filters/forecasts over "Period 143" are unacceptable for empirical work; converters accept an optional `time_index` and `_json` learns `Date`. Where the estimation drops initial observations (VAR lags, Hamilton/BK trimming), the axis must reflect the *calendar* position of each point, which the `offset` mechanism must map correctly.
- Every panel has x and y axis labels with units (`"Horizon (quarters)"` when known, `"Response (%)"` for percent-scaled IRFs).
- Figure title = object + method + inference statement (`"Impulse Response Functions (bootstrap 90% CI)"`). Panel title = the specific slice (`"gdp ← monetary"` convention: *response ← shock*).
- Single-panel figures must not repeat the same string as both figure title and panel title; give the panel a specific subtitle or suppress one.
- **Legend policy:** ≥2 series → legend always; 1 series → no legend (the title names it). Legend layout must be width-aware — fixed 90–130 px steps overflow with long names; wrap to a second row or truncate with full name in the tooltip. A legend may hold at most ~6 entries; beyond that use direct labels (scatter) or fold categories.
- Anything drawn that is not a series also needs a legend entry when ambiguous: overlapping CI bands, binding-region shading, equality lines.

### Heatmaps

- Always render a **color scale legend** (gradient bar with min/mid/max ticks). A heatmap without a scale is unreadable in print.
- Missing values get the neutral grey plus a legend entry "missing".
- Diverging scale ⇒ symmetric domain around the meaningful midpoint; sequential scale ⇒ `[0 or data-min, data-max]` single hue.

---

## Interaction

- Every renderer ships a tooltip (crosshair-nearest for line/area, per-mark for bar/cell/point). Tooltip text is assembled from series names + `fmt()` values; prefix comes from `xlabel`/parameter (A2).
- Hover targets are at least as large as the visual mark; scatter marks enlarge on hover (already standard — keep).
- Interactivity degrades gracefully: the chart must be complete without hovering (tooltips reveal precision, never information that exists nowhere else). This is what makes the same HTML acceptable as a static screenshot in papers.

---

## Robustness

Every plot method and renderer must survive, with sensible output:

| Input | Required behavior |
|---|---|
| `NaN` / `Inf` / `missing` values | `null` in JSON → visible gap (`.defined()`), never a broken SVG path; never `undefined` reaching a scale (a *missing key* in a data row is a bug — emit explicit `null`) |
| Empty series / single observation | no JS exception; degenerate extents padded (`\|\| 1` fallbacks) |
| Flat (constant) series | padded y-domain, line visible mid-panel |
| Names containing `'`, `"`, `<`, `&`, newlines, unicode | rendered verbatim, document intact (A8) |
| Very long names | truncated with ellipsis in axis/legend, full name in tooltip |
| More series than palette | fold or error per Color rules; never an out-of-bounds slice |
| Huge N (scatter/Q-Q with 10⁵ points) | subsample to ≤ ~2,000 marks with a visible note (C7); never emit multi-MB JSON silently |
| Wide panels (n_vars × n_shocks large) | cap the grid and say so (C7); flex-wrap must not silently destroy the shock-column alignment |
| Misaligned auxiliary series (`original` shorter than trend+offset) | throw `ArgumentError` naming the expected length — never silently shift or truncate the drawn series |
| Two plots in one notebook page | both render (A11) |

Performance rule of thumb: data-assembly loops in Julia must be O(total points drawn); scanning the full panel per (group, time) cell (`findfirst` over N rows inside a t×g loop) is a defect — pre-index once.

---

## Theming & Accessibility

- The HTML theme (background, ink colors, grid) is defined once in `_render_css` as CSS custom properties. Dark mode ships via `@media (prefers-color-scheme: dark)` overrides of those properties so docs iframes match the site's Solarized light/dark toggle; series colors get dark-surface-validated variants, not an automatic inversion.
- Contrast: axis/label ink ≥ WCAG AA against both surfaces.
- Print/export: the layout must not depend on hover; `svg { overflow: visible }` keeps legends printable.

---

## Testing Rules

Every new or changed plot method requires tests in `test/plotting/`:

1. **Smoke:** returns a `PlotOutput`; `save_path` writes a file that starts with `<!DOCTYPE html>`.
2. **Content:** HTML contains the expected panel count, panel titles, series names, and auto-title.
3. **Kwargs:** `var`/`shock` selection by Int *and* by String; bad name → `ArgumentError`; bad index → `ArgumentError`; `ncols`, `title` override, `view` dispatch incl. the error branch.
4. **Data edge cases:** input containing `NaN` produces `null` (assert `"null"` present, no `"NaN"` in the JS); constant series; minimal length.
5. **Escaping:** a variable named `a"b<c>` round-trips into valid HTML (no unescaped `<` inside the panel title element, no unescaped `"` inside script strings).
6. **JSON validity:** embedded `const data = [...]` blocks parse as strict JSON (extract-and-parse in the test; duplicate keys fail).
7. **Caps:** whatever C7 note the method emits is asserted.

Renderer changes additionally require one test per option (bands, ref lines incl. `axis:"x"`, overlays, modes).

---

## Documentation Rules

- Every `plot_result` method appears on `docs/src/plotting.md` with a generated example embedded per `docrule.md` ("Embedding Plot Iframes"). The page's supported-type count and dispatch table must match `length(methods(plot_result))` — generate the table with `@eval` rather than maintaining it by hand.
- Example plots are produced by `docs/generate_plots.jl` — never hand-edited HTML — into `docs/src/assets/plots/`, and regenerated whenever plotting code changes (the release workflow's "regenerate plots if plotting/docs changed" step). Every generated asset is embedded by some page; orphan assets are deleted. The generator should run in CI as a smoke test so that plot methods without unit tests still get exercised.
- The docstring lists every kwarg with defaults and states the visual anatomy (panels, lines, bands, reference marks) precisely (C6).

---

## Review Checklist

Copy into the PR description for any change under `src/plotting/`:

```
Architecture
[ ] Renderers only in render.jl; no domain nouns inside renderers
[ ] No scale re-derivation outside renderers; overlays via renderer options
[ ] Converters in helpers.jl; no inline row-building loops in plot methods
[ ] All interpolated strings escaped (HTML sink, JS sink, JSON sink)
[ ] No positional struct-literal adapters
API
[ ] plot_result only; standard kwarg names; String+Int selection; bounds-checked
[ ] No dead kwargs; view dispatch errors on unknown views
[ ] Docstring matches the pixels; numbers in titles rounded, not _json'd
[ ] No silent caps — truncation visible in output
Design
[ ] Correct form for the data's job (table above)
[ ] Fixed color order; entity-stable colors; no red-as-series next to red refs
[ ] Sequential vs diverging palette matches data sign; heatmap has color legend
[ ] Zero/reference lines present; axes labeled; integer ticks on horizon/lag axes
[ ] Legend policy respected; width-aware; bands/shading legended when ambiguous
Robustness & tests
[ ] NaN/empty/flat/long-name/quote-in-name cases handled + tested
[ ] Embedded JSON parses strictly; no duplicate keys
[ ] Big-N subsampling with note where applicable
[ ] test/plotting updated per Testing Rules
```

---

## Anti-Patterns

Concrete failures this guide exists to prevent (all observed in review):

1. **The lying docstring** — "horizontal bar chart" rendered vertical; "vertical line at posterior mean" never drawn; "reference period marker" absent.
2. **The leaked noun** — generic heatmap tooltip printing `Period Agriculture` because "Period" was hardcoded in the renderer.
3. **The dead kwarg** — `stat=:mean` accepted and ignored; `group_names` accepted and ignored.
4. **The duplicate-key bridge** — appending a second `"fc"` pair to a JSON row instead of replacing it.
5. **The phantom original** — reconstructing `trend .+ cycle` and indexing it with the full-series offset, drawing a shifted, truncated "Original" line (Hamilton/BK without `original=`).
6. **The scale clone** — post-hoc `<script>` that rebuilds margins and scales to add one vertical line, doubling the data payload and breaking on the next renderer edit.
7. **The silent cap** — plotting 5 of 12 factors, 10 of 500 panel groups, or point-sampled 60 of 1000 grid masses with no indication.
8. **The unrounded title** — `TWFE = 0.12345678901234567` via `_json` interpolation; `breakdown = null` when the value is `NaN`.
9. **The legend march** — one legend entry per sector at fixed 130 px spacing, walking off the right edge of the panel.
10. **The diverging rainbow** — RdBu on an all-positive matrix; midpoint at max/2 means nothing.
11. **The CDN coin-flip** — plots blank on planes, behind proxies, and in archived supplementary materials because d3 came from a CDN.
12. **The notebook collision** — second plot in a Jupyter page throws `Identifier 'tooltip' has already been declared`.
