# Nowcast Plotting Enhancement — Design Spec

**Date:** 2026-05-24
**Version:** 0.4.2
**Reference:** ECB Nowcasting Toolbox (Linzenich & Meunier 2024, WP 3004)

## Overview

Enhance MacroEconometricModels.jl nowcasting visualizations from 2 plot types to 7, matching the core visualization patterns of the ECB Nowcasting Toolbox. All new plots use the existing `plot_result` convention with a `view` keyword for dispatch.

## Scope

5 new plot views, 1 struct change, 1 new D3.js renderer. No new exported functions.

Out of scope for this iteration: vintage tracking (NowcastHistory), range of alternative models (leave-group-out), accuracy comparison (RMSE), and fixing `impact_revision` computation (hardcoded to zero).

## Struct Change

### `NowcastNews{T}` — add `group_names` field

```julia
struct NowcastNews{T<:AbstractFloat}
    old_nowcast::T
    new_nowcast::T
    impact_news::Vector{T}
    impact_revision::T
    impact_reestimation::T
    group_impacts::Vector{T}
    group_names::Vector{String}      # NEW
    variable_names::Vector{String}
end
```

The `nowcast_news()` function gains a `group_names::Union{Vector{String},Nothing}=nothing` keyword. Auto-generation rules:

- `groups` provided, `group_names` not: generate `["Group 1", "Group 2", ...]`
- Neither provided (default one-group-per-variable): use `"Var1"`, `"Var2"`, etc. derived from variable index

All existing constructor call sites updated.

## New Renderer

### `_render_heatmap_js` in `render.jl`

D3.js heatmap renderer for color-coded matrices.

**Signature:**
```julia
_render_heatmap_js(id, data_json, row_labels_json, col_labels_json;
                   xlabel="", ylabel="", color_domain=[-3, 3])
```

**Behavior:**

- Diverging blue-white-red color scale (`d3.interpolateRdBu` reversed: blue=positive, red=negative)
- `null` values rendered in `#d9d9d9` (grey) — this visualizes the ragged edge
- No numeric labels inside cells; values shown via tooltip on hover
- Row labels on left (truncated to ~20 chars), column labels on bottom
- `color_domain` defaults to `[-3, 3]` (standard deviations), values clamped

**Data format:** JSON array of objects `{x: colLabel, y: rowLabel, v: numericOrNull}`.

## Plot Types

### API Signatures

```julia
# NowcastNews plots
plot_result(nn::NowcastNews{T};
            view::Symbol=:releases,        # :releases | :groups | :individual
            title::String="",
            save_path::Union{String,Nothing}=nothing) where {T}

# NowcastResult plots
plot_result(nr::NowcastResult{T};
            view::Symbol=:default,         # :default | :heatmap | :contributions
            ncols::Int=0,
            title::String="",
            save_path::Union{String,Nothing}=nothing,
            groups::Union{Vector{Int},Nothing}=nothing,
            group_names::Union{Vector{String},Nothing}=nothing,
            variable_names::Union{Vector{String},Nothing}=nothing,
            n_periods::Int=18) where {T}
```

Invalid `view` values throw `ArgumentError`.

### 1. `view=:releases` (existing, default for NowcastNews)

Existing horizontal grouped bar chart of per-release impacts. No changes except it becomes the explicit default when `view` is not specified.

### 2. `view=:groups` (new, NowcastNews)

Stacked diverging bar chart of `nn.group_impacts`.

- Uses `_render_bar_js` with `mode="stacked"`
- X-axis: single bar "News" (or two bars "News" + "Revision + Re-est." if nonzero)
- Series: one per group, colored from `_PLOT_COLORS`, labeled with `nn.group_names`
- Title: old → new nowcast with delta

### 3. `view=:individual` (new, NowcastNews)

Horizontal bar chart sorted by absolute impact.

- Uses `_render_bar_js` with `mode="grouped"`
- Bars sorted by `|impact_news[i]|` descending
- X-axis: variable names from `nn.variable_names`
- Single series "News Impact"
- Same title format as existing news plot

### 4. `view=:default` (enhanced, NowcastResult)

Existing behavior unchanged: target variable line chart with nowcast/forecast extension + additional variable panels.

**Enhancement:** When `nr.model isa NowcastDFM`, append `r` additional panels showing each extracted factor time series from `model.F[:, 1:r]`. Panels labeled "Factor 1", "Factor 2", etc. using `_render_line_js`.

No change when model is BVAR or Bridge.

### 5. `view=:heatmap` (new, NowcastResult)

Color-coded z-score matrix using `_render_heatmap_js`.

- Computes z-scores from `nr.model.data` (original with NaN): `(x - mean) / std` per column using non-NaN values
- Shows last `n_periods` periods (default 18, matching ECB convention)
- Rows = variables (all N), columns = period indices
- NaN cells rendered grey — this IS the ragged edge visualization
- Optional row reordering by `groups` kwarg (variables within same group adjacent)
- Row labels from `variable_names` kwarg (auto-generate `"Var 1"`, ... if not provided)

### 6. `view=:contributions` (new, NowcastResult)

Group contributions stacked bar chart.

- Requires `nr.model isa NowcastDFM` — throws `ArgumentError` for BVAR/Bridge
- Computes group contributions: for each group g, sum `C[target_var, factor_cols] * F[end, factor_cols]` where `factor_cols` are factors assigned to group g via `model.blocks`
- Two stacked bars: "Nowcast" and "Forecast" periods
- Series colored by group plus "Mean" bar (unconditional mean `model.Mx[target_var]`)
- Uses `_render_bar_js` with `mode="stacked"`
- Needs `groups` and `group_names` kwargs

**Default auto-generation:** When `groups` is `nothing`, assign one group per block column from `model.blocks`. When `group_names` is `nothing`, generate `["Block 1", ...]`.

## Files Modified

| File | Change |
|---|---|
| `src/nowcast/types.jl` | Add `group_names::Vector{String}` to `NowcastNews` |
| `src/nowcast/news.jl` | Add `group_names` kwarg, pass to constructor |
| `src/plotting/render.jl` | Add `_render_heatmap_js()` |
| `src/plotting/nowcast.jl` | Rewrite with `view` dispatch: existing code → `:releases`/`:default`, add 4 new views |
| `test/plotting/test_plot_result.jl` | Tests for all 5 new views |

No new files. No new exports.

## Testing

Each new view gets a `@testset`:

1. Construct minimal data (small matrices, synthetic NowcastDFM/NowcastNews)
2. Call `plot_result` with the view
3. Assert: returns `PlotOutput`, HTML contains expected keywords, `save_plot` round-trips

Additional tests:

- `NowcastNews` struct: verify `group_names` populated correctly with/without explicit names
- `ArgumentError` for invalid `view` values
- `ArgumentError` for `:contributions` on non-DFM models
- Heatmap with all-NaN column (grey rendering)
- Factor panels only appear for DFM models

## Non-Goals

- `impact_revision` computation fix (currently hardcoded to zero) — separate task
- Vintage tracking / NowcastHistory type
- Leave-group-out range plots
- RMSE accuracy comparison plots
- Documentation updates (done separately per CLAUDE.md rules)
