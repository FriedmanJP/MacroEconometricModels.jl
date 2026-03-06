# [Visualization](@id plotting_page)

MacroEconometricModels.jl includes a zero-dependency visualization system that renders interactive HTML/SVG charts using inline [D3.js](https://d3js.org/) v7. The unified `plot_result()` function dispatches on 41 result types, producing self-contained HTML documents with interactive tooltips.

## Quick Start

**Recipe 1: Plot and save**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "UNRATE", "CPIAUCSL"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
m = estimate_var(Y, 4)
r = irf(m, 20; ci_type=:bootstrap, reps=500)
p = plot_result(r)
save_plot(p, "irf_plot.html")
```

**Recipe 2: Display in browser**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
d = fred[:, ["INDPRO", "UNRATE", "CPIAUCSL"]]
p = plot_result(d)
display_plot(p)
```

---

## PlotOutput Type

All `plot_result()` methods return a `PlotOutput` struct containing a complete self-contained HTML document.

| Context | How it works |
|---------|--------------|
| **Jupyter/IJulia** | Automatic inline display via `MIME"text/html"` |
| **REPL** | `display_plot(p)` opens in default browser |
| **File** | `save_plot(p, "path.html")` writes HTML to disk |
| **Programmatic** | Access `p.html` directly |

---

## Common Options

All `plot_result()` methods accept these keyword arguments:

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `title` | `String` | `""` | Override auto-generated title |
| `save_path` | `String` or `nothing` | `nothing` | Auto-save HTML to path |
| `ncols` | `Int` | `0` (auto) | Number of columns in multi-panel grid |

Type-specific kwargs (e.g., `var`, `shock`, `history`, `stat`, `bias_corrected`) are documented on each section page.

---

## Chart Types

| Chart | Used for | Features |
|-------|----------|----------|
| **Line** | IRF, forecasts, filters, volatility, data series | CI bands, dashed lines, zero reference, tooltips |
| **Stacked area** | FEVD | Proportions summing to 1.0, per-shock coloring |
| **Bar** | Historical decomposition, nowcast news | Stacked/grouped, diverging for negative values |

---

## Where to Find Visualizations

Inline visualizations are embedded on each section page:

| Visualization | Page |
|---------------|------|
| IRF (frequentist, Bayesian, LP, structural LP) | [Impulse Responses](@ref ia_irf_page) |
| FEVD (frequentist, Bayesian, LP) | [Variance Decomposition](@ref ia_fevd_page) |
| Historical decomposition (frequentist, Bayesian) | [Historical Decomposition](@ref ia_hd_page) |
| Filters (HP, Hamilton, BN, BK, boosted HP) | [Time Series Filters](@ref filters_page) |
| ARIMA forecasts | [ARIMA](@ref arima_page) |
| Volatility models and forecasts | [Volatility Models](@ref volatility_page) |
| VECM forecasts | [VECM](@ref vecm_page) |
| Factor models and forecasts | [Factor Models](@ref factor_page) |
| LP forecasts | [Local Projections](@ref lp_page) |
| Data containers (TimeSeriesData, PanelData) | [Data Management](@ref data_page) |
| Nowcast results and news | [Nowcasting](@ref nowcast_page) |
| DSGE IRF, FEVD, OccBin | [DSGE Overview](dsge.md) |

---

## References

- Bostock, M., Ogievetsky, V., & Heer, J. (2011). D3: Data-Driven Documents.
  *IEEE Transactions on Visualization and Computer Graphics*, 17(12), 2301-2309.
  [DOI: 10.1109/TVCG.2011.185](https://doi.org/10.1109/TVCG.2011.185)
