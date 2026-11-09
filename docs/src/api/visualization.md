# [Visualization API](@id api_visualization)

D3.js interactive plotting via the unified `plot_result` dispatch. See [Visualization](@ref plotting_page) for the dispatch catalog and worked examples, and [Utilities & Display API](@ref api_utilities) for the display-backend and table infrastructure.

---

## Plotting Types

```@docs
PlotOutput
```

---

## Core Plot Functions

MacroEconometricModels.jl provides `save_plot` and `display_plot` for exporting and viewing interactive D3.js figures. Both functions support single figures as well as batched multi-plot vectors and tuples.

### Function Signatures and Overloads

| Function / Overload | Arguments | Description |
|---------------------|-----------|-------------|
| `save_plot(p, path)` | `p::PlotOutput`, `path::String` | Write a single interactive visualization to disk as a self-contained HTML document. |
| `save_plot(plots, path; title=...)` | `plots::AbstractVector{PlotOutput}`, `path::String; title::String="MacroEconometricModels Plots"` | Combine multiple figures into a single self-contained HTML page with bundled D3.js runtime, unified styles, and styled figure dividers. |
| `save_plot(plots, path; kwargs...)` | `plots::Tuple{Vararg{PlotOutput}}`, `path::String; kwargs...` | Convenience tuple overload forwarding to the vector method (`save_plot(PlotOutput[plots...], path; kwargs...)`). |
| `display_plot(p)` | `p::PlotOutput` | Save a single figure to a temporary HTML file and open it in the system default web browser (`open` on macOS, `xdg-open` on Linux, `cmd /c start` on Windows). |
| `display_plot(plots; kwargs...)` | `plots::AbstractVector{PlotOutput}; kwargs...` | Combine multiple figures into a temporary HTML document and open it in the system default web browser. |
| `display_plot(plots; kwargs...)` | `plots::Tuple{Vararg{PlotOutput}}; kwargs...` | Convenience tuple overload forwarding to the vector method (`display_plot(PlotOutput[plots...]; kwargs...)`). |

Multi-plot saving bundles the D3.js runtime and CSS stylesheet once at the top of the combined document, deduplicating dependencies and isolating panel layouts across figures using scoped CSS custom properties (`--mem-panel-width`).

```@docs
save_plot
display_plot
```

---

## Plot Dispatches

Every `plot_result` method, grouped by the source file that defines it. Each docstring lists the `view` keywords that method accepts.

### Innovation Accounting and Forecasting

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["plotting/irf.jl", "plotting/fevd.jl", "plotting/hd.jl", "plotting/forecast.jl", "plotting/fceval.jl", "plotting/nowcast.jl", "plotting/midas.jl"]
Order   = [:function]
```

### Time Series and Univariate Models

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["plotting/timeseries.jl", "plotting/filters.jl", "plotting/arima.jl", "plotting/spectral.jl", "plotting/models.jl", "plotting/mgarch.jl", "plotting/nonlinear.jl", "plotting/nonparametric.jl", "plotting/diagnostics.jl", "plotting/ardl.jl"]
Order   = [:function]
```

### Structural Identification

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["plotting/svar_setid.jl", "plotting/svar_statid.jl"]
Order   = [:function]
```

### Cross-Section, Panel, and Causal Inference

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["plotting/reg.jl", "plotting/micro_coef.jl", "plotting/penalized.jl", "plotting/crosssection.jl", "plotting/panel.jl", "plotting/pvar.jl", "plotting/did.jl"]
Order   = [:function]
```

### DSGE, Heterogeneous Agents, and Bayesian Diagnostics

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["plotting/dsge_extra.jl", "plotting/ha_dynamics.jl", "plotting/ct_olg.jl", "plotting/bayes.jl", "plotting/bayesfan.jl", "plotting/mcmc.jl"]
Order   = [:function]
```

### Hypothesis Tests, GMM, and Input-Output

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["plotting/teststat.jl", "plotting/teststat_breaks.jl", "plotting/gmm.jl", "plotting/io.jl"]
Order   = [:function]
```

### Policy Counterfactuals

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["plotting/counterfactual.jl"]
Order   = [:function]
```
