# [Visualization](@id plotting_page)

MacroEconometricModels.jl includes a zero-dependency visualization system that renders interactive HTML/SVG charts using inline [D3.js](https://d3js.org/) v7. The unified `plot_result()` function dispatches on every plottable result type in the package, producing self-contained HTML documents with interactive tooltips.

## Quick Start

**Recipe 1: Plot and save**

```julia
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

---

## Chart Types

| Chart | Used for | Features |
|-------|----------|----------|
| **Line** | IRF, forecasts, filters, volatility, data series | CI bands, dashed lines, zero reference, tooltips |
| **Stacked area** | FEVD | Proportions summing to 1.0, per-shock coloring |
| **Bar** | Historical decomposition, nowcast news | Stacked/grouped, diverging for negative values |

---

## Dispatched Result Types

`plot_result()` is defined for every result type the package produces. The table below is generated at build time from `methods(plot_result)`, so it never drifts from the implementation. The **Category** column names the source module that carries each method.

```@eval
import MacroEconometricModels
import Markdown
rows = Tuple{String,String}[]
for m in methods(MacroEconometricModels.plot_result)
    sig = Base.unwrap_unionall(m.sig)
    (sig isa DataType && length(sig.parameters) >= 2) || continue
    arg = sig.parameters[2]
    arg isa UnionAll && (arg = Base.unwrap_unionall(arg))
    arg isa DataType || continue
    fam = replace(basename(String(m.file)), r"\.jl$" => "")
    push!(rows, (string(arg.name.name), fam))
end
sort!(rows, by = x -> (x[2], x[1]))
io = IOBuffer()
println(io, "| Result type | Category |")
println(io, "|-------------|----------|")
for (t, f) in rows
    println(io, "| `$t` | `$f` |")
end
Markdown.parse(String(take!(io)))
```

---

## Type-Specific Keyword Arguments

Beyond the **Common Options** above, several result types accept a `view=` selector or type-specific keywords that control what is plotted.

**`view=` selectors:**

| Result type | `view=` values | Selects |
|-------------|----------------|---------|
| `HASteadyState` | `:distribution` (default), `:lorenz`, `:policy` | Wealth histogram, Lorenz curve, or policy functions |
| `NowcastResult` | `:default`, `:heatmap`, `:contributions` | Factor path, ragged-edge z-scores, or grouped contributions |
| `NowcastNews` | `:releases` (default), `:groups`, `:individual` | News impact by release, by group, or per series |

**Other type-specific keywords:**

| Keyword | Type | Default | Applies to | Description |
|---------|------|---------|------------|-------------|
| `var` | `Int`/`String`/`nothing` | `nothing` | FEVD, HD, VAR/factor forecasts | Select one variable (`nothing` = all) |
| `shock` | `Int`/`String`/`nothing` | `nothing` | `ImpulseResponse` | Select one shock (`nothing` = all) |
| `stat` | `Symbol` | `:mean` | `BayesianFEVD`, `BayesianHistoricalDecomposition` | Posterior summary statistic |
| `bias_corrected` | `Bool` | `true` | `LPFEVD` | Plot the bias-corrected proportions |
| `history` / `n_history` | `Vector`/`nothing`, `Int` | `nothing`, `50` | `ARIMAForecast`, `VolatilityForecast` | Prepend observed history to the forecast |
| `original` | `Vector`/`nothing` | `nothing` | Filter results | Overlay the raw series on trend/cycle |
| `type` / `n_obs` | `Symbol`, `Int` | `:both`, `6` | `FactorForecast` | Plot factors, observables, or both (and how many) |
| `vars` | `Vector`/`nothing` | `nothing` | `TimeSeriesData`, `PanelData` | Subset of columns to plot |

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

## Complete Example

Estimate a VAR, identify shocks recursively, plot the impulse responses, and save the result to a self-contained HTML file. Because `plot_result()` returns a `PlotOutput` rather than printing, the call stays in a static block; the saved asset is embedded below with an `@raw html` iframe.

```julia
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "UNRATE", "CPIAUCSL"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
m = estimate_var(Y, 4)
r = irf(m, 20; ci_type=:bootstrap, reps=500)
p = plot_result(r)
save_plot(p, "irf_freq.html")   # written to docs/src/assets/plots/
```

```@raw html
<iframe src="../assets/plots/irf_freq.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## Common Pitfalls

1. **Unsupported `view=` value** — Passing a `view` a type does not define (e.g. `view=:lorenz` on a `NowcastResult`) throws an `ArgumentError`. Consult the **Type-Specific Keyword Arguments** table above for the accepted values.
2. **Forgetting `save_plot` leaves the iframe empty** — An `@raw html` iframe only resolves if the referenced HTML file exists on disk. Call `save_plot(p, "name.html")` (writing into `docs/src/assets/plots/`) before embedding.
3. **Iframe `src` must be `../assets/plots/…`** — Documenter builds with `prettyurls=true`, rendering each page as `page_name/index.html`, so the asset path is one directory up. A bare `assets/plots/…` (no `../`) yields a 404.
4. **`plot_result()` inside `@example` double-renders** — In an executed `@example` block the returned `PlotOutput` is displayed as raw HTML *and* the iframe below repeats it. Keep `plot_result()` in a static ```` ```julia ```` block and show the visualization once through the iframe.

---

## References

- Bostock, M., Ogievetsky, V., & Heer, J. (2011). D3: Data-Driven Documents.
  *IEEE Transactions on Visualization and Computer Graphics*, 17(12), 2301-2309.
  [DOI: 10.1109/TVCG.2011.185](https://doi.org/10.1109/TVCG.2011.185)
