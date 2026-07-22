# [Visualization](@id plotting_page)

MacroEconometricModels.jl renders every plottable result through a single entry point, `plot_result`, which returns a `PlotOutput` wrapping one **self-contained HTML document**. The charts are drawn with [D3.js](https://d3js.org/) v7, whose source is **vendored offline** and inlined into each document — no content delivery network, no external stylesheet, no runtime fetch — so a saved plot renders identically on an air-gapped machine, behind a corporate proxy, or embedded as a Documenter iframe. Every document is **theme-aware**: it reads the viewer's light/dark preference (and the docs site's theme toggle) and restyles its surface, ink, and grid accordingly.

The authoring standard for all plotting code — renderers, converters, kwargs, color, and testing — is [`docs/plotrule.md`](https://github.com/FriedmanJP/MacroEconometricModels.jl/blob/main/docs/plotrule.md). This page documents the user-facing API and catalogs the shipped dispatches.

## Quick Start

Every recipe stays in a static `julia` block because `plot_result` returns a `PlotOutput` rather than printing; the resulting chart is shown once through an iframe (running these calls inside an executed block would render the plot twice).

**Recipe 1: Plot and save**

```julia
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
m = estimate_var(Y, 4)
r = irf(m, 20; ci_type=:bootstrap, reps=500)
p = plot_result(r)
save_plot(p, "irf_plot.html")
```

**Recipe 2: Display in the browser**

```julia
fred = load_example(:fred_md)
d = fred[:, ["INDPRO", "UNRATE", "CPIAUCSL"]]
display_plot(plot_result(d))
```

**Recipe 3: Switch views**

Multi-view results select an exhibit with `view=`:

```julia
ss = compute_steady_state(load_ha_example(:krusell_smith))
plot_result(ss; view=:lorenz)      # or :distribution, :policy
```

**Recipe 4: Select one response or shock by name or index**

Wherever a result carries names, `var` and `shock` accept **both** an `Int` position and a `String` name:

```julia
plot_result(r; var="INDPRO", shock="FEDFUNDS")   # equivalently var=1, shock=3
```

**Recipe 5: Save and keep the object**

`save_path` writes the file *and* returns the `PlotOutput`, so a plot can be embedded and reused in one call:

```julia
p = plot_result(r; save_path="irf.html")   # p is still a PlotOutput
```

---

## The `plot_result` contract

Users plot any result with `plot_result(result; kwargs...)`. The exported plotting surface is exactly `plot_result`, `PlotOutput`, `save_plot`, and `display_plot`; every other plotting name is internal.

A `PlotOutput` carries a complete HTML document in its `html` field and displays itself according to context:

| Context | Behavior |
|---------|----------|
| **Jupyter / IJulia** | Automatic inline display via `MIME"text/html"` |
| **REPL** | `display_plot(p)` opens the document in the default browser |
| **File** | `save_plot(p, "path.html")` writes the document to disk |
| **Programmatic** | Access `p.html` directly |

The keyword vocabulary is standardized — these names carry these exact meanings on every dispatch that accepts them:

| Keyword | Type | Default | Semantics |
|---------|------|---------|-----------|
| `var` | `Int` / `String` / `nothing` | `nothing` | Select one response variable; `nothing` plots all |
| `shock` | `Int` / `String` / `nothing` | `nothing` | Select one shock; `nothing` plots all |
| `vars` | `Vector` / `nothing` | `nothing` | Select several columns (data containers) |
| `view` | `Symbol` | type-specific | Switch between named views of one result |
| `ncols` | `Int` | `0` (auto) | Columns in a multi-panel grid |
| `title` | `String` | `""` | Override the auto-generated figure title |
| `save_path` | `String` / `nothing` | `nothing` | Also write the HTML to this path |
| `history` / `n_history` | `Vector` / `Int` | `nothing` / `50` | Prepend observed history to a forecast fan |
| `stat` | `Symbol` | `:mean` | Posterior central line (`:mean` or `:median`) on Bayesian IRF/FEVD/HD |

**Name-and-index selection.** `var`, `shock`, and `vars` accept an `Int` position or a `String` name wherever the result stores names. An out-of-range index or an unknown name raises an `ArgumentError` rather than a downstream `BoundsError`.

**`save_path` semantics.** When given, `plot_result` writes the file and still returns the `PlotOutput`; it never opens a browser implicitly — that is `display_plot`'s job.

---

## Type-specific views and keywords

Results that carry more than one natural exhibit dispatch on `view::Symbol`. Passing a `view` a type does not define raises an `ArgumentError` that lists the accepted values, so the valid set is always discoverable. The most common multi-view types:

| Result type | `view=` values |
|-------------|----------------|
| `HASteadyState` | `:distribution` (default), `:lorenz`, `:policy` |
| `HADSGESolution` | `:distribution_dynamics` (default; alias `:distribution`), `:inequality` |
| `NowcastResult` | `:default`, `:heatmap`, `:contributions` |
| `NowcastNews` | `:releases` (default), `:groups`, `:individual` |
| `PVARModel` | `:oirf` (default), `:girf`, `:fevd`, `:stability` |
| `CTSteadyState` | `:distribution` (default), `:policy`, `:lorenz` |
| `CTTwoAssetSolution` | `:consumption`, `:density`, `:deposit`, `:liquid_drift`, `:illiquid_drift` |
| `MarkovSwitchingSVARResult` | `:regimes` (default), `:B0`, `:transition` |
| `ICASVARResult` | `:mixing` (default), `:unmixing`, `:shocks` |
| `GARCHSVARResult` | `:variance` (default), `:shocks` |
| GARCH-family models | `:default`, `:diagnostics`, `:news_impact` |
| `PerfectForesightPath` | `:levels` (default), `:deviations` |
| `BaiPerronResult` | `:criteria` (default), `:breaks` |
| `HeckmanModel` | `:outcome` (default), `:selection` |
| `ForecastEvaluation` | `:metrics` (default), `:theil` |
| `TimeSeriesData` | `:line` (default), `:corr` |
| `CrossSectionData` | `:hist` (default), `:corr`, `:binscatter` |
| `PanelData` | `:lines` (default), `:binscatter` |
| `BVARPosterior`, `BayesianDSGE` | `:trace` |

Reduced-form and estimated models additionally expose `view=:diagnostics`, a shared four-panel residual figure (fitted-vs-residual, residual ACF, histogram, and Q-Q), so a model's specification can be screened without leaving the plotting API.

Other keywords are scoped to the results that use them: `stat=:mean`/`:median` selects the central line of a Bayesian `irf`/`fevd`/`historical_decomposition` fan; `history=`/`n_history=` prepend observed data to any forecast fan (VAR, BVAR, VECM, factor, ARIMA, volatility, LP); `type=:both`/`:factors`/`:observables` and `n_obs` control a `FactorForecast`; and `original=` overlays the raw series on a filter's trend/cycle panels.

---

## Dispatched result types

`plot_result` is defined for every result type the package produces. The table below is generated at build time from `methods(plot_result)`, grouped into five user-facing categories. A build-time guard asserts that the table lists **every** method, so it can never silently drift from the implementation: if a new dispatch is added but the generation filter fails to capture it, the counts diverge and the documentation build fails here rather than shipping a stale catalog.

```@eval
import MacroEconometricModels
import Markdown
const _P = MacroEconometricModels

# Each plotting source file (basename) maps to one of the five categories.
_cat = Dict(
    "irf" => "Innovation accounting & forecasting",
    "fevd" => "Innovation accounting & forecasting",
    "hd" => "Innovation accounting & forecasting",
    "filters" => "Innovation accounting & forecasting",
    "forecast" => "Innovation accounting & forecasting",
    "nowcast" => "Innovation accounting & forecasting",
    "spectral" => "Innovation accounting & forecasting",
    "arima" => "Innovation accounting & forecasting",
    "diagnostics" => "Innovation accounting & forecasting",
    "models" => "Estimated models & the view API",
    "mgarch" => "Estimated models & the view API",
    "nonlinear" => "Estimated models & the view API",
    "penalized" => "Estimated models & the view API",
    "midas" => "Estimated models & the view API",
    "ardl" => "Estimated models & the view API",
    "gmm" => "Estimated models & the view API",
    "io" => "Estimated models & the view API",
    "timeseries" => "Data-analysis & cross-section",
    "panel" => "Data-analysis & cross-section",
    "crosssection" => "Data-analysis & cross-section",
    "binscatter" => "Data-analysis & cross-section",
    "reg" => "Data-analysis & cross-section",
    "micro_coef" => "Data-analysis & cross-section",
    "nonparametric" => "Data-analysis & cross-section",
    "bayes" => "Bayesian & residual diagnostics",
    "bayesfan" => "Bayesian & residual diagnostics",
    "mcmc" => "Bayesian & residual diagnostics",
    "pvar" => "Structural ID, HA/DSGE dynamics & tests",
    "svar_setid" => "Structural ID, HA/DSGE dynamics & tests",
    "svar_statid" => "Structural ID, HA/DSGE dynamics & tests",
    "ha_dynamics" => "Structural ID, HA/DSGE dynamics & tests",
    "ct_olg" => "Structural ID, HA/DSGE dynamics & tests",
    "dsge_extra" => "Structural ID, HA/DSGE dynamics & tests",
    "fceval" => "Structural ID, HA/DSGE dynamics & tests",
    "teststat" => "Structural ID, HA/DSGE dynamics & tests",
    "teststat_breaks" => "Structural ID, HA/DSGE dynamics & tests",
    "did" => "Structural ID, HA/DSGE dynamics & tests",
)
_tn(t) = string((t isa UnionAll ? Base.unwrap_unionall(t) : t).name.name)

rows = Tuple{String,String}[]
for m in methods(_P.plot_result)
    sig = Base.unwrap_unionall(m.sig)
    (sig isa DataType && length(sig.parameters) >= 2) || continue
    arg = sig.parameters[2]
    fam = replace(basename(String(m.file)), r"\.jl$" => "")
    cat = get(_cat, fam, "Other dispatches")
    if arg isa Union
        nm = join(sort(unique(String[_tn(t) for t in Base.uniontypes(arg)])), " / ")
        push!(rows, (cat, nm))
    else
        ua = arg isa UnionAll ? Base.unwrap_unionall(arg) : arg
        ua isa DataType && push!(rows, (cat, _tn(ua)))
    end
end

# Build-time drift guard: the table must list EVERY plot_result method.
_total = length(collect(methods(_P.plot_result)))
length(rows) == _total ||
    error("plotting.md dispatch table drift: listed $(length(rows)) of $_total " *
          "plot_result methods — the @eval filter dropped a real dispatch.")

sort!(rows, by = x -> (x[1], x[2]))
io = IOBuffer()
println(io, "The package ships **$(length(rows))** `plot_result` methods.")
println(io)
println(io, "| Category | Result type |")
println(io, "|----------|-------------|")
for (c, t) in rows
    println(io, "| $c | `$t` |")
end
Markdown.parse(String(take!(io)))
```

---

## Gallery

The examples below are produced by `docs/generate_plots.jl` — never hand-edited HTML — into `docs/src/assets/plots/`, and regenerated whenever plotting code changes. They are organized by the five categories above.

### 1. Innovation accounting & forecasting

Impulse responses, variance decompositions, historical decompositions, trend-cycle filters, and forecast fans — the core exhibits, with confidence/credible bands and mandatory zero references.

```@raw html
<iframe src="../assets/plots/irf_freq.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/fevd_freq.html" width="100%" height="460" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/hd_freq.html" width="100%" height="460" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/filter_hp.html" width="100%" height="460" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/forecast_var.html" width="100%" height="460" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### 2. Estimated models & the view API

Volatility, factor, DSGE, input-output, and nonlinear models. The same result type exposes several exhibits through `view=`: a GARCH fit renders its `:default` diagnostics or its `:news_impact` curve; a Leontief inverse renders as a sequential single-hue heatmap with a color-scale legend.

```@raw html
<iframe src="../assets/plots/model_garch.html" width="100%" height="520" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/news_impact_curve.html" width="100%" height="460" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/leontief_heatmap.html" width="100%" height="480" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/gmm_moment_fit.html" width="100%" height="440" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

An occasionally-binding-constraint (OccBin) solution path shades the periods in which the constraint binds:

```@raw html
<iframe src="../assets/plots/occbin_solution.html" width="100%" height="460" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### 3. Data-analysis & cross-section plotting

Container-level exploration: time-series correlation heatmaps, panel small-multiples, and binscatters with within-group demeaning and controls.

```@raw html
<iframe src="../assets/plots/data_timeseries_corr.html" width="100%" height="480" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/binscatter.html" width="100%" height="440" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/data_panel.html" width="100%" height="480" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### 4. Bayesian & residual diagnostics

Model-adequacy screens and posterior diagnostics: the shared four-panel OLS residual figure, MCMC trace plots, and prior/posterior overlap identification screens.

```@raw html
<iframe src="../assets/plots/reg_ols_diagnostics.html" width="100%" height="520" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/mcmc_trace.html" width="100%" height="480" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/prior_posterior.html" width="100%" height="440" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

A binary-choice model additionally renders a classification-diagnostics panel (confusion counts and threshold sweep):

```@raw html
<iframe src="../assets/plots/reg_classification.html" width="100%" height="460" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### 5. New model & test dispatches

Panel VAR structural analysis, set-identified and statistically-identified SVARs, heterogeneous-agent and continuous-time dynamics, and micro/forecast-evaluation exhibits.

```@raw html
<iframe src="../assets/plots/pvar_irf.html" width="100%" height="460" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/pvar_stability.html" width="100%" height="460" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/svar_setid_band.html" width="100%" height="460" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/ms_regime_probs.html" width="100%" height="440" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/ha_distribution_dynamics.html" width="100%" height="460" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/ha_inequality.html" width="100%" height="460" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/ct_distribution.html" width="100%" height="440" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/ct_policy.html" width="100%" height="460" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/micro_coef_forest.html" width="100%" height="440" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/odds_ratio_forest.html" width="100%" height="440" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/mincer_zarnowitz.html" width="100%" height="440" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/state_lp.html" width="100%" height="460" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## Where else visualizations appear

Beyond this gallery, worked visualizations are embedded on each section page:

| Visualization | Page |
|---------------|------|
| IRF (frequentist, Bayesian, LP, structural LP) | [Impulse Responses](@ref ia_irf_page) |
| FEVD (frequentist, Bayesian, LP) | [Variance Decomposition](@ref ia_fevd_page) |
| Historical decomposition | [Historical Decomposition](@ref ia_hd_page) |
| Filters (HP, Hamilton, BN, BK, boosted HP, X-13) | [Time Series Filters](@ref filters_page) |
| ARIMA and volatility forecasts | [ARIMA](@ref arima_page), [Volatility Models](@ref volatility_page) |
| VECM, factor, and LP forecasts | [VECM](@ref vecm_page), [Factor Models](@ref factor_page), [Local Projections](@ref lp_page) |
| Data containers | [Data Management](@ref data_page) |
| Nowcast results and news | [Nowcasting](@ref nowcast_page) |
| DSGE IRF, FEVD, OccBin | [DSGE Overview](dsge.md) |

---

## Complete Example

Estimate a VAR, identify shocks recursively, plot the impulse responses, and save the result to a self-contained HTML file. Because `plot_result` returns a `PlotOutput` rather than printing, the call stays in a static block; the saved asset is embedded below with an `@raw html` iframe.

```julia
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
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

1. **Unsupported `view=` value** — Passing a `view` a type does not define (e.g. `view=:lorenz` on a `NowcastResult`) raises an `ArgumentError` that lists the accepted values. Consult the **Type-specific views** table above.
2. **Forgetting `save_plot` leaves the iframe empty** — An `@raw html` iframe resolves only if the referenced HTML file exists. Call `save_plot(p, "name.html")` (writing into `docs/src/assets/plots/`) before embedding.
3. **Iframe `src` must be `../assets/plots/…`** — Documenter builds with `prettyurls=true`, rendering each page as `page_name/index.html`, so the asset path is one directory up. A bare `assets/plots/…` yields a 404.
4. **`plot_result` inside `@example` double-renders** — In an executed `@example` block the returned `PlotOutput` is displayed as raw HTML *and* the iframe below repeats it. Keep `plot_result` in a static ```` ```julia ```` block and show the visualization once through the iframe.

---

## References

- Bostock, M., Ogievetsky, V., & Heer, J. (2011). D3: Data-Driven Documents.
  *IEEE Transactions on Visualization and Computer Graphics*, 17(12), 2301-2309.
  [DOI: 10.1109/TVCG.2011.185](https://doi.org/10.1109/TVCG.2011.185)
- Plotting authoring standard: [`docs/plotrule.md`](https://github.com/FriedmanJP/MacroEconometricModels.jl/blob/main/docs/plotrule.md).
</content>
</invoke>
