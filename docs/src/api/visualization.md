# [Visualization API](@id api_visualization)

D3.js interactive plotting via the unified `plot_result` dispatch. See [Visualization](@ref plotting_page) for the dispatch catalog and worked examples, and [Utilities & Display API](@ref api_utilities) for the display-backend and table infrastructure.

---

## Plotting Types

```@docs
PlotOutput
```

---

## Core Plot Functions

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
